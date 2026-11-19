"""Checks of reweighting: k-hat, its flags, the evidence error and the engine."""

import functools
import itertools
import math
from collections.abc import Callable, Sequence
from typing import NamedTuple

import jax
import jax.numpy as jnp
import polars as pl
from jax.typing import ArrayLike
from jaxtyping import Array, PRNGKeyArray

from mixed_signals import zoo
from mixed_signals.core.samples import (
    SampleSet,
    kish_effective_sample_size,
    weighted_covariance,
)
from mixed_signals.engines.reweight import Proposal, Reweighting, reweight
from mixed_signals.mcmc import reduce_chains
from mixed_signals.reweight.importance import log_evidence_ratio
from mixed_signals.reweight.psis import (
    pareto_k_threshold,
    pareto_tail,
    pareto_tail_length,
)
from zoo_validation.report import (
    Case,
    Cell,
    Check,
    Context,
    Measurement,
    Section,
    Table,
    claim_cases,
    claim_cases_above,
    claim_true,
    grouped,
    role,
)

EXPONENTIAL_RATES = (1.5, 2.0, 3.0, 4.0, 10.0)
"""Rates λ of the exponential proposals; the shape of the weights is 1 − 1/λ."""
NARROWER_RATIOS = (0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3)
"""Width ratios r of the narrower Gaussian proposals; on two axes k = 1 − r²."""
SHIFTS = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0)
"""Shifts δ of the proposal against the target, in widths; the true shape is 0."""
DRAW_COUNTS = (500, 2_000, 10_000)
"""Numbers of draws S of one weight set."""
LARGEST_DRAWS = DRAW_COUNTS[-1]
FULL_REPLICATES = 200
FULL_Z_REPLICATES = 400
"""Replicates of the z-scores of the evidence ratio, which resolve a spread of 1.3."""
QUICK_REPLICATES = 20
PARAMETER_DIMENSION = 8
"""Parameters of the Gaussian cases that probe other numbers of axes."""

PARETO_RECORD_TOLERANCE = 0.01
"""Allowance of the plain k-hat for the bias of the fit; the record found 0.01."""
PRIOR_TOLERANCE = 0.03
"""Allowance of k-hat with the shape prior: its shrinkage (5 − 10 k)/(M + 10) is −0.013
at k = 0.91 and M = 300, and the fit's own bias reaches about 0.02 at k = 0.84."""
SHIFT_RECORD = {1.0: 0.25, 1.5: 0.42, 2.0: 0.60, 2.5: 0.80, 3.0: 0.95}
"""Mean k-hat recorded in session 13 for a shift of δ widths at S = 10 000."""
SHIFT_RECORD_TOLERANCE = 0.03
"""Allowance for the digits of the record, its own error and the shape prior."""
SPREAD_TOLERANCE = 0.1
"""Distance of a calibrated spread of z from one that a claim allows."""
RATIO_TOLERANCE = 0.1
"""Allowed relative distance of the prediction ratio from its closed form."""
FLAG_LEVEL = 0.01
"""Rate at which a weight set that should not flag may still do so, by chance."""

CHAIN_COEFFICIENTS = (0.9, 0.9)
CHAIN_WALKERS = 32
CHAIN_STEPS = 1_900
"""Steps per walker: 100 autocorrelation times of 19."""
CHUNK_ELEMENTS = 2_000_000
"""Most numbers that one batch of zoo draws holds, which bounds its memory."""
DRAW_LABEL = "reweighting-draws"
FULL_CHAIN_REPLICATES = 600
QUICK_CHAIN_REPLICATES = 60

EXACT_LABEL = "pareto-k-exact"
GAUSSIAN_LABEL = "pareto-k-gaussians"
FLAG_LABEL = "pareto-k-flags"
ERROR_LABEL = "evidence-error"
EFFICIENCY_LABEL = "reweighted-n-eff"
RECOVERY_LABEL = "reweighting-recovery"

PARETO_TAIL = role("func", "mixed_signals.reweight.psis.pareto_tail")
PARETO_THRESHOLD = role("func", "mixed_signals.reweight.psis.pareto_k_threshold")
EVIDENCE_RATIO = role("func", "mixed_signals.reweight.importance.log_evidence_ratio")
REWEIGHT = role("func", "mixed_signals.engines.reweight.reweight")
SCALED_GAUSSIANS = role("func", "mixed_signals.zoo.cases.scaled_gaussians")
SHIFTED_GAUSSIANS = role("func", "mixed_signals.zoo.cases.shifted_gaussians")


# * Cases of weights with a known tail.


class WeightCase(NamedTuple):
    """A proposal and target pair whose weights have a known tail shape."""

    family: str
    """Name of the family, shown in the tables."""
    setting: str
    """The setting of the case in the family, such as ``r = 0.6``."""
    truth: float
    """The true shape k of the tail, 1 − 1/λ, 1 − r² or 0."""
    exact: bool
    """Whether the weights are exactly Pareto above their minimum."""
    log_ratios: Callable[[PRNGKeyArray, int, int], Array]
    """Draws log weights from a key, shaped (replicates, draws)."""

    @property
    def name(self) -> str:
        """Family and setting, which identify the case."""
        return f"{self.family}, {self.setting}"


def _exponential_log_ratios(
    rate: float, key: PRNGKeyArray, replicates: int, draws: int
) -> Array:
    """Return log weights of exponential(1) over exponential(rate), drawn from it."""
    theta = jax.random.exponential(key, (replicates, draws)) / rate
    return (rate - 1) * theta - math.log(rate)


@functools.cache
def _scaled_case(ratio: float, param_count: int, axes: tuple[int, ...]) -> zoo.ZooCase:
    return zoo.scaled_gaussians(ratio, param_count=param_count, axes=axes)


@functools.cache
def _shifted_case(shift: float) -> zoo.ZooCase:
    return zoo.shifted_gaussians(shift, param_count=2)


def _zoo_log_ratios(
    case: zoo.ZooCase,
    proposal: str,
    key: PRNGKeyArray,
    replicates: int,
    draws: int,
) -> Array:
    """Draw from the proposal of a zoo case; return log reference less log proposal."""
    chunk = max(1, CHUNK_ELEMENTS // (draws * len(case.space) * len(case.names)))
    proposal_index = case.names.index(proposal)
    reference_index = case.names.index("reference")
    chunks = []
    for start in range(0, replicates, chunk):
        size = min(chunk, replicates - start)
        sample_set = case.sample(jax.random.fold_in(key, start), draws, batch_size=size)
        log_density = case.log_prob(sample_set.samples[:, proposal_index])
        chunks.append(
            log_density[..., reference_index] - log_density[..., proposal_index]
        )
    return jnp.concatenate(chunks)


def _narrower_log_ratios(
    ratio: float,
    param_count: int,
    axes: tuple[int, ...],
    key: PRNGKeyArray,
    replicates: int,
    draws: int,
) -> Array:
    return _zoo_log_ratios(
        _scaled_case(ratio, param_count, axes), "scaled", key, replicates, draws
    )


def _shifted_log_ratios(
    shift: float, key: PRNGKeyArray, replicates: int, draws: int
) -> Array:
    return _zoo_log_ratios(_shifted_case(shift), "shifted", key, replicates, draws)


def _setting(symbol: str, value: float) -> str:
    return f"{symbol} = {value:g}"


@functools.cache
def weight_cases() -> tuple[WeightCase, ...]:
    """Return every case of the shape checks, in table order."""
    cases = [
        WeightCase(
            "exponential pair",
            _setting("λ", rate),
            1 - 1 / rate,
            True,
            functools.partial(_exponential_log_ratios, rate),
        )
        for rate in EXPONENTIAL_RATES
    ]
    families = (
        ("narrower on two axes", 2, (0, 1), True),
        ("narrower on one axis, D = 1", 1, (0,), False),
        (
            f"narrower on one axis, D = {PARAMETER_DIMENSION}",
            PARAMETER_DIMENSION,
            (0,),
            False,
        ),
        (
            f"narrower on all {PARAMETER_DIMENSION} axes",
            PARAMETER_DIMENSION,
            tuple(range(PARAMETER_DIMENSION)),
            False,
        ),
    )
    for family, param_count, axes, exact in families:
        cases.extend(
            WeightCase(
                family,
                _setting("r", ratio),
                1 - ratio**2,
                exact,
                functools.partial(_narrower_log_ratios, ratio, param_count, axes),
            )
            for ratio in NARROWER_RATIOS
        )
    cases.extend(
        WeightCase(
            "shifted",
            _setting("δ", shift),
            0.0,
            False,
            functools.partial(_shifted_log_ratios, shift),
        )
        for shift in SHIFTS
    )
    return tuple(cases)


def _exact_cases() -> list[WeightCase]:
    """Return the exponential pairs and the zoo pairs narrower on two axes."""
    return [case for case in weight_cases() if case.exact]


def _case_key(context: Context, case: WeightCase) -> PRNGKeyArray:
    """Return the key of the draws of a case, which every check that reads it shares."""
    return context.key(DRAW_LABEL, case.family, case.setting)


# * Measurements of the replicates.


def _mean(values: Array) -> Measurement:
    """Return the mean of replicates with the standard error of the mean."""
    return Measurement.from_replicates(values)


def _spread(values: Array) -> Measurement:
    """Return the standard deviation of replicates with its standard error."""
    count = int(values.shape[0])
    spread = float(jnp.std(values, ddof=1))
    return Measurement(spread, spread / math.sqrt(2 * (count - 1)), count)


# * Shapes of the weight sets, shared by the three shape checks.


class CaseShapes(NamedTuple):
    """The k-hat of every replicate of one case, at every number of draws."""

    plain: dict[int, Array]
    """Without the shape prior, by number of draws, shaped (replicates,)."""
    shrunk: dict[int, Array]
    """With the shape prior, which the engine uses."""


@functools.partial(jax.jit, static_argnames=("tail_draws", "shape_prior"))
def _tail_shapes(log_weights: Array, *, tail_draws: int, shape_prior: bool) -> Array:
    """Return k-hat of every row of a stack of weight sets."""
    return jax.vmap(
        lambda row: (
            pareto_tail(row, tail_draws=tail_draws, shape_prior=shape_prior).shape
        )
    )(log_weights)


def _replicate_count(context: Context) -> int:
    return context.count(FULL_REPLICATES, QUICK_REPLICATES)


@functools.cache
def _shapes_of_case(seed: int, quick: bool, case_name: str) -> CaseShapes:
    context = Context.create(seed, quick=quick)
    case = next(entry for entry in weight_cases() if entry.name == case_name)
    # * Smaller sets are the first draws of the largest, so one draw serves all sizes.
    log_ratios = case.log_ratios(
        _case_key(context, case), _replicate_count(context), LARGEST_DRAWS
    )
    plain, shrunk = {}, {}
    for draws in DRAW_COUNTS:
        tail_draws = pareto_tail_length(draws)
        rows = log_ratios[:, :draws]
        plain[draws] = _tail_shapes(rows, tail_draws=tail_draws, shape_prior=False)
        shrunk[draws] = _tail_shapes(rows, tail_draws=tail_draws, shape_prior=True)
    return CaseShapes(plain, shrunk)


def _shapes(context: Context, case: WeightCase) -> CaseShapes:
    return _shapes_of_case(context.seed, context.quick, case.name)


# * Check pareto-k-exact.


def _exact_shapes(context: Context) -> Section:
    replicates = _replicate_count(context)
    cases = _exact_cases()
    rows: list[list[Cell]] = []
    plain_cases: list[Case] = []
    prior_cases: list[Case] = []
    identity_error = 0.0
    prior_lowers_scatter = True
    prior_pulls_towards_half = True
    worst_scatter = ("", 1.0)
    for case in cases:
        shapes = _shapes(context, case)
        for draws in DRAW_COUNTS:
            tail_draws = pareto_tail_length(draws)
            plain, shrunk = shapes.plain[draws], shapes.shrunk[draws]
            plain_mean, shrunk_mean = _mean(plain), _mean(shrunk)
            plain_spread, shrunk_spread = _spread(plain), _spread(shrunk)
            rows.append(
                [
                    case.family,
                    case.setting,
                    case.truth,
                    draws,
                    tail_draws,
                    plain_mean,
                    shrunk_mean,
                    plain_spread.value,
                    shrunk_spread.value,
                ]
            )
            weight = tail_draws / (tail_draws + 10)
            identity_error = max(
                identity_error,
                float(jnp.max(jnp.abs((shrunk - 0.5) - weight * (plain - 0.5)))),
            )
            if draws == DRAW_COUNTS[0]:
                prior_lowers_scatter &= shrunk_spread.value < plain_spread.value
                prior_pulls_towards_half &= (
                    abs(shrunk_mean.value - 0.5) <= abs(plain_mean.value - 0.5) + 1e-12
                )
                ratio = shrunk_spread.value / plain_spread.value
                if ratio < worst_scatter[1] or not worst_scatter[0]:
                    worst_scatter = (case.name, ratio)
            if draws == LARGEST_DRAWS:
                plain_cases.append(
                    Case(
                        case.name,
                        plain_mean.value - case.truth,
                        plain_mean.margin(),
                        PARETO_RECORD_TOLERANCE,
                    )
                )
                prior_cases.append(
                    Case(
                        case.name,
                        shrunk_mean.value - case.truth,
                        shrunk_mean.margin(),
                        PRIOR_TOLERANCE,
                    )
                )
    table = Table.of(
        "Mean k-hat of exactly Pareto weights without and with the shape prior,"
        " against the true shape, with the standard deviation over replicates.",
        (
            "Weights",
            "Setting",
            "True k",
            "Draws S",
            "Tail draws M",
            "Mean k-hat",
            "Mean k-hat with prior",
            "Scatter",
            "Scatter with prior",
        ),
        rows,
    )
    claims = [
        claim_cases(
            f"At S = {grouped(LARGEST_DRAWS)} the mean k-hat of the plain fit"
            " recovers the true shape.",
            plain_cases,
            tolerance_text=f"{PARETO_RECORD_TOLERANCE:g}, for the bias of the fit",
            replicates=replicates,
        ),
        claim_cases(
            f"At S = {grouped(LARGEST_DRAWS)} the mean k-hat with the shape prior"
            " recovers the true shape.",
            prior_cases,
            tolerance_text=f"{PRIOR_TOLERANCE:g}, for the shrinkage and the bias of"
            " the fit",
            replicates=replicates,
        ),
        claim_true(
            "With the prior, k-hat less one half is M/(M + 10) times the plain"
            " k-hat less one half, in every replicate.",
            "a largest difference below 1e-12",
            f"largest difference {identity_error:.2g}",
            identity_error < 1e-12,
        ),
        claim_true(
            f"At S = {DRAW_COUNTS[0]} the prior lowers the scatter and pulls the"
            " mean towards one half in every case.",
            "a lower scatter and a mean no farther from 0.5, in every case",
            f"scatter ratio down to {worst_scatter[1]:.3f} ({worst_scatter[0]})",
            prior_lowers_scatter and prior_pulls_towards_half,
        ),
    ]
    setup = (
        f"{replicates} replicates of every case, each a set of"
        f" {grouped(LARGEST_DRAWS)} log weights, of which the first"
        f" {DRAW_COUNTS[0]} and {grouped(DRAW_COUNTS[1])} make the smaller sets."
        " The exponential pair has the target exponential(1) and the proposal"
        " exponential(λ), with the log ratio (λ − 1) θ − ln λ for draws θ of the"
        " proposal, and the true shape 1 − 1/λ (Appendix D.1 of"
        " {cite:t}`vehtari2024pareto`). The zoo pair"
        f" ({SCALED_GAUSSIANS}, two parameters, both scaled by r) has the log"
        " ratio of the reference over the narrower posterior for draws of the"
        " latter. Then (z₁² + z₂²)/2 is exponential, so the weights are exactly"
        " Pareto with the true shape 1 − r². The tail holds M ="
        " ⌊min(0.2 S, 3 √S)⌋ weights, and"
        f" {PARETO_TAIL} fits it with and without the weakly informative prior"
        " of {cite:t}`vehtari2024pareto` (Appendix G). A claim lets the mean"
        " k-hat of every case at S ="
        f" {grouped(LARGEST_DRAWS)} lie within its allowance plus three standard"
        " errors of the mean (the Student t factor for few replicates) of the"
        " truth. The prior moves k-hat linearly, so it shrinks the scatter and"
        " the distance of the mean from one half by the factor M/(M + 10), which"
        " the third claim checks replicate by replicate."
    )
    return Section(setup, (table,), tuple(claims))


# * Check pareto-k-gaussians.


def _gaussian_shapes(context: Context) -> Section:
    replicates = _replicate_count(context)
    rows: list[list[Cell]] = []
    below: list[Case] = []
    above: list[Case] = []
    shift_means: dict[float, Measurement] = {}
    for case in weight_cases():
        if case.exact:
            continue
        shapes = _shapes(context, case)
        shrunk = shapes.shrunk[LARGEST_DRAWS]
        mean = _mean(shrunk)
        rows.append(
            [case.family, case.setting, case.truth, mean, mean.value - case.truth]
        )
        if case.family.startswith("narrower on one axis") and case.truth >= 0.64:
            below.append(
                Case(
                    f"{case.name}, shortfall",
                    case.truth - mean.value,
                    mean.margin(),
                    0.0,
                )
            )
        if case.family.startswith("narrower on all"):
            above.append(Case(case.name, mean.value - case.truth, mean.margin(), 0.0))
        if case.family == "shifted":
            shift_means[float(case.setting.split("=")[1])] = mean
    table = Table.of(
        f"Mean k-hat with the shape prior at S = {grouped(LARGEST_DRAWS)} for"
        " Gaussian proposals that the exact Pareto cases do not cover, against"
        " the true shape 1 − r² of the tail of the narrowed axes, or 0 for a"
        " shift, and the difference of the two.",
        ("Weights", "Setting", "True k", "Mean k-hat", "Mean k-hat less true k"),
        rows,
    )
    shifts = sorted(shift_means)
    rising = all(
        shift_means[later].value > shift_means[earlier].value
        for earlier, later in itertools.pairwise(shifts)
    )
    record_cases = [
        Case(
            f"shift {shift:g}",
            shift_means[shift].value - recorded,
            shift_means[shift].margin(),
            SHIFT_RECORD_TOLERANCE,
        )
        for shift, recorded in SHIFT_RECORD.items()
    ]
    claims = [
        claim_cases_above(
            "From a true shape of 0.64, one narrowed axis reads below 1 − r², in"
            " one and in eight parameters: the shortfall of the mean k-hat below"
            " the truth is positive.",
            below,
            tolerance_text=None,
            replicates=replicates,
            reference_text="zero",
        ),
        claim_cases_above(
            f"With all {PARAMETER_DIMENSION} axes narrowed, k-hat reads above 1 − r²"
            " at every r.",
            above,
            tolerance_text=None,
            replicates=replicates,
            reference_text="the truth",
        ),
        claim_true(
            "For a shift, which has the true shape 0, k-hat rises with the shift.",
            "a higher mean k-hat at every larger shift",
            "; ".join(f"{shift:g}: {shift_means[shift].render()}" for shift in shifts),
            rising,
        ),
        claim_cases(
            "The mean k-hat of a shift reproduces the record of session 13.",
            record_cases,
            tolerance_text=f"{SHIFT_RECORD_TOLERANCE:g}, for the digits of the"
            " record and the prior",
            replicates=replicates,
        ),
    ]
    setup = (
        f"{replicates} replicates of sets of {grouped(LARGEST_DRAWS)} weights."
        f" The zoo pair {SCALED_GAUSSIANS} narrows one axis of a one-parameter"
        f" case, one axis of an {PARAMETER_DIMENSION}-parameter case or all"
        f" {PARAMETER_DIMENSION} axes, with the same ratios r as the exact"
        f" cases, and {SHIFTED_GAUSSIANS} shifts the proposal by δ widths, which"
        " gives lognormal weights with every moment finite. The weights of the"
        " narrowed axes are those of a chi-squared variable with one or eight"
        " degrees of freedom times a factor that varies slowly in its tail, so"
        " k-hat reads the tail index only in the limit; only two axes give an"
        " exact Pareto tail (the check on exact Pareto weights). The first claim"
        " lets the mean k-hat of one narrowed axis lie below the truth by more"
        " than three standard errors, for true shapes from 0.64; the second lets"
        " it lie above by more than three standard errors, for all ratios. The"
        " third claim wants a mean k-hat that rises at every larger shift. The"
        " last claim compares it to the record of session 13, which found 0.25,"
        " 0.42, 0.60, 0.80 and 0.95 at shifts of 1, 1.5, 2, 2.5 and 3."
    )
    return Section(setup, (table,), tuple(claims))


# * Check pareto-k-flags.


def _flag_rates(context: Context) -> Section:
    cases = weight_cases()
    flag_counts: dict[tuple[str, int], int] = {}
    rows: list[list[Cell]] = []
    replicates = _replicate_count(context)
    for case in cases:
        shapes = _shapes(context, case)
        row: list[Cell] = [case.family, case.setting, case.truth]
        for draws in DRAW_COUNTS:
            flagged = int(jnp.sum(shapes.shrunk[draws] > pareto_k_threshold(draws)))
            flag_counts[(case.name, draws)] = flagged
            row.append(Measurement.from_rate(flagged, replicates))
        rows.append(row)
    thresholds = ", ".join(
        f"{pareto_k_threshold(draws):.3f} at S = {grouped(draws)}"
        for draws in DRAW_COUNTS
    )
    table = Table.of(
        f"Fraction of {replicates} replicates whose k-hat with the shape prior"
        f" exceeds the threshold ({thresholds}).",
        (
            "Weights",
            "Setting",
            "True k",
            *(f"Flagged at S = {grouped(draws)}" for draws in DRAW_COUNTS),
        ),
        rows,
    )

    def rate(case: WeightCase, draws: int) -> Measurement:
        return Measurement.from_rate(flag_counts[(case.name, draws)], replicates)

    by_name = {case.name: case for case in cases}
    heaviest = by_name[f"narrower on two axes, {_setting('r', 0.3)}"]
    quiet = [
        case
        for case in cases
        if case.exact or case.family.startswith("narrower on one axis")
        if case.truth <= 0.36 + 1e-9
    ] + [
        by_name[f"shifted, {_setting('δ', 0.5)}"],
        by_name[f"shifted, {_setting('δ', 1.0)}"],
    ]
    quiet_draws = DRAW_COUNTS[1:]
    quiet_worst = max(
        ((case, draws) for case in quiet for draws in quiet_draws),
        key=lambda entry: flag_counts[(entry[0].name, entry[1])],
    )
    quiet_bound = FLAG_LEVEL + 3 * math.sqrt(FLAG_LEVEL * (1 - FLAG_LEVEL) / replicates)
    shifted_three = by_name[f"shifted, {_setting('δ', 3.0)}"]
    three_quarters = [
        by_name[f"exponential pair, {_setting('λ', 4.0)}"],
        by_name[f"narrower on two axes, {_setting('r', 0.5)}"],
    ]
    two_thirds_cases = [
        Case(
            f"{case.name}, S = {grouped(draws)}",
            rate(case, draws).value - 2 / 3,
            3 * (rate(case, draws).error or 0.0),
            0.05,
        )
        for case in three_quarters
        for draws in DRAW_COUNTS
    ]
    claims = [
        claim_true(
            f"At S = {grouped(LARGEST_DRAWS)}, a true shape of 0.91 flags in most"
            " replicates.",
            "a flagged fraction of at least 0.85",
            f"{rate(heaviest, LARGEST_DRAWS).render()} ({heaviest.name})",
            rate(heaviest, LARGEST_DRAWS).value >= 0.85,
            full_only=True,
        ),
        claim_true(
            "A true shape of 0.36 or less, and a shift of one width or less,"
            f" almost never flags from S = {grouped(quiet_draws[0])}.",
            f"a flagged fraction at most {quiet_bound:.3f}, the level"
            f" {FLAG_LEVEL:g} plus three binomial errors",
            f"worst of {len(quiet) * len(quiet_draws)}:"
            f" {quiet_worst[0].name}, S = {grouped(quiet_worst[1])}:"
            f" {rate(*quiet_worst).render()}",
            all(
                rate(case, draws).value <= quiet_bound
                for case in quiet
                for draws in quiet_draws
            ),
            full_only=True,
        ),
        claim_true(
            "A shift of three widths flags in nearly every replicate.",
            "a flagged fraction of at least 0.95 at every S",
            "; ".join(
                f"S = {grouped(draws)}: {rate(shifted_three, draws).render()}"
                for draws in DRAW_COUNTS
            ),
            all(rate(shifted_three, draws).value >= 0.95 for draws in DRAW_COUNTS),
            full_only=True,
        ),
        claim_cases(
            "A true shape of 0.75 flags in about two thirds of the replicates, the"
            " scatter of k-hat around its threshold.",
            two_thirds_cases,
            tolerance_text="0.05",
            replicates=None,
        ),
    ]
    setup = (
        f"The replicates of the shape checks ({replicates} per case), whose k-hat"
        f" with the shape prior is compared with {PARETO_THRESHOLD}. The"
        " fraction that flags is shown with its binomial error. The threshold"
        " falls below 0.7 for fewer draws, to 0.629 at S = 500, so a true shape"
        " of 0.75 lies above it at every S but close to it, and about two"
        " thirds of the replicates flag, because k-hat scatters by about 0.1"
        " around the true shape. The"
        " claim on true shapes of 0.36 or less takes the exact Pareto cases,"
        " one narrowed axis and shifts of at most one width, which have shapes"
        " up to 0.36, and does not take narrowing along all eight axes, whose"
        " k-hat reads high. A claim with a tolerance lets a fraction lie within"
        " that tolerance plus three binomial errors."
    )
    return Section(setup, (table,), tuple(claims))


# * Check evidence-error.


@jax.jit
def _evidence_z_scores(log_ratios: Array) -> Array:
    """Return the z-score of ln Z for rows of log ratios, whose true ln Z is 0."""

    def one(row: Array) -> Array:
        evidence = log_evidence_ratio(jnp.zeros_like(row), row)
        return evidence.value / evidence.error

    return jax.vmap(one)(log_ratios)


class ChainRun(NamedTuple):
    """A reduced AR(1) chain, which is a sample of N(0, I) with correlated draws."""

    samples: Array
    n_eff: float


@functools.cache
def chain_runs(seed: int, quick: bool) -> tuple[ChainRun, ...]:
    """Return the reduced chains of every replicate, shared by two checks."""
    context = Context.create(seed, quick=quick)
    case = zoo.ar1_chains(CHAIN_COEFFICIENTS)
    runs = []
    for replicate in range(
        context.count(FULL_CHAIN_REPLICATES, QUICK_CHAIN_REPLICATES)
    ):
        chains = case.sample(
            context.key(ERROR_LABEL, "chains", replicate),
            CHAIN_STEPS,
            walker_count=CHAIN_WALKERS,
        )
        sample_set = reduce_chains(chains).sample_set
        assert sample_set.n_eff is not None
        runs.append(ChainRun(sample_set.samples[0, 0], float(sample_set.n_eff[0, 0])))
    return tuple(runs)


class ChainTarget(NamedTuple):
    """A target density against the chain posterior N(0, I), and a function of it."""

    family: str
    setting: str
    log_ratio: Callable[[Array], Array]
    """Log of target over N(0, I) at samples shaped (N, 2)."""
    function: Callable[[Array], Array]
    """The function whose mean is estimated, at samples shaped (N, 2)."""
    function_name: str
    parameter: float
    """The shift δ or the width ratio r of the target."""
    function_variance: float
    """The variance of the function under the target, in closed form."""
    kish_factor: float
    """For independent draws, the Eq. 6 prediction over the Kish prediction."""


def _shifted_kish_factor(shift: float) -> float:
    """Return 1 + δ², E_p[w (x − δ)²] over E_p[w] for the target p = N(δ, 1).

    The chain density is N(0, 1) and the variance of x under p is 1.
    """
    return 1 + shift**2


def _narrower_kish_factor(ratio: float) -> float:
    """Return E_p[w (x² − r²)²] over Var_p(x²) E_p[w] for p = N(0, r²).

    The density of p times w is proportional to N(0, s²) with s² = r²/(2 − r²), so
    the factor is (2 a² + (a − 1)²)/2 for a = s²/r² = 1/(2 − r²).
    """
    ratio_of_variances = 1 / (2 - ratio**2)
    return (2 * ratio_of_variances**2 + (ratio_of_variances - 1) ** 2) / 2


def _narrower_target_log_ratio(ratio: float, samples: Array) -> Array:
    squared = jnp.sum(samples**2, axis=-1)
    return squared / 2 - squared / (2 * ratio**2) - 2 * math.log(ratio)


def _shifted_target_log_ratio(shift: float, samples: Array) -> Array:
    return shift * samples[..., 0] - shift**2 / 2


def _first_square(samples: Array) -> Array:
    return samples[..., 0] ** 2


def _first_coordinate(samples: Array) -> Array:
    return samples[..., 0]


CHAIN_TARGETS = (
    *(
        ChainTarget(
            "shifted",
            _setting("δ", shift),
            functools.partial(_shifted_target_log_ratio, shift),
            _first_coordinate,
            "x",
            shift,
            1.0,
            _shifted_kish_factor(shift),
        )
        for shift in (0.25, 0.5, 1.0)
    ),
    *(
        ChainTarget(
            "narrower on two axes",
            _setting("r", ratio),
            functools.partial(_narrower_target_log_ratio, ratio),
            _first_square,
            "x squared",
            ratio,
            2 * ratio**4,
            _narrower_kish_factor(ratio),
        )
        for ratio in (0.8, 0.6)
    ),
)
"""Targets of the chains: shifts N(δ, 1) of the mean of x, whose variance is 1, and
narrower Gaussians N(0, r²) of the mean of x squared, whose variance is 2 r⁴."""


class ChainStatistics(NamedTuple):
    """Per replicate: the z-score of ln Z with and without R, and a mean estimate."""

    z_with_efficiency: Array
    z_without_efficiency: Array
    estimates: Array
    eq5_variances: Array
    """Squared error of each estimate for independent draws (Eq. 5)."""
    efficiencies: Array
    kish_sizes: Array
    """Kish size of the new weights of each replicate."""
    prediction_ratios: Array
    """Eq. 6 over the Kish prediction of each replicate, in which R cancels."""


def _chain_statistics(runs: Sequence[ChainRun], target: ChainTarget) -> ChainStatistics:
    z_with, z_without, estimates, variances = [], [], [], []
    efficiencies, kish_sizes, prediction_ratios = [], [], []
    for run in runs:
        count = run.samples.shape[0]
        efficiency = run.n_eff / count
        log_weights = target.log_ratio(run.samples)
        zeros = jnp.zeros(count)
        with_efficiency = log_evidence_ratio(zeros, log_weights, efficiency)
        without = log_evidence_ratio(zeros, log_weights)
        z_with.append(with_efficiency.value / with_efficiency.error)
        z_without.append(without.value / without.error)
        weights = jax.nn.softmax(log_weights)
        values = target.function(run.samples)
        estimate = jnp.sum(weights * values)
        estimates.append(estimate)
        variances.append(jnp.sum(weights**2 * (values - estimate) ** 2))
        efficiencies.append(efficiency)
        kish_size = kish_effective_sample_size(log_weights)
        kish_sizes.append(kish_size)
        # * Eq. 6 over the Kish prediction: R cancels, and Σ v² is 1/Kish.
        prediction_ratios.append(variances[-1] * kish_size / target.function_variance)
    return ChainStatistics(
        jnp.stack(z_with),
        jnp.stack(z_without),
        jnp.stack(estimates),
        jnp.stack(variances),
        jnp.asarray(efficiencies),
        jnp.stack(kish_sizes),
        jnp.stack(prediction_ratios),
    )


def _evidence_error(context: Context) -> Section:
    replicates = context.count(FULL_Z_REPLICATES, QUICK_REPLICATES)
    exact_rows: list[list[Cell]] = []
    exact_spreads: dict[str, Measurement] = {}
    exact_means: dict[str, Measurement] = {}
    for case in _exact_cases():
        log_ratios = case.log_ratios(
            _case_key(context, case), replicates, LARGEST_DRAWS
        )
        z_scores = _evidence_z_scores(log_ratios)
        exact_means[case.name] = _mean(z_scores)
        exact_spreads[case.name] = _spread(z_scores)
        exact_rows.append(
            [
                case.family,
                case.setting,
                case.truth,
                exact_means[case.name],
                exact_spreads[case.name],
            ]
        )
    exact_table = Table.of(
        f"The z-score (ln Z-hat − ln Z)/error of the log evidence ratio of exactly"
        f" Pareto weights at S = {grouped(LARGEST_DRAWS)}, whose true ln Z is 0:"
        " its mean and its standard deviation over replicates, which is 1 for a"
        " calibrated error.",
        ("Weights", "Setting", "True k", "Mean z", "Spread of z"),
        exact_rows,
    )
    calibrated = [case for case in _exact_cases() if case.truth <= 0.51 + 1e-9]
    heavy = [case for case in _exact_cases() if case.truth >= 0.64 - 1e-9]
    runs = chain_runs(context.seed, context.quick)
    chain_rows: list[list[Cell]] = []
    pools: dict[str, dict[str, list[Array]]] = {}
    spreads: dict[tuple[str, str], Measurement] = {}
    for target in CHAIN_TARGETS:
        statistics = _chain_statistics(runs, target)
        for label, z_scores in (
            ("R from n_eff", statistics.z_with_efficiency),
            ("R = 1", statistics.z_without_efficiency),
        ):
            spread = _spread(z_scores)
            spreads[(f"{target.family}, {target.setting}", label)] = spread
            pools.setdefault(target.family, {}).setdefault(label, []).append(z_scores)
            chain_rows.append(
                [
                    target.family,
                    target.setting,
                    label,
                    _mean(z_scores),
                    spread,
                ]
            )
    pooled: dict[tuple[str, str], Measurement] = {}
    for family, by_label in pools.items():
        for label, stacks in by_label.items():
            pooled[(family, label)] = _spread(jnp.concatenate(stacks))
            chain_rows.append([family, "pooled", label, None, pooled[(family, label)]])
    chain_table = Table.of(
        "The z-score of the log evidence ratio of a target against the chain"
        " posterior N(0, I), whose true ln Z is 0, on AR(1) chains reduced"
        f" with {role('func', 'mixed_signals.mcmc.reduce.reduce_chains')}: the"
        " mean and the spread of z over replicates, with the relative"
        " efficiency R from the n_eff of each set and with R = 1.",
        ("Target", "Setting", "Relative efficiency", "Mean z", "Spread of z"),
        chain_rows,
    )
    spread_tolerance_cases = [
        Case(
            case.name,
            exact_spreads[case.name].value - 1.0,
            exact_spreads[case.name].margin(),
            SPREAD_TOLERANCE,
        )
        for case in calibrated
    ]
    heavy_cases = [
        Case(
            case.name,
            exact_spreads[case.name].value - 1.0,
            exact_spreads[case.name].margin(),
            0.0,
        )
        for case in heavy
    ]
    negative_cases = [
        Case(
            f"{case.name}, minus the mean of z",
            -exact_means[case.name].value,
            exact_means[case.name].margin(),
            0.0,
        )
        for case in heavy
    ]
    shifted_with = pooled[("shifted", "R from n_eff")]
    shifted_without = pooled[("shifted", "R = 1")]
    narrower_with = pooled[("narrower on two axes", "R from n_eff")]
    claims = [
        claim_cases(
            "The error of the log evidence ratio is calibrated for a true shape up"
            " to 0.51: the spread of z is 1.",
            spread_tolerance_cases,
            tolerance_text=f"{SPREAD_TOLERANCE:g}",
            replicates=replicates,
        ),
        claim_cases_above(
            "From a true shape of 0.64, the error is too small: the spread of z"
            " lies above 1.",
            heavy_cases,
            tolerance_text=None,
            replicates=replicates,
            reference_text="1",
        ),
        claim_cases_above(
            "From a true shape of 0.64, the log evidence ratio is biased low:"
            " the mean of z is negative.",
            negative_cases,
            tolerance_text=None,
            replicates=replicates,
            reference_text="0",
        ),
        claim_cases(
            "On chains with shifted targets, whose estimates are linear in a"
            " parameter, the spread of z pooled over the shifts is 1 with the"
            " relative efficiency from n_eff.",
            [
                Case(
                    "pooled shifts",
                    shifted_with.value - 1.0,
                    shifted_with.margin(),
                    SPREAD_TOLERANCE,
                )
            ],
            tolerance_text=f"{SPREAD_TOLERANCE:g}",
            replicates=int(shifted_with.replicates or 0),
        ),
        claim_cases_above(
            "On the same chains without the relative efficiency, the pooled"
            " spread of z lies above 1.",
            [
                Case(
                    "pooled shifts",
                    shifted_without.value - 1.0,
                    shifted_without.margin(),
                    0.0,
                )
            ],
            tolerance_text=None,
            replicates=int(shifted_without.replicates or 0),
            reference_text="1",
        ),
        claim_true(
            "With narrower targets, whose estimates are functions of a square with"
            " a shorter autocorrelation time, the relative efficiency from n_eff"
            " overcorrects: the pooled spread of z is at most 1.",
            "a spread of at most 1 plus three standard errors",
            narrower_with.render(),
            narrower_with.value <= 1.0 + narrower_with.margin(),
        ),
    ]
    setup = (
        f"Two sets of replicates. First, {replicates} replicates of the exact"
        f" Pareto cases of the shape checks at S = {grouped(LARGEST_DRAWS)},"
        f" whose weights come from a proposal q that is normalised with a"
        f" normalised target, so that the true ln Z is 0, and"
        f" {EVIDENCE_RATIO} gives ln Z-hat and its error, from the delta method"
        " and Eq. 5 of {cite:t}`vehtari2024pareto`. Second, "
        f"{len(runs)} replicates of AR(1) chains"
        f" ({CHAIN_WALKERS} walkers, {grouped(CHAIN_STEPS)} steps, 100"
        " autocorrelation times, with coefficients 0.9) reduced by thinning at"
        " the autocorrelation time, which leaves about 3 600 samples with"
        " n_eff/N about 0.72. Targets are N(δ, I) shifted by δ in the first"
        " coordinate, and N(0, r² I) narrower on both coordinates, so that the"
        " weights of the chain are bounded or lognormal and the error is the"
        " variance of a weighted mean (Eq. 6 of the paper divides it by the"
        " relative efficiency R of correlated draws). The z-scores come with R"
        " = n_eff over the Kish value of the input weights, as"
        " the engine takes it, and with R = 1. A claim on a spread lets it lie"
        " within the stated tolerance plus three standard errors (the standard"
        " error of a standard deviation is the spread over the root of twice"
        " the replicates less one). Calibration holds up to a true shape of 0.5,"
        " where the weights have a finite variance, and fails above it."
    )
    return Section(setup, (exact_table, chain_table), tuple(claims))


# * Check reweighted-n-eff.


def _reweighted_variance(context: Context) -> Section:
    runs = chain_runs(context.seed, context.quick)
    replicates = len(runs)
    relative_error = math.sqrt(2 / (replicates - 1))
    rows: list[list[Cell]] = []
    with_cases: list[tuple[str, Measurement]] = []
    without_cases: list[tuple[str, Measurement]] = []
    kish_ratios: dict[str, Measurement] = {}
    closed_form_cases: list[Case] = []
    for target in CHAIN_TARGETS:
        statistics = _chain_statistics(runs, target)
        observed = float(jnp.var(statistics.estimates, ddof=1))
        eq5 = float(jnp.mean(statistics.eq5_variances))
        eq6 = float(jnp.mean(statistics.eq5_variances / statistics.efficiencies))
        kish = float(
            jnp.mean(
                target.function_variance
                / (statistics.efficiencies * statistics.kish_sizes)
            )
        )
        ratio_eq5, ratio_eq6, ratio_kish = (
            Measurement(
                observed / prediction,
                observed / prediction * relative_error,
                replicates,
            )
            for prediction in (eq5, eq6, kish)
        )
        prediction_ratio = _mean(statistics.prediction_ratios)
        name = f"{target.family}, {target.setting}"
        with_cases.append((name, ratio_eq6))
        without_cases.append((name, ratio_eq5))
        kish_ratios[name] = ratio_kish
        closed_form_cases.append(
            Case(
                name,
                prediction_ratio.value / target.kish_factor - 1.0,
                prediction_ratio.margin() / target.kish_factor,
                RATIO_TOLERANCE,
            )
        )
        rows.append(
            [
                target.family,
                target.setting,
                target.function_name,
                observed,
                eq5,
                eq6,
                kish,
                ratio_eq5,
                ratio_eq6,
                ratio_kish,
                prediction_ratio,
                target.kish_factor,
            ]
        )
    table = Table.of(
        "The variance over replicates of the self-normalised mean of a function on"
        " reweighted chains, against the mean prediction of Eq. 5 of"
        " {cite:t}`vehtari2024pareto` for independent draws, of Eq. 6 with the"
        " relative efficiency R, and of the Kish size, the variance of the"
        " function under the target over R times the Kish size of the new"
        " weights, and the ratio of the observed variance to each; then the"
        " Eq. 6 prediction over the Kish prediction, averaged over replicates,"
        " and its closed form for independent draws.",
        (
            "Target",
            "Setting",
            "Function",
            "Observed variance",
            "Eq. 5",
            "Eq. 6 with R",
            "Kish prediction",
            "Observed over Eq. 5",
            "Observed over Eq. 6",
            "Observed over the Kish prediction",
            "Eq. 6 over the Kish prediction",
            "Closed form of that ratio",
        ),
        rows,
    )
    allowed_above = [
        Case(name, ratio.value - 1.0, ratio.margin(), 0.0) for name, ratio in with_cases
    ]
    allowed_below = [
        Case(name, 1.0 - ratio.value, ratio.margin(), 0.0)
        for name, ratio in without_cases
    ]
    unit_shift = kish_ratios[f"shifted, {_setting('δ', 1.0)}"]
    narrower_ratios = [
        (name, ratio)
        for name, ratio in kish_ratios.items()
        if name.startswith("narrower")
    ]
    claims = [
        claim_true(
            "With the relative efficiency R, the variance of Eq. 6 is not smaller"
            " than the observed one.",
            "an observed variance of at most the prediction plus three standard errors",
            _worst(with_cases, highest=True),
            all(case.deviation <= case.error_part for case in allowed_above),
        ),
        claim_true(
            "Without R, the variance of Eq. 5 is not larger than the observed one.",
            "an observed variance of at least the prediction less three standard"
            " errors",
            _worst(without_cases, highest=False),
            all(case.deviation <= case.error_part for case in allowed_below),
        ),
        claim_cases(
            "The Eq. 6 prediction over the Kish prediction, in which R cancels,"
            " takes its closed form for independent draws, so the chain"
            " correlation does not confound it.",
            closed_form_cases,
            tolerance_text=f"{RATIO_TOLERANCE:g} of the closed form",
            replicates=replicates,
        ),
        claim_cases_above(
            "At a shift of one width, the observed variance exceeds the Kish"
            " prediction.",
            [Case("shift 1", unit_shift.value - 1.0, unit_shift.margin(), 0.0)],
            tolerance_text=None,
            replicates=replicates,
            reference_text="the prediction",
        ),
        claim_cases_above(
            "For the narrower targets, the observed variance lies below the Kish"
            " prediction: the shortfall is positive.",
            [
                Case(name, 1.0 - ratio.value, ratio.margin(), 0.0)
                for name, ratio in narrower_ratios
            ],
            tolerance_text=None,
            replicates=replicates,
            reference_text="zero",
        ),
    ]
    setup = (
        f"The {replicates} replicates of the chains of the check on the error of"
        " the log evidence ratio, reweighted to each target. A replicate gives"
        " the self-normalised mean of a function h, x squared for the narrower"
        " targets and x for the shifted ones, and the prediction Σ vᵢ² (hᵢ − mean)²"
        " of Eq. 5 for the normalised weights vᵢ; Eq. 6 divides it by R, the"
        " n_eff of the set over its Kish value. The Kish prediction is the"
        " variance of h under the target, 1 for a shifted target N(δ, 1) and"
        " 2 r⁴ for a narrower target N(0, r²), over R times the Kish size of the"
        " new weights, which is the variance that the size of the"
        " `importance_ess` row implies. The variance of the mean over"
        " replicates is the truth. The standard error of a variance ratio is the"
        " ratio times the root of two over the replicates less one. The first two"
        " claims are that Eq. 6 does not underestimate the variance and that"
        " Eq. 5 without R does not overestimate it, each within three standard"
        " errors of that side. The Kish size is the same for every function, so"
        " it misstates the variance of a reweighted mean in either direction."
        " Per replicate, the Eq. 6 prediction over the Kish prediction is"
        " Σ vᵢ² (hᵢ − mean)² over Var_p(h) Σ vᵢ², in which R cancels, so it is a"
        " property of the weights and of h on the same draws. For independent"
        " draws it tends to E_p[w (h − μ)²]/(Var_p(h) E_p[w]), for the target p,"
        " the chain density q and the weight w = p/q. For a shifted target and"
        " h = x this is 1 + δ², because E_p[w (x − δ)²] = exp(δ²) (1 + δ²) and"
        " E_p[w] = exp(δ²): the variance of the reweighted mean is 1 + δ² times"
        " larger than the Kish size implies. For a narrower target and h = x²,"
        " p w is proportional to N(0, s²) with s² = r²/(2 − r²), so the ratio is"
        " (2 a² + (a − 1)²)/2 for a = 1/(2 − r²), 0.5757 at r = 0.8 and 0.4480"
        " at r = 0.6: the variance is about half of what the Kish size implies,"
        " and the Kish prediction about twice too large. The observed column"
        " carries the effect of R on the chains on top of this factor. The third"
        " claim lets the mean ratio of every case lie within"
        f" {RATIO_TOLERANCE:g} of its closed form, relative to it, plus three"
        " standard errors. The fourth claim wants the observed variance at a"
        " shift of one width to exceed the Kish prediction by more than three"
        " standard errors, and the fifth that the observed variance of the"
        " narrower targets lie below it by more than three standard errors."
    )
    return Section(setup, (table,), tuple(claims))


def _worst(entries: Sequence[tuple[str, Measurement]], *, highest: bool) -> str:
    """Name the entry with the largest or the smallest ratio."""
    name, measurement = (max if highest else min)(
        entries, key=lambda entry: entry[1].value
    )
    word = "highest" if highest else "lowest"
    return f"{word} of {len(entries)}: {name}: {measurement.render()}"


# * Check reweighting-recovery.

RECOVERY_SAMPLES = 5_000
RECOVERY_BIAS_TOLERANCE = 0.0005
"""Allowance for the bias of a self-normalised estimate, about 1/n_eff = 0.0002."""
RECOVERY_SPREAD_TOLERANCE = 0.15
LIKELIHOOD_OFFSETS = (2.0, -1.0)
"""Constants added to the stored and to the new log-likelihood, which move ln Z."""
LIKELIHOOD_SHIFT = (0.5, 0.0, 0.0)
CONJUGATE_MEAN = (0.4, -0.2)
CONJUGATE_WIDTH = 0.3
OLD_PRIOR_VARIANCE = 1.0
NEW_PRIOR_VARIANCE = 2.0


def _gaussian_log_density(points: Array, mean: Array, covariance: Array) -> Array:
    """Return the log density of a Gaussian at points shaped (N, D), in JAX."""
    offsets = points - mean
    squared = jnp.einsum("ni,ij,nj->n", offsets, jnp.linalg.inv(covariance), offsets)
    log_determinant = jnp.linalg.slogdet(covariance)[1]
    density: Array = -0.5 * (
        squared + log_determinant + mean.shape[0] * math.log(2 * math.pi)
    )
    return density


class GaussianPrior:
    """A zero-mean Gaussian prior N(0, variance I) with the two methods of a prior."""

    def __init__(self, variance: float, param_count: int) -> None:
        """Store the variance and the number of parameters."""
        self.variance = variance
        self.param_count = param_count

    def logpdf(self, samples: ArrayLike) -> Array:
        """Return the log density at samples shaped (N, D)."""
        return _gaussian_log_density(
            jnp.asarray(samples),
            jnp.zeros(self.param_count),
            self.variance * jnp.eye(self.param_count),
        )

    def rvs(self, size: int | tuple[int, ...], *, key: PRNGKeyArray) -> Array:
        """Draw from the prior with a JAX key."""
        shape = (size,) if isinstance(size, int) else size
        return math.sqrt(self.variance) * jax.random.normal(
            key, (*shape, self.param_count)
        )


class Recovery(NamedTuple):
    """Per replicate: the first reweighted mean and width and the log evidence ratio."""

    means: Array
    widths: Array
    log_evidence: Array
    errors: Array


class RecoveryTruth(NamedTuple):
    """What a reweighting should recover, for the first parameter."""

    mean: float
    width: float
    log_evidence: float


def _recovered(outcome: Reweighting) -> Recovery:
    """Read the moments of the reweighted samples and the evidence rows."""
    mean, covariance = weighted_covariance(
        outcome.sample_set.samples[:, 0], outcome.sample_set.log_weights[:, 0]
    )
    evidence = outcome.result.table.filter(pl.col("metric") == "log_evidence_ratio")
    return Recovery(
        mean[:, 0],
        jnp.sqrt(covariance[:, 0, 0]),
        jnp.asarray(evidence["value"].to_list()),
        jnp.asarray(evidence["uncertainty"].to_list()),
    )


def _wider_approximation(context: Context, replicates: int) -> Recovery:
    case = zoo.scaled_gaussians(1.3, param_count=2)
    draws = case.sample(
        context.key(RECOVERY_LABEL, "approximation"),
        RECOVERY_SAMPLES,
        batch_size=replicates,
    )
    scaled, reference = case.names.index("scaled"), case.names.index("reference")
    points = draws.samples[:, scaled : scaled + 1]
    log_density = case.log_prob(points)
    sample_set = SampleSet(
        points,
        log_like=log_density[..., reference],
        log_prior=jnp.zeros(points.shape[:-1]),
        log_q=log_density[..., scaled],
    )
    return _recovered(
        reweight(sample_set, ["q"], case.space, proposal=Proposal.APPROXIMATION)
    )


def _shifted_likelihood(context: Context, replicates: int) -> Recovery:
    case = zoo.shifted_gaussians(LIKELIHOOD_SHIFT[0], param_count=3)
    draws = case.sample(
        context.key(RECOVERY_LABEL, "likelihood"),
        RECOVERY_SAMPLES,
        batch_size=replicates,
    )
    reference = case.names.index("reference")
    points = draws.samples[:, reference : reference + 1]
    stored_offset, new_offset = LIKELIHOOD_OFFSETS
    stored = case.log_prob(points)[..., reference] + stored_offset
    shifted_mean = jnp.asarray(LIKELIHOOD_SHIFT)

    def log_like(samples: Array) -> Array:
        return _gaussian_log_density(samples, shifted_mean, jnp.eye(3)) + new_offset

    # * Independent problems have their own data, so every problem gets a callable.
    return _recovered(
        reweight(
            SampleSet(points, log_like=stored),
            ["reference"],
            case.space,
            log_like=[log_like] * replicates,
            backend="jax",
        )
    )


def _conjugate_posterior(prior_variance: float) -> tuple[Array, Array]:
    """Return the posterior mean and covariance of a Gaussian likelihood and prior."""
    likelihood_precision = jnp.eye(2) / CONJUGATE_WIDTH**2
    covariance = jnp.linalg.inv(likelihood_precision + jnp.eye(2) / prior_variance)
    return covariance @ likelihood_precision @ jnp.asarray(CONJUGATE_MEAN), covariance


def _conjugate_log_evidence_ratio() -> float:
    """Return ln N(m; 0, C + B) − ln N(m; 0, C + A), the true ln Z_new/Z_old."""
    data = jnp.asarray(CONJUGATE_MEAN)[None]

    def log_evidence(prior_variance: float) -> Array:
        covariance = (CONJUGATE_WIDTH**2 + prior_variance) * jnp.eye(2)
        return _gaussian_log_density(data, jnp.zeros(2), covariance)[0]

    return float(log_evidence(NEW_PRIOR_VARIANCE) - log_evidence(OLD_PRIOR_VARIANCE))


def _conjugate_prior(context: Context, replicates: int) -> Recovery:
    old_mean, old_covariance = _conjugate_posterior(OLD_PRIOR_VARIANCE)
    case = zoo.gaussians(old_mean[None], old_covariance[None], names=["old"])
    draws = case.sample(
        context.key(RECOVERY_LABEL, "prior"), RECOVERY_SAMPLES, batch_size=replicates
    )
    likelihood_covariance = CONJUGATE_WIDTH**2 * jnp.eye(2)
    likelihood_mean = jnp.asarray(CONJUGATE_MEAN)

    def stored_prior(block: Array) -> Array:
        return _gaussian_log_density(
            block[0], jnp.zeros(2), OLD_PRIOR_VARIANCE * jnp.eye(2)
        )

    def stored_like(block: Array) -> Array:
        return _gaussian_log_density(block[0], likelihood_mean, likelihood_covariance)

    sample_set = SampleSet(
        draws.samples,
        log_like=jax.vmap(stored_like)(draws.samples)[:, None],
        log_prior=jax.vmap(stored_prior)(draws.samples)[:, None],
    )
    return _recovered(
        reweight(
            sample_set,
            ["old"],
            case.space,
            prior=GaussianPrior(NEW_PRIOR_VARIANCE, 2),
            backend="jax",
        )
    )


class RecoveryScenario(NamedTuple):
    """One reweighting through the engine with its known target."""

    name: str
    run: Callable[[Context, int], Recovery]
    truth: RecoveryTruth


def _recovery_scenarios() -> tuple[RecoveryScenario, ...]:
    new_mean, new_covariance = _conjugate_posterior(NEW_PRIOR_VARIANCE)
    return (
        RecoveryScenario(
            "approximation from a wider q",
            _wider_approximation,
            RecoveryTruth(0.0, 1.0, 0.0),
        ),
        RecoveryScenario(
            "likelihood between shifted Gaussians",
            _shifted_likelihood,
            RecoveryTruth(
                LIKELIHOOD_SHIFT[0], 1.0, LIKELIHOOD_OFFSETS[1] - LIKELIHOOD_OFFSETS[0]
            ),
        ),
        RecoveryScenario(
            "conjugate prior",
            _conjugate_prior,
            RecoveryTruth(
                float(new_mean[0]),
                math.sqrt(float(new_covariance[0, 0])),
                _conjugate_log_evidence_ratio(),
            ),
        ),
    )


def _recovery(context: Context) -> Section:
    replicates = context.count(100, 10)
    rows: list[list[Cell]] = []
    cases: list[Case] = []
    spread_cases: list[Case] = []
    for scenario in _recovery_scenarios():
        recovery = scenario.run(context, replicates)
        truth = scenario.truth
        quantities = (
            ("mean of the first parameter", recovery.means, truth.mean),
            ("width of the first parameter", recovery.widths, truth.width),
            ("log evidence ratio", recovery.log_evidence, truth.log_evidence),
        )
        for quantity, values, expected in quantities:
            measurement = _mean(values)
            rows.append([scenario.name, quantity, expected, measurement])
            cases.append(
                Case(
                    f"{scenario.name}, {quantity}",
                    measurement.value - expected,
                    measurement.margin(),
                    RECOVERY_BIAS_TOLERANCE,
                )
            )
        spread = _spread((recovery.log_evidence - truth.log_evidence) / recovery.errors)
        rows.append([scenario.name, "spread of the z-score of ln Z", 1.0, spread])
        spread_cases.append(
            Case(
                scenario.name,
                spread.value - 1.0,
                spread.margin(),
                RECOVERY_SPREAD_TOLERANCE,
            )
        )
    table = Table.of(
        "The mean over replicates of the reweighted mean and width of the first"
        " parameter, of the log evidence ratio and of the spread of its z-score,"
        " from the engine, against the truth.",
        ("Reweighting", "Quantity", "Truth", "Mean over replicates"),
        rows,
    )
    claims = [
        claim_cases(
            "The engine recovers the target: the reweighted mean and width and the"
            " log evidence ratio agree with their truths.",
            cases,
            tolerance_text=f"{RECOVERY_BIAS_TOLERANCE:g}, for the bias of a"
            " self-normalised estimate",
            replicates=replicates,
        ),
        claim_cases(
            "The error of the log evidence ratio is calibrated for these"
            " light-tailed weights: the spread of z is 1.",
            spread_cases,
            tolerance_text=f"{RECOVERY_SPREAD_TOLERANCE:g}",
            replicates=replicates,
        ),
    ]
    stored_offset, new_offset = LIKELIHOOD_OFFSETS
    setup = (
        f"{replicates} replicates of three reweightings through {REWEIGHT}, each"
        f" of {grouped(RECOVERY_SAMPLES)} samples, run as the problems of one"
        " batch. First, draws of the zoo Gaussian that is wider by 1.3 on the"
        f" first axis ({SCALED_GAUSSIANS}) are an approximation q of the"
        " reference, which is the target, so the true ln Z is 0. Second, samples"
        f" of the reference of {SHIFTED_GAUSSIANS} (three parameters, shift"
        f" {LIKELIHOOD_SHIFT[0]:g}) with a stored log-likelihood that is the"
        f" reference density with an offset of {stored_offset:+g} are reweighted"
        f" to a callable that gives the shifted density with an offset of"
        f" {new_offset:+g}, so ln Z changes by {new_offset - stored_offset:g}."
        " Third, a Gaussian likelihood N(θ; m, C)"
        f" with m = {CONJUGATE_MEAN} and C = {CONJUGATE_WIDTH}² I and an old prior"
        " N(0, I) give a Gaussian posterior, which is reweighted to the posterior"
        " of a new prior N(0, 2 I): the evidence ratio is ln N(m; 0, C + 2 I) −"
        " ln N(m; 0, C + I), and the moments are those of the new posterior. The"
        " table gives the first parameter. A claim lets each mean lie within"
        " three standard errors (Student t for few replicates) of its truth, plus"
        f" {RECOVERY_BIAS_TOLERANCE:g} for the bias of a self-normalised estimate,"
        " which is of the order of one over the effective sample size."
    )
    return Section(setup, (table,), tuple(claims))


CHECKS: tuple[Check, ...] = (
    Check(
        label=EXACT_LABEL,
        title="k-hat on exact Pareto weights",
        session="13",
        supports=(
            role("func", "mixed_signals.reweight.psis.pareto_tail", short=False),
            role("func", "mixed_signals.reweight.psis.pareto_tail_length", short=False),
            role(
                "func",
                "mixed_signals.sigma.pareto.generalized_pareto_fit",
                short=False,
            ),
            "the weakly informative prior on the shape, an option of the fit",
        ),
        run=_exact_shapes,
    ),
    Check(
        label=GAUSSIAN_LABEL,
        title="k-hat of narrower and shifted Gaussian proposals",
        session="13",
        supports=(
            role("func", "mixed_signals.reweight.psis.pareto_tail", short=False),
            "the reading of k-hat on Gaussian proposals with several axes",
        ),
        run=_gaussian_shapes,
    ),
    Check(
        label=FLAG_LABEL,
        title="Flag rates at the k-hat threshold",
        session="13",
        supports=(
            role("func", "mixed_signals.reweight.psis.pareto_k_threshold", short=False),
            role("data", "mixed_signals.reweight.psis.PARETO_K_LIMIT", short=False),
            "the done criterion of session 13: k-hat flags heavy-tailed weights",
        ),
        run=_flag_rates,
    ),
    Check(
        label=ERROR_LABEL,
        title="Error of the log evidence ratio",
        session="13",
        supports=(
            role(
                "func",
                "mixed_signals.reweight.importance.log_evidence_ratio",
                short=False,
            ),
            "the delta-method error of the evidence ratio, calibrated up to k = 0.5",
        ),
        run=_evidence_error,
    ),
    Check(
        label=EFFICIENCY_LABEL,
        title="Variance of reweighted means on correlated draws",
        session="13",
        supports=(
            role(
                "func",
                "mixed_signals.reweight.importance.relative_efficiency",
                short=False,
            ),
            role(
                "func",
                "mixed_signals.reweight.importance.reweighted_effective_sample_size",
                short=False,
            ),
            "the informational `importance_ess` row: Kish's size does not depend on"
            " the function, and the variance of a reweighted mean can be larger or"
            " smaller than it implies: (1 + δ²) times larger for the mean of a"
            " shifted target and about half for the square of a narrower one",
        ),
        run=_reweighted_variance,
    ),
    Check(
        label=RECOVERY_LABEL,
        title="Recovery of the target",
        session="13",
        supports=(
            role("func", "mixed_signals.engines.reweight.reweight", short=False),
            "the done criterion of session 13: reweighting recovers the analytic"
            " target",
        ),
        run=_recovery,
    ),
)
