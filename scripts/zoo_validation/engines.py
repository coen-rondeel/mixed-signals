"""Checks of the engines: kNN rows of weighted sets, nulls, errors and recovery."""

import functools
import math
import warnings
from collections.abc import Callable, Generator, Sequence
from contextlib import contextmanager
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import polars as pl
from jaxtyping import Array, PRNGKeyArray

from mixed_signals import zoo
from mixed_signals.core.samples import kish_effective_sample_size, rejection_subset
from mixed_signals.core.space import ParamSpace
from mixed_signals.diagnostics import THRESHOLDS, dependence_statistics
from mixed_signals.engines import compare, diagnose
from mixed_signals.errors import ValidityWarning
from mixed_signals.estimators.gaussian import GaussianDensity
from mixed_signals.metrics.divergence import (
    HELLINGER_NEIGHBOURS,
    JS_NEIGHBOURS,
    KL_NEIGHBOURS,
    knn_hellinger_distance,
    knn_js_divergence,
    knn_kl_divergence,
)
from mixed_signals.metrics.kernel import (
    energy_distance,
    maximum_mean_discrepancy,
    median_bandwidth,
)
from mixed_signals.metrics.marginal import ks_statistic
from mixed_signals.metrics.moments import gaussian_kl, mean_shift, width_ratio
from mixed_signals.metrics.ot.sliced import (
    SLICED_DIRECTIONS,
    sliced_directions,
    sliced_wasserstein_distance,
)
from mixed_signals.nulls import permutation_null
from mixed_signals.nulls.plan import Statistic
from mixed_signals.sigma.uncertainty import half_sample_draws, half_sample_error
from zoo_validation.common import (
    LEVELS,
    Builder,
    compare_on_plan,
    compare_with_constants,
    empirical_p_values,
    first_parameter_statistics,
    rate_counts,
    rate_measurements,
    real_space,
)
from zoo_validation.nulls import PROPOSAL_WIDTH
from zoo_validation.report import (
    Case,
    Cell,
    Check,
    Claim,
    Context,
    Measurement,
    Section,
    Table,
    claim_cases,
    claim_cases_above,
    claim_rates_within,
    claim_true,
    grouped,
    role,
)
from zoo_validation.reweighting import chain_runs

LEVEL_COLUMNS = tuple(f"Rate at {level:.0%}" for level in LEVELS)
"""Headers of the rejection rate columns."""
KNN_LEVELS = (0.05, 0.10)
"""The levels of the claims about the kNN rows."""
INFORMATION_TOLERANCE = 0.005
"""Nats that a mean excess information may lie from zero: a tenth of the flag threshold.

Equal-weight draws of independent axes read the excess about 0.0015 nats below
zero at about 400 samples in four dimensions, near the mean 1/(2N) of the
Gaussian value −½ ln(1 − r²) that the excess subtracts, so a claim of zero needs
a tolerance."""

REJECTION_SUBSET = role("func", "mixed_signals.core.samples.rejection_subset")
COMPARE = role("func", "mixed_signals.engines.compare.compare")
DIAGNOSE = role("func", "mixed_signals.engines.diagnose.diagnose")
PERMUTATION_NULL = role("func", "mixed_signals.nulls.permutation.permutation_null")
SHIFTED_GAUSSIANS = role("func", "mixed_signals.zoo.cases.shifted_gaussians")
SCALED_GAUSSIANS = role("func", "mixed_signals.zoo.cases.scaled_gaussians")
R_EFF = role("func", "mixed_signals.sigma.maps.r_eff")


# * Cases shared by the checks of weighted sets.


@functools.cache
def _proposal_case(dimension: int) -> zoo.ZooCase:
    """Return the target N(0, I) and a proposal PROPOSAL_WIDTH times wider."""
    identity = jnp.eye(dimension)
    return zoo.gaussians(
        jnp.zeros((2, dimension)),
        jnp.stack([identity, PROPOSAL_WIDTH**2 * identity]),
        names=("target", "proposal"),
    )


class _WeightedDraws(NamedTuple):
    """Replicates of proposal draws with importance weights, and target draws."""

    proposal: Array
    """Proposal samples shaped (replicates, count, D)."""
    log_weights: Array
    """Log of target over proposal at the samples, shaped (replicates, count)."""
    target: Array
    """Equal-weight target draws shaped (replicates, count, D)."""


def _weighted_draws(
    context: Context, label: str, dimension: int, count: int, replicates: int
) -> _WeightedDraws:
    """Draw ``count`` samples of the proposal and of the target per replicate."""
    case = _proposal_case(dimension)
    draws = case.sample(
        context.key(label, "draws", dimension), count, batch_size=replicates
    ).samples
    proposal = draws[:, 1]
    log_densities = case.log_prob(proposal)
    return _WeightedDraws(
        proposal, log_densities[..., 0] - log_densities[..., 1], draws[:, 0]
    )


def _plural(count: int, noun: str) -> str:
    """Write a count with its noun, in the plural unless the count is one."""
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"


def _value_text(values: Sequence[float]) -> str:
    """Write numbers as ``0.5 or 1``."""
    names = [f"{value:g}" for value in values]
    return f"{', '.join(names[:-1])} or {names[-1]}" if len(names) > 1 else names[0]


def _dimension_text(dimensions: Sequence[int]) -> str:
    """Write dimensions as ``2, 4 and 8``."""
    names = [str(dimension) for dimension in dimensions]
    return f"{', '.join(names[:-1])} and {names[-1]}" if len(names) > 1 else names[0]


@contextmanager
def _expected_zero_distances() -> Generator[None]:
    """Hide the zero-distance warnings of copies, whose NaN the check counts."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ValidityWarning)
        yield


# * Check knn-weights.

KNN_LABEL = "knn-weights"
KNN_FUNCTIONS: dict[str, Callable[..., Array]] = {
    "kl": knn_kl_divergence,
    "hellinger": knn_hellinger_distance,
    "js": knn_js_divergence,
}
KNN_RANKS = {
    "kl": KL_NEIGHBOURS,
    "hellinger": HELLINGER_NEIGHBOURS,
    "js": JS_NEIGHBOURS,
}
KNN_NAMES = {"kl": "KL", "hellinger": "Hellinger", "js": "JS"}
COPY_DIMENSIONS = (2, 4, 8)
COPY_WEIGHTED, COPY_TARGET = 4_000, 2_000
"""Proposal draws that are resampled, and target draws that they meet."""
INFORMATION_DIMENSIONS = (2, 4)
INFORMATION_COUNT = 2_000
NULL_DIMENSIONS = (2, 4)
NULL_WEIGHTED, NULL_TARGET = 1_000, 500
CUT_QUANTUM = 25
"""The cut count is rounded down to this multiple, to reuse the null program."""
NULL_DRAWS = 99


@functools.cache
def _knn_builder(metric: str, dimension: int) -> Builder:
    """Return the builder of the kNN statistic of a metric at its default rank."""
    function = KNN_FUNCTIONS[metric]
    space = real_space(dimension)

    def build(key: PRNGKeyArray) -> Statistic:
        def statistic(
            first: Array, first_weights: Array, second: Array, second_weights: Array
        ) -> Array:
            return function(
                first,
                first_weights,
                second,
                second_weights,
                space,
                key=key,
                neighbours=KNN_RANKS[metric],
            )

        return statistic

    return build


class _Copies(NamedTuple):
    """Per dimension: the copies after resampling and the NaN of the kNN estimates."""

    distinct: Array
    """Share of the resampled samples that are distinct, per replicate."""
    over_rank: dict[str, Array]
    """Share of the distinct samples with more copies than the rank, per metric."""
    not_a_number: dict[str, Array]
    """Whether the estimate of each metric is NaN, per replicate."""
    kish: Array
    kept: Array
    """The Kish size of the weights and the samples that rejection keeps."""


def _copies(context: Context, dimension: int, replicates: int) -> _Copies:
    """Resample weighted sets with replacement and read the copies and the kNN rows."""
    space = real_space(dimension)
    draws = _weighted_draws(context, KNN_LABEL, dimension, COPY_WEIGHTED, replicates)
    distinct, kish, kept = [], [], []
    over_rank: dict[str, list[float]] = {metric: [] for metric in KNN_FUNCTIONS}
    not_a_number: dict[str, list[bool]] = {metric: [] for metric in KNN_FUNCTIONS}
    for replicate in range(replicates):
        log_weights = draws.log_weights[replicate]
        indices = jax.random.choice(
            context.key(KNN_LABEL, "resample", dimension, replicate),
            COPY_WEIGHTED,
            shape=(COPY_WEIGHTED,),
            p=jax.nn.softmax(log_weights),
        )
        copies = jnp.bincount(indices, length=COPY_WEIGHTED)
        present = jnp.sum(copies > 0)
        distinct.append(float(present) / COPY_WEIGHTED)
        kish.append(float(kish_effective_sample_size(log_weights)))
        kept.append(float(jnp.sum(jnp.exp(log_weights - jnp.max(log_weights)))))
        resampled = draws.proposal[replicate][indices]
        target = draws.target[replicate, :COPY_TARGET]
        for metric, function in KNN_FUNCTIONS.items():
            # * A sample with more than k copies has a zero k-th neighbour distance.
            over_rank[metric].append(
                float(jnp.sum(copies > KNN_RANKS[metric]) / present)
            )
            with _expected_zero_distances():
                value = function(
                    resampled,
                    None,
                    target,
                    None,
                    space,
                    key=context.key(KNN_LABEL, "subsample", dimension, replicate),
                    neighbours=KNN_RANKS[metric],
                )
            not_a_number[metric].append(bool(jnp.isnan(value)))
    return _Copies(
        jnp.asarray(distinct),
        {metric: jnp.asarray(values) for metric, values in over_rank.items()},
        {metric: jnp.asarray(values) for metric, values in not_a_number.items()},
        jnp.asarray(kish),
        jnp.asarray(kept),
    )


class _Information(NamedTuple):
    """The mutual information of independent axes, per replicate and kind of set."""

    resampled: Array
    rejection: Array
    control: Array
    """Equal-weight draws of the target, as many as rejection kept."""
    kept: Array
    """The samples that rejection kept, per replicate."""


def _mean_excess_information(
    samples: Array, space: ParamSpace, key: PRNGKeyArray
) -> float:
    """Return the mean over the pairs of the KSG excess over the Gaussian value."""
    result = dependence_statistics(samples, None, space, key=key)
    return float(jnp.mean(result.mutual_information))


def _information(context: Context, dimension: int, replicates: int) -> _Information:
    """Read the excess mutual information of resampled, rejected and control sets."""
    space = real_space(dimension)
    draws = _weighted_draws(
        context, KNN_LABEL, dimension, INFORMATION_COUNT, replicates
    )
    values: dict[str, list[float]] = {"resampled": [], "rejection": [], "control": []}
    kept_counts = []
    for replicate in range(replicates):
        log_weights = draws.log_weights[replicate]
        proposal = draws.proposal[replicate]
        indices = jax.random.choice(
            context.key(KNN_LABEL, "information-resample", dimension, replicate),
            INFORMATION_COUNT,
            shape=(INFORMATION_COUNT,),
            p=jax.nn.softmax(log_weights),
        )
        kept = rejection_subset(
            proposal,
            log_weights,
            key=context.key(KNN_LABEL, "information-rejection", dimension, replicate),
        )
        control = (
            _proposal_case(dimension)
            .sample(
                context.key(KNN_LABEL, "control", dimension, replicate), kept.shape[0]
            )
            .samples[0, 0]
        )
        kept_counts.append(kept.shape[0])
        key = context.key(KNN_LABEL, "information", dimension, replicate)
        for kind, samples in (
            ("resampled", proposal[indices]),
            ("rejection", kept),
            ("control", control),
        ):
            values[kind].append(_mean_excess_information(samples, space, key))
    return _Information(
        jnp.asarray(values["resampled"]),
        jnp.asarray(values["rejection"]),
        jnp.asarray(values["control"]),
        jnp.asarray(kept_counts, dtype=jnp.float64),
    )


def _cut(key: PRNGKeyArray, samples: Array, count: int) -> Array:
    """Keep ``count`` samples chosen uniformly, in random order."""
    return samples[jax.random.permutation(key, samples.shape[0])[:count]]


def _null_p_values(
    context: Context, dimension: int, replicates: int, draws: int
) -> tuple[dict[str, Array], Array]:
    """Run the kNN rows on rejected and cut sets, and return their p-values."""
    weighted = _weighted_draws(context, KNN_LABEL, dimension, NULL_WEIGHTED, replicates)
    values: dict[str, list[Array]] = {metric: [] for metric in KNN_FUNCTIONS}
    null_values: dict[str, list[Array]] = {metric: [] for metric in KNN_FUNCTIONS}
    counts = []
    for replicate in range(replicates):
        kept = rejection_subset(
            weighted.proposal[replicate],
            weighted.log_weights[replicate],
            key=context.key(KNN_LABEL, "null-rejection", dimension, replicate),
        )
        count = min(kept.shape[0], NULL_TARGET) // CUT_QUANTUM * CUT_QUANTUM
        counts.append(count)
        first = _cut(
            context.key(KNN_LABEL, "cut-first", dimension, replicate), kept, count
        )
        second = _cut(
            context.key(KNN_LABEL, "cut-second", dimension, replicate),
            weighted.target[replicate, :NULL_TARGET],
            count,
        )
        plan = permutation_null(
            first,
            None,
            second,
            None,
            key=context.key(KNN_LABEL, "null", dimension, replicate),
            draws=draws,
        )
        subsample = context.key(KNN_LABEL, "null-subsample", dimension, replicate)
        for metric in KNN_FUNCTIONS:
            result = compare_with_constants(
                _knn_builder(metric, dimension),
                (subsample,),
                plan,
                first,
                None,
                second,
                None,
            )
            values[metric].append(result.value)
            null_values[metric].append(result.draws)
    p_values = {
        metric: empirical_p_values(
            jnp.stack(values[metric]), jnp.stack(null_values[metric]), metric
        )
        for metric in KNN_FUNCTIONS
    }
    return p_values, jnp.asarray(counts, dtype=jnp.float64)


def _first_dimensions(context: Context, dimensions: Sequence[int]) -> tuple[int, ...]:
    """Return the dimensions of a full run, or the first one in a quick run."""
    return tuple(dimensions[: context.count(len(dimensions), 1)])


def _knn_weights(context: Context) -> Section:
    copy_dimensions = _first_dimensions(context, COPY_DIMENSIONS)
    information_dimensions = _first_dimensions(context, INFORMATION_DIMENSIONS)
    null_dimensions = _first_dimensions(context, NULL_DIMENSIONS)
    copy_replicates = context.count(20, 2)
    information_replicates = context.count(200, 3)
    null_replicates = context.count(100, 3)
    draws = context.count(NULL_DRAWS, 19)
    copy_rows: list[list[Cell]] = []
    nan_counts: dict[str, list[int]] = {metric: [] for metric in KNN_FUNCTIONS}
    kish_text, kept_text = [], []
    for dimension in copy_dimensions:
        copies = _copies(context, dimension, copy_replicates)
        row: list[Cell] = [dimension, Measurement.from_replicates(copies.distinct)]
        row.extend(
            Measurement.from_replicates(copies.over_rank[metric])
            for metric in ("kl", "hellinger", "js")
        )
        for metric in ("kl", "hellinger", "js"):
            nan_count = int(jnp.sum(copies.not_a_number[metric]))
            nan_counts[metric].append(nan_count)
            row.append(Measurement.from_rate(nan_count, copy_replicates))
        copy_rows.append(row)
        kish_text.append(Measurement.from_replicates(copies.kish).render())
        kept_text.append(Measurement.from_replicates(copies.kept).render())
    copy_table = Table.of(
        "Resampling a weighted set with replacement: the share of the samples that"
        " are distinct, the share of the distinct samples with more copies than the"
        " rank k of the estimate, and the share of the sets for which the estimate"
        " against equal-weight draws of the target is NaN.",
        (
            "Dimensions",
            "Distinct",
            "More than 3 copies",
            "More than 5 copies",
            "More than 10 copies",
            "KL (k = 3) NaN",
            "Hellinger (k = 5) NaN",
            "JS (k = 10) NaN",
        ),
        copy_rows,
    )
    information_rows: list[list[Cell]] = []
    information_cases: list[Case] = []
    control_cases: list[Case] = []
    rejection_cases: list[Case] = []
    threshold = THRESHOLDS["mutual_information"].value
    for dimension in information_dimensions:
        information = _information(context, dimension, information_replicates)
        kept = Measurement.from_replicates(information.kept)
        for kind, name, values, size in (
            (
                "resampled",
                "resampled with replacement",
                information.resampled,
                INFORMATION_COUNT,
            ),
            ("rejection", "rejection", information.rejection, kept),
            (
                "control",
                "equal-weight draws of the rejection count",
                information.control,
                kept,
            ),
        ):
            measurement = Measurement.from_replicates(values)
            information_rows.append([dimension, name, size, measurement])
            if kind == "resampled":
                information_cases.append(
                    Case(
                        f"{dimension} dimensions",
                        measurement.value - threshold,
                        measurement.margin(),
                        0.0,
                    )
                )
            if kind == "rejection":
                rejection_cases.append(
                    Case(
                        f"{dimension} dimensions",
                        measurement.value,
                        measurement.margin(),
                        INFORMATION_TOLERANCE,
                    )
                )
        difference = Measurement.from_replicates(
            information.rejection - information.control
        )
        control_cases.append(
            Case(
                f"{dimension} dimensions",
                difference.value,
                difference.margin(),
                0.0,
            )
        )
    information_table = Table.of(
        "The mean over the parameter pairs of the excess mutual information of"
        " independent axes, in nats, whose truth is zero.",
        ("Dimensions", "Set", "Samples", "Excess mutual information"),
        information_rows,
    )
    null_rows: list[list[Cell]] = []
    null_claims: list[Claim] = []
    cut_text = []
    for dimension in null_dimensions:
        p_values, counts = _null_p_values(context, dimension, null_replicates, draws)
        cut_text.append(Measurement.from_replicates(counts).render())
        for metric in KNN_FUNCTIONS:
            null_rows.append(
                [dimension, KNN_NAMES[metric], *rate_measurements(p_values[metric])]
            )
            counts_at_levels = [
                rate_counts(p_values[metric], KNN_LEVELS)[position]
                for position in range(len(KNN_LEVELS))
            ]
            null_claims.append(
                claim_rates_within(
                    f"In {dimension} dimensions, the {KNN_NAMES[metric]} row after"
                    " rejection rejects at its levels.",
                    counts_at_levels,
                    KNN_LEVELS,
                    full_only=True,
                )
            )
    null_table = Table.of(
        "Rejection rates of the kNN rows of a weighted set against equal-weight"
        " draws of its target, after rejection and a cut to equal counts.",
        ("Dimensions", "Row", *LEVEL_COLUMNS),
        null_rows,
    )
    dimension_names = _dimension_text(copy_dimensions)
    claims = [
        claim_true(
            f"After resampling with replacement, the {KNN_NAMES[metric]} estimate"
            " is NaN in every set.",
            f"NaN in every replicate, in {dimension_names} dimensions",
            "; ".join(
                f"{dimension}: {count} of {copy_replicates}"
                for dimension, count in zip(
                    copy_dimensions, nan_counts[metric], strict=True
                )
            ),
            all(count == copy_replicates for count in nan_counts[metric]),
        )
        for metric in ("kl", "hellinger")
    ]
    claims.append(
        claim_cases_above(
            "After resampling with replacement, the mutual information of independent"
            " axes reads above the flag threshold.",
            information_cases,
            tolerance_text=None,
            replicates=information_replicates,
            reference_text=f"the threshold of {threshold:g} nats",
        )
    )
    claims.append(
        claim_cases(
            "After rejection, the excess mutual information of independent axes lies"
            " within a tenth of the flag threshold of zero.",
            rejection_cases,
            tolerance_text=f"{INFORMATION_TOLERANCE:g} nats, a tenth of the flag"
            " threshold",
            replicates=information_replicates,
        )
    )
    claims.append(
        claim_cases(
            "After rejection, the mutual information of independent axes equals that"
            " of equal-weight draws of the same count.",
            control_cases,
            tolerance_text=None,
            replicates=information_replicates,
        )
    )
    claims.extend(null_claims)
    setup = (
        f"Zoo Gaussians: a target N(0, I) and a proposal {PROPOSAL_WIDTH} times wider,"
        " whose draws carry the importance weights target over proposal. First,"
        f" {grouped(COPY_WEIGHTED)} proposal draws are resampled with replacement"
        f" to {grouped(COPY_WEIGHTED)} samples (the weighted bootstrap of Smith and"
        " Gelfand) in"
        f" {dimension_names} dimensions, with {copy_replicates} replicates each. The"
        f" Kish sizes of the weights are {', '.join(kish_text)}, and"
        f" {REJECTION_SUBSET} keeps {', '.join(kept_text)} samples. The table gives"
        " the share of distinct samples, the share of them with more than k copies"
        " for the ranks of the kNN KL, Hellinger and JS estimates, and the share of"
        " the sets for which each estimate against"
        f" {grouped(COPY_TARGET)} target draws is NaN. Second, the excess mutual"
        " information (KSG, k = 10) of the independent axes of"
        f" {grouped(INFORMATION_COUNT)} draws, in"
        f" {_dimension_text(information_dimensions)} dimensions with"
        f" {information_replicates} replicates, is read after resampling with"
        " replacement, after rejection, and for equal-weight draws of the target"
        " with the count that rejection kept in the same replicate. Third, the"
        f" kNN rows run as {COMPARE} runs them: {grouped(NULL_WEIGHTED)} weighted"
        f" draws are cut by rejection and both sets to the smaller count, rounded"
        f" down to a multiple of {CUT_QUANTUM} so that one compiled null program"
        f" serves the replicates (counts {', '.join(cut_text)} in"
        f" {_dimension_text(null_dimensions)} dimensions), against {NULL_TARGET}"
        f" target draws, with {draws} draws of {PERMUTATION_NULL}"
        f" and {null_replicates} replicates per dimension. A claim lets a mean lie"
        " within three standard errors (Student t for few replicates) of its"
        " expectation, and a rate within three binomial errors of its level."
    )
    return Section(setup, (copy_table, information_table, null_table), tuple(claims))


# * Check weighted-joint-null.

JOINT_LABEL = "weighted-joint-null"
JOINT_DIMENSIONS = (2, 4)
JOINT_WEIGHTED, JOINT_TARGET = 1_000, 500
JOINT_NAMES = {
    "mean_shift": "mean shift",
    "ks": "KS",
    "energy": "energy distance",
    "mmd": "MMD squared",
    "kl": "Gaussian KL",
}
JOINT_CONTROLS = ("mean_shift", "ks")
"""The per-parameter statistics that the weighted null was first checked on."""


def _gaussian_kl_statistic(space: ParamSpace) -> Statistic:
    """Return the Gaussian KL from the moment-matched Gaussians of the two sets."""

    def statistic(
        first: Array, first_weights: Array, second: Array, second_weights: Array
    ) -> Array:
        one = GaussianDensity.fit(first, first_weights, space)
        two = GaussianDensity.fit(second, second_weights, space)
        return gaussian_kl(
            one.mean, one.covariance, two.mean, two.covariance, space
        ).total

    return statistic


def _mmd_builder(space: ParamSpace, key: PRNGKeyArray) -> Builder:
    """Return the builder of the MMD squared statistic from its bandwidth."""

    def build(bandwidth: Array) -> Statistic:
        def statistic(
            first: Array, first_weights: Array, second: Array, second_weights: Array
        ) -> Array:
            return maximum_mean_discrepancy(
                first,
                first_weights,
                second,
                second_weights,
                space,
                bandwidth=bandwidth,
                key=key,
            )

        return statistic

    return build


def _joint_p_values(
    context: Context, dimension: int, replicates: int, draws: int
) -> tuple[dict[str, Array], Array]:
    """Return the p-values of the joint statistics and the Kish sizes."""
    space = real_space(dimension)
    key = context.key(JOINT_LABEL, "subsample", dimension)

    def energy(
        first: Array, first_weights: Array, second: Array, second_weights: Array
    ) -> Array:
        return energy_distance(
            first, first_weights, second, second_weights, space, key=key
        )

    statistics = {
        name: statistic
        for name, statistic in first_parameter_statistics(space).items()
        if name in JOINT_CONTROLS
    }
    statistics["energy"] = energy
    statistics["kl"] = _gaussian_kl_statistic(space)
    mmd_builder = _mmd_builder(space, key)
    draws_of = _weighted_draws(
        context, JOINT_LABEL, dimension, JOINT_WEIGHTED, replicates
    )
    values: dict[str, list[Array]] = {name: [] for name in JOINT_NAMES}
    null_values: dict[str, list[Array]] = {name: [] for name in JOINT_NAMES}
    kish = []
    for replicate in range(replicates):
        weighted = draws_of.proposal[replicate]
        log_weights = draws_of.log_weights[replicate]
        target = draws_of.target[replicate, :JOINT_TARGET]
        kish.append(float(kish_effective_sample_size(log_weights)))
        plan = permutation_null(
            weighted,
            log_weights,
            target,
            None,
            key=context.key(JOINT_LABEL, "null", dimension, replicate),
            draws=draws,
        )
        for name, statistic in statistics.items():
            result = compare_on_plan(
                statistic, plan, weighted, log_weights, target, None
            )
            values[name].append(result.value)
            null_values[name].append(result.draws)
        # * The bandwidth is read once on the observed sets, as the engine does.
        bandwidth = median_bandwidth(
            weighted, log_weights, target, None, space, key=key
        )
        result = compare_with_constants(
            mmd_builder, (bandwidth,), plan, weighted, log_weights, target, None
        )
        values["mmd"].append(result.value)
        null_values["mmd"].append(result.draws)
    p_values = {
        name: empirical_p_values(
            jnp.stack(values[name]), jnp.stack(null_values[name]), name
        )
        for name in JOINT_NAMES
    }
    return p_values, jnp.asarray(kish)


def _weighted_joint_null(context: Context) -> Section:
    replicates = context.count(200, 5)
    draws = context.count(NULL_DRAWS, 19)
    dimensions = _first_dimensions(context, JOINT_DIMENSIONS)
    rows: list[list[Cell]] = []
    claims: list[Claim] = []
    kish_text = []
    for dimension in dimensions:
        p_values, kish = _joint_p_values(context, dimension, replicates, draws)
        kish_text.append(Measurement.from_replicates(kish).render())
        for name, display in JOINT_NAMES.items():
            rows.append([dimension, display, *rate_measurements(p_values[name])])
            claims.append(
                claim_rates_within(
                    f"In {dimension} dimensions, the {display} of a weighted set"
                    " against an equal-weight set rejects at its levels.",
                    rate_counts(p_values[name]),
                    LEVELS,
                    full_only=True,
                )
            )
    table = Table.of(
        "Rejection rates of the statistics of an importance-weighted set against"
        " equal-weight draws of its target. The mean shift and the KS statistic read"
        " the first parameter.",
        ("Dimensions", "Statistic", *LEVEL_COLUMNS),
        rows,
    )
    setup = (
        f"Zoo Gaussians: a target N(0, I) and a proposal {PROPOSAL_WIDTH} times"
        f" wider. The weighted set holds {grouped(JOINT_WEIGHTED)} proposal draws"
        " with the importance weights target over proposal (Kish sizes"
        f" {', '.join(kish_text)} in {_dimension_text(dimensions)} dimensions); the"
        f" other set holds {JOINT_TARGET} equal-weight draws of the target. Each of"
        f" {replicates} replicates uses {draws} draws of {PERMUTATION_NULL}, whose"
        " sides follow the Kish sizes and in which every sample keeps its weight, so"
        " the joint statistics need no resampling. The energy distance, MMD"
        " squared (median bandwidth of the observed sets) and the Gaussian KL of"
        " moment-matched Gaussians are the joint statistics; the mean shift and the"
        " KS statistic of the first parameter are the controls of the first check"
        " of the weighted null. A claim lets each rate lie within three binomial"
        " errors of its level."
    )
    return Section(setup, (table,), tuple(claims))


# * Check half-sample-errors.

HALF_LABEL = "half-sample-errors"
HALF_NAMES = (
    "mean shift",
    "log width ratio",
    "KS",
    "energy distance",
    "Gaussian KL",
    "sliced W2",
)
LOG_WIDTH_RATIO = "log width ratio"
CHAIN_SHIFT = 0.3
"""Shift along the first axis between the two chains, in widths."""
INDEPENDENT_SHIFT = 0.5
INDEPENDENT_COUNT = 1_000
HALF_WEIGHTED, HALF_TARGET = 4_000, 2_000
HALF_SHIFT = 0.3
SUBSAMPLING_SCALE = math.sqrt(0.5)
"""The scale of subsampling with b much smaller than n, applied to halves."""


def _half_statistic(space: ParamSpace, key: PRNGKeyArray) -> Statistic:
    """Return the six statistics of the check as one vector, jitted."""
    directions = sliced_directions(space, SLICED_DIRECTIONS, key=key)

    def statistic(
        first: Array, first_weights: Array, second: Array, second_weights: Array
    ) -> Array:
        one = GaussianDensity.fit(first, first_weights, space)
        two = GaussianDensity.fit(second, second_weights, space)
        return jnp.stack(
            [
                mean_shift(first, first_weights, second, second_weights, space)[0],
                jnp.log(
                    width_ratio(first, first_weights, second, second_weights, space)[0]
                ),
                ks_statistic(first[:, 0], first_weights, second[:, 0], second_weights),
                energy_distance(
                    first, first_weights, second, second_weights, space, key=key
                ),
                gaussian_kl(
                    one.mean, one.covariance, two.mean, two.covariance, space
                ).total,
                sliced_wasserstein_distance(
                    first,
                    first_weights,
                    second,
                    second_weights,
                    space,
                    directions=directions,
                ),
            ]
        )

    return jax.jit(statistic)


class _HalfCase(NamedTuple):
    """A case of the check: its sets, weights and effective sizes per replicate."""

    name: str
    firsts: Array
    second_sets: Array
    first_log_weights: Array
    second_log_weights: Array
    first_n_eff: Array | None
    """Effective sizes of the first sets, per replicate; ``None`` for Kish sizes."""
    second_n_eff: Array | None


def _independent_case(context: Context, replicates: int) -> _HalfCase:
    case = zoo.shifted_gaussians(INDEPENDENT_SHIFT, param_count=2)
    sets = case.sample(
        context.key(HALF_LABEL, "independent"), INDEPENDENT_COUNT, batch_size=replicates
    ).samples
    zeros = jnp.zeros((replicates, INDEPENDENT_COUNT))
    return _HalfCase("independent", sets[:, 0], sets[:, 1], zeros, zeros, None, None)


def _weighted_case(context: Context, replicates: int) -> _HalfCase:
    case = zoo.gaussians(
        jnp.zeros((3, 2)).at[2, 0].set(HALF_SHIFT),
        jnp.stack(
            [jnp.eye(2), PROPOSAL_WIDTH**2 * jnp.eye(2), jnp.eye(2)],
        ),
        names=("target", "proposal", "shifted"),
    )
    sets = case.sample(
        context.key(HALF_LABEL, "weighted"), HALF_WEIGHTED, batch_size=replicates
    ).samples
    proposal = sets[:, 1]
    log_densities = case.log_prob(proposal)
    log_weights = log_densities[..., 0] - log_densities[..., 1]
    return _HalfCase(
        "weighted",
        proposal,
        sets[:, 2, :HALF_TARGET],
        log_weights,
        jnp.zeros((replicates, HALF_TARGET)),
        None,
        None,
    )


def _chain_case(context: Context, replicates: int) -> _HalfCase:
    runs = chain_runs(context.seed, context.quick)
    pairs = [(runs[2 * index], runs[2 * index + 1]) for index in range(replicates)]
    # * One compiled program serves the replicates, so cut every set to the smallest.
    count = min(run.samples.shape[0] for pair in pairs for run in pair)
    firsts = jnp.stack([first.samples[:count] for first, _ in pairs])
    seconds = jnp.stack(
        [second.samples[:count].at[:, 0].add(CHAIN_SHIFT) for _, second in pairs]
    )
    sizes = (
        jnp.asarray(
            [first.n_eff * count / first.samples.shape[0] for first, _ in pairs]
        ),
        jnp.asarray(
            [second.n_eff * count / second.samples.shape[0] for _, second in pairs]
        ),
    )
    zeros = jnp.zeros((replicates, count))
    return _HalfCase("chains", firsts, seconds, zeros, zeros, sizes[0], sizes[1])


class _HalfRun(NamedTuple):
    """The true scatter and the half-sample errors of one case."""

    true_scatter: Array
    """The standard deviation of every statistic over the replicates."""
    raw: Array
    """Half-sample errors without the correlation factor, per half replicate."""
    corrected: Array
    """Half-sample errors with the correlation factor, per half replicate."""


def _half_run(
    context: Context, case: _HalfCase, replicates: int, half_replicates: int, pairs: int
) -> _HalfRun:
    """Evaluate the statistics on every replicate and on halves of the first ones."""
    space = real_space(2)
    statistic = _half_statistic(space, context.key(HALF_LABEL, "statistic", case.name))
    values = jnp.stack(
        [
            statistic(
                case.firsts[replicate],
                case.first_log_weights[replicate],
                case.second_sets[replicate],
                case.second_log_weights[replicate],
            )
            for replicate in range(replicates)
        ]
    )
    raw, corrected = [], []
    for replicate in range(half_replicates):
        draws = half_sample_draws(
            statistic,
            case.firsts[replicate],
            case.first_log_weights[replicate],
            case.second_sets[replicate],
            case.second_log_weights[replicate],
            key=context.key(HALF_LABEL, "halves", case.name, replicate),
            pairs=pairs,
        )
        raw.append(
            half_sample_error(
                draws,
                case.first_log_weights[replicate],
                case.second_log_weights[replicate],
            )
        )
        corrected.append(
            half_sample_error(
                draws,
                case.first_log_weights[replicate],
                case.second_log_weights[replicate],
                first_n_eff=(
                    None
                    if case.first_n_eff is None
                    else float(case.first_n_eff[replicate])
                ),
                second_n_eff=(
                    None
                    if case.second_n_eff is None
                    else float(case.second_n_eff[replicate])
                ),
            )
        )
    return _HalfRun(
        jnp.std(values, axis=0, ddof=1),
        jnp.stack(raw),
        jnp.stack(corrected),
    )


def _ratio(errors: Array, true_scatter: Array, replicates: int) -> Measurement:
    """Return the mean error over the true scatter with the error of both parts.

    The error of the true scatter of ``replicates`` values is the relative
    1/sqrt(2 (replicates - 1)) of a standard deviation of normal values.
    """
    count = int(errors.shape[0])
    mean = float(jnp.mean(errors))
    error_of_mean = float(jnp.std(errors, ddof=1)) / math.sqrt(count)
    ratio = mean / float(true_scatter)
    relative = math.hypot(error_of_mean / mean, 1 / math.sqrt(2 * (replicates - 1)))
    return Measurement(ratio, ratio * relative, count)


def _scaled(measurement: Measurement, factor: float) -> Measurement:
    """Return a measurement multiplied by a constant."""
    assert measurement.error is not None
    return Measurement(
        factor * measurement.value, factor * measurement.error, measurement.replicates
    )


def _half_sample_errors(context: Context) -> Section:
    replicates = context.count(200, 20)
    half_replicates = context.count(20, 4)
    pairs = context.count(100, 10)
    cases = (
        _independent_case(context, replicates),
        _weighted_case(context, replicates),
        _chain_case(context, replicates),
    )
    rows: list[list[Cell]] = []
    scatter_rows: list[list[Cell]] = []
    ratio_cases: dict[str, list[Case]] = {case.name: [] for case in cases}
    scale_cases: list[Case] = []
    for case in cases:
        run = _half_run(context, case, replicates, half_replicates, pairs)
        for position, name in enumerate(HALF_NAMES):
            true_scatter = run.true_scatter[position]
            raw = _ratio(run.raw[:, position], true_scatter, replicates)
            corrected = _ratio(run.corrected[:, position], true_scatter, replicates)
            scaled = _scaled(raw, SUBSAMPLING_SCALE)
            scatter_rows.append(
                [
                    case.name,
                    name,
                    Measurement(
                        float(true_scatter),
                        float(true_scatter) / math.sqrt(2 * (replicates - 1)),
                    ),
                ]
            )
            rows.append([case.name, name, raw, corrected, scaled])
            linear = name != LOG_WIDTH_RATIO or case.name != "chains"
            used = corrected if case.name == "chains" else raw
            if linear:
                ratio_cases[case.name].append(
                    Case(name, used.value - 1.0, used.margin(), 0.0)
                )
            scale_cases.append(
                Case(
                    f"{case.name}, {name}",
                    1.0 - scaled.value,
                    scaled.margin(),
                    0.0,
                )
            )
    table = Table.of(
        "The mean half-sample error over the true scatter of the full-sample value,"
        " without and with the correlation factor, and without the factor times"
        " the subsampling scale of one over the square root of two.",
        ("Sets", "Statistic", "Without factor", "With factor", "Subsampling scale"),
        rows,
    )
    scatter_table = Table.of(
        "The true scatter of the full-sample value over the replicates.",
        ("Sets", "Statistic", "Standard deviation"),
        scatter_rows,
    )
    claims = [
        claim_cases(
            "On independent sets, the half-sample error equals the true scatter.",
            ratio_cases["independent"],
            tolerance_text=None,
            replicates=half_replicates,
        ),
        claim_cases(
            "On importance-weighted sets, the half-sample error equals the true"
            " scatter.",
            ratio_cases["weighted"],
            tolerance_text=None,
            replicates=half_replicates,
        ),
        claim_cases(
            "On thinned chains, the half-sample error with the correlation factor"
            " equals the true scatter of the statistics other than the log width"
            " ratio.",
            ratio_cases["chains"],
            tolerance_text=None,
            replicates=half_replicates,
        ),
        claim_cases_above(
            "The subsampling scale of one over the square root of two reads below"
            " the true scatter.",
            scale_cases,
            tolerance_text=None,
            replicates=half_replicates,
            reference_text="zero for one minus the ratio",
        ),
    ]
    chain_count = cases[2].firsts.shape[1]
    setup = (
        "Three kinds of sets with two parameters give two sets per replicate, and"
        " the standard deviation of each full-sample statistic over"
        f" {replicates} replicates is its true scatter. Independent sets:"
        f" {SHIFTED_GAUSSIANS} with a shift of {INDEPENDENT_SHIFT:g} and"
        f" {grouped(INDEPENDENT_COUNT)} samples per set. Weighted sets:"
        f" {grouped(HALF_WEIGHTED)} draws of a proposal {PROPOSAL_WIDTH} times wider"
        " than the target with importance weights, against"
        f" {grouped(HALF_TARGET)} equal-weight draws of the target shifted by"
        f" {HALF_SHIFT:g}. Thinned chains: pairs of the AR(1) runs of the evidence"
        " check, reduced by"
        f" {role('func', 'mixed_signals.mcmc.reduce.reduce_chains')}, the second"
        f" shifted by {CHAIN_SHIFT:g}, each cut to {grouped(chain_count)} samples so"
        " that one compiled program serves all replicates, with the effective"
        " sizes scaled to the cut. On the first"
        f" {half_replicates} replicates, {pairs} pairs of random half-samples"
        f" ({role('func', 'mixed_signals.sigma.uncertainty.half_sample_draws')})"
        " give the half-sample error"
        f" ({role('func', 'mixed_signals.sigma.uncertainty.half_sample_error')}),"
        " which the table divides by the true scatter. The correlation factor"
        " uses the effective sizes of the chains and is one otherwise. The error"
        " of a ratio combines the scatter of the half-sample errors over the"
        f" {half_replicates} replicates with the error 1/sqrt(2 ({replicates} - 1))"
        " of the true scatter. A claim lets a ratio lie within three standard errors"
        " (Student t for few replicates) of one; the claim about chains leaves out"
        " the log width ratio, a function of squares, which the table gives."
    )
    return Section(setup, (table, scatter_table), tuple(claims))


# * Check engine-recovery.

RECOVERY_LABEL = "engine-recovery"
RECOVERY_SAMPLES = 2_000
RECOVERY_SHIFTS = (0.5, 1.0)
RECOVERY_RATIO = 1.25
SHAPE_METRICS = ("width_ratio", "kl_shape", "w2_shape", "hellinger_shape")
"""Rows whose effect size is r_eff, a width ratio."""
LOCATION_METRICS = ("kl_location", "w2_location", "hellinger_location")
"""Rows of the location terms, which a change of width leaves at zero."""
PARAMETER_METRICS = ("mean_shift", "ks", "w1", "ad", "js", "hellinger")
"""Per-parameter rows that a change of width of x0 leaves at zero on x1."""
MODES = {"A": "reference", "B": None}
"""Mode A names the reference; mode B compares every pair."""


class _Row(NamedTuple):
    """One row of a result table: its pair, metric, estimator and parameters."""

    pair: tuple[str, ...]
    metric: str
    estimator: str
    parameters: tuple[str, ...]

    def label(self) -> str:
        """Name the row for a table: metric, estimator and parameters."""
        return f"{self.metric} ({self.estimator}), {'+'.join(self.parameters)}"


def _effect_values(result: Any, effect: str) -> dict[_Row, list[float]]:
    """Group an effect size of the result table by row, over the problems."""
    groups: dict[_Row, list[float]] = {}
    for row in result.table.to_dicts():
        value = row[effect]
        if value is None:
            continue
        key = _Row(
            tuple(row["pair"]),
            row["metric"],
            row["estimator"],
            tuple(row["parameters"]),
        )
        groups.setdefault(key, []).append(float(value))
    return groups


def _reference_values(
    case: zoo.ZooCase, metric: str
) -> dict[tuple[tuple[str, ...], tuple[str, ...]], float]:
    """Return the reference values of a metric by pair and parameters."""
    return {
        (tuple(row["pair"]), tuple(row["parameters"])): float(row["value"])
        for row in case.reference.to_dicts()
        if row["metric"] == metric
    }


class _Recovery(NamedTuple):
    """The rows of one case in both modes, with their effects and truths."""

    truths: dict[_Row, tuple[str, float]]
    """Per row, the effect size (``sigma_eff`` or ``r_eff``) and its truth."""
    means: dict[str, dict[_Row, Measurement]]
    """Per mode, the mean over the problems of the effect size, a width ratio as log."""
    cases: list[Case]


def _log_if(effect: str, value: float) -> float:
    """Return the value, or its logarithm for a width ratio."""
    return math.log(value) if effect == "r_eff" else value


def _recovered(
    results: dict[str, Any],
    truths: dict[_Row, tuple[str, float]],
    problems: int,
    label: str,
) -> _Recovery:
    """Join the effect sizes of both modes with their truths into cases of a claim.

    A width ratio enters as its logarithm, so that its error is symmetric.
    """
    means: dict[str, dict[_Row, Measurement]] = {mode: {} for mode in results}
    cases: list[Case] = []
    for mode, result in results.items():
        grouped_values = {
            effect: _effect_values(result, effect) for effect in ("sigma_eff", "r_eff")
        }
        for row, (effect, truth) in truths.items():
            values = grouped_values[effect].get(row, [])
            if len(values) != problems:
                raise ValueError(f"{label}: row {row.label()} has {len(values)} values")
            measurement = Measurement.from_replicates(
                [_log_if(effect, value) for value in values]
            )
            means[mode][row] = measurement
            cases.append(
                Case(
                    f"{label}, mode {mode}, {row.label()}",
                    measurement.value - _log_if(effect, truth),
                    measurement.margin(),
                    0.0,
                )
            )
    return _Recovery(truths, means, cases)


def _shift_truths(case: zoo.ZooCase, result: Any) -> dict[_Row, tuple[str, float]]:
    """Return the sigma_eff truth of every row of a shifted pair that has one."""
    per_parameter = _reference_values(case, "mean_shift")
    joint = _reference_values(case, "mahalanobis_shift")
    truths: dict[_Row, tuple[str, float]] = {}
    for row in _effect_values(result, "sigma_eff"):
        if row.metric in SHAPE_METRICS:
            continue
        table = per_parameter if len(row.parameters) == 1 else joint
        truths[row] = ("sigma_eff", abs(table[(row.pair, row.parameters)]))
    return truths


def _scale_truths(case: zoo.ZooCase, result: Any) -> dict[_Row, tuple[str, float]]:
    """Return the effect-size truth of the rows of a pair with a scaled width."""
    pair = ("reference", "scaled")
    ratios = _reference_values(case, "width_ratio")
    largest = max(value for (key, _), value in ratios.items() if key == pair)
    truths: dict[_Row, tuple[str, float]] = {}
    for row in _effect_values(result, "r_eff"):
        if row.pair != pair:
            continue
        if row.metric == "width_ratio":
            truths[row] = ("r_eff", ratios[(pair, row.parameters)])
        elif row.metric in SHAPE_METRICS:
            truths[row] = ("r_eff", largest)
    for row in _effect_values(result, "sigma_eff"):
        if row.pair != pair:
            continue
        if row.metric in LOCATION_METRICS or (
            row.metric in PARAMETER_METRICS and row.parameters == ("x1",)
        ):
            truths[row] = ("sigma_eff", 0.0)
    return truths


def _compare_modes(
    context: Context, case: zoo.ZooCase, name: str, problems: int, draws: int
) -> dict[str, Any]:
    """Run ``compare`` on a pair of the zoo in both modes."""
    samples = case.sample(
        context.key(RECOVERY_LABEL, "sets", name),
        context.count(RECOVERY_SAMPLES, 500),
        batch_size=problems,
    )
    return {
        mode: compare(
            samples,
            case.names,
            case.space,
            key=context.key(RECOVERY_LABEL, "compare", name, mode),
            reference=reference,
            draws=draws,
        )
        for mode, reference in MODES.items()
    }


def _recovery_table(caption: str, recovery: _Recovery) -> Table:
    """Tabulate the truths and the mean effect sizes of both modes."""
    rows: list[list[Cell]] = []
    for row, (effect, truth) in recovery.truths.items():
        shown: list[Cell] = []
        for mode in recovery.means:
            measurement = recovery.means[mode][row]
            if effect == "r_eff":
                ratio = math.exp(measurement.value)
                measurement = Measurement(
                    ratio, ratio * (measurement.error or 0.0), measurement.replicates
                )
            shown.append(measurement)
        rows.append(["/".join(row.pair), row.label(), effect, truth, *shown])
    return Table.of(
        caption,
        (
            "Pair",
            "Row",
            "Effect",
            "Truth",
            *(f"Mode {mode}" for mode in recovery.means),
        ),
        rows,
    )


class _FlagCase(NamedTuple):
    """A zoo posterior and the rows that the diagnostics must and must not flag."""

    name: str
    case: zoo.ZooCase
    posterior: str
    required: tuple[tuple[str, str], ...]
    """Rows that must flag, as (metric, parameters joined by commas)."""
    absent_metrics: tuple[str, ...] = ()
    """Metrics of which no row may flag."""
    absent_rows: tuple[tuple[str, str], ...] = ()
    """Rows that must not flag."""
    quiet: bool = False
    """Whether no row at all may flag."""


def _flag_cases() -> tuple[_FlagCase, ...]:
    covariance = jnp.asarray([[1.0, 0.8, 0.0], [0.8, 1.0, -0.3], [0.0, -0.3, 1.0]])
    return (
        _FlagCase(
            "Gaussian with correlations",
            zoo.gaussians(jnp.zeros((1, 3)), covariance[None], names=["gaussian"]),
            "gaussian",
            (),
            quiet=True,
        ),
        _FlagCase(
            "Banana",
            zoo.banana(),
            "banana",
            (
                ("distance_correlation", "x0,x1"),
                ("mutual_information", "x0,x1"),
                ("mardia_skewness", "x0,x1"),
                ("skewness", "x1"),
                ("gmm_bic", "x0,x1"),
            ),
            absent_metrics=("mode_dip",),
            absent_rows=(("spearman", "x0,x1"),),
        ),
        _FlagCase(
            "Mixture, separation 4",
            zoo.gaussian_mixture(4.0),
            "mixture",
            (
                ("mode_dip", "x0"),
                ("mode_dip", "x0,x1"),
                ("kurtosis", "x0"),
                ("gmm_bic", "x0,x1"),
            ),
            absent_metrics=("distance_correlation", "mutual_information"),
        ),
        _FlagCase(
            "Normal at its bound",
            zoo.railed_gaussians([0.0]),
            "railed_0",
            (("railing_lower", "x0"), ("skewness", "x0")),
            absent_metrics=("mode_dip",),
        ),
        _FlagCase(
            "Normal one width from its bound",
            zoo.railed_gaussians([1.0]),
            "railed_0",
            (("railing_lower", "x0"), ("skewness", "x0")),
            absent_metrics=("mode_dip",),
        ),
    )


def _flagged_rows(result: Any, problem: int) -> set[tuple[str, str]]:
    """Return the (metric, parameters) of the rows of a problem that flag."""
    flagged = result.table.filter((pl.col("problem") == problem) & pl.col("flag"))
    return {(row["metric"], ",".join(row["parameters"])) for row in flagged.to_dicts()}


def _flags_met(flag_case: _FlagCase, flagged: set[tuple[str, str]]) -> bool:
    """Return whether the flags of one problem are those that the case expects."""
    if flag_case.quiet:
        return not flagged
    metrics = {metric for metric, _ in flagged}
    return (
        set(flag_case.required) <= flagged
        and not metrics & set(flag_case.absent_metrics)
        and not flagged & set(flag_case.absent_rows)
    )


def _diagnose_flags(
    context: Context, flag_case: _FlagCase, problems: int, draws: int
) -> tuple[int, set[tuple[str, str]]]:
    """Run ``diagnose`` on a zoo posterior; count the problems that meet the flags."""
    samples = flag_case.case.sample(
        context.key(RECOVERY_LABEL, "diagnose", flag_case.name),
        context.count(RECOVERY_SAMPLES // 2, 400),
        batch_size=problems,
    ).select(posteriors=[flag_case.case.names.index(flag_case.posterior)])
    diagnosis = diagnose(
        samples,
        [flag_case.posterior],
        flag_case.case.space,
        key=context.key(RECOVERY_LABEL, "diagnose-key", flag_case.name),
        draws=draws,
    )
    met = 0
    seen: set[tuple[str, str]] = set()
    for problem in range(problems):
        flagged = _flagged_rows(diagnosis.result, problem)
        met += int(_flags_met(flag_case, flagged))
        seen |= flagged
    return met, seen


def _flag_text(rows: set[tuple[str, str]]) -> str:
    """List the flagged rows, or say that none flagged."""
    if not rows:
        return "none"
    return ", ".join(f"{metric} {parameters}" for metric, parameters in sorted(rows))


def _engine_recovery(context: Context) -> Section:
    problems = context.count(8, 2)
    draws = context.count(39, 9)
    flag_problems = context.count(4, 1)
    flag_draws = context.count(99, 19)
    # * A quick run keeps the largest shift and two flag cases, to save compile time.
    shifts = RECOVERY_SHIFTS[
        len(RECOVERY_SHIFTS) - context.count(len(RECOVERY_SHIFTS), 1) :
    ]
    flag_cases = _flag_cases()[: context.count(5, 2)]
    blocks: list[str | Table] = []
    claims: list[Claim] = []
    for shift in shifts:
        case = zoo.shifted_gaussians(shift, param_count=2)
        results = _compare_modes(context, case, f"shift-{shift:g}", problems, draws)
        recovery = _recovered(
            results, _shift_truths(case, results["A"]), problems, f"shift {shift:g}"
        )
        blocks.append(
            _recovery_table(
                f"The mean over {problems} problems of sigma_eff of every row of a"
                f" shift of {shift:g} widths, in both modes, against its truth.",
                recovery,
            )
        )
        claims.append(
            claim_cases(
                f"A shift of {shift:g} widths: every sigma_eff row of both modes"
                " equals the shift, or zero on the axis without shift.",
                recovery.cases,
                tolerance_text=None,
                replicates=problems,
            )
        )
    scaled = zoo.scaled_gaussians(RECOVERY_RATIO, param_count=2)
    results = _compare_modes(context, scaled, "scaled", problems, draws)
    recovery = _recovered(
        results,
        _scale_truths(scaled, results["A"]),
        problems,
        f"ratio {RECOVERY_RATIO:g}",
    )
    blocks.append(
        _recovery_table(
            f"The mean over {problems} problems of the effect size of the rows of a"
            f" width ratio of {RECOVERY_RATIO:g} along x0, in both modes: r_eff, the"
            " width ratio, and sigma_eff where the truth is zero.",
            recovery,
        )
    )
    claims.append(
        claim_cases(
            f"A width ratio of {RECOVERY_RATIO:g}: every r_eff row of both modes equals"
            " the ratio, in logarithm, and the rows without a change of location read"
            " no shift.",
            recovery.cases,
            tolerance_text=None,
            replicates=problems,
        )
    )
    flag_rows: list[list[Cell]] = []
    for flag_case in flag_cases:
        met, seen = _diagnose_flags(context, flag_case, flag_problems, flag_draws)
        flag_rows.append(
            [
                flag_case.name,
                ", ".join(
                    f"{metric} {parameters}"
                    for metric, parameters in flag_case.required
                )
                or "none",
                f"{met} of {flag_problems}",
                _flag_text(seen),
            ]
        )
        claims.append(
            claim_true(
                f"{flag_case.name}: the diagnostics raise the expected flags.",
                "the required rows flag and the excluded rows do not, in every problem",
                f"{met} of {flag_problems} problems",
                met == flag_problems,
                full_only=True,
            )
        )
    blocks.append(
        Table.of(
            "The rows that the diagnostics flag on zoo posteriors, against the rows"
            " that a case must flag.",
            (
                "Posterior",
                "Must flag",
                "Problems that meet the flags",
                "Flagged in any problem",
            ),
            flag_rows,
        )
    )
    setup = (
        f"{COMPARE} with the default recipes runs on the pairs of the zoo,"
        f" {grouped(context.count(RECOVERY_SAMPLES, 500))} samples per posterior, two"
        f" parameters, {_plural(problems, 'independent problem')} in one call,"
        f" {draws} null draws per row, once in mode A (the first posterior is the"
        " reference) and once in mode B (every pair, pooled whitening). The pairs are"
        f" {SHIFTED_GAUSSIANS} with a shift of {_value_text(shifts)} widths and"
        f" {SCALED_GAUSSIANS} with a width ratio"
        f" of {RECOVERY_RATIO:g} along x0. The table joins each row of the result"
        " with the reference table of the case: the mean shift or the Mahalanobis"
        " shift gives the truth of sigma_eff, and the width ratio the truth of"
        " r_eff, whose shape terms read the largest ratio; in mode B the W2 shape"
        f" term uses the pooled relation of {R_EFF}. Rows on the axis that a"
        " case leaves unchanged and the location terms of the width ratio have the"
        " truth zero, and the shape rows of the shifts are left out. A claim lets"
        " each mean over the problems lie within three standard errors (Student t"
        " for few problems) of its truth. Then"
        f" {DIAGNOSE} runs on zoo posteriors with"
        f" {grouped(context.count(RECOVERY_SAMPLES // 2, 400))} samples,"
        f" {_plural(flag_problems, 'problem')} and {flag_draws} surrogate draws; a"
        " claim is true when every problem flags the required rows and none flags an"
        " excluded row, and for the Gaussian, when no row flags."
    )
    return Section(setup, tuple(blocks), tuple(claims))


CHECKS: tuple[Check, ...] = (
    Check(
        label=KNN_LABEL,
        title="kNN rows of weighted sets",
        session="14",
        supports=(
            role("func", "mixed_signals.core.samples.rejection_subset", short=False),
            role("func", "mixed_signals.engines.compare.compare", short=False),
        ),
        run=_knn_weights,
    ),
    Check(
        label=JOINT_LABEL,
        title="The weighted permutation null for joint metrics",
        session="14",
        supports=(
            role(
                "func", "mixed_signals.nulls.permutation.permutation_null", short=False
            ),
            role("func", "mixed_signals.engines.compare.compare", short=False),
            "the weighted null of the joint rows of compare: the energy distance, MMD"
            " squared and Gaussian KL keep the weights in the pool",
        ),
        run=_weighted_joint_null,
    ),
    Check(
        label=HALF_LABEL,
        title="Half-sample standard errors",
        session="14",
        supports=(
            role(
                "func", "mixed_signals.sigma.uncertainty.half_sample_draws", short=False
            ),
            role(
                "func", "mixed_signals.sigma.uncertainty.half_sample_error", short=False
            ),
            "the subsampling bootstrap of compare",
        ),
        run=_half_sample_errors,
    ),
    Check(
        label=RECOVERY_LABEL,
        title="Zoo truths through the engines",
        session="14",
        supports=(
            role("func", "mixed_signals.engines.compare.compare", short=False),
            role("func", "mixed_signals.engines.diagnose.diagnose", short=False),
            "the done criterion of session 14: the engines recover the zoo truths",
        ),
        run=_engine_recovery,
    ),
)
"""Checks of the engines."""
