"""Checks of data space: linear signals, linear order, nulls and capped sets."""

import functools
import itertools
import math
from collections.abc import Sequence
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import ndtri
from jaxtyping import Array, PRNGKeyArray

from mixed_signals import zoo
from mixed_signals.core.samples import subsample_effective_size
from mixed_signals.engines.dataspace import compare_signals
from mixed_signals.nulls.permutation import permutation_side_size
from zoo_validation.common import LEVELS, rate_measurements, rejection_count
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
    claim_rates_within,
    claim_true,
    claim_within,
    error_text,
    grouped,
    role,
)

LEVEL_COLUMNS = tuple(f"Rate at {level:.0%}" for level in LEVELS)
"""Headers of the rejection rate columns."""

LINEAR_LABEL = "dataspace-linear"
ORDER_LABEL = "dataspace-linear-order"
NULL_LABEL = "dataspace-null"
CAPPED_LABEL = "capped-effective-size"

LINEAR_VARIANCES = (1.0, 4.0, 9.0, 16.0)
"""The diagonal of the Fisher matrix of the linear signal before correlation."""
LINEAR_CORRELATION = 0.5
"""The correlation between every two parameters of the linear signal."""
LINEAR_DATA_SIZE = 64
LINEAR_SHIFTS = (0.0, 0.5, 1.0)
LINEAR_DIRECTION = (1.0, -0.5, 0.5, 1.0)
"""An oblique direction in the parameters, along which the posteriors shift."""
EQUALITY_LIMIT = 1e-9
"""Largest relative difference of the data row and the chart row of a linear signal."""
BIAS_ALLOWANCE = 0.01
"""Allowance of the mean σ_eff for the O(1/N) bias of the energy statistic."""

ORDER_SNRS = (5.0, 10.0, 30.0, 100.0)
ORDER_SHIFT = 0.5
ORDER_DIRECTION = (0.0, 1.0, 0.0)
"""The frequency axis of the sinusoid, along which the posteriors shift."""
ORDER_DATA_SIZE = 1_000
HIGH_SNR_TOLERANCE = 0.01
"""Distance of the ratio from one that a claim allows at the highest SNR."""

NULL_SAMPLES = 1_000
NULL_SNR = 30.0
NULL_LEVEL = 0.05

CAPPED_WALKERS = 24
CAPPED_STEPS = 500
CAPPED_COEFFICIENTS = (0.135, 0.3)
"""AR(1) coefficients φ of the walkers.

The integrated autocorrelation time of an AR(1) chain is τ = (1 + φ)/(1 − φ), 1.31
and 1.86 here, so n_eff/N = 1/τ. Thinning a strongly correlated AR(1) chain by its
autocorrelation time leaves φ close to e⁻² ≈ 0.135, because for a coefficient ρ near 1
the time is τ ≈ 2/(1 − ρ) and ρ^τ tends to e⁻². A coefficient of 0.3 stands for a
chain that is thinned less.
"""
CAPPED_CAPS = (2_000, 6_000, 10_000, 12_000)
CAPPED_CHUNK = 1_000
"""Replicates of one batch of simulated runs, which bounds their memory."""
CAPPED_RULES = ("design effect", "kept ratio", "no correlation")

LINEAR_SIGNAL = role("func", "mixed_signals.zoo.signals.linear_signal")
SINUSOID_SIGNAL = role("func", "mixed_signals.zoo.signals.sinusoid_signal")
FISHER_GAUSSIANS = role("meth", "mixed_signals.zoo.signals.SignalCase.fisher_gaussians")
COMPARE_SIGNALS = role("func", "mixed_signals.engines.dataspace.compare_signals")
SUBSAMPLE_SIZE = role("func", "mixed_signals.core.samples.subsample_effective_size")
SIDE_SIZE = role("func", "mixed_signals.nulls.permutation.permutation_side_size")


# * Shared: running compare_signals on the replicates of one case.


def _linear_fisher() -> np.ndarray:
    """Return the Fisher matrix with the diagonal 1, 4, 9, 16 and correlations 0.5."""
    root = np.sqrt(np.asarray(LINEAR_VARIANCES))
    correlation = np.full((root.size, root.size), LINEAR_CORRELATION)
    np.fill_diagonal(correlation, 1.0)
    fisher: np.ndarray = correlation * np.outer(root, root)
    return fisher


def _compare_replicates(
    context: Context,
    label: str,
    case: zoo.SignalCase,
    shift: float,
    direction: Sequence[float],
    *,
    count: int,
    replicates: int,
    draws: int,
    path: Sequence[str | int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return the data rows and the chart rows of ``replicates`` problems."""
    gaussians = case.fisher_gaussians(shift, direction=direction)
    samples = gaussians.sample(
        context.key(label, "samples", *path), count, batch_size=replicates
    )
    size = len(case.space)
    result = compare_signals(
        samples,
        list(gaussians.names),
        case.space,
        generator=case.generator,
        noise=case.noise,
        key=context.key(label, "compare", *path),
        reference="reference",
        fisher=np.broadcast_to(np.asarray(case.fisher), (replicates, size, size)),
        centre=np.broadcast_to(np.asarray(case.centre), (replicates, size)),
        draws=draws,
    )
    rows = result.table.to_dicts()
    return rows[0::2], rows[1::2]


def _column(rows: Sequence[dict[str, Any]], name: str) -> np.ndarray:
    """Return one numeric column of the rows as an array."""
    return np.asarray([row[name] for row in rows], dtype=np.float64)


def _largest_relative_difference(
    data: Sequence[dict[str, Any]], chart: Sequence[dict[str, Any]]
) -> float:
    """Return the largest relative difference of the observed values and null draws."""
    differences = []
    for name in ("value", "null_draws"):
        first, second = _column(data, name), _column(chart, name)
        differences.append(float(np.max(np.abs(first - second) / np.abs(second))))
    return max(differences)


# * Check dataspace-linear.


def _dataspace_linear(context: Context) -> Section:
    replicates = context.count(50, 5)
    count = context.count(2_000, 500)
    draws = context.count(39, 9)
    case = zoo.linear_signal(_linear_fisher(), data_size=LINEAR_DATA_SIZE)
    rows: list[list[Cell]] = []
    cases: list[Case] = []
    largest = 0.0
    for shift in LINEAR_SHIFTS:
        data, chart = _compare_replicates(
            context,
            LINEAR_LABEL,
            case,
            shift,
            LINEAR_DIRECTION,
            count=count,
            replicates=replicates,
            draws=draws,
            path=(f"shift-{shift:g}",),
        )
        difference = _largest_relative_difference(data, chart)
        largest = max(largest, difference)
        data_effect = Measurement.from_replicates(_column(data, "sigma_eff"))
        chart_effect = Measurement.from_replicates(_column(chart, "sigma_eff"))
        rows.append([shift, data_effect, chart_effect, difference])
        cases.append(
            Case(
                f"shift {shift:g}",
                data_effect.value - shift,
                data_effect.margin(),
                BIAS_ALLOWANCE,
            )
        )
    table = Table.of(
        f"The mean over {replicates} replicates of σ_eff of the data row and of the"
        " Fisher chart row of a linear signal, and the largest relative difference of"
        " the two rows in the observed value and in every null draw.",
        (
            "Shift",
            "σ_eff, data row",
            "σ_eff, chart row",
            "Largest relative difference",
        ),
        rows,
    )
    claims: list[Claim] = [
        claim_true(
            "On a linear signal the data row equals the Fisher chart row, in its value"
            " and in every null draw.",
            f"the largest relative difference at most {EQUALITY_LIMIT:g} (rounding of"
            " two distance matrices)",
            f"{largest:.3g}",
            largest <= EQUALITY_LIMIT,
        ),
        claim_cases(
            "The mean σ_eff of the data row equals the Mahalanobis shift of the"
            " posteriors, in units of the noise standard deviation.",
            cases,
            tolerance_text=f"{BIAS_ALLOWANCE:g} for the O(1/N) bias of the energy"
            " statistic",
            replicates=replicates,
        ),
    ]
    setup = (
        f"{LINEAR_SIGNAL} gives a signal h = Aθ in white noise of unit variance with"
        f" {LINEAR_DATA_SIZE} data entries and four parameters, whose Fisher matrix is"
        " exactly Γ, with the diagonal"
        f" {', '.join(f'{value:g}' for value in LINEAR_VARIANCES)} and the"
        f" correlation {LINEAR_CORRELATION:g} between every two parameters."
        f" {FISHER_GAUSSIANS} gives two Gaussians of covariance Γ⁻¹ a Mahalanobis"
        f" length of {', '.join(f'{shift:g}' for shift in LINEAR_SHIFTS)} apart, along"
        f" the oblique direction {LINEAR_DIRECTION}. {COMPARE_SIGNALS} runs on"
        f" {replicates} independent problems in one call, {grouped(count)} samples per"
        f" posterior and {draws} null draws, with the Fisher matrix and its centre,"
        " so each pair has a data row and a chart row. For a linear signal the"
        " templates differ by Aδθ, whose whitened norm squared is δθᵀΓδθ, so the"
        " chart is an isometry of the templates and the two rows agree to rounding."
        " At linear order the energy distance of Gaussians of the same covariance"
        " follows the curve of the Mahalanobis shift in four dimensions, so σ_eff is"
        f" the shift; a mean over {replicates} replicates may lie within three"
        " standard errors (Student t for few replicates) and"
        f" {BIAS_ALLOWANCE:g} of it, the allowance for the O(1/N) bias of the"
        " V-statistic that the null subtraction leaves."
    )
    return Section(setup, (table,), tuple(claims))


# * Check dataspace-linear-order.


def _dataspace_linear_order(context: Context) -> Section:
    replicates = context.count(50, 5)
    count = context.count(2_000, 500)
    draws = context.count(39, 9)
    value_ratios: list[Measurement] = []
    effect_ratios: list[Measurement] = []
    rows: list[list[Cell]] = []
    for snr in ORDER_SNRS:
        case = zoo.sinusoid_signal(snr, data_size=ORDER_DATA_SIZE)
        data, chart = _compare_replicates(
            context,
            ORDER_LABEL,
            case,
            ORDER_SHIFT,
            ORDER_DIRECTION,
            count=count,
            replicates=replicates,
            draws=draws,
            path=(f"snr-{snr:g}",),
        )
        value_ratio = Measurement.from_replicates(
            _column(data, "value") / _column(chart, "value")
        )
        effect_ratio = Measurement.from_replicates(
            _column(data, "sigma_eff") / _column(chart, "sigma_eff")
        )
        value_ratios.append(value_ratio)
        effect_ratios.append(effect_ratio)
        rows.append(
            [
                snr,
                Measurement.from_replicates(_column(data, "sigma_eff")),
                Measurement.from_replicates(_column(chart, "sigma_eff")),
                value_ratio,
                effect_ratio,
            ]
        )
    table = Table.of(
        f"The mean over {replicates} replicates of σ_eff of the data row and of the"
        " Fisher chart row of a sinusoid shifted by half a noise standard deviation"
        " along its frequency, with the ratio of the observed values and the ratio of"
        " the σ_eff of the two rows on the same samples.",
        (
            "Signal-to-noise ratio",
            "σ_eff, data row",
            "σ_eff, chart row",
            "Ratio of values",
            "Ratio of σ_eff",
        ),
        rows,
    )
    claims: list[Claim] = []
    for name, ratios in (("values", value_ratios), ("σ_eff", effect_ratios)):
        means = [ratio.value for ratio in ratios]
        claims.append(
            claim_true(
                f"The ratio of the {name} of the data row to the chart row rises with"
                " the signal-to-noise ratio.",
                "each mean ratio above the one at the next lower signal-to-noise ratio",
                ", ".join(ratio.render() for ratio in ratios),
                all(later > earlier for earlier, later in itertools.pairwise(means)),
                full_only=True,
            )
        )
        claims.append(
            claim_true(
                f"The ratio of the {name} is below one at the lowest signal-to-noise"
                " ratio, where the template manifold curves.",
                f"one minus the mean ratio above {error_text(replicates)}",
                f"{ratios[0].render()}, one minus the mean {1 - means[0]:.3g} against"
                f" {ratios[0].margin():.3g}",
                1 - means[0] > ratios[0].margin(),
                full_only=True,
            )
        )
        claims.append(
            claim_within(
                f"The ratio of the {name} is one at the highest signal-to-noise ratio,"
                " where the signal is linear in the parameters.",
                ratios[-1],
                1.0,
                HIGH_SNR_TOLERANCE,
                tolerance_text="the signal is linear to this order at the highest SNR",
                full_only=True,
            )
        )
    setup = (
        f"{SINUSOID_SIGNAL} gives a sinusoid A sin(2π f t + φ) on {ORDER_DATA_SIZE}"
        " times in one unit of observing time, in white noise of unit variance, at the"
        f" signal-to-noise ratios {', '.join(f'{snr:g}' for snr in ORDER_SNRS)}; its"
        " Fisher matrix comes from automatic differentiation."
        f" {FISHER_GAUSSIANS} gives two Gaussians of covariance Γ⁻¹ half a Mahalanobis"
        f" unit apart along the frequency. {COMPARE_SIGNALS} runs on {replicates}"
        f" independent problems, {grouped(count)} samples per posterior and {draws}"
        " null draws, with the Fisher matrix, so each pair has a data row, which"
        " measures the distance of the templates, and a chart row, which measures"
        " ΔθᵀΓΔθ, the same distance at linear order. The ratio of the two is one"
        " where the template manifold is flat over the posterior and falls below one"
        " where it curves, so it should rise with the signal-to-noise ratio, as a"
        " probe found (0.90, 0.97 and 0.994 at 5, 10 and 30). The claims, in the"
        " full run only, apply to the ratio of the observed values and to that of"
        " σ_eff: each mean ratio rises with the signal-to-noise ratio; the ratio at"
        f" {ORDER_SNRS[0]:g} lies below one by more than three standard errors"
        " (Student t for few replicates); and at"
        f" {ORDER_SNRS[-1]:g} it lies within {HIGH_SNR_TOLERANCE:g} of one."
    )
    return Section(setup, (table,), tuple(claims))


# * Check dataspace-null.


def _null_cases() -> tuple[tuple[str, zoo.SignalCase], ...]:
    """Return the linear case and the sinusoid at the signal-to-noise ratio 30."""
    return (
        (
            "linear signal",
            zoo.linear_signal(_linear_fisher(), data_size=LINEAR_DATA_SIZE),
        ),
        (
            f"sinusoid, SNR {NULL_SNR:g}",
            zoo.sinusoid_signal(NULL_SNR, data_size=ORDER_DATA_SIZE),
        ),
    )


def _dataspace_null(context: Context) -> Section:
    replicates = context.count(200, 20)
    draws = context.count(199, 99)
    rows: list[list[Cell]] = []
    claims: list[Claim] = []
    for name, case in _null_cases():
        data, chart = _compare_replicates(
            context,
            NULL_LABEL,
            case,
            0.0,
            (1.0,) + (0.0,) * (len(case.space) - 1),
            count=NULL_SAMPLES,
            replicates=replicates,
            draws=draws,
            path=(name,),
        )
        counts = []
        for space, replicate_rows in (("data", data), ("chart", chart)):
            p_values = jnp.asarray(_column(replicate_rows, "p_value"))
            rows.append([name, space, *rate_measurements(p_values)])
            counts.append((rejection_count(p_values, NULL_LEVEL), replicates))
        claims.append(
            claim_rates_within(
                f"On identical posteriors of the {name}, the data row and the chart"
                " row reject at the 5% level as often as the level says.",
                counts,
                (NULL_LEVEL, NULL_LEVEL),
                full_only=True,
            )
        )
    table = Table.of(
        f"The share of {replicates} pairs of identical posteriors that each row"
        " rejects at three levels, from the p-value of the row.",
        ("Case", "Row", *LEVEL_COLUMNS),
        rows,
    )
    setup = (
        f"{COMPARE_SIGNALS} compares two posteriors that are the same distribution,"
        " the Gaussians of the Fisher matrix with a shift of zero"
        f" ({FISHER_GAUSSIANS}), {grouped(NULL_SAMPLES)} samples each, for"
        f" {replicates} independent problems"
        f" and {draws} permutations per row, with the Fisher matrix. The cases are"
        f" the linear signal of the first check and the sinusoid at the"
        f" signal-to-noise ratio {NULL_SNR:g}. A row rejects when its p-value"
        " (the +1 corrected permutation p-value, with the tail fit that the package"
        " uses below ten exceedances) is at most the level. The claim, in the full"
        " run only, lets the rate at the 5% level of both rows lie within three"
        " binomial errors of 5%."
    )
    return Section(setup, (table,), tuple(claims))


# * Check capped-effective-size.


def _ar1_runs(key: PRNGKeyArray, coefficient: float) -> Array:
    """Return two runs of stationary AR(1) walkers, flattened, shaped (2, N)."""
    start_key, noise_key = jax.random.split(key)
    start = jax.random.normal(start_key, (2, CAPPED_WALKERS))
    noise = jax.random.normal(noise_key, (CAPPED_STEPS - 1, 2, CAPPED_WALKERS))
    scale = math.sqrt(1 - coefficient**2)

    def step(previous: Array, innovation: Array) -> tuple[Array, Array]:
        current = coefficient * previous + scale * innovation
        return current, current

    _, rest = jax.lax.scan(step, start, noise)
    path = jnp.concatenate([start[None], rest])
    return jnp.transpose(path, (1, 2, 0)).reshape(2, CAPPED_WALKERS * CAPPED_STEPS)


def _capped_replicate(
    key: PRNGKeyArray, coefficient: float, caps: tuple[int, ...]
) -> tuple[Array, Array]:
    """Return, per cap, the difference of the capped means and the variance of the pool.

    Each run keeps the first ``cap`` entries of one random permutation, a
    uniform subset, so the caps are nested subsets of the same run.
    """
    chain_key, first_key, second_key = jax.random.split(key, 3)
    runs = _ar1_runs(chain_key, coefficient)
    count = runs.shape[1]
    orders = (
        jax.random.permutation(first_key, count),
        jax.random.permutation(second_key, count),
    )
    differences, variances = [], []
    for cap in caps:
        first = runs[0][orders[0][:cap]]
        second = runs[1][orders[1][:cap]]
        differences.append(first.mean() - second.mean())
        variances.append(jnp.var(jnp.concatenate([first, second])))
    return jnp.stack(differences), jnp.stack(variances)


@functools.partial(jax.jit, static_argnames=("coefficient", "caps"))
def _capped_batch(
    keys: Array, coefficient: float, caps: tuple[int, ...]
) -> tuple[Array, Array]:
    """Run a batch of replicates, shaped (replicates, caps) in both outputs."""
    function = jax.vmap(_capped_replicate, in_axes=(0, None, None))
    differences, variances = function(keys, coefficient, caps)
    return differences, variances


def _rejection_rate(
    differences: Array, variances: Array, cap: int, effective_size: float
) -> tuple[int, int, int]:
    """Return the rejections at 5%, the replicates and the side size of a null.

    The permutation null of the difference of the means of two disjoint sides
    of s samples from a pool of P has the variance 2 σ² P / (s (P − 1)) for the
    variance σ² of the pool.
    """
    pool = 2 * cap
    side = permutation_side_size(
        jnp.zeros(cap),
        jnp.zeros(cap),
        first_n_eff=effective_size,
        second_n_eff=effective_size,
    )
    null_variance = 2 * variances * pool / (side * (pool - 1))
    critical = float(ndtri(1 - NULL_LEVEL / 2))
    rejected = jnp.abs(differences) > critical * jnp.sqrt(null_variance)
    return int(jnp.sum(rejected)), int(differences.shape[0]), side


def _capped_effective_size(context: Context) -> Section:
    replicates = context.count(20_000, 1_000)
    count = CAPPED_WALKERS * CAPPED_STEPS
    rows: list[list[Cell]] = []
    design_counts: list[tuple[int, int]] = []
    below: list[bool] = []
    above: list[bool] = []
    observed_below: list[str] = []
    observed_above: list[str] = []
    for coefficient in CAPPED_COEFFICIENTS:
        correlation_time = (1 + coefficient) / (1 - coefficient)
        effective_size = count / correlation_time
        keys = jax.random.split(
            context.key(CAPPED_LABEL, "runs", f"{coefficient:g}"), replicates
        )
        parts = [
            _capped_batch(chunk, coefficient, CAPPED_CAPS)
            for chunk in jnp.split(keys, replicates // CAPPED_CHUNK)
        ]
        differences = jnp.concatenate([part[0] for part in parts])
        variances = jnp.concatenate([part[1] for part in parts])
        for position, cap in enumerate(CAPPED_CAPS):
            sizes = (
                subsample_effective_size(effective_size, float(count), float(cap)),
                cap * effective_size / count,
                float(cap),
            )
            for rule, size in zip(CAPPED_RULES, sizes, strict=True):
                rejected, trials, side = _rejection_rate(
                    differences[:, position], variances[:, position], cap, size
                )
                measurement = Measurement.from_rate(rejected, trials, level=NULL_LEVEL)
                rows.append(
                    [
                        coefficient,
                        effective_size / count,
                        cap,
                        rule,
                        side,
                        measurement,
                    ]
                )
                margin = 3 * (measurement.error or 0.0)
                label = f"φ {coefficient:g}, cap {grouped(cap)}"
                if rule == CAPPED_RULES[0]:
                    design_counts.append((rejected, trials))
                elif rule == CAPPED_RULES[1] and 2 * cap <= count:
                    below.append(measurement.value < NULL_LEVEL - margin)
                    observed_below.append(f"{label}: {measurement.value:.2%}")
                elif rule == CAPPED_RULES[2] and 2 * cap >= count:
                    above.append(measurement.value > NULL_LEVEL + margin)
                    observed_above.append(f"{label}: {measurement.value:.2%}")
    table = Table.of(
        f"The share of {grouped(replicates)} pairs of identical capped runs that the"
        " permutation null rejects at the 5% level, with the side size that the null"
        " takes from the effective size under three rules.",
        (
            "Coefficient φ",
            "n_eff/N",
            "Cap",
            "Rule",
            "Side size",
            "Rejections at 5%",
        ),
        rows,
    )
    claims = [
        claim_rates_within(
            "With the design effect, the permutation null of a capped chain rejects at"
            " 5% as often as the level says, at every cap.",
            design_counts,
            (NULL_LEVEL,) * len(design_counts),
            full_only=True,
        ),
        claim_true(
            "Keeping the ratio m n_eff/N makes the null too wide where the cap keeps at"
            " most half of the samples, so there it rejects below 5% by more than"
            " three binomial errors.",
            "every rate at a cap up to N/2 below 5% by more than three binomial errors",
            "; ".join(observed_below),
            all(below),
            full_only=True,
        ),
        claim_true(
            "Ignoring the correlation makes the null too narrow where the cap keeps at"
            " least half of the samples, so there it rejects above 5% by more than"
            " three binomial errors.",
            "every rate at a cap from N/2 up above 5% by more than three binomial"
            " errors",
            "; ".join(observed_above),
            all(above),
            full_only=True,
        ),
    ]
    setup = (
        f"Two runs of {CAPPED_WALKERS} walkers of {CAPPED_STEPS} steps share one"
        " distribution and are flattened to"
        f" N = {grouped(count)} samples each. Each walker is a stationary AR(1)"
        " chain with the coefficient φ, whose integrated autocorrelation time is"
        " τ = (1 + φ)/(1 − φ), so n_eff/N = 1/τ. Thinning a strongly correlated"
        " AR(1) chain by its autocorrelation time leaves φ close to e⁻² ≈ 0.135,"
        " because for a coefficient ρ near 1 the time is τ ≈ 2/(1 − ρ) and ρ^τ tends"
        " to e⁻²; φ = 0.3 stands for a chain that is thinned less. The coefficients"
        f" are {', '.join(f'{value:g}' for value in CAPPED_COEFFICIENTS)}. Each run is"
        " capped to a uniform subset of"
        f" {', '.join(grouped(cap) for cap in CAPPED_CAPS)} samples and the two are"
        " pooled. For the difference of the means, the permutation null of two"
        " disjoint sides of s samples from a pool of P has the variance 2 σ² P / (s"
        " (P − 1)) for the pool variance σ², so the two-sided rejection at 5% follows"
        " without drawing permutations, from the normal quantile. The side size s is"
        f" that of {SIDE_SIZE} for the effective size of a rule: the design effect"
        f" ({SUBSAMPLE_SIZE}), the kept ratio m n_eff/N, and the Kish size m, which"
        " ignores the correlation. Each wrong rule is exact in one limit: the kept"
        " ratio gives n_eff without a cap (m = N), and the Kish size gives the size of"
        " a nearly independent subset for a cap far below N. The design effect joins"
        " the two limits. The claims therefore test each wrong rule away from its own"
        f" limit. {grouped(replicates)} replicates per coefficient give the rates, each"
        " with the binomial error at 5%. The claims, in the full run only, are: the"
        " design effect within three binomial errors of 5% at every cap; the kept"
        " ratio below 5% by more than three errors where the cap keeps at most half of"
        " the samples; and the Kish size above 5% by more than three errors where the"
        " cap keeps at least half of the samples."
    )
    return Section(setup, (table,), tuple(claims))


CHECKS: tuple[Check, ...] = (
    Check(
        label=LINEAR_LABEL,
        title="Data space and Fisher chart of a linear signal",
        session="15",
        supports=(
            role(
                "func", "mixed_signals.engines.dataspace.compare_signals", short=False
            ),
            role("class", "mixed_signals.dataspace.chart.FisherChart", short=False),
            "the done criterion of session 15: on a linear signal the data row and"
            " the Fisher chart row agree, and σ_eff recovers the shift",
        ),
        run=_dataspace_linear,
    ),
    Check(
        label=ORDER_LABEL,
        title="The linear-order limit on a sinusoid",
        session="15",
        supports=(
            role(
                "func", "mixed_signals.engines.dataspace.compare_signals", short=False
            ),
            "the data row reading below the chart row at low signal-to-noise ratio",
        ),
        run=_dataspace_linear_order,
    ),
    Check(
        label=NULL_LABEL,
        title="Null calibration of compare_signals",
        session="15",
        supports=(
            role(
                "func", "mixed_signals.engines.dataspace.compare_signals", short=False
            ),
            "the permutation null of the data row and the chart row",
        ),
        run=_dataspace_null,
    ),
    Check(
        label=CAPPED_LABEL,
        title="Effective size of a capped chain",
        session="15",
        supports=(
            role(
                "func",
                "mixed_signals.core.samples.subsample_effective_size",
                short=False,
            ),
            "the effective size of the null sides of a capped set in data space",
        ),
        run=_capped_effective_size,
    ),
)
"""Checks of data space."""
