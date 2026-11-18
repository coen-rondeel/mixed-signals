"""Checks of the diagnostics: surrogate nulls, KSG, BIC, modes, chains and flags."""

import functools
import math
import warnings
from collections.abc import Callable, Sequence
from typing import NamedTuple, cast

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from jaxtyping import Array, PRNGKeyArray

from mixed_signals import errors, zoo
from mixed_signals.core.samples import circular_mean, unwrap_about
from mixed_signals.core.space import Parameter, ParamSpace
from mixed_signals.diagnostics import (
    MODE_FLOOR,
    MODE_SEPARATION,
    PAIR_CAP,
    THRESHOLDS,
    anderson_darling_normality,
    circle_centres,
    dependence_statistics,
    discriminant_directions,
    evaluate,
    gaussian_distance_correlation,
    gaussian_mutual_information,
    gaussian_spearman,
    gmm_bic,
    mardia_moments,
    marginal_modes,
    marginal_moments,
    moment_metric_names,
    parameter_pairs,
    projection_modes,
    railed_parameters,
    railing_scores,
)
from mixed_signals.diagnostics.modes import MAX_COMPONENTS, _grid, _valley_step
from mixed_signals.estimators.kde import KernelDensity
from mixed_signals.mcmc import split_chain_kuiper
from mixed_signals.nulls.surrogates import gaussian_surrogates, surrogate_size
from mixed_signals.zoo.closed_forms import _LEGENDRE_NODES, _LEGENDRE_WEIGHTS
from zoo_validation.common import (
    LEVELS,
    ResolutionRuns,
    pooled_resolution,
    rate_counts,
    rate_measurements,
    real_space,
    resolution_ratio,
)
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
    error_text,
    grouped,
    role,
)

SHAPE_RECORD = "0.8 to 1.2"
RAW_RECORD = "0.4 to 0.5"
EXCESS_RECORD = "0.96 to 1.33"


def _spread_measurement(values: Array) -> Measurement:
    """Return the standard deviation of the finite values with its standard error."""
    finite = values[jnp.isfinite(values)]
    count = int(finite.shape[0])
    spread = float(jnp.std(finite, ddof=1))
    return Measurement(spread, spread / math.sqrt(2 * (count - 1)), count)


def _mean_measurement(values: Array) -> Measurement:
    """Return the mean of the finite values with its standard error."""
    return Measurement.from_replicates(values[jnp.isfinite(values)])


def _limits_claim(
    statement: str,
    entries: Sequence[tuple[str, Measurement]],
    *,
    low: float | None,
    high: float,
    expected: str,
) -> Claim:
    """State that every entry lies in a band, or below a limit, without a margin.

    The observed text names the entry with the least room, with its error.
    """

    def room(measurement: Measurement) -> float:
        if not math.isfinite(measurement.value):
            return -math.inf
        above = high - measurement.value
        return above if low is None else min(measurement.value - low, above)

    worst_name, worst = min(entries, key=lambda entry: room(entry[1]))
    return claim_true(
        statement,
        expected,
        f"worst of {len(entries)}: {worst_name}: {worst.render()}",
        all(room(measurement) >= 0 for _, measurement in entries),
        full_only=True,
    )


class _Entry(NamedTuple):
    """One statistic of the calibration: its name, its form and the recorded spread."""

    name: str
    group: str
    record: str


# * Check surrogate-calibration.

CALIBRATION_LABEL = "surrogate-calibration"
CALIBRATION_CORRELATION = 0.6
SPREAD_BAND = (0.7, 1.4)
RAW_SPREAD_LIMIT = 0.6
PEARSON_SPREAD_LIMIT = 0.3
RAW_FORMS = (
    ("Spearman", "spearman"),
    ("distance correlation", "distance_correlation"),
    ("mutual information", "mutual_information"),
)


def _calibration_entries(space: ParamSpace) -> list[_Entry]:
    """Name the statistics of :func:`_calibration_vector`, in its order."""
    names = space.names
    pairs = [
        f"{names[first]}-{names[second]}"
        for first, second in parameter_pairs(len(space))
    ]
    entries = [
        _Entry(f"{kind} {name}", "shape", SHAPE_RECORD)
        for kind in ("skewness", "kurtosis", "Anderson-Darling")
        for name in names
    ]
    entries += [_Entry("Mardia b₁", "shape", SHAPE_RECORD)]
    entries += [_Entry("Mardia b₂", "shape", SHAPE_RECORD)]
    entries += [_Entry(f"Pearson r {pair}", "pearson", "0.12") for pair in pairs]
    entries += [
        _Entry(f"{label}, raw {pair}", "raw", RAW_RECORD)
        for label, _ in RAW_FORMS
        for pair in pairs
    ]
    entries += [
        _Entry(f"{label}, excess {pair}", "excess", EXCESS_RECORD)
        for label, _ in RAW_FORMS
        for pair in pairs
    ]
    return entries


def _calibration_vector(samples: Array, key: PRNGKeyArray, space: ParamSpace) -> Array:
    """Return every statistic of one set: shape rows, raw and excess dependence rows."""
    moments = marginal_moments(samples, None, space)
    distances = anderson_darling_normality(samples, None, space)
    mardia = mardia_moments(samples, None, space, key=key)
    dependence = dependence_statistics(samples, None, space, key=key)
    pearson = dependence.pearson
    # * The package returns excesses, so a raw value adds back the Gaussian at r.
    raw = [
        dependence.spearman + gaussian_spearman(pearson),
        dependence.distance_correlation + gaussian_distance_correlation(pearson),
        dependence.mutual_information + gaussian_mutual_information(pearson),
    ]
    excess = [
        dependence.spearman,
        dependence.distance_correlation,
        dependence.mutual_information,
    ]
    return jnp.concatenate(
        [
            moments.skewness,
            moments.kurtosis,
            distances,
            jnp.stack([mardia.skewness, mardia.kurtosis]),
            pearson,
            *raw,
            *excess,
        ]
    )


_calibration_observed = jax.jit(_calibration_vector, static_argnames=("space",))


@functools.partial(jax.jit, static_argnames=("space",))
def _calibration_null(surrogates: Array, key: PRNGKeyArray, space: ParamSpace) -> Array:
    """Return the statistics of every surrogate set, shaped (draws, statistics)."""
    return cast(
        Array,
        jax.lax.map(lambda values: _calibration_vector(values, key, space), surrogates),
    )


def _z_scores(observed: Array, null: Array) -> Array:
    """Return (value minus null mean) over the null standard deviation."""
    return (observed - jnp.mean(null, axis=0)) / jnp.std(null, axis=0, ddof=1)


def _surrogate_calibration(context: Context) -> Section:
    label = CALIBRATION_LABEL
    replicates = context.count(30, 2)
    draws = context.count(100, 5)
    sample_count = context.count(PAIR_CAP, 500)
    covariance = jnp.array(
        [[1.0, CALIBRATION_CORRELATION], [CALIBRATION_CORRELATION, 1.0]]
    )
    case = zoo.gaussians(jnp.zeros((1, 2)), covariance[None], names=["gaussian"])
    space = case.space
    sets = case.sample(
        context.key(label, "sets"), sample_count, batch_size=replicates
    ).samples[:, 0]
    entries = _calibration_entries(space)
    size = surrogate_size(float(sample_count), cap=PAIR_CAP)
    subsample_key = context.key(label, "subsample")
    scores = []
    for replicate in range(replicates):
        samples = sets[replicate]
        surrogates = gaussian_surrogates(
            samples,
            None,
            space,
            size=size,
            key=context.key(label, "surrogates", replicate),
            draws=draws,
        )
        observed = _calibration_observed(samples, subsample_key, space)
        null = _calibration_null(surrogates, subsample_key, space)
        scores.append(_z_scores(observed, null))
    z_values = jnp.stack(scores)
    missing = int(jnp.sum(~jnp.isfinite(z_values)))
    means = [_mean_measurement(z_values[:, index]) for index in range(len(entries))]
    spreads = [_spread_measurement(z_values[:, index]) for index in range(len(entries))]
    table = Table.of(
        f"Calibration of the Gaussian surrogate null: the mean and the spread of z ="
        f" (value minus surrogate mean) over the surrogate standard deviation, over"
        f" {replicates} replicates of {grouped(sample_count)} samples with {draws}"
        " surrogates each, with the standard errors, and the spread that session 11"
        " recorded for the group.",
        ("Statistic", "Form", "Mean z", "Spread of z", "Record, spread"),
        [
            [entry.name, entry.group, means[index], spreads[index], entry.record]
            for index, entry in enumerate(entries)
        ],
    )
    by_group: dict[str, list[tuple[str, Measurement]]] = {}
    for index, entry in enumerate(entries):
        by_group.setdefault(entry.group, []).append((entry.name, spreads[index]))
    low, high = SPREAD_BAND
    band_text = f"between {low:g} and {high:g}"
    claims = [
        _limits_claim(
            "The spread of z of every shape statistic is that of a calibrated null.",
            by_group["shape"],
            low=low,
            high=high,
            expected=band_text,
        ),
        _limits_claim(
            "The spread of z of every excess dependence statistic is that of a"
            " calibrated null.",
            by_group["excess"],
            low=low,
            high=high,
            expected=band_text,
        ),
        _limits_claim(
            "The raw Spearman, distance correlation and mutual information sit at"
            " the centre of their surrogate null.",
            by_group["raw"],
            low=None,
            high=RAW_SPREAD_LIMIT,
            expected=f"at most {RAW_SPREAD_LIMIT:g}",
        ),
        _limits_claim(
            "The raw Pearson correlation sits at the centre of its surrogate null.",
            by_group["pearson"],
            low=None,
            high=PEARSON_SPREAD_LIMIT,
            expected=f"at most {PEARSON_SPREAD_LIMIT:g}",
        ),
    ]
    spread_error = 100 / math.sqrt(2 * (replicates - 1))
    missing_text = (
        "No z was NaN or infinite."
        if missing == 0
        else f"{missing} values of z were NaN or infinite and are left out."
    )
    setup = (
        f"{replicates} replicates of {grouped(sample_count)} samples from a zoo"
        f" Gaussian ({role('func', 'mixed_signals.zoo.cases.gaussians')}) with"
        f" two parameters of correlation {CALIBRATION_CORRELATION:g}, so that the"
        " raw dependence statistics do not vanish. For each replicate,"
        f" {role('func', 'mixed_signals.nulls.surrogates.gaussian_surrogates')}"
        f" draws {draws} surrogate sets of the replicate's effective sample size,"
        f" which is {grouped(sample_count)} for independent draws and at most the"
        f" cap of {grouped(PAIR_CAP)} of the pairwise statistics. Every statistic of"
        " the replicate and of each surrogate set comes from the package: the"
        " marginal moments, the Anderson-Darling distance, Mardia's moments and the"
        " dependence rows. The dependence rows are returned as excesses over their"
        " Gaussian values at the set's own r, so a raw value adds that Gaussian"
        " value back. The spread of z over the replicates is a calibration: a"
        f" calibrated null gives one. A spread from {replicates} replicates has a"
        f" standard error of about {spread_error:.0f}%, which the table states for"
        " every row; the claims read each spread against its band as the record"
        " states it, without a margin for that error. "
        f"{missing_text}"
    )
    return Section(setup, (table,), tuple(claims))


# * Check circle-drift.

CIRCLE_LABEL = "circle-drift"
CIRCLE_WIDTHS = (0.25, 0.5, 0.75, 1.0, 1.25, 1.5)
CIRCLE_CORRELATION = 0.5
MEAN_BAND = 0.5
PEARSON_CLAIMED_WIDTH = 0.75
"""Largest width, in radians, up to which the mean z of Pearson's r is claimed."""
CIRCLE_STATISTICS = (
    ("Pearson r", "linear", "pearson"),
    ("Spearman excess", "linear", "spearman"),
    ("skewness, unwrapped", "linear", "skewness"),
    ("kurtosis, unwrapped", "linear", "kurtosis"),
    ("circular skewness", "circular", "circular_skewness"),
    ("circular kurtosis", "circular", "circular_kurtosis"),
)
"""Each statistic, whether it is linear on unwrapped values, and its flag metric."""
LINE_SPACE = real_space(1)


def _circle_vector(samples: Array, key: PRNGKeyArray, space: ParamSpace) -> Array:
    """Return the linear statistics on unwrapped values and the circular moments."""
    dependence = dependence_statistics(samples, None, space, key=key)
    lower, upper = space.parameters[0].bounds
    period = upper - lower
    centre = circular_mean(samples[:, 0], jnp.zeros(samples.shape[0]), period, lower)
    unwrapped = unwrap_about(samples[:, 0], centre, period)
    linear = marginal_moments(unwrapped[:, None], None, LINE_SPACE)
    circular = marginal_moments(samples, None, space)
    return jnp.stack(
        [
            dependence.pearson[0],
            dependence.spearman[0],
            linear.skewness[0],
            linear.kurtosis[0],
            circular.skewness[0],
            circular.kurtosis[0],
        ]
    )


@functools.partial(jax.jit, static_argnames=("space",))
def _circle_null(surrogates: Array, key: PRNGKeyArray, space: ParamSpace) -> Array:
    """Return the statistics of every surrogate set, shaped (draws, statistics)."""
    return cast(
        Array,
        jax.lax.map(lambda values: _circle_vector(values, key, space), surrogates),
    )


def _circle_drift(context: Context) -> Section:
    label = CIRCLE_LABEL
    replicates = context.count(100, 2)
    draws = context.count(100, 5)
    sample_count = context.count(PAIR_CAP, 500)
    z_by_width: dict[float, Array] = {}
    warning_counts: dict[float, int] = {}
    subsample_key = context.key(label, "subsample")
    for index, width in enumerate(CIRCLE_WIDTHS):
        case = zoo.wrapped_gaussians(
            0.0, width=width, param_count=2, correlation=CIRCLE_CORRELATION
        )
        space = case.space
        sets = case.sample(
            context.key(label, "sets", index), sample_count, batch_size=replicates
        ).samples[:, 0]
        scores = []
        counted = 0
        for replicate in range(replicates):
            samples = sets[replicate]
            # * The observed set runs outside jit, so that the package warns.
            with errors.capture_warnings() as records:
                observed = _circle_vector(samples, subsample_key, space)
            counted += len(records)
            for record in records:
                warnings.warn(
                    getattr(errors, record.category)(record.message), stacklevel=1
                )
            surrogates = gaussian_surrogates(
                samples,
                None,
                space,
                size=sample_count,
                key=context.key(label, "surrogates", index, replicate),
                draws=draws,
            )
            scores.append(
                _z_scores(observed, _circle_null(surrogates, subsample_key, space))
            )
        z_by_width[width] = jnp.stack(scores)
        warning_counts[width] = counted
    names = [name for name, _, _ in CIRCLE_STATISTICS]
    means = {
        width: [_mean_measurement(z_values[:, index]) for index in range(len(names))]
        for width, z_values in z_by_width.items()
    }
    spreads = {
        width: [_spread_measurement(z_values[:, index]) for index in range(len(names))]
        for width, z_values in z_by_width.items()
    }
    width_columns = tuple(f"{width:g} rad" for width in CIRCLE_WIDTHS)
    mean_table = Table.of(
        "Mean of z of each statistic by circular standard deviation, with the"
        " standard error of the mean.",
        ("Statistic", *width_columns),
        [
            [name, *[means[width][index] for width in CIRCLE_WIDTHS]]
            for index, name in enumerate(names)
        ],
    )
    spread_table = Table.of(
        "Spread of z of each statistic by circular standard deviation, with its"
        " standard error.",
        ("Statistic", *width_columns),
        [
            [name, *[spreads[width][index] for width in CIRCLE_WIDTHS]]
            for index, name in enumerate(names)
        ],
    )
    warning_table = Table.of(
        "Package warnings raised by the statistics of the observed sets, by"
        " circular standard deviation.",
        ("Circular standard deviation", "Warnings"),
        [[f"{width:g} rad", warning_counts[width]] for width in CIRCLE_WIDTHS],
    )
    limit = 1.0
    below = [width for width in CIRCLE_WIDTHS if width <= limit]
    widest = CIRCLE_WIDTHS[-1]
    linear = [
        index
        for index, (_, kind, _) in enumerate(CIRCLE_STATISTICS)
        if kind == "linear"
    ]
    circular = [
        index
        for index, (_, kind, _) in enumerate(CIRCLE_STATISTICS)
        if kind == "circular"
    ]
    # * Only a statistic with a flag threshold can flag; Pearson's r has none.
    flaggable = [index for index in linear if CIRCLE_STATISTICS[index][2] in THRESHOLDS]
    informational = [index for index in linear if index not in flaggable]
    pearson_widths = [
        width for width in CIRCLE_WIDTHS if width <= PEARSON_CLAIMED_WIDTH
    ]
    low, high = SPREAD_BAND
    centre, half_width = 0.5 * (low + high), 0.5 * (high - low)

    def mean_cases(widths: Sequence[float], indices: Sequence[int]) -> list[Case]:
        return [
            Case(
                f"{names[index]} at {width:g} rad",
                means[width][index].value,
                means[width][index].margin(),
                MEAN_BAND,
            )
            for width in widths
            for index in indices
        ]

    def spread_cases(widths: Sequence[float], indices: Sequence[int]) -> list[Case]:
        return [
            Case(
                f"{names[index]} at {width:g} rad",
                spreads[width][index].value - centre,
                spreads[width][index].margin(),
                half_width,
            )
            for width in widths
            for index in indices
        ]

    def leaves_the_band(index: int) -> bool:
        mean, spread = means[widest][index], spreads[widest][index]
        outside_mean = abs(mean.value) > MEAN_BAND + mean.margin()
        outside_spread = (
            index in flaggable
            and abs(spread.value - centre) > half_width + spread.margin()
        )
        return outside_mean or outside_spread

    departing = [names[index] for index in linear if leaves_the_band(index)]
    pearson_at_limit = ", ".join(
        means[limit][index].render() for index in informational
    )
    spread_text = f"{low:g} to {high:g}"
    claims = [
        claim_cases(
            f"At and below {limit:g} rad, the mean of z of every linear statistic"
            " that can flag stays near zero.",
            mean_cases(below, flaggable),
            tolerance_text=f"±{MEAN_BAND:g}",
            replicates=replicates,
        ),
        claim_cases(
            f"Up to {PEARSON_CLAIMED_WIDTH:g} rad, the mean of z of Pearson's r stays"
            " near zero.",
            mean_cases(pearson_widths, informational),
            tolerance_text=f"±{MEAN_BAND:g}",
            replicates=replicates,
        ),
        claim_cases(
            f"At and below {limit:g} rad, the spread of z of every linear statistic"
            f" that can flag lies in {spread_text}.",
            spread_cases(below, flaggable),
            tolerance_text=f"the band {spread_text}",
            replicates=replicates,
        ),
        claim_true(
            f"At {widest:g} rad, at least one linear statistic leaves the band: a mean"
            f" of z beyond ±{MEAN_BAND:g} or a spread outside {spread_text}, by"
            " more than"
            " the standard errors.",
            f"at least one of {len(linear)} statistics outside the band by more than"
            f" {error_text(replicates)}",
            (
                f"{len(departing)} of {len(linear)}: {', '.join(departing)}"
                if departing
                else f"0 of {len(linear)}"
            ),
            bool(departing),
            full_only=True,
        ),
        claim_cases(
            "The mean of z of the circular moments stays near zero at every width.",
            mean_cases(CIRCLE_WIDTHS, circular),
            tolerance_text=f"±{MEAN_BAND:g}",
            replicates=replicates,
        ),
        claim_cases(
            f"The spread of z of the circular moments lies in {spread_text} at every"
            " width.",
            spread_cases(CIRCLE_WIDTHS, circular),
            tolerance_text=f"the band {spread_text}",
            replicates=replicates,
        ),
    ]
    setup = (
        "Wrapped normals with a real parameter"
        f" ({role('func', 'mixed_signals.zoo.cases.wrapped_gaussians')}, two"
        f" parameters, correlation {CIRCLE_CORRELATION:g}), circular standard"
        f" deviations {', '.join(f'{width:g}' for width in CIRCLE_WIDTHS)} rad,"
        f" {grouped(sample_count)} samples, {replicates} replicates per width and"
        f" {draws} Gaussian surrogates per replicate. The linear statistics are"
        " Pearson's r, the excess of Spearman's ρ_s, and the skewness and kurtosis,"
        " all of values unwrapped about the circular mean; the circular statistics"
        " are the circular skewness and kurtosis of the package. The observed set"
        " of each replicate runs outside jit, so that the package raises its"
        f" warning above the limit of"
        f" {role('data', 'mixed_signals.diagnostics._common.CIRCLE_WIDTH_LIMIT')} rad;"
        " the table counts them. Session 11 recorded linear statistics that kept"
        " their calibration up to 1 rad and drifted from 1.25 rad, and circular"
        " moments that stayed calibrated at every width. The linear statistics that"
        " can flag are those with a threshold in"
        f" {role('data', 'mixed_signals.diagnostics.flags.THRESHOLDS')}:"
        f" {', '.join(names[index] for index in flaggable)}. Pearson's r has no"
        " threshold and the surrogates reproduce it by construction, so only its"
        f" mean of z enters the claims, and only up to {PEARSON_CLAIMED_WIDTH:g} rad:"
        f" at {limit:g} rad its mean of z is {pearson_at_limit}, which biases the"
        " σ_sig of an informational row, and of no row that can flag. The developers"
        f" kept the limit at {limit:g} rad for that reason. The claims allow the"
        " standard errors of the means and of the spreads."
    )
    return Section(setup, (mean_table, spread_table, warning_table), tuple(claims))


# * Check ksg-bananas.

KSG_LABEL = "ksg-bananas"
BANANA_VARIANCE = 100.0
TWISTS = (0.001, 0.002, 0.003, 0.01, 0.03, 0.1)
FAINT_TWIST = 0.003
"""The banana of about 0.08 nats, whose resolution the check compares."""
KSG_SIZES = (2_000, 1_000)
KSG_RANKS = (10, 3)
KSG_TOLERANCE = 0.10
LEGENDRE_NODES = jnp.asarray(_LEGENDRE_NODES)
LEGENDRE_WEIGHTS = jnp.asarray(_LEGENDRE_WEIGHTS)
QUADRATURE_Z_MAX = 7.0
"""Largest z of the quadrature, where exp(−z²/2) is 2e-11."""
QUADRATURE_PANEL = 0.2
"""Width of a panel along y at the largest z, in standard deviations of the noise."""
QUADRATURE_STEP = 0.05
"""Step of the uniform grid along y."""
QUADRATURE_MARGIN = 8.0
"""Standard deviations of the noise that the grid reaches beyond the support."""
QUADRATURE_CHUNK = 512


def banana_mutual_information(
    product: float,
    *,
    refinement: float = 1.0,
) -> float:
    """Return the mutual information of the zoo's banana, in nats, by quadrature.

    The banana twists x₁ ~ N(0, v) into y₂ = x₂ + b(x₁² − v), so y₂ = x₂ +
    t(z² − 1) with z standard normal and t = b·v. Given y₁ = x₁, y₂ is a
    unit normal at a known place, so I(y₁; y₂) = h(y₂) − ½ ln(2πe). The density
    of y₂ is p(y) = E_z[φ(y − t(z² − 1))]. It comes from composite 8-point
    Gauss-Legendre panels in z whose edges are uniform in z², each spanning a
    shift of at most ``QUADRATURE_PANEL`` along y, and h(y₂) from the trapezoid
    rule on a uniform grid, which converges geometrically for a smooth density.

    Args:
        product: The twist times the variance, t, positive.
        refinement: A factor that divides the panel width and the grid step, to
            test the accuracy of the quadrature.

    Returns:
        The mutual information in nats.
    """
    panel = QUADRATURE_PANEL / refinement
    step = QUADRATURE_STEP / refinement
    z_squared_step = min(panel / product, 1.0)
    panels = math.ceil(QUADRATURE_Z_MAX**2 / z_squared_step)
    edges = jnp.sqrt(z_squared_step * jnp.arange(panels + 1))
    half = 0.5 * (edges[1:] - edges[:-1])
    middle = 0.5 * (edges[1:] + edges[:-1])
    nodes = (middle[:, None] + half[:, None] * LEGENDRE_NODES[None, :]).reshape(-1)
    weights = (half[:, None] * LEGENDRE_WEIGHTS[None, :]).reshape(-1)
    # * The factor two counts z and −z, which give the same shift.
    mass = 2.0 * weights * jnp.exp(-0.5 * nodes**2) / math.sqrt(2 * math.pi)
    shifts = product * (nodes**2 - 1.0)
    y_start = -product - QUADRATURE_MARGIN
    y_stop = product * (QUADRATURE_Z_MAX**2 - 1.0) + QUADRATURE_MARGIN
    count = int((y_stop - y_start) / step) + 1
    chunks = -(-count // QUADRATURE_CHUNK)
    grid = y_start + step * jnp.arange(chunks * QUADRATURE_CHUNK)

    def density(points: Array) -> Array:
        kernels = jnp.exp(-0.5 * (points[:, None] - shifts[None, :]) ** 2)
        return jnp.sum(mass[None, :] * kernels, axis=1) / math.sqrt(2 * math.pi)

    values = jax.lax.map(density, grid.reshape(chunks, QUADRATURE_CHUNK)).reshape(-1)
    # * The padding past the grid's stop holds negligible density and no weight.
    inside = jnp.arange(chunks * QUADRATURE_CHUNK) < count
    entropy = -step * jnp.sum(
        jnp.where(inside & (values > 0), values * jnp.log(values), 0.0)
    )
    return float(entropy - 0.5 * math.log(2 * math.pi * math.e))


@functools.partial(jax.jit, static_argnames=("space",))
def _ksg_values(samples: Array, key: PRNGKeyArray, space: ParamSpace) -> Array:
    """Return the KSG excesses at each rank, then the estimates at each rank."""
    excesses, estimates = [], []
    for rank in KSG_RANKS:
        dependence = dependence_statistics(
            samples, None, space, key=key, neighbours=rank
        )
        excess = dependence.mutual_information[0]
        excesses.append(excess)
        estimates.append(excess + gaussian_mutual_information(dependence.pearson[0]))
    return jnp.stack([*excesses, *estimates])


@functools.partial(jax.jit, static_argnames=("space",))
def _ksg_null(surrogates: Array, key: PRNGKeyArray, space: ParamSpace) -> Array:
    """Return the KSG values of every surrogate set, shaped (draws, values)."""
    return cast(
        Array,
        jax.lax.map(lambda values: _ksg_values(values, key, space), surrogates),
    )


class _Faint(NamedTuple):
    """The replicates of the faint banana at the first size, for the resolution."""

    sets: Array
    space: ParamSpace
    observed: Array
    """The excess of the KSG estimate of each replicate at each rank."""


def _ksg_bananas(context: Context) -> Section:
    label = KSG_LABEL
    replicates = context.count(30, 2)
    resolution_replicates = context.count(16, 2)
    draws = context.count(30, 5)
    sizes = KSG_SIZES
    truths = {
        twist: banana_mutual_information(twist * BANANA_VARIANCE) for twist in TWISTS
    }
    accuracy = max(
        abs(
            banana_mutual_information(twist * BANANA_VARIANCE, refinement=2.0)
            - truths[twist]
        )
        for twist in TWISTS
    )
    ratios: dict[tuple[float, int, int], Measurement] = {}
    faint: _Faint | None = None
    for twist_index, twist in enumerate(TWISTS):
        case = zoo.banana(twist, variance=BANANA_VARIANCE)
        space = case.space
        for size in sizes:
            sets = case.sample(
                context.key(label, "sets", twist_index, size),
                size,
                batch_size=replicates,
            ).samples[:, case.names.index("banana")]
            subsample_key = context.key(label, "subsample", twist_index, size)
            values = jnp.stack(
                [
                    _ksg_values(sets[replicate], subsample_key, space)
                    for replicate in range(replicates)
                ]
            )
            for position, rank in enumerate(KSG_RANKS):
                ratios[(twist, size, rank)] = Measurement.from_replicates(
                    values[:, len(KSG_RANKS) + position] / truths[twist]
                )
            if twist == FAINT_TWIST and size == sizes[0]:
                faint = _Faint(sets, space, values[:, :2])
    runs: dict[int, tuple[list[float], list[float]]] = {
        rank: ([], []) for rank in KSG_RANKS
    }
    if faint is None:
        raise ValueError("the faint banana is not among the twists and sizes")
    for replicate in range(resolution_replicates):
        samples = faint.sets[replicate]
        surrogates = gaussian_surrogates(
            samples,
            None,
            faint.space,
            size=surrogate_size(float(sizes[0]), cap=PAIR_CAP),
            key=context.key(label, "surrogates", replicate),
            draws=draws,
        )
        null = _ksg_null(
            surrogates, context.key(label, "subsample", "null"), faint.space
        )
        for position, rank in enumerate(KSG_RANKS):
            excess_null = null[:, position]
            runs[rank][0].append(
                float(faint.observed[replicate, position] - jnp.mean(excess_null))
            )
            runs[rank][1].append(float(jnp.var(excess_null, ddof=1)))
    resolution = {
        rank: pooled_resolution(ResolutionRuns(*runs[rank])) for rank in KSG_RANKS
    }
    advantage = resolution_ratio(
        ResolutionRuns(*runs[KSG_RANKS[0]]), ResolutionRuns(*runs[KSG_RANKS[1]])
    )
    ratio_table = Table.of(
        f"KSG estimate over the true mutual information, mean of {replicates}"
        " replicates with the standard error, by twist, sample count and rank k."
        " The true value is the quadrature of the script.",
        (
            "Twist b",
            "True mutual information (nats)",
            "N",
            *[f"k = {rank}" for rank in KSG_RANKS],
        ),
        [
            [
                twist,
                truths[twist],
                size,
                *[ratios[(twist, size, rank)] for rank in KSG_RANKS],
            ]
            for twist in TWISTS
            for size in sizes
        ],
    )
    resolution_rows: list[list[Cell]] = [[rank, resolution[rank]] for rank in KSG_RANKS]
    resolution_rows.append(
        [f"Ratio of k = {KSG_RANKS[0]} to k = {KSG_RANKS[1]}", advantage]
    )
    resolution_table = Table.of(
        f"Resolution of the banana of twist {FAINT_TWIST:g}, whose mutual"
        f" information is {truths[FAINT_TWIST]:.4f} nats, at {grouped(sizes[0])}"
        f" samples: (value minus surrogate mean) over the surrogate width, pooled"
        f" over {resolution_replicates} replicates with {draws} surrogates each. The"
        " two ranks share their samples and surrogates.",
        ("Rank k", "Resolution (null widths)"),
        resolution_rows,
    )
    claimed = [twist for twist in TWISTS if twist >= FAINT_TWIST]
    claims = [
        claim_cases(
            f"The KSG estimate at k = {KSG_RANKS[0]} recovers the mutual information"
            f" of every banana of twist at least {FAINT_TWIST:g} at"
            f" {grouped(sizes[0])} samples.",
            [
                Case(
                    f"b = {twist:g}",
                    ratios[(twist, sizes[0], KSG_RANKS[0])].value - 1.0,
                    ratios[(twist, sizes[0], KSG_RANKS[0])].margin(),
                    KSG_TOLERANCE,
                )
                for twist in claimed
            ],
            tolerance_text=f"{KSG_TOLERANCE:.0%} of the truth",
            replicates=replicates,
        ),
        claim_cases_above(
            f"The KSG estimate at k = {KSG_RANKS[0]} resolves the faint banana better"
            f" than at k = {KSG_RANKS[1]}.",
            [
                Case(
                    f"N = {grouped(sizes[0])}",
                    advantage.value - 1.0,
                    advantage.margin(),
                    0.0,
                )
            ],
            tolerance_text=None,
            reference_text=f"the resolution at k = {KSG_RANKS[1]}",
            replicates=advantage.replicates,
        ),
    ]
    twist_text = ", ".join(f"{twist:g}" for twist in TWISTS)
    size_text = ", ".join(grouped(size) for size in sizes)
    dependence_role = role(
        "func", "mixed_signals.diagnostics.dependence.dependence_statistics"
    )
    rank_role = role(
        "data", "mixed_signals.diagnostics.dependence.MUTUAL_INFORMATION_NEIGHBOURS"
    )
    setup = (
        "The zoo's twisted Gaussian"
        f" ({role('func', 'mixed_signals.zoo.cases.banana')}, variance"
        f" {BANANA_VARIANCE:g}) at the twists {twist_text}."
        " The zoo does not tabulate its mutual information, so the script computes it."
        " With t the twist times the variance, the second parameter is a unit normal"
        " plus t(z² − 1) for z standard normal, so the mutual information is the"
        " entropy of that sum minus ½ ln(2πe). Its density comes from composite"
        f" {len(_LEGENDRE_NODES)}-point Gauss-Legendre panels in z, with edges uniform"
        f" in z² up to z = {QUADRATURE_Z_MAX:g} and a shift of at most"
        f" {QUADRATURE_PANEL:g} along y per panel, and its entropy from the trapezoid"
        f" rule with a step of {QUADRATURE_STEP:g}; halving both changes every"
        f" value by at most {accuracy:.1e} nats. The"
        f" estimate is the KSG mutual information of {dependence_role}, which"
        " returns it as an excess over the Gaussian value at the set's Pearson r;"
        f" the table adds that value back. Sample counts {size_text},"
        f" {replicates} replicates per cell, ranks {KSG_RANKS[0]} (the default,"
        f" which {rank_role} sets) and {KSG_RANKS[1]}. The claims cover the bananas"
        f" of twist at least {FAINT_TWIST:g}, the faint one included, whose value of"
        f" {truths[FAINT_TWIST]:.4f} nats is the record's banana of 0.08 nats."
        " Session 11 recorded k = 10 within −9 to +4% at 2 000 samples and −26% at"
        " 1 000 samples for the faint banana, and a resolution 1.7 times that of"
        " k = 3."
    )
    return Section(setup, (ratio_table, resolution_table), tuple(claims))


# * Check bic-neff.

BIC_LABEL = "bic-neff"
BIC_REPEATS = 5


def _components_text(counts: Sequence[int]) -> str:
    """Write the chosen component counts as ``count: replicates`` pairs."""
    return ", ".join(
        f"{count}: {sum(1 for entry in counts if entry == count)}"
        for count in sorted(set(counts))
    )


def _bic_neff(context: Context) -> Section:
    label = BIC_LABEL
    replicates = context.count(20, 1)
    sample_count = context.count(1_000, 400)
    cases = (
        ("Gaussian", zoo.shifted_gaussians(0.0, param_count=2), "reference"),
        ("Banana", zoo.banana(), "banana"),
        ("Mixture, separation 4", zoo.gaussian_mixture(4.0), "mixture"),
    )
    settings = (
        "independent draws, n = n_eff",
        f"repeated {BIC_REPEATS} times, n = N",
        f"repeated {BIC_REPEATS} times, n = n_eff",
    )
    rows: list[list[Cell]] = []
    verdicts: dict[str, tuple[list[int], list[int], list[int]]] = {}
    for index, (name, case, posterior) in enumerate(cases):
        sets = case.sample(
            context.key(label, "sets", index), sample_count, batch_size=replicates
        ).samples[:, case.names.index(posterior)]
        components: tuple[list[int], list[int], list[int]] = ([], [], [])
        deltas: tuple[list[float], list[float], list[float]] = ([], [], [])
        for replicate in range(replicates):
            samples = sets[replicate]
            repeated = jnp.tile(samples, (BIC_REPEATS, 1))
            key = context.key(label, "fits", index, replicate)
            repeated_key = context.key(label, "repeated", index, replicate)
            fits = (
                gmm_bic(
                    samples,
                    None,
                    case.space,
                    effective_sample_size=float(sample_count),
                    key=key,
                ),
                gmm_bic(
                    repeated,
                    None,
                    case.space,
                    effective_sample_size=float(BIC_REPEATS * sample_count),
                    key=repeated_key,
                ),
                gmm_bic(
                    repeated,
                    None,
                    case.space,
                    effective_sample_size=float(sample_count),
                    key=repeated_key,
                ),
            )
            for position, fit in enumerate(fits):
                components[position].append(fit.components)
                deltas[position].append(fit.delta_bic)
        verdicts[name] = components
        reference_delta = sum(deltas[0]) / replicates
        for position, setting in enumerate(settings):
            matching = sum(
                1
                for chosen, independent in zip(
                    components[position], components[0], strict=True
                )
                if chosen == independent
            )
            rows.append(
                [
                    name,
                    setting,
                    Measurement.from_replicates(jnp.asarray(deltas[position])),
                    (sum(deltas[position]) / replicates) / reference_delta
                    if reference_delta != 0.0
                    else None,
                    _components_text(components[position]),
                    f"{matching} of {replicates}",
                ]
            )
    table = Table.of(
        "ΔBIC = BIC(1) minus the smallest BIC and the number of components that BIC"
        f" chooses, over {replicates} replicates of each case and setting. The ratio"
        " divides the mean ΔBIC by that of the independent draws, and the last"
        " column counts the replicates whose chosen number equals that of the"
        " independent draws.",
        (
            "Case",
            "Setting",
            "ΔBIC",
            "ΔBIC over that of the independent draws",
            "Components chosen (count: replicates)",
            "Same count as independent draws",
        ),
        rows,
    )
    equal_everywhere = all(
        chosen == independent
        for components in verdicts.values()
        for chosen, independent in zip(components[2], components[0], strict=True)
    )
    matching_total = sum(
        1
        for components in verdicts.values()
        for chosen, independent in zip(components[2], components[0], strict=True)
        if chosen == independent
    )
    exceeding: dict[str, tuple[int, int]] = {}
    for name, components in verdicts.items():
        applicable = [
            (inflated, independent)
            for inflated, independent in zip(components[1], components[0], strict=True)
            if independent < MAX_COMPONENTS
        ]
        exceeding[name] = (
            sum(1 for inflated, independent in applicable if inflated > independent),
            len(applicable),
        )
    exceeds_where_possible = all(
        found >= 1 for found, possible in exceeding.values() if possible > 0
    )
    exceeding_text = "; ".join(
        f"{name}: {found} of {possible}" if possible > 0 else f"{name}: not applicable"
        for name, (found, possible) in exceeding.items()
    )
    claims = [
        claim_true(
            "With n = n_eff, the chosen number of components equals that of the"
            " independent draws in every replicate.",
            "every replicate of every case",
            f"{matching_total} of {replicates * len(cases)} replicates",
            equal_everywhere,
            full_only=True,
        ),
        claim_true(
            "With n = N, the chosen number exceeds that of the independent draws in"
            " at least one replicate of every case where the independent draws"
            f" choose fewer than {MAX_COMPONENTS}.",
            "at least one replicate per applicable case",
            exceeding_text,
            exceeds_where_possible,
            full_only=True,
        ),
    ]
    setup = (
        "Independent draws of the zoo's Gaussian"
        f" ({role('func', 'mixed_signals.zoo.cases.shifted_gaussians')} without a"
        f" shift, 2 parameters), banana and mixture at separation 4, with"
        f" {grouped(sample_count)} samples and {replicates} replicates each, and the"
        f" same draws repeated {BIC_REPEATS} times, which hold"
        f" {grouped(BIC_REPEATS * sample_count)} samples and the effective sample size"
        f" of the independent draws, {grouped(sample_count)}."
        f" {role('func', 'mixed_signals.diagnostics.modes.gmm_bic')} runs with n"
        " set to the repeated count N and to n_eff; the two share their fits, so"
        " only the n of BIC differs. Components range from 1 to"
        f" {MAX_COMPONENTS}. The record found that n = N inflated ΔBIC tenfold and"
        " chose too many components, while n_eff gave the result of the independent"
        " draws. The second claim does not apply to a case whose independent draws"
        f" already choose {MAX_COMPONENTS} components."
    )
    return Section(setup, (table,), tuple(claims))


# * Check mode-counting.

MODES_LABEL = "mode-counting"
MODE_ITERATIONS = 1_000
MODE_MERGE = 1e-2
"""Distance, in overall standard deviations, below which two mixture modes merge."""
FLOOR_CONTRAST = 0.01
STUDENT_DEGREES = 3.0
EIGHT_PARAMETERS = 8


def mixture_modes(
    log_weights: ArrayLike, means: ArrayLike, covariances: ArrayLike
) -> int:
    """Return the number of modes of a Gaussian mixture.

    The fixed-point iteration of Carreira-Perpiñán (2000), x ← (Σ_c r_c Σ_c⁻¹)⁻¹
    Σ_c r_c Σ_c⁻¹ μ_c with r_c the responsibilities at x, starts at every
    component mean and runs a fixed number of iterations. Converged points
    closer than a small tolerance, in units of the overall standard deviation,
    count as one mode.

    Args:
        log_weights: Log weights of the components, shaped (C,).
        means: Component means, shaped (C, D).
        covariances: Component covariances, shaped (C, D, D).

    Returns:
        The number of distinct modes reached from the component means.
    """
    return int(
        _mixture_mode_count(
            *(jnp.asarray(entry) for entry in (log_weights, means, covariances))
        )
    )


@jax.jit
def _mixture_mode_count(log_weights: Array, means: Array, covariances: Array) -> Array:
    precisions = jnp.linalg.inv(covariances)
    log_normalisers = 0.5 * jnp.linalg.slogdet(precisions)[1]

    def update(point: Array) -> Array:
        offsets = point[None, :] - means
        quadratic = jnp.einsum("cd,cde,ce->c", offsets, precisions, offsets)
        weights = jax.nn.softmax(log_weights + log_normalisers - 0.5 * quadratic)
        precision = jnp.einsum("c,cde->de", weights, precisions)
        target = jnp.einsum("c,cde,ce->d", weights, precisions, means)
        return cast(Array, jnp.linalg.solve(precision, target))

    def converge(start: Array) -> Array:
        return cast(
            Array,
            jax.lax.fori_loop(
                0, MODE_ITERATIONS, lambda _, point: update(point), start
            ),
        )

    modes = jax.vmap(converge)(means)
    weights = jnp.exp(log_weights)
    mean = weights @ means
    variances = weights @ jnp.diagonal(covariances, axis1=1, axis2=2) + weights @ (
        (means - mean) ** 2
    )
    scale = jnp.sqrt(jnp.mean(variances))
    distances = jnp.linalg.norm(modes[:, None, :] - modes[None, :, :], axis=-1)
    first_close = jnp.argmax(distances <= MODE_MERGE * scale, axis=1)
    return jnp.sum(first_close == jnp.arange(means.shape[0]))


@functools.partial(jax.jit, static_argnames=("parameter", "floor"))
def modes_at_floor(values: Array, parameter: Parameter, floor: float) -> Array:
    """Return the modes of the package's 1D estimate with another peak floor.

    The estimate, its grid and the merging of shallow valleys are the package's;
    only the fraction of the highest peak that a peak must reach changes.
    """
    kde = KernelDensity.fit(
        values[:, None], jnp.zeros(values.shape[0]), ParamSpace(parameters=(parameter,))
    )
    # * The covariance of JAX's estimator already holds the factor squared.
    width = jnp.sqrt(kde.kde.covariance[0, 0])
    grid = _grid(values, parameter, width)
    density = jnp.exp(kde.log_density(grid[:, None]))
    previous = jnp.concatenate([jnp.full(1, -jnp.inf), density[:-1]])
    following = jnp.concatenate([density[1:], jnp.full(1, -jnp.inf)])
    peaks = (density > previous) & (density >= following)
    counted = peaks & (density >= floor * jnp.max(density))
    initial = (
        jnp.asarray(jnp.nan),
        jnp.asarray(jnp.inf),
        jnp.asarray(0.0),
        jnp.asarray(0.0),
    )
    (_, _, _, separations), _ = jax.lax.scan(_valley_step, initial, (density, counted))
    return jnp.where(jnp.isfinite(kde.log_normaliser), 1 + separations, jnp.nan)


class _ModeCase(NamedTuple):
    """A case of the mode check: its space, its draws and the modes that it has."""

    name: str
    kind: str
    space: ParamSpace
    draw: Callable[[PRNGKeyArray, int, int], Array]
    expected: int | None
    """The number of modes the check claims, or ``None`` where it makes no claim."""
    true_dip: float | None = None
    """1 − f(valley)/f(peak) of an equal mixture from the zoo density."""


class _ModeRun(NamedTuple):
    """The mode counts of one replicate."""

    components: int
    largest: int
    dip: float
    mixture_modes: int
    floor_modes: int
    """Largest count over the marginals at the contrast floor."""
    package_modes: int
    """Largest count over the marginals at the package's floor."""


def _zoo_draw(
    case: zoo.ZooCase, posterior: str
) -> Callable[[PRNGKeyArray, int, int], Array]:
    position = case.names.index(posterior)

    def draw(key: PRNGKeyArray, count: int, batch: int) -> Array:
        return case.sample(key, count, batch_size=batch).samples[:, position]

    return draw


def _student_draw(
    degrees: float, param_count: int
) -> Callable[[PRNGKeyArray, int, int], Array]:
    def draw(key: PRNGKeyArray, count: int, batch: int) -> Array:
        return jax.random.t(key, degrees, (batch, count, param_count))

    return draw


TRUE_DIP_POINTS = 100_001
"""Points of the grid on which the true dip of a mixture is read."""
TRUE_DIP_MARGIN = 8.0
"""Component widths that the grid of the true dip reaches beyond the components."""
CLAIMED_SEPARATION = 3.0
"""Smallest separation of a mixture whose two modes the check claims."""


def _true_dip(case: zoo.ZooCase, separation: float) -> float:
    """Return 1 − f(valley)/f(peak) of the mixture along its separation axis.

    The density is the zoo's, on a grid symmetric about the valley, so the valley
    is the middle point; the other parameters are independent of the first.
    """
    reach = 0.5 * separation + TRUE_DIP_MARGIN
    grid = jnp.linspace(-reach, reach, TRUE_DIP_POINTS)
    points = jnp.zeros((TRUE_DIP_POINTS, len(case.space))).at[:, 0].set(grid)
    density = jnp.exp(case.log_prob(points)[:, case.names.index("mixture")])
    return float(1.0 - density[TRUE_DIP_POINTS // 2] / jnp.max(density))


def _mode_cases() -> list[_ModeCase]:
    cases = []
    for twist in (0.003, 0.03):
        banana = zoo.banana(twist)
        cases.append(
            _ModeCase(
                f"Banana, b = {twist:g}",
                "banana",
                banana.space,
                _zoo_draw(banana, "banana"),
                1,
            )
        )
    railed = zoo.railed_gaussians([0.0, 1.0])
    for position, offset in enumerate((0.0, 1.0)):
        cases.append(
            _ModeCase(
                f"Normal railed at {offset:g}σ",
                "railed",
                railed.space,
                _zoo_draw(railed, f"railed_{position}"),
                1,
            )
        )
    cases.append(
        _ModeCase(
            f"Student t, {STUDENT_DEGREES:g} degrees of freedom",
            "heavy",
            real_space(2),
            _student_draw(STUDENT_DEGREES, 2),
            1,
        )
    )
    for separation in (2.0, 2.5, 3.0, 4.0):
        mixture = zoo.gaussian_mixture(separation)
        cases.append(
            _ModeCase(
                f"Mixture, separation {separation:g}",
                "mixture",
                mixture.space,
                _zoo_draw(mixture, "mixture"),
                None if separation < CLAIMED_SEPARATION else 2,
                _true_dip(mixture, separation),
            )
        )
    eight = zoo.gaussian_mixture(3.0, param_count=EIGHT_PARAMETERS)
    cases.append(
        _ModeCase(
            f"Mixture, separation 3, {EIGHT_PARAMETERS} parameters",
            "mixture",
            eight.space,
            _zoo_draw(eight, "mixture"),
            2,
            _true_dip(eight, 3.0),
        )
    )
    return cases


def _analyse_modes(samples: Array, space: ParamSpace, key: PRNGKeyArray) -> _ModeRun:
    """Count the modes of one replicate and fit the mixture that BIC chooses."""
    count = samples.shape[0]
    structure = gmm_bic(
        samples, None, space, effective_sample_size=float(count), key=key
    )
    marginal = marginal_modes(samples, None, space)
    largest, dip = float(jnp.max(marginal.modes)), float(jnp.max(marginal.dip))
    if structure.components >= 2:
        projected = projection_modes(
            samples,
            None,
            space,
            directions=discriminant_directions(structure.mixture),
            centres=circle_centres(samples, None, space),
        )
        largest = max(largest, float(jnp.max(projected.modes)))
        dip = max(dip, float(jnp.max(projected.dip)))
    floor_counts = [
        int(
            modes_at_floor(
                samples[:, position],
                Parameter(
                    name="value",
                    topology=space.parameters[position].topology,
                    bounds=space.parameters[position].bounds,
                ),
                FLOOR_CONTRAST,
            )
        )
        for position in range(len(space))
    ]
    mixture = structure.mixture
    return _ModeRun(
        structure.components,
        round(largest),
        dip,
        mixture_modes(mixture.log_weights, mixture.means, mixture.covariances),
        max(floor_counts),
        round(float(jnp.max(marginal.modes))),
    )


def _mode_counting(context: Context) -> Section:
    label = MODES_LABEL
    replicates = context.count(12, 1)
    sample_count = context.count(2_000, 500)
    cases = _mode_cases()
    runs: dict[str, list[_ModeRun]] = {}
    for index, case in enumerate(cases):
        sets = case.draw(context.key(label, "sets", index), sample_count, replicates)
        runs[case.name] = [
            _analyse_modes(
                sets[replicate], case.space, context.key(label, "bic", index, replicate)
            )
            for replicate in range(replicates)
        ]
    structure_rows: list[list[Cell]] = []
    contrast_rows: list[list[Cell]] = []
    dip_means: dict[str, Measurement] = {}
    for case in cases:
        entries = runs[case.name]
        dip_means[case.name] = Measurement.from_replicates(
            jnp.asarray([run.dip for run in entries])
        )
        structure_rows.append(
            [
                case.name,
                len(case.space),
                Measurement.from_replicates(
                    jnp.asarray([float(run.components) for run in entries])
                ),
                sum(1 for run in entries if run.largest == 1),
                sum(1 for run in entries if run.largest == 2),
                sum(1 for run in entries if run.largest > 2),
                dip_means[case.name],
                case.true_dip,
            ]
        )
        contrast_rows.append(
            [
                case.name,
                f"{sum(1 for run in entries if run.floor_modes > 1)} of {replicates}",
                f"{sum(1 for run in entries if run.package_modes > 1)} of {replicates}",
                Measurement.from_replicates(
                    jnp.asarray([float(run.mixture_modes) for run in entries])
                ),
                max(run.mixture_modes for run in entries),
            ]
        )
    structure_table = Table.of(
        f"Components that BIC chooses and the modes of the 1D estimates, the largest"
        f" count over the marginals and over the discriminant directions of the"
        f" chosen mixture, over {replicates} replicates of {grouped(sample_count)}"
        " samples. The dip is the deepest valley; the true dip of an equal mixture"
        " is 1 − f(valley)/f(peak) of the zoo density, read on a grid of"
        f" {grouped(TRUE_DIP_POINTS)} points.",
        (
            "Case",
            "Parameters",
            "BIC components",
            "Replicates with one mode",
            "with two modes",
            "with more",
            "Deepest dip",
            "True dip",
        ),
        structure_rows,
    )
    contrast_table = Table.of(
        f"Contrasts. The replicates in which some marginal has a second mode with a"
        f" peak floor of {FLOOR_CONTRAST:.0%} and with the package's floor of"
        f" {MODE_FLOOR:.0%}, and the modes of the mixture that BIC chooses, by the"
        " fixed-point iteration, with their largest count.",
        (
            "Case",
            f"Second marginal mode, floor {FLOOR_CONTRAST:.0%}",
            f"Second marginal mode, floor {MODE_FLOOR:.0%}",
            "Modes of the BIC mixture",
            "Largest count",
        ),
        contrast_rows,
    )

    def all_runs(kinds: Sequence[str]) -> list[_ModeRun]:
        return [run for case in cases if case.kind in kinds for run in runs[case.name]]

    unimodal = all_runs(("banana", "railed", "heavy"))
    counted_mixtures = [
        run
        for case in cases
        if case.kind == "mixture" and case.expected is not None
        for run in runs[case.name]
    ]
    banana_runs = all_runs(("banana",))
    one_mode = sum(1 for run in unimodal if run.largest == 1)
    two_modes = sum(1 for run in counted_mixtures if run.largest == 2)
    floor_second = sum(1 for run in unimodal if run.floor_modes > 1)
    banana_mixture = max(run.mixture_modes for run in banana_runs)
    claims = [
        claim_true(
            "Every banana, railed and heavy-tailed replicate has one mode.",
            "one mode in every replicate",
            f"{one_mode} of {len(unimodal)} replicates",
            one_mode == len(unimodal),
            full_only=True,
        ),
        claim_true(
            f"Every replicate of a mixture from separation {CLAIMED_SEPARATION:g}, and"
            f" of the {EIGHT_PARAMETERS}-parameter mixture, has two modes.",
            "two modes in every replicate",
            f"{two_modes} of {len(counted_mixtures)} replicates",
            two_modes == len(counted_mixtures),
            full_only=True,
        ),
        claim_true(
            f"With a peak floor of {FLOOR_CONTRAST:.0%}, at least one unimodal"
            " replicate counts a second mode.",
            "at least one replicate",
            f"{floor_second} of {len(unimodal)} replicates",
            floor_second >= 1,
            full_only=True,
        ),
        claim_true(
            "The modes of the mixture that BIC chooses exceed one for at least one"
            " banana replicate.",
            "more than one mode in at least one replicate",
            f"largest count {banana_mixture} in {len(banana_runs)} replicates",
            banana_mixture > 1,
            full_only=True,
        ),
    ]
    separation_25 = next(case for case in cases if case.name.endswith("separation 2.5"))
    true_dip_25 = separation_25.true_dip
    dip_25 = dip_means[separation_25.name]
    setup = (
        "Cases: two bananas"
        f" ({role('func', 'mixed_signals.zoo.cases.banana')}), normals railed at the"
        " bound by 0σ and 1σ"
        f" ({role('func', 'mixed_signals.zoo.cases.railed_gaussians')}), a Student"
        f" t distribution with {STUDENT_DEGREES:g} degrees of freedom in two"
        " independent parameters, drawn with `jax.random.t` because the zoo has none,"
        " mixtures at the separations 2, 2.5, 3 and 4"
        f" ({role('func', 'mixed_signals.zoo.cases.gaussian_mixture')}) and one in"
        f" {EIGHT_PARAMETERS} parameters at separation 3, with"
        f" {grouped(sample_count)} samples and {replicates} replicates each."
        f" {role('func', 'mixed_signals.diagnostics.modes.marginal_modes')} counts the"
        " modes along every parameter, and"
        f" {role('func', 'mixed_signals.diagnostics.modes.projection_modes')} along"
        " the discriminant directions of the mixture that"
        f" {role('func', 'mixed_signals.diagnostics.modes.gmm_bic')} chooses; the"
        " table gives the largest count of both. Two contrasts come from the script."
        " The first counts the marginal modes with the package's own estimate, grid"
        " and valley rule but a peak floor of"
        f" {FLOOR_CONTRAST:.0%} instead of"
        f" {role('data', 'mixed_signals.diagnostics.modes.MODE_FLOOR')}; no private"
        " helper of the package takes the floor, so the counting is repeated for the"
        " contrast. The second finds the modes of the mixture that BIC chooses by the"
        " fixed-point iteration of Carreira-Perpiñán"
        " ({cite:t}`carreiraperpinan2000mode`) from every component mean, with"
        f" {grouped(MODE_ITERATIONS)} iterations, merging points closer than"
        f" {MODE_MERGE:g} overall standard deviations. The mixture at separation 2"
        " is not claimed, because an equal mixture has two modes only above 2."
        " The mixture at separation 2.5 stays in the tables without a claim: its"
        f" true dip is {true_dip_25:.3f}, against {MODE_SEPARATION:g} for"
        f" {role('data', 'mixed_signals.diagnostics.modes.MODE_SEPARATION')}, and"
        f" the mean dip of its estimate is {dip_25.render()} at"
        f" {grouped(sample_count)} samples, because the bandwidth of the estimate"
        " fills the valley. Session 11 recorded two modes from separation 2.5; the"
        f" developers moved the claim to separation {CLAIMED_SEPARATION:g}."
        " Session 11 recorded one mode for every banana and railed case and two"
        " from separation 2.5, with dips of 0.10 to 0.57 and including the"
        " 8-parameter mixture that BIC missed; a floor of 1% let single samples in"
        " heavy tails count as modes in 5 to 25% of marginals against none at 5%;"
        " and mixture modes numbered up to five for a unimodal banana."
    )
    return Section(setup, (structure_table, contrast_table), tuple(claims))


# * Check split-chain-kuiper.

KUIPER_LABEL = "split-chain-kuiper"
KUIPER_COEFFICIENT = 0.5
KUIPER_WIDTH = 2.5
KUIPER_STEPS = 1_000
KUIPER_WALKERS = 16
KUIPER_THIN = 3
TWO_PI = 2 * math.pi


def _split_chain_kuiper(context: Context) -> Section:
    label = KUIPER_LABEL
    runs = context.count(300, 20)
    case = zoo.ar1_chains([KUIPER_COEFFICIENT])
    thin = jnp.asarray([KUIPER_THIN])
    p_values = []
    for run in range(runs):
        chains = case.sample(
            context.key(label, "chains", run), KUIPER_STEPS, walker_count=KUIPER_WALKERS
        )
        values = chains.samples[:, 0, :, 0, :] * KUIPER_WIDTH
        # * Wrapping onto [−π, π) makes walkers cross the cut, which V ignores.
        wrapped = jnp.mod(values + math.pi, TWO_PI) - math.pi
        p_values.append(split_chain_kuiper(wrapped, thin=thin).p_value[0])
    p_array = jnp.stack(p_values)
    table = Table.of(
        f"Rejection rate of the split-chain Kuiper test on {runs} stationary chains"
        " on a circle: the share of runs whose p-value is at or below the level,"
        " with the binomial error at the level, and the mean p-value.",
        ("Runs", *[f"Rate at {level:.0%}" for level in LEVELS], "Mean p-value"),
        [
            [
                runs,
                *rate_measurements(p_array),
                Measurement.from_replicates(p_array),
            ]
        ],
    )
    claims = [
        claim_rates_within(
            "The split-chain Kuiper test rejects stationary circle chains at its"
            " levels.",
            rate_counts(p_array),
            LEVELS,
            full_only=True,
        )
    ]
    setup = (
        f"{runs} runs of AR(1) chains with a coefficient of {KUIPER_COEFFICIENT:g}"
        f" ({role('func', 'mixed_signals.zoo.cases.ar1_chains')}), {KUIPER_WALKERS}"
        f" walkers of {grouped(KUIPER_STEPS)} steps, scaled by {KUIPER_WIDTH:g} and"
        " wrapped onto [−π, π), as the test of the package builds them."
        f" {role('func', 'mixed_signals.mcmc.ess.split_chain_kuiper')} runs with the"
        f" thinning factor {KUIPER_THIN}, the autocorrelation time of the chains."
        " The chains are stationary, so the p-values are uniform if the test is"
        " exact. Session 11 recorded 5.7 to 6.3% below 0.05 in 300 runs."
    )
    return Section(setup, (table,), tuple(claims))


# * Check diagnostic-flags.

FLAGS_LABEL = "diagnostic-flags"
FLAG_SAMPLE_COUNT = PAIR_CAP
POLICY_LABELS = ("sets", "plain", "truncated", "statistics", "bic")


class _FlagCase(NamedTuple):
    """A zoo case for the flags and the rows that the record says it flags."""

    name: str
    case: zoo.ZooCase
    posterior: str
    required: tuple[tuple[str, str], ...]
    """Rows that must flag in every replicate."""
    forbidden: tuple[str, ...]
    """Metrics of which no row may flag."""
    silent: bool
    """Whether no row at all may flag."""


def _flag_cases() -> list[_FlagCase]:
    covariance = jnp.array([[1.0, 0.8, 0.0], [0.8, 1.0, -0.3], [0.0, -0.3, 1.0]])
    correlated = zoo.gaussians(jnp.zeros((1, 3)), covariance[None], names=["gaussian"])
    return [
        _FlagCase(
            "Gaussian, independent, 2 parameters",
            zoo.shifted_gaussians(0.0, param_count=2),
            "reference",
            (),
            (),
            True,
        ),
        _FlagCase(
            "Gaussian, correlated, 3 parameters", correlated, "gaussian", (), (), True
        ),
        _FlagCase(
            "Wrapped normal, width 0.5 rad, 2 parameters",
            zoo.wrapped_gaussians(0.0, width=0.5, param_count=2, correlation=0.5),
            "reference",
            (),
            (),
            True,
        ),
        _FlagCase(
            "Normal 4σ from its bound, 2 parameters",
            zoo.railed_gaussians([4.0]),
            "railed_0",
            (),
            (),
            True,
        ),
        _FlagCase(
            "Banana, 2 parameters",
            zoo.banana(),
            "banana",
            (
                ("distance_correlation", "x0-x1"),
                ("mutual_information", "x0-x1"),
                ("mardia_skewness", "all"),
                ("gmm_bic", "all"),
            ),
            ("mode_dip",),
            False,
        ),
        _FlagCase(
            "Mixture, separation 4, 2 parameters",
            zoo.gaussian_mixture(4.0),
            "mixture",
            (("mode_dip", "x0"), ("kurtosis", "x0")),
            (),
            False,
        ),
        _FlagCase(
            "Normal railed at 0σ, 2 parameters",
            zoo.railed_gaussians([0.0]),
            "railed_0",
            (("railing_lower", "x0"), ("skewness", "x0")),
            (),
            False,
        ),
        _FlagCase(
            "Normal railed at 1σ, 2 parameters",
            zoo.railed_gaussians([1.0]),
            "railed_0",
            (("railing_lower", "x0"), ("skewness", "x0")),
            (),
            False,
        ),
    ]


class _SetStatistics(NamedTuple):
    """The surrogate-null statistics of one set, besides the mode dips."""

    skewness: Array
    kurtosis: Array
    anderson: Array
    mardia_skewness: Array
    mardia_kurtosis: Array
    spearman: Array
    distance: Array
    information: Array


def _set_statistics(
    samples: Array, key: PRNGKeyArray, space: ParamSpace
) -> _SetStatistics:
    moments = marginal_moments(samples, None, space)
    mardia = mardia_moments(samples, None, space, key=key)
    dependence = dependence_statistics(samples, None, space, key=key)
    return _SetStatistics(
        moments.skewness,
        moments.kurtosis,
        anderson_darling_normality(samples, None, space),
        mardia.skewness,
        mardia.kurtosis,
        dependence.spearman,
        dependence.distance_correlation,
        dependence.mutual_information,
    )


_observed_statistics = jax.jit(_set_statistics, static_argnames=("space",))


@functools.partial(jax.jit, static_argnames=("space",))
def _null_statistics(
    surrogates: Array, key: PRNGKeyArray, space: ParamSpace
) -> _SetStatistics:
    return cast(
        _SetStatistics,
        jax.lax.map(lambda values: _set_statistics(values, key, space), surrogates),
    )


@functools.partial(jax.jit, static_argnames=("space",))
def _dip_null(surrogates: Array, space: ParamSpace) -> Array:
    return cast(
        Array,
        jax.lax.map(lambda values: marginal_modes(values, None, space).dip, surrogates),
    )


@functools.partial(jax.jit, static_argnames=("space",))
def _projection_null(
    surrogates: Array, directions: Array, centres: Array, space: ParamSpace
) -> Array:
    return cast(
        Array,
        jax.lax.map(
            lambda values: (
                projection_modes(
                    values, None, space, directions=directions, centres=centres
                ).dip
            ),
            surrogates,
        ),
    )


def _flagged_rows(
    context: Context,
    index: int,
    entry: _FlagCase,
    samples: Array,
    replicate: int,
    draws: int,
) -> set[tuple[str, str]]:
    """Run every diagnostic of one replicate and return the rows that flag."""
    label = FLAGS_LABEL
    space = entry.case.space
    names = space.names
    count = samples.shape[0]
    path = (index, replicate)
    key = context.key(label, "statistics", *path)
    flagged: set[tuple[str, str]] = set()

    def record(metric: str, row: str, value: Array, null: Array | None = None) -> None:
        if evaluate(metric, float(value), null).flagged:
            flagged.add((metric, row))

    plain = gaussian_surrogates(
        samples,
        None,
        space,
        size=count,
        draws=draws,
        key=context.key(label, "plain", *path),
    )
    truncated = gaussian_surrogates(
        samples,
        None,
        space,
        size=count,
        draws=draws,
        key=context.key(label, "truncated", *path),
        truncate=True,
    )
    observed = _observed_statistics(samples, key, space)
    null = _null_statistics(plain, key, space)
    for position, (skew_name, kurtosis_name) in enumerate(moment_metric_names(space)):
        name = names[position]
        record(skew_name, name, observed.skewness[position], null.skewness[:, position])
        record(
            kurtosis_name, name, observed.kurtosis[position], null.kurtosis[:, position]
        )
        record(
            "anderson_darling",
            name,
            observed.anderson[position],
            null.anderson[:, position],
        )
    record("mardia_skewness", "all", observed.mardia_skewness, null.mardia_skewness)
    record("mardia_kurtosis", "all", observed.mardia_kurtosis, null.mardia_kurtosis)
    for pair, (first, second) in enumerate(parameter_pairs(len(space))):
        row = f"{names[first]}-{names[second]}"
        record("spearman", row, observed.spearman[pair], null.spearman[:, pair])
        record(
            "distance_correlation", row, observed.distance[pair], null.distance[:, pair]
        )
        record(
            "mutual_information",
            row,
            observed.information[pair],
            null.information[:, pair],
        )
    dips = marginal_modes(samples, None, space).dip
    dip_null = _dip_null(truncated, space)
    for position, name in enumerate(names):
        record("mode_dip", name, dips[position], dip_null[:, position])
    structure = gmm_bic(
        samples,
        None,
        space,
        effective_sample_size=float(count),
        key=context.key(label, "bic", *path),
    )
    record("gmm_bic", "all", jnp.asarray(structure.delta_bic))
    directions = discriminant_directions(structure.mixture)
    if directions.shape[0]:
        centres = circle_centres(samples, None, space)
        projected = projection_modes(
            samples, None, space, directions=directions, centres=centres
        ).dip
        projected_null = _projection_null(plain, directions, centres, space)
        for direction in range(directions.shape[0]):
            record(
                "mode_dip",
                f"direction {direction}",
                projected[direction],
                projected_null[:, direction],
            )
    scores = railing_scores(samples, None, space)
    for position in railed_parameters(space):
        for metric, values in (
            ("railing_lower", scores.lower),
            ("railing_upper", scores.upper),
        ):
            if not math.isnan(float(values[position])):
                record(metric, names[position], values[position])
    return flagged


def _row_text(rows: set[tuple[str, str]]) -> str:
    """Write the flagged rows as ``metric row`` items, or ``none``."""
    if not rows:
        return "none"
    return "; ".join(f"{metric} {row}" for metric, row in sorted(rows))


def _diagnostic_flags(context: Context) -> Section:
    replicates = context.count(3, 1)
    draws = context.count(100, 19)
    sample_count = context.count(FLAG_SAMPLE_COUNT, 500)
    entries = _flag_cases()
    table_rows: list[list[Cell]] = []
    claims: list[Claim] = []
    for index, entry in enumerate(entries):
        space = entry.case.space
        sets = entry.case.sample(
            context.key(FLAGS_LABEL, "sets", index), sample_count, batch_size=replicates
        ).samples[:, entry.case.names.index(entry.posterior)]
        satisfied = 0
        strays: list[str] = []
        for replicate in range(replicates):
            rows = _flagged_rows(
                context, index, entry, sets[replicate], replicate, draws
            )
            table_rows.append([entry.name, len(space), replicate, _row_text(rows)])
            if entry.silent:
                met = not rows
                strays.extend(f"{metric} {row}" for metric, row in sorted(rows))
            else:
                forbidden = {
                    (metric, row) for metric, row in rows if metric in entry.forbidden
                }
                met = set(entry.required) <= rows and not forbidden
                strays.extend(f"{metric} {row}" for metric, row in sorted(forbidden))
            satisfied += int(met)
        if entry.silent:
            statement = f"{entry.name}: no row flags."
            expected = "no row flags in any replicate"
        else:
            wanted = ", ".join(f"{metric} {row}" for metric, row in entry.required)
            statement = f"{entry.name}: the rows {wanted} flag."
            expected = "every named row flags in every replicate" + (
                f", and no {', '.join(entry.forbidden)} row" if entry.forbidden else ""
            )
        observed = f"{satisfied} of {replicates} replicates" + (
            f"; stray rows: {', '.join(sorted(set(strays)))}" if strays else ""
        )
        claims.append(
            claim_true(
                statement, expected, observed, satisfied == replicates, full_only=True
            )
        )
    table = Table.of(
        f"Rows that flag, by zoo case and replicate, with {draws} Gaussian surrogates"
        f" per set and {grouped(sample_count)} samples.",
        ("Case", "Parameters", "Replicate", "Rows that flag"),
        table_rows,
    )
    setup = (
        "Zoo cases with the diagnostics of session 11 assembled as the tests of the"
        f" package assemble them: the marginal moments, the Anderson-Darling"
        " distance, Mardia's moments, the dependence rows and the mode dips, each"
        f" against {draws} Gaussian surrogates of the set's effective sample size,"
        f" which is {grouped(sample_count)} for independent draws; GMM-BIC with its"
        " ΔBIC; the dips along the discriminant directions of the chosen mixture;"
        " and the railing scores, which have no surrogate null. The mode rows on"
        " intervals take surrogates truncated to the interval."
        f" {role('func', 'mixed_signals.diagnostics.flags.evaluate')} flags each row"
        f" against {role('data', 'mixed_signals.diagnostics.flags.THRESHOLDS')};"
        f" {replicates} replicates per case. Session 11 recorded that the"
        " Gaussians, a narrow wrapped normal and a normal 4σ from its bound raised no"
        " flag, that bananas flagged their dependence, Mardia's skewness and"
        " GMM-BIC but no mode, that mixtures at separation 4 flagged their mode and"
        " kurtosis, and that normals railed at 0σ and 1σ flagged their bound and"
        " skewness. A claim names the rows of the record and holds only if they flag"
        " in every replicate; the table lists every row that flags, so a stray row"
        " is visible, and a stray row in a case that must raise no flag fails its"
        " claim."
    )
    return Section(setup, (table,), tuple(claims))


CHECKS: tuple[Check, ...] = (
    Check(
        label=CALIBRATION_LABEL,
        title="Calibration of the Gaussian surrogate null",
        session="11",
        supports=(
            role(
                "func",
                "mixed_signals.nulls.surrogates.gaussian_surrogates",
                short=False,
            ),
            role(
                "func",
                "mixed_signals.diagnostics.dependence.dependence_statistics",
                short=False,
            ),
        ),
        run=_surrogate_calibration,
    ),
    Check(
        label=CIRCLE_LABEL,
        title="Drift of linear statistics on wide circles",
        session="11",
        supports=(
            role(
                "data",
                "mixed_signals.diagnostics._common.CIRCLE_WIDTH_LIMIT",
                short=False,
            ),
        ),
        run=_circle_drift,
    ),
    Check(
        label=KSG_LABEL,
        title="KSG mutual information on bananas",
        session="11",
        supports=(
            role(
                "data",
                "mixed_signals.diagnostics.dependence.MUTUAL_INFORMATION_NEIGHBOURS",
                short=False,
            ),
            role("data", "mixed_signals.diagnostics.dependence.PAIR_CAP", short=False),
        ),
        run=_ksg_bananas,
    ),
    Check(
        label=BIC_LABEL,
        title="BIC with the effective sample size",
        session="11",
        supports=(
            role("func", "mixed_signals.diagnostics.modes.gmm_bic", short=False),
        ),
        run=_bic_neff,
    ),
    Check(
        label=MODES_LABEL,
        title="Counting modes",
        session="11",
        supports=(
            role("data", "mixed_signals.diagnostics.modes.MODE_FLOOR", short=False),
            role(
                "data", "mixed_signals.diagnostics.modes.MODE_SEPARATION", short=False
            ),
            "the choice of the modes of 1D kernel estimates over those of the fitted"
            " mixture",
        ),
        run=_mode_counting,
    ),
    Check(
        label=KUIPER_LABEL,
        title="Split-chain Kuiper test on stationary chains",
        session="11",
        supports=(
            role("func", "mixed_signals.mcmc.ess.split_chain_kuiper", short=False),
        ),
        run=_split_chain_kuiper,
    ),
    Check(
        label=FLAGS_LABEL,
        title="Default flags on the zoo",
        session="11",
        supports=(
            role("data", "mixed_signals.diagnostics.flags.THRESHOLDS", short=False),
            role("func", "mixed_signals.diagnostics.flags.evaluate", short=False),
            "the done criterion of session 11: the zoo's cases flag as recorded",
        ),
        run=_diagnostic_flags,
    ),
)
"""Checks of the diagnostics."""
