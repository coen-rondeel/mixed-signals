"""Checks of the effect sizes: shifts, width ratios, signs, thresholds, kNN KL."""

import functools
import math
from collections.abc import Mapping, Sequence
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray
from matplotlib.figure import Figure as MatplotlibFigure

from mixed_signals import zoo
from mixed_signals.core.samples import weighted_covariance
from mixed_signals.core.space import ParamSpace
from mixed_signals.metrics.divergence import KL_NEIGHBOURS, knn_kl_divergence
from mixed_signals.metrics.kernel import (
    energy_distance,
    maximum_mean_discrepancy,
    median_bandwidth,
)
from mixed_signals.metrics.marginal import (
    anderson_darling_distance,
    ks_statistic,
    marginal_kde_divergences,
    shared_bandwidth_factor,
    wasserstein_distance,
)
from mixed_signals.metrics.moments import (
    gaussian_hellinger,
    gaussian_kl,
    gaussian_w2,
    width_ratio,
)
from mixed_signals.metrics.ot.sinkhorn import (
    SINKHORN_EPSILON_FACTOR,
    sinkhorn_divergence,
)
from mixed_signals.metrics.ot.sliced import (
    SLICED_DIRECTIONS,
    gaussian_sliced_wasserstein_distance,
    sliced_directions,
    sliced_wasserstein_distance,
)
from mixed_signals.nulls import permutation_null
from mixed_signals.nulls.plan import Statistic
from mixed_signals.prng import derive_key
from mixed_signals.sigma.calibration import knn_kl_curve
from mixed_signals.sigma.maps import (
    SUBTRACTION_SPACES,
    SubtractionSpace,
    null_subtract,
    r_eff,
    sigma_eff,
)
from mixed_signals.sigma.thresholds import LOG_R_EFF_THRESHOLD, SIGMA_EFF_THRESHOLD
from mixed_signals.zoo import closed_forms
from zoo_validation import figures
from zoo_validation.common import Builder, compare_on_plan, compare_with_constants
from zoo_validation.report import (
    Case,
    Cell,
    Check,
    Claim,
    Context,
    Figure,
    Measurement,
    Section,
    Table,
    claim_cases,
    claim_true,
    error_text,
    grouped,
    role,
)

RECOVERY_LABEL = "effect-recovery"

SHIFTED_GAUSSIANS = role("func", "mixed_signals.zoo.cases.shifted_gaussians")
WIDTH_LABEL = "width-recovery"
SIGNED_LABEL = "signed-maps"
THRESHOLD_LABEL = "threshold-basis"
KNN_LABEL = "knn-kl-calibration"

PARAM_COUNT = 4
SAMPLE_COUNT = 1_000
LARGE_SAMPLE_COUNT = 10_000
SHIFTS = (0.0, 0.2, 0.5, 1.0, 2.0)
QUICK_SHIFTS = (0.0, 0.5, 2.0)
RELATIVE_TOLERANCE = 0.03
KS_RELATIVE_TOLERANCE = 0.06
MINIMUM_CLAIMED_SHIFT = 0.5
SIGNED_REPLICATES = (40, 4)
"""Replicates of the cheap rows of ``signed-maps`` in a full and a quick run."""
OUTSIDE_LEGEND_HEIGHT = 1.4
"""Inches that the recovery figure adds for the legends below its panels."""
SMALL_SHIFT = 0.2
"""The shift that no recovery claim covers, where the default threshold sits."""


class Row(NamedTuple):
    """A comparison row: the statistic that computes it and the map that reads it."""

    key: str
    display: str
    group: str
    """The statistic group whose null draws give the row."""
    position: int
    """Position of the row in the vector that its group's statistic returns."""
    metric: str
    """Name for ``null_subtract`` and ``sigma_eff``."""
    estimator: str | None = None


ROWS = (
    Row("ks", "KS", "fast", 0, "ks"),
    Row("w1", "W1", "fast", 1, "w1"),
    Row("ad", "AD", "fast", 2, "ad"),
    Row("js kde", "JS, KDE", "kde", 0, "js", "kde"),
    Row("hellinger kde", "Hellinger, KDE", "kde", 1, "hellinger", "kde"),
    Row("kl gaussian", "KL, Gaussian", "fast", 3, "kl"),
    Row("w2 gaussian", "W2, Gaussian", "fast", 4, "w2"),
    Row("hellinger gaussian", "Hellinger, Gaussian", "fast", 5, "hellinger"),
    Row("energy", "Energy distance", "fast", 6, "energy"),
    Row("mmd", "MMD squared", "fast", 7, "mmd"),
    Row("sliced gaussian", "Sliced W2, Gaussian", "fast", 8, "sliced_w2", "gaussian"),
    Row("sliced empirical", "Sliced W2, empirical", "sliced", 0, "sliced_w2"),
    Row("sinkhorn", "Sinkhorn divergence", "sinkhorn", 0, "sinkhorn"),
)
ROW_BY_KEY = {row.key: row for row in ROWS}

PANELS = (
    ("Marginal metrics", ("ks", "w1", "ad")),
    ("Marginal divergences, KDE", ("js kde", "hellinger kde")),
    ("Gaussian metrics", ("kl gaussian", "w2 gaussian", "hellinger gaussian")),
    ("Kernel and transport", ("energy", "mmd", "sinkhorn")),
    ("Sliced W2", ("sliced empirical", "sliced gaussian")),
)
"""Panels of the recovery figure, at most three series each."""


def is_squared(row: Row) -> bool:
    """Return whether the metric of a row is subtracted in squares."""
    return SUBTRACTION_SPACES[row.metric] is SubtractionSpace.SQUARED


class Outcome(NamedTuple):
    """The effect size of one row of one replicate."""

    signed: float
    """σ_eff after the null subtraction of the metric's own space."""
    plain: float
    """σ_eff after plain subtraction, for contrast."""
    subtracted: float
    """The null-subtracted value, before the map."""
    sample_shift: float
    """Difference of the sample means along the shifted axis, the shift of the sets."""


class Fixed(NamedTuple):
    """Settings that a comparison fixes once and every null draw reuses."""

    bandwidth: Array
    directions: Array
    kde_factor: float


def _marginal_vector(
    first: Array, first_weights: Array, second: Array, second_weights: Array
) -> Array:
    first_column, second_column = first[:, 0], second[:, 0]
    return jnp.stack(
        [
            ks_statistic(first_column, first_weights, second_column, second_weights),
            wasserstein_distance(
                first_column, first_weights, second_column, second_weights
            ),
            anderson_darling_distance(
                first_column, first_weights, second_column, second_weights
            ),
        ]
    )


@functools.cache
def _group_builders(space: ParamSpace, kde_factor: float) -> dict[str, Builder]:
    """Return, for each statistic group, the function that builds its statistic.

    A builder takes the settings of one comparison as arrays, so the compiled
    draws of a group serve every replicate.
    """
    first_space = ParamSpace(parameters=(space.parameters[0],))

    def fast(bandwidth: Array, directions: Array, key: PRNGKeyArray) -> Statistic:
        def statistic(
            first: Array, first_weights: Array, second: Array, second_weights: Array
        ) -> Array:
            mean_first, covariance_first = weighted_covariance(first, first_weights)
            mean_second, covariance_second = weighted_covariance(second, second_weights)
            gaussian = [
                function(
                    mean_first, covariance_first, mean_second, covariance_second
                ).total
                for function in (gaussian_kl, gaussian_w2, gaussian_hellinger)
            ]
            joint = [
                energy_distance(
                    first, first_weights, second, second_weights, space, key=key
                ),
                maximum_mean_discrepancy(
                    first,
                    first_weights,
                    second,
                    second_weights,
                    space,
                    bandwidth=bandwidth,
                    key=key,
                ),
                gaussian_sliced_wasserstein_distance(
                    first,
                    first_weights,
                    second,
                    second_weights,
                    space,
                    directions=directions,
                ),
            ]
            return jnp.concatenate(
                [
                    _marginal_vector(first, first_weights, second, second_weights),
                    jnp.stack(gaussian),
                    jnp.stack(joint),
                ]
            )

        return statistic

    def kde() -> Statistic:
        def statistic(
            first: Array, first_weights: Array, second: Array, second_weights: Array
        ) -> Array:
            divergences = marginal_kde_divergences(
                first[:, :1],
                first_weights,
                second[:, :1],
                second_weights,
                first_space,
                factor=kde_factor,
            )
            return jnp.stack([divergences.js[0], divergences.hellinger[0]])

        return statistic

    def sliced(directions: Array) -> Statistic:
        def statistic(
            first: Array, first_weights: Array, second: Array, second_weights: Array
        ) -> Array:
            return sliced_wasserstein_distance(
                first,
                first_weights,
                second,
                second_weights,
                space,
                directions=directions,
            )[None]

        return statistic

    def sinkhorn(bandwidth: Array, key: PRNGKeyArray) -> Statistic:
        def statistic(
            first: Array, first_weights: Array, second: Array, second_weights: Array
        ) -> Array:
            return sinkhorn_divergence(
                first,
                first_weights,
                second,
                second_weights,
                space,
                epsilon=SINKHORN_EPSILON_FACTOR * bandwidth**2,
                key=key,
            )[None]

        return statistic

    return {"fast": fast, "kde": kde, "sliced": sliced, "sinkhorn": sinkhorn}


def _group_constants(group: str, fixed: Fixed, key: PRNGKeyArray) -> tuple[Array, ...]:
    """Return the settings that a group's builder takes, in its order."""
    return {
        "fast": (fixed.bandwidth, fixed.directions, key),
        "kde": (),
        "sliced": (fixed.directions,),
        "sinkhorn": (fixed.bandwidth, key),
    }[group]


def _map_effect(row: Row, subtracted: Array, fixed: Fixed) -> float:
    """Apply the σ_eff map of a row, with the settings of the comparison."""
    kernel_like = row.metric in ("energy", "mmd", "sliced_w2")
    mapped = sigma_eff(
        row.metric,
        subtracted,
        estimator=row.estimator if row.estimator == "kde" else None,
        bandwidth=(
            fixed.kde_factor
            if row.estimator == "kde"
            else fixed.bandwidth
            if row.metric == "mmd"
            else None
        ),
        dimension=PARAM_COUNT if kernel_like else None,
    )
    return float(mapped)


def _sample_shift(first: Array, second: Array) -> float:
    """Return the difference of the means along the first axis, in widths."""
    return float(jnp.mean(second[:, 0]) - jnp.mean(first[:, 0]))


def replicate_outcomes(
    context: Context,
    label: str,
    space: ParamSpace,
    pair: tuple[Array, Array],
    *,
    tag: Sequence[str | int],
    draws: Mapping[str, int],
) -> dict[str, Outcome]:
    """Return the effect size of every row of one replicate.

    Only the statistic groups that ``draws`` names run.

    Args:
        context: The build context.
        label: Label of the check, the root of every key.
        space: Space of the samples.
        pair: The two sample sets of the replicate.
        tag: Names that identify the replicate in the keys.
        draws: Number of null draws of each statistic group that runs.

    Returns:
        The outcome of each row whose group runs, by row key.
    """
    first, second = pair
    path = tuple(tag)
    fixed = Fixed(
        bandwidth=median_bandwidth(
            first, None, second, None, space, key=context.key(label, "bandwidth", *path)
        ),
        directions=sliced_directions(
            space, SLICED_DIRECTIONS, key=context.key(label, "directions", *path)
        ),
        kde_factor=float(
            shared_bandwidth_factor(
                jnp.zeros(first.shape[0]), jnp.zeros(second.shape[0])
            )
        ),
    )
    builders = _group_builders(space, fixed.kde_factor)
    subsample_key = context.key(label, "subsample")
    outcomes: dict[str, Outcome] = {}
    for group, draw_count in draws.items():
        plan = permutation_null(
            first,
            None,
            second,
            None,
            key=context.key(label, "null", *path, group),
            draws=draw_count,
        )
        result = compare_with_constants(
            builders[group],
            _group_constants(group, fixed, subsample_key),
            plan,
            first,
            None,
            second,
            None,
        )
        for row in ROWS:
            if row.group != group:
                continue
            value, null = result.value[row.position], result.draws[row.position]
            subtracted = null_subtract(row.metric, value, null)
            finite = jnp.isfinite(null)
            plain = value - jnp.sum(jnp.where(finite, null, 0.0)) / jnp.sum(finite)
            outcomes[row.key] = Outcome(
                signed=_map_effect(row, subtracted, fixed),
                plain=_map_effect(row, plain, fixed),
                subtracted=float(subtracted),
                sample_shift=_sample_shift(first, second),
            )
    return outcomes


def _large_marginal_outcomes(
    context: Context,
    label: str,
    space: ParamSpace,
    pair: tuple[Array, Array],
    *,
    tag: Sequence[str | int],
    draw_count: int,
) -> dict[str, Outcome]:
    """Return the KS, W1 and AD effect sizes of one replicate at large N."""
    first, second = pair
    plan = permutation_null(
        first,
        None,
        second,
        None,
        key=context.key(label, "null", *tag, "large"),
        draws=draw_count,
    )
    result = compare_on_plan(_marginal_vector, plan, first, None, second, None)
    outcomes: dict[str, Outcome] = {}
    for row in ROWS[:3]:
        value, null = result.value[row.position], result.draws[row.position]
        subtracted = null_subtract(row.metric, value, null)
        plain = value - jnp.mean(null)
        fixed = Fixed(jnp.asarray(1.0), jnp.zeros((1, PARAM_COUNT)), 0.0)
        outcomes[row.key] = Outcome(
            _map_effect(row, subtracted, fixed),
            _map_effect(row, plain, fixed),
            float(subtracted),
            _sample_shift(first, second),
        )
    return outcomes


def _deviation_claim(
    statement: str,
    labels: Sequence[str],
    deviations: Sequence[Measurement],
    scales: Sequence[float],
    relative: float,
    *,
    full_only: bool = True,
) -> Claim:
    """State that each mean deviation is within a share of its scale plus its errors.

    Args:
        statement: The claim.
        labels: Name of each case.
        deviations: Mean deviation of each case from its expectation, with the
            standard error of the mean: a paired difference, or a mean minus
            its truth.
        scales: The truth that the relative tolerance applies to, per case.
        relative: Relative tolerance.
        full_only: Whether the quick replicates are too few for the tolerance.

    Returns:
        The claim, which names its worst case.
    """
    cases = [
        Case(label, deviation.value, deviation.margin(), relative * abs(scale))
        for label, deviation, scale in zip(labels, deviations, scales, strict=True)
    ]
    return claim_cases(
        statement,
        cases,
        tolerance_text=f"{relative:.0%} of the truth",
        replicates=deviations[0].replicates,
        full_only=full_only,
    )


def _zero_claim(statement: str, measurement: Measurement) -> Claim:
    """State that a mean lies within its standard errors of zero."""
    return claim_cases(
        statement,
        [Case("mean", measurement.value, measurement.margin(), 0.0)],
        tolerance_text=None,
        replicates=measurement.replicates,
    )


# * Check effect-recovery.


def _without_zero_shift(by_shift: dict[float, Measurement]) -> list[Cell]:
    """Return the cells of a row over the shifts, with a dash at the zero shift."""
    return [None if shift == 0.0 else entry for shift, entry in by_shift.items()]


def _small_shift_text(
    paired_means: dict[str, dict[float, Measurement]], claimed: Sequence[float]
) -> str:
    """Name the metrics that deviate most from the sample shift at the small shift.

    Returns:
        A paragraph, empty when the run holds no such shift.
    """
    if not any(SMALL_SHIFT in by_shift for by_shift in paired_means.values()):
        return ""
    ranked = sorted(
        (
            (
                by_shift[SMALL_SHIFT].value / SMALL_SHIFT,
                (by_shift[SMALL_SHIFT].error or 0.0) / SMALL_SHIFT,
                key,
            )
            for key, by_shift in paired_means.items()
        ),
        key=lambda entry: abs(entry[0]),
        reverse=True,
    )
    exceeding = [
        f"{ROW_BY_KEY[key].display} ({100 * deviation:+.1f} ± {100 * error:.1f}%)"
        for deviation, error, key in ranked
        if abs(deviation)
        > (KS_RELATIVE_TOLERANCE if key == "ks" else RELATIVE_TOLERANCE)
    ]
    ending = f", largest first: {', '.join(exceeding)}." if exceeding else "."
    return (
        f"No claim covers the shift of {SMALL_SHIFT:g}, which lies below the smallest"
        f" claimed shift of {claimed[0]:g} and is where the default threshold of the"
        " shift sits. There the paired deviation over the shift exceeds the"
        f" tolerance of the claims for {len(exceeding)} of {len(ranked)} metrics"
        + ending
    )


def _effect_recovery(context: Context) -> Section:
    label = RECOVERY_LABEL
    shifts = QUICK_SHIFTS if context.quick else SHIFTS
    large_count = context.count(LARGE_SAMPLE_COUNT, 3_000)
    replicates = context.count(12, 2)
    large_replicates = context.count(5, 2)
    draws = {
        "fast": context.count(99, 19),
        "kde": context.count(49, 9),
        "sliced": context.count(30, 5),
        "sinkhorn": context.count(20, 4),
    }
    large_draws = context.count(49, 19)
    space = zoo.shifted_gaussians(0.0, param_count=PARAM_COUNT).space
    keys = [row.key for row in ROWS]
    signed: dict[str, dict[float, list[float]]] = {
        key: {shift: [] for shift in shifts} for key in keys
    }
    plain: dict[str, dict[float, list[float]]] = {
        key: {shift: [] for shift in shifts} for key in keys
    }
    paired: dict[str, dict[float, list[float]]] = {
        key: {shift: [] for shift in shifts} for key in keys
    }
    sample_shifts: dict[float, list[float]] = {shift: [] for shift in shifts}
    large: dict[str, dict[float, list[float]]] = {
        key: {shift: [] for shift in shifts} for key in keys[:3]
    }
    large_paired: dict[str, dict[float, list[float]]] = {
        key: {shift: [] for shift in shifts} for key in keys[:3]
    }
    large_shifts: dict[float, list[float]] = {shift: [] for shift in shifts}
    for shift_index, shift in enumerate(shifts):
        case = zoo.shifted_gaussians(shift, param_count=PARAM_COUNT)
        sets = case.sample(
            context.key(label, "sets", shift_index),
            SAMPLE_COUNT,
            batch_size=replicates,
        ).samples
        for replicate in range(replicates):
            outcomes = replicate_outcomes(
                context,
                label,
                space,
                (sets[replicate, 0], sets[replicate, 1]),
                tag=(shift_index, replicate),
                draws=draws,
            )
            for key, outcome in outcomes.items():
                signed[key][shift].append(outcome.signed)
                plain[key][shift].append(outcome.plain)
                paired[key][shift].append(outcome.signed - outcome.sample_shift)
            sample_shifts[shift].append(outcomes[keys[0]].sample_shift)
        large_sets = case.sample(
            context.key(label, "large-sets", shift_index),
            large_count,
            batch_size=large_replicates,
        ).samples
        for replicate in range(large_replicates):
            outcomes = _large_marginal_outcomes(
                context,
                label,
                space,
                (large_sets[replicate, 0], large_sets[replicate, 1]),
                tag=(shift_index, replicate),
                draw_count=large_draws,
            )
            for key, outcome in outcomes.items():
                large[key][shift].append(outcome.signed)
                large_paired[key][shift].append(outcome.signed - outcome.sample_shift)
            large_shifts[shift].append(outcomes[keys[0]].sample_shift)
    means = _measured(signed)
    paired_means = _measured(paired)
    large_means = _measured(large)
    large_paired_means = _measured(large_paired)
    plain_means = _measured(
        {key: by for key, by in plain.items() if is_squared(ROW_BY_KEY[key])}
    )
    shift_means = [
        Measurement.from_replicates(sample_shifts[shift]) for shift in shifts
    ]
    large_shift_means = [
        Measurement.from_replicates(large_shifts[shift]) for shift in shifts
    ]
    shift_columns = tuple(f"Shift {shift:g}" for shift in shifts)
    recovery_rows: list[list[Cell]] = [["Sample shift", *shift_means]]
    recovery_rows.extend([row.display, *means[row.key].values()] for row in ROWS)
    large_rows: list[list[Cell]] = [["Sample shift", *large_shift_means]]
    large_rows.extend(
        [ROW_BY_KEY[key].display, *by.values()] for key, by in large_means.items()
    )
    large_rows.extend(
        [f"{ROW_BY_KEY[key].display}, paired", *_without_zero_shift(by)]
        for key, by in large_paired_means.items()
    )
    recovery = Table.of(
        f"Mean σ_eff over replicates of {grouped(SAMPLE_COUNT)} samples per set,"
        " with the standard error of the mean, by true shift. The first row is"
        " the mean difference of the sample means along the shifted axis, the"
        " shift that the sets hold.",
        ("Metric", *shift_columns),
        recovery_rows,
    )
    paired_table = Table.of(
        f"Paired deviation, σ_eff minus the sample shift of its own replicate, at"
        f" {grouped(SAMPLE_COUNT)} samples per set, with the standard error of the"
        " paired differences. A dash marks the zero shift, which no claim covers.",
        ("Metric", *shift_columns),
        [[row.display, *_without_zero_shift(paired_means[row.key])] for row in ROWS],
    )
    large_table = Table.of(
        f"Mean σ_eff of the marginal metrics at {grouped(large_count)} samples per"
        " set, and the paired deviation from the sample shift in the second block,"
        " which has a dash at the zero shift.",
        ("Metric", *shift_columns),
        large_rows,
    )
    plain_table = Table.of(
        "Mean σ_eff of the metrics that subtract in squares, when the null mean of"
        " the values is subtracted plainly instead: they read low.",
        ("Metric", *shift_columns),
        [[ROW_BY_KEY[key].display, *by.values()] for key, by in plain_means.items()],
    )
    claims: list[Claim] = []
    claimed = [shift for shift in shifts if shift >= MINIMUM_CLAIMED_SHIFT]
    shift_labels = [f"shift {shift:g}" for shift in claimed]
    for row in ROWS:
        relative = KS_RELATIVE_TOLERANCE if row.key == "ks" else RELATIVE_TOLERANCE
        claims.append(
            _deviation_claim(
                f"{row.display} recovers the sample shift at"
                f" {', '.join(f'{shift:g}' for shift in claimed)}.",
                shift_labels,
                [paired_means[row.key][shift] for shift in claimed],
                claimed,
                relative,
            )
        )
    for key in keys[:3]:
        relative = KS_RELATIVE_TOLERANCE if key == "ks" else RELATIVE_TOLERANCE
        claims.append(
            _deviation_claim(
                f"{ROW_BY_KEY[key].display} recovers the sample shift at"
                f" {grouped(large_count)} samples per set.",
                shift_labels,
                [large_paired_means[key][shift] for shift in claimed],
                claimed,
                relative,
            )
        )
    plotted = [shift for shift in shifts if shift > 0]
    data = {
        "shifts": shifts,
        "sigma_eff": {
            key: [[by[shift].value, by[shift].error] for shift in shifts]
            for key, by in means.items()
        },
        "paired_deviation": {
            key: [[by[shift].value, by[shift].error] for shift in shifts]
            for key, by in paired_means.items()
        },
        "sample_shift": [[entry.value, entry.error] for entry in shift_means],
    }

    def draw(style: figures.FigureStyle) -> MatplotlibFigure:
        figure, panels = figures.new_figure(
            style, rows=2, columns=3, sharey=False, extra_height=OUTSIDE_LEGEND_HEIGHT
        )
        for axes, (title, panel_keys) in zip(panels, PANELS, strict=False):
            figures.plot_tolerance_band(axes, style, RELATIVE_TOLERANCE)
            for series, key in enumerate(panel_keys):
                figures.plot_series(
                    axes,
                    style,
                    series,
                    plotted,
                    [paired_means[key][shift].value / shift for shift in plotted],
                    label=ROW_BY_KEY[key].display,
                    errors=[
                        (paired_means[key][shift].error or 0.0) / shift
                        for shift in plotted
                    ],
                )
            figures.plot_reference(axes, style, y_value=0.0)
            figures.finish_panel(
                axes,
                style,
                title=title,
                x_label="true shift",
                y_label="paired deviation over shift",
            )
            axes.set_xlim(0.0, 2.1)
            figures.add_legend(axes, style, outside=True)
        figures.hide_unused(panels, len(PANELS))
        return figure

    unclaimed_note = (
        f"; the points at {SMALL_SHIFT:g} carry no claim"
        if SMALL_SHIFT in shifts
        else ""
    )
    figure = Figure(
        label=f"{label}-shifts",
        caption="Paired deviation of σ_eff from the sample shift of its own"
        " replicate, over the true shift, against the true shift. The line is zero,"
        f" the band is the {RELATIVE_TOLERANCE:.0%} tolerance of the claims"
        f" ({KS_RELATIVE_TOLERANCE:.0%} for the KS statistic) and the bars are"
        " standard errors of the paired differences. The claims cover the shifts"
        f" {claimed[0]:g} to {claimed[-1]:g}{unclaimed_note}.",
        alt="Five panels of the paired relative deviation against the true shift,"
        " scattered around zero within a narrow band.",
        draw=draw,
        data=data,
    )
    setup = (
        f"Shifted zoo Gaussians ({SHIFTED_GAUSSIANS},"
        f" {PARAM_COUNT} parameters) with true shifts"
        f" {', '.join(f'{shift:g}' for shift in shifts)}, {grouped(SAMPLE_COUNT)}"
        f" samples per set and {replicates} replicates per shift. Each row uses the"
        " default permutation null, then"
        f" {role('func', 'mixed_signals.sigma.maps.null_subtract')} and"
        f" {role('func', 'mixed_signals.sigma.maps.sigma_eff')} with the settings of"
        " the row. The marginal metrics read the first parameter, which carries the"
        " shift. The JS divergence and the Hellinger distance come from kernel"
        " estimates with the bandwidth factor of"
        f" {role('func', 'mixed_signals.metrics.marginal.shared_bandwidth_factor')},"
        " and the Gaussian KL, W2 and Hellinger distance from the weighted moments."
        " MMD squared takes the median bandwidth of the comparison, fixed once, and"
        " the Sinkhorn divergence takes ε at"
        f" {SINKHORN_EPSILON_FACTOR} times its square. Both sliced W2 estimators"
        f" share {SLICED_DIRECTIONS} directions. The null has"
        f" {draws['fast']} draws for the cheap metrics, {draws['kde']} for the"
        f" kernel estimates, {draws['sliced']} for the empirical sliced W2 and"
        f" {draws['sinkhorn']} for the Sinkhorn divergence, fewer than the default"
        f" 200. The marginal KS, W1 and AD also run at {grouped(large_count)}"
        f" samples per set with {large_replicates} replicates. The sample means of"
        " a replicate differ by a shift of their own, which scatters by"
        f" about 0.045 at {grouped(SAMPLE_COUNT)} samples, and every metric follows"
        " it, so a mean σ_eff compared with the true shift carries that scatter. A"
        " claim therefore tests the mean of the paired difference, σ_eff minus the"
        " sample shift of the same replicate, which has the same expectation as"
        " σ_eff minus the truth and a far smaller scatter. The tolerance is 3% of"
        " the true shift (6% for the KS statistic, which keeps a small positive"
        " bias), plus the standard errors of the paired differences. The zero shift"
        f" has no claim here; the check {{ref}}`validation-{SIGNED_LABEL}` covers it"
        f" with {context.count(*SIGNED_REPLICATES)} replicates."
    )
    return Section(
        setup,
        (
            recovery,
            paired_table,
            _small_shift_text(paired_means, claimed),
            figure,
            large_table,
            plain_table,
        ),
        tuple(claims),
    )


def _measured(
    values: dict[str, dict[float, list[float]]],
) -> dict[str, dict[float, Measurement]]:
    """Return the mean and the standard error of every list of replicate values."""
    return {
        key: {
            shift: Measurement.from_replicates(entries) for shift, entries in by.items()
        }
        for key, by in values.items()
    }


# * Check width-recovery.

WIDTH_ROWS = (
    ("width_ratio", "Width ratio"),
    ("kl_shape", "KL, shape term"),
    ("w2_shape", "W2, shape term"),
    ("hellinger_shape", "Hellinger, shape term"),
)
WIDTH_RATIOS = (0.8, 1.25, 1.5)


def _width_statistic(
    space: ParamSpace,
) -> Statistic:
    def statistic(
        first: Array, first_weights: Array, second: Array, second_weights: Array
    ) -> Array:
        mean_first, covariance_first = weighted_covariance(first, first_weights)
        mean_second, covariance_second = weighted_covariance(second, second_weights)
        terms = [
            function(mean_first, covariance_first, mean_second, covariance_second)
            for function in (gaussian_kl, gaussian_w2, gaussian_hellinger)
        ]
        return jnp.stack(
            [
                width_ratio(first, first_weights, second, second_weights, space)[0],
                *[term.shape for term in terms],
                terms[0].log_det_ratio,
            ]
        )

    return statistic


def _width_recovery(context: Context) -> Section:
    label = WIDTH_LABEL
    replicates = context.count(60, 4)
    draws = context.count(99, 19)
    results: dict[str, dict[float, list[float]]] = {
        key: {ratio: [] for ratio in WIDTH_RATIOS} for key, _ in WIDTH_ROWS
    }
    for ratio_index, ratio in enumerate(WIDTH_RATIOS):
        case = zoo.scaled_gaussians(ratio, param_count=PARAM_COUNT, axes=(0,))
        statistic = _width_statistic(case.space)
        sets = case.sample(
            context.key(label, "sets", ratio_index),
            SAMPLE_COUNT,
            batch_size=replicates,
        ).samples
        for replicate in range(replicates):
            first, second = sets[replicate, 0], sets[replicate, 1]
            plan = permutation_null(
                first,
                None,
                second,
                None,
                key=context.key(label, "null", ratio_index, replicate),
                draws=draws,
            )
            result = compare_on_plan(statistic, plan, first, None, second, None)
            log_det_ratio = result.value[4]
            for index, (key, _) in enumerate(WIDTH_ROWS):
                subtracted = null_subtract(
                    key, result.value[index], result.draws[index]
                )
                effect = (
                    r_eff(key, subtracted)
                    if key == "width_ratio"
                    else r_eff(key, subtracted, log_det_ratio)
                )
                results[key][ratio].append(float(effect))
    means = {
        key: {
            ratio: Measurement.from_replicates(values) for ratio, values in by.items()
        }
        for key, by in results.items()
    }
    truths = [
        zoo.scaled_gaussians(ratio, param_count=PARAM_COUNT, axes=(0,)).reference_value(
            "width_ratio", ("reference", "scaled"), "x0"
        )
        for ratio in WIDTH_RATIOS
    ]
    table = Table.of(
        f"Mean r_eff over {replicates} replicates, with the standard error of the"
        " mean, by true width ratio.",
        ("Metric", *[f"Ratio {ratio:g}" for ratio in WIDTH_RATIOS]),
        [[display, *means[key].values()] for key, display in WIDTH_ROWS],
    )
    claims = [
        _deviation_claim(
            f"{display} recovers the width ratios"
            f" {', '.join(f'{ratio:g}' for ratio in WIDTH_RATIOS)}.",
            [f"ratio {ratio:g}" for ratio in WIDTH_RATIOS],
            [
                Measurement(entry.value - truth, entry.error, entry.replicates)
                for entry, truth in zip(means[key].values(), truths, strict=True)
            ],
            truths,
            RELATIVE_TOLERANCE,
        )
        for key, display in WIDTH_ROWS
    ]
    setup = (
        f"Zoo Gaussians ({role('func', 'mixed_signals.zoo.cases.scaled_gaussians')},"
        f" {PARAM_COUNT} parameters) with the first axis scaled by 0.8, 1.25 and 1.5,"
        f" {grouped(SAMPLE_COUNT)} samples per set, {replicates} replicates and"
        f" {draws} null draws per replicate. The width ratio of the first parameter"
        " is subtracted in logarithms, and the shape terms of the Gaussian KL, W2"
        " and Hellinger distance in the space of their metric; the sign of the"
        " log-determinant ratio chooses the side of one that r_eff takes. A claim"
        " is consistent if the mean lies within 3% of the true ratio plus its"
        " standard errors."
    )
    return Section(setup, (table,), tuple(claims))


# * Check signed-maps.


def _scatter_of(values: Array) -> Measurement:
    """Return a standard deviation with the standard error of a normal sample."""
    spread = float(jnp.std(values, ddof=1))
    return Measurement(spread, spread / math.sqrt(2 * (values.shape[0] - 1)))


def _signed_maps(context: Context) -> Section:
    label = SIGNED_LABEL
    fast_replicates = context.count(*SIGNED_REPLICATES)
    slow_replicates = context.count(10, 2)
    draws = {
        "fast": context.count(49, 9),
        "kde": context.count(29, 9),
        "sliced": context.count(20, 5),
        "sinkhorn": context.count(15, 4),
    }
    space = zoo.shifted_gaussians(0.0, param_count=PARAM_COUNT).space
    sets = (
        zoo.shifted_gaussians(0.0, param_count=PARAM_COUNT)
        .sample(context.key(label, "sets"), SAMPLE_COUNT, batch_size=fast_replicates)
        .samples
    )
    collected: dict[str, list[Outcome]] = {row.key: [] for row in ROWS}
    for replicate in range(fast_replicates):
        running = (
            draws
            if replicate < slow_replicates
            else {group: count for group, count in draws.items() if group == "fast"}
        )
        outcomes = replicate_outcomes(
            context,
            label,
            space,
            (sets[replicate, 0], sets[replicate, 1]),
            tag=(replicate,),
            draws=running,
        )
        for key, outcome in outcomes.items():
            collected[key].append(outcome)
    rows: list[list[Cell]] = []
    claims: list[Claim] = []
    for row in ROWS:
        row_outcomes = collected[row.key]
        subtracted = jnp.array([outcome.subtracted for outcome in row_outcomes])
        signed = jnp.array([outcome.signed for outcome in row_outcomes])
        clipped = jnp.maximum(signed, 0.0)
        negative = Measurement.from_rate(
            int(jnp.sum(subtracted < 0)), len(row_outcomes)
        )
        signed_mean = Measurement.from_replicates(signed)
        clipped_mean = Measurement.from_replicates(clipped)
        rows.append(
            [
                row.display,
                len(row_outcomes),
                negative,
                signed_mean,
                _scatter_of(signed),
                clipped_mean,
                _scatter_of(clipped),
            ]
        )
        claims.append(_zero_claim(f"{row.display}: signed σ_eff.", signed_mean))
    table = Table.of(
        "Identical sets: the share of negative null-subtracted values, and the mean"
        " and scatter of the signed σ_eff and of σ_eff clipped at zero.",
        (
            "Metric",
            "Replicates",
            "Negative share",
            "Signed mean",
            "Signed scatter",
            "Clipped mean",
            "Clipped scatter",
        ),
        rows,
    )
    names = {
        group: ", ".join(row.display for row in ROWS if row.group == group)
        for group in draws
    }
    setup = (
        f"Two sets of {grouped(SAMPLE_COUNT)} samples of one zoo Gaussian"
        f" ({PARAM_COUNT} parameters). The rows of the cheap group ({names['fast']})"
        f" take {fast_replicates} replicates with {draws['fast']} null draws. The"
        f" other rows take {slow_replicates} replicates: the kernel estimates"
        f" ({names['kde']}) with {draws['kde']} draws, the empirical sliced W2 with"
        f" {draws['sliced']} and the Sinkhorn divergence with {draws['sinkhorn']},"
        f" so the null draws run from {min(draws.values())} to"
        f" {max(draws.values())}. Each row is null-subtracted and mapped to σ_eff"
        " with the signed map of the package. For contrast the script clips the"
        " same σ_eff at zero, which a map that gives NaN or zero below the null"
        " would do. Clipping raises the mean by construction, because a clipped"
        " value is never below its signed value; the table shows by how much. The"
        " claims cover the signed σ_eff only."
    )
    return Section(setup, (table,), tuple(claims))


# * Check threshold-basis.

THRESHOLD_SHIFT = 0.2
BISECTION_STEPS = 60
WIDTH_RATIO_LIMIT = 4.0


def _width_ratio_with_js(target: float, *, above: bool) -> float:
    """Return the width ratio whose JS divergence with one equals ``target`` (nats)."""
    lower, upper = (1.0, WIDTH_RATIO_LIMIT) if above else (1 / WIDTH_RATIO_LIMIT, 1.0)
    for _ in range(BISECTION_STEPS):
        middle = 0.5 * (lower + upper)
        divergence = float(closed_forms.normal_js(0.0, 1.0, 0.0, middle))
        # * The divergence grows with the distance of the ratio from one.
        too_far = divergence > target
        if above:
            lower, upper = (lower, middle) if too_far else (middle, upper)
        else:
            lower, upper = (middle, upper) if too_far else (lower, middle)
    return 0.5 * (lower + upper)


def _threshold_basis(context: Context) -> Section:
    del context
    shift_js = float(closed_forms.normal_js(0.0, 1.0, THRESHOLD_SHIFT, 1.0))
    shift_bits = shift_js / math.log(2)
    wider = _width_ratio_with_js(shift_js, above=True)
    narrower = _width_ratio_with_js(shift_js, above=False)
    rows: list[list[Cell]] = [
        ["Shift of 0.2σ", THRESHOLD_SHIFT, shift_bits, "-"],
        ["Width ratio above one", wider, shift_bits, math.log(wider)],
        ["Width ratio below one", narrower, shift_bits, math.log(narrower)],
    ]
    table = Table.of(
        "Gaussians with the JS divergence of a 0.2σ shift: the shift, the"
        " width ratios on either side of one and their natural logarithms.",
        ("Case", "Shift or ratio", "JS divergence (bits)", "ln r"),
        rows,
    )
    claims = [
        claim_true(
            "A 0.2σ shift has the JS divergence that GWTC-2 deems significant.",
            "0.0072 bits, at least the 0.007 bits of the catalogue",
            f"{shift_bits:.5f} bits",
            abs(shift_bits - 0.0072) < 5e-5 and shift_bits >= 0.007,
        ),
        claim_true(
            "The shift threshold is 0.2.",
            f"{SIGMA_EFF_THRESHOLD}",
            f"{THRESHOLD_SHIFT}",
            math.isclose(SIGMA_EFF_THRESHOLD, THRESHOLD_SHIFT),
        ),
        claim_true(
            "The width ratios of the same JS divergence are 1.153 and 0.867.",
            "1.153 and 0.867 to three digits",
            f"{wider:.3f} and {narrower:.3f}",
            round(wider, 3) == 1.153 and round(narrower, 3) == 0.867,
        ),
        claim_true(
            "The threshold of the logarithm of r_eff is 0.14.",
            f"{LOG_R_EFF_THRESHOLD}",
            f"{abs(math.log(wider)):.4f} and {abs(math.log(narrower)):.4f}",
            all(
                round(abs(math.log(ratio)), 2) == LOG_R_EFF_THRESHOLD
                for ratio in (wider, narrower)
            ),
        ),
    ]
    setup = (
        "The GWTC-2 catalogue ({cite:t}`abbott2021gwtc`, Appendix A) deems a JS"
        " divergence of 0.007 bits between waveform models significant, which for"
        " a Gaussian corresponds to a 20% shift in the mean, measured in units of"
        " one standard deviation. The check is exact and has no replicates. The"
        f" JS divergence of N(0, 1) and N({THRESHOLD_SHIFT}, 1) comes from"
        f" {role('func', 'mixed_signals.zoo.closed_forms.normal_js')} in nats, and"
        " the width ratios on either side of one that give the same divergence"
        f" come from a bisection on `normal_js(0, 1, 0, r)` ({BISECTION_STEPS}"
        " steps)."
    )
    return Section(setup, (table,), tuple(claims))


# * Check knn-kl-calibration.

KNN_DIMENSIONS = (1, 4, 16)
KNN_SHIFTS = (0.5, 1.0, 2.0, 3.0)
KNN_RECOVERY_SHIFTS = (1.0, 2.0)
KNN_RECOVERY_DIMENSION = 4
KNN_RECOVERY_COUNT = 1_000


def _knn_estimates(
    space: ParamSpace,
    first: Array,
    second: Array,
    shifts: Sequence[float],
    key: PRNGKeyArray,
) -> Array:
    """Return the kNN KL averaged over both directions at each shift."""
    estimates = []
    for shift in shifts:
        moved = second.at[:, 0].add(shift)
        forward = knn_kl_divergence(
            first, None, moved, None, space, key=key, neighbours=KL_NEIGHBOURS
        )
        backward = knn_kl_divergence(
            moved, None, first, None, space, key=key, neighbours=KL_NEIGHBOURS
        )
        estimates.append(0.5 * (forward + backward))
    return jnp.stack(estimates)


def _knn_ratios(
    context: Context, label: str, size: int, replicates: int
) -> dict[int, list[Measurement]]:
    """Return the null-subtracted estimate over δ²/2, by dimension and shift."""
    ratios: dict[int, list[Measurement]] = {}
    shifts = (0.0, *KNN_SHIFTS)
    for dimension in KNN_DIMENSIONS:
        space = ParamSpace.model_validate(
            {"parameters": [{"name": f"x{index}"} for index in range(dimension)]}
        )
        differences = []
        for replicate in range(replicates):
            key = context.key(label, "sets", size, dimension, replicate)
            first = jax.random.normal(derive_key(key, "first"), (size, dimension))
            second = jax.random.normal(derive_key(key, "second"), (size, dimension))
            estimates = _knn_estimates(
                space, first, second, shifts, derive_key(key, "subsample")
            )
            differences.append(estimates[1:] - estimates[0])
        stacked = jnp.stack(differences)
        ratios[dimension] = [
            Measurement.from_replicates(stacked[:, index] / (shift**2 / 2))
            for index, shift in enumerate(KNN_SHIFTS)
        ]
    return ratios


@functools.cache
def _knn_builder(space: ParamSpace) -> Builder:
    """Return the builder of the kNN KL statistic for a space."""

    def build(key: PRNGKeyArray) -> Statistic:
        def statistic(
            first: Array, first_weights: Array, second: Array, second_weights: Array
        ) -> Array:
            return knn_kl_divergence(
                first, first_weights, second, second_weights, space, key=key
            )

        return statistic

    return build


def _knn_calibration(context: Context) -> Section:
    label = KNN_LABEL
    sizes = (1_000, 3_000, 10_000) if not context.quick else (1_000,)
    replicate_counts = {1_000: context.count(16, 3), 3_000: 4, 10_000: 2}
    ratio_rows: list[list[Cell]] = []
    ratios_by_size: dict[int, dict[int, list[Measurement]]] = {}
    for size in sizes:
        ratios = _knn_ratios(context, label, size, replicate_counts[size])
        ratios_by_size[size] = ratios
        for dimension, measurements in ratios.items():
            ratio_rows.append([size, dimension, replicate_counts[size], *measurements])
    ratio_table = Table.of(
        "The null-subtracted kNN KL (k = 3, both directions averaged) over the true"
        " KL of unit Gaussians, δ²/2, by sample count, dimension and shift.",
        (
            "N",
            "D",
            "Replicates",
            *[f"Shift {shift:g}" for shift in KNN_SHIFTS],
        ),
        ratio_rows,
    )
    recovery_replicates = context.count(16, 2)
    draws = context.count(30, 9)
    recovery_rows: list[list[Cell]] = []
    claims: list[Claim] = []
    space = ParamSpace.model_validate(
        {
            "parameters": [
                {"name": f"x{index}"} for index in range(KNN_RECOVERY_DIMENSION)
            ]
        }
    )
    for shift in KNN_RECOVERY_SHIFTS:
        calibrated, plain, sample_shifts = [], [], []
        for replicate in range(recovery_replicates):
            key = context.key(label, "recovery", int(shift), replicate)
            first = jax.random.normal(
                derive_key(key, "first"), (KNN_RECOVERY_COUNT, KNN_RECOVERY_DIMENSION)
            )
            second = (
                jax.random.normal(
                    derive_key(key, "second"),
                    (KNN_RECOVERY_COUNT, KNN_RECOVERY_DIMENSION),
                )
                .at[:, 0]
                .add(shift)
            )
            subsample_key = derive_key(key, "subsample")

            plan = permutation_null(
                first, None, second, None, key=derive_key(key, "null"), draws=draws
            )
            result = compare_with_constants(
                _knn_builder(space), (subsample_key,), plan, first, None, second, None
            )
            subtracted = null_subtract("kl", result.value, result.draws)
            curve = knn_kl_curve(
                KNN_RECOVERY_DIMENSION,
                KNN_RECOVERY_COUNT,
                KNN_RECOVERY_COUNT,
                key=derive_key(key, "calibration"),
            )
            calibrated.append(
                float(sigma_eff("kl", subtracted, estimator="knn", curve=curve))
            )
            plain.append(float(sigma_eff("kl", subtracted)))
            sample_shifts.append(_sample_shift(first, second))
        calibrated_paired = Measurement.from_replicates(
            [
                entry - held
                for entry, held in zip(calibrated, sample_shifts, strict=True)
            ]
        )
        plain_paired = Measurement.from_replicates(
            [entry - held for entry, held in zip(plain, sample_shifts, strict=True)]
        )
        recovery_rows.append(
            [
                shift,
                Measurement.from_replicates(sample_shifts),
                Measurement.from_replicates(calibrated),
                calibrated_paired,
                Measurement.from_replicates(plain),
                plain_paired,
            ]
        )
        claims.append(
            _deviation_claim(
                f"The calibrated σ_eff recovers the sample shift of {shift:g}.",
                [f"shift {shift:g}"],
                [calibrated_paired],
                [shift],
                0.05,
            )
        )
        claims.append(
            claim_true(
                f"The map √(2 KL) reads the sample shift of {shift:g} low.",
                "paired deviation below zero by more than"
                f" {error_text(recovery_replicates)}",
                plain_paired.render(),
                plain_paired.value + plain_paired.margin() < 0,
                full_only=True,
            )
        )
    recovery_table = Table.of(
        f"σ_eff of the kNN KL at D = {KNN_RECOVERY_DIMENSION} and"
        f" N = {grouped(KNN_RECOVERY_COUNT)}: the run-time calibration curve against"
        " √(2 KL), each with its paired deviation from the sample shift.",
        (
            "True shift",
            "Sample shift",
            "Calibrated σ_eff",
            "Paired deviation",
            "√(2 KL)",
            "Paired deviation, √(2 KL)",
        ),
        recovery_rows,
    )

    def draw(style: figures.FigureStyle) -> MatplotlibFigure:
        figure, panels = figures.new_figure(
            style, columns=len(ratios_by_size), sharey=True
        )
        for index, (axes, (size, ratios)) in enumerate(
            zip(panels, ratios_by_size.items(), strict=True)
        ):
            for series, dimension in enumerate(KNN_DIMENSIONS):
                figures.plot_series(
                    axes,
                    style,
                    series,
                    KNN_SHIFTS,
                    [measurement.value for measurement in ratios[dimension]],
                    label=f"D = {dimension}",
                    errors=[
                        measurement.error or 0.0 for measurement in ratios[dimension]
                    ],
                )
            figures.plot_reference(axes, style, y_value=1.0)
            figures.finish_panel(
                axes,
                style,
                title=f"N = {grouped(size)}",
                x_label="shift",
                y_label="estimate over truth" if index == 0 else None,
            )
            axes.set_ylim(0.4, 1.1)
            if index == 0:
                figures.add_legend(axes, style, location="lower left")
        return figure

    figure = Figure(
        label=f"{label}-ratio",
        caption="Null-subtracted kNN KL over its truth against the shift, by sample"
        " count and dimension; the line is one.",
        alt="Lines of the estimate over the truth against the shift, below one for"
        " four and sixteen dimensions and falling with the shift.",
        draw=draw,
        data={
            "shifts": KNN_SHIFTS,
            "ratios": {
                str(size): {
                    str(dimension): [
                        [entry.value, entry.error] for entry in measurements
                    ]
                    for dimension, measurements in ratios.items()
                }
                for size, ratios in ratios_by_size.items()
            },
        },
    )
    counts_text = ", ".join(
        f"{replicate_counts[size]} at N = {grouped(size)}" for size in sizes
    )
    setup = (
        "Unit Gaussians, the second set shifted along one axis, with common random"
        " numbers across shifts as"
        f" {role('func', 'mixed_signals.sigma.calibration.knn_kl_curve')} draws"
        " them. The estimate is"
        f" {role('func', 'mixed_signals.metrics.divergence.knn_kl_divergence')}"
        f" with k = {KL_NEIGHBOURS}, averaged over both directions, and its null"
        " is the mean estimate at zero shift of the same replicate. Replicates:"
        f" {counts_text}."
        f" The recovery uses {recovery_replicates} comparisons per shift with"
        f" {draws} permutation draws, each with its own calibration curve."
        " A claim tests the mean of the paired difference between σ_eff and the"
        " sample shift of the same replicate, the difference of the sample means"
        " along the shifted axis; the tolerance is 5% of the true shift plus the"
        " standard errors of the paired differences."
    )
    return Section(setup, (ratio_table, figure, recovery_table), tuple(claims))


CHECKS: tuple[Check, ...] = (
    Check(
        label=RECOVERY_LABEL,
        title="Recovery of known shifts",
        session="12a",
        supports=(
            role("func", "mixed_signals.sigma.maps.sigma_eff", short=False),
            role("func", "mixed_signals.sigma.maps.null_subtract", short=False),
            role("data", "mixed_signals.sigma.maps.SUBTRACTION_SPACES", short=False),
            "the done criterion of session 12: σ_eff recovers the zoo's shifts",
        ),
        run=_effect_recovery,
    ),
    Check(
        label=WIDTH_LABEL,
        title="Recovery of known width ratios",
        session="12a",
        supports=(
            role("func", "mixed_signals.sigma.maps.r_eff", short=False),
            "the done criterion of session 12: r_eff recovers the zoo's width ratios",
        ),
        run=_width_recovery,
    ),
    Check(
        label=SIGNED_LABEL,
        title="Signed maps on identical sets",
        session="12a",
        supports=(
            role("func", "mixed_signals.sigma.maps.sigma_eff", short=False),
            "the signed maps for values below the null mean",
        ),
        run=_signed_maps,
    ),
    Check(
        label=THRESHOLD_LABEL,
        title="Basis of the default thresholds",
        session="12a",
        supports=(
            role(
                "data",
                "mixed_signals.sigma.thresholds.SIGMA_EFF_THRESHOLD",
                short=False,
            ),
            role(
                "data",
                "mixed_signals.sigma.thresholds.LOG_R_EFF_THRESHOLD",
                short=False,
            ),
        ),
        run=_threshold_basis,
    ),
    Check(
        label=KNN_LABEL,
        title="Calibration of the kNN KL map",
        session="12a",
        supports=(
            role("func", "mixed_signals.sigma.calibration.knn_kl_curve", short=False),
            role("data", "mixed_signals.sigma.calibration.KNN_KL_SHIFTS", short=False),
        ),
        run=_knn_calibration,
    ),
)
