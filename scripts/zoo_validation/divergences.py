"""Check of the divergences: the ranks of the nearest-neighbour estimates."""

import functools
import math
from typing import NamedTuple

import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray
from matplotlib.figure import Figure as MatplotlibFigure

from mixed_signals import zoo
from mixed_signals.metrics.divergence import (
    HELLINGER_NEIGHBOURS,
    JS_NEIGHBOURS,
    KL_NEIGHBOURS,
    knn_hellinger_distance,
    knn_js_divergence,
    knn_kl_divergence,
)
from mixed_signals.nulls import permutation_null
from mixed_signals.nulls.plan import Statistic
from mixed_signals.sigma.maps import null_subtract
from mixed_signals.zoo import closed_forms
from zoo_validation import figures
from zoo_validation.common import (
    Builder,
    ResolutionRuns,
    compare_with_constants,
    pooled_resolution,
    real_space,
    resolution_ratio,
    sample_shift,
)
from zoo_validation.report import (
    Case,
    Check,
    Claim,
    Context,
    Figure,
    Measurement,
    Section,
    Table,
    claim_cases,
    claim_cases_above,
    grouped,
    role,
)

KNN_LABEL = "knn-ranks"
DIMENSIONS = (1, 4, 16)
KNN_SAMPLE_COUNT = 1_000
RANKS = (1, 3, 5, 10)
REFERENCE_RANK = 3
SHIFTS = (1.0, 2.0)
"""True shifts of the bias rows; the resolution rows read the first one."""
METRICS = ("kl", "js", "hellinger")
DISPLAY = {"kl": "KL", "js": "JS", "hellinger": "Hellinger"}
BIAS_DISPLAY = {"kl": "KL", "js": "JS", "hellinger": "Hellinger (H²)"}
DEFAULT_RANKS = {
    "kl": KL_NEIGHBOURS,
    "js": JS_NEIGHBOURS,
    "hellinger": HELLINGER_NEIGHBOURS,
}
KNN_FUNCTIONS = {
    "kl": knn_kl_divergence,
    "js": knn_js_divergence,
    "hellinger": knn_hellinger_distance,
}
TOLERANCE = 0.10
CLAIMED_BIAS_DIMENSIONS = (1, 4)
"""Dimensions of the bias claims, up to the one that the record states."""
OUTSIDE_LEGEND_HEIGHT = 0.5
"""Inches that the figure adds for its legend below the panels."""


@functools.cache
def _knn_builder(metric: str, neighbours: int, dimension: int) -> Builder:
    """Return the builder of a kNN statistic; one object per metric, rank and D."""
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
                neighbours=neighbours,
            )

        return statistic

    return build


class _Run(NamedTuple):
    """The value and the null draws of one metric at one rank, for one pair of sets."""

    value: Array
    draws: Array


def _signal_and_variance(metric: str, run: _Run) -> tuple[float, float]:
    """Return the value minus the null mean, and the variance of the null draws.

    The Hellinger rows are read in H² = sign(H)·H², the space in which the package
    subtracts their null; the other rows are read as they are.
    """
    value, draws = run.value, run.draws
    if metric == "hellinger":
        value, draws = value * jnp.abs(value), draws * jnp.abs(draws)
    finite = jnp.isfinite(draws)
    count = jnp.sum(finite)
    mean = jnp.sum(jnp.where(finite, draws, 0.0)) / count
    variance = jnp.sum(jnp.where(finite, (draws - mean) ** 2, 0.0)) / (count - 1)
    return float(value - mean), float(variance)


def _true_value(metric: str, shift: float) -> float:
    """Return the truth of a metric between unit Gaussians a shift apart."""
    if metric == "kl":
        return shift**2 / 2
    if metric == "js":
        return float(closed_forms.normal_js(0.0, 1.0, shift, 1.0))
    return 1.0 - math.exp(-(shift**2) / 8)


def _subtracted_value(metric: str, run: _Run) -> float:
    """Return the null-subtracted estimate, as H squared for the Hellinger rows."""
    subtracted = null_subtract(metric, run.value, run.draws)
    if metric == "hellinger":
        # * The estimate is H, the signed root of 1 - BC; its square is H².
        return float(subtracted * jnp.abs(subtracted))
    return float(subtracted)


class _Dimension(NamedTuple):
    """The runs of one dimension, by replicate."""

    signals: dict[tuple[str, int], list[float]]
    """Value minus null mean at the 1σ shift, by metric and rank."""
    variances: dict[tuple[str, int], list[float]]
    """Null variance at the 1σ shift, by metric and rank."""
    deviations: dict[tuple[str, float], list[float]]
    """Estimate over the truth at the sample's own shift, minus one, at the default."""
    non_finite: int
    hellinger_null: list[Array]
    """Null draws of H at its default rank and the 1σ shift, one array per replicate."""


def _run_dimension(
    context: Context, dimension: int, replicates: int, draws: int
) -> _Dimension:
    """Run every replicate of one dimension at every rank and shift."""
    signals: dict[tuple[str, int], list[float]] = {
        (metric, rank): [] for metric in METRICS for rank in RANKS
    }
    variances: dict[tuple[str, int], list[float]] = {
        (metric, rank): [] for metric in METRICS for rank in RANKS
    }
    deviations: dict[tuple[str, float], list[float]] = {
        (metric, shift): [] for metric in METRICS for shift in SHIFTS
    }
    non_finite = 0
    hellinger_null: list[Array] = []
    for shift_index, shift in enumerate(SHIFTS):
        case = zoo.shifted_gaussians(shift, param_count=dimension)
        sets = case.sample(
            context.key(KNN_LABEL, "sets", dimension, shift_index),
            KNN_SAMPLE_COUNT,
            batch_size=replicates,
        ).samples
        for replicate in range(replicates):
            first, second = sets[replicate, 0], sets[replicate, 1]
            path = (dimension, shift_index, replicate)
            plan = permutation_null(
                first,
                None,
                second,
                None,
                key=context.key(KNN_LABEL, "null", *path),
                draws=draws,
            )
            subsample_key = context.key(KNN_LABEL, "subsample", *path)
            held = sample_shift(first, second)
            for metric in METRICS:
                ranks = RANKS if shift_index == 0 else (DEFAULT_RANKS[metric],)
                for rank in ranks:
                    comparison = compare_with_constants(
                        _knn_builder(metric, rank, dimension),
                        (subsample_key,),
                        plan,
                        first,
                        None,
                        second,
                        None,
                    )
                    run = _Run(comparison.value, comparison.draws)
                    non_finite += int(jnp.sum(~jnp.isfinite(run.draws))) + int(
                        not bool(jnp.isfinite(run.value))
                    )
                    if shift_index == 0:
                        signal, variance = _signal_and_variance(metric, run)
                        signals[(metric, rank)].append(signal)
                        variances[(metric, rank)].append(variance)
                        if metric == "hellinger" and rank == DEFAULT_RANKS[metric]:
                            hellinger_null.append(run.draws)
                    if rank == DEFAULT_RANKS[metric]:
                        deviations[(metric, shift)].append(
                            _subtracted_value(metric, run) / _true_value(metric, held)
                            - 1.0
                        )
    return _Dimension(signals, variances, deviations, non_finite, hellinger_null)


def _knn_ranks(context: Context) -> Section:
    replicates = context.count(16, 2)
    draws = context.count(14, 4)
    dimensions = DIMENSIONS if not context.quick else (1, 4)
    runs = {
        dimension: _run_dimension(context, dimension, replicates, draws)
        for dimension in dimensions
    }
    ratios: dict[tuple[str, int], dict[int, Measurement]] = {}
    absolute: dict[tuple[str, int], Measurement] = {}
    for dimension, run in runs.items():
        for metric in METRICS:
            reference = ResolutionRuns(
                run.signals[(metric, REFERENCE_RANK)],
                run.variances[(metric, REFERENCE_RANK)],
            )
            absolute[(metric, dimension)] = pooled_resolution(reference)
            ratios[(metric, dimension)] = {
                rank: (
                    Measurement(1.0)
                    if rank == REFERENCE_RANK
                    else resolution_ratio(
                        ResolutionRuns(
                            run.signals[(metric, rank)], run.variances[(metric, rank)]
                        ),
                        reference,
                    )
                )
                for rank in RANKS
            }
    resolution_table = Table.of(
        "Resolution of a 1σ shift, (value minus null mean) over the null standard"
        f" deviation, relative to k = {REFERENCE_RANK}: the ratio of the mean signal"
        f" over the pooled null width of {replicates} replicates, which share their"
        " samples and permutations across ranks, with the leave-one-out error over"
        f" the replicates. The third column is the resolution at k = {REFERENCE_RANK},"
        " in null widths. The Hellinger rows read the signal and the null in"
        " H² = sign(H)·H², where the package subtracts the null of that metric.",
        (
            "Metric",
            "D",
            f"Resolution at k = {REFERENCE_RANK}",
            *[f"k = {rank}" for rank in RANKS],
        ),
        [
            [
                DISPLAY[metric],
                dimension,
                absolute[(metric, dimension)],
                *ratios[(metric, dimension)].values(),
            ]
            for metric in METRICS
            for dimension in dimensions
        ],
    )
    deviation_measurements = {
        (metric, dimension, shift): Measurement.from_replicates(
            jnp.asarray(runs[dimension].deviations[(metric, shift)])
        )
        for metric in METRICS
        for dimension in dimensions
        for shift in SHIFTS
    }
    bias_table = Table.of(
        "Null-subtracted estimate at the default rank over the truth at the sample's"
        " own shift, minus one, with the standard error of the paired deviations: KL"
        " against δ²/2, JS against the closed form and H² against 1 − exp(−δ²/8).",
        ("Metric", "Rank", "D", *[f"Shift {shift:g}" for shift in SHIFTS]),
        [
            [
                BIAS_DISPLAY[metric],
                DEFAULT_RANKS[metric],
                dimension,
                *[
                    deviation_measurements[(metric, dimension, shift)]
                    for shift in SHIFTS
                ],
            ]
            for metric in METRICS
            for dimension in dimensions
        ],
    )
    claims: list[Claim] = []
    for metric in ("js", "hellinger"):
        label = "H²" if metric == "hellinger" else DISPLAY[metric]
        cases = [
            Case(
                f"D = {dimension}, shift {shift:g}",
                deviation_measurements[(metric, dimension, shift)].value,
                deviation_measurements[(metric, dimension, shift)].margin(),
                TOLERANCE,
            )
            for dimension in dimensions
            if dimension in CLAIMED_BIAS_DIMENSIONS
            for shift in SHIFTS
        ]
        claims.append(
            claim_cases(
                f"The {label} at its default rank k = {DEFAULT_RANKS[metric]} recovers"
                " the truth at the sample's own shift up to D = "
                f"{max(CLAIMED_BIAS_DIMENSIONS)}.",
                cases,
                tolerance_text=f"{TOLERANCE:.0%} of the truth",
                replicates=replicates,
            )
        )
    for metric in ("js", "hellinger"):
        rank = DEFAULT_RANKS[metric]
        cases = [
            Case(
                f"D = {dimension}",
                ratios[(metric, dimension)][rank].value - 1.0,
                ratios[(metric, dimension)][rank].margin(),
                0.0,
            )
            for dimension in dimensions
        ]
        claims.append(
            claim_cases_above(
                f"The {DISPLAY[metric]} at its default rank k = {rank} resolves a 1σ"
                f" shift better than k = {REFERENCE_RANK}.",
                cases,
                tolerance_text=None,
                reference_text=f"the resolution at k = {REFERENCE_RANK}",
                replicates=replicates,
            )
        )

    def draw(style: figures.FigureStyle) -> MatplotlibFigure:
        figure, panels = figures.new_figure(
            style,
            columns=len(METRICS),
            sharey=False,
            extra_height=OUTSIDE_LEGEND_HEIGHT,
        )
        for index, (axes, metric) in enumerate(zip(panels, METRICS, strict=True)):
            for series, dimension in enumerate(dimensions):
                by_rank = ratios[(metric, dimension)]
                figures.plot_series(
                    axes,
                    style,
                    series,
                    RANKS,
                    [entry.value for entry in by_rank.values()],
                    label=f"D = {dimension}",
                    errors=[entry.error or 0.0 for entry in by_rank.values()],
                )
            figures.plot_reference(axes, style, y_value=1.0)
            figures.finish_panel(
                axes,
                style,
                title=DISPLAY[metric],
                x_label="rank k",
                y_label=f"resolution over that at k = {REFERENCE_RANK}"
                if index == 0
                else None,
                log_x=True,
            )
            axes.set_xticks(RANKS, [str(rank) for rank in RANKS])
        figures.add_figure_legend(figure, panels[0], style, columns=len(dimensions))
        return figure

    figure = Figure(
        label=f"{KNN_LABEL}-resolution",
        caption=f"Resolution of a 1σ shift over its value at k = {REFERENCE_RANK}"
        " against the rank k, one panel per metric and the dimension D as the"
        " series; the line is one and the bars are the errors of the ratio.",
        alt="Three panels of the resolution relative to k = 3 against the rank,"
        " rising with the rank and falling below one at k = 1.",
        draw=draw,
        data={
            "ranks": RANKS,
            "ratios": {
                f"{metric} D={dimension}": [
                    [entry.value, entry.error]
                    for entry in ratios[(metric, dimension)].values()
                ]
                for metric in METRICS
                for dimension in dimensions
            },
        },
    )
    missing = sum(run.non_finite for run in runs.values())
    largest = max(runs)
    null_draws = jnp.concatenate(runs[largest].hellinger_null)
    null_draws = null_draws[jnp.isfinite(null_draws)]
    null_mean = float(jnp.mean(null_draws))
    negative = int(jnp.sum(null_draws < 0))
    hellinger_text = (
        f"At D = {largest}, the null draws of the Hellinger estimate H at"
        f" k = {DEFAULT_RANKS['hellinger']} have the mean {null_mean:.3f}, and"
        f" {negative} of {int(null_draws.shape[0])} of them lie below zero."
    )
    default_text = ", ".join(
        f"{DISPLAY[metric]} {DEFAULT_RANKS[metric]}" for metric in METRICS
    )
    missing_text = (
        "No estimate or null draw was NaN or infinite."
        if missing == 0
        else f"{grouped(missing)} estimates or null draws were NaN or infinite and"
        " are left out of the means."
    )
    setup = (
        "Unit Gaussians in"
        f" {', '.join(str(dimension) for dimension in dimensions)} parameters, the"
        " second set shifted along one axis"
        f" ({role('func', 'mixed_signals.zoo.cases.shifted_gaussians')}), with"
        f" {grouped(KNN_SAMPLE_COUNT)} samples per set and {replicates} replicates"
        f" per dimension and shift. The estimates are"
        f" {role('func', 'mixed_signals.metrics.divergence.knn_kl_divergence')},"
        f" {role('func', 'mixed_signals.metrics.divergence.knn_js_divergence')} and"
        f" {role('func', 'mixed_signals.metrics.divergence.knn_hellinger_distance')}"
        f" at the ranks {', '.join(str(rank) for rank in RANKS)}, against the"
        f" permutation null with {draws} draws, fewer than the default 200 and the"
        " 40 of the original check; every rank of a replicate uses the same sets"
        " and the same permutations. The resolution at the 1σ shift is the mean"
        " of (value minus null mean) over the root of the mean null variance of"
        " the replicates, because every replicate draws the same null again; it is"
        f" read relative to k = {REFERENCE_RANK}. The Hellinger resolution is read in"
        " H² = sign(H)·H², the space in which the package subtracts the null of"
        " that metric. In H, where the null draws lie below zero, the slope of the"
        " signed root narrows the null spread against the signal, and the"
        f" resolution reads high. {hellinger_text} The bias rows subtract the null"
        " mean"
        " as the σ_eff maps do, in squares for the Hellinger distance, and compare"
        " with the truth at the shift that the two sample sets hold, the"
        " difference of their means along the shifted axis, which has the same"
        " expectation as the true shift and a far smaller scatter. The default"
        f" ranks are {default_text}."
        " The record, at 3 000 and 10 000 samples, gave a resolution against k = 3"
        " of 0.54 and 1.62 at k = 1 and 10 for KL, 0.58 and 1.87 for JS, and 0.17,"
        " 1.41 and 2.08 at k = 1, 5 and 10 for Hellinger; at the defaults JS lay"
        " within −1 to +3% of the truth up to D = 8, H² within 0 to −9%, and KL read"
        " −9 to −17% low at D = 4. The nearest-neighbour searches cost O(N²), and"
        " the JS estimate costs most, so this check runs at a smaller N and fewer"
        " replicates than the record. "
        f"{missing_text}"
    )
    return Section(setup, (resolution_table, figure, bias_table), tuple(claims))


CHECKS: tuple[Check, ...] = (
    Check(
        label=KNN_LABEL,
        title="Ranks of the nearest-neighbour divergences",
        session="9",
        supports=(
            role("data", "mixed_signals.metrics.divergence.KL_NEIGHBOURS", short=False),
            role("data", "mixed_signals.metrics.divergence.JS_NEIGHBOURS", short=False),
            role(
                "data",
                "mixed_signals.metrics.divergence.HELLINGER_NEIGHBOURS",
                short=False,
            ),
        ),
        run=_knn_ranks,
    ),
)
"""Checks of the divergences."""
