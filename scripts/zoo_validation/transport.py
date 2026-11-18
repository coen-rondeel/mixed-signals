"""Checks of the transport metrics, the chordal metrics and the sphere width."""

import dataclasses
import functools
import math
import time
from collections.abc import Callable, Sequence

import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray
from matplotlib.figure import Figure as MatplotlibFigure

from mixed_signals import errors, zoo
from mixed_signals.core.samples import circular_mean, spherical_width
from mixed_signals.core.space import ParamSpace
from mixed_signals.core.topology import PolarForm, unit_vectors, wrap
from mixed_signals.metrics.kernel import (
    energy_distance,
    maximum_mean_discrepancy,
    median_bandwidth,
)
from mixed_signals.metrics.ot.sinkhorn import (
    SINKHORN_EPSILON_FACTOR,
    sinkhorn_divergence,
)
from mixed_signals.metrics.ot.sliced import (
    SLICED_DIRECTIONS,
    SLICED_SINKHORN_DIRECTIONS,
    gaussian_sliced_wasserstein_distance,
    sliced_directions,
    sliced_sinkhorn_divergence,
    sliced_sinkhorn_epsilon,
    sliced_wasserstein_distance,
    sliced_wasserstein_of_gaussians,
)
from mixed_signals.nulls import NullPlan, permutation_null
from mixed_signals.nulls.plan import Statistic
from mixed_signals.prng import derive_key
from mixed_signals.sigma.maps import (
    CHORDAL_CIRCLE_WIDTH,
    CHORDAL_METRICS,
    CHORDAL_SPHERE_WIDTH,
    null_subtract,
    sigma_eff,
    warn_wide_chordal_axes,
)
from mixed_signals.transforms.whiten import Whitening
from mixed_signals.zoo import closed_forms
from mixed_signals.zoo.distributions import VonMisesFisher
from zoo_validation import figures
from zoo_validation.common import (
    Builder,
    Comparison,
    compare_on_plan,
    compare_with_constants,
    dimension_of,
    embedding_dimension_of,
    equal_log_weights,
    finite_measurement,
    real_space,
    sliced_w2_builder,
)
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
    critical_value,
    error_text,
    grouped,
    role,
)

SINKHORN_LABEL = "sinkhorn-epsilon"

NULL_DRAWS = role("func", "mixed_signals.nulls.plan.null_draws")
WRAPPED_GAUSSIANS = role("func", "mixed_signals.zoo.cases.wrapped_gaussians")
VON_MISES_FISHER = role("func", "mixed_signals.zoo.cases.von_mises_fisher")
DIRECTIONS_LABEL = "sliced-directions"
GAUSSIAN_SLICED_LABEL = "gaussian-sliced-w2"
CHORDAL_LABEL = "chordal-widths"
SPHERE_WIDTH_LABEL = "sphere-width"
CHORDS_LABEL = "chords-and-arcs"

SAMPLE_COUNT = 1_000
SLICED_SINKHORN_CHECK_CAP = 500
"""Samples per set of the sliced Sinkhorn divergence in these checks, for its cost."""
MEAN_FRACTION = 0.5
"""Quantile of the null that the figures draw as its centre."""
OUTSIDE_LEGEND_HEIGHT = 1.8
"""Inches that a two-row figure adds for the legends below its panels."""


def _resolution(value: Array, null: Array) -> Array:
    """Return (value minus the null mean) over the null standard deviation."""
    mean = jnp.nanmean(null, axis=-1)
    spread = jnp.nanstd(null, axis=-1, ddof=1)
    return (value - mean) / spread


@functools.cache
def _sinkhorn_builder(space: ParamSpace) -> Builder:
    """Return the builder of the Sinkhorn statistic of the vector of ε."""

    def build(epsilon: Array, key: PRNGKeyArray) -> Statistic:
        def statistic(
            first: Array, first_weights: Array, second: Array, second_weights: Array
        ) -> Array:
            return sinkhorn_divergence(
                first,
                first_weights,
                second,
                second_weights,
                space,
                epsilon=epsilon,
                key=key,
            )

        return statistic

    return build


@functools.cache
def _sliced_sinkhorn_builder(space: ParamSpace) -> Builder:
    """Return the builder of the sliced Sinkhorn statistic."""

    def build(directions: Array, epsilon: Array, key: PRNGKeyArray) -> Statistic:
        def statistic(
            first: Array, first_weights: Array, second: Array, second_weights: Array
        ) -> Array:
            return sliced_sinkhorn_divergence(
                first,
                first_weights,
                second,
                second_weights,
                space,
                directions=directions,
                epsilon=epsilon,
                key=key,
                cap=SLICED_SINKHORN_CHECK_CAP,
            )

        return statistic

    return build


def _seconds_of_call(statistic: Statistic, first: Array, second: Array) -> float:
    """Time one call of a statistic that has run before, so that it is compiled."""
    started = time.perf_counter()
    statistic(
        first,
        equal_log_weights(first.shape[0]),
        second,
        equal_log_weights(second.shape[0]),
    ).block_until_ready()
    return time.perf_counter() - started


# * Check sinkhorn-epsilon.

EPSILON_FACTORS = (0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0)
QUICK_EPSILON_FACTORS = (0.25, 1.0, 4.0)
SINKHORN_DIMENSIONS = (2, 8, 16)
SINKHORN_SHIFT = 0.5
SINKHORN_WIDTH_RATIO = 1.3
SHIFT_TOLERANCE = 0.03
MINIMUM_CLAIMED_FACTOR = 0.25
"""Smallest ε, in units of ℓ², that the claim on the recovered shift covers."""
LOW_ROOT_DIMENSION = 16
LOW_ROOT_FACTORS = (0.05, 0.1)
"""The ε, in units of ℓ², at which the root reads low in ``LOW_ROOT_DIMENSION``."""


@dataclasses.dataclass(frozen=True)
class CurveResult:
    """The curves S_ε(ε) of the replicates of one case, with their null draws."""

    values: Array
    """Shaped (replicates, epsilons)."""
    draws: Array
    """Shaped (replicates, epsilons, draws)."""
    seconds_per_call: list[float]
    sample_shifts: Array
    """Difference of the sample means along the first axis, shaped (replicates,)."""


def _sinkhorn_curves(
    context: Context,
    case: zoo.ZooCase,
    *,
    case_name: str,
    factors: Sequence[float],
    replicates: int,
    draw_count: int,
    sample_count: int,
) -> CurveResult:
    label = SINKHORN_LABEL
    dimension = len(case.space)
    sets = case.sample(
        context.key(label, "sets", case_name, dimension),
        sample_count,
        batch_size=replicates,
    ).samples
    values, draws, seconds = [], [], []
    for replicate in range(replicates):
        first, second = sets[replicate, 0], sets[replicate, 1]
        bandwidth = median_bandwidth(
            first,
            None,
            second,
            None,
            case.space,
            key=context.key(label, "bandwidth", case_name, dimension, replicate),
        )
        epsilon = jnp.asarray(factors) * bandwidth**2
        constants = (epsilon, context.key(label, "subsample"))
        builder = _sinkhorn_builder(case.space)
        plan = permutation_null(
            first,
            None,
            second,
            None,
            key=context.key(label, "null", case_name, dimension, replicate),
            draws=draw_count,
        )
        result = compare_with_constants(
            builder, constants, plan, first, None, second, None
        )
        seconds.append(_seconds_of_call(builder(*constants), first, second))
        values.append(result.value)
        draws.append(result.draws)
    sample_shifts = jnp.mean(sets[:, 1, :, 0], axis=1) - jnp.mean(
        sets[:, 0, :, 0], axis=1
    )
    return CurveResult(jnp.stack(values), jnp.stack(draws), seconds, sample_shifts)


def _sinkhorn_epsilon(context: Context) -> Section:
    label = SINKHORN_LABEL
    factors = QUICK_EPSILON_FACTORS if context.quick else EPSILON_FACTORS
    replicates = {"shift": context.count(16, 2), "width": context.count(6, 2)}
    draw_counts = {"shift": context.count(4, 3), "width": context.count(6, 3)}
    sample_count = context.count(SAMPLE_COUNT, 300)
    rows: list[list[Cell]] = []
    cost_rows: list[list[Cell]] = []
    claims: list[Claim] = []
    bands: dict[int, dict[str, list[float]]] = {}
    paired_by_dimension: dict[int, dict[float, Measurement | None]] = {}
    for dimension in SINKHORN_DIMENSIONS:
        results = {
            "shift": _sinkhorn_curves(
                context,
                zoo.shifted_gaussians(SINKHORN_SHIFT, param_count=dimension),
                case_name="shift",
                factors=factors,
                replicates=replicates["shift"],
                draw_count=draw_counts["shift"],
                sample_count=sample_count,
            ),
            "width": _sinkhorn_curves(
                context,
                zoo.scaled_gaussians(
                    SINKHORN_WIDTH_RATIO, param_count=dimension, axes=(0,)
                ),
                case_name="width",
                factors=factors,
                replicates=replicates["width"],
                draw_count=draw_counts["width"],
                sample_count=sample_count,
            ),
        }
        resolution = {
            name: _resolution(result.values, result.draws)
            for name, result in results.items()
        }
        recovered = sigma_eff(
            "sinkhorn",
            null_subtract("sinkhorn", results["shift"].values, results["shift"].draws),
        )
        paired = recovered - results["shift"].sample_shifts[:, None]
        sample_shift = Measurement.from_replicates(results["shift"].sample_shifts)
        width_means: list[Measurement | None] = []
        paired_means: list[Measurement | None] = []
        for position, factor in enumerate(factors):
            converged = (
                f"{int(jnp.sum(jnp.isfinite(results['shift'].values[:, position])))}"
                f" of {replicates['shift']}"
            )
            shift_resolution = finite_measurement(resolution["shift"][:, position])
            width_resolution = finite_measurement(resolution["width"][:, position])
            shift_recovered = finite_measurement(recovered[:, position])
            shift_paired = finite_measurement(paired[:, position])
            width_means.append(width_resolution)
            paired_means.append(shift_paired)
            rows.append(
                [
                    dimension,
                    factor,
                    converged,
                    shift_resolution or "not converged",
                    width_resolution or "not converged",
                    shift_recovered or "not converged",
                    sample_shift,
                    shift_paired or "not converged",
                ]
            )
        seconds = Measurement.from_replicates(
            results["shift"].seconds_per_call + results["width"].seconds_per_call
        )
        cost_rows.append([dimension, len(factors), seconds])
        claims.append(_epsilon_width_claim(dimension, factors, width_means))
        claims.append(_epsilon_shift_claim(dimension, factors, paired_means))
        paired_by_dimension[dimension] = dict(zip(factors, paired_means, strict=True))
        if dimension == LOW_ROOT_DIMENSION:
            claims.append(_epsilon_low_claim(dimension, paired_by_dimension[dimension]))
        shift_result = results["shift"]
        lower, median, upper = (
            jnp.nanmean(
                jnp.nanpercentile(shift_result.draws, percentile, axis=-1), axis=0
            )
            for percentile in (16, 50, 84)
        )
        bands[dimension] = {
            "value": [
                float(entry) for entry in jnp.nanmean(shift_result.values, axis=0)
            ],
            "lower": [float(entry) for entry in lower],
            "median": [float(entry) for entry in median],
            "upper": [float(entry) for entry in upper],
        }
    table = Table.of(
        "Resolution, in null widths, of a 0.5σ shift and of a width ratio of"
        " 1.3, the shift recovered as the root of the null-subtracted S_ε, the"
        " difference of the sample means along the shifted axis and the paired"
        " deviation of the two, by dimension and ε in units of ℓ², the squared"
        " median bandwidth.",
        (
            "D",
            "ε over ℓ²",
            "Converged",
            "Shift resolution",
            "Width resolution",
            "Recovered shift",
            "Sample shift",
            "Paired deviation",
        ),
        rows,
    )
    cost_table = Table.of(
        "Wall time of one call of the Sinkhorn divergence with the vector of ε.",
        ("D", "ε values", "Seconds per call"),
        cost_rows,
    )

    def draw(style: figures.FigureStyle) -> MatplotlibFigure:
        figure, panels = figures.new_figure(
            style, columns=len(SINKHORN_DIMENSIONS), sharey=False
        )
        for axes, dimension in zip(panels, SINKHORN_DIMENSIONS, strict=True):
            band = bands[dimension]
            figures.plot_band(axes, style, 1, factors, band["lower"], band["upper"])
            figures.plot_series(axes, style, 0, factors, band["value"], label="value")
            figures.plot_series(
                axes, style, 1, factors, band["median"], label="null median"
            )
            figures.plot_reference(axes, style, y_value=SINKHORN_SHIFT**2)
            figures.finish_panel(
                axes,
                style,
                title=f"D = {dimension}",
                x_label="ε over ℓ²",
                y_label=r"$S_\varepsilon$"
                if dimension == SINKHORN_DIMENSIONS[0]
                else None,
                log_x=True,
                log_y=True,
            )
            ticks = [factor for factor in (0.05, 0.25, 1.0, 4.0) if factor in factors]
            axes.set_xticks(ticks, [f"{factor:g}" for factor in ticks])
        figures.add_figure_legend(figure, panels[0], style)
        return figure

    figure = Figure(
        label=f"{label}-curve",
        caption="The Sinkhorn divergence of a 0.5σ shift against ε, with the"
        " 16th to 84th percentile band of the permutation null; the line is the"
        " true value, the squared shift. Values are means over replicates.",
        alt="Three panels of S_ε against ε on logarithmic axes, the value above the"
        " null band and approaching the squared shift at large ε.",
        draw=draw,
        data={"factors": factors, "bands": {str(key): bands[key] for key in bands}},
    )
    setup = (
        "Whitened zoo Gaussians (unit covariance) in"
        f" {', '.join(str(dimension) for dimension in SINKHORN_DIMENSIONS)}"
        f" dimensions with {grouped(sample_count)} samples per set, a shift of"
        f" {SINKHORN_SHIFT}σ and a width ratio of {SINKHORN_WIDTH_RATIO} on"
        " one axis. ε runs over"
        f" {', '.join(f'{factor:g}' for factor in factors)} times ℓ², with ℓ the"
        " median bandwidth of the comparison, fixed once. One call of"
        f" {role('func', 'mixed_signals.metrics.ot.sinkhorn.sinkhorn_divergence')}"
        " with the vector of ε gives the curve, and"
        f" {NULL_DRAWS} gives the curve of each of"
        f" {draw_counts['shift']} draws of the default permutation null for the"
        f" shift, with {replicates['shift']} replicates, which the claim on the"
        f" shift needs, and of {draw_counts['width']} draws with"
        f" {replicates['width']} replicates for the width ratio. The default ε of"
        " the package is 0.25 ℓ². Iterations that do"
        " not converge give NaN; the table counts the replicates that converged"
        " and averages the others. A claim on the width compares the resolution at"
        " 0.25 ℓ² with the best one, allowing the standard errors of both. A"
        " claim on the shift tests the mean paired deviation of the recovered"
        " shift from the sample shift of the same replicate, which is within"
        f" {SHIFT_TOLERANCE:.0%} of the true shift plus its margin of"
        f" {error_text(replicates['shift'])}, at every ε of"
        f" {MINIMUM_CLAIMED_FACTOR:g} ℓ² and above in each dimension."
        f"{_small_epsilon_text(paired_by_dimension, replicates['shift'])}"
    )
    return Section(setup, (table, cost_table, figure), tuple(claims))


def _epsilon_width_claim(
    dimension: int, factors: Sequence[float], means: Sequence[Measurement | None]
) -> Claim:
    present = [
        (index, measurement)
        for index, measurement in enumerate(means)
        if measurement is not None
    ]
    default_index = list(factors).index(SINKHORN_EPSILON_FACTOR)
    best_index, best = max(present, key=lambda entry: entry[1].value)
    default = means[default_index]
    consistent = default is not None and (
        best.value - default.value <= (best.error or 0.0) + (default.error or 0.0)
    )
    return Claim(
        f"D = {dimension}: the default ε is near the best width resolution.",
        "resolution at 0.25 over ℓ² within the standard errors of the best",
        f"best {best.render()} at {factors[best_index]:g}; default"
        f" {'none' if default is None else default.render()}",
        consistent,
        True,
    )


def _epsilon_shift_claim(
    dimension: int, factors: Sequence[float], paired: Sequence[Measurement | None]
) -> Claim:
    cases = [
        Case(
            f"ε = {factor:g} ℓ²",
            measurement.value,
            measurement.margin(),
            SHIFT_TOLERANCE * SINKHORN_SHIFT,
        )
        for factor, measurement in zip(factors, paired, strict=True)
        if measurement is not None and factor >= MINIMUM_CLAIMED_FACTOR
    ]
    return claim_cases(
        f"D = {dimension}: the root of S_ε recovers the sample shift at every ε of"
        f" {MINIMUM_CLAIMED_FACTOR:g} ℓ² and above.",
        cases,
        tolerance_text=f"{SHIFT_TOLERANCE:.0%} of the truth",
        replicates=next(
            measurement.replicates for measurement in paired if measurement is not None
        ),
    )


def _epsilon_low_claim(
    dimension: int, paired: dict[float, Measurement | None]
) -> Claim:
    """State that the root reads low at the small ε, by more than its margin."""
    entries = [(factor, paired.get(factor)) for factor in LOW_ROOT_FACTORS]
    observed = "; ".join(
        f"ε = {factor:g} ℓ²: "
        + (
            "not run"
            if measurement is None
            else f"{measurement.render()} (margin {measurement.margin():.4f})"
        )
        for factor, measurement in entries
    )
    holds = all(
        measurement is not None and measurement.value < -measurement.margin()
        for _, measurement in entries
    )
    factors_text = " and ".join(f"{factor:g} ℓ²" for factor in LOW_ROOT_FACTORS)
    return claim_true(
        f"D = {dimension}: at ε of {factors_text} the root of S_ε reads low.",
        "paired deviation below zero by more than its margin at both ε",
        observed,
        holds,
        True,
    )


def _small_epsilon_text(
    paired_by_dimension: dict[int, dict[float, Measurement | None]], replicates: int
) -> str:
    """Say which small-ε cells carry no claim, with the D = 8 numbers of the run."""
    middle = SINKHORN_DIMENSIONS[1]
    cells = [
        paired_by_dimension.get(middle, {}).get(factor) for factor in LOW_ROOT_FACTORS
    ]
    present = [cell for cell in cells if cell is not None and cell.error]
    if len(present) != len(LOW_ROOT_FACTORS):
        return ""
    ratios = [abs(cell.value) / cell.error for cell in present if cell.error]
    return (
        f" At D = {LOW_ROOT_DIMENSION} a further claim says that the root reads low"
        f" at {' and '.join(f'{factor:g}' for factor in LOW_ROOT_FACTORS)} ℓ², by"
        " more than its margin. No claim covers the smaller ε at D ="
        f" {SINKHORN_DIMENSIONS[0]} or D = {middle}; at D = {middle} the paired"
        f" deviation lies {min(ratios):.1f} to {max(ratios):.1f} standard errors"
        f" below zero, too close to the margin of {critical_value(replicates):.1f}"
        " standard errors for a claim, as the table shows."
    )


# * Check sliced-directions.

DIRECTION_DIMENSIONS = (4, 8, 16, 32)
DIRECTION_COUNTS = (20, 50, 100, 200, 500, 1_000, 2_000)
QUICK_DIRECTION_DIMENSIONS = (4,)
QUICK_DIRECTION_COUNTS = (20, 200)
WIDER_RATIO = 1.5
SCATTER_TOLERANCE = 0.2
ZERO_SCATTER = 1e-9
RESOLUTION_DIMENSIONS = (2, 8, 16)
RESOLUTION_COUNTS = (20, 100, 500, 2_000)
QUICK_RESOLUTION_COUNTS = (20, 500)
QUICK_RESOLUTION_DIMENSIONS = (2, 16)
RESOLUTION_SHIFT = 0.5
RESOLUTION_WIDTH_RATIO = 1.3


def _gaussian_values_of_directions(
    key: PRNGKeyArray, space: ParamSpace, count: int, *, orthonormal: bool
) -> Array:
    """Return SW2² of a 1σ shift and of a wider axis for one draw of directions."""
    size = len(space)
    directions = sliced_directions(space, count, key=key, orthonormal=orthonormal)
    zeros, identity = jnp.zeros(size), jnp.eye(size)
    wider = jnp.diag(jnp.ones(size).at[0].set(WIDER_RATIO**2))
    return jnp.stack(
        [
            sliced_wasserstein_of_gaussians(
                zeros,
                identity,
                zeros.at[0].set(1.0),
                identity,
                directions=directions,
            ),
            sliced_wasserstein_of_gaussians(
                zeros, identity, zeros, wider, directions=directions
            ),
        ]
    )


def _scatter_over_draws(
    context: Context, *, size: int, count: int, orthonormal: bool, draws: int
) -> tuple[Array, Array]:
    """Return the mean and the standard deviation of SW2² over direction draws.

    Both cases, the shift and the wider axis, share each draw of the directions;
    the results are shaped (2,).
    """
    space = real_space(size)
    keys = jax.random.split(
        context.key(DIRECTIONS_LABEL, size, count, int(orthonormal)), draws
    )

    def values(key: PRNGKeyArray) -> Array:
        return _gaussian_values_of_directions(
            key, space, count, orthonormal=orthonormal
        )

    drawn = jax.lax.map(values, keys, batch_size=50)
    return jnp.mean(drawn, axis=0), jnp.std(drawn, axis=0, ddof=1)


def _scatter_name(case_name: str, size: int, count: int, orthonormal: bool) -> str:
    basis = "orthonormal" if orthonormal else "independent"
    return f"{case_name}/n={size}/L={count}/{basis}"


def _scatter_measurement(mean: Array, spread: Array, draws: int) -> Measurement:
    relative = float(spread / mean)
    return Measurement(relative, relative / math.sqrt(2 * (draws - 1)))


def _directions_scatter(
    context: Context,
) -> tuple[list[Table], Figure, list[Claim], int]:
    dimensions = QUICK_DIRECTION_DIMENSIONS if context.quick else DIRECTION_DIMENSIONS
    counts = QUICK_DIRECTION_COUNTS if context.quick else DIRECTION_COUNTS
    draws = context.count(200, 30)
    truths: dict[str, Callable[[int], float]] = {
        "shift": lambda size: 1.0 / size,
        "wider": lambda size: float(
            closed_forms.gaussian_sliced_w2(0.0, WIDER_RATIO, 1, size)
        ),
    }
    scatter: dict[tuple[str, int, int, bool], Measurement] = {}
    relative_mean: dict[tuple[str, int, int, bool], float] = {}
    for size in dimensions:
        for count in counts:
            for orthonormal in (False, True):
                mean, spread = _scatter_over_draws(
                    context,
                    size=size,
                    count=count,
                    orthonormal=orthonormal,
                    draws=draws,
                )
                for position, case_name in enumerate(truths):
                    key = (case_name, size, count, orthonormal)
                    scatter[key] = _scatter_measurement(
                        mean[position], spread[position], draws
                    )
                    relative_mean[key] = float(mean[position]) / truths[case_name](size)
    tables = []
    for case_name, title in (
        ("shift", "a 1σ shift along one axis"),
        ("wider", f"one axis {WIDER_RATIO} times wider"),
    ):
        rows: list[list[Cell]] = []
        for size in dimensions:
            # * An orthonormal shift scatters by round-off only, so it has no row.
            for orthonormal in (False,) if case_name == "shift" else (False, True):
                rows.append(
                    [
                        size,
                        "orthonormal" if orthonormal else "independent",
                        *[
                            Measurement(
                                100
                                * scatter[(case_name, size, count, orthonormal)].value,
                                100
                                * (
                                    scatter[(case_name, size, count, orthonormal)].error
                                    or 0.0
                                ),
                            )
                            for count in counts
                        ],
                    ]
                )
        tables.append(
            Table.of(
                f"Relative scatter, in per cent, of the sliced W2 value of two"
                f" Gaussians over {draws} draws of the directions, for {title};"
                " the columns are the numbers of directions asked for."
                + (
                    " Orthonormal bases leave no scatter for a shift beyond"
                    " round-off, which the first claim checks."
                    if case_name == "shift"
                    else ""
                ),
                ("n", "Directions", *[str(count) for count in counts]),
                rows,
            )
        )
    claims: list[Claim] = []
    orthonormal_cases = [
        (f"n = {size}, {count} directions", scatter[("shift", size, count, True)].value)
        for size in dimensions
        for count in counts
    ]
    worst_label, worst_orthonormal = max(orthonormal_cases, key=lambda entry: entry[1])
    claims.append(
        claim_true(
            "Complete orthonormal bases leave no scatter for a shift.",
            f"relative scatter at most {ZERO_SCATTER:g} for every n and count",
            f"worst of {len(orthonormal_cases)}: {worst_label}:"
            f" {worst_orthonormal:.2g}",
            worst_orthonormal <= ZERO_SCATTER,
        )
    )
    beta_cases = []
    for size in dimensions:
        for count in counts:
            predicted = math.sqrt(2 * (size - 1) / ((size + 2) * count))
            observed = scatter[("shift", size, count, False)].value
            beta_cases.append(
                (f"n = {size}, {count} directions", observed / predicted - 1)
            )
    beta_label, beta_deviation = max(beta_cases, key=lambda entry: abs(entry[1]))
    claims.append(
        claim_true(
            "Independent directions scatter as the Beta law predicts.",
            f"within {SCATTER_TOLERANCE:.0%} of the root of 2(n - 1)/((n + 2)L)",
            f"worst of {len(beta_cases)}: {beta_label}: {beta_deviation:+.1%} against"
            f" ±{SCATTER_TOLERANCE:.0%}",
            abs(beta_deviation) <= SCATTER_TOLERANCE,
            True,
        )
    )
    data = {
        "draws": draws,
        "scatter": {
            _scatter_name(case_name, size, count, orthonormal): [
                value.value,
                value.error,
            ]
            for (case_name, size, count, orthonormal), value in scatter.items()
        },
        "mean_over_truth": {
            _scatter_name(case_name, size, count, orthonormal): value
            for (case_name, size, count, orthonormal), value in relative_mean.items()
        },
    }

    def draw(style: figures.FigureStyle) -> MatplotlibFigure:
        figure, panels = figures.new_figure(style, rows=2, columns=2, sharey=True)
        for axes, size in zip(panels, dimensions, strict=False):
            predicted = [
                math.sqrt(2 * (size - 1) / ((size + 2) * count)) for count in counts
            ]
            figures.plot_series(
                axes,
                style,
                0,
                counts,
                [scatter[("shift", size, count, False)].value for count in counts],
                label="shift, independent",
            )
            figures.plot_series(
                axes,
                style,
                1,
                counts,
                [scatter[("wider", size, count, False)].value for count in counts],
                label="wider axis, independent",
            )
            figures.plot_series(
                axes,
                style,
                2,
                counts,
                [scatter[("wider", size, count, True)].value for count in counts],
                label="wider axis, orthonormal",
            )
            axes.plot(
                counts,
                predicted,
                color=style.reference,
                linewidth=figures.REFERENCE_WIDTH,
                zorder=0,
            )
            figures.finish_panel(
                axes,
                style,
                title=f"n = {size}",
                x_label="directions",
                y_label="relative scatter",
                log_x=True,
                log_y=True,
            )
            if size == dimensions[0]:
                figures.add_legend(axes, style, location="lower left")
        figures.hide_unused(panels, len(dimensions))
        return figure

    figure = Figure(
        label=f"{DIRECTIONS_LABEL}-scatter",
        caption="Relative scatter of the sliced W2 value over draws of the directions"
        " against the number of directions, one panel per number of coordinates n;"
        " the line is the root of 2(n - 1)/((n + 2)L). The scatter of a shift with"
        " orthonormal bases is zero and is not drawn.",
        alt="Four panels of relative scatter against the number of directions on"
        " logarithmic axes, falling as the inverse square root of the number of"
        " directions for independent directions and lower for orthonormal bases.",
        draw=draw,
        data=data,
    )
    return tables, figure, claims, draws


@dataclasses.dataclass(frozen=True)
class ResolutionRow:
    """The resolution of one comparison by one method, and its cost."""

    dimension: int
    case: str
    method: str
    directions: int
    resolution: Measurement | None
    relative: Measurement | None
    """Resolution over the resolution at the default directions, per replicate."""
    seconds: Measurement
    nan_values: int = 0
    """Replicates whose value is NaN."""
    replicates: int = 0
    nan_draws: int = 0
    """Null draws that are NaN, over all replicates."""
    draw_total: int = 0


def _resolution_rows(context: Context, sample_count: int) -> list[ResolutionRow]:
    label = DIRECTIONS_LABEL
    counts = QUICK_RESOLUTION_COUNTS if context.quick else RESOLUTION_COUNTS
    replicates = context.count(4, 2)
    draw_count = context.count(10, 3)
    slow_replicates = context.count(3, 1)
    slow_draws = context.count(5, 2)
    rows: list[ResolutionRow] = []
    dimensions = QUICK_RESOLUTION_DIMENSIONS if context.quick else RESOLUTION_DIMENSIONS
    for dimension in dimensions:
        for case_name, case in (
            ("shift", zoo.shifted_gaussians(RESOLUTION_SHIFT, param_count=dimension)),
            (
                "width",
                zoo.scaled_gaussians(
                    RESOLUTION_WIDTH_RATIO, param_count=dimension, axes=(0,)
                ),
            ),
        ):
            sets = case.sample(
                context.key(label, "sets", case_name, dimension),
                sample_count,
                batch_size=replicates,
            ).samples
            space = case.space
            builder = sliced_w2_builder(space)
            resolutions: dict[int, list[float]] = {count: [] for count in counts}
            seconds: dict[int, list[float]] = {count: [] for count in counts}
            nan_values = dict.fromkeys(counts, 0)
            nan_draws = dict.fromkeys(counts, 0)
            for replicate in range(replicates):
                first, second = sets[replicate, 0], sets[replicate, 1]
                # * One plan per replicate: every count sees the same sets and draws.
                plan = permutation_null(
                    first,
                    None,
                    second,
                    None,
                    key=context.key(label, "null", case_name, dimension, replicate),
                    draws=draw_count,
                )
                for count in counts:
                    directions = sliced_directions(
                        space,
                        count,
                        key=context.key(
                            label, "directions", case_name, dimension, count, replicate
                        ),
                    )
                    result = compare_with_constants(
                        builder, (directions,), plan, first, None, second, None
                    )
                    seconds[count].append(
                        _seconds_of_call(builder(directions), first, second)
                    )
                    resolutions[count].append(
                        float(_resolution(result.value, result.draws))
                    )
                    nan_values[count] += int(jnp.isnan(result.value))
                    nan_draws[count] += int(jnp.sum(jnp.isnan(result.draws)))
            reference = jnp.array(resolutions[SLICED_DIRECTIONS])
            for count in counts:
                resolution = jnp.array(resolutions[count])
                rows.append(
                    ResolutionRow(
                        dimension,
                        case_name,
                        "sliced W2",
                        count,
                        finite_measurement(resolution),
                        finite_measurement(resolution / reference),
                        Measurement.from_replicates(seconds[count]),
                        nan_values[count],
                        replicates,
                        nan_draws[count],
                        replicates * draw_count,
                    )
                )
            rows.append(
                _sliced_sinkhorn_row(
                    context,
                    case_name,
                    case,
                    sets,
                    replicates=slow_replicates,
                    draw_count=slow_draws,
                )
            )
    return rows


def _sliced_sinkhorn_row(
    context: Context,
    case_name: str,
    case: zoo.ZooCase,
    sets: Array,
    *,
    replicates: int,
    draw_count: int,
) -> ResolutionRow:
    label = DIRECTIONS_LABEL
    space = case.space
    dimension = len(space)
    resolutions, seconds = [], []
    nan_values = nan_draws = 0
    for replicate in range(replicates):
        first, second = sets[replicate, 0], sets[replicate, 1]
        key = context.key(label, "sliced-sinkhorn", case_name, dimension, replicate)
        directions = sliced_directions(
            space, SLICED_SINKHORN_DIRECTIONS, key=derive_key(key, "directions")
        )
        epsilon = sliced_sinkhorn_epsilon(
            first,
            None,
            second,
            None,
            space,
            directions=directions,
            key=derive_key(key, "epsilon"),
        )

        plan = permutation_null(
            first, None, second, None, key=derive_key(key, "null"), draws=draw_count
        )
        builder = _sliced_sinkhorn_builder(space)
        constants = (directions, epsilon, derive_key(key, "subsample"))
        result = compare_with_constants(
            builder, constants, plan, first, None, second, None
        )
        seconds.append(_seconds_of_call(builder(*constants), first, second))
        resolutions.append(float(_resolution(result.value, result.draws)))
        nan_values += int(jnp.isnan(result.value))
        nan_draws += int(jnp.sum(jnp.isnan(result.draws)))
    return ResolutionRow(
        dimension,
        case_name,
        "sliced Sinkhorn",
        int(directions.shape[0]),
        finite_measurement(jnp.array(resolutions)),
        None,
        Measurement.from_replicates(seconds),
        nan_values,
        replicates,
        nan_draws,
        replicates * draw_count,
    )


def _sliced_directions(context: Context) -> Section:
    gaussian_form = role(
        "func", "mixed_signals.metrics.ot.sliced.sliced_wasserstein_of_gaussians"
    )
    tables, figure, claims, draws = _directions_scatter(context)
    sample_count = context.count(SAMPLE_COUNT, 200)
    rows = _resolution_rows(context, sample_count)
    resolution_table = Table.of(
        "Resolution, in null widths, of a 0.5σ shift and of a width ratio of"
        f" 1.3 with {grouped(sample_count)} samples per set, by sliced W2 with"
        " growing numbers of directions and by the sliced Sinkhorn divergence at"
        " its default directions. For sliced W2 every number of directions uses the"
        " same samples and the same permutations, and the relative resolution is"
        f" the ratio of each replicate's resolution to its own at {SLICED_DIRECTIONS}"
        " directions, with the standard error of the ratio. The seconds are the"
        " wall time of one call. The last two columns count the replicates with a"
        " NaN value and the NaN draws of the null, over all replicates.",
        (
            "D",
            "Case",
            "Method",
            "Directions",
            "Resolution",
            "Relative to the default",
            "Seconds per call",
            "NaN values",
            "NaN null draws",
        ),
        [
            [
                row.dimension,
                row.case,
                row.method,
                row.directions,
                row.resolution or "not converged",
                row.relative,
                row.seconds,
                f"{row.nan_values} of {row.replicates}",
                f"{row.nan_draws} of {grouped(row.draw_total)}",
            ]
            for row in rows
        ],
    )
    setup = (
        "The Gaussian form"
        f" {gaussian_form}"
        " gives the sliced W2 value of two Gaussians at no sampling cost, so the"
        f" script draws the directions {draws} times for each number of"
        " coordinates n and each number of directions, with"
        f" {role('func', 'mixed_signals.metrics.ot.sliced.sliced_directions')}"
        " as independent unit vectors and as complete orthonormal bases. The cases"
        " are a 1σ shift along one axis and one axis"
        f" {WIDER_RATIO} times wider, in n real coordinates. The relative scatter"
        " is the standard deviation over the draws divided by the mean, with the"
        " standard error of a standard deviation. For independent directions the"
        " mean of L squared projections of a shift, each distributed as"
        " Beta(1/2, (n - 1)/2), has the relative scatter"
        " √(2(n - 1)/((n + 2)L)). The second part reruns the resolution of a"
        " permutation null with the empirical sliced W2, paired over the number of"
        " directions, and with the sliced Sinkhorn divergence at its default"
        " directions. The cost of the latter limits its replicates and draws, and"
        f" it runs at a cap of {SLICED_SINKHORN_CHECK_CAP} samples per set. A NaN"
        " draw is left out of the null mean and spread, and a replicate with a NaN"
        " value is left out of the resolution; the table counts both."
    )
    return Section(setup, (*tables, figure, resolution_table), tuple(claims))


# * Check gaussian-sliced-w2.

BIAS_DIMENSION = 8
BIAS_COUNTS = (1_000, 5_000, 20_000)
BIAS_RELATIVE_TOLERANCE = 0.05


@functools.cache
def _both_estimators_builder(space: ParamSpace) -> Builder:
    """Return the builder of the empirical and the Gaussian sliced W2 together."""

    def build(directions: Array) -> Statistic:
        def statistic(
            first: Array, first_weights: Array, second: Array, second_weights: Array
        ) -> Array:
            return jnp.stack(
                [
                    sliced_wasserstein_distance(
                        first,
                        first_weights,
                        second,
                        second_weights,
                        space,
                        directions=directions,
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
            )

        return statistic

    return build


def _pooled_whitening(
    pair: Array, space: ParamSpace
) -> tuple[Array, Array, ParamSpace]:
    whitening = Whitening.from_samples(pair, None, space)
    return (
        whitening.forward(pair[0]),
        whitening.forward(pair[1]),
        whitening.map_space(space),
    )


def _estimators_effect(
    context: Context,
    label: str,
    pair: Array,
    space: ParamSpace,
    *,
    tag: Sequence[str | int],
    draw_count: int,
) -> tuple[float, float]:
    first, second, whitened = _pooled_whitening(pair, space)
    directions = sliced_directions(
        whitened, SLICED_DIRECTIONS, key=context.key(label, "directions", *tag)
    )

    plan = permutation_null(
        first,
        None,
        second,
        None,
        key=context.key(label, "null", *tag),
        draws=draw_count,
    )
    result = compare_with_constants(
        _both_estimators_builder(whitened),
        (directions,),
        plan,
        first,
        None,
        second,
        None,
    )
    subtracted = null_subtract("sliced_w2", result.value, result.draws)
    effect = sigma_eff(
        "sliced_w2", subtracted, dimension=embedding_dimension_of(whitened)
    )
    return float(effect[0]), float(effect[1])


def _gaussian_sliced_w2(context: Context) -> Section:
    label = GAUSSIAN_SLICED_LABEL
    replicates = context.count(10, 3)
    draw_count = context.count(30, 9)
    cases = (
        ("banana against its Gaussian", zoo.banana(), 0, 1),
        (
            "mixture at separation 4 against its Gaussian",
            zoo.gaussian_mixture(4.0),
            0,
            1,
        ),
        ("identical Gaussians", zoo.shifted_gaussians(0.0, param_count=2), 0, 1),
    )
    rows: list[list[Cell]] = []
    measurements: dict[str, tuple[Measurement, Measurement]] = {}
    for case_index, (name, case, first_index, second_index) in enumerate(cases):
        sets = case.sample(
            context.key(label, "sets", case_index), SAMPLE_COUNT, batch_size=replicates
        ).samples
        empirical, gaussian = [], []
        for replicate in range(replicates):
            pair = sets[replicate, jnp.array([first_index, second_index])]
            effects = _estimators_effect(
                context,
                label,
                pair,
                case.space,
                tag=(case_index, replicate),
                draw_count=draw_count,
            )
            empirical.append(effects[0])
            gaussian.append(effects[1])
        measurements[name] = (
            Measurement.from_replicates(empirical),
            Measurement.from_replicates(gaussian),
        )
        rows.append([name, replicates, *measurements[name]])
    table = Table.of(
        "Mean σ_eff of the empirical and the Gaussian estimator of the sliced W2"
        " row, null-subtracted, with the standard error of the mean.",
        ("Case", "Replicates", "Empirical σ_eff", "Gaussian σ_eff"),
        rows,
    )
    bias_rows: list[list[Cell]] = []
    bias_cases: list[Case] = []
    space = real_space(BIAS_DIMENSION)
    bias_replicates = context.count(100, 5)
    bias_counts = BIAS_COUNTS if not context.quick else BIAS_COUNTS[:2]
    for count in bias_counts:
        sample_key = context.key(label, "bias", count)
        values = []
        for replicate in range(bias_replicates):
            replicate_key = derive_key(sample_key, replicate)
            first = jax.random.normal(
                derive_key(replicate_key, "first"), (count, BIAS_DIMENSION)
            )
            second = jax.random.normal(
                derive_key(replicate_key, "second"), (count, BIAS_DIMENSION)
            )
            directions = sliced_directions(
                space, SLICED_DIRECTIONS, key=derive_key(replicate_key, "directions")
            )
            values.append(
                float(
                    gaussian_sliced_wasserstein_distance(
                        first, None, second, None, space, directions=directions
                    )
                )
            )
        bias = Measurement.from_replicates(values)
        expected = 3.0 / count
        bias_rows.append([count, bias, expected, bias.value / expected])
        bias_cases.append(
            Case(
                f"N = {grouped(count)}",
                bias.value - expected,
                bias.margin(),
                BIAS_RELATIVE_TOLERANCE * expected,
            )
        )
    bias_table = Table.of(
        "Gaussian estimator before subtraction on identical"
        f" {BIAS_DIMENSION}-dimensional Gaussians: the mean value, the value 3/N"
        " and their ratio.",
        ("N per set", "Mean value", "3/N", "Ratio"),
        bias_rows,
    )
    claims: list[Claim] = []
    zero = "identical Gaussians"
    for index, estimator in enumerate(("empirical", "Gaussian")):
        measurement = measurements[zero][index]
        claims.append(
            claim_cases(
                f"Identical Gaussians: the {estimator} estimator is at zero.",
                [Case("mean σ_eff", measurement.value, measurement.margin(), 0.0)],
                tolerance_text=None,
                replicates=measurement.replicates,
            )
        )
    banana = measurements["banana against its Gaussian"]
    claims.append(
        claim_cases(
            "Banana: the Gaussian estimator is consistent with zero.",
            [Case("mean σ_eff", banana[1].value, banana[1].margin(), 0.0)],
            tolerance_text=None,
            replicates=banana[1].replicates,
        )
    )
    claims.append(
        claim_true(
            "Banana: the empirical estimator is not consistent with zero.",
            f"farther than {error_text(banana[0].replicates)} from 0",
            banana[0].render(),
            abs(banana[0].value) > banana[0].margin(),
            True,
        )
    )
    claims.append(
        claim_cases(
            "The Gaussian form before subtraction has a bias of 3/N.",
            bias_cases,
            tolerance_text=f"{BIAS_RELATIVE_TOLERANCE:.0%} of 3/N",
            replicates=bias_replicates,
        )
    )
    setup = (
        f"{replicates} replicates of {grouped(SAMPLE_COUNT)} samples per set and"
        f" {draw_count} permutation draws. Each pair is whitened with its pooled"
        " covariance, the two estimators share"
        f" {SLICED_DIRECTIONS} directions, and both are null-subtracted and mapped"
        " by the sliced W2 map with n coordinates. The zoo banana"
        f" ({role('func', 'mixed_signals.zoo.cases.banana')}) and the mixture"
        f" ({role('func', 'mixed_signals.zoo.cases.gaussian_mixture')}) share their"
        " mean and covariance with their Gaussians, so the Gaussian estimator, a"
        " lower bound of the sliced W2 for these moments"
        " ({cite:t}`gelbrich1990formula`),"
        " reads zero, while the empirical estimator sees the gap. The second table"
        " fits the Gaussian form to identical Gaussians without subtraction."
        f" {bias_replicates} replicates give its bias, which the fitted moments raise"
        " by about 3/N."
    )
    return Section(setup, (table, bias_table), tuple(claims))


# * Check chordal-widths.

CIRCLE_WIDTHS = (0.1, 0.3, 0.5, 1.0)
SPHERE_WIDTHS = (0.05, 0.2, 0.5)
QUICK_CIRCLE_WIDTHS = (0.1, 1.0)
QUICK_SPHERE_WIDTHS = (0.05, 0.5)
CHORDAL_SHIFTS = (0.5, 1.0, 2.0)
PILOT_COUNT = 10_000
SLICED_SINKHORN_SHIFT = 1.0
CHORDAL_TOLERANCE = 0.05
GEOMETRIES = ("circle", "sphere")
CHEAP_METRICS = ("energy", "mmd", "sliced_w2", "sinkhorn")
CHORDAL_NAMES = {
    "energy": "Energy distance",
    "mmd": "MMD squared",
    "sinkhorn": "Sinkhorn divergence",
    "sliced_w2": "Sliced W2",
    "sliced_sinkhorn": "Sliced Sinkhorn",
}


def _chordal_case(geometry: str, shift: float, width: float) -> zoo.ZooCase:
    if geometry == "circle":
        return zoo.wrapped_gaussians(shift, width=width, param_count=1)
    return zoo.von_mises_fisher(shift, width=width)


@functools.cache
def _cheap_builder(space: ParamSpace) -> Builder:
    """Return the builder of the four chordal metrics that run on every comparison."""

    def build(bandwidth: Array, directions: Array, key: PRNGKeyArray) -> Statistic:
        def statistic(
            first: Array, first_weights: Array, second: Array, second_weights: Array
        ) -> Array:
            return jnp.stack(
                [
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
                    sliced_wasserstein_distance(
                        first,
                        first_weights,
                        second,
                        second_weights,
                        space,
                        directions=directions,
                    ),
                    sinkhorn_divergence(
                        first,
                        first_weights,
                        second,
                        second_weights,
                        space,
                        epsilon=SINKHORN_EPSILON_FACTOR * bandwidth**2,
                        key=key,
                    ),
                ]
            )

        return statistic

    return build


def _estimated_width(whitening: Whitening, space: ParamSpace) -> float:
    """Return the width in radians that a whitening measured on its axis."""
    if whitening.blocks.circles:
        lower, upper = space.parameters[whitening.blocks.circles[0]].bounds
        return 2 * math.pi * float(whitening.circle_scales[0]) / (upper - lower)
    return float(whitening.sphere_scales[0])


def _chordal_effects(
    context: Context,
    pair: Array,
    space: ParamSpace,
    pilot: Whitening,
    *,
    tag: Sequence[str | int],
    draw_count: int,
    with_sliced_sinkhorn: bool,
    sliced_sinkhorn_draws: int,
    shift: float,
) -> tuple[dict[str, float], bool, float]:
    """Return σ_eff over the shift of each metric, the warning and the width.

    The metrics run in the coordinates of the pilot whitening of the width, so
    that every replicate shares one whitened space and one compiled program;
    the warning and the estimated width come from the whitening of the
    replicate itself.
    """
    label = CHORDAL_LABEL
    own = Whitening.from_samples(pair, None, space)
    with errors.capture_warnings() as records:
        warn_wide_chordal_axes("energy", own, space)
    first, second = pilot.forward(pair[0]), pilot.forward(pair[1])
    whitened = pilot.map_space(space)
    bandwidth = median_bandwidth(
        first, None, second, None, whitened, key=context.key(label, "bandwidth", *tag)
    )
    directions = sliced_directions(
        whitened, SLICED_DIRECTIONS, key=context.key(label, "directions", *tag)
    )
    subsample_key = context.key(label, "subsample")
    plan = permutation_null(
        first,
        None,
        second,
        None,
        key=context.key(label, "null", *tag),
        draws=draw_count,
    )
    result = compare_with_constants(
        _cheap_builder(whitened),
        (bandwidth, directions, subsample_key),
        plan,
        first,
        None,
        second,
        None,
    )
    dimension, embedding = dimension_of(whitened), embedding_dimension_of(whitened)
    bandwidths = {"mmd": bandwidth}
    dimensions = {"energy": dimension, "mmd": dimension, "sliced_w2": embedding}
    ratios: dict[str, float] = {}
    for position, metric in enumerate(CHEAP_METRICS):
        subtracted = null_subtract(
            metric, result.value[position], result.draws[position]
        )
        ratios[metric] = (
            float(
                sigma_eff(
                    metric,
                    subtracted,
                    bandwidth=bandwidths.get(metric),
                    dimension=dimensions.get(metric),
                )
            )
            / shift
        )
    if with_sliced_sinkhorn:
        sliced_directions_array = sliced_directions(
            whitened,
            SLICED_SINKHORN_DIRECTIONS,
            key=context.key(label, "sliced-directions", *tag),
        )
        epsilon = sliced_sinkhorn_epsilon(
            first,
            None,
            second,
            None,
            whitened,
            directions=sliced_directions_array,
            key=context.key(label, "sliced-epsilon", *tag),
        )
        sliced_plan = permutation_null(
            first,
            None,
            second,
            None,
            key=context.key(label, "sliced-null", *tag),
            draws=sliced_sinkhorn_draws,
        )
        sliced_result = compare_with_constants(
            _sliced_sinkhorn_builder(whitened),
            (sliced_directions_array, epsilon, subsample_key),
            sliced_plan,
            first,
            None,
            second,
            None,
        )
        subtracted = null_subtract(
            "sliced_sinkhorn", sliced_result.value, sliced_result.draws
        )
        ratios["sliced_sinkhorn"] = (
            float(sigma_eff("sliced_sinkhorn", subtracted, dimension=embedding)) / shift
        )
    return ratios, bool(records), _estimated_width(own, space)


def _chordal_sample_shift(case: zoo.ZooCase, pair: Array, pilot: Whitening) -> float:
    """Return the shift of the two sample sets in the widths of the pilot whitening.

    On a circle it is the difference of the two circular means, wrapped to the
    shorter arc, over the circle's scale. On a sphere it is the difference of
    the angles of the two mean directions in the plane of the true shift, which
    holds the mean direction of the reference and the direction towards the
    shifted mean, over the sphere's width.
    """
    space = case.space
    if pilot.blocks.circles:
        position = pilot.blocks.circles[0]
        lower, upper = space.parameters[position].bounds
        period = upper - lower
        means = [
            circular_mean(
                pair[index][:, position],
                equal_log_weights(pair.shape[1]),
                period,
                lower,
            )
            for index in (0, 1)
        ]
        gap = wrap(means[1] - means[0], -period / 2, period)
        return float(gap / pilot.circle_scales[0])
    sphere = space.spheres[0]
    reference, shifted = case.posteriors[:2]
    assert isinstance(reference, VonMisesFisher)
    assert isinstance(shifted, VonMisesFisher)
    first_axis = reference.direction
    second_axis = (
        shifted.direction - jnp.dot(shifted.direction, first_axis) * first_axis
    )
    second_axis = second_axis / jnp.linalg.norm(second_axis)
    azimuth, polar = space.axis_blocks.spheres[0]
    angles = []
    for index in (0, 1):
        units, _, _ = unit_vectors(
            pair[index][:, azimuth], pair[index][:, polar], sphere.polar_form
        )
        mean = jnp.mean(units, axis=0)
        angles.append(jnp.arctan2(mean @ second_axis, mean @ first_axis))
    return float((angles[1] - angles[0]) / pilot.sphere_scales[0])


def _chordal_widths(context: Context) -> Section:
    label = CHORDAL_LABEL
    shifts = CHORDAL_SHIFTS if not context.quick else (SLICED_SINKHORN_SHIFT,)
    sample_count = context.count(SAMPLE_COUNT, 200)
    pilot_count = context.count(PILOT_COUNT, 2_000)
    replicates = context.count(6, 1)
    draw_count = context.count(6, 3)
    sliced_replicates = context.count(2, 1)
    sliced_draws = context.count(3, 2)
    widths: dict[str, tuple[float, ...]] = (
        {"circle": QUICK_CIRCLE_WIDTHS, "sphere": QUICK_SPHERE_WIDTHS}
        if context.quick
        else {"circle": CIRCLE_WIDTHS, "sphere": SPHERE_WIDTHS}
    )
    limits = {"circle": CHORDAL_CIRCLE_WIDTH, "sphere": CHORDAL_SPHERE_WIDTH}
    ratios: dict[tuple[str, float, float, str], list[float]] = {}
    paired: dict[tuple[str, float, float, str], list[float]] = {}
    held: dict[tuple[str, float, float], list[float]] = {}
    fired: dict[tuple[str, float], list[bool]] = {}
    estimated: dict[tuple[str, float], list[float]] = {}
    for geometry in GEOMETRIES:
        for width_index, width in enumerate(widths[geometry]):
            pilot_case = _chordal_case(geometry, SLICED_SINKHORN_SHIFT, width)
            pilot = Whitening.from_samples(
                pilot_case.sample(
                    context.key(label, "pilot", geometry, width_index), pilot_count
                ).samples[0],
                None,
                pilot_case.space,
            )
            for shift_index, shift in enumerate(shifts):
                case = _chordal_case(geometry, shift, width)
                sets = case.sample(
                    context.key(label, "sets", geometry, width_index, shift_index),
                    sample_count,
                    batch_size=replicates,
                ).samples
                for replicate in range(replicates):
                    with_sliced = (
                        shift == SLICED_SINKHORN_SHIFT and replicate < sliced_replicates
                    )
                    effects, did_fire, estimate = _chordal_effects(
                        context,
                        sets[replicate],
                        case.space,
                        pilot,
                        tag=(geometry, width_index, shift_index, replicate),
                        draw_count=draw_count,
                        with_sliced_sinkhorn=with_sliced,
                        sliced_sinkhorn_draws=sliced_draws,
                        shift=shift,
                    )
                    sample_ratio = (
                        _chordal_sample_shift(case, sets[replicate], pilot) / shift
                    )
                    held.setdefault((geometry, width, shift), []).append(sample_ratio)
                    for metric, ratio in effects.items():
                        key = (geometry, width, shift, metric)
                        ratios.setdefault(key, []).append(ratio)
                        paired.setdefault(key, []).append(ratio - sample_ratio)
                    fired.setdefault((geometry, width), []).append(did_fire)
                    estimated.setdefault((geometry, width), []).append(estimate)
    measured = _finite_measurements(ratios)
    paired_means = _finite_measurements(paired)
    held_means = {
        key: Measurement.from_replicates(values) for key, values in held.items()
    }
    shift_columns = tuple(f"Shift {shift:g}" for shift in shifts)
    rows: list[list[Cell]] = []
    paired_rows: list[list[Cell]] = []
    for geometry in GEOMETRIES:
        for width in widths[geometry]:
            for metric in CHORDAL_METRICS:
                for target, source in ((rows, measured), (paired_rows, paired_means)):
                    target.append(
                        [
                            geometry,
                            width,
                            CHORDAL_NAMES[metric],
                            *[
                                source.get((geometry, width, shift, metric))
                                for shift in shifts
                            ],
                        ]
                    )
    held_rows: list[list[Cell]] = [
        [geometry, width, *[held_means[(geometry, width, shift)] for shift in shifts]]
        for geometry in GEOMETRIES
        for width in widths[geometry]
    ]
    table = Table.of(
        "σ_eff over the true shift, by geometry, width in radians, metric and shift"
        " in widths, with the standard error over replicates. The sliced Sinkhorn"
        " divergence ran at a shift of 1 only. A dash marks a value that was not"
        " computed or whose iterations did not converge.",
        ("Geometry", "Width (rad)", "Metric", *shift_columns),
        rows,
    )
    held_table = Table.of(
        "The shift that the sample sets hold, over the true shift: on a circle the"
        " difference of the circular means over the width of the whitening, on a"
        " sphere the difference of the angles of the mean directions in the plane of"
        " the true shift over the sphere width of the whitening.",
        ("Geometry", "Width (rad)", *shift_columns),
        held_rows,
    )
    paired_table = Table.of(
        "Paired deviation over the true shift: σ_eff minus the sample shift of the"
        " same replicate, over the true shift, with the standard error of the"
        " paired differences.",
        ("Geometry", "Width (rad)", "Metric", *shift_columns),
        paired_rows,
    )
    claims = _chordal_claims(paired_means, widths, limits, replicates, shifts)
    warning_rows: list[list[Cell]] = []
    for geometry in GEOMETRIES:
        for width in widths[geometry]:
            warned = sum(fired[(geometry, width)])
            total = len(fired[(geometry, width)])
            size = Measurement.from_replicates(estimated[(geometry, width)])
            warning_rows.append(
                [geometry, width, limits[geometry], size, f"{warned} of {total}"]
            )
            claims.append(
                _warning_claim(geometry, width, limits[geometry], warned, total, size)
            )
    warning_table = Table.of(
        "Width estimated by the pooled whitening, in radians, and the comparisons"
        " for which `warn_wide_chordal_axes` fired.",
        ("Geometry", "Width (rad)", "Limit (rad)", "Estimated width", "Warned"),
        warning_rows,
    )
    figure = _chordal_figure(paired_means, widths, limits)
    circle_widths = ", ".join(f"{width:g}" for width in widths["circle"])
    sphere_widths = ", ".join(f"{width:g}" for width in widths["sphere"])
    setup = (
        f"Zoo wrapped Gaussians ({WRAPPED_GAUSSIANS})"
        " of width"
        f" {circle_widths} rad and von Mises-Fisher"
        f" distributions ({VON_MISES_FISHER}) of"
        f" width {sphere_widths} rad, shifted by"
        f" {', '.join(f'{shift:g}' for shift in shifts)} widths, with"
        f" {grouped(sample_count)} samples per set and {replicates} replicates."
        " The pooled whitening of"
        f" {role('func', 'mixed_signals.transforms.whiten.Whitening.from_samples')}"
        " averages the widths of the two posteriors, so it does not depend on the"
        " shift; the script fixes it once per width from a pilot pair of"
        f" {grouped(pilot_count)} samples, and the five chordal metrics run on the"
        " whitened samples with the"
        f" default permutation null of {draw_count} draws. The sliced Sinkhorn"
        f" divergence, the costliest, runs at a shift of 1 with {sliced_replicates}"
        f" replicates, {sliced_draws} null draws and a cap of"
        f" {SLICED_SINKHORN_CHECK_CAP} samples per set; its numbers are in the"
        " tables and carry no claim, because two replicates cannot show a"
        " deviation of 5%. The limits are"
        f" {role('data', 'mixed_signals.sigma.maps.CHORDAL_CIRCLE_WIDTH')} and"
        f" {role('data', 'mixed_signals.sigma.maps.CHORDAL_SPHERE_WIDTH')}. The sets"
        " of a replicate hold a shift of their own, which differs from the true"
        " shift by sampling noise and which every metric follows, so a claim tests"
        " the mean paired deviation over the shift, σ_eff minus the sample shift of"
        " the same replicate: it is within 5% of the true shift, plus the standard"
        " errors of the paired differences, at and below the limit. A width at the"
        " limit falls on either side of it by sampling, so the warning claim asks"
        " the estimated width to match the limit within its standard errors there."
    )
    return Section(
        setup, (table, held_table, paired_table, figure, warning_table), tuple(claims)
    )


def _finite_measurements(
    values: dict[tuple[str, float, float, str], list[float]],
) -> dict[tuple[str, float, float, str], Measurement]:
    """Return the mean and standard error of the finite values of each key."""
    return {
        key: Measurement.from_replicates(finite)
        for key, entries in values.items()
        if (finite := [entry for entry in entries if math.isfinite(entry)])
    }


def _chordal_claims(
    paired: dict[tuple[str, float, float, str], Measurement],
    widths: dict[str, tuple[float, ...]],
    limits: dict[str, float],
    replicates: int,
    shifts: Sequence[float],
) -> list[Claim]:
    claims: list[Claim] = []
    for geometry in GEOMETRIES:
        # * The sliced Sinkhorn has two replicates, so its claim could never fail.
        for metric in CHEAP_METRICS:
            cases = [
                Case(
                    f"{geometry} {width:g} rad, shift {shift:g}",
                    measurement.value,
                    measurement.margin(),
                    CHORDAL_TOLERANCE,
                )
                for width in widths[geometry]
                if width <= limits[geometry]
                for shift in shifts
                if (measurement := paired.get((geometry, width, shift, metric)))
                is not None
            ]
            claims.append(
                claim_cases(
                    f"{CHORDAL_NAMES[metric]} on {geometry}s at and below the limit"
                    " reads the sample shift.",
                    cases,
                    tolerance_text=f"{CHORDAL_TOLERANCE:.0%} of the truth",
                    replicates=replicates,
                )
            )
    return claims


def _warning_claim(
    geometry: str,
    width: float,
    limit: float,
    warned: int,
    total: int,
    size: Measurement,
) -> Claim:
    counts = f"{warned} of {total}"
    if width > limit:
        return claim_true(
            f"A {geometry} of width {width:g} rad above the limit warns.",
            "the warning fires in every comparison",
            counts,
            warned == total,
        )
    if width < limit:
        return claim_true(
            f"A {geometry} of width {width:g} rad below the limit stays silent.",
            "the warning never fires",
            counts,
            warned == 0,
        )
    return claim_true(
        f"A {geometry} of width {width:g} rad at the limit: the width is measured"
        " at the limit.",
        f"estimated width within {error_text(size.replicates)} of {limit:g}; the"
        " warning fires on either side of the limit by sampling",
        f"width {size.render()}; warned {counts}",
        abs(size.value - limit) <= size.margin(),
        True,
    )


def _chordal_figure(
    paired: dict[tuple[str, float, float, str], Measurement],
    widths: dict[str, tuple[float, ...]],
    limits: dict[str, float],
) -> Figure:
    groups = (CHORDAL_METRICS[:3], CHORDAL_METRICS[3:])
    panels_spec = [(geometry, group) for geometry in GEOMETRIES for group in groups]

    def draw(style: figures.FigureStyle) -> MatplotlibFigure:
        figure, panels = figures.new_figure(
            style, rows=2, columns=2, sharey=True, extra_height=OUTSIDE_LEGEND_HEIGHT
        )
        for axes, (geometry, group) in zip(panels, panels_spec, strict=True):
            figures.plot_tolerance_band(axes, style, CHORDAL_TOLERANCE)
            for series, metric in enumerate(group):
                points = [
                    (
                        width,
                        paired.get((geometry, width, SLICED_SINKHORN_SHIFT, metric)),
                    )
                    for width in widths[geometry]
                ]
                present = [
                    (width, entry) for width, entry in points if entry is not None
                ]
                figures.plot_series(
                    axes,
                    style,
                    series,
                    [width for width, _ in present],
                    [entry.value for _, entry in present],
                    label=CHORDAL_NAMES[metric],
                    errors=[entry.error or 0.0 for _, entry in present],
                )
            figures.plot_reference(axes, style, y_value=0.0)
            figures.plot_reference(axes, style, x_value=limits[geometry])
            figures.finish_panel(
                axes,
                style,
                title=f"{geometry}s",
                x_label="width (rad)",
                y_label="paired deviation over the shift",
                log_x=True,
            )
            axes.set_xticks(
                list(widths[geometry]), [f"{width:g}" for width in widths[geometry]]
            )
            axes.set_ylim(-0.5, 0.2)
            figures.add_legend(axes, style, outside=True)
        return figure

    data = {
        "shift": SLICED_SINKHORN_SHIFT,
        "paired_deviation": {
            f"{geometry}/{width:g}/{metric}": [entry.value, entry.error]
            for (geometry, width, shift, metric), entry in paired.items()
            if shift == SLICED_SINKHORN_SHIFT
        },
    }
    return Figure(
        label=f"{CHORDAL_LABEL}-ratio",
        caption="Paired deviation of σ_eff from the sample shift of its own"
        " replicate, over the true shift, against the width, at a shift of one"
        " width, for circles and spheres. The band is the 5% tolerance, the"
        " horizontal line zero and the vertical line the limit above which the"
        " package warns.",
        alt="Four panels of the paired deviation against the width, within the"
        " band up to the limit and falling below it above the limit.",
        draw=draw,
        data=data,
    )


# * Check sphere-width.

CONCENTRATIONS = (1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 400.0)
SPHERE_WIDTH_TOLERANCE = 0.01


def _sphere_width(context: Context) -> Section:
    label = SPHERE_WIDTH_LABEL
    sample_count = context.count(4_000, 1_000)
    replicates = context.count(24, 6)
    resultant_length = role(
        "func", "mixed_signals.zoo.closed_forms.von_mises_fisher_resultant_length"
    )
    rows: list[list[Cell]] = []
    claims: list[Claim] = []
    width_cases: list[Case] = []
    low_wide: list[bool] = []
    for index, concentration in enumerate(CONCENTRATIONS):
        width = 1 / math.sqrt(concentration)
        case = zoo.von_mises_fisher(0.0, width=width)
        samples = case.sample(
            context.key(label, "sets", index), sample_count, batch_size=replicates
        ).samples[:, 0]
        measured = spherical_width(
            samples[..., 0],
            samples[..., 1],
            jnp.zeros((replicates, sample_count)),
            PolarForm.LATITUDE,
        )
        measurement = Measurement.from_replicates(measured)
        exact = float(
            jnp.sqrt(1 - closed_forms.von_mises_fisher_resultant_length(concentration))
        )
        width_cases.append(
            Case(
                f"κ = {concentration:g}",
                measurement.value - exact,
                measurement.margin(),
                SPHERE_WIDTH_TOLERANCE * exact,
            )
        )
        if concentration <= CONCENTRATIONS[1]:
            low_wide.append(measurement.value + measurement.margin() < width)
        rows.append(
            [concentration, measurement, width, exact, measurement.value / width]
        )
    table = Table.of(
        "Sphere width √(1 - R) of von Mises-Fisher samples against the width"
        " 1/√κ and the exact width from the mean resultant length.",
        ("κ", "Measured width", "1/√κ", "Exact width", "Measured over 1/√κ"),
        rows,
    )
    claims.append(
        claim_cases(
            "The measured width equals the exact one.",
            width_cases,
            tolerance_text=f"{SPHERE_WIDTH_TOLERANCE:.0%} of the exact width",
            replicates=replicates,
        )
    )
    claims.append(
        claim_true(
            "On wide spheres the width falls below 1/√κ.",
            f"below 1/√κ by more than {error_text(replicates)} at κ ="
            f" {CONCENTRATIONS[0]:g} and {CONCENTRATIONS[1]:g}",
            f"{sum(low_wide)} of {len(low_wide)}",
            all(low_wide),
        )
    )
    setup = (
        "Zoo von Mises-Fisher distributions"
        f" ({VON_MISES_FISHER})"
        f" with κ from {CONCENTRATIONS[0]:g} to {CONCENTRATIONS[-1]:g},"
        f" {grouped(sample_count)} samples and {replicates} replicates each. The"
        " width is"
        f" {role('func', 'mixed_signals.core.samples.spherical_width')}, √(1 - R̄)"
        " for the mean resultant length R̄. A concentrated distribution has the"
        " width 1/√κ ({cite:t}`fisher1953dispersion`), and in general"
        " R̄ = coth κ - 1/κ, which"
        f" {resultant_length} gives exactly."
    )
    return Section(setup, (table,), tuple(claims))


# * Check chords-and-arcs.

ARC_NAMES = ("chords", "arcs")
BIMODAL_WIDTH = 0.3
RESOLVED = 3.0


def _circle_arcs(first: Array, second: Array, period: float) -> Array:
    """Return the shorter arcs between all pairs of angles."""
    gap = jnp.abs(first[:, None] - second[None, :]) % period
    return jnp.minimum(gap, period - gap)


def _sphere_arcs(first: Array, second: Array) -> Array:
    """Return the great-circle angles between all pairs of unit vectors."""
    return jnp.arccos(jnp.clip(first @ second.T, -1.0, 1.0))


def _arc_statistic(
    kind: str, period: float, polar_form: PolarForm, bandwidth: Array
) -> Statistic:
    """Return a statistic of the energy distance and the MMD squared along arcs."""

    def distances(first: Array, second: Array) -> Array:
        if kind == "circle":
            return _circle_arcs(first[:, 0], second[:, 0], period)
        units_first, _, _ = unit_vectors(first[:, 0], first[:, 1], polar_form)
        units_second, _, _ = unit_vectors(second[:, 0], second[:, 1], polar_form)
        return _sphere_arcs(units_first, units_second)

    def statistic(
        first: Array, first_weights: Array, second: Array, second_weights: Array
    ) -> Array:
        del first_weights, second_weights
        within_first = distances(first, first)
        within_second = distances(second, second)
        between = distances(first, second)

        def kernel(arc: Array) -> Array:
            return jnp.exp(-(arc**2) / (2 * bandwidth**2))

        energy = (
            2 * jnp.mean(between) - jnp.mean(within_first) - jnp.mean(within_second)
        )
        mmd = (
            jnp.mean(kernel(within_first))
            + jnp.mean(kernel(within_second))
            - 2 * jnp.mean(kernel(between))
        )
        return jnp.stack([energy, mmd])

    return statistic


def _arc_bandwidth(
    kind: str, period: float, polar_form: PolarForm, first: Array, second: Array
) -> Array:
    """Return the median arc between the pooled samples."""
    pooled = jnp.concatenate([first, second])
    if kind == "circle":
        arcs = _circle_arcs(pooled[:, 0], pooled[:, 0], period)
    else:
        units, _, _ = unit_vectors(pooled[:, 0], pooled[:, 1], polar_form)
        arcs = _sphere_arcs(units, units)
    rows, columns = jnp.triu_indices(pooled.shape[0], 1)
    return jnp.median(arcs[rows, columns])


def _chord_statistic(
    space: ParamSpace, bandwidth: Array, key: PRNGKeyArray
) -> Statistic:
    def statistic(
        first: Array, first_weights: Array, second: Array, second_weights: Array
    ) -> Array:
        return jnp.stack(
            [
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
            ]
        )

    return statistic


def _bimodal_sets(key: PRNGKeyArray, count: int, centres: tuple[float, float]) -> Array:
    """Return angles of an equal mixture of two wrapped normals, shaped (count, 1)."""
    component_key, noise_key = jax.random.split(key)
    choice = jax.random.bernoulli(component_key, 0.5, (count,))
    centre = jnp.where(choice, centres[0], centres[1])
    angles = centre + BIMODAL_WIDTH * jax.random.normal(noise_key, (count,))
    return (angles % (2 * math.pi))[:, None]


def _chords_and_arcs(context: Context) -> Section:
    label = CHORDS_LABEL
    sample_count = context.count(500, 200)
    replicates = context.count(6, 2)
    draw_count = context.count(20, 6)
    circle_case = zoo.wrapped_gaussians(1.0, width=0.3, param_count=1)
    sphere_case = zoo.von_mises_fisher(1.0, width=0.1)
    bimodal_space = circle_case.space
    cases = ("wrapped normals", "von Mises-Fisher", "bimodal circle")
    resolutions: dict[tuple[str, str, str], list[float]] = {}
    for case_name in cases:
        for replicate in range(replicates):
            key = context.key(label, case_name, replicate)
            if case_name == "wrapped normals":
                pair = circle_case.sample(
                    derive_key(key, "sets"), sample_count
                ).samples[0]
                space, kind = circle_case.space, "circle"
            elif case_name == "von Mises-Fisher":
                pair = sphere_case.sample(
                    derive_key(key, "sets"), sample_count
                ).samples[0]
                space, kind = sphere_case.space, "sphere"
            else:
                pair = jnp.stack(
                    [
                        _bimodal_sets(
                            derive_key(key, "first"), sample_count, (0.0, math.pi)
                        ),
                        _bimodal_sets(
                            derive_key(key, "second"),
                            sample_count,
                            (math.pi / 2, -math.pi / 2),
                        ),
                    ]
                )
                space, kind = bimodal_space, "circle"
            first, second = pair[0], pair[1]
            chord_bandwidth = median_bandwidth(
                first, None, second, None, space, key=derive_key(key, "bandwidth")
            )
            arc_bandwidth = _arc_bandwidth(
                kind, 2 * math.pi, PolarForm.LATITUDE, first, second
            )
            statistics = {
                "chords": _chord_statistic(
                    space, chord_bandwidth, derive_key(key, "subsample")
                ),
                "arcs": _arc_statistic(
                    kind, 2 * math.pi, PolarForm.LATITUDE, arc_bandwidth
                ),
            }
            plan: NullPlan = permutation_null(
                first,
                None,
                second,
                None,
                key=derive_key(key, "null"),
                draws=draw_count,
            )
            for distance_name, statistic in statistics.items():
                result: Comparison = compare_on_plan(
                    statistic, plan, first, None, second, None
                )
                resolution = _resolution(result.value, result.draws)
                for position, metric in enumerate(("energy", "mmd")):
                    resolutions.setdefault(
                        (case_name, distance_name, metric), []
                    ).append(float(resolution[position]))
    rows: list[list[Cell]] = []
    for case_name in cases:
        for metric, display in (("energy", "Energy distance"), ("mmd", "MMD squared")):
            rows.append(
                [
                    case_name,
                    display,
                    *[
                        Measurement.from_replicates(
                            resolutions[(case_name, distance_name, metric)]
                        )
                        for distance_name in ARC_NAMES
                    ],
                ]
            )
    table = Table.of(
        "Resolution, in null widths, of the energy distance and the MMD squared"
        " along chords (the package) and along arc lengths (the script).",
        ("Case", "Metric", "Chords", "Arcs"),
        rows,
    )
    claims: list[Claim] = []
    for metric, display in (("energy", "energy distance"), ("mmd", "MMD squared")):
        chords = Measurement.from_replicates(
            resolutions[("bimodal circle", "chords", metric)]
        )
        arcs = Measurement.from_replicates(
            resolutions[("bimodal circle", "arcs", metric)]
        )
        claims.append(
            claim_true(
                f"The {display} along chords resolves the bimodal circle.",
                f"resolution above {RESOLVED:g} null widths",
                chords.render(),
                chords.value > RESOLVED,
                True,
            )
        )
        claims.append(
            claim_true(
                f"The {display} along arcs does not resolve the bimodal circle.",
                f"resolution below {RESOLVED:g} null widths",
                arcs.render(),
                arcs.value < RESOLVED,
                True,
            )
        )
    arc_mmd = Measurement.from_replicates(
        resolutions[("bimodal circle", "arcs", "mmd")]
    )
    arc_text = (
        "Along arcs the MMD squared of the bimodal circle falls far below its null,"
        f" at {arc_mmd.render()} null widths. The upper-tail test reads that as no"
        " difference. The sign comes from the Gaussian kernel of arc lengths, which"
        " is not positive definite."
        if arc_mmd.value < -RESOLVED
        else "Along arcs the MMD squared of the bimodal circle reads"
        f" {arc_mmd.render()} null widths."
    )
    setup = (
        f"{sample_count} samples per set and {replicates} replicates, with"
        f" {draw_count} permutation draws. The shifts are one width of a wrapped"
        " normal of 0.3 rad and of a von Mises-Fisher distribution of 0.1 rad. The"
        " bimodal circle compares an equal mixture of two wrapped normals of 0.3 rad"
        " with modes at 0 and π against modes at ±π/2, which differ in even"
        " harmonics only. Along chords the package computes the distances in the"
        " embedding; along arcs the script computes the shorter arc on the circle and"
        " the great-circle angle on the sphere, a rejected alternative that the"
        " package does not offer. Both use the median of the pooled pairwise"
        " distances as the MMD bandwidth. The arc of two points on a circle is not"
        " negative type in the strict sense ({cite:t}`lyons2020strong`), and a"
        " Gaussian kernel of the geodesic distance is not positive definite"
        " ({cite:t}`feragen2015geodesic`)."
    )
    return Section(setup, (table, arc_text), tuple(claims))


CHECKS: tuple[Check, ...] = (
    Check(
        label=SINKHORN_LABEL,
        title="Sinkhorn regularisation and its null band",
        session="10",
        supports=(
            role(
                "data",
                "mixed_signals.metrics.ot.sinkhorn.SINKHORN_EPSILON_FACTOR",
                short=False,
            ),
            role(
                "func",
                "mixed_signals.metrics.ot.sinkhorn.sinkhorn_divergence",
                short=False,
            ),
        ),
        run=_sinkhorn_epsilon,
    ),
    Check(
        label=DIRECTIONS_LABEL,
        title="Directions of the sliced metrics",
        session="12b",
        supports=(
            role(
                "func", "mixed_signals.metrics.ot.sliced.sliced_directions", short=False
            ),
            role(
                "data", "mixed_signals.metrics.ot.sliced.SLICED_DIRECTIONS", short=False
            ),
            role(
                "data",
                "mixed_signals.metrics.ot.sliced.SLICED_SINKHORN_DIRECTIONS",
                short=False,
            ),
        ),
        run=_sliced_directions,
    ),
    Check(
        label=GAUSSIAN_SLICED_LABEL,
        title="The Gaussian estimator of sliced W2",
        session="12b",
        supports=(
            role(
                "func",
                "mixed_signals.metrics.ot.sliced.gaussian_sliced_wasserstein_distance",
                short=False,
            ),
            role(
                "func",
                "mixed_signals.metrics.ot.sliced.sliced_wasserstein_of_gaussians",
                short=False,
            ),
        ),
        run=_gaussian_sliced_w2,
    ),
    Check(
        label=CHORDAL_LABEL,
        title="Chordal metrics on wide circles and spheres",
        session="12b",
        supports=(
            role("data", "mixed_signals.sigma.maps.CHORDAL_CIRCLE_WIDTH", short=False),
            role("data", "mixed_signals.sigma.maps.CHORDAL_SPHERE_WIDTH", short=False),
            role(
                "func", "mixed_signals.sigma.maps.warn_wide_chordal_axes", short=False
            ),
        ),
        run=_chordal_widths,
    ),
    Check(
        label=SPHERE_WIDTH_LABEL,
        title="Width of a sphere",
        session="10",
        supports=(
            role("func", "mixed_signals.core.samples.spherical_width", short=False),
        ),
        run=_sphere_width,
    ),
    Check(
        label=CHORDS_LABEL,
        title="Chords against arc lengths",
        session="9",
        supports=(
            role(
                "func",
                "mixed_signals.metrics.kernel.maximum_mean_discrepancy",
                short=False,
            ),
            role("func", "mixed_signals.metrics.kernel.energy_distance", short=False),
        ),
        run=_chords_and_arcs,
    ),
)
