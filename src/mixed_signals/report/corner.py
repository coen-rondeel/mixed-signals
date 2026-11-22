"""Corner plots of posteriors: kernel contours, Gaussian ellipses and bounds.

:func:`corner_plot` lays out the plot with cantuccio (Santini 2026) and draws each
posterior with the estimate that the comparison rows read, so that a contour of
the plot is the density behind the numbers of the report.

Kernel estimate
    Every panel is a Gaussian kernel density estimate at one bandwidth factor,
    the multiplier of the standard deviations of the samples that sets the
    kernel widths. cantuccio draws a panel through its exact SciPy path. On a
    two-dimensional panel the density equals
    :class:`~mixed_signals.estimators.kde.KernelDensity` to 1e-10 away from bounds.
    On a diagonal panel cantuccio evaluates the estimate on a grid padded by 10% of
    the data range and interpolates it linearly onto the grid of the panel, which
    agrees with the package's estimate to below 1e-4 of the peak, about 1e-3
    relative. Contours enclose the highest-density regions of Hyndman (1996), at the
    masses :data:`CONTOUR_LEVELS`, 68% and 95%.
Bandwidth
    Silverman's one-dimensional factor (0.75 n)^(−1/5), with n the Kish
    effective sample size of the weights (Silverman 1986), is the factor of
    :class:`~mixed_signals.estimators.kde.KernelDensity` and of the marginal JS
    and Hellinger rows. :func:`silverman_factor` gives it for one posterior, and
    :func:`overview_factor` for several at their smallest n. A pair plot uses the
    factor that the rows of the pair report (:func:`pair_factor`), so its
    diagonals are the estimates behind its JS and Hellinger values. The factors
    always come from the full sets, before any subsample.
Bounded parameters
    cantuccio knows no bounds and draws circles on 800 by 800 grids. Circles
    are therefore unwrapped about their pooled circular mean, so that a posterior
    across the cut stays whole, and every panel with a circle, or with an
    interval parameter that a bound reaches, is redrawn from the package's
    estimate on a grid: the non-negative boundary correction of Jones and Foster
    (1996) on a one-dimensional interval, reflection in two dimensions (Schuster
    1985) and windings on circles. A bound is reached when it lies within
    :data:`~mixed_signals.estimators.base.IMAGE_REACH` kernel widths of the
    nearest sample, the reach at which the estimate adds images itself.
Size
    A plot of D parameters is 2 D inches square. marimo saves a figure at twice its
    dpi and shows it at 100 pixels per inch, so the page shows the plot 200 D
    pixels wide, from an image of 4 D times the dpi pixels. The dpi is the smaller
    of 100 and 2 400 / (4 D), which caps the image at 2 400 pixels across: a plot of
    more than six parameters is saved at a lower resolution than 100 dpi.
Gaussian ellipses
    The moment-matched Gaussian of each posterior, the fit behind the Gaussian KL,
    W2, Hellinger and sliced W2 rows, is drawn dashed in the colour of the
    posterior at the same masses. The radius of an ellipse at mass p is the root
    of the χ² quantile with two degrees of freedom, and the interval on a
    diagonal panel is the one with one degree of freedom (Johnson and Wichern
    2007).

References:
    :cite:t:`santini2026cantuccio`; :cite:t:`hyndman1996computing`;
    :cite:t:`silverman1986density`; :cite:t:`jones1996simple`;
    :cite:t:`schuster1985incorporating`; :cite:t:`johnson2007applied`.
"""

import json
from collections.abc import Sequence
from typing import Final

import cantuccio
import jax
import jax.numpy as jnp
import numpy as np
import polars as pl
from cantuccio.kde import hdi_levels
from jax.typing import ArrayLike
from jaxtyping import Float, PRNGKeyArray
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.mathtext import MathTextParser

from mixed_signals.core.samples import (
    SampleSet,
    circular_mean,
    kish_effective_sample_size,
    unwrap_about,
    weighted_covariance,
)
from mixed_signals.core.space import Parameter, ParamSpace
from mixed_signals.core.topology import Topology
from mixed_signals.engines._common import flat_space
from mixed_signals.errors import capture_warnings
from mixed_signals.estimators.base import IMAGE_REACH
from mixed_signals.estimators.gaussian import GaussianDensity
from mixed_signals.estimators.kde import KernelDensity, KernelDensitySettings
from mixed_signals.metrics.marginal import shared_bandwidth_factor
from mixed_signals.report._style import (
    CONTOUR_WIDTH,
    INK,
    LINE_WIDTH,
    REFERENCE,
    REFERENCE_WIDTH,
    REPORT_STYLE,
    TITLE_SIZE,
    new_figure,
    style_context,
)
from mixed_signals.report.tables import pair_rows

__all__ = [
    "CONTOUR_LEVELS",
    "OVERVIEW_LIMIT",
    "bounded_parameters",
    "corner_plot",
    "overview_factor",
    "pair_factor",
    "plot_subsample",
    "row_bandwidth",
    "silverman_factor",
]

CONTOUR_LEVELS: Final = (0.68, 0.95)
"""The masses of the kernel contours and of the Gaussian ellipses."""

OVERVIEW_LIMIT: Final = 3
"""The most posteriors that one plot draws together.

Beyond three, the reference palette no longer keeps every pair of colours apart
for colour-vision deficiencies when all of them overlap, so the overview is left
out and the pair plots carry the comparison.
"""

_DIAGONAL_POINTS: Final = 512
_PANEL_POINTS: Final = 80
_MAXIMUM_PIXELS: Final = 2_400
_MAXIMUM_DPI: Final = 100
# * marimo saves a figure at twice its dpi (marimo/_output/mpl.py).
_MARIMO_RESOLUTION: Final = 2
_PANEL_INCHES: Final = 2.0
# * Room for the title above the panels, in inches.
_TITLE_ROOM: Final = 0.45
_MATH_PARSER: Final = MathTextParser("path")


def silverman_factor(log_weights: Float[ArrayLike, " sample"]) -> float:
    """Return Silverman's one-dimensional bandwidth factor of one posterior.

    The factor is (0.75 n)^(−1/5) with n the Kish effective sample size, the
    factor of :class:`~mixed_signals.estimators.kde.KernelDensity` for one
    parameter and of
    :func:`~mixed_signals.metrics.marginal.shared_bandwidth_factor`.

    Args:
        log_weights: Natural-log weights of the samples, shaped (N,).

    Returns:
        The factor.

    References:
        :cite:t:`silverman1986density`.
    """
    return _factor_at(float(kish_effective_sample_size(jnp.asarray(log_weights))))


def overview_factor(sets: Sequence[SampleSet]) -> float:
    """Return Silverman's factor at the smallest Kish size of several posteriors.

    Args:
        sets: Sets of one problem and one posterior each.

    Returns:
        The factor of the posterior with the fewest effective samples, which no
        panel of the overview then undersmooths.

    References:
        :cite:t:`silverman1986density`.
    """
    sizes = [
        float(kish_effective_sample_size(sample_set.log_weights[0, 0]))
        for sample_set in sets
    ]
    return _factor_at(min(sizes))


def row_bandwidth(
    table: pl.DataFrame, problem: int, pair: tuple[str, str]
) -> float | None:
    """Return the bandwidth factor that the marginal rows of a pair report, if any.

    Args:
        table: A result table.
        problem: The problem index.
        pair: The two posterior names, in either order.

    Returns:
        The ``bandwidth`` in the settings of the first ``js`` or ``hellinger`` row
        of the pair that used the estimator ``kde``, or ``None`` when no such row
        holds one.
    """
    rows = pair_rows(table, problem, pair, ("compare",)).filter(
        pl.col("metric").is_in(["js", "hellinger"]) & (pl.col("estimator") == "kde")
    )
    for settings in rows["estimator_settings"]:
        bandwidth = json.loads(settings).get("bandwidth")
        if bandwidth is not None:
            return float(bandwidth)
    return None


def pair_factor(
    table: pl.DataFrame,
    problem: int,
    pair: tuple[str, str],
    first: SampleSet,
    second: SampleSet,
) -> float:
    """Return the bandwidth factor that the marginal rows of a pair share.

    Args:
        table: A result table.
        problem: The problem index.
        pair: The two posterior names, in either order.
        first: The samples of the first posterior, of one problem and posterior.
        second: The samples of the second posterior, likewise.

    Returns:
        The ``bandwidth`` of the ``js`` and ``hellinger`` rows of the pair that
        used the estimator ``kde`` (:func:`row_bandwidth`); when those rows did
        not run, the factor of
        :func:`~mixed_signals.metrics.marginal.shared_bandwidth_factor` of the
        two sets.

    References:
        :cite:t:`silverman1986density`.
    """
    bandwidth = row_bandwidth(table, problem, pair)
    if bandwidth is not None:
        return bandwidth
    shared = shared_bandwidth_factor(first.log_weights[0, 0], second.log_weights[0, 0])
    return float(shared)


def bounded_parameters(
    sets: Sequence[tuple[str, SampleSet]], space: ParamSpace, factor: float
) -> tuple[str, ...]:
    """Return the parameters whose panels come from the package's bounded estimate.

    Args:
        sets: The name and the samples of each posterior, each set of one
            problem and one posterior.
        space: The parameter space of the samples.
        factor: The bandwidth factor of the plot.

    Returns:
        The names of the circles, and of the interval parameters with a finite
        bound within :data:`~mixed_signals.estimators.base.IMAGE_REACH` kernel
        widths of the nearest sample of a posterior.
    """
    flat = flat_space(space)
    values = [np.asarray(sample_set.samples[0, 0]) for _, sample_set in sets]
    marked = _marked_positions(flat, sets, values, factor)
    return tuple(
        name for name, is_marked in zip(flat.names, marked, strict=True) if is_marked
    )


def plot_subsample(sample_set: SampleSet, cap: int, key: PRNGKeyArray) -> SampleSet:
    """Return a uniform subsample of a set, for a plot of a large posterior.

    Args:
        sample_set: A set of one problem and one posterior.
        cap: The most samples to keep.
        key: The PRNG key of the draw.

    Returns:
        The set unchanged when it holds at most ``cap`` samples. Above it, the
        samples of zero weight leave first, then ``cap`` of the others are drawn
        uniformly without replacement, with their weights kept, in their
        original order. The set then has no ``n_eff``.
    """
    if sample_set.shape.sample <= cap:
        return sample_set
    positive = jnp.flatnonzero(sample_set.log_weights[0, 0] > -jnp.inf)
    if positive.shape[0] > cap:
        positive = jnp.sort(jax.random.choice(key, positive, (cap,), replace=False))
    return sample_set.take_samples(positive)


def corner_plot(
    sets: Sequence[tuple[str, SampleSet]],
    space: ParamSpace,
    *,
    factor: float,
    colours: Sequence[str],
    title: str,
    gaussians: bool = True,
    bounds: bool = False,
    levels: Sequence[float] = CONTOUR_LEVELS,
) -> Figure:
    """Draw the posteriors of one problem in one corner plot.

    The module docstring describes the estimates and the size. The caller
    subsamples a large set first and passes the factor of the full set. The
    estimates and the Gaussian fits of the plot raise package warnings, such as a
    sample size below Silverman's table or an unresolved circle. The plot captures
    and drops them, because the rows of the engines carry the same warnings and the
    report lists them in its warnings section.

    Args:
        sets: The name and the samples of each posterior, each set of one
            problem and one posterior.
        space: The parameter space of the samples. Declarations of spheres are
            dropped, as in every marginal row, so that the azimuth of a sphere is
            a circle and its polar axis an interval.
        factor: The bandwidth factor of every panel.
        colours: One colour per posterior.
        title: The title above the plot.
        gaussians: Whether to draw the moment-matched Gaussian of each posterior,
            dashed.
        bounds: Whether to draw each finite bound of an interval parameter as a
            thin line.
        levels: The masses of the contours and the ellipses.

    Returns:
        The figure, which is not registered with pyplot.

    Raises:
        ValueError: If the number of colours differs from the number of
            posteriors.

    References:
        :cite:t:`santini2026cantuccio`; :cite:t:`hyndman1996computing`;
        :cite:t:`jones1996simple`; :cite:t:`schuster1985incorporating`;
        :cite:t:`johnson2007applied`.
    """
    if len(colours) != len(sets):
        raise ValueError(
            f"give one colour per posterior: {len(sets)} posteriors, {len(colours)}"
            " colours"
        )
    flat = flat_space(space)
    count = len(flat)
    names = [name for name, _ in sets]
    labels = _axis_labels(flat)
    centres = _pooled_centres([sample_set for _, sample_set in sets], flat)
    values = [_unwrapped_samples(sample_set, flat, centres) for _, sample_set in sets]
    weights = [
        np.asarray(sample_set.normalized_weights()[0, 0]) for _, sample_set in sets
    ]
    marked = _marked_positions(flat, sets, values, factor)
    inches = _PANEL_INCHES * count
    with style_context():
        figure = new_figure(
            inches,
            inches,
            dpi=_figure_dpi(inches),
        )
        axes = figure.subplots(count, count, squeeze=False)
        cantuccio.cornerplot(
            [dict(zip(labels, array.T, strict=True)) for array in values],
            columns=labels,
            weights=weights,
            labels=names,
            colors=list(colours),
            diag_mode="kde",
            offdiag_mode="kde",
            contour_levels=tuple(levels),
            kde_kwargs={"bandwidth": factor, "fast": False, "linewidth": LINE_WIDTH},
            offdiag_kwargs={"linewidths": CONTOUR_WIDTH},
            periodic=None,
            stylefile=REPORT_STYLE,
            fig=figure,
            axes=axes,
        )
        for position in range(count):
            # * Freeze the diagonal limits, which the later lines must not change.
            axes[position, position].set_xlim(axes[position, position].get_xlim())
        # * The rows carry the warnings of these fits already, so the plot is silent.
        with capture_warnings():
            _redraw(axes, flat, sets, values, centres, marked, factor, colours, levels)
            if gaussians:
                _draw_gaussians(figure, flat, sets, centres, colours, levels)
        if bounds:
            _draw_bounds(axes, flat)
        figure.subplots_adjust(top=1 - _TITLE_ROOM / inches)
        figure.suptitle(
            title, color=INK, fontsize=TITLE_SIZE, y=1 - 0.1 / inches, va="top"
        )
    return figure


def _figure_dpi(inches: float) -> float:
    """Return the dpi that keeps the saved image of a figure within 2 400 pixels.

    marimo saves a figure at twice its dpi, so the cap divides by that factor.
    """
    return min(float(_MAXIMUM_DPI), _MAXIMUM_PIXELS / (_MARIMO_RESOLUTION * inches))


def _factor_at(size: float) -> float:
    """Return Silverman's one-dimensional factor (0.75 n)^(−1/5) at a size n."""
    return float((0.75 * size) ** -0.2)


def _axis_labels(space: ParamSpace) -> list[str]:
    """Return the axis label of each parameter: its label and unit, or its name."""
    labels = []
    for parameter in space.parameters:
        text = _label_text(parameter)
        labels.append(text if parameter.unit is None else f"{text} [{parameter.unit}]")
    # * cantuccio keys its columns by label, so labels must be unique.
    return labels if len(set(labels)) == len(labels) else list(space.names)


def _label_text(parameter: Parameter) -> str:
    """Return the LaTeX label of a parameter as mathtext, or its name."""
    if parameter.label == parameter.name:
        return parameter.name
    text = f"${parameter.label}$"
    try:
        _MATH_PARSER.parse(text)
    except ValueError:
        return parameter.name
    return text


def _circle_positions(space: ParamSpace) -> list[int]:
    """Return the positions of the circle parameters."""
    return [
        position
        for position, parameter in enumerate(space.parameters)
        if parameter.topology is Topology.CIRCLE
    ]


def _pooled_centres(sets: Sequence[SampleSet], space: ParamSpace) -> dict[int, float]:
    """Return the circular mean of each circle over all posteriors, equally weighted."""
    centres = {}
    log_share = -float(jnp.log(len(sets)))
    pooled_weights = jnp.concatenate(
        [sample_set.normalized_log_weights()[0, 0] + log_share for sample_set in sets]
    )
    for position in _circle_positions(space):
        parameter = space.parameters[position]
        assert parameter.period is not None
        pooled = jnp.concatenate(
            [sample_set.samples[0, 0, :, position] for sample_set in sets]
        )
        centres[position] = float(
            circular_mean(pooled, pooled_weights, parameter.period, parameter.bounds[0])
        )
    return centres


def _unwrapped_samples(
    sample_set: SampleSet, space: ParamSpace, centres: dict[int, float]
) -> np.ndarray:
    """Return the samples (N, D) with each circle unwrapped about the pooled mean."""
    samples = sample_set.samples[0, 0]
    for position, centre in centres.items():
        period = space.parameters[position].period
        assert period is not None
        samples = samples.at[:, position].set(
            unwrap_about(samples[:, position], centre, period)
        )
    return np.asarray(samples)


def _marked_positions(
    space: ParamSpace,
    sets: Sequence[tuple[str, SampleSet]],
    values: Sequence[np.ndarray],
    factor: float,
) -> list[bool]:
    """Return which parameters need the package's estimate."""
    marked = []
    for position, parameter in enumerate(space.parameters):
        if parameter.topology is Topology.CIRCLE:
            marked.append(True)
        elif parameter.topology is Topology.INTERVAL:
            marked.append(
                any(
                    _reaches_bound(
                        array[:, position],
                        sample_set.log_weights[0, 0],
                        parameter,
                        factor,
                    )
                    for (_, sample_set), array in zip(sets, values, strict=True)
                )
            )
        else:
            marked.append(False)
    return marked


def _reaches_bound(
    column: np.ndarray, log_weights: jax.Array, parameter: Parameter, factor: float
) -> bool:
    """Return whether a finite bound lies within the kernel reach of a column."""
    lower, upper = parameter.bounds
    _, covariance = weighted_covariance(jnp.asarray(column)[:, None], log_weights)
    reach = IMAGE_REACH * factor * float(jnp.sqrt(covariance[0, 0]))
    held = column[np.asarray(log_weights) > -np.inf]
    return bool(held.min() - lower <= reach or upper - held.max() <= reach)


def _grid_limits(
    limits: tuple[float, float], parameter: Parameter, centre: float | None
) -> tuple[float, float]:
    """Return the limits of a panel clipped to its bounds, or to one period."""
    low, high = sorted(limits)
    if centre is not None:
        assert parameter.period is not None
        half = parameter.period / 2
        return max(low, centre - half), min(high, centre + half)
    lower, upper = parameter.bounds
    return max(low, lower), min(high, upper)


def _redraw(
    axes: np.ndarray,
    space: ParamSpace,
    sets: Sequence[tuple[str, SampleSet]],
    values: Sequence[np.ndarray],
    centres: dict[int, float],
    marked: Sequence[bool],
    factor: float,
    colours: Sequence[str],
    levels: Sequence[float],
) -> None:
    """Replace every panel of a marked parameter by the package's own estimate."""
    settings = KernelDensitySettings(bandwidth=factor)
    for row in range(len(space)):
        for column in range(row + 1):
            if not (marked[row] or marked[column]):
                continue
            panel = axes[row, column]
            positions = [row] if row == column else [column, row]
            sub_space = ParamSpace(
                parameters=tuple(space.parameters[position] for position in positions)
            )
            limits = [
                _grid_limits(
                    panel.get_xlim() if index == 0 else panel.get_ylim(),
                    space.parameters[position],
                    centres.get(position),
                )
                for index, position in enumerate(positions)
            ]
            densities = [
                _grid_density(
                    array[:, positions],
                    sample_set.log_weights[0, 0],
                    sub_space,
                    settings,
                    limits,
                )
                for (_, sample_set), array in zip(sets, values, strict=True)
            ]
            if row == column:
                _draw_diagonal(panel, limits[0], densities, colours)
            else:
                _draw_contours(panel, limits, densities, colours, levels)


def _grid_density(
    samples: np.ndarray,
    log_weights: jax.Array,
    space: ParamSpace,
    settings: KernelDensitySettings,
    limits: Sequence[tuple[float, float]],
) -> np.ndarray:
    """Return the estimate on a grid: (M,) in one dimension, else (M, M)."""
    estimator = KernelDensity.fit(samples, log_weights, space, settings)
    if len(limits) == 1:
        grid = np.linspace(*limits[0], _DIAGONAL_POINTS)
        return np.exp(np.asarray(estimator.log_density(grid[:, None])))
    x_grid = np.linspace(*limits[0], _PANEL_POINTS)
    y_grid = np.linspace(*limits[1], _PANEL_POINTS)
    mesh_x, mesh_y = np.meshgrid(x_grid, y_grid)
    points = np.stack([mesh_x.ravel(), mesh_y.ravel()], axis=-1)
    return np.exp(np.asarray(estimator.log_density(points))).reshape(mesh_x.shape)


def _draw_diagonal(
    panel: Axes,
    limits: tuple[float, float],
    densities: Sequence[np.ndarray],
    colours: Sequence[str],
) -> None:
    """Replace the lines of a diagonal panel; its floor stays 0 and its top follows."""
    for line in list(panel.lines):
        line.remove()
    grid = np.linspace(*limits, _DIAGONAL_POINTS)
    for density, colour in zip(densities, colours, strict=True):
        panel.plot(grid, density, color=colour, linewidth=LINE_WIDTH)
    panel.set_ylim(0.0, 1.05 * max(float(density.max()) for density in densities))


def _draw_contours(
    panel: Axes,
    limits: Sequence[tuple[float, float]],
    densities: Sequence[np.ndarray],
    colours: Sequence[str],
    levels: Sequence[float],
) -> None:
    """Replace the contours of an off-diagonal panel, keeping its limits."""
    x_limits, y_limits = panel.get_xlim(), panel.get_ylim()
    for collection in list(panel.collections):
        collection.remove()
    mesh_x, mesh_y = np.meshgrid(
        np.linspace(*limits[0], _PANEL_POINTS), np.linspace(*limits[1], _PANEL_POINTS)
    )
    for density, colour in zip(densities, colours, strict=True):
        # * The same levels as cantuccio: the highest-density regions of the grid.
        thresholds = np.unique(hdi_levels(density, levels))
        thresholds = thresholds[thresholds < density.max()]
        if thresholds.size:
            panel.contour(
                mesh_x,
                mesh_y,
                density,
                levels=thresholds,
                colors=[colour],
                linewidths=CONTOUR_WIDTH,
            )
    panel.set_xlim(x_limits)
    panel.set_ylim(y_limits)


def _draw_gaussians(
    figure: Figure,
    space: ParamSpace,
    sets: Sequence[tuple[str, SampleSet]],
    centres: dict[int, float],
    colours: Sequence[str],
    levels: Sequence[float],
) -> None:
    """Overlay the dashed moment-matched Gaussian of each posterior."""
    for (_, sample_set), colour in zip(sets, colours, strict=True):
        estimator = GaussianDensity.fit(
            sample_set.samples[0, 0], sample_set.log_weights[0, 0], space
        )
        means = np.array(estimator.mean)
        for position, centre in centres.items():
            period = space.parameters[position].period
            assert period is not None
            # * The fit wraps a circle's mean onto the circle; the plot is unwrapped.
            means[position] = float(
                unwrap_about(jnp.asarray(means[position]), centre, period)
            )
        cantuccio.overlay_covariance(
            figure,
            np.asarray(estimator.covariance),
            means,
            levels=tuple(levels),
            plot_1d=True,
            colors=colour,
            linestyles="--",
            linewidths=CONTOUR_WIDTH,
        )


def _draw_bounds(axes: np.ndarray, space: ParamSpace) -> None:
    """Draw each finite bound inside the limits of a panel as a thin line."""
    for row in range(len(space)):
        for column in range(row + 1):
            panel = axes[row, column]
            for line_position, limits, vertical in (
                (column, panel.get_xlim(), True),
                (row, panel.get_ylim(), False),
            ):
                if not vertical and row == column:
                    continue
                parameter = space.parameters[line_position]
                if parameter.topology is not Topology.INTERVAL:
                    continue
                for bound in parameter.bounds:
                    if np.isfinite(bound) and min(limits) <= bound <= max(limits):
                        draw = panel.axvline if vertical else panel.axhline
                        draw(bound, color=REFERENCE, linewidth=REFERENCE_WIDTH)
