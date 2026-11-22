"""Figures of the comparison and of the nulls: heatmap, null histograms, ε curve.

Every function returns a :class:`matplotlib.figure.Figure` that is not registered
with pyplot, drawn under the style of :mod:`mixed_signals.report._style`, or
``None`` where a figure has nothing to show.

Heatmap
    :func:`effect_heatmap` shows the effect size of every comparison row over its
    threshold, so that one colour scale serves every metric: a cell at 1 has
    reached its threshold. The text of a cell is σ_eff or r_eff, in bold where
    σ_sig reaches the significance level, and in ink or white, whichever has the
    larger contrast ratio against the fill of the cell (WCAG 2.2). A frame marks a
    row that fails, so that a flag does not depend on colour alone. The labels of
    the rows and columns name the cells and are drawn in the secondary ink.
Null histograms
    :func:`null_histograms` draws the null of every row that stores its draws,
    with the value of the row and the null mean. A row that flags comes first and
    is framed, with the word "flags" in its title.
Entropic regularisation
    :func:`epsilon_plot` shows the Sinkhorn divergence S_ε against ε/ℓ², with ℓ
    the median bandwidth that sets the scale of ε. The band covers the null
    between the Φ(−L) and Φ(L) quantiles for the significance level L, so a row
    reaches the level where its value passes the upper edge of the band
    (Cowan et al. 2011).

References:
    :cite:t:`cowan2011asymptotic`; :cite:t:`feydy2019interpolating`.
"""

import json
import math
from collections.abc import Mapping, Sequence
from typing import Any, Final

import jax.numpy as jnp
import numpy as np
import polars as pl
from jax.scipy.special import ndtr
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_rgb
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator, NullFormatter

from mixed_signals.report._style import (
    BACKGROUND,
    BAND_OPACITY,
    CRITICAL,
    GRID,
    HAIRLINE_WIDTH,
    INK,
    LINE_WIDTH,
    MUTED_INK,
    PANEL_TITLE_SIZE,
    REFERENCE,
    REFERENCE_WIDTH,
    SECONDARY_INK,
    SEQUENTIAL_BLUE,
    SERIES_COLOURS,
    TICK_SIZE,
    TITLE_SIZE,
    new_figure,
    style_context,
)
from mixed_signals.transforms.base import COMPOSITION_SEPARATOR

__all__ = ["effect_heatmap", "epsilon_plot", "null_histograms"]

HEATMAP_CEILING: Final = 2.0
"""The effect over threshold at which the colour ramp ends: step 700 of the blue."""

HISTOGRAM_COLUMNS: Final = 4
MAXIMUM_BINS: Final = 40
_RAMP: Final = LinearSegmentedColormap.from_list("effect", SEQUENTIAL_BLUE)
_RAMP_NORM: Final = Normalize(vmin=0.0, vmax=HEATMAP_CEILING, clip=True)
_CELL_INSET: Final = 0.04
_CELL_TEXT_SIZE: Final = 7.5
_KL_METRICS: Final = ("kl", "kl_location", "kl_shape")
_PANEL_INCHES: Final = (2.0, 1.7)
_JOINT: Final = "joint"
_WHITE: Final = "#ffffff"
_MAXIMUM_NAMED_PARAMETERS: Final = 2
# * WCAG 2.2 linearises sRGB below this value of a channel with a straight line.
_SRGB_KNEE: Final = 0.040_45


def effect_heatmap(
    comparison: pl.DataFrame, parameters: Sequence[str], *, title: str
) -> Figure | None:
    """Draw the effect over threshold of every comparison row of one pair.

    A row of the heatmap is one metric with its estimator and, for the
    Kullback-Leibler family, its direction, and for the Sinkhorn rows their
    ε/ℓ². A column is a parameter that has a cell, in the order of ``parameters``,
    then ``joint`` for rows over more than one parameter; a parameter without a
    row, such as in a run of one joint metric, leaves no empty column. A cell is
    coloured by effect over threshold on the blue ramp from 0 to 2 and above, so
    values below zero take the lightest step. Its text is σ_eff to two decimals,
    or r_eff to three, in bold where σ_sig reaches the level, in ink or in white,
    whichever has the larger contrast ratio against the fill by the relative
    luminance of WCAG 2.2. A rectangle in ink frames each failing cell. A cell
    without a row stays empty, and a NaN effect reads ``NaN``. The labels of the
    rows and columns are in the secondary ink.

    Args:
        comparison: The output of
            :func:`~mixed_signals.report.tables.comparison_table`.
        parameters: The parameter names in space order.
        title: The title of the figure.

    Returns:
        The figure, or ``None`` for an empty table.
    """
    if comparison.is_empty():
        return None
    entries = comparison.to_dicts()
    keys: list[tuple[str, str, str | None, float | None]] = []
    for entry in entries:
        key = _row_key(entry)
        if key not in keys:
            keys.append(key)
    named = {entry["parameters"] for entry in entries}
    columns = [name for name in parameters if name in named]
    if any(entry["parameters"] not in parameters for entry in entries):
        columns.append(_JOINT)
    with style_context():
        figure = new_figure(
            2.6 + 0.85 * len(columns) + 1.0,
            0.9 + 0.3 * len(keys),
            layout="constrained",
        )
        axes = figure.subplots()
        for entry in entries:
            row = keys.index(_row_key(entry))
            column = (
                columns.index(entry["parameters"])
                if entry["parameters"] in parameters
                else len(columns) - 1
            )
            _draw_cell(axes, row, column, entry)
        axes.set_xlim(0, len(columns))
        axes.set_ylim(len(keys), 0)
        axes.set_xticks([position + 0.5 for position in range(len(columns))], columns)
        axes.set_yticks(
            [position + 0.5 for position in range(len(keys))],
            [_row_label(*key) for key in keys],
        )
        # * The labels name each cell, so they take the ink of axis labels.
        axes.tick_params(length=0, labelcolor=SECONDARY_INK)
        for spine in axes.spines.values():
            spine.set_visible(False)
        axes.set_title(title, loc="left")
        colour_bar = figure.colorbar(
            ScalarMappable(norm=_RAMP_NORM, cmap=_RAMP),
            ax=axes,
            ticks=[0.0, 0.5, 1.0, 1.5, 2.0],
            shrink=0.6,
        )
        colour_bar.ax.set_yticklabels(["0", "0.5", "1", "1.5", "2+"])
        colour_bar.set_label("effect / threshold")
        colour_bar.outline.set_visible(False)
    return figure


def null_histograms(rows: pl.DataFrame, *, title: str) -> Figure | None:
    """Draw the null of every row that stores finite draws, one panel per row.

    A panel shows the histogram of the draws, the value of the row as a solid
    line in ink and the null mean as a thin line in muted ink. Its title has two
    lines: the metric and estimator, with the direction of a Kullback-Leibler row
    and ``data`` or ``chart`` for a row of data space; then the parameters,
    joined with ``", "`` for one or two of them and ``joint`` for more, σ_sig,
    ``(tail fit)`` when the p-value comes from the fit to the tail and, for a
    flagged row, the word ``flags``. σ_sig has two decimals, the minus sign − of
    the axes, ``∞`` or ``−∞`` where infinite, and ``NaN`` where undefined. A
    flagged panel has a frame in the status colour and comes first.

    Args:
        rows: Result rows with the columns of
            :func:`~mixed_signals.report.tables.with_effects`.
        title: The title of the figure.

    Returns:
        The figure with four columns of panels, or ``None`` when no row holds a
        finite draw.
    """
    panels = []
    for row in rows.iter_rows(named=True):
        draws = np.asarray(row["null_draws"] or [], dtype=float)
        draws = draws[np.isfinite(draws)]
        if draws.size:
            panels.append((row, draws))
    if not panels:
        return None
    # * A stable partition: the rows that flag first, each group in table order.
    panels = [panel for panel in panels if panel[0]["flag"]] + [
        panel for panel in panels if not panel[0]["flag"]
    ]
    line_count = math.ceil(len(panels) / HISTOGRAM_COLUMNS)
    width, height = _PANEL_INCHES
    with style_context():
        figure = new_figure(
            width * HISTOGRAM_COLUMNS, height * line_count + 0.5, layout="constrained"
        )
        axes = figure.subplots(line_count, HISTOGRAM_COLUMNS, squeeze=False)
        for panel_axes in axes.ravel()[len(panels) :]:
            panel_axes.set_visible(False)
        for panel_axes, (row, draws) in zip(
            axes.ravel()[: len(panels)], panels, strict=True
        ):
            _draw_null(panel_axes, row, draws)
        figure.suptitle(title, color=INK, fontsize=TITLE_SIZE)
    return figure


def epsilon_plot(rows: pl.DataFrame, *, level: float) -> Figure | None:
    """Draw the Sinkhorn divergence against ε/ℓ² with its null and its σ_eff.

    The left panel shows S_ε, the value of each ``sinkhorn`` row, against the
    factor ε/ℓ² on a logarithmic axis, with the median of the null as a thin line
    and a band between the Φ(−L) and Φ(L) quantiles of its draws, with L the
    significance level. The right panel shows σ_eff against the factor, with the
    threshold as a reference line.

    Args:
        rows: The ``sinkhorn`` rows of one pair, as result rows.
        level: The significance level L, in σ.

    Returns:
        The figure, or ``None`` below two distinct values of ``epsilon_factor``
        in the settings of the rows.

    References:
        :cite:t:`cowan2011asymptotic`; :cite:t:`feydy2019interpolating`.
    """
    entries = []
    for row in rows.filter(pl.col("metric") == "sinkhorn").iter_rows(named=True):
        factor = json.loads(row["estimator_settings"]).get("epsilon_factor")
        if factor is not None:
            entries.append((float(factor), row))
    if len({factor for factor, _ in entries}) < 2:
        return None
    entries.sort(key=_first_item)
    factors = np.array([factor for factor, _ in entries])
    lower_mass, upper_mass = float(ndtr(-level)), float(ndtr(level))
    medians, lows, highs = [], [], []
    for _, row in entries:
        draws = jnp.asarray(row["null_draws"] or [], dtype=jnp.float64)
        draws = draws[jnp.isfinite(draws)]
        if draws.size:
            low, high = jnp.quantile(draws, jnp.asarray([lower_mass, upper_mass]))
            medians.append(float(jnp.median(draws)))
            lows.append(float(low))
            highs.append(float(high))
        else:
            medians.append(math.nan)
            lows.append(math.nan)
            highs.append(math.nan)
    with style_context():
        figure = new_figure(7.0, 3.2, layout="constrained")
        value_axes, effect_axes = figure.subplots(1, 2)
        value_axes.plot(
            factors,
            [row["value"] for _, row in entries],
            color=SERIES_COLOURS[0],
            marker="o",
            markersize=4,
            linewidth=LINE_WIDTH,
        )
        value_axes.plot(factors, medians, color=MUTED_INK, linewidth=REFERENCE_WIDTH)
        value_axes.fill_between(
            factors, lows, highs, color=MUTED_INK, alpha=BAND_OPACITY, linewidth=0
        )
        effect_axes.plot(
            factors,
            [row["sigma_eff"] for _, row in entries],
            color=SERIES_COLOURS[0],
            marker="o",
            markersize=4,
            linewidth=LINE_WIDTH,
        )
        threshold = entries[0][1]["threshold"]
        if threshold is not None:
            effect_axes.axhline(threshold, color=REFERENCE, linewidth=REFERENCE_WIDTH)
        for panel, label in (
            (value_axes, r"$S_\varepsilon$"),
            (effect_axes, r"$\sigma_\mathrm{eff}$"),
        ):
            _finish_epsilon_axes(panel, factors, label)
    return figure


def _first_item(entry: tuple[float, Mapping[str, Any]]) -> float:
    """Return the factor of an entry, by which the entries sort."""
    return entry[0]


def _row_key(entry: Mapping[str, Any]) -> tuple[str, str, str | None, float | None]:
    """Return what tells the heatmap rows apart: metric, estimator, direction, ε."""
    return (
        entry["metric"],
        entry["estimator"],
        entry["direction"],
        entry.get("epsilon_factor"),
    )


def _row_label(
    metric: str, estimator: str, direction: str | None, epsilon_factor: float | None
) -> str:
    """Return the label of a heatmap row: metric, estimator, direction and ε/ℓ²."""
    label = f"{metric}:{estimator}"
    if direction is not None:
        label += f" {direction}"
    if epsilon_factor is not None:
        label += f" ε/ℓ² {epsilon_factor:g}"
    return label


def _cell_text(entry: Mapping[str, Any]) -> str:
    """Return σ_eff to two decimals, r_eff to three, or NaN for an undefined effect."""
    effect = entry["effect"]
    if effect is None or math.isnan(effect):
        return "NaN"
    if entry["sigma_eff"] is not None:
        return f"{entry['sigma_eff']:.2f}"
    return f"{entry['r_eff']:.3f}"


def _draw_cell(axes: Axes, row: int, column: int, entry: Mapping[str, Any]) -> None:
    """Draw the fill, the text and the frame of one heatmap cell."""
    effect, threshold = entry["effect"], entry["threshold"]
    ratio = math.nan
    if effect is not None and threshold:
        ratio = effect / threshold
    has_colour = not math.isnan(ratio)
    # * Values below zero take the lightest step, values above the ceiling the darkest.
    colour = _RAMP(_RAMP_NORM(max(ratio, 0.0))) if has_colour else "none"
    corner = (column + _CELL_INSET, row + _CELL_INSET)
    size = 1 - 2 * _CELL_INSET
    axes.add_patch(Rectangle(corner, size, size, facecolor=colour, edgecolor="none"))
    if entry["flag"]:
        axes.add_patch(
            Rectangle(corner, size, size, fill=False, edgecolor=INK, linewidth=1.5)
        )
    fill = to_rgb(colour) if has_colour else to_rgb(BACKGROUND)
    axes.text(
        column + 0.5,
        row + 0.5,
        _cell_text(entry),
        ha="center",
        va="center",
        fontsize=_CELL_TEXT_SIZE,
        fontweight="bold" if entry["resolved"] else "normal",
        color=_text_colour(fill),
    )


# * Text colour by the relative luminance and contrast ratio of WCAG 2.2.
def _relative_luminance(colour: tuple[float, float, float]) -> float:
    """Return the relative luminance of an sRGB colour, as WCAG 2.2 defines it."""
    red, green, blue = (
        channel / 12.92 if channel <= _SRGB_KNEE else ((channel + 0.055) / 1.055) ** 2.4
        for channel in colour
    )
    return 0.2126 * red + 0.7152 * green + 0.0722 * blue


def _contrast_ratio(
    first: tuple[float, float, float], second: tuple[float, float, float]
) -> float:
    """Return the contrast ratio of two sRGB colours, as WCAG 2.2 defines it."""
    lighter, darker = sorted(
        (_relative_luminance(first), _relative_luminance(second)), reverse=True
    )
    return (lighter + 0.05) / (darker + 0.05)


def _text_colour(fill: tuple[float, float, float]) -> str:
    """Return ink or white, whichever has the larger contrast ratio on the fill."""
    ink = _contrast_ratio(to_rgb(INK), fill)
    white = _contrast_ratio(to_rgb(_WHITE), fill)
    return INK if ink >= white else _WHITE


def _format_sigma(value: float) -> str:
    """Return σ_sig as the axes write numbers: two decimals and the minus sign −."""
    if math.isnan(value):
        return "NaN"
    if math.isinf(value):
        return "∞" if value > 0 else "−∞"
    return f"{value:.2f}".replace("-", "−")


def _draw_null(axes: Axes, row: Mapping[str, Any], draws: np.ndarray) -> None:
    """Draw the null of one row: histogram, value, null mean, title and frame."""
    edges = np.histogram_bin_edges(draws, bins="auto")
    if len(edges) - 1 > MAXIMUM_BINS:
        edges = np.histogram_bin_edges(draws, bins=MAXIMUM_BINS)
    counts, _ = np.histogram(draws, bins=edges)
    axes.stairs(counts, edges, fill=True, color=SEQUENTIAL_BLUE[3])
    low, high = float(draws.min()), float(draws.max())
    value = row["value"]
    if math.isfinite(value):
        axes.axvline(value, color=INK, linewidth=LINE_WIDTH)
        low, high = min(low, value), max(high, value)
    if row["null_mean"] is not None and math.isfinite(row["null_mean"]):
        axes.axvline(row["null_mean"], color=MUTED_INK, linewidth=REFERENCE_WIDTH)
    padding = 0.05 * (high - low) or 0.5
    axes.set_xlim(low - padding, high + padding)
    axes.set_yticks([])
    axes.xaxis.set_major_locator(MaxNLocator(nbins=3))
    axes.tick_params(axis="x", labelsize=TICK_SIZE - 1.5)
    axes.set_title(_null_title(row), fontsize=PANEL_TITLE_SIZE, loc="left")
    if row["flag"]:
        for spine in axes.spines.values():
            spine.set_visible(True)
            spine.set_color(CRITICAL)
            spine.set_linewidth(1.2)


def _null_title(row: Mapping[str, Any]) -> str:
    """Return the two-line title of a null panel."""
    first = f"{row['metric']}:{row['estimator']}"
    if row["metric"] in _KL_METRICS:
        first += f" {row['pair'][0]}‖{row['pair'][1]}"
    space = row["space"].split(COMPOSITION_SEPARATOR)[-1]
    if space.startswith("data"):
        first += " data"
    elif space.startswith("chart"):
        first += " chart"
    names = row["parameters"]
    second = ", ".join(names) if len(names) <= _MAXIMUM_NAMED_PARAMETERS else _JOINT
    if row["sigma_sig"] is not None:
        second += f"  σ_sig {_format_sigma(row['sigma_sig'])}"
    if row["extrapolated"]:
        second += " (tail fit)"
    if row["flag"]:
        second += "  flags"
    return f"{first}\n{second}"


def _finish_epsilon_axes(axes: Axes, factors: np.ndarray, label: str) -> None:
    """Give a panel of the ε curve its log axis, labels and value-axis gridlines."""
    axes.set_xscale("log")
    axes.set_xticks(factors, [f"{factor:g}" for factor in factors])
    axes.xaxis.set_minor_formatter(NullFormatter())
    axes.set_xlabel(r"$\varepsilon / \ell^2$")
    axes.set_ylabel(label)
    axes.yaxis.grid(True, color=GRID, linewidth=HAIRLINE_WIDTH, linestyle="-")
