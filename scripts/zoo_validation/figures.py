"""Figure style of the validation page, drawn once in a light and once in a dark theme.

Every figure is drawn twice from the same data, because the docs theme (furo)
has a light and a dark mode and shows the matching file. The palette, the line
widths and the layout follow one reference style: an opaque background, hairline
gridlines on the value axis only, at most three series per panel in a fixed
order, a thin solid reference line, and text in neutral ink, never in a series
colour.
"""

import dataclasses
from collections.abc import Iterator, Sequence
from contextlib import AbstractContextManager
from typing import Literal

import matplotlib
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure as MatplotlibFigure
from matplotlib.ticker import NullFormatter
from matplotlib.transforms import ScaledTranslation
from numpy.typing import ArrayLike

type LegendLocation = Literal[
    "best", "upper left", "upper right", "lower left", "lower right", "center right"
]
"""Places of a legend that the figures use."""

FIGURE_WIDTH = 6.4
"""Width of every figure, in inches."""

SINGLE_PANEL_HEIGHT = 3.6
"""Height of a figure with one panel, in inches."""

ROW_HEIGHT = 2.6
"""Height of one row of small multiples, in inches."""

LEGEND_DROP = 0.5
"""Distance of an outside legend below its axes, in inches: ticks and label."""

MAXIMUM_SERIES = 3
"""Most series in one panel; more series go into further panels."""

LINE_WIDTH = 1.5
MARKER_SIZE = 6.0
MARKER_RING_WIDTH = 1.5
HAIRLINE_WIDTH = 0.5
REFERENCE_WIDTH = 0.8
BAND_OPACITY = 0.10
LABEL_SIZE = 8.5
TICK_SIZE = 8.0
TITLE_SIZE = 9.0
SVG_HASH_SALT = "mixed-signals-validation"
"""Fixed salt of the SVG identifiers, so that equal data give equal files."""


@dataclasses.dataclass(frozen=True)
class FigureStyle:
    """The colours of one theme of the docs."""

    name: str
    """``"light"`` or ``"dark"``, the suffix of the file name."""
    background: str
    """Background of the figure and its axes."""
    ink: str
    """Primary ink, for titles and values."""
    secondary_ink: str
    """Secondary ink, for axis labels and legend text."""
    muted_ink: str
    """Muted ink, for tick labels."""
    grid: str
    """Gridlines."""
    axis: str
    """Axis lines and ticks."""
    series: tuple[str, str, str]
    """Series 1 (blue), 2 (orange) and 3 (aqua), in this fixed order."""
    reference: str
    """Reference lines: a truth, a threshold or the identity."""


LIGHT = FigureStyle(
    name="light",
    background="#fcfcfb",
    ink="#0b0b0b",
    secondary_ink="#52514e",
    muted_ink="#898781",
    grid="#e1e0d9",
    axis="#c3c2b7",
    series=("#2a78d6", "#eb6834", "#1baf7a"),
    reference="#898781",
)

DARK = FigureStyle(
    name="dark",
    background="#1a1a19",
    ink="#f0efec",
    secondary_ink="#c3c2b7",
    muted_ink="#898781",
    grid="#2c2c2a",
    axis="#383835",
    series=("#3987e5", "#d95926", "#199e70"),
    reference="#898781",
)

STYLES = (LIGHT, DARK)
"""Both themes, in the order in which the files are written."""


def style_context() -> AbstractContextManager[None]:
    """Return the matplotlib settings that every figure of the page is drawn under."""
    return matplotlib.rc_context(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
            # * Text stays text, so the browser draws the system sans font.
            "svg.fonttype": "none",
            "svg.hashsalt": SVG_HASH_SALT,
            "axes.unicode_minus": True,
        }
    )


def new_figure(
    style: FigureStyle,
    *,
    rows: int = 1,
    columns: int = 1,
    sharey: bool = False,
    extra_height: float = 0.0,
) -> tuple[MatplotlibFigure, list[Axes]]:
    """Return a figure with a grid of panels, and the panels row by row.

    ``extra_height`` adds inches to the figure, for legends outside the panels.
    """
    height = SINGLE_PANEL_HEIGHT if rows * columns == 1 else ROW_HEIGHT * rows
    height += extra_height
    figure = MatplotlibFigure(
        figsize=(FIGURE_WIDTH, height),
        facecolor=style.background,
        layout="constrained",
    )
    grid = figure.subplots(rows, columns, squeeze=False, sharey=sharey)
    return figure, [axes for row in grid for axes in row]


def _as_floats(values: ArrayLike) -> np.ndarray:
    return np.asarray(values, dtype=float)


def finish_panel(
    axes: Axes,
    style: FigureStyle,
    *,
    title: str | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    log_x: bool = False,
    log_y: bool = False,
) -> None:
    """Apply the axes style: no top or right line, hairline gridlines on y, ink."""
    axes.set_facecolor(style.background)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axes.spines[side].set_color(style.axis)
        axes.spines[side].set_linewidth(REFERENCE_WIDTH)
    axes.tick_params(
        axis="both",
        which="both",
        color=style.axis,
        labelcolor=style.muted_ink,
        labelsize=TICK_SIZE,
    )
    axes.set_axisbelow(True)
    axes.yaxis.grid(True, color=style.grid, linewidth=HAIRLINE_WIDTH, linestyle="-")
    axes.xaxis.grid(False)
    if log_x:
        axes.set_xscale("log")
        axes.xaxis.set_minor_formatter(NullFormatter())
    if log_y:
        axes.set_yscale("log")
        axes.yaxis.set_minor_formatter(NullFormatter())
    if title is not None:
        axes.set_title(title, color=style.ink, fontsize=TITLE_SIZE, loc="left")
    if x_label is not None:
        axes.set_xlabel(x_label, color=style.secondary_ink, fontsize=LABEL_SIZE)
    if y_label is not None:
        axes.set_ylabel(y_label, color=style.secondary_ink, fontsize=LABEL_SIZE)


def plot_series(
    axes: Axes,
    style: FigureStyle,
    index: int,
    x_values: ArrayLike,
    y_values: ArrayLike,
    *,
    label: str | None = None,
    errors: ArrayLike | None = None,
    markers: bool = True,
) -> None:
    """Draw series ``index`` (0 to 2) as a line, with markers and error bars."""
    if not 0 <= index < MAXIMUM_SERIES:
        raise ValueError(f"a panel has at most {MAXIMUM_SERIES} series, got {index}")
    colour = style.series[index]
    horizontal, vertical = _as_floats(x_values), _as_floats(y_values)
    if errors is not None:
        axes.errorbar(
            horizontal,
            vertical,
            yerr=_as_floats(errors),
            fmt="none",
            ecolor=colour,
            elinewidth=LINE_WIDTH,
            capsize=0,
            zorder=2,
        )
    axes.plot(
        horizontal,
        vertical,
        color=colour,
        linewidth=LINE_WIDTH,
        solid_capstyle="round",
        solid_joinstyle="round",
        marker="o" if markers else None,
        markersize=MARKER_SIZE,
        markerfacecolor=colour,
        markeredgecolor=style.background,
        markeredgewidth=MARKER_RING_WIDTH,
        label=label,
        zorder=3,
    )


def plot_steps(
    axes: Axes,
    style: FigureStyle,
    index: int,
    x_values: ArrayLike,
    y_values: ArrayLike,
    *,
    label: str | None = None,
) -> None:
    """Draw series ``index`` as a step line without markers."""
    if not 0 <= index < MAXIMUM_SERIES:
        raise ValueError(f"a panel has at most {MAXIMUM_SERIES} series, got {index}")
    axes.step(
        _as_floats(x_values),
        _as_floats(y_values),
        where="post",
        color=style.series[index],
        linewidth=LINE_WIDTH,
        solid_capstyle="round",
        solid_joinstyle="round",
        label=label,
        zorder=3,
    )


def plot_band(
    axes: Axes,
    style: FigureStyle,
    index: int,
    x_values: ArrayLike,
    lower: ArrayLike,
    upper: ArrayLike,
) -> None:
    """Draw an interval as the colour of series ``index`` at 10% opacity."""
    axes.fill_between(
        _as_floats(x_values),
        _as_floats(lower),
        _as_floats(upper),
        color=style.series[index],
        alpha=BAND_OPACITY,
        linewidth=0,
        zorder=1,
    )


def plot_reference(
    axes: Axes,
    style: FigureStyle,
    *,
    x_value: float | None = None,
    y_value: float | None = None,
    identity: bool = False,
) -> None:
    """Draw a thin solid reference line: vertical, horizontal or the identity."""
    colour, width = style.reference, REFERENCE_WIDTH
    if x_value is not None:
        axes.axvline(x_value, color=colour, linewidth=width, zorder=0)
    if y_value is not None:
        axes.axhline(y_value, color=colour, linewidth=width, zorder=0)
    if identity:
        axes.axline((0.0, 0.0), slope=1.0, color=colour, linewidth=width, zorder=0)


def plot_tolerance_band(axes: Axes, style: FigureStyle, half_width: float) -> None:
    """Shade the interval of a tolerance around zero in the reference ink."""
    axes.axhspan(
        -half_width,
        half_width,
        color=style.reference,
        alpha=BAND_OPACITY,
        linewidth=0,
        zorder=0,
    )


def add_figure_legend(
    figure: MatplotlibFigure, axes: Axes, style: FigureStyle, *, columns: int = 2
) -> None:
    """Add one legend below all panels, in the drawing order of ``axes``."""
    handles, labels = axes.get_legend_handles_labels()
    if len(handles) < 2:
        return
    figure.legend(
        handles,
        labels,
        loc="outside lower center",
        ncol=columns,
        frameon=False,
        labelcolor=style.secondary_ink,
        fontsize=TICK_SIZE,
        handlelength=1.6,
    )


def add_legend(
    axes: Axes,
    style: FigureStyle,
    *,
    location: LegendLocation = "best",
    outside: bool = False,
) -> None:
    """Add a legend when the panel has two or more labelled series.

    With ``outside`` the legend hangs below the axis label, clear of the data.
    """
    handles, labels = axes.get_legend_handles_labels()
    if len(handles) < 2:
        return
    drop = ScaledTranslation(0.0, -LEGEND_DROP, axes.figure.dpi_scale_trans)
    axes.legend(
        handles,
        labels,
        loc="upper center" if outside else location,
        bbox_to_anchor=(0.5, 0.0) if outside else None,
        bbox_transform=axes.transAxes + drop if outside else None,
        frameon=False,
        labelcolor=style.secondary_ink,
        fontsize=TICK_SIZE,
        handlelength=1.6,
    )


def hide_unused(axes_list: Sequence[Axes], used: int) -> None:
    """Hide the panels of a grid beyond the first ``used`` ones."""
    for axes in axes_list[used:]:
        axes.set_visible(False)


def panels_of(items: Sequence[object], size: int = MAXIMUM_SERIES) -> Iterator[range]:
    """Yield the index ranges that split ``items`` into groups of at most ``size``."""
    for start in range(0, len(items), size):
        yield range(start, min(start + size, len(items)))
