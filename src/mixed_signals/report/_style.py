"""Colours, fonts and matplotlib settings that every figure of the report shares.

The palette, the inks and the fonts follow the figures of the validation page in
its light theme (``scripts/zoo_validation/figures.py``): an opaque background,
hairline gridlines on the value axis only, a thin solid reference line, and text
in neutral ink, never in a series colour. The report extends the three series
colours of that page to eight, with the fourth to eighth slots of the same
reference palette. The report has no dark theme.

Series
    Posterior k of a run, in the order of its config, always takes slot k of
    :data:`SERIES_COLOURS`, in every figure.
Status
    :data:`CRITICAL` frames a flagged panel, always beside the word "flags" in
    its title, so that colour never carries the meaning alone.
Settings
    :data:`REPORT_STYLE` holds the matplotlib settings that every figure is
    drawn under. It also goes to ``cantuccio.cornerplot`` as its style, because
    the style of cantuccio switches on LaTeX, which a machine without LaTeX
    cannot draw.
"""

import contextlib
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, Final, Literal

import matplotlib
from matplotlib.figure import Figure

SERIES_COLOURS: Final = (
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#e87ba4",  # magenta
    "#008300",  # green
    "#6250d6",  # violet
    "#e34948",  # red
)
"""The eight series colours of the light theme, in the order of their slots."""

BACKGROUND: Final = "#fcfcfb"
INK: Final = "#0b0b0b"
SECONDARY_INK: Final = "#52514e"
MUTED_INK: Final = "#898781"
GRID: Final = "#e1e0d9"
AXIS: Final = "#c3c2b7"
REFERENCE: Final = "#898781"
"""Colour of reference lines: a bound, a threshold or a null mean."""
CRITICAL: Final = "#d03b3b"
"""Status colour of the frame of a flagged panel."""

SEQUENTIAL_BLUE: Final = (
    "#cde2fb",  # step 100
    "#b7d3f6",  # step 150
    "#9ec5f4",  # step 200
    "#86b6ef",  # step 250
    "#6da7ec",  # step 300
    "#5598e7",  # step 350
    "#3987e5",  # step 400
    "#2a78d6",  # step 450
    "#256abf",  # step 500
    "#1c5cab",  # step 550
    "#184f95",  # step 600
    "#104281",  # step 650
    "#0d366b",  # step 700
)
"""The blue ramp of the reference palette, steps 100 to 700 in steps of 50."""

LINE_WIDTH: Final = 1.5
CONTOUR_WIDTH: Final = 1.0
HAIRLINE_WIDTH: Final = 0.5
REFERENCE_WIDTH: Final = 0.8
BAND_OPACITY: Final = 0.10
LABEL_SIZE: Final = 8.5
TICK_SIZE: Final = 8.0
TITLE_SIZE: Final = 9.0
PANEL_TITLE_SIZE: Final = 7.0

# * Fonts, sizes and line widths follow scripts/zoo_validation/figures.py, light theme.
REPORT_STYLE: Final[Mapping[str, Any]] = MappingProxyType(
    {
        # ! cantuccio's own style turns LaTeX on, and CI has none.
        "text.usetex": False,
        "text.color": INK,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": TICK_SIZE,
        "axes.unicode_minus": True,
        "figure.dpi": 100,
        "figure.facecolor": BACKGROUND,
        "savefig.facecolor": BACKGROUND,
        "axes.facecolor": BACKGROUND,
        "axes.edgecolor": AXIS,
        "axes.linewidth": REFERENCE_WIDTH,
        "axes.labelcolor": SECONDARY_INK,
        "axes.labelsize": LABEL_SIZE,
        "axes.titlecolor": INK,
        "axes.titlesize": TITLE_SIZE,
        "axes.grid": False,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": AXIS,
        "ytick.color": AXIS,
        "xtick.labelcolor": MUTED_INK,
        "ytick.labelcolor": MUTED_INK,
        "xtick.labelsize": TICK_SIZE,
        "ytick.labelsize": TICK_SIZE,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.top": False,
        "ytick.right": False,
        "grid.color": GRID,
        "grid.linewidth": HAIRLINE_WIDTH,
        "grid.linestyle": "-",
        "lines.linewidth": LINE_WIDTH,
        "legend.fontsize": LABEL_SIZE,
        "legend.frameon": False,
    }
)
"""The matplotlib settings that every figure of the report is drawn under."""


def style_context() -> contextlib.AbstractContextManager[None]:
    """Return the context in which every figure and its text are created.

    Matplotlib reads most settings when it creates an artist, so a figure
    function creates its figure, axes and text inside this context.
    """
    # * rc_context types its keys as the literal names of the settings.
    settings: dict[Any, Any] = dict(REPORT_STYLE)
    return matplotlib.rc_context(settings)


def new_figure(
    width: float,
    height: float,
    *,
    dpi: float | None = None,
    layout: Literal["constrained", "compressed", "tight"] | None = None,
) -> Figure:
    """Return a new figure under the report style, not registered with pyplot.

    Args:
        width: Width in inches.
        height: Height in inches.
        dpi: Dots per inch; the dpi of the style when ``None``.
        layout: A matplotlib layout engine, such as ``"constrained"``; none
            when ``None``.

    Returns:
        The empty figure.
    """
    with style_context():
        return Figure(figsize=(width, height), dpi=dpi, layout=layout)
