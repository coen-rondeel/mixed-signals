"""Tests for the palette, the style mapping and the figure helper of the report."""

import matplotlib
import matplotlib.style
from matplotlib import pyplot
from matplotlib.colors import to_hex

from mixed_signals.report import _style


def test_the_series_colours_are_the_eight_slots_of_the_reference_palette() -> None:
    """The first three are the validation page's series, in the fixed order."""
    assert _style.SERIES_COLOURS == (
        "#2a78d6",
        "#eb6834",
        "#1baf7a",
        "#eda100",
        "#e87ba4",
        "#008300",
        "#6250d6",
        "#e34948",
    )


def test_the_blue_ramp_has_the_thirteen_steps_from_100_to_700() -> None:
    """Step 100 is the lightest, step 700 the darkest, in steps of 50."""
    assert len(_style.SEQUENTIAL_BLUE) == 13
    assert _style.SEQUENTIAL_BLUE[0] == "#cde2fb"
    assert _style.SEQUENTIAL_BLUE[3] == "#86b6ef"
    assert _style.SEQUENTIAL_BLUE[6] == "#3987e5"
    assert _style.SEQUENTIAL_BLUE[-1] == "#0d366b"


def test_the_style_turns_off_latex_and_follows_the_validation_page() -> None:
    """LaTeX is off, because CI has none; sizes and widths are those of the page."""
    style = _style.REPORT_STYLE
    assert style["text.usetex"] is False
    assert style["axes.labelsize"] == 8.5
    assert style["xtick.labelsize"] == 8.0
    assert style["axes.titlesize"] == 9.0
    assert style["lines.linewidth"] == 1.5
    assert style["axes.unicode_minus"] is True
    assert style["axes.grid"] is False
    assert style["axes.spines.top"] is False
    assert style["axes.spines.right"] is False
    assert style["xtick.direction"] == "out"
    assert style["figure.facecolor"] == "#fcfcfb"
    assert style["text.color"] == "#0b0b0b"
    assert style["font.sans-serif"][-1] == "DejaVu Sans"


def test_matplotlib_accepts_the_style_without_a_word() -> None:
    """Every key is a style setting, so cantuccio can hand the mapping to matplotlib."""
    before = matplotlib.rcParams["text.usetex"]
    # ? The stubs type the style as a dict, while matplotlib takes any mapping.
    with matplotlib.style.context(_style.REPORT_STYLE):  # type: ignore[arg-type]
        assert matplotlib.rcParams["text.usetex"] is False
        assert matplotlib.rcParams["axes.facecolor"] == "#fcfcfb"
    assert matplotlib.rcParams["text.usetex"] == before


def test_the_context_restores_the_settings_of_the_session() -> None:
    """A figure function leaves the global settings as it found them."""
    before = matplotlib.rcParams["axes.labelsize"]
    with _style.style_context():
        assert matplotlib.rcParams["axes.labelsize"] == 8.5
    assert matplotlib.rcParams["axes.labelsize"] == before


def test_a_new_figure_has_the_style_and_no_pyplot_registry() -> None:
    """Size, dpi and face follow the arguments and the style; pyplot never sees it."""
    figure = _style.new_figure(3.0, 2.0)
    assert tuple(figure.get_size_inches()) == (3.0, 2.0)
    assert figure.dpi == 100
    assert to_hex(figure.get_facecolor()) == "#fcfcfb"
    assert figure.get_layout_engine() is None
    sized = _style.new_figure(3.0, 2.0, dpi=50.0, layout="constrained")
    assert sized.dpi == 50.0
    assert sized.get_layout_engine() is not None
    assert pyplot.get_fignums() == []
