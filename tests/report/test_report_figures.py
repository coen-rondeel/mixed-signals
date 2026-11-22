"""Tests for the heatmap of effects, the null histograms and the epsilon curve."""

import json
import math
from collections.abc import Callable
from typing import Any

import numpy as np
import polars as pl
import pytest
from matplotlib.colors import to_rgb, to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, StepPatch
from scipy.stats import norm

from mixed_signals.core.result import Result
from mixed_signals.report._style import (
    BACKGROUND,
    CRITICAL,
    INK,
    MUTED_INK,
    REFERENCE,
    SECONDARY_INK,
    SEQUENTIAL_BLUE,
    SERIES_COLOURS,
)
from mixed_signals.report.figures import (
    MAXIMUM_BINS,
    effect_heatmap,
    epsilon_plot,
    null_histograms,
)
from mixed_signals.report.tables import with_effects

PARAMETERS = ("x0", "x1", "x2")
COMPARISON_SCHEMA = {
    "metric": pl.String,
    "estimator": pl.String,
    "direction": pl.String,
    "parameters": pl.String,
    "sigma_eff": pl.Float64,
    "r_eff": pl.Float64,
    "effect": pl.Float64,
    "threshold": pl.Float64,
    "resolved": pl.Boolean,
    "flag": pl.Boolean,
    "epsilon_factor": pl.Float64,
}


def comparison_of(*rows: dict[str, Any]) -> pl.DataFrame:
    """Return a comparison table of rows that name what they need."""
    defaults = {
        "metric": "mean_shift",
        "estimator": "empirical",
        "direction": None,
        "parameters": "x0",
        "sigma_eff": 0.0,
        "r_eff": None,
        "effect": 0.0,
        "threshold": 0.2,
        "resolved": False,
        "flag": False,
        "epsilon_factor": None,
    }
    return pl.DataFrame([{**defaults, **row} for row in rows], schema=COMPARISON_SCHEMA)


def xdata(line: Line2D) -> np.ndarray:
    """Return the x values of a line as an array."""
    return np.asarray(line.get_xdata())


def ydata(line: Line2D) -> np.ndarray:
    """Return the y values of a line as an array."""
    return np.asarray(line.get_ydata())


def steps(axes: Any) -> tuple[np.ndarray, np.ndarray]:
    """Return the heights and the edges of the histogram of a panel."""
    patch = axes.patches[0]
    assert isinstance(patch, StepPatch)
    values, edges, _ = patch.get_data()
    return np.asarray(values), np.asarray(edges)


def filled(figure: Any) -> list[Rectangle]:
    """Return the filled cells of the heatmap."""
    return [patch for patch in figure.axes[0].patches if patch.get_fill()]


def outlined(figure: Any) -> list[Rectangle]:
    """Return the frames of the failing cells of the heatmap."""
    return [patch for patch in figure.axes[0].patches if not patch.get_fill()]


def cell_of(patch: Rectangle) -> tuple[int, int]:
    """Return the (row, column) of a rectangle of the heatmap."""
    return int(patch.get_y()), int(patch.get_x())


def ramp(step: int) -> tuple[float, ...]:
    """Return the colour of a step of the blue ramp, as RGBA."""
    return to_rgba(SEQUENTIAL_BLUE[step])


def luminance(colour: Any) -> float:
    """Return the relative luminance of a colour by the definition of WCAG 2.2."""
    red, green, blue = (
        value / 12.92 if value <= 0.040_45 else ((value + 0.055) / 1.055) ** 2.4
        for value in to_rgb(colour)
    )
    return 0.2126 * red + 0.7152 * green + 0.0722 * blue


def contrast(first: Any, second: Any) -> float:
    """Return the contrast ratio of two colours by the definition of WCAG 2.2."""
    lighter, darker = sorted((luminance(first), luminance(second)), reverse=True)
    return (lighter + 0.05) / (darker + 0.05)


# * The heatmap


def test_the_heatmap_colours_follow_the_ramp_from_zero_to_two() -> None:
    """Effect over threshold 0, 1 and above 2 take the first, middle and last step."""
    comparison = comparison_of(
        {"parameters": "x0", "effect": 0.0, "sigma_eff": 0.0},
        {"parameters": "x1", "effect": 0.2, "sigma_eff": 0.2},
        {"parameters": "x2", "effect": 0.8, "sigma_eff": 0.8},
        {"metric": "ks", "parameters": "x0", "effect": -0.3, "sigma_eff": -0.3},
        {"metric": "ks", "parameters": "x1", "effect": 0.1, "sigma_eff": 0.1},
    )
    figure = effect_heatmap(comparison, PARAMETERS, title="a pair")
    assert figure is not None
    colours = {
        cell_of(patch): to_rgba(patch.get_facecolor()) for patch in filled(figure)
    }
    np.testing.assert_allclose(colours[(0, 0)], ramp(0), atol=0.01)
    np.testing.assert_allclose(colours[(0, 1)], ramp(6), atol=0.01)
    np.testing.assert_allclose(colours[(0, 2)], ramp(12), atol=0.01)
    np.testing.assert_allclose(colours[(1, 0)], ramp(0), atol=0.01)
    # * A half of the threshold sits a quarter along the ramp: step 250.
    np.testing.assert_allclose(colours[(1, 1)], ramp(3), atol=0.01)
    assert len(filled(figure)) == 5


def test_the_cell_text_bold_and_frames_follow_resolved_and_flag() -> None:
    """σ_eff to two decimals, r_eff to three; bold if resolved; a frame if flagged."""
    comparison = comparison_of(
        {"parameters": "x0", "effect": 0.0, "sigma_eff": 0.0},
        {"parameters": "x1", "effect": 0.2, "sigma_eff": 0.2, "resolved": True},
        {
            "parameters": "x2",
            "effect": 0.8,
            "sigma_eff": 0.8,
            "resolved": True,
            "flag": True,
        },
        {
            "metric": "width_ratio",
            "threshold": 0.14,
            "parameters": "x0",
            "effect": math.log(1.25),
            "sigma_eff": None,
            "r_eff": 1.25,
        },
    )
    figure = effect_heatmap(comparison, PARAMETERS, title="a pair")
    assert figure is not None
    texts = {
        (int(text.get_position()[1]), int(text.get_position()[0])): text
        for text in figure.axes[0].texts
    }
    assert texts[(0, 0)].get_text() == "0.00"
    assert texts[(0, 1)].get_text() == "0.20"
    assert texts[(1, 0)].get_text() == "1.250"
    assert [texts[key].get_fontweight() for key in sorted(texts)] == [
        "normal",
        "bold",
        "bold",
        "normal",
    ]
    # * Ink up to step 400, white on the darker fills of the ratios 4 and 1.6.
    assert to_rgb(texts[(0, 0)].get_color()) == pytest.approx(to_rgb(INK))
    assert to_rgb(texts[(0, 1)].get_color()) == pytest.approx(to_rgb(INK))
    assert to_rgb(texts[(0, 2)].get_color()) == pytest.approx((1.0, 1.0, 1.0))
    assert to_rgb(texts[(1, 0)].get_color()) == pytest.approx((1.0, 1.0, 1.0))
    frames = outlined(figure)
    assert [cell_of(frame) for frame in frames] == [(0, 2)]
    assert to_rgb(frames[0].get_edgecolor()) == pytest.approx(to_rgb(INK))


def test_the_text_has_the_larger_contrast_ratio_against_the_fill() -> None:
    """Ink at ratio 1 (step 400), white at 1.5 and above 2: the better of the two."""
    ratios = (0.0, 0.5, 1.0, 1.25, 1.5, 2.0, 3.0)
    comparison = comparison_of(
        *[
            {"metric": f"metric_{row}", "effect": 0.2 * ratio, "sigma_eff": 0.2 * ratio}
            for row, ratio in enumerate(ratios)
        ]
    )
    figure = effect_heatmap(comparison, PARAMETERS, title="a pair")
    assert figure is not None
    fills = {cell_of(patch)[0]: patch.get_facecolor() for patch in filled(figure)}
    texts = {int(text.get_position()[1]): text for text in figure.axes[0].texts}
    for row, ratio in enumerate(ratios):
        chosen = texts[row].get_color()
        other = "#ffffff" if to_rgb(chosen) == to_rgb(INK) else INK
        assert contrast(chosen, fills[row]) >= contrast(other, fills[row]), ratio
    assert to_rgb(texts[2].get_color()) == pytest.approx(to_rgb(INK))
    assert to_rgb(texts[4].get_color()) == pytest.approx((1.0, 1.0, 1.0))
    assert to_rgb(texts[5].get_color()) == pytest.approx((1.0, 1.0, 1.0))
    assert to_rgb(texts[6].get_color()) == pytest.approx((1.0, 1.0, 1.0))
    # * The two colours cross near step 450, which the ramp reaches at ratio 1.17.
    assert contrast(INK, ramp(6)) > contrast("#ffffff", ramp(6))
    assert contrast(INK, ramp(8)) < contrast("#ffffff", ramp(8))
    # * A cell without a fill takes ink, which has the larger contrast on the page.
    empty = effect_heatmap(
        comparison_of({"effect": math.nan, "sigma_eff": math.nan}),
        PARAMETERS,
        title="t",
    )
    assert empty is not None
    assert to_rgb(empty.axes[0].texts[0].get_color()) == pytest.approx(to_rgb(INK))
    assert contrast(INK, BACKGROUND) > contrast("#ffffff", BACKGROUND)


def test_a_nan_effect_reads_nan_and_a_missing_cell_stays_empty() -> None:
    """An undefined effect has text and no fill; a cell without a row has neither."""
    comparison = comparison_of(
        {"parameters": "x1", "effect": math.nan, "sigma_eff": math.nan},
        {"parameters": "x2", "effect": None, "sigma_eff": None},
        {"parameters": "x0", "effect": 0.4, "sigma_eff": 0.4, "threshold": None},
    )
    figure = effect_heatmap(comparison, PARAMETERS, title="a pair")
    assert figure is not None
    texts = [text.get_text() for text in figure.axes[0].texts]
    assert texts == ["NaN", "NaN", "0.40"]
    assert len(figure.axes[0].texts) == 3
    transparent = [to_rgba(patch.get_facecolor())[3] for patch in filled(figure)]
    assert transparent == [0.0, 0.0, 0.0]
    assert len(filled(figure)) == 3


def test_the_rows_and_columns_are_the_metrics_parameters_and_the_joint() -> None:
    """Directions and epsilon factors tell rows apart; joint rows end the row."""
    comparison = comparison_of(
        {"metric": "mean_shift", "parameters": "x0", "effect": 0.1},
        {"metric": "mean_shift", "parameters": "x2", "effect": 0.1},
        {
            "metric": "kl",
            "estimator": "gaussian",
            "direction": "a‖b",
            "parameters": "x0, x1, x2",
        },
        {
            "metric": "kl",
            "estimator": "gaussian",
            "direction": "b‖a",
            "parameters": "x0, x1, x2",
        },
        {"metric": "sinkhorn", "parameters": "x0, x1, x2", "epsilon_factor": 0.25},
        {"metric": "sinkhorn", "parameters": "x0, x1, x2", "epsilon_factor": 1.0},
    )
    figure = effect_heatmap(comparison, PARAMETERS, title="a title")
    assert figure is not None
    axes = figure.axes[0]
    # * No cell lies in the column of x1, so the column is left out.
    assert [label.get_text() for label in axes.get_xticklabels()] == [
        "x0",
        "x2",
        "joint",
    ]
    assert [label.get_text() for label in axes.get_yticklabels()] == [
        "mean_shift:empirical",
        "kl:gaussian a‖b",
        "kl:gaussian b‖a",
        "sinkhorn:empirical ε/ℓ² 0.25",
        "sinkhorn:empirical ε/ℓ² 1",
    ]
    cells = sorted(cell_of(patch) for patch in filled(figure))
    assert cells == [(0, 0), (0, 1), (1, 2), (2, 2), (3, 2), (4, 2)]
    assert axes.get_title(loc="left") == "a title"
    assert axes.get_ylim() == (5.0, 0.0)
    without = effect_heatmap(comparison.head(2), PARAMETERS, title="t")
    assert without is not None
    assert [label.get_text() for label in without.axes[0].get_xticklabels()] == [
        "x0",
        "x2",
    ]
    only_joint = effect_heatmap(comparison.tail(2), PARAMETERS, title="t")
    assert only_joint is not None
    assert [label.get_text() for label in only_joint.axes[0].get_xticklabels()] == [
        "joint"
    ]
    assert sorted(cell_of(patch) for patch in filled(only_joint)) == [(0, 0), (1, 0)]


def test_the_labels_of_rows_and_columns_are_in_the_secondary_ink() -> None:
    """They name the cells, as axis labels name axes, so they are not muted."""
    figure = effect_heatmap(
        comparison_of({"parameters": "x0"}, {"metric": "ks", "parameters": "x1"}),
        PARAMETERS,
        title="t",
    )
    assert figure is not None
    axes = figure.axes[0]
    assert to_rgb(axes.get_yticklabels()[0].get_color()) == pytest.approx(
        to_rgb(SECONDARY_INK)
    )
    assert to_rgb(axes.get_xticklabels()[0].get_color()) == pytest.approx(
        to_rgb(SECONDARY_INK)
    )
    assert to_rgb(SECONDARY_INK) != to_rgb(MUTED_INK)


def test_the_colour_bar_and_the_size() -> None:
    """The bar has its label and ticks, and the size grows with the cells."""
    small = effect_heatmap(comparison_of({}), PARAMETERS, title="t")
    large = effect_heatmap(
        comparison_of(*[{"metric": f"metric_{row}"} for row in range(12)]),
        PARAMETERS,
        title="t",
    )
    assert small is not None
    assert large is not None
    bar = small.axes[1]
    assert bar.get_ylabel() == "effect / threshold"
    assert [label.get_text() for label in bar.get_yticklabels()] == [
        "0",
        "0.5",
        "1",
        "1.5",
        "2+",
    ]
    assert large.get_size_inches()[1] > small.get_size_inches()[1]
    names = ("x0", "x1", "x2", "x3", "x4")
    wide = effect_heatmap(
        comparison_of(*[{"parameters": name} for name in names]), names, title="t"
    )
    assert wide is not None
    assert wide.get_size_inches()[0] > small.get_size_inches()[0]
    assert effect_heatmap(comparison_of(), PARAMETERS, title="t") is None


# * The null histograms


def null_rows(result_of: Callable[..., Result], *rows: dict[str, Any]) -> pl.DataFrame:
    """Return rows with effects, as the histograms take them, at the level 2."""
    base = {
        "pair": ["a", "b"],
        "metric": "mean_shift",
        "space": "zscore(reference)",
        "threshold": 0.2,
        "sigma_sig": 1.0,
        "flag": False,
        "null_mean": 0.0,
        "null_draws": [-0.2, -0.1, 0.0, 0.1, 0.2],
        "value": 0.1,
    }
    result = result_of(*({**base, **row} for row in rows))
    return with_effects(result.table, {"compare": 2.0, "compare_signals": 2.0})


def titles(figure: Any) -> list[str]:
    """Return the titles of the visible panels, in panel order."""
    return [axes.get_title(loc="left") for axes in figure.axes if axes.get_visible()]


def test_a_flagged_row_comes_first_and_is_framed(
    result_of: Callable[..., Result],
) -> None:
    """Rows that flag lead, each group in table order, with critical spines."""
    rows = null_rows(
        result_of,
        {"parameters": ["x0"]},
        {"parameters": ["x1"], "flag": True, "sigma_sig": 2.31},
        {"parameters": ["x2"]},
        {"parameters": ["x3"], "flag": True, "sigma_sig": 3.0},
    )
    figure = null_histograms(rows, title="nulls")
    assert figure is not None
    assert [title.splitlines()[1].split()[0] for title in titles(figure)] == [
        "x1",
        "x3",
        "x0",
        "x2",
    ]
    assert titles(figure)[0].endswith("flags")
    assert "flags" not in titles(figure)[2]
    for position, axes in enumerate(figure.axes[:4]):
        spine = axes.spines["bottom"]
        if position < 2:
            assert to_rgb(spine.get_edgecolor()) == pytest.approx(to_rgb(CRITICAL))
            assert all(axes.spines[side].get_visible() for side in ("top", "right"))
        else:
            assert to_rgb(spine.get_edgecolor()) != pytest.approx(to_rgb(CRITICAL))
            assert not axes.spines["top"].get_visible()
    assert figure.get_suptitle() == "nulls"


def test_a_panel_holds_the_value_the_null_mean_and_the_draws(
    result_of: Callable[..., Result],
) -> None:
    """The value is a solid ink line, the null mean a thin one; limits hold both."""
    draws = list(np.random.default_rng(0).normal(size=300))
    rows = null_rows(
        result_of,
        {"value": 50.0, "null_draws": draws, "null_mean": 0.05, "sigma_sig": 12.0},
    )
    figure = null_histograms(rows, title="t")
    assert figure is not None
    axes = figure.axes[0]
    lines = {to_rgb(line.get_color()): line for line in axes.lines}
    assert set(lines) == {to_rgb(INK), to_rgb(MUTED_INK)}
    assert list(xdata(lines[to_rgb(INK)])) == [50.0, 50.0]
    assert list(xdata(lines[to_rgb(MUTED_INK)])) == [0.05, 0.05]
    assert lines[to_rgb(INK)].get_linewidth() > lines[to_rgb(MUTED_INK)].get_linewidth()
    low, high = axes.get_xlim()
    assert low <= min(draws)
    assert high >= 50.0
    assert len(axes.get_yticks()) == 0
    values, edges = steps(axes)
    assert values.sum() == 300
    assert to_rgb(axes.patches[0].get_facecolor()) == pytest.approx(
        to_rgb(SEQUENTIAL_BLUE[3])
    )
    assert edges[0] <= min(draws)
    assert edges[-1] >= max(draws)


def test_the_titles_name_the_metric_the_parameters_and_the_significance(
    result_of: Callable[..., Result],
) -> None:
    """Direction, data or chart, the parameters, sigma_sig and a note for a tail fit."""
    warning = [{"category": "ExtrapolationWarning", "message": "tail"}]
    rows = null_rows(
        result_of,
        {
            "metric": "kl",
            "estimator": "gaussian",
            "parameters": ["x0", "x1"],
            "sigma_sig": 2.5,
        },
        {"metric": "energy", "space": "shift | data(white)", "warnings": warning},
        {"metric": "energy", "space": "chart(fisher)", "sigma_sig": 0.5},
        {"metric": "w1", "sigma_sig": None, "threshold": None},
    )
    figure = null_histograms(rows, title="t")
    assert figure is not None
    assert titles(figure) == [
        "kl:gaussian a‖b\nx0, x1  σ_sig 2.50",
        "energy:empirical data\nx0  σ_sig 1.00 (tail fit)",
        "energy:empirical chart\nx0  σ_sig 0.50",
        "w1:empirical\nx0",
    ]


def test_one_or_two_parameters_are_named_and_more_are_joint(
    result_of: Callable[..., Result],
) -> None:
    """The three Spearman rows of a posterior differ by their pair of parameters."""
    rows = null_rows(
        result_of,
        {"parameters": ["x0"]},
        {"parameters": ["x0", "x1"]},
        {"parameters": ["x0", "x2"]},
        {"parameters": ["x0", "x1", "x2"]},
        {"parameters": ["a", "b", "c", "d"]},
    )
    figure = null_histograms(rows, title="t")
    assert figure is not None
    names = [title.splitlines()[1].split("  ")[0] for title in titles(figure)]
    assert names == ["x0", "x0, x1", "x0, x2", "joint", "joint"]


@pytest.mark.parametrize(
    ("sigma", "text"),
    [
        (2.314, "2.31"),
        (-0.7124, "−0.71"),
        (0.0, "0.00"),
        (math.inf, "∞"),
        (-math.inf, "−∞"),
        (math.nan, "NaN"),
    ],
)
def test_sigma_sig_is_written_as_the_axes_write_numbers(
    result_of: Callable[..., Result], sigma: float, text: str
) -> None:
    """Two decimals and the minus sign U+2212; infinity and NaN in words or symbols."""
    rows = null_rows(result_of, {"sigma_sig": sigma})
    figure = null_histograms(rows, title="t")
    assert figure is not None
    assert titles(figure)[0].endswith(f"σ_sig {text}")
    assert "-" not in titles(figure)[0].split("σ_sig")[1]


def test_many_draws_make_at_most_forty_bins(result_of: Callable[..., Result]) -> None:
    """The automatic rule would give more bins than the panel can show."""
    draws = list(np.random.default_rng(1).normal(size=20_000))
    figure = null_histograms(null_rows(result_of, {"null_draws": draws}), title="t")
    assert figure is not None
    edges = steps(figure.axes[0])[1]
    assert len(edges) - 1 == MAXIMUM_BINS
    few = null_histograms(null_rows(result_of, {"null_draws": draws[:30]}), title="t")
    assert few is not None
    assert len(steps(few.axes[0])[1]) - 1 < MAXIMUM_BINS


def test_rows_without_a_finite_draw_leave_no_panel(
    result_of: Callable[..., Result],
) -> None:
    """Only finite draws count; with none anywhere, there is no figure."""
    nan = math.nan
    rows = null_rows(
        result_of,
        {"null_draws": [nan, nan]},
        {"null_draws": None},
        {"null_draws": []},
    )
    assert null_histograms(rows, title="t") is None
    mixed = null_rows(
        result_of,
        {"null_draws": [nan, 0.1, 0.2, math.inf]},
        {"null_draws": None},
    )
    figure = null_histograms(mixed, title="t")
    assert figure is not None
    assert steps(figure.axes[0])[0].sum() == 2
    assert null_histograms(mixed.head(0), title="t") is None


def test_unused_panels_are_hidden_and_awkward_values_leave_no_line(
    result_of: Callable[..., Result],
) -> None:
    """Four columns; no value line for NaN or inf; no mean line for a missing mean."""
    rows = null_rows(
        result_of,
        {"value": math.nan},
        {"value": math.inf, "null_mean": None},
        {"null_draws": [1.0, 1.0, 1.0], "value": 1.0, "null_mean": math.nan},
        {},
        {},
    )
    figure = null_histograms(rows, title="t")
    assert figure is not None
    assert len(figure.axes) == 8
    assert [axes.get_visible() for axes in figure.axes] == [True] * 5 + [False] * 3
    assert len(figure.axes[0].lines) == 1
    assert len(figure.axes[1].lines) == 0
    assert len(figure.axes[2].lines) == 1
    low, high = figure.axes[2].get_xlim()
    assert (low, high) == (0.5, 1.5)


# * The epsilon curve


def sinkhorn_rows(
    result_of: Callable[..., Result], factors: list[float], **changes: Any
) -> tuple[pl.DataFrame, dict[float, np.ndarray]]:
    """Return sinkhorn rows at the factors, in that order, and the draws of each."""
    generator = np.random.default_rng(2)
    draws = {factor: generator.gamma(2.0, 0.1 / factor, size=400) for factor in factors}
    rows = [
        {
            "pair": ["a", "b"],
            "metric": "sinkhorn",
            "space": "whiten(reference)",
            "estimator_settings": json.dumps({"epsilon_factor": factor}),
            "value": 0.5 / factor,
            "sigma_eff": 0.3 + factor,
            "threshold": 0.2,
            "null_draws": list(draws[factor]),
            **changes,
        }
        for factor in factors
    ]
    return result_of(*rows).table, draws


def test_the_band_is_the_quantile_range_of_the_null_at_the_level(
    result_of: Callable[..., Result],
) -> None:
    """The band runs from the Phi(-L) to the Phi(L) quantile of each row's draws."""
    rows, draws = sinkhorn_rows(result_of, [1.0, 0.25, 2.0, 0.5])
    level = 1.5
    figure = epsilon_plot(rows, level=level)
    assert figure is not None
    value_axes, effect_axes = figure.axes
    factors = sorted(draws)
    low, high = norm.cdf(-level), norm.cdf(level)
    expected_low = [np.quantile(draws[factor], low) for factor in factors]
    expected_high = [np.quantile(draws[factor], high) for factor in factors]
    band = np.asarray(value_axes.collections[0].get_paths()[0].vertices)
    for factor, lower, upper in zip(factors, expected_low, expected_high, strict=True):
        heights = band[np.isclose(band[:, 0], factor), 1]
        assert heights.min() == pytest.approx(lower, rel=1e-9)
        assert heights.max() == pytest.approx(upper, rel=1e-9)
    assert value_axes.collections[0].get_alpha() == pytest.approx(0.10)
    lines = value_axes.lines
    series, median = lines[0], lines[1]
    assert list(xdata(series)) == factors
    assert list(ydata(series)) == [0.5 / factor for factor in factors]
    assert to_rgb(series.get_color()) == pytest.approx(to_rgb(SERIES_COLOURS[0]))
    np.testing.assert_allclose(
        ydata(median), [np.median(draws[factor]) for factor in factors]
    )
    assert value_axes.get_xscale() == "log"
    assert effect_axes.get_xscale() == "log"
    assert value_axes.get_xlabel() == r"$\varepsilon / \ell^2$"
    assert value_axes.get_ylabel() == r"$S_\varepsilon$"
    assert effect_axes.get_ylabel() == r"$\sigma_\mathrm{eff}$"
    assert list(ydata(effect_axes.lines[0])) == [0.3 + factor for factor in factors]
    reference = effect_axes.lines[1]
    assert list(ydata(reference)) == [0.2, 0.2]
    assert to_rgb(reference.get_color()) == pytest.approx(to_rgb(REFERENCE))
    assert [label.get_text() for label in value_axes.get_xticklabels()] == [
        "0.25",
        "0.5",
        "1",
        "2",
    ]
    assert value_axes.yaxis.get_gridlines()[0].get_visible()


def test_the_curve_needs_two_factors_and_leaves_out_rows_without_draws(
    result_of: Callable[..., Result],
) -> None:
    """None below two factors, for other metrics, and for settings without one."""
    one, _ = sinkhorn_rows(result_of, [0.25, 0.25])
    assert epsilon_plot(one, level=2.0) is None
    none = result_of(
        {"pair": ["a", "b"], "metric": "sinkhorn", "estimator_settings": "{}"}
    )
    assert epsilon_plot(none.table, level=2.0) is None
    other, _ = sinkhorn_rows(result_of, [0.25, 1.0], metric="energy")
    assert epsilon_plot(other, level=2.0) is None
    rows, _ = sinkhorn_rows(result_of, [0.25, 1.0, 2.0])
    without_draws = rows.with_columns(
        pl.when(pl.col("estimator_settings").str.contains("0.25"))
        .then(None)
        .otherwise(pl.col("null_draws"))
        .alias("null_draws")
    )
    figure = epsilon_plot(without_draws, level=2.0)
    assert figure is not None
    median = ydata(figure.axes[0].lines[1])
    assert math.isnan(median[0])
    assert not math.isnan(median[1])
    no_threshold = rows.with_columns(pl.lit(None, dtype=pl.Float64).alias("threshold"))
    bare = epsilon_plot(no_threshold, level=2.0)
    assert bare is not None
    assert len(bare.axes[1].lines) == 1
