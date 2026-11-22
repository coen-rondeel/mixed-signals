"""Tests for the corner plots: factors, subsamples, estimates, ellipses, bounds."""

import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import cantuccio
import jax
import jax.numpy as jnp
import numpy as np
import polars as pl
import pytest
from cantuccio.kde import hdi_levels
from matplotlib.colors import to_rgb
from matplotlib.figure import Figure
from scipy.stats import chi2

from mixed_signals import prng, zoo
from mixed_signals.core.result import Result
from mixed_signals.core.samples import (
    SampleSet,
    circular_mean,
    kish_effective_sample_size,
)
from mixed_signals.core.space import Parameter, ParamSpace
from mixed_signals.core.topology import Topology
from mixed_signals.estimators import KernelDensity, KernelDensitySettings
from mixed_signals.metrics.marginal import shared_bandwidth_factor
from mixed_signals.report import corner
from mixed_signals.report._style import REFERENCE, SERIES_COLOURS
from mixed_signals.report.corner import (
    CONTOUR_LEVELS,
    bounded_parameters,
    corner_plot,
    overview_factor,
    pair_factor,
    plot_subsample,
    row_bandwidth,
    silverman_factor,
)
from mixed_signals.report.folder import read_run_folder

COLOURS = SERIES_COLOURS[:2]
SAMPLES = 500


def posterior_sets(
    case: zoo.ZooCase, key: int = 0, size: int = SAMPLES
) -> list[tuple[str, SampleSet]]:
    """Return the posteriors of a zoo case as sets of one problem and one posterior."""
    draws = case.sample(jax.random.key(key), size)
    return [
        (name, draws.select(problems=0, posteriors=position))
        for position, name in enumerate(case.names)
    ]


def draw(
    sets: list[tuple[str, SampleSet]], space: ParamSpace, **changes: Any
) -> Figure:
    """Draw a corner plot of sets at their overview factor, with two colours."""
    settings: dict[str, Any] = {
        "factor": overview_factor([sample_set for _, sample_set in sets]),
        "colours": COLOURS[: len(sets)],
        "title": "a title",
    }
    return corner_plot(sets, space, **{**settings, **changes})


def density_lines(panel: Any) -> list[Any]:
    """Return the lines of a diagonal panel that hold a density, not a vertical mark."""
    return [line for line in panel.lines if len(line.get_xdata()) > 2]


def kernel_density(
    sample_set: SampleSet, positions: list[int], space: ParamSpace, factor: float
) -> KernelDensity:
    """Fit the package's estimate to some parameters of one posterior."""
    sub_space = ParamSpace(
        parameters=tuple(space.parameters[position] for position in positions)
    )
    return KernelDensity.fit(
        sample_set.samples[0, 0][:, positions],
        sample_set.log_weights[0, 0],
        sub_space,
        KernelDensitySettings(bandwidth=factor),
    )


def weighted_set(size: int, zero_weights: int, key: int = 0) -> SampleSet:
    """Return a set whose first samples have no weight, of one problem and posterior."""
    draws = jax.random.normal(jax.random.key(key), (size, 2))
    log_weights = jnp.zeros(size).at[:zero_weights].set(-jnp.inf)
    return SampleSet(draws, log_weights=log_weights)


# * Factors


def test_silverman_factor_follows_the_kish_size() -> None:
    """The factor is (0.75 n) to the power minus one fifth, with n the Kish size."""
    equal = SampleSet(np.zeros((400, 1)) + np.arange(400)[:, None])
    assert silverman_factor(equal.log_weights[0, 0]) == pytest.approx(
        (0.75 * 400) ** -0.2
    )
    log_weights = jnp.log(jnp.arange(1.0, 401.0))
    kish = float(kish_effective_sample_size(log_weights))
    assert silverman_factor(log_weights) == pytest.approx((0.75 * kish) ** -0.2)
    shared = shared_bandwidth_factor(log_weights, log_weights)
    assert silverman_factor(log_weights) == pytest.approx(float(shared), rel=1e-12)


def test_the_overview_factor_is_that_of_the_smallest_set() -> None:
    """No posterior of the overview is undersmoothed."""
    big, small = weighted_set(2_000, 0), weighted_set(300, 0)
    skewed = weighted_set(1_000, 400)
    expected = silverman_factor(small.log_weights[0, 0])
    assert overview_factor([big, small]) == pytest.approx(expected)
    smallest = min(
        float(kish_effective_sample_size(sample_set.log_weights[0, 0]))
        for sample_set in (big, skewed)
    )
    assert overview_factor([big, skewed]) == pytest.approx((0.75 * smallest) ** -0.2)


def test_the_pair_factor_is_the_bandwidth_of_the_rows(report_run_folder: Path) -> None:
    """The js and hellinger rows of the estimator kde fix it, else the shared one."""
    folder = read_run_folder(report_run_folder)
    table = folder.result.table
    sets = folder.samples["input"].sets
    first, second = sets["mcmc"], sets["table"]
    rows = table.filter(
        (pl.col("metric") == "js")
        & (pl.col("estimator") == "kde")
        & (pl.col("pair").list.len() == 2)
    )
    reported = float(rows["estimator_settings"].str.json_path_match("$.bandwidth")[0])
    assert row_bandwidth(table, 0, ("mcmc", "table")) == reported
    assert row_bandwidth(table, 0, ("table", "mcmc")) == reported
    assert pair_factor(table, 0, ("mcmc", "table"), first, second) == reported
    # * Without those rows the shared factor of the two sets applies.
    kept = table.filter(~pl.col("metric").is_in(["js", "hellinger"]))
    assert row_bandwidth(kept, 0, ("mcmc", "table")) is None
    shared = shared_bandwidth_factor(first.log_weights[0, 0], second.log_weights[0, 0])
    assert pair_factor(kept, 0, ("mcmc", "table"), first, second) == pytest.approx(
        float(shared)
    )
    assert reported == pytest.approx(float(shared), rel=0.3)


def test_rows_without_a_bandwidth_are_skipped(result_of: Callable[..., Result]) -> None:
    """A row that did not run has no settings; the next row that has one decides."""
    base = {"pair": ["a", "b"], "estimator": "kde", "space": "zscore(reference)"}
    result = result_of(
        {**base, "metric": "js", "estimator_settings": "{}"},
        {**base, "metric": "hellinger", "estimator_settings": '{"bandwidth":0.31}'},
        {**base, "metric": "js", "estimator_settings": '{"bandwidth":0.5}'},
        {**base, "metric": "ks", "estimator_settings": '{"bandwidth":0.9}'},
    )
    assert row_bandwidth(result.table, 0, ("a", "b")) == 0.31
    only_empty = result_of({**base, "metric": "js", "estimator_settings": "{}"})
    assert row_bandwidth(only_empty.table, 0, ("a", "b")) is None


# * The subsample


def test_a_set_at_or_below_the_cap_is_returned_unchanged() -> None:
    """The cap acts only above it."""
    sample_set = weighted_set(100, 30)
    key = prng.root_key(1)
    assert plot_subsample(sample_set, 100, key) is sample_set
    assert plot_subsample(sample_set, 500, key) is sample_set


def test_above_the_cap_a_subsample_is_drawn_without_repeats_for_a_key() -> None:
    """Uniform, without replacement, in the original order, deterministic for a key."""
    draws = jnp.arange(1_000.0)[:, None] * jnp.array([[1.0, 0.0]]) + jnp.array(
        [[0.0, 1.0]]
    )
    sample_set = SampleSet(draws, log_weights=jnp.log(1 + jnp.arange(1_000.0) % 3))
    key = prng.root_key(2)
    taken = plot_subsample(sample_set, 100, key)
    assert taken.shape.sample == 100
    positions = np.asarray(taken.samples[0, 0][:, 0]).astype(int)
    assert len(set(positions)) == 100
    assert (np.diff(positions) > 0).all()
    np.testing.assert_array_equal(
        np.asarray(taken.log_weights[0, 0]),
        np.asarray(sample_set.log_weights[0, 0])[positions],
    )
    assert taken.n_eff is None
    again = plot_subsample(sample_set, 100, key)
    np.testing.assert_array_equal(taken.samples, again.samples)
    other = plot_subsample(sample_set, 100, prng.root_key(3))
    assert not np.array_equal(taken.samples, other.samples)
    # ? Uniform draws keep the mean of a linear ramp near its centre.
    assert abs(float(positions.mean()) - 499.5) < 100


def test_samples_without_weight_leave_before_the_draw() -> None:
    """With few samples of weight, all of them stay; with many, none of zero weight."""
    key = prng.root_key(4)
    sparse = weighted_set(200, 150)
    kept = plot_subsample(sparse, 100, key)
    assert kept.shape.sample == 50
    assert bool(jnp.all(kept.log_weights > -jnp.inf))
    np.testing.assert_array_equal(kept.samples[0, 0], sparse.samples[0, 0][150:])
    mostly = weighted_set(400, 100, key=1)
    drawn = plot_subsample(mostly, 100, key)
    assert drawn.shape.sample == 100
    assert bool(jnp.all(drawn.log_weights > -jnp.inf))


# * The plot


def test_a_diagonal_panel_is_the_kernel_estimate_of_the_package() -> None:
    """Away from bounds a diagonal is the estimate of the rows, to the interpolation."""
    case = zoo.shifted_gaussians(1.0, param_count=2)
    sets = posterior_sets(case)
    figure = draw(sets, case.space)
    factor = overview_factor([sample_set for _, sample_set in sets])
    panel = figure.axes[0]
    lines = density_lines(panel)
    assert len(lines) == len(sets)
    for line, (_, sample_set) in zip(lines, sets, strict=True):
        grid, density = np.asarray(line.get_xdata()), np.asarray(line.get_ydata())
        estimate = kernel_density(sample_set, [0], case.space, factor)
        expected = np.exp(np.asarray(estimate.log_density(np.asarray(grid)[:, None])))
        # * cantuccio interpolates its estimate from a padded grid onto the panel grid.
        np.testing.assert_allclose(
            density, expected, rtol=1e-3, atol=1e-5 * expected.max()
        )
        assert line.get_linewidth() == 1.5
    assert [to_rgb(line.get_color()) for line in lines] == [
        to_rgb(colour) for colour in COLOURS
    ]


def test_an_interval_diagonal_is_the_corrected_estimate() -> None:
    """A bound within reach redraws the diagonal with the Jones-Foster estimate."""
    case = zoo.railed_gaussians([0.3, 1.0], param_count=2)
    sets = posterior_sets(case)
    figure = draw(sets, case.space)
    factor = overview_factor([sample_set for _, sample_set in sets])
    panel = figure.axes[0]
    lines = density_lines(panel)
    assert len(lines) == 2
    assert panel.get_ylim()[0] == 0.0
    for line, (_, sample_set) in zip(lines, sets, strict=True):
        grid = np.asarray(line.get_xdata())
        assert len(grid) == 512
        assert grid[0] >= 0.0
        estimate = kernel_density(sample_set, [0], case.space, factor)
        expected = np.exp(np.asarray(estimate.log_density(grid[:, None])))
        np.testing.assert_allclose(np.asarray(line.get_ydata()), expected, rtol=1e-8)
    top = max(float(np.max(line.get_ydata())) for line in lines)
    assert panel.get_ylim()[1] == pytest.approx(1.05 * top)
    # ? The corrected estimate differs from the plain one at the bound.
    plain = np.exp(np.asarray(estimate.kde.logpdf(jnp.zeros((1, 1)))))  # type: ignore[no-untyped-call]
    assert float(np.asarray(lines[1].get_ydata())[0]) != pytest.approx(
        float(plain[0]), rel=1e-3
    )


def test_a_redrawn_panel_has_the_contours_at_the_levels_of_the_package_grid() -> None:
    """The highest-density masses of Hyndman, on the grid of the package's estimate."""
    case = zoo.railed_gaussians([0.3, 1.0], param_count=2)
    sets = posterior_sets(case)
    figure = draw(sets, case.space)
    factor = overview_factor([sample_set for _, sample_set in sets])
    panel = figure.axes[2]
    x_limits, y_limits = panel.get_xlim(), panel.get_ylim()
    grid_x = np.linspace(max(x_limits[0], 0.0), x_limits[1], 80)
    grid_y = np.linspace(y_limits[0], y_limits[1], 80)
    mesh_x, mesh_y = np.meshgrid(grid_x, grid_y)
    points = np.stack([mesh_x.ravel(), mesh_y.ravel()], axis=-1)
    contours = [
        collection for collection in panel.collections if hasattr(collection, "levels")
    ]
    assert len(contours) == len(sets)
    for contour, (_, sample_set) in zip(contours, sets, strict=True):
        estimate = kernel_density(sample_set, [0, 1], case.space, factor)
        density = np.exp(np.asarray(estimate.log_density(points))).reshape(mesh_x.shape)
        levels = np.unique(hdi_levels(density, CONTOUR_LEVELS))
        levels = levels[levels < density.max()]
        np.testing.assert_allclose(contour.levels, levels, rtol=1e-8)
        assert np.asarray(contour.get_linewidth())[0] == 1.0
    assert tuple(panel.get_xlim()) == x_limits
    assert tuple(panel.get_ylim()) == y_limits


def test_a_panel_without_bounds_or_circles_is_not_redrawn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only the panels of a marked parameter come from the package's estimate."""
    calls: list[int] = []
    original = corner._grid_density

    def recorded(*arguments: Any, **keywords: Any) -> Any:
        calls.append(arguments[0].shape[1])
        return original(*arguments, **keywords)

    monkeypatch.setattr(corner, "_grid_density", recorded)
    plain = zoo.shifted_gaussians(1.0, param_count=2)
    draw(posterior_sets(plain), plain.space)
    assert calls == []
    railed = zoo.railed_gaussians([0.3, 1.0], param_count=3)
    draw(posterior_sets(railed), railed.space)
    # * Parameter 0 is railed: its diagonal and its two panels, for two posteriors.
    assert sorted(calls) == [1, 1, 2, 2, 2, 2]


def test_a_panel_whose_levels_reach_the_maximum_draws_no_contour(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """As in cantuccio, a level at the top of the grid leaves nothing to draw."""
    monkeypatch.setattr(
        corner,
        "hdi_levels",
        lambda density, levels: [float(density.max())] * len(levels),
    )
    case = zoo.railed_gaussians([0.3, 1.0], param_count=2)
    figure = draw(posterior_sets(case), case.space)
    panel = figure.axes[2]
    assert not [item for item in panel.collections if hasattr(item, "levels")]
    assert len(density_lines(figure.axes[0])) == 2


def test_a_parameter_far_from_its_bound_is_not_marked() -> None:
    """The reach is twelve kernel widths of the nearest sample."""
    near = zoo.railed_gaussians([0.3, 1.0], param_count=2)
    assert bounded_parameters(posterior_sets(near), near.space, 0.25) == ("x0",)
    gaussians = zoo.shifted_gaussians(1.0, param_count=2)
    interval = ParamSpace(
        parameters=(
            Parameter(name="x0", topology=Topology.INTERVAL, bounds=(-60.0, 60.0)),
            Parameter(name="x1", topology=Topology.INTERVAL, bounds=(-60.0, math.inf)),
        )
    )
    assert bounded_parameters(posterior_sets(gaussians), interval, 0.25) == ()
    upper = ParamSpace(
        parameters=(
            Parameter(name="x0", topology=Topology.INTERVAL, bounds=(-math.inf, 3.0)),
            Parameter(name="x1"),
        )
    )
    assert bounded_parameters(posterior_sets(gaussians), upper, 0.25) == ("x0",)


def test_the_gaussians_are_dashed_in_the_colour_of_the_posterior() -> None:
    """One ellipse per mass and posterior on a panel, two dashed lines on a diagonal."""
    case = zoo.shifted_gaussians(1.0, param_count=2)
    sets = posterior_sets(case)
    figure = draw(sets, case.space)
    panel = figure.axes[2]
    assert len(panel.patches) == len(sets) * len(CONTOUR_LEVELS)
    for position, patch in enumerate(panel.patches):
        assert patch.get_linestyle() == "--"
        colour = COLOURS[position // len(CONTOUR_LEVELS)]
        assert to_rgb(patch.get_edgecolor()) == pytest.approx(to_rgb(colour))
    diagonal = figure.axes[0]
    dashed = [line for line in diagonal.lines if line.get_linestyle() == "--"]
    assert len(dashed) == 2 * len(sets) * len(CONTOUR_LEVELS)
    # * The interval of the first mass has the radius of chi-squared with one degree.
    mean = float(jnp.mean(sets[0][1].samples[0, 0][:, 0]))
    spread = float(jnp.std(sets[0][1].samples[0, 0][:, 0], ddof=1))
    expected = [
        mean - spread * np.sqrt(chi2.ppf(0.68, 1)),
        mean + spread * np.sqrt(chi2.ppf(0.68, 1)),
    ]
    first = sorted(np.asarray(line.get_xdata())[0] for line in dashed[:2])
    np.testing.assert_allclose(first, expected, rtol=1e-6)
    bare = draw(sets, case.space, gaussians=False)
    assert len(bare.axes[2].patches) == 0
    assert not [line for line in bare.axes[0].lines if line.get_linestyle() == "--"]


def test_bound_lines_appear_only_when_asked() -> None:
    """Each finite bound inside the limits is a thin solid reference line."""
    rng = np.random.default_rng(0)
    space = ParamSpace(
        parameters=(
            Parameter(name="a", topology=Topology.INTERVAL, bounds=(0.0, 1.0)),
            Parameter(name="b", topology=Topology.INTERVAL, bounds=(0.0, 1.0)),
            Parameter(name="c"),
        )
    )
    samples = np.concatenate(
        [rng.uniform(0.0, 1.0, (400, 2)), rng.normal(size=(400, 1))], axis=1
    )
    sets = [("only", SampleSet(samples))]
    plain = corner_plot(sets, space, factor=0.2, colours=COLOURS[:1], title="t")
    reference = to_rgb(REFERENCE)

    def marks(figure: Figure, row: int, column: int) -> list[Any]:
        panel = figure.axes[3 * row + column]
        return [line for line in panel.lines if to_rgb(line.get_color()) == reference]

    assert all(
        marks(plain, row, column) == [] for row in range(3) for column in range(row + 1)
    )
    with_bounds = corner_plot(
        sets, space, factor=0.2, colours=COLOURS[:1], title="t", bounds=True
    )
    assert len(marks(with_bounds, 0, 0)) == 2
    assert len(marks(with_bounds, 2, 2)) == 0
    # * The panel of b over a has a vertical and a horizontal line at each bound.
    lines = marks(with_bounds, 1, 0)
    assert len(lines) == 4
    vertical = [line for line in lines if len(set(np.asarray(line.get_xdata()))) == 1]
    assert sorted(np.asarray(line.get_xdata())[0] for line in vertical) == [0.0, 1.0]
    assert all(
        line.get_linewidth() == 0.8 and line.get_linestyle() == "-" for line in lines
    )
    # * The panel of c over a has only the bounds of a; c has none.
    assert len(marks(with_bounds, 2, 0)) == 2


def test_a_circle_across_its_cut_stays_whole(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every circle is unwrapped about the pooled mean, so no posterior splits."""
    case = zoo.wrapped_gaussians(1.0, width=0.5, param_count=2)
    sets = posterior_sets(case)
    raw = np.asarray(sets[0][1].samples[0, 0][:, 0])
    assert np.ptp(raw) > 0.9 * 2 * math.pi, "the reference must straddle the cut"
    recorded: list[Any] = []
    original = cantuccio.cornerplot

    def spy(
        samples: list[dict[str, np.ndarray]], *arguments: Any, **keywords: Any
    ) -> Any:
        recorded.extend(samples)
        return original(samples, *arguments, **keywords)

    monkeypatch.setattr(cantuccio, "cornerplot", spy)
    figure = draw(sets, case.space)
    period = case.space.parameters[0].period
    assert period is not None
    pooled = jnp.concatenate([sample_set.samples[0, 0][:, 0] for _, sample_set in sets])
    log_weights = jnp.full(pooled.shape, -math.log(pooled.size))
    centre = float(
        circular_mean(pooled, log_weights, period, case.space.parameters[0].bounds[0])
    )
    for columns in recorded:
        circle = columns["x0"]
        assert np.ptp(circle) < 0.7 * period
        assert abs(np.mean(circle) - centre) < period / 2
        assert (circle >= centre - period / 2).all()
        assert (circle < centre + period / 2).all()
    panel = figure.axes[0]
    for line in density_lines(panel):
        assert np.ptp(line.get_xdata()) <= period
    # * The Gaussian mean of a circle is unwrapped into the plotted range too.
    dashed = [line for line in panel.lines if line.get_linestyle() == "--"]
    for line in dashed:
        assert abs(np.asarray(line.get_xdata())[0] - centre) < period / 2
    assert bounded_parameters(sets, case.space, 0.3) == ("x0",)


def test_a_sphere_is_drawn_as_a_circle_and_an_interval() -> None:
    """The declaration of the sphere is dropped, as in every marginal row."""
    case = zoo.von_mises_fisher(1.0, width=0.2)
    assert case.space.spheres
    sets = posterior_sets(case)
    marked = bounded_parameters(sets, case.space, 0.3)
    assert case.space.names[0] in marked
    figure = draw(sets, case.space)
    assert len(figure.axes) == len(case.space) ** 2


def test_the_layout_follows_the_number_of_parameters_and_the_arguments() -> None:
    """Size, resolution, title, legend, labels and colours."""
    case = zoo.shifted_gaussians(1.0, param_count=3)
    sets = posterior_sets(case, size=200)
    figure = draw(sets, case.space, title="three parameters")
    assert tuple(figure.get_size_inches()) == (6.0, 6.0)
    assert figure.dpi == 100
    assert figure.get_suptitle() == "three parameters"
    assert [text.get_text() for text in figure.legends[0].get_texts()] == list(
        case.names
    )
    assert figure.axes[-3].get_xlabel() == "x0"
    assert figure.axes[-1].get_xlabel() == "x2"
    assert figure.axes[3].get_ylabel() == "x1"
    one = draw(
        posterior_sets(zoo.shifted_gaussians(1.0, param_count=1)),
        zoo.shifted_gaussians(1.0, param_count=1).space,
    )
    assert tuple(one.get_size_inches()) == (2.0, 2.0)
    assert len(one.axes) == 1
    with pytest.raises(
        ValueError, match="one colour per posterior: 2 posteriors, 1 colours"
    ):
        corner_plot(sets, case.space, factor=0.2, colours=COLOURS[:1], title="t")


def test_the_resolution_is_capped_for_the_doubling_of_marimo() -> None:
    """Marimo saves at twice the dpi, so the saved width stays within 2 400 pixels."""
    assert corner._figure_dpi(2.0) == 100.0
    assert corner._figure_dpi(12.0) == 100.0
    # * Thirteen parameters make a figure of 26 inches.
    assert corner._figure_dpi(26.0) == pytest.approx(2_400 / (2 * 26.0))
    for parameters in (1, 6, 7, 13, 16):
        inches = 2.0 * parameters
        saved_pixels = 2 * corner._figure_dpi(inches) * inches
        assert saved_pixels <= 2_400 + 1e-9
        assert saved_pixels == pytest.approx(min(4.0 * 100 * parameters, 2_400))


def test_the_labels_carry_the_unit_and_fall_back_to_the_names() -> None:
    """A LaTeX label becomes mathtext, a bad one the name, a clash the names."""
    rng = np.random.default_rng(1)
    samples = rng.normal(size=(300, 3))
    sets = [("one", SampleSet(samples))]

    def labels_of(parameters: tuple[Parameter, ...]) -> list[str]:
        figure = corner_plot(
            sets,
            ParamSpace(parameters=parameters),
            factor=0.3,
            colours=COLOURS[:1],
            title="t",
        )
        return [
            figure.axes[6].get_xlabel(),
            figure.axes[7].get_xlabel(),
            figure.axes[8].get_xlabel(),
        ]

    plain = labels_of(
        (
            Parameter(name="phase", label=r"\phi_0", unit="rad"),
            Parameter(name="mass", label="mass", unit="Msun"),
            Parameter(name="bad", label=r"\notacommand"),
        )
    )
    assert plain == [r"$\phi_0$ [rad]", "mass [Msun]", "bad"]
    clash = labels_of(
        (
            Parameter(name="first", label="m"),
            Parameter(name="second", label="m"),
            Parameter(name="third", label="m", unit="kg"),
        )
    )
    assert clash == ["first", "second", "third"]
