"""Tests for the joint local chart of a Fisher matrix."""

import math

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mixed_signals.core.space import ParamSpace
from mixed_signals.dataspace import CHART_CUT_WIDTHS, FisherChart
from mixed_signals.errors import PrecisionWarning, ValidityWarning
from mixed_signals.transforms.base import Transform

TWO_PI = 2 * math.pi
# * A real line, a circle from 0, a circle from -π/2 with period π, and an interval.
SPACE = ParamSpace.model_validate(
    {
        "parameters": [
            {"name": "amplitude", "label": "A", "unit": "strain"},
            {"name": "phase", "topology": "circle", "bounds": [0.0, TWO_PI]},
            {
                "name": "psi",
                "topology": "circle",
                "bounds": [-math.pi / 2, math.pi / 2],
            },
            {"name": "fraction", "topology": "interval", "bounds": [0.0, 1.0]},
        ]
    }
)
SKY_SPACE = ParamSpace.model_validate(
    {
        "parameters": [
            {"name": "longitude", "topology": "circle", "bounds": [0.0, TWO_PI]},
            {"name": "sin_latitude", "topology": "interval", "bounds": [-1.0, 1.0]},
            {"name": "frequency"},
        ],
        "spheres": [
            {
                "azimuth": "longitude",
                "polar": "sin_latitude",
                "polar_form": "sin_latitude",
            }
        ],
    }
)
CENTRE = np.array([1.0, 0.2, 0.1, 0.5])
PERIODS = np.array([np.inf, TWO_PI, math.pi, np.inf])


def correlated_fisher(seed: int = 3) -> np.ndarray:
    """Return a positive definite matrix with strong correlations between all axes."""
    generator = np.random.default_rng(seed)
    factor = generator.normal(size=(4, 4))
    return factor @ factor.T + 4 * np.eye(4)


def chart() -> FisherChart:
    """Return the chart of the correlated matrix at the centre."""
    return FisherChart.from_fisher(correlated_fisher(), CENTRE, SPACE)


def arcs_from_centre(points: np.ndarray) -> np.ndarray:
    """Return the signed arcs from the centre in NumPy, as the chart defines them."""
    offsets = points - CENTRE
    circle = np.isfinite(PERIODS)
    wrapped = (offsets + np.where(circle, PERIODS, 1.0) / 2) % np.where(
        circle, PERIODS, 1.0
    ) - np.where(circle, PERIODS, 1.0) / 2
    arcs: np.ndarray = np.where(circle, wrapped, offsets)
    return arcs


# * Construction


def test_the_matrix_satisfies_w_transpose_w_equal_to_the_fisher_matrix() -> None:
    """WᵀW = Γ, W⁻¹ inverts W, and ln det W = ½ ln det Γ."""
    fisher = correlated_fisher()
    result = chart()
    np.testing.assert_allclose(result.matrix.T @ result.matrix, fisher, rtol=1e-10)
    np.testing.assert_allclose(
        result.matrix @ result.inverse_matrix, np.eye(4), atol=1e-12
    )
    assert float(result.log_abs_det) == pytest.approx(
        0.5 * np.linalg.slogdet(fisher)[1], rel=1e-12
    )
    assert result.periods == (None, TWO_PI, math.pi, None)
    assert result.lower_bounds == (None, 0.0, -math.pi / 2, None)
    assert result.names == ("amplitude", "phase", "psi", "fraction")


def test_the_matrix_is_the_zca_cor_whitening_of_the_covariance() -> None:
    """W = R^(−1/2) V^(−1/2) for the correlation matrix R: axes keep their names."""
    covariance = np.linalg.inv(correlated_fisher())
    widths = np.sqrt(np.diag(covariance))
    correlation = covariance / np.outer(widths, widths)
    values, vectors = np.linalg.eigh(correlation)
    expected = (vectors / np.sqrt(values)) @ vectors.T / widths
    np.testing.assert_allclose(chart().matrix, expected, rtol=1e-9)
    assert np.all(np.diag(np.asarray(chart().matrix)) > 0)


def test_a_sphere_azimuth_is_a_circle_and_its_polar_axis_a_line() -> None:
    """The azimuth of a sphere unwraps with the period 2π; the polar axis does not."""
    sky = FisherChart.from_fisher(np.diag([4.0, 9.0, 1.0]), np.zeros(3), SKY_SPACE)
    assert sky.periods == (TWO_PI, None, None)
    assert sky.lower_bounds == (0.0, None, None)
    across = sky.forward(jnp.array([[TWO_PI - 0.1, 0.3, 0.0], [0.1, 0.3, 0.0]]))
    np.testing.assert_allclose(across[0, 0], -0.2, rtol=1e-12)
    np.testing.assert_allclose(across[1, 0], 0.2, rtol=1e-12)


def test_the_chart_is_a_transform_with_its_description() -> None:
    """The chart is one Transform, and the space column names it chart(fisher)."""
    result = chart()
    assert isinstance(result, Transform)
    assert result.description == "chart(fisher)"
    assert CHART_CUT_WIDTHS == 3.0


# * The maps


def test_chart_distances_are_the_fisher_quadratic_form_with_wrapped_arcs() -> None:
    """‖y₁ − y₂‖² = ΔθᵀΓΔθ for the arcs from the centre, across the boundary too."""
    fisher = correlated_fisher()
    result = chart()
    generator = np.random.default_rng(11)
    first = CENTRE + generator.normal(scale=0.4, size=(50, 4))
    second = CENTRE + generator.normal(scale=0.4, size=(50, 4))
    # * A pair across the boundary of the phase circle, close to the centre in arc.
    first[0, 1], second[0, 1] = TWO_PI - 0.05, 0.3
    differences = arcs_from_centre(first) - arcs_from_centre(second)
    expected = np.einsum("ni,ij,nj->n", differences, fisher, differences)
    charted_first = np.asarray(result.forward(jnp.asarray(first)))
    charted_second = np.asarray(result.forward(jnp.asarray(second)))
    distances = np.sum((charted_first - charted_second) ** 2, axis=1)
    np.testing.assert_allclose(distances, expected, rtol=1e-10)
    # * The pair across the boundary is 0.35 apart in the phase, not 6.
    assert abs(differences[0, 1]) == pytest.approx(0.35, abs=1e-12)


def test_the_centre_maps_to_zero_also_a_period_away() -> None:
    """The centre is the origin of the chart, and one period away on a circle too."""
    result = chart()
    np.testing.assert_allclose(result.forward(jnp.asarray(CENTRE)), 0.0, atol=1e-14)
    shifted = CENTRE + np.array([0.0, TWO_PI, 0.0, 0.0])
    np.testing.assert_allclose(result.forward(jnp.asarray(shifted)), 0.0, atol=1e-12)


def test_the_forward_map_takes_any_leading_shape() -> None:
    """A batch of points maps as each point alone."""
    result = chart()
    points = jnp.asarray(CENTRE + np.random.default_rng(1).normal(size=(3, 5, 4)) * 0.3)
    batched = result.forward(points)
    assert batched.shape == (3, 5, 4)
    np.testing.assert_allclose(batched[1, 2], result.forward(points[1, 2]), rtol=1e-13)


def test_the_inverse_restores_points_inside_the_circle_ranges() -> None:
    """Round trips of points anywhere in the ranges, also far from the centre."""
    result = chart()
    generator = np.random.default_rng(5)
    points = np.stack(
        [
            generator.normal(size=40),
            generator.uniform(0.0, TWO_PI, 40),
            generator.uniform(-math.pi / 2, math.pi / 2, 40),
            generator.uniform(0.0, 1.0, 40),
        ],
        axis=1,
    )
    restored = np.asarray(result.inverse(result.forward(jnp.asarray(points))))
    np.testing.assert_allclose(restored, points, rtol=1e-10, atol=1e-10)
    # * Outside the range, a circle value returns as its representative inside.
    outside = jnp.asarray(CENTRE + np.array([0.0, TWO_PI + 0.3, -math.pi, 0.0]))
    returned = np.asarray(result.inverse(result.forward(outside)))
    np.testing.assert_allclose(returned[1], 0.5, atol=1e-12)
    np.testing.assert_allclose(returned[2], 0.1, atol=1e-12)
    # * The real and the interval axes are not wrapped.
    far = jnp.asarray(CENTRE + np.array([50.0, 0.0, 0.0, 3.0]))
    np.testing.assert_allclose(
        result.inverse(result.forward(far))[jnp.array([0, 3])], far[jnp.array([0, 3])]
    )


def test_the_log_determinant_is_the_constant_broadcast_to_the_points() -> None:
    """The constant ln det W at every point, shaped like the leading axes."""
    result = chart()
    points = jnp.zeros((2, 3, 4))
    values = result.log_abs_det_jacobian(points)
    assert values.shape == (2, 3)
    np.testing.assert_allclose(values, float(result.log_abs_det), rtol=1e-14)


def test_the_jacobian_is_w_away_from_the_cut_and_matches_autodiff() -> None:
    """jax.jacfwd of the forward map equals W at points off the cut."""
    result = chart()
    points = jnp.asarray(CENTRE + np.random.default_rng(2).normal(size=(6, 4)) * 0.3)
    jacobians = result.jacobian(points)
    autodiff = jax.vmap(jax.jacfwd(result.forward))(points)
    np.testing.assert_allclose(jacobians, autodiff, rtol=1e-13)
    np.testing.assert_allclose(jacobians[0], result.matrix, rtol=1e-12)
    log_dets = jnp.linalg.slogdet(jacobians)[1]
    np.testing.assert_allclose(log_dets, float(result.log_abs_det), rtol=1e-10)


def test_the_fisher_matrix_of_the_chart_is_the_identity() -> None:
    """Γ maps to Jᵀ Γ J = I at the centre, also for a batch of copies."""
    fisher = correlated_fisher()
    result = chart()
    mapped = result.map_fisher(jnp.asarray(fisher), jnp.asarray(CENTRE))
    np.testing.assert_allclose(mapped, np.eye(4), atol=1e-10)
    batched = result.map_fisher(
        jnp.broadcast_to(jnp.asarray(fisher), (2, 4, 4)),
        jnp.broadcast_to(jnp.asarray(CENTRE), (2, 4)),
    )
    np.testing.assert_allclose(
        batched, np.broadcast_to(np.eye(4), (2, 4, 4)), atol=1e-10
    )


def test_the_chart_builds_and_maps_inside_jit() -> None:
    """A closed-over matrix passes its checks; a chart crosses jit as an argument."""
    fisher = jnp.asarray(correlated_fisher())
    centre = jnp.asarray(CENTRE)
    points = jnp.asarray(CENTRE + 0.2)

    def build_and_map(values: jax.Array) -> jax.Array:
        return FisherChart.from_fisher(fisher, centre, SPACE).forward(values)

    expected = chart().forward(points)
    np.testing.assert_allclose(jax.jit(build_and_map)(points), expected, rtol=1e-13)
    np.testing.assert_allclose(
        eqx.filter_jit(chart().forward)(points), expected, rtol=1e-13
    )
    np.testing.assert_allclose(
        jax.jit(lambda built, point: built.forward(point))(chart(), points),
        expected,
        rtol=1e-13,
    )


def test_an_invalid_traced_fisher_matrix_gives_nan() -> None:
    """Values are unknown inside jit, so nothing raises and a singular matrix is NaN."""
    space = ParamSpace.model_validate({"parameters": [{"name": "a"}, {"name": "b"}]})
    compiled = jax.jit(
        lambda fisher: FisherChart.from_fisher(fisher, jnp.zeros(2), space).matrix
    )
    assert bool(jnp.all(jnp.isnan(compiled(jnp.ones((2, 2))))))
    assert bool(jnp.all(jnp.isfinite(compiled(jnp.eye(2)))))


# * The space of the chart


def test_the_space_of_the_chart_has_real_lines_with_the_names_and_labels() -> None:
    """Every axis is a real line, without unit, bounds, topology or sphere."""
    mapped = chart().map_space(SPACE)
    assert mapped.names == SPACE.names
    assert [parameter.label for parameter in mapped.parameters] == [
        parameter.label for parameter in SPACE.parameters
    ]
    assert mapped.parameters[0].label == "A"
    assert all(parameter.unit is None for parameter in mapped.parameters)
    assert all(parameter.topology == "real" for parameter in mapped.parameters)
    assert mapped.spheres == ()
    assert mapped.axis_blocks.circles == ()


def test_a_space_with_other_parameters_or_circles_is_rejected() -> None:
    """The chart maps only the space that it was built for, in its circle axes."""
    result = chart()
    three = ParamSpace.model_validate(
        {"parameters": [{"name": "a"}, {"name": "b"}, {"name": "c"}]}
    )
    with pytest.raises(ValueError, match="the space has 3 parameters, but the chart"):
        result.map_space(three)
    no_circles = ParamSpace.model_validate(
        {"parameters": [{"name": name} for name in SPACE.names]}
    )
    with pytest.raises(ValueError, match="other circle axes than the chart"):
        result.map_space(no_circles)
    other_period = ParamSpace.model_validate(
        {
            "parameters": [
                {"name": "amplitude"},
                {"name": "phase", "topology": "circle", "bounds": [0.0, 1.0]},
                {"name": "psi", "topology": "circle", "bounds": [-1.0, 1.0]},
                {"name": "fraction"},
            ]
        }
    )
    with pytest.raises(ValueError, match="other circle axes than the chart"):
        result.map_space(other_period)


# * Invalid inputs


@pytest.mark.parametrize(
    ("fisher", "centre", "message"),
    [
        (np.eye(3), CENTRE, r"fisher must be shaped \(4, 4\)"),
        (np.eye(4), CENTRE[:3], r"centre must be shaped \(4,\)"),
        (np.diag([1.0, np.nan, 1.0, 1.0]), CENTRE, "fisher must be finite"),
        (np.eye(4), np.array([0.0, np.inf, 0.0, 0.0]), "centre must be finite"),
        (np.diag([1.0, -1.0, 1.0, 1.0]), CENTRE, "information of 'phase' is not"),
        (
            np.array([[1, 0.5, 0, 0], [0.4, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1.0]]),
            CENTRE,
            "must be symmetric",
        ),
        (np.ones((4, 4)), CENTRE, "singular or not positive definite"),
        (
            np.array([[1, 3, 0, 0], [3, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1.0]]),
            CENTRE,
            "not positive definite",
        ),
    ],
)
def test_invalid_fisher_matrices_and_centres_raise(
    fisher: np.ndarray, centre: np.ndarray, message: str
) -> None:
    """Wrong shapes, and matrices not finite, symmetric or positive definite."""
    with pytest.raises(ValueError, match=message):
        FisherChart.from_fisher(fisher, centre, SPACE)


def test_complex_inputs_are_rejected() -> None:
    """A Fisher matrix must hold real numbers."""
    with pytest.raises(TypeError, match="fisher must hold real numbers"):
        FisherChart.from_fisher(np.eye(4) + 0j, CENTRE, SPACE)


def test_an_ill_conditioned_matrix_warns_and_still_builds_the_chart() -> None:
    """A condition number above 1/√ε loses digits: a ValidityWarning, a finite chart."""
    space = ParamSpace.model_validate({"parameters": [{"name": "a"}, {"name": "b"}]})
    correlation = 1 - 1e-9
    scales = np.array([1e10, 1.0])
    fisher = np.array([[1.0, correlation], [correlation, 1.0]]) * np.outer(
        scales, scales
    )
    with pytest.warns(ValidityWarning, match=r"condition number 2e\+09"):
        result = FisherChart.from_fisher(fisher, np.zeros(2), space)
    assert bool(jnp.all(jnp.isfinite(result.matrix)))


def test_a_float32_matrix_warns_about_precision() -> None:
    """The matrix is cast to float64 with the package's PrecisionWarning."""
    with pytest.warns(PrecisionWarning, match="fisher"):
        FisherChart.from_fisher(np.eye(4, dtype=np.float32), CENTRE, SPACE)


# * The warning about wide circles


def spread_samples(
    phase_width: float, psi_width: float, count: int = 4_000, seed: int = 0
) -> np.ndarray:
    """Return Gaussian samples about the centre, wide in the phase and in psi."""
    generator = np.random.default_rng(seed)
    widths = np.array([5.0, phase_width, psi_width, 0.1])
    samples: np.ndarray = CENTRE + generator.normal(size=(count, 4)) * widths
    return samples


def test_narrow_circles_do_not_warn() -> None:
    """Arcs within P/(2·3) of the centre leave the chart silent, a wide line too."""
    samples = spread_samples(phase_width=0.5, psi_width=0.2)
    chart().warn_wide_axes(samples, np.zeros(len(samples)))


def test_wide_circles_warn_once_and_name_every_axis() -> None:
    """One warning names the phase and psi, with their arcs and limits."""
    samples = spread_samples(phase_width=1.5, psi_width=0.9)
    with pytest.warns(ValidityWarning, match="'phase'.*'psi'") as records:
        chart().warn_wide_axes(samples, np.zeros(len(samples)))
    assert len(records) == 1
    message = str(records[0].message)
    assert "half a period" in message
    assert "'amplitude'" not in message
    assert "'fraction'" not in message


def test_only_the_wide_circle_is_named() -> None:
    """A narrow psi stays out of the message when the phase is wide."""
    samples = spread_samples(phase_width=1.5, psi_width=0.1)
    with pytest.warns(ValidityWarning, match="'phase'") as records:
        chart().warn_wide_axes(samples, np.zeros(len(samples)))
    assert "'psi'" not in str(records[0].message)


def test_the_weights_set_the_root_mean_square_arc() -> None:
    """Weights that remove the far samples silence the warning."""
    samples = spread_samples(phase_width=1.5, psi_width=0.1)
    near = np.abs(arcs_from_centre(samples)[:, 1]) < 0.3
    log_weights = np.where(near, 0.0, -np.inf)
    chart().warn_wide_axes(samples, log_weights)


def test_the_warning_reads_concrete_values_only() -> None:
    """Inside jax.jit the values are unknown, so the check does nothing."""
    samples = jnp.asarray(spread_samples(phase_width=1.5, psi_width=0.9))
    result = chart()

    def check(values: jax.Array) -> jax.Array:
        result.warn_wide_axes(values, jnp.zeros(values.shape[0]))
        return values.sum()

    jax.jit(check)(samples)
