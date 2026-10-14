"""Tests for ``Whitening``: blocks, sources, rotation, conditioning, Fisher."""

import math
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mixed_signals import zoo
from mixed_signals.core.samples import SampleSet, circular_standard_deviation
from mixed_signals.core.space import ParamSpace
from mixed_signals.core.topology import Topology
from mixed_signals.errors import PrecisionWarning, ValidityWarning
from mixed_signals.transforms import (
    CONDITION_NUMBER_LIMIT,
    Compose,
    Whitening,
    WhiteningSource,
)
from mixed_signals.zoo import ZooCase

SAMPLE_COUNT = 20_000
STANDARD_ERRORS = 5.0
# * A galactic-binary-like pair: σ_f = 2e-11 Hz at f = 3 mHz, and a unit axis.
GW_MEAN = np.array([3e-3, 1.0])
GW_COVARIANCE = np.array([[4e-22, 1e-11], [1e-11, 1.0]])
CORRELATED_COVARIANCE = np.array([[2.0, 0.6, -0.3], [0.6, 1.0, 0.2], [-0.3, 0.2, 0.5]])


def zca_cor(covariance: np.ndarray) -> np.ndarray:
    """Return P^(−1/2) V^(−1/2), computed independently of the package."""
    widths = np.sqrt(np.diag(covariance))
    eigenvalues, eigenvectors = np.linalg.eigh(covariance / np.outer(widths, widths))
    inverse_root = eigenvectors @ np.diag(eigenvalues**-0.5) @ eigenvectors.T
    return np.asarray(inverse_root / widths)


def real_space(count: int) -> ParamSpace:
    """Return a space of real parameters x0, x1, ..."""
    return ParamSpace.model_validate(
        {"parameters": [{"name": f"x{index}"} for index in range(count)]}
    )


def circle_space() -> ParamSpace:
    """Return a circle x0 of period 2π followed by two real parameters."""
    return ParamSpace.model_validate(
        {
            "parameters": [
                {"name": "x0", "topology": "circle", "bounds": (0.0, 2 * math.pi)},
                {"name": "x1"},
                {"name": "x2"},
            ]
        }
    )


def draws(case: ZooCase, seed: int = 0, count: int = SAMPLE_COUNT) -> SampleSet:
    """Return equally weighted draws of one problem of a zoo case."""
    return case.sample(jax.random.key(seed), count)


def whiten_posteriors(
    case: ZooCase, sample_set: SampleSet, posteriors: list[int], **options: Any
) -> Whitening:
    """Return the whitening from some posteriors of the first problem."""
    index = np.asarray(posteriors)
    return Whitening.from_samples(
        sample_set.samples[0][index],
        sample_set.log_weights[0][index],
        case.space,
        **options,
    )


def test_reference_whitening_makes_its_posterior_white() -> None:
    """The reference's own samples have zero mean and unit covariance afterwards."""
    case = zoo.shifted_gaussians(1.0, param_count=3, covariance=CORRELATED_COVARIANCE)
    sample_set = draws(case)
    whitening = whiten_posteriors(case, sample_set, [0])
    assert whitening.source is WhiteningSource.REFERENCE
    assert whitening.description == "whiten(reference)"
    whitened = np.asarray(whitening.forward(sample_set.samples[0, 0]))
    np.testing.assert_allclose(whitened.mean(axis=0), 0.0, atol=1e-12)
    np.testing.assert_allclose(np.cov(whitened.T), np.eye(3), atol=1e-12)


def test_zca_cor_axes_correlate_with_their_own_parameters() -> None:
    """The cross-correlation of whitened and original axes is P^(1/2).

    That symmetric positive-definite form is what makes ZCA-cor the whitening
    whose axes correlate most with their own parameters (Kessy, Lewin and
    Strimmer 2018).
    """
    case = zoo.shifted_gaussians(0.0, param_count=3, covariance=CORRELATED_COVARIANCE)
    original = np.asarray(draws(case).samples[0, 0])
    whitening = Whitening.from_samples(original[None], None, case.space)
    whitened = np.asarray(whitening.forward(jnp.asarray(original)))
    cross = np.corrcoef(whitened.T, original.T)[:3, 3:]
    eigenvalues, eigenvectors = np.linalg.eigh(np.corrcoef(original.T))
    root = eigenvectors @ np.diag(np.sqrt(eigenvalues)) @ eigenvectors.T
    np.testing.assert_allclose(cross, root, atol=1e-10)


@pytest.mark.parametrize("shift", [0.8, 3.0])
def test_reference_and_pooled_whitening_recover_the_zoo_shift(shift: float) -> None:
    """Both sources give the Mahalanobis shift, not the pooled-sample d/√(1 + d²/4)."""
    case = zoo.shifted_gaussians(shift, param_count=3, covariance=CORRELATED_COVARIANCE)
    sample_set = draws(case)
    truth = case.reference_value("mahalanobis_shift", ("reference", "shifted"))
    tolerance = STANDARD_ERRORS * math.sqrt(2 / SAMPLE_COUNT) * (1 + shift)
    for posteriors in ([0], [0, 1]):
        whitened = whiten_posteriors(case, sample_set, posteriors).forward(
            sample_set.samples[0]
        )
        gap = whitened[1].mean(axis=0) - whitened[0].mean(axis=0)
        assert float(jnp.linalg.norm(gap)) == pytest.approx(truth, abs=tolerance)
    shrunk = shift / math.sqrt(1 + shift**2 / 4)
    assert abs(truth - shrunk) > STANDARD_ERRORS * tolerance or shift < 1


def test_pooled_covariance_is_the_mean_of_the_two_covariances() -> None:
    """Pooled whitening uses (Σ_a + Σ_b)/2, each posterior weighted equally."""
    case = zoo.scaled_gaussians(
        0.5, param_count=3, axes=(0, 2), covariance=CORRELATED_COVARIANCE
    )
    sample_set = draws(case, count=500)
    whitening = whiten_posteriors(case, sample_set, [0, 1])
    assert whitening.description == "whiten(pooled)"
    samples = np.asarray(sample_set.samples[0])
    pooled = 0.5 * (np.cov(samples[0].T) + np.cov(samples[1].T))
    np.testing.assert_allclose(whitening.matrix, zca_cor(pooled), rtol=1e-10)
    np.testing.assert_allclose(
        whitening.centre, samples.mean(axis=1).mean(axis=0), rtol=1e-12
    )


def test_z_score_scales_by_the_root_mean_square_width() -> None:
    """Without decorrelation, each axis is divided by √((σ_a² + σ_b²)/2)."""
    case = zoo.scaled_gaussians(0.5, param_count=2)
    sample_set = draws(case, count=500)
    whitening = whiten_posteriors(case, sample_set, [0, 1], decorrelate=False)
    assert whitening.description == "zscore(pooled)"
    samples = np.asarray(sample_set.samples[0])
    pooled_width = np.sqrt(0.5 * samples.var(axis=1, ddof=1).sum(axis=0))
    np.testing.assert_allclose(whitening.matrix, np.diag(1 / pooled_width), rtol=1e-12)


def test_z_score_ignores_collinear_parameters() -> None:
    """Scaling alone needs no inverse correlation matrix, so collinearity is fine."""
    values = np.random.default_rng(8).normal(size=(1, 300, 1))
    samples = np.concatenate([values, -2 * values], axis=-1)
    whitening = Whitening.from_samples(samples, None, real_space(2), decorrelate=False)
    assert bool(jnp.all(jnp.isfinite(whitening.matrix)))


def test_zca_cor_does_not_depend_on_units() -> None:
    """Rescaling and shifting parameters leaves the whitened samples unchanged."""
    case = zoo.shifted_gaussians(1.0, param_count=3, covariance=CORRELATED_COVARIANCE)
    samples = draws(case, count=2_000).samples[0]
    scale, offset = jnp.array([1e-3, 1e8, 1.0]), jnp.array([5.0, -3e8, 0.0])
    original = Whitening.from_samples(samples, None, case.space)
    rescaled = Whitening.from_samples(samples * scale + offset, None, case.space)
    np.testing.assert_allclose(
        rescaled.forward(samples * scale + offset),
        original.forward(samples),
        rtol=1e-9,
        atol=1e-9,
    )


def test_gw_scale_whitening_round_trips_and_recovers_the_shift() -> None:
    """At σ_f/f ~ 1e-8 the round trip stays far below one width, and the shift holds."""
    case = zoo.shifted_gaussians(
        0.8, direction=np.array([1.0, 1.0]), mean=GW_MEAN, covariance=GW_COVARIANCE
    )
    sample_set = draws(case)
    whitening = whiten_posteriors(case, sample_set, [0])
    samples = sample_set.samples[0]
    back = whitening.inverse(whitening.forward(samples))
    widths = np.sqrt(np.diag(GW_COVARIANCE))
    assert float(jnp.max(jnp.abs(back - samples) / widths)) < 1e-6
    whitened = whitening.forward(samples)
    gap = float(jnp.linalg.norm(whitened[1].mean(axis=0) - whitened[0].mean(axis=0)))
    assert gap == pytest.approx(0.8, abs=STANDARD_ERRORS * math.sqrt(2 / SAMPLE_COUNT))


def test_circles_are_only_scaled_by_their_circular_width() -> None:
    """A circle correlated with a real axis is scaled alone, never mixed."""
    case = zoo.wrapped_gaussians(
        -2.5, width=0.4, param_count=3, correlation=0.6, period=math.pi, lower=-1.0
    )
    sample_set = draws(case)
    whitening = whiten_posteriors(case, sample_set, [0])
    assert whitening.blocks.circles == (0,)
    assert float(whitening.circle_scales[0]) == pytest.approx(0.4, rel=0.02)
    jacobian = np.asarray(whitening.jacobian(sample_set.samples[0, 0, 0]))
    np.testing.assert_array_equal(jacobian[0, 1:], 0.0)
    np.testing.assert_array_equal(jacobian[1:, 0], 0.0)
    assert jacobian[0, 0] == pytest.approx(1 / float(whitening.circle_scales[0]))
    euclidean = np.asarray(whitening.forward(sample_set.samples[0, 0]))[:, 1:]
    np.testing.assert_allclose(np.cov(euclidean.T), np.eye(2), atol=1e-12)


def test_pooled_circles_take_the_root_mean_square_width() -> None:
    """Two circular widths pool as √((s_a² + s_b²)/2)."""
    space = circle_space()
    rng = np.random.default_rng(12)
    samples = rng.normal(size=(2, 400, 3))
    samples[:, :, 0] = np.mod(samples[:, :, 0] * np.array([[0.2], [0.5]]), 2 * np.pi)
    whitening = Whitening.from_samples(samples, None, space)
    widths = circular_standard_deviation(
        jnp.asarray(samples[:, :, 0]), jnp.zeros((2, 400)), 2 * math.pi
    )
    expected = math.sqrt(float(jnp.mean(widths**2)))
    assert float(whitening.circle_scales[0]) == pytest.approx(expected, rel=1e-12)


def test_whitened_space_drops_units_and_scales_circle_periods() -> None:
    """Decorrelated axes become real lines; circles keep their topology at P/s."""
    case = zoo.wrapped_gaussians(1.0, param_count=3, correlation=0.3)
    whitening = whiten_posteriors(case, draws(case, count=2_000), [0])
    mapped = whitening.map_space(case.space)
    scale = float(whitening.circle_scales[0])
    assert mapped["x0"].topology is Topology.CIRCLE
    assert mapped["x0"].period == pytest.approx(2 * math.pi / scale)
    assert all(mapped[name].topology is Topology.REAL for name in ("x1", "x2"))
    assert all(parameter.unit is None for parameter in mapped.parameters)
    assert mapped.axis_blocks == case.space.axis_blocks


def test_z_score_space_maps_interval_bounds() -> None:
    """Without decorrelation an interval stays an interval, with mapped bounds."""
    case = zoo.railed_gaussians([0.5, 1.0], param_count=2)
    whitening = whiten_posteriors(
        case, draws(case, count=2_000), [0, 1], decorrelate=False
    )
    mapped = whitening.map_space(case.space)
    expected_lower = -float(whitening.centre[0]) * float(whitening.matrix[0, 0])
    assert mapped["x0"].topology is Topology.INTERVAL
    assert mapped["x0"].bounds == pytest.approx((expected_lower, math.inf))
    assert mapped["x1"].topology is Topology.REAL
    decorrelated = whiten_posteriors(case, draws(case, count=2_000), [0, 1])
    assert decorrelated.map_space(case.space)["x0"].topology is Topology.REAL


def test_spheres_pass_unchanged() -> None:
    """Sky positions keep their coordinates, topology and units."""
    case = zoo.von_mises_fisher(
        3.0, width=0.05, colatitude=0.1, polar_form="colatitude", euclidean_count=1
    )
    sample_set = draws(case, count=2_000)
    whitening = whiten_posteriors(case, sample_set, [0, 1])
    assert whitening.blocks.spheres == ((0, 1),)
    whitened = whitening.forward(sample_set.samples[0])
    np.testing.assert_array_equal(whitened[..., :2], sample_set.samples[0][..., :2])
    mapped = whitening.map_space(case.space)
    assert mapped.spheres == case.space.spheres
    assert mapped.parameters[:2] == case.space.parameters[:2]


def test_whitening_without_a_euclidean_block() -> None:
    """A space of one circle gives an empty matrix and only a scale."""
    case = zoo.wrapped_gaussians(1.0)
    sample_set = draws(case, count=2_000)
    whitening = whiten_posteriors(case, sample_set, [0])
    assert whitening.matrix.shape == (0, 0)
    scale = float(whitening.circle_scales[0])
    assert float(whitening.log_abs_det) == pytest.approx(-math.log(scale))
    np.testing.assert_allclose(
        whitening.forward(sample_set.samples[0]), sample_set.samples[0] / scale
    )


def test_jacobian_is_the_block_matrix_and_log_det_its_determinant() -> None:
    """jax.jacfwd gives W on the Euclidean block and 1/s on circles."""
    case = zoo.wrapped_gaussians(
        -2.5, width=0.4, param_count=3, correlation=0.6, period=math.pi, lower=-1.0
    )
    sample_set = draws(case, count=2_000)
    whitening = whiten_posteriors(case, sample_set, [0, 1])
    expected = np.zeros((3, 3))
    expected[0, 0] = 1 / float(whitening.circle_scales[0])
    expected[1:, 1:] = np.asarray(whitening.matrix)
    jacobian = whitening.jacobian(sample_set.samples[0, :, :4])
    np.testing.assert_allclose(jacobian, np.broadcast_to(expected, (2, 4, 3, 3)))
    np.testing.assert_allclose(
        whitening.log_abs_det_jacobian(sample_set.samples[0, :, :4]),
        np.linalg.slogdet(expected)[1],
        rtol=1e-13,
    )


@pytest.mark.parametrize(
    "build",
    [
        lambda: zoo.shifted_gaussians(1.5, param_count=3),
        lambda: zoo.shifted_gaussians(
            0.8, direction=np.array([1.0, 1.0]), mean=GW_MEAN, covariance=GW_COVARIANCE
        ),
        lambda: zoo.banana(0.1, param_count=4, variance=4.0),
        lambda: zoo.railed_gaussians([2.0, -0.5], param_count=2),
        lambda: zoo.wrapped_gaussians(-2.5, width=0.4, param_count=3, correlation=0.6),
        lambda: zoo.von_mises_fisher(1.0, polar_form="sin_latitude", euclidean_count=2),
    ],
)
@pytest.mark.parametrize("decorrelate", [True, False])
def test_inverse_round_trips_on_zoo_cases(
    build: Callable[[], ZooCase], decorrelate: bool
) -> None:
    """inverse(forward(x)) returns x to a tiny fraction of every width."""
    case = build()
    samples = draws(case, count=2_000).samples[0]
    whitening = Whitening.from_samples(
        samples, None, case.space, decorrelate=decorrelate
    )
    back = whitening.inverse(whitening.forward(samples))
    widths = jnp.std(samples, axis=(0, 1))
    assert float(jnp.max(jnp.abs(back - samples) / widths)) < 1e-9


def test_fisher_whitening_matches_covariance_whitening_at_gw_scale() -> None:
    """With Γ = Σ⁻¹, the whitening is ZCA-cor of Σ and whitens Γ to the identity."""
    space = real_space(2)
    fisher = np.linalg.inv(GW_COVARIANCE)
    whitening = Whitening.from_fisher(fisher, GW_MEAN, space)
    assert whitening.source is WhiteningSource.FISHER
    assert whitening.description == "whiten(fisher)"
    np.testing.assert_allclose(whitening.matrix, zca_cor(GW_COVARIANCE), rtol=1e-9)
    np.testing.assert_allclose(whitening.centre, GW_MEAN)
    mapped = whitening.map_fisher(jnp.asarray(fisher), jnp.asarray(GW_MEAN))
    np.testing.assert_allclose(mapped, np.eye(2), atol=1e-9)


def test_fisher_whitening_maps_the_density_to_a_unit_normal() -> None:
    """Log N(x; μ, Σ) − ln|det ∂y/∂x| is the unit-normal log density of y."""
    case = zoo.shifted_gaussians(
        0.0, direction=np.array([1.0, 1.0]), mean=GW_MEAN, covariance=GW_COVARIANCE
    )
    samples = draws(case, count=200).samples[0, 0]
    log_q = case.log_prob(samples)[:, 0]
    sample_set = SampleSet(samples, log_q=log_q)
    whitening = Whitening.from_fisher(np.linalg.inv(GW_COVARIANCE), GW_MEAN, case.space)
    mapped = whitening.map_sample_set(sample_set)
    assert mapped.log_q is not None
    whitened = mapped.samples[0, 0]
    unit_normal = -0.5 * jnp.sum(whitened**2, axis=-1) - math.log(2 * math.pi)
    np.testing.assert_allclose(mapped.log_q[0, 0], unit_normal, rtol=1e-9)


def test_fisher_whitening_uses_the_marginal_euclidean_block() -> None:
    """The Euclidean block takes its block of Γ⁻¹, not the inverse of its Γ block."""
    space = circle_space()
    covariance = np.array([[0.04, 0.1, -0.05], [0.1, 1.0, 0.4], [-0.05, 0.4, 2.0]])
    fisher = np.linalg.inv(covariance)
    whitening = Whitening.from_fisher(fisher, np.array([1.0, 0.0, 0.0]), space)
    np.testing.assert_allclose(whitening.matrix, zca_cor(covariance[1:, 1:]), rtol=1e-9)
    assert float(whitening.circle_scales[0]) == pytest.approx(0.2, rel=1e-12)
    conditional = zca_cor(np.linalg.inv(fisher[1:, 1:]))
    assert not np.allclose(whitening.matrix, conditional, rtol=1e-3)
    z_score = Whitening.from_fisher(fisher, np.zeros(3), space, decorrelate=False)
    assert z_score.description == "zscore(fisher)"
    np.testing.assert_allclose(
        np.diag(z_score.matrix), 1 / np.sqrt(np.diag(covariance)[1:]), rtol=1e-12
    )


def test_ill_conditioned_fisher_matrices_warn() -> None:
    """A condition number above 1/√ε warns, and the whitening is still built."""
    correlation = 1 - 1e-9
    scales = np.array([1e10, 1.0])
    fisher = np.array([[1.0, correlation], [correlation, 1.0]]) * np.outer(
        scales, scales
    )
    with pytest.warns(
        ValidityWarning, match=r"condition number 2e\+09.*along 0\.707 x0 - 0\.707 x1$"
    ):
        whitening = Whitening.from_fisher(fisher, np.zeros(2), real_space(2))
    assert bool(jnp.all(jnp.isfinite(whitening.matrix)))
    assert pytest.approx(2.0**26) == CONDITION_NUMBER_LIMIT


@pytest.mark.parametrize(
    ("fisher", "centre", "message"),
    [
        (np.eye(2), np.zeros(3), r"centre must be shaped \(2,\)"),
        (np.eye(2)[:1], np.zeros(2), r"fisher must be shaped \(2, 2\)"),
        (np.array([[1.0, 0.0], [0.0, np.nan]]), np.zeros(2), "fisher must be finite"),
        (np.eye(2), np.array([0.0, np.inf]), "centre must be finite"),
        (np.diag([1.0, -2.0]), np.zeros(2), r"information of 'x1' is not positive"),
        (np.array([[1.0, 0.5], [0.4, 1.0]]), np.zeros(2), "must be symmetric"),
        (np.ones((2, 2)), np.zeros(2), "singular or not positive definite"),
        (np.array([[1.0, 2.0], [2.0, 1.0]]), np.zeros(2), "not positive definite"),
    ],
)
def test_invalid_fisher_inputs_raise(
    fisher: np.ndarray, centre: np.ndarray, message: str
) -> None:
    """Fisher matrices must fit the space and be finite, symmetric and definite."""
    with pytest.raises(ValueError, match=message):
        Whitening.from_fisher(fisher, centre, real_space(2))


def test_invalid_fisher_inside_jit_gives_nan() -> None:
    """Traced Fisher matrices are not checked; a singular one gives NaN."""
    space = real_space(2)
    compiled = jax.jit(
        lambda fisher: Whitening.from_fisher(fisher, jnp.zeros(2), space).matrix
    )
    assert bool(jnp.all(jnp.isnan(compiled(jnp.ones((2, 2))))))
    assert bool(jnp.all(jnp.isfinite(compiled(jnp.eye(2)))))


def degenerate_samples(kind: str) -> tuple[np.ndarray, np.ndarray | None, ParamSpace]:
    """Return samples of one posterior with a singular width, and their space."""
    values = np.random.default_rng(21).normal(size=(1, 200, 3))
    space = real_space(3)
    log_weights = None
    if kind == "constant":
        values[..., 1] = 0.25
    elif kind == "two constants":
        values[..., :2] = 0.25
    elif kind == "collinear":
        values[..., 2] = -2 * values[..., 0]
    elif kind == "one weight":
        log_weights = np.full((1, 200), -np.inf)
        log_weights[0, 3] = 0.0
    else:
        space = circle_space()
        values[..., 0] = 1.5
    return values, log_weights, space


@pytest.mark.parametrize(
    ("kind", "message"),
    [
        ("constant", r"NaN: 'x1' has no spread"),
        ("two constants", r"NaN: 'x0', 'x1' have no spread"),
        (
            "collinear",
            r"NaN: the Euclidean block is singular.* along 0\.707 x0 \+ 0\.707 x2$",
        ),
        ("one weight", "fewer than two samples with positive weight"),
        ("circle", r"circular standard deviation of 'x0' is zero or infinite"),
    ],
)
def test_singular_widths_give_nan_and_a_warning(kind: str, message: str) -> None:
    """A singular covariance warns and gives NaN instead of raising."""
    samples, log_weights, space = degenerate_samples(kind)
    with pytest.warns(ValidityWarning, match=message):
        whitening = Whitening.from_samples(samples, log_weights, space)
    assert math.isnan(float(whitening.log_abs_det))
    assert bool(jnp.all(jnp.isnan(whitening.matrix)))
    with pytest.raises(ValueError, match="the whitening is NaN"):
        whitening.map_space(space)


def test_ill_conditioned_samples_warn_but_whiten() -> None:
    """Nearly collinear parameters warn about lost digits, and the whitening holds."""
    rng = np.random.default_rng(5)
    values = rng.normal(size=(1, 500, 2))
    values[..., 1] = values[..., 0] + 1e-5 * rng.normal(size=(1, 500))
    with pytest.warns(
        ValidityWarning, match=r"condition number .* along 0\.707 x0 - 0\.707 x1$"
    ):
        whitening = Whitening.from_samples(values, None, real_space(2))
    assert bool(jnp.all(jnp.isfinite(whitening.matrix)))


def test_nothing_warns_inside_jit_and_vmap() -> None:
    """Traced widths are unknown, so a singular covariance gives NaN silently."""
    samples, _, space = degenerate_samples("constant")
    compiled = jax.jit(lambda values: Whitening.from_samples(values, None, space))
    assert bool(jnp.all(jnp.isnan(compiled(jnp.asarray(samples)).matrix)))


def test_vmap_builds_one_whitening_per_problem() -> None:
    """jax.vmap over problems gives the whitenings of the problems one by one."""
    case = zoo.shifted_gaussians(1.0, param_count=3, covariance=CORRELATED_COVARIANCE)
    sample_set = case.sample(jax.random.key(2), 300, batch_size=3)
    batched = jax.vmap(
        lambda samples, log_weights: Whitening.from_samples(
            samples, log_weights, case.space
        )
    )(sample_set.samples, sample_set.log_weights)
    assert batched.matrix.shape == (3, 3, 3)
    single = Whitening.from_samples(
        sample_set.samples[1], sample_set.log_weights[1], case.space
    )
    np.testing.assert_allclose(batched.matrix[1], single.matrix, rtol=1e-12)
    whitened = jax.vmap(lambda whitening, points: whitening.forward(points))(
        batched, sample_set.samples
    )
    np.testing.assert_allclose(
        whitened[1], single.forward(sample_set.samples[1]), rtol=1e-12
    )


@pytest.mark.parametrize(
    ("samples", "log_weights", "message"),
    [
        (np.zeros((200, 3)), None, r"samples must be shaped \(K, N, 3\)"),
        (np.zeros((1, 200, 2)), None, r"samples must be shaped \(K, N, 3\)"),
        (np.zeros((1, 0, 3)), None, r"samples must be shaped \(K, N, 3\)"),
        (np.zeros((2, 5, 3)), np.zeros((2, 4)), r"log_weights must be shaped \(2, 5\)"),
        (np.full((1, 5, 3), np.nan), None, "samples must be finite"),
        (
            np.zeros((1, 5, 3)),
            np.full((1, 5), np.nan),
            "must not contain NaN or \\+inf",
        ),
        (np.zeros((1, 5, 3)), np.full((1, 5), -np.inf), "positive weight"),
    ],
)
def test_invalid_sample_inputs_raise(
    samples: np.ndarray, log_weights: np.ndarray | None, message: str
) -> None:
    """Shapes, finite values and a positive weight per posterior are checked."""
    with pytest.raises(ValueError, match=message):
        Whitening.from_samples(samples, log_weights, real_space(3))


def test_input_types_are_checked_and_cast() -> None:
    """Integers are cast silently, narrow floats warn and complex inputs fail."""
    integers = np.arange(30).reshape(1, 10, 3) % 7
    integers[..., 1] = integers[..., 1] ** 2
    integers[..., 2] = (integers[..., 2] * 3) % 5
    Whitening.from_samples(integers, None, real_space(3), decorrelate=False)
    with pytest.warns(PrecisionWarning, match=r"cast fisher \(float32\)"):
        Whitening.from_fisher(np.eye(2, dtype=np.float32), np.zeros(2), real_space(2))
    with pytest.raises(TypeError, match="samples must hold real numbers"):
        Whitening.from_samples(np.ones((1, 4, 3), dtype=complex), None, real_space(3))


def test_map_space_needs_the_blocks_of_its_whitening() -> None:
    """A space with other axis blocks cannot be mapped."""
    whitening = Whitening.from_fisher(np.eye(3), np.zeros(3), real_space(3))
    with pytest.raises(ValueError, match="but the whitening was built for"):
        whitening.map_space(circle_space())


def test_composed_whitenings_describe_and_map_in_order() -> None:
    """A chain of whitenings reads left to right and maps the space step by step."""
    space = circle_space()
    first = Whitening.from_fisher(
        np.diag([25.0, 1.0, 0.5]), np.zeros(3), space, decorrelate=False
    )
    second = Whitening.from_fisher(np.diag([4.0, 2.0, 2.0]), np.ones(3), space)
    chain = Compose((first, second))
    assert chain.description == "zscore(fisher) | whiten(fisher)"
    mapped = chain.map_space(space)
    assert mapped["x0"].period == pytest.approx(2 * math.pi * 5 * 2)
    point = jnp.array([0.3, -1.0, 2.0])
    np.testing.assert_allclose(
        chain.forward(point), second.forward(first.forward(point)), rtol=1e-14
    )
