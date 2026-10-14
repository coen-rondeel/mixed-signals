"""Tests for ``Transform`` and ``Compose``: Jacobians, Fisher maps and densities."""

import math

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jaxtyping import Array, Float
from scipy import stats

from mixed_signals.core.samples import SampleSet
from mixed_signals.core.space import ParamSpace
from mixed_signals.transforms import COMPOSITION_SEPARATOR, Compose, Transform

PARAM_COUNT = 3
# * An invertible, non-symmetric matrix, so that row and column conventions matter.
SHEAR_MATRIX = np.array([[2.0, 0.5, 0.0], [-0.3, 1.0, 0.7], [0.1, 0.0, 0.5]])
SHEAR_OFFSET = np.array([1.0, -2.0, 0.5])


class Sinh(Transform):
    """Elementwise sinh, a nonlinear bijection of the real line, for the tests."""

    def forward(
        self, points: Float[Array, "*leading param"]
    ) -> Float[Array, "*leading param"]:
        """Return sinh(x)."""
        return jnp.sinh(points)

    def inverse(
        self, points: Float[Array, "*leading param"]
    ) -> Float[Array, "*leading param"]:
        """Return arsinh(y)."""
        return jnp.arcsinh(points)

    def log_abs_det_jacobian(
        self, points: Float[Array, "*leading param"]
    ) -> Float[Array, " *leading"]:
        """Return Σ ln cosh(x)."""
        return jnp.sum(jnp.log(jnp.cosh(points)), axis=-1)

    @property
    def description(self) -> str:
        """Name of the test transform."""
        return "sinh"


class Shear(Transform):
    """The affine map y = A x + b with a fixed invertible matrix, for the tests."""

    matrix: Float[Array, "param param"]
    offset: Float[Array, " param"]

    def forward(
        self, points: Float[Array, "*leading param"]
    ) -> Float[Array, "*leading param"]:
        """Return A x + b."""
        return points @ self.matrix.T + self.offset

    def inverse(
        self, points: Float[Array, "*leading param"]
    ) -> Float[Array, "*leading param"]:
        """Return A⁻¹ (y − b)."""
        restored: Array = (points - self.offset) @ jnp.linalg.inv(self.matrix).T
        return restored

    def log_abs_det_jacobian(
        self, points: Float[Array, "*leading param"]
    ) -> Float[Array, " *leading"]:
        """Return ln|det A| at every point."""
        return jnp.broadcast_to(jnp.linalg.slogdet(self.matrix)[1], points.shape[:-1])

    @property
    def description(self) -> str:
        """Name of the test transform."""
        return "shear"


@pytest.fixture
def shear() -> Shear:
    """The test shear."""
    return Shear(jnp.asarray(SHEAR_MATRIX), jnp.asarray(SHEAR_OFFSET))


@pytest.fixture
def points() -> Array:
    """Points shaped (2, 5, D), with two leading axes."""
    return jax.random.normal(jax.random.key(11), (2, 5, PARAM_COUNT))


def test_transform_is_abstract() -> None:
    """The base class needs its maps from a subclass."""
    with pytest.raises(TypeError, match="abstract"):
        Transform()  # type: ignore[abstract]


def test_default_jacobian_differentiates_forward(points: Array) -> None:
    """The default Jacobian is jax.jacfwd of forward, for any leading axes."""
    jacobian = Sinh().jacobian(points)
    assert jacobian.shape == (2, 5, PARAM_COUNT, PARAM_COUNT)
    expected = jnp.cosh(points)[..., None] * jnp.eye(PARAM_COUNT)
    np.testing.assert_allclose(jacobian, expected, rtol=1e-14)


def test_affine_jacobian_rows_follow_the_outputs(shear: Shear, points: Array) -> None:
    """Row i of the Jacobian holds the derivatives of the output yᵢ."""
    np.testing.assert_allclose(
        shear.jacobian(points[0, 0]), SHEAR_MATRIX, rtol=1e-14, atol=1e-15
    )


def test_inverse_round_trips(shear: Shear, points: Array) -> None:
    """Each member and their composition invert exactly, up to rounding."""
    for transform in (Sinh(), shear, Compose((shear, Sinh()))):
        np.testing.assert_allclose(
            transform.inverse(transform.forward(points)), points, rtol=1e-12
        )


def test_compose_applies_its_members_from_left_to_right(
    shear: Shear, points: Array
) -> None:
    """Compose((first, second)) applies first, then second."""
    composed = Compose((shear, Sinh()))
    np.testing.assert_allclose(
        composed.forward(points), jnp.sinh(shear.forward(points)), rtol=1e-14
    )
    assert composed.description == f"shear{COMPOSITION_SEPARATOR}sinh"
    assert Compose([Sinh(), shear]).description == "sinh | shear"


def test_compose_log_det_and_jacobian_match_jacfwd(shear: Shear, points: Array) -> None:
    """The composition's log-determinant and Jacobian agree with jax.jacfwd."""
    composed = Compose((Sinh(), shear, Compose((Sinh(),))))
    direct = jnp.vectorize(
        jax.jacfwd(composed.forward), signature="(param)->(param,param)"
    )(points)
    np.testing.assert_allclose(composed.jacobian(points), direct, rtol=1e-12)
    np.testing.assert_allclose(
        composed.log_abs_det_jacobian(points),
        jnp.linalg.slogdet(direct)[1],
        rtol=1e-12,
    )


def test_compose_rejects_empty_and_foreign_members(shear: Shear) -> None:
    """A composition needs at least one member, each a Transform."""
    with pytest.raises(ValueError, match="at least one"):
        Compose(())
    with pytest.raises(TypeError, match="must be a Transform, got str"):
        Compose((shear, "sinh"))


def test_map_fisher_is_the_pullback_by_the_inverse(shear: Shear, points: Array) -> None:
    """Γ_y = Jᵀ Γ_x J with J = ∂x/∂y, which equals J_f⁻ᵀ Γ_x J_f⁻¹."""
    composed = Compose((shear, Sinh()))
    factor = jax.random.normal(jax.random.key(5), (PARAM_COUNT, PARAM_COUNT))
    fisher = factor @ factor.T + jnp.eye(PARAM_COUNT)
    mapped = composed.map_fisher(jnp.broadcast_to(fisher, (2, 5, 3, 3)), points)
    forward_jacobian = np.asarray(composed.jacobian(points))
    inverse_jacobian = np.linalg.inv(forward_jacobian)
    expected = (
        np.swapaxes(inverse_jacobian, -1, -2) @ np.asarray(fisher) @ inverse_jacobian
    )
    np.testing.assert_allclose(mapped, expected, rtol=1e-10)
    default = Transform.map_fisher(composed, fisher, points[0, 0])
    np.testing.assert_allclose(default, expected[0, 0], rtol=1e-10)


def test_map_fisher_keeps_the_information_form(shear: Shear, points: Array) -> None:
    """dxᵀ Γ_x dx equals dyᵀ Γ_y dy for dy = (∂f/∂x) dx."""
    composed = Compose((Sinh(), shear))
    point = points[1, 2]
    fisher = jnp.diag(jnp.array([4.0, 0.5, 2.0])) + 0.3
    step = jnp.array([1e-3, -2e-3, 5e-4])
    mapped_step = composed.jacobian(point) @ step
    mapped_fisher = composed.map_fisher(fisher, point)
    assert float(mapped_step @ mapped_fisher @ mapped_step) == pytest.approx(
        float(step @ fisher @ step), rel=1e-12
    )


def test_map_space_is_undefined_by_default() -> None:
    """A transform that does not map spaces says so."""
    space = ParamSpace.model_validate({"parameters": [{"name": "x0"}]})
    with pytest.raises(NotImplementedError, match="Sinh does not define"):
        Compose((Sinh(),)).map_space(space)


def test_map_sample_set_moves_samples_and_densities(shear: Shear) -> None:
    """Densities lose ln|det J|; likelihoods, weights and n_eff stay."""
    mean = np.array([0.5, -1.0, 2.0])
    covariance = np.array([[1.0, 0.3, 0.0], [0.3, 2.0, -0.4], [0.0, -0.4, 0.8]])
    draws = np.random.default_rng(4).multivariate_normal(mean, covariance, size=50)
    log_density = stats.multivariate_normal(mean, covariance).logpdf(draws)
    sample_set = SampleSet(
        draws,
        log_weights=-0.01 * np.arange(50),
        n_eff=30.0,
        log_like=-np.arange(50.0),
        log_prior=log_density,
        log_q=log_density,
    )
    mapped = shear.map_sample_set(sample_set)
    assert mapped.log_q is not None
    moved = draws @ SHEAR_MATRIX.T + SHEAR_OFFSET
    np.testing.assert_allclose(mapped.samples[0, 0], moved, rtol=1e-14)
    # * The image of N(μ, Σ) under y = A x + b is N(A μ + b, A Σ Aᵀ).
    image = stats.multivariate_normal(
        SHEAR_MATRIX @ mean + SHEAR_OFFSET, SHEAR_MATRIX @ covariance @ SHEAR_MATRIX.T
    )
    np.testing.assert_allclose(mapped.log_q[0, 0], image.logpdf(moved), rtol=1e-12)
    np.testing.assert_allclose(np.asarray(mapped.log_prior), mapped.log_q, rtol=1e-14)
    assert eqx.tree_equal(
        (mapped.log_like, mapped.log_weights, mapped.n_eff),
        (sample_set.log_like, sample_set.log_weights, sample_set.n_eff),
    )


def test_map_sample_set_keeps_absent_densities_absent() -> None:
    """Without log densities, only the samples move."""
    sample_set = SampleSet(np.linspace(-1.0, 1.0, 12).reshape(4, 3))
    mapped = Sinh().map_sample_set(sample_set)
    assert mapped.log_prior is None
    assert mapped.log_q is None
    np.testing.assert_allclose(mapped.samples, jnp.sinh(sample_set.samples))


def test_transforms_run_under_jit_vmap_and_grad(shear: Shear, points: Array) -> None:
    """Transforms are pytrees that jit, vmap and differentiate."""
    composed = Compose((shear, Sinh()))
    compiled = jax.jit(lambda transform, values: transform.forward(values))
    np.testing.assert_allclose(
        compiled(composed, points), composed.forward(points), rtol=1e-14
    )
    mapped = jax.vmap(composed.log_abs_det_jacobian)(points)
    np.testing.assert_allclose(
        mapped, composed.log_abs_det_jacobian(points), rtol=1e-14
    )
    gradient = jax.grad(lambda point: jnp.sum(composed.forward(point)))(points[0, 0])
    expected = composed.jacobian(points[0, 0]).sum(axis=0)
    np.testing.assert_allclose(gradient, expected, rtol=1e-12)
    assert math.isfinite(float(composed.log_abs_det_jacobian(points[0, 0])))
