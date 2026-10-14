"""Tests for the moment-matched Gaussian estimator."""

import math
from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import stats

from mixed_signals import zoo
from mixed_signals.core.space import ParamSpace
from mixed_signals.errors import PrecisionWarning, ValidityWarning
from mixed_signals.estimators import GaussianDensity, GaussianSettings

type GridIntegral = Callable[..., float]

STANDARD_ERRORS = 5.0
SAMPLE_COUNT = 40_000
REAL_PAIR = ParamSpace.model_validate({"parameters": [{"name": "x0"}, {"name": "x1"}]})


def correlated_case() -> zoo.ZooCase:
    """Three correlated Gaussian parameters of different widths."""
    covariance = np.array([[4.0, 1.2, -0.3], [1.2, 1.0, 0.2], [-0.3, 0.2, 0.25]])
    return zoo.gaussians([[1.0, -2.0, 0.5]], [covariance], names=["only"])


# * Moments and density


def test_fit_recovers_the_zoo_moments() -> None:
    """Mean and covariance agree with the zoo truth within five standard errors."""
    case = correlated_case()
    samples = case.sample(jax.random.key(0), SAMPLE_COUNT).samples[0, 0]
    fitted = GaussianDensity.fit(samples, None, case.space)
    truth = case.posteriors[0]
    assert isinstance(truth, zoo.distributions.Gaussian)
    true_covariance = np.asarray(truth.covariance)
    widths = np.sqrt(np.diag(true_covariance))
    mean_error = widths / math.sqrt(SAMPLE_COUNT)
    assert np.all(
        np.abs(np.asarray(fitted.mean) - np.asarray(truth.mean))
        < STANDARD_ERRORS * mean_error
    )
    covariance_error = np.sqrt(
        (np.outer(widths**2, widths**2) + true_covariance**2) / SAMPLE_COUNT
    )
    assert np.all(
        np.abs(np.asarray(fitted.covariance) - true_covariance)
        < STANDARD_ERRORS * covariance_error
    )


def test_moments_are_the_weighted_numpy_estimates() -> None:
    """The weighted mean and covariance are numpy's with reliability weights."""
    rng = np.random.default_rng(1)
    samples = rng.normal(size=(300, 2)) * [2.0, 0.5]
    log_weights = rng.normal(size=300)
    fitted = GaussianDensity.fit(samples, log_weights, REAL_PAIR)
    weights = np.exp(log_weights)
    np.testing.assert_allclose(
        fitted.mean, np.average(samples, axis=0, weights=weights), rtol=1e-12
    )
    np.testing.assert_allclose(
        fitted.covariance,
        np.cov(samples, rowvar=False, aweights=weights),
        rtol=1e-12,
    )


def test_log_density_is_the_normal_of_the_fitted_moments() -> None:
    """On the real line the density is SciPy's normal with the fitted moments."""
    case = correlated_case()
    samples = case.sample(jax.random.key(2), 2_000).samples[0, 0]
    fitted = GaussianDensity.fit(samples, None, case.space)
    points = np.asarray(samples[:7]) + 0.4
    expected = stats.multivariate_normal(
        np.asarray(fitted.mean), np.asarray(fitted.covariance)
    ).logpdf(points)
    np.testing.assert_allclose(fitted.log_density(points), expected, rtol=1e-12)
    assert fitted.log_density(points[0]).shape == ()
    assert fitted.log_density(points.reshape(7, 1, 3)).shape == (7, 1)


def test_gravitational_wave_scales_keep_the_density() -> None:
    """At 3 mHz ± 1e-11 Hz the log density shifts by the log Jacobian only."""
    rng = np.random.default_rng(3)
    standard = rng.multivariate_normal([0.0, 0.0], [[1.0, 0.9], [0.9, 1.0]], 5_000)
    scales = np.array([1e-11, 1e-23])
    physical = np.array([3e-3, 1e-22]) + scales * standard
    unit = GaussianDensity.fit(standard, None, REAL_PAIR)
    scaled = GaussianDensity.fit(physical, None, REAL_PAIR)
    shift = -np.sum(np.log(scales))
    # * At 3 mHz a float64 step is 4e-8 of the spread, so values agree to about 1e-7.
    np.testing.assert_allclose(
        scaled.log_density(physical[:5]),
        unit.log_density(standard[:5]) + shift,
        rtol=0.0,
        atol=1e-6,
    )


# * Circles, spheres and intervals


def test_circles_are_unwrapped_and_their_density_wrapped() -> None:
    """A circle across the boundary gets unwrapped moments and a wrapped density."""
    case = zoo.wrapped_gaussians(0.0, width=0.6, param_count=2, correlation=0.5)
    samples = case.sample(jax.random.key(4), SAMPLE_COUNT).samples[0, 0]
    fitted = GaussianDensity.fit(samples, None, case.space)
    distance = min(float(fitted.mean[0]), 2 * math.pi - float(fitted.mean[0]))
    assert distance < STANDARD_ERRORS * 0.6 / math.sqrt(SAMPLE_COUNT)
    assert float(jnp.sqrt(fitted.covariance[0, 0])) == pytest.approx(0.6, rel=0.02)
    points = np.asarray(samples[:6])
    normal = stats.multivariate_normal(
        np.asarray(fitted.mean), np.asarray(fitted.covariance)
    )
    windings = np.arange(-10, 11)[:, None, None] * np.array([2 * math.pi, 0.0])
    expected = np.log(np.sum(normal.pdf(points + windings), axis=0))
    np.testing.assert_allclose(fitted.log_density(points), expected, rtol=1e-10)
    np.testing.assert_allclose(
        fitted.log_density(points + np.array([2 * math.pi, 0.0])),
        expected,
        rtol=1e-10,
    )


def test_wrapped_density_integrates_to_one(grid_integral: GridIntegral) -> None:
    """Over one period of the circle, the wrapped density has unit mass."""
    case = zoo.wrapped_gaussians(1.0, width=1.5, param_count=2, correlation=-0.4)
    samples = case.sample(jax.random.key(5), 5_000).samples[0, 1]
    fitted = GaussianDensity.fit(samples, None, case.space)
    assert fitted.windings[0] >= 2
    mass = grid_integral(
        lambda points: np.exp(np.asarray(fitted.log_density(points))),
        (0.0, 2 * math.pi, 64),
        (-12.0, 12.0, 96),
    )
    assert mass == pytest.approx(1.0, abs=1e-10)


def test_sphere_axes_follow_their_topologies() -> None:
    """A sphere's azimuth is unwrapped across 2π, its polar axis is taken as is."""
    case = zoo.von_mises_fisher(0.0, width=0.2, colatitude=1.0)
    samples = case.sample(jax.random.key(6), 10_000).samples[0, 0]
    fitted = GaussianDensity.fit(samples, None, case.space)
    azimuth = float(fitted.mean[0])
    assert min(azimuth, 2 * math.pi - azimuth) < 0.02
    assert float(fitted.mean[1]) == pytest.approx(math.pi / 2 - 1.0, abs=0.02)
    assert np.all(np.isfinite(np.asarray(fitted.log_density(samples[:5]))))


def test_interval_bounds_do_not_cut_the_gaussian() -> None:
    """The moment-matched Gaussian keeps its mass beyond an interval bound."""
    case = zoo.railed_gaussians([0.5], param_count=1)
    samples = case.sample(jax.random.key(7), 2_000).samples[0, 0]
    fitted = GaussianDensity.fit(samples, None, case.space)
    assert math.isfinite(float(fitted.log_density(np.array([-0.5]))))


# * Draws


def test_draws_follow_the_fit_and_wrap_onto_the_circle() -> None:
    """Draws have the fitted moments, and circle values lie on the circle."""
    case = zoo.wrapped_gaussians(0.0, width=0.5, param_count=2, correlation=0.3)
    fitted = GaussianDensity.fit(
        case.sample(jax.random.key(8), 5_000).samples[0, 0], None, case.space
    )
    draws = fitted.sample(jax.random.key(9), (SAMPLE_COUNT,))
    assert draws.shape == (SAMPLE_COUNT, 2)
    assert bool(jnp.all((draws[:, 0] >= 0) & (draws[:, 0] < 2 * math.pi)))
    refit = GaussianDensity.fit(draws, None, case.space)
    np.testing.assert_allclose(
        refit.covariance, fitted.covariance, rtol=0.05, atol=0.01
    )
    assert fitted.sample(jax.random.key(9)).shape == (2,)


# * Validity


@pytest.mark.parametrize(
    ("samples", "log_weights", "message"),
    [
        (np.c_[np.zeros(20), np.arange(20.0)], None, "'x0' has no spread"),
        (np.arange(8.0).reshape(4, 2), [0.0, -np.inf, -np.inf, -np.inf], "fewer than"),
        (np.c_[np.arange(20.0), 2 * np.arange(20.0)], None, "collinear"),
    ],
    ids=["no spread", "one sample", "collinear"],
)
def test_singular_covariances_give_nan_and_warn(
    samples: np.ndarray, log_weights: list[float] | None, message: str
) -> None:
    """A singular covariance gives a NaN Gaussian with a reason."""
    weights = None if log_weights is None else np.asarray(log_weights)
    with pytest.warns(ValidityWarning, match=message):
        fitted = GaussianDensity.fit(samples, weights, REAL_PAIR)
    assert math.isnan(float(fitted.log_density(np.zeros(2))))


def test_narrow_inputs_warn() -> None:
    """float32 samples are cast with a PrecisionWarning."""
    samples = np.random.default_rng(10).normal(size=(50, 2)).astype(np.float32)
    with pytest.warns(PrecisionWarning, match="float32"):
        GaussianDensity.fit(samples, None, REAL_PAIR)


def test_settings_have_nothing_to_choose() -> None:
    """The Gaussian estimator writes an empty settings object."""
    fitted = GaussianDensity.fit(np.eye(3)[:, :2] + [0.0, 0.5], None, REAL_PAIR)
    assert fitted.settings == GaussianSettings()
    assert fitted.resolved_settings().to_json() == "{}"


# * Transformations


def test_vmap_fits_one_gaussian_per_problem() -> None:
    """Batched fits under vmap match concrete fits, wider winding sums included."""
    case = zoo.wrapped_gaussians(0.5, width=0.4, param_count=2, correlation=0.2)
    batch = case.sample(jax.random.key(11), 3_000, batch_size=3).samples[:, 0]
    batched = jax.vmap(lambda values: GaussianDensity.fit(values, None, case.space))(
        batch
    )
    concrete = GaussianDensity.fit(batch[1], None, case.space)
    assert batched.windings[0] > concrete.windings[0]
    values = jax.vmap(lambda estimator, points: estimator.log_density(points))(
        batched, batch[:, :5]
    )
    np.testing.assert_allclose(
        values[1], concrete.log_density(batch[1, :5]), rtol=1e-12
    )


def test_log_density_runs_under_jit() -> None:
    """The fitted module passes through jit with its points."""
    fitted = GaussianDensity.fit(
        np.random.default_rng(12).normal(size=(100, 2)), None, REAL_PAIR
    )
    compiled = jax.jit(lambda estimator, points: estimator.log_density(points))
    np.testing.assert_allclose(
        compiled(fitted, jnp.zeros((3, 2))), fitted.log_density(np.zeros((3, 2)))
    )
