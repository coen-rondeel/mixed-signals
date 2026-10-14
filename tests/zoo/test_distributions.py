"""Tests for the zoo distributions: densities, normalisation, draws and checks."""

import math
from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.typing import ArrayLike
from scipy import stats

from mixed_signals.core.topology import AZIMUTH_PERIOD, PolarForm, unit_vectors
from mixed_signals.zoo import closed_forms
from mixed_signals.zoo.distributions import (
    Gaussian,
    GaussianMixture,
    TruncatedGaussian,
    TwistedGaussian,
    VonMisesFisher,
    WrappedGaussian,
    ZooDistribution,
)

type GridIntegral = Callable[..., float]

STANDARD_ERRORS = 5.0
SAMPLE_COUNT = 100_000


def assert_mean_within_error(draws: np.ndarray, expected: ArrayLike) -> None:
    """Check a sample mean against its expectation within five standard errors."""
    standard_error = draws.std() / math.sqrt(len(draws))
    assert (
        abs(draws.mean() - float(jnp.asarray(expected)))
        < STANDARD_ERRORS * standard_error
    )


def density_of(distribution: ZooDistribution) -> Callable[[np.ndarray], np.ndarray]:
    """Return the density of a zoo distribution as a function of grid points."""

    def density(points: np.ndarray) -> np.ndarray:
        return np.exp(np.asarray(distribution.log_prob(jnp.asarray(points))))

    return density


@pytest.fixture
def key() -> jax.Array:
    """A fixed JAX key for draws."""
    return jax.random.key(2026)


# * Gaussian


def test_gaussian_log_prob_matches_scipy() -> None:
    """The log density agrees with scipy.stats.multivariate_normal."""
    mean = np.array([0.5, -1.0, 2.0])
    covariance = np.array([[2.0, 0.7, 0.1], [0.7, 1.0, -0.2], [0.1, -0.2, 0.5]])
    points = np.random.default_rng(1).normal(size=(4, 5, 3))
    np.testing.assert_allclose(
        Gaussian(mean, covariance).log_prob(points),
        stats.multivariate_normal(mean, covariance).logpdf(points),
        rtol=1e-12,
    )


def test_gaussian_draws_have_the_given_moments(key: jax.Array) -> None:
    """Sample means and covariances match μ and Σ."""
    mean = np.array([0.5, -1.0])
    covariance = np.array([[2.0, 0.7], [0.7, 1.0]])
    draws = np.asarray(Gaussian(mean, covariance).sample(key, (SAMPLE_COUNT,)))
    assert draws.shape == (SAMPLE_COUNT, 2)
    for axis in range(2):
        assert_mean_within_error(draws[:, axis], mean[axis])
    np.testing.assert_allclose(np.cov(draws.T), covariance, atol=0.04)


def test_gaussian_works_at_gravitational_wave_scales(key: jax.Array) -> None:
    """A frequency posterior with σ_f/f ~ 1e-8 keeps its width in float64."""
    frequency, width = 3e-3, 3e-11
    gaussian = Gaussian(np.array([frequency]), np.array([[width**2]]))
    draws = np.asarray(gaussian.sample(key, (SAMPLE_COUNT,)))[:, 0]
    assert draws.std() == pytest.approx(width, rel=0.02)
    assert float(gaussian.log_prob(jnp.array([frequency]))) == pytest.approx(
        -math.log(width) - 0.5 * math.log(2 * math.pi), rel=1e-12
    )


def test_log_prob_runs_under_jit() -> None:
    """Distributions are pytrees, so their log densities compile under equinox jit."""
    gaussian = Gaussian(np.zeros(2), np.eye(2))
    points = jnp.ones((3, 2))
    np.testing.assert_allclose(
        eqx.filter_jit(gaussian.log_prob)(points), gaussian.log_prob(points)
    )


# * Twisted Gaussian


def test_twisted_gaussian_integrates_to_one(grid_integral: GridIntegral) -> None:
    """The unit Jacobian of the twist keeps the density normalised."""
    banana = TwistedGaussian(0.03, 100.0, 2)
    normalisation = grid_integral(
        density_of(banana), (-80.0, 80.0, 80), (-30.0, 200.0, 460)
    )
    assert normalisation == pytest.approx(1.0, abs=1e-9)


def test_twisted_gaussian_draws_have_the_closed_form_moments(key: jax.Array) -> None:
    """Draws have zero mean, variances diag(v, 1 + 2b²v², 1) and no correlation."""
    banana = TwistedGaussian(0.1, 4.0, 3)
    draws = np.asarray(banana.sample(key, (SAMPLE_COUNT,)))
    _, covariance = closed_forms.twisted_gaussian_moments(0.1, 4.0, 3)
    for axis in range(3):
        assert_mean_within_error(draws[:, axis], 0.0)
        centred = draws[:, axis] ** 2
        assert_mean_within_error(centred, covariance[axis, axis])
    assert_mean_within_error(draws[:, 0] * draws[:, 1], 0.0)


def test_twisted_gaussian_log_prob_untwists_the_point() -> None:
    """The log density is the base Gaussian at the untwisted point."""
    banana = TwistedGaussian(0.5, 2.0, 3)
    point = np.array([1.2, 0.3, -0.4])
    untwisted = point.copy()
    untwisted[1] -= 0.5 * (point[0] ** 2 - 2.0)
    expected = stats.multivariate_normal(np.zeros(3), np.diag([2.0, 1, 1])).logpdf(
        untwisted
    )
    assert float(banana.log_prob(point)) == pytest.approx(expected, rel=1e-12)


# * Gaussian mixture


def test_gaussian_mixture_log_prob_matches_scipy() -> None:
    """The log density is the log of the weighted sum of component densities."""
    covariances = [np.eye(2), np.array([[2.0, 0.3], [0.3, 0.5]])]
    mixture = GaussianMixture(
        np.array([3.0, 7.0]), np.array([[-1.5, 0.0], [1.5, 0.5]]), covariances
    )
    points = np.random.default_rng(2).normal(size=(6, 2))
    expected = np.log(
        0.3 * stats.multivariate_normal([-1.5, 0.0], covariances[0]).pdf(points)
        + 0.7 * stats.multivariate_normal([1.5, 0.5], covariances[1]).pdf(points)
    )
    np.testing.assert_allclose(mixture.log_prob(points), expected, rtol=1e-12)


def test_gaussian_mixture_draws_follow_the_weights(key: jax.Array) -> None:
    """Components are drawn in proportion to their normalised weights."""
    mixture = GaussianMixture(
        np.array([0.2, 0.8]), np.array([[-30.0], [30.0]]), np.ones((2, 1, 1))
    )
    draws = np.asarray(mixture.sample(key, (SAMPLE_COUNT,)))[:, 0]
    assert_mean_within_error((draws > 0).astype(float), 0.8)


# * Truncated Gaussian


def test_truncated_gaussian_draws_respect_bounds_and_moments(key: jax.Array) -> None:
    """Draws stay in each interval, also 9σ into a tail, with closed-form moments."""
    truncated = TruncatedGaussian(
        np.array([1.5, 0.0, -2.0, 0.0]),
        np.ones(4),
        np.array([0.0, -np.inf, 7.0, -np.inf]),
        np.array([np.inf, np.inf, np.inf, -1.0]),
    )
    draws = np.asarray(truncated.sample(key, (SAMPLE_COUNT,)))
    bounds = [(0.0, np.inf), (-np.inf, np.inf), (7.0, np.inf), (-np.inf, -1.0)]
    for axis, (centre, (lower, upper)) in enumerate(
        zip([1.5, 0.0, -2.0, 0.0], bounds, strict=True)
    ):
        assert np.all((draws[:, axis] >= lower) & (draws[:, axis] <= upper))
        mean, std = closed_forms.truncated_normal_moments(centre, 1.0, lower, upper)
        assert_mean_within_error(draws[:, axis], mean)
        assert draws[:, axis].std() == pytest.approx(std, rel=0.02)


def test_truncated_gaussian_log_prob_matches_scipy() -> None:
    """Inside the box the density is the product of truncated normals; outside −inf."""
    truncated = TruncatedGaussian(
        np.array([1.5, 0.0]),
        np.array([1.0, 2.0]),
        np.array([0.0, -np.inf]),
        np.array([np.inf, np.inf]),
    )
    point = np.array([0.7, 0.3])
    expected = stats.truncnorm(-1.5, np.inf, loc=1.5).logpdf(0.7) + stats.norm.logpdf(
        0.3, scale=2.0
    )
    assert float(truncated.log_prob(point)) == pytest.approx(expected, rel=1e-12)
    assert float(truncated.log_prob(np.array([-0.1, 0.3]))) == -math.inf


# * Wrapped Gaussian


def test_wrapped_gaussian_integrates_to_one_over_the_circle(
    grid_integral: GridIntegral,
) -> None:
    """The winding sum gives a density on the circle times the real line."""
    wrapped = WrappedGaussian(
        np.array([0.0, 0.0]),
        np.array([[0.36, 0.3], [0.3, 1.0]]),
        lower=-math.pi,
        period=AZIMUTH_PERIOD,
    )
    normalisation = grid_integral(
        density_of(wrapped), (-math.pi, math.pi, 32), (-12.0, 12.0, 48)
    )
    assert normalisation == pytest.approx(1.0, abs=1e-10)


def test_wide_wrapped_gaussian_sums_enough_windings(
    grid_integral: GridIntegral,
) -> None:
    """A wrapped normal wider than its period still integrates to one."""
    wrapped = WrappedGaussian(np.array([1.0]), np.array([[9.0]]), period=1.0)
    assert wrapped.windings == math.ceil(12 * 3.0 / 1.0) + 1
    normalisation = grid_integral(density_of(wrapped), (0.0, 1.0, 4))
    assert normalisation == pytest.approx(1.0, abs=1e-10)


def test_wrapped_gaussian_density_is_periodic() -> None:
    """Points one period apart have the same density."""
    wrapped = WrappedGaussian(np.array([0.2, 1.0]), np.array([[0.5, 0.1], [0.1, 1.0]]))
    point = jnp.array([0.4, 0.9])
    shifted = point.at[0].add(AZIMUTH_PERIOD)
    assert float(wrapped.log_prob(shifted)) == pytest.approx(
        float(wrapped.log_prob(point)), rel=1e-12
    )


def test_wrapped_gaussian_draws_lie_on_the_circle(key: jax.Array) -> None:
    """Draws centred on the cut fill both ends of [0, 2π) with the right spread."""
    wrapped = WrappedGaussian(np.zeros(1), np.array([[0.36]]))
    angles = np.asarray(wrapped.sample(key, (SAMPLE_COUNT,)))[:, 0]
    assert np.all((angles >= 0) & (angles < AZIMUTH_PERIOD))
    assert np.mean(angles < 1) == pytest.approx(
        np.mean(angles > AZIMUTH_PERIOD - 1), abs=0.01
    )
    assert_mean_within_error(
        np.cos(angles),
        closed_forms.wrapped_normal_resultant_length(0.6, AZIMUTH_PERIOD),
    )


# * von Mises-Fisher


@pytest.mark.parametrize("polar_form", list(PolarForm))
def test_von_mises_fisher_integrates_to_one_in_every_polar_form(
    polar_form: PolarForm, grid_integral: GridIntegral
) -> None:
    """The area element of each polar form makes the coordinate density normalised."""
    direction = np.array([math.sin(0.3), 0.0, math.cos(0.3)])
    distribution = VonMisesFisher(direction, 20.0, polar_form=polar_form)
    lower, upper = polar_form.bounds
    normalisation = grid_integral(
        density_of(distribution), (0.0, AZIMUTH_PERIOD, 32), (lower, upper, 64)
    )
    assert normalisation == pytest.approx(1.0, abs=1e-9)


def test_von_mises_fisher_log_prob_matches_scipy() -> None:
    """In the sine form the density is the surface density of SciPy's distribution."""
    direction = np.array([0.1, -0.2, 0.97])
    unit = direction / np.linalg.norm(direction)
    distribution = VonMisesFisher(direction, 60.0, polar_form="sin_latitude")
    points = np.array([[0.3, 0.95], [2.0, 0.9], [5.9, 0.99]])
    vectors, _, _ = unit_vectors(
        jnp.asarray(points[:, 0]), jnp.asarray(points[:, 1]), PolarForm.SIN_LATITUDE
    )
    np.testing.assert_allclose(
        distribution.log_prob(points),
        stats.vonmises_fisher(unit, 60.0).logpdf(np.asarray(vectors)),
        rtol=1e-11,
    )


def test_von_mises_fisher_draws_have_the_mean_resultant_length(key: jax.Array) -> None:
    """The mean unit vector points along μ with length coth κ − 1/κ."""
    direction = np.array([math.sin(0.2), 0.0, math.cos(0.2)])
    distribution = VonMisesFisher(direction, 50.0, euclidean_count=2)
    draws = distribution.sample(key, (SAMPLE_COUNT,))
    assert draws.shape == (SAMPLE_COUNT, 4)
    vectors, _, inside = unit_vectors(draws[:, 0], draws[:, 1], PolarForm.LATITUDE)
    assert bool(jnp.all(inside))
    along = np.asarray(vectors @ jnp.asarray(direction))
    assert_mean_within_error(
        along, closed_forms.von_mises_fisher_resultant_length(50.0)
    )
    for axis in (2, 3):
        assert_mean_within_error(np.asarray(draws[:, axis]), 0.0)


def test_von_mises_fisher_polar_forms_describe_the_same_points(key: jax.Array) -> None:
    """The same key gives the same unit vectors in every polar form."""
    direction = np.array([0.0, 0.0, 1.0])
    vectors = []
    for polar_form in PolarForm:
        draws = VonMisesFisher(direction, 5.0, polar_form=polar_form).sample(key, (50,))
        unit, _, _ = unit_vectors(draws[:, 0], draws[:, 1], polar_form)
        vectors.append(np.asarray(unit))
    for other in vectors[1:]:
        np.testing.assert_allclose(other, vectors[0], atol=1e-12)


def test_von_mises_fisher_density_is_zero_outside_the_polar_range() -> None:
    """A polar value outside the range of its form has log density −inf."""
    distribution = VonMisesFisher(np.array([0.0, 0.0, 1.0]), 5.0, polar_form="latitude")
    assert float(distribution.log_prob(jnp.array([0.0, 2.0]))) == -math.inf


# * Parameter checks


def test_distribution_parameters_are_checked() -> None:
    """Invalid parameters fail with messages that name them."""
    with pytest.raises(ValueError, match="covariance must be shaped"):
        Gaussian(np.zeros(2), np.eye(3))
    with pytest.raises(ValueError, match="positive definite"):
        Gaussian(np.zeros(2), -np.eye(2))
    with pytest.raises(ValueError, match="covariance must be finite"):
        Gaussian(np.zeros(2), np.array([[np.nan, 0.0], [0.0, 1.0]]))
    with pytest.raises(ValueError, match="mean must be finite"):
        Gaussian(np.array([np.nan, 0.0]), np.eye(2))
    with pytest.raises(ValueError, match="at least 2"):
        TwistedGaussian(0.1, 1.0, 1)
    with pytest.raises(ValueError, match="variance must be finite and positive"):
        TwistedGaussian(0.1, 0.0, 2)
    with pytest.raises(ValueError, match="weights must be positive"):
        GaussianMixture(np.array([1.0, 0.0]), np.zeros((2, 1)), np.ones((2, 1, 1)))
    with pytest.raises(ValueError, match="means must be shaped"):
        GaussianMixture(np.ones(2), np.zeros((3, 1)), np.ones((2, 1, 1)))
    with pytest.raises(ValueError, match="covariances must be shaped"):
        GaussianMixture(np.ones(2), np.zeros((2, 1)), np.ones((2, 2, 2)))
    with pytest.raises(ValueError, match="std must be positive"):
        TruncatedGaussian(np.zeros(1), np.zeros(1), np.zeros(1), np.ones(1))
    with pytest.raises(ValueError, match="lower bound must lie below"):
        TruncatedGaussian(np.zeros(1), np.ones(1), np.ones(1), np.zeros(1))
    with pytest.raises(ValueError, match="upper must be shaped"):
        TruncatedGaussian(np.zeros(1), np.ones(1), np.zeros(1), np.ones(2))
    with pytest.raises(ValueError, match="circle 2 is not an axis"):
        WrappedGaussian(np.zeros(2), np.eye(2), circle=2)
    with pytest.raises(ValueError, match="period must be finite and positive"):
        WrappedGaussian(np.zeros(1), np.eye(1), period=-1.0)
    with pytest.raises(ValueError, match="zero vector"):
        VonMisesFisher(np.zeros(3), 1.0)
    with pytest.raises(ValueError, match="concentration must be finite and positive"):
        VonMisesFisher(np.ones(3), 0.0)
    with pytest.raises(ValueError, match="'east' is not a valid PolarForm"):
        VonMisesFisher(np.ones(3), 1.0, polar_form="east")
