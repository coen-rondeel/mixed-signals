"""Tests for the Gaussian surrogates of the diagnostics."""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import stats

from mixed_signals import zoo
from mixed_signals.core.samples import circular_mean
from mixed_signals.errors import ValidityWarning
from mixed_signals.estimators import GaussianDensity
from mixed_signals.nulls.surrogates import gaussian_surrogates, surrogate_size
from mixed_signals.prng import derive_key

STANDARD_ERRORS = 5.0


def test_surrogates_follow_the_moment_matched_gaussian() -> None:
    """The pooled sets have the weighted mean and covariance of the samples."""
    case = zoo.banana(0.03)
    samples = case.sample(jax.random.key(0), 2_000).samples[0, 0]
    surrogates = gaussian_surrogates(
        samples, None, case.space, size=500, draws=40, key=jax.random.key(1)
    )
    assert surrogates.shape == (40, 500, 2)
    fitted = GaussianDensity.fit(samples, None, case.space)
    pooled = np.asarray(surrogates).reshape(-1, 2)
    errors = np.sqrt(np.diagonal(np.asarray(fitted.covariance)) / pooled.shape[0])
    assert np.all(
        np.abs(pooled.mean(axis=0) - np.asarray(fitted.mean)) < STANDARD_ERRORS * errors
    )
    np.testing.assert_allclose(
        np.cov(pooled.T), np.asarray(fitted.covariance), rtol=0.03, atol=0.03
    )


def test_each_set_draws_from_its_own_key() -> None:
    """Set i draws from derive_key(key, "surrogate", i), whatever the number of sets."""
    case = zoo.shifted_gaussians(1.0, param_count=3)
    samples = case.sample(jax.random.key(2), 300).samples[0, 0]
    key = jax.random.key(3)
    few = gaussian_surrogates(samples, None, case.space, size=50, draws=2, key=key)
    many = gaussian_surrogates(samples, None, case.space, size=50, draws=5, key=key)
    np.testing.assert_array_equal(few, many[:2])
    fitted = GaussianDensity.fit(samples, None, case.space)
    np.testing.assert_array_equal(
        many[4], fitted.sample(derive_key(key, "surrogate", 4), (50,))
    )


def test_weights_enter_the_fit() -> None:
    """Weighted samples give the surrogates of their weighted moments."""
    rng = np.random.default_rng(4)
    case = zoo.shifted_gaussians(0.0, param_count=1)
    samples = rng.normal(size=(4_000, 1))
    log_weights = -0.5 * (samples[:, 0] - 1.0) ** 2 * 3.0
    surrogates = gaussian_surrogates(
        samples, log_weights, case.space, size=1_000, draws=20, key=jax.random.key(5)
    )
    fitted = GaussianDensity.fit(samples, log_weights, case.space)
    assert float(jnp.mean(surrogates)) == pytest.approx(float(fitted.mean[0]), abs=0.02)


def test_circles_stay_on_their_period() -> None:
    """Circle values are wrapped back onto the circle and keep the mean direction."""
    case = zoo.wrapped_gaussians(0.0, width=0.8, param_count=2)
    samples = case.sample(jax.random.key(6), 2_000).samples[0, 0]
    surrogates = np.asarray(
        gaussian_surrogates(
            samples, None, case.space, size=1_000, draws=4, key=jax.random.key(7)
        )
    )
    assert surrogates[..., 0].min() >= 0.0
    assert surrogates[..., 0].max() < 2 * math.pi
    direction = float(
        circular_mean(
            jnp.asarray(surrogates[..., 0].ravel()), jnp.zeros(4_000), 2 * math.pi, 0.0
        )
    )
    distance = abs((direction + math.pi) % (2 * math.pi) - math.pi)
    assert distance < 0.1


def test_a_singular_posterior_gives_nan_surrogates() -> None:
    """A parameter without spread gives NaN sets and a ValidityWarning."""
    case = zoo.shifted_gaussians(0.0, param_count=2)
    samples = np.array(case.sample(jax.random.key(8), 100).samples[0, 0])
    samples[:, 1] = 2.0
    with pytest.warns(ValidityWarning, match="no spread"):
        surrogates = gaussian_surrogates(
            samples, None, case.space, size=10, draws=2, key=jax.random.key(9)
        )
    assert bool(jnp.all(jnp.isnan(surrogates)))


@pytest.mark.parametrize(
    ("size", "draws", "error"),
    [
        (0, 2, ValueError),
        (10, 0, ValueError),
        (2.5, 2, TypeError),
        (10, True, TypeError),
    ],
)
def test_counts_are_checked(size: object, draws: object, error: type) -> None:
    """The set size and the number of sets are integers of at least one."""
    case = zoo.shifted_gaussians(0.0, param_count=1)
    samples = case.sample(jax.random.key(10), 50).samples[0, 0]
    with pytest.raises(error):
        gaussian_surrogates(
            samples,
            None,
            case.space,
            size=size,  # type: ignore[arg-type]
            draws=draws,  # type: ignore[arg-type]
            key=jax.random.key(11),
        )


def test_surrogate_size_rounds_the_effective_sample_size() -> None:
    """The size is round(n_eff), at least two, and at most the cap."""
    assert surrogate_size(1_234.4) == 1_234
    assert surrogate_size(1.2) == 2
    assert surrogate_size(5_000.0, cap=2_000) == 2_000
    for invalid in (0.5, math.nan, math.inf):
        with pytest.raises(ValueError, match="at least 1"):
            surrogate_size(invalid)


def test_truncated_surrogates_stay_inside_interval_bounds() -> None:
    """Interval parameters follow their normal truncated to the interval."""
    case = zoo.railed_gaussians([0.0])
    samples = case.sample(jax.random.key(12), 3_000).samples[0, 0]
    surrogates = np.asarray(
        gaussian_surrogates(
            samples,
            None,
            case.space,
            size=2_000,
            draws=10,
            key=jax.random.key(13),
            truncate=True,
        )
    )
    assert surrogates[..., 0].min() >= 0.0
    fitted = GaussianDensity.fit(samples, None, case.space)
    mean = float(fitted.mean[0])
    width = math.sqrt(float(fitted.covariance[0, 0]))
    truncated = stats.truncnorm((0.0 - mean) / width, np.inf, loc=mean, scale=width)
    pooled = surrogates[..., 0].ravel()
    assert pooled.mean() == pytest.approx(truncated.mean(), abs=0.01)
    assert pooled.std() == pytest.approx(truncated.std(), rel=0.02)
    untruncated = np.asarray(
        gaussian_surrogates(
            samples, None, case.space, size=2_000, draws=10, key=jax.random.key(13)
        )
    )
    np.testing.assert_array_equal(untruncated[..., 1], surrogates[..., 1])
    assert untruncated[..., 0].min() < 0.0
