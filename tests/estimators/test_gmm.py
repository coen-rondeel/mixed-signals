"""Tests for the Gaussian mixture estimator."""

import math
import warnings
from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mixed_signals import zoo
from mixed_signals.core.space import ParamSpace
from mixed_signals.errors import ConvergenceWarning, ValidityWarning
from mixed_signals.estimators import DEFAULT_STARTS, MixtureDensity, MixtureSettings

type GridIntegral = Callable[..., float]

STANDARD_ERRORS = 5.0
SAMPLE_COUNT = 20_000
TWO_PI = 2 * math.pi
REAL_PAIR = ParamSpace.model_validate({"parameters": [{"name": "x0"}, {"name": "x1"}]})
PHASE = ParamSpace.model_validate(
    {"parameters": [{"name": "phase", "topology": "circle", "bounds": [0.0, TWO_PI]}]}
)


def two_components(**changes: object) -> MixtureSettings:
    """Return settings for two components, with optional changes."""
    return MixtureSettings.model_validate({"components": 2, **changes})


# * Recovery


def test_separated_modes_of_the_zoo_are_recovered() -> None:
    """Weights, means and covariances agree with the zoo within five standard errors."""
    case = zoo.gaussian_mixture(6.0, param_count=2)
    samples = case.sample(jax.random.key(0), SAMPLE_COUNT).samples[0, 0]
    fitted = MixtureDensity.fit(
        samples, None, case.space, two_components(starts=4), key=jax.random.key(1)
    )
    order = np.argsort(np.asarray(fitted.means[:, 0]))
    weights = np.exp(np.asarray(fitted.log_weights))[order]
    means = np.asarray(fitted.means)[order]
    covariances = np.asarray(fitted.covariances)[order]
    half = SAMPLE_COUNT / 2
    assert np.all(
        np.abs(weights - 0.5) < STANDARD_ERRORS * math.sqrt(0.25 / SAMPLE_COUNT)
    )
    assert np.all(
        np.abs(means - [[-3.0, 0.0], [3.0, 0.0]]) < STANDARD_ERRORS / math.sqrt(half)
    )
    assert np.all(
        np.abs(covariances - np.eye(2)) < STANDARD_ERRORS * math.sqrt(2 / half)
    )
    assert bool(fitted.converged)


def test_mean_log_likelihood_matches_the_truth() -> None:
    """The weighted mean log density of the fit is the true one, within its error."""
    case = zoo.gaussian_mixture(4.0, param_count=2)
    samples = case.sample(jax.random.key(2), SAMPLE_COUNT).samples[0, 0]
    fitted = MixtureDensity.fit(
        samples, None, case.space, two_components(starts=4), key=jax.random.key(3)
    )
    true_log = np.asarray(case.log_prob(samples)[:, 0])
    error = true_log.std() / math.sqrt(SAMPLE_COUNT)
    assert abs(float(fitted.mean_log_likelihood) - true_log.mean()) < (
        STANDARD_ERRORS * error
    )
    np.testing.assert_allclose(
        fitted.mean_log_likelihood,
        jnp.mean(fitted.log_density(samples)),
        rtol=1e-12,
    )


def test_one_component_is_the_maximum_likelihood_gaussian() -> None:
    """With one component, EM gives the weighted mean and the biased covariance."""
    rng = np.random.default_rng(4)
    samples = rng.multivariate_normal([1.0, -1.0], [[2.0, 0.7], [0.7, 1.0]], size=500)
    log_weights = rng.normal(scale=0.5, size=500)
    fitted = MixtureDensity.fit(
        samples,
        log_weights,
        REAL_PAIR,
        MixtureSettings(components=1),
        key=jax.random.key(5),
    )
    weights = np.exp(log_weights)
    biased = np.cov(samples, rowvar=False, aweights=weights, bias=True)
    unbiased = np.cov(samples, rowvar=False, aweights=weights)
    # * The 1e-6 regularisation acts in z-scored units, so it scales with each variance.
    expected = biased + 1e-6 * np.diag(np.diag(unbiased))
    np.testing.assert_allclose(fitted.covariances[0], expected, rtol=1e-9)
    np.testing.assert_allclose(
        fitted.means[0], np.average(samples, axis=0, weights=weights), rtol=1e-12
    )
    assert float(fitted.log_weights[0]) == pytest.approx(0.0, abs=1e-12)


def test_fit_does_not_depend_on_parameter_scales() -> None:
    """At 3 mHz ± 1e-11 Hz fits agree; the likelihood shifts by the Jacobian."""
    case = zoo.gaussian_mixture(5.0, param_count=2)
    samples = case.sample(jax.random.key(6), 5_000).samples[0, 0]
    scales = jnp.array([1e-11, 1e-23])
    physical = jnp.array([3e-3, 1e-22]) + scales * samples
    plain = MixtureDensity.fit(
        samples, None, case.space, two_components(starts=4), key=jax.random.key(7)
    )
    scaled = MixtureDensity.fit(
        physical, None, case.space, two_components(starts=4), key=jax.random.key(7)
    )
    shift = -float(jnp.sum(jnp.log(scales)))
    assert float(scaled.mean_log_likelihood) == pytest.approx(
        float(plain.mean_log_likelihood) + shift, rel=1e-9
    )
    np.testing.assert_allclose(
        (scaled.means - jnp.array([3e-3, 1e-22])) / scales, plain.means, atol=1e-6
    )


# * Circles


def test_modes_on_a_circle_are_recovered(grid_integral: GridIntegral) -> None:
    """Two modes, one across the boundary, keep their means, with unit total mass."""
    rng = np.random.default_rng(8)
    phases = np.r_[rng.normal(0.1, 0.2, 5_000), rng.normal(2.0, 0.3, 5_000)] % TWO_PI
    fitted = MixtureDensity.fit(
        phases[:, None], None, PHASE, two_components(starts=4), key=jax.random.key(9)
    )
    means = np.sort(np.asarray(fitted.means[:, 0]) % TWO_PI)
    np.testing.assert_allclose(means, [0.1, 2.0], atol=0.02)
    mass = grid_integral(
        lambda points: np.exp(np.asarray(fitted.log_density(points))),
        (0.0, TWO_PI, 64),
    )
    assert mass == pytest.approx(1.0, abs=1e-9)


def test_opposite_modes_on_a_circle_warn() -> None:
    """With no resolved mean direction, the cut may split a mode."""
    rng = np.random.default_rng(10)
    phases = np.r_[rng.normal(1.0, 0.2, 2_000), rng.normal(1.0 + math.pi, 0.2, 2_000)]
    with pytest.warns(ValidityWarning, match="Rayleigh"):
        MixtureDensity.fit(
            phases[:, None] % TWO_PI,
            None,
            PHASE,
            two_components(starts=4),
            key=jax.random.key(11),
        )


# * Stopping and starts


def test_hitting_the_iteration_cap_warns() -> None:
    """A winning start that stops at the cap raises a ConvergenceWarning."""
    case = zoo.gaussian_mixture(3.0, param_count=2)
    samples = case.sample(jax.random.key(12), 2_000).samples[0, 0]
    with pytest.warns(ConvergenceWarning, match="after 1 iterations"):
        fitted = MixtureDensity.fit(
            samples,
            None,
            case.space,
            two_components(max_iterations=1),
            key=jax.random.key(13),
        )
    assert not bool(fitted.converged)
    assert int(fitted.iterations) == 1


def test_the_key_fixes_the_fit() -> None:
    """The same key repeats a fit exactly."""
    case = zoo.gaussian_mixture(3.0, param_count=2)
    samples = case.sample(jax.random.key(14), 2_000).samples[0, 0]
    first, second = (
        MixtureDensity.fit(
            samples, None, case.space, two_components(starts=2), key=jax.random.key(15)
        )
        for _ in range(2)
    )
    np.testing.assert_array_equal(first.means, second.means)
    np.testing.assert_array_equal(first.covariances, second.covariances)


def test_default_starts_find_the_modes_that_single_starts_miss() -> None:
    """Under scikit-learn's stopping rule, single starts often end near one Gaussian."""
    case = zoo.gaussian_mixture(3.0, param_count=4)
    samples = case.sample(jax.random.key(16), 4_000).samples[0, 0]
    reference = MixtureDensity.fit(
        samples,
        None,
        case.space,
        two_components(starts=20, tolerance=1e-10, max_iterations=5_000),
        key=jax.random.key(17),
    )

    def gap(settings: MixtureSettings, key: jax.Array) -> jax.Array:
        fitted = MixtureDensity.fit(samples, None, case.space, settings, key=key)
        return reference.mean_log_likelihood - fitted.mean_log_likelihood

    keys = jax.random.split(jax.random.key(18), 24)
    single = jax.vmap(lambda key: gap(two_components(starts=1), key))(keys)
    defaults = jax.vmap(lambda key: gap(two_components(), key))(keys[:5])
    assert DEFAULT_STARTS == 18
    assert float(jnp.mean(single > 0.01)) > 0.25
    assert bool(jnp.all(defaults < 0.01))


# * Validity and transformations


def test_a_parameter_without_spread_gives_nan_and_warns() -> None:
    """z-scoring needs every parameter to spread."""
    samples = np.c_[np.zeros(40), np.random.default_rng(19).normal(size=40)]
    with pytest.warns(ValidityWarning, match="'x0' has no spread"):
        fitted = MixtureDensity.fit(
            samples, None, REAL_PAIR, two_components(starts=4), key=jax.random.key(20)
        )
    assert math.isnan(float(fitted.mean_log_likelihood))


def test_vmap_over_problems_and_keys() -> None:
    """Batched fits run under vmap, also with concrete samples and traced keys."""
    case = zoo.gaussian_mixture(5.0, param_count=2)
    batch = case.sample(jax.random.key(21), 2_000, batch_size=3).samples[:, 0]
    keys = jax.random.split(jax.random.key(22), 3)
    batched = jax.vmap(
        lambda values, key: MixtureDensity.fit(
            values, None, case.space, two_components(starts=4), key=key
        )
    )(batch, keys)
    concrete = MixtureDensity.fit(
        batch[1], None, case.space, two_components(starts=4), key=keys[1]
    )
    np.testing.assert_allclose(
        batched.mean_log_likelihood[1], concrete.mean_log_likelihood, rtol=1e-12
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        over_keys = jax.vmap(
            lambda key: (
                MixtureDensity.fit(
                    batch[0], None, case.space, two_components(starts=2), key=key
                ).mean_log_likelihood
            )
        )(keys)
    assert over_keys.shape == (3,)


def test_parameter_count_and_settings() -> None:
    """P counts C − 1 weights, C·D means and C·D(D + 1)/2 covariance entries."""
    space = ParamSpace.model_validate(
        {"parameters": [{"name": f"x{index}"} for index in range(4)]}
    )
    samples = np.random.default_rng(23).normal(size=(200, 4))
    fitted = MixtureDensity.fit(
        samples,
        None,
        space,
        MixtureSettings(components=3, starts=2),
        key=jax.random.key(24),
    )
    assert fitted.parameter_count == 2 + 12 + 30
    assert two_components().to_json() == (
        '{"components":2,"starts":18,"max_iterations":100,"tolerance":0.001,'
        '"regularization":1e-6}'
    )
