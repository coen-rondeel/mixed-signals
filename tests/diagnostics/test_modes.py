"""Tests for GMM-BIC and the modes of one-dimensional estimates."""

import math
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mixed_signals import zoo
from mixed_signals.core.space import ParamSpace
from mixed_signals.diagnostics.modes import (
    MODE_SEPARATION,
    circle_centres,
    discriminant_directions,
    gmm_bic,
    marginal_modes,
    projection_modes,
)
from mixed_signals.errors import ConvergenceWarning, ValidityWarning
from mixed_signals.estimators import MixtureDensity, MixtureSettings
from mixed_signals.prng import derive_key

TWO_PI = 2 * math.pi
CIRCLE = ParamSpace.model_validate(
    {"parameters": [{"name": "phase", "topology": "circle", "bounds": [0.0, TWO_PI]}]}
)
PLANE = ParamSpace.model_validate({"parameters": [{"name": "x0"}, {"name": "x1"}]})


def draws(case: zoo.ZooCase, name: str, count: int, seed: int) -> np.ndarray:
    """Return samples of one posterior of a zoo case."""
    index = case.names.index(name)
    return np.asarray(case.sample(jax.random.key(seed), count).samples[0, index])


# * GMM-BIC


def test_bic_is_formed_with_the_effective_sample_size() -> None:
    """BIC(c) = −2 n_eff ℓ̄ + p ln n_eff from the fit of c components."""
    case = zoo.gaussian_mixture(4.0)
    samples = draws(case, "mixture", 1_000, 0)
    key = jax.random.key(1)
    result = gmm_bic(samples, None, case.space, effective_sample_size=400.0, key=key)
    fit = MixtureDensity.fit(
        samples,
        None,
        case.space,
        MixtureSettings(components=2),
        key=derive_key(key, "components", 2),
    )
    expected = -2 * 400 * float(
        fit.mean_log_likelihood
    ) + fit.parameter_count * math.log(400)
    assert result.bic[1] == pytest.approx(expected, rel=1e-12)
    assert result.components == 2
    assert result.delta_bic == pytest.approx(result.bic[0] - result.bic[1], rel=1e-12)
    assert len(result.bic) == 5


def test_one_gaussian_wins_for_a_gaussian() -> None:
    """A Gaussian posterior needs no second component."""
    case = zoo.shifted_gaussians(0.0, param_count=3)
    samples = draws(case, "reference", 2_000, 2)
    result = gmm_bic(
        samples, None, case.space, effective_sample_size=2_000.0, key=jax.random.key(3)
    )
    assert result.components == 1
    assert result.delta_bic == 0.0


def test_repeated_draws_with_their_n_eff_give_the_independent_result() -> None:
    """Draws repeated five times, with the originals' n_eff, choose as the originals."""
    case = zoo.gaussian_mixture(2.5)
    samples = draws(case, "mixture", 1_000, 4)
    key = jax.random.key(5)
    original = gmm_bic(
        samples, None, case.space, effective_sample_size=1_000.0, key=key
    )
    repeated = gmm_bic(
        np.repeat(samples, 5, axis=0),
        None,
        case.space,
        effective_sample_size=1_000.0,
        key=key,
    )
    assert repeated.components == original.components == 2
    assert repeated.delta_bic == pytest.approx(original.delta_bic, rel=0.05)


def test_bic_needs_two_effective_samples_and_a_component() -> None:
    """The penalty needs n_eff ≥ 2; the search needs at least one component."""
    case = zoo.shifted_gaussians(0.0, param_count=1)
    samples = draws(case, "reference", 100, 6)
    with pytest.raises(ValueError, match="at least 2"):
        gmm_bic(
            samples, None, case.space, effective_sample_size=1.5, key=jax.random.key(0)
        )
    with pytest.raises(ValueError, match="max_components"):
        gmm_bic(
            samples,
            None,
            case.space,
            effective_sample_size=100.0,
            key=jax.random.key(0),
            max_components=0,
        )


def test_a_parameter_without_spread_warns_once() -> None:
    """A constant parameter gives NaN mixtures and one warning, not one per fit."""
    samples = np.c_[np.random.default_rng(7).normal(size=(300, 1)), np.full(300, 2.0)]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = gmm_bic(
            samples, None, PLANE, effective_sample_size=300.0, key=jax.random.key(8)
        )
    validity = [
        entry for entry in caught if issubclass(entry.category, ValidityWarning)
    ]
    assert len(validity) == 1
    assert "mixtures are NaN" in str(validity[0].message)
    assert math.isnan(result.delta_bic)


def test_discriminant_directions_are_unit_vectors_between_components() -> None:
    """Components separated along x0 give a direction along x0."""
    case = zoo.gaussian_mixture(5.0, param_count=3)
    samples = draws(case, "mixture", 2_000, 9)
    result = gmm_bic(
        samples, None, case.space, effective_sample_size=2_000.0, key=jax.random.key(10)
    )
    directions = np.asarray(discriminant_directions(result.mixture))
    assert directions.shape == (1, 3)
    np.testing.assert_allclose(np.linalg.norm(directions, axis=1), 1.0, rtol=1e-12)
    assert abs(directions[0, 0]) > 0.99
    single = MixtureDensity.fit(
        samples, None, case.space, MixtureSettings(components=1), key=jax.random.key(0)
    )
    assert discriminant_directions(single).shape == (0, 3)


# * Modes of 1D estimates


def test_marginal_modes_find_the_mixture_and_spare_the_banana() -> None:
    """Separated modes give two peaks and a dip; a banana's marginals keep one."""
    mixture = zoo.gaussian_mixture(4.0)
    counts = marginal_modes(draws(mixture, "mixture", 2_000, 11), None, mixture.space)
    np.testing.assert_array_equal(counts.modes, [2.0, 1.0])
    assert float(counts.dip[0]) > 0.3
    assert float(counts.dip[1]) == 0.0
    banana = zoo.banana(0.1)
    curved = marginal_modes(draws(banana, "banana", 2_000, 12), None, banana.space)
    np.testing.assert_array_equal(curved.modes, [1.0, 1.0])


def test_railed_marginals_keep_one_mode_at_the_bound() -> None:
    """The corrected estimate at a bound adds no mode to a railed posterior."""
    case = zoo.railed_gaussians([0.0, 1.0])
    for name in case.names:
        counts = marginal_modes(draws(case, name, 2_000, 13), None, case.space)
        assert float(counts.modes[0]) == 1.0
        assert float(counts.dip[0]) < MODE_SEPARATION


def test_circle_modes_do_not_depend_on_the_cut() -> None:
    """Two modes on a circle are found wherever the period starts."""
    rng = np.random.default_rng(14)
    angles = np.r_[rng.normal(0.0, 0.3, 1_000), rng.normal(math.pi, 0.3, 1_000)]
    for turn in (0.0, 1.0, 2.5):
        counts = marginal_modes(np.mod(angles + turn, TWO_PI)[:, None], None, CIRCLE)
        assert float(counts.modes[0]) == 2.0
        assert float(counts.dip[0]) > 0.9
    single = marginal_modes(
        np.mod(rng.normal(0.0, 0.3, 2_000), TWO_PI)[:, None], None, CIRCLE
    )
    assert float(single.modes[0]) == 1.0
    assert float(single.dip[0]) == 0.0


def test_shallow_valleys_merge_and_deep_ones_separate() -> None:
    """Three separated modes on a line count three; a flat top counts one."""
    rng = np.random.default_rng(15)
    line = ParamSpace.model_validate({"parameters": [{"name": "x0"}]})
    three = np.r_[
        rng.normal(-4, 1, 1_000), rng.normal(0, 1, 1_000), rng.normal(4, 1, 1_000)
    ]
    counts = marginal_modes(three[:, None], None, line)
    assert float(counts.modes[0]) == 3.0
    flat = np.r_[rng.normal(-1, 1, 2_000), rng.normal(1, 1, 2_000)]
    assert float(marginal_modes(flat[:, None], None, line).modes[0]) == 1.0


def test_a_mode_that_hides_in_the_marginals_shows_along_its_direction() -> None:
    """Modes at ±(1, 1), long across the diagonal, blur each marginal into one mode."""
    rng = np.random.default_rng(16)
    centres = np.where(rng.uniform(size=4_000)[:, None] < 0.5, -1.0, 1.0)
    along = 0.3 * rng.normal(size=(4_000, 1))
    across = 2.0 * rng.normal(size=(4_000, 1))
    samples = centres + (
        along * np.array([1.0, 1.0]) + across * np.array([1.0, -1.0])
    ) / math.sqrt(2)
    marginal = marginal_modes(samples, None, PLANE)
    np.testing.assert_array_equal(marginal.modes, [1.0, 1.0])
    result = gmm_bic(
        samples, None, PLANE, effective_sample_size=4_000.0, key=jax.random.key(17)
    )
    assert result.components >= 2
    projected = projection_modes(
        samples,
        None,
        PLANE,
        directions=discriminant_directions(result.mixture),
        centres=circle_centres(samples, None, PLANE),
    )
    assert float(jnp.max(projected.modes)) == 2.0
    assert float(jnp.max(projected.dip)) > MODE_SEPARATION


def test_projections_unwrap_circles_about_the_given_centres() -> None:
    """A circle is unwrapped about the given centre before it is projected."""
    case = zoo.wrapped_gaussians(0.0, width=0.3, param_count=2)
    samples = draws(case, "reference", 1_000, 18)
    centres = circle_centres(samples, None, case.space)
    along_circle = projection_modes(
        samples, None, case.space, directions=np.array([[1.0, 0.0]]), centres=centres
    )
    assert float(along_circle.modes[0]) == 1.0
    empty = projection_modes(
        samples, None, case.space, directions=np.zeros((0, 2)), centres=centres
    )
    assert empty.modes.shape == (0,)


def test_mode_counts_map_over_surrogate_sets() -> None:
    """Under jax.lax.map the counts equal concrete calls."""
    case = zoo.gaussian_mixture(4.0)
    sets = case.sample(jax.random.key(19), 800, batch_size=2).samples[:, 0]
    mapped = jax.lax.map(lambda values: marginal_modes(values, None, case.space), sets)
    direct = marginal_modes(sets[1], None, case.space)
    np.testing.assert_allclose(mapped.dip[1], direct.dip, rtol=1e-12)
    np.testing.assert_array_equal(mapped.modes[1], direct.modes)


def test_a_fit_that_stops_at_its_cap_warns() -> None:
    """EM settings pass through to the fits; one iteration leaves EM unconverged."""
    case = zoo.gaussian_mixture(3.0)
    with pytest.warns(ConvergenceWarning, match="stopped at 1 iterations"):
        result = gmm_bic(
            draws(case, "mixture", 500, 20),
            None,
            case.space,
            effective_sample_size=500.0,
            key=jax.random.key(21),
            max_components=2,
            settings=MixtureSettings(components=1, max_iterations=1, starts=2),
        )
    assert result.mixture.settings.max_iterations == 1


def test_upper_bounds_close_the_grid() -> None:
    """A railed posterior below an upper bound keeps one mode at the bound."""
    case = zoo.railed_gaussians([0.0], side="upper")
    counts = marginal_modes(draws(case, "railed_0", 2_000, 22), None, case.space)
    assert float(counts.modes[0]) == 1.0
