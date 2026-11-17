"""Tests for the generalized Pareto fit of Zhang and Stephens and its survival."""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import stats

from mixed_signals.sigma.pareto import (
    ParetoFit,
    generalized_pareto_fit,
    generalized_pareto_log_survival,
)

SCALE = 2.0
SHAPES = [-0.3, 0.0, 0.4]
SHAPE_TOLERANCE = 0.05
SCALE_TOLERANCE = 0.05


def genpareto_sample(shape: float, size: int, seed: int) -> np.ndarray:
    """Draw excesses from a generalized Pareto distribution with SciPy."""
    return np.asarray(
        stats.genpareto.rvs(shape, scale=SCALE, size=size, random_state=seed)
    )


# * The estimator


@pytest.mark.parametrize("shape", SHAPES)
def test_the_fit_recovers_shape_and_scale(shape: float) -> None:
    """5 000 excesses give ξ within 0.05 and σ within 5% (about 3.5 errors in ξ)."""
    fit = generalized_pareto_fit(genpareto_sample(shape, 5_000, seed=1))
    assert abs(float(fit.shape) - shape) < SHAPE_TOLERANCE
    assert float(fit.scale) == pytest.approx(SCALE, rel=SCALE_TOLERANCE)


@pytest.mark.parametrize("shape", SHAPES)
def test_the_fit_agrees_with_the_maximum_likelihood_fit(shape: float) -> None:
    """At 5 000 excesses ξ differs by under 0.01 and σ by under 1% from SciPy's fit."""
    excesses = genpareto_sample(shape, 5_000, seed=2)
    shape_ml, _, scale_ml = stats.genpareto.fit(excesses, floc=0)
    fit = generalized_pareto_fit(excesses)
    assert abs(float(fit.shape) - shape_ml) < 0.01
    assert float(fit.scale) == pytest.approx(scale_ml, rel=0.01)


def test_the_fit_does_not_depend_on_the_order() -> None:
    """Shuffling the excesses leaves the fit unchanged."""
    excesses = genpareto_sample(0.2, 400, seed=3)
    shuffled = np.random.default_rng(0).permutation(excesses)
    first, second = generalized_pareto_fit(excesses), generalized_pareto_fit(shuffled)
    np.testing.assert_allclose(first.shape, second.shape, rtol=1e-12)
    np.testing.assert_allclose(first.scale, second.scale, rtol=1e-12)


def test_the_bound_moves_the_end_point_above_it() -> None:
    """A bound above the largest excess keeps the end point of a ξ < 0 fit above it."""
    excesses = genpareto_sample(-0.3, 300, seed=4)
    free = generalized_pareto_fit(excesses)
    bound = 1.5 * SCALE / 0.3
    bounded = generalized_pareto_fit(excesses, bound=bound)
    assert float(free.shape) < 0
    assert float(bounded.shape) < 0
    assert -float(bounded.scale) / float(bounded.shape) > bound
    assert -float(free.scale) / float(free.shape) < bound


def test_a_bound_below_the_largest_excess_changes_nothing() -> None:
    """The grid uses the larger of the largest excess and the bound."""
    excesses = genpareto_sample(0.1, 300, seed=5)
    free = generalized_pareto_fit(excesses)
    bounded = generalized_pareto_fit(excesses, bound=0.5 * float(excesses.max()))
    np.testing.assert_array_equal(free.shape, bounded.shape)
    np.testing.assert_array_equal(free.scale, bounded.scale)


def test_the_mask_selects_the_fitted_excesses() -> None:
    """Masked entries are never read, and the fit equals that of the selected ones."""
    excesses = genpareto_sample(-0.2, 120, seed=6)
    valid = np.ones(120, dtype=bool)
    valid[::8] = False
    garbage = excesses.copy()
    garbage[~valid] = np.array([np.nan, -np.inf, 1e300, -5.0] * 15)[: (~valid).sum()]
    masked = generalized_pareto_fit(excesses, valid)
    ignored = generalized_pareto_fit(garbage, valid)
    selected = generalized_pareto_fit(excesses[valid])
    np.testing.assert_array_equal(masked.shape, ignored.shape)
    np.testing.assert_array_equal(masked.scale, ignored.scale)
    # * Both arrays hold 105 to 120 entries, so they share the grid of 40 values.
    np.testing.assert_allclose(masked.shape, selected.shape, rtol=1e-10)
    np.testing.assert_allclose(masked.scale, selected.scale, rtol=1e-10)


def test_the_fit_runs_under_jit_and_vmap() -> None:
    """A traced mask and bound work, and vmap matches a loop over rows."""
    rows = np.stack([genpareto_sample(0.1, 80, seed=seed) for seed in range(7, 10)])
    masks = np.arange(80)[None, :] < np.array([[80], [60], [40]])
    bounds = jnp.array([20.0, 30.0, 40.0])
    mapped = jax.vmap(
        lambda row, mask, bound: generalized_pareto_fit(row, mask, bound=bound)
    )(rows, masks, bounds)
    assert mapped.shape.shape == (3,)
    for index in range(3):
        single = generalized_pareto_fit(
            rows[index], masks[index], bound=float(bounds[index])
        )
        np.testing.assert_allclose(mapped.shape[index], single.shape, rtol=1e-12)
        np.testing.assert_allclose(mapped.scale[index], single.scale, rtol=1e-12)
    jitted = jax.jit(generalized_pareto_fit)(rows[0], masks[0])
    direct = generalized_pareto_fit(rows[0], masks[0])
    np.testing.assert_allclose(jitted.shape, direct.shape, rtol=1e-12)
    assert isinstance(jitted, ParetoFit)


def test_too_few_excesses_give_nan() -> None:
    """Fewer than five valid excesses give NaN shape and scale."""
    excesses = genpareto_sample(0.0, 30, seed=11)
    four = generalized_pareto_fit(excesses[:4])
    assert math.isnan(float(four.shape))
    assert math.isnan(float(four.scale))
    mask = np.arange(30) < 4
    masked = generalized_pareto_fit(excesses, mask)
    assert math.isnan(float(masked.shape))
    assert math.isnan(float(masked.scale))
    five = generalized_pareto_fit(excesses[:5])
    assert math.isfinite(float(five.shape))


def test_a_non_positive_first_quartile_gives_nan() -> None:
    """Mostly tied zero excesses leave a first quartile of zero, so the fit is NaN."""
    excesses = np.concatenate([np.zeros(30), genpareto_sample(0.0, 10, seed=12)])
    fit = generalized_pareto_fit(excesses)
    assert math.isnan(float(fit.shape))
    assert math.isnan(float(fit.scale))


# * The survival function


@pytest.mark.parametrize("shape", [-0.4, -1e-3, 0.0, 1e-3, 0.5])
def test_the_log_survival_matches_scipy(shape: float) -> None:
    """The log survival equals SciPy's, with the exponential limit at ξ = 0."""
    excess = np.array([0.0, 0.3, 1.0, 4.0, 25.0])
    expected = stats.genpareto.logsf(excess, shape, scale=SCALE)
    values = generalized_pareto_log_survival(excess, shape, SCALE)
    np.testing.assert_allclose(values, expected, rtol=1e-12, atol=1e-15)


def test_the_log_survival_is_minus_infinity_beyond_the_end_point() -> None:
    """With ξ < 0 the survival is zero at and beyond σ/|ξ|, and 1 below zero."""
    end_point = SCALE / 0.5
    values = generalized_pareto_log_survival(
        jnp.array([-1.0, 0.0, end_point - 0.01, end_point, end_point + 3.0, jnp.inf]),
        -0.5,
        SCALE,
    )
    assert float(values[0]) == 0.0
    assert float(values[1]) == 0.0
    assert math.isfinite(float(values[2]))
    assert all(float(value) == -math.inf for value in values[3:])


def test_the_log_survival_broadcasts_and_maps() -> None:
    """Excesses, shapes and scales broadcast, and the function works under vmap."""
    excess = jnp.array([[0.5], [2.0]])
    shape = jnp.array([-0.2, 0.0, 0.3])
    assert generalized_pareto_log_survival(excess, shape, 1.5).shape == (2, 3)
    mapped = jax.vmap(generalized_pareto_log_survival)(
        jnp.array([0.5, 2.0]), jnp.array([-0.2, 0.3]), jnp.array([1.0, 2.0])
    )
    np.testing.assert_allclose(
        mapped,
        stats.genpareto.logsf([0.5, 2.0], [-0.2, 0.3], scale=[1.0, 2.0]),
        rtol=1e-12,
    )


def test_nan_inputs_give_nan() -> None:
    """A NaN excess or a NaN fit gives NaN, not a finite value."""
    assert math.isnan(float(generalized_pareto_log_survival(jnp.nan, 0.1, 1.0)))
    assert math.isnan(float(generalized_pareto_log_survival(1.0, jnp.nan, jnp.nan)))
