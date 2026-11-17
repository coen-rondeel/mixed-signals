"""Tests for the permutation null: side sizes, pooled weights and calibration."""

import math
from collections.abc import Callable
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mixed_signals import zoo
from mixed_signals.core.samples import kish_effective_sample_size
from mixed_signals.nulls import (
    Construction,
    null_draws,
    permutation_null,
    permutation_side_size,
    random_sides,
)
from mixed_signals.prng import derive_key
from mixed_signals.sigma.significance import empirical_p_value

STANDARD_ERRORS = 4.0
LEVEL = 0.1


def two_valued_weights(count: int, kept: int) -> jax.Array:
    """Return log weights of `count` samples, `kept` of them equal and the rest zero."""
    return jnp.where(jnp.arange(count) < kept, 0.0, -jnp.inf)


# * Side sizes


def test_equal_weights_and_sizes_give_the_set_size() -> None:
    """Two equal-weight sets of N samples give sides of N samples."""
    assert permutation_side_size(jnp.zeros(1_000), jnp.zeros(1_000)) == 1_000


def test_equal_weight_sets_of_different_sizes_get_equal_halves() -> None:
    """Sets of 4 000 and 1 000 equal-weight samples give sides of 2 500."""
    assert permutation_side_size(jnp.zeros(4_000), jnp.zeros(1_000)) == 2_500


def test_effective_sample_sizes_give_their_harmonic_mean() -> None:
    """Equal-weight sets of equal size with given n_eff give the harmonic mean."""
    size = permutation_side_size(
        jnp.zeros(1_000), jnp.zeros(1_000), first_n_eff=400.0, second_n_eff=600.0
    )
    assert size == round(2 / (1 / 400 + 1 / 600))
    assert size == 480


def test_independent_weighted_sets_give_half_the_pool() -> None:
    """With n_eff equal to the Kish size, the side holds half the pooled samples."""
    rng = np.random.default_rng(0)
    first = jnp.asarray(rng.normal(size=1_000))
    second = jnp.asarray(rng.normal(size=600) * 0.5)
    assert float(kish_effective_sample_size(first)) < 1_000
    assert permutation_side_size(first, second) == 800


def test_general_formula_with_weights_and_effective_sizes() -> None:
    """Weights and given effective sizes enter as N, K and n of the formula."""
    rng = np.random.default_rng(1)
    first = rng.normal(size=900) * 0.7
    second = rng.normal(size=500)
    kish = [
        float(kish_effective_sample_size(jnp.asarray(log_weights)))
        for log_weights in (first, second)
    ]
    n_eff = [300.0, 250.0]
    expected = round(
        0.5 * 1_400 * (1 / kish[0] + 1 / kish[1]) / (1 / n_eff[0] + 1 / n_eff[1])
    )
    size = permutation_side_size(
        first, second, first_n_eff=n_eff[0], second_n_eff=n_eff[1]
    )
    assert size == expected
    assert 2 < size < 700


def test_the_size_is_clipped_to_two_and_to_half_the_pool() -> None:
    """The side holds at least 2 samples and at most half of the pooled samples."""
    assert (
        permutation_side_size(
            jnp.zeros(1_000), jnp.zeros(1_000), first_n_eff=1.0, second_n_eff=1.0
        )
        == 2
    )
    assert (
        permutation_side_size(
            jnp.zeros(1_000), jnp.zeros(600), first_n_eff=1e9, second_n_eff=1e9
        )
        == 800
    )


def test_side_size_errors() -> None:
    """Effective sizes below one or not finite, bad shapes and tiny pools raise."""
    ones = jnp.zeros(10)
    for invalid in (0.5, math.nan, math.inf):
        with pytest.raises(ValueError, match="at least 1"):
            permutation_side_size(ones, ones, first_n_eff=invalid)
        with pytest.raises(ValueError, match="at least 1"):
            permutation_side_size(ones, ones, second_n_eff=invalid)
    with pytest.raises(ValueError, match="shaped"):
        permutation_side_size(jnp.zeros((3, 3)), ones)
    with pytest.raises(ValueError, match="Kish"):
        permutation_side_size(jnp.full(5, -jnp.inf), ones)
    with pytest.raises(ValueError, match="pool of 4"):
        permutation_side_size(jnp.zeros(1), jnp.zeros(2))


# * The plan


def test_pooled_weights_sum_to_one_in_each_set() -> None:
    """Each set's weights are normalised before pooling; None means equal weights."""
    rng = np.random.default_rng(2)
    case = zoo.shifted_gaussians(0.0, param_count=2)
    sets = case.sample(jax.random.key(3), 80).samples[0]
    first_log_weights = rng.normal(size=80) + 5.0
    plan = permutation_null(
        sets[0],
        first_log_weights,
        sets[1],
        None,
        key=jax.random.key(4),
        draws=5,
    )
    weights = np.exp(np.asarray(plan.log_weights))
    assert weights[:80].sum() == pytest.approx(1.0)
    assert weights[80:].sum() == pytest.approx(1.0)
    np.testing.assert_allclose(weights[80:], 1 / 80)
    np.testing.assert_allclose(
        weights[:80] / weights[:80].sum(),
        np.exp(first_log_weights) / np.exp(first_log_weights).sum(),
    )
    np.testing.assert_array_equal(plan.samples, jnp.concatenate([sets[0], sets[1]]))


def test_plan_holds_disjoint_sides_inside_the_pool() -> None:
    """Both sides of every draw lie in the pool and share no sample."""
    rng = np.random.default_rng(5)
    first = rng.normal(size=(300, 2))
    second = rng.normal(size=(100, 2))
    plan = permutation_null(first, None, second, None, key=jax.random.key(6), draws=12)
    assert plan.construction == Construction.PERMUTATION
    assert plan.samples.shape == (400, 2)
    assert plan.first.shape == plan.second.shape == (12, 200)
    assert plan.recipe() == {
        "construction": "permutation",
        "draws": 12,
        "sample_size": 200,
    }
    for first_side, second_side in zip(plan.first, plan.second, strict=True):
        assert len(set(first_side.tolist()) | set(second_side.tolist())) == 400


def test_plan_uses_the_side_size_and_the_given_effective_sizes() -> None:
    """The side size of the plan is the one of permutation_side_size."""
    rng = np.random.default_rng(7)
    first = rng.normal(size=(500, 1))
    second = rng.normal(size=(500, 1))
    plan = permutation_null(
        first,
        None,
        second,
        None,
        key=jax.random.key(8),
        first_n_eff=200.0,
        second_n_eff=300.0,
        draws=3,
    )
    assert plan.first.shape == (3, 240)


def test_the_plan_does_not_depend_on_the_number_of_draws() -> None:
    """Draw i of the plan is the draw of random_sides with the same key."""
    rng = np.random.default_rng(9)
    first = rng.normal(size=(60, 1))
    second = rng.normal(size=(40, 1))
    key = jax.random.key(10)
    few = permutation_null(first, None, second, None, key=key, draws=2)
    many = permutation_null(first, None, second, None, key=key, draws=5)
    np.testing.assert_array_equal(few.first, many.first[:2])
    sides = random_sides(key, 100, 50, draws=5)
    np.testing.assert_array_equal(many.first, sides[0])
    np.testing.assert_array_equal(many.second, sides[1])


def test_inputs_are_checked() -> None:
    """Shapes, types and values of the sets are validated."""
    good = np.zeros((10, 2))
    key = jax.random.key(11)
    with pytest.raises(ValueError, match="same parameters"):
        permutation_null(good, None, np.zeros((10, 3)), None, key=key)
    with pytest.raises(ValueError, match="shaped"):
        permutation_null(np.zeros(10), None, good, None, key=key)
    with pytest.raises(ValueError, match="shaped"):
        permutation_null(good, np.zeros(9), good, None, key=key)
    with pytest.raises(ValueError, match="finite"):
        permutation_null(np.full((10, 2), np.nan), None, good, None, key=key)
    with pytest.raises(ValueError, match="NaN"):
        permutation_null(good, np.full(10, np.nan), good, None, key=key)
    with pytest.raises(ValueError, match="positive weight"):
        permutation_null(good, np.full(10, -np.inf), good, None, key=key)
    with pytest.raises(TypeError, match="real"):
        permutation_null(good + 1j, None, good, None, key=key)
    with pytest.raises(TypeError, match="draws"):
        permutation_null(good, None, good, None, key=key, draws=2.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="at least 1"):
        permutation_null(good, None, good, None, key=key, first_n_eff=0.2)


# * Calibration: identical distributions give uniform p-values

REPLICATES = 2_000
PERMUTATIONS = 99
SAMPLES = 400

type Statistic = Callable[[jax.Array, jax.Array, jax.Array, jax.Array], jax.Array]


def weighted_widths(
    samples: jax.Array, log_weights: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """Return the weighted mean and width of the first parameter."""
    weights = jax.nn.softmax(log_weights)
    mean = jnp.sum(weights * samples[:, 0])
    return mean, jnp.sqrt(jnp.sum(weights * (samples[:, 0] - mean) ** 2))


def mean_difference(
    first: jax.Array,
    first_log_weights: jax.Array,
    second: jax.Array,
    second_log_weights: jax.Array,
) -> jax.Array:
    """Return the absolute difference of the weighted means."""
    first_mean, _ = weighted_widths(first, first_log_weights)
    second_mean, _ = weighted_widths(second, second_log_weights)
    return jnp.abs(first_mean - second_mean)


def log_width_ratio(
    first: jax.Array,
    first_log_weights: jax.Array,
    second: jax.Array,
    second_log_weights: jax.Array,
) -> jax.Array:
    """Return the absolute log of the ratio of the weighted widths."""
    _, first_width = weighted_widths(first, first_log_weights)
    _, second_width = weighted_widths(second, second_log_weights)
    return jnp.abs(jnp.log(second_width / first_width))


def rejection_error(
    statistic: Statistic,
    pool: jax.Array,
    pool_log_weights: jax.Array,
    observed: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
    *,
    side_size: int,
    key: jax.Array,
) -> float:
    """Return the distance of the rejection rate at LEVEL from LEVEL, in errors.

    Every replicate cuts its own pool into PERMUTATIONS random pairs of sides,
    evaluates the statistic on them and takes the empirical p-value of the
    observed value. The rate at which p is at most LEVEL is LEVEL for exchangeable
    sets, with binomial error sqrt(LEVEL (1 - LEVEL) / replicates).

    Args:
        statistic: Statistic of two weighted sides, large for a departure.
        pool: Pooled samples of every replicate, shaped (R, T, 1).
        pool_log_weights: Their log weights, shaped (R, T).
        observed: Samples and log weights of the observed first and second
            set of every replicate, each with a leading axis of R.
        side_size: Samples per side.
        key: JAX PRNG key.

    Returns:
        The distance in binomial standard errors.
    """
    replicate_count, pool_size = pool.shape[:2]
    keys = jnp.stack(
        [derive_key(key, "replicate", index) for index in range(replicate_count)]
    )
    first_sides, second_sides = jax.vmap(
        partial(
            random_sides,
            pool_size=pool_size,
            side_size=side_size,
            draws=PERMUTATIONS,
        )
    )(keys)

    def p_value(
        replicate: tuple[jax.Array, ...],
    ) -> jax.Array:
        samples, log_weights, first, second, *seen = replicate
        draws = jax.lax.map(
            lambda positions: statistic(
                samples[positions[0]],
                log_weights[positions[0]],
                samples[positions[1]],
                log_weights[positions[1]],
            ),
            (first, second),
        )
        return empirical_p_value(statistic(*seen), draws, "upper")

    p_values = jax.lax.map(
        p_value,
        (pool, pool_log_weights, first_sides, second_sides, *observed),
        batch_size=100,
    )
    rate = float(jnp.mean(p_values <= LEVEL))
    return abs(rate - LEVEL) / math.sqrt(LEVEL * (1 - LEVEL) / replicate_count)


def pooled(
    first: jax.Array,
    first_log_weights: jax.Array,
    second: jax.Array,
    second_log_weights: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Pool every replicate's two sets, each with weights summing to one."""
    return (
        jnp.concatenate([first, second], axis=1),
        jnp.concatenate(
            [
                jax.nn.log_softmax(first_log_weights, axis=1),
                jax.nn.log_softmax(second_log_weights, axis=1),
            ],
            axis=1,
        ),
    )


@pytest.mark.parametrize("statistic", [mean_difference, log_width_ratio])
def test_equal_weight_sets_of_one_distribution_are_calibrated(
    statistic: Statistic,
) -> None:
    """Identical zoo Gaussians reject at the 10% level at a rate of 10%."""
    case = zoo.shifted_gaussians(0.0, param_count=1)
    sets = case.sample(jax.random.key(20), SAMPLES, batch_size=REPLICATES).samples
    first, second = sets[:, 0], sets[:, 1]
    weights = jnp.zeros((REPLICATES, SAMPLES))
    pool, pool_log_weights = pooled(first, weights, second, weights)
    size = permutation_side_size(weights[0], weights[0])
    assert size == SAMPLES
    error = rejection_error(
        statistic,
        pool,
        pool_log_weights,
        (first, weights, second, weights),
        side_size=size,
        key=jax.random.key(21),
    )
    assert error < STANDARD_ERRORS


@pytest.mark.parametrize("statistic", [mean_difference, log_width_ratio])
def test_a_weighted_set_against_an_equal_weight_set_is_calibrated(
    statistic: Statistic,
) -> None:
    """Importance weights of a proposal 1.5 times wider keep the null calibrated."""
    case = zoo.scaled_gaussians(1.5, param_count=1)
    proposal = case.sample(jax.random.key(22), SAMPLES, batch_size=REPLICATES)
    target = case.sample(jax.random.key(23), SAMPLES // 2, batch_size=REPLICATES)
    weighted = proposal.samples[:, 1]
    log_densities = case.log_prob(weighted)
    weighted_log_weights = log_densities[..., 0] - log_densities[..., 1]
    equal = target.samples[:, 0]
    equal_log_weights = jnp.zeros((REPLICATES, SAMPLES // 2))
    pool, pool_log_weights = pooled(
        weighted, weighted_log_weights, equal, equal_log_weights
    )
    size = permutation_side_size(weighted_log_weights[0], equal_log_weights[0])
    assert size == (SAMPLES + SAMPLES // 2) // 2
    error = rejection_error(
        statistic,
        pool,
        pool_log_weights,
        (weighted, weighted_log_weights, equal, equal_log_weights),
        side_size=size,
        key=jax.random.key(24),
    )
    assert error < STANDARD_ERRORS


def test_the_public_path_reproduces_the_calibration_kernel() -> None:
    """permutation_null with null_draws gives the draws that the kernel evaluates."""
    case = zoo.shifted_gaussians(0.0, param_count=1)
    sets = case.sample(jax.random.key(25), 50).samples[0]
    key = jax.random.key(26)
    plan = permutation_null(sets[0], None, sets[1], None, key=key, draws=7)
    draws = null_draws(mean_difference, plan)
    first_sides, second_sides = random_sides(key, 100, 50, draws=7)
    pool = jnp.concatenate([sets[0], sets[1]])
    weights = jnp.zeros(100)
    expected = jnp.stack(
        [
            mean_difference(pool[first], weights[first], pool[second], weights[second])
            for first, second in zip(first_sides, second_sides, strict=True)
        ]
    )
    np.testing.assert_allclose(draws, expected, rtol=1e-12)
