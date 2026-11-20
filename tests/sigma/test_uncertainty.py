"""Tests for the half-sample draws and the half-sample standard error."""

import math
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mixed_signals import zoo
from mixed_signals.core.samples import kish_effective_sample_size
from mixed_signals.errors import PrecisionWarning
from mixed_signals.metrics.moments import mean_shift
from mixed_signals.nulls.plan import Statistic
from mixed_signals.prng import derive_key
from mixed_signals.sigma.uncertainty import (
    HALF_SAMPLE_PAIRS,
    half_sample_draws,
    half_sample_error,
)

KEY = jax.random.key(7)
FIRST_COUNT = 600
SECOND_COUNT = 400


def weighted_mean(samples: jax.Array, log_weights: jax.Array) -> jax.Array:
    """Return the weighted mean of every parameter."""
    return jnp.sum(jax.nn.softmax(log_weights)[:, None] * samples, axis=0)


def mean_difference(
    first: jax.Array,
    first_log_weights: jax.Array,
    second: jax.Array,
    second_log_weights: jax.Array,
) -> jax.Array:
    """Return the difference of the weighted means of the first parameter."""
    return (
        weighted_mean(first, first_log_weights)[0]
        - weighted_mean(second, second_log_weights)[0]
    )


def mean_differences(
    first: jax.Array,
    first_log_weights: jax.Array,
    second: jax.Array,
    second_log_weights: jax.Array,
) -> jax.Array:
    """Return the difference of the weighted means of every parameter."""
    return weighted_mean(first, first_log_weights) - weighted_mean(
        second, second_log_weights
    )


def mean_difference_matrix(
    first: jax.Array,
    first_log_weights: jax.Array,
    second: jax.Array,
    second_log_weights: jax.Array,
) -> jax.Array:
    """Return two rows of three values, to test a statistic with two leading axes."""
    differences = mean_differences(first, first_log_weights, second, second_log_weights)
    return jnp.stack([differences, 2 * differences])


def half_sizes(
    first: jax.Array,
    first_log_weights: jax.Array,
    second: jax.Array,
    second_log_weights: jax.Array,
) -> jax.Array:
    """Return the numbers of samples of the two halves."""
    assert first.shape[0] == first_log_weights.shape[0]
    assert second.shape[0] == second_log_weights.shape[0]
    return jnp.array([first.shape[0], second.shape[0]], dtype=jnp.float64)


def index_sums(
    first: jax.Array,
    first_log_weights: jax.Array,
    second: jax.Array,
    second_log_weights: jax.Array,
) -> jax.Array:
    """Return the sums of the positions in the two halves, which the samples hold."""
    return jnp.array([jnp.sum(first[:, 0]), jnp.sum(second[:, 0])])


def normal_sets(
    seed: int, first_count: int = FIRST_COUNT, second_count: int = SECOND_COUNT
) -> tuple[np.ndarray, np.ndarray]:
    """Return two independent standard normal sets in two parameters."""
    rng = np.random.default_rng(seed)
    return rng.normal(size=(first_count, 2)), rng.normal(size=(second_count, 2))


def test_the_default_number_of_pairs_is_two_hundred() -> None:
    """The subsampling bootstrap draws 200 pairs unless asked otherwise."""
    assert HALF_SAMPLE_PAIRS == 200
    first, second = normal_sets(0, 30, 30)
    assert half_sample_draws(
        mean_difference, first, None, second, None, key=KEY
    ).shape == (HALF_SAMPLE_PAIRS,)


def test_the_error_of_a_mean_difference_matches_its_analytic_value() -> None:
    """Halves of independent sets give the standard error of the mean difference."""
    first, second = normal_sets(1)
    pairs = 1_000
    draws = half_sample_draws(
        mean_difference, first, None, second, None, key=KEY, pairs=pairs
    )
    error = float(
        half_sample_error(draws, jnp.zeros(FIRST_COUNT), jnp.zeros(SECOND_COUNT))
    )
    # * Halves of a fixed set vary by the finite-population variance of its own samples.
    analytic = math.sqrt(
        np.var(first[:, 0], ddof=1) / FIRST_COUNT
        + np.var(second[:, 0], ddof=1) / SECOND_COUNT
    )
    monte_carlo_error = 1 / math.sqrt(2 * (pairs - 1))
    assert error == pytest.approx(analytic, rel=5 * monte_carlo_error)
    # * The sample variances scatter by sqrt(2/N), so the true σ matches more loosely.
    sample_scatter = math.sqrt(2 / FIRST_COUNT**3 + 2 / SECOND_COUNT**3) / (
        2 * (1 / FIRST_COUNT + 1 / SECOND_COUNT)
    )
    truth = math.sqrt(1 / FIRST_COUNT + 1 / SECOND_COUNT)
    assert error == pytest.approx(
        truth, rel=5 * math.hypot(monte_carlo_error, sample_scatter)
    )


def test_the_correlation_factor_follows_its_formula() -> None:
    """Kish sizes alone leave the spread as it is; n_eff scales it by the Kish ratio."""
    rng = np.random.default_rng(2)
    log_weights = [0.4 * rng.normal(size=FIRST_COUNT), 0.8 * rng.normal(size=60)]
    draws = jnp.asarray(rng.normal(size=(3, 50)))
    spread = np.std(np.asarray(draws), axis=-1, ddof=1)
    plain = half_sample_error(draws, log_weights[0], log_weights[1])
    np.testing.assert_allclose(plain, spread, rtol=1e-12)
    kish = [
        float(kish_effective_sample_size(jnp.asarray(entry))) for entry in log_weights
    ]
    n_eff = (kish[0] / 2, kish[1] / 3)
    corrected = half_sample_error(
        draws,
        log_weights[0],
        log_weights[1],
        first_n_eff=n_eff[0],
        second_n_eff=n_eff[1],
    )
    factor = math.sqrt((1 / n_eff[0] + 1 / n_eff[1]) / (1 / kish[0] + 1 / kish[1]))
    assert factor > 1.0
    np.testing.assert_allclose(corrected, factor * spread, rtol=1e-12)
    # * Only one effective size given: the other stays at its Kish size.
    first_only = half_sample_error(
        draws, log_weights[0], log_weights[1], first_n_eff=n_eff[0]
    )
    one_sided = math.sqrt((1 / n_eff[0] + 1 / kish[1]) / (1 / kish[0] + 1 / kish[1]))
    np.testing.assert_allclose(first_only, one_sided * spread, rtol=1e-12)


def test_weighted_sets_without_correlation_keep_the_spread() -> None:
    """Importance weights alone leave the factor at one, as the Kish sizes cancel."""
    rng = np.random.default_rng(3)
    first_weights = 0.7 * rng.normal(size=500)
    second_weights = jnp.zeros(300)
    draws = jnp.asarray(rng.normal(size=40))
    error = half_sample_error(draws, first_weights, second_weights)
    assert float(error) == pytest.approx(float(np.std(np.asarray(draws), ddof=1)))


def test_a_pair_does_not_depend_on_the_number_of_pairs() -> None:
    """Pair i keeps its halves, and so its value, whatever the number of pairs."""
    first, second = normal_sets(4, 80, 50)
    few = half_sample_draws(
        mean_differences, first, None, second, None, key=KEY, pairs=5
    )
    many = half_sample_draws(
        mean_differences, first, None, second, None, key=KEY, pairs=12
    )
    np.testing.assert_array_equal(few, many[:, :5])


def test_the_halves_follow_the_key_path_of_their_pair() -> None:
    """Pair i cuts the first floor(N/2) positions of permutations from its own keys."""
    first_count, second_count = 51, 40
    first = jnp.arange(float(first_count))[:, None] * jnp.ones((1, 2))
    second = jnp.arange(float(second_count))[:, None] * jnp.ones((1, 2))
    draws = half_sample_draws(index_sums, first, None, second, None, key=KEY, pairs=6)
    assert draws.shape == (2, 6)
    for index in range(6):
        pair_key = derive_key(KEY, "pair", index)
        first_order = jax.random.permutation(derive_key(pair_key, "first"), first_count)
        second_order = jax.random.permutation(
            derive_key(pair_key, "second"), second_count
        )
        assert float(draws[0, index]) == float(jnp.sum(first_order[:25]))
        assert float(draws[1, index]) == float(jnp.sum(second_order[:20]))
    other = half_sample_draws(
        index_sums, first, None, second, None, key=jax.random.key(8), pairs=6
    )
    assert not np.array_equal(draws, other)
    # * A half never holds a position twice, so its sum has an upper bound.
    assert float(jnp.max(draws[0])) <= float(jnp.sum(jnp.arange(26.0, 51.0)))


def test_pairs_differ_from_each_other() -> None:
    """Different pairs give different halves."""
    first, second = normal_sets(5, 100, 100)
    draws = half_sample_draws(
        mean_difference, first, None, second, None, key=KEY, pairs=4
    )
    assert len(set(np.asarray(draws).tolist())) == 4


@pytest.mark.parametrize(
    ("statistic", "shape"),
    [
        (mean_difference, ()),
        (mean_differences, (2,)),
        (mean_difference_matrix, (2, 2)),
    ],
)
def test_the_draws_axis_comes_last(
    statistic: Statistic, shape: tuple[int, ...]
) -> None:
    """Scalar, vector and matrix statistics give (..., pairs), with draws last."""
    first, second = normal_sets(6, 40, 30)
    pairs = 7
    draws = half_sample_draws(
        statistic, first, None, second, None, key=KEY, pairs=pairs
    )
    assert draws.shape == (*shape, pairs)
    error = half_sample_error(draws, jnp.zeros(40), jnp.zeros(30))
    assert error.shape == shape
    assert bool(jnp.all(jnp.isfinite(error)))


def test_a_batch_size_changes_nothing_but_the_evaluation_order() -> None:
    """Evaluating several pairs at once gives the same values."""
    first, second = normal_sets(7, 60, 45)
    single = half_sample_draws(
        mean_differences, first, None, second, None, key=KEY, pairs=10
    )
    batched = half_sample_draws(
        mean_differences, first, None, second, None, key=KEY, pairs=10, batch_size=4
    )
    np.testing.assert_allclose(batched, single, rtol=1e-12)


@pytest.mark.parametrize(
    ("first_count", "second_count", "first_half", "second_half"),
    [(101, 40, 50, 20), (2, 3, 1, 1), (7, 8, 3, 4)],
)
def test_halves_of_unequal_sets_hold_half_of_each_set(
    first_count: int, second_count: int, first_half: int, second_half: int
) -> None:
    """Each half holds the integer part of half of its own set."""
    first, second = normal_sets(8, first_count, second_count)
    draws = half_sample_draws(half_sizes, first, None, second, None, key=KEY, pairs=3)
    np.testing.assert_array_equal(draws[0], first_half)
    np.testing.assert_array_equal(draws[1], second_half)


def test_zero_weight_samples_never_enter_a_weighted_statistic() -> None:
    """Rows with zero weight sit in the halves but carry no weight."""
    first, second = normal_sets(9, 200, 150)
    first_weights = np.where(np.arange(200) % 5 == 0, -np.inf, 0.0)
    first = np.where(first_weights[:, None] == -np.inf, 1e6, first)
    draws = half_sample_draws(
        mean_difference, first, first_weights, second, None, key=KEY, pairs=30
    )
    assert bool(jnp.all(jnp.isfinite(draws)))
    assert float(jnp.max(jnp.abs(draws))) < 1.0
    error = half_sample_error(draws, jnp.asarray(first_weights), jnp.zeros(150))
    assert 0.0 < float(error) < 1.0


def test_a_package_metric_runs_on_the_halves() -> None:
    """A package metric, with its space, gives one value per parameter and pair."""
    case = zoo.shifted_gaussians(0.5, param_count=2)
    sets = case.sample(jax.random.key(1), 300).samples[0]
    statistic = partial(mean_shift, space=case.space)
    draws = half_sample_draws(
        statistic, sets[0], None, sets[1], None, key=KEY, pairs=20
    )
    assert draws.shape == (2, 20)
    error = half_sample_error(draws, jnp.zeros(300), jnp.zeros(300))
    # * Two sets of 300 unit-variance draws: the error is near sqrt(2/300).
    assert float(error[0]) == pytest.approx(math.sqrt(2 / 300), rel=0.5)


def test_the_error_is_nan_where_fewer_than_two_draws_are_finite() -> None:
    """Draws that are not finite are ignored, and fewer than two leave NaN."""
    draws = jnp.array(
        [
            [1.0, jnp.nan, jnp.nan],
            [1.0, 2.0, jnp.nan],
            [1.0, 2.0, 4.0],
            [jnp.nan, jnp.inf, -jnp.inf],
            [jnp.inf, 3.0, 5.0],
        ]
    )
    error = half_sample_error(draws, jnp.zeros(10), jnp.zeros(10))
    assert math.isnan(float(error[0]))
    assert float(error[1]) == pytest.approx(np.std([1.0, 2.0], ddof=1))
    assert float(error[2]) == pytest.approx(np.std([1.0, 2.0, 4.0], ddof=1))
    assert math.isnan(float(error[3]))
    assert float(error[4]) == pytest.approx(np.std([3.0, 5.0], ddof=1))
    assert math.isnan(
        float(half_sample_error(jnp.array([2.0]), jnp.zeros(4), jnp.zeros(4)))
    )


@pytest.mark.parametrize("pairs", [True, 2.5, "3"])
def test_pairs_must_be_an_integer(pairs: object) -> None:
    """Booleans, floats and text are not counts of pairs."""
    first, second = normal_sets(10, 10, 10)
    with pytest.raises(TypeError):
        half_sample_draws(
            mean_difference,
            first,
            None,
            second,
            None,
            key=KEY,
            pairs=pairs,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("pairs", [0, -3])
def test_pairs_must_be_positive(pairs: int) -> None:
    """At least one pair is needed."""
    first, second = normal_sets(11, 10, 10)
    with pytest.raises(ValueError, match="pairs must be at least 1"):
        half_sample_draws(
            mean_difference, first, None, second, None, key=KEY, pairs=pairs
        )


def test_the_sets_are_checked_before_any_pair_is_drawn() -> None:
    """Shapes, parameter counts, sizes and values of the sets raise ValueError."""
    first, second = normal_sets(12, 10, 10)
    with pytest.raises(ValueError, match="same parameters, got 2 and 3"):
        half_sample_draws(
            mean_difference, first, None, np.zeros((10, 3)), None, key=KEY
        )
    with pytest.raises(ValueError, match="the first set needs at least 2 samples"):
        half_sample_draws(mean_difference, first[:1], None, second, None, key=KEY)
    with pytest.raises(ValueError, match="the second set needs at least 2 samples"):
        half_sample_draws(mean_difference, first, None, second[:1], None, key=KEY)
    with pytest.raises(ValueError, match="first samples must be shaped"):
        half_sample_draws(mean_difference, first[:, 0], None, second, None, key=KEY)
    with pytest.raises(ValueError, match="second log_weights must be shaped"):
        half_sample_draws(mean_difference, first, None, second, np.zeros(9), key=KEY)
    with pytest.raises(ValueError, match="first samples must be finite"):
        half_sample_draws(
            mean_difference, np.full((10, 2), np.nan), None, second, None, key=KEY
        )
    with pytest.raises(ValueError, match="second needs at least one sample"):
        half_sample_draws(
            mean_difference, first, None, second, np.full(10, -np.inf), key=KEY
        )
    with pytest.raises(TypeError, match="first samples must hold real numbers"):
        half_sample_draws(
            mean_difference,
            np.zeros((10, 2), dtype=complex),
            None,
            second,
            None,
            key=KEY,
        )


def test_narrow_floats_warn_and_are_cast() -> None:
    """Float32 sets raise a PrecisionWarning, as every package entry point does."""
    first, second = normal_sets(13, 20, 20)
    with pytest.warns(PrecisionWarning, match=r"first samples \(float32\)"):
        draws = half_sample_draws(
            mean_difference,
            first.astype(np.float32),
            None,
            second,
            None,
            key=KEY,
            pairs=3,
        )
    assert draws.dtype == jnp.float64


def test_the_error_checks_its_weights_and_sizes() -> None:
    """Weights must be vectors and sizes finite and at least one."""
    draws = jnp.arange(5.0)
    with pytest.raises(ValueError, match="log weights of the first set must be shaped"):
        half_sample_error(draws, jnp.zeros((4, 2)), jnp.zeros(4))
    with pytest.raises(
        ValueError, match="log weights of the second set must be shaped"
    ):
        half_sample_error(draws, jnp.zeros(4), jnp.zeros((4, 2)))
    with pytest.raises(ValueError, match="effective sample size of the first set"):
        half_sample_error(draws, jnp.zeros(4), jnp.zeros(4), first_n_eff=0.5)
    with pytest.raises(ValueError, match="effective sample size of the second set"):
        half_sample_error(draws, jnp.zeros(4), jnp.zeros(4), second_n_eff=math.nan)
    with pytest.raises(ValueError, match="Kish sample size of the first set"):
        half_sample_error(draws, jnp.full(4, -jnp.inf), jnp.zeros(4))
