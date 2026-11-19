"""Tests for Pareto smoothed importance sampling: tail length, k̂ and smoothing."""

import inspect
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import stats

from mixed_signals import prng, zoo
from mixed_signals.reweight.psis import (
    MINIMUM_TAIL_DRAWS,
    PARETO_K_LIMIT,
    ParetoTail,
    SmoothedWeights,
    _excess_quantile,
    pareto_k_threshold,
    pareto_smooth,
    pareto_tail,
    pareto_tail_length,
)
from mixed_signals.sigma.pareto import generalized_pareto_fit

REPLICATES = 40
DRAWS = 10_000
TAIL_DRAWS = 300
STANDARD_ERRORS = 4
ROOT_KEY = prng.root_key(2026)


def exponential_pair_log_ratios(rate: float) -> np.ndarray:
    """Return log ratios of exponential(1) over exponential(rate), shaped (R, S).

    The ratio is exp((rate − 1) θ)/rate for θ drawn from the proposal, so its
    tail has the true shape 1 − 1/rate.
    """
    key = prng.derive_key(ROOT_KEY, "exponential-pair", int(10 * rate))
    theta = jax.random.exponential(key, (REPLICATES, DRAWS)) / rate
    return np.asarray((rate - 1) * theta - math.log(rate))


def narrower_gaussian_log_ratios(ratio: float) -> np.ndarray:
    """Return log ratios of the zoo reference over its narrower scaled posterior.

    Both posteriors are Gaussians of two parameters. With the draws z of the
    scaled one, (z₁² + z₂²)/2 is exponential, so the weights are exactly Pareto
    with the true shape 1 − ratio².
    """
    case = zoo.scaled_gaussians(ratio, param_count=2, axes=(0, 1))
    key = prng.derive_key(ROOT_KEY, "narrower-gaussians", int(10 * ratio))
    sample_set = case.sample(key, DRAWS, batch_size=REPLICATES)
    points = sample_set.samples[:, case.names.index("scaled")]
    log_density = case.log_prob(points)
    log_ratio: np.ndarray = np.asarray(
        log_density[..., case.names.index("reference")]
        - log_density[..., case.names.index("scaled")]
    )
    return log_ratio


def tail_shapes(log_ratios: np.ndarray) -> jax.Array:
    """Return k̂ of every replicate row of log ratios."""
    return jax.vmap(lambda row: pareto_tail(row, tail_draws=TAIL_DRAWS).shape)(
        jnp.asarray(log_ratios)
    )


def raw_tail_excesses(log_weights: np.ndarray, tail_draws: int) -> np.ndarray:
    """Return the excesses over the cutoff, in units of the largest weight."""
    weights = np.sort(np.exp(log_weights - log_weights.max()))
    excesses: np.ndarray = weights[-tail_draws:] - weights[-tail_draws - 1]
    return excesses


# * Tail length and threshold


@pytest.mark.parametrize(
    ("draws", "relative_efficiency", "expected"),
    [
        (100, 1.0, 20),
        (1_000, 1.0, 94),
        (2_000, 1.0, 134),
        (10_000, 1.0, 300),
        (10_000, 0.5, 424),
        (100, 0.5, 20),
        (24, 1.0, 4),
        (25, 1.0, 5),
        (1, 1.0, 0),
    ],
)
def test_the_tail_length_follows_the_rule(
    draws: int, relative_efficiency: float, expected: int
) -> None:
    """M is the floor of min(0.2 S, 3 sqrt(S/R)), such as 94 at S = 1 000."""
    assert pareto_tail_length(draws, relative_efficiency) == expected


def test_the_tail_length_errors() -> None:
    """Draws below one, non-integers and a bad efficiency raise ValueError.

    The test session checks the annotations first, so the test calls the
    unwrapped function for the types, as runs without runtime checks do.
    """
    unchecked = inspect.unwrap(pareto_tail_length)
    with pytest.raises(ValueError, match="draws must be at least 1, got 0"):
        pareto_tail_length(0)
    with pytest.raises(ValueError, match=r"draws must be an integer, got 10\.5"):
        unchecked(10.5)
    with pytest.raises(ValueError, match="draws must be an integer, got True"):
        unchecked(True)
    for bad in (0.0, -1.0, math.nan, math.inf):
        with pytest.raises(ValueError, match="relative_efficiency must be finite"):
            pareto_tail_length(100, bad)


@pytest.mark.parametrize(
    ("draws", "expected"),
    [(100, 0.5), (1_000, 2 / 3), (2_000, 0.697_07), (10_000, 0.7), (2, -2.3219)],
)
def test_the_threshold_follows_the_rule(draws: int, expected: float) -> None:
    """The threshold is min(1 − 1/log10 S, 0.7): 0.5 at 100 and 0.7 from 2 154 up."""
    assert pareto_k_threshold(draws) == pytest.approx(expected, abs=5e-5)
    assert pareto_k_threshold(draws) <= PARETO_K_LIMIT


def test_the_threshold_reaches_its_limit_at_about_2154_draws() -> None:
    """1 − 1/log10 S equals 0.7 at S = 10^(10/3) = 2 154.4."""
    assert pareto_k_threshold(2_154) < PARETO_K_LIMIT
    assert pareto_k_threshold(2_155) == PARETO_K_LIMIT


def test_the_threshold_needs_two_draws() -> None:
    """One draw has log10 S = 0, so the rule is undefined."""
    with pytest.raises(ValueError, match="draws must be at least 2, got 1"):
        pareto_k_threshold(1)


# * k̂ of exact Pareto weights


@pytest.mark.parametrize("rate", [1.5, 2.0, 4.0])
def test_the_exponential_pair_gives_its_true_shape(rate: float) -> None:
    """A proposal exponential(rate) has k = 1 − 1/rate; mean k̂ is within 4 errors."""
    shapes = np.asarray(tail_shapes(exponential_pair_log_ratios(rate)))
    error = shapes.std(ddof=1) / math.sqrt(REPLICATES)
    assert abs(shapes.mean() - (1 - 1 / rate)) < STANDARD_ERRORS * error


@pytest.mark.parametrize("ratio", [0.8, 0.6, 0.4])
def test_narrower_gaussians_give_their_true_shape(ratio: float) -> None:
    """Narrowing two axes by r gives exactly Pareto weights with k = 1 − r²."""
    shapes = np.asarray(tail_shapes(narrower_gaussian_log_ratios(ratio)))
    error = shapes.std(ddof=1) / math.sqrt(REPLICATES)
    assert abs(shapes.mean() - (1 - ratio**2)) < STANDARD_ERRORS * error


def test_the_shape_is_the_fit_of_the_excesses_with_or_without_the_prior() -> None:
    """Without the prior k̂ is the plain fit; with it, (M k + 5)/(M + 10) of that."""
    log_weights = exponential_pair_log_ratios(2.0)[0]
    excesses = raw_tail_excesses(log_weights, TAIL_DRAWS)
    plain = generalized_pareto_fit(excesses)
    unshrunk = pareto_tail(log_weights, tail_draws=TAIL_DRAWS, shape_prior=False)
    shrunk = pareto_tail(log_weights, tail_draws=TAIL_DRAWS)
    np.testing.assert_allclose(unshrunk.shape, plain.shape, rtol=1e-10)
    np.testing.assert_allclose(unshrunk.scale, plain.scale, rtol=1e-10)
    expected = (TAIL_DRAWS * float(plain.shape) + 5) / (TAIL_DRAWS + 10)
    np.testing.assert_allclose(shrunk.shape, expected, rtol=1e-10)
    np.testing.assert_array_equal(shrunk.scale, unshrunk.scale)
    cutoff = np.sort(log_weights - log_weights.max())[-TAIL_DRAWS - 1]
    np.testing.assert_allclose(shrunk.cutoff, cutoff, rtol=1e-12)


def test_the_weights_enter_through_their_logarithm_up_to_a_scale() -> None:
    """Adding a constant to every log weight leaves k̂ and the cutoff unchanged."""
    log_weights = exponential_pair_log_ratios(2.0)[1]
    reference = pareto_tail(log_weights, tail_draws=TAIL_DRAWS)
    shifted = pareto_tail(log_weights + 700.0, tail_draws=TAIL_DRAWS)
    np.testing.assert_allclose(shifted.shape, reference.shape, rtol=1e-10)
    np.testing.assert_allclose(shifted.cutoff, reference.cutoff, rtol=1e-10)


# * Flat and undefined tails


def test_identical_weights_have_a_flat_tail() -> None:
    """Equal weights are bounded at the cutoff: k̂ is −∞ and the scale 0."""
    tail = pareto_tail(jnp.zeros(100), tail_draws=20)
    assert float(tail.shape) == -math.inf
    assert float(tail.scale) == 0.0
    assert float(tail.cutoff) == 0.0


def test_weights_of_zero_or_one_value_have_a_flat_tail() -> None:
    """More than M + 1 weights at one value, the rest zero, are flat."""
    log_weights = jnp.where(jnp.arange(100) < 30, math.log(3.0), -jnp.inf)
    tail = pareto_tail(log_weights, tail_draws=20)
    assert float(tail.shape) == -math.inf
    assert float(tail.scale) == 0.0
    # * Exactly M + 1 equal positive weights: the cutoff is the common value.
    boundary = jnp.where(jnp.arange(100) < 21, 1.0, -jnp.inf)
    assert float(pareto_tail(boundary, tail_draws=20).shape) == -math.inf


def test_fewer_than_one_more_than_the_tail_positive_weights_give_nan() -> None:
    """With M positive weights there is no cutoff weight, so k̂ is NaN."""
    log_weights = jnp.where(jnp.arange(100) < 20, 0.0, -jnp.inf)
    tail = pareto_tail(log_weights, tail_draws=20)
    assert math.isnan(float(tail.shape))
    assert math.isnan(float(tail.scale))
    assert float(tail.cutoff) == -math.inf


def test_weights_that_are_all_zero_give_nan() -> None:
    """A set without a positive weight has no tail."""
    tail = pareto_tail(jnp.full(100, -jnp.inf), tail_draws=20)
    assert math.isnan(float(tail.shape))
    assert math.isnan(float(tail.scale))


def test_the_flat_tolerance_separates_flat_from_nearly_flat_tails() -> None:
    """A largest weight 1e-13 above the rest is flat; 1e-11 above is not."""
    flat = pareto_tail(jnp.zeros(100).at[0].set(1e-13), tail_draws=20)
    assert float(flat.shape) == -math.inf
    # * One excess of 1e-11 and 19 of zero leave a zero quartile, so the fit is NaN.
    nearly_flat = pareto_tail(jnp.zeros(100).at[0].set(1e-11), tail_draws=20)
    assert math.isnan(float(nearly_flat.shape))
    assert math.isnan(float(nearly_flat.scale))


def test_the_tail_length_is_checked_statically() -> None:
    """Fewer than five tail draws, or no weight left for the cutoff, raise."""
    log_weights = jnp.zeros(50)
    with pytest.raises(ValueError, match="tail_draws must lie between 5 and 49"):
        pareto_tail(log_weights, tail_draws=MINIMUM_TAIL_DRAWS - 1)
    with pytest.raises(ValueError, match="tail_draws must lie between 5 and 49"):
        pareto_tail(log_weights, tail_draws=50)
    with pytest.raises(ValueError, match="tail_draws must lie between 5 and 49"):
        pareto_smooth(log_weights, tail_draws=50)
    assert float(pareto_tail(log_weights, tail_draws=49).shape) == -math.inf
    assert float(pareto_tail(log_weights, tail_draws=5).shape) == -math.inf


# * Smoothing


def smoothing_case(
    rate: float = 2.0, draws: int = 1_000, tail_draws: int = 60
) -> tuple[np.ndarray, np.ndarray, ParetoTail, SmoothedWeights]:
    """Return log weights, their order, their tail fit and their smoothed version."""
    key = prng.derive_key(ROOT_KEY, "smoothing", int(10 * rate), draws)
    theta = np.asarray(jax.random.exponential(key, (draws,))) / rate
    log_weights = (rate - 1) * theta - math.log(rate)
    tail = pareto_tail(log_weights, tail_draws=tail_draws)
    smoothed = pareto_smooth(log_weights, tail_draws=tail_draws)
    return log_weights, np.argsort(log_weights), tail, smoothed


@pytest.mark.parametrize("rate", [1.5, 4.0])
def test_the_smoothed_tail_is_the_pareto_quantiles(rate: float) -> None:
    """The tail equals SciPy's quantiles at (z − 1/2)/M, up to the truncation."""
    tail_draws = 60
    log_weights, order, tail, smoothed = smoothing_case(rate, 1_000, tail_draws)
    scaled = np.exp(np.asarray(smoothed.log_weights) - log_weights.max())
    probabilities = (np.arange(1, tail_draws + 1) - 0.5) / tail_draws
    expected = stats.genpareto.ppf(
        probabilities,
        c=float(tail.shape),
        loc=math.exp(float(tail.cutoff)),
        scale=float(tail.scale),
    )
    smoothed_tail = scaled[order[-tail_draws:]]
    below_maximum = expected < 1.0
    assert below_maximum.any()
    np.testing.assert_allclose(
        smoothed_tail[below_maximum], expected[below_maximum], rtol=1e-12
    )
    np.testing.assert_allclose(smoothed_tail[~below_maximum], 1.0, rtol=1e-12)


def test_smoothing_truncates_at_the_largest_raw_weight() -> None:
    """With a heavy tail the top fitted quantiles pass the largest weight: capped."""
    log_weights, _, tail, smoothed = smoothing_case(4.0, 1_000, 60)
    probabilities = (np.arange(1, 61) - 0.5) / 60
    untruncated = stats.genpareto.ppf(
        probabilities,
        c=float(tail.shape),
        loc=math.exp(float(tail.cutoff)),
        scale=float(tail.scale),
    )
    assert (untruncated > 1.0).any()
    assert float(jnp.max(smoothed.log_weights)) == pytest.approx(log_weights.max())
    assert np.asarray(smoothed.log_weights).max() <= log_weights.max()


def test_smoothing_leaves_the_rest_of_the_weights_and_the_ranks_alone() -> None:
    """Weights outside the tail keep their exact values, and no rank reverses."""
    log_weights, order, tail, smoothed = smoothing_case(2.0, 1_000, 60)
    new = np.asarray(smoothed.log_weights)
    np.testing.assert_array_equal(new[order[:-60]], log_weights[order[:-60]])
    assert np.all(np.diff(new[order]) >= 0)
    assert not np.array_equal(new[order[-60:]], log_weights[order[-60:]])
    assert float(smoothed.shape) == float(tail.shape)


def test_smoothing_with_and_without_the_prior_returns_the_matching_shape() -> None:
    """The returned shape is that of pareto_tail with the same prior option."""
    key = prng.derive_key(ROOT_KEY, "smoothing-prior")
    log_weights = jax.random.normal(key, (500,)) * 1.5
    for shape_prior in (True, False):
        smoothed = pareto_smooth(log_weights, tail_draws=40, shape_prior=shape_prior)
        tail = pareto_tail(log_weights, tail_draws=40, shape_prior=shape_prior)
        assert float(smoothed.shape) == float(tail.shape)
    assert float(pareto_tail(log_weights, tail_draws=40).shape) != float(
        pareto_tail(log_weights, tail_draws=40, shape_prior=False).shape
    )


def test_smoothing_keeps_zero_weights_at_zero() -> None:
    """A weight of zero (−∞) outside the tail stays zero, and the tail stays finite."""
    key = prng.derive_key(ROOT_KEY, "smoothing-zeros")
    log_weights = np.array(jax.random.normal(key, (300,)) * 1.5)
    log_weights[:50] = -np.inf
    smoothed = pareto_smooth(log_weights, tail_draws=30)
    new = np.asarray(smoothed.log_weights)
    assert np.all(new[:50] == -np.inf)
    assert np.all(np.isfinite(new[50:]))
    assert math.isfinite(float(smoothed.shape))


@pytest.mark.parametrize(
    "log_weights",
    [
        jnp.zeros(100),
        jnp.where(jnp.arange(100) < 15, 0.0, -jnp.inf),
        jnp.full(100, -jnp.inf),
    ],
    ids=["flat", "undefined", "all-zero"],
)
def test_flat_and_undefined_tails_are_returned_unchanged(
    log_weights: jax.Array,
) -> None:
    """A shape that is −∞ or NaN leaves every weight as it was."""
    smoothed = pareto_smooth(log_weights, tail_draws=20)
    np.testing.assert_array_equal(smoothed.log_weights, log_weights)
    assert not math.isfinite(float(smoothed.shape))


def test_the_quantile_term_has_its_exponential_limit_at_zero_shape() -> None:
    """At k = 0 the quantile is −σ log(1 − p), and a tiny k agrees with it."""
    probabilities = jnp.array([0.05, 0.5, 0.95])
    scale = jnp.asarray(2.0)
    exponential = _excess_quantile(probabilities, jnp.asarray(0.0), scale)
    np.testing.assert_allclose(
        exponential, stats.genpareto.ppf(np.asarray(probabilities), 0.0, scale=2.0)
    )
    nearby = _excess_quantile(probabilities, jnp.asarray(1e-9), scale)
    np.testing.assert_allclose(nearby, exponential, rtol=1e-8)
    for shape in (-0.4, 0.7):
        np.testing.assert_allclose(
            _excess_quantile(probabilities, jnp.asarray(shape), scale),
            stats.genpareto.ppf(np.asarray(probabilities), shape, scale=2.0),
            rtol=1e-12,
        )


# * Transformations


def test_the_functions_run_under_jit_and_vmap() -> None:
    """Static tail length and prior option compile, and vmap matches a loop."""
    rows = jnp.asarray(exponential_pair_log_ratios(2.0)[:3, :2_000])
    mapped_tail = jax.vmap(lambda row: pareto_tail(row, tail_draws=134))(rows)
    mapped_smooth = jax.vmap(lambda row: pareto_smooth(row, tail_draws=134))(rows)
    jitted_tail = jax.jit(pareto_tail, static_argnames=("tail_draws", "shape_prior"))
    jitted_smooth = jax.jit(
        pareto_smooth, static_argnames=("tail_draws", "shape_prior")
    )
    assert mapped_tail.shape.shape == (3,)
    assert mapped_smooth.log_weights.shape == (3, 2_000)
    for index in range(3):
        single = jitted_tail(rows[index], tail_draws=134, shape_prior=True)
        smoothed = jitted_smooth(rows[index], tail_draws=134, shape_prior=True)
        np.testing.assert_allclose(mapped_tail.shape[index], single.shape, rtol=1e-12)
        np.testing.assert_allclose(
            mapped_smooth.log_weights[index], smoothed.log_weights, rtol=1e-12
        )
    assert isinstance(single, ParetoTail)
    assert isinstance(smoothed, SmoothedWeights)
