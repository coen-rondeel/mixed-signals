"""Tests for importance weights: the evidence ratio, its error and the efficiency."""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import logsumexp

from mixed_signals import prng
from mixed_signals.core.samples import kish_effective_sample_size
from mixed_signals.reweight.importance import (
    EvidenceRatio,
    log_evidence_ratio,
    relative_efficiency,
    reweighted_effective_sample_size,
)

ROOT_KEY = prng.root_key(13)
SAMPLE_COUNT = 400


def random_log_weights(*path: str | int, shape: tuple[int, ...]) -> np.ndarray:
    """Return reproducible log weights of moderate spread."""
    key = prng.derive_key(ROOT_KEY, *path)
    return np.asarray(0.7 * jax.random.normal(key, shape))


def brute_force_error(input_log_weights: np.ndarray, log_ratio: np.ndarray) -> float:
    """Return sqrt(sum w² (r/Z − 1)²) for normalised input weights, from the formula."""
    weights = np.exp(input_log_weights - logsumexp(input_log_weights))
    ratios = np.exp(log_ratio)
    evidence = np.sum(weights * ratios)
    return float(np.sqrt(np.sum(weights**2 * (ratios / evidence - 1) ** 2)))


# * The evidence ratio and its error


def test_equal_input_weights_give_the_mean_ratio_and_the_error_of_dax() -> None:
    """The value is ln mean(r); the squared error is (1 − ε)/(N ε), ε = Kish/N."""
    log_ratio = random_log_weights("equal", shape=(SAMPLE_COUNT,))
    result = log_evidence_ratio(np.zeros(SAMPLE_COUNT), log_ratio)
    np.testing.assert_allclose(
        result.value, np.log(np.mean(np.exp(log_ratio))), rtol=1e-13
    )
    kish = float(kish_effective_sample_size(jnp.asarray(log_ratio)))
    epsilon = kish / SAMPLE_COUNT
    expected = math.sqrt((1 - epsilon) / (SAMPLE_COUNT * epsilon))
    np.testing.assert_allclose(result.error, expected, rtol=1e-11)
    assert isinstance(result, EvidenceRatio)


def test_the_relative_efficiency_divides_the_variance() -> None:
    """With R the squared error is the independent one over R."""
    log_ratio = random_log_weights("efficiency", shape=(SAMPLE_COUNT,))
    plain = log_evidence_ratio(np.zeros(SAMPLE_COUNT), log_ratio)
    correlated = log_evidence_ratio(np.zeros(SAMPLE_COUNT), log_ratio, 0.25)
    np.testing.assert_allclose(correlated.error**2, plain.error**2 / 0.25, rtol=1e-13)
    np.testing.assert_array_equal(correlated.value, plain.value)


def test_unequal_input_weights_match_the_brute_force_sum() -> None:
    """The error is sqrt(sum w² (r/Z − 1)²) for the normalised input weights w."""
    input_log_weights = random_log_weights("input", shape=(SAMPLE_COUNT,))
    log_ratio = random_log_weights("ratio", shape=(SAMPLE_COUNT,))
    result = log_evidence_ratio(input_log_weights, input_log_weights + log_ratio)
    np.testing.assert_allclose(
        result.error, brute_force_error(input_log_weights, log_ratio), rtol=1e-11
    )
    weights = np.exp(input_log_weights - logsumexp(input_log_weights))
    np.testing.assert_allclose(
        result.value, np.log(np.sum(weights * np.exp(log_ratio))), rtol=1e-12
    )


def test_weights_of_any_scale_give_the_same_ratio() -> None:
    """Adding a constant to both sets of log weights changes neither value nor error."""
    input_log_weights = random_log_weights("scale-input", shape=(SAMPLE_COUNT,))
    new_log_weights = input_log_weights + random_log_weights(
        "scale-ratio", shape=(SAMPLE_COUNT,)
    )
    reference = log_evidence_ratio(input_log_weights, new_log_weights)
    shifted = log_evidence_ratio(input_log_weights - 900.0, new_log_weights + 400.0)
    np.testing.assert_allclose(shifted.error, reference.error, rtol=1e-12)
    np.testing.assert_allclose(shifted.value, reference.value + 1_300.0, rtol=1e-12)


def test_zero_input_weights_drop_out() -> None:
    """Samples of zero input and new weight change neither the value nor the error."""
    input_log_weights = random_log_weights("zero-input", shape=(SAMPLE_COUNT,))
    log_ratio = random_log_weights("zero-ratio", shape=(SAMPLE_COUNT,))
    with_zeros = np.concatenate([input_log_weights, np.full(25, -np.inf)])
    new_with_zeros = np.concatenate(
        [input_log_weights + log_ratio, np.full(25, -np.inf)]
    )
    plain = log_evidence_ratio(input_log_weights, input_log_weights + log_ratio)
    padded = log_evidence_ratio(with_zeros, new_with_zeros)
    np.testing.assert_allclose(padded.value, plain.value, rtol=1e-13)
    np.testing.assert_allclose(padded.error, plain.error, rtol=1e-13)


def test_leading_axes_broadcast_and_the_function_runs_under_jit() -> None:
    """Each leading row is its own set; R may be one value or one per set."""
    input_log_weights = random_log_weights("lead-input", shape=(2, 3, 120))
    new_log_weights = input_log_weights + random_log_weights(
        "lead-ratio", shape=(2, 3, 120)
    )
    efficiency = np.array([[1.0, 0.5, 0.25], [0.8, 0.6, 0.4]])
    mapped = log_evidence_ratio(input_log_weights, new_log_weights, efficiency)
    assert mapped.value.shape == mapped.error.shape == (2, 3)
    scalar = log_evidence_ratio(input_log_weights, new_log_weights, 0.5)
    assert scalar.error.shape == (2, 3)
    default = log_evidence_ratio(input_log_weights, new_log_weights)
    np.testing.assert_allclose(scalar.error**2, 2 * default.error**2, rtol=1e-12)
    jitted = jax.jit(log_evidence_ratio)(input_log_weights, new_log_weights, efficiency)
    for row in range(2):
        for column in range(3):
            single = log_evidence_ratio(
                input_log_weights[row, column],
                new_log_weights[row, column],
                efficiency[row, column],
            )
            np.testing.assert_allclose(mapped.value[row, column], single.value)
            np.testing.assert_allclose(mapped.error[row, column], single.error)
    np.testing.assert_allclose(jitted.error, mapped.error, rtol=1e-12)


# * Efficiency and effective sample size


def test_the_relative_efficiency_is_n_eff_over_the_kish_value() -> None:
    """R is n_eff over the Kish value of the weights, and one without n_eff."""
    log_weights = random_log_weights("kish", shape=(2, 3, 200))
    kish = np.asarray(kish_effective_sample_size(jnp.asarray(log_weights)))
    n_eff = 0.6 * kish
    np.testing.assert_allclose(relative_efficiency(log_weights, n_eff), 0.6)
    np.testing.assert_array_equal(relative_efficiency(log_weights), np.ones((2, 3)))
    np.testing.assert_array_equal(
        relative_efficiency(log_weights[0, 0]), np.asarray(1.0)
    )


def test_the_reweighted_size_is_the_efficiency_times_the_kish_value() -> None:
    """With R = 0.5 the size is half the Kish value of the new weights."""
    log_weights = random_log_weights("size", shape=(2, 150))
    kish = np.asarray(kish_effective_sample_size(jnp.asarray(log_weights)))
    np.testing.assert_allclose(
        reweighted_effective_sample_size(log_weights), kish, rtol=1e-13
    )
    np.testing.assert_allclose(
        reweighted_effective_sample_size(log_weights, 0.5), 0.5 * kish, rtol=1e-13
    )


def test_the_reweighted_size_is_clipped_to_one_and_the_positive_count() -> None:
    """An efficiency cannot lift the size above the positive count or below one."""
    equal = jnp.where(jnp.arange(60) < 40, 0.0, -jnp.inf)
    assert float(reweighted_effective_sample_size(equal)) == pytest.approx(40.0)
    assert float(reweighted_effective_sample_size(equal, 3.0)) == 40.0
    assert float(reweighted_effective_sample_size(equal, 0.001)) == 1.0
    dominated = jnp.array([50.0, 0.0, 0.0, 0.0])
    assert float(reweighted_effective_sample_size(dominated)) == pytest.approx(1.0)
