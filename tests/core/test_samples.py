"""Tests for ``SampleSet``: layouts, weights, checks, operations and conversions."""

import dataclasses
import math
from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import stats

from mixed_signals.core.samples import (
    N_EFF_RELATIVE_TOLERANCE,
    SampleSet,
    SampleShape,
    circular_standard_deviation,
    kish_effective_sample_size,
    weighted_covariance,
)
from mixed_signals.errors import PrecisionWarning

SAMPLE_COUNT = 40
WEIGHT_SLOPE = -1e-3


@pytest.fixture
def rng() -> np.random.Generator:
    """A seeded NumPy generator for test data."""
    return np.random.default_rng(2026)


def labelled_set(batch_size: int = 2, posterior_count: int = 3) -> SampleSet:
    """Return a set whose fields all encode the identity of each sample.

    The first parameter is a running sample identity; every per-sample field is
    a fixed function of it, so a misaligned field shows up as a wrong value.
    """
    sample_id = np.arange(
        batch_size * posterior_count * SAMPLE_COUNT, dtype=float
    ).reshape(batch_size, posterior_count, SAMPLE_COUNT)
    return SampleSet(
        np.stack([sample_id, -sample_id, sample_id / 7], axis=-1),
        log_weights=WEIGHT_SLOPE * sample_id,
        n_eff=np.arange(batch_size * posterior_count, dtype=float).reshape(
            batch_size, posterior_count
        )
        + 10,
        log_like=sample_id,
        log_prior=-sample_id,
        log_q=2 * sample_id,
    )


def assert_fields_follow_samples(sample_set: SampleSet) -> None:
    """Check that every per-sample field still matches the sample identity."""
    sample_id = np.asarray(sample_set.samples[..., 0])
    np.testing.assert_array_equal(sample_set.samples[..., 1], -sample_id)
    np.testing.assert_allclose(sample_set.log_weights, WEIGHT_SLOPE * sample_id)
    np.testing.assert_array_equal(sample_set.log_like, sample_id)
    np.testing.assert_array_equal(sample_set.log_prior, -sample_id)
    np.testing.assert_array_equal(sample_set.log_q, 2 * sample_id)


# * Layouts


def test_every_input_layout_builds_the_same_container(
    rng: np.random.Generator,
) -> None:
    """One posterior given as (N, D), (1, N, D) or (1, 1, N, D) is one container."""
    draws = rng.normal(size=(SAMPLE_COUNT, 3))
    weights = rng.normal(size=SAMPLE_COUNT)
    from_flat = SampleSet(draws, log_weights=weights, n_eff=5.0)
    from_stack = SampleSet(draws[None], log_weights=weights[None], n_eff=[5.0])
    from_batch = SampleSet(
        draws[None, None], log_weights=weights[None, None], n_eff=[[5.0]]
    )
    assert from_flat.shape == (1, 1, SAMPLE_COUNT, 3)
    assert eqx.tree_equal(from_flat, from_stack)
    assert eqx.tree_equal(from_flat, from_batch)


def test_stacked_posteriors_get_a_batch_axis(rng: np.random.Generator) -> None:
    """K posteriors shaped (K, N, D) become one problem with K posteriors."""
    sample_set = SampleSet(
        rng.normal(size=(4, SAMPLE_COUNT, 2)),
        log_like=np.zeros((4, SAMPLE_COUNT)),
        n_eff=np.full(4, 3.0),
    )
    assert sample_set.shape == SampleShape(1, 4, SAMPLE_COUNT, 2)
    assert sample_set.log_like is not None
    assert sample_set.log_like.shape == (1, 4, SAMPLE_COUNT)
    assert sample_set.n_eff is not None
    assert sample_set.n_eff.shape == (1, 4)


def test_batches_keep_their_layout() -> None:
    """Inputs shaped (B, K, N, D) keep their shape, field by field."""
    sample_set = labelled_set(batch_size=2, posterior_count=3)
    assert sample_set.shape == SampleShape(
        batch=2, posterior=3, sample=SAMPLE_COUNT, param=3
    )
    assert sample_set.log_weights.shape == (2, 3, SAMPLE_COUNT)
    assert sample_set.n_eff is not None
    assert sample_set.n_eff.shape == (2, 3)


def test_shape_compares_equal_to_a_plain_tuple() -> None:
    """The shape is a named tuple that also compares equal to a plain tuple."""
    shape = labelled_set(batch_size=1, posterior_count=2).shape
    assert shape == (1, 2, SAMPLE_COUNT, 3)
    assert (shape.batch, shape.posterior, shape.sample, shape.param) == shape


def test_optional_fields_default_to_none_and_equal_weights() -> None:
    """Without optional inputs, the weights are equal and the rest is unset."""
    sample_set = SampleSet(np.zeros((SAMPLE_COUNT, 2)))
    np.testing.assert_array_equal(sample_set.log_weights, 0.0)
    assert sample_set.n_eff is None
    assert sample_set.log_like is None
    assert sample_set.log_prior is None
    assert sample_set.log_q is None


def test_integer_and_bool_inputs_become_float64_without_warning() -> None:
    """Integer and bool inputs lose no precision in float64, so nothing warns."""
    sample_set = SampleSet(
        np.arange(6).reshape(3, 2), log_weights=np.array([True, True, False])
    )
    assert sample_set.samples.dtype == jnp.float64
    assert sample_set.log_weights.dtype == jnp.float64


def test_narrow_floats_warn_once_and_name_each_field() -> None:
    """Narrower floats are cast to float64 with one warning naming each field."""
    with pytest.warns(PrecisionWarning) as records:
        sample_set = SampleSet(
            np.zeros((SAMPLE_COUNT, 2), dtype=np.float32),
            log_q=jnp.zeros(SAMPLE_COUNT, dtype=jnp.bfloat16),
        )
    assert len(records) == 1
    message = str(records[0].message)
    assert "samples (float32)" in message
    assert "log_q (bfloat16)" in message
    assert sample_set.samples.dtype == jnp.float64
    assert sample_set.log_q is not None
    assert sample_set.log_q.dtype == jnp.float64


# * Weights and the effective sample size


def test_kish_size_of_equal_weights_is_the_sample_count() -> None:
    """Equal weights give the number of samples, whatever their scale."""
    for log_weight in (0.0, 700.0, -800.0):
        size = kish_effective_sample_size(jnp.full(SAMPLE_COUNT, log_weight))
        assert float(size) == pytest.approx(SAMPLE_COUNT, rel=1e-12)


def test_kish_size_matches_the_direct_formula(rng: np.random.Generator) -> None:
    """The log-space evaluation equals (Σ w)² / Σ w² on ordinary weights."""
    weights = rng.uniform(0.1, 2.0, size=(3, SAMPLE_COUNT))
    expected = weights.sum(axis=-1) ** 2 / (weights**2).sum(axis=-1)
    computed = kish_effective_sample_size(jnp.log(weights))
    np.testing.assert_allclose(computed, expected, rtol=1e-12)


def test_kish_size_counts_positive_weights_only() -> None:
    """Zero weights, given as log weights of -inf, do not count."""
    log_weights = jnp.log(jnp.array([1.0, 1.0, 0.0, 0.0]))
    assert float(kish_effective_sample_size(log_weights)) == pytest.approx(2.0)


def test_kish_size_of_one_dominant_weight_is_one() -> None:
    """A weight that dominates all others gives one effective sample."""
    log_weights = jnp.array([0.0, -200.0, -200.0, -200.0])
    assert float(kish_effective_sample_size(log_weights)) == pytest.approx(1.0)


def test_weighted_covariance_matches_numpy_with_aweights(
    rng: np.random.Generator,
) -> None:
    """The estimator is numpy.cov with aweights, posterior by posterior."""
    samples = rng.normal(size=(3, 200, 4)) @ rng.normal(size=(4, 4))
    log_weights = rng.normal(size=(3, 200))
    mean, covariance = weighted_covariance(
        jnp.asarray(samples), jnp.asarray(log_weights)
    )
    for posterior in range(3):
        weights = np.exp(log_weights[posterior])
        np.testing.assert_allclose(
            mean[posterior],
            np.average(samples[posterior], axis=0, weights=weights),
            rtol=1e-12,
        )
        np.testing.assert_allclose(
            covariance[posterior],
            np.cov(samples[posterior].T, aweights=weights),
            rtol=1e-12,
        )


def test_weighted_covariance_of_equal_weights_divides_by_n_minus_one(
    rng: np.random.Generator,
) -> None:
    """Equal weights give the usual unbiased sample covariance."""
    samples = rng.normal(size=(SAMPLE_COUNT, 3))
    _, covariance = weighted_covariance(jnp.asarray(samples), jnp.zeros(SAMPLE_COUNT))
    np.testing.assert_allclose(covariance, np.cov(samples.T, ddof=1), rtol=1e-12)


def test_weighted_covariance_ignores_zero_weights(rng: np.random.Generator) -> None:
    """Samples with log weight −∞ drop out; one positive weight gives NaN."""
    samples = rng.normal(size=(SAMPLE_COUNT, 2))
    log_weights = np.zeros(SAMPLE_COUNT)
    log_weights[::2] = -np.inf
    _, covariance = weighted_covariance(jnp.asarray(samples), jnp.asarray(log_weights))
    np.testing.assert_allclose(covariance, np.cov(samples[1::2].T), rtol=1e-12)
    log_weights[1:] = -np.inf
    _, single = weighted_covariance(jnp.asarray(samples), jnp.asarray(log_weights))
    assert bool(jnp.all(jnp.isnan(single)))


def test_circular_standard_deviation_matches_scipy(rng: np.random.Generator) -> None:
    """Equal weights reproduce scipy.stats.circstd on a circle of any period."""
    lower, period = -1.0, 3.0
    values = lower + rng.vonmises(0.4, 2.0, size=(2, 300)) * period / (2 * math.pi)
    computed = circular_standard_deviation(
        jnp.asarray(values), jnp.zeros(values.shape), period
    )
    expected = stats.circstd(values, high=lower + period, low=lower, axis=-1)
    np.testing.assert_allclose(computed, expected, rtol=1e-12)


def test_circular_standard_deviation_honours_weights(rng: np.random.Generator) -> None:
    """Integer weights act like repeated values."""
    values = rng.uniform(0.0, 1.0, size=12)
    repeats = rng.integers(1, 5, size=12)
    weighted = circular_standard_deviation(
        jnp.asarray(values), jnp.log(jnp.asarray(repeats, dtype=float)), 1.0
    )
    repeated = circular_standard_deviation(
        jnp.asarray(np.repeat(values, repeats)), jnp.zeros(int(repeats.sum())), 1.0
    )
    assert float(weighted) == pytest.approx(float(repeated), rel=1e-12)


def test_circular_standard_deviation_of_wrapped_normals_is_their_width() -> None:
    """A wrapped normal of width σ has circular standard deviation σ."""
    width, period = 0.4, math.pi
    draws = width * jax.random.normal(jax.random.key(3), (200_000,)) + 1.3
    computed = circular_standard_deviation(draws, jnp.zeros(draws.shape), period)
    assert float(computed) == pytest.approx(width, rel=0.01)


def test_coinciding_values_have_exactly_zero_spread(rng: np.random.Generator) -> None:
    """Shifting by a sample keeps rounding out of constant parameters."""
    values = (
        jnp.full((1_000, 2), 0.1234).at[:, 1].set(jnp.asarray(rng.normal(size=1_000)))
    )
    log_weights = jnp.asarray(rng.normal(size=1_000))
    _, covariance = weighted_covariance(values, log_weights)
    assert float(covariance[0, 0]) == 0.0
    assert float(covariance[0, 1]) == 0.0
    width = circular_standard_deviation(values[:, 0], log_weights, 2 * math.pi)
    assert float(width) == 0.0


def test_narrow_circles_keep_their_width(rng: np.random.Generator) -> None:
    """A width of 1e-7 rad, where 1 − R̄ is near ε, matches the linear width."""
    values = 1.0 + 1e-7 * rng.normal(size=1_000)
    width = circular_standard_deviation(
        jnp.asarray(values), jnp.zeros(1_000), 2 * math.pi
    )
    assert float(width) == pytest.approx(float(np.std(values)), rel=1e-6)


def test_gw_scale_means_stay_within_one_float64_step(
    rng: np.random.Generator,
) -> None:
    """A 3 mHz frequency known to 2e-11 Hz keeps its weighted mean to one step."""
    values = 3e-3 + 2e-11 * rng.normal(size=5_000)
    log_weights = rng.normal(size=5_000)
    mean, _ = weighted_covariance(
        jnp.asarray(values)[:, None], jnp.asarray(log_weights)
    )
    weights = np.exp(log_weights) / np.exp(log_weights).sum()
    # * Differences of values within a factor of two are exact in floating point.
    exact = 3e-3 + np.sum(weights * (values - 3e-3))
    assert abs(float(mean[0]) - exact) <= np.spacing(3e-3)


def test_effective_sample_size_prefers_a_given_n_eff() -> None:
    """A given n_eff replaces the Kish value of the weights."""
    draws = np.zeros((SAMPLE_COUNT, 1))
    assert float(SampleSet(draws).effective_sample_size()[0, 0]) == pytest.approx(
        SAMPLE_COUNT
    )
    with_n_eff = SampleSet(draws, n_eff=12.5)
    assert float(with_n_eff.effective_sample_size()[0, 0]) == 12.5


def test_normalized_weights_sum_to_one_per_posterior(
    rng: np.random.Generator,
) -> None:
    """Normalisation acts on each problem and posterior separately."""
    log_weights = rng.normal(scale=30.0, size=(2, 3, SAMPLE_COUNT))
    log_weights[0, 1, :5] = -np.inf
    sample_set = SampleSet(
        rng.normal(size=(2, 3, SAMPLE_COUNT, 2)), log_weights=log_weights
    )
    weights = sample_set.normalized_weights()
    np.testing.assert_allclose(weights.sum(axis=-1), 1.0, rtol=1e-12)
    np.testing.assert_array_equal(weights[0, 1, :5], 0.0)
    np.testing.assert_allclose(
        jnp.exp(sample_set.normalized_log_weights()), weights, rtol=1e-12
    )


# * Checks of shapes and types


@pytest.mark.parametrize("shape", [(SAMPLE_COUNT,), (1, 1, 1, SAMPLE_COUNT, 2)])
def test_samples_need_two_to_four_axes(shape: tuple[int, ...]) -> None:
    """Samples with too few or too many axes fail with the accepted layouts."""
    with pytest.raises(ValueError, match=r"\(N, D\), \(K, N, D\) or \(B, K, N, D\)"):
        SampleSet(np.zeros(shape))


@pytest.mark.parametrize("shape", [(0, 2), (SAMPLE_COUNT, 0), (2, 0, 3)])
def test_empty_axes_are_rejected(shape: tuple[int, ...]) -> None:
    """Every axis needs at least one entry."""
    with pytest.raises(ValueError, match="entries on every axis"):
        SampleSet(np.zeros(shape))


@pytest.mark.parametrize(
    ("field_name", "bad_value"),
    [
        ("log_weights", np.zeros(SAMPLE_COUNT + 1)),
        ("log_weights", np.zeros((1, SAMPLE_COUNT))),
        ("n_eff", np.ones(2)),
        ("log_like", np.zeros(SAMPLE_COUNT - 1)),
        ("log_prior", np.zeros((SAMPLE_COUNT, 2))),
        ("log_q", np.zeros(())),
    ],
)
def test_field_shapes_must_fit_the_samples(
    field_name: str, bad_value: np.ndarray
) -> None:
    """A field whose shape does not fit the samples fails with both shapes."""
    with pytest.raises(ValueError, match=f"{field_name} has shape"):
        SampleSet(np.zeros((SAMPLE_COUNT, 2)), **{field_name: bad_value})


@pytest.mark.parametrize(
    "complex_input",
    [
        {"samples": np.zeros((SAMPLE_COUNT, 2), dtype=complex)},
        {"log_weights": np.zeros(SAMPLE_COUNT, dtype=complex)},
        {"n_eff": np.ones((), dtype=complex)},
    ],
)
def test_complex_inputs_are_rejected(complex_input: dict[str, np.ndarray]) -> None:
    """Complex numbers are not real-valued samples, weights or sizes."""
    inputs = {"samples": np.zeros((SAMPLE_COUNT, 2))} | complex_input
    with pytest.raises(TypeError, match="real numbers"):
        SampleSet(**inputs)


# * Checks of values


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_non_finite_samples_fail_with_their_location(bad_value: float) -> None:
    """NaN or infinite samples fail, naming the problem and posterior."""
    draws = np.zeros((2, 3, SAMPLE_COUNT, 2))
    draws[1, 2, 7, 0] = bad_value
    with pytest.raises(ValueError, match="problem 1, posterior 2"):
        SampleSet(draws)


def test_the_first_failure_in_c_order_is_named() -> None:
    """With several invalid samples, the message names the first one."""
    draws = np.zeros((2, 3, SAMPLE_COUNT, 2))
    draws[1, 0, 0, 0] = np.nan
    draws[0, 2, 9, 1] = np.nan
    draws[0, 2, 3, 0] = np.inf
    with pytest.raises(ValueError, match="problem 0, posterior 2"):
        SampleSet(draws)


@pytest.mark.parametrize(
    "field_name", ["log_weights", "log_like", "log_prior", "log_q"]
)
@pytest.mark.parametrize("bad_value", [np.nan, np.inf])
def test_log_fields_reject_nan_and_positive_infinity(
    field_name: str, bad_value: float
) -> None:
    """Log weights and log densities may not hold NaN or +inf."""
    values = np.zeros((2, SAMPLE_COUNT))
    values[1, 3] = bad_value
    with pytest.raises(ValueError, match=f"{field_name} of problem 0, posterior 1"):
        SampleSet(np.zeros((2, SAMPLE_COUNT, 2)), **{field_name: values})


def test_log_densities_may_be_negative_infinity() -> None:
    """A log density of -inf is a valid value, for example for padding."""
    log_like = np.zeros(SAMPLE_COUNT)
    log_like[0] = -np.inf
    sample_set = SampleSet(np.zeros((SAMPLE_COUNT, 2)), log_like=log_like)
    assert sample_set.log_like is not None
    assert float(sample_set.log_like[0, 0, 0]) == -math.inf


def test_a_posterior_without_positive_weight_is_rejected() -> None:
    """Each posterior needs at least one sample with positive weight."""
    log_weights = np.zeros((2, SAMPLE_COUNT))
    log_weights[1] = -np.inf
    with pytest.raises(ValueError, match="posterior 1 of problem 0 has no sample"):
        SampleSet(np.zeros((2, SAMPLE_COUNT, 2)), log_weights=log_weights)


@pytest.mark.parametrize(
    "n_eff", [0.5, SAMPLE_COUNT - 9.5, math.nan, math.inf, -1.0, 0.0]
)
def test_n_eff_must_lie_between_one_and_the_positive_count(n_eff: float) -> None:
    """n_eff lies in [1, number of positive weights], here 30 of 40 samples."""
    log_weights = np.zeros(SAMPLE_COUNT)
    log_weights[:10] = -np.inf
    with pytest.raises(ValueError, match="between 1 and 30"):
        SampleSet(np.zeros((SAMPLE_COUNT, 2)), log_weights=log_weights, n_eff=n_eff)


def test_n_eff_at_its_bounds_is_accepted(rng: np.random.Generator) -> None:
    """The bounds themselves and a rounded Kish value pass the range check."""
    draws = np.zeros((SAMPLE_COUNT, 1))
    for n_eff in (
        1.0,
        float(SAMPLE_COUNT),
        SAMPLE_COUNT * (1 + N_EFF_RELATIVE_TOLERANCE / 2),
    ):
        SampleSet(draws, n_eff=n_eff)
    log_weights = rng.normal(size=SAMPLE_COUNT)
    kish_value = kish_effective_sample_size(jnp.asarray(log_weights))
    SampleSet(draws, log_weights=log_weights, n_eff=kish_value)


def test_values_are_not_checked_inside_jit() -> None:
    """Under jit the values are unknown, so only shapes and types are checked."""

    @jax.jit
    def build(draws: jax.Array) -> SampleSet:
        return SampleSet(draws, n_eff=jnp.zeros(()))

    built = build(jnp.full((SAMPLE_COUNT, 2), jnp.nan))
    assert built.shape == (1, 1, SAMPLE_COUNT, 2)

    @jax.jit
    def build_misshaped(draws: jax.Array) -> SampleSet:
        return SampleSet(draws, log_weights=jnp.zeros(3))

    with pytest.raises(ValueError, match="log_weights has shape"):
        build_misshaped(jnp.zeros((SAMPLE_COUNT, 2)))


# * Pytree behaviour


def test_sets_pass_through_jit_unchanged() -> None:
    """A set is a pytree: jit takes it as input and returns it intact."""
    sample_set = labelled_set()
    returned = jax.jit(lambda passed: passed)(sample_set)
    assert eqx.tree_equal(returned, sample_set)
    sizes = jax.jit(lambda passed: passed.effective_sample_size())(sample_set)
    np.testing.assert_array_equal(sizes, sample_set.n_eff)


def test_sets_are_immutable() -> None:
    """Fields cannot be reassigned after construction."""
    sample_set = labelled_set()
    with pytest.raises(dataclasses.FrozenInstanceError):
        sample_set.n_eff = None


def test_dataclass_replace_checks_the_new_fields() -> None:
    """dataclasses.replace builds a new set through the same checks."""
    sample_set = labelled_set()
    without_q = dataclasses.replace(sample_set, log_q=None)
    assert without_q.log_q is None
    assert eqx.tree_equal(without_q.log_weights, sample_set.log_weights)
    with pytest.raises(ValueError, match="n_eff"):
        dataclasses.replace(sample_set, n_eff=jnp.full((2, 3), 0.1))


# * Operations


def test_select_keeps_every_field_aligned() -> None:
    """Selecting problems and posteriors moves every field together."""
    sample_set = labelled_set()
    selected = sample_set.select(problems=[1], posteriors=[2, 0])
    assert selected.shape == (1, 2, SAMPLE_COUNT, 3)
    assert_fields_follow_samples(selected)
    np.testing.assert_array_equal(
        selected.samples, sample_set.samples[jnp.array([1])][:, jnp.array([2, 0])]
    )
    assert sample_set.n_eff is not None
    np.testing.assert_array_equal(selected.n_eff, sample_set.n_eff[1:2, [2, 0]])


def test_single_and_negative_indices_keep_their_axis() -> None:
    """One index keeps its axis with size one; negative indices count back."""
    sample_set = labelled_set()
    last = sample_set.select(problems=-1, posteriors=np.int64(1))
    assert last.shape == (1, 1, SAMPLE_COUNT, 3)
    np.testing.assert_array_equal(last.samples[0, 0], sample_set.samples[1, 1])
    assert_fields_follow_samples(last)


def test_select_params_keeps_the_log_densities() -> None:
    """Parameter subsets keep the per-sample fields of the full posterior."""
    sample_set = labelled_set()
    reordered = sample_set.select(params=jnp.array([2, 0]))
    np.testing.assert_array_equal(reordered.samples, sample_set.samples[..., [2, 0]])
    for name in ("log_weights", "log_like", "log_prior", "log_q", "n_eff"):
        assert eqx.tree_equal(getattr(reordered, name), getattr(sample_set, name))


@pytest.mark.parametrize(
    ("selection", "error", "pattern"),
    [
        ({"problems": [2]}, IndexError, "problem index 2 is out of range"),
        ({"posteriors": [-4]}, IndexError, "posterior index -4 is out of range"),
        ({"params": [0, 3]}, IndexError, "parameter index 3 is out of range"),
        ({"params": []}, ValueError, "select at least one parameter"),
        ({"posteriors": np.zeros((1, 1), dtype=int)}, ValueError, "one-dimensional"),
        ({"problems": np.array([0.0])}, TypeError, "must be integers"),
        ({"problems": True}, TypeError, "must be integers, got a bool"),
    ],
)
def test_select_rejects_invalid_indices(
    selection: dict[str, object], error: type[Exception], pattern: str
) -> None:
    """Indices outside an axis, empty or non-integer selections fail clearly."""
    with pytest.raises(error, match=pattern):
        labelled_set().select(**selection)  # type: ignore[arg-type]


def test_select_inside_jit_accepts_traced_indices() -> None:
    """Traced indices skip the range check and still select."""
    sample_set = labelled_set()
    selected = jax.jit(lambda passed, index: passed.select(posteriors=index))(
        sample_set, jnp.array([-1])
    )
    np.testing.assert_array_equal(selected.samples, sample_set.samples[:, 2:3])


def test_take_samples_keeps_fields_aligned_and_drops_n_eff() -> None:
    """Shared positions keep every per-sample field aligned and reset n_eff."""
    sample_set = labelled_set()
    taken = sample_set.take_samples([5, 0, -1, 5])
    assert taken.shape == (2, 3, 4, 3)
    assert taken.n_eff is None
    assert_fields_follow_samples(taken)
    np.testing.assert_array_equal(
        taken.samples, sample_set.samples[:, :, [5, 0, SAMPLE_COUNT - 1, 5]]
    )


def test_take_samples_accepts_one_row_per_posterior() -> None:
    """Indices shaped (B, K, M) select different samples for each posterior."""
    sample_set = labelled_set()
    positions = np.arange(2 * 3 * 2).reshape(2, 3, 2)
    taken = sample_set.take_samples(positions)
    assert_fields_follow_samples(taken)
    for problem in range(2):
        for posterior in range(3):
            np.testing.assert_array_equal(
                taken.samples[problem, posterior],
                sample_set.samples[problem, posterior, positions[problem, posterior]],
            )


@pytest.mark.parametrize(
    ("indices", "error", "pattern"),
    [
        ([SAMPLE_COUNT], IndexError, f"sample index {SAMPLE_COUNT} is out of range"),
        ([], ValueError, "select at least one sample"),
        (np.zeros((3, 3, 2), dtype=int), ValueError, r"leading axes \(2, 3\)"),
        (np.zeros((2, 2), dtype=int), ValueError, r"\(M,\) or \(B, K, M\)"),
        (np.array([0.5]), TypeError, "must be integers"),
    ],
)
def test_take_samples_rejects_invalid_indices(
    indices: object, error: type[Exception], pattern: str
) -> None:
    """Positions outside the sample axis or of the wrong shape fail clearly."""
    with pytest.raises(error, match=pattern):
        labelled_set().take_samples(indices)  # type: ignore[arg-type]


def test_take_samples_needs_a_positive_weight_per_posterior() -> None:
    """Keeping only zero-weight samples leaves a posterior without weight."""
    log_weights = np.zeros(SAMPLE_COUNT)
    log_weights[:3] = -np.inf
    sample_set = SampleSet(np.zeros((SAMPLE_COUNT, 2)), log_weights=log_weights)
    with pytest.raises(ValueError, match="no sample with positive weight"):
        sample_set.take_samples([0, 1, 2])


def test_take_samples_inside_jit_accepts_traced_indices() -> None:
    """Traced positions skip the range check and still select samples."""
    sample_set = labelled_set()
    taken = jax.jit(lambda passed, index: passed.take_samples(index))(
        sample_set, jnp.array([3, -2])
    )
    np.testing.assert_array_equal(
        taken.samples, sample_set.samples[:, :, [3, SAMPLE_COUNT - 2]]
    )


def test_weights_survive_every_operation() -> None:
    """Each operation, conversion and transfer keeps the weights with the samples."""
    sample_set = labelled_set()
    operations: dict[str, Callable[[SampleSet], SampleSet]] = {
        "select": lambda passed: passed.select(problems=[1, 0], posteriors=[1]),
        "take": lambda passed: passed.take_samples([9, 3, 3]),
        "jit": jax.jit(lambda passed: passed),
        "device_put": lambda passed: jax.device_put(passed, jax.devices("cpu")[0]),
        "numpy round trip": lambda passed: SampleSet(**passed.to_arrays()),
        "replace": lambda passed: dataclasses.replace(passed, n_eff=None),
    }
    for name, operation in operations.items():
        result = operation(sample_set)
        assert isinstance(result, SampleSet), name
        assert_fields_follow_samples(result)


# * Conversions


def test_to_arrays_round_trips_through_numpy() -> None:
    """The arrays rebuild an equal set, and unset fields are left out."""
    sample_set = SampleSet(np.ones((SAMPLE_COUNT, 2)), log_like=np.zeros(SAMPLE_COUNT))
    arrays = sample_set.to_arrays()
    assert list(arrays) == ["samples", "log_weights", "log_like"]
    assert all(isinstance(array, np.ndarray) for array in arrays.values())
    assert eqx.tree_equal(SampleSet(**arrays), sample_set)


def test_to_arrays_copies_on_request() -> None:
    """copy=True gives writable arrays that are independent of the set."""
    sample_set = labelled_set()
    arrays = sample_set.to_arrays(copy=True)
    arrays["samples"][0, 0, 0, 0] = -1.0
    assert float(sample_set.samples[0, 0, 0, 0]) == 0.0
    assert not sample_set.to_arrays()["samples"].flags.writeable


def test_to_arrays_round_trips_through_torch() -> None:
    """PyTorch tensors rebuild an equal set, without any precision warning."""
    torch = pytest.importorskip("torch")
    sample_set = labelled_set()
    tensors = sample_set.to_arrays("torch")
    assert all(isinstance(tensor, torch.Tensor) for tensor in tensors.values())
    assert eqx.tree_equal(SampleSet(**tensors), sample_set)


def test_float32_tensors_warn_and_are_cast() -> None:
    """float32 tensors, as from many SBI codes, are cast with a warning."""
    torch = pytest.importorskip("torch")
    draws = torch.randn(SAMPLE_COUNT, 2, generator=torch.Generator().manual_seed(3))
    with pytest.warns(PrecisionWarning, match=r"samples \(float32\)"):
        sample_set = SampleSet(draws)
    np.testing.assert_array_equal(sample_set.samples[0, 0], draws.double().numpy())
