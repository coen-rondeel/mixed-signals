"""Tests for ``SampleSet``: layouts, weights, checks, operations and conversions."""

import dataclasses
import math
from collections.abc import Callable
from functools import partial
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import stats

from mixed_signals import zoo
from mixed_signals.core.samples import (
    N_EFF_RELATIVE_TOLERANCE,
    SampleSet,
    SampleShape,
    _is_concrete,
    circular_mean,
    circular_standard_deviation,
    kish_effective_sample_size,
    mean_resultant_length,
    rejection_subset,
    spherical_width,
    unwrap_about,
    weighted_covariance,
)
from mixed_signals.core.topology import PolarForm, sphere_coordinates
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


def test_circular_mean_matches_scipy(rng: np.random.Generator) -> None:
    """Equal weights reproduce scipy.stats.circmean on a circle of any period."""
    lower, period = -1.0, 3.0
    values = lower + rng.vonmises(0.4, 2.0, size=(2, 300)) * period / (2 * math.pi)
    computed = circular_mean(
        jnp.asarray(values), jnp.zeros(values.shape), period, lower
    )
    expected = stats.circmean(values, high=lower + period, low=lower, axis=-1)
    np.testing.assert_allclose(computed, expected, rtol=1e-12)
    assert bool(jnp.all((computed >= lower) & (computed < lower + period)))


def test_circular_mean_and_resultant_honour_weights(rng: np.random.Generator) -> None:
    """Integer weights act like repeated values."""
    values = rng.uniform(0.0, 1.0, size=12)
    repeats = rng.integers(1, 5, size=12)
    log_weights = jnp.log(jnp.asarray(repeats, dtype=float))
    repeated = jnp.asarray(np.repeat(values, repeats))
    flat = jnp.zeros(int(repeats.sum()))
    assert float(circular_mean(jnp.asarray(values), log_weights, 1.0, 0.0)) == (
        pytest.approx(float(circular_mean(repeated, flat, 1.0, 0.0)), rel=1e-12)
    )
    assert float(mean_resultant_length(jnp.asarray(values), log_weights, 1.0)) == (
        pytest.approx(float(mean_resultant_length(repeated, flat, 1.0)), rel=1e-12)
    )


def test_mean_resultant_length_matches_the_mean_unit_vector(
    rng: np.random.Generator,
) -> None:
    """R̄ is the length of the mean of the unit vectors."""
    values = rng.vonmises(1.0, 0.7, size=500)
    expected = abs(np.mean(np.exp(1j * values)))
    computed = mean_resultant_length(jnp.asarray(values), jnp.zeros(500), 2 * math.pi)
    assert float(computed) == pytest.approx(float(expected), rel=1e-12)


def test_opposite_clusters_have_no_mean_resultant() -> None:
    """Two equal opposite clusters give R̄ = 0, where the mean direction is undefined."""
    values = jnp.array([0.5, 0.5 + math.pi])
    resultant = mean_resultant_length(values, jnp.zeros(2), 2 * math.pi)
    assert float(resultant) == pytest.approx(0.0, abs=1e-8)


def test_narrow_circles_keep_their_mean(rng: np.random.Generator) -> None:
    """A circle 1e-9 rad wide next to the boundary keeps its mean to rounding."""
    values = 2 * math.pi - 2e-9 + 1e-9 * rng.normal(size=1_000)
    computed = circular_mean(jnp.asarray(values), jnp.zeros(1_000), 2 * math.pi, 0.0)
    expected = float(np.mean(values))
    assert abs(float(computed) - expected) < 1e-15


def test_unwrap_about_gives_representatives_within_half_a_period() -> None:
    """Every value maps to the point of the same angle nearest the centre."""
    values = jnp.array([0.1, 3.0, 6.2, -6.2, 12.0])
    centre = 0.05
    unwrapped = unwrap_about(values, centre, 2 * math.pi)
    assert bool(jnp.all(jnp.abs(unwrapped - centre) <= math.pi))
    np.testing.assert_allclose(
        np.cos(np.asarray(unwrapped)), np.cos(np.asarray(values)), atol=1e-12
    )
    np.testing.assert_allclose(
        np.sin(np.asarray(unwrapped)), np.sin(np.asarray(values)), atol=1e-12
    )


@pytest.mark.parametrize("polar_form", list(PolarForm))
def test_spherical_width_matches_scipy_directional_stats(
    rng: np.random.Generator, polar_form: PolarForm
) -> None:
    """Equal weights give √(1 − R̄) with SciPy's mean resultant length, in any form."""
    direction = np.array([0.3, -0.2, 0.9]) / np.linalg.norm([0.3, -0.2, 0.9])
    vectors = stats.vonmises_fisher(direction, 12.0).rvs(500, random_state=rng)
    resultant = stats.directional_stats(vectors).mean_resultant_length
    azimuth, polar = sphere_coordinates(jnp.asarray(vectors), polar_form)
    width = spherical_width(azimuth, polar, jnp.zeros(500), polar_form)
    assert float(width) == pytest.approx(math.sqrt(1 - resultant), rel=1e-10)


def test_spherical_width_honours_weights(rng: np.random.Generator) -> None:
    """Integer weights act like repeated points, and zero weights remove them."""
    azimuth = rng.uniform(0.0, 1.0, size=12)
    latitude = rng.uniform(-0.5, 0.5, size=12)
    repeats = rng.integers(0, 4, size=12)
    with np.errstate(divide="ignore"):
        log_weights = np.log(repeats.astype(float))
    weighted = spherical_width(
        jnp.asarray(azimuth),
        jnp.asarray(latitude),
        jnp.asarray(log_weights),
        PolarForm.LATITUDE,
    )
    repeated = spherical_width(
        jnp.asarray(np.repeat(azimuth, repeats)),
        jnp.asarray(np.repeat(latitude, repeats)),
        jnp.zeros(int(repeats.sum())),
        PolarForm.LATITUDE,
    )
    assert float(weighted) == pytest.approx(float(repeated), rel=1e-12)


def test_spherical_width_of_von_mises_fisher_is_its_width() -> None:
    """A concentrated von Mises-Fisher distribution has the width 1/√κ."""
    for width in (0.02, 0.1, 0.5):
        case = zoo.von_mises_fisher(1.0, width=width)
        draws = case.sample(jax.random.key(4), 50_000).samples[0, 0]
        computed = spherical_width(
            draws[:, 0],
            draws[:, 1],
            jnp.zeros(50_000),
            case.space.spheres[0].polar_form,
        )
        # * The relative standard error of the width is 1/√(2N), 0.3% here.
        assert float(computed) == pytest.approx(width, rel=0.015)


def test_coinciding_points_have_exactly_zero_width() -> None:
    """Points that coincide have no spread, and an even spread has width one."""
    same = spherical_width(
        jnp.full(30, 2.0), jnp.full(30, -0.4), jnp.zeros(30), PolarForm.LATITUDE
    )
    assert float(same) == 0.0
    opposite = spherical_width(
        jnp.array([0.0, math.pi]),
        jnp.array([0.0, 0.0]),
        jnp.zeros(2),
        PolarForm.LATITUDE,
    )
    assert float(opposite) == pytest.approx(1.0, abs=1e-15)


def test_narrow_spheres_keep_their_width(rng: np.random.Generator) -> None:
    """A sky position known to 1e-7 rad keeps its width to a few parts in 1e6."""
    latitude = 0.3 + 1e-7 * rng.normal(size=2_000)
    azimuth = 1.0 + 1e-7 * rng.normal(size=2_000) / math.cos(0.3)
    computed = spherical_width(
        jnp.asarray(azimuth),
        jnp.asarray(latitude),
        jnp.zeros(2_000),
        PolarForm.LATITUDE,
    )
    east = (azimuth - azimuth.mean()) * math.cos(0.3)
    north = latitude - latitude.mean()
    # * For small spreads 1 − R̄ is half the mean squared tangent offset.
    expected = math.sqrt(0.5 * np.mean(east**2 + north**2))
    assert float(computed) == pytest.approx(expected, rel=1e-5)


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


# * The probe that tells whether a check can run eagerly


def test_is_concrete_follows_the_staging_of_the_trace() -> None:
    """True where jnp operations run eagerly, false inside jit, map and scan.

    The probe is a fresh ``jnp.zeros(())``. A JAX release that changes where it
    is staged must fail this test, since every concrete check relies on it.
    """
    closed_over = jnp.ones(3)
    seen: dict[str, bool] = {}

    def record(name: str, *arrays: jax.Array) -> None:
        seen[name] = _is_concrete(*arrays)

    record("plain", closed_over)
    assert seen["plain"]
    assert _is_concrete()

    def under_vmap(row: jax.Array) -> jax.Array:
        record("vmap", closed_over)
        record("vmap mapped", row)
        return row

    jax.vmap(under_vmap)(jnp.ones((2, 3)))
    assert seen["vmap"]
    assert not seen["vmap mapped"]

    def under_grad(value: jax.Array) -> jax.Array:
        record("grad", closed_over)
        return value

    jax.grad(under_grad)(1.0)
    assert seen["grad"]

    def under_jit(value: jax.Array) -> jax.Array:
        record("jit", closed_over)
        record("jit argument", value)
        with jax.ensure_compile_time_eval():
            record("jit eager", closed_over)
        return value

    jax.jit(under_jit)(jnp.ones(3))
    assert not seen["jit"]
    assert not seen["jit argument"]
    assert seen["jit eager"]

    def under_map(value: jax.Array) -> jax.Array:
        record("map", closed_over)
        return value

    jax.lax.map(under_map, jnp.ones(2))
    assert not seen["map"]

    def under_scan(carry: jax.Array, value: jax.Array) -> tuple[jax.Array, jax.Array]:
        record("scan", closed_over)
        return carry, value

    jax.lax.scan(under_scan, jnp.zeros(()), jnp.ones(2))
    assert not seen["scan"]


# * Rejection subsets of weighted samples

PROPOSAL_WIDTH = 1.5
"""Width of the Gaussian proposal of the rejection tests, against a unit target."""


def weighted_proposal(count: int, param_count: int, seed: int) -> tuple[Any, Any]:
    """Draw a wide Gaussian proposal with the log weights of a unit Gaussian target."""
    proposal = stats.norm(scale=PROPOSAL_WIDTH)
    target = stats.norm()
    draws = proposal.rvs(
        size=(count, param_count), random_state=np.random.default_rng(seed)
    )
    log_weights = (target.logpdf(draws) - proposal.logpdf(draws)).sum(axis=1)
    return jnp.asarray(draws), jnp.asarray(log_weights)


def test_rejection_keeps_every_positive_weight_sample_of_equal_weights() -> None:
    """Equal positive weights, at any scale, keep every sample of positive weight."""
    samples = jnp.arange(24.0).reshape(12, 2)
    key = jax.random.key(0)
    for level in (0.0, 3.5, -40.0):
        kept = rejection_subset(samples, jnp.full(12, level), key=key)
        np.testing.assert_array_equal(kept, samples)
    log_weights = jnp.where(jnp.arange(12) % 3 == 0, -jnp.inf, 1.0)
    kept = rejection_subset(samples, log_weights, key=key)
    np.testing.assert_array_equal(kept, samples[jnp.arange(12) % 3 != 0])


def test_rejection_never_keeps_a_zero_weight_and_keeps_the_input_order() -> None:
    """A weight of zero never stays; the kept rows keep their order and values."""
    count = 400
    index = jnp.arange(float(count))
    samples = jnp.stack([index, -index], axis=1)
    log_weights = jnp.where(jnp.arange(count) % 2 == 0, -jnp.inf, -0.2 * index / 100)
    for seed in range(5):
        kept = rejection_subset(samples, log_weights, key=jax.random.key(seed))
        positions = np.asarray(kept[:, 0]).astype(int)
        assert kept.shape[1] == 2
        assert np.all(positions % 2 == 1)
        assert np.all(np.diff(positions) > 0)
        np.testing.assert_array_equal(kept, samples[positions])
        assert positions.size > 0


def test_rejection_always_keeps_the_largest_weight() -> None:
    """The sample of the largest weight stays under every key."""
    samples = jnp.arange(30.0).reshape(30, 1)
    log_weights = jnp.where(jnp.arange(30) == 17, 0.0, -6.0)
    for seed in range(20):
        kept = rejection_subset(samples, log_weights, key=jax.random.key(seed))
        assert 17.0 in np.asarray(kept[:, 0])


def test_rejection_repeats_for_a_fixed_key_and_changes_with_the_key() -> None:
    """The same key keeps the same samples; another key keeps others."""
    samples, log_weights = weighted_proposal(300, 2, seed=1)
    first = rejection_subset(samples, log_weights, key=jax.random.key(3))
    again = rejection_subset(samples, log_weights, key=jax.random.key(3))
    other = rejection_subset(samples, log_weights, key=jax.random.key(4))
    np.testing.assert_array_equal(first, again)
    assert first.shape != other.shape or not np.array_equal(first, other)


def test_rejection_returns_float64_rows_of_the_input() -> None:
    """The output is a float64 JAX array of selected rows; integers are cast."""
    samples = jnp.arange(20).reshape(10, 2)
    kept = rejection_subset(samples, jnp.zeros(10), key=jax.random.key(0))
    assert isinstance(kept, jax.Array)
    assert kept.dtype == jnp.float64
    np.testing.assert_array_equal(kept, samples)


def test_the_kept_count_matches_the_sum_of_weights_over_the_largest_weight() -> None:
    """Over many keys the kept count averages to Σw/max w, within binomial error."""
    samples, log_weights = weighted_proposal(500, 2, seed=2)
    probabilities = np.exp(np.asarray(log_weights - jnp.max(log_weights)))
    expected = probabilities.sum()
    variance = (probabilities * (1 - probabilities)).sum()
    key_count = 60
    counts = [
        rejection_subset(samples, log_weights, key=jax.random.key(seed)).shape[0]
        for seed in range(key_count)
    ]
    assert abs(np.mean(counts) - expected) < 5 * math.sqrt(variance / key_count)
    kish = float(kish_effective_sample_size(log_weights))
    assert expected <= kish


def test_the_kept_samples_follow_the_target_of_a_wider_proposal() -> None:
    """A proposal 1.5 times wider gives a unit Gaussian: mean 0 and variance 1."""
    param_count = 2
    samples, log_weights = weighted_proposal(4_000, param_count, seed=3)
    kept = rejection_subset(samples, log_weights, key=jax.random.key(5))
    kept_count = kept.shape[0]
    assert 0.3 * 4_000 < kept_count < 0.8 * 4_000
    mean_error = 1 / math.sqrt(kept_count)
    variance_error = math.sqrt(2 / kept_count)
    np.testing.assert_allclose(kept.mean(axis=0), 0.0, atol=5 * mean_error)
    np.testing.assert_allclose(kept.var(axis=0), 1.0, atol=5 * variance_error)
    # * The proposal itself is wider, so the test can tell a wrong subset apart.
    assert float(jnp.var(samples)) > 1 + 5 * variance_error


def test_rejection_makes_no_copies() -> None:
    """Every kept row is a distinct input row, unlike resampling with replacement."""
    samples, log_weights = weighted_proposal(1_000, 2, seed=4)
    kept = rejection_subset(samples, log_weights, key=jax.random.key(6))
    assert np.unique(np.asarray(kept), axis=0).shape[0] == kept.shape[0]


def test_rejection_needs_concrete_inputs() -> None:
    """Traced samples, weights or key raise, since the output size depends on values."""
    samples = jnp.arange(20.0).reshape(10, 2)
    log_weights = jnp.zeros(10)
    key = jax.random.key(0)
    with pytest.raises(ValueError, match=r"call rejection_subset outside jax\.jit"):
        jax.jit(partial(rejection_subset, key=key))(samples, log_weights)
    with pytest.raises(ValueError, match=r"call rejection_subset outside jax\.jit"):
        jax.jit(partial(rejection_subset, samples, key=key))(log_weights)
    keys = jax.random.split(key, 3)
    with pytest.raises(ValueError, match=r"call rejection_subset outside jax\.jit"):
        jax.vmap(partial(rejection_subset, samples, log_weights))(key=keys)


@pytest.mark.parametrize(
    ("samples", "log_weights", "message"),
    [
        (jnp.full((4, 2), jnp.nan), jnp.zeros(4), "samples must be finite"),
        (jnp.full((4, 2), jnp.inf), jnp.zeros(4), "samples must be finite"),
        (jnp.zeros((4, 2)), jnp.array([0.0, jnp.nan, 0.0, 0.0]), "NaN or"),
        (jnp.zeros((4, 2)), jnp.array([0.0, jnp.inf, 0.0, 0.0]), "NaN or \\+inf"),
        (jnp.zeros((4, 2)), jnp.full(4, -jnp.inf), "at least one sample with a pos"),
        (jnp.zeros(4), jnp.zeros(4), r"shaped \(N, D\).*shape \(4,\)"),
        (jnp.zeros((0, 2)), jnp.zeros(0), r"shaped \(N, D\).*shape \(0, 2\)"),
        (jnp.zeros((4, 2, 1)), jnp.zeros(4), r"shaped \(N, D\).*\(4, 2, 1\)"),
        (jnp.zeros((4, 2)), jnp.zeros(3), r"log_weights must be shaped \(4,\)"),
        (jnp.zeros((4, 2)), jnp.zeros((4, 1)), r"got shape \(4, 1\)"),
    ],
)
def test_rejection_rejects_invalid_values_and_shapes(
    samples: Any, log_weights: Any, message: str
) -> None:
    """Non-finite samples, invalid weights and misfit shapes each raise ValueError."""
    with pytest.raises(ValueError, match=message):
        rejection_subset(samples, log_weights, key=jax.random.key(0))


def test_rejection_rejects_values_that_are_not_real() -> None:
    """Complex samples or weights raise TypeError with the argument's name."""
    key = jax.random.key(0)
    with pytest.raises(TypeError, match="samples must hold real numbers"):
        rejection_subset(jnp.zeros((4, 2), dtype=complex), jnp.zeros(4), key=key)
    with pytest.raises(TypeError, match="log_weights must hold real numbers"):
        rejection_subset(jnp.zeros((4, 2)), jnp.zeros(4, dtype=complex), key=key)


def test_rejection_warns_about_float32_input() -> None:
    """Float32 samples or weights are cast and raise a PrecisionWarning."""
    samples = jnp.ones((6, 2), dtype=jnp.float32)
    log_weights = jnp.zeros(6, dtype=jnp.float32)
    with pytest.warns(PrecisionWarning, match=r"samples \(float32\)") as records:
        kept = rejection_subset(samples, log_weights, key=jax.random.key(0))
    assert kept.dtype == jnp.float64
    assert "log_weights (float32)" in str(records[0].message)
