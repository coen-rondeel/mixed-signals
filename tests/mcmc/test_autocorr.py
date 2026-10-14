"""Tests for the autocorrelation function and the integrated autocorrelation time."""

import math
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mixed_signals import zoo
from mixed_signals.errors import PrecisionWarning, ValidityWarning
from mixed_signals.mcmc import (
    MINIMUM_LENGTH_IN_TAU,
    autocorrelation_function,
    integrated_autocorrelation_time,
)

STANDARD_ERRORS = 5.0
STEP_COUNT = 20_000
WALKER_COUNT = 64
# * Exact τ = (1 + φ)/(1 − φ): 19, 3, 1, 0.538, 0.333 and 0.053, below the floor.
COEFFICIENTS = (0.9, 0.5, 0.0, -0.3, -0.5, -0.9)


def cold_samples(
    coefficients: tuple[float, ...],
    seed: int,
    step_count: int,
    walker_count: int,
    start: float | None = None,
) -> jax.Array:
    """Return zoo AR(1) chains shaped (step, walker, param)."""
    chains = zoo.ar1_chains(coefficients).sample(
        jax.random.key(seed), step_count, walker_count=walker_count, start=start
    )
    return chains.samples[:, 0, :, 0, :]


def tau_error(tau: float, draw_count: int) -> float:
    """Return a conservative standard error of τ̂ from S·W draws.

    The a priori error of a sum over the lags up to M is τ √(2(2M + 1)/(S W))
    (Madras and Sokal 1988; Wolff 2004); M = 5τ + 10 bounds the window used.
    """
    window = 5 * tau + 10
    return tau * math.sqrt(2 * (2 * window + 1) / draw_count)


@pytest.fixture(scope="module")
def ar1_samples() -> jax.Array:
    """Stationary AR(1) chains of every coefficient, 20 000 steps of 64 walkers."""
    return cold_samples(COEFFICIENTS, 0, STEP_COUNT, WALKER_COUNT)


@pytest.fixture(scope="module")
def ar1_tau(ar1_samples: jax.Array) -> np.ndarray:
    """τ of the stationary AR(1) chains."""
    return np.asarray(integrated_autocorrelation_time(ar1_samples))


# * Values against closed forms


@pytest.mark.parametrize("index", range(len(COEFFICIENTS) - 1))
def test_tau_matches_the_ar1_closed_form(ar1_tau: np.ndarray, index: int) -> None:
    """τ recovers (1 + φ)/(1 − φ), also below 1 for anticorrelated chains."""
    case = zoo.ar1_chains(COEFFICIENTS)
    exact = case.reference_value("tau", f"x{index}")
    error = tau_error(exact, STEP_COUNT * WALKER_COUNT)
    assert abs(ar1_tau[index] - exact) < STANDARD_ERRORS * error


def test_tau_stops_at_its_floor_for_strongly_antithetic_chains(
    ar1_tau: np.ndarray,
) -> None:
    """τ = 0.053 lies below 1/log₁₀ S, so the estimate is the floor itself."""
    floor = 1 / math.log10(STEP_COUNT * WALKER_COUNT)
    assert ar1_tau[-1] == pytest.approx(floor, rel=1e-12)


def test_autocorrelation_function_decays_as_phi_to_the_lag(
    ar1_samples: jax.Array,
) -> None:
    """ρ(t) = φ^t at the first lags, within its sampling error."""
    function = np.asarray(autocorrelation_function(ar1_samples))
    assert function.shape == (STEP_COUNT, len(COEFFICIENTS))
    np.testing.assert_array_equal(function[0], np.ones(len(COEFFICIENTS)))
    lags = np.arange(1, 6)
    # * The error of ρ̂(t) is about √(τ/(S W)), with τ of the chain.
    for index, coefficient in enumerate(COEFFICIENTS[:3]):
        tau = (1 + coefficient) / (1 - coefficient)
        error = math.sqrt(tau / (STEP_COUNT * WALKER_COUNT))
        assert np.all(
            np.abs(function[lags, index] - coefficient**lags) < STANDARD_ERRORS * error
        )


# * Reference estimators


def test_function_equals_the_walker_average_of_emcee_and_eryn() -> None:
    """The function is emcee's, and eryn's, per-walker ACF averaged over walkers."""
    emcee = pytest.importorskip("emcee")
    eryn_utility = pytest.importorskip("eryn.utils.utility")
    samples = cold_samples((0.8, -0.4), 1, 3_000, 8)
    host = np.asarray(samples)
    function = np.asarray(autocorrelation_function(samples))
    for index in range(2):
        emcee_function = np.mean(
            [emcee.autocorr.function_1d(host[:, walker, index]) for walker in range(8)],
            axis=0,
        )
        np.testing.assert_allclose(function[:, index], emcee_function, atol=1e-12)
    eryn_function = eryn_utility.get_acf(host, axis=0).mean(axis=1)
    np.testing.assert_allclose(function, eryn_function, atol=1e-12)


def test_eryn_fixed_window_is_a_partial_sum_of_the_function() -> None:
    """The τ of eryn is 1 + 2 Σ ρ(t) for t < 50 and agrees with ours for small τ."""
    eryn_utility = pytest.importorskip("eryn.utils.utility")
    samples = cold_samples((0.5,), 2, STEP_COUNT, 16)
    host = np.asarray(samples)
    function = np.asarray(autocorrelation_function(samples))
    eryn_tau = eryn_utility.get_integrated_act(host[:, None], window=50)[0]
    np.testing.assert_allclose(eryn_tau, 1 + 2 * function[1:50].sum(axis=0), rtol=1e-12)
    tau = float(integrated_autocorrelation_time(samples)[0])
    error = tau_error(3.0, STEP_COUNT * 16)
    assert abs(tau - float(eryn_tau[0])) < STANDARD_ERRORS * error


@pytest.mark.parametrize("coefficient", [0.5, 0.9])
def test_tau_agrees_with_emcee_for_positive_correlation(coefficient: float) -> None:
    """Where Sokal's window works, emcee's τ and ours agree within their error."""
    emcee = pytest.importorskip("emcee")
    samples = cold_samples((coefficient,), 3, STEP_COUNT, 16)
    emcee_tau = float(emcee.autocorr.integrated_time(np.asarray(samples))[0])
    tau = float(integrated_autocorrelation_time(samples)[0])
    exact = (1 + coefficient) / (1 - coefficient)
    assert abs(tau - emcee_tau) < STANDARD_ERRORS * tau_error(exact, STEP_COUNT * 16)


# * Validity checks


def test_short_chains_warn_and_keep_their_value() -> None:
    """Fewer than 50τ steps warn; τ is still returned."""
    samples = cold_samples((0.9,), 4, 400, 32)
    with pytest.warns(
        ValidityWarning, match=f"fewer than {MINIMUM_LENGTH_IN_TAU} autocorrelation"
    ):
        tau = integrated_autocorrelation_time(samples)
    assert np.isfinite(tau).all()


def test_pairs_positive_to_the_end_warn_and_reach_the_floor() -> None:
    """Pairs that never turn negative put τ at its floor, which warns."""
    samples = np.random.default_rng(5).normal(size=(3, 8, 1))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        tau = float(integrated_autocorrelation_time(samples)[0])
    messages = [str(record.message) for record in caught]
    assert any("stay positive to the end" in message for message in messages)
    assert all(record.category is ValidityWarning for record in caught)
    assert tau == pytest.approx(1 / math.log10(3 * 8))


def test_autocorrelations_of_each_walker_sum_to_one_half() -> None:
    """Centring on the walker mean makes Σ ρ(t) over all lags equal ½."""
    samples = cold_samples((0.9, -0.5), 10, 1_000, 4)
    function = np.asarray(autocorrelation_function(samples))
    np.testing.assert_allclose(function.sum(axis=0), 0.5, atol=1e-9)


def test_walkers_that_never_move_are_left_out() -> None:
    """A stuck walker leaves the average; the others still give the right τ."""
    samples = np.array(cold_samples((0.5, 0.5), 5, 4_000, 16))
    samples[:, 3, 1] = samples[0, 3, 1]
    with pytest.warns(ValidityWarning, match="walker 3 of parameter 1"):
        function = np.asarray(autocorrelation_function(samples))
    moving = np.delete(samples[:, :, 1], 3, axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        expected = np.asarray(autocorrelation_function(moving[:, :, None]))[:, 0]
    np.testing.assert_allclose(function[:, 1], expected, atol=1e-12)


def test_a_parameter_without_moving_walkers_gives_nan() -> None:
    """When no walker moves in a parameter, its τ is NaN and the rest stays."""
    samples = np.array(cold_samples((0.5, 0.0), 6, 4_000, 8))
    samples[:, :, 0] = 2.5
    with pytest.warns(ValidityWarning, match="no walker moves in parameter 0"):
        tau = np.asarray(integrated_autocorrelation_time(samples))
    assert math.isnan(tau[0])
    assert abs(tau[1] - 1.0) < STANDARD_ERRORS * tau_error(1.0, 4_000 * 8)


def test_warnings_count_long_lists() -> None:
    """A warning names a few walkers and counts the others."""
    samples = np.array(cold_samples((0.0, 0.0), 7, 200, 12))
    samples[:, :10, 1] = 1.0
    with pytest.warns(ValidityWarning, match=r"walkers 0, 1, .*, 7 and 2 more"):
        autocorrelation_function(samples)


# * Inputs, jit and vmap


@pytest.mark.parametrize(
    ("samples", "error", "message"),
    [
        (np.zeros((10, 3)), ValueError, r"\(step, walker, param\)"),
        (np.zeros((10, 0, 2)), ValueError, "entries on every axis"),
        (np.zeros((1, 4, 2)), ValueError, "at least 2 steps"),
        (np.full((10, 2, 1), np.nan), ValueError, "finite"),
        (np.zeros((10, 2, 1), dtype=complex), TypeError, "real numbers"),
    ],
)
def test_invalid_samples_raise(
    samples: np.ndarray, error: type[Exception], message: str
) -> None:
    """Shapes, values and types are checked with clear messages."""
    with pytest.raises(error, match=message):
        integrated_autocorrelation_time(samples)


def test_float32_samples_warn_once() -> None:
    """Narrow floats are cast to float64 with a PrecisionWarning."""
    samples = np.asarray(cold_samples((0.0,), 8, 1_000, 4)).astype(np.float32)
    with pytest.warns(PrecisionWarning, match=r"samples \(float32\)"):
        integrated_autocorrelation_time(samples)


def test_two_steps_are_enough_to_run() -> None:
    """The shortest chains run, warn, and stop at the floor."""
    samples = np.random.default_rng(9).normal(size=(2, 5, 1))
    with pytest.warns(ValidityWarning):
        tau = float(integrated_autocorrelation_time(samples)[0])
    assert tau == pytest.approx(1 / math.log10(10))


def test_jit_and_vmap_give_the_same_values_without_warnings() -> None:
    """Under jit and vmap τ is the same, and traced values never warn."""
    batch = jnp.stack(
        [cold_samples((0.9, 0.2), seed, 300, 8) for seed in range(3)], axis=0
    )
    with pytest.warns(ValidityWarning):
        direct = np.asarray(integrated_autocorrelation_time(batch[0]))
    stuck = batch[0].at[:, 0, 1].set(0.0)
    with pytest.warns(ValidityWarning, match="never move"):
        direct_function = np.asarray(autocorrelation_function(stuck))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        compiled = np.asarray(jax.jit(integrated_autocorrelation_time)(batch[0]))
        mapped = np.asarray(jax.vmap(integrated_autocorrelation_time)(batch))
        compiled_function = np.asarray(jax.jit(autocorrelation_function)(stuck))
    np.testing.assert_allclose(compiled, direct, rtol=1e-12)
    np.testing.assert_allclose(mapped[0], direct, rtol=1e-12)
    np.testing.assert_allclose(compiled_function, direct_function, atol=1e-14)
    assert mapped.shape == (3, 2)
