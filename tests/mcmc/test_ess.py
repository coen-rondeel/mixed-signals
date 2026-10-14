"""Tests for the effective sample size, the split-chain KS test and R̂."""

import math
import warnings
from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.special
import scipy.stats

from mixed_signals import zoo
from mixed_signals.errors import CoupledChainsWarning, ValidityWarning
from mixed_signals.mcmc import (
    MINIMUM_SPLIT_EFFECTIVE_SAMPLE_SIZE,
    effective_sample_size,
    integrated_autocorrelation_time,
    rank_normalized_split_rhat,
    split_chain_ks,
)
from mixed_signals.mcmc.ess import _kolmogorov_survival

STANDARD_ERRORS = 5.0


def cold_samples(
    coefficients: tuple[float, ...],
    seed: int,
    step_count: int,
    walker_count: int,
    start: float | None = None,
) -> np.ndarray:
    """Return zoo AR(1) chains shaped (step, walker, param) as a writable array."""
    chains = zoo.ar1_chains(coefficients).sample(
        jax.random.key(seed), step_count, walker_count=walker_count, start=start
    )
    return np.array(chains.samples[:, 0, :, 0, :])


def quiet[**Parameters, Result](
    function: Callable[Parameters, Result],
    *arguments: Parameters.args,
    **options: Parameters.kwargs,
) -> Result:
    """Call a function with the package's validity warnings silenced."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", CoupledChainsWarning)
        warnings.simplefilter("ignore", ValidityWarning)
        return function(*arguments, **options)


def reference_scale_reduction(chains: np.ndarray) -> np.ndarray:
    """Return √(var⁺/W) of chains shaped (draw, chain, param), as Gelman and Rubin."""
    draws = chains.shape[0]
    within = chains.var(axis=0, ddof=1).mean(axis=0)
    between = draws * chains.mean(axis=0).var(axis=0, ddof=1)
    pooled = (draws - 1) / draws * within + between / draws
    return np.asarray(np.sqrt(pooled / within))


def reference_scores(values: np.ndarray) -> np.ndarray:
    """Return Φ⁻¹((r − 3/8)/(S + 1/4)) of SciPy's average ranks over all draws."""
    flat = values.reshape(-1, values.shape[-1])
    ranks = scipy.stats.rankdata(flat, axis=0)
    scores = scipy.special.ndtri((ranks - 0.375) / (flat.shape[0] + 0.25))
    return np.asarray(scores).reshape(values.shape)


def reference_rhat(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return bulk and folded rank-normalised split-R̂ (Vehtari et al. 2021)."""
    step_count = samples.shape[0]
    half = step_count // 2
    split = np.concatenate([samples[:half], samples[step_count - half :]], axis=1)
    median = np.median(split.reshape(-1, split.shape[-1]), axis=0)
    return (
        reference_scale_reduction(reference_scores(split)),
        reference_scale_reduction(reference_scores(np.abs(split - median))),
    )


# * Effective sample size


def test_effective_sample_size_is_draws_over_tau() -> None:
    """ESS = S·W/τ for every parameter, also above S·W for antithetic chains."""
    samples = cold_samples((0.9, 0.0, -0.5), 0, 5_000, 32)
    ess = np.asarray(effective_sample_size(samples))
    tau = np.asarray(integrated_autocorrelation_time(samples))
    np.testing.assert_allclose(ess, 5_000 * 32 / tau, rtol=1e-12)
    draw_count = 5_000 * 32
    assert ess[2] > draw_count
    assert ess[0] == pytest.approx(draw_count / 19, rel=0.1)


def test_effective_sample_size_is_bounded_by_the_floor_of_tau() -> None:
    """Strongly antithetic chains reach S·W·log₁₀(S·W), the bound of the floor."""
    samples = cold_samples((-0.95,), 1, 5_000, 32)
    draw_count = 5_000 * 32
    ess = float(effective_sample_size(samples)[0])
    assert ess == pytest.approx(draw_count * math.log10(draw_count), rel=1e-12)


# * Split-chain Kolmogorov-Smirnov test


def test_split_test_matches_scipy_on_the_thinned_halves() -> None:
    """D and p equal SciPy's on the halves thinned by the returned factors."""
    samples = cold_samples((0.9, 0.3), 2, 3_001, 8)
    result = quiet(split_chain_ks, samples)
    half = samples.shape[0] // 2
    for index in range(2):
        thin = int(result.thin[index])
        first = samples[:half:thin, :, index].ravel()
        second = samples[samples.shape[0] - half :: thin, :, index].ravel()
        assert int(result.half_size[index]) == first.size == second.size
        expected = scipy.stats.ks_2samp(first, second).statistic
        assert float(result.statistic[index]) == pytest.approx(expected, abs=1e-12)
        scaled = math.sqrt(first.size / 2) * expected
        assert float(result.p_value[index]) == pytest.approx(
            float(scipy.special.kolmogorov(scaled)), abs=1e-12
        )
    assert int(result.thin[1]) < int(result.thin[0])


def test_split_test_p_values_are_uniform_for_stationary_chains() -> None:
    """Without a transient, the p-values spread evenly over (0, 1)."""
    p_values = np.array(
        [
            quiet(split_chain_ks, cold_samples((0.9, 0.0), seed, 1_000, 16)).p_value
            for seed in range(100)
        ]
    )
    # * With 100 draws the mean of uniform p-values has the standard error 0.029.
    for column in p_values.T:
        assert abs(column.mean() - 0.5) < STANDARD_ERRORS * math.sqrt(1 / 12 / 100)
        assert np.mean(column < 0.05) < 0.05 + STANDARD_ERRORS * math.sqrt(
            0.05 * 0.95 / 100
        )


def test_split_test_flags_a_transient_that_burn_in_removes() -> None:
    """A start 20σ away fails the test; dropping the transient passes it."""
    samples = cold_samples((0.95,), 3, 600, 32, start=20.0)
    flagged = quiet(split_chain_ks, samples)
    assert float(flagged.p_value[0]) < 1e-3
    cleaned = quiet(split_chain_ks, samples[200:])
    assert float(cleaned.p_value[0]) > 1e-3


def test_split_test_of_a_constant_parameter_passes() -> None:
    """Identical halves give D = 0 and p = 1."""
    samples = cold_samples((0.5, 0.5), 4, 400, 4)
    samples[:, :, 1] = 3.0
    result = quiet(split_chain_ks, samples)
    assert float(result.statistic[1]) == pytest.approx(0.0, abs=1e-12)
    assert float(result.p_value[1]) == 1.0
    assert int(result.thin[1]) == 1


def test_kolmogorov_survival_matches_scipy() -> None:
    """Q(λ) equals SciPy's kolmogorov on both sides of the series crossover."""
    grid = np.concatenate([[0.0, 0.01, 0.1], np.linspace(0.2, 4.0, 77), [10.0, 30.0]])
    values = np.asarray(_kolmogorov_survival(jnp.asarray(grid)))
    np.testing.assert_allclose(values, scipy.special.kolmogorov(grid), atol=1e-15)


# * R̂


def test_rhat_matches_an_independent_implementation() -> None:
    """R̂ is the larger of bulk and folded split-R̂, with average ranks for ties."""
    rng = np.random.default_rng(5)
    samples = rng.normal(size=(101, 6, 2)).round(1)
    samples[:, 0, 1] += 0.4
    rhat = np.asarray(quiet(rank_normalized_split_rhat, samples))
    bulk, folded = reference_rhat(samples)
    np.testing.assert_allclose(rhat, np.maximum(bulk, folded), rtol=1e-12)


def test_rhat_is_one_for_chains_that_agree() -> None:
    """Independent walkers of one distribution give R̂ below 1.01."""
    samples = cold_samples((0.0, 0.5), 6, 2_000, 8)
    rhat = np.asarray(quiet(rank_normalized_split_rhat, samples))
    assert np.all(rhat < 1.01)


def test_rhat_flags_a_shifted_walker() -> None:
    """One walker shifted by 1σ raises the bulk R̂ above 1.01."""
    samples = cold_samples((0.0,), 7, 2_000, 8)
    samples[:, 0] += 1.0
    assert float(quiet(rank_normalized_split_rhat, samples)[0]) > 1.01


def test_folding_flags_walkers_with_another_spread() -> None:
    """Walkers at half the width pass the bulk R̂ but not the folded one."""
    samples = cold_samples((0.0,), 8, 2_000, 8)
    samples[:, :4] *= 0.5
    bulk, folded = reference_rhat(samples)
    assert bulk[0] < 1.01 < folded[0]
    assert float(quiet(rank_normalized_split_rhat, samples)[0]) == pytest.approx(
        folded[0], rel=1e-12
    )


def test_rhat_always_warns_about_coupled_walkers() -> None:
    """Every concrete call raises a CoupledChainsWarning."""
    samples = cold_samples((0.0,), 9, 2_000, 8)
    with pytest.warns(CoupledChainsWarning, match="informational"):
        rank_normalized_split_rhat(samples)


def test_short_split_chains_warn() -> None:
    """Split chains with fewer than 50 effective samples make R̂ unreliable."""
    samples = cold_samples((0.9,), 10, 200, 4)
    with (
        pytest.warns(CoupledChainsWarning),
        pytest.warns(
            ValidityWarning, match=f"fewer than {MINIMUM_SPLIT_EFFECTIVE_SAMPLE_SIZE}"
        ),
    ):
        rank_normalized_split_rhat(samples)


def test_rhat_of_a_constant_parameter_is_nan() -> None:
    """A parameter that never moves gives NaN and a warning; the rest stays."""
    samples = cold_samples((0.0, 0.0), 11, 2_000, 8)
    samples[:, :, 0] = 1.0
    with (
        pytest.warns(CoupledChainsWarning),
        pytest.warns(ValidityWarning, match="never move: parameter 0"),
    ):
        rhat = np.asarray(rank_normalized_split_rhat(samples))
    assert math.isnan(rhat[0])
    assert rhat[1] < 1.01


def test_rhat_needs_four_steps() -> None:
    """Each half needs two draws for its variance."""
    with pytest.raises(ValueError, match="at least 4 steps"):
        rank_normalized_split_rhat(np.zeros((3, 4, 1)))


def test_traced_calls_give_the_same_values_without_warnings() -> None:
    """Under jit the functions agree with concrete calls and never warn."""
    samples = jnp.asarray(cold_samples((0.9, 0.2), 12, 3_000, 8))
    direct_rhat = quiet(rank_normalized_split_rhat, samples)
    direct_test = quiet(split_chain_ks, samples)
    direct_ess = quiet(effective_sample_size, samples)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        compiled_rhat = jax.jit(rank_normalized_split_rhat)(samples)
        compiled_test = jax.jit(split_chain_ks)(samples)
        compiled_ess = jax.jit(effective_sample_size)(samples)
    np.testing.assert_allclose(compiled_rhat, direct_rhat, rtol=1e-12)
    np.testing.assert_allclose(compiled_test.p_value, direct_test.p_value, rtol=1e-12)
    np.testing.assert_array_equal(compiled_test.thin, direct_test.thin)
    np.testing.assert_allclose(compiled_ess, direct_ess, rtol=1e-12)
