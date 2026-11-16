"""Tests for the chain-health diagnostics."""

import math
import warnings

import jax
import numpy as np
import pytest

from mixed_signals import zoo
from mixed_signals.core.chains import Chains
from mixed_signals.core.space import ParamSpace
from mixed_signals.diagnostics.chains import (
    ChainHealth,
    chain_health,
    split_chain_metric_names,
)
from mixed_signals.errors import CoupledChainsWarning, ValidityWarning
from mixed_signals.mcmc import (
    integrated_autocorrelation_time,
    rank_normalized_split_rhat,
    split_chain_ks,
    split_chain_kuiper,
)

TWO_PI = 2 * math.pi
LINE_AND_CIRCLE = ParamSpace.model_validate(
    {
        "parameters": [
            {"name": "x0"},
            {"name": "phase", "topology": "circle", "bounds": [0.0, TWO_PI]},
        ]
    }
)


def ar1_samples(coefficients: tuple[float, ...], seed: int, steps: int) -> np.ndarray:
    """Return AR(1) chains shaped (step, walker, param) as a writable array."""
    chains = zoo.ar1_chains(coefficients).sample(jax.random.key(seed), steps)
    return np.array(chains.samples[:, 0, :, 0, :])


def quiet_health(chains: Chains, space: ParamSpace, burn_in: int = 0) -> ChainHealth:
    """Return the chain health with the package's warnings silenced."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", CoupledChainsWarning)
        warnings.simplefilter("ignore", ValidityWarning)
        return chain_health(chains, space, burn_in=burn_in)


def test_lines_use_the_mcmc_kernels_after_the_burn_in() -> None:
    """τ, ESS, the KS split test and R̂ of a line are those of the kernels."""
    samples = ar1_samples((0.9, 0.0), 0, 3_000)
    space = ParamSpace.model_validate({"parameters": [{"name": "x0"}, {"name": "x1"}]})
    health = quiet_health(Chains(samples), space, burn_in=200)
    kept = samples[200:]
    tau = np.asarray(integrated_autocorrelation_time(kept))
    np.testing.assert_allclose(health.tau, tau, rtol=1e-12)
    np.testing.assert_allclose(
        health.effective_sample_size, kept.shape[0] * kept.shape[1] / tau, rtol=1e-12
    )
    expected = split_chain_ks(kept)
    np.testing.assert_allclose(health.split_statistic, expected.statistic, rtol=1e-12)
    np.testing.assert_allclose(health.split_p_value, expected.p_value, rtol=1e-12)
    np.testing.assert_array_equal(health.split_thin, expected.thin)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", CoupledChainsWarning)
        np.testing.assert_allclose(
            health.rhat, rank_normalized_split_rhat(kept), rtol=1e-12
        )
    assert health.acceptance_rate is None
    assert health.swap_rate is None


def test_a_narrow_circle_across_the_cut_keeps_its_tau() -> None:
    """Angles that wrap at the cut keep the τ of the series; raw angles read it low."""
    samples = ar1_samples((0.0, 0.5), 1, 4_000)
    samples[..., 1] = np.mod(0.3 * samples[..., 1], TWO_PI)
    health = quiet_health(Chains(samples), LINE_AND_CIRCLE)
    assert float(health.tau[1]) == pytest.approx(3.0, rel=0.1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ValidityWarning)
        raw = float(integrated_autocorrelation_time(samples[..., 1:])[0])
    # * Jumps of a period at the cut act as white noise and shorten the raw τ.
    assert raw < 0.8 * float(health.tau[1])


def test_circles_take_kuiper_on_unwrapped_values() -> None:
    """The circle's split test is Kuiper's V of values unwrapped about the mean."""
    samples = ar1_samples((0.5, 0.5), 2, 2_000)
    samples[..., 1] = np.mod(0.4 * samples[..., 1], TWO_PI)
    health = quiet_health(Chains(samples), LINE_AND_CIRCLE)
    unwrapped = np.angle(np.exp(1j * samples[..., 1]))
    expected = split_chain_kuiper(
        unwrapped[..., None], thin=np.asarray(health.split_thin[1:])
    )
    assert float(health.split_statistic[1]) == pytest.approx(
        float(expected.statistic[0]), abs=1e-12
    )
    assert float(health.split_p_value[1]) == pytest.approx(
        float(expected.p_value[0]), abs=1e-12
    )
    assert split_chain_metric_names(LINE_AND_CIRCLE) == (
        "split_chain_ks",
        "split_chain_kuiper_ks",
    )


def test_rates_come_from_the_counts_of_the_run() -> None:
    """Acceptance per temperature pools the walkers; swaps divide by their proposals."""
    rng = np.random.default_rng(3)
    samples = rng.normal(size=(200, 2, 4, 1, 2))
    chains = Chains(
        samples,
        betas=np.tile([1.0, 0.5], (200, 1)),
        accepted=np.array([[10, 20, 30, 40], [5, 5, 5, 5]]),
        proposed=np.full((2, 4), 100),
        swaps_accepted=np.array([30]),
        swaps_proposed=np.array([120]),
    )
    space = ParamSpace.model_validate({"parameters": [{"name": "x0"}, {"name": "x1"}]})
    health = quiet_health(chains, space)
    assert health.acceptance_rate is not None
    assert health.swap_rate is not None
    np.testing.assert_allclose(health.acceptance_rate, np.array([0.25, 0.05]))
    np.testing.assert_allclose(health.swap_rate, np.array([0.25]))


def test_r_hat_always_warns_about_coupled_walkers() -> None:
    """R̂ is informational, so every call raises a CoupledChainsWarning."""
    samples = ar1_samples((0.3, 0.3), 4, 2_000)
    space = ParamSpace.model_validate({"parameters": [{"name": "x0"}, {"name": "x1"}]})
    with pytest.warns(CoupledChainsWarning, match="informational"):
        chain_health(Chains(samples), space)


def test_inputs_are_checked() -> None:
    """The space must fit the chains, and the burn-in must leave enough steps."""
    samples = ar1_samples((0.3, 0.3), 5, 100)
    one = ParamSpace.model_validate({"parameters": [{"name": "x0"}]})
    with pytest.raises(ValueError, match="2 parameters, the space 1"):
        chain_health(Chains(samples), one)
    two = ParamSpace.model_validate({"parameters": [{"name": "x0"}, {"name": "x1"}]})
    with pytest.raises(ValueError, match="burn_in must be at least 0"):
        chain_health(Chains(samples), two, burn_in=-1)
    with pytest.raises(ValueError, match="at least 4 steps"):
        chain_health(Chains(samples), two, burn_in=98)
