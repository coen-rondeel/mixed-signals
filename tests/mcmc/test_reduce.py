"""Tests for the chain reduction: cold chain, burn-in, thinning, n_eff and record."""

import math
import warnings

import equinox as eqx
import jax
import numpy as np
import pytest

from mixed_signals import zoo
from mixed_signals.core.chains import Chains
from mixed_signals.errors import ValidityWarning
from mixed_signals.mcmc import (
    ReductionRecord,
    hash_chains,
    integrated_autocorrelation_time,
    reduce_chains,
    split_chain_ks,
)

STANDARD_ERRORS = 5.0


def ar1_chains(
    coefficients: tuple[float, ...],
    seed: int,
    step_count: int,
    walker_count: int,
    start: float | None = None,
) -> Chains:
    """Return zoo AR(1) chains with one temperature."""
    return zoo.ar1_chains(coefficients).sample(
        jax.random.key(seed), step_count, walker_count=walker_count, start=start
    )


def with_log_densities(chains: Chains) -> Chains:
    """Return the chains with log_like = −|x|²/2 and log_prior = Σ x per walker."""
    samples = np.asarray(chains.samples[:, 0, :, 0, :])
    return Chains(
        samples,
        log_like=-0.5 * np.sum(samples**2, axis=-1),
        log_prior=np.sum(samples, axis=-1),
    )


# * Steps of the reduction


def test_kept_steps_are_flattened_in_step_order() -> None:
    """Every k-th step after the burn-in is kept, walkers inside each step."""
    chains = ar1_chains((0.5, -0.2), 0, 1_000, 6)
    reduction = reduce_chains(chains, burn_in=100, thin=7)
    samples = np.asarray(chains.samples[:, 0, :, 0, :])
    expected = samples[100::7].reshape(-1, 2)
    assert reduction.sample_set.shape == (1, 1, expected.shape[0], 2)
    np.testing.assert_array_equal(reduction.sample_set.samples[0, 0], expected)
    record = reduction.record
    assert (record.burn_in, record.thin, record.kept_steps) == (100, 7, 129)
    assert record.sample_count == 129 * 6
    assert record.chain_shape == (1_000, 1, 6, 1, 2)


def test_tau_thinning_takes_the_largest_factor_over_parameters() -> None:
    """thin='tau' uses the slowest parameter's factor, as the split test does."""
    chains = ar1_chains((0.9, 0.0), 1, 4_000, 16)
    reduction = reduce_chains(chains, burn_in=50)
    after_burn_in = np.asarray(chains.samples[50:, 0, :, 0, :])
    factors = np.asarray(split_chain_ks(after_burn_in).thin)
    assert reduction.record.thin == int(factors.max()) == int(factors[0])
    np.testing.assert_allclose(
        reduction.record.tau, integrated_autocorrelation_time(after_burn_in), rtol=1e-12
    )
    # * The rule subtracts twice the error of τ, so it never thins beyond ⌈τ⌉.
    assert 1 < reduction.record.thin <= math.ceil(reduction.record.tau[0])


def test_n_eff_counts_the_correlation_left_after_thinning() -> None:
    """n_eff = N/τ_k, and τ_k matches (1 + φᵏ)/(1 − φᵏ) of AR(1) chains."""
    coefficient = 0.9
    chains = ar1_chains((coefficient,), 2, 20_000, 32)
    record = reduce_chains(chains).record
    exact = (1 + coefficient**record.thin) / (1 - coefficient**record.thin)
    # * A priori error of τ_k over about ten kept lags (Madras and Sokal 1988).
    error = exact * math.sqrt(2 * (2 * 10 + 1) / record.sample_count)
    assert abs(record.thinned_tau[0] - exact) < STANDARD_ERRORS * error
    assert record.n_eff == pytest.approx(record.sample_count / record.thinned_tau[0])
    assert record.n_eff < record.sample_count


def test_antithetic_chains_keep_every_step_and_every_sample() -> None:
    """τ < 1 gives k = 1, and n_eff is capped at the number of samples."""
    reduction = reduce_chains(ar1_chains((-0.5,), 3, 2_000, 8))
    assert reduction.record.thin == 1
    assert reduction.record.n_eff == reduction.record.sample_count == 2_000 * 8
    assert reduction.sample_set.n_eff is not None
    assert float(reduction.sample_set.n_eff[0, 0]) == 2_000 * 8


def test_log_densities_follow_their_samples() -> None:
    """log_like and log_prior are kept and flattened with their samples."""
    chains = with_log_densities(ar1_chains((0.3, 0.6), 4, 600, 5))
    sample_set = reduce_chains(chains, burn_in=10, thin=3).sample_set
    kept = np.asarray(sample_set.samples[0, 0])
    assert sample_set.log_like is not None
    assert sample_set.log_prior is not None
    expected_log_like = -0.5 * np.sum(kept**2, axis=-1)
    np.testing.assert_allclose(sample_set.log_like[0, 0], expected_log_like)
    np.testing.assert_allclose(sample_set.log_prior[0, 0], np.sum(kept, axis=-1))


def test_the_cold_chain_of_tempered_chains_is_reduced() -> None:
    """With β = (0.5, 1), the second temperature is the cold chain."""
    rng = np.random.default_rng(5)
    samples = rng.normal(size=(500, 2, 4, 1, 2))
    samples[:, 1] += 10.0
    chains = Chains(samples, betas=np.tile([0.5, 1.0], (500, 1)))
    reduction = reduce_chains(chains, thin=1)
    assert reduction.record.temperature == 1
    np.testing.assert_array_equal(
        reduction.sample_set.samples[0, 0], samples[:, 1, :, 0, :].reshape(-1, 2)
    )


def test_burn_in_removes_a_transient() -> None:
    """A start 20σ away biases the mean; dropping 300 steps removes the bias."""
    chains = ar1_chains((0.95,), 6, 2_000, 32, start=20.0)
    biased = reduce_chains(chains)
    cleaned = reduce_chains(chains, burn_in=300)
    for reduction, expect_bias in ((biased, True), (cleaned, False)):
        mean = float(np.mean(np.asarray(reduction.sample_set.samples)))
        error = 1 / math.sqrt(reduction.record.n_eff)
        assert (abs(mean) > STANDARD_ERRORS * error) is expect_bias


# * Repeating and recording


def test_a_reduction_repeats_from_its_record() -> None:
    """The recorded burn-in and thinning give equal samples, n_eff and record."""
    chains = with_log_densities(ar1_chains((0.9, 0.1), 7, 3_000, 8, start=5.0))
    first = reduce_chains(chains, burn_in=200)
    again = reduce_chains(chains, burn_in=first.record.burn_in, thin=first.record.thin)
    assert eqx.tree_equal(first.sample_set, again.sample_set)
    assert first.record == again.record
    assert hash_chains(chains) == first.record.chains_hash


def test_the_hash_identifies_the_input_chains() -> None:
    """Other values or other log densities give another hash."""
    chains = ar1_chains((0.5,), 8, 300, 4)
    shifted = Chains(np.asarray(chains.samples) + 1e-12)
    densities = with_log_densities(chains)
    digests = {hash_chains(chains), hash_chains(shifted), hash_chains(densities)}
    assert len(digests) == 3


def test_records_round_trip_through_json() -> None:
    """A record, NaN values included, survives JSON as text and as values."""
    samples = np.array(ar1_chains((0.5, 0.5), 9, 500, 4).samples)
    samples[..., 1] = 2.0
    with pytest.warns(ValidityWarning, match="no walker moves in parameter 1"):
        record = reduce_chains(Chains(samples)).record
    assert math.isnan(record.tau[1])
    text = record.model_dump_json()
    assert "NaN" in text
    again = ReductionRecord.model_validate_json(text)
    assert again.model_dump_json() == text
    assert again.tau[0] == record.tau[0]


# * Parameters that never move


def test_a_parameter_that_never_moves_does_not_set_n_eff() -> None:
    """n_eff and the thinning come from the parameters that move."""
    samples = np.array(ar1_chains((0.5, 0.5), 10, 2_000, 8).samples)
    samples[..., 0] = 1.0
    with pytest.warns(ValidityWarning, match="no walker moves in parameter 0"):
        record = reduce_chains(Chains(samples)).record
    assert math.isnan(record.thinned_tau[0])
    assert record.n_eff == pytest.approx(record.sample_count / record.thinned_tau[1])


def test_n_eff_is_one_when_no_parameter_moves() -> None:
    """Identical samples hold the information of a single one."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        reduction = reduce_chains(Chains(np.ones((100, 4, 2))))
    messages = [str(record.message) for record in caught]
    assert any("n_eff is 1" in message for message in messages)
    assert reduction.record.n_eff == 1.0
    assert reduction.record.thin == 1


# * Invalid arguments


@pytest.mark.parametrize(
    ("arguments", "error", "message"),
    [
        ({"burn_in": True}, TypeError, "burn_in must be an integer"),
        ({"burn_in": -1}, ValueError, "burn_in must be at least 0"),
        ({"burn_in": 99}, ValueError, "leaves 1 of 100 steps"),
        ({"thin": "auto"}, ValueError, "thin must be 'tau' or an integer"),
        ({"thin": 0}, ValueError, "thin must be at least 1"),
        ({"thin": True}, TypeError, "thin must be an integer"),
        ({"thin": 100}, ValueError, "keeps 1 of the 100 steps"),
    ],
)
def test_invalid_arguments_raise(
    arguments: dict[str, object], error: type[Exception], message: str
) -> None:
    """Arguments are checked before the chains are read."""
    chains = ar1_chains((0.0,), 11, 100, 4)
    with pytest.raises(error, match=message):
        reduce_chains(chains, **arguments)  # type: ignore[arg-type]


def test_chains_without_a_cold_chain_raise() -> None:
    """A ladder without β = 1 has no posterior to reduce."""
    chains = Chains(np.zeros((10, 2, 3, 1, 1)), betas=np.full((10, 2), 0.5))
    with pytest.raises(ValueError, match="no cold chain"):
        reduce_chains(chains)
