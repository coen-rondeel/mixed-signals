"""Tests for the empirical p-values and σ_sig."""

import math

import jax.numpy as jnp
import numpy as np
import pytest
from scipy import stats

from mixed_signals.sigma.significance import Tail, empirical_p_value, sigma_sig

STANDARD_ERRORS = 5.0


def test_upper_p_value_counts_draws_at_or_above_the_value() -> None:
    """The p-value is (1 + draws ≥ value)/(1 + M) (Phipson and Smyth 2010)."""
    draws = jnp.array([1.0, 2.0, 3.0, 4.0])
    assert float(empirical_p_value(2.0, draws, "upper")) == pytest.approx(4 / 5)
    assert float(empirical_p_value(5.0, draws, Tail.UPPER)) == pytest.approx(1 / 5)
    assert float(empirical_p_value(0.0, draws, "upper")) == 1.0


def test_two_sided_p_value_counts_draws_as_far_from_their_mean() -> None:
    """Two sides compare distances from the mean of the draws."""
    draws = jnp.array([-2.0, -1.0, 1.0, 2.0])
    assert float(empirical_p_value(1.5, draws, "two_sided")) == pytest.approx(3 / 5)
    assert float(empirical_p_value(-1.5, draws, "two_sided")) == pytest.approx(3 / 5)
    assert float(empirical_p_value(3.0, draws, "two_sided")) == pytest.approx(1 / 5)


def test_p_values_broadcast_over_leading_axes() -> None:
    """Each leading index has its own value and draws."""
    draws = jnp.array([[0.0, 1.0, 2.0], [5.0, 6.0, 7.0]])
    values = jnp.array([1.5, 4.0])
    np.testing.assert_allclose(empirical_p_value(values, draws, "upper"), [0.5, 1.0])


def test_non_finite_draws_leave_the_count() -> None:
    """NaN draws are left out of M; a NaN value or no finite draw gives NaN."""
    draws = jnp.array([1.0, jnp.nan, 3.0, jnp.inf])
    assert float(empirical_p_value(2.0, draws, "upper")) == pytest.approx(2 / 3)
    assert math.isnan(float(empirical_p_value(jnp.nan, draws, "upper")))
    assert math.isnan(float(empirical_p_value(1.0, jnp.full(3, jnp.nan), "two_sided")))


def test_p_values_are_uniform_for_exchangeable_draws() -> None:
    """When the value is one more draw of the null, P(p ≤ α) = α to resolution."""
    rng = np.random.default_rng(0)
    trials = rng.normal(size=(20_000, 100))
    for tail in ("upper", "two_sided"):
        p_values = np.asarray(
            empirical_p_value(
                jnp.asarray(trials[:, 0]), jnp.asarray(trials[:, 1:]), tail
            )
        )
        for level in (0.05, 0.2, 0.5):
            error = math.sqrt(level * (1 - level) / len(p_values))
            assert abs(np.mean(p_values <= level) - level) < STANDARD_ERRORS * error


def test_sigma_sig_matches_the_normal_quantiles() -> None:
    """Upper: Φ⁻¹(1 − p); two sides: Φ⁻¹(1 − p/2)."""
    p_values = np.array([1e-12, 1e-4, 0.005, 0.05, 0.3, 0.999])
    np.testing.assert_allclose(sigma_sig(p_values, "upper"), stats.norm.isf(p_values))
    np.testing.assert_allclose(
        sigma_sig(p_values, "two_sided"), stats.norm.isf(p_values / 2)
    )
    assert float(sigma_sig(1.0, "upper")) == -math.inf
    assert float(sigma_sig(1.0, "two_sided")) == 0.0
    assert bool(jnp.all(jnp.isnan(sigma_sig(jnp.array([0.0, 1.5, jnp.nan]), "upper"))))


def test_two_sided_sigma_sig_reads_the_z_of_a_gaussian_null() -> None:
    """With many Gaussian draws, a value z from their mean gives σ_sig ≈ |z|."""
    draws = jnp.asarray(stats.norm.ppf((np.arange(100_000) + 0.5) / 100_000))
    for score in (-2.0, 1.0, 2.5):
        p_value = empirical_p_value(score, draws, "two_sided")
        assert float(sigma_sig(p_value, "two_sided")) == pytest.approx(
            abs(score), abs=1e-3
        )


def test_an_unknown_tail_raises() -> None:
    """Only the two tails of Tail are accepted."""
    with pytest.raises(ValueError, match="lower"):
        empirical_p_value(1.0, jnp.ones(3), "lower")
    with pytest.raises(ValueError, match="lower"):
        sigma_sig(0.1, "lower")
