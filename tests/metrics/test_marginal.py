"""Tests for the marginal metrics: the weighted two-sample KS statistic."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.stats

from mixed_signals.metrics.marginal import ks_statistic


@pytest.fixture
def rng() -> np.random.Generator:
    """A seeded NumPy generator for test data."""
    return np.random.default_rng(2026)


def statistic(
    first: np.ndarray,
    second: np.ndarray,
    first_log_weights: np.ndarray | None = None,
    second_log_weights: np.ndarray | None = None,
) -> float:
    """Return the package's statistic of two one-dimensional samples."""
    return float(
        ks_statistic(
            jnp.asarray(first),
            jnp.zeros(first.shape)
            if first_log_weights is None
            else jnp.asarray(first_log_weights),
            jnp.asarray(second),
            jnp.zeros(second.shape)
            if second_log_weights is None
            else jnp.asarray(second_log_weights),
        )
    )


@pytest.mark.parametrize(
    ("first_size", "second_size", "decimals"),
    [(300, 200, None), (50, 500, 1), (1_000, 1_000, 0)],
)
def test_equal_weights_give_the_scipy_statistic(
    rng: np.random.Generator, first_size: int, second_size: int, decimals: int | None
) -> None:
    """With equal weights, D equals SciPy's ks_2samp, also for samples with ties."""
    first = rng.normal(size=first_size)
    second = rng.normal(0.3, 1.2, size=second_size)
    if decimals is not None:
        first, second = first.round(decimals), second.round(decimals)
    expected = scipy.stats.ks_2samp(first, second).statistic
    assert statistic(first, second) == pytest.approx(expected, abs=1e-12)


def test_integer_weights_count_as_repeated_samples(rng: np.random.Generator) -> None:
    """A weight of k counts like k copies of a sample; the scale does not matter."""
    first = rng.normal(size=40)
    copies = rng.integers(1, 5, size=40)
    second = rng.normal(0.5, 1.0, size=60)
    expected = scipy.stats.ks_2samp(np.repeat(first, copies), second).statistic
    log_weights = np.log(copies) + 7.0
    assert statistic(first, second, log_weights) == pytest.approx(expected, abs=1e-12)


def test_zero_weights_remove_samples(rng: np.random.Generator) -> None:
    """A log weight of −∞ removes a sample from its distribution function."""
    first = rng.normal(size=80)
    second = rng.normal(size=70)
    keep = np.arange(80) % 3 != 0
    expected = scipy.stats.ks_2samp(first[keep], second).statistic
    log_weights = np.where(keep, 0.0, -np.inf)
    assert statistic(first, second, log_weights) == pytest.approx(expected, abs=1e-12)


def test_leading_axes_give_one_statistic_per_row(rng: np.random.Generator) -> None:
    """Leading axes map: every row of (3, 2) pairs gets its own statistic."""
    first = rng.normal(size=(3, 2, 100))
    second = rng.normal(0.2, 1.0, size=(3, 2, 80))
    values = ks_statistic(
        jnp.asarray(first),
        jnp.zeros(first.shape),
        jnp.asarray(second),
        jnp.zeros(second.shape),
    )
    assert values.shape == (3, 2)
    for row in np.ndindex(3, 2):
        expected = scipy.stats.ks_2samp(first[row], second[row]).statistic
        assert float(values[row]) == pytest.approx(expected, abs=1e-12)


def test_identical_disjoint_and_swapped_samples(rng: np.random.Generator) -> None:
    """Identical samples give 0, separated ones 1, and swapping changes nothing."""
    first = rng.normal(size=200)
    second = rng.normal(size=150)
    assert statistic(first, first) == pytest.approx(0.0, abs=1e-12)
    assert statistic(first, first + 100.0) == pytest.approx(1.0)
    assert statistic(first, second) == pytest.approx(statistic(second, first))


def test_statistic_runs_under_vmap(rng: np.random.Generator) -> None:
    """The kernel maps over a batch axis like any JAX function."""
    first = jnp.asarray(rng.normal(size=(4, 50)))
    second = jnp.asarray(rng.normal(size=(4, 60)))
    mapped = jax.vmap(ks_statistic)(
        first, jnp.zeros((4, 50)), second, jnp.zeros((4, 60))
    )
    direct = ks_statistic(first, jnp.zeros((4, 50)), second, jnp.zeros((4, 60)))
    np.testing.assert_allclose(mapped, direct, rtol=0, atol=1e-15)
