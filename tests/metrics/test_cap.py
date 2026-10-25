"""Tests for the subsampling cap of metrics whose cost grows as N²."""

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import stats

from mixed_signals._config import SAMPLE_CAP
from mixed_signals.errors import SubsamplingWarning
from mixed_signals.metrics._cap import cap_pair, cap_samples
from mixed_signals.prng import derive_key


def test_samples_below_the_cap_come_back_unchanged() -> None:
    """At or below the cap nothing is drawn and nothing warns."""
    samples = jnp.arange(24.0).reshape(2, 6, 2)
    log_weights = jnp.zeros((2, 6))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        kept, kept_weights = cap_samples(
            samples, log_weights, key=jax.random.key(0), metric="mmd", cap=6
        )
    assert kept is samples or bool(jnp.all(kept == samples))
    assert bool(jnp.all(kept_weights == log_weights))
    assert SAMPLE_CAP == 10_000


def test_above_the_cap_each_row_keeps_a_subset_in_order() -> None:
    """Every problem and posterior keeps cap distinct samples, with their weights."""
    count = 50
    samples = jnp.broadcast_to(jnp.arange(float(count))[:, None], (2, 3, count, 2))
    log_weights = jnp.broadcast_to(-jnp.arange(float(count)) / 10, (2, 3, count))
    with pytest.warns(SubsamplingWarning, match="energy: 50 samples per posterior"):
        kept, kept_weights = cap_samples(
            samples, log_weights, key=jax.random.key(1), metric="energy", cap=12
        )
    assert kept.shape == (2, 3, 12, 2)
    positions = np.asarray(kept[..., 0]).astype(int)
    for row in positions.reshape(-1, 12):
        assert np.all(np.diff(row) > 0)
    np.testing.assert_allclose(kept_weights, -positions / 10)
    assert len({tuple(row) for row in positions.reshape(-1, 12)}) > 1


def test_zero_weights_are_kept_only_when_too_few_others_remain() -> None:
    """Samples of zero weight follow every sample with a positive weight."""
    samples = jnp.arange(20.0).reshape(1, 20, 1)
    log_weights = jnp.zeros((1, 20)).at[0, :8].set(-jnp.inf)
    with pytest.warns(SubsamplingWarning):
        kept, kept_weights = cap_samples(
            samples, log_weights, key=jax.random.key(2), metric="mmd", cap=10
        )
    assert bool(jnp.all(kept[0, :, 0] >= 8))
    assert bool(jnp.all(jnp.isfinite(kept_weights)))
    with pytest.warns(SubsamplingWarning):
        crowded, crowded_weights = cap_samples(
            samples, log_weights, key=jax.random.key(2), metric="mmd", cap=15
        )
    assert int(jnp.sum(jnp.isfinite(crowded_weights))) == 12
    assert crowded.shape == (1, 15, 1)


def test_every_sample_is_kept_with_the_same_probability() -> None:
    """Over many keys, inclusion counts follow the binomial law of cap/N."""
    count, cap, draws = 40, 10, 2_000
    samples = jnp.arange(float(count)).reshape(count, 1)
    included = np.zeros(count)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SubsamplingWarning)
        for index in range(draws):
            kept, _ = cap_samples(
                samples,
                jnp.zeros(count),
                key=jax.random.key(index),
                metric="mmd",
                cap=cap,
            )
            included[np.asarray(kept[:, 0]).astype(int)] += 1
    expected = draws * cap / count
    statistic = np.sum((included - expected) ** 2 / expected)
    assert stats.chi2.sf(statistic, count - 1) > 1e-3


def test_the_same_key_draws_the_same_subset() -> None:
    """A subset depends on the key only."""
    samples = jnp.arange(30.0).reshape(30, 1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SubsamplingWarning)
        first, _ = cap_samples(
            samples, jnp.zeros(30), key=jax.random.key(4), metric="mmd", cap=7
        )
        again, _ = cap_samples(
            samples, jnp.zeros(30), key=jax.random.key(4), metric="mmd", cap=7
        )
    np.testing.assert_array_equal(first, again)


def test_cap_checks_its_inputs() -> None:
    """A cap below one and shapes that do not fit fail with clear messages."""
    with pytest.raises(ValueError, match="cap must be at least 1"):
        cap_samples(
            jnp.zeros((5, 1)), jnp.zeros(5), key=jax.random.key(0), metric="mmd", cap=0
        )
    with pytest.raises(ValueError, match=r"shaped \(\.\.\., N, D\)"):
        cap_samples(jnp.zeros(5), jnp.zeros(5), key=jax.random.key(0), metric="mmd")
    with pytest.raises(ValueError, match=r"log_weights \(\.\.\., N\)"):
        cap_samples(
            jnp.zeros((5, 1)), jnp.zeros(4), key=jax.random.key(0), metric="mmd"
        )


def test_a_pair_below_the_cap_comes_back_unchanged() -> None:
    """Both posteriors at or below the cap pass through without a warning."""
    first, second = jnp.ones((6, 2)), jnp.zeros((4, 2))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        kept = cap_pair(
            first,
            jnp.zeros(6),
            second,
            jnp.zeros(4),
            key=jax.random.key(0),
            metric="mmd",
            cap=6,
        )
    assert [part.shape for part in kept] == [(6, 2), (6,), (4, 2), (4,)]


def test_a_pair_above_the_cap_warns_once_and_draws_each_from_its_key() -> None:
    """Each posterior above the cap draws from its own derived key; the other stays."""
    first = jnp.arange(40.0).reshape(20, 2)
    second = jnp.arange(16.0).reshape(8, 2)
    key = jax.random.key(3)
    with pytest.warns(SubsamplingWarning, match="kl: 20 and 8 samples") as caught:
        kept = cap_pair(
            first, jnp.zeros(20), second, jnp.zeros(8), key=key, metric="kl", cap=10
        )
    assert len(caught) == 1
    expected, _ = cap_samples_quietly(first, derive_key(key, "first"), 10)
    np.testing.assert_array_equal(kept[0], expected)
    np.testing.assert_array_equal(kept[2], second)
    assert kept[1].shape == (10,)


def test_a_pair_checks_its_shapes() -> None:
    """Both posteriors need (N, D) samples, (N,) weights and the same D."""
    key = jax.random.key(0)
    with pytest.raises(ValueError, match="cap must be at least 1"):
        cap_pair(
            jnp.zeros((3, 1)),
            jnp.zeros(3),
            jnp.zeros((3, 1)),
            jnp.zeros(3),
            key=key,
            metric="kl",
            cap=0,
        )
    with pytest.raises(ValueError, match=r"shaped \(N, D\) and its log_weights"):
        cap_pair(
            jnp.zeros(3),
            jnp.zeros(3),
            jnp.zeros((3, 1)),
            jnp.zeros(3),
            key=key,
            metric="kl",
        )
    with pytest.raises(ValueError, match="same parameters, got 1 and 2"):
        cap_pair(
            jnp.zeros((3, 1)),
            jnp.zeros(3),
            jnp.zeros((3, 2)),
            jnp.zeros(3),
            key=key,
            metric="kl",
        )


def cap_samples_quietly(
    samples: jax.Array, key: jax.Array, cap: int
) -> tuple[jax.Array, jax.Array]:
    """Return cap_samples of equally weighted samples without its warning."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SubsamplingWarning)
        return cap_samples(
            samples, jnp.zeros(samples.shape[0]), key=key, metric="test", cap=cap
        )
