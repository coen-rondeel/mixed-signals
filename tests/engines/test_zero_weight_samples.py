"""Tests that zero-weight samples, such as padding, change no row of an engine."""

from typing import Any

import jax
import jax.numpy as jnp
import pytest
from polars.testing import assert_frame_equal

from mixed_signals import zoo
from mixed_signals.core.result import Result
from mixed_signals.core.samples import SampleSet
from mixed_signals.engines._common import _kept, gather_posteriors, problem_samples
from mixed_signals.engines.compare import DEFAULT_METRICS, compare
from mixed_signals.engines.cost import (
    estimate_compare_cost,
    estimate_diagnose_cost,
)
from mixed_signals.engines.diagnose import diagnose

CASE = zoo.shifted_gaussians(0.8, param_count=2)
NAMES = ["reference", "other"]
KEY = 2027
SAMPLE_COUNT = 70
PADDING = 9
DRAWS = 20
SMALL_CAP = 40
# * The fast set plus the rows that read rejection subsets, and the Sinkhorn divergence.
METRICS = [*DEFAULT_METRICS, "kl:knn", "js:knn", "sinkhorn"]


def draw(seed: int, count: int = SAMPLE_COUNT, batch_size: int = 2) -> SampleSet:
    """Draw the two posteriors of the zoo case in a few problems."""
    return CASE.sample(jax.random.key(seed), count, batch_size=batch_size)


def pad(
    sample_set: SampleSet, extra: int = PADDING, *, front: bool = False
) -> SampleSet:
    """Add zero-weight copies of the first sample of every problem and posterior."""
    copies = jnp.repeat(sample_set.samples[:, :, :1], extra, axis=2)
    no_weight = jnp.full((*sample_set.log_weights.shape[:2], extra), -jnp.inf)
    pieces = (
        ((copies, sample_set.samples), (no_weight, sample_set.log_weights))
        if front
        else ((sample_set.samples, copies), (sample_set.log_weights, no_weight))
    )
    return SampleSet(
        jnp.concatenate(pieces[0], axis=2),
        log_weights=jnp.concatenate(pieces[1], axis=2),
    )


def assert_same_rows(plain: Result, padded: Result) -> None:
    """Check that two result tables agree bit for bit, provenance included."""
    assert_frame_equal(plain.table, padded.table, check_exact=True)


@pytest.mark.parametrize("front", [False, True])
def test_padding_changes_no_row_in_mode_b(front: bool) -> None:
    """Zero-weight copies, after or before the samples, leave every row identical."""
    plain = draw(1)
    keywords: dict[str, Any] = {"draws": DRAWS, "cap": SMALL_CAP, "metrics": METRICS}
    rows = compare(plain, NAMES, CASE.space, key=KEY, **keywords)
    padded_rows = compare(
        pad(plain, front=front), NAMES, CASE.space, key=KEY, **keywords
    )
    assert len(rows) > 20
    assert_same_rows(rows, padded_rows)


def test_padding_changes_no_row_in_mode_a_with_the_split_half_null() -> None:
    """The reference whitening and the split-half null read the kept samples only."""
    plain = draw(2)
    keywords: dict[str, Any] = {
        "reference": "reference",
        "null": "split_half",
        "draws": DRAWS,
        "cap": SMALL_CAP,
        "metrics": METRICS,
    }
    rows = compare(plain, NAMES, CASE.space, key=KEY, **keywords)
    assert_same_rows(rows, compare(pad(plain), NAMES, CASE.space, key=KEY, **keywords))


def test_padding_changes_no_row_in_mode_a_with_the_permutation_null() -> None:
    """The shared whitening of the reference and the permutation null agree too."""
    plain = draw(3)
    keywords: dict[str, Any] = {
        "reference": "reference",
        "draws": DRAWS,
        "cap": SMALL_CAP,
        "metrics": METRICS,
    }
    rows = compare(plain, NAMES, CASE.space, key=KEY, **keywords)
    assert_same_rows(rows, compare(pad(plain), NAMES, CASE.space, key=KEY, **keywords))


def test_padding_changes_no_row_of_the_subsampling_bootstrap() -> None:
    """The half-sample errors draw from the kept samples, so padding leaves them."""
    plain = draw(4)
    keywords: dict[str, Any] = {
        "uncertainty": "subsampling_bootstrap",
        "half_samples": 4,
        "draws": DRAWS,
        "cap": SMALL_CAP,
        "metrics": METRICS,
    }
    rows = compare(plain, NAMES, CASE.space, key=KEY, **keywords)
    assert_same_rows(rows, compare(pad(plain), NAMES, CASE.space, key=KEY, **keywords))


def test_padding_changes_no_row_for_unequal_numbers_of_samples() -> None:
    """Two sets of different N, each padded on its own, keep their rows."""
    first = draw(5, SAMPLE_COUNT).select(posteriors=0)
    second = draw(6, SAMPLE_COUNT - 20).select(posteriors=1)
    keywords: dict[str, Any] = {"draws": DRAWS, "cap": SMALL_CAP, "metrics": METRICS}
    rows = compare([first, second], NAMES, CASE.space, key=KEY, **keywords)
    padded_rows = compare(
        [pad(first, 5), pad(second, 17)], NAMES, CASE.space, key=KEY, **keywords
    )
    assert_same_rows(rows, padded_rows)


def test_padding_changes_no_row_of_weighted_posteriors() -> None:
    """Weighted samples keep their rows, with the Kish size of the kept weights."""
    plain = draw(7)
    weights = -0.01 * jnp.arange(SAMPLE_COUNT) * jnp.ones((2, 2, 1))
    weighted = SampleSet(plain.samples, log_weights=weights)
    padded = pad(weighted)
    keywords: dict[str, Any] = {"draws": DRAWS, "cap": SMALL_CAP, "metrics": METRICS}
    rows = compare(weighted, NAMES, CASE.space, key=KEY, **keywords)
    assert_same_rows(rows, compare(padded, NAMES, CASE.space, key=KEY, **keywords))


def test_padding_changes_no_row_of_diagnose() -> None:
    """Every diagnostic row of flat sets is bit-identical with padding."""
    plain = draw(8)
    keywords: dict[str, Any] = {"draws": DRAWS, "cap": SMALL_CAP, "pair_cap": SMALL_CAP}
    rows = diagnose(plain, NAMES, CASE.space, key=KEY, **keywords).result
    padded_rows = diagnose(pad(plain), NAMES, CASE.space, key=KEY, **keywords).result
    assert len(rows) > 10
    assert_same_rows(rows, padded_rows)


def ragged_set(plain: SampleSet, *, front: bool) -> SampleSet:
    """Keep problem 0 whole and pad problem 1 to the same length from fewer samples."""
    kept_count = plain.shape.sample - PADDING
    kept = plain.samples[1:, :, :kept_count]
    copies = jnp.repeat(kept[:, :, :1], PADDING, axis=2)
    pieces = (copies, kept) if front else (kept, copies)
    no_weight = jnp.full((1, 2, PADDING), -jnp.inf)
    weights = (no_weight, jnp.zeros((1, 2, kept_count)))
    second = SampleSet(
        jnp.concatenate(pieces, axis=2),
        log_weights=jnp.concatenate(weights if front else weights[::-1], axis=2),
    )
    return SampleSet(
        jnp.concatenate([plain.samples[:1], second.samples]),
        log_weights=jnp.concatenate([plain.log_weights[:1], second.log_weights]),
    )


def test_problems_with_different_numbers_of_kept_samples_are_read_one_by_one() -> None:
    """Padding that differs between problems, and moves, leaves every row as it is."""
    plain = draw(9, SAMPLE_COUNT + PADDING)
    keywords: dict[str, Any] = {
        "draws": DRAWS,
        "cap": SMALL_CAP,
        "metrics": ["mean_shift", "energy"],
    }
    rows = compare(
        ragged_set(plain, front=False), NAMES, CASE.space, key=KEY, **keywords
    )
    moved = compare(
        ragged_set(plain, front=True), NAMES, CASE.space, key=KEY, **keywords
    )
    assert_same_rows(rows, moved)


def test_the_dry_runs_count_the_samples_with_a_weight() -> None:
    """The estimates size the statistics by the kept samples, padding excluded."""
    plain = draw(10)
    keywords: dict[str, Any] = {
        "draws": DRAWS,
        "cap": 10_000,
        "metrics": ["mean_shift", "energy"],
    }
    estimate = estimate_compare_cost(plain, NAMES, CASE.space, **keywords)
    padded = estimate_compare_cost(pad(plain), NAMES, CASE.space, **keywords)
    assert_frame_equal(estimate.table, padded.table)
    diagnosis = estimate_diagnose_cost(plain, NAMES, CASE.space, draws=DRAWS)
    padded_diagnosis = estimate_diagnose_cost(
        pad(plain), NAMES, CASE.space, draws=DRAWS
    )
    assert_frame_equal(diagnosis.table, padded_diagnosis.table)


def test_the_kept_samples_of_a_problem_drop_the_zero_weights() -> None:
    """The helper returns the samples with a finite log weight, in their order."""
    plain = draw(11, count=12, batch_size=1)
    padded = pad(plain, 3, front=True)
    (posterior, _) = gather_posteriors(padded, NAMES, CASE.space)
    samples, log_weights = problem_samples(posterior, 0)
    assert samples.shape == (12, 2)
    assert log_weights.shape == (12,)
    assert bool(jnp.all(samples == plain.samples[0, 0]))
    assert bool(jnp.all(log_weights == 0.0))


def test_without_zero_weights_the_kept_samples_are_the_given_arrays() -> None:
    """With no zero weight the helper returns the very arrays that it was given."""
    samples = jnp.arange(24.0).reshape(12, 2)
    log_weights = jnp.zeros(12)
    kept_samples, kept_log_weights = _kept(samples, log_weights)
    assert kept_samples is samples
    assert kept_log_weights is log_weights
    assert kept_log_weights.shape == (12,)


def test_ragged_padding_gives_each_posterior_its_own_kish_size() -> None:
    """Posteriors with different kept counts get the Kish size of their kept weights."""
    plain = draw(13, count=20, batch_size=2)
    log_weights = plain.log_weights.at[0, 0, -5:].set(-jnp.inf)
    ragged = SampleSet(plain.samples, log_weights=log_weights)
    first, second = gather_posteriors(ragged, NAMES, CASE.space)
    assert float(first.n_eff[0]) == pytest.approx(15.0)
    assert float(first.n_eff[1]) == pytest.approx(20.0)
    assert float(second.n_eff[0]) == pytest.approx(20.0)
    assert len(first.hashes) == len(second.hashes) == 2
    assert first.hashes[0] != first.hashes[1]
