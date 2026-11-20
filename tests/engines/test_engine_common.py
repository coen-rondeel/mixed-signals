"""Tests for the private helpers shared by the engines."""

import math
import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mixed_signals import prng, zoo
from mixed_signals.core.result import Provenance, hash_arrays
from mixed_signals.core.samples import SampleSet, kish_effective_sample_size
from mixed_signals.core.space import ParamSpace
from mixed_signals.engines._common import (
    Posterior,
    flat_space,
    gather_posteriors,
    null_moments,
    provenance,
    read_key,
    space_text,
    unique_names,
)
from mixed_signals.transforms.base import COMPOSITION_SEPARATOR, Compose
from mixed_signals.transforms.whiten import Whitening

CASE = zoo.shifted_gaussians(0.5, param_count=3)
SPACE = CASE.space
SAMPLE_COUNT = 40


def sample_set(seed: int, count: int = SAMPLE_COUNT, batch_size: int = 1) -> SampleSet:
    """Draw the two posteriors of the zoo case, in the given number of problems."""
    return CASE.sample(jax.random.key(seed), count, batch_size=batch_size)


# * The key of a call


def test_an_integer_seed_gives_its_root_key_and_is_reported() -> None:
    """An int becomes root_key(seed), and the seed is returned for the provenance."""
    key, seed = read_key(2027)
    assert seed == 2027
    np.testing.assert_array_equal(
        jax.random.key_data(key), jax.random.key_data(prng.root_key(2027))
    )
    assert read_key(0)[1] == 0


def test_a_typed_key_is_returned_unchanged_without_a_seed() -> None:
    """A typed key passes through, and no seed is reported."""
    typed = jax.random.key(5)
    key, seed = read_key(typed)
    assert key is typed
    assert seed is None


def test_a_legacy_key_is_returned_unchanged_without_a_seed() -> None:
    """A legacy uint32 key of shape (2,) is a key too."""
    legacy = jax.random.PRNGKey(5)
    assert legacy.shape == (2,)
    key, seed = read_key(legacy)
    assert key is legacy
    assert seed is None


@pytest.mark.parametrize("seed", [-1, 2**63, 2**70])
def test_seeds_outside_the_range_of_root_key_raise(seed: int) -> None:
    """root_key judges the range, and read_key lets its error through."""
    with pytest.raises(ValueError, match="the seed must lie in"):
        read_key(seed)


@pytest.mark.parametrize(
    ("key", "named"),
    [
        (True, "bool"),
        (2.0, "float"),
        ("7", "str"),
        (None, "NoneType"),
        (np.int64(3), "int64"),
        (np.zeros(2, dtype=np.uint32), "ndarray"),
        (jax.random.split(jax.random.key(0), 3), "key<fry> and shape (3,)"),
        (jnp.zeros(3, dtype=jnp.uint32), "dtype uint32 and shape (3,)"),
        (jnp.zeros(2, dtype=jnp.float64), "dtype float64 and shape (2,)"),
        (jnp.int32(3), "dtype int32 and shape ()"),
    ],
)
def test_anything_but_an_integer_or_a_key_raises_a_type_error(
    key: object, named: str
) -> None:
    """The message names the type, and the dtype and shape of an array."""
    with pytest.raises(
        TypeError, match="key must be a JAX PRNG key or an integer seed"
    ):
        read_key(key)
    with pytest.raises(TypeError, match=re.escape(named)):
        read_key(key)


# * The names of the posteriors


def test_unique_names_returns_a_tuple_of_the_names() -> None:
    """Any sequence of the right length of distinct strings passes."""
    assert unique_names("names", ["a", "b"], 2) == ("a", "b")
    assert unique_names("names", ("run",), 1) == ("run",)


@pytest.mark.parametrize(
    ("names", "count", "message"),
    [
        ("ab", 2, "names must be a sequence of strings, got str"),
        (3, 1, "names must be a sequence of strings, got int"),
        ({"a", "b"}, 2, "names must be a sequence of strings, got set"),
        (["a"], 2, "names must name each of the 2 posteriors, got 1 names"),
        (["a", "b", "c"], 2, "names must name each of the 2 posteriors, got 3 names"),
        (["a", 3], 2, r"names must hold strings, got \['a', 3\]"),
        (["a", "a"], 2, r"names must be unique, got \['a', 'a'\]"),
    ],
)
def test_unique_names_rejects_invalid_names(
    names: object, count: int, message: str
) -> None:
    """Non-sequences, strings, wrong counts, non-strings and repeats are rejected."""
    with pytest.raises(ValueError, match=message):
        unique_names("names", names, count)


def test_unique_names_names_its_argument() -> None:
    """The message names the argument that the engine received."""
    with pytest.raises(ValueError, match="target_names must be unique"):
        unique_names("target_names", ["x", "x"], 2)


# * The posteriors of one or several sets


def test_one_set_gives_one_posterior_per_axis_entry() -> None:
    """A set of two posteriors gives two named posteriors in axis order."""
    samples = sample_set(0, batch_size=3)
    posteriors = gather_posteriors(samples, ["first", "second"], SPACE)
    assert [posterior.name for posterior in posteriors] == ["first", "second"]
    for index, posterior in enumerate(posteriors):
        assert isinstance(posterior, Posterior)
        np.testing.assert_array_equal(posterior.samples, samples.samples[:, index])
        np.testing.assert_array_equal(
            posterior.log_weights, samples.log_weights[:, index]
        )
        assert posterior.samples.shape == (3, SAMPLE_COUNT, 3)
        assert posterior.log_weights.shape == (3, SAMPLE_COUNT)
        assert posterior.n_eff.shape == (3,)


def test_several_sets_join_in_order_with_their_own_sizes() -> None:
    """Sets with different sample counts and posterior counts join in order."""
    first = sample_set(1, 30, batch_size=2)
    second = sample_set(2, 50, batch_size=2).select(posteriors=[0])
    third = sample_set(3, 20, batch_size=2)
    posteriors = gather_posteriors([first, second, third], list("abcde"), SPACE)
    assert [posterior.name for posterior in posteriors] == list("abcde")
    assert [posterior.samples.shape[1] for posterior in posteriors] == [
        30,
        30,
        50,
        20,
        20,
    ]
    np.testing.assert_array_equal(posteriors[2].samples, second.samples[:, 0])
    np.testing.assert_array_equal(posteriors[3].samples, third.samples[:, 0])
    assert gather_posteriors((first,), ["a", "b"], SPACE)[1].name == "b"


def test_hashes_are_those_of_each_problem_of_the_input() -> None:
    """One digest per problem, of the samples and log weights before any transform."""
    samples = sample_set(4, batch_size=3)
    (_, second) = gather_posteriors(samples, ["a", "b"], SPACE)
    assert len(second.hashes) == 3
    for problem, digest in enumerate(second.hashes):
        assert digest == hash_arrays(
            samples.samples[problem, 1], samples.log_weights[problem, 1]
        )
    assert len(set(second.hashes)) == 3


def test_the_effective_sample_size_is_the_given_one_or_the_kish_value() -> None:
    """n_eff follows the set: given, or the Kish size of the weights."""
    base = sample_set(5)
    weighted = SampleSet(
        base.samples, log_weights=0.3 * jax.random.normal(jax.random.key(6), (1, 2, 40))
    )
    kish = kish_effective_sample_size(weighted.log_weights)
    posteriors = gather_posteriors(weighted, ["a", "b"], SPACE)
    for index, posterior in enumerate(posteriors):
        np.testing.assert_allclose(posterior.n_eff, kish[:, index])
    correlated = SampleSet(base.samples, n_eff=jnp.array([[11.0, 17.0]]))
    posteriors = gather_posteriors(correlated, ["a", "b"], SPACE)
    assert [float(posterior.n_eff[0]) for posterior in posteriors] == [11.0, 17.0]


@pytest.mark.parametrize("bad", [np.zeros((1, 2, 5, 3)), "samples", 3, None])
def test_inputs_that_are_not_sample_sets_raise_a_type_error(bad: object) -> None:
    """Arrays, strings, numbers and None are no SampleSet and no sequence of them."""
    with pytest.raises(
        TypeError, match="sample_sets must be a SampleSet or a sequence"
    ):
        gather_posteriors(bad, ["a", "b"], SPACE)


def test_a_sequence_must_hold_sample_sets_only() -> None:
    """Every entry is checked and the first offender is named."""
    with pytest.raises(TypeError, match="entry 1 is ndarray"):
        gather_posteriors([sample_set(0), np.zeros((1, 2, 5, 3))], list("abcd"), SPACE)


def test_an_empty_sequence_raises_a_value_error() -> None:
    """There is nothing to compare in an empty sequence."""
    with pytest.raises(ValueError, match="at least one SampleSet"):
        gather_posteriors([], [], SPACE)


@pytest.mark.parametrize(
    ("batch_size", "param_count", "message"),
    [
        (2, 3, r"\(B, D\) = \(1, 3\), but set 1 has \(B, D\) = \(2, 3\)"),
        (1, 4, r"\(B, D\) = \(1, 3\), but set 1 has \(B, D\) = \(1, 4\)"),
    ],
)
def test_sets_must_share_the_problem_and_parameter_counts(
    batch_size: int, param_count: int, message: str
) -> None:
    """A mismatched set raises ValueError naming both shapes."""
    other = zoo.shifted_gaussians(0.5, param_count=param_count).sample(
        jax.random.key(1), 5, batch_size=batch_size
    )
    with pytest.raises(ValueError, match=message):
        gather_posteriors([sample_set(0), other], list("abcd"), SPACE)


def test_the_space_must_fit_the_parameter_count() -> None:
    """A space of another size raises ValueError, naming both counts."""
    two_parameters = zoo.shifted_gaussians(0.5, param_count=2).space
    with pytest.raises(ValueError, match="the space has 2 parameters, but the samples"):
        gather_posteriors(sample_set(0), ["a", "b"], two_parameters)


def test_the_names_must_fit_the_posteriors() -> None:
    """The count of the names follows the posteriors of every set together."""
    with pytest.raises(ValueError, match="names must name each of the 2 posteriors"):
        gather_posteriors(sample_set(0), ["a"], SPACE)
    with pytest.raises(ValueError, match="names must name each of the 4 posteriors"):
        gather_posteriors([sample_set(0), sample_set(1)], ["a", "b"], SPACE)
    with pytest.raises(ValueError, match="names must be unique"):
        gather_posteriors(sample_set(0), ["a", "a"], SPACE)


# * The transform of the posteriors


def whitening_of(samples: SampleSet) -> Whitening:
    """Return the reference whitening of the first posterior of the first problem."""
    return Whitening.from_samples(samples.samples[0, :1], None, SPACE)


def test_a_transform_maps_the_arrays_but_not_the_hashes() -> None:
    """Samples come from the mapped set; the hashes still read the input."""
    base = sample_set(7, batch_size=2)
    samples = SampleSet(base.samples, n_eff=jnp.full((2, 2), 12.0))
    transform = whitening_of(samples)
    plain = gather_posteriors(samples, ["a", "b"], SPACE)
    mapped = gather_posteriors(samples, ["a", "b"], SPACE, transform=transform)
    for original, moved in zip(plain, mapped, strict=True):
        assert moved.name == original.name
        assert moved.hashes == original.hashes
        np.testing.assert_allclose(
            moved.samples, transform.forward(original.samples), rtol=1e-13
        )
        assert not np.allclose(moved.samples, original.samples)
        np.testing.assert_array_equal(moved.log_weights, original.log_weights)
        np.testing.assert_array_equal(moved.n_eff, original.n_eff)
    assert [float(posterior.n_eff[0]) for posterior in mapped] == [12.0, 12.0]


def test_a_transform_keeps_the_kish_size_of_weighted_sets() -> None:
    """Without a given n_eff, the mapped set has the Kish size of the weights."""
    weights = 0.3 * jax.random.normal(jax.random.key(8), (1, 2, SAMPLE_COUNT))
    samples = SampleSet(sample_set(9).samples, log_weights=weights)
    mapped = gather_posteriors(
        samples, ["a", "b"], SPACE, transform=whitening_of(samples)
    )
    kish = kish_effective_sample_size(weights)
    for index, posterior in enumerate(mapped):
        np.testing.assert_allclose(posterior.n_eff, kish[:, index])
        np.testing.assert_array_equal(posterior.log_weights, weights[:, index])


def test_a_transform_applies_to_every_set_and_composes() -> None:
    """Each set of a sequence is mapped, and a composed transform acts in turn."""
    first, second = sample_set(10, 30), sample_set(11, 50)
    transform = Compose([whitening_of(first), whitening_of(second)])
    posteriors = gather_posteriors(
        [first, second], list("abcd"), SPACE, transform=transform
    )
    expected = transform.forward(second.samples[:, 1])
    np.testing.assert_allclose(posteriors[3].samples, expected, rtol=1e-12)
    assert posteriors[3].samples.shape == (1, 50, 3)
    plain = gather_posteriors([first, second], list("abcd"), SPACE)
    assert [posterior.hashes for posterior in posteriors] == [
        posterior.hashes for posterior in plain
    ]


# * The flat space, the space text and the provenance


def test_the_flat_space_drops_the_sphere_declarations_only() -> None:
    """The parameters stay, with their topologies and bounds; the spheres go."""
    case = zoo.von_mises_fisher(0.5, euclidean_count=1)
    assert len(case.space.spheres) == 1
    flat = flat_space(case.space)
    assert isinstance(flat, ParamSpace)
    assert flat.spheres == ()
    assert flat.parameters == case.space.parameters
    assert flat.names == case.space.names
    assert flat.axis_blocks.spheres == ()
    assert len(flat.axis_blocks.circles) == 1
    assert flat_space(SPACE) == SPACE


def test_space_text_joins_descriptions_with_the_composition_separator() -> None:
    """Descriptions join as the space column of a row writes a chain of transforms."""
    assert space_text("a", "b") == f"a{COMPOSITION_SEPARATOR}b"
    assert space_text("identity") == "identity"
    assert space_text("log | whiten(pooled)", "x") == "log | whiten(pooled) | x"


def test_the_provenance_records_the_seed_the_key_and_the_hashes() -> None:
    """The row provenance holds the seed, the formatted key, hashes and config hash."""
    key = prng.root_key(9)
    hashes = [hash_arrays(np.zeros(3)), hash_arrays(np.ones(3))]
    config_hash = hash_arrays(np.arange(4))
    record = provenance(9, key, hashes, config_hash)
    assert isinstance(record, Provenance)
    assert record.seed == 9
    assert record.prng_key == prng.format_key(key)
    assert record.input_hashes == tuple(hashes)
    assert record.config_hash == config_hash
    without = provenance(None, key, hashes, None)
    assert without.seed is None
    assert without.config_hash is None
    assert without.input_hashes == tuple(hashes)


def test_null_moments_ignore_draws_that_are_not_finite() -> None:
    """Mean and ddof = 1 deviation of the finite draws; NaN where they are too few."""
    mean, deviation = null_moments(jnp.array([1.0, 2.0, jnp.nan, 4.0, jnp.inf]))
    assert mean == pytest.approx(7 / 3)
    assert deviation == pytest.approx(np.std([1.0, 2.0, 4.0], ddof=1))
    one_mean, one_deviation = null_moments(jnp.array([3.0, jnp.nan]))
    assert one_mean == 3.0
    assert math.isnan(one_deviation)
    none_mean, none_deviation = null_moments(jnp.array([jnp.nan, jnp.inf]))
    assert math.isnan(none_mean)
    assert math.isnan(none_deviation)
