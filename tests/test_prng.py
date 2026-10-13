"""Tests for root keys, named key derivation and the text form of keys."""

import itertools

import jax
import numpy as np
import pytest
from jaxtyping import PRNGKeyArray

from mixed_signals import prng


def key_words(key: PRNGKeyArray) -> list[int]:
    """Return the key data as a list of 32-bit words."""
    return [int(word) for word in np.asarray(jax.random.key_data(key)).ravel()]


def test_root_key_is_threefry_of_the_seed() -> None:
    """The root key is the Threefry-2x32 key of the seed."""
    expected = jax.random.key(42, impl="threefry2x32")
    assert key_words(prng.root_key(42)) == key_words(expected)
    assert str(jax.random.key_impl(prng.root_key(42))) == "threefry2x32"


def test_root_key_ignores_the_process_default() -> None:
    """A different JAX default generator does not change the root key of a seed."""
    with jax.default_prng_impl("rbg"):
        root = prng.root_key(7)
    assert str(jax.random.key_impl(root)) == "threefry2x32"
    assert key_words(root) == key_words(jax.random.key(7, impl="threefry2x32"))


@pytest.mark.parametrize("seed", [-1, 2**63])
def test_root_key_rejects_seeds_outside_the_range(seed: int) -> None:
    """Seeds outside [0, 2**63) fail with the allowed range in the message."""
    with pytest.raises(ValueError, match=r"\[0, 2\*\*63\)"):
        prng.root_key(seed)


def test_root_key_rejects_bool_seeds() -> None:
    """A bool is not accepted as a seed, although Python treats it as an int."""
    with pytest.raises(TypeError, match="bool"):
        prng.root_key(True)


def test_derived_keys_are_reproducible() -> None:
    """The same parent key and path always give the same key."""
    first = prng.derive_key(prng.root_key(3), "compare", 0, "kl", "null")
    second = prng.derive_key(prng.root_key(3), "compare", 0, "kl", "null")
    assert key_words(first) == key_words(second)


def test_derived_keys_match_the_reference_values() -> None:
    """The derivation scheme gives fixed key data for a fixed seed and path."""
    # ! A change here changes every derived key: check the JAX changelog for fold_in.
    derived = prng.derive_key(prng.root_key(0), "compare")
    assert key_words(derived) == [3_545_282_203, 426_908_566]


def test_paths_compose() -> None:
    """Deriving in two steps equals deriving along the joined path."""
    root = prng.root_key(11)
    in_one_step = prng.derive_key(root, "compare", 2, "mmd")
    in_two_steps = prng.derive_key(prng.derive_key(root, "compare"), 2, "mmd")
    assert key_words(in_one_step) == key_words(in_two_steps)


def test_order_and_type_of_path_elements_matter() -> None:
    """Reordered paths, and an integer against its decimal string, differ."""
    root = prng.root_key(5)
    assert key_words(prng.derive_key(root, "a", "b")) != key_words(
        prng.derive_key(root, "b", "a")
    )
    assert key_words(prng.derive_key(root, 3)) != key_words(prng.derive_key(root, "3"))


def test_distinct_paths_give_distinct_keys() -> None:
    """A grid of 600 paths, as a run would use, gives 600 different keys."""
    root = prng.root_key(0)
    paths = itertools.product(range(10), ["kl", "mmd", "energy"], range(20))
    keys = {tuple(key_words(prng.derive_key(root, *path))) for path in paths}
    assert len(keys) == 600


def test_derived_keys_do_not_depend_on_other_calls() -> None:
    """A key depends on its own path only, not on keys derived before it."""
    root = prng.root_key(9)
    alone = prng.derive_key(root, "compare", "kl")
    for metric in ["mmd", "energy", "sinkhorn"]:
        prng.derive_key(root, "compare", metric)
    assert key_words(prng.derive_key(root, "compare", "kl")) == key_words(alone)


def test_legacy_keys_derive_like_typed_keys() -> None:
    """A legacy uint32 key gives the same derived data as the typed key."""
    legacy = jax.random.PRNGKey(42)
    derived_legacy = prng.derive_key(legacy, "null")
    derived_typed = prng.derive_key(prng.root_key(42), "null")
    assert [int(word) for word in np.asarray(derived_legacy)] == key_words(
        derived_typed
    )


def test_empty_path_is_rejected() -> None:
    """An empty path would hand out the parent key again."""
    with pytest.raises(ValueError, match="at least one element"):
        prng.derive_key(prng.root_key(0))


@pytest.mark.parametrize("element", [True, 1.5])
def test_path_elements_must_be_str_or_int(element: object) -> None:
    """Bools and floats are not path elements."""
    with pytest.raises(TypeError):
        prng.derive_key(prng.root_key(0), element)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "key",
    [
        prng.root_key(42),
        prng.derive_key(prng.root_key(42), "null"),
        jax.random.PRNGKey(42),
        jax.random.key(42, impl="rbg"),
    ],
)
def test_key_text_round_trips(key: PRNGKeyArray) -> None:
    """``parse_key`` rebuilds the key that ``format_key`` wrote."""
    text = prng.format_key(key)
    assert prng.KEY_TEXT_PATTERN.fullmatch(text)
    rebuilt = prng.parse_key(text)
    assert prng.format_key(rebuilt) == text


def test_key_text_names_the_implementation() -> None:
    """The text form starts with the generator name and lists the words in hex."""
    assert prng.format_key(prng.root_key(42)) == "threefry2x32:000000000000002a"


@pytest.mark.parametrize(
    "text",
    [
        "000000000000002a",
        "threefry2x32:2a",
        "threefry2x32:00000000000000000000002a",
        "unknown:000000000000002a",
        "Threefry2x32:000000000000002A",
    ],
)
def test_malformed_key_text_is_rejected(text: str) -> None:
    """Text that is not a formatted key, or does not fit its generator, fails."""
    with pytest.raises(ValueError, match="key"):
        prng.parse_key(text)
