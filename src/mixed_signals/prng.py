"""PRNG keys: one config seed gives every stochastic call its own JAX key.

Every stochastic function takes an explicit JAX key. A run starts from one root
key, :func:`root_key` of the config seed, and derives the key of each call from
a path of names with :func:`derive_key`::

    root = root_key(42)
    null_key = derive_key(root, "compare", 3, "kl", "knn", "sbi|mcmc", "null")
    draw_keys = jax.random.split(null_key, 200)

Each path element is hashed with SHA-256 together with its type, and the first
eight bytes of the digest enter the key as two 32-bit words through
``jax.random.fold_in``. A derived key therefore depends only on the root key
and its own path, not on which other calls run or in which order: adding a
metric to a config leaves the null draws of every other row unchanged. Paths
compose, so ``derive_key(derive_key(root, "a"), "b")`` equals
``derive_key(root, "a", "b")``, and an engine can hand a derived key down to
the functions it calls.

JAX combines the counter-based Threefry generator (Salmon et al. 2011) with a
functional, splittable key model (Claessen and Pałka 2013), in which split and
folded keys give independent streams. Root keys always use Threefry-2x32,
whatever the process default. For a fixed key, the values that JAX samplers
return may change between JAX releases, so result provenance records the JAX
version next to the key.

References:
    :cite:t:`salmon2011parallel`; :cite:t:`claessen2013splittable`;
    :cite:t:`jax2019prng`; :cite:t:`jax2022api`; :cite:t:`nist2015secure`.
"""

import hashlib
import re
from typing import Final

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import PRNGKeyArray

from mixed_signals._config import PRNG_IMPLEMENTATION

SEED_LIMIT: Final = 2**63
"""Seeds are integers in ``[0, SEED_LIMIT)``, the non-negative signed 64-bit range."""

KEY_TEXT_PATTERN: Final = re.compile(
    r"(?P<implementation>[a-z0-9_]+):(?P<words>(?:[0-9a-f]{8})+)"
)
"""Pattern of the text that :func:`format_key` writes, a name and hex words."""

# * A path element enters the key as the first two 32-bit words of its SHA-256 digest.
_WORD_BYTES: Final = 4
_WORDS_PER_ELEMENT: Final = 2
_HEX_DIGITS_PER_WORD: Final = 2 * _WORD_BYTES


def root_key(seed: int) -> PRNGKeyArray:
    """Return the root key of a seed.

    Args:
        seed: Non-negative integer below :data:`SEED_LIMIT`.

    Returns:
        A typed Threefry-2x32 key.

    Raises:
        TypeError: If the seed is not an integer.
        ValueError: If the seed lies outside ``[0, SEED_LIMIT)``.

    References:
        :cite:t:`salmon2011parallel`; :cite:t:`jax2019prng`.
    """
    if isinstance(seed, bool):
        raise TypeError("the seed must be an integer, not a bool")
    if not 0 <= seed < SEED_LIMIT:
        raise ValueError(f"the seed must lie in [0, 2**63), got {seed}")
    return jax.random.key(seed, impl=PRNG_IMPLEMENTATION)


def derive_key(key: PRNGKeyArray, *path: str | int) -> PRNGKeyArray:
    """Derive the key of one stochastic call from a parent key and a path.

    Each element is hashed with SHA-256 as ``"str:<text>"`` or ``"int:<value>"``,
    so ``3`` and ``"3"`` give different keys, and the first eight digest bytes
    are folded into the key as two big-endian 32-bit words.

    Args:
        key: Parent key, typed or a legacy ``uint32`` key.
        *path: One or more names or integers that identify the call.

    Returns:
        The derived key, of the same kind as the parent key.

    Raises:
        TypeError: If a path element is not a string or an integer.
        ValueError: If the path is empty.

    References:
        :cite:t:`claessen2013splittable`; :cite:t:`nist2015secure`.
    """
    if not path:
        raise ValueError(
            "the path needs at least one element, so that no call reuses its parent key"
        )
    derived_key = key
    for element in path:
        for word in _element_words(element):
            derived_key = jax.random.fold_in(derived_key, word)
    return derived_key


def format_key(key: PRNGKeyArray) -> str:
    """Write a key as text, for example ``"threefry2x32:000000000000002a"``.

    Args:
        key: Typed or legacy ``uint32`` key.

    Returns:
        The implementation name, a colon and the key data as 8-digit hex words.
    """
    implementation = str(jax.random.key_impl(key))
    key_data = jax.random.key_data(key) if _is_typed_key(key) else key
    words = np.asarray(key_data, dtype=np.uint32).ravel()
    return implementation + ":" + "".join(f"{int(word):08x}" for word in words)


def parse_key(text: str) -> PRNGKeyArray:
    """Rebuild a typed key from the text that :func:`format_key` writes.

    Args:
        text: Implementation name, a colon and the key data as 8-digit hex words.

    Returns:
        The typed key.

    Raises:
        ValueError: If the text does not have the format of :func:`format_key`,
            or its words do not fit the implementation.
    """
    text_match = KEY_TEXT_PATTERN.fullmatch(text)
    if text_match is None:
        raise ValueError(
            f"{text!r} is not a formatted key; expected '<implementation>:<hex words>',"
            " for example 'threefry2x32:000000000000002a'"
        )
    hex_words = text_match["words"]
    words = np.array(
        [
            int(hex_words[start : start + _HEX_DIGITS_PER_WORD], 16)
            for start in range(0, len(hex_words), _HEX_DIGITS_PER_WORD)
        ],
        dtype=np.uint32,
    )
    try:
        key: PRNGKeyArray = jax.random.wrap_key_data(
            jnp.asarray(words), impl=text_match["implementation"]
        )
    except (TypeError, ValueError) as error:
        raise ValueError(f"{text!r} does not give a valid key: {error}") from None
    return key


def _element_words(element: str | int) -> tuple[int, ...]:
    if isinstance(element, bool) or not isinstance(element, str | int):
        raise TypeError(
            f"path elements must be str or int, got {type(element).__name__}"
        )
    tag = "str" if isinstance(element, str) else "int"
    digest = hashlib.sha256(f"{tag}:{element}".encode()).digest()
    return tuple(
        int.from_bytes(digest[start : start + _WORD_BYTES], "big")
        for start in range(0, _WORDS_PER_ELEMENT * _WORD_BYTES, _WORD_BYTES)
    )


def _is_typed_key(key: PRNGKeyArray) -> bool:
    return bool(jnp.issubdtype(key.dtype, jax.dtypes.prng_key))
