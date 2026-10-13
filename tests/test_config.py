"""Tests for the process-wide JAX settings and the package defaults."""

import jax
import jax.numpy as jnp
import pytest
from jax.scipy.special import ndtri

from mixed_signals import _config


def test_import_enables_float64() -> None:
    """Importing the package makes float64 the default JAX float type."""
    assert _config.float64_enabled()
    assert jnp.zeros(()).dtype == jnp.float64


def test_default_null_draws_resolve_the_documented_significance() -> None:
    """200 draws floor the p-value at 1/201, a one-sided significance of 2.58σ."""
    smallest_p_value = 1 / (_config.NULL_DRAWS + 1)
    assert float(ndtri(1 - smallest_p_value)) == pytest.approx(2.58, abs=0.005)


def test_root_keys_use_the_jax_default_generator() -> None:
    """The pinned generator is the one JAX uses by default, as documented."""
    assert jax.config.jax_default_prng_impl == _config.PRNG_IMPLEMENTATION
