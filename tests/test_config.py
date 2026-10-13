"""Tests for the process-wide JAX settings."""

import jax.numpy as jnp

from mixed_signals import _config


def test_import_enables_float64() -> None:
    """Importing the package makes float64 the default JAX float type."""
    assert _config.float64_enabled()
    assert jnp.zeros(()).dtype == jnp.float64
