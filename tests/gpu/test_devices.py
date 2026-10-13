"""GPU smoke tests, run by hand on a CUDA machine with ``pytest -m gpu``."""

import jax
import jax.numpy as jnp
import pytest

from mixed_signals import _config

pytestmark = pytest.mark.gpu


def test_jax_computes_float64_on_gpu() -> None:
    """JAX finds a GPU and keeps float64 precision on it."""
    assert _config.float64_enabled()
    gpu_device = jax.devices("gpu")[0]
    ones = jax.device_put(jnp.ones(3), gpu_device)
    total = jnp.sum(ones)
    assert total.dtype == jnp.float64
    assert gpu_device in total.devices()
    assert float(total) == 3.0
