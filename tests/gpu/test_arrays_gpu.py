"""GPU tests of zero-copy DLPack exchange, run by hand with ``pytest -m gpu``.

They need JAX with CUDA support. The CuPy tests need the ``cuda12`` or
``cuda13`` extra, and the torch tests a CUDA build of torch; development
environments install the CPU build, so those tests skip there.
"""

from typing import Any

import jax
import numpy as np
import pytest

from mixed_signals.core.samples import SampleSet
from mixed_signals.io.arrays import from_jax, to_jax

pytestmark = pytest.mark.gpu


def cuda_torch() -> Any:
    """Return torch if it can use a CUDA device, otherwise skip the test."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("torch has no CUDA device; install a CUDA build of torch")
    return torch


def test_cupy_arrays_reach_jax_without_a_copy() -> None:
    """A CuPy array and its JAX import share one GPU buffer, both ways."""
    cupy = pytest.importorskip("cupy")
    original = cupy.arange(6, dtype=cupy.float64)
    imported = to_jax(original)
    assert imported.devices().pop().platform == "gpu"
    assert imported.unsafe_buffer_pointer() == original.data.ptr
    exported = from_jax(imported, "cupy")
    assert exported.data.ptr == original.data.ptr


def test_cupy_copies_are_independent() -> None:
    """copy=True gives a separate GPU buffer."""
    cupy = pytest.importorskip("cupy")
    original = cupy.zeros(4, dtype=cupy.float64)
    copied = to_jax(original, copy=True)
    original[0] = 1.0
    assert float(copied[0]) == 0.0


def test_cuda_tensors_reach_jax_without_a_copy() -> None:
    """A CUDA tensor and its JAX import share one GPU buffer, both ways."""
    torch = cuda_torch()
    original = torch.arange(6, dtype=torch.float64, device="cuda")
    imported = to_jax(original)
    assert imported.devices().pop().platform == "gpu"
    assert imported.unsafe_buffer_pointer() == original.data_ptr()
    assert from_jax(imported, "torch").data_ptr() == original.data_ptr()


def test_sample_sets_from_cupy_stay_on_the_gpu() -> None:
    """A set built from CuPy arrays keeps its fields on the GPU."""
    cupy = pytest.importorskip("cupy")
    draws = cupy.asarray(np.random.default_rng(3).normal(size=(2, 100, 3)))
    sample_set = SampleSet(draws, log_weights=cupy.zeros((2, 100)))
    assert sample_set.samples.devices() == {jax.devices("gpu")[0]}
    np.testing.assert_array_equal(np.asarray(sample_set.samples[0]), draws.get())
