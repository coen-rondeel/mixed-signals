"""Tests for the conversions between JAX and NumPy, PyTorch or CuPy arrays."""

import inspect
import sys
import types
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mixed_signals.errors import MissingDependencyError, PrecisionWarning
from mixed_signals.io.arrays import call_in_chunks, from_jax, to_jax


class HostArray:
    """An array-like object with ``__array__`` but without DLPack methods."""

    def __array__(self, dtype: Any = None, copy: Any = None) -> np.ndarray:
        """Return the values as a NumPy array."""
        return np.arange(3.0)


class UnreadableArray:
    """A DLPack array on a device that JAX does not use, Apple Metal."""

    def __dlpack__(self, **kwargs: Any) -> Any:
        """Refuse the export, as a producer on an unknown device would."""
        raise BufferError("the array lies on a device that JAX does not use")

    def __dlpack_device__(self) -> tuple[int, int]:
        """Report the Metal device type of the DLPack specification."""
        return (8, 0)


# * NumPy and Python inputs


@pytest.mark.parametrize("dtype", [np.float64, np.float32, np.int64, np.int32, bool])
def test_numpy_round_trips_keep_values_shape_and_type(dtype: type) -> None:
    """NumPy to JAX and back keeps values, shape and type."""
    original: np.ndarray = (np.arange(24).reshape(2, 3, 4) % 5).astype(dtype)
    converted = to_jax(original)
    assert isinstance(converted, jax.Array)
    assert converted.dtype == original.dtype
    np.testing.assert_array_equal(from_jax(converted, "numpy"), original)


def test_numpy_inputs_are_copied() -> None:
    """JAX gets its own copy of a NumPy input, so later edits do not reach it."""
    original = np.zeros(4)
    converted = to_jax(original)
    original[0] = 1.0
    assert float(converted[0]) == 0.0


def test_non_native_byte_order_and_strides_convert() -> None:
    """Big-endian and transposed NumPy arrays convert to equal JAX arrays."""
    big_endian = np.arange(6.0).reshape(2, 3).astype(">f8")
    np.testing.assert_array_equal(to_jax(big_endian), big_endian)
    transposed = np.arange(6.0).reshape(2, 3).T
    np.testing.assert_array_equal(to_jax(transposed), transposed)


def test_python_sequences_scalars_and_array_likes_convert() -> None:
    """Nested lists, scalars and objects with __array__ become JAX arrays."""
    np.testing.assert_array_equal(to_jax([[1.0, 2.0], [3.0, 4.0]]), [[1, 2], [3, 4]])
    assert to_jax(2.5).shape == ()
    np.testing.assert_array_equal(to_jax(HostArray()), np.arange(3.0))


def test_jax_arrays_pass_through_unless_copied() -> None:
    """A JAX input is returned as it is, or as a new array with copy=True."""
    original = jnp.arange(3.0)
    assert to_jax(original) is original
    copied = to_jax(original, copy=True)
    assert copied is not original
    np.testing.assert_array_equal(copied, original)
    assert from_jax(original, "jax") is original
    assert from_jax(original, "jax", copy=True) is not original


def test_numpy_results_are_read_only_views_unless_copied() -> None:
    """A NumPy view of a JAX array is read-only; a copy is writable."""
    original = jnp.arange(3.0)
    view = from_jax(original, "numpy")
    assert not view.flags.writeable
    copied = from_jax(original, "numpy", copy=True)
    copied[0] = 7.0
    assert float(original[0]) == 0.0


def test_unreadable_dlpack_arrays_fail_with_advice() -> None:
    """An array that JAX cannot import fails with a message that says why."""
    with pytest.raises(ValueError, match="UnreadableArray through DLPack"):
        to_jax(UnreadableArray())


def test_unknown_backends_are_rejected() -> None:
    """Only NumPy, JAX, torch and CuPy are known backends.

    The test session checks the backend annotation first, so the test calls the
    unwrapped function, as runs without runtime type checks do.
    """
    unchecked_from_jax = inspect.unwrap(from_jax)
    with pytest.raises(ValueError, match="unknown array backend 'tensorflow'"):
        unchecked_from_jax(jnp.zeros(2), "tensorflow")


@pytest.mark.parametrize(
    ("backend", "extra"), [("torch", r"mixed-signals\[torch\]"), ("cupy", "cuda12")]
)
def test_missing_libraries_name_their_extra(
    backend: str, extra: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A missing torch or CuPy fails with the extra that installs it."""
    monkeypatch.setitem(sys.modules, backend, None)
    with pytest.raises(MissingDependencyError, match=extra):
        from_jax(jnp.zeros(2), backend)  # type: ignore[arg-type]


def test_cupy_results_come_from_its_dlpack_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The CuPy branch imports through cupy.from_dlpack and copies on request."""
    # * A NumPy stand-in for CuPy, which needs CUDA; the GPU suite runs the real one.
    stand_in = types.ModuleType("cupy")
    monkeypatch.setattr(stand_in, "from_dlpack", np.from_dlpack, raising=False)
    monkeypatch.setitem(sys.modules, "cupy", stand_in)
    original = jnp.arange(3.0)
    assert not from_jax(original, "cupy").flags.writeable
    copied = from_jax(original, "cupy", copy=True)
    copied[0] = 5.0
    assert float(original[0]) == 0.0


# * PyTorch, run where torch is installed


@pytest.mark.parametrize("dtype_name", ["float64", "float32", "int64", "bool"])
def test_torch_round_trips_keep_values_shape_and_type(dtype_name: str) -> None:
    """PyTorch to JAX and back keeps values, shape and type."""
    torch = pytest.importorskip("torch")
    original = (torch.arange(24).reshape(2, 3, 4) % 5).to(getattr(torch, dtype_name))
    converted = to_jax(original)
    assert str(converted.dtype) == dtype_name
    back = from_jax(converted, "torch")
    assert isinstance(back, torch.Tensor)
    assert back.dtype == original.dtype
    assert torch.equal(back, original)


def test_torch_tensors_share_memory_unless_copied() -> None:
    """Without copy, the JAX array and the tensor are one buffer."""
    torch = pytest.importorskip("torch")
    tensor = torch.zeros(4, dtype=torch.float64)
    shared = to_jax(tensor)
    independent = to_jax(tensor, copy=True)
    tensor[0] = 1.0
    assert float(shared[0]) == 1.0
    assert float(independent[0]) == 0.0
    copied_back = from_jax(independent, "torch", copy=True)
    copied_back[1] = 5.0
    assert float(independent[1]) == 0.0


def test_tensors_that_track_gradients_are_detached() -> None:
    """A tensor that requires gradients converts after detaching."""
    torch = pytest.importorskip("torch")
    tensor = torch.ones(3, dtype=torch.float64, requires_grad=True)
    np.testing.assert_array_equal(to_jax(tensor), np.ones(3))


def test_non_contiguous_tensors_convert() -> None:
    """Transposed tensors keep their values through DLPack."""
    torch = pytest.importorskip("torch")
    tensor = torch.arange(6, dtype=torch.float64).reshape(2, 3).T
    np.testing.assert_array_equal(to_jax(tensor), tensor.numpy())


# * Calling user code in chunks


class RowSum:
    """A function of the caller that records the type and size of every call."""

    def __init__(self) -> None:
        """Start without calls."""
        self.types: list[type] = []
        self.sizes: list[int] = []

    def __call__(self, points: Any) -> Any:
        """Return the row sums, as the array library of the points."""
        self.types.append(type(points))
        self.sizes.append(len(points))
        return points.sum(axis=-1)


def test_chunks_cover_every_row_in_order() -> None:
    """Ten rows in chunks of three give calls of 3, 3, 3 and 1 rows, order kept."""
    points = np.arange(30.0).reshape(10, 3)
    function = RowSum()
    values = call_in_chunks(function, points, chunk_size=3)
    assert function.sizes == [3, 3, 3, 1]
    np.testing.assert_array_equal(values, points.sum(axis=-1))
    assert values.dtype == jnp.float64


def test_without_a_chunk_size_one_call_gets_every_row() -> None:
    """The default sends all rows at once; a chunk larger than N does the same."""
    points = np.arange(12.0).reshape(6, 2)
    function = RowSum()
    call_in_chunks(function, points)
    call_in_chunks(function, points, chunk_size=100)
    assert function.sizes == [6, 6]


@pytest.mark.parametrize("backend", ["numpy", "jax"])
def test_functions_receive_arrays_of_the_backend(backend: str) -> None:
    """NumPy and JAX backends hand over arrays of their own type."""
    function = RowSum()
    values = call_in_chunks(
        function,
        jnp.arange(8.0).reshape(4, 2),
        backend=backend,  # type: ignore[arg-type]
        chunk_size=2,
    )
    expected_type = np.ndarray if backend == "numpy" else jax.Array
    assert all(issubclass(received, expected_type) for received in function.types)
    np.testing.assert_array_equal(values, [1.0, 5.0, 9.0, 13.0])


def test_torch_functions_receive_tensors() -> None:
    """The torch backend hands over tensors, and their results convert back."""
    torch = pytest.importorskip("torch")
    function = RowSum()
    values = call_in_chunks(function, np.arange(8.0).reshape(4, 2), backend="torch")
    assert function.types == [torch.Tensor]
    np.testing.assert_array_equal(values, [1.0, 5.0, 9.0, 13.0])


def test_results_of_any_real_type_come_back_as_float64() -> None:
    """Integer and boolean results cast silently to float64."""
    points = np.arange(6.0).reshape(3, 2)
    integers = call_in_chunks(lambda chunk: np.arange(len(chunk)), points)
    booleans = call_in_chunks(lambda chunk: np.ones(len(chunk), dtype=bool), points)
    assert integers.dtype == booleans.dtype == jnp.float64
    np.testing.assert_array_equal(integers, [0.0, 1.0, 2.0])
    np.testing.assert_array_equal(booleans, np.ones(3))


def test_a_narrow_float_result_warns_once_per_call() -> None:
    """Float32 results of several chunks give one warning and float64 values."""
    points = np.arange(10.0).reshape(5, 2)
    with pytest.warns(PrecisionWarning, match=r"results \(float32\)") as records:
        values = call_in_chunks(
            lambda chunk: chunk.sum(axis=-1).astype(np.float32), points, chunk_size=2
        )
    assert len(records) == 1
    assert values.dtype == jnp.float64
    np.testing.assert_array_equal(values, points.sum(axis=-1))


def test_results_must_have_one_value_per_point() -> None:
    """A result of another shape raises, naming the shapes."""
    points = np.arange(10.0).reshape(5, 2)
    with pytest.raises(ValueError, match=r"returned shape \(2, 2\) for a chunk of 2"):
        call_in_chunks(lambda chunk: np.zeros((len(chunk), 2)), points, chunk_size=2)
    with pytest.raises(ValueError, match=r"returned shape \(\) for a chunk of 5"):
        call_in_chunks(lambda chunk: np.float64(0.0), points)
    with pytest.raises(ValueError, match=r"returned shape \(4,\) for a chunk of 5"):
        call_in_chunks(lambda chunk: np.zeros(4), points)


def test_results_must_be_real() -> None:
    """Complex and string results raise TypeError."""
    points = np.arange(4.0).reshape(2, 2)
    with pytest.raises(TypeError, match="real numbers, got dtype complex128"):
        call_in_chunks(lambda chunk: np.zeros(len(chunk), dtype=complex), points)


def test_unknown_backends_and_bad_chunk_sizes_are_rejected() -> None:
    """The arguments are checked before the function is called.

    The test session checks the annotations first, so the test calls the
    unwrapped function, as runs without runtime type checks do.
    """
    unchecked = inspect.unwrap(call_in_chunks)
    points = np.zeros((3, 2))
    calls: list[int] = []

    def function(chunk: Any) -> Any:
        calls.append(len(chunk))
        return np.zeros(len(chunk))

    with pytest.raises(ValueError, match="unknown array backend 'tensorflow'"):
        unchecked(function, points, backend="tensorflow")
    with pytest.raises(ValueError, match="chunk_size must be at least 1, got 0"):
        unchecked(function, points, chunk_size=0)
    with pytest.raises(
        TypeError, match="chunk_size must be an integer or None, got str"
    ):
        unchecked(function, points, chunk_size="3")
    with pytest.raises(
        TypeError, match="chunk_size must be an integer or None, got bool"
    ):
        unchecked(function, points, chunk_size=True)
    assert calls == []


def test_at_least_one_point_is_needed() -> None:
    """An empty set of points raises instead of returning nothing."""
    with pytest.raises(ValueError, match="at least one point"):
        call_in_chunks(lambda chunk: np.zeros(len(chunk)), np.zeros((0, 2)))
