"""Conversion between JAX arrays and NumPy, CuPy or PyTorch arrays.

JAX reads CuPy and PyTorch arrays through DLPack, the data-interchange protocol
of the Python array API standard, which shares memory where possible and copies
only where needed (Consortium for Python Data API Standards 2021). A CuPy array
or a CUDA tensor therefore reaches JAX on the same GPU without a copy, and a JAX
array reaches torch or CuPy in the same way. NumPy arrays, Python sequences and
scalars are copied into JAX, as :func:`jax.numpy.asarray` does.

A shared array is one memory buffer under two names. JAX assumes that its
arrays never change, so do not modify a shared torch or CuPy array in place
while the JAX array is in use; pass ``copy=True`` when the two must be
independent.

The result of a conversion lives on the device of its input. JAX cannot read
arrays on devices that it does not use, such as Apple MPS tensors, or CUDA
arrays in a process whose JAX has no CUDA support; move those to the host
first.

User code
    :func:`call_in_chunks` calls a function of the caller, such as a
    log-likelihood, a prior or a signal whitener, at this boundary and outside
    ``jax.jit``: it hands the function arrays of the chosen library in chunks
    of rows, checks what comes back and returns one JAX array of float64, with
    one value per row or an array of a fixed shape per row.

torch and CuPy are imported inside :func:`from_jax`, so the package imports
without them. The ``torch`` extra installs torch, and the ``cuda12`` and
``cuda13`` extras install CuPy together with the matching JAX CUDA plugin.

References:
    :cite:t:`dataapis2021python`.
"""

import importlib
import os
import warnings
from collections.abc import Callable
from types import ModuleType
from typing import Any, Final, Literal, get_args

import jax
import jax.dlpack
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike
from jaxtyping import Array, Float

from mixed_signals.errors import MissingDependencyError, PrecisionWarning

type ArrayBackend = Literal["numpy", "jax", "torch", "cupy"]
"""Array libraries that :func:`from_jax` converts to."""

# * Frames in the package are skipped, so a warning points at the caller's line.
_PACKAGE_DIRECTORY: Final = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_INSTALL_HINTS = {
    "torch": "install it with the torch extra: pip install 'mixed-signals[torch]'",
    "cupy": (
        "install it with the CUDA extra that matches the machine:"
        " pip install 'mixed-signals[cuda12]' or 'mixed-signals[cuda13]'"
    ),
}


def to_jax(array: object, *, copy: bool = False) -> Array:
    """Convert an array to a JAX array with the same values, shape and type.

    Args:
        array: A JAX, NumPy, CuPy or PyTorch array, another object with the
            DLPack methods, or a Python scalar or nested sequence. A torch
            tensor that tracks gradients is detached first.
        copy: If true, always return a copy. Otherwise a JAX array is returned
            unchanged, and a DLPack array shares memory with the result where
            possible. NumPy arrays and Python sequences are copied either way.

    Returns:
        The JAX array, on the device of the input.

    Raises:
        ValueError: If JAX cannot import the array through DLPack, for example
            because the array lies on a device that JAX does not use.

    References:
        :cite:t:`dataapis2021python`.
    """
    if isinstance(array, jax.Array):
        return jnp.array(array, copy=True) if copy else array
    if isinstance(array, np.ndarray | np.generic):
        # ! JAX rejects non-native byte order, which FITS data use; swap it first.
        if not array.dtype.isnative:
            array = array.astype(array.dtype.newbyteorder("="))
        return jnp.asarray(array)
    if not hasattr(array, "__dlpack__"):
        return jnp.asarray(array)
    # * A torch tensor that tracks gradients refuses DLPack export.
    # * Detaching it keeps its memory.
    detach = getattr(array, "detach", None)
    if getattr(array, "requires_grad", False) is True and callable(detach):
        array = detach()
    try:
        imported: Array = jax.dlpack.from_dlpack(array)
    except (BufferError, RuntimeError, TypeError, ValueError) as error:
        raise ValueError(
            f"JAX cannot import this {type(array).__name__} through DLPack: {error}."
            " Arrays on a device that JAX does not use, such as Apple MPS, must"
            " move to the host first"
        ) from error
    # ! JAX 0.11 shares CPU memory even when from_dlpack gets copy=True, so copy here.
    return jnp.array(imported, copy=True) if copy else imported


def from_jax(array: Array, backend: ArrayBackend, *, copy: bool = False) -> Any:
    """Convert a JAX array to a NumPy, CuPy or PyTorch array, or keep it in JAX.

    Args:
        array: The JAX array.
        backend: ``"numpy"``, ``"jax"``, ``"torch"`` or ``"cupy"``.
        copy: If true, return an independent copy. Otherwise the result shares
            memory with ``array`` where possible: a NumPy result of an array on
            the host is a read-only view, and a torch or CuPy result is a
            writable view that must not be modified in place.

    Returns:
        The array in the requested library, on the device of ``array``. A NumPy
        result always lives on the host.

    Raises:
        ValueError: If the backend is unknown.
        MissingDependencyError: If torch or CuPy is not installed.

    References:
        :cite:t:`dataapis2021python`.
    """
    known_backends = get_args(ArrayBackend.__value__)
    if backend not in known_backends:
        raise ValueError(
            f"unknown array backend {backend!r}; choose one of {known_backends}"
        )
    if backend == "jax":
        return jnp.array(array, copy=True) if copy else array
    if backend == "numpy":
        return np.array(array) if copy else np.asarray(array)
    if backend == "torch":
        tensor = _import_backend("torch").from_dlpack(array)
        return tensor.clone() if copy else tensor
    cupy_array = _import_backend("cupy").from_dlpack(array)
    return cupy_array.copy() if copy else cupy_array


def call_in_chunks(
    function: Callable[[Any], object],
    points: Float[ArrayLike, "sample param"],
    *,
    backend: ArrayBackend = "numpy",
    chunk_size: int | None = None,
    value_shape: tuple[int, ...] | None = (),
) -> Float[Array, "sample ..."]:
    """Call a function of the caller on rows of points and gather its values per row.

    The function may belong to any array library, such as a NumPy
    log-likelihood or a torch prior. For consecutive chunks of at most
    ``chunk_size`` rows, the function gets the chunk as an array of
    ``backend``, and what it returns converts with :func:`to_jax`. The call
    runs on the host, outside ``jax.jit``, because the function is not JAX
    code. Values are not checked here, so callers check them, for example for
    NaN.

    Args:
        function: Maps points shaped (M, D), as arrays of ``backend``, to M
            real values, or to real arrays shaped ``(M, *value_shape)``.
        points: The points, shaped (N, D), with at least one row.
        backend: ``"numpy"``, ``"jax"``, ``"torch"`` or ``"cupy"``: the library
            of the arrays that the function receives.
        chunk_size: The most rows in one call; all rows in one call when
            ``None``. A smaller chunk bounds the memory of the function.
        value_shape: The shape of the result of one row: ``()`` for one value
            per row, a tuple of positive integers such as (D, D) for an array
            per row, or ``None`` for any trailing shape, which the first chunk
            fixes and every later chunk must match.

    Returns:
        The values in the order of the points, as float64, shaped
        ``(N, *value_shape)``; (N,) by default.

    Raises:
        ValueError: If the backend is unknown, ``chunk_size`` is below one,
            an entry of ``value_shape`` is below one, there are no points, or
            a result is not shaped ``(M, *value_shape)`` for its chunk of M rows.
        TypeError: If ``chunk_size`` is not an integer, ``value_shape`` is not
            a tuple of integers or ``None``, or a result is not real.
        MissingDependencyError: If torch or CuPy is not installed.

    Warns:
        PrecisionWarning: Once per call, if a result is a float narrower than
            float64, which is cast to float64.

    References:
        :cite:t:`dataapis2021python`.
    """
    known_backends = get_args(ArrayBackend.__value__)
    if backend not in known_backends:
        raise ValueError(
            f"unknown array backend {backend!r}; choose one of {known_backends}"
        )
    if chunk_size is not None:
        if isinstance(chunk_size, bool) or not isinstance(chunk_size, int | np.integer):
            raise TypeError(
                "chunk_size must be an integer or None, got"
                f" {type(chunk_size).__name__}"
            )
        if chunk_size < 1:
            raise ValueError(f"chunk_size must be at least 1, got {chunk_size}")
    _check_value_shape(value_shape)
    rows = to_jax(points)
    row_count = rows.shape[0]
    if row_count == 0:
        raise ValueError("call the function on at least one point, got none")
    step = row_count if chunk_size is None else int(chunk_size)
    values: list[Array] = []
    narrow_dtypes: set[str] = set()
    # * With value_shape None, the first chunk fixes the trailing shape.
    trailing_shape = value_shape
    for start in range(0, row_count, step):
        chunk = rows[start : start + step]
        result = to_jax(function(from_jax(chunk, backend)))
        dtype = result.dtype
        if not (
            jnp.issubdtype(dtype, jnp.floating)
            or jnp.issubdtype(dtype, jnp.integer)
            or jnp.issubdtype(dtype, jnp.bool_)
        ):
            raise TypeError(f"the function must return real numbers, got dtype {dtype}")
        trailing_shape = _trailing_shape(result.shape, chunk.shape[0], trailing_shape)
        if jnp.issubdtype(dtype, jnp.floating) and dtype.itemsize < 8:
            narrow_dtypes.add(str(dtype))
        values.append(jnp.asarray(result, dtype=jnp.float64))
    if narrow_dtypes:
        warnings.warn(
            PrecisionWarning(
                f"cast the function results ({', '.join(sorted(narrow_dtypes))}) to"
                " float64; the cast cannot restore precision lost before it:"
                " neighbouring float32 values differ by up to 1.2e-7 of their"
                " magnitude, and narrower types by more"
            ),
            skip_file_prefixes=(_PACKAGE_DIRECTORY,),
        )
    return jnp.concatenate(values)


def _check_value_shape(value_shape: object) -> None:
    """Raise unless ``value_shape`` is ``None`` or a tuple of positive integers."""
    if value_shape is None:
        return
    if not isinstance(value_shape, tuple):
        raise TypeError(
            "value_shape must be a tuple of integers or None, got"
            f" {type(value_shape).__name__}"
        )
    for size in value_shape:
        if isinstance(size, bool) or not isinstance(size, int | np.integer):
            raise TypeError(
                "value_shape must hold integers, got an entry of type"
                f" {type(size).__name__}"
            )
        if size < 1:
            raise ValueError(
                f"every entry of value_shape must be at least 1, got {value_shape}"
            )


def _trailing_shape(
    shape: tuple[int, ...], row_count: int, trailing_shape: tuple[int, ...] | None
) -> tuple[int, ...]:
    """Return the trailing shape of a result shaped (row_count, *trailing_shape).

    ``None`` accepts any trailing shape and returns the one that this result has.
    """
    if (
        len(shape) == 0
        or shape[0] != row_count
        or (trailing_shape is not None and shape[1:] != trailing_shape)
    ):
        expected = (
            f"({row_count}, ...)"
            if trailing_shape is None
            else str((row_count, *trailing_shape))
        )
        raise ValueError(
            f"the function returned shape {shape} for a chunk of {row_count} points;"
            f" it must return shape {expected}"
        )
    return shape[1:] if trailing_shape is None else trailing_shape


def _import_backend(module_name: str) -> ModuleType:
    try:
        return importlib.import_module(module_name)
    except ImportError as error:
        raise MissingDependencyError(
            f"{module_name} is not installed; {_INSTALL_HINTS[module_name]}"
        ) from error
