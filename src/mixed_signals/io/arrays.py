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

torch and CuPy are imported inside :func:`from_jax`, so the package imports
without them. The ``torch`` extra installs torch, and the ``cuda12`` and
``cuda13`` extras install CuPy together with the matching JAX CUDA plugin.

References:
    :cite:t:`dataapis2021python`.
"""

import importlib
from types import ModuleType
from typing import Any, Literal, get_args

import jax
import jax.dlpack
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from mixed_signals.errors import MissingDependencyError

type ArrayBackend = Literal["numpy", "jax", "torch", "cupy"]
"""Array libraries that :func:`from_jax` converts to."""

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


def _import_backend(module_name: str) -> ModuleType:
    try:
        return importlib.import_module(module_name)
    except ImportError as error:
        raise MissingDependencyError(
            f"{module_name} is not installed; {_INSTALL_HINTS[module_name]}"
        ) from error
