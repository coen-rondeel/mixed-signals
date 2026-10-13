"""Checks of zoo inputs, shared by the closed forms, distributions and builders.

NumPy reads the parameters that users pass to constructors and builders;
:func:`require` checks the values of JAX computations, and only when they are
concrete, because traced values are unknown inside ``jax.jit``.
"""

import math
from typing import Final

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

SYMMETRY_TOLERANCE: Final = 1e-10
"""Largest asymmetry of a covariance matrix, relative to its largest entry."""


def require(condition: object, message: str) -> None:
    """Raise ValueError unless a concrete condition holds everywhere.

    A traced condition, inside ``jax.jit`` or ``jax.vmap``, is skipped: there
    the values are unknown, and invalid inputs give NaN instead.
    """
    if isinstance(condition, jax.core.Tracer):
        return
    if not bool(jnp.all(jnp.asarray(condition))):
        raise ValueError(message)


def require_covariance(name: str, covariance: Float[Array, "param param"]) -> None:
    """Check that a concrete covariance is finite, symmetric and positive definite."""
    require(jnp.all(jnp.isfinite(covariance)), f"{name} must be finite")
    scale = jnp.max(jnp.abs(covariance))
    require(
        jnp.max(jnp.abs(covariance - covariance.T)) <= SYMMETRY_TOLERANCE * scale,
        f"{name} must be symmetric",
    )
    require(
        jnp.all(jnp.isfinite(jnp.linalg.cholesky(covariance))),
        f"{name} must be positive definite",
    )


def cholesky(covariance: Float[Array, "param param"], name: str) -> Array:
    """Return the lower Cholesky factor, after checking a concrete covariance."""
    require_covariance(name, covariance)
    return jnp.asarray(jnp.linalg.cholesky(covariance))


def vector(name: str, value: object, size: int | None = None) -> np.ndarray:
    """Read a finite float64 vector, shaped (size,) when a size is given."""
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 1 or (size is not None and array.shape != (size,)):
        expected = "(D,)" if size is None else f"({size},)"
        raise ValueError(f"{name} must be shaped {expected}, got shape {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    return array


def matrix(name: str, value: object, size: int) -> np.ndarray:
    """Read a float64 matrix shaped (size, size); its values are checked later."""
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (size, size):
        raise ValueError(
            f"{name} must be shaped ({size}, {size}), got shape {array.shape}"
        )
    return array


def check_finite(name: str, value: float) -> None:
    """Raise ValueError unless the value is a finite number."""
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value!r}")


def check_positive(name: str, value: float) -> None:
    """Raise ValueError unless the value is finite and positive."""
    if not (math.isfinite(value) and value > 0):
        raise ValueError(f"{name} must be finite and positive, got {value!r}")


def check_count(name: str, value: int, minimum: int = 1) -> None:
    """Raise TypeError or ValueError unless the value is an integer >= minimum."""
    if isinstance(value, bool) or not isinstance(value, int | np.integer):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}, got {value}")
