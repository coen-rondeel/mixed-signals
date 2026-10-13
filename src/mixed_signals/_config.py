"""Process-wide JAX settings and package defaults.

The package works in float64 throughout: galactic-binary frequency posteriors
have a relative width near 1e-8, below the resolution of float32.
"""

import jax


def enable_float64() -> None:
    """Switch JAX to float64 for the whole process.

    The JAX setting is global, so it also applies to JAX code outside this
    package that runs in the same process.
    """
    jax.config.update("jax_enable_x64", True)  # type: ignore[no-untyped-call]


def float64_enabled() -> bool:
    """Return whether JAX currently uses float64 as its default float type."""
    return bool(jax.config.jax_enable_x64)
