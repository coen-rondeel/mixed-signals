"""Whitened templates: signals of a generator, whitened by a noise model, as JAX rows.

A data-space comparison needs the templates of its samples as real vectors whose
dot products are the noise-weighted inner products (Finn 1992; Cutler and
Flanagan 1994). :func:`whitened_templates` calls the generator and the noise
model of the caller on chunks of samples, at the boundary and outside
``jax.jit`` (:func:`~mixed_signals.io.arrays.call_in_chunks`), so the templates
and their whitening run in the library of the caller and only the real whitened
rows cross into JAX, through DLPack on a GPU (Consortium for Python Data API
Standards 2021). A complex template, which the data domain of a
frequency-domain waveform holds, never leaves the library of the caller.

References:
    :cite:t:`finn1992detection`; :cite:t:`cutler1994gravitational`;
    :cite:t:`dataapis2021python`.
"""

from functools import partial
from typing import Any

import jax.numpy as jnp
from jax.typing import ArrayLike
from jaxtyping import Array, Float

from mixed_signals.generators.capabilities import Capability, require_capabilities
from mixed_signals.generators.protocol import NoiseModel, SignalGenerator
from mixed_signals.io.arrays import ArrayBackend, call_in_chunks


def whitened_templates(
    generator: SignalGenerator,
    noise: NoiseModel,
    points: Float[ArrayLike, "sample param"],
    *,
    backend: ArrayBackend = "numpy",
    chunk_size: int | None = None,
) -> Float[Array, "sample data"]:
    """Return the whitened templates of parameter vectors, shaped (N, M).

    For consecutive chunks of at most ``chunk_size`` points, the generator
    gets the chunk as an array of ``backend`` and returns the signals, which
    the noise model whitens to real vectors in the same library. The M entries
    of a whitened vector are the entries of the data vector; templates of one
    problem share one data grid.

    Args:
        generator: A signal generator, with ``generate_template(params)``.
        noise: A noise model, with ``whiten(signals)``.
        points: Parameter vectors shaped (N, D), with at least one row.
        backend: ``"numpy"``, ``"jax"``, ``"torch"`` or ``"cupy"``: the library
            of the arrays that the generator receives.
        chunk_size: The most points in one call; all points in one call when
            ``None``. A smaller chunk bounds the memory of the generator.

    Returns:
        The whitened templates as float64, shaped (N, M).

    Raises:
        CapabilityError: If the generator lacks ``generate_template`` or the
            noise model lacks ``whiten``.
        ValueError: If ``whiten`` does not return real vectors shaped (n, M)
            with M of one or more, a whitened template is not finite, or
            ``call_in_chunks`` rejects the points, the backend or the chunk
            size.
        TypeError: If the whitened templates are not real.
        MissingDependencyError: If torch or CuPy is not installed.

    Warns:
        PrecisionWarning: Once per call, if the whitened vectors are narrower
            than float64.

    References:
        :cite:t:`finn1992detection`; :cite:t:`cutler1994gravitational`;
        :cite:t:`dataapis2021python`.
    """
    require_capabilities(
        generator,
        [Capability.TEMPLATES],
        engine="whitened_templates",
        role="generator",
    )
    require_capabilities(
        noise, [Capability.WHITEN], engine="whitened_templates", role="noise model"
    )
    whitened = call_in_chunks(
        partial(_whiten_chunk, generator, noise),
        points,
        backend=backend,
        chunk_size=chunk_size,
        value_shape=None,
    )
    if whitened.ndim != 2 or whitened.shape[1] < 1:
        raise ValueError(
            f"whiten must return real vectors shaped (n, M), got {whitened.shape}"
        )
    not_finite = ~jnp.all(jnp.isfinite(whitened), axis=1)
    if bool(jnp.any(not_finite)):
        raise ValueError(
            f"the whitened templates are not finite in {int(jnp.sum(not_finite))} of"
            f" {whitened.shape[0]} rows"
        )
    return whitened


def _whiten_chunk(generator: SignalGenerator, noise: NoiseModel, chunk: Any) -> Any:
    """Return the whitened templates of one chunk, in the library of the chunk."""
    return noise.whiten(generator.generate_template(chunk))
