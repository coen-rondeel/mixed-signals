"""Chordal embedding of a parameter space, shared by the distance-based metrics.

The Euclidean block stays as it is, each circle of period P becomes a circle
of circumference P in a plane, and each sphere a sphere of its radius in
three dimensions. Euclidean distances between embedded points are chords on
circles and spheres and equal the arc lengths locally. Euclidean spaces have
strong negative type (Lyons 2013), and on them the Gaussian kernel is
positive definite and characteristic (Gretton et al. 2012), so the energy
distance, the maximum mean discrepancy and the Sinkhorn divergence of the
squared chord keep the properties that arc lengths would lose on circles and
spheres (Lyons 2020; Feragen, Lauze and Hauberg 2015).

References:
    :cite:t:`lyons2013distance`; :cite:t:`gretton2012kernel`;
    :cite:t:`lyons2020strong`; :cite:t:`feragen2015geodesic`.
"""

import math

import jax.numpy as jnp
from jaxtyping import Array, Float

from mixed_signals.core.space import ParamSpace
from mixed_signals.core.topology import unit_vectors


def embedding_size(space: ParamSpace) -> int:
    """Return the number of embedded coordinates, E + 2C + 3S."""
    blocks = space.axis_blocks
    return len(blocks.euclidean) + 2 * len(blocks.circles) + 3 * len(blocks.spheres)


def embed(
    samples: Float[Array, "sample param"], space: ParamSpace
) -> Float[Array, "sample _"]:
    """Return points whose Euclidean distances are chordal on circles and spheres.

    Args:
        samples: Samples in the parameter order of ``space``, shaped (N, D).
        space: Space of the samples.

    Returns:
        The embedded points, shaped (N, E + 2C + 3S): the Euclidean block,
        then two coordinates per circle and three per sphere.
    """
    blocks = space.axis_blocks
    columns = []
    if blocks.euclidean:
        columns.append(samples[:, list(blocks.euclidean)])
    for position in blocks.circles:
        lower, upper = space.parameters[position].bounds
        period = upper - lower
        angle = 2 * math.pi * (samples[:, position] - lower) / period
        radius = period / (2 * math.pi)
        columns.append(radius * jnp.stack([jnp.cos(angle), jnp.sin(angle)], axis=1))
    for (azimuth, polar), sphere in zip(blocks.spheres, space.spheres, strict=True):
        vectors, _, _ = unit_vectors(
            samples[:, azimuth], samples[:, polar], sphere.polar_form
        )
        columns.append(sphere.radius * vectors)
    return jnp.concatenate(columns, axis=1)
