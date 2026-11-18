"""Prior railing: the marginal posterior density at a finite bound over its peak.

A posterior rails when a prior bound cuts into it. :func:`railing_scores`
measures this for every finite bound of every interval parameter as the
density of the one-dimensional :class:`~mixed_signals.estimators.kde.KernelDensity`
at the bound divided by its largest value. The estimate takes the
non-negative boundary correction of Jones and Foster (1996), whose bias at a
bound is of order h², so the score of a normal whose peak lies z widths inside
the bound, exp(−z²/2), is recovered within a few per cent down to z = 2.5,
where reflection read 10 to 30 per cent high (zoo check, see
:ref:`validation-kde-railing`). No paper defines a
railing diagnostic. The LVK tool pe-configurator reports railing of the chirp
mass, the mass ratio and the distance when the edge bin of a 50-bin histogram
of the samples exceeds 2% of the highest bin, and the default flag threshold
takes the same 0.02 for the score, a peak within about 2.8σ of the bound.

Rows
    ``railing_lower`` and ``railing_upper`` hold the score at each finite
    bound. They carry no surrogate null, because a Gaussian surrogate knows no
    bound.
Spheres and circles
    Circles have no bounds. The polar axis of a sphere ends at the poles,
    which are not prior bounds, so it gets no railing rows.

References:
    :cite:t:`jones1993simple`; :cite:t:`jones1996simple`;
    :cite:t:`silverman1986density`; :cite:t:`estelles2023peconfigurator`.
"""

import math
from typing import Final, NamedTuple

import jax.numpy as jnp
from jax.typing import ArrayLike
from jaxtyping import Array, Float

from mixed_signals.core.space import Parameter, ParamSpace
from mixed_signals.core.topology import Topology
from mixed_signals.diagnostics._common import _jit
from mixed_signals.estimators.base import _read_samples
from mixed_signals.estimators.kde import KernelDensity

GRID_POINTS: Final = 2_048
"""Grid points over the data, inside the bounds, at which the peak density is sought."""

# * The grid reaches this many kernel widths beyond the data, within the bounds.
_GRID_MARGIN: Final = 3.0


class RailingScores(NamedTuple):
    """Railing score at the lower and the upper bound of every parameter."""

    lower: Float[Array, " param"]
    """Density at the lower bound over the peak; NaN without a finite lower bound."""
    upper: Float[Array, " param"]
    """Density at the upper bound over the peak; NaN without a finite upper bound."""


def railed_parameters(space: ParamSpace) -> tuple[int, ...]:
    """Return the positions that get railing rows: intervals that are no polar axis.

    Args:
        space: Parameter space.

    Returns:
        The positions of interval parameters outside the polar axes of spheres.
    """
    polar_axes = {polar for _, polar in space.axis_blocks.spheres}
    return tuple(
        position
        for position, parameter in enumerate(space.parameters)
        if parameter.topology is Topology.INTERVAL and position not in polar_axes
    )


def railing_scores(
    samples: ArrayLike, log_weights: ArrayLike | None, space: ParamSpace
) -> RailingScores:
    """Return the railing score at every finite bound of every interval parameter.

    Args:
        samples: Samples of one posterior, shaped (N, D), inside the bounds.
        log_weights: Their log weights, shaped (N,); equal weights when
            ``None``.
        space: Parameter space of the samples.

    Returns:
        The scores at the lower and upper bounds, each shaped (D,); NaN
        where a parameter has no finite bound on that side, is no interval,
        or is the polar axis of a sphere.

    Raises:
        ValueError: If the shapes do not fit the space, or concrete values
            are invalid.
        TypeError: If an input is not real-valued.

    Warns:
        ValidityWarning: As
            :meth:`~mixed_signals.estimators.kde.KernelDensity.fit`.
        PrecisionWarning: If an input is narrower than float64.

    References:
        :cite:t:`jones1996simple`.
    """
    samples_array, weights_array = _read_samples(samples, log_weights, space)
    lower = jnp.full(len(space), jnp.nan)
    upper = jnp.full(len(space), jnp.nan)
    for position in railed_parameters(space):
        parameter = space.parameters[position]
        scores = _scores(
            samples_array[:, position],
            weights_array,
            parameter=Parameter(
                name="value", topology=parameter.topology, bounds=parameter.bounds
            ),
        )
        lower = lower.at[position].set(scores[0])
        upper = upper.at[position].set(scores[1])
    return RailingScores(lower=lower, upper=upper)


def _scores_kernel(
    values: Float[Array, " sample"],
    log_weights: Float[Array, " sample"],
    *,
    parameter: Parameter,
) -> Float[Array, " 2"]:
    """Return the density at each finite bound over the largest density between them."""
    bounds = parameter.bounds
    kde = KernelDensity.fit(
        values[:, None], log_weights, ParamSpace(parameters=(parameter,))
    )
    # * The covariance of JAX's estimator already holds the factor squared.
    width = jnp.sqrt(kde.kde.covariance[0, 0])
    low = jnp.clip(jnp.min(values) - _GRID_MARGIN * width, *bounds)
    high = jnp.clip(jnp.max(values) + _GRID_MARGIN * width, *bounds)
    # * The bounds join the grid, so a peak at a bound and a bound far away both count.
    edges = jnp.asarray([bound if math.isfinite(bound) else 0.0 for bound in bounds])
    points = jnp.concatenate([edges, jnp.linspace(low, high, GRID_POINTS)])
    density = jnp.exp(kde.log_density(points[:, None]))
    finite = jnp.asarray([math.isfinite(bound) for bound in bounds])
    peak = jnp.max(jnp.where(jnp.arange(points.shape[0]) < 2, 0.0, density))
    peak = jnp.maximum(peak, jnp.max(jnp.where(finite, density[:2], 0.0)))
    return jnp.where(finite, density[:2] / peak, jnp.nan)


_scores = _jit(_scores_kernel, "parameter")
