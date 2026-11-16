"""Gaussian surrogates: the null of the posterior-shape diagnostics.

A shape or dependence diagnostic asks how far a posterior departs from a
Gaussian. Its null comes from surrogate sets drawn from the posterior's
moment-matched Gaussian, a parametric bootstrap (Efron and Tibshirani 1993):
:func:`gaussian_surrogates` fits
:class:`~mixed_signals.estimators.gaussian.GaussianDensity` to the weighted
samples and draws sets of equally weighted samples from it, each circle
wrapped back onto its period. A statistic of the surrogates then scatters as
it would for a Gaussian posterior, finite-sample bias included.

Size
    Each set holds the posterior's effective sample size, so that the
    surrogates scatter as the posterior's statistic does; a statistic that
    runs on at most a cap of samples takes the smaller of the two.
Truncation
    With ``truncate=True`` each interval parameter is drawn from its
    moment-matched normal truncated to the interval, by the inverse
    distribution function applied to the same normal deviates (Devroye
    1986), so statistics that need values inside the support, such as the
    bounded KDE of the mode rows, can run on the surrogates. The other
    parameters, and the dependence between all of them, stay as they are.
Keys
    Set i draws from ``derive_key(key, "surrogate", i)``, so a set does not
    depend on how many others are drawn.
Recipe
    Rows name the construction :data:`CONSTRUCTION` and the tail of their
    p-value in the ``null_recipe`` column.

References:
    :cite:t:`efron1993introduction`; :cite:t:`mardia2000directional`;
    :cite:t:`devroye1986non`.
"""

import math
import operator
from typing import Final

import jax
import jax.numpy as jnp
from jax.scipy.special import ndtr, ndtri
from jax.typing import ArrayLike
from jaxtyping import Array, Float, PRNGKeyArray

from mixed_signals._config import NULL_DRAWS
from mixed_signals.core.space import ParamSpace
from mixed_signals.core.topology import Topology
from mixed_signals.estimators.gaussian import GaussianDensity
from mixed_signals.prng import derive_key

CONSTRUCTION: Final = "gaussian_surrogate"
"""Name of the construction in the ``null_recipe`` column of diagnostic rows."""


def gaussian_surrogates(
    samples: ArrayLike,
    log_weights: ArrayLike | None,
    space: ParamSpace,
    *,
    size: int,
    key: PRNGKeyArray,
    draws: int = NULL_DRAWS,
    truncate: bool = False,
) -> Float[Array, "draw sample param"]:
    """Draw surrogate sets from the moment-matched Gaussian of one posterior.

    Args:
        samples: Samples of one posterior, shaped (N, D).
        log_weights: Their log weights, shaped (N,); equal weights when
            ``None``.
        space: Parameter space of the samples.
        size: Samples per set, usually the effective sample size.
        key: JAX PRNG key; set i draws from ``derive_key(key, "surrogate",
            i)``.
        draws: Number of sets.
        truncate: Whether to draw each interval parameter from its normal
            truncated to the interval.

    Returns:
        The sets, shaped (draws, size, D), with circles on their period; NaN
        if the covariance of the samples is singular.

    Raises:
        TypeError: If ``size`` or ``draws`` is not an integer.
        ValueError: If ``size`` or ``draws`` is below one, or the samples do
            not fit the space.

    Warns:
        ValidityWarning: As
            :meth:`~mixed_signals.estimators.gaussian.GaussianDensity.fit`, for
            a singular covariance or a circle whose mean direction is not
            resolved.

    References:
        :cite:t:`efron1993introduction`.
    """
    for name, count in (("size", size), ("draws", draws)):
        if isinstance(count, bool) or not isinstance(count, int):
            raise TypeError(f"{name} must be an integer, got {type(count).__name__}")
        if count < 1:
            raise ValueError(f"{name} must be at least 1, got {count}")
    gaussian = GaussianDensity.fit(samples, log_weights, space)
    keys = jnp.stack([derive_key(key, "surrogate", index) for index in range(draws)])
    sets = jax.vmap(
        lambda draw_key: gaussian.sample(draw_key, (size,)),
    )(keys)
    if truncate:
        for position, parameter in enumerate(space.parameters):
            if parameter.topology is Topology.INTERVAL:
                sets = sets.at[..., position].set(
                    _truncate(
                        sets[..., position],
                        gaussian.mean[position],
                        jnp.sqrt(gaussian.covariance[position, position]),
                        parameter.bounds,
                    )
                )
    return sets


def _truncate(
    values: Float[Array, " *leading"],
    mean: Float[Array, ""],
    width: Float[Array, ""],
    bounds: tuple[float, float],
) -> Float[Array, " *leading"]:
    """Map normal draws to the same normal truncated to the bounds, by inverse CDF."""
    lower, upper = (
        ndtr((bound - mean) / width) if math.isfinite(bound) else jnp.asarray(edge)
        for bound, edge in zip(bounds, (0.0, 1.0), strict=True)
    )
    levels = lower + ndtr((values - mean) / width) * (upper - lower)
    return jnp.clip(mean + width * ndtri(levels), bounds[0], bounds[1])


def surrogate_size(effective_sample_size: float, cap: int | None = None) -> int:
    """Return the samples per surrogate set: the effective sample size, rounded.

    Args:
        effective_sample_size: The posterior's n_eff, at least one.
        cap: Largest number of samples the statistic runs on, if any.

    Returns:
        round(n_eff), at least two, and at most ``cap``.

    Raises:
        ValueError: If the effective sample size is below one or not finite.
    """
    if not (math.isfinite(effective_sample_size) and effective_sample_size >= 1):
        raise ValueError(
            f"the effective sample size must be at least 1, got {effective_sample_size}"
        )
    size = max(2, round(effective_sample_size))
    return size if cap is None else min(size, operator.index(cap))
