"""Empirical p-values and the significance σ_sig of a value against its null draws.

p-value
    With M null draws, the p-value is the number of draws at least as extreme
    as the value, plus one, over M + 1 (Phipson and Smyth 2010). It never
    falls to zero, and it is exact when the value and the draws are
    exchangeable under the null. An upper tail counts the draws at or above
    the value; two sides count the draws at least as far from the mean of the
    draws as the value, for rows whose departures count in both directions,
    such as a skewness. Draws that are not finite leave the count and M.
σ_sig
    The p-value as a Gaussian significance: Φ⁻¹(1 − p) for an upper tail
    (Cowan et al. 2011), and Φ⁻¹(1 − p/2) for two sides, so that σ_sig equals
    the absolute z-score for a symmetric Gaussian null. M draws resolve σ_sig up to
    Φ⁻¹(1 − 1/(M + 1)), 2.58 for the default
    :data:`~mixed_signals._config.NULL_DRAWS` of 200; a fit to the tail of the
    null beyond that resolution is not part of this module yet.

References:
    :cite:t:`phipson2010permutation`; :cite:t:`cowan2011asymptotic`.
"""

from enum import StrEnum

import jax.numpy as jnp
from jax.scipy.special import ndtri
from jax.typing import ArrayLike
from jaxtyping import Array, Float


class Tail(StrEnum):
    """Which departures from the null count as extreme.

    ``upper``
        Values above the null, for statistics that grow with the departure,
        such as a distance or an excess dependence.
    ``two_sided``
        Values far from the mean of the null in either direction, for signed
        statistics such as a skewness or an excess kurtosis.
    """

    UPPER = "upper"
    TWO_SIDED = "two_sided"


def empirical_p_value(
    value: Float[ArrayLike, " *leading"],
    draws: Float[ArrayLike, "*leading draw"],
    tail: Tail | str,
) -> Float[Array, " *leading"]:
    """Return the empirical p-value of values against their null draws.

    Args:
        value: The observed values, shaped (...).
        draws: The null draws of each value along the last axis, shaped
            (..., M).
        tail: ``"upper"`` or ``"two_sided"``; see :class:`Tail`.

    Returns:
        (1 + number of draws at least as extreme) / (1 + M), with M the
        number of finite draws; NaN where the value is not finite or no draw
        is.

    Raises:
        ValueError: If ``tail`` is not a :class:`Tail` value.

    References:
        :cite:t:`phipson2010permutation`.
    """
    side = Tail(tail)
    values = jnp.asarray(value, dtype=jnp.float64)
    null = jnp.asarray(draws, dtype=jnp.float64)
    finite = jnp.isfinite(null)
    count = jnp.sum(finite, axis=-1)
    if side is Tail.UPPER:
        extreme = null >= values[..., None]
    else:
        centre = jnp.sum(jnp.where(finite, null, 0.0), axis=-1) / count
        extreme = (
            jnp.abs(null - centre[..., None]) >= jnp.abs(values - centre)[..., None]
        )
    exceed = jnp.sum(finite & extreme, axis=-1)
    p_value = (1.0 + exceed) / (1.0 + count)
    return jnp.where(jnp.isfinite(values) & (count > 0), p_value, jnp.nan)


def sigma_sig(
    p_value: Float[ArrayLike, " *leading"], tail: Tail | str
) -> Float[Array, " *leading"]:
    """Return the Gaussian significance of p-values.

    Args:
        p_value: p-values in (0, 1].
        tail: ``"upper"`` for Φ⁻¹(1 − p), ``"two_sided"`` for Φ⁻¹(1 − p/2).

    Returns:
        σ_sig for each p-value, −∞ for p = 1 in the upper tail; NaN outside
        (0, 1].

    Raises:
        ValueError: If ``tail`` is not a :class:`Tail` value.

    References:
        :cite:t:`cowan2011asymptotic`.
    """
    side = Tail(tail)
    values = jnp.asarray(p_value, dtype=jnp.float64)
    inside = (values > 0) & (values <= 1)
    probability = values if side is Tail.UPPER else values / 2
    # * −Φ⁻¹(p) equals Φ⁻¹(1 − p) and keeps its digits for small p.
    return jnp.where(inside, -ndtri(probability), jnp.nan)
