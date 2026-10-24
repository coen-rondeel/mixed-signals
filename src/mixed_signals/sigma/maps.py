"""Closed-form σ_eff and r_eff maps of the moment and marginal metrics.

σ_eff
    The mean shift between two unit Gaussians that gives the same value of a
    metric, in units of the whitening σ. :func:`sigma_eff` inverts each
    metric for that pair: the mean shift, W1 and the W2 total and location
    terms are the shift itself; the Gaussian KL total and location terms give
    √(2 KL); KS and Kuiper's V give 2Φ⁻¹((1 + D)/2), because for a shift F − G
    keeps one sign and V = D; the Hellinger total and location terms give
    √(−8 ln(1 − H²)) (Bhattacharyya 1943; Fukunaga 1990).
r_eff
    The ratio of widths along one axis between two unit Gaussians that gives
    the same value, for rows that measure only a difference in width. The
    width ratio is its own r_eff. For the shape terms, :func:`r_eff` solves
    KL(N(0, 1) ‖ N(0, r²)) = ½(1/r² − 1 + ln r²), the W2 distance, 1 − r in
    absolute value, and
    the Hellinger distance √(1 − √(2r/(1 + r²))) for r. Each value has two
    roots, r and a root below one; the sign of ln(det Σ_q/det Σ_p), which the
    Gaussian divergences of :mod:`mixed_signals.metrics.moments` return,
    chooses r above one when the second posterior is wider.
Domains
    Each map is the inverse of its metric for that pair, so a value outside
    the metric's range, such as a negative divergence or a KS statistic above
    one, gives NaN. The values a row passes are null-subtracted; the null
    construction settles how values below zero enter.
Maps that need numbers
    The Anderson-Darling and JS maps are tabulated curves, built from the zoo
    with the nulls, and are not in these tables yet.

References:
    :cite:t:`kullback1951information`; :cite:t:`dowson1982frechet`;
    :cite:t:`givens1984class`; :cite:t:`vallender1974calculation`;
    :cite:t:`smirnov1948table`; :cite:t:`kuiper1960tests`;
    :cite:t:`hellinger1909neue`; :cite:t:`bhattacharyya1943measure`;
    :cite:t:`fukunaga1990introduction`; :cite:t:`nakagawa2015meta`.
"""

import functools
from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import Final

import jax
import jax.numpy as jnp
from jax.scipy.special import ndtri
from jax.typing import ArrayLike
from jaxtyping import Array, Float

_NEWTON_STEPS: Final = 64
"""Newton steps of the KL shape inversion; near zero, where the root is double,
each step at least halves the error, so 64 steps reach float64 resolution."""


def _shift(value: Array) -> Array:
    return value


def _distance(value: Array) -> Array:
    return jnp.where(value >= 0, value, jnp.nan)


def _kl_sigma(value: Array) -> Array:
    return jnp.sqrt(jnp.where(value >= 0, 2 * value, jnp.nan))


def _ks_sigma(value: Array) -> Array:
    inside = (value >= 0) & (value <= 1)
    return jnp.where(inside, 2 * ndtri(0.5 * (1 + value)), jnp.nan)


def _hellinger_sigma(value: Array) -> Array:
    inside = (value >= 0) & (value <= 1)
    return jnp.where(inside, jnp.sqrt(-8 * jnp.log1p(-(value**2))), jnp.nan)


SIGMA_EFF_MAPS: Final[Mapping[str, Callable[[Array], Array]]] = MappingProxyType(
    {
        "mean_shift": _shift,
        "w1": _distance,
        "w2": _distance,
        "w2_location": _distance,
        "kl": _kl_sigma,
        "kl_location": _kl_sigma,
        "ks": _ks_sigma,
        "kuiper_ks": _ks_sigma,
        "hellinger": _hellinger_sigma,
        "hellinger_location": _hellinger_sigma,
    }
)
"""The closed-form σ_eff map of each metric, by metric name.

:meta hide-value:
"""


def sigma_eff(
    metric: str, value: Float[ArrayLike, " *leading"]
) -> Float[Array, " *leading"]:
    """Return the Gaussian-equivalent shift σ_eff of metric values.

    Args:
        metric: Name of the metric, a key of :data:`SIGMA_EFF_MAPS`.
        value: Values of the metric, in the units of the whitening.

    Returns:
        σ_eff for each value; NaN outside the range of the metric.

    Raises:
        KeyError: If the metric has no closed-form map.

    References:
        :cite:t:`kullback1951information`; :cite:t:`smirnov1948table`;
        :cite:t:`kuiper1960tests`; :cite:t:`bhattacharyya1943measure`;
        :cite:t:`fukunaga1990introduction`.
    """
    try:
        mapping = SIGMA_EFF_MAPS[metric]
    except KeyError:
        raise KeyError(
            f"no closed-form sigma_eff map for {metric!r}; the maps cover"
            f" {', '.join(sorted(SIGMA_EFF_MAPS))}"
        ) from None
    return mapping(jnp.asarray(value, dtype=jnp.float64))


def r_eff(
    metric: str,
    value: Float[ArrayLike, " *leading"],
    log_det_ratio: Float[ArrayLike, " *leading"] | None = None,
) -> Float[Array, " *leading"]:
    """Return the Gaussian-equivalent width ratio r_eff of metric values.

    Args:
        metric: ``"width_ratio"``, ``"kl_shape"``, ``"w2_shape"`` or
            ``"hellinger_shape"``.
        value: Values of the metric.
        log_det_ratio: ln(det Σ_q/det Σ_p) of each row, whose sign chooses
            the root above or below one; needed for every shape term.

    Returns:
        r_eff for each value, below one when the second posterior is
        narrower; NaN outside the range of the metric.

    Raises:
        KeyError: If the metric has no r_eff map.
        ValueError: If a shape term comes without ``log_det_ratio``.

    References:
        :cite:t:`nakagawa2015meta`; :cite:t:`kullback1951information`;
        :cite:t:`givens1984class`; :cite:t:`bhattacharyya1943measure`.
    """
    values = jnp.asarray(value, dtype=jnp.float64)
    if metric == "width_ratio":
        return jnp.where(values > 0, values, jnp.nan)
    if metric not in _SHAPE_MAPS:
        raise KeyError(
            f"no r_eff map for {metric!r}; the maps cover width_ratio,"
            f" {', '.join(sorted(_SHAPE_MAPS))}"
        )
    if log_det_ratio is None:
        raise ValueError(f"the r_eff of {metric!r} needs log_det_ratio")
    wider = jnp.asarray(log_det_ratio, dtype=jnp.float64) > 0
    return _SHAPE_MAPS[metric](values, wider)


def _kl_shape_ratio(value: Array, wider: Array) -> Array:
    """Return r with ½(1/r² − 1 + ln r²) = value, above one where ``wider``."""
    # * With s = ln r², g(s) = e^(−s) + s − 1 − 2 KL is convex, one root on each side.
    target = 2 * jnp.where(value >= 0, value, jnp.nan)
    # * Each start lies where g > 0, so Newton steps approach the root from one side.
    start = jnp.where(wider, 1 + target, -jnp.log1p(target) - 1)
    log_square = jax.lax.fori_loop(
        0, _NEWTON_STEPS, functools.partial(_newton_step, target=target), start
    )
    return jnp.where(value == 0, 1.0, jnp.exp(0.5 * log_square))


def _newton_step(_: Array, log_square: Array, target: Array) -> Array:
    """Return one Newton step on g(s) = e^(−s) + s − 1 − 2 KL, kept where g' = 0."""
    excess = jnp.expm1(-log_square) + log_square - target
    slope = -jnp.expm1(-log_square)
    safe = jnp.where(slope == 0, 1.0, slope)
    return jnp.where(slope == 0, log_square, log_square - excess / safe)


def _w2_shape_ratio(value: Array, wider: Array) -> Array:
    ratio = jnp.where(wider, 1 + value, 1 - value)
    return jnp.where((value >= 0) & (ratio > 0), ratio, jnp.nan)


def _hellinger_shape_ratio(value: Array, wider: Array) -> Array:
    """Return r with √(1 − √(2r/(1 + r²))) = value, above one where ``wider``."""
    inside = (value >= 0) & (value < 1)
    # * c = 1 − H² and 1 − c⁴ come from log1p, so small distances keep their digits.
    log_coefficient = jnp.log1p(-(jnp.where(inside, value, 0.0) ** 2))
    spread = jnp.sqrt(-jnp.expm1(4 * log_coefficient))
    root = jnp.where(wider, 1 + spread, 1 - spread) / jnp.exp(2 * log_coefficient)
    return jnp.where(inside, root, jnp.nan)


_SHAPE_MAPS: Final[Mapping[str, Callable[[Array, Array], Array]]] = MappingProxyType(
    {
        "kl_shape": _kl_shape_ratio,
        "w2_shape": _w2_shape_ratio,
        "hellinger_shape": _hellinger_shape_ratio,
    }
)
