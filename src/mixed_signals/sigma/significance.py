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
    the absolute z-score for a symmetric Gaussian null. M draws resolve σ_sig
    up to the one of the smallest p-value, 1/(M + 1): Φ⁻¹(1 − 1/(M + 1)) in the
    upper tail, 2.58 for the default
    :data:`~mixed_signals._config.NULL_DRAWS` of 200.
Tail fit
    :func:`null_significance` resolves smaller p-values than that. When fewer than
    :data:`MINIMUM_EXCEEDANCES` draws are at least as extreme as the value
    (the rule of Knijnenburg et al. 2009), it fits a generalized Pareto
    distribution (GPD) to the excesses of the most extreme
    :data:`TAIL_FRACTION` of the draws over a threshold between the last draw
    of the tail and the first one outside it, with the estimator of Zhang and
    Stephens (2009), :func:`~mixed_signals.sigma.pareto.generalized_pareto_fit`.
    The p-value is the share of draws in the tail times the survival function
    of the fit at the value's excess, which interpolates between the last
    draws and extrapolates beyond them, and its row is flagged as extrapolated.
    The fit keeps the end point of its support above the value, by the margin
    :data:`SUPPORT_MARGIN` (the support constraint of Peschel et al. 2026), so
    the p-value never falls to zero, as it does for the unconstrained maximum
    likelihood fit of Knijnenburg et al. (see the zoo check). A fitted σ_sig
    reads low far in the tail: against 200 normal draws, a value at 10σ gave
    4.9 and one at 100σ gave 7.7, so it is a lower estimate there.
Zoo check
    On known nulls, a normal, a Kolmogorov and a χ² distribution with 4 degrees
    of freedom, with 1 000 replicates, 200 draws gave a median σ_sig of 2.9 at
    a true 3σ, 3.4 to 3.6 at 4σ and 3.8 to 4.1 at 5σ, with a range of about
    ±0.3 between the 16th and 84th percentiles and no infinite value; 1 000
    draws gave 3.0, 4.05 and 4.7 to 4.9. The unconstrained maximum likelihood
    fit returned p = 0 in 18 to 31%, 40 to 64% and 51 to 79% of the cases at
    200 draws. A support margin between 1.001 and 1.05 changed the medians by
    at most 0.05.

References:
    :cite:t:`phipson2010permutation`; :cite:t:`cowan2011asymptotic`;
    :cite:t:`knijnenburg2009fewer`; :cite:t:`peschel2026permapprox`;
    :cite:t:`zhang2009new`.
"""

import os
import sys
import warnings
from collections.abc import Callable
from enum import StrEnum
from functools import partial
from typing import Final, NamedTuple, cast

import jax
import jax.numpy as jnp
from jax.scipy.special import ndtri
from jax.typing import ArrayLike
from jaxtyping import Array, Bool, Float, Int

from mixed_signals.core.samples import _is_concrete
from mixed_signals.errors import ExtrapolationWarning
from mixed_signals.sigma.pareto import (
    generalized_pareto_fit,
    generalized_pareto_log_survival,
)

TAIL_FRACTION: Final = 0.25
"""Share of the null draws, the most extreme ones, that the tail fit uses.

The starting share of permApprox (Peschel et al. 2026), without its
goodness-of-fit search for a smaller share: 200 draws leave 50 in the tail.
The accuracy of the resulting σ_sig is in the zoo check of the module
docstring.
"""

MINIMUM_EXCEEDANCES: Final = 10
"""Fewest draws at least as extreme as the value for which the p-value is empirical.

With fewer, the p-value comes from the tail fit (Knijnenburg et al. 2009).
"""

SUPPORT_MARGIN: Final = 1.05
"""Factor by which the support of the tail fit must exceed the value's excess.

The fitted end point lies above this multiple of the value's excess over the
threshold, the support constraint of Peschel et al. (2026). Margins between
1.001 and 1.05 changed the median σ_sig by at most 0.05 in the zoo check of the
module docstring.
"""

_PACKAGE_DIRECTORY: Final = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _jit[**Parameters, Result](
    function: Callable[Parameters, Result], *static_names: str
) -> Callable[Parameters, Result]:
    """Compile a kernel with :func:`jax.jit` and keep its signature for type checks."""
    return cast(
        Callable[Parameters, Result], jax.jit(function, static_argnames=static_names)
    )


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


class Significance(NamedTuple):
    """The p-value and σ_sig of values against their nulls, shaped like the values."""

    p_value: Float[Array, " *leading"]
    """The empirical p-value, or the tail fit's where the row is extrapolated."""
    sigma_sig: Float[Array, " *leading"]
    """The p-value as a Gaussian significance."""
    exceedances: Int[Array, " *leading"]
    """Number of finite draws at least as extreme as the value."""
    extrapolated: Bool[Array, " *leading"]
    """Whether the p-value comes from the tail fit, which interpolates between the
    last draws and extrapolates beyond them."""


def null_significance(
    value: Float[ArrayLike, " *leading"],
    draws: Float[ArrayLike, "*leading draw"],
    tail: Tail | str,
    *,
    tail_fit: bool = True,
) -> Significance:
    """Return the p-value and σ_sig of values against their null draws.

    The p-value is the empirical one of :func:`empirical_p_value`, unless
    fewer than :data:`MINIMUM_EXCEEDANCES` draws are at least as extreme as
    the value and the tail of :data:`TAIL_FRACTION` of the M finite draws
    holds at least that many. Then the p-value is
    (n_tail/M) S(d − t), with n_tail = round(TAIL_FRACTION · M) draws in the
    tail, t midway between the n_tail-th and the next largest distance, d the
    value's distance, and S the survival function of the generalized Pareto
    distribution fitted to the excesses over t of the tail, with its end point
    above :data:`SUPPORT_MARGIN` times d − t. Distances are the values for an
    upper tail, and the absolute differences from the mean of the finite draws
    for two sides. A degenerate tail, such as tied draws, keeps the empirical
    p-value.

    Args:
        value: The observed values, shaped (...).
        draws: The null draws of each value along the last axis, shaped
            (..., M).
        tail: ``"upper"`` or ``"two_sided"``; see :class:`Tail`.
        tail_fit: Whether to replace the empirical p-value by the tail fit
            where the rule above asks for it.

    Returns:
        The p-value, σ_sig, the number of finite draws at least as extreme
        as the value and whether the row is extrapolated, each shaped like
        ``value``. A p-value from the fit is not smaller than the smallest
        normal float64, so σ_sig stays finite. p-value and σ_sig are NaN
        where the value is not finite or no draw is.

    Raises:
        ValueError: If ``tail`` is not a :class:`Tail` value.

    Warns:
        ExtrapolationWarning: If concrete inputs give an extrapolated row. The
            kernel runs under :func:`jax.jit` and :func:`jax.vmap`, but no
            warning is raised for traced inputs.

    References:
        :cite:t:`phipson2010permutation`; :cite:t:`knijnenburg2009fewer`;
        :cite:t:`peschel2026permapprox`; :cite:t:`zhang2009new`.
    """
    side = Tail(tail)
    values = jnp.asarray(value, dtype=jnp.float64)
    null = jnp.asarray(draws, dtype=jnp.float64)
    result = _significance(values, null, side, tail_fit)
    if _is_concrete(result.extrapolated):
        _warn_about_extrapolation(result.extrapolated, null, side)
    return result


def _significance_kernel(
    value: Float[Array, ""],
    draws: Float[Array, " draw"],
    side: Tail,
    tail_fit: bool,
) -> Significance:
    """Return the p-value and σ_sig of one value against its draws."""
    finite = jnp.isfinite(draws)
    count = jnp.sum(finite).astype(jnp.float64)
    if side is Tail.UPPER:
        value_distance, draw_distance = value, draws
    else:
        centre = jnp.sum(jnp.where(finite, draws, 0.0)) / jnp.maximum(count, 1.0)
        value_distance = jnp.abs(value - centre)
        draw_distance = jnp.abs(draws - centre)
    exceedances = jnp.sum(finite & (draw_distance >= value_distance))
    p_value = (1.0 + exceedances) / (1.0 + count)
    fitted = jnp.asarray(False)
    if tail_fit:
        tail_size = jnp.round(TAIL_FRACTION * count)
        # * Non-finite draws sort last, below every distance.
        ranked = -jnp.sort(-jnp.where(finite, draw_distance, -jnp.inf))
        last = draws.shape[0] - 1
        threshold = 0.5 * (
            ranked[jnp.clip(tail_size - 1, 0, last).astype(int)]
            + ranked[jnp.clip(tail_size, 0, last).astype(int)]
        )
        excess_value = value_distance - threshold
        fit = generalized_pareto_fit(
            ranked - threshold,
            jnp.arange(draws.shape[0]) < tail_size,
            bound=SUPPORT_MARGIN * excess_value,
        )
        survival = jnp.exp(
            generalized_pareto_log_survival(excess_value, fit.shape, fit.scale)
        )
        # * An underflowing survival would give σ_sig = NaN; p stays at or above
        # * the smallest normal float.
        tail_p_value = jnp.maximum(
            tail_size / jnp.maximum(count, 1.0) * survival, sys.float_info.min
        )
        fitted = (
            (exceedances < MINIMUM_EXCEEDANCES)
            & (tail_size >= MINIMUM_EXCEEDANCES)
            & jnp.isfinite(value)
            & jnp.isfinite(tail_p_value)
        )
        p_value = jnp.where(fitted, tail_p_value, p_value)
    resolved = jnp.isfinite(value) & (count > 0)
    p_value = jnp.where(resolved, p_value, jnp.nan)
    return Significance(
        p_value=p_value,
        sigma_sig=sigma_sig(p_value, side),
        exceedances=exceedances,
        extrapolated=fitted & resolved,
    )


def _significance_kernel_over_leading_axes(
    values: Float[Array, " *leading"],
    draws: Float[Array, "*leading draw"],
    side: Tail,
    tail_fit: bool,
) -> Significance:
    """Map the kernel over every leading index of the values."""
    draw_count = draws.shape[-1]
    if draw_count == 0:
        nan = jnp.full(values.shape, jnp.nan)
        return Significance(
            nan, nan, jnp.zeros(values.shape, dtype=int), jnp.zeros(values.shape, bool)
        )
    flat = jax.vmap(partial(_significance_kernel, side=side, tail_fit=tail_fit))(
        values.reshape(values.size), draws.reshape(values.size, draw_count)
    )
    return Significance(*(field.reshape(values.shape) for field in flat))


_significance = _jit(_significance_kernel_over_leading_axes, "side", "tail_fit")


def _warn_about_extrapolation(
    extrapolated: Bool[Array, " *leading"],
    draws: Float[Array, "*leading draw"],
    side: Tail,
) -> None:
    """Warn that some significances come from the tail fit, with its resolution."""
    entries = int(jnp.sum(extrapolated))
    if entries == 0:
        return
    # * The smallest finite count among the extrapolated entries sets the resolution.
    draw_count = int(
        jnp.min(
            jnp.where(extrapolated, jnp.sum(jnp.isfinite(draws), axis=-1), draws.size)
        )
    )
    resolution = float(sigma_sig(1.0 / (draw_count + 1), side))
    warnings.warn(
        ExtrapolationWarning(
            f"{entries} of {extrapolated.size} significances come from a generalized"
            f" Pareto fit to the tail of the null, because fewer than"
            f" {MINIMUM_EXCEEDANCES} of the draws reach their values; {draw_count}"
            f" draws alone resolve σ_sig up to {resolution:.2f}"
        ),
        skip_file_prefixes=(_PACKAGE_DIRECTORY,),
    )
