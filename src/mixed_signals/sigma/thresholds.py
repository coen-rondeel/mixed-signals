"""Default effect-size thresholds of the comparison rows, and the verdict of a row.

A comparison row fails when its effect size reaches the threshold of its metric
and its σ_sig reaches :data:`SIGNIFICANCE_LEVEL`. The effect size is σ_eff for
the rows that measure a shift or a distance, and the log of r_eff for the rows
that measure only a difference in width. The thresholds follow the effect
size, not the raw value, so that one default serves every metric. The YAML
config of a pipeline overrides them.

Effect size
    :func:`verdict` takes the absolute σ_eff for ``mean_shift``, which is
    signed, and σ_eff itself for the other σ_eff rows, so a null-subtracted
    value below zero never fails. For ``width_ratio`` it takes the absolute
    value of ln r_eff, since a width off by a factor r or 1/r is as far from
    the reference. For the three shape terms of the joint Gaussian metrics it
    takes the absolute value of ln r_eff with the sign of the subtracted
    value, so a shape term below its null never fails either.
Tails
    The p-value of a row compares its value, before subtraction, with the null
    draws. :func:`comparison_tail` is two-sided for the signed metrics
    (``mean_shift`` and ``width_ratio``, where a departure in either direction
    counts) and upper for the others.
Basis of the defaults
    :data:`SIGMA_EFF_THRESHOLD` is 0.2: the GWTC-2 catalogue (Abbott et al.
    2021, Appendix A) deems a JS divergence of 0.007 bit significant, "which
    for a Gaussian corresponds to a 20% shift in the mean, measured in units of
    one standard deviation". By the zoo quadrature a shift of 0.2σ has a JS
    divergence of 0.0072 bit. Cohen (1988) calls 0.2 a small effect.
    :data:`LOG_R_EFF_THRESHOLD` is 0.14: the width ratio of two Gaussians with
    that JS divergence is 1.153 or 0.867, so abs(ln r) = 0.142, rounded to
    0.14.

References:
    :cite:t:`abbott2021gwtc`; :cite:t:`cohen1988statistical`;
    :cite:t:`phipson2010permutation`; :cite:t:`cowan2011asymptotic`;
    :cite:t:`lin1991divergence`.
"""

import math
from collections.abc import Mapping
from types import MappingProxyType
from typing import Final, NamedTuple

import jax.numpy as jnp
from jax.typing import ArrayLike
from jaxtyping import Float

from mixed_signals.sigma.maps import (
    KERNEL_METRICS,
    QUADRATURE_METRICS,
    SIGMA_EFF_MAPS,
    SLICED_METRICS,
)
from mixed_signals.sigma.significance import Tail

SIGMA_EFF_THRESHOLD: Final = 0.2
"""Default threshold of σ_eff, the Gaussian-equivalent shift in units of σ.

The GWTC-2 catalogue (Abbott et al. 2021, Appendix A) deems a JS divergence of
0.007 bit significant, "which for a Gaussian corresponds to a 20% shift in the
mean, measured in units of one standard deviation"; by the zoo quadrature a
0.2σ shift has a JS divergence of 0.0072 bit. Cohen (1988) calls 0.2 a small
effect.

References:
    :cite:t:`abbott2021gwtc`; :cite:t:`cohen1988statistical`.
"""

LOG_R_EFF_THRESHOLD: Final = 0.14
"""Default threshold of the absolute log of r_eff, the Gaussian-equivalent width ratio.

Two Gaussians with the JS divergence of a 0.2σ shift, 0.0072 bit, differ in
width by a ratio of 1.153 or 0.867, so abs(ln r) = 0.142, rounded to 0.14.

References:
    :cite:t:`abbott2021gwtc`.
"""

SIGNIFICANCE_LEVEL: Final = 2.0
"""σ_sig that a row with a null or a p-value of its own must reach to flag or fail.

The default 200 null draws resolve σ_sig up to 2.58 without a fit to the tail of
the null (Phipson and Smyth 2010; Cowan et al. 2011), so a level of 2 lies
inside their resolution.

References:
    :cite:t:`phipson2010permutation`; :cite:t:`cowan2011asymptotic`.
"""

R_EFF_METRICS: Final = ("width_ratio", "kl_shape", "w2_shape", "hellinger_shape")
"""Metrics whose effect size is r_eff: rows that measure only a difference in width."""

SIGNED_METRICS: Final = ("mean_shift", "width_ratio")
"""Metrics whose departures count in both directions, and whose null is two-sided."""

THRESHOLDS: Final[Mapping[str, float]] = MappingProxyType(
    {
        **dict.fromkeys(
            (
                *SIGMA_EFF_MAPS,
                *KERNEL_METRICS,
                *SLICED_METRICS,
                *QUADRATURE_METRICS,
            ),
            SIGMA_EFF_THRESHOLD,
        ),
        **dict.fromkeys(R_EFF_METRICS, LOG_R_EFF_THRESHOLD),
    }
)
"""Default threshold of the effect size of every comparison metric.

:data:`SIGMA_EFF_THRESHOLD` for each metric with a σ_eff map, and
:data:`LOG_R_EFF_THRESHOLD` for each of :data:`R_EFF_METRICS`.

:meta hide-value:
"""


class Verdict(NamedTuple):
    """The effect size, the resolution and the failure of one comparison row."""

    effect: float
    """The effect size that the threshold acts on; NaN where it is undefined."""
    resolved: bool
    """Whether σ_sig reaches the significance level."""
    failed: bool
    """Whether the effect reaches the threshold and σ_sig the level."""


def comparison_tail(metric: str) -> Tail:
    """Return the tail of the p-value of a comparison metric.

    Args:
        metric: Name of the metric.

    Returns:
        :attr:`~mixed_signals.sigma.significance.Tail.TWO_SIDED` for the
        metrics of :data:`SIGNED_METRICS`, whose departures count in both
        directions, and :attr:`~mixed_signals.sigma.significance.Tail.UPPER`
        for the others.
    """
    return Tail.TWO_SIDED if metric in SIGNED_METRICS else Tail.UPPER


def verdict(
    metric: str,
    *,
    sigma_eff: Float[ArrayLike, ""] | None = None,
    r_eff: Float[ArrayLike, ""] | None = None,
    subtracted: Float[ArrayLike, ""] | None = None,
    sigma_sig: Float[ArrayLike, ""],
    thresholds: Mapping[str, float] = THRESHOLDS,
    level: float = SIGNIFICANCE_LEVEL,
) -> Verdict:
    """Return the effect size and the verdict of one comparison row.

    The effect size is abs(σ_eff) for ``mean_shift``; σ_eff itself for the other
    σ_eff rows, so a negative one never fails; abs(ln r_eff) for
    ``width_ratio``; and sign(subtracted) · abs(ln r_eff) for the shape terms.
    The row fails when the effect size reaches the threshold and σ_sig reaches
    the level. A NaN effect size or a NaN σ_sig never fails.

    Args:
        metric: Name of the metric: a key of ``thresholds``.
        sigma_eff: σ_eff of the row; for the metrics that are not in
            :data:`R_EFF_METRICS`.
        r_eff: r_eff of the row; for the metrics in :data:`R_EFF_METRICS`.
        subtracted: The null-subtracted value of the row, whose sign gives the
            direction of a shape term; ignored for the other metrics.
        sigma_sig: σ_sig of the row.
        thresholds: Threshold of each metric.
        level: σ_sig that the row must reach.

    Returns:
        The effect size, whether σ_sig reaches the level and whether the row
        failed.

    Raises:
        KeyError: If the metric has no threshold.
        ValueError: If the effect size given does not fit the metric: σ_eff
            for an r_eff metric or r_eff for another one, both or neither of
            them, or a shape term without ``subtracted``.

    References:
        :cite:t:`abbott2021gwtc`; :cite:t:`cohen1988statistical`.
    """
    try:
        threshold = thresholds[metric]
    except KeyError:
        raise KeyError(
            f"no threshold for {metric!r}; the thresholds cover"
            f" {', '.join(sorted(thresholds))}"
        ) from None
    effect = _effect(metric, sigma_eff, r_eff, subtracted)
    resolved = _as_float(sigma_sig) >= level
    # * A NaN effect compares false, so it never fails.
    return Verdict(
        effect=effect, resolved=resolved, failed=resolved and effect >= threshold
    )


def _effect(
    metric: str,
    sigma_eff: Float[ArrayLike, ""] | None,
    r_eff: Float[ArrayLike, ""] | None,
    subtracted: Float[ArrayLike, ""] | None,
) -> float:
    """Return the effect size that the threshold of a metric acts on."""
    if metric in R_EFF_METRICS:
        if r_eff is None or sigma_eff is not None:
            raise ValueError(f"the effect size of {metric!r} is r_eff; pass r_eff only")
        ratio = _as_float(r_eff)
        magnitude = (
            abs(math.log(ratio)) if ratio > 0 and math.isfinite(ratio) else math.nan
        )
        if metric == "width_ratio":
            return magnitude
        if subtracted is None:
            raise ValueError(
                f"the shape term {metric!r} needs its subtracted value, whose sign"
                " gives the direction"
            )
        return _sign(_as_float(subtracted)) * magnitude
    if sigma_eff is None or r_eff is not None:
        raise ValueError(
            f"the effect size of {metric!r} is sigma_eff; pass sigma_eff only"
        )
    return abs(_as_float(sigma_eff)) if metric == "mean_shift" else _as_float(sigma_eff)


def _sign(number: float) -> float:
    """Return −1, 0 or 1 by the sign of a number, and NaN for NaN."""
    if math.isnan(number):
        return math.nan
    return float((number > 0) - (number < 0))


def _as_float(number: ArrayLike) -> float:
    """Return a real scalar, such as a JAX scalar array, as a Python float."""
    return float(jnp.asarray(number, dtype=jnp.float64))
