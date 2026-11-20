"""Null subtraction and the σ_eff and r_eff maps of the comparison metrics.

σ_eff
    The mean shift between two unit Gaussians that gives the same value of a
    metric, in units of the whitening σ. :func:`sigma_eff` inverts each
    metric for that pair: the mean shift, W1 and the W2 total and location
    terms are the shift itself; the Gaussian KL total and location terms give
    √(2 KL); KS and Kuiper's V give 2Φ⁻¹((1 + D)/2), because for a shift F − G
    keeps one sign and V = D; the Hellinger total and location terms give
    √(−8 ln(1 − H²)) (Bhattacharyya 1943; Fukunaga 1990). The kNN rows of
    ``hellinger`` share the last map.
Kernel metrics
    Two unit Gaussians in D dimensions shifted by Δ differ by
    x − y ~ N(Δ, 2I), so ‖x − y‖²/2 follows a noncentral χ² with D degrees
    of freedom and noncentrality Δ²/2, and ‖x − y‖/√2 the noncentral χ
    distribution χ_D(Δ/√2) (Johnson, Kotz and Balakrishnan 1995). The
    moment generating function of the noncentral χ² gives the squared MMD of
    the Gaussian kernel of width ℓ (Gretton et al. 2012) as
    2c(1 − exp(−Δ²/(2(ℓ² + 2)))) with c = (ℓ²/(ℓ² + 2))^(D/2), so
    σ_eff = √(−2(ℓ² + 2) ln(1 − MMD²/(2c))). The energy distance is
    2√2 (E[χ_D(Δ/√2)] − E[χ_D(0)]) (Székely and Rizzo 2013), which rises
    monotonically in Δ and is inverted by bisection. The mean of χ_D(λ) is
    the Poisson mixture Σⱼ Pois(j; λ²/2) √2 Γ((D + 1)/2 + j)/Γ(D/2 + j)
    (Johnson, Kotz and Balakrishnan 1995), summed directly because JAX's
    confluent hypergeometric function loses every digit for large λ, and
    replaced by its large-λ expansion, checked against the series, where
    λ²/2 exceeds 6 000. D is the dimension of the space, E + C + 2S for E
    Euclidean axes, C circles and S spheres, whose chordal distances equal
    the Euclidean ones locally.
Optimal transport
    With the squared cost, a shift Δ of any distribution gives the Sinkhorn
    divergence S_ε = Δ² at every ε (Janati et al. 2020, Lemma 1; Peyré and
    Cuturi 2019), so σ_eff = √S_ε for ``sinkhorn`` rows. Over uniform
    directions in n dimensions, the mean squared projection of a shift is
    Δ²/n, so the sliced rows ``sliced_w2``, which hold SW2², and
    ``sliced_sinkhorn`` give σ_eff = √(n · value). n counts the coordinates
    that the directions span, E + 2C + 3S in the chordal embedding. The
    ``sliced_w2`` rows of the ``gaussian`` estimator, the sliced W2 between
    Gaussian fits, take the same map. Circles and spheres enter along chords,
    shorter than arcs, so wide ones read a shift low, as the next paragraph
    describes.
Wide circles and spheres
    The chordal metrics, :data:`CHORDAL_METRICS`, measure circles and spheres
    along chords, so they read a shift low where a circle or a sphere is wide.
    :func:`warn_wide_chordal_axes` gives a row of such a metric a
    :class:`~mixed_signals.errors.ValidityWarning` above a circular standard
    deviation of :data:`CHORDAL_CIRCLE_WIDTH` = 0.3 rad or a sphere width of
    :data:`CHORDAL_SPHERE_WIDTH` = 0.2 rad. In a zoo check with pooled
    whitening, 6 replicates of 2 000 samples and shifts of 0.5, 1 and 2σ,
    σ_eff over the true shift for sliced W2, the Sinkhorn divergence and the
    energy distance was, for wrapped Gaussians of width 0.1 rad, 0.97 to 1.02,
    0.97 to 1.10 and 0.99 to 1.01; at 0.3 rad, 0.96 to 1.02, 0.98 to 1.07 and
    0.96 to 1.01; at 0.5 rad, 0.88 to 0.95, 0.91 to 1.01 and 0.89 to 0.95; at
    1.0 rad, 0.53 to 0.73, 0.55 to 0.85 and 0.60 to 0.77. For von
    Mises-Fisher distributions of width 0.05 rad it was 1.00 to 1.01, 0.99 and
    0.99 to 1.00; at 0.2 rad, 0.95 to 0.96, 0.98 and 0.96 to 0.97; at 0.5 rad,
    0.73 to 0.78, 0.77 to 0.87 and 0.78 to 0.82. The chord of the shift alone,
    (2/(wδ)) sin(wδ/2) for a width w and a shift of δ widths, gives 0.84 to
    0.99 at 1 rad, which falls short of the measured loss, so no closed-form
    correction applies (see :ref:`validation-chordal-widths`).
JS and Anderson-Darling
    The shift δ ≥ 0 for which the JS divergence (Lin 1991) or the
    Anderson-Darling distance (Pettitt 1976; Scholz and Stephens 1987) of
    N(0, 1) and N(δ, 1) equals the value, found by bisection on the
    quadratures of :mod:`mixed_signals.zoo.closed_forms`. JS reaches its upper
    limit ln 2 and the distance its limit 8 ln 2 − 4 to float64 precision near
    δ = 17 and δ = 12, and both are within 1e-6 of it at 10σ, so beyond about
    10σ the maps do not resolve σ_eff.
Kernel density estimates
    A kernel estimate with bandwidth factor f, the factor of Silverman's rule
    that :func:`~mixed_signals.metrics.marginal.shared_bandwidth_factor`
    gives, widens each unit Gaussian to the variance 1 + f², so the marginal
    JS and Hellinger rows read the shift δ/√(1 + f²). With ``estimator="kde"``
    the maps of ``js`` and ``hellinger`` multiply by √(1 + f²) to undo it.
kNN KL
    The kNN estimate of the KL divergence (Wang, Kulkarni and Verdú 2009)
    reads low from four dimensions on in the zoo check of
    :mod:`mixed_signals.sigma.calibration` (see
    :ref:`validation-knn-kl-calibration`), by a ratio that depends on the
    dimension, the sample counts and the shift. With ``estimator="knn"``,
    :func:`sigma_eff` inverts a curve that
    :func:`~mixed_signals.sigma.calibration.knn_kl_curve` measures on unit
    Gaussians for the row.
r_eff
    The ratio of widths along one axis between two unit Gaussians that gives
    the same value, for rows that measure only a difference in width. The
    width ratio is its own r_eff. For the shape terms, :func:`r_eff` solves
    KL(N(0, 1) ‖ N(0, r²)) = ½(1/r² − 1 + ln r²), the W2 distance, 1 − r in
    absolute value, and the Hellinger distance √(1 − √(2r/(1 + r²))) for r.
    Each value has two roots, r and a root below one; the sign of
    ln(det Σ_q/det Σ_p), which the Gaussian divergences of
    :mod:`mixed_signals.metrics.moments` return, chooses r above one when the
    second posterior is wider. The W2 map takes the first posterior to have unit
    width, as a reference whitening makes it. In a pooled whitening the two whitened
    covariances sum to twice the identity, and the W2 shape term of a width
    ratio r along one axis is √2 · abs(r − 1)/√(1 + r²); ``pooled=True`` inverts that
    relation. KL and Hellinger do not depend on the frame.
Null subtraction
    The null mean, the bias of a metric between two draws of one
    distribution, is subtracted before a map applies. :func:`null_subtract`
    subtracts it in the space where a metric is quadratic in the shift between
    two unit Gaussians, :data:`SUBTRACTION_SPACES`: plainly for the metrics
    that already grow as the shift squared (KL, JS, AD, the energy distance,
    MMD², SW2², the sliced Sinkhorn divergence and S_ε); in squares for those
    that grow linearly with the shift (KS, Kuiper's V, W1, W2 and every
    Hellinger form), with the signed root of the result; and in logarithms for
    the width ratio. For a metric that grows linearly with the shift, the
    noise adds to the shift in quadrature, so plain subtraction reads low.
Signs
    The subtracted value of an identical pair is negative for about half the
    draws, so every map is signed: σ_eff is sign(v) times the map of the
    magnitude of v, and r_eff is the reciprocal of the root for a value below
    zero. A value above the range of its metric, such as a KS statistic above
    one, still gives NaN.
Zoo check
    Shifted unit Gaussians with N = 1 000 per set, 30 to 200 replicates and
    true shifts of 0.2, 0.5 and 1.0 gave, after plain subtraction of the
    permutation null mean, KS 0.15 / 0.44 / 0.93, W1 0.15 / 0.45 / 0.94,
    Gaussian W2 with 4 parameters 0.11 / 0.38 / 0.87, KDE Hellinger
    0.12 / 0.40 / 0.89 and Gaussian Hellinger 0.09 / 0.36 / 0.84. Subtracting
    the squares gave KS 0.22 / 0.53 / 1.03, W1 0.20 / 0.50 / 1.00, W2
    0.19 / 0.50 / 1.00, KDE Hellinger 0.20 / 0.51 / 1.01, Gaussian Hellinger
    0.19 / 0.50 / 1.00 and kNN Hellinger 0.14 / 0.49 / 0.99. Plain subtraction
    gave 0.19 to 0.20 / 0.50 / 1.00 for the Gaussian KL, KDE JS, AD, the
    energy distance, MMD² and SW2². KS keeps a small positive bias after the
    subtraction: 0.529 and 1.031 for true shifts of 0.5 and 1.0 at N = 1 000,
    0.508 and 1.009 at N = 10 000. With the factor √(1 + f²), f = 0.266 at
    N = 1 000, KDE JS gave 0.201 / 0.502 / 1.003 and KDE Hellinger
    0.203 / 0.505 / 1.010. For identical inputs 46 to 57% of the subtracted
    values were negative. At zero shift (N = 1 000) the signed map gave a mean
    σ_eff of +0.007 for the energy distance (scatter 0.069) and 0.000 for W1
    (scatter 0.046), and clipping at zero gave +0.035 and +0.021 (see
    :ref:`validation-effect-recovery` and :ref:`validation-signed-maps`).
Domains
    Each map is the inverse of its metric for that pair, so a value outside
    the metric's range, such as a KS statistic above one, gives NaN: KS and
    Hellinger above 1, MMD² above 2c, JS above ln 2 and AD above 8 ln 2 − 4.
    Values below zero are not outside the range, as the paragraph on signs
    says.

References:
    :cite:t:`kullback1951information`; :cite:t:`dowson1982frechet`;
    :cite:t:`givens1984class`; :cite:t:`vallender1974calculation`;
    :cite:t:`smirnov1948table`; :cite:t:`kuiper1960tests`;
    :cite:t:`hellinger1909neue`; :cite:t:`bhattacharyya1943measure`;
    :cite:t:`fukunaga1990introduction`; :cite:t:`nakagawa2015meta`;
    :cite:t:`gretton2012kernel`; :cite:t:`szekely2013energy`;
    :cite:t:`johnson1995continuous`; :cite:t:`janati2020entropic`;
    :cite:t:`peyre2019computational`; :cite:t:`lin1991divergence`;
    :cite:t:`pettitt1976two`; :cite:t:`scholz1987k`;
    :cite:t:`silverman1986density`; :cite:t:`wang2009divergence`.
"""

import functools
import math
from collections.abc import Callable, Mapping
from enum import StrEnum
from types import MappingProxyType
from typing import Final, cast

import jax
import jax.numpy as jnp
from jax.scipy.special import gammaln, ndtri
from jax.typing import ArrayLike
from jaxtyping import Array, Float

from mixed_signals.core.space import ParamSpace
from mixed_signals.estimators.base import _warn
from mixed_signals.sigma.calibration import CalibrationCurve
from mixed_signals.transforms.whiten import Whitening
from mixed_signals.zoo.closed_forms import normal_anderson_darling, normal_js

_NEWTON_STEPS: Final = 64
"""Newton steps of the KL shape inversion; near zero, where the root is double,
each step at least halves the error, so 64 steps reach float64 resolution."""

_BISECTION_STEPS: Final = 64
"""Bisection steps of the energy, JS and AD inversions: each halves an interval
of a width below the shift, so 64 steps reach float64 resolution."""

_SATURATION_SHIFT: Final = 20.0
"""Upper end of the shift interval of the JS and AD inversions.

The quadratures reach their limits to float64 precision near 17 and 12, so a
larger shift adds no resolution.
"""

_POISSON_TERMS: Final = 8_192
"""Terms of the Poisson mixture of the noncentral χ mean; for rates up to
6 000 the terms left out carry less than e^(−360) of the mass."""

_ASYMPTOTIC_RATE: Final = 6_000.0
"""Rate λ²/2 above which the mean of χ_D(λ) takes its large-λ expansion
λ + (D − 1)/(2λ) − (D − 1)(D − 3)/(8λ³), within 1e-7 of the series there for
D up to 32."""

_QUADRATURE_LIMITS: Final[Mapping[str, float]] = MappingProxyType(
    {"ad": 8 * math.log(2) - 4, "js": math.log(2)}
)
"""Largest value of the JS divergence and the AD distance of two normals."""

_KDE_METRICS: Final = ("hellinger", "js")
"""Metrics whose rows can come from a marginal kernel density estimate."""


class SubtractionSpace(StrEnum):
    """The space in which :func:`null_subtract` subtracts the mean of the null.

    ``linear``
        The value and the draws as they are, for metrics that already grow as
        the shift squared.
    ``squared``
        The signed squares, with the signed root of the difference, for
        metrics that grow linearly with the shift.
    ``log``
        Logarithms, for the width ratio.
    """

    LINEAR = "linear"
    SQUARED = "squared"
    LOG = "log"


SUBTRACTION_SPACES: Final[Mapping[str, SubtractionSpace]] = MappingProxyType(
    {
        **dict.fromkeys(
            (
                "mean_shift",
                "kl",
                "kl_location",
                "kl_shape",
                "js",
                "ad",
                "energy",
                "mmd",
                "sinkhorn",
                "sliced_w2",
                "sliced_sinkhorn",
            ),
            SubtractionSpace.LINEAR,
        ),
        **dict.fromkeys(
            (
                "ks",
                "kuiper_ks",
                "w1",
                "w2",
                "w2_location",
                "w2_shape",
                "hellinger",
                "hellinger_location",
                "hellinger_shape",
            ),
            SubtractionSpace.SQUARED,
        ),
        "width_ratio": SubtractionSpace.LOG,
    }
)
"""The space of the null subtraction of each metric, by name.

:meta hide-value:
"""


def _jit[**Parameters, Result](
    function: Callable[Parameters, Result], *static_names: str
) -> Callable[Parameters, Result]:
    """Compile a map with :func:`jax.jit` and keep its signature for type checks."""
    return cast(
        Callable[Parameters, Result], jax.jit(function, static_argnames=static_names)
    )


def null_subtract(
    metric: str,
    value: Float[ArrayLike, " *leading"],
    draws: Float[ArrayLike, "*leading draw"],
) -> Float[Array, " *leading"]:
    """Subtract the mean of the null draws from metric values.

    The space of the subtraction follows the metric, :data:`SUBTRACTION_SPACES`.
    Linear: v − mean(d). Squared: t = v·abs(v) − mean(d·abs(d)), and the result
    is sign(t)·√abs(t); the signed square keeps values that can be negative,
    such as 1 − BC of the kNN Hellinger rows, with BC the Bhattacharyya
    coefficient. Log: exp(ln v − mean(ln d)),
    where a non-positive value gives NaN and non-positive draws leave the
    mean. Draws that are not finite leave the mean in every space.

    Args:
        metric: Name of the metric: a key of :data:`SUBTRACTION_SPACES`.
        value: The observed values, shaped (...).
        draws: The null draws of each value along the last axis, shaped
            (..., M).

    Returns:
        The null-subtracted values, shaped like ``value``; NaN where the
        value is not finite or no draw is usable. The function runs under
        :func:`jax.jit` and :func:`jax.vmap`.

    Raises:
        KeyError: If the metric has no subtraction space.

    References:
        See the zoo check in the module description and
        :ref:`validation-effect-recovery`.
    """
    try:
        space = SUBTRACTION_SPACES[metric]
    except KeyError:
        raise KeyError(
            f"no null subtraction for {metric!r}; the spaces cover"
            f" {', '.join(sorted(SUBTRACTION_SPACES))}"
        ) from None
    return _null_subtract(
        space,
        jnp.asarray(value, dtype=jnp.float64),
        jnp.asarray(draws, dtype=jnp.float64),
    )


def _null_subtract_kernel(
    space: SubtractionSpace,
    values: Float[Array, " *leading"],
    draws: Float[Array, "*leading draw"],
) -> Float[Array, " *leading"]:
    """Subtract the mean of the usable draws in the given space."""
    usable = jnp.isfinite(draws)
    if space is SubtractionSpace.LOG:
        usable = usable & (draws > 0)
    count = jnp.sum(usable, axis=-1)
    if space is SubtractionSpace.LINEAR:
        transformed = draws
    elif space is SubtractionSpace.SQUARED:
        transformed = draws * jnp.abs(draws)
    else:
        transformed = jnp.log(jnp.where(usable, draws, 1.0))
    mean = jnp.sum(jnp.where(usable, transformed, 0.0), axis=-1) / jnp.maximum(count, 1)
    if space is SubtractionSpace.LINEAR:
        subtracted = values - mean
    elif space is SubtractionSpace.SQUARED:
        difference = values * jnp.abs(values) - mean
        subtracted = jnp.sign(difference) * jnp.sqrt(jnp.abs(difference))
    else:
        positive = values > 0
        subtracted = jnp.where(
            positive, jnp.exp(jnp.log(jnp.where(positive, values, 1.0)) - mean), jnp.nan
        )
    return jnp.where(jnp.isfinite(values) & (count > 0), subtracted, jnp.nan)


_null_subtract = _jit(_null_subtract_kernel, "space")


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


def _root(value: Array) -> Array:
    return jnp.sqrt(jnp.where(value >= 0, value, jnp.nan))


SIGMA_EFF_MAPS: Final[Mapping[str, Callable[[Array], Array]]] = MappingProxyType(
    {
        "w1": _distance,
        "w2": _distance,
        "w2_location": _distance,
        "kl": _kl_sigma,
        "kl_location": _kl_sigma,
        "ks": _ks_sigma,
        "kuiper_ks": _ks_sigma,
        "hellinger": _hellinger_sigma,
        "hellinger_location": _hellinger_sigma,
        "sinkhorn": _root,
    }
)
"""The closed-form σ_eff map of each metric that needs only its value, by name.

The maps take the magnitude of a value; :func:`sigma_eff` restores the sign.
``mean_shift`` has no entry: its value is σ_eff, and :func:`sigma_eff` returns
it before it reads the maps. The kernel metrics ``mmd`` and ``energy`` also
need the dimension of the space, and ``mmd`` its bandwidth; :func:`sigma_eff`
takes both.

:meta hide-value:
"""

KERNEL_METRICS: Final = ("energy", "mmd")
"""Metrics whose σ_eff map also needs the dimension, and for ``mmd`` the bandwidth."""

SLICED_METRICS: Final = ("sliced_sinkhorn", "sliced_w2")
"""Metrics whose σ_eff map needs the number of coordinates the directions span."""

CHORDAL_METRICS: Final = ("energy", "mmd", "sinkhorn", "sliced_sinkhorn", "sliced_w2")
"""Metrics that measure circles and spheres along chords and read wide ones low.

In a zoo check with pooled whitening, 6 replicates of 2 000 samples and shifts
of 0.5, 1 and 2σ, σ_eff over the true shift for sliced W2, the Sinkhorn
divergence and the energy distance fell with the width of wrapped Gaussians
from 0.96 to 1.07 at 0.3 rad to 0.88 to 1.01 at 0.5 rad and 0.53 to 0.85 at
1.0 rad, and with the width of von Mises-Fisher distributions from 0.95 to
0.98 at 0.2 rad to 0.73 to 0.87 at 0.5 rad (see
:ref:`validation-chordal-widths`).
"""

CHORDAL_CIRCLE_WIDTH: Final = 0.3
"""Circular standard deviation in radians above which a chordal metric warns.

At 0.1 and 0.3 rad, σ_eff over the true shift of wrapped Gaussians was 0.97 to
1.02 and 0.96 to 1.02 for sliced W2, 0.97 to 1.10 and 0.98 to 1.07 for the
Sinkhorn divergence, and 0.99 to 1.01 and 0.96 to 1.01 for the energy
distance; at 0.5 rad it was 0.88 to 0.95, 0.91 to 1.01 and 0.89 to 0.95, and
at 1.0 rad 0.53 to 0.73, 0.55 to 0.85 and 0.60 to 0.77. The chord of the shift
alone explains only part of the loss at 1 rad, so no closed-form correction
applies.
"""

CHORDAL_SPHERE_WIDTH: Final = 0.2
"""Angular width in radians above which a chordal metric warns about a sphere.

At 0.05 and 0.2 rad, σ_eff over the true shift of von Mises-Fisher
distributions was 1.00 to 1.01 and 0.95 to 0.96 for sliced W2, 0.99 and 0.98
for the Sinkhorn divergence, and 0.99 to 1.00 and 0.96 to 0.97 for the energy
distance; at 0.5 rad it was 0.73 to 0.78, 0.77 to 0.87 and 0.78 to 0.82.
"""

QUADRATURE_METRICS: Final = ("ad", "js")
"""Metrics whose σ_eff map inverts a quadrature of the zoo by bisection."""


def sigma_eff(
    metric: str,
    value: Float[ArrayLike, " *leading"],
    *,
    estimator: str | None = None,
    bandwidth: Float[ArrayLike, ""] | None = None,
    dimension: int | None = None,
    curve: CalibrationCurve | None = None,
) -> Float[Array, " *leading"]:
    """Return the Gaussian-equivalent shift σ_eff of null-subtracted metric values.

    Every map but that of ``mean_shift``, which is signed already, applies to
    the magnitude of the value and gives the result the sign of the value, so
    a subtracted value below zero gives a negative σ_eff. A magnitude above
    the range of the metric gives NaN: KS and Hellinger above 1, MMD² above
    2c, JS above ln 2 and AD above 8 ln 2 − 4. The JS and AD maps invert the
    quadratures of :mod:`mixed_signals.zoo.closed_forms` by bisection on
    shifts up to 20; both quadratures saturate beyond about 10σ, where σ_eff
    is not resolved.

    Args:
        metric: Name of the metric: ``mean_shift``, a key of
            :data:`SIGMA_EFF_MAPS`, or one of :data:`KERNEL_METRICS`,
            :data:`SLICED_METRICS` or :data:`QUADRATURE_METRICS`.
        value: Null-subtracted values of the metric, in the units of the
            whitening; the squared MMD for ``mmd`` and SW2² for
            ``sliced_w2``.
        estimator: Name of the estimator of the row. ``"kde"``, valid for
            ``js`` and ``hellinger`` only, multiplies the map by
            √(1 + f²) for the bandwidth factor f, because the kernel estimate
            widens each unit Gaussian to the variance 1 + f². ``"knn"`` with
            ``kl`` inverts ``curve``. Any other name, such as ``"empirical"``
            or ``"gaussian"``, and ``None``, leave the maps as they are.
        bandwidth: For ``mmd``, the kernel width ℓ; for ``estimator="kde"``,
            the bandwidth factor f of the row.
        dimension: For ``mmd`` and ``energy``, the dimension D = E + C + 2S
            of the space; for the sliced metrics, the number of coordinates
            the directions span, E + 2C + 3S.
        curve: For ``kl`` with ``estimator="knn"``, the calibration curve of
            :func:`~mixed_signals.sigma.calibration.knn_kl_curve`.

    Returns:
        σ_eff for each value; NaN outside the range of the metric.

    Raises:
        KeyError: If the metric has no map.
        ValueError: If a kernel or sliced metric lacks its bandwidth or
            dimension, the dimension is not a positive integer, the ``kde``
            estimator comes with another metric or without a bandwidth, or
            ``kl`` with the ``knn`` estimator comes without a curve.

    References:
        :cite:t:`kullback1951information`; :cite:t:`smirnov1948table`;
        :cite:t:`kuiper1960tests`; :cite:t:`bhattacharyya1943measure`;
        :cite:t:`fukunaga1990introduction`; :cite:t:`gretton2012kernel`;
        :cite:t:`szekely2013energy`; :cite:t:`johnson1995continuous`;
        :cite:t:`janati2020entropic`; :cite:t:`lin1991divergence`;
        :cite:t:`pettitt1976two`; :cite:t:`scholz1987k`;
        :cite:t:`silverman1986density`; :cite:t:`wang2009divergence`.
    """
    values = jnp.asarray(value, dtype=jnp.float64)
    if estimator == "kde" and metric not in _KDE_METRICS:
        raise ValueError(
            f"the kde estimator has a sigma_eff map for {', '.join(_KDE_METRICS)}"
            f" only, got {metric!r}"
        )
    if metric == "mean_shift":
        return values
    magnitudes = jnp.abs(values)
    mapped = _magnitude_sigma(
        metric, magnitudes, estimator, bandwidth, dimension, curve
    )
    return jnp.sign(values) * mapped


def _magnitude_sigma(
    metric: str,
    magnitudes: Float[Array, " *leading"],
    estimator: str | None,
    bandwidth: Float[ArrayLike, ""] | None,
    dimension: int | None,
    curve: CalibrationCurve | None,
) -> Float[Array, " *leading"]:
    """Return the σ_eff map of non-negative values, after reading the settings."""
    if metric in SLICED_METRICS:
        _check_dimension(metric, dimension, needs_bandwidth=False, bandwidth=None)
        return _root(cast(int, dimension) * magnitudes)
    if metric in KERNEL_METRICS:
        _check_dimension(
            metric, dimension, needs_bandwidth=metric == "mmd", bandwidth=bandwidth
        )
        if metric == "mmd":
            return _mmd_sigma(
                magnitudes,
                jnp.asarray(bandwidth, dtype=jnp.float64),
                cast(int, dimension),
            )
        return _energy_sigma(magnitudes, cast(int, dimension))
    if metric in QUADRATURE_METRICS:
        mapped = _quadrature_sigma(magnitudes, metric)
    elif metric == "kl" and estimator == "knn":
        if curve is None:
            raise ValueError(
                "the sigma_eff of 'kl' with the knn estimator needs its calibration"
                " curve, from knn_kl_curve"
            )
        return _curve_sigma(magnitudes, curve)
    else:
        try:
            mapped = SIGMA_EFF_MAPS[metric](magnitudes)
        except KeyError:
            names = sorted(
                [
                    "mean_shift",
                    *SIGMA_EFF_MAPS,
                    *KERNEL_METRICS,
                    *SLICED_METRICS,
                    *QUADRATURE_METRICS,
                ]
            )
            raise KeyError(
                f"no closed-form sigma_eff map for {metric!r}; the maps cover"
                f" {', '.join(names)}"
            ) from None
    if estimator == "kde":
        if bandwidth is None:
            raise ValueError(
                f"the sigma_eff of {metric!r} with the kde estimator needs its"
                " bandwidth factor"
            )
        return jnp.sqrt(1 + jnp.asarray(bandwidth, dtype=jnp.float64) ** 2) * mapped
    return mapped


def _check_dimension(
    metric: str,
    dimension: int | None,
    *,
    needs_bandwidth: bool,
    bandwidth: Float[ArrayLike, ""] | None,
) -> None:
    """Raise ValueError unless a kernel or sliced metric has its settings."""
    if dimension is None or (needs_bandwidth and bandwidth is None):
        needs = "bandwidth and dimension" if needs_bandwidth else "dimension"
        raise ValueError(f"the sigma_eff of {metric!r} needs its {needs}")
    if isinstance(dimension, bool) or dimension < 1:
        raise ValueError(f"dimension must be a positive integer, got {dimension!r}")


def warn_wide_chordal_axes(
    metric: str, whitening: Whitening, space: ParamSpace
) -> None:
    """Warn if a circle or sphere is too wide for a chordal metric to read shifts well.

    Does nothing for a metric outside :data:`CHORDAL_METRICS`. Otherwise it
    raises one :class:`~mixed_signals.errors.ValidityWarning` that names every
    circle whose circular standard deviation exceeds
    :data:`CHORDAL_CIRCLE_WIDTH` and every sphere whose angular width exceeds
    :data:`CHORDAL_SPHERE_WIDTH`, each with its width in radians. The widths
    are the scales that the whitening stores: for a circle of period P and
    scale s, in the units of the circle, the width is 2π s/P rad, and a
    sphere's scale is its width in radians already. The function reads
    concrete values, so it runs outside ``jax.jit``, once per row after
    :meth:`~mixed_signals.transforms.whiten.Whitening.from_samples` or
    :meth:`~mixed_signals.transforms.whiten.Whitening.from_fisher`.

    Args:
        metric: Name of the metric of the row.
        whitening: Whitening of the comparison.
        space: Space of the original coordinates, before the whitening, whose
            circle bounds give the periods.

    Warns:
        ValidityWarning: If a circle or a sphere is wider than its limit.

    Raises:
        ValueError: If the space has other axis blocks than the whitening.

    References:
        See the zoo check of the wide circles and spheres in the module
        description and :ref:`validation-chordal-widths`.
    """
    if metric not in CHORDAL_METRICS:
        return
    if space.axis_blocks != whitening.blocks:
        raise ValueError(
            f"the space has the axis blocks {space.axis_blocks}, but the"
            f" whitening was built for {whitening.blocks}"
        )
    wide: list[str] = []
    for position, scale in zip(
        whitening.blocks.circles, whitening.circle_scales, strict=True
    ):
        parameter = space.parameters[position]
        lower, upper = parameter.bounds
        width = 2 * math.pi * float(scale) / (upper - lower)
        if width > CHORDAL_CIRCLE_WIDTH:
            wide.append(f"circle {parameter.name!r} ({width:.2f} rad)")
    for sphere, scale in zip(space.spheres, whitening.sphere_scales, strict=True):
        width = float(scale)
        if width > CHORDAL_SPHERE_WIDTH:
            wide.append(
                f"sphere ({sphere.azimuth!r}, {sphere.polar!r}) ({width:.2f} rad)"
            )
    if wide:
        _warn(
            f"{metric}: chordal distances read shifts low on wide circles and"
            f" spheres, above {CHORDAL_CIRCLE_WIDTH} rad for a circle and"
            f" {CHORDAL_SPHERE_WIDTH} rad for a sphere; wide here: {', '.join(wide)}"
        )


def r_eff(
    metric: str,
    value: Float[ArrayLike, " *leading"],
    log_det_ratio: Float[ArrayLike, " *leading"] | None = None,
    *,
    pooled: bool = False,
) -> Float[Array, " *leading"]:
    """Return the Gaussian-equivalent width ratio r_eff of null-subtracted values.

    The shape terms are signed: for a magnitude v of the value, r above one
    where ``log_det_ratio`` is positive and below one otherwise, a value
    below zero gives the reciprocal of that root, so that
    ln r_eff = sign(value) · ln r(v).

    Args:
        metric: ``"width_ratio"``, ``"kl_shape"``, ``"w2_shape"`` or
            ``"hellinger_shape"``.
        value: Null-subtracted values of the metric; for ``width_ratio``, the
            subtracted ratio.
        log_det_ratio: ln(det Σ_q/det Σ_p) of each row, whose sign chooses
            the root above or below one; needed for every shape term.
        pooled: Whether the values come from a pooled whitening, in which the
            whitened covariances A and B satisfy A + B = 2I. They commute, so
            the W2 distance separates per axis, and for a width ratio r along
            one axis the W2 shape term is g = √2 · abs(r − 1)/√(1 + r²); ``w2_shape``
            then solves that relation instead of g = abs(r − 1). The other
            metrics ignore it.

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
    shape_map = (
        _w2_shape_pooled_ratio
        if pooled and metric == "w2_shape"
        else _SHAPE_MAPS[metric]
    )
    root = shape_map(jnp.abs(values), wider)
    return jnp.where(values < 0, 1 / root, root)


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


def _w2_shape_pooled_ratio(value: Array, wider: Array) -> Array:
    """Return r with √2 · abs(r − 1)/√(1 + r²) = value, above one where ``wider``."""
    inside = (value >= 0) & (value < math.sqrt(2))
    magnitude = jnp.where(inside, value, 0.0)
    # * The wide root has no cancellation; (1 − spread) would lose digits towards √2.
    spread = magnitude * jnp.sqrt(1 - magnitude**2 / 4)
    root = (1 + spread) / (1 - magnitude**2 / 2)
    return jnp.where(inside, jnp.where(wider, root, 1 / root), jnp.nan)


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


def _mmd_sigma(value: Array, bandwidth: Array, dimension: int) -> Array:
    """Return Δ with 2c(1 − exp(−Δ²/(2(ℓ² + 2)))) = MMD², c = (ℓ²/(ℓ² + 2))^(D/2)."""
    squared_width = bandwidth**2
    scale = (squared_width / (squared_width + 2)) ** (0.5 * dimension)
    ratio = value / (2 * scale)
    inside = (value >= 0) & (ratio < 1)
    # * log1p keeps the digits of small squared discrepancies.
    log_term = jnp.log1p(-jnp.where(inside, ratio, 0.0))
    return jnp.where(inside, jnp.sqrt(-2 * (squared_width + 2) * log_term), jnp.nan)


def _central_chi_mean(dimension: int) -> float:
    """Return E[χ_D(0)] = √2 Γ((D + 1)/2)/Γ(D/2)."""
    return math.sqrt(2) * math.exp(
        math.lgamma(0.5 * (dimension + 1)) - math.lgamma(0.5 * dimension)
    )


def _chi_mean_excess(rate: Array, dimension: int) -> Array:
    """Return E[χ_D(λ)] − E[χ_D(0)] for the rate λ²/2, from the Poisson mixture."""
    base = _central_chi_mean(dimension)
    terms = jnp.arange(1, _POISSON_TERMS, dtype=jnp.float64)
    # * Each term carries m_j − m_0 > 0, so small rates keep their digits.
    excess_means = (
        jnp.exp(
            0.5 * math.log(2)
            + gammaln(0.5 * (dimension + 1) + terms)
            - gammaln(0.5 * dimension + terms)
        )
        - base
    )
    safe_rate = jnp.where(rate > 0, rate, 1.0)[..., None]
    log_weights = -safe_rate + terms * jnp.log(safe_rate) - gammaln(terms + 1)
    series = jnp.sum(jnp.exp(log_weights) * excess_means, axis=-1)
    noncentrality = jnp.sqrt(2 * jnp.where(rate > 0, rate, 1.0))
    expansion = (
        noncentrality
        + (dimension - 1) / (2 * noncentrality)
        - (dimension - 1) * (dimension - 3) / (8 * noncentrality**3)
        - base
    )
    return jnp.where(
        rate > 0, jnp.where(rate > _ASYMPTOTIC_RATE, expansion, series), 0.0
    )


def _energy_of_shift(shift: Array, dimension: int) -> Array:
    """Return 2√2 (E[χ_D(Δ/√2)] − E[χ_D(0)]), the energy distance of a shift Δ."""
    return 2 * math.sqrt(2) * _chi_mean_excess(0.25 * shift**2, dimension)


def _bisection_step(
    _: Array,
    bounds: tuple[Array, Array],
    target: Array,
    dimension: int,
) -> tuple[Array, Array]:
    """Halve the interval that holds the shift whose energy distance is the target."""
    lower, upper = bounds
    middle = 0.5 * (lower + upper)
    below = _energy_of_shift(middle, dimension) < target
    return jnp.where(below, middle, lower), jnp.where(below, upper, middle)


def _energy_sigma_kernel(value: Array, dimension: int) -> Array:
    """Return Δ with 2√2 (E[χ_D(Δ/√2)] − E[χ_D(0)]) = value, by bisection."""
    inside = value >= 0
    target = jnp.where(inside, value, 0.0)
    # * E‖z + μ‖ lies between ‖μ‖ and ‖μ‖ + E‖z‖, which brackets Δ.
    lower = 0.5 * target
    upper = lower + math.sqrt(2) * _central_chi_mean(dimension)
    lower, upper = jax.lax.fori_loop(
        0,
        _BISECTION_STEPS,
        functools.partial(_bisection_step, target=target, dimension=dimension),
        (lower, upper),
    )
    shift = jnp.where(target == 0, 0.0, 0.5 * (lower + upper))
    return jnp.where(inside, shift, jnp.nan)


_energy_sigma = _jit(_energy_sigma_kernel, "dimension")


def _quadrature_shift(value: Float[Array, ""], metric: str) -> Float[Array, ""]:
    """Return the shift δ whose JS or AD quadrature equals the value, by bisection."""
    quadrature = normal_js if metric == "js" else normal_anderson_darling
    inside = (value >= 0) & (value <= _QUADRATURE_LIMITS[metric])
    target = jnp.where(inside, value, 0.0)
    lower, upper = jax.lax.fori_loop(
        0,
        _BISECTION_STEPS,
        functools.partial(_quadrature_step, target=target, quadrature=quadrature),
        (jnp.zeros_like(target), jnp.full_like(target, _SATURATION_SHIFT)),
    )
    shift = jnp.where(target == 0, 0.0, 0.5 * (lower + upper))
    return jnp.where(inside, shift, jnp.nan)


def _quadrature_step(
    _: Array,
    bounds: tuple[Array, Array],
    target: Array,
    quadrature: Callable[[float, float, Array, float], Array],
) -> tuple[Array, Array]:
    """Halve the interval that holds the shift whose quadrature equals the target."""
    lower, upper = bounds
    middle = 0.5 * (lower + upper)
    below = quadrature(0.0, 1.0, middle, 1.0) < target
    return jnp.where(below, middle, lower), jnp.where(below, upper, middle)


def _quadrature_sigma_kernel(
    values: Float[Array, " *leading"], metric: str
) -> Float[Array, " *leading"]:
    """Map the scalar bisection over every index of the values."""
    shifts: Array = jnp.vectorize(functools.partial(_quadrature_shift, metric=metric))(
        values
    )
    return shifts


_quadrature_sigma = _jit(_quadrature_sigma_kernel, "metric")


def _curve_sigma(
    magnitudes: Float[Array, " *leading"], curve: CalibrationCurve
) -> Float[Array, " *leading"]:
    """Invert a calibration curve, carrying its last ratio beyond its last point."""
    shifts = jnp.asarray(curve.shifts, dtype=jnp.float64)
    curve_values = jnp.asarray(curve.values, dtype=jnp.float64)
    if shifts.ndim != 1 or shifts.shape != curve_values.shape or shifts.size < 2:
        raise ValueError(
            "the calibration curve needs shifts and values of one shape (grid,)"
            f" with at least two points, got {shifts.shape} and {curve_values.shape}"
        )
    last_shift, last_value = shifts[-1], curve_values[-1]
    inside = jnp.interp(magnitudes, curve_values, shifts)
    # * Past the last point the shift grows as √value, as for a quadratic metric.
    safe_last = jnp.where(last_value > 0, last_value, 1.0)
    beyond = last_shift * jnp.sqrt(magnitudes / safe_last)
    mapped = jnp.where(magnitudes <= last_value, inside, beyond)
    return jnp.where(last_value > 0, mapped, jnp.nan)
