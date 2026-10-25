"""Closed-form σ_eff and r_eff maps of the moment, marginal, kernel and OT metrics.

σ_eff
    The mean shift between two unit Gaussians that gives the same value of a
    metric, in units of the whitening σ. :func:`sigma_eff` inverts each
    metric for that pair: the mean shift, W1 and the W2 total and location
    terms are the shift itself; the Gaussian KL total and location terms give
    √(2 KL); KS and Kuiper's V give 2Φ⁻¹((1 + D)/2), because for a shift F − G
    keeps one sign and V = D; the Hellinger total and location terms give
    √(−8 ln(1 − H²)) (Bhattacharyya 1943; Fukunaga 1990). The kNN rows of
    ``kl`` and ``hellinger`` share these maps.
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
    that the directions span, E + 2C + 3S in the chordal embedding. Circles
    and spheres enter along chords, shorter than arcs, so their shifts read
    low until the calibration on the zoo.
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
    :cite:t:`fukunaga1990introduction`; :cite:t:`nakagawa2015meta`;
    :cite:t:`gretton2012kernel`; :cite:t:`szekely2013energy`;
    :cite:t:`johnson1995continuous`; :cite:t:`janati2020entropic`;
    :cite:t:`peyre2019computational`.
"""

import functools
import math
from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import Final, cast

import jax
import jax.numpy as jnp
from jax.scipy.special import gammaln, ndtri
from jax.typing import ArrayLike
from jaxtyping import Array, Float

_NEWTON_STEPS: Final = 64
"""Newton steps of the KL shape inversion; near zero, where the root is double,
each step at least halves the error, so 64 steps reach float64 resolution."""

_BISECTION_STEPS: Final = 64
"""Bisection steps of the energy inversion: each halves an interval of a
width below the shift, so 64 steps reach float64 resolution."""

_POISSON_TERMS: Final = 8_192
"""Terms of the Poisson mixture of the noncentral χ mean; for rates up to
6 000 the terms left out carry less than e^(−360) of the mass."""

_ASYMPTOTIC_RATE: Final = 6_000.0
"""Rate λ²/2 above which the mean of χ_D(λ) takes its large-λ expansion
λ + (D − 1)/(2λ) − (D − 1)(D − 3)/(8λ³), within 1e-7 of the series there for
D up to 32."""


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


def _root(value: Array) -> Array:
    return jnp.sqrt(jnp.where(value >= 0, value, jnp.nan))


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
        "sinkhorn": _root,
    }
)
"""The closed-form σ_eff map of each metric that needs only its value, by name.

The kernel metrics ``mmd`` and ``energy`` also need the dimension of the
space, and ``mmd`` its bandwidth; :func:`sigma_eff` takes both.

:meta hide-value:
"""

KERNEL_METRICS: Final = ("energy", "mmd")
"""Metrics whose σ_eff map also needs the dimension, and for ``mmd`` the bandwidth."""

SLICED_METRICS: Final = ("sliced_sinkhorn", "sliced_w2")
"""Metrics whose σ_eff map needs the number of coordinates the directions span."""


def sigma_eff(
    metric: str,
    value: Float[ArrayLike, " *leading"],
    *,
    bandwidth: Float[ArrayLike, ""] | None = None,
    dimension: int | None = None,
) -> Float[Array, " *leading"]:
    """Return the Gaussian-equivalent shift σ_eff of metric values.

    Args:
        metric: Name of the metric: a key of :data:`SIGMA_EFF_MAPS`, or one
            of :data:`KERNEL_METRICS` or :data:`SLICED_METRICS`.
        value: Values of the metric, in the units of the whitening; the
            squared MMD for ``mmd`` and SW2² for ``sliced_w2``.
        bandwidth: Kernel width ℓ of ``mmd`` values.
        dimension: For ``mmd`` and ``energy``, the dimension D = E + C + 2S
            of the space; for the sliced metrics, the number of coordinates
            the directions span, E + 2C + 3S.

    Returns:
        σ_eff for each value; NaN outside the range of the metric.

    Raises:
        KeyError: If the metric has no closed-form map.
        ValueError: If a kernel or sliced metric lacks its bandwidth or
            dimension, or the dimension is not a positive integer.

    References:
        :cite:t:`kullback1951information`; :cite:t:`smirnov1948table`;
        :cite:t:`kuiper1960tests`; :cite:t:`bhattacharyya1943measure`;
        :cite:t:`fukunaga1990introduction`; :cite:t:`gretton2012kernel`;
        :cite:t:`szekely2013energy`; :cite:t:`johnson1995continuous`;
        :cite:t:`janati2020entropic`.
    """
    values = jnp.asarray(value, dtype=jnp.float64)
    if metric in SLICED_METRICS:
        if dimension is None:
            raise ValueError(f"the sigma_eff of {metric!r} needs its dimension")
        if isinstance(dimension, bool) or dimension < 1:
            raise ValueError(f"dimension must be a positive integer, got {dimension!r}")
        return _root(dimension * values)
    if metric in KERNEL_METRICS:
        if dimension is None or (metric == "mmd" and bandwidth is None):
            needs = "bandwidth and dimension" if metric == "mmd" else "dimension"
            raise ValueError(f"the sigma_eff of {metric!r} needs its {needs}")
        if isinstance(dimension, bool) or dimension < 1:
            raise ValueError(f"dimension must be a positive integer, got {dimension!r}")
        if metric == "mmd":
            return _mmd_sigma(
                values, jnp.asarray(bandwidth, dtype=jnp.float64), dimension
            )
        return _energy_sigma(values, dimension)
    try:
        mapping = SIGMA_EFF_MAPS[metric]
    except KeyError:
        names = sorted([*SIGMA_EFF_MAPS, *KERNEL_METRICS, *SLICED_METRICS])
        raise KeyError(
            f"no closed-form sigma_eff map for {metric!r}; the maps cover"
            f" {', '.join(names)}"
        ) from None
    return mapping(values)


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


def _jit[**Parameters, Result](
    function: Callable[Parameters, Result], *static_names: str
) -> Callable[Parameters, Result]:
    """Compile a map with :func:`jax.jit` and keep its signature for type checks."""
    return cast(
        Callable[Parameters, Result], jax.jit(function, static_argnames=static_names)
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
