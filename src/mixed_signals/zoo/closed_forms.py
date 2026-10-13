"""Reference values of the validation zoo, in closed form or by quadrature.

Every function returns a population value for distributions with known
parameters as a float64 JAX array, so the values compute under
:func:`jax.jit`, map over batches with :func:`jax.vmap` and differentiate
with :func:`jax.grad`. Called with concrete inputs, a function checks them and
raises ValueError for invalid ones; inside ``jax.jit`` the values are unknown,
and invalid inputs give NaN.

Gaussian divergences
    The divergences between two Gaussians split into a location term, which
    only the means change, and a shape term, which only the covariances
    change. The KL divergence is their sum (Kullback and Leibler 1951;
    Rasmussen and Williams 2006). The squared W2 distance is the squared
    Euclidean distance between the means plus the squared Bures distance
    between the covariances (Dowson and Landau 1982; Givens and Shortt 1984).
    The Hellinger distance follows from the Bhattacharyya distance D_B as
    H² = 1 − exp(−D_B), and D_B is a location term plus a shape term
    (Hellinger 1909; Bhattacharyya 1943; Fukunaga 1990).
One-dimensional normals
    The Kolmogorov-Smirnov distance, the largest absolute gap between the two
    distribution functions, is reached where the two densities cross
    (Smirnov 1948). The W1 distance is the integral of the absolute
    difference of the quantile functions (Vallender 1974), which for two
    normals is the mean of a folded normal (Leone, Nelson and Nottingham
    1961). The Jensen-Shannon divergence in nats (Lin 1991) and the
    Anderson-Darling distance ∫ (F − G)² / (H(1 − H)) dH, with
    H = (F + G)/2, need quadrature. The second is the limit of the two-sample
    Anderson-Darling statistic divided by nm/(n + m) for equal sample sizes
    (Pettitt 1976; Scholz and Stephens 1987).
Quadrature
    Integrals use composite 8-point Gauss-Legendre rules (Davis and
    Rabinowitz 1984) on panels that the distributions define. Each normal
    component adds panels of width σ/2 out to 12σ from its mean, where its
    density falls below 1e-31 of its peak. Each point where two weighted
    component densities cross adds panels at the width of that crossing,
    over which a log-density switches from one component to the other. Every
    panel is then small against every feature of the integrand, also for
    components of very different widths, and the arrays keep fixed shapes, as
    ``jax.jit`` and ``jax.vmap`` need. The tests check the values against
    adaptive QUADPACK (Piessens et al. 1983) through SciPy
    (Virtanen et al. 2020), given the same break points; without them,
    QUADPACK can step over a narrow component without noticing.

References:
    :cite:t:`kullback1951information`; :cite:t:`rasmussen2006gaussian`;
    :cite:t:`dowson1982frechet`; :cite:t:`givens1984class`;
    :cite:t:`hellinger1909neue`; :cite:t:`bhattacharyya1943measure`;
    :cite:t:`fukunaga1990introduction`; :cite:t:`smirnov1948table`;
    :cite:t:`vallender1974calculation`; :cite:t:`leone1961folded`;
    :cite:t:`lin1991divergence`; :cite:t:`pettitt1976two`;
    :cite:t:`scholz1987k`; :cite:t:`davis1984methods`;
    :cite:t:`piessens1983quadpack`; :cite:t:`virtanen2020scipy`.
"""

import functools
import math
from collections.abc import Callable
from typing import Final, NamedTuple, cast

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.linalg import solve_triangular
from jax.scipy.special import erf, log_ndtr, logsumexp, ndtr
from jax.typing import ArrayLike
from jaxtyping import Array, Float

from mixed_signals.zoo._checks import require, require_covariance

QUADRATURE_REACH: Final = 12.0
"""Standard deviations from a component's mean to the end of its panels."""

PANELS_PER_FEATURE: Final = 48
"""Panels across ±12 widths of every component or crossing: each spans half a width."""

GAUSS_LEGENDRE_ORDER: Final = 8
"""Nodes of the Gauss-Legendre rule on every panel."""

SMALL_CONCENTRATION: Final = 1e-2
"""Concentration below which the mean resultant length uses its Taylor series."""

CROSSING_RESOLUTION: Final = 1e-3
"""Narrowest crossing width, as a fraction of the narrower of its two components."""

_HALF_LOG_TWO_PI: Final = 0.5 * math.log(2 * math.pi)
# * Fixed node table: the panels move with the distributions, the rule does not.
_LEGENDRE_NODES, _LEGENDRE_WEIGHTS = np.polynomial.legendre.leggauss(
    GAUSS_LEGENDRE_ORDER
)
_PANEL_OFFSETS: Final = np.linspace(
    -QUADRATURE_REACH, QUADRATURE_REACH, PANELS_PER_FEATURE + 1
)


def _jit[**Parameters, Result](
    function: Callable[Parameters, Result],
) -> Callable[Parameters, Result]:
    """Compile a kernel with :func:`jax.jit` and keep its signature for type checks."""
    return cast(Callable[Parameters, Result], jax.jit(function))


class Terms(NamedTuple):
    """A divergence between two Gaussians, with its location and shape terms."""

    total: Array
    """The divergence."""
    location: Array
    """The part that only the means change."""
    shape: Array
    """The part that only the covariances change."""


def gaussian_kl(
    mean_p: Float[ArrayLike, " param"],
    cov_p: Float[ArrayLike, "param param"],
    mean_q: Float[ArrayLike, " param"],
    cov_q: Float[ArrayLike, "param param"],
) -> Terms:
    """Return the KL divergence KL(p ‖ q) between two Gaussians, in nats.

    The location term is ½ ΔμᵀΣ_q⁻¹Δμ and the shape term
    ½ [tr(Σ_q⁻¹Σ_p) − D + ln(det Σ_q / det Σ_p)]; the total is their sum.
    Equal covariances give a shape term of exactly zero.

    Args:
        mean_p: Mean of p, shaped (D,).
        cov_p: Covariance of p, shaped (D, D).
        mean_q: Mean of q, shaped (D,).
        cov_q: Covariance of q, shaped (D, D).

    Returns:
        The divergence and its location and shape terms.

    Raises:
        ValueError: If concrete inputs are not finite, or a covariance is not
            symmetric positive definite.

    References:
        :cite:t:`kullback1951information`; :cite:t:`rasmussen2006gaussian`.
    """
    return _gaussian_kl(*_gaussian_inputs(mean_p, cov_p, mean_q, cov_q))


def gaussian_w2(
    mean_p: Float[ArrayLike, " param"],
    cov_p: Float[ArrayLike, "param param"],
    mean_q: Float[ArrayLike, " param"],
    cov_q: Float[ArrayLike, "param param"],
) -> Terms:
    """Return the W2 distance between two Gaussians.

    The location term is the Euclidean distance between the means, and the
    shape term the Bures distance between the covariances,
    [tr Σ_p + tr Σ_q − 2 tr((Σ_p^½ Σ_q Σ_p^½)^½)]^½. The total is the root of
    the sum of their squares. The distance is in the units of the parameters,
    so it changes when the parameters are rescaled. The matrix roots come
    from eigendecompositions, whose gradients are undefined where two
    eigenvalues coincide.

    Args:
        mean_p: Mean of p, shaped (D,).
        cov_p: Covariance of p, shaped (D, D).
        mean_q: Mean of q, shaped (D,).
        cov_q: Covariance of q, shaped (D, D).

    Returns:
        The distance and its location and shape terms.

    Raises:
        ValueError: If concrete inputs are not finite, or a covariance is not
            symmetric positive definite.

    References:
        :cite:t:`dowson1982frechet`; :cite:t:`givens1984class`.
    """
    return _gaussian_w2(*_gaussian_inputs(mean_p, cov_p, mean_q, cov_q))


def gaussian_hellinger(
    mean_p: Float[ArrayLike, " param"],
    cov_p: Float[ArrayLike, "param param"],
    mean_q: Float[ArrayLike, " param"],
    cov_q: Float[ArrayLike, "param param"],
) -> Terms:
    """Return the Hellinger distance between two Gaussians.

    With Σ = (Σ_p + Σ_q)/2, the Bhattacharyya distance has the location term
    ⅛ ΔμᵀΣ⁻¹Δμ and the shape term ½ ln(det Σ / √(det Σ_p det Σ_q)). Each
    term converts to a distance as H = √(1 − exp(−D_B)), so that
    1 − H² of the total is the product of 1 − H² of the two terms.

    Args:
        mean_p: Mean of p, shaped (D,).
        cov_p: Covariance of p, shaped (D, D).
        mean_q: Mean of q, shaped (D,).
        cov_q: Covariance of q, shaped (D, D).

    Returns:
        The distance and its location and shape terms, each in [0, 1].

    Raises:
        ValueError: If concrete inputs are not finite, or a covariance is not
            symmetric positive definite.

    References:
        :cite:t:`hellinger1909neue`; :cite:t:`bhattacharyya1943measure`;
        :cite:t:`fukunaga1990introduction`.
    """
    return _gaussian_hellinger(*_gaussian_inputs(mean_p, cov_p, mean_q, cov_q))


def mahalanobis_shift(
    mean_p: Float[ArrayLike, " param"],
    cov_p: Float[ArrayLike, "param param"],
    mean_q: Float[ArrayLike, " param"],
) -> Float[Array, ""]:
    """Return the shift between two means in units of the covariance of p.

    The shift is √(ΔμᵀΣ_p⁻¹Δμ) with Δμ = μ_q − μ_p. When q is p shifted, it
    is the shift between two unit Gaussians after whitening by Σ_p, which is
    the definition of σ_eff.

    Args:
        mean_p: Mean of p, shaped (D,).
        cov_p: Covariance of p, shaped (D, D).
        mean_q: Mean of q, shaped (D,).

    Returns:
        The shift, non-negative.

    Raises:
        ValueError: If concrete inputs are not finite, or the covariance is
            not symmetric positive definite.
    """
    mean_p_array, cov_p_array, mean_q_array, _ = _gaussian_inputs(
        mean_p, cov_p, mean_q, cov_p
    )
    return _mahalanobis_shift(mean_p_array, cov_p_array, mean_q_array)


def normal_ks(
    mean_p: Float[ArrayLike, ""],
    std_p: Float[ArrayLike, ""],
    mean_q: Float[ArrayLike, ""],
    std_q: Float[ArrayLike, ""],
) -> Float[Array, ""]:
    """Return the Kolmogorov-Smirnov distance between two normals.

    The distance is the largest absolute gap between F_p and F_q, which lies
    where the two densities cross. For equal widths that is the midpoint of
    the means, which gives 2Φ(Δ/(2σ)) − 1 for an absolute mean difference Δ;
    otherwise the densities cross twice, at the roots of a quadratic.

    Args:
        mean_p: Mean of p.
        std_p: Standard deviation of p, positive.
        mean_q: Mean of q.
        std_q: Standard deviation of q, positive.

    Returns:
        The distance, in [0, 1].

    Raises:
        ValueError: If a concrete mean is not finite or a width not positive.

    References:
        :cite:t:`smirnov1948table`.
    """
    return _normal_ks(*_normal_inputs(mean_p, std_p, mean_q, std_q))


def normal_w1(
    mean_p: Float[ArrayLike, ""],
    std_p: Float[ArrayLike, ""],
    mean_q: Float[ArrayLike, ""],
    std_q: Float[ArrayLike, ""],
) -> Float[Array, ""]:
    """Return the W1 distance between two normals.

    Coupling the quantiles, the distance is the mean absolute value of
    a + bZ for a standard normal Z, with a the absolute difference of the
    means and b that of the standard deviations. That is the mean of a folded
    normal, b √(2/π) exp(−a²/(2b²)) + a erf(a/(b√2)).

    Args:
        mean_p: Mean of p.
        std_p: Standard deviation of p, positive.
        mean_q: Mean of q.
        std_q: Standard deviation of q, positive.

    Returns:
        The distance, in the units of the parameter.

    Raises:
        ValueError: If a concrete mean is not finite or a width not positive.

    References:
        :cite:t:`vallender1974calculation`; :cite:t:`leone1961folded`.
    """
    return _normal_w1(*_normal_inputs(mean_p, std_p, mean_q, std_q))


def normal_hellinger(
    mean_p: Float[ArrayLike, ""],
    std_p: Float[ArrayLike, ""],
    mean_q: Float[ArrayLike, ""],
    std_q: Float[ArrayLike, ""],
) -> Float[Array, ""]:
    """Return the Hellinger distance between two normals.

    The Bhattacharyya coefficient of two normals is
    √(2σ_pσ_q/(σ_p² + σ_q²)) exp(−Δμ²/(4(σ_p² + σ_q²))), and H = √(1 − BC).

    Args:
        mean_p: Mean of p.
        std_p: Standard deviation of p, positive.
        mean_q: Mean of q.
        std_q: Standard deviation of q, positive.

    Returns:
        The distance, in [0, 1].

    Raises:
        ValueError: If a concrete mean is not finite or a width not positive.

    References:
        :cite:t:`hellinger1909neue`; :cite:t:`bhattacharyya1943measure`.
    """
    return _normal_hellinger(*_normal_inputs(mean_p, std_p, mean_q, std_q))


def normal_js(
    mean_p: Float[ArrayLike, ""],
    std_p: Float[ArrayLike, ""],
    mean_q: Float[ArrayLike, ""],
    std_q: Float[ArrayLike, ""],
) -> Float[Array, ""]:
    """Return the Jensen-Shannon divergence between two normals, in nats.

    The divergence is ½ KL(p ‖ m) + ½ KL(q ‖ m) with m = (p + q)/2, integrated
    by quadrature. It lies between 0 and ln 2, and identical normals give
    exactly zero.

    Args:
        mean_p: Mean of p.
        std_p: Standard deviation of p, positive.
        mean_q: Mean of q.
        std_q: Standard deviation of q, positive.

    Returns:
        The divergence in nats.

    Raises:
        ValueError: If a concrete mean is not finite or a width not positive.

    References:
        :cite:t:`lin1991divergence`; :cite:t:`davis1984methods`.
    """
    return _normal_js(*_normal_inputs(mean_p, std_p, mean_q, std_q))


def normal_anderson_darling(
    mean_p: Float[ArrayLike, ""],
    std_p: Float[ArrayLike, ""],
    mean_q: Float[ArrayLike, ""],
    std_q: Float[ArrayLike, ""],
) -> Float[Array, ""]:
    """Return the Anderson-Darling distance between two normals.

    The distance is ∫ (F − G)² / (H(1 − H)) dH with H = (F + G)/2, integrated
    by quadrature. It is the limit of the two-sample Anderson-Darling
    statistic divided by nm/(n + m) when both samples have the same size.
    Fully separated normals give 8 ln 2 − 4.

    Args:
        mean_p: Mean of p.
        std_p: Standard deviation of p, positive.
        mean_q: Mean of q.
        std_q: Standard deviation of q, positive.

    Returns:
        The distance, non-negative.

    Raises:
        ValueError: If a concrete mean is not finite or a width not positive.

    References:
        :cite:t:`pettitt1976two`; :cite:t:`scholz1987k`;
        :cite:t:`davis1984methods`.
    """
    return _normal_anderson_darling(*_normal_inputs(mean_p, std_p, mean_q, std_q))


def normal_mixture_kl(
    weights_p: Float[ArrayLike, " component"],
    means_p: Float[ArrayLike, " component"],
    stds_p: Float[ArrayLike, " component"],
    weights_q: Float[ArrayLike, " other"],
    means_q: Float[ArrayLike, " other"],
    stds_q: Float[ArrayLike, " other"],
) -> Float[Array, ""]:
    """Return KL(p ‖ q) between two one-dimensional normal mixtures, in nats.

    A normal is a mixture with one component. The divergence is integrated by
    quadrature, with panels around every component of both mixtures and
    around every point where two weighted components cross, where ln p or
    ln q switches from one component to the other.

    Args:
        weights_p: Component weights of p, positive; they are normalised.
        means_p: Component means of p.
        stds_p: Component standard deviations of p, positive.
        weights_q: Component weights of q, positive; they are normalised.
        means_q: Component means of q.
        stds_q: Component standard deviations of q, positive.

    Returns:
        The divergence in nats.

    Raises:
        ValueError: If a concrete weight or width is not positive, or a mean
            is not finite.

    References:
        :cite:t:`kullback1951information`; :cite:t:`davis1984methods`.
    """
    arrays = [
        jnp.asarray(value, dtype=jnp.float64)
        for value in (weights_p, means_p, stds_p, weights_q, means_q, stds_q)
    ]
    for name, (weights, means, stds) in (("p", arrays[:3]), ("q", arrays[3:])):
        require(jnp.all(jnp.isfinite(means)), f"the means of {name} must be finite")
        require(
            jnp.all((weights > 0) & jnp.isfinite(weights) & (stds > 0)),
            f"the weights and standard deviations of {name} must be positive",
        )
    return _normal_mixture_kl(*arrays)


def twisted_gaussian_moments(
    twist: Float[ArrayLike, ""], variance: Float[ArrayLike, ""], param_count: int
) -> tuple[Float[Array, " param"], Float[Array, "param param"]]:
    """Return the mean and covariance of the twisted Gaussian B(b, v).

    The twisted Gaussian maps x ~ N(0, diag(v, 1, …, 1)) to y with
    y₂ = x₂ + b(x₁² − v) and every other coordinate unchanged
    (Haario, Saksman and Tamminen 1999). Its mean is zero and its covariance
    diag(v, 1 + 2b²v², 1, …, 1): the twist adds the variance b² Var(x₁²)
    to y₂ and leaves y₁ and y₂ uncorrelated, because E[x₁³] = 0.

    Args:
        twist: The twist b.
        variance: The variance v of the first coordinate, positive.
        param_count: Number of parameters D, at least 2.

    Returns:
        The mean, shaped (D,), and the covariance, shaped (D, D).

    Raises:
        ValueError: If there are fewer than 2 parameters, or a concrete
            parameter is out of range.

    References:
        :cite:t:`haario1999adaptive`; :cite:t:`sejdinovic2014kernel`.
    """
    if param_count < 2:
        raise ValueError(
            f"a twisted Gaussian needs at least 2 parameters, got {param_count}"
        )
    twist_array, variance_array = _twisted_inputs(twist, variance)
    diagonal = (
        jnp.ones(param_count)
        .at[0]
        .set(variance_array)
        .at[1]
        .set(1 + 2 * twist_array**2 * variance_array**2)
    )
    return jnp.zeros(param_count), jnp.diag(diagonal)


def twisted_gaussian_kl(
    twist: Float[ArrayLike, ""], variance: Float[ArrayLike, ""]
) -> tuple[Float[Array, ""], Float[Array, ""]]:
    """Return the KL divergences between B(b, v) and its moment-matched Gaussian.

    With c = 2b²v², KL(B ‖ N) = ½ ln(1 + c) and KL(N ‖ B) = c − ½ ln(1 + c),
    exactly. The first holds because the twist has a unit Jacobian, so B has
    the entropy of its base Gaussian, and because KL to the moment-matched
    Gaussian is the entropy gap that the Gaussian maximises (Cover and Thomas
    2006). The second holds because ln B is a polynomial in y, whose mean
    under N follows from Gaussian moments. Neither depends on the number of
    parameters.

    Args:
        twist: The twist b.
        variance: The variance v of the first coordinate, positive.

    Returns:
        KL(B ‖ N) and KL(N ‖ B), in nats.

    Raises:
        ValueError: If a concrete parameter is out of range.

    References:
        :cite:t:`haario1999adaptive`; :cite:t:`cover2006elements`.
    """
    twist_array, variance_array = _twisted_inputs(twist, variance)
    excess = 2 * twist_array**2 * variance_array**2
    banana_to_gaussian = 0.5 * jnp.log1p(excess)
    return banana_to_gaussian, excess - banana_to_gaussian


def truncated_normal_moments(
    mean: Float[ArrayLike, " *leading"],
    std: Float[ArrayLike, " *leading"],
    lower: Float[ArrayLike, " *leading"],
    upper: Float[ArrayLike, " *leading"],
) -> tuple[Float[Array, " *leading"], Float[Array, " *leading"]]:
    """Return the mean and standard deviation of a truncated normal.

    The normal N(μ, σ²) is truncated to [lower, upper], where one bound may be
    infinite. With a and b the bounds in units of σ from μ, Z = Φ(b) − Φ(a)
    and the ratios λ_a = φ(a)/Z and λ_b = φ(b)/Z, the mean is
    μ + σ(λ_a − λ_b) and the variance σ²[1 + aλ_a − bλ_b − (λ_a − λ_b)²].
    Each ratio is evaluated in log space, so bounds far in a tail keep their
    precision.

    Args:
        mean: Mean μ of the normal before truncation.
        std: Standard deviation σ before truncation, positive.
        lower: Lower bound, which may be −inf.
        upper: Upper bound, above ``lower``, which may be inf.

    Returns:
        The mean and the standard deviation after truncation.

    Raises:
        ValueError: If a concrete mean is not finite, a width not positive or
            a bound pair not ordered.

    References:
        :cite:t:`barr1999mean`.
    """
    return _truncated_normal_moments(*_truncated_inputs(mean, std, lower, upper))


def truncated_normal_kl(
    mean_p: Float[ArrayLike, ""],
    mean_q: Float[ArrayLike, ""],
    std: Float[ArrayLike, ""],
    lower: Float[ArrayLike, ""],
    upper: Float[ArrayLike, ""],
) -> Float[Array, ""]:
    """Return KL(p ‖ q) between two normals truncated to the same interval.

    Both have the standard deviation σ before truncation. With Z the mass
    that each keeps and m_p the mean of p after truncation,
    KL = [2 m_p (μ_p − μ_q) + μ_q² − μ_p²]/(2σ²) + ln(Z_q / Z_p).

    Args:
        mean_p: Mean of p before truncation.
        mean_q: Mean of q before truncation.
        std: Standard deviation of both before truncation, positive.
        lower: Lower bound, which may be −inf.
        upper: Upper bound, above ``lower``, which may be inf.

    Returns:
        The divergence in nats.

    Raises:
        ValueError: If a concrete mean is not finite, the width not positive
            or the bounds not ordered.

    References:
        :cite:t:`kullback1951information`; :cite:t:`barr1999mean`.
    """
    mean_p_array, std_array, lower_array, upper_array = _truncated_inputs(
        mean_p, std, lower, upper
    )
    mean_q_array = jnp.asarray(mean_q, dtype=jnp.float64)
    require(jnp.isfinite(mean_q_array), "mean_q must be finite")
    return _truncated_normal_kl(
        mean_p_array, mean_q_array, std_array, lower_array, upper_array
    )


@_jit
def log_normal_mass(
    lower: Float[ArrayLike, " *leading"], upper: Float[ArrayLike, " *leading"]
) -> Float[Array, " *leading"]:
    """Return ln(Φ(upper) − Φ(lower)), the log-mass of a standard normal interval.

    Both bounds are mirrored into the lower tail when ``lower`` > 0, where Φ
    keeps its relative precision, so that intervals far in a tail do not
    round to zero mass.

    Args:
        lower: Lower bound in standard deviations, which may be −inf.
        upper: Upper bound, above ``lower``, which may be inf.

    Returns:
        The log-mass, at most zero.
    """
    lower_array = jnp.asarray(lower, dtype=jnp.float64)
    upper_array = jnp.asarray(upper, dtype=jnp.float64)
    mirror = lower_array > 0
    low = jnp.where(mirror, -upper_array, lower_array)
    high = jnp.where(mirror, -lower_array, upper_array)
    log_high, log_low = log_ndtr(high), log_ndtr(low)
    return log_high + jnp.log1p(-jnp.exp(log_low - log_high))


def railing_score(offset: Float[ArrayLike, " *leading"]) -> Float[Array, " *leading"]:
    """Return the density at the bound over the peak density of a railed normal.

    A normal with its peak ``offset`` standard deviations inside a bound,
    truncated there, has the score exp(−offset²/2). When the peak lies at or
    beyond the bound (offset ≤ 0), the density is largest at the bound and
    the score is 1.

    Args:
        offset: Distance from the bound to the peak of the normal before
            truncation, in its standard deviations; positive inside the
            support.

    Returns:
        The score, in (0, 1].

    Raises:
        ValueError: If a concrete offset is not finite.
    """
    offset_array = jnp.asarray(offset, dtype=jnp.float64)
    require(jnp.isfinite(offset_array), "the offset must be finite")
    return jnp.where(offset_array > 0, jnp.exp(-0.5 * offset_array**2), 1.0)


def wrapped_normal_resultant_length(
    std: Float[ArrayLike, " *leading"], period: Float[ArrayLike, " *leading"]
) -> Float[Array, " *leading"]:
    """Return the mean resultant length of a wrapped normal.

    A normal of standard deviation σ wrapped onto a circle of period P has
    the mean resultant length R = exp(−½ (2πσ/P)²), so its circular standard
    deviation √(−2 ln R) is σ in units of P/(2π) (Mardia and Jupp 2000).

    Args:
        std: Standard deviation σ of the normal before wrapping, positive.
        period: Period P of the circle, positive.

    Returns:
        The mean resultant length, in (0, 1).

    Raises:
        ValueError: If a concrete width or period is not finite and positive.

    References:
        :cite:t:`mardia2000directional`.
    """
    std_array = jnp.asarray(std, dtype=jnp.float64)
    period_array = jnp.asarray(period, dtype=jnp.float64)
    require(
        (std_array > 0) & jnp.isfinite(std_array), "std must be finite and positive"
    )
    require(
        (period_array > 0) & jnp.isfinite(period_array),
        "period must be finite and positive",
    )
    return jnp.exp(-0.5 * (2 * math.pi * std_array / period_array) ** 2)


def von_mises_fisher_resultant_length(
    concentration: Float[ArrayLike, " *leading"],
) -> Float[Array, " *leading"]:
    """Return the mean resultant length of a von Mises-Fisher distribution on S².

    On the 2-sphere E[x] = A(κ) μ with A(κ) = coth κ − 1/κ (Fisher 1953;
    Mardia and Jupp 2000). Below κ = 0.01 the Taylor series
    κ/3 − κ³/45 + 2κ⁵/945 avoids the cancellation of the two terms.

    Args:
        concentration: Concentration κ, non-negative.

    Returns:
        The mean resultant length, in [0, 1).

    Raises:
        ValueError: If a concrete concentration is negative or not finite.

    References:
        :cite:t:`fisher1953dispersion`; :cite:t:`mardia2000directional`.
    """
    return _resultant_length(_concentration_input(concentration))


def von_mises_fisher_log_normaliser(
    concentration: Float[ArrayLike, " *leading"],
) -> Float[Array, " *leading"]:
    """Return ln C(κ) of the von Mises-Fisher density C(κ) exp(κ μᵀx) on S².

    On the 2-sphere C(κ) = κ / (4π sinh κ), and C(0) = 1/(4π) for the uniform
    distribution; ln sinh κ is evaluated as κ − ln 2 + ln(1 − e^(−2κ)).

    Args:
        concentration: Concentration κ, non-negative.

    Returns:
        The log-normaliser, with respect to the area on the unit sphere.

    Raises:
        ValueError: If a concrete concentration is negative or not finite.

    References:
        :cite:t:`fisher1953dispersion`; :cite:t:`mardia2000directional`.
    """
    return _log_normaliser(_concentration_input(concentration))


def von_mises_fisher_kl(
    direction_p: Float[ArrayLike, " 3"],
    concentration_p: Float[ArrayLike, ""],
    direction_q: Float[ArrayLike, " 3"],
    concentration_q: Float[ArrayLike, ""],
) -> Float[Array, ""]:
    """Return KL(p ‖ q) between two von Mises-Fisher distributions on S².

    With ln f = ln C(κ) + κ μᵀx and E_p[x] = A(κ_p) μ_p,
    KL = ln C(κ_p) − ln C(κ_q) + A(κ_p)(κ_p − κ_q μ_pᵀμ_q).

    Args:
        direction_p: Mean direction of p, shaped (3,); it is normalised.
        concentration_p: Concentration of p, non-negative.
        direction_q: Mean direction of q, shaped (3,); it is normalised.
        concentration_q: Concentration of q, non-negative.

    Returns:
        The divergence in nats.

    Raises:
        ValueError: If a concrete direction is zero or not finite, or a
            concentration is negative or not finite.

    References:
        :cite:t:`kullback1951information`; :cite:t:`fisher1953dispersion`;
        :cite:t:`mardia2000directional`.
    """
    return _von_mises_fisher_kl(
        *_von_mises_fisher_inputs(
            direction_p, concentration_p, direction_q, concentration_q
        )
    )


def von_mises_fisher_hellinger(
    direction_p: Float[ArrayLike, " 3"],
    concentration_p: Float[ArrayLike, ""],
    direction_q: Float[ArrayLike, " 3"],
    concentration_q: Float[ArrayLike, ""],
) -> Float[Array, ""]:
    """Return the Hellinger distance between two von Mises-Fisher distributions on S².

    The product √(f_p f_q) is a von Mises-Fisher kernel of concentration
    κ = ‖κ_p μ_p + κ_q μ_q‖ / 2, so the Bhattacharyya coefficient is
    √(C(κ_p) C(κ_q)) / C(κ) and H = √(1 − BC).

    Args:
        direction_p: Mean direction of p, shaped (3,); it is normalised.
        concentration_p: Concentration of p, non-negative.
        direction_q: Mean direction of q, shaped (3,); it is normalised.
        concentration_q: Concentration of q, non-negative.

    Returns:
        The distance, in [0, 1].

    Raises:
        ValueError: If a concrete direction is zero or not finite, or a
            concentration is negative or not finite.

    References:
        :cite:t:`hellinger1909neue`; :cite:t:`bhattacharyya1943measure`;
        :cite:t:`fisher1953dispersion`.
    """
    return _von_mises_fisher_hellinger(
        *_von_mises_fisher_inputs(
            direction_p, concentration_p, direction_q, concentration_q
        )
    )


def ar1_autocorrelation_time(
    coefficient: Float[ArrayLike, " *leading"],
) -> Float[Array, " *leading"]:
    """Return the integrated autocorrelation time of an AR(1) chain.

    The autocorrelation of x_t = φ x_{t−1} + ε_t at lag t is φ^|t|, so
    τ = Σ_t φ^|t| = (1 + φ)/(1 − φ), with τ summed over all lags as in emcee,
    so that N/τ is the effective sample size. Negative φ gives τ < 1.

    Args:
        coefficient: The coefficient φ, in (−1, 1).

    Returns:
        The autocorrelation time τ.

    Raises:
        ValueError: If a concrete coefficient lies outside (−1, 1).

    References:
        :cite:t:`thompson2010comparison`;
        :cite:t:`foremanmackey2017autocorrelation`.
    """
    coefficient_array = jnp.asarray(coefficient, dtype=jnp.float64)
    require(
        (coefficient_array > -1) & (coefficient_array < 1),
        "an AR(1) coefficient must lie in (-1, 1)",
    )
    return (1 + coefficient_array) / (1 - coefficient_array)


# * Input checks, run before the jitted kernels so that concrete inputs can raise.


def _gaussian_inputs(
    mean_p: ArrayLike, cov_p: ArrayLike, mean_q: ArrayLike, cov_q: ArrayLike
) -> tuple[Array, Array, Array, Array]:
    mean_p_array, cov_p_array, mean_q_array, cov_q_array = (
        jnp.asarray(value, dtype=jnp.float64)
        for value in (mean_p, cov_p, mean_q, cov_q)
    )
    require(jnp.all(jnp.isfinite(mean_p_array)), "mean_p must be finite")
    require(jnp.all(jnp.isfinite(mean_q_array)), "mean_q must be finite")
    require_covariance("cov_p", cov_p_array)
    require_covariance("cov_q", cov_q_array)
    return mean_p_array, cov_p_array, mean_q_array, cov_q_array


def _normal_inputs(
    mean_p: ArrayLike, std_p: ArrayLike, mean_q: ArrayLike, std_q: ArrayLike
) -> tuple[Array, Array, Array, Array]:
    mean_p_array, std_p_array, mean_q_array, std_q_array = (
        jnp.asarray(value, dtype=jnp.float64)
        for value in (mean_p, std_p, mean_q, std_q)
    )
    require(jnp.isfinite(mean_p_array), "mean_p must be finite")
    require(jnp.isfinite(mean_q_array), "mean_q must be finite")
    for name, value in (("std_p", std_p_array), ("std_q", std_q_array)):
        require(
            (value > 0) & jnp.isfinite(value), f"{name} must be finite and positive"
        )
    return mean_p_array, std_p_array, mean_q_array, std_q_array


def _twisted_inputs(twist: ArrayLike, variance: ArrayLike) -> tuple[Array, Array]:
    twist_array = jnp.asarray(twist, dtype=jnp.float64)
    variance_array = jnp.asarray(variance, dtype=jnp.float64)
    require(jnp.isfinite(twist_array), "the twist must be finite")
    require(
        (variance_array > 0) & jnp.isfinite(variance_array),
        "the variance must be finite and positive",
    )
    return twist_array, variance_array


def _truncated_inputs(
    mean: ArrayLike, std: ArrayLike, lower: ArrayLike, upper: ArrayLike
) -> tuple[Array, Array, Array, Array]:
    mean_array, std_array, lower_array, upper_array = (
        jnp.asarray(value, dtype=jnp.float64) for value in (mean, std, lower, upper)
    )
    require(jnp.all(jnp.isfinite(mean_array)), "the mean must be finite")
    require(
        jnp.all((std_array > 0) & jnp.isfinite(std_array)),
        "std must be finite and positive",
    )
    require(jnp.all(lower_array < upper_array), "the bounds must satisfy lower < upper")
    return mean_array, std_array, lower_array, upper_array


def _concentration_input(concentration: ArrayLike) -> Array:
    array = jnp.asarray(concentration, dtype=jnp.float64)
    require(
        jnp.all((array >= 0) & jnp.isfinite(array)),
        "the concentration must be finite and non-negative",
    )
    return array


def _von_mises_fisher_inputs(
    direction_p: ArrayLike,
    concentration_p: ArrayLike,
    direction_q: ArrayLike,
    concentration_q: ArrayLike,
) -> tuple[Array, Array, Array, Array]:
    units = []
    for name, direction in (("direction_p", direction_p), ("direction_q", direction_q)):
        array = jnp.asarray(direction, dtype=jnp.float64)
        require(jnp.all(jnp.isfinite(array)), f"{name} must be finite")
        require(jnp.any(array != 0), f"{name} must not be the zero vector")
        units.append(array / jnp.linalg.norm(array))
    return (
        units[0],
        _concentration_input(concentration_p),
        units[1],
        _concentration_input(concentration_q),
    )


# * Jitted kernels.


@_jit
def _gaussian_kl(mean_p: Array, cov_p: Array, mean_q: Array, cov_q: Array) -> Terms:
    cholesky_p, cholesky_q = jnp.linalg.cholesky(cov_p), jnp.linalg.cholesky(cov_q)
    whitened_shift = solve_triangular(cholesky_q, mean_q - mean_p, lower=True)
    location = 0.5 * whitened_shift @ whitened_shift
    whitened_root = solve_triangular(cholesky_q, cholesky_p, lower=True)
    log_det_ratio = _log_det(cholesky_q) - _log_det(cholesky_p)
    shape = 0.5 * (jnp.sum(whitened_root**2) - mean_p.shape[0] + log_det_ratio)
    # * Equal covariances give an exact zero, not a rounding residue.
    shape = jnp.where(jnp.all(cov_p == cov_q), 0.0, shape)
    return Terms(location + shape, location, shape)


@_jit
def _gaussian_w2(mean_p: Array, cov_p: Array, mean_q: Array, cov_q: Array) -> Terms:
    location = jnp.linalg.norm(mean_q - mean_p)
    eigenvalues, eigenvectors = jnp.linalg.eigh(cov_p)
    root_p = (eigenvectors * jnp.sqrt(jnp.clip(eigenvalues, 0.0))) @ eigenvectors.T
    middle = root_p @ cov_q @ root_p
    middle_eigenvalues = jnp.linalg.eigvalsh(0.5 * (middle + middle.T))
    cross_trace = jnp.sum(jnp.sqrt(jnp.clip(middle_eigenvalues, 0.0)))
    traces = jnp.trace(cov_p) + jnp.trace(cov_q)
    bures_squared = jnp.maximum(traces - 2 * cross_trace, 0.0)
    shape = jnp.where(jnp.all(cov_p == cov_q), 0.0, jnp.sqrt(bures_squared))
    return Terms(jnp.hypot(location, shape), location, shape)


@_jit
def _gaussian_hellinger(
    mean_p: Array, cov_p: Array, mean_q: Array, cov_q: Array
) -> Terms:
    cholesky_middle = jnp.linalg.cholesky(0.5 * (cov_p + cov_q))
    whitened_shift = solve_triangular(cholesky_middle, mean_q - mean_p, lower=True)
    location_distance = 0.125 * whitened_shift @ whitened_shift
    mean_log_det = 0.5 * (
        _log_det(jnp.linalg.cholesky(cov_p)) + _log_det(jnp.linalg.cholesky(cov_q))
    )
    shape_distance = jnp.maximum(0.5 * (_log_det(cholesky_middle) - mean_log_det), 0.0)
    return Terms(
        _hellinger_from_bhattacharyya(location_distance + shape_distance),
        _hellinger_from_bhattacharyya(location_distance),
        _hellinger_from_bhattacharyya(shape_distance),
    )


@_jit
def _mahalanobis_shift(mean_p: Array, cov_p: Array, mean_q: Array) -> Array:
    whitened_shift = solve_triangular(
        jnp.linalg.cholesky(cov_p), mean_q - mean_p, lower=True
    )
    return jnp.sqrt(whitened_shift @ whitened_shift)


@_jit
def _normal_ks(mean_p: Array, std_p: Array, mean_q: Array, std_q: Array) -> Array:
    shift, ratio = (mean_q - mean_p) / std_p, std_q / std_p
    # * In units of p: p is N(0, 1), q is N(shift, ratio²); ln p = ln q is quadratic.
    quadratic = 1.0 - 1.0 / ratio**2
    linear = 2.0 * shift / ratio**2
    constant = -(shift**2) / ratio**2 - 2.0 * jnp.log(ratio)
    # * Two different normals always cross; the roots clamp a rounding-negative term.
    first, second, _ = _quadratic_roots(quadratic, linear, constant)
    gap = jnp.maximum(
        _normal_cdf_gap(first, (first - shift) / ratio),
        _normal_cdf_gap(second, (second - shift) / ratio),
    )
    return jnp.where((quadratic == 0.0) & (linear == 0.0), 0.0, gap)


@_jit
def _normal_w1(mean_p: Array, std_p: Array, mean_q: Array, std_q: Array) -> Array:
    mean_gap, width_gap = jnp.abs(mean_q - mean_p), jnp.abs(std_q - std_p)
    safe_gap = jnp.where(width_gap == 0.0, 1.0, width_gap)
    folded = safe_gap * math.sqrt(2 / math.pi) * jnp.exp(
        -(mean_gap**2) / (2 * safe_gap**2)
    ) + mean_gap * erf(mean_gap / (safe_gap * math.sqrt(2)))
    return jnp.where(width_gap == 0.0, mean_gap, folded)


@_jit
def _normal_hellinger(
    mean_p: Array, std_p: Array, mean_q: Array, std_q: Array
) -> Array:
    variance_sum = std_p**2 + std_q**2
    log_coefficient = 0.5 * jnp.log(2 * std_p * std_q / variance_sum) - (
        mean_q - mean_p
    ) ** 2 / (4 * variance_sum)
    return jnp.sqrt(jnp.maximum(-jnp.expm1(log_coefficient), 0.0))


@_jit
def _normal_js(mean_p: Array, std_p: Array, mean_q: Array, std_q: Array) -> Array:
    shift, ratio = (mean_q - mean_p) / std_p, std_q / std_p
    value = _quadrature(
        functools.partial(_js_integrand, shift=shift, ratio=ratio),
        jnp.log(jnp.array([0.5, 0.5])),
        jnp.stack([jnp.zeros_like(shift), shift]),
        jnp.stack([jnp.ones_like(ratio), ratio]),
    )
    # * Rounding can carry the integral a few ulps past the exact range [0, ln 2].
    value = jnp.clip(value, 0.0, math.log(2))
    return jnp.where((shift == 0.0) & (ratio == 1.0), 0.0, value)


@_jit
def _normal_anderson_darling(
    mean_p: Array, std_p: Array, mean_q: Array, std_q: Array
) -> Array:
    shift, ratio = (mean_q - mean_p) / std_p, std_q / std_p
    value = _quadrature(
        functools.partial(_anderson_darling_integrand, shift=shift, ratio=ratio),
        jnp.log(jnp.array([0.5, 0.5])),
        jnp.stack([jnp.zeros_like(shift), shift]),
        jnp.stack([jnp.ones_like(ratio), ratio]),
    )
    return jnp.where((shift == 0.0) & (ratio == 1.0), 0.0, jnp.maximum(value, 0.0))


@_jit
def _normal_mixture_kl(
    weights_p: Array,
    means_p: Array,
    stds_p: Array,
    weights_q: Array,
    means_q: Array,
    stds_q: Array,
) -> Array:
    log_weights_p = jnp.log(weights_p / jnp.sum(weights_p))
    log_weights_q = jnp.log(weights_q / jnp.sum(weights_q))
    value = _quadrature(
        functools.partial(
            _mixture_kl_integrand,
            components_p=(log_weights_p, means_p, stds_p),
            components_q=(log_weights_q, means_q, stds_q),
        ),
        jnp.concatenate([log_weights_p, log_weights_q]),
        jnp.concatenate([means_p, means_q]),
        jnp.concatenate([stds_p, stds_q]),
    )
    return jnp.maximum(value, 0.0)


@_jit
def _truncated_normal_moments(
    mean: Array, std: Array, lower: Array, upper: Array
) -> tuple[Array, Array]:
    alpha, beta = (lower - mean) / std, (upper - mean) / std
    log_mass = log_normal_mass(alpha, beta)
    ratio_lower, bound_lower = _bound_terms(alpha, log_mass)
    ratio_upper, bound_upper = _bound_terms(beta, log_mass)
    shift = ratio_lower - ratio_upper
    variance = jnp.maximum(1.0 + bound_lower - bound_upper - shift**2, 0.0)
    return mean + std * shift, std * jnp.sqrt(variance)


@_jit
def _truncated_normal_kl(
    mean_p: Array, mean_q: Array, std: Array, lower: Array, upper: Array
) -> Array:
    mean_after, _ = _truncated_normal_moments(mean_p, std, lower, upper)
    log_mass_p = log_normal_mass((lower - mean_p) / std, (upper - mean_p) / std)
    log_mass_q = log_normal_mass((lower - mean_q) / std, (upper - mean_q) / std)
    quadratic_part = (2 * mean_after * (mean_p - mean_q) + mean_q**2 - mean_p**2) / (
        2 * std**2
    )
    return quadratic_part + log_mass_q - log_mass_p


@_jit
def _resultant_length(concentration: Array) -> Array:
    small = concentration < SMALL_CONCENTRATION
    safe = jnp.where(small, 1.0, concentration)
    series = concentration / 3 - concentration**3 / 45 + 2 * concentration**5 / 945
    return jnp.where(small, series, 1 / jnp.tanh(safe) - 1 / safe)


@_jit
def _log_normaliser(concentration: Array) -> Array:
    zero = concentration == 0
    safe = jnp.where(zero, 1.0, concentration)
    log_sinh = safe - math.log(2) + jnp.log(-jnp.expm1(-2 * safe))
    value = jnp.log(safe) - math.log(4 * math.pi) - log_sinh
    return jnp.where(zero, -math.log(4 * math.pi), value)


@_jit
def _von_mises_fisher_kl(
    unit_p: Array, concentration_p: Array, unit_q: Array, concentration_q: Array
) -> Array:
    cosine = jnp.clip(unit_p @ unit_q, -1.0, 1.0)
    return (
        _log_normaliser(concentration_p)
        - _log_normaliser(concentration_q)
        + _resultant_length(concentration_p)
        * (concentration_p - concentration_q * cosine)
    )


@_jit
def _von_mises_fisher_hellinger(
    unit_p: Array, concentration_p: Array, unit_q: Array, concentration_q: Array
) -> Array:
    joint = 0.5 * jnp.linalg.norm(concentration_p * unit_p + concentration_q * unit_q)
    log_coefficient = 0.5 * (
        _log_normaliser(concentration_p) + _log_normaliser(concentration_q)
    ) - _log_normaliser(joint)
    return jnp.sqrt(jnp.maximum(-jnp.expm1(jnp.minimum(log_coefficient, 0.0)), 0.0))


# * Quadrature and its integrands.


def _quadrature(
    integrand: Callable[[Array], Array],
    log_weights: Float[Array, " component"],
    means: Float[Array, " component"],
    stds: Float[Array, " component"],
) -> Float[Array, ""]:
    """Integrate over the real line on panels refined at components and crossings."""
    crossing_centres, crossing_widths = _crossings(log_weights, means, stds)
    centres = jnp.concatenate([means, crossing_centres])
    widths = jnp.concatenate([stds, crossing_widths])
    edges = jnp.sort((centres[:, None] + widths[:, None] * _PANEL_OFFSETS).ravel())
    half_widths = 0.5 * jnp.diff(edges)
    midpoints = 0.5 * (edges[:-1] + edges[1:])
    points = midpoints[:, None] + half_widths[:, None] * _LEGENDRE_NODES
    return jnp.sum(half_widths[:, None] * _LEGENDRE_WEIGHTS * integrand(points))


def _crossings(
    log_weights: Float[Array, " component"],
    means: Float[Array, " component"],
    stds: Float[Array, " component"],
) -> tuple[Array, Array]:
    """Return where each pair of weighted components cross, and the switch width.

    Two weighted log-densities f_i and f_j are equal at the roots of a
    quadratic; around a root, a log-sum-exp switches from one to the other
    over a width of one over the difference of their slopes. Pairs without a
    real crossing repeat the panels of their first component, which changes
    nothing.
    """
    first, second = np.triu_indices(means.shape[0], 1)
    precision = 1 / stds**2
    offset = log_weights - jnp.log(stds) - 0.5 * means**2 * precision
    root_one, root_two, exists = _quadratic_roots(
        0.5 * (precision[second] - precision[first]),
        means[first] * precision[first] - means[second] * precision[second],
        offset[first] - offset[second],
    )
    roots = jnp.concatenate([root_one, root_two])
    owner, partner = np.concatenate([first, first]), np.concatenate([second, second])
    valid = jnp.concatenate([exists, exists]) & jnp.isfinite(roots)
    slope = (roots - means[partner]) * precision[partner] - (
        roots - means[owner]
    ) * precision[owner]
    widest = jnp.maximum(stds[owner], stds[partner])
    narrowest = jnp.minimum(stds[owner], stds[partner])
    inverse_slope = 1 / jnp.where(slope == 0, 1 / widest, jnp.abs(slope))
    switch_width = jnp.clip(inverse_slope, CROSSING_RESOLUTION * narrowest, widest)
    span_lower = jnp.min(means - QUADRATURE_REACH * stds)
    span_upper = jnp.max(means + QUADRATURE_REACH * stds)
    centres = jnp.where(valid, jnp.clip(roots, span_lower, span_upper), means[owner])
    return centres, jnp.where(valid, switch_width, stds[owner])


def _quadratic_roots(
    quadratic: Array, linear: Array, constant: Array
) -> tuple[Array, Array, Array]:
    """Return both real roots of a x² + b x + c, and whether any root exists.

    The stable form avoids cancellation; with a = 0 both roots are the root of
    the linear equation, and with a = b = 0 there is none.
    """
    is_linear = quadratic == 0
    discriminant = linear**2 - 4 * quadratic * constant
    stable = -0.5 * (
        linear + jnp.copysign(jnp.sqrt(jnp.maximum(discriminant, 0.0)), linear)
    )
    root_one = jnp.where(
        is_linear,
        -constant / jnp.where(linear == 0, 1.0, linear),
        stable / jnp.where(is_linear, 1.0, quadratic),
    )
    root_two = jnp.where(
        is_linear, root_one, constant / jnp.where(stable == 0, 1.0, stable)
    )
    exists = jnp.where(is_linear, linear != 0, discriminant >= 0)
    return root_one, root_two, exists


def _log_normal(points: Array, mean: ArrayLike, std: ArrayLike) -> Array:
    return -0.5 * ((points - mean) / std) ** 2 - jnp.log(std) - _HALF_LOG_TWO_PI


def _js_integrand(points: Array, shift: Array, ratio: Array) -> Array:
    """Return the Jensen-Shannon integrand of N(0, 1) and N(shift, ratio²)."""
    log_p = _log_normal(points, 0.0, 1.0)
    log_q = _log_normal(points, shift, ratio)
    log_mixture = jnp.logaddexp(log_p, log_q) - math.log(2)
    return 0.5 * (
        jnp.exp(log_p) * (log_p - log_mixture) + jnp.exp(log_q) * (log_q - log_mixture)
    )


def _anderson_darling_integrand(points: Array, shift: Array, ratio: Array) -> Array:
    """Return (F − G)²/(H(1 − H)) times dH/dx for N(0, 1) and N(shift, ratio²)."""
    standard_q = (points - shift) / ratio
    cdf_p, cdf_q = ndtr(points), ndtr(standard_q)
    survival_p, survival_q = ndtr(-points), ndtr(-standard_q)
    pooled = 0.5 * (cdf_p + cdf_q)
    pooled_survival = 0.5 * (survival_p + survival_q)
    # * Each difference in the tail where both terms are small loses no digits.
    gap = jnp.where(pooled <= 0.5, cdf_p - cdf_q, survival_q - survival_p)
    denominator = pooled * pooled_survival
    density = 0.5 * (
        jnp.exp(_log_normal(points, 0.0, 1.0))
        + jnp.exp(_log_normal(points, shift, ratio))
    )
    # ! Far beyond every component both Φ terms underflow; there the integrand is zero.
    safe = jnp.where(denominator > 0, denominator, 1.0)
    return jnp.where(denominator > 0, gap**2 / safe * density, 0.0)


def _mixture_kl_integrand(
    points: Array,
    components_p: tuple[Array, Array, Array],
    components_q: tuple[Array, Array, Array],
) -> Array:
    """Return p (ln p − ln q) for two normal mixtures given by their components."""
    log_p = _log_mixture(points, *components_p)
    return jnp.exp(log_p) * (log_p - _log_mixture(points, *components_q))


def _log_mixture(points: Array, log_weights: Array, means: Array, stds: Array) -> Array:
    return logsumexp(log_weights + _log_normal(points[..., None], means, stds), axis=-1)


def _normal_cdf_gap(standard_p: Array, standard_q: Array) -> Array:
    """Return |Φ(z_p) − Φ(z_q)|, using survival functions in the upper tail."""
    upper = standard_p + standard_q > 0
    return jnp.abs(
        jnp.where(
            upper,
            ndtr(-standard_q) - ndtr(-standard_p),
            ndtr(standard_p) - ndtr(standard_q),
        )
    )


def _bound_terms(bound: Array, log_mass: Array) -> tuple[Array, Array]:
    """Return φ(bound)/Z and bound φ(bound)/Z, both zero at an infinite bound."""
    finite = jnp.isfinite(bound)
    safe = jnp.where(finite, bound, 0.0)
    ratio = jnp.where(
        finite, jnp.exp(-0.5 * safe**2 - _HALF_LOG_TWO_PI - log_mass), 0.0
    )
    return ratio, safe * ratio


def _log_det(cholesky_factor: Array) -> Array:
    return 2.0 * jnp.sum(jnp.log(jnp.diag(cholesky_factor)))


def _hellinger_from_bhattacharyya(distance: Array) -> Array:
    return jnp.sqrt(-jnp.expm1(-distance))
