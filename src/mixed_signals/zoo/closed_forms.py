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
    (Hellinger 1909; Bhattacharyya 1943; Fukunaga 1990). The kernels are
    those of :mod:`mixed_signals.metrics.moments`, which the metrics share;
    the functions here check concrete inputs first. Each result is a
    :class:`~mixed_signals.metrics.moments.Terms`.
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
from typing import Final, cast

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.linalg import solve_triangular
from jax.scipy.special import erf, log_ndtr, logsumexp, ndtr
from jax.typing import ArrayLike
from jaxtyping import Array, Float

from mixed_signals.metrics import moments
from mixed_signals.metrics.moments import Terms
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

TRACED_WINDINGS: Final = 13
"""Windings summed on each side of a wrapped normal inside ``jax.jit``.

Outside ``jax.jit`` the windings follow the width; inside, these 13 cover 12
widths beyond the circle for widths up to one period, and wider ones give NaN.
"""

_HALF_LOG_TWO_PI: Final = 0.5 * math.log(2 * math.pi)
# * Fixed node table: the panels move with the distributions, the rule does not.
_LEGENDRE_NODES, _LEGENDRE_WEIGHTS = np.polynomial.legendre.leggauss(
    GAUSS_LEGENDRE_ORDER
)
_PANEL_OFFSETS: Final = np.linspace(
    -QUADRATURE_REACH, QUADRATURE_REACH, PANELS_PER_FEATURE + 1
)


def _jit[**Parameters, Result](
    function: Callable[Parameters, Result], *static_names: str
) -> Callable[Parameters, Result]:
    """Compile a kernel with :func:`jax.jit` and keep its signature for type checks."""
    return cast(
        Callable[Parameters, Result], jax.jit(function, static_argnames=static_names)
    )


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
        The divergence, its location and shape terms and the log-determinant
        ratio ln(det Σ_q/det Σ_p).

    Raises:
        ValueError: If concrete inputs are not finite, or a covariance is not
            symmetric positive definite.

    References:
        :cite:t:`kullback1951information`; :cite:t:`rasmussen2006gaussian`.
    """
    return moments.gaussian_kl(*_gaussian_inputs(mean_p, cov_p, mean_q, cov_q))


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
        The distance, its location and shape terms and the log-determinant
        ratio ln(det Σ_q/det Σ_p).

    Raises:
        ValueError: If concrete inputs are not finite, or a covariance is not
            symmetric positive definite.

    References:
        :cite:t:`dowson1982frechet`; :cite:t:`givens1984class`.
    """
    return moments.gaussian_w2(*_gaussian_inputs(mean_p, cov_p, mean_q, cov_q))


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
        The distance and its location and shape terms, each in [0, 1], and
        the log-determinant ratio ln(det Σ_q/det Σ_p).

    Raises:
        ValueError: If concrete inputs are not finite, or a covariance is not
            symmetric positive definite.

    References:
        :cite:t:`hellinger1909neue`; :cite:t:`bhattacharyya1943measure`;
        :cite:t:`fukunaga1990introduction`.
    """
    return moments.gaussian_hellinger(*_gaussian_inputs(mean_p, cov_p, mean_q, cov_q))


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


def truncated_normal_ks(
    mean_p: Float[ArrayLike, ""],
    mean_q: Float[ArrayLike, ""],
    std: Float[ArrayLike, ""],
    lower: Float[ArrayLike, ""],
    upper: Float[ArrayLike, ""],
) -> Float[Array, ""]:
    """Return the Kolmogorov-Smirnov distance between two normals truncated alike.

    Both have the standard deviation σ before truncation and the same
    interval, so their log densities differ by a linear function and cross
    once inside it, at x* = (μ_p + μ_q)/2 + σ² ln(Z_p/Z_q)/(μ_p − μ_q), with Z
    the mass that each keeps. The absolute gap between F_p and F_q is largest
    there.

    Args:
        mean_p: Mean of p before truncation.
        mean_q: Mean of q before truncation.
        std: Standard deviation of both before truncation, positive.
        lower: Lower bound, which may be −inf.
        upper: Upper bound, above ``lower``, which may be inf.

    Returns:
        The distance, in [0, 1].

    Raises:
        ValueError: If a concrete mean is not finite, the width not positive
            or the bounds not ordered.

    References:
        :cite:t:`smirnov1948table`; :cite:t:`barr1999mean`.
    """
    return _truncated_normal_ks(
        *_truncated_pair_inputs(mean_p, mean_q, std, lower, upper)
    )


def truncated_normal_w1(
    mean_p: Float[ArrayLike, ""],
    mean_q: Float[ArrayLike, ""],
    std: Float[ArrayLike, ""],
    lower: Float[ArrayLike, ""],
    upper: Float[ArrayLike, ""],
) -> Float[Array, ""]:
    """Return the W1 distance between two normals truncated alike.

    The densities cross once, so F_p − F_q keeps one sign, and the integral of
    its absolute value is the gap between the means after truncation
    (Vallender 1974; Barr and Sherrill 1999).

    Args:
        mean_p: Mean of p before truncation.
        mean_q: Mean of q before truncation.
        std: Standard deviation of both before truncation, positive.
        lower: Lower bound, which may be −inf.
        upper: Upper bound, above ``lower``, which may be inf.

    Returns:
        The distance, in the units of the parameter.

    Raises:
        ValueError: If a concrete mean is not finite, the width not positive
            or the bounds not ordered.

    References:
        :cite:t:`vallender1974calculation`; :cite:t:`barr1999mean`.
    """
    return _truncated_normal_w1(
        *_truncated_pair_inputs(mean_p, mean_q, std, lower, upper)
    )


def truncated_normal_hellinger(
    mean_p: Float[ArrayLike, ""],
    mean_q: Float[ArrayLike, ""],
    std: Float[ArrayLike, ""],
    lower: Float[ArrayLike, ""],
    upper: Float[ArrayLike, ""],
) -> Float[Array, ""]:
    """Return the Hellinger distance between two normals truncated alike.

    √(f_p f_q) is a normal of mean (μ_p + μ_q)/2 times exp(−(μ_p − μ_q)²/(8σ²)),
    so the Bhattacharyya coefficient is that factor times the mass Z of the
    middle normal in the interval over √(Z_p Z_q), and H = √(1 − BC).

    Args:
        mean_p: Mean of p before truncation.
        mean_q: Mean of q before truncation.
        std: Standard deviation of both before truncation, positive.
        lower: Lower bound, which may be −inf.
        upper: Upper bound, above ``lower``, which may be inf.

    Returns:
        The distance, in [0, 1].

    Raises:
        ValueError: If a concrete mean is not finite, the width not positive
            or the bounds not ordered.

    References:
        :cite:t:`hellinger1909neue`; :cite:t:`bhattacharyya1943measure`.
    """
    return _truncated_normal_hellinger(
        *_truncated_pair_inputs(mean_p, mean_q, std, lower, upper)
    )


def truncated_normal_js(
    mean_p: Float[ArrayLike, ""],
    mean_q: Float[ArrayLike, ""],
    std: Float[ArrayLike, ""],
    lower: Float[ArrayLike, ""],
    upper: Float[ArrayLike, ""],
) -> Float[Array, ""]:
    """Return the JS divergence between two normals truncated alike, in nats.

    The divergence is integrated by quadrature over the interval, with
    panels around both means and an edge at each finite bound.

    Args:
        mean_p: Mean of p before truncation.
        mean_q: Mean of q before truncation.
        std: Standard deviation of both before truncation, positive.
        lower: Lower bound, which may be −inf.
        upper: Upper bound, above ``lower``, which may be inf.

    Returns:
        The divergence in nats, in [0, ln 2].

    Raises:
        ValueError: If a concrete mean is not finite, the width not positive
            or the bounds not ordered.

    References:
        :cite:t:`lin1991divergence`; :cite:t:`davis1984methods`.
    """
    return _truncated_normal_js(
        *_truncated_pair_inputs(mean_p, mean_q, std, lower, upper)
    )


def truncated_normal_anderson_darling(
    mean_p: Float[ArrayLike, ""],
    mean_q: Float[ArrayLike, ""],
    std: Float[ArrayLike, ""],
    lower: Float[ArrayLike, ""],
    upper: Float[ArrayLike, ""],
) -> Float[Array, ""]:
    """Return the Anderson-Darling distance between two normals truncated alike.

    The distance ∫ (F − G)²/(H(1 − H)) dH, with H = (F + G)/2, is integrated by
    quadrature over the interval; each distribution function and its
    complement come from the log-mass of the interval, so tails keep their
    precision.

    Args:
        mean_p: Mean of p before truncation.
        mean_q: Mean of q before truncation.
        std: Standard deviation of both before truncation, positive.
        lower: Lower bound, which may be −inf.
        upper: Upper bound, above ``lower``, which may be inf.

    Returns:
        The distance, non-negative.

    Raises:
        ValueError: If a concrete mean is not finite, the width not positive
            or the bounds not ordered.

    References:
        :cite:t:`pettitt1976two`; :cite:t:`scholz1987k`;
        :cite:t:`davis1984methods`.
    """
    return _truncated_normal_anderson_darling(
        *_truncated_pair_inputs(mean_p, mean_q, std, lower, upper)
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


def wrapped_normal_kuiper(
    mean_p: Float[ArrayLike, ""],
    mean_q: Float[ArrayLike, ""],
    std: Float[ArrayLike, ""],
    period: Float[ArrayLike, ""],
) -> Float[Array, ""]:
    """Return Kuiper's V between two wrapped normals of equal width.

    Two wrapped normals of width σ whose means lie an arc a apart cross at
    the midpoint of that arc and opposite it, where F − G has its extremes.
    Reflection about the midpoint swaps the two, so
    V = max(F − G) − min(F − G) is the absolute value of 2M − 1, with M the
    mass that the second puts on the half circle that starts at the midpoint
    and contains its mean. On a line the same pair gives the KS distance
    2Φ(a/(2σ)) − 1.

    Args:
        mean_p: Mean direction of p, in the units of the circle.
        mean_q: Mean direction of q.
        std: Width σ of both before wrapping, positive.
        period: Period P of the circle, positive.

    Returns:
        The statistic, in [0, 1].

    Raises:
        ValueError: If a concrete mean is not finite, or the width or the
            period is not finite and positive.

    References:
        :cite:t:`kuiper1960tests`; :cite:t:`mardia2000directional`.
    """
    arrays, windings = _wrapped_pair_inputs(mean_p, mean_q, std, period)
    return _wrapped_normal_kuiper(*arrays, windings)


def wrapped_normal_w1(
    mean_p: Float[ArrayLike, ""],
    mean_q: Float[ArrayLike, ""],
    std: Float[ArrayLike, ""],
    period: Float[ArrayLike, ""],
) -> Float[Array, ""]:
    """Return the W1 distance along the circle between equal-width wrapped normals.

    With the arc length as cost, W1 is the smallest integral, over one period,
    of the absolute value of F − G − c for a constant c (Rabin, Delon and
    Gousseau 2011). For means an arc b apart, F − G is, up to a constant, the
    mass m(x) of the first on the arc of length b that ends at x, symmetric
    and unimodal about the midpoint of the means. Its median then splits the
    circle into the half around that midpoint and the opposite half, so
    W1 = ∫ m over the first half minus ∫ m over the second, each a sum of
    integrals of Φ: ∫ Φ(t) dt = tΦ(t) + φ(t).

    Args:
        mean_p: Mean direction of p, in the units of the circle.
        mean_q: Mean direction of q.
        std: Width σ of both before wrapping, positive.
        period: Period P of the circle, positive.

    Returns:
        The distance, in the units of the circle.

    Raises:
        ValueError: If a concrete mean is not finite, or the width or the
            period is not finite and positive.

    References:
        :cite:t:`rabin2011transportation`; :cite:t:`mardia2000directional`.
    """
    arrays, windings = _wrapped_pair_inputs(mean_p, mean_q, std, period)
    return _wrapped_normal_w1(*arrays, windings)


def wrapped_normal_anderson_darling(
    mean_p: Float[ArrayLike, ""],
    mean_q: Float[ArrayLike, ""],
    std: Float[ArrayLike, ""],
    period: Float[ArrayLike, ""],
) -> Float[Array, ""]:
    """Return the Anderson-Darling distance between two wrapped normals of equal width.

    The circle is cut opposite the midpoint of the shorter arc between the
    means, the circular mean of the two pooled with equal weight, and
    ∫ (F − G)²/(H(1 − H)) dH, with H = (F + G)/2, is integrated by quadrature
    over the period that the cut leaves.

    Args:
        mean_p: Mean direction of p, in the units of the circle.
        mean_q: Mean direction of q.
        std: Width σ of both before wrapping, positive.
        period: Period P of the circle, positive.

    Returns:
        The distance, non-negative.

    Raises:
        ValueError: If a concrete mean is not finite, or the width or the
            period is not finite and positive.

    References:
        :cite:t:`pettitt1976two`; :cite:t:`scholz1987k`;
        :cite:t:`mardia2000directional`; :cite:t:`davis1984methods`.
    """
    arrays, windings = _wrapped_pair_inputs(mean_p, mean_q, std, period)
    return _wrapped_normal_anderson_darling(*arrays, windings)


def wrapped_normal_js(
    mean_p: Float[ArrayLike, ""],
    mean_q: Float[ArrayLike, ""],
    std: Float[ArrayLike, ""],
    period: Float[ArrayLike, ""],
) -> Float[Array, ""]:
    """Return the Jensen-Shannon divergence between two wrapped normals of equal width.

    The divergence, in nats, is integrated by quadrature over one period of
    the wrapped densities.

    Args:
        mean_p: Mean direction of p, in the units of the circle.
        mean_q: Mean direction of q.
        std: Width σ of both before wrapping, positive.
        period: Period P of the circle, positive.

    Returns:
        The divergence in nats, in [0, ln 2].

    Raises:
        ValueError: If a concrete mean is not finite, or the width or the
            period is not finite and positive.

    References:
        :cite:t:`lin1991divergence`; :cite:t:`mardia2000directional`.
    """
    arrays, windings = _wrapped_pair_inputs(mean_p, mean_q, std, period)
    return _wrapped_normal_js(*arrays, windings)


def wrapped_normal_hellinger(
    mean_p: Float[ArrayLike, ""],
    mean_q: Float[ArrayLike, ""],
    std: Float[ArrayLike, ""],
    period: Float[ArrayLike, ""],
) -> Float[Array, ""]:
    """Return the Hellinger distance between two wrapped normals of equal width.

    H² = ½ ∫ (√f_p − √f_q)² is integrated by quadrature over one period of
    the wrapped densities.

    Args:
        mean_p: Mean direction of p, in the units of the circle.
        mean_q: Mean direction of q.
        std: Width σ of both before wrapping, positive.
        period: Period P of the circle, positive.

    Returns:
        The distance, in [0, 1].

    Raises:
        ValueError: If a concrete mean is not finite, or the width or the
            period is not finite and positive.

    References:
        :cite:t:`hellinger1909neue`; :cite:t:`mardia2000directional`.
    """
    arrays, windings = _wrapped_pair_inputs(mean_p, mean_q, std, period)
    return _wrapped_normal_hellinger(*arrays, windings)


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


def _truncated_pair_inputs(
    mean_p: ArrayLike,
    mean_q: ArrayLike,
    std: ArrayLike,
    lower: ArrayLike,
    upper: ArrayLike,
) -> tuple[Array, Array, Array, Array, Array]:
    mean_p_array, std_array, lower_array, upper_array = _truncated_inputs(
        mean_p, std, lower, upper
    )
    mean_q_array = jnp.asarray(mean_q, dtype=jnp.float64)
    require(jnp.isfinite(mean_q_array), "mean_q must be finite")
    return mean_p_array, mean_q_array, std_array, lower_array, upper_array


def _wrapped_pair_inputs(
    mean_p: ArrayLike, mean_q: ArrayLike, std: ArrayLike, period: ArrayLike
) -> tuple[tuple[Array, Array, Array, Array], int]:
    """Check two wrapped normals and size their winding sums from concrete widths."""
    mean_p_array, mean_q_array, std_array, period_array = (
        jnp.asarray(value, dtype=jnp.float64) for value in (mean_p, mean_q, std, period)
    )
    require(jnp.isfinite(mean_p_array), "mean_p must be finite")
    require(jnp.isfinite(mean_q_array), "mean_q must be finite")
    for name, value in (("std", std_array), ("period", period_array)):
        require(
            (value > 0) & jnp.isfinite(value), f"{name} must be finite and positive"
        )
    if isinstance(std_array, jax.core.Tracer) or isinstance(
        period_array, jax.core.Tracer
    ):
        windings = TRACED_WINDINGS
    else:
        windings = (
            math.ceil(QUADRATURE_REACH * float(std_array) / float(period_array)) + 1
        )
    return (mean_p_array, mean_q_array, std_array, period_array), windings


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


@_jit
def _truncated_normal_ks(
    mean_p: Array, mean_q: Array, std: Array, lower: Array, upper: Array
) -> Array:
    log_mass_p = log_normal_mass((lower - mean_p) / std, (upper - mean_p) / std)
    log_mass_q = log_normal_mass((lower - mean_q) / std, (upper - mean_q) / std)
    difference = mean_p - mean_q
    safe = jnp.where(difference == 0, 1.0, difference)
    crossing = jnp.clip(
        0.5 * (mean_p + mean_q) + std**2 * (log_mass_p - log_mass_q) / safe,
        lower,
        upper,
    )
    cdf_p, survival_p = _truncated_cdfs(crossing, mean_p, std, lower, upper, log_mass_p)
    cdf_q, survival_q = _truncated_cdfs(crossing, mean_q, std, lower, upper, log_mass_q)
    gap = jnp.where(cdf_p + cdf_q <= 1, cdf_p - cdf_q, survival_q - survival_p)
    return jnp.where(difference == 0, 0.0, jnp.abs(gap))


@_jit
def _truncated_normal_w1(
    mean_p: Array, mean_q: Array, std: Array, lower: Array, upper: Array
) -> Array:
    mean_after_p, _ = _truncated_normal_moments(mean_p, std, lower, upper)
    mean_after_q, _ = _truncated_normal_moments(mean_q, std, lower, upper)
    return jnp.where(mean_p == mean_q, 0.0, jnp.abs(mean_after_q - mean_after_p))


@_jit
def _truncated_normal_hellinger(
    mean_p: Array, mean_q: Array, std: Array, lower: Array, upper: Array
) -> Array:
    middle = 0.5 * (mean_p + mean_q)
    log_coefficient = (
        -((mean_p - mean_q) ** 2) / (8 * std**2)
        + log_normal_mass((lower - middle) / std, (upper - middle) / std)
        - 0.5 * log_normal_mass((lower - mean_p) / std, (upper - mean_p) / std)
        - 0.5 * log_normal_mass((lower - mean_q) / std, (upper - mean_q) / std)
    )
    return jnp.sqrt(jnp.maximum(-jnp.expm1(jnp.minimum(log_coefficient, 0.0)), 0.0))


@_jit
def _truncated_normal_js(
    mean_p: Array, mean_q: Array, std: Array, lower: Array, upper: Array
) -> Array:
    log_mass_p = log_normal_mass((lower - mean_p) / std, (upper - mean_p) / std)
    log_mass_q = log_normal_mass((lower - mean_q) / std, (upper - mean_q) / std)
    value = _window_quadrature(
        functools.partial(
            _truncated_js_integrand,
            means=(mean_p, mean_q),
            std=std,
            log_masses=(log_mass_p, log_mass_q),
        ),
        jnp.stack([mean_p, mean_q]),
        jnp.stack([std, std]),
        lower,
        upper,
    )
    return jnp.where(mean_p == mean_q, 0.0, jnp.clip(value, 0.0, math.log(2)))


@_jit
def _truncated_normal_anderson_darling(
    mean_p: Array, mean_q: Array, std: Array, lower: Array, upper: Array
) -> Array:
    log_mass_p = log_normal_mass((lower - mean_p) / std, (upper - mean_p) / std)
    log_mass_q = log_normal_mass((lower - mean_q) / std, (upper - mean_q) / std)
    value = _window_quadrature(
        functools.partial(
            _truncated_anderson_darling_integrand,
            means=(mean_p, mean_q),
            std=std,
            bounds=(lower, upper),
            log_masses=(log_mass_p, log_mass_q),
        ),
        jnp.stack([mean_p, mean_q]),
        jnp.stack([std, std]),
        lower,
        upper,
    )
    return jnp.where(mean_p == mean_q, 0.0, jnp.maximum(value, 0.0))


def _wrapped_normal_kuiper_kernel(
    mean_p: Array, mean_q: Array, std: Array, period: Array, windings: int
) -> Array:
    half = 0.5 * jnp.abs(_shortest_arc(mean_p, mean_q, period))
    # * In the frame of q, the half circle from the midpoint spans −a/2 to P/2 − a/2.
    mass = _wrapped_arc_mass(-half, 0.5 * period - half, 0.0, std, period, windings)
    value = jnp.where(half == 0, 0.0, jnp.abs(2 * mass - 1))
    return _within_windings(value, std, period, windings)


def _wrapped_normal_w1_kernel(
    mean_p: Array, mean_q: Array, std: Array, period: Array, windings: int
) -> Array:
    gap = jnp.abs(_shortest_arc(mean_p, mean_q, period))
    centre, quarter = 0.5 * gap, 0.25 * period
    inner = _arc_mass_integral(
        centre - quarter, centre + quarter, gap, std, period, windings
    )
    outer = _arc_mass_integral(
        centre + quarter, centre + 3 * quarter, gap, std, period, windings
    )
    value = jnp.where(gap == 0, 0.0, jnp.maximum(inner - outer, 0.0))
    return _within_windings(value, std, period, windings)


def _wrapped_normal_anderson_darling_kernel(
    mean_p: Array, mean_q: Array, std: Array, period: Array, windings: int
) -> Array:
    half = 0.5 * _shortest_arc(mean_p, mean_q, period)
    integrand = functools.partial(
        _wrapped_anderson_darling_integrand,
        half=half,
        std=std,
        period=period,
        windings=windings,
    )
    value = _wrapped_quadrature(integrand, half, std, period, windings)
    value = jnp.where(half == 0, 0.0, jnp.maximum(value, 0.0))
    return _within_windings(value, std, period, windings)


def _wrapped_normal_js_kernel(
    mean_p: Array, mean_q: Array, std: Array, period: Array, windings: int
) -> Array:
    half = 0.5 * _shortest_arc(mean_p, mean_q, period)
    integrand = functools.partial(
        _wrapped_js_integrand, half=half, std=std, period=period, windings=windings
    )
    value = _wrapped_quadrature(integrand, half, std, period, windings)
    value = jnp.where(half == 0, 0.0, jnp.clip(value, 0.0, math.log(2)))
    return _within_windings(value, std, period, windings)


def _wrapped_normal_hellinger_kernel(
    mean_p: Array, mean_q: Array, std: Array, period: Array, windings: int
) -> Array:
    half = 0.5 * _shortest_arc(mean_p, mean_q, period)
    integrand = functools.partial(
        _wrapped_hellinger_integrand,
        half=half,
        std=std,
        period=period,
        windings=windings,
    )
    squared = _wrapped_quadrature(integrand, half, std, period, windings)
    value = jnp.where(half == 0, 0.0, jnp.sqrt(jnp.clip(squared, 0.0, 1.0)))
    return _within_windings(value, std, period, windings)


_wrapped_normal_kuiper = _jit(_wrapped_normal_kuiper_kernel, "windings")
_wrapped_normal_w1 = _jit(_wrapped_normal_w1_kernel, "windings")
_wrapped_normal_anderson_darling = _jit(
    _wrapped_normal_anderson_darling_kernel, "windings"
)
_wrapped_normal_js = _jit(_wrapped_normal_js_kernel, "windings")
_wrapped_normal_hellinger = _jit(_wrapped_normal_hellinger_kernel, "windings")


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


# * Marginals of truncated and wrapped normals.


def _window_quadrature(
    integrand: Callable[[Array], Array],
    centres: Float[Array, " component"],
    widths: Float[Array, " component"],
    lower: Float[ArrayLike, ""],
    upper: Float[ArrayLike, ""],
) -> Float[Array, ""]:
    """Integrate over [lower, upper] on panels refined around every component."""
    component_edges = (centres[:, None] + widths[:, None] * _PANEL_OFFSETS).ravel()
    bounds = jnp.stack([jnp.asarray(lower), jnp.asarray(upper)])
    # * An infinite end adds no edge of its own: it moves onto the outermost edge.
    outermost = jnp.stack([jnp.min(component_edges), jnp.max(component_edges)])
    bounds = jnp.where(jnp.isfinite(bounds), bounds, outermost)
    edges = jnp.sort(jnp.clip(jnp.concatenate([component_edges, bounds]), lower, upper))
    half_widths = 0.5 * jnp.diff(edges)
    midpoints = 0.5 * (edges[:-1] + edges[1:])
    points = midpoints[:, None] + half_widths[:, None] * _LEGENDRE_NODES
    return jnp.sum(half_widths[:, None] * _LEGENDRE_WEIGHTS * integrand(points))


def _truncated_cdfs(
    points: Array, mean: Array, std: Array, lower: Array, upper: Array, log_mass: Array
) -> tuple[Array, Array]:
    """Return F and 1 − F of a truncated normal, each from its own log-mass."""
    standard = (points - mean) / std
    low, high = (
        jnp.broadcast_to((bound - mean) / std, standard.shape)
        for bound in (lower, upper)
    )
    cdf = jnp.exp(log_normal_mass(low, standard) - log_mass)
    survival = jnp.exp(log_normal_mass(standard, high) - log_mass)
    return cdf, survival


def _truncated_js_integrand(
    points: Array,
    means: tuple[Array, Array],
    std: Array,
    log_masses: tuple[Array, Array],
) -> Array:
    """Return the Jensen-Shannon integrand of two truncated normals in the interval."""
    log_p = _log_normal(points, means[0], std) - log_masses[0]
    log_q = _log_normal(points, means[1], std) - log_masses[1]
    return _js_terms(log_p, log_q)


def _truncated_anderson_darling_integrand(
    points: Array,
    means: tuple[Array, Array],
    std: Array,
    bounds: tuple[Array, Array],
    log_masses: tuple[Array, Array],
) -> Array:
    """Return (F − G)²/(H(1 − H)) times dH/dx for two truncated normals."""
    cdf_p, survival_p = _truncated_cdfs(points, means[0], std, *bounds, log_masses[0])
    cdf_q, survival_q = _truncated_cdfs(points, means[1], std, *bounds, log_masses[1])
    density = 0.5 * (
        jnp.exp(_log_normal(points, means[0], std) - log_masses[0])
        + jnp.exp(_log_normal(points, means[1], std) - log_masses[1])
    )
    return _anderson_darling_terms(cdf_p, survival_p, cdf_q, survival_q, density)


def _shortest_arc(origin: Array, target: Array, period: Array) -> Array:
    """Return the shortest signed arc from origin to target, in [−P/2, P/2)."""
    return jnp.mod(target - origin + 0.5 * period, period) - 0.5 * period


def _winding_shifts(period: Array, windings: int) -> Array:
    return jnp.arange(-windings, windings + 1) * period


def _wrapped_arc_mass(
    start: ArrayLike,
    end: ArrayLike,
    mean: ArrayLike,
    std: Array,
    period: Array,
    windings: int,
) -> Array:
    """Return the mass of a wrapped normal on the arc [start, end], at most a period."""
    shifts = _winding_shifts(period, windings)
    low, high = jnp.broadcast_arrays(
        (jnp.asarray(start)[..., None] - mean + shifts) / std,
        (jnp.asarray(end)[..., None] - mean + shifts) / std,
    )
    return jnp.sum(jnp.exp(log_normal_mass(low, high)), axis=-1)


def _wrapped_log_density(
    points: Array, mean: ArrayLike, std: Array, period: Array, windings: int
) -> Array:
    shifts = _winding_shifts(period, windings)
    return logsumexp(_log_normal(points[..., None] + shifts, mean, std), axis=-1)


def _arc_mass_integral(
    start: Array, end: Array, gap: Array, std: Array, period: Array, windings: int
) -> Array:
    """Return ∫ m dx on [start, end], m the wrapped N(0, σ²) mass on (x − gap, x]."""
    shifts = _winding_shifts(period, windings)
    return std * (
        _phi_antiderivative(end, shifts, std)
        - _phi_antiderivative(start, shifts, std)
        - _phi_antiderivative(end - gap, shifts, std)
        + _phi_antiderivative(start - gap, shifts, std)
    )


def _phi_antiderivative(offset: Array, shifts: Array, std: Array) -> Array:
    """Return Σ J((offset + kP)/σ) over windings, with J(t) = tΦ(t) + φ(t) = ∫ Φ."""
    standard = (offset + shifts) / std
    return jnp.sum(standard * ndtr(standard) + jnp.exp(_log_normal(standard, 0.0, 1.0)))


def _wrapped_quadrature(
    integrand: Callable[[Array], Array],
    half: Array,
    std: Array,
    period: Array,
    windings: int,
) -> Array:
    """Integrate over the period centred on the midpoint of the two means."""
    shifts = _winding_shifts(period, windings)
    centres = jnp.concatenate([-half + shifts, half + shifts])
    return _window_quadrature(
        integrand, centres, jnp.full_like(centres, std), -0.5 * period, 0.5 * period
    )


def _wrapped_pair_densities(
    points: Array, half: Array, std: Array, period: Array, windings: int
) -> tuple[Array, Array]:
    """Return the log densities of the pair, centred at ∓half around the midpoint."""
    return (
        _wrapped_log_density(points, -half, std, period, windings),
        _wrapped_log_density(points, half, std, period, windings),
    )


def _wrapped_js_integrand(
    points: Array, half: Array, std: Array, period: Array, windings: int
) -> Array:
    return _js_terms(*_wrapped_pair_densities(points, half, std, period, windings))


def _wrapped_hellinger_integrand(
    points: Array, half: Array, std: Array, period: Array, windings: int
) -> Array:
    log_p, log_q = _wrapped_pair_densities(points, half, std, period, windings)
    return 0.5 * (jnp.exp(0.5 * log_p) - jnp.exp(0.5 * log_q)) ** 2


def _wrapped_anderson_darling_integrand(
    points: Array, half: Array, std: Array, period: Array, windings: int
) -> Array:
    """Return (F − G)²/(H(1 − H)) times dH/dx on the period from the cut."""
    start, end = -0.5 * period, 0.5 * period
    cdf_p = _wrapped_arc_mass(start, points, -half, std, period, windings)
    survival_p = _wrapped_arc_mass(points, end, -half, std, period, windings)
    cdf_q = _wrapped_arc_mass(start, points, half, std, period, windings)
    survival_q = _wrapped_arc_mass(points, end, half, std, period, windings)
    log_p, log_q = _wrapped_pair_densities(points, half, std, period, windings)
    density = 0.5 * (jnp.exp(log_p) + jnp.exp(log_q))
    return _anderson_darling_terms(cdf_p, survival_p, cdf_q, survival_q, density)


def _within_windings(value: Array, std: Array, period: Array, windings: int) -> Array:
    """Return the value, or NaN where the windings miss 12 widths beyond the circle."""
    return jnp.where(QUADRATURE_REACH * std <= (windings - 1) * period, value, jnp.nan)


def _js_terms(log_p: Array, log_q: Array) -> Array:
    """Return ½ p ln(2p/(p + q)) + ½ q ln(2q/(p + q)), zero where a density is."""
    log_sum = jnp.logaddexp(log_p, log_q)
    terms = jnp.zeros_like(log_sum)
    for log_density in (log_p, log_q):
        positive = log_density > -jnp.inf
        safe = jnp.where(positive, log_density, 0.0)
        terms = terms + jnp.where(
            positive, 0.5 * jnp.exp(safe) * (math.log(2) + safe - log_sum), 0.0
        )
    return terms


def _anderson_darling_terms(
    cdf_p: Array, survival_p: Array, cdf_q: Array, survival_q: Array, density: Array
) -> Array:
    """Return (F − G)²/(H(1 − H)) times the pooled density, zero where H(1 − H) is."""
    pooled = 0.5 * (cdf_p + cdf_q)
    pooled_survival = 0.5 * (survival_p + survival_q)
    # * Each difference in the tail where both terms are small loses no digits.
    gap = jnp.where(pooled <= 0.5, cdf_p - cdf_q, survival_q - survival_p)
    denominator = pooled * pooled_survival
    safe = jnp.where(denominator > 0, denominator, 1.0)
    return jnp.where(denominator > 0, gap**2 / safe * density, 0.0)
