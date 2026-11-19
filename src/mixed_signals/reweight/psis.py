"""Pareto smoothed importance sampling: k̂, its threshold and smoothed weights.

The right tail of importance weights follows a generalized Pareto distribution
with shape k, so the weights have finite moments only below the order 1/k: the
variance is finite for k below 1/2 and the mean for k below 1 (Vehtari et al.
2024). :func:`pareto_tail` fits the shape of the tail, k̂, with the estimator of
Zhang and Stephens (2009), and :func:`pareto_k_threshold` says when k̂ is too
high for estimates from the weights to be reliable.

Tail length
    The tail holds the M largest weights, M = ⌊min(0.2 S, 3 √(S/R))⌋ for S
    draws and a relative efficiency R of the draws (:func:`pareto_tail_length`).
    Independent draws have R = 1, which is Algorithm 1 of Vehtari et al.
    (2024); the factor 1/R lengthens the tail for correlated draws, as in
    Appendix H of that paper and in the ``n_pareto`` function of the loo
    package (Vehtari et al. 2026). Fewer than 5 tail draws are too few for a
    fit.
Shape prior
    The fit uses the weakly informative prior on the shape of Vehtari et al.
    (2024, Appendix G), k ← (L k + 10 · 0.5)/(L + 10) for L excesses, which
    :func:`~mixed_signals.sigma.pareto.generalized_pareto_fit` offers as an
    option. The scale comes from the plain shape, as in the posterior R
    package (Bürkner et al. 2026).
Threshold
    :func:`pareto_k_threshold` is min(1 − 1/log₁₀ S, 0.7), the shape above
    which estimates from S draws are likely unstable or biased (Vehtari et
    al. 2024, Algorithm 1 and Eq. 13). For S above about 2 154 it is 0.7, and
    it falls for fewer draws.
Draws
    S counts the samples whose weight is positive. Samples of weight zero lie
    below every weight in the tail, so the right tail of the ratios has the
    same index with or without them, and fewer draws lower the threshold.
Flat tails
    When the largest weight exceeds the (M + 1)-th largest by less than 1e-12
    of itself, the weights are bounded at the cutoff, such as weights that
    take one value, or the ratio of two uniform priors. That is the limit k →
    −∞ of a generalized Pareto tail, so k̂ is −∞, with no flag, and smoothing
    leaves the weights unchanged. When fewer than M + 1 weights are positive,
    k̂ is NaN.
Smoothing
    :func:`pareto_smooth` replaces the M largest weights by the expected order
    statistics of the fitted tail, u + (σ/k)((1 − p)^(−k) − 1) at p = (z −
    1/2)/M for z = 1, ..., M (Vehtari et al. 2024, Algorithm 1; the
    percentile points of Blom 1958), with u the largest weight outside the
    tail, truncated at the largest raw weight. It never changes the weights
    outside the tail. Weights stay raw unless a caller asks for smoothing.

Zoo checks
    The reweighting checks of the validation build
    (``scripts/zoo_validation/reweighting.py``) test k̂ on zoo cases with known
    weights. On exact Pareto weights, those of exponential proposals and of
    Gaussians narrower along two axes, k̂ recovers the true shape, and the
    shape prior moves it towards 0.5 by the factor M/(M + 10). Narrowing one
    axis of a Gaussian gives a k̂ below 1 − r² from a true shape of about 0.64,
    and narrowing all eight axes a k̂ above it, because the weights are
    chi-squared tails with one or eight degrees of freedom, whose slowly
    varying factor biases the fit at finite S. A shift between two Gaussians
    has the true shape 0, but k̂ rises with the shift and flags in most sets
    from about 2.5 widths.

References:
    :cite:t:`vehtari2024pareto`; :cite:t:`zhang2009new`;
    :cite:t:`vehtari2026loo`; :cite:t:`burkner2026posterior`;
    :cite:t:`blom1958statistical`.
"""

import math
from typing import Final, NamedTuple

import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike
from jaxtyping import Array, Float

from mixed_signals.sigma.pareto import _jit, generalized_pareto_fit

PARETO_K_LIMIT: Final = 0.7
"""Threshold of k̂ from about 2 154 draws up, where 1 − 1/log₁₀ S reaches it."""

MINIMUM_TAIL_DRAWS: Final = 5
"""Fewest tail draws; the enough-tail-samples rule of loo and the floor of the fit."""

_TAIL_FRACTION: Final = 0.2
"""Largest share of the draws in the tail, from Algorithm 1 of Vehtari et al. (2024)."""

_TAIL_FACTOR: Final = 3
"""Factor of √(S/R) in the tail length, from Vehtari et al. (2024)."""

# * A tail that spans less than this share of its largest weight counts as flat.
_FLAT_TAIL_TOLERANCE: Final = 1e-12


class ParetoTail(NamedTuple):
    """The generalized Pareto fit to the right tail of a set of weights."""

    shape: Float[Array, ""]
    """The shape k̂; −∞ for a flat tail and NaN when it is undefined."""
    scale: Float[Array, ""]
    """The scale σ of the excesses, in units of the largest weight; 0 when flat."""
    cutoff: Float[Array, ""]
    """Log of the largest weight outside the tail, less that of the largest weight."""


class SmoothedWeights(NamedTuple):
    """Log weights after Pareto smoothing, with the shape of their tail."""

    log_weights: Float[Array, " sample"]
    """The smoothed log weights, in the order of the input."""
    shape: Float[Array, ""]
    """The shape k̂ of the tail, as :class:`ParetoTail` has it."""


def pareto_tail_length(draws: int, relative_efficiency: float = 1.0) -> int:
    """Return the number of draws M in the tail of S weights.

    M = ⌊min(0.2 S, 3 √(S/R))⌋ for a relative efficiency R of the draws. For
    independent draws, R = 1, this is Algorithm 1 of Vehtari et al. (2024),
    and the factor 1/R is the generalization to MCMC draws, as the
    ``n_pareto`` function of loo applies it. At most a fifth of the draws, and
    for S above 225 fewer than that, so that the tail stays a tail.

    Args:
        draws: The number of draws S, at least 1.
        relative_efficiency: The relative efficiency R, finite and positive;
            the effective sample size over the Kish value of the weights.

    Returns:
        The tail length M, which is below 5, too few for a fit, when S is
        below 25.

    Raises:
        ValueError: If ``draws`` is not an integer of at least 1, or the
            relative efficiency is not finite and positive.

    Example:
        >>> [pareto_tail_length(count) for count in (100, 1_000, 10_000)]
        [20, 94, 300]

    References:
        :cite:t:`vehtari2024pareto`; :cite:t:`vehtari2026loo`.
    """
    if isinstance(draws, bool) or not isinstance(draws, int | np.integer):
        raise ValueError(f"draws must be an integer, got {draws!r}")
    if draws < 1:
        raise ValueError(f"draws must be at least 1, got {draws}")
    if not (math.isfinite(relative_efficiency) and relative_efficiency > 0):
        raise ValueError(
            "relative_efficiency must be finite and positive, got"
            f" {relative_efficiency}"
        )
    # * Integer division gives the floor of a fifth without rounding in floats.
    return min(
        int(draws) // 5,
        math.floor(_TAIL_FACTOR * math.sqrt(draws / relative_efficiency)),
    )


def pareto_k_threshold(draws: int) -> float:
    """Return the k̂ above which estimates from S weights are not reliable.

    The threshold is min(1 − 1/log₁₀ S, 0.7) (Vehtari et al. 2024, Algorithm 1
    and Eq. 13): the shape up to which the error of the estimate stays small
    for S draws. It is 0.5 at S = 100, 2/3 at S = 1 000, and 0.7 from about
    2 154 draws on.

    Args:
        draws: The number of draws S, at least 2.

    Returns:
        The threshold.

    Raises:
        ValueError: If ``draws`` is below 2, where the formula is not defined.

    Example:
        >>> round(pareto_k_threshold(1_000), 4)
        0.6667

    References:
        :cite:t:`vehtari2024pareto`.
    """
    if draws < 2:
        raise ValueError(f"draws must be at least 2, got {draws}")
    return min(1 - 1 / math.log10(draws), PARETO_K_LIMIT)


@_jit("tail_draws", "shape_prior")
def pareto_tail(
    log_weights: Float[ArrayLike, " sample"],
    *,
    tail_draws: int,
    shape_prior: bool = True,
) -> ParetoTail:
    """Fit a generalized Pareto distribution to the right tail of one set of weights.

    The weights are scaled by their maximum, and the tail holds the
    ``tail_draws`` largest. Its excesses over the next largest weight, the
    cutoff, go to
    :func:`~mixed_signals.sigma.pareto.generalized_pareto_fit`. A flat tail,
    whose largest weight exceeds the cutoff by less than 1e-12 of itself, has
    k̂ = −∞ and scale 0, the limit of bounded weights; fewer than
    ``tail_draws`` + 1 positive weights, or no positive weight, give NaN. The
    function acts on the last axis and runs under :func:`jax.jit` and
    :func:`jax.vmap` for a static ``tail_draws``.

    Args:
        log_weights: Natural logarithms of the weights of one set, shaped
            (N,); −∞ for a weight of zero.
        tail_draws: The number of draws M in the tail, from
            :func:`pareto_tail_length`; at least 5 and at most N − 1.
        shape_prior: Whether to shrink k̂ with the weakly informative prior
            of Vehtari et al. (2024, Appendix G). Static under
            :func:`jax.jit`.

    Returns:
        The shape, the scale and the cutoff of the tail.

    Raises:
        ValueError: If ``tail_draws`` lies outside 5 to N − 1.

    References:
        :cite:t:`vehtari2024pareto`; :cite:t:`zhang2009new`.
    """
    shifted, _ = _scaled_log_weights(log_weights)
    sample_count = shifted.shape[0]
    _check_tail_draws(tail_draws, sample_count)
    ordered = jnp.sort(shifted)
    cutoff = ordered[sample_count - tail_draws - 1]
    # * Excesses in units of the largest weight; a log weight of −∞ gives zero.
    excesses = jnp.exp(ordered[sample_count - tail_draws :]) - jnp.exp(cutoff)
    fit = generalized_pareto_fit(excesses, shape_prior=shape_prior)
    undefined = cutoff == -jnp.inf
    flat = -jnp.expm1(cutoff) < _FLAT_TAIL_TOLERANCE
    shape = jnp.where(undefined, jnp.nan, jnp.where(flat, -jnp.inf, fit.shape))
    scale = jnp.where(undefined, jnp.nan, jnp.where(flat, 0.0, fit.scale))
    return ParetoTail(shape=shape, scale=scale, cutoff=cutoff)


@_jit("tail_draws", "shape_prior")
def pareto_smooth(
    log_weights: Float[ArrayLike, " sample"],
    *,
    tail_draws: int,
    shape_prior: bool = True,
) -> SmoothedWeights:
    """Smooth the right tail of one set of weights with its fitted Pareto tail.

    The ``tail_draws`` largest weights, in rank order, become the expected
    order statistics of the fitted tail, u + (σ/k)((1 − p)^(−k) − 1) at
    p = (z − 1/2)/M for z = 1, ..., M, with z = 1 for the smallest tail weight
    and u the cutoff weight, in units of the largest weight. None exceeds the
    largest raw weight. The quantile term is −σ log(1 − p) for k = 0. The
    weights outside the tail never change, and the set is returned unchanged
    when the shape is not finite: a flat tail needs no smoothing, and an
    undefined one has no fit. Smoothing lowers the variance of estimates at
    the price of a bias that grows with k̂ (Vehtari et al. 2024).

    Args:
        log_weights: Natural logarithms of the weights of one set, shaped
            (N,); −∞ for a weight of zero.
        tail_draws: The number of draws M in the tail; at least 5 and at most
            N − 1.
        shape_prior: Whether the fit uses the weakly informative prior on the
            shape. Static under :func:`jax.jit`.

    Returns:
        The smoothed log weights in the order of the input, and the shape k̂ of
        :func:`pareto_tail`.

    Raises:
        ValueError: If ``tail_draws`` lies outside 5 to N − 1.

    References:
        :cite:t:`vehtari2024pareto`; :cite:t:`blom1958statistical`.
    """
    shifted, anchor = _scaled_log_weights(log_weights)
    sample_count = shifted.shape[0]
    tail = pareto_tail(log_weights, tail_draws=tail_draws, shape_prior=shape_prior)
    positions = jnp.argsort(shifted)[sample_count - tail_draws :]
    # * Blom's percentile points (z − 1/2)/M, from the smallest weight of the tail.
    probabilities = (jnp.arange(1, tail_draws + 1) - 0.5) / tail_draws
    quantile_term = _excess_quantile(probabilities, tail.shape, tail.scale)
    smoothed_tail = jnp.minimum(jnp.log(jnp.exp(tail.cutoff) + quantile_term), 0.0)
    original = jnp.asarray(log_weights, dtype=jnp.float64)
    # * Only tail positions are written, so every other weight keeps its exact value.
    replaced = original.at[positions].set(smoothed_tail + anchor)
    smoothed = jnp.where(jnp.isfinite(tail.shape), replaced, original)
    return SmoothedWeights(log_weights=smoothed, shape=tail.shape)


def _excess_quantile(
    probability: Float[Array, " draw"],
    shape: Float[Array, ""],
    scale: Float[Array, ""],
) -> Float[Array, " draw"]:
    """Return the quantiles (σ/k)((1 − p)^(−k) − 1) of the excesses of a Pareto tail.

    The limit −σ log(1 − p) holds at k = 0.
    """
    log_survival = jnp.log1p(-probability)
    # * A zero shape takes the exponential limit; the other branch divides by one.
    zero_shape = shape == 0
    safe_shape = jnp.where(zero_shape, 1.0, shape)
    return jnp.where(
        zero_shape,
        -scale * log_survival,
        scale / safe_shape * jnp.expm1(-safe_shape * log_survival),
    )


def _scaled_log_weights(
    log_weights: Float[ArrayLike, " sample"],
) -> tuple[Float[Array, " sample"], Float[Array, ""]]:
    """Return the log weights less their maximum, and that maximum.

    A set without a positive weight has no maximum to subtract, so it keeps its
    values and the shift is zero.
    """
    values = jnp.asarray(log_weights, dtype=jnp.float64)
    peak = jnp.max(values)
    anchor = jnp.where(jnp.isfinite(peak), peak, 0.0)
    return values - anchor, anchor


def _check_tail_draws(tail_draws: int, sample_count: int) -> None:
    """Raise ValueError unless 5 <= tail_draws <= N - 1."""
    if not MINIMUM_TAIL_DRAWS <= tail_draws <= sample_count - 1:
        raise ValueError(
            f"tail_draws must lie between {MINIMUM_TAIL_DRAWS} and"
            f" {sample_count - 1}, one below the number of samples, got {tail_draws}"
        )
