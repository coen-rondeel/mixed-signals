"""Permutation null of a comparison: the pooled samples of two sets, split at random.

Construction
    The samples of both sets are pooled, and each draw splits the pool at
    random into two disjoint sides of equal size. If both sets come from one
    distribution, a random split is as likely as the observed one, so a
    statistic of the two sides scatters as the observed statistic does
    (Good 2005). :func:`permutation_null` builds the
    :class:`~mixed_signals.nulls.plan.NullPlan` and
    :func:`~mixed_signals.nulls.plan.null_draws` evaluates a statistic on its
    draws.
Weights
    Every sample keeps its weight. Each set's weights are normalised to sum to
    one, so that neither set outweighs the other by the scale of its weights,
    and the sides of a draw are weighted samples of the pool. Zoo check: a set
    with importance weights (a proposal 1.5 times wider than the target,
    N = 4 000, Kish effective sample size 2 766) against 2 000 equal-weight
    draws of the same target, 2 000 replicates and 199 permutations rejected
    at the 5% level in 4.8, 5.1 and 4.3% of the replicates for the mean shift,
    the KS statistic and the log width ratio (binomial error 0.5%). Sides with
    the sample counts of the two sets gave 3.0, 2.7 and 3.3%, and resampling
    both sets to equal weights before permuting gave 16.6, 33.7 and 16.6%.
    Equal-weight sets of 4 000 and 1 000 samples rejected in 5.2, 5.4 and 4.9%
    of the replicates with equal sides, and in 1.0, 0.3 and 1.0% with the
    original side sizes.
Correlation
    Chains stay thinned by the autocorrelation time τ. The observed value uses
    all samples, and the sides of the null carry the effective sample size.
    Zoo check: AR(1) chains with τ = 19, reduced by
    :func:`~mixed_signals.mcmc.reduce.reduce_chains`, keep τ_k = 1.37
    (N = 2 944, effective sample size 2 116). Over 2 000 pairs of identical
    runs, a permutation of all samples rejected at the 5% level in 8.0, 7.3 and
    5.3% of the pairs (mean shift, KS, width), cutting both sets to the
    effective sample size at random first gave 7.7, 6.6 and 4.7%, and null
    sides of the effective sample size 3.6, 1.8 and 1.5%. The last form is the
    conservative one that the developers chose. One effective sample size, that
    of the slowest parameter's mean, over-corrects other parameters and
    statistics.
Side size
    With N_i samples, Kish sizes K_i of the weights and effective sample sizes
    n_i (the given ones, else K_i), each side holds
    s = round(½ (N₁ + N₂) (1/K₁ + 1/K₂) / (1/n₁ + 1/n₂)),
    clipped to [2, ⌊(N₁ + N₂)/2⌋]. To first order, it makes the variance of a
    weighted mean difference between the two sides equal to that between the
    two sets, σ²(1/n₁ + 1/n₂), for samples of variance σ² whose variance of a
    weighted mean is σ² Σwᵢ²/(Σwᵢ)² (Kish 1965). It gives ½(N₁ + N₂) for
    independent weighted samples, and the harmonic mean of n₁ and n₂ for
    equal-weight sets of equal size.

References:
    :cite:t:`good2005permutation`; :cite:t:`kish1965survey`.
"""

import jax.numpy as jnp
from jax.typing import ArrayLike
from jaxtyping import PRNGKeyArray

from mixed_signals._config import NULL_DRAWS
from mixed_signals.nulls.plan import (
    Construction,
    NullPlan,
    _read_set,
    _set_sizes,
    random_sides,
)


def permutation_side_size(
    first_log_weights: ArrayLike,
    second_log_weights: ArrayLike,
    *,
    first_n_eff: float | None = None,
    second_n_eff: float | None = None,
) -> int:
    """Return the number of samples in each side of a permutation null.

    The size is round(½ (N₁ + N₂) (1/K₁ + 1/K₂) / (1/n₁ + 1/n₂)) for N_i
    samples, Kish sizes K_i and effective sample sizes n_i, clipped to
    [2, ⌊(N₁ + N₂)/2⌋]. Call this function outside ``jax.jit``, with concrete
    weights.

    Args:
        first_log_weights: Log weights of the first set, shaped (N₁,).
        second_log_weights: Log weights of the second set, shaped (N₂,).
        first_n_eff: Effective sample size of the first set, for correlated
            samples; its Kish size when ``None``.
        second_n_eff: Effective sample size of the second set; its Kish size
            when ``None``.

    Returns:
        The size of each side.

    Raises:
        ValueError: If a weight array is not one-dimensional, a Kish or
            effective sample size is below one or not finite, or the two sets
            hold fewer than four samples together.

    References:
        :cite:t:`kish1965survey`; :cite:t:`good2005permutation`.
    """
    first_count, first_kish, first_effective = _set_sizes(
        "the first set", first_log_weights, first_n_eff
    )
    second_count, second_kish, second_effective = _set_sizes(
        "the second set", second_log_weights, second_n_eff
    )
    pool_size = first_count + second_count
    largest = pool_size // 2
    if largest < 2:
        raise ValueError(
            f"two sides of at least 2 samples need a pool of 4, got {pool_size}"
        )
    size = round(
        0.5
        * pool_size
        * (1 / first_kish + 1 / second_kish)
        / (1 / first_effective + 1 / second_effective)
    )
    return min(max(2, size), largest)


def permutation_null(
    first: ArrayLike,
    first_log_weights: ArrayLike | None,
    second: ArrayLike,
    second_log_weights: ArrayLike | None,
    *,
    key: PRNGKeyArray,
    first_n_eff: float | None = None,
    second_n_eff: float | None = None,
    draws: int = NULL_DRAWS,
) -> NullPlan:
    """Pool two weighted sets and split the pool at random, for every draw.

    The pool holds the first set and then the second, each with its weights
    normalised to sum to one. Every draw takes two disjoint sides of
    :func:`permutation_side_size` samples. The function only moves rows, so it
    needs no parameter space. Call it outside ``jax.jit``, with concrete
    weights.

    Args:
        first: Samples of the first set, shaped (N, D).
        first_log_weights: Their log weights, shaped (N,); equal weights when
            ``None``.
        second: Samples of the second set, shaped (M, D).
        second_log_weights: Their log weights, shaped (M,); equal weights when
            ``None``.
        key: JAX PRNG key, derived with :func:`~mixed_signals.prng.derive_key`
            from the path of the row, ending in ``"null"``.
        first_n_eff: Effective sample size of the first set, for correlated
            samples; its Kish size when ``None``.
        second_n_eff: Effective sample size of the second set; its Kish size
            when ``None``.
        draws: Number of draws.

    Returns:
        The plan: the pool, its log weights and the positions of both sides of
        every draw.

    Raises:
        TypeError: If an input is not real-valued, or ``draws`` is not an
            integer.
        ValueError: If the shapes do not fit, the sets have different numbers
            of parameters, concrete values are invalid, or an effective sample
            size is below one or not finite.

    Warns:
        PrecisionWarning: If an input is narrower than float64.

    References:
        :cite:t:`good2005permutation`; :cite:t:`kish1965survey`.
    """
    first_samples, first_weights = _read_set("first", first, first_log_weights)
    second_samples, second_weights = _read_set("second", second, second_log_weights)
    if first_samples.shape[1] != second_samples.shape[1]:
        raise ValueError(
            "both sets need the same parameters, got"
            f" {first_samples.shape[1]} and {second_samples.shape[1]}"
        )
    size = permutation_side_size(
        first_weights,
        second_weights,
        first_n_eff=first_n_eff,
        second_n_eff=second_n_eff,
    )
    pool = jnp.concatenate([first_samples, second_samples])
    first_sides, second_sides = random_sides(key, pool.shape[0], size, draws=draws)
    return NullPlan(
        construction=Construction.PERMUTATION,
        samples=pool,
        log_weights=jnp.concatenate([first_weights, second_weights]),
        first=first_sides,
        second=second_sides,
    )
