"""Split-half null of a comparison to a reference: the reference alone, cut in two.

Construction
    The pool is the reference with its own weights. Each draw cuts it at
    random into two disjoint sides, and a statistic of the two sides scatters
    as a comparison of two independent samples of one distribution does
    (Good 2005). It is the alternative to the permutation null of
    :mod:`mixed_signals.nulls.permutation` for a comparison of an
    approximation to a reference that is much larger.
Side size
    With N_r reference samples, the Kish size K_r of their weights and the
    effective sample sizes n_r of the reference (the given one, else K_r) and
    n_a of the approximation, each side holds
    s = round((N_r/K_r) · 2/(1/n_r + 1/n_a)),
    clipped to [2, ⌊N_r/2⌋] (:func:`split_half_side_size`). The sides then
    carry the effective sample sizes whose weighted mean difference has the
    variance of the comparison, σ²(1/n_r + 1/n_a) (Kish 1965).
Clip
    Two disjoint sides cannot hold more than ⌊N_r/2⌋ samples each. The clip
    acts unless the reference holds about three times the effective samples of
    the approximation, and then the null is wider than the comparison, so its
    p-values are conservative; :func:`split_half_side_size` warns.
Zoo check
    A reference of 10 000 and an approximation of 1 000 equal-weight samples,
    1 500 replicates, rejected at the 5% level in 4.6, 4.7 and 5.7% of the
    replicates for the mean shift, the KS statistic and the log width ratio
    with these sides, against 0.5, 0.2 and 0.3% with sides of the
    approximation's size (see :ref:`validation-split-half`).

References:
    :cite:t:`good2005permutation`; :cite:t:`kish1965survey`.
"""

import os
import warnings
from typing import Final

from jax.typing import ArrayLike
from jaxtyping import PRNGKeyArray

from mixed_signals._config import NULL_DRAWS
from mixed_signals.errors import ValidityWarning
from mixed_signals.nulls.plan import (
    Construction,
    NullPlan,
    _check_size,
    _read_set,
    _set_sizes,
    random_sides,
)

_PACKAGE_DIRECTORY: Final = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def split_half_side_size(
    reference_log_weights: ArrayLike,
    *,
    approximation_n_eff: float,
    reference_n_eff: float | None = None,
) -> int:
    """Return the number of samples in each side of a split-half null.

    The size is round((N_r/K_r) · 2/(1/n_r + 1/n_a)) for N_r samples, a Kish
    size K_r and an effective sample size n_r of the reference and n_a of the
    approximation, clipped to [2, ⌊N_r/2⌋]. Call this function outside
    ``jax.jit``, with concrete weights.

    Args:
        reference_log_weights: Log weights of the reference, shaped (N_r,).
        approximation_n_eff: Effective sample size n_a of the approximation.
        reference_n_eff: Effective sample size n_r of the reference, for
            correlated samples; its Kish size when ``None``.

    Returns:
        The size of each side.

    Raises:
        ValueError: If the weights are not one-dimensional, a Kish or
            effective sample size is below one or not finite, or the
            reference holds fewer than four samples.

    Warns:
        ValidityWarning: If the size exceeds ⌊N_r/2⌋ and is clipped to it, so
            that the null is wider than the comparison.

    References:
        :cite:t:`kish1965survey`; :cite:t:`good2005permutation`.
    """
    count, kish_size, reference_effective = _set_sizes(
        "the reference", reference_log_weights, reference_n_eff
    )
    approximation_effective = float(approximation_n_eff)
    _check_size("effective", "the approximation", approximation_effective)
    largest = count // 2
    if largest < 2:
        raise ValueError(
            f"two sides of at least 2 samples need a reference of 4, got {count}"
        )
    size = round(
        count / kish_size * 2 / (1 / reference_effective + 1 / approximation_effective)
    )
    if size > largest:
        warnings.warn(
            ValidityWarning(
                f"split-half null: a reference of {count} samples (effective size"
                f" {reference_effective:.0f}) cannot match an approximation of"
                f" {approximation_effective:.0f} effective samples; its sides are"
                f" limited to {largest} samples, so the null is wider than the"
                " comparison and the p-values are conservative"
            ),
            skip_file_prefixes=(_PACKAGE_DIRECTORY,),
        )
    return min(max(2, size), largest)


def split_half_null(
    reference: ArrayLike,
    reference_log_weights: ArrayLike | None,
    *,
    approximation_n_eff: float,
    key: PRNGKeyArray,
    reference_n_eff: float | None = None,
    draws: int = NULL_DRAWS,
) -> NullPlan:
    """Cut a weighted reference at random into two sides, for every draw.

    The pool is the reference with its log weights normalised to sum to one.
    Every draw takes two disjoint sides of :func:`split_half_side_size`
    samples. The function only moves rows, so it needs no parameter space.
    Call it outside ``jax.jit``, with concrete weights.

    Args:
        reference: Samples of the reference, shaped (N, D).
        reference_log_weights: Their log weights, shaped (N,); equal weights
            when ``None``.
        approximation_n_eff: Effective sample size of the approximation.
        key: JAX PRNG key, derived with :func:`~mixed_signals.prng.derive_key`
            from the path of the row, ending in ``"null"``.
        reference_n_eff: Effective sample size of the reference, for
            correlated samples; its Kish size when ``None``.
        draws: Number of draws.

    Returns:
        The plan: the pool, its log weights and the positions of both sides of
        every draw.

    Raises:
        TypeError: If an input is not real-valued, or ``draws`` is not an
            integer.
        ValueError: If the shapes do not fit, concrete values are invalid, or
            an effective sample size is below one or not finite.

    Warns:
        PrecisionWarning: If an input is narrower than float64.
        ValidityWarning: As :func:`split_half_side_size`, if the sides are
            clipped.

    References:
        :cite:t:`good2005permutation`; :cite:t:`kish1965survey`.
    """
    samples, log_weights = _read_set("reference", reference, reference_log_weights)
    size = split_half_side_size(
        log_weights,
        approximation_n_eff=approximation_n_eff,
        reference_n_eff=reference_n_eff,
    )
    first_sides, second_sides = random_sides(key, samples.shape[0], size, draws=draws)
    return NullPlan(
        construction=Construction.SPLIT_HALF,
        samples=samples,
        log_weights=log_weights,
        first=first_sides,
        second=second_sides,
    )
