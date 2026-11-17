"""Finite-sample nulls: permutation, split-half and Gaussian surrogates.

A null gives the values that a statistic takes when the quantity it measures is
absent, so that an observed value can be compared with it. Every null here
gives its sides or sets the effective sample size of the inputs it compares,
not their sample counts, so that the null scatters as the observed value does.

Permutation
    :func:`permutation_null` pools the samples of the two sets of a comparison
    and splits the pool at random into two disjoint sides, for every draw. Each
    set keeps its weights, and the side size follows from the sample counts,
    the weights and the effective sample sizes of both sets
    (:func:`permutation_side_size`).
Split-half
    :func:`split_half_null` cuts the reference alone into two sides, with its
    own weights, whose size follows from the effective sample sizes of the
    reference and the approximation (:func:`split_half_side_size`). It is the
    alternative for a comparison to a much larger reference.
Gaussian surrogates
    :func:`gaussian_surrogates` draws sets from the moment-matched Gaussian of
    one posterior, for the diagnostics of its shape and dependence, each set
    of the posterior's effective sample size (:func:`surrogate_size`).

The permutation and split-half constructions return a :class:`NullPlan`, the
pool and the positions of the two sides of every draw, and
:func:`null_draws` evaluates a statistic on them::

    import jax
    from mixed_signals import zoo
    from mixed_signals.metrics.moments import mean_shift
    from mixed_signals.nulls import null_draws, permutation_null

    case = zoo.shifted_gaussians(0.5, param_count=2)
    sets = case.sample(jax.random.key(0), 1_000).samples[0]
    plan = permutation_null(sets[0], None, sets[1], None, key=jax.random.key(1))
    null = null_draws(
        lambda first, first_weights, second, second_weights: mean_shift(
            first, first_weights, second, second_weights, case.space
        ),
        plan,
    )  # shaped (2, 200): one row of draws per parameter
"""

from mixed_signals.nulls.permutation import permutation_null, permutation_side_size
from mixed_signals.nulls.plan import (
    Construction,
    NullPlan,
    null_draws,
    random_sides,
)
from mixed_signals.nulls.split_half import split_half_null, split_half_side_size
from mixed_signals.nulls.surrogates import (
    CONSTRUCTION,
    gaussian_surrogates,
    surrogate_size,
)

__all__ = [
    "CONSTRUCTION",
    "Construction",
    "NullPlan",
    "gaussian_surrogates",
    "null_draws",
    "permutation_null",
    "permutation_side_size",
    "random_sides",
    "split_half_null",
    "split_half_side_size",
    "surrogate_size",
]
