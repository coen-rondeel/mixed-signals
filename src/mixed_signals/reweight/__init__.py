"""Importance reweighting to new priors or likelihoods, evidence ratios and PSIS.

:mod:`mixed_signals.reweight.importance` holds the evidence ratio of old and new
log weights with its error, the relative efficiency of correlated draws and the
effective sample size of reweighted draws, and
:mod:`mixed_signals.reweight.psis` the Pareto fit to the tail of the weights,
its threshold and Pareto smoothing. The engine function
:func:`mixed_signals.engines.reweight.reweight` joins them with the new
densities of a prior and a likelihood::

    import math

    import jax

    from mixed_signals.reweight import (
        pareto_k_threshold,
        pareto_tail,
        pareto_tail_length,
    )

    # Draws from N(0, 0.3²) reweighted to N(0, 1) have heavy-tailed weights.
    draws = 0.3 * jax.random.normal(jax.random.key(0), (10_000,))
    log_ratio = 0.5 * (1 / 0.3**2 - 1) * draws**2 + math.log(0.3)
    tail = pareto_tail(log_ratio, tail_draws=pareto_tail_length(10_000))
    float(tail.shape) > pareto_k_threshold(10_000)
"""

from mixed_signals.reweight.importance import (
    EvidenceRatio,
    log_evidence_ratio,
    relative_efficiency,
    reweighted_effective_sample_size,
)
from mixed_signals.reweight.psis import (
    MINIMUM_TAIL_DRAWS,
    PARETO_K_LIMIT,
    ParetoTail,
    SmoothedWeights,
    pareto_k_threshold,
    pareto_smooth,
    pareto_tail,
    pareto_tail_length,
)

__all__ = [
    "MINIMUM_TAIL_DRAWS",
    "PARETO_K_LIMIT",
    "EvidenceRatio",
    "ParetoTail",
    "SmoothedWeights",
    "log_evidence_ratio",
    "pareto_k_threshold",
    "pareto_smooth",
    "pareto_tail",
    "pareto_tail_length",
    "relative_efficiency",
    "reweighted_effective_sample_size",
]
