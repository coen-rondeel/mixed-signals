"""MCMC tools: autocorrelation time, effective sample size and chain reduction.

The kernels take samples shaped (step, walker, param) after burn-in, the layout
of emcee and of the cold chain of :class:`~mixed_signals.core.chains.Chains`;
:func:`~mixed_signals.mcmc.reduce.reduce_chains` takes the chains themselves
and returns a :class:`~mixed_signals.core.samples.SampleSet` with its
``n_eff``::

    import jax
    from mixed_signals import zoo
    from mixed_signals.mcmc import integrated_autocorrelation_time, reduce_chains

    chains = zoo.ar1_chains([0.9]).sample(jax.random.key(0), 5_000)
    integrated_autocorrelation_time(chains.cold().samples[:, 0, :, 0, :])  # ≈ 19
    reduction = reduce_chains(chains, burn_in=100)
    reduction.record.thin  # about τ after the burn-in
    reduction.sample_set.n_eff  # N/τ_k of the kept samples

:mod:`mixed_signals.mcmc.autocorr` estimates τ, :mod:`mixed_signals.mcmc.ess`
holds the effective sample size, the split-chain Kolmogorov-Smirnov test and
the informational R̂, and :mod:`mixed_signals.mcmc.reduce` the reduction.
"""

from mixed_signals.mcmc.autocorr import (
    MINIMUM_LENGTH_IN_TAU,
    MINIMUM_STEPS,
    autocorrelation_function,
    integrated_autocorrelation_time,
)
from mixed_signals.mcmc.ess import (
    MINIMUM_RHAT_STEPS,
    MINIMUM_SPLIT_EFFECTIVE_SAMPLE_SIZE,
    SplitChainTest,
    effective_sample_size,
    rank_normalized_split_rhat,
    split_chain_ks,
)
from mixed_signals.mcmc.reduce import (
    TAU_THINNING,
    Reduction,
    ReductionRecord,
    hash_chains,
    reduce_chains,
)

__all__ = [
    "MINIMUM_LENGTH_IN_TAU",
    "MINIMUM_RHAT_STEPS",
    "MINIMUM_SPLIT_EFFECTIVE_SAMPLE_SIZE",
    "MINIMUM_STEPS",
    "TAU_THINNING",
    "Reduction",
    "ReductionRecord",
    "SplitChainTest",
    "autocorrelation_function",
    "effective_sample_size",
    "hash_chains",
    "integrated_autocorrelation_time",
    "rank_normalized_split_rhat",
    "reduce_chains",
    "split_chain_ks",
]
