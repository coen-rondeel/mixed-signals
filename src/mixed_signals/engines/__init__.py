"""SampleSet-level engines: ``diagnose``, ``compare``, ``reweight``, data space.

So far :func:`~mixed_signals.engines.reweight.reweight` exists: it reweights a
:class:`~mixed_signals.core.samples.SampleSet` to a new prior, a new likelihood
or the target of an approximation, and returns the reweighted set with a result
table of the Pareto k̂, the effective sample size and the evidence ratio.
"""

from mixed_signals.engines.reweight import (
    TARGET_SUFFIX,
    Proposal,
    Reweighting,
    reweight,
)

__all__ = ["TARGET_SUFFIX", "Proposal", "Reweighting", "reweight"]
