"""SampleSet-level engines: ``compare``, ``diagnose``, ``reweight`` and their dry run.

The engines are plain functions that take samples and a parameter space and
return a result table, alone or with the objects it describes; ``compare`` and
``diagnose`` also take a PRNG key or an integer seed.

:func:`~mixed_signals.engines.compare.compare`
    Compares pairs of posteriors, from one set or several, in mode A (the
    reference against each other posterior) or mode B (all pairs): the metrics
    of :data:`RECIPES` on whitened samples, each with a finite-sample null, σ_sig
    and an effect size, σ_eff or r_eff, with the thresholds and the flag of the
    row. :data:`DEFAULT_METRICS` is the fast set.
:func:`~mixed_signals.engines.diagnose.diagnose`
    Checks posteriors one by one, from MCMC chains or flat samples: the health of
    the chains, the shape, dependence, modes and railing of each posterior,
    against Gaussian surrogates. It returns a
    :class:`~mixed_signals.engines.diagnose.Diagnosis` with the table and the
    reduction of each run of chains. :data:`DIAGNOSTIC_FAMILIES` names its
    families.
:func:`~mixed_signals.engines.reweight.reweight`
    Reweights samples to a new prior, a new likelihood or the target of an
    approximation, with the rows ``pareto_k``, ``importance_ess`` and
    ``log_evidence_ratio``.
Dry run
    :func:`~mixed_signals.engines.cost.estimate_compare_cost` and
    :func:`~mixed_signals.engines.cost.estimate_diagnose_cost` give the estimated
    evaluations and seconds of a call, from counts and shapes only. ``compare``
    and ``diagnose`` call the estimate first and raise a
    :class:`~mixed_signals.errors.CostWarning` above their ``budget``,
    :data:`COST_BUDGET` by default.
"""

from mixed_signals._config import COST_BUDGET
from mixed_signals.engines.compare import DEFAULT_METRICS, RECIPES, compare
from mixed_signals.engines.cost import (
    CostEstimate,
    estimate_compare_cost,
    estimate_diagnose_cost,
)
from mixed_signals.engines.diagnose import DIAGNOSTIC_FAMILIES, Diagnosis, diagnose
from mixed_signals.engines.reweight import (
    TARGET_SUFFIX,
    Proposal,
    Reweighting,
    reweight,
)

__all__ = [
    "COST_BUDGET",
    "DEFAULT_METRICS",
    "DIAGNOSTIC_FAMILIES",
    "RECIPES",
    "TARGET_SUFFIX",
    "CostEstimate",
    "Diagnosis",
    "Proposal",
    "Reweighting",
    "compare",
    "diagnose",
    "estimate_compare_cost",
    "estimate_diagnose_cost",
    "reweight",
]
