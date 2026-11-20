"""The dry run of the engines: an estimate of their run time before they start.

:func:`estimate_compare_cost`, :func:`estimate_diagnose_cost` and
:func:`estimate_compare_signals_cost` take the arguments of
:func:`~mixed_signals.engines.compare.compare`,
:func:`~mixed_signals.engines.diagnose.diagnose` and
:func:`~mixed_signals.engines.dataspace.compare_signals` that change the cost,
read and check them with the code of the engines
(:mod:`mixed_signals.engines._arguments`), and return a table of the
evaluations of every statistic with their estimated seconds. They compute no
metric and use only counts and shapes; the data-space estimate also probes the
generator and the noise model on one sample to learn the number of data
entries. A transform does not change counts, so the estimates do not rely on
it. The engines call the same estimate first and
raise one :class:`~mixed_signals.errors.CostWarning` when it exceeds their
``budget``, which defaults to :data:`COST_BUDGET`, one hour.

Model
    One evaluation of a statistic costs c (n/10 000)^a · u (D + d₀)/(16 + d₀)
    seconds, with n the samples per set that it runs on, after its cap, D the
    number of parameters (E + C + 2S for E Euclidean axes, C circles and S
    spheres), u its work units, and no dimension factor where d₀ is "none".
    The table below gives c, a, d₀ and u. A statistic is evaluated once on
    the observed sets and once per null draw, and with the subsampling
    bootstrap once per half-sample pair on half the samples. The cost of a
    half-sample evaluation is that of min(⌈n/2⌉, cap) samples, which is
    2^(−a) times the cost of n samples where the cap does not act.
Grid and machine
    The constants were fitted on 2026-11-19 to the timings of every statistic
    at N of 1 000, 3 000 and 10 000 samples and D of 2, 8 and 16 parameters on
    the Apple M4 Pro of the developers, in High Power mode, with JAX 0.11.2 and
    float64, each statistic jitted as in the null draws. The estimate is for
    one CPU of that machine; another machine, a GPU or a different JAX version
    changes the seconds, though the ratios between statistics are likely to
    carry over. The two statistics of data space were fitted on 2026-11-20 on
    the same machine and mode to the timings at n of 1 000, 3 000 and 10 000
    samples per set, with 2n pooled rows for the matrix at M of 16, 256 and
    4 096 data entries, and for the evaluations in batches of 200 null draws,
    as the engine runs them, divided by 200. The costs check of
    ``scripts/zoo_validation/costs.py`` (see :ref:`validation-costs`) times
    every statistic at four points, (3 000, 16), (10 000, 2), (10 000, 8) and
    (10 000, 16), and checks the costly rows against the model; the page gains
    it at its next full build.
Misfits
    The worst misfit is the largest factor between the model and a measured
    time on the grid of the fit. The statistics that run per null draw or
    surrogate and take at least 10 ms, which set the cost of a call, fit
    within 1.36, the Sinkhorn divergence (1.36) and the mode projections (1.34)
    worst. Three cheap rows fit worse, because a fixed cost of dispatch
    dominates them: the mean shift and width ratio (2.87) and the marginal
    moments (2.14) below a millisecond, and the Anderson-Darling distance
    (2.11) below 5 ms, whose cost the model takes as linear in D. The GMM-BIC
    fit, which runs once per posterior, fits within 1.55: its time does not
    follow a power law in N (0.73 s at 3 000 samples and 0.67 s at 10 000 in
    16 parameters). The distance matrix of data space fits within 1.32, worst
    at 1 000 samples and 16 data entries (0.9 ms), and within 1.20 at the
    other eight points; the evaluations of the energy distance from a matrix
    fit within 1.03.
Constants
    ::

        Statistic                        c      a    d0   u          n cap    Misfit
        moments (shift, width ratio)     4.5e-5 0.62 none D                   2.87
        marginal (KS/Kuiper, AD, W1)     0.0022 1.13 none D                   1.18
        kde (JS and Hellinger)           0.031  1.01 none D                   1.27
        gaussian (every Gaussian term)   4.9e-4 0.58 9.4  1                   1.13
        knn_kl (one direction)           2.54   2.09 43   1          cap      1.05
        knn_js                           8.9    2.06 40   1          cap      1.09
        knn_hellinger                    5.18   2.10 42   1          cap      1.05
        mmd                              0.27   1.94 10.5 1          cap      1.15
        energy                           0.26   1.95 7.0  1          cap      1.13
        sinkhorn (one epsilon)           6.7    2.19 none 1          cap      1.36
        sliced_w2 (empirical)            0.57   1.10 none 1                   1.02
        sliced_w2 (gaussian)             5.2e-4 0.65 7.4  1                   1.07
        sliced_sinkhorn (per direction)  5.34   1.77 none directions sliced   1.08
        median_bandwidth (once)          0.30   0    none 1                   1.00
        sliced_sinkhorn_epsilon (once)   0.08   0    none directions          1.03
        marginal_moments                 3.9e-5 0.72 none D                   2.14
        anderson_darling                 5.5e-4 0.70 none D                   2.11
        mardia                           0.019  1.62 45   1          cap      1.17
        dependence (per pair)            3.6    2.63 none D(D-1)/2   pair_cap 1.31
        marginal_modes (per parameter)   0.0153 0.98 none D                   1.24
        projection_modes (per direction) 0.0159 0.96 27   directions          1.34
        gmm_bic (once per posterior)     0.69   0.32 0    1                   1.55
        signal_matrix (one pair)         0.146  2.10 56   1                   1.32
        signal_draw (one evaluation)     3.86e-3 2.00 none 1                  1.03

    In the column of the caps, ``cap`` and ``pair_cap`` are the arguments of the
    engines, and ``sliced`` is
    :data:`~mixed_signals.metrics.ot.sliced.SLICED_SINKHORN_CAP`. The directions
    of the sliced Sinkhorn divergence are the rows that
    :func:`~mixed_signals.metrics.ot.sliced.sliced_directions` returns for
    :data:`~mixed_signals.metrics.ot.sliced.SLICED_SINKHORN_DIRECTIONS` in that
    space, ⌈20/n⌉ · n for n = E + 2C + 3S coordinates. The calibration curve of
    the kNN KL map costs (len(KNN_KL_SHIFTS) + 1) · KNN_KL_REPLICATES
    evaluations of ``knn_kl`` at the counts of its direction, once per
    direction.
Comparisons
    A group is one statistic of one pair in one problem. A comparison runs the
    observed evaluation and ``draws`` null draws of each selected statistic,
    ``half_samples`` more at half size with the bootstrap, ``sinkhorn`` once
    per ε factor, and per comparison the median bandwidth (for ``mmd`` and
    ``sinkhorn``), the ε of the sliced Sinkhorn divergence and the KL curves.
    A statistic that serves several recipes counts once. The samples n are
    the larger of the two counts of the pair, an upper bound for sets of
    different sizes, and for the kNN rows the smaller of the positive counts
    of the two sets, also an upper bound, since rejection keeps fewer samples
    of a weighted set. The sides of a null draw are taken as long as the
    sets, an upper bound for correlated samples.
Diagnostics
    A group is one statistic of one posterior in one problem. The shape rows
    and the mode rows take one observed evaluation at N samples (Mardia's
    moments at min(N, cap)) and ``draws`` evaluations at the size of the
    surrogate sets, :func:`~mixed_signals.nulls.surrogates.surrogate_size` of
    the effective sample size and the cap. The dependence rows take one at
    min(N, pair_cap) and ``draws`` at the surrogate size for ``pair_cap``.
    The GMM-BIC fit runs once, and the mode projections run along at most
    MAX_COMPONENTS (MAX_COMPONENTS − 1)/2 directions. The health of chains,
    their reduction and the railing rows are not modelled; they cost a
    fraction of a second.
Chains
    A run of chains enters the estimate before its reduction, with the upper
    bound N = (steps − burn-in) · walkers and an effective sample size of N,
    so the estimate overstates the cost of the thinned samples that the
    engine reads.
Data space
    A pair of ``compare_signals`` costs one distance matrix of the pooled
    whitened templates and 1 + ``draws`` evaluations of the energy distance from
    it (and ``half_samples`` more with the bootstrap), once for the data row and
    once for the Fisher row where a Fisher matrix is available. The fit
    ``signal_matrix`` gives the seconds of one matrix, with n the larger count
    of the pair and the dimension M of the templates or D of the chart, and
    ``signal_draw`` the seconds of one evaluation for one pair of sides, with no
    dimension factor, timed in batches of 200 as the engine runs them. The
    generator and the noise model are user code, so their rows give the number
    of generator calls and the memory of the templates, 8·m·M bytes, with no
    seconds; the matrix of a pair needs 8·(m_p + m_q)² bytes.
Table
    ``engine``, ``problem``, ``pair`` (the names of the posteriors), ``metric``,
    ``estimator``, ``samples`` (the samples of the null draws, after the cap),
    ``dimension`` (D, or M in the data rows of data space), ``evaluations``
    (observed, null and half-sample evaluations of the group), ``seconds``,
    ``space`` and ``memory`` (bytes). The last two are null in the rows of
    ``compare`` and ``diagnose``, ``seconds`` is null in the generator rows, and
    the total sums the seconds that are not null.

References:
    :cite:t:`phipson2010permutation`; :cite:t:`politis1999subsampling`.
"""

import math
import os
import warnings
from collections.abc import Sequence
from typing import Final, NamedTuple

import jax.numpy as jnp
import polars as pl
from jax.typing import ArrayLike

from mixed_signals import errors
from mixed_signals._config import COST_BUDGET, NULL_DRAWS, SAMPLE_CAP
from mixed_signals.core.chains import Chains
from mixed_signals.core.result import UncertaintyMethod
from mixed_signals.core.samples import SampleSet
from mixed_signals.core.space import ParamSpace
from mixed_signals.dataspace.chart import CHART_DESCRIPTION
from mixed_signals.dataspace.templates import whitened_templates
from mixed_signals.diagnostics import flags
from mixed_signals.diagnostics.dependence import PAIR_CAP
from mixed_signals.diagnostics.modes import MAX_COMPONENTS
from mixed_signals.engines._arguments import (
    FAMILIES,
    RECIPES,
    CompareArguments,
    DiagnoseArguments,
    DiagnoseSource,
    SignalArguments,
    comparison_pairs,
    read_compare_arguments,
    read_diagnose_arguments,
    read_signal_arguments,
)
from mixed_signals.engines._common import Posterior, data_space_text, space_text
from mixed_signals.errors import CostWarning
from mixed_signals.generators.protocol import NoiseModel, SignalGenerator
from mixed_signals.io.arrays import ArrayBackend
from mixed_signals.metrics._embedding import embedding_size
from mixed_signals.metrics.ot.sinkhorn import SINKHORN_EPSILON_FACTOR
from mixed_signals.metrics.ot.sliced import (
    SLICED_SINKHORN_CAP,
    SLICED_SINKHORN_DIRECTIONS,
)
from mixed_signals.nulls.plan import Construction
from mixed_signals.nulls.surrogates import surrogate_size
from mixed_signals.sigma.calibration import KNN_KL_REPLICATES, KNN_KL_SHIFTS
from mixed_signals.sigma.thresholds import SIGNIFICANCE_LEVEL, THRESHOLDS
from mixed_signals.sigma.uncertainty import HALF_SAMPLE_PAIRS
from mixed_signals.transforms.base import Transform
from mixed_signals.transforms.whiten import WhiteningSource

__all__ = [
    "COST_BUDGET",
    "CostEstimate",
    "estimate_compare_cost",
    "estimate_compare_signals_cost",
    "estimate_diagnose_cost",
]

# * Frames in the package are skipped, so a warning points at the caller's line.
_PACKAGE_DIRECTORY: Final = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_REFERENCE_SAMPLES: Final = 10_000
"""Samples per set at which the constants of the model are scaled."""

_REFERENCE_DIMENSION: Final = 16
"""Number of parameters at which the constants of the model are scaled."""

_TABLE_SCHEMA: Final = pl.Schema(
    {
        "engine": pl.String(),
        "problem": pl.Int64(),
        "pair": pl.List(pl.String()),
        "metric": pl.String(),
        "estimator": pl.String(),
        "samples": pl.Int64(),
        "dimension": pl.Int64(),
        "evaluations": pl.Int64(),
        "seconds": pl.Float64(),
        "space": pl.String(),
        "memory": pl.Int64(),
    }
)
"""Columns and types of the table of an estimate.

The columns ``space`` and ``memory`` are null in the rows of ``compare`` and
``diagnose``; ``seconds`` is null in the rows of user code.
"""

_PROJECTION_DIRECTIONS: Final = MAX_COMPONENTS * (MAX_COMPONENTS - 1) // 2
"""Upper bound of the directions of the mode projections: the pairs of components."""


class CostEstimate(NamedTuple):
    """The estimated evaluations and run time of an engine call."""

    table: pl.DataFrame
    """One row per statistic of every pair or posterior, with the columns engine,
    problem, pair, metric, estimator, samples, dimension, evaluations, seconds,
    space and memory."""
    seconds: float
    """The estimated run time on one CPU of the developers' machine, in seconds.

    It sums the seconds of the rows that have them: the user code of data space has
    none."""


class _Fit(NamedTuple):
    """The constants of the model of one statistic."""

    coefficient: float
    """The cost c in seconds of one evaluation at 10 000 samples and 16 parameters."""
    exponent: float
    """The exponent a of the samples."""
    dimension_offset: float | None
    """The offset d₀ of the dimension factor; ``None`` for no dimension factor."""
    cap: str | None
    """The cap on the samples: ``cap``, ``pair_cap`` or ``sliced_sinkhorn_cap``."""
    misfit: float
    """The largest factor between the model and a timing of the grid of the fit."""


_FITS: Final = {
    "moments": _Fit(4.5e-5, 0.62, None, None, 2.87),
    "marginal": _Fit(0.0022, 1.13, None, None, 1.18),
    "kde": _Fit(0.031, 1.01, None, None, 1.27),
    "gaussian": _Fit(4.9e-4, 0.58, 9.4, None, 1.13),
    "knn_kl": _Fit(2.54, 2.09, 43.0, "cap", 1.05),
    "knn_js": _Fit(8.9, 2.06, 40.0, "cap", 1.09),
    "knn_hellinger": _Fit(5.18, 2.10, 42.0, "cap", 1.05),
    "mmd": _Fit(0.27, 1.94, 10.5, "cap", 1.15),
    "energy": _Fit(0.26, 1.95, 7.0, "cap", 1.13),
    "sinkhorn": _Fit(6.7, 2.19, None, "cap", 1.36),
    "sliced_w2": _Fit(0.57, 1.10, None, None, 1.02),
    "sliced_w2_gaussian": _Fit(5.2e-4, 0.65, 7.4, None, 1.07),
    "sliced_sinkhorn": _Fit(5.34, 1.77, None, "sliced_sinkhorn_cap", 1.08),
    "median_bandwidth": _Fit(0.30, 0.0, None, None, 1.00),
    "sliced_sinkhorn_epsilon": _Fit(0.08, 0.0, None, None, 1.03),
    "marginal_moments": _Fit(3.9e-5, 0.72, None, None, 2.14),
    "anderson_darling": _Fit(5.5e-4, 0.70, None, None, 2.11),
    "mardia": _Fit(0.019, 1.62, 45.0, "cap", 1.17),
    "dependence": _Fit(3.6, 2.63, None, "pair_cap", 1.31),
    "marginal_modes": _Fit(0.0153, 0.98, None, None, 1.24),
    "projection_modes": _Fit(0.0159, 0.96, 27.0, None, 1.34),
    "gmm_bic": _Fit(0.69, 0.32, 0.0, None, 1.55),
    "signal_matrix": _Fit(0.146, 2.10, 56.0, None, 1.32),
    "signal_draw": _Fit(3.86e-3, 2.00, None, None, 1.03),
}
"""The constants of the model of every statistic.

The first 22 were fitted on 2026-11-19, the two of data space on 2026-11-20."""


class _Group(NamedTuple):
    """One statistic of one pair or posterior, with what its cost depends on."""

    metric: str
    estimator: str
    statistic: str
    """The key of :data:`_FITS`."""
    units: float
    """The work units u of one evaluation."""
    runs: int
    """Repeats of every evaluation: the directions of a KL pair, the ε factors."""
    observed_samples: int
    """The samples of the observed evaluation, before the cap."""
    draw_samples: int
    """The samples of a null draw, before the cap."""
    draws: int
    """The number of null draws; zero for a statistic that runs once."""
    halves: int = 0
    """The number of half-sample evaluations."""


def estimate_compare_cost(
    sample_sets: SampleSet | Sequence[SampleSet],
    names: Sequence[str],
    space: ParamSpace,
    *,
    reference: str | None = None,
    whitening: WhiteningSource | str | None = None,
    transform: Transform | None = None,
    metrics: Sequence[str] | None = None,
    null: Construction | str = Construction.PERMUTATION,
    draws: int = NULL_DRAWS,
    cap: int = SAMPLE_CAP,
    sinkhorn_epsilon_factors: Sequence[float] = (SINKHORN_EPSILON_FACTOR,),
    uncertainty: UncertaintyMethod | str = UncertaintyMethod.NULL_STD,
    half_samples: int = HALF_SAMPLE_PAIRS,
) -> CostEstimate:
    """Estimate the run time of :func:`~mixed_signals.engines.compare.compare`.

    The arguments are those of ``compare`` that change the cost, read and
    checked by the same code. The estimate computes no metric; the module
    docstring gives the model, the constants and their limits. A Fisher
    whitening costs what a pooled one does, so it needs no matrices here.

    Args:
        sample_sets: A :class:`~mixed_signals.core.samples.SampleSet`, or a
            sequence of them, as ``compare`` reads them.
        names: One unique name per posterior.
        space: The parameter space of the samples.
        reference: The name of the reference posterior, for mode A.
        whitening: ``"reference"``, ``"pooled"`` or ``"fisher"``.
        transform: A transform of the coordinates; it changes no count.
        metrics: Metric names and recipes; the default selection when
            ``None``.
        null: ``"permutation"`` or ``"split_half"``.
        draws: Number of null draws per row.
        cap: Largest number of samples of a statistic with a cap.
        sinkhorn_epsilon_factors: The ε factors of the Sinkhorn divergence.
        uncertainty: ``"null_std"`` or ``"subsampling_bootstrap"``.
        half_samples: Number of half-sample pairs of the bootstrap.

    Returns:
        The table of groups and the total in seconds.

    Raises:
        ValueError: As :func:`~mixed_signals.engines.compare.compare`, for the
            arguments that the estimate takes.
        TypeError: As ``compare``.
        KeyError: As ``compare``.
        NotImplementedError: If the metrics hold the classifier test.

    References:
        :cite:t:`politis1999subsampling`.
    """
    reads_fisher = whitening == WhiteningSource.FISHER
    arguments = read_compare_arguments(
        sample_sets,
        names,
        space,
        key=0,
        reference=reference,
        whitening="pooled" if reads_fisher else whitening,
        fisher=None,
        centre=None,
        transform=transform,
        metrics=metrics,
        null=null,
        draws=draws,
        cap=cap,
        sinkhorn_epsilon_factors=sinkhorn_epsilon_factors,
        uncertainty=uncertainty,
        half_samples=half_samples,
        thresholds=THRESHOLDS,
        level=SIGNIFICANCE_LEVEL,
        budget=COST_BUDGET,
        config_hash=None,
    )
    return _compare_cost(arguments)


def estimate_diagnose_cost(
    sources: Chains | SampleSet | Sequence[Chains | SampleSet],
    names: Sequence[str],
    space: ParamSpace,
    *,
    burn_in: int | Sequence[int] = 0,
    transform: Transform | None = None,
    diagnostics: Sequence[str] | None = None,
    draws: int = NULL_DRAWS,
    cap: int = SAMPLE_CAP,
    pair_cap: int = PAIR_CAP,
) -> CostEstimate:
    """Estimate the run time of :func:`~mixed_signals.engines.diagnose.diagnose`.

    The arguments are those of ``diagnose`` that change the cost, read and
    checked by the same code. A run of chains enters with N = (steps −
    burn-in) · walkers samples and an effective sample size of N, before its
    reduction, so the estimate overstates the cost of the thinned samples.

    Args:
        sources: Chains, a :class:`~mixed_signals.core.samples.SampleSet` or a
            sequence of them, as ``diagnose`` reads them.
        names: One unique name per posterior.
        space: The parameter space.
        burn_in: Steps dropped from the start of the chains.
        transform: A transform of the coordinates; it changes no count.
        diagnostics: The families to run; all when ``None``.
        draws: Number of surrogate sets per posterior.
        cap: Largest number of samples of a statistic with a cap.
        pair_cap: Largest number of samples of the dependence rows.

    Returns:
        The table of groups and the total in seconds.

    Raises:
        ValueError: As :func:`~mixed_signals.engines.diagnose.diagnose`, for
            the arguments that the estimate takes.
        TypeError: As ``diagnose``.

    References:
        :cite:t:`efron1993introduction`.
    """
    arguments = read_diagnose_arguments(
        sources,
        names,
        space,
        key=0,
        burn_in=burn_in,
        transform=transform,
        diagnostics=diagnostics,
        draws=draws,
        cap=cap,
        pair_cap=pair_cap,
        thresholds=flags.THRESHOLDS,
        level=SIGNIFICANCE_LEVEL,
        budget=COST_BUDGET,
        config_hash=None,
    )
    return _diagnose_cost(arguments)


def estimate_compare_signals_cost(
    sample_sets: SampleSet | Sequence[SampleSet],
    names: Sequence[str],
    space: ParamSpace,
    *,
    generator: SignalGenerator | Sequence[SignalGenerator],
    noise: NoiseModel | Sequence[NoiseModel],
    reference: str | None = None,
    fisher: ArrayLike | None = None,
    centre: ArrayLike | None = None,
    transform: Transform | None = None,
    null: Construction | str = Construction.PERMUTATION,
    draws: int = NULL_DRAWS,
    cap: int = SAMPLE_CAP,
    uncertainty: UncertaintyMethod | str = UncertaintyMethod.NULL_STD,
    half_samples: int = HALF_SAMPLE_PAIRS,
    backend: ArrayBackend = "numpy",
    chunk_size: int | None = None,
) -> CostEstimate:
    """Estimate the run time of ``compare_signals``, the data-space comparison.

    The arguments are those of
    :func:`~mixed_signals.engines.dataspace.compare_signals` that change the
    cost, read and checked by the same code. The estimate probes the generator and the
    noise model of each problem on one sample, the first sample with a positive
    weight of the first posterior after the transform, to learn the number M of
    data entries, and computes nothing else. The module docstring gives the
    model; the rows of user code have no seconds.

    Args:
        sample_sets: A :class:`~mixed_signals.core.samples.SampleSet`, or a
            sequence of them, as ``compare_signals`` reads them.
        names: One unique name per posterior.
        space: The parameter space of the samples.
        generator: A signal generator, or one per problem.
        noise: A noise model, or one per problem.
        reference: The name of the reference posterior, for mode A.
        fisher: Fisher matrices, which switch on the Fisher rows with
            ``centre``; only their presence changes the estimate.
        centre: The points of the Fisher matrices, or of the matrices that the
            generators give.
        transform: A transform of the coordinates; it changes no count.
        null: ``"permutation"`` or ``"split_half"``.
        draws: Number of null draws per row.
        cap: Largest number of samples of a posterior of a problem.
        uncertainty: ``"null_std"`` or ``"subsampling_bootstrap"``.
        half_samples: Number of half-sample pairs of the bootstrap.
        backend: The array library of the arrays that the generator receives.
        chunk_size: The most samples in one call of the generator.

    Returns:
        The table of groups and the total in seconds.

    Raises:
        ValueError: As :func:`~mixed_signals.engines.dataspace.compare_signals`,
            for the arguments that the estimate takes, and if the templates of
            a problem or the probe are not finite real vectors.
        TypeError: As ``compare_signals``.
        KeyError: As ``compare_signals``.
        CapabilityError: As ``compare_signals``.

    References:
        :cite:t:`szekely2013energy`.
    """
    arguments = read_signal_arguments(
        sample_sets,
        names,
        space,
        generator=generator,
        noise=noise,
        key=0,
        reference=reference,
        fisher=fisher,
        centre=centre,
        transform=transform,
        null=null,
        draws=draws,
        cap=cap,
        uncertainty=uncertainty,
        half_samples=half_samples,
        thresholds=THRESHOLDS,
        level=SIGNIFICANCE_LEVEL,
        budget=COST_BUDGET,
        backend=backend,
        chunk_size=chunk_size,
        config_hash=None,
    )
    return _compare_signals_cost(arguments)


def warn_if_over_budget(engine: str, estimate: CostEstimate, budget: float) -> None:
    """Raise one cost warning if the estimate exceeds the budget.

    The warning names the estimate, the budget and the three costliest
    statistics, with their seconds summed over problems and pairs. It names
    only statistics with an estimate: the rows of user code, whose seconds are
    null, are left out. It goes to the caller, not into the rows of a result.

    Args:
        engine: Name of the engine, for the message.
        estimate: The estimate of the engine call.
        budget: The budget in seconds; infinity never warns.

    Warns:
        CostWarning: If the estimated seconds exceed the budget.
    """
    if not estimate.seconds > budget:
        return
    costliest = (
        estimate.table.filter(pl.col("seconds").is_not_null())
        .group_by(["metric", "estimator"], maintain_order=True)
        .agg(pl.col("seconds").sum())
        .sort("seconds", descending=True, maintain_order=True)
        .head(3)
    )
    groups = ", ".join(
        f"{metric} ({estimator}) {_format_seconds(float(seconds))}"
        for metric, estimator, seconds in costliest.iter_rows()
    )
    warnings.warn(
        CostWarning(
            f"the {engine} run is estimated at {_format_seconds(estimate.seconds)}"
            " on one CPU of the developers' machine, above the budget of"
            f" {_format_seconds(budget)}; the costliest statistics are {groups}"
        ),
        skip_file_prefixes=(_PACKAGE_DIRECTORY,),
    )


def _format_seconds(seconds: float) -> str:
    """Write seconds from a Python float, as whole numbers above 10 s."""
    if seconds >= 10:
        return f"{seconds:,.0f} s"
    return f"{seconds:.3g} s"


# * The model


def _evaluation_seconds(
    fit: _Fit, samples: int, dimension: int, units: float, caps: dict[str, int]
) -> float:
    """Return the seconds of one evaluation: c (n/10 000)^a · u · (D + d₀)/(16 + d₀)."""
    kept = float(samples if fit.cap is None else min(samples, caps[fit.cap]))
    scale = math.pow(max(kept, 1.0) / _REFERENCE_SAMPLES, fit.exponent)
    factor = 1.0
    if fit.dimension_offset is not None:
        factor = (dimension + fit.dimension_offset) / (
            _REFERENCE_DIMENSION + fit.dimension_offset
        )
    return fit.coefficient * scale * units * factor


def _group_row(
    engine: str,
    problem: int,
    pair: list[str],
    group: _Group,
    dimension: int,
    caps: dict[str, int],
) -> dict[str, object]:
    """Return the table row of a group: its evaluations and their seconds."""
    fit = _FITS[group.statistic]
    half_samples = math.ceil(group.draw_samples / 2)
    per_run = (
        _evaluation_seconds(fit, group.observed_samples, dimension, group.units, caps)
        + group.draws
        * _evaluation_seconds(fit, group.draw_samples, dimension, group.units, caps)
        + group.halves
        * _evaluation_seconds(fit, half_samples, dimension, group.units, caps)
    )
    drawn = (
        group.draw_samples
        if fit.cap is None
        else min(group.draw_samples, caps[fit.cap])
    )
    return {
        "engine": engine,
        "problem": problem,
        "pair": pair,
        "metric": group.metric,
        "estimator": group.estimator,
        "samples": drawn,
        "dimension": dimension,
        "evaluations": group.runs * (1 + group.draws + group.halves),
        "seconds": group.runs * per_run,
    }


def _estimate(rows: list[dict[str, object]]) -> CostEstimate:
    """Return the estimate of a list of rows: the table and its total."""
    table = pl.DataFrame(rows, schema=_TABLE_SCHEMA)
    return CostEstimate(table=table, seconds=float(table["seconds"].sum()))


# * The cost of a comparison


def _positive_count(posterior: Posterior, problem: int) -> int:
    """Return the number of samples with a positive weight."""
    return int(jnp.sum(posterior.log_weights[problem] > -jnp.inf))


def _compare_cost(arguments: CompareArguments) -> CostEstimate:
    """Estimate the run time of a comparison from its read arguments."""
    dimension = len(arguments.space)
    embedding = embedding_size(arguments.space)
    caps = {"cap": arguments.cap, "pair_cap": arguments.cap}
    caps["sliced_sinkhorn_cap"] = SLICED_SINKHORN_CAP
    selected = {
        family: tuple(recipe for recipe in members if recipe in arguments.recipes)
        for family, members in FAMILIES.items()
    }
    families = [family for family, recipes in selected.items() if recipes]
    batch_count = arguments.posteriors[0].samples.shape[0]
    rows: list[dict[str, object]] = []
    for problem in range(batch_count):
        for first_index, second_index in comparison_pairs(
            len(arguments.posteriors), arguments.reference
        ):
            first = arguments.posteriors[first_index]
            second = arguments.posteriors[second_index]
            groups = _compare_groups(
                arguments, selected, families, first, second, problem, embedding
            )
            pair = [first.name, second.name]
            rows.extend(
                _group_row("compare", problem, pair, group, dimension, caps)
                for group in groups
            )
    return _estimate(rows)


# * Per family of compare: the statistic of the model, and what its units count
_COMPARE_STATISTICS: Final = {
    "mean_shift": ("moments", "dimension"),
    "width_ratio": ("moments", "dimension"),
    "marginal": ("marginal", "dimension"),
    "kde": ("kde", "dimension"),
    "gaussian": ("gaussian", "one"),
    "mmd": ("mmd", "one"),
    "energy": ("energy", "one"),
    "sinkhorn": ("sinkhorn", "one"),
    "sliced_w2": ("sliced_w2", "one"),
    "sliced_w2_gaussian": ("sliced_w2_gaussian", "one"),
    "sliced_sinkhorn": ("sliced_sinkhorn", "directions"),
}


def _compare_groups(
    arguments: CompareArguments,
    selected: dict[str, tuple[str, ...]],
    families: list[str],
    first: Posterior,
    second: Posterior,
    problem: int,
    embedding: int,
) -> list[_Group]:
    """Return the statistics of one comparison with the sizes they run on."""
    samples = max(first.samples.shape[1], second.samples.shape[1])
    bootstrap = arguments.uncertainty is UncertaintyMethod.SUBSAMPLING_BOOTSTRAP
    halves = arguments.half_samples if bootstrap else 0
    units = {
        "dimension": float(len(arguments.space)),
        "one": 1.0,
        "directions": float(
            math.ceil(SLICED_SINKHORN_DIRECTIONS / embedding) * embedding
        ),
    }
    groups: list[_Group] = []
    for family in families:
        recipes = selected[family]
        if family == "knn":
            continue
        statistic, unit_kind = _COMPARE_STATISTICS[family]
        runs = len(arguments.factors) if family == "sinkhorn" else 1
        groups.append(
            _Group(
                "/".join(dict.fromkeys(RECIPES[recipe].metric for recipe in recipes)),
                RECIPES[recipes[0]].estimator,
                statistic,
                units[unit_kind],
                runs,
                samples,
                samples,
                arguments.draws,
                halves,
            )
        )
    if {"mmd", "sinkhorn"} & set(families):
        groups.append(
            _Group(
                "median_bandwidth",
                "empirical",
                "median_bandwidth",
                1.0,
                1,
                samples,
                samples,
                0,
            )
        )
    if "sliced_sinkhorn" in families:
        groups.append(
            _Group(
                "sliced_sinkhorn_epsilon",
                "empirical",
                "sliced_sinkhorn_epsilon",
                units["directions"],
                1,
                samples,
                samples,
                0,
            )
        )
    if "knn" in families:
        positive = min(
            _positive_count(first, problem), _positive_count(second, problem)
        )
        groups.extend(_knn_groups(selected["knn"], positive, arguments.draws, halves))
    return groups


def _knn_groups(
    recipes: tuple[str, ...], samples: int, draws: int, halves: int
) -> list[_Group]:
    """Return the kNN statistics: two KL directions, JS, Hellinger and the KL curves."""
    groups = []
    for recipe in recipes:
        metric = RECIPES[recipe].metric
        runs = 2 if metric == "kl" else 1
        groups.append(
            _Group(
                metric,
                "knn",
                f"knn_{metric}",
                1.0,
                runs,
                samples,
                samples,
                draws,
                halves,
            )
        )
        if metric == "kl":
            curve = (len(KNN_KL_SHIFTS) + 1) * KNN_KL_REPLICATES
            groups.append(
                _Group(
                    "kl_calibration",
                    "knn",
                    "knn_kl",
                    1.0,
                    2,
                    samples,
                    samples,
                    curve - 1,
                )
            )
    return groups


# * The cost of a data-space comparison


def _compare_signals_cost(arguments: SignalArguments) -> CostEstimate:
    """Estimate the run time of a data-space comparison from its read arguments."""
    rows: list[dict[str, object]] = []
    prefix = [] if arguments.transform_text is None else [arguments.transform_text]
    evaluations = 1 + arguments.draws
    if arguments.uncertainty is UncertaintyMethod.SUBSAMPLING_BOOTSTRAP:
        evaluations += arguments.half_samples
    batch_count = arguments.posteriors[0].samples.shape[0]
    for problem in range(batch_count):
        generator, noise = arguments.generators[problem], arguments.noises[problem]
        data_size = _probe_data_size(arguments, problem, generator, noise)
        counts = [
            min(_positive_count(posterior, problem), arguments.cap)
            for posterior in arguments.posteriors
        ]
        for posterior, count in zip(arguments.posteriors, counts, strict=True):
            calls = (
                1
                if arguments.chunk_size is None
                else math.ceil(count / arguments.chunk_size)
            )
            rows.append(
                {
                    "engine": "compare_signals",
                    "problem": problem,
                    "pair": [posterior.name],
                    "metric": "generate_template",
                    "estimator": "user",
                    "samples": count,
                    "dimension": data_size,
                    "evaluations": calls,
                    "seconds": None,
                    "space": None,
                    "memory": 8 * count * data_size,
                }
            )
        for first_index, second_index in comparison_pairs(
            len(arguments.posteriors), arguments.reference
        ):
            pair = [
                arguments.posteriors[first_index].name,
                arguments.posteriors[second_index].name,
            ]
            sizes = (counts[first_index], counts[second_index])
            spaces = [(space_text(*prefix, data_space_text(noise)), data_size)]
            if arguments.centre is not None:
                spaces.append(
                    (space_text(*prefix, CHART_DESCRIPTION), len(arguments.space))
                )
            rows.extend(
                _signal_row(problem, pair, sizes, evaluations, text, dimension)
                for text, dimension in spaces
            )
    return _estimate(rows)


def _probe_data_size(
    arguments: SignalArguments,
    problem: int,
    generator: SignalGenerator,
    noise: NoiseModel,
) -> int:
    """Return the number of data entries M, from one template of the first posterior."""
    first = arguments.posteriors[0]
    start = int(jnp.argmax(first.log_weights[problem] > -jnp.inf))
    # * The estimate needs only M; the engine records the warnings of the templates.
    with errors.capture_warnings():
        probe = whitened_templates(
            generator,
            noise,
            first.samples[problem][start : start + 1],
            backend=arguments.backend,
            chunk_size=arguments.chunk_size,
        )
    return int(probe.shape[1])


def _signal_row(
    problem: int,
    pair: list[str],
    sizes: tuple[int, int],
    evaluations: int,
    space: str,
    dimension: int,
) -> dict[str, object]:
    """Return the row of the energy distances of one pair in one space of the data."""
    larger = max(sizes)
    matrix = _evaluation_seconds(_FITS["signal_matrix"], larger, dimension, 1.0, {})
    draw = _evaluation_seconds(_FITS["signal_draw"], larger, dimension, 1.0, {})
    return {
        "engine": "compare_signals",
        "problem": problem,
        "pair": pair,
        "metric": "energy",
        "estimator": "empirical",
        "samples": larger,
        "dimension": dimension,
        "evaluations": evaluations,
        "seconds": matrix + evaluations * draw,
        "space": space,
        "memory": 8 * sum(sizes) ** 2,
    }


# * The cost of diagnostics


def _diagnose_cost(arguments: DiagnoseArguments) -> CostEstimate:
    """Estimate the run time of the diagnostics from their read arguments."""
    dimension = len(arguments.row_space)
    caps = {"cap": arguments.cap, "pair_cap": arguments.pair_cap}
    caps["sliced_sinkhorn_cap"] = SLICED_SINKHORN_CAP
    rows: list[dict[str, object]] = []
    for source in arguments.sources:
        for position, name in enumerate(source.names):
            for problem in range(arguments.batch_count):
                samples, n_eff = _diagnosed_size(source, position, problem)
                groups = _diagnose_groups(arguments, dimension, samples, n_eff)
                rows.extend(
                    _group_row("diagnose", problem, [name], group, dimension, caps)
                    for group in groups
                )
    return _estimate(rows)


def _diagnosed_size(
    source: DiagnoseSource, position: int, problem: int
) -> tuple[int, float]:
    """Return the samples and the effective sample size that a posterior enters with."""
    if source.chains is not None:
        steps = source.chains.shape.step - source.burn_in
        count = max(steps, 1) * source.chains.shape.walker
        return count, float(count)
    assert source.sample_set is not None
    n_eff = float(source.sample_set.effective_sample_size()[problem, position])
    return source.sample_set.shape.sample, n_eff


def _diagnose_groups(
    arguments: DiagnoseArguments, dimension: int, samples: int, n_eff: float
) -> list[_Group]:
    """Return the statistics of the selected families for one posterior."""
    families = arguments.families
    shape_size = surrogate_size(n_eff, arguments.cap)
    groups: list[_Group] = []
    if "shape" in families:
        for statistic, units, observed in (
            ("marginal_moments", dimension, samples),
            ("anderson_darling", dimension, samples),
            ("mardia", 1, min(samples, arguments.cap)),
        ):
            groups.append(
                _Group(
                    statistic,
                    "empirical",
                    statistic,
                    float(units),
                    1,
                    observed,
                    shape_size,
                    arguments.draws,
                )
            )
    if "dependence" in families and dimension >= 2:
        pair_size = surrogate_size(n_eff, arguments.pair_cap)
        pairs = dimension * (dimension - 1) // 2
        groups.append(
            _Group(
                "dependence",
                "empirical",
                "dependence",
                float(pairs),
                1,
                min(samples, arguments.pair_cap),
                pair_size,
                arguments.draws,
            )
        )
    if "modes" in families:
        groups.append(_Group("gmm_bic", "gmm", "gmm_bic", 1.0, 1, samples, samples, 0))
        groups.append(
            _Group(
                "marginal_modes",
                "kde",
                "marginal_modes",
                float(dimension),
                1,
                samples,
                shape_size,
                arguments.draws,
            )
        )
        groups.append(
            _Group(
                "projection_modes",
                "kde",
                "projection_modes",
                float(_PROJECTION_DIRECTIONS),
                1,
                samples,
                shape_size,
                arguments.draws,
            )
        )
    return groups
