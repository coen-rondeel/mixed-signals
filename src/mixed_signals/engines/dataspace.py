"""``compare_signals``: posteriors compared through the signals that they imply.

:func:`compare_signals` compares pairs of posteriors, from one
:class:`~mixed_signals.core.samples.SampleSet` or from several that share their
problems and parameters, in data space: it calls a signal generator on the
samples, whitens the templates by a noise model and compares the two clouds of
whitened templates with the energy distance. Distances come in units of the
noise standard deviation, and ‖δh‖ ≲ 1 is indistinguishable in the data
(Lindblom, Owen and Brown 2008). It is a plain function that returns one
:class:`~mixed_signals.core.result.Result` table, with a finite-sample null,
σ_sig and σ_eff in every row, as :func:`~mixed_signals.engines.compare.compare`
does; the package warnings raised in a row go into its ``warnings`` column.

Inputs
    The modes, the null, the transform and the thresholds are those of
    ``compare``: ``reference=<name>`` pairs the reference with every other
    posterior, otherwise every pair (names[i], names[j]) with i < j runs;
    ``null="split_half"`` needs the reference. One generator and one noise
    model serve every problem, or a sequence of B of each gives one per
    problem, since templates and noise depend on the signal model and the
    observation. The thresholds must hold ``"energy"``.
Templates
    A generator implements ``generate_template(params)``, a noise model
    ``whiten(signals)`` (:mod:`mixed_signals.generators.protocol`). They are
    user code, so they are called at the boundary, in chunks of ``chunk_size``
    samples and in the library named by ``backend``, outside ``jax.jit``
    (:func:`~mixed_signals.dataspace.templates.whitened_templates`); only the
    real whitened rows cross into JAX. The templates of one problem must share
    one data grid, so every posterior of a problem gives the same number M of
    data entries, or the call raises. A capability that the objects lack raises
    :class:`~mixed_signals.errors.CapabilityError` before any call.
Cap and effective size
    Zero-weight samples enter no estimate and are never sent to the generator.
    A posterior above ``cap`` positive samples keeps a uniform subset without
    replacement, drawn once per problem, and each of its pairs records the
    :class:`~mixed_signals.errors.SubsamplingWarning`. Every row of a pair uses
    the same capped samples, null plan and half-samples. The null sides of a
    capped set take the design-effect effective size of
    :func:`~mixed_signals.core.samples.subsample_effective_size`, which keeps the
    correlation of a chain between its samples (Kish 1965); a set without a cap
    keeps its own effective size. The zoo check ``capped-effective-size`` of
    ``scripts/zoo_validation/dataspace.py``, which joins the validation page at
    its next full build, found 4.69 to 5.34% rejections at the 5% level with
    this size on capped AR(1) chains, against 1.15 to 4.56% when the ratio
    m·n_eff/N was kept and 5.75 to 15.11% when the correlation was ignored.
Distances
    The energy distance (Székely and Rizzo 2013) of the pooled samples of a pair
    needs the distances between all pairs of pooled templates, one matrix of
    8P² bytes for P pooled samples
    (:func:`~mixed_signals.dataspace.distances.distance_matrix`). The observed
    value, every null draw and every half-sample are weighted sums over that
    matrix, evaluated as matrix products
    (:func:`~mixed_signals.metrics.kernel.energy_distance_from_distances`), so
    the result is exact and equals the energy distance of the templates, whose
    direct evaluation takes about a minute at the cap. In the Hilbert space of
    the noise-weighted inner product, the energy distance vanishes only for equal
    distributions, because a separable Hilbert space has strong negative type
    (Lyons 2013), and its root is a metric. A probe on the developers' machine
    with M = 4 000 data entries took 0.3, 1.9 and 8.1 s for the matrix (0.13,
    0.8 and 3.2 GB) at 2 000, 5 000 and 10 000 samples per posterior, and 0.03,
    0.2 and 0.75 s for 200 null draws. The matrix is released before the next
    one is built.
Nulls
    ``null="permutation"`` pools the two capped sets and cuts the pool into two
    random sides; ``null="split_half"``, in mode A, cuts the reference alone
    (Good 2005). The sides carry the effective sample sizes, every sample keeps
    its weight, and the p-value has the +1 correction (Phipson and Smyth 2010),
    with σ_sig a Gaussian significance (Cowan et al. 2011). The null plan only
    chooses positions in the pool, which the data row and the Fisher row share.
Fisher row
    With a Fisher matrix, each pair has a second row: the energy distance of
    the parameters in the joint local chart of
    :class:`~mixed_signals.dataspace.chart.FisherChart`, y = W·arc(θ − c) with
    WᵀW = Γ. At linear order its distances are ΔθᵀΓΔθ, the squared data-space
    distance (Vallisneri 2008), so the two rows agree where the signal is
    linear in the parameters. The chart is joint, not the block form of
    :class:`~mixed_signals.transforms.whiten.Whitening`, which drops the
    correlations between circles and other axes; W is the ZCA-cor whitening of
    Γ⁻¹ over all axes (Kessy, Lewin and Strimmer 2018). The matrix comes from
    ``fisher=`` with ``centre=``, in the coordinates of the input as for
    ``compare``, or from ``centre=`` alone through the ``information_matrix`` of
    the generator; with neither, there is no Fisher row. A matrix that is not
    finite, symmetric and positive definite gives a
    :class:`~mixed_signals.errors.ValidityWarning` and NaN Fisher rows for its
    problem, while the data rows and the other problems go on; a wrong shape
    raises.
Effect sizes
    σ_eff of both rows is the energy-distance curve for D parameters of
    :mod:`mixed_signals.sigma.maps`, applied to the null-subtracted value, so
    the data row reads in units of the noise standard deviation. It is valid at
    linear order: where the template manifold curves, as for a sinusoid of low
    signal-to-noise ratio, the data row reads below the chart row, and the
    check ``dataspace-linear-order`` of ``scripts/zoo_validation/dataspace.py``,
    which joins the validation page at its next full build, measures how the
    ratio of the two rises towards one with the signal-to-noise ratio. The
    thresholds act on σ_eff, and ``flag`` says whether it reaches the threshold
    while σ_sig reaches ``level``.
Uncertainty
    ``uncertainty="null_std"`` reports the standard deviation of the null draws.
    ``"subsampling_bootstrap"`` reports the half-sample error of
    :func:`~mixed_signals.sigma.uncertainty.half_sample_error`, from
    ``half_samples`` pairs of random half-samples evaluated on the same matrix
    (Politis, Romano and Wolf 1999; Shao and Wu 1989).
Keys
    A key or an integer seed seeds the call. Every random choice derives its key
    with :func:`~mixed_signals.prng.derive_key` from a path that starts with
    ``("compare_signals", problem, ...)`` and names the posteriors involved, so
    a row does not depend on the other rows of the call.
Dry run
    The call first runs the dry run
    :func:`~mixed_signals.engines.cost.estimate_compare_signals_cost`, which
    probes each problem's generator and noise model on one sample to learn M,
    and raises one :class:`~mixed_signals.errors.CostWarning` above ``budget``.
Space
    The ``space`` column reads ``<transform> | data(<description>)`` and
    ``<transform> | chart(fisher)``; the description is the string attribute
    ``description`` of the noise model, ``data`` without it, and the transform
    part is left out without a transform.

References:
    :cite:t:`szekely2013energy`; :cite:t:`lyons2013distance`;
    :cite:t:`lindblom2008model`; :cite:t:`vallisneri2008use`;
    :cite:t:`finn1992detection`; :cite:t:`cutler1994gravitational`;
    :cite:t:`kish1965survey`; :cite:t:`good2005permutation`;
    :cite:t:`phipson2010permutation`; :cite:t:`cowan2011asymptotic`;
    :cite:t:`kessy2018optimal`; :cite:t:`politis1999subsampling`;
    :cite:t:`shao1989general`.
"""

import math
import os
import warnings
from collections.abc import Mapping, Sequence
from functools import partial
from typing import Any, Final, NamedTuple

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from jaxtyping import Array, Float, Int, PRNGKeyArray

from mixed_signals import errors
from mixed_signals._config import COST_BUDGET, NULL_DRAWS, SAMPLE_CAP
from mixed_signals.core.result import Provenance, Result, UncertaintyMethod
from mixed_signals.core.samples import (
    SampleSet,
    kish_effective_sample_size,
    subsample_effective_size,
)
from mixed_signals.core.space import ParamSpace
from mixed_signals.dataspace.chart import CHART_DESCRIPTION, FisherChart
from mixed_signals.dataspace.distances import distance_matrix
from mixed_signals.dataspace.templates import whitened_templates
from mixed_signals.engines._arguments import (
    SignalArguments,
    comparison_pairs,
    read_signal_arguments,
)
from mixed_signals.engines._common import (
    KeyInput,
    Posterior,
    data_space_text,
    null_moments,
    provenance,
    settings_text,
    space_text,
)
from mixed_signals.engines.cost import _compare_signals_cost, warn_if_over_budget
from mixed_signals.errors import ValidityWarning, WarningRecord
from mixed_signals.estimators.base import CHUNK_ENTRIES
from mixed_signals.generators.protocol import NoiseModel, SignalGenerator
from mixed_signals.io.arrays import ArrayBackend, call_in_chunks
from mixed_signals.metrics._cap import cap_samples
from mixed_signals.metrics.kernel import energy_distance_from_distances
from mixed_signals.nulls.permutation import permutation_null
from mixed_signals.nulls.plan import Construction, NullPlan
from mixed_signals.nulls.split_half import split_half_null
from mixed_signals.prng import derive_key
from mixed_signals.sigma.maps import null_subtract, sigma_eff
from mixed_signals.sigma.significance import null_significance
from mixed_signals.sigma.thresholds import (
    SIGNIFICANCE_LEVEL,
    THRESHOLDS,
    comparison_tail,
    verdict,
)
from mixed_signals.sigma.uncertainty import (
    HALF_SAMPLE_PAIRS,
    _half_positions,
    half_sample_error,
)
from mixed_signals.transforms.base import Transform

__all__ = ["compare_signals"]

# * Frames in the package are skipped, so a warning points at the caller's line.
_PACKAGE_DIRECTORY: Final = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_METRIC: Final = "energy"
_ESTIMATOR: Final = "empirical"
_NAN: Final = math.nan


class _Prepared(NamedTuple):
    """One posterior of one problem: capped samples, weights, size and templates."""

    name: str
    samples: Float[Array, "sample param"]
    log_weights: Float[Array, " sample"]
    """Log weights normalised to sum to one."""
    n_eff: float
    """The effective sample size that the sides of a null carry."""
    templates: Float[Array, "sample data"]
    records: list[WarningRecord]
    """The warnings of the cap and of the templates."""


class _ChartState(NamedTuple):
    """The Fisher chart of one problem, or why it is missing, with its warnings."""

    chart: FisherChart | None
    records: list[WarningRecord]
    settings: str
    """The ``estimator_settings`` text of the Fisher rows."""


class _Sides(NamedTuple):
    """Log weights of the two sides over the pool, one row per evaluation."""

    first: Float[Array, "draw sample"]
    """The rows are the observed value, the null draws and the half-samples."""
    second: Float[Array, "draw sample"]
    draws: int
    half_samples: int


class _Values(NamedTuple):
    """The observed energy distance, its null draws and its half-sample values."""

    observed: float
    draws: Float[Array, " draw"]
    half_values: Float[Array, " draw"] | None


class _PairInfo(NamedTuple):
    """What the rows of one pair share."""

    problem: int
    names: tuple[str, str]
    provenance: Provenance


def compare_signals(
    sample_sets: SampleSet | Sequence[SampleSet],
    names: Sequence[str],
    space: ParamSpace,
    *,
    generator: SignalGenerator | Sequence[SignalGenerator],
    noise: NoiseModel | Sequence[NoiseModel],
    key: KeyInput,
    reference: str | None = None,
    fisher: ArrayLike | None = None,
    centre: ArrayLike | None = None,
    transform: Transform | None = None,
    null: Construction | str = Construction.PERMUTATION,
    draws: int = NULL_DRAWS,
    cap: int = SAMPLE_CAP,
    uncertainty: UncertaintyMethod | str = UncertaintyMethod.NULL_STD,
    half_samples: int = HALF_SAMPLE_PAIRS,
    thresholds: Mapping[str, float] = THRESHOLDS,
    level: float = SIGNIFICANCE_LEVEL,
    budget: float = COST_BUDGET,
    backend: ArrayBackend = "numpy",
    chunk_size: int | None = None,
    config_hash: str | None = None,
) -> Result:
    """Compare posteriors pairwise in data space, with a null and σ_eff per row.

    The module docstring describes the inputs, the templates, the cap, the
    distances, the nulls, the Fisher row and the effect sizes. The package
    warnings that a row raises, such as a
    :class:`~mixed_signals.errors.SubsamplingWarning` or a
    :class:`~mixed_signals.errors.ValidityWarning` for an invalid Fisher matrix,
    are not raised but go into the ``warnings`` column of the row.

    Args:
        sample_sets: A :class:`~mixed_signals.core.samples.SampleSet` shaped
            (B, K, N, D), or a sequence of sets with the same B and D, whose
            posteriors join in order. The numbers of samples may differ.
        names: One unique name per posterior, in the order of the posteriors.
        space: The parameter space of the samples, with D parameters.
        generator: A signal generator with ``generate_template(params)``, for
            every problem, or a sequence of B generators, one per problem. It
            gets parameters in the coordinates after ``transform``.
        noise: A noise model with ``whiten(signals)``, for every problem, or a
            sequence of B noise models.
        key: A JAX PRNG key, or an integer seed that becomes
            :func:`~mixed_signals.prng.root_key` and fills the provenance seed.
        reference: The name of the reference posterior, for mode A; ``None``
            compares every pair, mode B.
        fisher: Fisher matrices for the Fisher row, shaped (B, D, D), or (D, D)
            when B = 1, in the coordinates of the input; needs ``centre``.
        centre: Points at which the Fisher matrices hold, shaped (B, D), or (D,)
            when B = 1, in the coordinates of the input. Without ``fisher``,
            the generators give the matrices with ``information_matrix``.
        transform: A transform that maps the samples to the coordinates of the
            generator first; ``None`` for the coordinates of the input.
        null: ``"permutation"``, or ``"split_half"`` in mode A.
        draws: Number of null draws per row.
        cap: Largest number of samples of a posterior of a problem.
        uncertainty: ``"null_std"`` or ``"subsampling_bootstrap"``.
        half_samples: Number of half-sample pairs of the subsampling bootstrap.
        thresholds: The threshold of each metric; it must hold ``"energy"``.
        level: The σ_sig that a row must reach to flag.
        budget: The estimated run time in seconds above which the dry run
            raises a :class:`~mixed_signals.errors.CostWarning`; infinity
            switches the warning off.
        backend: The array library of the arrays that the generator and the
            noise model receive: ``"numpy"``, ``"jax"``, ``"torch"`` or
            ``"cupy"``.
        chunk_size: The most samples in one call of the generator; all samples
            of a posterior in one call by default.
        config_hash: SHA-256 digest of the pipeline config, for the provenance.

    Returns:
        The result table: problems, then pairs, with the data row before the
        Fisher row of each pair.

    Raises:
        ValueError: If the sets, names, space, mode, null, counts, uncertainty
            method, thresholds or the Fisher input do not fit, a sequence of
            generators or noise models has the wrong length, the templates of
            a problem differ in their number of data entries or are not finite,
            or the generator returns a Fisher matrix of the wrong shape.
        TypeError: If a count is not an integer or an input is of another type
            than the arguments need.
        KeyError: If the thresholds lack ``"energy"``.
        CapabilityError: If a generator lacks ``generate_template``, a noise
            model lacks ``whiten``, or ``centre`` comes alone for a generator
            without ``information_matrix``.

    References:
        :cite:t:`szekely2013energy`; :cite:t:`lindblom2008model`;
        :cite:t:`vallisneri2008use`; :cite:t:`good2005permutation`;
        :cite:t:`phipson2010permutation`; :cite:t:`cowan2011asymptotic`.
    """
    arguments = read_signal_arguments(
        sample_sets,
        names,
        space,
        generator=generator,
        noise=noise,
        key=key,
        reference=reference,
        fisher=fisher,
        centre=centre,
        transform=transform,
        null=null,
        draws=draws,
        cap=cap,
        uncertainty=uncertainty,
        half_samples=half_samples,
        thresholds=thresholds,
        level=level,
        budget=budget,
        backend=backend,
        chunk_size=chunk_size,
        config_hash=config_hash,
    )
    warn_if_over_budget(
        "compare_signals", _compare_signals_cost(arguments), arguments.budget
    )
    pairs = comparison_pairs(len(arguments.posteriors), arguments.reference)
    rows: list[dict[str, Any]] = []
    for problem in range(arguments.posteriors[0].samples.shape[0]):
        rows.extend(_compare_problem(arguments, problem, pairs))
    return Result.from_rows(rows)


# * One problem: the capped sets with their templates, the chart, then the pairs


def _path_key(
    arguments: SignalArguments, problem: int, *tail: str | int
) -> PRNGKeyArray:
    """Return the key of a path below ``("compare_signals", problem)``."""
    return derive_key(arguments.root, "compare_signals", problem, *tail)


def _warn_validity(message: str) -> None:
    """Raise a validity warning that points at the caller's line."""
    warnings.warn(ValidityWarning(message), skip_file_prefixes=(_PACKAGE_DIRECTORY,))


def _compare_problem(
    arguments: SignalArguments, problem: int, pairs: list[tuple[int, int]]
) -> list[dict[str, Any]]:
    """Return the rows of every pair of one problem."""
    generator, noise = arguments.generators[problem], arguments.noises[problem]
    prepared = [
        _prepare_posterior(arguments, problem, posterior, generator, noise)
        for posterior in arguments.posteriors
    ]
    _check_data_grid(problem, prepared)
    state = _chart_state(arguments, problem, generator, noise)
    rows: list[dict[str, Any]] = []
    for first_index, second_index in pairs:
        hashes = [
            arguments.posteriors[first_index].hashes[problem],
            arguments.posteriors[second_index].hashes[problem],
        ]
        info = _PairInfo(
            problem,
            (prepared[first_index].name, prepared[second_index].name),
            provenance(arguments.seed, arguments.root, hashes, arguments.config_hash),
        )
        rows.extend(
            _pair_rows(
                arguments,
                info,
                prepared[first_index],
                prepared[second_index],
                state,
                data_space_text(noise),
            )
        )
    return rows


def _prepare_posterior(
    arguments: SignalArguments,
    problem: int,
    posterior: Posterior,
    generator: SignalGenerator,
    noise: NoiseModel,
) -> _Prepared:
    """Cap the positive samples of a posterior and whiten the templates of the rest."""
    positive = posterior.log_weights[problem] > -jnp.inf
    samples = posterior.samples[problem][positive]
    log_weights = posterior.log_weights[problem][positive]
    n_eff = float(posterior.n_eff[problem])
    records: list[WarningRecord] = []
    if samples.shape[0] > arguments.cap:
        kish_size = float(kish_effective_sample_size(log_weights))
        with errors.capture_warnings() as cap_records:
            samples, log_weights = cap_samples(
                samples,
                log_weights,
                key=_path_key(arguments, problem, posterior.name, "subsample"),
                metric=_METRIC,
                cap=arguments.cap,
            )
        records.extend(cap_records)
        subsample_kish_size = float(kish_effective_sample_size(log_weights))
        # * A subset without a heavy sample can exceed K, which the design effect bars.
        n_eff = subsample_effective_size(
            n_eff, kish_size, min(subsample_kish_size, kish_size)
        )
    with errors.capture_warnings() as template_records:
        templates = whitened_templates(
            generator,
            noise,
            samples,
            backend=arguments.backend,
            chunk_size=arguments.chunk_size,
        )
    records.extend(template_records)
    return _Prepared(
        posterior.name,
        samples,
        jax.nn.log_softmax(log_weights),
        n_eff,
        templates,
        records,
    )


def _check_data_grid(problem: int, prepared: list[_Prepared]) -> None:
    """Raise unless every posterior of a problem has the same number of data entries."""
    reference = prepared[0]
    for other in prepared[1:]:
        if other.templates.shape[1] != reference.templates.shape[1]:
            raise ValueError(
                f"the templates of problem {problem} must share one data grid, but"
                f" {reference.name!r} gives M = {reference.templates.shape[1]} data"
                f" entries and {other.name!r} gives M = {other.templates.shape[1]}"
            )


def _information_matrix(generator: Any, noise: NoiseModel, chunk: Any) -> Any:
    """Return the Fisher matrices of a chunk of parameters, in the caller's library."""
    return generator.information_matrix(chunk, noise=noise)


def _chart_state(
    arguments: SignalArguments,
    problem: int,
    generator: SignalGenerator,
    noise: NoiseModel,
) -> _ChartState | None:
    """Build the Fisher chart of one problem, or find that its matrix is invalid."""
    if arguments.centre is None:
        return None
    centre = arguments.centre[problem]
    size = len(arguments.space)
    chart = None
    with errors.capture_warnings() as records:
        if arguments.fisher is None:
            matrix = call_in_chunks(
                partial(_information_matrix, generator, noise),
                centre[None],
                backend=arguments.backend,
                value_shape=(size, size),
            )[0]
        else:
            matrix = arguments.fisher[problem]
        try:
            chart = FisherChart.from_fisher(matrix, centre, arguments.space)
        except ValueError as error:
            _warn_validity(
                f"the Fisher matrix of problem {problem} is not valid: {error}"
            )
    return _ChartState(
        chart, records, settings_text({"fisher": arguments.fisher_source})
    )


# * One pair: the pool, the null plan, the sides, and the two rows


def _side_rows(
    pool_size: int,
    positions: Int[Array, "draw selected"],
    log_weights: Float[Array, " _"],
) -> Float[Array, "draw sample"]:
    """Return one row of log weights over the pool per row of positions.

    The log weight of a position is that of ``log_weights``, and −∞ marks every
    position outside the side.
    """
    count = positions.shape[0]
    empty = jnp.full((count, pool_size), -jnp.inf)
    return empty.at[jnp.arange(count)[:, None], positions].set(log_weights[positions])


def _null_plan(
    arguments: SignalArguments, info: _PairInfo, first: _Prepared, second: _Prepared
) -> tuple[NullPlan, list[WarningRecord]]:
    """Return the null plan of a pair, with the warnings that it raised."""
    key = _path_key(arguments, info.problem, *info.names, "null")
    with errors.capture_warnings() as records:
        if arguments.construction is Construction.SPLIT_HALF:
            plan = split_half_null(
                first.samples,
                first.log_weights,
                approximation_n_eff=second.n_eff,
                key=key,
                reference_n_eff=first.n_eff,
                draws=arguments.draws,
            )
        else:
            plan = permutation_null(
                first.samples,
                first.log_weights,
                second.samples,
                second.log_weights,
                key=key,
                first_n_eff=first.n_eff,
                second_n_eff=second.n_eff,
                draws=arguments.draws,
            )
    return plan, records


def _pair_sides(
    arguments: SignalArguments,
    info: _PairInfo,
    sizes: tuple[int, int],
    pool_log_weights: Float[Array, " sample"],
    plan: NullPlan,
) -> _Sides:
    """Return the log weights of the observed sides, the draws and the halves."""
    pool_size = pool_log_weights.shape[0]
    first_count, second_count = sizes
    first_positions = [jnp.arange(first_count)[None]]
    second_positions = [first_count + jnp.arange(second_count)[None]]
    first_rows = [_side_rows(pool_size, first_positions[0], pool_log_weights)]
    second_rows = [_side_rows(pool_size, second_positions[0], pool_log_weights)]
    first_rows.append(_side_rows(pool_size, plan.first, plan.log_weights))
    second_rows.append(_side_rows(pool_size, plan.second, plan.log_weights))
    half_samples = 0
    if arguments.uncertainty is UncertaintyMethod.SUBSAMPLING_BOOTSTRAP:
        key = _path_key(arguments, info.problem, *info.names, "half_samples")
        pairs = arguments.half_samples
        half_first = _half_positions(key, "first", first_count, pairs)
        half_second = first_count + _half_positions(key, "second", second_count, pairs)
        first_rows.append(_side_rows(pool_size, half_first, pool_log_weights))
        second_rows.append(_side_rows(pool_size, half_second, pool_log_weights))
        half_samples = pairs
    return _Sides(
        jnp.concatenate(first_rows),
        jnp.concatenate(second_rows),
        arguments.draws,
        half_samples,
    )


def _energy_values(distances: Float[Array, "sample sample"], sides: _Sides) -> _Values:
    """Evaluate the energy distance of every row of sides on one distance matrix."""
    pool_size = distances.shape[0]
    block = max(1, CHUNK_ENTRIES // pool_size)
    values = jnp.concatenate(
        [
            energy_distance_from_distances(
                distances,
                sides.first[start : start + block],
                sides.second[start : start + block],
            )
            for start in range(0, sides.first.shape[0], block)
        ]
    )
    return _Values(
        float(values[0]),
        values[1 : 1 + sides.draws],
        values[1 + sides.draws :] if sides.half_samples else None,
    )


def _pair_rows(
    arguments: SignalArguments,
    info: _PairInfo,
    first: _Prepared,
    second: _Prepared,
    state: _ChartState | None,
    description: str,
) -> list[dict[str, Any]]:
    """Return the data row and, with a Fisher matrix, the Fisher row of one pair."""
    pool_samples = jnp.concatenate([first.samples, second.samples])
    pool_log_weights = jnp.concatenate([first.log_weights, second.log_weights])
    plan, plan_records = _null_plan(arguments, info, first, second)
    sides = _pair_sides(
        arguments,
        info,
        (first.samples.shape[0], second.samples.shape[0]),
        pool_log_weights,
        plan,
    )
    prefix = [] if arguments.transform_text is None else [arguments.transform_text]
    head = [*first.records, *second.records, *plan_records]
    distances = distance_matrix(jnp.concatenate([first.templates, second.templates]))
    values = _energy_values(distances, sides)
    # * One matrix at a time: the next one needs the memory of this one.
    del distances
    data_size = first.templates.shape[1]
    rows = [
        _energy_row(
            arguments,
            _common(
                arguments,
                info,
                settings_text({"data_size": data_size}),
                space_text(*prefix, description),
            ),
            values,
            plan,
            (first, second),
            head,
        )
    ]
    if state is None:
        return rows
    common = _common(
        arguments, info, state.settings, space_text(*prefix, CHART_DESCRIPTION)
    )
    if state.chart is None:
        rows.append(_unavailable_row(common, [*head, *state.records]))
        return rows
    with errors.capture_warnings() as wide_records:
        state.chart.warn_wide_axes(pool_samples, pool_log_weights)
    distances = distance_matrix(state.chart.forward(pool_samples))
    values = _energy_values(distances, sides)
    del distances
    rows.append(
        _energy_row(
            arguments,
            common,
            values,
            plan,
            (first, second),
            [*head, *state.records, *wide_records],
        )
    )
    return rows


def _common(
    arguments: SignalArguments, info: _PairInfo, settings: str, space: str
) -> dict[str, Any]:
    """Return the columns that every row of a pair has."""
    return {
        "problem": info.problem,
        "pair": list(info.names),
        "metric": _METRIC,
        "estimator": _ESTIMATOR,
        "estimator_settings": settings,
        "space": space,
        "parameters": list(arguments.space.names),
        "threshold": float(arguments.thresholds[_METRIC]),
        "provenance": info.provenance,
    }


def _unavailable_row(
    common: dict[str, Any], records: list[WarningRecord]
) -> dict[str, Any]:
    """Return a row without a chart: its value and effect size are NaN."""
    return common | {
        "value": _NAN,
        "p_value": _NAN,
        "sigma_sig": _NAN,
        "sigma_eff": _NAN,
        "flag": False,
        "warnings": records,
    }


def _energy_row(
    arguments: SignalArguments,
    common: dict[str, Any],
    values: _Values,
    plan: NullPlan,
    sets: tuple[_Prepared, _Prepared],
    records: list[WarningRecord],
) -> dict[str, Any]:
    """Return a row with its null, significance, effect size and verdict."""
    draws = values.draws
    tail = comparison_tail(_METRIC)
    null_mean, null_std = null_moments(draws)
    with errors.capture_warnings() as significance_records:
        significance = null_significance(jnp.asarray(values.observed), draws, tail)
    sigma_sig = float(significance.sigma_sig)
    subtracted = null_subtract(_METRIC, values.observed, draws)
    sigma_value = float(
        sigma_eff(
            _METRIC,
            subtracted,
            estimator=_ESTIMATOR,
            dimension=len(arguments.space),
        )
    )
    outcome = verdict(
        _METRIC,
        sigma_eff=sigma_value,
        sigma_sig=sigma_sig,
        thresholds=arguments.thresholds,
        level=arguments.level,
    )
    if values.half_values is None:
        uncertainty, method = null_std, UncertaintyMethod.NULL_STD
    else:
        first, second = sets
        uncertainty = float(
            half_sample_error(
                values.half_values,
                first.log_weights,
                second.log_weights,
                first_n_eff=first.n_eff,
                second_n_eff=second.n_eff,
            )
        )
        method = UncertaintyMethod.SUBSAMPLING_BOOTSTRAP
    return common | {
        "value": values.observed,
        "uncertainty": uncertainty,
        "uncertainty_method": method,
        "null_mean": null_mean,
        "null_std": null_std,
        "null_draws": draws.tolist(),
        "null_recipe": plan.recipe(tail=tail),
        "p_value": float(significance.p_value),
        "sigma_sig": sigma_sig,
        "sigma_eff": sigma_value,
        "r_eff": None,
        "flag": outcome.failed,
        "warnings": [*records, *significance_records],
    }
