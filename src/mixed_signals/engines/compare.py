"""``compare``: the metrics, nulls and effect sizes of posteriors of one data set.

:func:`compare` compares pairs of posteriors, from one
:class:`~mixed_signals.core.samples.SampleSet` or from several that share their
problems and parameters, and returns one :class:`~mixed_signals.core.result.Result`
table. Every number comes with a finite-sample null, its significance σ_sig and
an effect size, σ_eff or r_eff, on which the thresholds act. It is a plain
function; the package warnings raised in a row go into its ``warnings`` column.

Modes
    ``reference=<name>`` selects mode A: the pairs (reference, other) of every
    other posterior in the order of ``names``, whitened by the reference. Without
    it, mode B takes all pairs (names[i], names[j]) with i < j and whitens each
    pair by both posteriors, pooled. ``whitening=`` overrides the source:
    ``"reference"`` (mode A only), ``"pooled"``, or ``"fisher"`` with ``fisher=``
    and ``centre=``. In a pair (p, q), a metric reads from p to q, so the
    reverse KL rows carry the pair [q, p].
Zero weights
    A sample with a log weight of −∞ has no weight, so every problem is read
    without such samples before each whitening, statistic, null, input hash and
    Kish size. Padding with zero weights, as for sources that are in a chain
    for part of a run, changes no row, bit for bit.
Transform
    A ``transform`` maps the sets, the space, the Fisher matrices (Γ_y = JᵀΓJ,
    Jeffreys 1946) and the centres first, and the whitening follows it, so the
    ``space`` column reads ``<transform> | whiten(<source>)``. The input hashes
    of the provenance are those of the untransformed input.
Whitening
    Each pair uses two whitenings built on concrete arrays (Kessy, Lewin and
    Strimmer 2018): a decorrelated one, in whose coordinates the joint rows run,
    and a z-score of every parameter by its own topology, a circle for a sphere's
    azimuth and an interval for its polar axis, for the per-parameter rows. A
    pooled whitening of sets with different sample counts pads the smaller set
    with zero-weight copies of its first sample. A whitening that is NaN, for a
    singular covariance, leaves its rows at NaN with its warning, and the other
    rows and problems go on. Such a row has a threshold, no null, and does not
    flag.
Recipes
    A recipe is a metric with an estimator, :data:`RECIPES`. ``metrics`` takes
    metric names, which select every estimator of the metric, or
    ``"metric:estimator"`` for one recipe. :data:`DEFAULT_METRICS` is the fast
    set; the kNN rows, the Sinkhorn divergence and the sliced Sinkhorn
    divergence run on request, and the classifier test is a placeholder. The
    metrics are those of :mod:`mixed_signals.metrics`. Joint metrics of one
    pair share one whitening and one null plan.
Nulls
    ``null="permutation"`` (the default) pools the two sets and cuts the pool
    into two random sides; ``null="split_half"``, in mode A, cuts the reference
    alone (Good 2005). The sides carry the effective sample size of the sets
    (Kish 1965), and every sample keeps its weight, also for the joint metrics.
    The p-value has the +1 correction (Phipson and Smyth 2010) and σ_sig is a
    Gaussian significance (Cowan et al. 2011); the tail is two-sided for the
    signed metrics, the mean shift and the width ratio, and upper for the others.
    The metric is evaluated on the same positions for every recipe, so a recipe
    added to the selection leaves the null of the others unchanged.
Effect sizes
    The null mean is subtracted in the space of the metric, then σ_eff or r_eff
    follows from the closed-form maps of :mod:`mixed_signals.sigma.maps`.
    The W2 shape term is the only shape term that depends on the frame: the maps
    take the first posterior to have unit width, as a reference whitening makes
    it. With a pooled whitening, in mode B or with ``whitening="pooled"``,
    r_eff solves the pooled relation instead; with a Fisher whitening it keeps
    the reference map, which assumes that the first posterior has the width of
    the Fisher matrix.
    ``threshold`` is the default of :data:`~mixed_signals.sigma.thresholds.THRESHOLDS`
    or the caller's override, and ``flag`` says whether the effect size reaches
    it while σ_sig reaches ``level``.
Nearest-neighbour rows
    The kNN estimators need equally weighted samples without copies, so each
    whitened set is converted once per comparison by
    :func:`~mixed_signals.core.samples.rejection_subset` (Smith and Gelfand
    1992; Ashton et al. 2019), with an effective size of M times the relative
    efficiency of the input. Both sets are cut to the smaller count and to
    ``cap``, one kNN null runs on the cut sets, and the observed values use the
    same sets. With ``null="split_half"``, each prepared set is cut to ``cap``
    only and the null cuts the prepared reference into two sides. The JS
    statistic takes the first min(N₁, N₂) samples of its two inputs, which are
    in random order, so that the sets of the observed value, the null sides and
    the half-samples may differ in size.
Uncertainty
    ``uncertainty="null_std"`` reports the standard deviation of the null draws.
    ``"subsampling_bootstrap"`` reports the half-sample error of
    :func:`~mixed_signals.sigma.uncertainty.half_sample_error`, from
    ``half_samples`` pairs of random half-samples (Politis, Romano and Wolf 1999;
    Shao and Wu 1989); ``null_std`` stays in its column.
Keys
    A key or an integer seed seeds the call. Every random choice derives its key
    with :func:`~mixed_signals.prng.derive_key` from a path of names that starts
    with ``("compare", problem, p_name, q_name)``, so a row does not depend on
    the other rows of the call.

References:
    :cite:t:`good2005permutation`; :cite:t:`kish1965survey`;
    :cite:t:`phipson2010permutation`; :cite:t:`cowan2011asymptotic`;
    :cite:t:`kessy2018optimal`; :cite:t:`jeffreys1946invariant`;
    :cite:t:`smith1992bayesian`; :cite:t:`ashton2019bilby`;
    :cite:t:`politis1999subsampling`; :cite:t:`shao1989general`.
"""

import math
import os
import warnings
from collections.abc import Callable, Mapping, Sequence
from functools import partial
from typing import Any, Final, NamedTuple

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from jaxtyping import Array, PRNGKeyArray

from mixed_signals import errors
from mixed_signals._config import COST_BUDGET, NULL_DRAWS, SAMPLE_CAP
from mixed_signals.core.result import Provenance, Result, UncertaintyMethod
from mixed_signals.core.samples import SampleSet, rejection_subset
from mixed_signals.core.space import ParamSpace
from mixed_signals.engines import _arguments
from mixed_signals.engines._arguments import (
    FAMILIES,
    SETTINGS_OF_FAMILY,
    CompareArguments,
    Recipe,
    Scope,
    comparison_pairs,
    read_compare_arguments,
)
from mixed_signals.engines._common import (
    KeyInput,
    Posterior,
    flat_space,
    null_moments,
    problem_samples,
    provenance,
    settings_text,
    space_text,
)
from mixed_signals.engines.cost import _compare_cost, warn_if_over_budget
from mixed_signals.errors import SubsamplingWarning, ValidityWarning, WarningRecord
from mixed_signals.estimators.gaussian import GaussianDensity, GaussianSettings
from mixed_signals.estimators.kde import KernelDensitySettings
from mixed_signals.estimators.knn import NearestNeighbourSettings
from mixed_signals.metrics._embedding import embedding_size
from mixed_signals.metrics.divergence import (
    HELLINGER_NEIGHBOURS,
    JS_NEIGHBOURS,
    KL_NEIGHBOURS,
    knn_hellinger_distance,
    knn_js_divergence,
    knn_kl_divergence,
)
from mixed_signals.metrics.kernel import (
    energy_distance,
    maximum_mean_discrepancy,
    median_bandwidth,
)
from mixed_signals.metrics.marginal import (
    ks_metric_names,
    marginal_distances,
    marginal_kde_divergences,
    shared_bandwidth_factor,
)
from mixed_signals.metrics.moments import (
    gaussian_hellinger,
    gaussian_kl,
    gaussian_w2,
    mean_shift,
    width_ratio,
)
from mixed_signals.metrics.ot.sinkhorn import (
    SINKHORN_EPSILON_FACTOR,
    sinkhorn_divergence,
)
from mixed_signals.metrics.ot.sliced import (
    SLICED_DIRECTIONS,
    SLICED_SINKHORN_CAP,
    SLICED_SINKHORN_DIRECTIONS,
    gaussian_sliced_wasserstein_distance,
    sliced_directions,
    sliced_sinkhorn_divergence,
    sliced_sinkhorn_epsilon,
    sliced_wasserstein_distance,
)
from mixed_signals.nulls.permutation import permutation_null
from mixed_signals.nulls.plan import Construction, NullPlan, Statistic, null_draws
from mixed_signals.nulls.split_half import split_half_null
from mixed_signals.prng import derive_key
from mixed_signals.reweight.importance import relative_efficiency
from mixed_signals.sigma.calibration import CalibrationCurve, knn_kl_curve
from mixed_signals.sigma.maps import (
    null_subtract,
    r_eff,
    sigma_eff,
    warn_wide_chordal_axes,
)
from mixed_signals.sigma.significance import null_significance
from mixed_signals.sigma.thresholds import (
    R_EFF_METRICS,
    SIGNIFICANCE_LEVEL,
    THRESHOLDS,
    comparison_tail,
    verdict,
)
from mixed_signals.sigma.uncertainty import (
    HALF_SAMPLE_PAIRS,
    half_sample_draws,
    half_sample_error,
)
from mixed_signals.transforms.base import Transform
from mixed_signals.transforms.whiten import Whitening, WhiteningSource

__all__ = ["DEFAULT_METRICS", "RECIPES", "Recipe", "Scope", "compare"]

RECIPES: Final = _arguments.RECIPES
"""Every recipe by its name ``"metric:estimator"``, in the order of the result rows.

The recipes are :class:`Recipe` records of the metric, the estimator and the
scope of the rows. ``metrics`` of :func:`compare` selects them by this name, or
by the name of a metric, which selects every estimator of the metric.

:meta hide-value:
"""

DEFAULT_METRICS: Final = _arguments.DEFAULT_METRICS
"""The recipes that ``compare`` runs by default, the fast set.

The kNN rows, the Sinkhorn divergence and the sliced Sinkhorn divergence cost
more and run on request.
"""

# * Frames in the package are skipped, so a warning points at the caller's line.
_PACKAGE_DIRECTORY: Final = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


_NAN: Final = math.nan


class _Row(NamedTuple):
    """What a result row needs beyond the evaluation of its family."""

    metric: str
    estimator: str
    position: int
    parameters: tuple[str, ...]
    settings: str
    reverse: bool = False
    log_det_position: int | None = None
    bandwidth: float | None = None
    dimension: int | None = None
    curve: CalibrationCurve | None = None


class _Layout(NamedTuple):
    """Names and sizes of a comparison that do not depend on its samples."""

    parameter_names: tuple[str, ...]
    ks_names: tuple[str, ...]
    joint_names: tuple[str, ...]
    dimension: int
    embedding: int
    cap: int
    factors: tuple[float, ...]


class _Computed(NamedTuple):
    """The settings that a comparison fixes once and its recipes share."""

    factor: float | None = None
    bandwidth: float | None = None
    directions: Array | None = None
    sinkhorn_directions: Array | None = None
    sinkhorn_epsilon: float | None = None
    curves: tuple[CalibrationCurve, CalibrationCurve] | None = None


class _Built(NamedTuple):
    """A whitening with the warnings of its construction."""

    whitening: Whitening
    records: list[WarningRecord]
    valid: bool


class _PairWhitenings(NamedTuple):
    """The two whitenings of a pair; ``None`` where no selected recipe needs one."""

    joint: _Built | None
    parameter: _Built | None


class _Whitened(NamedTuple):
    """A whitening applied to the two sets of a pair."""

    built: _Built
    text: str
    space: ParamSpace | None
    first: Array | None
    second: Array | None


class _Sets(NamedTuple):
    """The two sets that a family evaluates, with their effective sample sizes."""

    first: Array
    first_log_weights: Array
    second: Array
    second_log_weights: Array
    first_n_eff: float
    second_n_eff: float


class _Evaluation(NamedTuple):
    """The observed values, null draws and half-sample errors of one family."""

    observed: Array
    draws: Array
    half_errors: Array | None
    records: list[WarningRecord]


class _Knn(NamedTuple):
    """The prepared sets, null plan and settings of the nearest-neighbour rows."""

    sets: _Sets
    plan: NullPlan
    curves: tuple[CalibrationCurve, CalibrationCurve] | None
    records: list[WarningRecord]
    curve_records: list[WarningRecord]


class _PairInfo(NamedTuple):
    """What every row of one comparison shares."""

    problem: int
    names: tuple[str, str]
    provenance: Provenance
    reverse_provenance: Provenance
    """The provenance of a row of the pair in reverse order, with its hashes."""


def compare(
    sample_sets: SampleSet | Sequence[SampleSet],
    names: Sequence[str],
    space: ParamSpace,
    *,
    key: KeyInput,
    reference: str | None = None,
    whitening: WhiteningSource | str | None = None,
    fisher: ArrayLike | None = None,
    centre: ArrayLike | None = None,
    transform: Transform | None = None,
    metrics: Sequence[str] | None = None,
    null: Construction | str = Construction.PERMUTATION,
    draws: int = NULL_DRAWS,
    cap: int = SAMPLE_CAP,
    sinkhorn_epsilon_factors: Sequence[float] = (SINKHORN_EPSILON_FACTOR,),
    uncertainty: UncertaintyMethod | str = UncertaintyMethod.NULL_STD,
    half_samples: int = HALF_SAMPLE_PAIRS,
    thresholds: Mapping[str, float] = THRESHOLDS,
    level: float = SIGNIFICANCE_LEVEL,
    budget: float = COST_BUDGET,
    config_hash: str | None = None,
) -> Result:
    """Compare posteriors pairwise, with a null, σ_sig and an effect size per row.

    The module docstring describes the modes, the whitenings, the recipes, the
    nulls and the rows. The package warnings that a row raises, such as a
    :class:`~mixed_signals.errors.ValidityWarning` of a singular whitening, are
    not raised but go into the ``warnings`` column of the row.

    Args:
        sample_sets: A :class:`~mixed_signals.core.samples.SampleSet` shaped
            (B, K, N, D), or a sequence of sets with the same B and D, whose
            posteriors join in order. The numbers of samples may differ.
        names: One unique name per posterior, in the order of the posteriors.
        space: The parameter space of the samples, with D parameters.
        key: A JAX PRNG key, or an integer seed that becomes
            :func:`~mixed_signals.prng.root_key` and fills the provenance seed.
        reference: The name of the reference posterior, for mode A; ``None``
            compares every pair, mode B.
        whitening: ``"reference"`` (mode A only), ``"pooled"`` or ``"fisher"``;
            ``"reference"`` in mode A and ``"pooled"`` in mode B by default.
        fisher: Fisher matrices for ``"fisher"``, shaped (B, D, D), or (D, D)
            when B = 1, in the coordinates of the input.
        centre: Points at which the Fisher matrices hold, shaped (B, D), or (D,)
            when B = 1, in the coordinates of the input.
        transform: A transform that maps the samples to the coordinates of the
            comparison first; ``None`` for the coordinates of the input.
        metrics: Metric names and ``"metric:estimator"`` recipes; the
            :data:`DEFAULT_METRICS` when ``None``.
        null: ``"permutation"``, or ``"split_half"`` in mode A.
        draws: Number of null draws per row.
        cap: Largest number of samples per posterior in a metric whose cost grows
            as N², and in the kNN rows.
        sinkhorn_epsilon_factors: The factors of the squared median bandwidth that
            give the regularisation ε; the ``sinkhorn`` metric has one row per
            factor.
        uncertainty: ``"null_std"`` or ``"subsampling_bootstrap"``.
        half_samples: Number of half-sample pairs of the subsampling bootstrap.
        thresholds: The threshold of each metric; the package defaults.
        level: The σ_sig that a row must reach to flag.
        budget: The estimated run time in seconds above which the dry run
            raises a :class:`~mixed_signals.errors.CostWarning`; infinity
            switches the warning off.
        config_hash: SHA-256 digest of the pipeline config, for the provenance.

    Returns:
        The result table: problems, then pairs, then recipes, in order.

    Raises:
        ValueError: If the sets, names, space, mode, whitening, null, metrics,
            counts, factors, uncertainty method or thresholds do not fit, or a
            Fisher whitening gets matrices that are not symmetric and positive
            definite.
        TypeError: If a count is not an integer or an input is of another type
            than the arguments need.
        KeyError: If the thresholds lack a metric of the selected rows.
        NotImplementedError: If the metrics hold the classifier test.

    References:
        :cite:t:`good2005permutation`; :cite:t:`phipson2010permutation`;
        :cite:t:`cowan2011asymptotic`; :cite:t:`kessy2018optimal`.
    """
    arguments = read_compare_arguments(
        sample_sets,
        names,
        space,
        key=key,
        reference=reference,
        whitening=whitening,
        fisher=fisher,
        centre=centre,
        transform=transform,
        metrics=metrics,
        null=null,
        draws=draws,
        cap=cap,
        sinkhorn_epsilon_factors=sinkhorn_epsilon_factors,
        uncertainty=uncertainty,
        half_samples=half_samples,
        thresholds=thresholds,
        level=level,
        budget=budget,
        config_hash=config_hash,
    )
    warn_if_over_budget("compare", _compare_cost(arguments), arguments.budget)
    layout = _layout(arguments)
    scopes = {RECIPES[recipe].scope for recipe in arguments.recipes}
    batch_count = arguments.posteriors[0].samples.shape[0]
    pairs = comparison_pairs(len(arguments.posteriors), arguments.reference)
    shared = {
        problem: _shared_whitenings(arguments, problem, scopes)
        for problem in range(batch_count)
        if arguments.source is not WhiteningSource.POOLED
    }
    rows: list[dict[str, Any]] = []
    for problem in range(batch_count):
        for first_index, second_index in pairs:
            first = arguments.posteriors[first_index]
            second = arguments.posteriors[second_index]
            built = shared.get(problem) or _pooled_whitenings(
                arguments, problem, first, second, scopes
            )
            rows.extend(_compare_pair(arguments, layout, problem, first, second, built))
    return Result.from_rows(rows)


def _layout(arguments: CompareArguments) -> _Layout:
    """Return the names and sizes of the comparison."""
    space = arguments.space
    flat = flat_space(space)
    return _Layout(
        parameter_names=flat.names,
        ks_names=ks_metric_names(flat),
        joint_names=space.names,
        dimension=len(space),
        embedding=embedding_size(space),
        cap=arguments.cap,
        factors=arguments.factors,
    )


# * Whitenings


def _samples_whitening(
    space: ParamSpace,
    decorrelate: bool,
    *,
    samples: Array,
    log_weights: Array,
) -> Whitening:
    """Whiten by the covariance of the posteriors in ``samples``, shaped (K, N, D)."""
    return Whitening.from_samples(samples, log_weights, space, decorrelate=decorrelate)


def _fisher_whitening(
    space: ParamSpace, decorrelate: bool, *, fisher: Array, centre: Array
) -> Whitening:
    """Whiten by the inverse of one Fisher matrix at one centre."""
    return Whitening.from_fisher(fisher, centre, space, decorrelate=decorrelate)


def _build_whitenings(
    builder: Callable[[ParamSpace, bool], Whitening],
    arguments: CompareArguments,
    scopes: set[Scope],
) -> _PairWhitenings:
    """Build the whitenings that the selected scopes need, with their warnings."""
    built: dict[Scope, _Built] = {}
    for scope, space, decorrelate in (
        (Scope.JOINT, arguments.space, True),
        (Scope.PARAMETER, flat_space(arguments.space), False),
    ):
        if scope not in scopes:
            continue
        with errors.capture_warnings() as records:
            whitening = builder(space, decorrelate)
        built[scope] = _Built(
            whitening, records, not math.isnan(float(whitening.log_abs_det))
        )
    return _PairWhitenings(built.get(Scope.JOINT), built.get(Scope.PARAMETER))


def _shared_whitenings(
    arguments: CompareArguments, problem: int, scopes: set[Scope]
) -> _PairWhitenings:
    """Return the whitenings of one problem from its reference or its Fisher matrix."""
    if arguments.source is WhiteningSource.REFERENCE:
        assert arguments.reference is not None
        reference_samples, reference_log_weights = problem_samples(
            arguments.posteriors[arguments.reference], problem
        )
        builder: Callable[[ParamSpace, bool], Whitening] = partial(
            _samples_whitening,
            samples=reference_samples[None],
            log_weights=reference_log_weights[None],
        )
    else:
        assert arguments.fisher is not None
        assert arguments.centre is not None
        builder = partial(
            _fisher_whitening,
            fisher=arguments.fisher[problem],
            centre=arguments.centre[problem],
        )
    return _build_whitenings(builder, arguments, scopes)


def _pooled_whitenings(
    arguments: CompareArguments,
    problem: int,
    first: Posterior,
    second: Posterior,
    scopes: set[Scope],
) -> _PairWhitenings:
    """Return the whitenings of the two posteriors of a pair, pooled."""
    samples, log_weights = _padded_pair(first, second, problem)
    builder = partial(_samples_whitening, samples=samples, log_weights=log_weights)
    return _build_whitenings(builder, arguments, scopes)


def _padded_pair(
    first: Posterior, second: Posterior, problem: int
) -> tuple[Array, Array]:
    """Stack the kept samples of two posteriors, padding the smaller by copies."""
    sets = [problem_samples(first, problem), problem_samples(second, problem)]
    count = max(samples.shape[0] for samples, _ in sets)
    padded_samples, padded_weights = [], []
    for samples, log_weights in sets:
        missing = count - samples.shape[0]
        padding = jnp.repeat(samples[:1], missing, axis=0)
        padded_samples.append(jnp.concatenate([samples, padding]))
        padded_weights.append(
            jnp.concatenate([log_weights, jnp.full(missing, -jnp.inf)])
        )
    return jnp.stack(padded_samples), jnp.stack(padded_weights)


def _whiten_pair(
    built: _Built | None,
    row_space: ParamSpace,
    first: Array,
    second: Array,
    transform_text: str | None,
) -> _Whitened | None:
    """Apply a whitening to both sets and map the space, if the whitening is valid."""
    if built is None:
        return None
    parts = [] if transform_text is None else [transform_text]
    text = space_text(*parts, built.whitening.description)
    if not built.valid:
        return _Whitened(built, text, None, None, None)
    return _Whitened(
        built,
        text,
        built.whitening.map_space(row_space),
        built.whitening.forward(first),
        built.whitening.forward(second),
    )


# * The statistics of the recipes, bound to their fixed settings with partial


def _mean_shift_statistic(
    first: Array,
    first_log_weights: Array,
    second: Array,
    second_log_weights: Array,
    *,
    space: ParamSpace,
) -> Array:
    """Return the mean shift of every parameter."""
    return mean_shift(first, first_log_weights, second, second_log_weights, space)


def _width_ratio_statistic(
    first: Array,
    first_log_weights: Array,
    second: Array,
    second_log_weights: Array,
    *,
    space: ParamSpace,
) -> Array:
    """Return the width ratio of every parameter."""
    return width_ratio(first, first_log_weights, second, second_log_weights, space)


def _marginal_statistic(
    first: Array,
    first_log_weights: Array,
    second: Array,
    second_log_weights: Array,
    *,
    space: ParamSpace,
    names: tuple[str, ...],
) -> Array:
    """Return the selected marginal distances, each shaped (D,), one after another."""
    distances = marginal_distances(
        first, first_log_weights, second, second_log_weights, space
    )
    return jnp.concatenate([getattr(distances, name) for name in names])


def _kde_statistic(
    first: Array,
    first_log_weights: Array,
    second: Array,
    second_log_weights: Array,
    *,
    space: ParamSpace,
    factor: float,
    names: tuple[str, ...],
) -> Array:
    """Return the selected kernel density divergences, each shaped (D,)."""
    divergences = marginal_kde_divergences(
        first, first_log_weights, second, second_log_weights, space, factor=factor
    )
    return jnp.concatenate([getattr(divergences, name) for name in names])


def _gaussian_statistic(
    first: Array,
    first_log_weights: Array,
    second: Array,
    second_log_weights: Array,
    *,
    space: ParamSpace,
    term_sets: tuple[str, ...],
) -> Array:
    """Return total, location, shape and log-determinant ratio of each term set."""
    first_fit = GaussianDensity.fit(first, first_log_weights, space)
    second_fit = GaussianDensity.fit(second, second_log_weights, space)
    blocks = []
    for term_set in term_sets:
        forward = term_set != "kl_reverse"
        metric = (
            gaussian_kl
            if term_set.startswith("kl")
            else (gaussian_w2 if term_set == "w2" else gaussian_hellinger)
        )
        origin, target = (first_fit, second_fit) if forward else (second_fit, first_fit)
        terms = metric(
            origin.mean, origin.covariance, target.mean, target.covariance, space
        )
        blocks.append(
            jnp.stack([terms.total, terms.location, terms.shape, terms.log_det_ratio])
        )
    return jnp.concatenate(blocks)


def _mmd_statistic(
    first: Array,
    first_log_weights: Array,
    second: Array,
    second_log_weights: Array,
    *,
    space: ParamSpace,
    bandwidth: float,
    key: PRNGKeyArray,
    cap: int,
) -> Array:
    """Return the squared MMD of a Gaussian kernel, as a vector of one value."""
    value = maximum_mean_discrepancy(
        first,
        first_log_weights,
        second,
        second_log_weights,
        space,
        bandwidth=bandwidth,
        key=key,
        cap=cap,
    )
    return jnp.atleast_1d(value)


def _energy_statistic(
    first: Array,
    first_log_weights: Array,
    second: Array,
    second_log_weights: Array,
    *,
    space: ParamSpace,
    key: PRNGKeyArray,
    cap: int,
) -> Array:
    """Return the energy distance, as a vector of one value."""
    value = energy_distance(
        first, first_log_weights, second, second_log_weights, space, key=key, cap=cap
    )
    return jnp.atleast_1d(value)


def _sinkhorn_statistic(
    first: Array,
    first_log_weights: Array,
    second: Array,
    second_log_weights: Array,
    *,
    space: ParamSpace,
    epsilons: Array,
    key: PRNGKeyArray,
    cap: int,
) -> Array:
    """Return the Sinkhorn divergence at each ε."""
    return sinkhorn_divergence(
        first,
        first_log_weights,
        second,
        second_log_weights,
        space,
        epsilon=epsilons,
        key=key,
        cap=cap,
    )


def _sliced_w2_statistic(
    first: Array,
    first_log_weights: Array,
    second: Array,
    second_log_weights: Array,
    *,
    space: ParamSpace,
    directions: Array,
) -> Array:
    """Return the sliced W2 squared of the samples, as a vector of one value."""
    value = sliced_wasserstein_distance(
        first,
        first_log_weights,
        second,
        second_log_weights,
        space,
        directions=directions,
    )
    return jnp.atleast_1d(value)


def _gaussian_sliced_w2_statistic(
    first: Array,
    first_log_weights: Array,
    second: Array,
    second_log_weights: Array,
    *,
    space: ParamSpace,
    directions: Array,
) -> Array:
    """Return the sliced W2 squared of the Gaussian fits, as a vector of one value."""
    value = gaussian_sliced_wasserstein_distance(
        first,
        first_log_weights,
        second,
        second_log_weights,
        space,
        directions=directions,
    )
    return jnp.atleast_1d(value)


def _sliced_sinkhorn_statistic(
    first: Array,
    first_log_weights: Array,
    second: Array,
    second_log_weights: Array,
    *,
    space: ParamSpace,
    directions: Array,
    epsilon: float,
    key: PRNGKeyArray,
) -> Array:
    """Return the sliced Sinkhorn divergence, as a vector of one value."""
    value = sliced_sinkhorn_divergence(
        first,
        first_log_weights,
        second,
        second_log_weights,
        space,
        directions=directions,
        epsilon=epsilon,
        key=key,
        cap=SLICED_SINKHORN_CAP,
    )
    return jnp.atleast_1d(value)


def _knn_statistic(
    first: Array,
    first_log_weights: Array,
    second: Array,
    second_log_weights: Array,
    *,
    space: ParamSpace,
    calls: tuple[tuple[str, PRNGKeyArray], ...],
    cap: int,
) -> Array:
    """Return the selected nearest-neighbour estimates, one value per call.

    The JS divergence needs equal counts, so it reads the first min(N₁, N₂)
    samples of its inputs; the sets are in random order, so these are a uniform
    subset.
    """
    values = []
    for name, key in calls:
        if name == "js":
            count = min(first.shape[0], second.shape[0])
            value = knn_js_divergence(
                first[:count],
                None,
                second[:count],
                None,
                space,
                key=key,
                neighbours=JS_NEIGHBOURS,
                cap=cap,
            )
        elif name == "hellinger":
            value = knn_hellinger_distance(
                first,
                None,
                second,
                None,
                space,
                key=key,
                neighbours=HELLINGER_NEIGHBOURS,
                cap=cap,
            )
        else:
            origin, target = (first, second) if name == "kl" else (second, first)
            value = knn_kl_divergence(
                origin,
                None,
                target,
                None,
                space,
                key=key,
                neighbours=KL_NEIGHBOURS,
                cap=cap,
            )
        values.append(value)
    return jnp.stack(values)


def _term_sets(recipes: Sequence[str]) -> tuple[str, ...]:
    """Return the Gaussian term sets of the recipes, in the order of the rows."""
    sets: list[str] = []
    for recipe in recipes:
        metric = RECIPES[recipe].metric
        sets.extend(("kl", "kl_reverse") if metric == "kl" else (metric,))
    return tuple(sets)


def _knn_names(recipes: Sequence[str]) -> tuple[str, ...]:
    """Return the nearest-neighbour calls of the recipes, in the order of the rows."""
    names: list[str] = []
    for recipe in recipes:
        metric = RECIPES[recipe].metric
        names.extend(("kl", "kl_reverse") if metric == "kl" else (metric,))
    return tuple(names)


def _family_statistic(
    family: str,
    recipes: Sequence[str],
    *,
    space: ParamSpace,
    computed: _Computed,
    layout: _Layout,
    key: Callable[..., PRNGKeyArray],
) -> Statistic:
    """Return the statistic of a family, bound to its settings, in a whitened space."""
    metrics = tuple(RECIPES[recipe].metric for recipe in recipes)
    if family == "mean_shift":
        return partial(_mean_shift_statistic, space=space)
    if family == "width_ratio":
        return partial(_width_ratio_statistic, space=space)
    if family == "marginal":
        return partial(_marginal_statistic, space=space, names=metrics)
    if family == "kde":
        assert computed.factor is not None
        return partial(
            _kde_statistic, space=space, factor=computed.factor, names=metrics
        )
    if family == "gaussian":
        return partial(_gaussian_statistic, space=space, term_sets=_term_sets(recipes))
    if family == "knn":
        calls = tuple(
            (name, key(name.removesuffix("_reverse"), "knn", "subsample"))
            for name in _knn_names(recipes)
        )
        return partial(_knn_statistic, space=space, calls=calls, cap=layout.cap)
    if family == "mmd":
        assert computed.bandwidth is not None
        return partial(
            _mmd_statistic,
            space=space,
            bandwidth=computed.bandwidth,
            key=key("mmd", "empirical", "subsample"),
            cap=layout.cap,
        )
    if family == "energy":
        return partial(
            _energy_statistic,
            space=space,
            key=key("energy", "empirical", "subsample"),
            cap=layout.cap,
        )
    if family == "sinkhorn":
        assert computed.bandwidth is not None
        epsilons = jnp.asarray(layout.factors) * computed.bandwidth**2
        return partial(
            _sinkhorn_statistic,
            space=space,
            epsilons=epsilons,
            key=key("sinkhorn", "empirical", "subsample"),
            cap=layout.cap,
        )
    if family == "sliced_sinkhorn":
        assert computed.sinkhorn_directions is not None
        assert computed.sinkhorn_epsilon is not None
        return partial(
            _sliced_sinkhorn_statistic,
            space=space,
            directions=computed.sinkhorn_directions,
            epsilon=computed.sinkhorn_epsilon,
            key=key("sliced_sinkhorn", "empirical", "subsample"),
        )
    assert computed.directions is not None
    sliced = (
        _sliced_w2_statistic if family == "sliced_w2" else _gaussian_sliced_w2_statistic
    )
    return partial(sliced, space=space, directions=computed.directions)


# * The rows of a family: names, positions in its values and settings


def _direction_count(count: int, embedding: int) -> int:
    """Return the number of directions in complete orthonormal bases."""
    return math.ceil(count / embedding) * embedding


def _family_rows(
    family: str,
    recipes: Sequence[str],
    layout: _Layout,
    computed: _Computed,
) -> dict[str, tuple[_Row, ...]]:
    """Return the rows of every recipe of a family, in the order of the table.

    The rows name positions in the vector of values that the statistic of the
    family returns. Settings that a failed whitening leaves undefined are null.
    """
    rows: dict[str, tuple[_Row, ...]] = {}
    if family in ("mean_shift", "width_ratio", "marginal", "kde"):
        count = len(layout.parameter_names)
        for block, recipe in enumerate(recipes):
            metric, estimator, _ = RECIPES[recipe]
            settings, bandwidth = "{}", None
            if estimator == "kde":
                assert computed.factor is not None
                settings = KernelDensitySettings(bandwidth=computed.factor).to_json()
                bandwidth = computed.factor
            rows[recipe] = tuple(
                _Row(
                    metric=layout.ks_names[index] if metric == "ks" else metric,
                    estimator=estimator,
                    position=block * count + index,
                    parameters=(name,),
                    settings=settings,
                    bandwidth=bandwidth,
                )
                for index, name in enumerate(layout.parameter_names)
            )
    elif family == "gaussian":
        offset = 0
        for recipe in recipes:
            metric = RECIPES[recipe].metric
            rows_of_recipe = []
            for reverse in (False, True) if metric == "kl" else (False,):
                for suffix, shift in (("", 0), ("_location", 1), ("_shape", 2)):
                    rows_of_recipe.append(
                        _Row(
                            metric=metric + suffix,
                            estimator="gaussian",
                            position=offset + shift,
                            parameters=layout.joint_names,
                            settings=GaussianSettings().to_json(),
                            reverse=reverse,
                            log_det_position=offset + 3 if suffix == "_shape" else None,
                        )
                    )
                offset += 4
            rows[recipe] = tuple(rows_of_recipe)
    elif family == "knn":
        position = 0
        for recipe in recipes:
            metric = RECIPES[recipe].metric
            neighbours = {
                "kl": KL_NEIGHBOURS,
                "js": JS_NEIGHBOURS,
                "hellinger": HELLINGER_NEIGHBOURS,
            }[metric]
            settings = NearestNeighbourSettings(neighbours=neighbours).to_json()
            rows_of_recipe = []
            for reverse in (False, True) if metric == "kl" else (False,):
                curve = None
                if metric == "kl" and computed.curves is not None:
                    curve = computed.curves[int(reverse)]
                rows_of_recipe.append(
                    _Row(
                        metric=metric,
                        estimator="knn",
                        position=position,
                        parameters=layout.joint_names,
                        settings=settings,
                        reverse=reverse,
                        curve=curve,
                    )
                )
                position += 1
            rows[recipe] = tuple(rows_of_recipe)
    else:
        rows[recipes[0]] = _joint_rows(family, recipes[0], layout, computed)
    return rows


def _joint_rows(
    family: str, recipe: str, layout: _Layout, computed: _Computed
) -> tuple[_Row, ...]:
    """Return the rows of a family whose statistic gives one value per row."""
    metric, estimator, _ = RECIPES[recipe]
    joint = layout.joint_names
    bandwidth = computed.bandwidth
    if family == "mmd":
        settings = settings_text({"bandwidth": bandwidth, "cap": layout.cap})
        return (
            _Row(
                metric,
                estimator,
                0,
                joint,
                settings,
                bandwidth=bandwidth,
                dimension=layout.dimension,
            ),
        )
    if family == "energy":
        settings = settings_text({"cap": layout.cap})
        return (
            _Row(metric, estimator, 0, joint, settings, dimension=layout.dimension),
        )
    if family == "sinkhorn":
        rows = []
        for position, factor in enumerate(layout.factors):
            epsilon = None if bandwidth is None else factor * bandwidth**2
            settings = settings_text(
                {
                    "bandwidth": bandwidth,
                    "cap": layout.cap,
                    "epsilon": epsilon,
                    "epsilon_factor": factor,
                }
            )
            rows.append(_Row(metric, estimator, position, joint, settings))
        return tuple(rows)
    if family == "sliced_sinkhorn":
        directions = computed.sinkhorn_directions
        count = (
            _direction_count(SLICED_SINKHORN_DIRECTIONS, layout.embedding)
            if directions is None
            else directions.shape[0]
        )
        settings = settings_text(
            {
                "cap": SLICED_SINKHORN_CAP,
                "directions": count,
                "epsilon": computed.sinkhorn_epsilon,
                "orthonormal": True,
            }
        )
        return (
            _Row(metric, estimator, 0, joint, settings, dimension=layout.embedding),
        )
    count = _direction_count(SLICED_DIRECTIONS, layout.embedding)
    settings = settings_text({"directions": count, "orthonormal": True})
    return (_Row(metric, estimator, 0, joint, settings, dimension=layout.embedding),)


# * Settings that a comparison fixes once, and the nulls


def _path_key(
    root: PRNGKeyArray, path: tuple[str | int, ...], *tail: str | int
) -> PRNGKeyArray:
    """Return the key of a path below the comparison path."""
    return derive_key(root, *path, *tail)


def _warn_validity(message: str) -> None:
    """Raise a validity warning that points at the caller's line."""
    warnings.warn(ValidityWarning(message), skip_file_prefixes=(_PACKAGE_DIRECTORY,))


def _warn_subsampling(message: str) -> None:
    """Raise a subsampling warning that points at the caller's line."""
    warnings.warn(SubsamplingWarning(message), skip_file_prefixes=(_PACKAGE_DIRECTORY,))


def _compute_settings(
    families: set[str],
    joint: _Whitened | None,
    first: Posterior,
    second: Posterior,
    problem: int,
    key: Callable[..., PRNGKeyArray],
) -> tuple[_Computed, dict[str, list[WarningRecord]]]:
    """Fix the bandwidth factor, kernel bandwidth, directions and ε of a comparison.

    Each is computed only when a selected recipe reads it, and the warnings of
    its computation go to the rows of those recipes.
    """
    records: dict[str, list[WarningRecord]] = {}
    factor = bandwidth = directions = sinkhorn_directions = sinkhorn_epsilon = None
    first_log_weights = problem_samples(first, problem)[1]
    second_log_weights = problem_samples(second, problem)[1]
    if "kde" in families:
        with errors.capture_warnings() as records["factor"]:
            factor = float(
                shared_bandwidth_factor(first_log_weights, second_log_weights)
            )
    usable = joint is not None and joint.space is not None
    if usable and {"mmd", "sinkhorn"} & families:
        assert joint is not None
        assert joint.space is not None
        assert joint.first is not None
        assert joint.second is not None
        with errors.capture_warnings() as records["bandwidth"]:
            value = float(
                median_bandwidth(
                    joint.first,
                    first_log_weights,
                    joint.second,
                    second_log_weights,
                    joint.space,
                    key=key("bandwidth"),
                )
            )
            bandwidth = value if math.isfinite(value) and value > 0 else None
            if bandwidth is None:
                _warn_validity(
                    f"the median bandwidth is {value}, not positive and finite, so the"
                    " kernel and Sinkhorn rows are NaN"
                )
    if usable and {"sliced_w2", "sliced_w2_gaussian"} & families:
        assert joint is not None
        assert joint.space is not None
        with errors.capture_warnings() as records["directions"]:
            directions = sliced_directions(
                joint.space, SLICED_DIRECTIONS, key=key("directions")
            )
    if usable and "sliced_sinkhorn" in families:
        assert joint is not None
        assert joint.space is not None
        assert joint.first is not None
        assert joint.second is not None
        with errors.capture_warnings() as records["sliced_sinkhorn"]:
            sinkhorn_directions = sliced_directions(
                joint.space,
                SLICED_SINKHORN_DIRECTIONS,
                key=key("sliced_sinkhorn", "directions"),
            )
            value = float(
                sliced_sinkhorn_epsilon(
                    joint.first,
                    first_log_weights,
                    joint.second,
                    second_log_weights,
                    joint.space,
                    directions=sinkhorn_directions,
                    key=key("sliced_sinkhorn", "epsilon"),
                )
            )
            sinkhorn_epsilon = value if math.isfinite(value) and value > 0 else None
            if sinkhorn_epsilon is None:
                _warn_validity(
                    f"the regularisation of the sliced Sinkhorn divergence is {value},"
                    " not positive and finite, so its row is NaN"
                )
    computed = _Computed(
        factor, bandwidth, directions, sinkhorn_directions, sinkhorn_epsilon
    )
    return computed, records


def _shared_plan(
    arguments: CompareArguments,
    problem: int,
    first: Posterior,
    second: Posterior,
    key: Callable[..., PRNGKeyArray],
) -> tuple[NullPlan, list[WarningRecord]]:
    """Return the null plan on the unwhitened samples, with the warnings it raised."""
    first_n_eff = float(first.n_eff[problem])
    second_n_eff = float(second.n_eff[problem])
    first_samples, first_log_weights = problem_samples(first, problem)
    second_samples, second_log_weights = problem_samples(second, problem)
    with errors.capture_warnings() as records:
        if arguments.construction is Construction.SPLIT_HALF:
            plan = split_half_null(
                first_samples,
                first_log_weights,
                approximation_n_eff=second_n_eff,
                key=key("null"),
                reference_n_eff=first_n_eff,
                draws=arguments.draws,
            )
        else:
            plan = permutation_null(
                first_samples,
                first_log_weights,
                second_samples,
                second_log_weights,
                key=key("null"),
                first_n_eff=first_n_eff,
                second_n_eff=second_n_eff,
                draws=arguments.draws,
            )
    return plan, records


def _prepare_set(
    posterior: Posterior,
    samples: Array,
    problem: int,
    key: Callable[..., PRNGKeyArray],
) -> tuple[Array, float]:
    """Convert a whitened set to equal weights, and give its effective size."""
    log_weights = problem_samples(posterior, problem)[1]
    kept = rejection_subset(
        samples, log_weights, key=key("knn", "prepare", posterior.name)
    )
    efficiency = float(relative_efficiency(log_weights, posterior.n_eff[problem]))
    count = float(kept.shape[0])
    return kept, min(max(count * efficiency, 1.0), count)


def _cut_set(
    name: str, kept: Array, size: float, count: int, key: Callable[..., PRNGKeyArray]
) -> tuple[Array, float]:
    """Keep the first ``count`` positions of a random permutation of a prepared set."""
    order = jax.random.permutation(key("knn", "equal_counts", name), kept.shape[0])
    scaled = size * count / kept.shape[0]
    return kept[order[:count]], min(max(scaled, 1.0), float(count))


def _prepare_knn(
    arguments: CompareArguments,
    layout: _Layout,
    joint: _Whitened,
    first: Posterior,
    second: Posterior,
    problem: int,
    key: Callable[..., PRNGKeyArray],
    recipes: Sequence[str],
) -> _Knn:
    """Prepare the sets, the null and the KL calibration of the kNN rows."""
    assert joint.first is not None
    assert joint.second is not None
    cap = arguments.cap
    with errors.capture_warnings() as records:
        kept_first, size_first = _prepare_set(first, joint.first, problem, key)
        kept_second, size_second = _prepare_set(second, joint.second, problem, key)
        counts = (kept_first.shape[0], kept_second.shape[0])
        if arguments.construction is Construction.SPLIT_HALF:
            targets = (min(counts[0], cap), min(counts[1], cap))
            cut_by_cap = counts[0] > cap or counts[1] > cap
        else:
            targets = (min(*counts, cap),) * 2
            cut_by_cap = cap < min(counts)
        if cut_by_cap:
            _warn_subsampling(
                f"kNN rows: after rejection {first.name!r} keeps {counts[0]} samples"
                f" and {second.name!r} {counts[1]}; the cap of {cap} cuts them to"
                f" {targets[0]} and {targets[1]}"
            )
        cut_first, n_first = _cut_set(
            first.name, kept_first, size_first, targets[0], key
        )
        cut_second, n_second = _cut_set(
            second.name, kept_second, size_second, targets[1], key
        )
        if arguments.construction is Construction.SPLIT_HALF:
            plan = split_half_null(
                cut_first,
                None,
                approximation_n_eff=n_second,
                key=key("knn", "null"),
                reference_n_eff=n_first,
                draws=arguments.draws,
            )
        else:
            plan = permutation_null(
                cut_first,
                None,
                cut_second,
                None,
                key=key("knn", "null"),
                first_n_eff=n_first,
                second_n_eff=n_second,
                draws=arguments.draws,
            )
    curves = None
    with errors.capture_warnings() as curve_records:
        if "kl:knn" in recipes:
            curves = (
                knn_kl_curve(
                    layout.dimension,
                    targets[0],
                    targets[1],
                    key=key("kl", "knn", first.name, second.name, "calibration"),
                    neighbours=KL_NEIGHBOURS,
                    cap=cap,
                ),
                knn_kl_curve(
                    layout.dimension,
                    targets[1],
                    targets[0],
                    key=key("kl", "knn", second.name, first.name, "calibration"),
                    neighbours=KL_NEIGHBOURS,
                    cap=cap,
                ),
            )
    sets = _Sets(
        cut_first,
        jnp.zeros(targets[0]),
        cut_second,
        jnp.zeros(targets[1]),
        n_first,
        n_second,
    )
    return _Knn(sets, plan, curves, records, curve_records)


def _evaluate(
    statistic: Statistic,
    sets: _Sets,
    plan: NullPlan,
    arguments: CompareArguments,
    half_key: PRNGKeyArray,
) -> _Evaluation:
    """Evaluate a statistic on the observed sets, the null plan and the half-samples."""
    with errors.capture_warnings() as records:
        observed = statistic(
            sets.first, sets.first_log_weights, sets.second, sets.second_log_weights
        )
        draws = null_draws(statistic, plan)
        half_errors = None
        if arguments.uncertainty is UncertaintyMethod.SUBSAMPLING_BOOTSTRAP:
            half_values = half_sample_draws(
                statistic,
                sets.first,
                sets.first_log_weights,
                sets.second,
                sets.second_log_weights,
                key=half_key,
                pairs=arguments.half_samples,
            )
            half_errors = half_sample_error(
                half_values,
                sets.first_log_weights,
                sets.second_log_weights,
                first_n_eff=sets.first_n_eff,
                second_n_eff=sets.second_n_eff,
            )
    return _Evaluation(observed, draws, half_errors, records)


# * The rows


def _logarithm(values: Array) -> Array:
    """Return the logarithm of positive values, and NaN of the others."""
    positive = values > 0
    return jnp.where(positive, jnp.log(jnp.where(positive, values, 1.0)), _NAN)


class _RecipeContext(NamedTuple):
    """What the rows of one recipe share."""

    whitened: _Whitened
    head: list[WarningRecord]
    evaluation: _Evaluation | None
    plan: NullPlan | None


def _comparison_rows(
    arguments: CompareArguments,
    info: _PairInfo,
    rows: Sequence[_Row],
    context: _RecipeContext,
) -> list[dict[str, Any]]:
    """Return the result rows of one recipe."""
    result = []
    for row in rows:
        names = info.names[::-1] if row.reverse else info.names
        common: dict[str, Any] = {
            "problem": info.problem,
            "pair": list(names),
            "metric": row.metric,
            "estimator": row.estimator,
            "estimator_settings": row.settings,
            "space": context.whitened.text,
            "parameters": list(row.parameters),
            "threshold": float(arguments.thresholds[row.metric]),
            "provenance": info.reverse_provenance if row.reverse else info.provenance,
        }
        if context.evaluation is None or context.plan is None:
            result.append(_unavailable_row(common, row, context))
        else:
            result.append(
                _evaluated_row(arguments, common, row, context.evaluation, context)
            )
    return result


def _unavailable_row(
    common: dict[str, Any], row: _Row, context: _RecipeContext
) -> dict[str, Any]:
    """Return a row that no null ran for: its value and effect size are NaN."""
    effect = {"r_eff": _NAN} if row.metric in R_EFF_METRICS else {"sigma_eff": _NAN}
    return (
        common
        | effect
        | {
            "value": _NAN,
            "p_value": _NAN,
            "sigma_sig": _NAN,
            "flag": False,
            "warnings": list(context.head),
        }
    )


def _evaluated_row(
    arguments: CompareArguments,
    common: dict[str, Any],
    row: _Row,
    evaluation: _Evaluation,
    context: _RecipeContext,
) -> dict[str, Any]:
    """Return a row with its null, significance, effect size and verdict."""
    assert context.plan is not None
    metric = row.metric
    value = float(evaluation.observed[row.position])
    draws = evaluation.draws[row.position]
    tail = comparison_tail(metric)
    null_mean, null_std = null_moments(draws)
    # * The width ratio is compared on the log scale, where its null is symmetric.
    scaled_value, scaled_draws = jnp.asarray(value), draws
    if metric == "width_ratio":
        scaled_value, scaled_draws = _logarithm(scaled_value), _logarithm(draws)
    with errors.capture_warnings() as significance_records:
        significance = null_significance(scaled_value, scaled_draws, tail)
    sigma_sig = float(significance.sigma_sig)
    subtracted = null_subtract(metric, value, draws)
    sigma_value = r_value = None
    if metric in R_EFF_METRICS:
        log_det = (
            None
            if row.log_det_position is None
            else evaluation.observed[row.log_det_position]
        )
        # * Only the W2 shape term depends on the frame, so it needs the whitening.
        r_value = float(
            r_eff(
                metric,
                subtracted,
                log_det,
                pooled=arguments.source is WhiteningSource.POOLED,
            )
        )
    else:
        sigma_value = float(
            sigma_eff(
                metric,
                subtracted,
                estimator=row.estimator,
                bandwidth=row.bandwidth,
                dimension=row.dimension,
                curve=row.curve,
            )
        )
    outcome = verdict(
        metric,
        sigma_eff=sigma_value,
        r_eff=r_value,
        subtracted=subtracted,
        sigma_sig=sigma_sig,
        thresholds=arguments.thresholds,
        level=arguments.level,
    )
    with errors.capture_warnings() as chordal_records:
        warn_wide_chordal_axes(
            metric, context.whitened.built.whitening, arguments.space
        )
    if evaluation.half_errors is None:
        uncertainty, method = null_std, UncertaintyMethod.NULL_STD
    else:
        uncertainty = float(evaluation.half_errors[row.position])
        method = UncertaintyMethod.SUBSAMPLING_BOOTSTRAP
    return common | {
        "value": value,
        "uncertainty": uncertainty,
        "uncertainty_method": method,
        "null_mean": null_mean,
        "null_std": null_std,
        "null_draws": draws.tolist(),
        "null_recipe": context.plan.recipe(tail=tail),
        "p_value": float(significance.p_value),
        "sigma_sig": sigma_sig,
        "sigma_eff": sigma_value,
        "r_eff": r_value,
        "flag": outcome.failed,
        "warnings": [
            *context.head,
            *evaluation.records,
            *chordal_records,
            *significance_records,
        ],
    }


# * One comparison: a pair of posteriors of one problem


def _usable(family: str, whitened: _Whitened | None, computed: _Computed) -> bool:
    """Return whether a family can run: its whitening and settings are defined."""
    if whitened is None or whitened.space is None:
        return False
    needed = {
        "mmd": computed.bandwidth,
        "sinkhorn": computed.bandwidth,
        "sliced_w2": computed.directions,
        "sliced_w2_gaussian": computed.directions,
        "sliced_sinkhorn": computed.sinkhorn_epsilon,
    }
    return family not in needed or needed[family] is not None


def _compare_pair(
    arguments: CompareArguments,
    layout: _Layout,
    problem: int,
    first: Posterior,
    second: Posterior,
    built: _PairWhitenings,
) -> list[dict[str, Any]]:
    """Return the result rows of one pair in one problem, recipe after recipe."""
    key = partial(
        _path_key, arguments.root, ("compare", problem, first.name, second.name)
    )
    hashes = [first.hashes[problem], second.hashes[problem]]
    info = _PairInfo(
        problem,
        (first.name, second.name),
        provenance(arguments.seed, arguments.root, hashes, arguments.config_hash),
        provenance(arguments.seed, arguments.root, hashes[::-1], arguments.config_hash),
    )
    first_kept = problem_samples(first, problem)
    second_kept = problem_samples(second, problem)
    samples = (first_kept[0], second_kept[0])
    joint = _whiten_pair(
        built.joint, arguments.space, *samples, arguments.transform_text
    )
    parameter = _whiten_pair(
        built.parameter, flat_space(arguments.space), *samples, arguments.transform_text
    )
    recipes_of = {
        family: tuple(recipe for recipe in members if recipe in arguments.recipes)
        for family, members in FAMILIES.items()
    }
    families = [family for family, recipes in recipes_of.items() if recipes]
    computed, setting_records = _compute_settings(
        set(families), joint, first, second, problem, key
    )
    whitened_of = {
        family: joint
        if RECIPES[recipes_of[family][0]].scope is Scope.JOINT
        else parameter
        for family in families
    }
    usable = {
        family: _usable(family, whitened_of[family], computed) for family in families
    }
    shared_plan: NullPlan | None = None
    plan_records: list[WarningRecord] = []
    if any(usable[family] for family in families if family != "knn"):
        shared_plan, plan_records = _shared_plan(arguments, problem, first, second, key)
    knn = None
    if usable.get("knn"):
        assert joint is not None
        knn = _prepare_knn(
            arguments, layout, joint, first, second, problem, key, recipes_of["knn"]
        )
        computed = computed._replace(curves=knn.curves)
    rows_by_recipe: dict[str, list[dict[str, Any]]] = {}
    for family in families:
        whitened = whitened_of[family]
        assert whitened is not None
        recipes = recipes_of[family]
        layouts = _family_rows(family, recipes, layout, computed)
        head = [
            *whitened.built.records,
            *setting_records.get(SETTINGS_OF_FAMILY.get(family, ""), []),
        ]
        evaluation = plan = None
        extra: dict[str, list[WarningRecord]] = {}
        if usable[family]:
            assert whitened.space is not None
            statistic = _family_statistic(
                family,
                recipes,
                space=whitened.space,
                computed=computed,
                layout=layout,
                key=key,
            )
            if family == "knn":
                assert knn is not None
                sets, plan, half_key = knn.sets, knn.plan, key("knn", "half_samples")
                head += knn.records
                extra = {"kl:knn": knn.curve_records}
            else:
                assert shared_plan is not None
                assert whitened.first is not None
                assert whitened.second is not None
                sets = _Sets(
                    whitened.first,
                    first_kept[1],
                    whitened.second,
                    second_kept[1],
                    float(first.n_eff[problem]),
                    float(second.n_eff[problem]),
                )
                plan = shared_plan._replace(
                    samples=whitened.built.whitening.forward(shared_plan.samples)
                )
                head += plan_records
                half_key = key("half_samples")
            evaluation = _evaluate(statistic, sets, plan, arguments, half_key)
        for recipe in recipes:
            context = _RecipeContext(
                whitened, [*head, *extra.get(recipe, [])], evaluation, plan
            )
            rows_by_recipe[recipe] = _comparison_rows(
                arguments, info, layouts[recipe], context
            )
    return [row for recipe in arguments.recipes for row in rows_by_recipe[recipe]]
