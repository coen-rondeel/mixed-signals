"""``diagnose``: the health, shape, dependence, modes and railing of posteriors.

:func:`diagnose` runs the single-posterior diagnostics of
:mod:`mixed_signals.diagnostics` on MCMC chains or on flat samples, compares
every shape, dependence and mode statistic with a null from Gaussian
surrogates, and returns one :class:`~mixed_signals.core.result.Result` table,
with the reduction of every run of chains. It is a plain function; the package
warnings raised in a family go into the ``warnings`` column of its rows.

Inputs
    A source is a :class:`~mixed_signals.core.chains.Chains` object, one run
    with one posterior, or a :class:`~mixed_signals.core.samples.SampleSet`
    with K posteriors and B problems. Chains count as one problem, so they
    mix only with sets of one problem. Each run of chains is reduced by
    :func:`~mixed_signals.mcmc.reduce.reduce_chains` after its burn-in and
    thinned by its autocorrelation time, and the reduced set, with its
    effective sample size, feeds every family but the chain rows.
Families
    ``chains`` (the rows of the chains, in the coordinates of the chains),
    ``shape``, ``dependence``, ``modes`` and ``railing``; see
    :data:`DIAGNOSTIC_FAMILIES`. A caller selects by name, and ``chains`` gives
    no rows without a Chains input. A ``transform`` maps every flat or reduced
    set and the space of the other families, whose ``space`` column then reads
    the description of the transform, or ``identity``.
Surrogates
    The null of the shape, dependence and mode rows is the statistic of
    ``draws`` sets drawn from the moment-matched Gaussian of the posterior, a
    parametric bootstrap (Efron and Tibshirani 1993;
    :func:`~mixed_signals.nulls.surrogates.gaussian_surrogates`). Each set
    holds the effective sample size of the posterior, at most ``cap`` samples
    for the shape and mode rows and at most ``pair_cap`` for the dependence
    rows; the observed statistics still use all samples. One draw serves each
    statistic as far as it reads: the first samples of a surrogate set are
    independent, so the first m of them are a set of m.
Weights
    The nearest-neighbour mutual information needs equally weighted samples
    without copies, so a set with unequal positive weights runs its dependence
    rows on its rejection subset
    (:func:`~mixed_signals.core.samples.rejection_subset`; Smith and Gelfand
    1992; Ashton et al. 2019), with surrogates of the effective size of that
    subset, M times the relative efficiency of the input.
Rows
    One row per parameter, per parameter pair, or per joint statistic, with
    the pair column holding the name of the posterior. A row with a threshold
    in ``thresholds`` carries its ``p_value``, ``sigma_sig``, ``threshold`` and
    ``flag`` from :func:`~mixed_signals.diagnostics.flags.evaluate`; rows with
    surrogate draws also carry the null columns, with the construction
    ``gaussian_surrogate``. The informational rows, such as τ and the mode
    count, carry the value only. ``sigma_eff`` and ``r_eff`` stay null.
Keys
    Every random choice derives its key with
    :func:`~mixed_signals.prng.derive_key` from a path that starts with
    ``("diagnose", problem, name)``.

References:
    :cite:t:`efron1993introduction`; :cite:t:`kish1965survey`;
    :cite:t:`smith1992bayesian`; :cite:t:`ashton2019bilby`;
    :cite:t:`phipson2010permutation`; :cite:t:`cowan2011asymptotic`.
"""

import json
import math
import os
import warnings
from collections.abc import Callable, Mapping, Sequence
from functools import partial
from types import MappingProxyType
from typing import Any, Final, NamedTuple

import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray

from mixed_signals import errors
from mixed_signals._config import COST_BUDGET, NULL_DRAWS, SAMPLE_CAP
from mixed_signals.core.chains import Chains
from mixed_signals.core.result import Provenance, Result, UncertaintyMethod
from mixed_signals.core.samples import SampleSet, rejection_subset
from mixed_signals.core.space import ParamSpace
from mixed_signals.diagnostics import flags
from mixed_signals.diagnostics._common import parameter_pairs
from mixed_signals.diagnostics.chains import (
    ChainHealth,
    chain_health,
    split_chain_metric_names,
)
from mixed_signals.diagnostics.dependence import (
    MUTUAL_INFORMATION_NEIGHBOURS,
    PAIR_CAP,
    dependence_statistics,
)
from mixed_signals.diagnostics.flags import Threshold
from mixed_signals.diagnostics.modes import (
    GmmBic,
    circle_centres,
    discriminant_directions,
    gmm_bic,
    marginal_modes,
    projection_modes,
)
from mixed_signals.diagnostics.railing import railed_parameters, railing_scores
from mixed_signals.diagnostics.shape import (
    anderson_darling_normality,
    mardia_moments,
    marginal_moments,
    moment_metric_names,
)
from mixed_signals.engines import _arguments
from mixed_signals.engines._arguments import (
    DiagnoseArguments,
    read_diagnose_arguments,
)
from mixed_signals.engines._common import (
    KeyInput,
    Posterior,
    gather_posteriors,
    null_moments,
    provenance,
    settings_text,
)
from mixed_signals.engines.cost import _diagnose_cost, warn_if_over_budget
from mixed_signals.errors import ValidityWarning, WarningRecord
from mixed_signals.estimators.kde import KernelDensitySettings
from mixed_signals.estimators.knn import WEIGHT_TOLERANCE, NearestNeighbourSettings
from mixed_signals.mcmc.reduce import Reduction, reduce_chains
from mixed_signals.nulls.surrogates import (
    CONSTRUCTION,
    gaussian_surrogates,
    surrogate_size,
)
from mixed_signals.prng import derive_key
from mixed_signals.reweight.importance import relative_efficiency
from mixed_signals.sigma.thresholds import SIGNIFICANCE_LEVEL
from mixed_signals.transforms.base import IDENTITY_DESCRIPTION, Transform

__all__ = ["DIAGNOSTIC_FAMILIES", "Diagnosis", "diagnose"]

DIAGNOSTIC_FAMILIES: Final = _arguments.DIAGNOSTIC_FAMILIES
"""The families of rows that ``diagnose`` gives, in the order of the result rows.

``chains``
    The health of MCMC chains: τ, the effective sample size, the split-chain
    test, R̂ and the rates of the run.
``shape``
    The skewness, kurtosis and Anderson-Darling distance of each parameter,
    and Mardia's moments.
``dependence``
    Pearson's r and the excess of Spearman's ρ_s, the distance correlation and
    the mutual information of each pair of parameters.
``modes``
    GMM-BIC, and the modes of the marginals and of the discriminant
    directions of the chosen mixture.
``railing``
    The railing scores at the finite bounds of interval parameters.
"""

# * Frames in the package are skipped, so a warning points at the caller's line.
_PACKAGE_DIRECTORY: Final = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Diagnosis(NamedTuple):
    """The result table of ``diagnose`` and the reductions of its chains."""

    result: Result
    """One row per diagnostic statistic of every problem and posterior."""
    reductions: Mapping[str, Reduction]
    """The reduction of every Chains input, by name."""


class _ChainEntry(NamedTuple):
    """The health and the reduction of one run of chains, with their warnings."""

    burn_in: int
    health: ChainHealth | None
    health_records: list[WarningRecord]
    reduction: Reduction
    reduction_records: list[WarningRecord]


class _Info(NamedTuple):
    """What every row of one posterior shares."""

    problem: int
    name: str
    space_text: str
    provenance: Provenance


class _Surrogates(NamedTuple):
    """Gaussian surrogate sets of one posterior, and the sizes each statistic reads."""

    plain: Array | None
    truncated: Array | None
    shape_size: int
    pair_size: int
    records: list[WarningRecord]


def diagnose(
    sources: Chains | SampleSet | Sequence[Chains | SampleSet],
    names: Sequence[str],
    space: ParamSpace,
    *,
    key: KeyInput,
    burn_in: int | Sequence[int] = 0,
    transform: Transform | None = None,
    diagnostics: Sequence[str] | None = None,
    draws: int = NULL_DRAWS,
    cap: int = SAMPLE_CAP,
    pair_cap: int = PAIR_CAP,
    thresholds: Mapping[str, Threshold] = flags.THRESHOLDS,
    level: float = SIGNIFICANCE_LEVEL,
    budget: float = COST_BUDGET,
    config_hash: str | None = None,
) -> Diagnosis:
    """Diagnose posteriors one by one, with a surrogate null and a flag per row.

    The module docstring describes the inputs, the families, the surrogates and
    the rows. The package warnings that a family raises, such as a
    :class:`~mixed_signals.errors.CoupledChainsWarning` of the chain rows, are
    not raised but go into the ``warnings`` column of its rows.

    Args:
        sources: One run of :class:`~mixed_signals.core.chains.Chains`, one
            :class:`~mixed_signals.core.samples.SampleSet` shaped (B, K, N,
            D), or a sequence of them. Chains hold one posterior and count as
            one problem.
        names: One unique name per posterior, in the order of the sources.
        space: The parameter space, with D parameters; the space of the chains
            and of the input sets.
        key: A JAX PRNG key, or an integer seed that becomes
            :func:`~mixed_signals.prng.root_key` and fills the provenance seed.
        burn_in: Steps to drop from the start of the chains: one number for
            every Chains input, or one per Chains input in order.
        transform: A transform that maps the flat and the reduced sets to the
            coordinates of the shape, dependence, mode and railing rows;
            ``None`` for the coordinates of the input.
        diagnostics: The families to run, by name; all of
            :data:`DIAGNOSTIC_FAMILIES` when ``None``.
        draws: Number of surrogate sets per posterior.
        cap: Largest number of samples of a surrogate set of the shape and
            mode rows, and of Mardia's observed statistic.
        pair_cap: Largest number of samples of a surrogate set of the
            dependence rows, and of their observed statistic.
        thresholds: The threshold of each flagging metric; the package
            defaults.
        level: The σ_sig that a row with a null must reach to flag.
        budget: The estimated run time in seconds above which the dry run
            raises a :class:`~mixed_signals.errors.CostWarning`; infinity
            switches the warning off.
        config_hash: SHA-256 digest of the pipeline config, for the provenance.

    Returns:
        The result table and the reduction of every Chains input.

    Raises:
        ValueError: If the sources, names, space, burn-in, families, counts or
            thresholds do not fit, or chains are too short for their burn-in.
        TypeError: If a source or a count is of another type than the
            arguments need.
        KeyError: If the thresholds lack a flagging metric of the selected
            rows.

    References:
        :cite:t:`efron1993introduction`; :cite:t:`phipson2010permutation`;
        :cite:t:`cowan2011asymptotic`.
    """
    arguments = read_diagnose_arguments(
        sources,
        names,
        space,
        key=key,
        burn_in=burn_in,
        transform=transform,
        diagnostics=diagnostics,
        draws=draws,
        cap=cap,
        pair_cap=pair_cap,
        thresholds=thresholds,
        level=level,
        budget=budget,
        config_hash=config_hash,
    )
    warn_if_over_budget("diagnose", _diagnose_cost(arguments), arguments.budget)
    entries = _reduce_chains(arguments)
    sets: list[SampleSet] = []
    for source in arguments.sources:
        if source.chains is None:
            assert source.sample_set is not None
            sets.append(source.sample_set)
        else:
            sets.append(entries[source.names[0]].reduction.sample_set)
    posteriors = gather_posteriors(
        sets, arguments.names, arguments.space, transform=arguments.transform
    )
    rows: list[dict[str, Any]] = []
    for problem in range(arguments.batch_count):
        for posterior in posteriors:
            entry = entries.get(posterior.name)
            rows.extend(_posterior_rows(arguments, problem, posterior, entry))
    reductions = MappingProxyType(
        {name: entry.reduction for name, entry in entries.items()}
    )
    return Diagnosis(Result.from_rows(rows), reductions)


# * Chains: their health and their reduction


def _reduce_chains(arguments: DiagnoseArguments) -> dict[str, _ChainEntry]:
    """Compute the health and the reduction of every run of chains, by name."""
    entries: dict[str, _ChainEntry] = {}
    for source in arguments.sources:
        if source.chains is None:
            continue
        health = None
        with errors.capture_warnings() as health_records:
            if "chains" in arguments.families:
                health = chain_health(
                    source.chains, arguments.space, burn_in=source.burn_in
                )
        with errors.capture_warnings() as reduction_records:
            reduction = reduce_chains(source.chains, burn_in=source.burn_in, thin="tau")
        entries[source.names[0]] = _ChainEntry(
            source.burn_in, health, health_records, reduction, reduction_records
        )
    return entries


def _chain_rows(
    arguments: DiagnoseArguments, info: _Info, entry: _ChainEntry
) -> list[dict[str, Any]]:
    """Return the chain rows of one run: per parameter, per temperature and swap."""
    health = entry.health
    assert health is not None
    names = arguments.space.names
    split_names = split_chain_metric_names(arguments.space)
    base: dict[str, object] = {"burn_in": entry.burn_in}
    chain_info = info._replace(space_text=IDENTITY_DESCRIPTION)
    records = entry.health_records
    rows = []
    for position, name in enumerate(names):
        parameter = (name,)
        thin = int(health.split_thin[position])
        for metric, value, settings, p_value in (
            ("tau", health.tau[position], base, None),
            ("ess", health.effective_sample_size[position], base, None),
            (
                split_names[position],
                health.split_statistic[position],
                {**base, "thin": thin},
                health.split_p_value[position],
            ),
            ("rhat", health.rhat[position], base, None),
        ):
            rows.append(
                _row(
                    arguments,
                    chain_info,
                    metric,
                    "empirical",
                    settings_text(settings),
                    parameter,
                    float(value),
                    records,
                    p_value=None if p_value is None else float(p_value),
                )
            )
    if health.acceptance_rate is not None:
        for temperature, rate in enumerate(health.acceptance_rate):
            settings = {**base, "temperature": temperature}
            rows.append(
                _row(
                    arguments,
                    chain_info,
                    "acceptance_rate",
                    "empirical",
                    settings_text(settings),
                    names,
                    float(rate),
                    records,
                )
            )
    if health.swap_rate is not None:
        for temperature, rate in enumerate(health.swap_rate):
            settings = {**base, "temperatures": [temperature, temperature + 1]}
            rows.append(
                _row(
                    arguments,
                    chain_info,
                    "swap_rate",
                    "empirical",
                    settings_text(settings),
                    names,
                    float(rate),
                    records,
                )
            )
    return rows


# * The statistics of the families, mapped over surrogate sets with jax.lax.map


def _shape_values(
    samples: Array,
    log_weights: Array | None,
    *,
    space: ParamSpace,
    key: PRNGKeyArray,
    cap: int,
) -> Array:
    """Return skewness, kurtosis and Anderson-Darling of each parameter, then Mardia."""
    moments = marginal_moments(samples, log_weights, space)
    distances = anderson_darling_normality(samples, log_weights, space)
    mardia = mardia_moments(samples, log_weights, space, key=key, cap=cap)
    return jnp.concatenate(
        [
            moments.skewness,
            moments.kurtosis,
            distances,
            jnp.stack([mardia.skewness, mardia.kurtosis]),
        ]
    )


def _dependence_values(
    samples: Array,
    log_weights: Array | None,
    *,
    space: ParamSpace,
    key: PRNGKeyArray,
    cap: int,
) -> Array:
    """Return the four dependence statistics of every pair, shaped (4, pairs)."""
    result = dependence_statistics(samples, log_weights, space, key=key, cap=cap)
    return jnp.stack(
        [
            result.pearson,
            result.spearman,
            result.distance_correlation,
            result.mutual_information,
        ]
    )


def _marginal_mode_values(
    samples: Array, log_weights: Array | None, *, space: ParamSpace
) -> Array:
    """Return the mode count and the dip of each parameter, shaped (2, D)."""
    counts = marginal_modes(samples, log_weights, space)
    return jnp.stack([counts.modes, counts.dip])


def _projection_values(
    samples: Array,
    log_weights: Array | None,
    *,
    space: ParamSpace,
    directions: Array,
    centres: Array,
) -> Array:
    """Return the mode count and the dip along each direction, shaped (2, P)."""
    counts = projection_modes(
        samples, log_weights, space, directions=directions, centres=centres
    )
    return jnp.stack([counts.modes, counts.dip])


def _map_over_sets(statistic: Callable[[Array], Array], sets: Array) -> Array:
    """Evaluate a statistic on every surrogate set, the set axis first."""
    values: Array = jax.lax.map(statistic, sets)
    return values


# * The rows of one posterior


def _path_key(
    root: PRNGKeyArray, path: tuple[str | int, ...], *tail: str | int
) -> PRNGKeyArray:
    """Return the key of a path below the posterior path."""
    return derive_key(root, *path, *tail)


def _warn_validity(message: str) -> None:
    """Raise a validity warning that points at the caller's line."""
    warnings.warn(ValidityWarning(message), skip_file_prefixes=(_PACKAGE_DIRECTORY,))


def _row(
    arguments: DiagnoseArguments,
    info: _Info,
    metric: str,
    estimator: str,
    settings: str,
    parameters: Sequence[str],
    value: float,
    records: Sequence[WarningRecord],
    *,
    draws: Array | None = None,
    size: int | None = None,
    p_value: float | None = None,
) -> dict[str, Any]:
    """Return one result row, with its null, p-value and flag where it has them."""
    row: dict[str, Any] = {
        "problem": info.problem,
        "pair": [info.name],
        "metric": metric,
        "estimator": estimator,
        "estimator_settings": settings,
        "space": info.space_text,
        "parameters": list(parameters),
        "value": value,
        "provenance": info.provenance,
    }
    threshold = arguments.thresholds.get(metric)
    if draws is not None:
        assert size is not None
        null_mean, null_std = null_moments(draws)
        row |= {
            "uncertainty": null_std,
            "uncertainty_method": UncertaintyMethod.NULL_STD,
            "null_mean": null_mean,
            "null_std": null_std,
            "null_draws": draws.tolist(),
            "null_recipe": {
                "construction": str(CONSTRUCTION),
                "draws": int(draws.shape[0]),
                "sample_size": size,
                "tail": None
                if threshold is None or threshold.tail is None
                else str(threshold.tail),
            },
        }
    extra: list[WarningRecord] = []
    if threshold is not None:
        with errors.capture_warnings() as extra:
            evaluation = flags.evaluate(
                metric,
                value,
                draws,
                p_value=p_value,
                thresholds=arguments.thresholds,
                level=arguments.level,
            )
        if draws is not None or p_value is not None:
            row["p_value"] = evaluation.p_value
            row["sigma_sig"] = evaluation.sigma_sig
        row["threshold"] = float(threshold.value)
        row["flag"] = evaluation.flagged
    row["warnings"] = [*records, *extra]
    return row


def _posterior_rows(
    arguments: DiagnoseArguments,
    problem: int,
    posterior: Posterior,
    entry: _ChainEntry | None,
) -> list[dict[str, Any]]:
    """Return the rows of every selected family for one posterior of one problem."""
    path = ("diagnose", problem, posterior.name)
    key = partial(_path_key, arguments.root, path)
    chains_hash = None if entry is None else entry.reduction.record.chains_hash
    hashes = [chains_hash] if chains_hash is not None else [posterior.hashes[problem]]
    transform = arguments.transform
    info = _Info(
        problem,
        posterior.name,
        IDENTITY_DESCRIPTION if transform is None else transform.description,
        provenance(arguments.seed, arguments.root, hashes, arguments.config_hash),
    )
    samples = posterior.samples[problem]
    log_weights = posterior.log_weights[problem]
    n_eff = float(posterior.n_eff[problem])
    families = arguments.families
    rows: list[dict[str, Any]] = []
    if "chains" in families and entry is not None:
        rows.extend(_chain_rows(arguments, info, entry))
    reduction_records = [] if entry is None else entry.reduction_records
    pair_input = None
    if "dependence" in families and len(arguments.row_space) >= 2:
        pair_input = _dependence_input(samples, log_weights, n_eff, key)
    surrogates = _surrogates(
        arguments, families, samples, log_weights, n_eff, pair_input, key
    )
    head = [*reduction_records, *surrogates.records]
    if "shape" in families:
        rows.extend(
            _shape_rows(arguments, info, samples, log_weights, surrogates, head, key)
        )
    if pair_input is not None:
        rows.extend(
            _dependence_rows(arguments, info, pair_input, surrogates, head, key)
        )
    if "modes" in families:
        rows.extend(
            _mode_rows(
                arguments, info, samples, log_weights, n_eff, surrogates, head, key
            )
        )
    if "railing" in families:
        rows.extend(
            _railing_rows(arguments, info, samples, log_weights, reduction_records)
        )
    return rows


class _PairInput(NamedTuple):
    """The samples that the dependence rows read, and the size of their surrogates."""

    samples: Array
    log_weights: Array | None
    size: float
    records: list[WarningRecord]


def _dependence_input(
    samples: Array,
    log_weights: Array,
    n_eff: float,
    key: Callable[..., PRNGKeyArray],
) -> _PairInput:
    """Return the samples of the dependence rows: the set, or its rejection subset."""
    present = log_weights[log_weights > -jnp.inf]
    equal = float(jnp.max(present) - jnp.min(present)) <= WEIGHT_TOLERANCE
    if equal:
        return _PairInput(samples, log_weights, n_eff, [])
    with errors.capture_warnings() as records:
        subset = rejection_subset(
            samples, log_weights, key=key("dependence", "prepare")
        )
    efficiency = float(relative_efficiency(log_weights, n_eff))
    count = float(subset.shape[0])
    return _PairInput(subset, None, min(max(count * efficiency, 1.0), count), records)


def _surrogates(
    arguments: DiagnoseArguments,
    families: Sequence[str],
    samples: Array,
    log_weights: Array,
    n_eff: float,
    pair_input: _PairInput | None,
    key: Callable[..., PRNGKeyArray],
) -> _Surrogates:
    """Draw the surrogate sets that the selected families read."""
    shape_size = surrogate_size(n_eff, arguments.cap)
    pair_size = (
        0 if pair_input is None else surrogate_size(pair_input.size, arguments.pair_cap)
    )
    needs_plain = bool({"shape", "dependence", "modes"} & set(families))
    plain = truncated = None
    with errors.capture_warnings() as records:
        if needs_plain:
            plain = gaussian_surrogates(
                samples,
                log_weights,
                arguments.row_space,
                size=max(shape_size, pair_size),
                key=key("surrogates"),
                draws=arguments.draws,
            )
        if "modes" in families:
            truncated = gaussian_surrogates(
                samples,
                log_weights,
                arguments.row_space,
                size=shape_size,
                key=key("surrogates"),
                draws=arguments.draws,
                truncate=True,
            )
    return _Surrogates(plain, truncated, shape_size, pair_size, records)


def _shape_rows(
    arguments: DiagnoseArguments,
    info: _Info,
    samples: Array,
    log_weights: Array,
    surrogates: _Surrogates,
    head: list[WarningRecord],
    key: Callable[..., PRNGKeyArray],
) -> list[dict[str, Any]]:
    """Return the skewness, kurtosis, Anderson-Darling and Mardia rows."""
    space = arguments.row_space
    mardia_key = key("mardia", "subsample")
    assert surrogates.plain is not None
    size = surrogates.shape_size
    statistic = partial(
        _shape_values, log_weights=None, space=space, key=mardia_key, cap=arguments.cap
    )
    with errors.capture_warnings() as records:
        observed = _shape_values(
            samples, log_weights, space=space, key=mardia_key, cap=arguments.cap
        )
        null = _map_over_sets(statistic, surrogates.plain[:, :size])
    count = len(space)
    rows = []
    for position, (skewness, kurtosis) in enumerate(moment_metric_names(space)):
        for metric, offset in (
            (skewness, 0),
            (kurtosis, count),
            ("anderson_darling", 2 * count),
        ):
            index = offset + position
            rows.append(
                _row(
                    arguments,
                    info,
                    metric,
                    "empirical",
                    "{}",
                    (space.names[position],),
                    float(observed[index]),
                    [*head, *records],
                    draws=null[:, index],
                    size=size,
                )
            )
    for offset, metric in enumerate(("mardia_skewness", "mardia_kurtosis")):
        index = 3 * count + offset
        rows.append(
            _row(
                arguments,
                info,
                metric,
                "empirical",
                settings_text({"cap": arguments.cap}),
                space.names,
                float(observed[index]),
                [*head, *records],
                draws=null[:, index],
                size=size,
            )
        )
    return rows


def _dependence_rows(
    arguments: DiagnoseArguments,
    info: _Info,
    pair_input: _PairInput,
    surrogates: _Surrogates,
    head: list[WarningRecord],
    key: Callable[..., PRNGKeyArray],
) -> list[dict[str, Any]]:
    """Return Pearson, Spearman, distance correlation and MI rows of each pair."""
    space = arguments.row_space
    subsample_key = key("dependence", "subsample")
    assert surrogates.plain is not None
    size = surrogates.pair_size
    statistic = partial(
        _dependence_values,
        log_weights=None,
        space=space,
        key=subsample_key,
        cap=arguments.pair_cap,
    )
    with errors.capture_warnings() as records:
        observed = _dependence_values(
            pair_input.samples,
            pair_input.log_weights,
            space=space,
            key=subsample_key,
            cap=arguments.pair_cap,
        )
        null = _map_over_sets(statistic, surrogates.plain[:, :size])
    neighbours = json.loads(
        NearestNeighbourSettings(
            neighbours=MUTUAL_INFORMATION_NEIGHBOURS, norm="max"
        ).to_json()
    )
    settings = {
        "pearson": ("empirical", settings_text({"cap": arguments.pair_cap})),
        "spearman": ("empirical", settings_text({"cap": arguments.pair_cap})),
        "distance_correlation": (
            "empirical",
            settings_text({"cap": arguments.pair_cap}),
        ),
        "mutual_information": (
            "knn",
            settings_text({**neighbours, "cap": arguments.pair_cap}),
        ),
    }
    rows = []
    for position, (first, second) in enumerate(parameter_pairs(len(space))):
        parameters = (space.names[first], space.names[second])
        for statistic_index, (metric, (estimator, text)) in enumerate(settings.items()):
            informational = metric == "pearson"
            rows.append(
                _row(
                    arguments,
                    info,
                    metric,
                    estimator,
                    text,
                    parameters,
                    float(observed[statistic_index, position]),
                    [*head, *pair_input.records, *records],
                    draws=None if informational else null[:, statistic_index, position],
                    size=None if informational else size,
                )
            )
    return rows


def _finite_or_none(values: Sequence[float]) -> list[float | None]:
    """Return numbers for JSON, with null for the ones that are not finite."""
    return [value if math.isfinite(value) else None for value in values]


def _mode_rows(
    arguments: DiagnoseArguments,
    info: _Info,
    samples: Array,
    log_weights: Array,
    n_eff: float,
    surrogates: _Surrogates,
    head: list[WarningRecord],
    key: Callable[..., PRNGKeyArray],
) -> list[dict[str, Any]]:
    """Return the GMM-BIC row and the mode rows of the marginals and projections."""
    space = arguments.row_space
    size = surrogates.shape_size
    assert surrogates.truncated is not None
    assert surrogates.plain is not None
    rows: list[dict[str, Any]] = []
    structure: GmmBic | None = None
    with errors.capture_warnings() as records:
        if n_eff >= 2:
            structure = gmm_bic(
                samples,
                log_weights,
                space,
                effective_sample_size=n_eff,
                key=key("gmm"),
            )
        else:
            _warn_validity(
                f"the effective sample size {n_eff:.3g} is below 2, so BIC is NaN"
            )
        marginal = _marginal_mode_values(samples, log_weights, space=space)
        marginal_null = _map_over_sets(
            partial(_marginal_mode_values, log_weights=None, space=space),
            surrogates.truncated[:, :size],
        )
    gmm_settings: dict[str, object] = {"bic": [], "components": None}
    delta = math.nan
    if structure is not None:
        gmm_settings = {
            "bic": _finite_or_none(structure.bic),
            "components": structure.components,
        }
        delta = structure.delta_bic
    rows.append(
        _row(
            arguments,
            info,
            "gmm_bic",
            "gmm",
            settings_text(gmm_settings),
            space.names,
            delta,
            [*head, *records],
        )
    )
    kde_settings = KernelDensitySettings().to_json()
    for position, name in enumerate(space.names):
        rows.append(
            _row(
                arguments,
                info,
                "mode_count",
                "kde",
                kde_settings,
                (name,),
                float(marginal[0, position]),
                [*head, *records],
            )
        )
        rows.append(
            _row(
                arguments,
                info,
                "mode_dip",
                "kde",
                kde_settings,
                (name,),
                float(marginal[1, position]),
                [*head, *records],
                draws=marginal_null[:, 1, position],
                size=size,
            )
        )
    if structure is not None and structure.components >= 2:
        rows.extend(
            _projection_rows(
                arguments, info, samples, log_weights, structure, surrogates, head
            )
        )
    return rows


def _projection_rows(
    arguments: DiagnoseArguments,
    info: _Info,
    samples: Array,
    log_weights: Array,
    structure: GmmBic,
    surrogates: _Surrogates,
    head: list[WarningRecord],
) -> list[dict[str, Any]]:
    """Return the mode rows along the discriminant directions of the mixture."""
    space = arguments.row_space
    size = surrogates.shape_size
    assert surrogates.plain is not None
    with errors.capture_warnings() as records:
        directions = discriminant_directions(structure.mixture)
        centres = circle_centres(samples, log_weights, space)
        observed = _projection_values(
            samples, log_weights, space=space, directions=directions, centres=centres
        )
        null = _map_over_sets(
            partial(
                _projection_values,
                log_weights=None,
                space=space,
                directions=directions,
                centres=centres,
            ),
            surrogates.plain[:, :size],
        )
    rows = []
    components = parameter_pairs(structure.components)
    for position, (first, second) in enumerate(components):
        settings = settings_text(
            {
                "bandwidth": "silverman",
                "components": [first, second],
                "direction": _finite_or_none(directions[position].tolist()),
            }
        )
        for index, metric in enumerate(("mode_count", "mode_dip")):
            rows.append(
                _row(
                    arguments,
                    info,
                    metric,
                    "kde",
                    settings,
                    space.names,
                    float(observed[index, position]),
                    [*head, *records],
                    draws=None if index == 0 else null[:, 1, position],
                    size=None if index == 0 else size,
                )
            )
    return rows


def _railing_rows(
    arguments: DiagnoseArguments,
    info: _Info,
    samples: Array,
    log_weights: Array,
    head: list[WarningRecord],
) -> list[dict[str, Any]]:
    """Return the railing rows at the finite bounds of the interval parameters."""
    space = arguments.row_space
    positions = railed_parameters(space)
    if not positions:
        return []
    with errors.capture_warnings() as records:
        scores = railing_scores(samples, log_weights, space)
    settings = KernelDensitySettings().to_json()
    rows = []
    for position in positions:
        for metric, values in (
            ("railing_lower", scores.lower),
            ("railing_upper", scores.upper),
        ):
            value = float(values[position])
            if math.isnan(value):
                continue
            rows.append(
                _row(
                    arguments,
                    info,
                    metric,
                    "kde",
                    settings,
                    (space.names[position],),
                    value,
                    [*head, *records],
                )
            )
    return rows
