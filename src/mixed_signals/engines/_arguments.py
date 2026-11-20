"""Argument reading of the engines ``compare`` and ``diagnose``, and the recipe table.

The engines and their dry-run estimates read and check the same arguments, so
the code lives here: the recipes of ``compare`` with their families and
settings (:data:`RECIPES`, :data:`DEFAULT_METRICS`), the checks of the
arguments of ``compare`` (:func:`read_compare_arguments`) and of ``diagnose``
(:func:`read_diagnose_arguments`, with :data:`DIAGNOSTIC_FAMILIES`), and the
shared checks of counts and numbers. The module imports no engine, so that the
dry-run estimates import it without a cycle. It is private: its names may
change with the engines, and the public constants are re-exported by the
engine modules.
"""

import math
from collections.abc import Mapping, Sequence
from enum import StrEnum
from types import MappingProxyType
from typing import Final, NamedTuple

import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray

from mixed_signals.core.chains import Chains
from mixed_signals.core.result import UncertaintyMethod
from mixed_signals.core.samples import SampleSet
from mixed_signals.core.space import ParamSpace
from mixed_signals.diagnostics.chains import split_chain_metric_names
from mixed_signals.diagnostics.flags import THRESHOLDS as DIAGNOSTIC_THRESHOLDS
from mixed_signals.diagnostics.flags import Threshold
from mixed_signals.diagnostics.railing import railed_parameters
from mixed_signals.diagnostics.shape import moment_metric_names
from mixed_signals.engines._common import (
    Posterior,
    gather_posteriors,
    read_key,
    unique_names,
)
from mixed_signals.metrics.classifier import _DEFERRED
from mixed_signals.metrics.marginal import ks_metric_names
from mixed_signals.nulls.plan import Construction
from mixed_signals.transforms.base import Transform
from mixed_signals.transforms.whiten import WhiteningSource


class Scope(StrEnum):
    """Which whitening and which rows a recipe has."""

    PARAMETER = "parameter"
    """One row per parameter, in the z-scored coordinates of each parameter."""
    JOINT = "joint"
    """Rows of the whole parameter vector, in the decorrelated coordinates."""


class Recipe(NamedTuple):
    """A metric with the estimator that computes it, and the rows' scope."""

    metric: str
    """Value of the ``metric`` column; ``ks`` also stands for ``kuiper_ks``."""
    estimator: str
    """Value of the ``estimator`` column."""
    scope: Scope
    """Whether the rows are per parameter or joint."""


_RECIPE_TABLE: Final = (
    ("mean_shift", "empirical", Scope.PARAMETER),
    ("width_ratio", "empirical", Scope.PARAMETER),
    ("ks", "empirical", Scope.PARAMETER),
    ("ad", "empirical", Scope.PARAMETER),
    ("w1", "empirical", Scope.PARAMETER),
    ("js", "kde", Scope.PARAMETER),
    ("hellinger", "kde", Scope.PARAMETER),
    ("kl", "gaussian", Scope.JOINT),
    ("w2", "gaussian", Scope.JOINT),
    ("hellinger", "gaussian", Scope.JOINT),
    ("kl", "knn", Scope.JOINT),
    ("js", "knn", Scope.JOINT),
    ("hellinger", "knn", Scope.JOINT),
    ("mmd", "empirical", Scope.JOINT),
    ("energy", "empirical", Scope.JOINT),
    ("sinkhorn", "empirical", Scope.JOINT),
    ("sliced_w2", "empirical", Scope.JOINT),
    ("sliced_w2", "gaussian", Scope.JOINT),
    ("sliced_sinkhorn", "empirical", Scope.JOINT),
    ("c2st", "classifier", Scope.JOINT),
)

RECIPES: Final[Mapping[str, Recipe]] = MappingProxyType(
    {
        f"{metric}:{estimator}": Recipe(metric, estimator, scope)
        for metric, estimator, scope in _RECIPE_TABLE
    }
)
"""Every recipe by its name ``"metric:estimator"``, in the order of the result rows.

:meta hide-value:
"""

DEFAULT_METRICS: Final = (
    "mean_shift:empirical",
    "width_ratio:empirical",
    "ks:empirical",
    "ad:empirical",
    "w1:empirical",
    "js:kde",
    "hellinger:kde",
    "kl:gaussian",
    "w2:gaussian",
    "hellinger:gaussian",
    "mmd:empirical",
    "energy:empirical",
    "sliced_w2:empirical",
    "sliced_w2:gaussian",
)
"""The recipes that ``compare`` runs by default, the fast set.

The kNN rows, the Sinkhorn divergence and the sliced Sinkhorn divergence cost
more and run on request.
"""

_TERM_TOTALS: Final = MappingProxyType(
    {
        "kl_location": "kl",
        "kl_shape": "kl",
        "w2_location": "w2",
        "w2_shape": "w2",
        "hellinger_location": "hellinger",
        "hellinger_shape": "hellinger",
    }
)
"""The location and shape terms of the Gaussian metrics, with their totals."""

# * Recipes that one statistic serves with one call, in the order of the result rows.
FAMILIES: Final[Mapping[str, tuple[str, ...]]] = MappingProxyType(
    {
        "mean_shift": ("mean_shift:empirical",),
        "width_ratio": ("width_ratio:empirical",),
        "marginal": ("ks:empirical", "ad:empirical", "w1:empirical"),
        "kde": ("js:kde", "hellinger:kde"),
        "gaussian": ("kl:gaussian", "w2:gaussian", "hellinger:gaussian"),
        "knn": ("kl:knn", "js:knn", "hellinger:knn"),
        "mmd": ("mmd:empirical",),
        "energy": ("energy:empirical",),
        "sinkhorn": ("sinkhorn:empirical",),
        "sliced_w2": ("sliced_w2:empirical",),
        "sliced_w2_gaussian": ("sliced_w2:gaussian",),
        "sliced_sinkhorn": ("sliced_sinkhorn:empirical",),
    }
)

# * The once-per-comparison settings that the recipes of a family read.
SETTINGS_OF_FAMILY: Final = MappingProxyType(
    {
        "kde": "factor",
        "mmd": "bandwidth",
        "sinkhorn": "bandwidth",
        "sliced_w2": "directions",
        "sliced_w2_gaussian": "directions",
        "sliced_sinkhorn": "sliced_sinkhorn",
    }
)


class CompareArguments(NamedTuple):
    """The arguments of ``compare`` after their checks, in the form the engine reads."""

    posteriors: tuple[Posterior, ...]
    space: ParamSpace
    transform_text: str | None
    reference: int | None
    source: WhiteningSource
    fisher: Array | None
    centre: Array | None
    recipes: tuple[str, ...]
    construction: Construction
    draws: int
    cap: int
    factors: tuple[float, ...]
    uncertainty: UncertaintyMethod
    half_samples: int
    thresholds: Mapping[str, float]
    level: float
    budget: float
    root: PRNGKeyArray
    seed: int | None
    config_hash: str | None


# * Reading and checking the arguments


def read_compare_arguments(
    sample_sets: object,
    names: object,
    space: ParamSpace,
    *,
    key: object,
    reference: object,
    whitening: object,
    fisher: object,
    centre: object,
    transform: Transform | None,
    metrics: object,
    null: object,
    draws: object,
    cap: object,
    sinkhorn_epsilon_factors: object,
    uncertainty: object,
    half_samples: object,
    thresholds: object,
    level: object,
    budget: object,
    config_hash: str | None,
) -> CompareArguments:
    """Check every argument and return them in the form the engine reads."""
    posteriors = gather_posteriors(sample_sets, names, space, transform=transform)
    if len(posteriors) < 2:
        raise ValueError(
            f"a comparison needs at least two posteriors, got {len(posteriors)}"
        )
    posterior_names = [posterior.name for posterior in posteriors]
    reference_index = _read_reference(reference, posterior_names)
    source = _read_whitening(whitening, reference_index, fisher, centre)
    construction = _read_null(null, reference_index)
    recipes = select_recipes(metrics)
    draw_count = read_count("draws", draws, 1)
    half_sample_count = read_count("half_samples", half_samples, 1)
    cap_count = read_count("cap", cap, 2)
    factors = _read_factors(sinkhorn_epsilon_factors)
    if not isinstance(uncertainty, str):
        raise TypeError(
            f"uncertainty must be a string, got {type(uncertainty).__name__}"
        )
    method = UncertaintyMethod(uncertainty)
    if method is UncertaintyMethod.DELTA_METHOD:
        raise ValueError(
            "uncertainty must be 'null_std' or 'subsampling_bootstrap'; the delta"
            " method is the error of the evidence ratio of reweighting"
        )
    row_space = space if transform is None else transform.map_space(space)
    _check_thresholds(thresholds, recipes, row_space)
    level_value = read_number("level", level)
    if not math.isfinite(level_value):
        raise ValueError(f"level must be finite, got {level_value}")
    budget_value = read_number("budget", budget)
    if not budget_value > 0:
        raise ValueError(f"budget must be positive, got {budget_value}")
    batch_count = posteriors[0].samples.shape[0]
    fisher_array, centre_array = _read_fisher(
        fisher, centre, batch_count, len(space), transform
    )
    root, seed = read_key(key)
    assert isinstance(thresholds, Mapping)
    return CompareArguments(
        posteriors=posteriors,
        space=row_space,
        transform_text=None if transform is None else transform.description,
        reference=reference_index,
        source=source,
        fisher=fisher_array,
        centre=centre_array,
        recipes=recipes,
        construction=construction,
        draws=draw_count,
        cap=cap_count,
        factors=factors,
        uncertainty=method,
        half_samples=half_sample_count,
        thresholds=thresholds,
        level=level_value,
        budget=budget_value,
        root=root,
        seed=seed,
        config_hash=config_hash,
    )


def comparison_pairs(
    posterior_count: int, reference: int | None
) -> list[tuple[int, int]]:
    """Return the posterior pairs: the reference against the others, or i < j."""
    if reference is not None:
        return [
            (reference, other) for other in range(posterior_count) if other != reference
        ]
    return [
        (first, second)
        for first in range(posterior_count)
        for second in range(first + 1, posterior_count)
    ]


def _read_reference(reference: object, names: list[str]) -> int | None:
    """Return the position of the reference posterior, or ``None`` for mode B."""
    if reference is None:
        return None
    if not isinstance(reference, str) or reference not in names:
        raise ValueError(f"reference must be one of {names}, got {reference!r}")
    return names.index(reference)


def _read_whitening(
    whitening: object, reference: int | None, fisher: object, centre: object
) -> WhiteningSource:
    """Return the whitening source, after checking it against the mode and Fisher."""
    if not isinstance(whitening, str | None):
        raise TypeError(
            f"whitening must be a string or None, got {type(whitening).__name__}"
        )
    if whitening is None:
        source = (
            WhiteningSource.POOLED if reference is None else WhiteningSource.REFERENCE
        )
    else:
        source = WhiteningSource(whitening)
    if source is WhiteningSource.REFERENCE and reference is None:
        raise ValueError("the reference whitening needs reference=<name>")
    if source is WhiteningSource.FISHER:
        if fisher is None or centre is None:
            raise ValueError("the fisher whitening needs fisher= and centre=")
    elif fisher is not None or centre is not None:
        raise ValueError(f"fisher and centre serve the fisher whitening, not {source}")
    return source


def _read_null(null: object, reference: int | None) -> Construction:
    """Return the construction of the null, checked against the mode."""
    if not isinstance(null, str):
        raise TypeError(f"null must be a string, got {type(null).__name__}")
    construction = Construction(null)
    if construction is Construction.GAUSSIAN_SURROGATE:
        raise ValueError(
            "null must be 'permutation' or 'split_half'; the Gaussian surrogate is"
            " the null of the diagnostics"
        )
    if construction is Construction.SPLIT_HALF and reference is None:
        raise ValueError("the split_half null needs reference=<name>")
    return construction


def read_count(name: str, value: object, minimum: int) -> int:
    """Return an integer argument, at least ``minimum``."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}, got {value}")
    return value


def read_number(name: str, value: object) -> float:
    """Return a real argument as a float."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{name} must be a number, got {type(value).__name__}")
    return float(value)


def _read_factors(factors: object) -> tuple[float, ...]:
    """Return the Sinkhorn ε factors: distinct, positive and finite numbers."""
    if isinstance(factors, str) or not isinstance(factors, Sequence) or not factors:
        raise ValueError(
            "sinkhorn_epsilon_factors must be a non-empty sequence of numbers, got"
            f" {factors!r}"
        )
    values = tuple(read_number("sinkhorn_epsilon_factors", entry) for entry in factors)
    if not all(math.isfinite(value) and value > 0 for value in values):
        raise ValueError(
            f"sinkhorn_epsilon_factors must be positive and finite, got {values}"
        )
    if len(set(values)) != len(values):
        raise ValueError(f"sinkhorn_epsilon_factors must be distinct, got {values}")
    return values


def _read_fisher(
    fisher: object,
    centre: object,
    batch_count: int,
    param_count: int,
    transform: Transform | None,
) -> tuple[Array | None, Array | None]:
    """Return the Fisher matrices and centres in the coordinates of the engine."""
    if fisher is None or centre is None:
        return None, None
    fisher_array = jnp.asarray(fisher, dtype=jnp.float64)
    centre_array = jnp.asarray(centre, dtype=jnp.float64)
    if batch_count == 1 and fisher_array.shape == (param_count, param_count):
        fisher_array = fisher_array[None]
    if batch_count == 1 and centre_array.shape == (param_count,):
        centre_array = centre_array[None]
    if fisher_array.shape != (batch_count, param_count, param_count):
        raise ValueError(
            f"fisher must be shaped ({batch_count}, {param_count}, {param_count}),"
            f" or ({param_count}, {param_count}) for one problem, got shape"
            f" {fisher_array.shape}"
        )
    if centre_array.shape != (batch_count, param_count):
        raise ValueError(
            f"centre must be shaped ({batch_count}, {param_count}), or"
            f" ({param_count},) for one problem, got shape {centre_array.shape}"
        )
    if transform is None:
        return fisher_array, centre_array
    # * The Jacobian is taken at the centre in the input coordinates.
    return transform.map_fisher(fisher_array, centre_array), transform.forward(
        centre_array
    )


def select_recipes(metrics: object) -> tuple[str, ...]:
    """Return the selected recipes in the order of the result rows."""
    if metrics is None:
        return DEFAULT_METRICS
    if isinstance(metrics, str) or not isinstance(metrics, Sequence):
        raise ValueError(
            f"metrics must be a sequence of strings, got {type(metrics).__name__}"
        )
    if not metrics:
        raise ValueError("metrics must not be empty")
    chosen: set[str] = set()
    for entry in metrics:
        if not isinstance(entry, str):
            raise ValueError(f"metrics must hold strings, got {entry!r}")
        chosen.update(_recipes_of(entry))
    if any(RECIPES[recipe].metric == "c2st" for recipe in chosen):
        raise NotImplementedError(_DEFERRED)
    return tuple(recipe for recipe in RECIPES if recipe in chosen)


def _recipes_of(entry: str) -> tuple[str, ...]:
    """Return the recipes that one entry of ``metrics`` selects."""
    metric, separator, estimator = entry.partition(":")
    metric = "ks" if metric == "kuiper_ks" else metric
    if metric in _TERM_TOTALS:
        total = _TERM_TOTALS[metric]
        raise ValueError(
            f"{entry!r} is a term of {total!r}; select {total!r}, whose rows"
            " carry its location and shape terms"
        )
    found: tuple[str, ...]
    if separator:
        recipe = f"{metric}:{estimator}"
        found = (recipe,) if recipe in RECIPES else ()
    else:
        found = tuple(key for key, value in RECIPES.items() if value.metric == metric)
    if not found:
        valid = sorted({recipe.metric for recipe in RECIPES.values()} | {"kuiper_ks"})
        raise ValueError(
            f"unknown metric {entry!r}; the metrics are {', '.join(valid)}, and the"
            f" recipes are {', '.join(RECIPES)}"
        )
    return found


def _row_metric_names(recipes: Sequence[str], space: ParamSpace) -> set[str]:
    """Return the metric names of the rows that the recipes give in a space."""
    names: set[str] = set()
    for recipe in recipes:
        metric = RECIPES[recipe].metric
        if metric == "ks":
            names.update(ks_metric_names(space))
        elif metric in ("kl", "w2", "hellinger") and recipe.endswith(":gaussian"):
            names.update((metric, f"{metric}_location", f"{metric}_shape"))
        else:
            names.add(metric)
    return names


def _check_thresholds(
    thresholds: object, recipes: Sequence[str], space: ParamSpace
) -> None:
    """Raise unless the thresholds hold every metric of the selected rows."""
    if not isinstance(thresholds, Mapping):
        raise TypeError(
            f"thresholds must be a mapping, got {type(thresholds).__name__}"
        )
    missing = sorted(_row_metric_names(recipes, space) - set(thresholds))
    if missing:
        raise KeyError(f"thresholds lack the metrics {', '.join(missing)}")


# * The arguments of ``diagnose``

DIAGNOSTIC_FAMILIES: Final = ("chains", "shape", "dependence", "modes", "railing")
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


class DiagnoseSource(NamedTuple):
    """One input of ``diagnose``: one run of chains, or one set of posteriors."""

    chains: Chains | None
    """The chains of a run, with one posterior; ``None`` for a sample set."""
    sample_set: SampleSet | None
    """The flat samples of the posteriors; ``None`` for chains."""
    names: tuple[str, ...]
    """The names of its posteriors."""
    burn_in: int
    """Steps dropped from the start of the chains; zero for a sample set."""


class DiagnoseArguments(NamedTuple):
    """The arguments of ``diagnose`` after their checks, as the engine reads them."""

    sources: tuple[DiagnoseSource, ...]
    names: tuple[str, ...]
    batch_count: int
    space: ParamSpace
    """The space of the chains and of the input sets."""
    row_space: ParamSpace
    """The space of the rows other than the chain rows: mapped by the transform."""
    transform: Transform | None
    families: tuple[str, ...]
    draws: int
    cap: int
    pair_cap: int
    thresholds: Mapping[str, Threshold]
    level: float
    budget: float
    root: PRNGKeyArray
    seed: int | None
    config_hash: str | None


def read_diagnose_arguments(
    sources: object,
    names: object,
    space: ParamSpace,
    *,
    key: object,
    burn_in: object,
    transform: Transform | None,
    diagnostics: object,
    draws: object,
    cap: object,
    pair_cap: object,
    thresholds: object,
    level: object,
    budget: object,
    config_hash: str | None,
) -> DiagnoseArguments:
    """Check every argument of ``diagnose`` and return them in the form it reads."""
    entries = _read_sources(sources)
    chain_count = sum(isinstance(entry, Chains) for entry in entries)
    burn_ins = iter(_read_burn_in(burn_in, chain_count))
    param_count = len(space)
    batch_counts = {
        1 if isinstance(entry, Chains) else entry.shape.batch for entry in entries
    }
    for position, entry in enumerate(entries):
        size = entry.shape.param
        if size != param_count:
            raise ValueError(
                f"the space has {param_count} parameters, but source {position} has"
                f" {size}"
            )
    if len(batch_counts) > 1:
        raise ValueError(
            "every source needs the same number of problems, and chains count as"
            f" one, got {sorted(batch_counts)}"
        )
    counts = [
        1 if isinstance(entry, Chains) else entry.shape.posterior for entry in entries
    ]
    posterior_names = unique_names("names", names, sum(counts))
    sources_read = []
    start = 0
    for entry, count in zip(entries, counts, strict=True):
        is_chains = isinstance(entry, Chains)
        sources_read.append(
            DiagnoseSource(
                chains=entry if isinstance(entry, Chains) else None,
                sample_set=None if isinstance(entry, Chains) else entry,
                names=posterior_names[start : start + count],
                burn_in=next(burn_ins) if is_chains else 0,
            )
        )
        start += count
    families = _read_families(diagnostics)
    draw_count = read_count("draws", draws, 1)
    cap_count = read_count("cap", cap, 2)
    pair_cap_count = read_count("pair_cap", pair_cap, 2)
    row_space = space if transform is None else transform.map_space(space)
    _check_diagnostic_thresholds(
        thresholds, families, chain_count > 0, space, row_space
    )
    level_value = read_number("level", level)
    if not math.isfinite(level_value):
        raise ValueError(f"level must be finite, got {level_value}")
    budget_value = read_number("budget", budget)
    if not budget_value > 0:
        raise ValueError(f"budget must be positive, got {budget_value}")
    root, seed = read_key(key)
    assert isinstance(thresholds, Mapping)
    return DiagnoseArguments(
        sources=tuple(sources_read),
        names=posterior_names,
        batch_count=batch_counts.pop(),
        space=space,
        row_space=row_space,
        transform=transform,
        families=families,
        draws=draw_count,
        cap=cap_count,
        pair_cap=pair_cap_count,
        thresholds=thresholds,
        level=level_value,
        budget=budget_value,
        root=root,
        seed=seed,
        config_hash=config_hash,
    )


def _read_sources(sources: object) -> tuple[Chains | SampleSet, ...]:
    """Return one source, or a sequence of sources, as a non-empty tuple."""
    if isinstance(sources, Chains | SampleSet):
        return (sources,)
    if isinstance(sources, str) or not isinstance(sources, Sequence):
        raise TypeError(
            "sources must be Chains, a SampleSet or a sequence of them, got"
            f" {type(sources).__name__}"
        )
    if not sources:
        raise ValueError("sources must hold at least one Chains or SampleSet")
    for position, entry in enumerate(sources):
        if not isinstance(entry, Chains | SampleSet):
            raise TypeError(
                "sources must hold Chains or SampleSet objects, but entry"
                f" {position} is {type(entry).__name__}"
            )
    return tuple(sources)


def _read_burn_in(burn_in: object, chain_count: int) -> tuple[int, ...]:
    """Return the burn-in of each Chains input: one value for all, or one for each."""
    if isinstance(burn_in, Sequence) and not isinstance(burn_in, str):
        if len(burn_in) != chain_count:
            raise ValueError(
                "burn_in must give one value for every Chains input, got"
                f" {len(burn_in)} values for {chain_count} Chains inputs"
            )
        return tuple(read_count("burn_in", value, 0) for value in burn_in)
    return (read_count("burn_in", burn_in, 0),) * chain_count


def _read_families(diagnostics: object) -> tuple[str, ...]:
    """Return the selected families of rows in the order of the result."""
    if diagnostics is None:
        return DIAGNOSTIC_FAMILIES
    if isinstance(diagnostics, str) or not isinstance(diagnostics, Sequence):
        raise ValueError(
            "diagnostics must be a sequence of family names, got"
            f" {type(diagnostics).__name__}"
        )
    if not diagnostics:
        raise ValueError("diagnostics must not be empty")
    unknown = [entry for entry in diagnostics if entry not in DIAGNOSTIC_FAMILIES]
    if unknown:
        raise ValueError(
            f"unknown diagnostic families {unknown}; the families are"
            f" {', '.join(DIAGNOSTIC_FAMILIES)}"
        )
    return tuple(family for family in DIAGNOSTIC_FAMILIES if family in diagnostics)


def _diagnostic_metric_names(
    families: Sequence[str], has_chains: bool, space: ParamSpace, row_space: ParamSpace
) -> set[str]:
    """Return the metrics of the selected rows that have a default threshold."""
    names: set[str] = set()
    if "chains" in families and has_chains:
        names.update(("ess", *split_chain_metric_names(space)))
    if "shape" in families:
        for skewness, kurtosis in moment_metric_names(row_space):
            names.update((skewness, kurtosis))
        names.update(("anderson_darling", "mardia_skewness", "mardia_kurtosis"))
    if "dependence" in families and len(row_space) >= 2:
        names.update(("spearman", "distance_correlation", "mutual_information"))
    if "modes" in families:
        names.update(("gmm_bic", "mode_dip"))
    if "railing" in families and railed_parameters(row_space):
        names.update(("railing_lower", "railing_upper"))
    return names & set(DIAGNOSTIC_THRESHOLDS)


def _check_diagnostic_thresholds(
    thresholds: object,
    families: Sequence[str],
    has_chains: bool,
    space: ParamSpace,
    row_space: ParamSpace,
) -> None:
    """Raise unless the thresholds hold every flagging metric of the selected rows."""
    if not isinstance(thresholds, Mapping):
        raise TypeError(
            f"thresholds must be a mapping, got {type(thresholds).__name__}"
        )
    required = _diagnostic_metric_names(families, has_chains, space, row_space)
    missing = sorted(required - set(thresholds))
    if missing:
        raise KeyError(f"thresholds lack the metrics {', '.join(missing)}")
