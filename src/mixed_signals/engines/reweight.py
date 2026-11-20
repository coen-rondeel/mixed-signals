"""``reweight``: importance weights for a new prior, a new likelihood or a new target.

:func:`reweight` turns the samples of a :class:`~mixed_signals.core.samples.SampleSet`
into samples of another target density by importance weights, estimates the
ratio of the evidences, and says how far the weights can be trusted. It is a
plain function that returns the reweighted set and a
:class:`~mixed_signals.core.result.Result` table.

Proposals
    ``proposal="posterior"`` (the default) takes the samples as draws of the
    stored posterior L_old π_old, with their weights. It needs a new prior,
    a new log-likelihood or both. A new prior needs the stored ``log_prior`` of
    the set, and a new likelihood its stored ``log_like``; the ratio is
    π_new/π_old, L_new/L_old or their product (Thrane and Talbot 2019; Payne,
    Talbot and Thrane 2019). ``proposal="approximation"`` takes the samples
    as equally weighted draws from an approximation q, such as a normalizing
    flow, and needs the stored ``log_q``. The target is L π, with the new
    log-likelihood and prior where given and the stored ones otherwise, so
    each must come from one of the two; the ratio is L π/q, and its mean is
    the evidence Z because q is normalised (Dax et al. 2023). The positive
    input weights of an approximation must be equal.
Coordinates
    A ratio of densities needs both in the same coordinates, and a density
    moves with the Jacobian of a change of coordinates (Papamakarios et al.
    2021). The stored ``log_prior`` and ``log_q`` are in the coordinates of the
    set, so a new prior must be given there. With ``transform`` the engine maps
    the set first, and the new densities are evaluated in the mapped
    coordinates. For a set in eryn's sampling coordinates, where an amplitude is
    sampled as ln A, ``transform=ErynPhysical(old_priors)`` and
    ``prior=ErynPrior(new_priors, coordinates="physical")`` reweight a
    log-uniform amplitude prior to a uniform one in the physical amplitude::

        reweighting = reweight(
            sample_set,
            ["run"],
            space,
            prior=ErynPrior(new_priors, coordinates="physical"),
            transform=ErynPhysical(old_priors),
        )
        reweighting.sample_set  # in the physical coordinates
Boundary calls
    A new prior and a new likelihood may be any callable of the array library
    named by ``backend``, such as NumPy, JAX, PyTorch or CuPy, so they are
    called at the boundary, outside ``jax.jit``, in chunks of ``chunk_size``
    samples (:func:`~mixed_signals.io.arrays.call_in_chunks`). Each takes
    samples shaped (n, D) and returns n values. A set of several problems
    takes one log-likelihood per problem, because independent problems have
    their own data. The densities are checked for NaN and +∞ at the samples
    with positive weight; −∞ is a zero density.
Weights
    The new log weights are the input ones plus the log ratios, and −∞ at
    samples that had no weight. Each weight set, one per problem and
    posterior, gets k̂, the shape of the Pareto tail of its weights
    (:func:`~mixed_signals.reweight.psis.pareto_tail`), against the threshold
    for its number of samples. The weights stay raw unless ``smooth=True``, which
    replaces the tail by its fitted quantiles
    (:func:`~mixed_signals.reweight.psis.pareto_smooth`).
Rows
    Each weight set gives three rows with the pair (source name, target
    name): ``pareto_k`` (estimator ``psis``) with k̂, ``importance_ess``
    (estimator ``kish``) with the reweighted effective sample size, and
    ``log_evidence_ratio`` (estimator ``importance``, or ``psis`` when the
    weights are smoothed) with ln Z_new/Z_old, its standard error as the
    ``uncertainty`` and the ``uncertainty_method`` ``delta_method``
    (:func:`~mixed_signals.reweight.importance.log_evidence_ratio`). The
    ``estimator_settings`` hold the number of draws, the tail length and the
    relative efficiency of the input set. The ``space`` column names the
    transform, or ``identity``. The provenance holds the hashes of the input
    and the returned samples with their weights; the engine is deterministic, so
    it has no seed and no key.
Returned set
    The reweighted set holds the samples, the new log weights, the effective
    sample size of the new weights, and the new log-likelihood and log-prior
    where given, the stored ones otherwise, so that it reweights again.
    Its ``log_q`` is the stored one. At samples of zero weight, a new density
    that is NaN or +∞ is stored as −∞, since the sample no longer enters any
    estimate.
Flags
    An :class:`~mixed_signals.errors.ImportanceWeightWarning` goes into every
    row of a weight set in three cases: k̂ passes its threshold; fewer than 25
    samples keep a weight, because k̂ needs 5 tail draws; or k̂ is undefined
    because the generalized Pareto fit of the tail failed, for example when many
    of the largest weights are equal, so the weights cannot be judged (they stay
    raw also when smoothing is asked, because smoothing leaves an undefined tail
    unchanged). A flat tail, whose k̂ is −∞, does not warn, as the weights are
    then bounded. Warnings of the density calls, such as a narrow float result,
    go into the rows of their problem. The package warnings are captured per
    row, not raised.

References:
    :cite:t:`payne2019higher`; :cite:t:`thrane2019introduction`;
    :cite:t:`dax2023neural`; :cite:t:`vehtari2024pareto`;
    :cite:t:`vehtari2026loo`; :cite:t:`kish1965survey`;
    :cite:t:`papamakarios2021normalizing`.
"""

import json
import math
import os
import warnings
from collections.abc import Callable, Sequence
from enum import StrEnum
from typing import Any, Final, NamedTuple

import jax.numpy as jnp
from jaxtyping import Array, Bool, Float

from mixed_signals import errors
from mixed_signals.core.result import Provenance, Result, UncertaintyMethod, hash_arrays
from mixed_signals.core.samples import SampleSet
from mixed_signals.core.space import ParamSpace
from mixed_signals.engines._common import unique_names
from mixed_signals.errors import CapabilityError, ImportanceWeightWarning, WarningRecord
from mixed_signals.io.arrays import ArrayBackend, call_in_chunks
from mixed_signals.priors.protocol import PriorDistribution
from mixed_signals.reweight.importance import (
    log_evidence_ratio,
    relative_efficiency,
    reweighted_effective_sample_size,
)
from mixed_signals.reweight.psis import (
    MINIMUM_TAIL_DRAWS,
    pareto_k_threshold,
    pareto_smooth,
    pareto_tail,
    pareto_tail_length,
)
from mixed_signals.transforms.base import IDENTITY_DESCRIPTION, Transform

TARGET_SUFFIX: Final = ":reweighted"
"""Text that follows a posterior name in the default name of its reweighted target."""

# * Frames in the package are skipped, so a warning points at the caller's line.
_PACKAGE_DIRECTORY: Final = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_EQUAL_WEIGHT_TOLERANCE: Final = 1e-12
"""Largest relative difference of the positive weights that counts as equal."""

# * Five tail draws need a fifth of the draws, whatever the relative efficiency.
_MINIMUM_DRAWS: Final = 5 * MINIMUM_TAIL_DRAWS


class Proposal(StrEnum):
    """What the samples of a set are draws from."""

    POSTERIOR = "posterior"
    """The stored posterior L_old π_old, with the weights of the set."""
    APPROXIMATION = "approximation"
    """An approximation q with stored ``log_q``, with equal weights."""


type LogLikelihood = Callable[[Any], object]
"""Maps samples shaped (n, D), as arrays of the chosen library, to n log-likelihoods."""


class Reweighting(NamedTuple):
    """The reweighted samples and the table of their weight diagnostics."""

    sample_set: SampleSet
    """The samples with the new log weights, in the coordinates of the rows."""
    result: Result
    """Three rows per problem and posterior: ``pareto_k``, ``importance_ess`` and
    ``log_evidence_ratio``."""


class _NewDensities(NamedTuple):
    """The new densities of one problem at its samples, shaped (K, N)."""

    log_prior: Float[Array, "posterior sample"] | None
    log_like: Float[Array, "posterior sample"] | None
    records: list[WarningRecord]


class _WeightSet(NamedTuple):
    """The assessment of one weight set: its tail, final weights and warnings."""

    shape: float
    draws: int
    tail_draws: int
    log_weights: Float[Array, " sample"]
    efficiency: float
    log_evidence: Float[Array, ""]
    evidence_error: Float[Array, ""]
    effective_sample_size: float
    records: list[WarningRecord]


def reweight(
    sample_set: SampleSet,
    names: Sequence[str],
    space: ParamSpace,
    *,
    prior: PriorDistribution | None = None,
    log_like: LogLikelihood | Sequence[LogLikelihood] | None = None,
    proposal: Proposal | str = Proposal.POSTERIOR,
    transform: Transform | None = None,
    smooth: bool = False,
    target_names: Sequence[str] | None = None,
    backend: ArrayBackend = "numpy",
    chunk_size: int | None = None,
) -> Reweighting:
    """Reweight samples to a new prior, a new likelihood or an approximation's target.

    The module docstring gives the cases, the coordinates and the rows. The
    package warnings, such as an
    :class:`~mixed_signals.errors.ImportanceWeightWarning`, are not raised but
    go into the ``warnings`` column of the rows of their weight set.

    Args:
        sample_set: The samples, shaped (B, K, N, D), with the stored densities
            that the proposal needs.
        names: The K names of the posteriors, unique.
        space: The parameter space of the samples, with D parameters.
        prior: The new prior, a
            :class:`~mixed_signals.priors.protocol.PriorDistribution` whose
            ``logpdf`` takes samples in the coordinates of the reweighting;
            ``None`` to keep the stored one.
        log_like: The new log-likelihood: one callable for a set of one
            problem, or one callable per problem, each called on all K
            posteriors of its problem; ``None`` to keep the stored one.
        proposal: ``"posterior"`` or ``"approximation"``; see
            :class:`Proposal`.
        transform: A transform that maps the set to the coordinates of the
            new densities first; ``None`` for the coordinates of the set.
        smooth: Whether to replace the tail of each weight set by its fitted
            Pareto quantiles. The weights stay raw by default.
        target_names: The names of the reweighted posteriors in the ``pair``
            column; ``<name>:reweighted`` for each name by default.
        backend: The array library of the arrays that the new densities
            receive: ``"numpy"``, ``"jax"``, ``"torch"`` or ``"cupy"``.
        chunk_size: The most samples in one call of a new density; all samples
            of a problem in one call by default.

    Returns:
        The reweighted set and its result table.

    Raises:
        ValueError: If the names or the space do not fit the set, the proposal
            is unknown, the proposal needs a change and gets none, a single
            log-likelihood serves several problems or a sequence has the wrong
            length, the weights of an approximation are unequal, a new density
            is NaN or +∞, or a stored density that enters as a denominator is
            not finite, at a sample with positive weight, or no sample of a
            posterior keeps a positive weight.
        TypeError: If the prior is not a ``PriorDistribution`` or a
            log-likelihood is not callable.
        CapabilityError: If the set lacks the stored ``log_prior``,
            ``log_like`` or ``log_q`` that the proposal needs.

    References:
        :cite:t:`vehtari2024pareto`; :cite:t:`payne2019higher`;
        :cite:t:`thrane2019introduction`; :cite:t:`dax2023neural`.
    """
    batch_count, posterior_count, _, param_count = sample_set.shape
    source_names, targets = _check_names(names, target_names, posterior_count)
    if len(space) != param_count:
        raise ValueError(
            f"the space has {len(space)} parameters, but the samples have {param_count}"
        )
    chosen = Proposal(proposal)
    if prior is not None and not isinstance(prior, PriorDistribution):
        raise TypeError(
            "prior must be a PriorDistribution with logpdf and rvs, got"
            f" {type(prior).__name__}"
        )
    likelihoods = _check_likelihoods(log_like, batch_count)
    if transform is None:
        working, row_space, space_text = sample_set, space, IDENTITY_DESCRIPTION
    else:
        working = transform.map_sample_set(sample_set)
        row_space = transform.map_space(space)
        space_text = transform.description
    _check_capabilities(
        working, chosen, has_prior=prior is not None, has_like=likelihoods is not None
    )
    if chosen is Proposal.APPROXIMATION:
        _check_equal_weights(working.log_weights, source_names)
    efficiency = relative_efficiency(working.log_weights, working.n_eff)
    parameter_names = list(row_space.names)

    new_log_weights, new_log_like, new_log_prior = [], [], []
    assessments: list[list[_WeightSet]] = []
    for problem in range(batch_count):
        densities = _new_densities(
            working,
            problem,
            prior,
            None if likelihoods is None else likelihoods[problem],
            source_names,
            backend=backend,
            chunk_size=chunk_size,
        )
        problem_weights = _new_log_weights(
            working, problem, chosen, densities, source_names
        )
        sets = [
            _assess_weight_set(
                working.log_weights[problem, posterior],
                problem_weights[posterior],
                float(efficiency[problem, posterior]),
                source_names[posterior],
                problem,
                smooth=smooth,
                density_records=densities.records,
            )
            for posterior in range(posterior_count)
        ]
        assessments.append(sets)
        new_log_weights.append(jnp.stack([entry.log_weights for entry in sets]))
        if densities.log_like is not None:
            new_log_like.append(densities.log_like)
        if densities.log_prior is not None:
            new_log_prior.append(densities.log_prior)

    new_set = SampleSet(
        working.samples,
        log_weights=jnp.stack(new_log_weights),
        n_eff=jnp.asarray(
            [[entry.effective_sample_size for entry in sets] for sets in assessments]
        ),
        log_like=jnp.stack(new_log_like) if new_log_like else working.log_like,
        log_prior=jnp.stack(new_log_prior) if new_log_prior else working.log_prior,
        log_q=working.log_q,
    )
    rows: list[dict[str, Any]] = []
    for problem, sets in enumerate(assessments):
        for posterior, entry in enumerate(sets):
            rows.extend(
                _set_rows(
                    entry,
                    problem=problem,
                    pair=[source_names[posterior], targets[posterior]],
                    space_text=space_text,
                    parameters=parameter_names,
                    provenance=Provenance.collect(
                        input_hashes=[
                            hash_arrays(
                                sample_set.samples[problem, posterior],
                                sample_set.log_weights[problem, posterior],
                            ),
                            hash_arrays(
                                new_set.samples[problem, posterior],
                                new_set.log_weights[problem, posterior],
                            ),
                        ]
                    ),
                    smoothed=smooth,
                )
            )
    return Reweighting(new_set, Result.from_rows(rows))


def _check_names(
    names: object, target_names: object, posterior_count: int
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return the source and target names, checked against the posteriors."""
    sources = unique_names("names", names, posterior_count)
    if target_names is None:
        return sources, tuple(f"{name}{TARGET_SUFFIX}" for name in sources)
    targets = unique_names("target_names", target_names, posterior_count)
    for source, target in zip(sources, targets, strict=True):
        if source == target:
            raise ValueError(
                f"target_names must differ from names, but {source!r} is both"
            )
    return sources, targets


def _check_likelihoods(
    log_like: object, batch_count: int
) -> tuple[LogLikelihood, ...] | None:
    """Return one log-likelihood per problem, or ``None`` when none is given."""
    if log_like is None:
        return None
    if callable(log_like):
        if batch_count != 1:
            raise ValueError(
                f"the set has {batch_count} problems, so pass one log-likelihood per"
                " problem: independent problems have their own data"
            )
        return (log_like,)
    if isinstance(log_like, str) or not isinstance(log_like, Sequence):
        raise TypeError(
            "log_like must be a callable or a sequence of callables, got"
            f" {type(log_like).__name__}"
        )
    functions = tuple(log_like)
    if len(functions) != batch_count:
        raise ValueError(
            f"log_like holds {len(functions)} log-likelihoods for {batch_count}"
            " problems; pass one per problem"
        )
    for index, function in enumerate(functions):
        if not callable(function):
            raise TypeError(
                f"log_like entry {index} is not callable, got {type(function).__name__}"
            )
    return functions


def _check_capabilities(
    working: SampleSet, proposal: Proposal, *, has_prior: bool, has_like: bool
) -> None:
    """Raise unless the set stores the densities that the proposal needs."""
    if proposal is Proposal.POSTERIOR:
        if not (has_prior or has_like):
            raise ValueError(
                "reweight needs a new prior, a new log-likelihood or both: the"
                " posterior proposal reweights a change"
            )
        if has_prior and working.log_prior is None:
            raise CapabilityError(
                "prior reweighting needs stored log-prior values, the log_prior of"
                " the sample set"
            )
        if has_like and working.log_like is None:
            raise CapabilityError(
                "likelihood reweighting needs stored log-likelihood values, the"
                " log_like of the sample set"
            )
        return
    if working.log_q is None:
        raise CapabilityError(
            "reweighting from an approximation needs the stored log_q of the sample"
            " set, the log density of each sample under the approximation"
        )
    if not has_like and working.log_like is None:
        raise CapabilityError(
            "reweighting from an approximation needs a target log-likelihood: pass"
            " log_like or store the log_like of the sample set"
        )
    if not has_prior and working.log_prior is None:
        raise CapabilityError(
            "reweighting from an approximation needs a target log-prior: pass prior"
            " or store the log_prior of the sample set"
        )


def _check_equal_weights(
    log_weights: Float[Array, "batch posterior sample"], names: Sequence[str]
) -> None:
    """Raise unless the positive weights of every posterior are equal."""
    positive = log_weights > -jnp.inf
    highest = jnp.max(log_weights, axis=-1)
    lowest = jnp.min(jnp.where(positive, log_weights, jnp.inf), axis=-1)
    unequal = highest - lowest > _EQUAL_WEIGHT_TOLERANCE
    if bool(jnp.any(unequal)):
        problem, posterior = (int(index[0]) for index in jnp.nonzero(unequal))
        raise ValueError(
            "reweighting from an approximation needs equal input weights, because"
            f" the samples are draws from q; posterior {names[posterior]!r} of"
            f" problem {problem} has unequal weights"
        )


def _new_densities(
    working: SampleSet,
    problem: int,
    prior: PriorDistribution | None,
    log_like: LogLikelihood | None,
    names: Sequence[str],
    *,
    backend: ArrayBackend,
    chunk_size: int | None,
) -> _NewDensities:
    """Evaluate the new prior and likelihood of one problem at the boundary."""
    positive = working.log_weights[problem] > -jnp.inf
    shape = (working.shape.posterior, working.shape.sample)
    points = working.samples[problem].reshape(-1, working.shape.param)
    new_prior = new_like = None
    with errors.capture_warnings() as records:
        if prior is not None:
            new_prior = call_in_chunks(
                prior.logpdf, points, backend=backend, chunk_size=chunk_size
            ).reshape(shape)
        if log_like is not None:
            new_like = call_in_chunks(
                log_like, points, backend=backend, chunk_size=chunk_size
            ).reshape(shape)
    for density_name, values in (
        ("log-prior", new_prior),
        ("log-likelihood", new_like),
    ):
        if values is not None:
            _check_new_values(values, positive, names, problem, density_name)
    return _NewDensities(
        _clean_density(new_prior), _clean_density(new_like), list(records)
    )


def _check_new_values(
    values: Float[Array, "posterior sample"],
    positive: Bool[Array, "posterior sample"],
    names: Sequence[str],
    problem: int,
    density_name: str,
) -> None:
    """Raise if a new density is NaN or +inf where the input weight is positive."""
    invalid = (jnp.isnan(values) | (values == jnp.inf)) & positive
    flagged = jnp.any(invalid, axis=-1)
    if bool(jnp.any(flagged)):
        raise ValueError(
            f"the new {density_name} of posterior {names[int(jnp.argmax(flagged))]!r}"
            f" in problem {problem} is NaN or +inf at a sample with positive weight"
        )


def _check_stored_values(
    values: Float[Array, "posterior sample"],
    positive: Bool[Array, "posterior sample"],
    names: Sequence[str],
    problem: int,
    density_name: str,
) -> None:
    """Raise if a stored density is not finite where the input weight is positive."""
    flagged = jnp.any(~jnp.isfinite(values) & positive, axis=-1)
    if bool(jnp.any(flagged)):
        raise ValueError(
            f"the stored {density_name} of posterior"
            f" {names[int(jnp.argmax(flagged))]!r} in problem {problem} is not"
            " finite at a sample with positive weight, but it is the denominator"
            " of the ratio"
        )


def _clean_density(
    values: Float[Array, "posterior sample"] | None,
) -> Float[Array, "posterior sample"] | None:
    """Return a new density with NaN and +inf at zero-weight samples set to −inf."""
    if values is None:
        return None
    return jnp.where(jnp.isnan(values) | (values == jnp.inf), -jnp.inf, values)


def _new_log_weights(
    working: SampleSet,
    problem: int,
    proposal: Proposal,
    densities: _NewDensities,
    names: Sequence[str],
) -> Float[Array, "posterior sample"]:
    """Return the new log weights of one problem, the input plus the log ratio."""
    input_weights = working.log_weights[problem]
    positive = input_weights > -jnp.inf
    if proposal is Proposal.POSTERIOR:
        log_ratio = jnp.zeros_like(input_weights)
        if densities.log_prior is not None:
            stored_prior = _stored(working.log_prior, problem)
            _check_stored_values(stored_prior, positive, names, problem, "log_prior")
            log_ratio = log_ratio + densities.log_prior - stored_prior
        if densities.log_like is not None:
            stored_like = _stored(working.log_like, problem)
            _check_stored_values(stored_like, positive, names, problem, "log_like")
            log_ratio = log_ratio + densities.log_like - stored_like
    else:
        log_q = _stored(working.log_q, problem)
        _check_stored_values(log_q, positive, names, problem, "log_q")
        target_like = (
            _stored(working.log_like, problem)
            if densities.log_like is None
            else densities.log_like
        )
        target_prior = (
            _stored(working.log_prior, problem)
            if densities.log_prior is None
            else densities.log_prior
        )
        log_ratio = target_like + target_prior - log_q
    # * Zero-weight samples stay at −∞, whatever their ratio, so −∞ + ∞ never occurs.
    return jnp.where(positive, input_weights + log_ratio, -jnp.inf)


def _stored(
    values: Float[Array, "batch posterior sample"] | None, problem: int
) -> Float[Array, "posterior sample"]:
    """Return the stored density of one problem, which the capability check ensured."""
    assert values is not None
    return values[problem]


def _assess_weight_set(
    input_log_weights: Float[Array, " sample"],
    new_log_weights: Float[Array, " sample"],
    efficiency: float,
    name: str,
    problem: int,
    *,
    smooth: bool,
    density_records: Sequence[WarningRecord],
) -> _WeightSet:
    """Fit the tail of one weight set, flag it, and find its evidence and size."""
    draws = int(jnp.sum(new_log_weights > -jnp.inf))
    if draws == 0:
        raise ValueError(
            f"no sample of posterior {name!r} in problem {problem} keeps a positive"
            " weight"
        )
    tail_draws = pareto_tail_length(draws, efficiency)
    final_weights = new_log_weights
    with errors.capture_warnings() as records:
        if tail_draws < MINIMUM_TAIL_DRAWS:
            shape = float("nan")
            _warn(
                f"k-hat of posterior {name!r} in problem {problem} needs at least"
                f" {MINIMUM_TAIL_DRAWS} tail draws, so at least {_MINIMUM_DRAWS}"
                f" samples with positive weight, but {draws} keep one, which gives a"
                f" tail of {tail_draws}; the weights stay raw"
            )
        else:
            if smooth:
                smoothed = pareto_smooth(new_log_weights, tail_draws=tail_draws)
                shape, final_weights = float(smoothed.shape), smoothed.log_weights
            else:
                shape = float(pareto_tail(new_log_weights, tail_draws=tail_draws).shape)
            threshold = pareto_k_threshold(draws)
            if math.isnan(shape):
                # * NaN at a hard limit warns; a flat tail (−inf) does not.
                _warn(
                    f"k-hat of posterior {name!r} in problem {problem} is undefined"
                    " because the generalized Pareto fit of its tail failed, for"
                    " example when many of the largest weights are equal, so the"
                    " weights cannot be judged"
                    + ("; smoothing leaves them unchanged" if smooth else "")
                )
            # * NaN and −inf compare false, so only a defined shape can flag.
            elif shape > threshold:
                _warn(
                    f"k-hat {shape:.3g} of posterior {name!r} in problem {problem}"
                    f" exceeds the threshold {threshold:.3g} for {draws} draws;"
                    " estimates from these weights are likely unstable or biased"
                )
    evidence = log_evidence_ratio(input_log_weights, final_weights, efficiency)
    size = reweighted_effective_sample_size(final_weights, efficiency)
    return _WeightSet(
        shape=shape,
        draws=draws,
        tail_draws=tail_draws,
        log_weights=final_weights,
        efficiency=efficiency,
        log_evidence=evidence.value,
        evidence_error=evidence.error,
        effective_sample_size=float(size),
        records=[*density_records, *records],
    )


def _warn(message: str) -> None:
    """Raise an importance-weight warning that points at the caller's line."""
    warnings.warn(
        ImportanceWeightWarning(message), skip_file_prefixes=(_PACKAGE_DIRECTORY,)
    )


def _set_rows(
    entry: _WeightSet,
    *,
    problem: int,
    pair: list[str],
    space_text: str,
    parameters: list[str],
    provenance: Provenance,
    smoothed: bool,
) -> list[dict[str, Any]]:
    """Return the three rows of one weight set, in order."""
    common: dict[str, Any] = {
        "problem": problem,
        "pair": pair,
        "space": space_text,
        "parameters": parameters,
        "warnings": entry.records,
        "provenance": provenance,
    }
    efficiency_settings = json.dumps({"relative_efficiency": entry.efficiency})
    return [
        common
        | {
            "metric": "pareto_k",
            "estimator": "psis",
            "estimator_settings": json.dumps(
                {
                    "draws": entry.draws,
                    "tail_draws": entry.tail_draws,
                    "shape_prior": True,
                    "relative_efficiency": entry.efficiency,
                }
            ),
            "value": entry.shape,
        },
        common
        | {
            "metric": "importance_ess",
            "estimator": "kish",
            "estimator_settings": efficiency_settings,
            "value": entry.effective_sample_size,
        },
        common
        | {
            "metric": "log_evidence_ratio",
            "estimator": "psis" if smoothed else "importance",
            "estimator_settings": efficiency_settings,
            "value": float(entry.log_evidence),
            "uncertainty": float(entry.evidence_error),
            "uncertainty_method": UncertaintyMethod.DELTA_METHOD,
        },
    ]
