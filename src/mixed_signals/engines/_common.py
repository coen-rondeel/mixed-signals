"""Helpers shared by the engines, their dry run and their argument readers.

The engines read the same arguments, so their checks and conversions live here:
the key of a call (:func:`read_key`), the names of the posteriors
(:func:`unique_names`), the posteriors of one or several
:class:`~mixed_signals.core.samples.SampleSet` objects
(:func:`gather_posteriors`), the samples of one problem that carry a weight
(:func:`problem_samples`), the space text of a row (:func:`space_text`, and
:func:`data_space_text` for the rows of data space) and its provenance
(:func:`provenance`). The module is private: its functions are not part of the
public interface and may change with the engines.

Zero weights
    A sample with a log weight of −∞ has no weight and no effect on any
    estimate. Padding with such samples, as for sources that are in the chain
    for part of a run, must not change a row, so the engines read each problem
    through :func:`problem_samples`, which keeps the samples with a finite log
    weight, and the input hashes and the Kish sizes of a posterior come from
    these samples alone.
"""

import json
import math
from collections.abc import Mapping, Sequence
from typing import Final, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, PRNGKeyArray

from mixed_signals.core.result import Provenance, hash_arrays
from mixed_signals.core.samples import SampleSet, kish_effective_sample_size
from mixed_signals.core.space import ParamSpace
from mixed_signals.prng import format_key, root_key
from mixed_signals.transforms.base import COMPOSITION_SEPARATOR, Transform

type KeyInput = PRNGKeyArray | int
"""A PRNG key, or an integer seed from which the root key follows."""

# * A legacy PRNG key is two uint32 words.
_LEGACY_KEY_SHAPE: Final = (2,)


class Posterior(NamedTuple):
    """One named posterior of every problem, as the engines read it."""

    name: str
    """Name of the posterior."""
    samples: Float[Array, "batch sample param"]
    """Its samples in every problem."""
    log_weights: Float[Array, "batch sample"]
    """Their log weights."""
    n_eff: Float[Array, " batch"]
    """The effective sample size of the posterior in every problem."""
    hashes: tuple[str, ...]
    """SHA-256 digest of the samples and log weights of every problem, as input.

    The digest covers the samples with a finite log weight only.
    """


def read_key(key: object) -> tuple[PRNGKeyArray, int | None]:
    """Return the PRNG key of a call and the seed it came from, if any.

    Args:
        key: An integer seed, which becomes
            :func:`~mixed_signals.prng.root_key` of the seed, or a JAX PRNG
            key, typed or a legacy ``uint32`` key of shape (2,).

    Returns:
        The key and the seed, which is ``None`` when the caller passed a key.

    Raises:
        TypeError: If the key is a bool or any other type than an integer or
            a JAX key.
        ValueError: If the seed lies outside the range of
            :func:`~mixed_signals.prng.root_key`.
    """
    if isinstance(key, int) and not isinstance(key, bool):
        return root_key(key), key
    if isinstance(key, jax.Array) and _is_single_key(key):
        return key, None
    description = type(key).__name__
    if isinstance(key, jax.Array):
        description = f"a JAX array of dtype {key.dtype} and shape {key.shape}"
    raise TypeError(f"key must be a JAX PRNG key or an integer seed, got {description}")


def unique_names(argument: str, names: object, count: int) -> tuple[str, ...]:
    """Return the names as a tuple, one unique string per posterior.

    Args:
        argument: Name of the argument, for the messages.
        names: The names, a sequence that is not a string.
        count: Number of posteriors to name.

    Returns:
        The names, in order.

    Raises:
        ValueError: If the names are not a sequence of strings, their number
            differs from the number of posteriors, or two of them are equal.
    """
    if isinstance(names, str) or not isinstance(names, Sequence):
        raise ValueError(
            f"{argument} must be a sequence of strings, got {type(names).__name__}"
        )
    values = tuple(names)
    if len(values) != count:
        raise ValueError(
            f"{argument} must name each of the {count} posteriors, got"
            f" {len(values)} names"
        )
    if not all(isinstance(value, str) for value in values):
        raise ValueError(f"{argument} must hold strings, got {list(values)!r}")
    if len(set(values)) != len(values):
        raise ValueError(f"{argument} must be unique, got {list(values)!r}")
    return values


def gather_posteriors(
    sample_sets: object,
    names: object,
    space: ParamSpace,
    *,
    transform: Transform | None = None,
) -> tuple[Posterior, ...]:
    """Return the named posteriors of one set, or of several sets joined in order.

    The posteriors keep the order of the sets and, within a set, of its
    posterior axis. The checks and the hashes read the input sets. With a
    transform, the samples, log weights and effective sample sizes of the
    posteriors come from the mapped sets, as
    :meth:`~mixed_signals.transforms.base.Transform.map_sample_set` gives them.

    Args:
        sample_sets: A :class:`~mixed_signals.core.samples.SampleSet`, or a
            sequence of them with the same number of problems and parameters.
            The number of samples may differ between sets.
        names: One unique name per posterior, in the order of the posteriors.
        space: Parameter space of the input samples.
        transform: Transform that maps the samples to the coordinates of the
            engine; the identity when ``None``.

    Returns:
        One :class:`Posterior` per name.

    Raises:
        TypeError: If ``sample_sets`` is neither a ``SampleSet`` nor a
            sequence of them.
        ValueError: If the sequence is empty, the sets differ in their number
            of problems or parameters, the space does not fit the number of
            parameters, or the names do not fit the posteriors.
    """
    sets = _read_sets(sample_sets)
    batch_count, param_count = sets[0].shape.batch, sets[0].shape.param
    for position, entry in enumerate(sets[1:], start=1):
        if (entry.shape.batch, entry.shape.param) != (batch_count, param_count):
            raise ValueError(
                "every set needs the number of problems and parameters of the"
                f" first, (B, D) = ({batch_count}, {param_count}), but set"
                f" {position} has (B, D) = ({entry.shape.batch}, {entry.shape.param})"
            )
    if len(space) != param_count:
        raise ValueError(
            f"the space has {len(space)} parameters, but the samples have {param_count}"
        )
    members = [
        (set_position, posterior)
        for set_position, entry in enumerate(sets)
        for posterior in range(entry.shape.posterior)
    ]
    posterior_names = unique_names("names", names, len(members))
    mapped_sets = (
        sets if transform is None else tuple(map(transform.map_sample_set, sets))
    )
    set_sizes = tuple(_effective_sizes(mapped_set) for mapped_set in mapped_sets)
    return tuple(
        _posterior(
            name,
            sets[set_position],
            mapped_sets[set_position],
            set_sizes[set_position],
            posterior,
        )
        for name, (set_position, posterior) in zip(
            posterior_names, members, strict=True
        )
    )


def problem_samples(
    posterior: Posterior, problem: int
) -> tuple[Float[Array, "selected param"], Float[Array, " selected"]]:
    """Return the samples and log weights of one problem that carry a weight.

    Samples with a log weight of −∞ are dropped, so that padding with zero
    weights never reaches a statistic, a null or a hash. The function reads
    concrete arrays, as the per-problem code of the engines does.

    Args:
        posterior: The posterior, as the engine reads it.
        problem: Index of the problem.

    Returns:
        The kept samples shaped (N_kept, D) and their log weights shaped
        (N_kept,), in their input order.
    """
    return _kept(posterior.samples[problem], posterior.log_weights[problem])


def flat_space(space: ParamSpace) -> ParamSpace:
    """Return the same parameters without the declarations of spheres.

    A sphere's azimuth is then a circle and its polar axis an interval, each
    with its own topology.
    """
    return ParamSpace(parameters=space.parameters)


def settings_text(settings: Mapping[str, object]) -> str:
    """Write estimator settings as a compact JSON object with sorted keys."""
    return json.dumps(settings, sort_keys=True, separators=(",", ":"))


def null_moments(draws: Array) -> tuple[float, float]:
    """Return the mean and the ddof = 1 standard deviation of the finite draws.

    The mean is NaN without a finite draw, and the deviation is NaN with fewer
    than two.
    """
    finite = jnp.isfinite(draws)
    count = int(jnp.sum(finite))
    if count == 0:
        return math.nan, math.nan
    mean = float(jnp.sum(jnp.where(finite, draws, 0.0)) / count)
    if count < 2:
        return mean, math.nan
    squares = jnp.sum(jnp.where(finite, (draws - mean) ** 2, 0.0))
    return mean, float(jnp.sqrt(squares / (count - 1)))


def space_text(*descriptions: str) -> str:
    """Return descriptions of transforms joined as the ``space`` column writes them."""
    return COMPOSITION_SEPARATOR.join(descriptions)


def data_space_text(noise: object) -> str:
    """Return the ``space`` text of a data-space row: ``data(<description>)``.

    The description is the string attribute ``description`` of the noise model;
    a model without a non-empty one gives the text ``data``.
    """
    description = getattr(noise, "description", None)
    if isinstance(description, str) and description:
        return f"data({description})"
    return "data"


def provenance(
    seed: int | None,
    key: PRNGKeyArray,
    hashes: Sequence[str],
    config_hash: str | None,
) -> Provenance:
    """Return the provenance of a row of an engine that received a key or a seed."""
    return Provenance.collect(
        seed=seed,
        prng_key=format_key(key),
        input_hashes=hashes,
        config_hash=config_hash,
    )


def _is_single_key(key: jax.Array) -> bool:
    """Return whether an array is one PRNG key, typed or legacy."""
    if jnp.issubdtype(key.dtype, jax.dtypes.prng_key):
        return key.shape == ()
    return key.dtype == jnp.uint32 and key.shape == _LEGACY_KEY_SHAPE


def _read_sets(sample_sets: object) -> tuple[SampleSet, ...]:
    """Return one set, or a sequence of sets, as a non-empty tuple of sets."""
    if isinstance(sample_sets, SampleSet):
        return (sample_sets,)
    if isinstance(sample_sets, str) or not isinstance(sample_sets, Sequence):
        raise TypeError(
            "sample_sets must be a SampleSet or a sequence of them, got"
            f" {type(sample_sets).__name__}"
        )
    if not sample_sets:
        raise ValueError("sample_sets must hold at least one SampleSet")
    for position, entry in enumerate(sample_sets):
        if not isinstance(entry, SampleSet):
            raise TypeError(
                f"sample_sets must hold SampleSet objects, but entry {position} is"
                f" {type(entry).__name__}"
            )
    return tuple(sample_sets)


def _kept(
    samples: Float[Array, "sample param"], log_weights: Float[Array, " sample"]
) -> tuple[Float[Array, "selected param"], Float[Array, " selected"]]:
    """Return the samples with a finite log weight and their log weights."""
    kept = jnp.flatnonzero(log_weights > -jnp.inf)
    if kept.shape[0] == log_weights.shape[0]:
        return samples, log_weights
    return samples[kept], log_weights[kept]


def _effective_sizes(
    sample_set: SampleSet,
) -> Float[Array, "batch posterior"]:
    """Return ``n_eff``, or the Kish size of the samples that carry a weight.

    A set whose every posterior keeps the same number of samples gives its Kish
    sizes as one array of the kept log weights, so that padding with zero
    weights leaves them bit for bit as without it. Where the kept counts
    differ, each posterior of each problem gives its own Kish size.
    """
    if sample_set.n_eff is not None:
        return sample_set.n_eff
    log_weights = sample_set.log_weights
    kept = log_weights > -jnp.inf
    counts = np.asarray(jnp.sum(kept, axis=-1))
    batch_size, posterior_count, _ = log_weights.shape
    if counts.min() == counts.max():
        return kish_effective_sample_size(
            log_weights[kept].reshape(batch_size, posterior_count, int(counts.min()))
        )
    sizes = [
        kish_effective_sample_size(
            log_weights[problem, posterior][kept[problem, posterior]]
        )
        for problem in range(batch_size)
        for posterior in range(posterior_count)
    ]
    return jnp.stack(sizes).reshape(batch_size, posterior_count)


def _posterior(
    name: str,
    input_set: SampleSet,
    mapped_set: SampleSet,
    sizes: Float[Array, "batch posterior"],
    posterior: int,
) -> Posterior:
    """Return one posterior: arrays of the mapped set, hashes of the input set."""
    input_samples = input_set.samples[:, posterior]
    input_log_weights = input_set.log_weights[:, posterior]
    return Posterior(
        name=name,
        samples=mapped_set.samples[:, posterior],
        log_weights=mapped_set.log_weights[:, posterior],
        n_eff=sizes[:, posterior],
        hashes=tuple(
            hash_arrays(*_kept(input_samples[problem], input_log_weights[problem]))
            for problem in range(input_samples.shape[0])
        ),
    )
