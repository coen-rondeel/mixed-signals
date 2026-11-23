"""Comparison of two holders of LISA sources, one problem per matched pair.

:func:`compare_sources` pairs the sources of two holders, as catpylisa matches
the catalogs of two global fits, and runs
:func:`~mixed_signals.engines.compare.compare` with one problem per pair, so
that every row of the result belongs to a pair of sources. It refuses holders
that disagree on a convention, a parameter set or a constant, and says how to
align them.

Matches
    A matched index array pairs the sources. Preferably it is shaped (M, 2),
    with an index in the first holder and one in the second; it may also have
    one entry per source of the first holder, with its index in the second or −1
    for none. Without one, the holders pair in order and must have the same
    length. A source may appear in several pairs. :func:`match_indices` builds
    the pairs from source ids, the form of the match tables of catpylisa.
Checks
    Both holders must be of one source type, with different names, and record
    equal conventions
    (:func:`~mixed_signals.gw.lisa.conventions.require_equal_conventions`).
    Their sampled parameters must agree, or ``parameters`` names a subset that
    both sample; a parameter that one holder samples and the other holds
    constant raises unless it is left out. The compared parameters must have
    equal names, labels, units, topologies and spheres; prior bounds may differ,
    and the space of the comparison takes the union of the two bounds, so that
    no sample lies outside it. The constants of every matched pair agree within
    a relative 10⁻¹².
"""

import math
from collections.abc import Iterable, Sequence
from typing import Any, Final, NamedTuple

import numpy as np
import polars as pl

from mixed_signals.core.result import Result
from mixed_signals.core.space import Parameter, ParamSpace
from mixed_signals.core.topology import Topology
from mixed_signals.engines import compare as run_compare
from mixed_signals.engines._common import KeyInput
from mixed_signals.errors import ConventionError
from mixed_signals.gw.lisa.conventions import require_equal_conventions
from mixed_signals.gw.lisa.sources import LisaSources, _decode_id

__all__ = ["SourceComparison", "compare_sources", "match_indices"]

_SETTING_RELATIVE_TOLERANCE: Final = 1e-12


class SourceComparison(NamedTuple):
    """The result of :func:`compare_sources` and the pairs that it compared."""

    result: Result
    """The result table of ``compare``, with one problem per pair."""
    pairs: pl.DataFrame
    """One row per problem: ``problem``, ``first_index``, ``first_source_id``,
    ``second_index`` and ``second_source_id``."""


def match_indices(
    first: LisaSources,
    second: LisaSources,
    source_id_pairs: Iterable[tuple[Any, Any]],
) -> np.ndarray:
    """Return the matched index array of pairs of source ids.

    Args:
        first: The first holder.
        second: The second holder.
        source_id_pairs: Pairs of a source id of the first holder and one of the
            second, such as a match table of catpylisa. An id is a string,
            UTF-8 bytes or an integer, which becomes its decimal string.

    Returns:
        The indices shaped (M, 2), the pairs in the order given.

    Raises:
        ValueError: If no pair is given or an id is unknown to its holder.
    """
    pairs = [(_decode_id(left), _decode_id(right)) for left, right in source_id_pairs]
    if not pairs:
        raise ValueError("give at least one pair of source ids")
    first_positions = {
        source: position for position, source in enumerate(first.source_ids)
    }
    second_positions = {
        source: position for position, source in enumerate(second.source_ids)
    }
    unknown_first = sorted({left for left, _ in pairs if left not in first_positions})
    unknown_second = sorted(
        {right for _, right in pairs if right not in second_positions}
    )
    if unknown_first or unknown_second:
        raise ValueError(
            f"unknown source ids: {unknown_first} in the first holder and"
            f" {unknown_second} in the second"
        )
    return np.asarray(
        [(first_positions[left], second_positions[right]) for left, right in pairs],
        dtype=np.int64,
    )


def compare_sources(
    first: LisaSources,
    second: LisaSources,
    *,
    matches: object = None,
    key: KeyInput,
    parameters: Sequence[str] | None = None,
    **compare_keywords: Any,
) -> SourceComparison:
    """Compare the matched sources of two holders, one problem per pair.

    The module docstring describes the matches and the checks. The sets of the
    two holders enter ``compare`` as two posteriors named after the holders, so
    mode A takes ``reference=`` as the name of one of them, and the other
    keywords pass through.

    Args:
        first: The first holder.
        second: The second holder, of the same type and with another name.
        matches: ``None`` to pair the sources in order, an integer array shaped
            (M, 2) of pairs of indices, or one integer per source of the first
            holder, its index in the second or −1 for none.
        key: A JAX PRNG key, or an integer seed.
        parameters: The sampled parameters to compare; every sampled parameter
            when ``None``.
        **compare_keywords: Keywords of
            :func:`~mixed_signals.engines.compare.compare`, such as ``reference``,
            ``metrics``, ``draws`` and ``cap``.

    Returns:
        The result and the table of pairs, with the problem of each.

    Raises:
        ConventionError: If the holders are of different types, differ in a
            convention, differ in their sampled parameters or in the definition
            of a compared parameter, or a matched pair differs in a setting.
        ValueError: If the holders share a name, the matches or the parameters
            are invalid, or ``compare`` rejects its arguments.
    """
    if type(first) is not type(second):
        raise ConventionError(
            f"the holders are of different types, {type(first).__name__} and"
            f" {type(second).__name__}"
        )
    if first.name == second.name:
        raise ValueError(
            f"the holders need different names, both are {first.name!r}; the name"
            " labels the posterior"
        )
    require_equal_conventions(
        first.conventions, second.conventions, epoch_shift=first.can_shift_epoch
    )
    pairs = _read_matches(matches, first, second)
    names = _compared_names(first, second, parameters)
    space = _compared_space(first, second, names)
    _check_settings(first, second, pairs)
    sets = [
        holder.values.select(
            problems=pairs[:, side],
            params=[holder.full_space.index(name) for name in names],
        )
        for side, holder in enumerate((first, second))
    ]
    result = run_compare(
        sets, [first.name, second.name], space, key=key, **compare_keywords
    )
    table = pl.DataFrame(
        {
            "problem": list(range(len(pairs))),
            "first_index": pairs[:, 0].tolist(),
            "first_source_id": [first.source_ids[position] for position in pairs[:, 0]],
            "second_index": pairs[:, 1].tolist(),
            "second_source_id": [
                second.source_ids[position] for position in pairs[:, 1]
            ],
        },
        schema={
            "problem": pl.Int64(),
            "first_index": pl.Int64(),
            "first_source_id": pl.String(),
            "second_index": pl.Int64(),
            "second_source_id": pl.String(),
        },
    )
    return SourceComparison(result, table)


def _read_matches(
    matches: object, first: LisaSources, second: LisaSources
) -> np.ndarray:
    """Return the matches as pairs shaped (M, 2), after checking them."""
    if matches is None:
        if len(first) != len(second):
            raise ValueError(
                f"without matches the holders pair in order, but they hold {len(first)}"
                f" and {len(second)} sources"
            )
        return np.stack([np.arange(len(first))] * 2, axis=1)
    array = np.asarray(matches)
    if not np.issubdtype(array.dtype, np.integer):
        raise ValueError(f"matches must hold integers, got dtype {array.dtype}")
    if array.ndim == 1:
        if array.shape[0] != len(first):
            raise ValueError(
                f"one match per source of the first holder, {len(first)}, was"
                f" expected, got {array.shape[0]}"
            )
        if (array < -1).any():
            raise ValueError("a source without a match is marked by -1, not lower")
        pairs = np.stack([np.arange(len(first)), array], axis=1)[array >= 0]
    elif array.ndim == 2 and array.shape[1] == 2:
        pairs = array
    else:
        raise ValueError(
            f"matches must be shaped (M, 2) or (B,), got shape {array.shape}"
        )
    if pairs.shape[0] == 0:
        raise ValueError("the matches pair no source")
    for side, holder in enumerate((first, second)):
        outside = (pairs[:, side] < 0) | (pairs[:, side] >= len(holder))
        if outside.any():
            raise ValueError(
                f"match index {int(pairs[outside, side][0])} lies outside the"
                f" {len(holder)} sources of holder {holder.name!r}"
            )
    return pairs.astype(np.int64)


def _compared_names(
    first: LisaSources, second: LisaSources, parameters: Sequence[str] | None
) -> list[str]:
    """Return the sampled parameters to compare, which both holders must sample."""
    sampled = [set(holder.space.names) for holder in (first, second)]
    if parameters is None:
        if sampled[0] != sampled[1]:
            raise ConventionError(
                "the holders sample different parameters: "
                + _describe_difference(first, second, sampled)
                + "; leave them out with parameters="
            )
        return list(first.space.names)
    names = list(parameters)
    if not names or len(set(names)) != len(names):
        raise ValueError(
            f"parameters must be a non-empty list of unique names, got {names}"
        )
    for name in names:
        if any(name not in holder.full_space.names for holder in (first, second)):
            raise ValueError(f"parameter {name!r} is not in both holders")
        if name not in sampled[0] | sampled[1]:
            raise ValueError(
                f"parameter {name!r} is a setting in both holders, so there is no"
                " distribution to compare"
            )
        if name not in sampled[0] & sampled[1]:
            raise ConventionError(
                f"parameter {name!r} is sampled in one holder and a setting in the"
                " other"
            )
    return names


def _describe_difference(
    first: LisaSources, second: LisaSources, sampled: list[set[str]]
) -> str:
    """Name the parameters that only one holder samples, and what the other holds."""
    parts = []
    for holder, other, own, others in (
        (first, second, sampled[0], sampled[1]),
        (second, first, sampled[1], sampled[0]),
    ):
        for name in sorted(own - others):
            elsewhere = "a setting" if name in other.full_space.names else "absent"
            parts.append(
                f"{name!r} is sampled in {holder.name!r} and {elsewhere} in"
                f" {other.name!r}"
            )
    return "; ".join(parts)


def _kind(parameter: Parameter) -> tuple[object, ...]:
    """Return what two compared parameters must share: not the bounds of a line."""
    on_circle = parameter.topology is Topology.CIRCLE
    return (
        parameter.name,
        parameter.label,
        parameter.unit,
        on_circle,
        parameter.bounds if on_circle else None,
    )


def _compared_space(
    first: LisaSources, second: LisaSources, names: list[str]
) -> ParamSpace:
    """Return the space of the compared parameters, with the union of the bounds."""
    parameters = []
    for name in names:
        own, other = first.full_space[name], second.full_space[name]
        if _kind(own) != _kind(other):
            raise ConventionError(
                f"parameter {name!r} is defined differently in the two holders:"
                f" {own} and {other}"
            )
        if own.topology is Topology.CIRCLE:
            parameters.append(own)
            continue
        lower = min(own.bounds[0], other.bounds[0])
        upper = max(own.bounds[1], other.bounds[1])
        topology = (
            Topology.REAL
            if math.isinf(lower) and math.isinf(upper)
            else Topology.INTERVAL
        )
        parameters.append(
            own.model_copy(update={"topology": topology, "bounds": (lower, upper)})
        )
    kept = set(names)
    spheres = [
        tuple(
            sphere
            for sphere in holder.full_space.spheres
            if {sphere.azimuth, sphere.polar} <= kept
        )
        for holder in (first, second)
    ]
    if spheres[0] != spheres[1]:
        raise ConventionError(
            "the spheres of the compared parameters differ:"
            f" {spheres[0]} and {spheres[1]}"
        )
    return ParamSpace(parameters=tuple(parameters), spheres=spheres[0])


def _check_settings(first: LisaSources, second: LisaSources, pairs: np.ndarray) -> None:
    """Check that the constants of every matched pair agree within a relative 10⁻¹²."""
    first_settings, second_settings = first.settings, second.settings
    shared = [
        name
        for name in first_settings.columns
        if name != "source_id" and name in second_settings.columns
    ]
    for name in shared:
        first_values = first_settings[name].to_list()
        second_values = second_settings[name].to_list()
        for left, right in pairs.tolist():
            if not math.isclose(
                first_values[left],
                second_values[right],
                rel_tol=_SETTING_RELATIVE_TOLERANCE,
            ):
                raise ConventionError(
                    f"the matched sources {first.source_ids[left]!r} and"
                    f" {second.source_ids[right]!r} differ in the setting {name!r}:"
                    f" {first_values[left]!r} and {second_values[right]!r}"
                )
