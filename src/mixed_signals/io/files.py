"""NumPy files and tables as sample sets and chains.

:func:`sample_set_from_npz`, :func:`sample_set_from_table` and
:func:`chains_from_npz` back the classmethods
:meth:`SampleSet.from_npz <mixed_signals.core.samples.SampleSet.from_npz>`,
:meth:`SampleSet.from_table <mixed_signals.core.samples.SampleSet.from_table>`
and :meth:`Chains.from_npz <mixed_signals.core.chains.Chains.from_npz>`.

NumPy files
    ``numpy.load`` reads a ``.npy`` array or a ``.npz`` archive with
    ``allow_pickle=False``, so an object array raises an error and no pickled
    code runs. A ``.npy`` file holds the samples alone. A ``.npz`` archive
    holds arrays under the keyword names of the container: ``samples``,
    ``log_weights``, ``n_eff``, ``log_like``, ``log_prior`` and ``log_q`` for a
    sample set, and ``samples``, ``betas``, ``log_like``, ``log_prior``,
    ``accepted``, ``proposed``, ``swaps_accepted`` and ``swaps_proposed`` for
    chains. Unknown keys raise an error, so a misspelt name never drops an
    array silently. The files carry no parameter names, so their last axis is
    read in order and must match the parameter space of the comparison.
Tables
    A parquet or CSV table holds one column per parameter and, if the posterior
    has weights, one column of natural-log weights. The columns are read by
    name: the names that the caller passes, which are the names of the
    parameter space, choose the columns in their order. The table may hold more
    columns, in any order, so a space may name a subset of the parameters.
    A missing name raises an error that lists the columns of the table.
One posterior per file
    A flat reader returns a sample set with one posterior, K = 1. Samples shaped
    (N, D) are one problem, and samples shaped (B, N, D) are B problems; the
    posterior axis is inserted for the caller. Per-sample arrays have the
    shape of the samples without the parameter axis, and ``n_eff`` is one number
    for one problem or one number per problem. The container then checks sizes
    and values as usual.
Precision
    Every field is stored in float64. A table column or an array narrower than
    float64 raises a :class:`~mixed_signals.errors.PrecisionWarning` before
    the cast, because the cast cannot restore the precision lost before it.
"""

import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Final

import jax.numpy as jnp
import numpy as np
import polars as pl
from jaxtyping import Array

from mixed_signals.core.chains import Chains
from mixed_signals.core.samples import SampleSet, _warn_narrow_floats
from mixed_signals.io.arrays import to_jax

__all__ = ["chains_from_npz", "sample_set_from_npz", "sample_set_from_table"]

_SAMPLE_SET_KEYS: Final = (
    "samples",
    "log_weights",
    "n_eff",
    "log_like",
    "log_prior",
    "log_q",
)
_CHAINS_KEYS: Final = (
    "samples",
    "betas",
    "log_like",
    "log_prior",
    "accepted",
    "proposed",
    "swaps_accepted",
    "swaps_proposed",
)
_FLAT_LAYOUTS: Final = "(N, D) for one problem or (B, N, D) for B problems"


def sample_set_from_npz(
    path: str | os.PathLike[str], *, n_eff: object = None
) -> SampleSet:
    """Read a sample set of one posterior from a ``.npy`` or ``.npz`` file.

    :meth:`SampleSet.from_npz <mixed_signals.core.samples.SampleSet.from_npz>`
    calls this function; the module docstring describes the layout.

    Args:
        path: A ``.npy`` file of samples, or a ``.npz`` archive with the
            keyword names of :class:`~mixed_signals.core.samples.SampleSet`;
            the suffix tells them apart, in any case.
        n_eff: The effective sample size, a number for one problem or one
            number per problem, for a file that holds none.

    Returns:
        The set, shaped (B, 1, N, D).

    Raises:
        ValueError: If the suffix is neither ``.npy`` nor ``.npz``, the archive
            lacks ``samples`` or holds other keys, ``n_eff`` is given for an
            archive that holds one, an array has the wrong rank, or the
            container rejects a size or a value. An object array also raises
            ValueError, because the file is read without pickle.
        TypeError: If an array is not real-valued.
        OSError: If the file cannot be read.

    Warns:
        PrecisionWarning: If an array is narrower than float64.
    """
    arrays: dict[str, object] = dict(_read_numpy_file(path, (".npy", ".npz")))
    _check_keys(arrays, _SAMPLE_SET_KEYS, path)
    if n_eff is not None:
        if "n_eff" in arrays:
            raise ValueError(
                f"{os.fspath(path)} holds n_eff, so the n_eff argument would"
                " overwrite it; give one of the two"
            )
        arrays["n_eff"] = n_eff
    return _flat_sample_set(arrays)


def sample_set_from_table(
    path: str | os.PathLike[str],
    names: Sequence[str],
    *,
    log_weights: str | None = None,
    n_eff: float | None = None,
) -> SampleSet:
    """Read a sample set of one problem and one posterior from a table.

    :meth:`SampleSet.from_table <mixed_signals.core.samples.SampleSet.from_table>`
    calls this function; the module docstring describes the layout.

    Args:
        path: A ``.parquet`` or ``.csv`` file with one column per parameter.
        names: The parameter columns, in the order of the last sample axis,
            which are the names of the parameter space.
        log_weights: The column of natural-log weights, not among ``names``;
            ``None`` for equal weights.
        n_eff: The effective sample size; the Kish value of the weights when
            ``None``.

    Returns:
        The set, shaped (1, 1, N, D).

    Raises:
        ValueError: If the suffix is neither ``.parquet`` nor ``.csv``, the
            names are empty, repeated or not strings, the log weights are among
            the names, a column is missing, a selected column holds a null, or
            the container rejects a size or a value.
        TypeError: If a selected column is not integer or float.
        OSError: If the file cannot be read.

    Warns:
        PrecisionWarning: If a selected column is narrower than float64.
    """
    parameter_names = _check_names(names)
    if log_weights is not None and log_weights in parameter_names:
        raise ValueError(
            f"log_weights names the column {log_weights!r}, which is also a"
            " parameter name"
        )
    table = _read_table(path)
    wanted = [*parameter_names, *([] if log_weights is None else [log_weights])]
    missing = [name for name in wanted if name not in table.columns]
    if missing:
        raise ValueError(
            f"{os.fspath(path)} has no columns {', '.join(map(repr, missing))}; its"
            f" columns are {', '.join(map(repr, table.columns))}"
        )
    columns = {name: _numeric_column(table, name) for name in wanted}
    _warn_narrow_arrays(columns, "column")
    arrays: dict[str, object] = {
        "samples": np.stack(
            [columns[name].astype(np.float64) for name in parameter_names], axis=-1
        ),
        "n_eff": n_eff,
    }
    if log_weights is not None:
        arrays["log_weights"] = columns[log_weights].astype(np.float64)
    return _flat_sample_set(arrays)


def chains_from_npz(path: str | os.PathLike[str]) -> Chains:
    """Read chains from a ``.npz`` archive.

    :meth:`Chains.from_npz <mixed_signals.core.chains.Chains.from_npz>` calls
    this function; the module docstring describes the keys.

    Args:
        path: A ``.npz`` archive with the keyword names of
            :class:`~mixed_signals.core.chains.Chains`. Its ``samples`` have
            the eryn layout (step, temp, walker, leaf, param), or (step,
            walker, param) for one temperature.

    Returns:
        The chains.

    Raises:
        ValueError: If the suffix is not ``.npz``, the archive lacks
            ``samples`` or holds other keys, or the container rejects a shape
            or a value. An object array also raises ValueError, because the
            file is read without pickle.
        TypeError: If an array is not real-valued.
        NotImplementedError: If the chains hold several leaves.
        OSError: If the file cannot be read.

    Warns:
        PrecisionWarning: If an array is narrower than float64.
    """
    arrays = _read_numpy_file(path, (".npz",))
    _check_keys(arrays, _CHAINS_KEYS, path)
    return Chains(**arrays)


def _check_names(names: object) -> tuple[str, ...]:
    """Return the parameter names as a non-empty tuple of unique strings."""
    if isinstance(names, str) or not isinstance(names, Sequence):
        raise ValueError(
            f"names must be a sequence of strings, got {type(names).__name__}"
        )
    values = tuple(names)
    if not values:
        raise ValueError("names must not be empty")
    if not all(isinstance(value, str) for value in values):
        raise ValueError(f"names must hold strings, got {list(values)!r}")
    if len(set(values)) != len(values):
        raise ValueError(f"names must be unique, got {list(values)!r}")
    return values


def _warn_narrow_arrays(arrays: Mapping[str, np.ndarray], kind: str) -> None:
    """Raise the precision warning once for the arrays narrower than float64.

    Only the narrow arrays reach JAX, so that a wide table is not copied for
    the check.
    """
    narrow = {
        f"{kind} {name!r}": to_jax(values)
        for name, values in arrays.items()
        if values.dtype.kind == "f" and values.dtype.itemsize < 8
    }
    _warn_narrow_floats(narrow)


def _flat_sample_set(arrays: Mapping[str, object]) -> SampleSet:
    """Build the sample set of one posterior from flat arrays.

    Args:
        arrays: The fields of the set by keyword name: ``samples`` shaped (N, D)
            or (B, N, D), per-sample arrays of one axis fewer, and ``n_eff`` as
            a number or one number per problem. ``None`` entries are left out.

    Returns:
        The set, shaped (B, 1, N, D).

    Raises:
        ValueError: If an array has the wrong rank, or the container rejects a
            size or a value.
    """
    given = {name: to_jax(value) for name, value in arrays.items() if value is not None}
    samples = given["samples"]
    if samples.ndim not in (2, 3):
        raise ValueError(
            f"samples must be shaped {_FLAT_LAYOUTS}, got shape {samples.shape}"
        )
    # * One problem needs both leading axes, B problems only the posterior axis.
    axes = (0, 1) if samples.ndim == 2 else 1
    batch_count = 1 if samples.ndim == 2 else samples.shape[0]
    fields: dict[str, Array] = {}
    for name, array in given.items():
        if name == "n_eff":
            fields[name] = _per_problem(array, batch_count)
        elif name == "samples" or array.ndim == samples.ndim - 1:
            fields[name] = jnp.expand_dims(array, axes)
        else:
            raise ValueError(
                f"{name} must have one axis fewer than the samples, so shape"
                f" {samples.shape[:-1]}, got shape {array.shape}"
            )
    return SampleSet(**fields)


def _per_problem(n_eff: Array, batch_count: int) -> Array:
    """Return n_eff shaped (B, 1) from a number, or from one number per problem."""
    values = n_eff.reshape(1) if n_eff.ndim == 0 else n_eff
    if values.shape != (batch_count,):
        raise ValueError(
            f"n_eff must be one number for each of the {batch_count} problems,"
            f" got shape {n_eff.shape}"
        )
    return values[:, None]


def _read_numpy_file(
    path: str | os.PathLike[str], suffixes: tuple[str, ...]
) -> dict[str, np.ndarray]:
    """Return the arrays of a ``.npz`` archive, or a ``.npy`` file as ``samples``."""
    file_path = Path(path)
    suffix = file_path.suffix.lower()
    if suffix not in suffixes:
        raise ValueError(
            f"{file_path} must end in {' or '.join(suffixes)}, got {suffix!r}"
        )
    if suffix == ".npy":
        return {"samples": np.load(file_path, allow_pickle=False)}
    with np.load(file_path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def _check_keys(
    arrays: Mapping[str, object],
    allowed: Sequence[str],
    path: str | os.PathLike[str],
) -> None:
    """Raise unless the keys of an archive are allowed and include ``samples``."""
    unknown = sorted(set(arrays) - set(allowed))
    if unknown:
        raise ValueError(
            f"{os.fspath(path)} holds the unknown keys"
            f" {', '.join(map(repr, unknown))}; the keys are"
            f" {', '.join(map(repr, allowed))}"
        )
    if "samples" not in arrays:
        holds = ", ".join(map(repr, arrays)) or "no arrays"
        raise ValueError(
            f"{os.fspath(path)} needs an array under the key 'samples'; it"
            f" holds {holds}"
        )


def _read_table(path: str | os.PathLike[str]) -> pl.DataFrame:
    """Read a parquet or CSV table, told apart by the suffix."""
    file_path = Path(path)
    suffix = file_path.suffix.lower()
    if suffix == ".parquet":
        return pl.read_parquet(file_path)
    if suffix == ".csv":
        return pl.read_csv(file_path)
    raise ValueError(f"{file_path} must end in .parquet or .csv, got {suffix!r}")


def _numeric_column(table: pl.DataFrame, name: str) -> np.ndarray:
    """Return a column as a NumPy array, after checking its type and its nulls."""
    column = table[name]
    if not (column.dtype.is_integer() or column.dtype.is_float()):
        raise TypeError(
            f"column {name!r} must be integer or float, got type {column.dtype}"
        )
    if column.null_count() > 0:
        raise ValueError(f"column {name!r} holds {column.null_count()} null values")
    return column.to_numpy()
