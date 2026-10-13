"""Result schema and provenance.

A :class:`Result` holds one long table, with one row per problem, pair, metric,
estimator, space and parameter subset. Every engine writes rows with the same
columns, so the results of different engines and runs concatenate.

``problem``
    Index of the problem on the batch axis of the samples.
``pair``
    Names of the posteriors, in the order of the metric's arguments: one name
    for a diagnostic row, two for a comparison.
``metric``
    The functional, such as ``"kl"``.
``estimator``
    How the functional is estimated, such as ``"knn"``, or ``"empirical"`` for
    a metric computed from the samples directly.
``estimator_settings``
    The estimator settings as a JSON object, such as ``{"k": 5}``.
``space``
    The transform chain in which the metric was computed.
``parameters``
    Names of the parameters that the row covers.
``value``
    The metric value.
``uncertainty``, ``uncertainty_method``
    The uncertainty of the value, and how it was estimated; see
    :class:`UncertaintyMethod`.
``null_mean``, ``null_std``, ``null_draws``
    Mean, standard deviation and draws of the metric under the null.
``null_recipe``
    The construction of the null, its number of draws and the sample size of
    each side.
``p_value``
    Empirical p-value of the value under the null, with the +1 correction of
    Phipson and Smyth (2010).
``sigma_sig``
    The p-value as a one-sided Gaussian significance, Φ⁻¹(1 − p)
    (Cowan et al. 2011).
``sigma_eff``
    The shift between two unit Gaussians that gives the same null-subtracted
    value, in units of the whitening σ.
``r_eff``
    The width ratio between two unit Gaussians that gives the same
    null-subtracted value, for rows that measure only a difference in width.
``warnings``
    Package warnings raised for the row, as
    :class:`~mixed_signals.errors.WarningRecord` entries.
``provenance``
    Where the row comes from; see :class:`Provenance`.

A null entry means that a column does not apply to the row, such as ``r_eff``
on a row that carries ``sigma_eff``. A NaN value means that the method is
undefined for the inputs of the row, at a hard limit of its validity domain.

References:
    :cite:t:`phipson2010permutation`; :cite:t:`cowan2011asymptotic`.
"""

import hashlib
import os
import re
from collections.abc import Iterable, Mapping, Sequence
from enum import StrEnum
from importlib.metadata import version
from pathlib import Path
from typing import Any, Final

import jax
import numpy as np
import polars as pl
from jax.typing import ArrayLike
from pydantic import BaseModel, ConfigDict, Field, field_validator

from mixed_signals.errors import SchemaError, WarningRecord
from mixed_signals.prng import KEY_TEXT_PATTERN, SEED_LIMIT

SHA256_HEX_PATTERN: Final = re.compile(r"[0-9a-f]{64}")
"""Pattern of a SHA-256 hex digest, 64 lowercase hex characters."""

NULL_RECIPE_DTYPE: Final = pl.Struct(
    {"construction": pl.String(), "draws": pl.Int64(), "sample_size": pl.Int64()}
)
"""Type of the ``null_recipe`` column."""

WARNING_DTYPE: Final = pl.Struct({"category": pl.String(), "message": pl.String()})
"""Type of one entry of the ``warnings`` column."""

PROVENANCE_DTYPE: Final = pl.Struct(
    {
        "package_version": pl.String(),
        "seed": pl.Int64(),
        "prng_key": pl.String(),
        "input_hashes": pl.List(pl.String()),
        "jax_version": pl.String(),
        "device_platform": pl.String(),
        "config_hash": pl.String(),
    }
)
"""Type of the ``provenance`` column."""

RESULT_SCHEMA: Final = pl.Schema(
    {
        "problem": pl.Int64(),
        "pair": pl.List(pl.String()),
        "metric": pl.String(),
        "estimator": pl.String(),
        "estimator_settings": pl.String(),
        "space": pl.String(),
        "parameters": pl.List(pl.String()),
        "value": pl.Float64(),
        "uncertainty": pl.Float64(),
        "uncertainty_method": pl.String(),
        "null_mean": pl.Float64(),
        "null_std": pl.Float64(),
        "null_draws": pl.List(pl.Float64()),
        "null_recipe": NULL_RECIPE_DTYPE,
        "p_value": pl.Float64(),
        "sigma_sig": pl.Float64(),
        "sigma_eff": pl.Float64(),
        "r_eff": pl.Float64(),
        "warnings": pl.List(WARNING_DTYPE),
        "provenance": PROVENANCE_DTYPE,
    }
)
"""Column names and types of every result table, in order.

:meta hide-value:
"""

REQUIRED_COLUMNS: Final = (
    "problem",
    "pair",
    "metric",
    "estimator",
    "estimator_settings",
    "space",
    "parameters",
    "value",
    "warnings",
    "provenance",
)
"""Columns that must not be null, because they identify the row and hold its value."""


class UncertaintyMethod(StrEnum):
    """How the ``uncertainty`` column of a row was estimated.

    ``null_std``
        The standard deviation of the null draws, used as an approximate
        uncertainty.
    ``subsampling_bootstrap``
        The spread of the metric over half-samples drawn without replacement
        (Politis, Romano and Wolf 1999), so that no estimator sees duplicate
        points.

    References:
        :cite:t:`politis1999subsampling`.
    """

    NULL_STD = "null_std"
    SUBSAMPLING_BOOTSTRAP = "subsampling_bootstrap"


class Provenance(BaseModel):
    """Where a result row comes from.

    Attributes:
        package_version: Version of mixed-signals.
        seed: Config seed of the run, or ``None`` when the caller passed a key.
        prng_key: Key that the engine received, from which the keys of the row
            derive, as written by :func:`~mixed_signals.prng.format_key`.
        input_hashes: SHA-256 hex digests of the input posteriors, one per name
            in ``pair``, from :func:`hash_arrays`.
        jax_version: Version of JAX. For a fixed key, the values that JAX
            samplers return may change between JAX releases.
        device_platform: JAX platform of the computation, such as ``"cpu"`` or
            ``"gpu"``.
        config_hash: SHA-256 hex digest of the pipeline config, or ``None``
            outside the pipeline.

    References:
        :cite:t:`jax2022api`; :cite:t:`nist2015secure`.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    package_version: str
    seed: int | None = Field(default=None, ge=0, lt=SEED_LIMIT)
    prng_key: str | None = Field(default=None, pattern=f"^{KEY_TEXT_PATTERN.pattern}$")
    input_hashes: tuple[str, ...] = ()
    jax_version: str
    device_platform: str
    config_hash: str | None = None

    @field_validator("input_hashes")
    @classmethod
    def _check_input_hashes(cls, digests: tuple[str, ...]) -> tuple[str, ...]:
        for digest in digests:
            _check_sha256(digest)
        return digests

    @field_validator("config_hash")
    @classmethod
    def _check_config_hash(cls, digest: str | None) -> str | None:
        if digest is not None:
            _check_sha256(digest)
        return digest

    @classmethod
    def collect(
        cls,
        *,
        seed: int | None = None,
        prng_key: str | None = None,
        input_hashes: Sequence[str] = (),
        config_hash: str | None = None,
    ) -> "Provenance":
        """Build the provenance of a row in the current process.

        The package version, the JAX version and the device platform come from
        the running process.

        Args:
            seed: Config seed of the run, if any.
            prng_key: Key that the engine received, as text.
            input_hashes: SHA-256 hex digests of the input posteriors.
            config_hash: SHA-256 hex digest of the pipeline config, if any.

        Returns:
            The validated provenance.
        """
        return cls(
            package_version=version("mixed-signals"),
            seed=seed,
            prng_key=prng_key,
            input_hashes=tuple(input_hashes),
            jax_version=jax.__version__,
            device_platform=jax.default_backend(),
            config_hash=config_hash,
        )

    def to_struct(self) -> dict[str, Any]:
        """Return the provenance as a value of the ``provenance`` column."""
        return self.model_dump(mode="json")


def hash_arrays(*arrays: ArrayLike) -> str:
    """Return one SHA-256 hex digest over one or more arrays.

    For each array in turn, the digest covers a header of its NumPy type string
    in little-endian byte order and its shape, written as ``"<f8:1000,3;"``,
    followed by its bytes in C order. Equal values with a different type or
    shape therefore give a different digest.

    Args:
        *arrays: Arrays to hash, in order; JAX arrays are copied to the host.

    Returns:
        The digest as 64 lowercase hex characters.

    Raises:
        TypeError: If an array has the object type, whose bytes are pointers.

    References:
        :cite:t:`nist2015secure`.
    """
    digest = hashlib.sha256()
    for array in arrays:
        host_array = np.ascontiguousarray(np.asarray(array))
        if host_array.dtype.hasobject:
            raise TypeError("cannot hash arrays of Python objects")
        little_endian = host_array.astype(
            host_array.dtype.newbyteorder("<"), copy=False
        )
        shape_text = ",".join(str(size) for size in little_endian.shape)
        digest.update(f"{little_endian.dtype.str}:{shape_text};".encode())
        digest.update(little_endian.data)
    return digest.hexdigest()


class Result:
    """A result table with the columns of :data:`RESULT_SCHEMA`.

    Args:
        table: Table with exactly the columns of :data:`RESULT_SCHEMA`, in
            order, and no nulls in :data:`REQUIRED_COLUMNS`.

    Raises:
        SchemaError: If the table does not match the schema.
    """

    __slots__ = ("_table",)

    def __init__(self, table: pl.DataFrame) -> None:
        """Check the table against the schema and wrap it."""
        problems = _schema_problems(table)
        if problems:
            raise SchemaError(
                "the table does not match the result schema: " + "; ".join(problems)
            )
        self._table = table

    @classmethod
    def empty(cls) -> "Result":
        """Return a result without rows."""
        return cls(pl.DataFrame(schema=RESULT_SCHEMA))

    @classmethod
    def from_rows(cls, rows: Iterable[Mapping[str, Any]]) -> "Result":
        """Build a result from rows given as mappings from column name to value.

        Columns that a row leaves out are null. A
        :class:`Provenance` and :class:`~mixed_signals.errors.WarningRecord`
        entries may be given as objects.

        Args:
            rows: The rows, in order.

        Returns:
            The result.

        Raises:
            SchemaError: If a row names a column outside the schema, or the
                rows leave a required column null.
        """
        row_values = [_column_values(row) for row in rows]
        return cls(pl.DataFrame(row_values, schema=RESULT_SCHEMA))

    @classmethod
    def concat(cls, results: Iterable["Result"]) -> "Result":
        """Stack the rows of several results, in order."""
        tables = [result.table for result in results]
        if not tables:
            return cls.empty()
        return cls(pl.concat(tables, how="vertical"))

    @classmethod
    def read_parquet(cls, path: str | os.PathLike[str]) -> "Result":
        """Read a result from a parquet file and check its schema."""
        return cls(pl.read_parquet(Path(path)))

    @property
    def table(self) -> pl.DataFrame:
        """The rows as a polars DataFrame."""
        return self._table

    def write_parquet(self, path: str | os.PathLike[str]) -> None:
        """Write the rows to a parquet file."""
        self._table.write_parquet(Path(path))

    def __len__(self) -> int:
        """Return the number of rows."""
        return self._table.height

    def __repr__(self) -> str:
        """Return a short description with the number of rows."""
        return f"Result(rows={len(self)})"


def _check_sha256(digest: str) -> None:
    if SHA256_HEX_PATTERN.fullmatch(digest) is None:
        raise ValueError(
            f"{digest!r} is not a SHA-256 hex digest of 64 lowercase hex characters"
        )


def _column_values(row: Mapping[str, Any]) -> dict[str, Any]:
    unknown = [name for name in row if name not in RESULT_SCHEMA]
    if unknown:
        raise SchemaError(f"columns outside the result schema: {', '.join(unknown)}")
    values = dict(row)
    if isinstance(values.get("provenance"), Provenance):
        values["provenance"] = values["provenance"].to_struct()
    if values.get("warnings") is not None:
        values["warnings"] = [
            entry.to_dict() if isinstance(entry, WarningRecord) else entry
            for entry in values["warnings"]
        ]
    return values


def _schema_problems(table: pl.DataFrame) -> list[str]:
    schema = table.schema
    problems = []
    missing = [name for name in RESULT_SCHEMA if name not in schema]
    if missing:
        problems.append(f"missing columns {', '.join(missing)}")
    unexpected = [name for name in schema if name not in RESULT_SCHEMA]
    if unexpected:
        problems.append(f"unexpected columns {', '.join(unexpected)}")
    problems.extend(
        f"column {name!r} has type {dtype}, expected {RESULT_SCHEMA[name]}"
        for name, dtype in schema.items()
        if name in RESULT_SCHEMA and dtype != RESULT_SCHEMA[name]
    )
    if problems:
        return problems
    if list(schema) != list(RESULT_SCHEMA):
        return ["columns are not in the order of the schema"]
    null_columns = [name for name in REQUIRED_COLUMNS if table[name].null_count() > 0]
    if null_columns:
        problems.append(f"nulls in required columns {', '.join(null_columns)}")
    return problems
