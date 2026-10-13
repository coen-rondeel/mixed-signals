"""Tests for the result schema, the result table and provenance."""

import hashlib
import math
from importlib.metadata import version
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import polars as pl
import pytest
from pydantic import ValidationError

from mixed_signals.core.result import (
    PROVENANCE_DTYPE,
    REQUIRED_COLUMNS,
    RESULT_SCHEMA,
    Provenance,
    Result,
    UncertaintyMethod,
    hash_arrays,
)
from mixed_signals.errors import SchemaError, WarningRecord

# * Written out in full, so that any change to the agreed schema fails here.
AGREED_SCHEMA = pl.Schema(
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
        "null_recipe": pl.Struct(
            {
                "construction": pl.String(),
                "draws": pl.Int64(),
                "sample_size": pl.Int64(),
            }
        ),
        "p_value": pl.Float64(),
        "sigma_sig": pl.Float64(),
        "sigma_eff": pl.Float64(),
        "r_eff": pl.Float64(),
        "warnings": pl.List(
            pl.Struct({"category": pl.String(), "message": pl.String()})
        ),
        "provenance": pl.Struct(
            {
                "package_version": pl.String(),
                "seed": pl.Int64(),
                "prng_key": pl.String(),
                "input_hashes": pl.List(pl.String()),
                "jax_version": pl.String(),
                "device_platform": pl.String(),
                "config_hash": pl.String(),
            }
        ),
    }
)


def comparison_row(**overrides: Any) -> dict[str, Any]:
    """Return a complete comparison row, with some columns replaced."""
    row: dict[str, Any] = {
        "problem": 0,
        "pair": ["sbi", "mcmc"],
        "metric": "kl",
        "estimator": "knn",
        "estimator_settings": '{"k": 5}',
        "space": "whiten(pooled)",
        "parameters": ["frequency", "amplitude"],
        "value": 0.12,
        "uncertainty": 0.01,
        "uncertainty_method": UncertaintyMethod.NULL_STD,
        "null_mean": 0.02,
        "null_std": 0.01,
        "null_draws": [0.01, 0.02, 0.03],
        "null_recipe": {"construction": "permutation", "draws": 3, "sample_size": 500},
        "p_value": 0.25,
        "sigma_sig": 0.67,
        "sigma_eff": 0.45,
        "r_eff": None,
        "warnings": [WarningRecord("SubsamplingWarning", "cap reached")],
        "provenance": Provenance.collect(
            seed=42,
            prng_key="threefry2x32:000000000000002a",
            input_hashes=[hash_arrays(np.zeros(3)), hash_arrays(np.ones(3))],
        ),
    }
    row.update(overrides)
    return row


def test_empty_result_has_the_agreed_schema() -> None:
    """An empty result has every agreed column, in order, with its type."""
    empty = Result.empty()
    assert len(empty) == 0
    assert empty.table.schema == AGREED_SCHEMA
    assert RESULT_SCHEMA == AGREED_SCHEMA


def test_required_columns_are_schema_columns() -> None:
    """Every required column exists in the schema."""
    assert set(REQUIRED_COLUMNS) <= set(RESULT_SCHEMA)


def test_provenance_fields_match_the_column() -> None:
    """The provenance model and the provenance column have the same fields."""
    field_names = [field.name for field in PROVENANCE_DTYPE.fields]
    assert list(Provenance.model_fields) == field_names


def test_rows_keep_their_values() -> None:
    """Rows given as mappings, with model objects, keep every value."""
    result = Result.from_rows([comparison_row()])
    (stored,) = result.table.to_dicts()
    assert stored["uncertainty_method"] == "null_std"
    assert stored["warnings"] == [
        {"category": "SubsamplingWarning", "message": "cap reached"}
    ]
    assert stored["provenance"]["seed"] == 42
    assert stored["r_eff"] is None


def test_parquet_round_trips(tmp_path: Path) -> None:
    """A result written to parquet reads back with the same rows and schema."""
    result = Result.from_rows(
        [comparison_row(), comparison_row(problem=1, value=math.nan)]
    )
    path = tmp_path / "results.parquet"
    result.write_parquet(path)
    read_back = Result.read_parquet(path)
    assert read_back.table.schema == AGREED_SCHEMA
    assert read_back.table.equals(result.table)


def test_empty_result_round_trips(tmp_path: Path) -> None:
    """An empty result keeps its schema through parquet."""
    path = tmp_path / "empty.parquet"
    Result.empty().write_parquet(path)
    assert Result.read_parquet(path).table.schema == AGREED_SCHEMA


def test_undefined_values_are_nan_and_missing_columns_are_null() -> None:
    """NaN marks an undefined value; columns that do not apply stay null."""
    row = comparison_row(value=math.nan)
    for column in ("uncertainty", "null_mean", "null_std", "p_value", "sigma_sig"):
        del row[column]
    (stored,) = Result.from_rows([row]).table.to_dicts()
    assert math.isnan(stored["value"])
    assert stored["p_value"] is None


def test_unknown_columns_are_rejected() -> None:
    """A row with a column outside the schema fails instead of losing data."""
    with pytest.raises(SchemaError, match="sigma_effective"):
        Result.from_rows([comparison_row(sigma_effective=0.4)])


@pytest.mark.parametrize("column", ["metric", "value", "warnings", "provenance"])
def test_required_columns_cannot_be_null(column: str) -> None:
    """Rows must name what they measured, hold a value and carry provenance."""
    with pytest.raises(SchemaError, match=f"nulls in required columns {column}"):
        Result.from_rows([comparison_row(**{column: None})])


def test_tables_with_a_wrong_schema_are_rejected() -> None:
    """Missing, unexpected, mistyped and reordered columns each fail clearly."""
    table = Result.from_rows([comparison_row()]).table
    with pytest.raises(SchemaError, match="missing columns r_eff"):
        Result(table.drop("r_eff"))
    with pytest.raises(SchemaError, match="unexpected columns verdict"):
        Result(table.with_columns(verdict=pl.lit("pass")))
    with pytest.raises(SchemaError, match="'problem' has type Int32"):
        Result(table.with_columns(pl.col("problem").cast(pl.Int32)))
    with pytest.raises(SchemaError, match="order"):
        Result(table.select(table.columns[::-1]))


def test_concat_stacks_rows_in_order() -> None:
    """Concatenation keeps the order of the results and of their rows."""
    first = Result.from_rows([comparison_row(problem=0)])
    second = Result.from_rows([comparison_row(problem=1), comparison_row(problem=2)])
    combined = Result.concat([first, Result.empty(), second])
    assert combined.table["problem"].to_list() == [0, 1, 2]
    assert len(Result.concat([])) == 0


def test_repr_counts_rows() -> None:
    """The representation shows the number of rows."""
    assert repr(Result.from_rows([comparison_row()])) == "Result(rows=1)"


def test_provenance_records_the_running_process() -> None:
    """Versions and platform come from the process that builds the row."""
    provenance = Provenance.collect(seed=3)
    assert provenance.package_version == version("mixed-signals")
    assert provenance.jax_version == jax.__version__
    assert provenance.device_platform == jax.default_backend()
    assert provenance.prng_key is None
    assert provenance.config_hash is None


@pytest.mark.parametrize(
    "fields",
    [
        {"seed": -1},
        {"seed": 2**63},
        {"prng_key": "0000002a"},
        {"input_hashes": ["abc"]},
        {"config_hash": "A" * 64},
    ],
)
def test_provenance_rejects_malformed_fields(fields: dict[str, Any]) -> None:
    """Seeds outside the key range and malformed keys or digests fail."""
    with pytest.raises(ValidationError):
        Provenance.collect(**fields)


def test_hash_follows_its_documented_definition() -> None:
    """The digest covers the type string, the shape and the C-ordered bytes."""
    samples = np.arange(6, dtype=np.float64).reshape(3, 2)
    expected = hashlib.sha256(b"<f8:3,2;" + samples.tobytes(order="C")).hexdigest()
    assert hash_arrays(samples) == expected


def test_hash_is_the_same_for_numpy_jax_and_byte_orders() -> None:
    """The digest depends on the values only, not on where or how they are stored."""
    samples = np.linspace(0.0, 1.0, 12).reshape(4, 3)
    digest = hash_arrays(samples)
    assert hash_arrays(jnp.asarray(samples)) == digest
    assert hash_arrays(samples.astype(">f8")) == digest
    assert hash_arrays(np.asfortranarray(samples)) == digest


def test_hash_tells_types_shapes_and_values_apart() -> None:
    """A different type, shape, value or array order changes the digest."""
    samples = np.linspace(0.0, 1.0, 12).reshape(4, 3)
    weights = np.zeros(4)
    digests = {
        hash_arrays(samples),
        hash_arrays(samples.astype(np.float32)),
        hash_arrays(samples.reshape(3, 4)),
        hash_arrays(samples + 1e-15),
        hash_arrays(samples, weights),
        hash_arrays(weights, samples),
    }
    assert len(digests) == 6


def test_hash_rejects_object_arrays() -> None:
    """Arrays of Python objects have no stable bytes to hash."""
    with pytest.raises(TypeError, match="Python objects"):
        hash_arrays(np.array([object()]))
