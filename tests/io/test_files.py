"""Tests for the NumPy-file and table readers and the classmethods that call them."""

import inspect
import os
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
import polars as pl
import pytest

from mixed_signals.core.chains import Chains
from mixed_signals.core.samples import SampleSet
from mixed_signals.errors import PrecisionWarning
from mixed_signals.io import files

unchecked_table_reader = inspect.unwrap(files.sample_set_from_table)
SAMPLES = 40
PARAMS = 3
BATCH = 2
NAMES = ("x0", "x1", "x2")
TRIPWIRE = "MIXED_SIGNALS_TRIPWIRE"
LAYOUTS = r"\(N, D\) for one problem or \(B, N, D\)"


class Payload:
    """An object whose pickle runs code when it is loaded."""

    def __reduce__(self) -> tuple[Any, tuple[str]]:
        """Return a call that sets an environment variable."""
        return exec, (f"import os; os.environ[{TRIPWIRE!r}] = '1'",)


def draws(*shape: int, seed: int = 0) -> np.ndarray:
    """Return normal draws of a shape."""
    return np.random.default_rng(seed).normal(size=shape)


def n_eff_of(sample_set: SampleSet) -> float:
    """Return the effective sample size of a set of one problem and posterior."""
    assert sample_set.n_eff is not None
    return float(sample_set.n_eff[0, 0])


def assert_same_set(first: SampleSet, second: SampleSet) -> None:
    """Check that two sets hold the same fields with the same values."""
    first_fields, second_fields = first.to_arrays(), second.to_arrays()
    assert first_fields.keys() == second_fields.keys()
    for name, values in first_fields.items():
        np.testing.assert_array_equal(values, second_fields[name], err_msg=name)


def flat_arrays(batch: int | None) -> dict[str, np.ndarray]:
    """Return every field of a flat set: (N, D) for one problem, or (B, N, D)."""
    leading = () if batch is None else (batch,)
    return {
        "samples": draws(*leading, SAMPLES, PARAMS),
        "log_weights": draws(*leading, SAMPLES, seed=1),
        "n_eff": np.full(leading, 25.0),
        "log_like": draws(*leading, SAMPLES, seed=2),
        "log_prior": draws(*leading, SAMPLES, seed=3),
        "log_q": draws(*leading, SAMPLES, seed=4),
    }


def parameter_table(extra: bool = True) -> pl.DataFrame:
    """Return a table with the three parameters in a shuffled order and more columns."""
    values = draws(SAMPLES, PARAMS)
    columns = {"x2": values[:, 2], "x0": values[:, 0], "x1": values[:, 1]}
    if extra:
        columns["log_w"] = draws(SAMPLES, seed=5)
        columns["other"] = draws(SAMPLES, seed=6)
    return pl.DataFrame(columns)


# * NumPy files


@pytest.mark.parametrize("batch", [None, BATCH])
def test_archives_equal_the_constructor_with_the_posterior_axis(
    tmp_path: Path, batch: int | None
) -> None:
    """Every optional array is carried, with one posterior inserted."""
    arrays: dict[str, Any] = flat_arrays(batch)
    path = tmp_path / "set.npz"
    np.savez(path, **arrays)
    read = SampleSet.from_npz(path)
    if batch is None:
        expected = SampleSet(**arrays)
    else:
        expected = SampleSet(
            arrays["samples"][:, None],
            log_weights=arrays["log_weights"][:, None],
            n_eff=arrays["n_eff"][:, None],
            log_like=arrays["log_like"][:, None],
            log_prior=arrays["log_prior"][:, None],
            log_q=arrays["log_q"][:, None],
        )
    assert read.shape == (batch or 1, 1, SAMPLES, PARAMS)
    assert_same_set(read, expected)


def test_a_npy_file_holds_the_samples_alone(tmp_path: Path) -> None:
    """The array is the samples, and the suffix is read in any case."""
    samples = draws(SAMPLES, PARAMS)
    names = ("lower.npy", "UPPER.NPY")
    for name in names:
        # ! A handle makes np.save write the name as given, whereas a path that does
        # ! not end in lowercase .npy gets that suffix appended.
        with (tmp_path / name).open("wb") as handle:
            np.save(handle, samples)
        read = SampleSet.from_npz(tmp_path / name)
        assert_same_set(read, SampleSet(samples))
    assert sorted(path.name for path in tmp_path.iterdir()) == sorted(names)
    assert read.n_eff is None


def test_n_eff_comes_from_the_argument_or_the_archive(tmp_path: Path) -> None:
    """The argument serves a file without n_eff, and conflicts with an archived one."""
    samples = draws(BATCH, SAMPLES, PARAMS)
    np.save(tmp_path / "samples.npy", samples)
    read = SampleSet.from_npz(tmp_path / "samples.npy", n_eff=[10.0, 12.0])
    np.testing.assert_array_equal(read.n_eff, [[10.0], [12.0]])
    np.savez(tmp_path / "archive.npz", samples=samples[0])
    one_problem = SampleSet.from_npz(tmp_path / "archive.npz", n_eff=7.0)
    assert n_eff_of(one_problem) == 7.0
    np.savez(tmp_path / "held.npz", samples=samples[0], n_eff=np.float64(9.0))
    with pytest.raises(ValueError, match="holds n_eff"):
        SampleSet.from_npz(tmp_path / "held.npz", n_eff=7.0)
    # * One problem takes a number; a one-problem batch also takes a list of one.
    np.save(tmp_path / "one.npy", samples[:1])
    batch_of_one = SampleSet.from_npz(tmp_path / "one.npy", n_eff=[8.0])
    assert n_eff_of(batch_of_one) == 8.0
    with pytest.raises(ValueError, match="n_eff must be one number for each of the 2"):
        SampleSet.from_npz(tmp_path / "samples.npy", n_eff=8.0)
    with pytest.raises(ValueError, match="n_eff must be one number for each of the 1"):
        SampleSet.from_npz(tmp_path / "archive.npz", n_eff=[8.0, 9.0])


def test_keys_and_layouts_are_checked(tmp_path: Path) -> None:
    """Unknown keys, a missing array, a wrong rank and a wrong suffix raise."""
    samples = draws(SAMPLES, PARAMS)
    np.savez(tmp_path / "unknown.npz", samples=samples, weights=samples[:, 0])
    with pytest.raises(ValueError, match=r"unknown keys 'weights'.*'log_weights'"):
        SampleSet.from_npz(tmp_path / "unknown.npz")
    np.savez(tmp_path / "missing.npz", log_weights=samples[:, 0])
    with pytest.raises(ValueError, match="key 'samples'; it holds 'log_weights'"):
        SampleSet.from_npz(tmp_path / "missing.npz")
    np.savez(tmp_path / "empty.npz")
    with pytest.raises(ValueError, match="key 'samples'; it holds no arrays"):
        SampleSet.from_npz(tmp_path / "empty.npz")
    for shape in ((PARAMS,), (BATCH, 1, SAMPLES, PARAMS)):
        np.save(tmp_path / "rank.npy", draws(*shape))
        with pytest.raises(ValueError, match=LAYOUTS):
            SampleSet.from_npz(tmp_path / "rank.npy")
    np.savez(
        tmp_path / "per_sample.npz",
        samples=draws(BATCH, SAMPLES, PARAMS),
        log_weights=draws(SAMPLES),
    )
    with pytest.raises(ValueError, match="log_weights must have one axis fewer"):
        SampleSet.from_npz(tmp_path / "per_sample.npz")
    np.savez(
        tmp_path / "size.npz",
        samples=draws(SAMPLES, PARAMS),
        log_weights=draws(SAMPLES + 1),
    )
    with pytest.raises(ValueError, match="log_weights has shape"):
        SampleSet.from_npz(tmp_path / "size.npz")
    (tmp_path / "samples.txt").write_text("1 2 3")
    with pytest.raises(ValueError, match=r"must end in \.npy or \.npz, got '\.txt'"):
        SampleSet.from_npz(tmp_path / "samples.txt")
    with pytest.raises(OSError, match="missing"):
        SampleSet.from_npz(tmp_path / "missing.npy")


def test_files_with_objects_fail_without_unpickling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The files load with allow_pickle=False, so no pickled code runs."""
    monkeypatch.delenv(TRIPWIRE, raising=False)
    objects = np.array([Payload()], dtype=object)
    np.save(tmp_path / "objects.npy", objects, allow_pickle=True)
    np.savez(tmp_path / "objects.npz", samples=objects)
    with pytest.raises(ValueError, match="allow_pickle=False"):
        SampleSet.from_npz(tmp_path / "objects.npy")
    with pytest.raises(ValueError, match="allow_pickle=False"):
        SampleSet.from_npz(tmp_path / "objects.npz")
    with pytest.raises(ValueError, match="allow_pickle=False"):
        Chains.from_npz(tmp_path / "objects.npz")
    assert TRIPWIRE not in os.environ


def test_narrow_arrays_warn_and_non_real_arrays_raise(tmp_path: Path) -> None:
    """A float32 array reaches the container, which warns; strings are not numbers."""
    np.save(tmp_path / "narrow.npy", draws(SAMPLES, PARAMS).astype(np.float32))
    with pytest.warns(PrecisionWarning, match="samples"):
        SampleSet.from_npz(tmp_path / "narrow.npy")
    np.save(tmp_path / "text.npy", np.array([["a", "b"]] * SAMPLES))
    with pytest.raises(TypeError):
        SampleSet.from_npz(tmp_path / "text.npy")


# * Chains


@pytest.mark.parametrize("layout", ["eryn", "emcee"])
def test_chain_archives_equal_the_constructor(tmp_path: Path, layout: str) -> None:
    """Both layouts of the constructor hold, with every optional array carried."""
    steps, walkers = 30, 6
    shape: tuple[int, ...]
    arrays: dict[str, Any]
    if layout == "eryn":
        shape = (steps, 2, walkers, 1, PARAMS)
        arrays = {
            "betas": np.tile([1.0, 0.5], (steps, 1)),
            "log_like": draws(steps, 2, walkers, seed=1),
            "log_prior": draws(steps, 2, walkers, seed=2),
            "accepted": np.full((2, walkers), 5),
            "proposed": np.full((2, walkers), 9),
            "swaps_accepted": np.array([3]),
            "swaps_proposed": np.array([8]),
        }
    else:
        shape = (steps, walkers, PARAMS)
        arrays = {
            "log_like": draws(steps, walkers, seed=1),
            "log_prior": draws(steps, walkers, seed=2),
            "accepted": np.full(walkers, 5),
            "proposed": np.full(walkers, 9),
        }
    samples = draws(*shape)
    np.savez(tmp_path / "chains.npz", samples=samples, **arrays)
    read = Chains.from_npz(tmp_path / "chains.npz")
    expected = Chains(samples, **arrays)
    for name in ("samples", "betas", "log_like", "log_prior", "accepted", "proposed"):
        np.testing.assert_array_equal(getattr(read, name), getattr(expected, name))
    for name in ("swaps_accepted", "swaps_proposed"):
        assert (getattr(read, name) is None) == (layout == "emcee")


def test_chain_archives_are_checked(tmp_path: Path) -> None:
    """Unknown keys, a missing array and a wrong suffix raise."""
    samples = draws(30, 6, PARAMS)
    np.savez(tmp_path / "unknown.npz", samples=samples, weights=samples[..., 0])
    with pytest.raises(ValueError, match=r"unknown keys 'weights'.*'swaps_proposed'"):
        Chains.from_npz(tmp_path / "unknown.npz")
    np.savez(tmp_path / "missing.npz", log_like=samples[..., 0])
    with pytest.raises(ValueError, match="needs an array under the key 'samples'"):
        Chains.from_npz(tmp_path / "missing.npz")
    np.save(tmp_path / "chains.npy", samples)
    with pytest.raises(ValueError, match=r"must end in \.npz, got '\.npy'"):
        Chains.from_npz(tmp_path / "chains.npy")


# * Tables


def test_parquet_and_csv_give_the_same_set(tmp_path: Path) -> None:
    """The table holds more columns in another order; the names choose and order."""
    frame = parameter_table()
    frame.write_parquet(tmp_path / "table.parquet")
    frame.write_csv(tmp_path / "table.csv")
    from_parquet = SampleSet.from_table(tmp_path / "table.parquet", NAMES)
    from_csv = SampleSet.from_table(tmp_path / "table.csv", list(NAMES))
    assert from_parquet.shape == (1, 1, SAMPLES, PARAMS)
    for position, name in enumerate(NAMES):
        np.testing.assert_array_equal(
            from_parquet.samples[0, 0, :, position], frame[name].to_numpy()
        )
    assert_same_set(from_parquet, from_csv)
    assert from_parquet.n_eff is None
    np.testing.assert_array_equal(from_parquet.log_weights, 0.0)
    subset = SampleSet.from_table(tmp_path / "table.parquet", ["x1", "x0"])
    np.testing.assert_array_equal(
        subset.samples[0, 0], frame.select("x1", "x0").to_numpy()
    )


def test_log_weights_and_n_eff_come_with_the_table(tmp_path: Path) -> None:
    """The weight column and the effective size reach the set."""
    frame = parameter_table()
    frame.write_parquet(tmp_path / "table.parquet")
    read = SampleSet.from_table(
        tmp_path / "table.parquet", NAMES, log_weights="log_w", n_eff=12.0
    )
    np.testing.assert_array_equal(read.log_weights[0, 0], frame["log_w"].to_numpy())
    assert n_eff_of(read) == 12.0
    with pytest.raises(ValueError, match="log_weights names the column 'x0'"):
        SampleSet.from_table(tmp_path / "table.parquet", NAMES, log_weights="x0")
    with pytest.raises(ValueError, match="n_eff must be one number for each of the 1"):
        unchecked_table_reader(tmp_path / "table.parquet", NAMES, n_eff=[1.0, 2.0])


def test_integer_columns_are_numbers(tmp_path: Path) -> None:
    """An integer column is read as float64."""
    frame = pl.DataFrame({"x0": np.arange(SAMPLES), "x1": draws(SAMPLES)})
    frame.write_parquet(tmp_path / "table.parquet")
    read = SampleSet.from_table(tmp_path / "table.parquet", ["x0", "x1"])
    np.testing.assert_array_equal(read.samples[0, 0, :, 0], np.arange(SAMPLES))


def test_missing_columns_are_listed(tmp_path: Path) -> None:
    """The message names the missing columns and the columns of the table."""
    parameter_table().write_parquet(tmp_path / "table.parquet")
    with pytest.raises(ValueError, match=r"no columns 'y0', 'y1'.*'x2', 'x0', 'x1'"):
        SampleSet.from_table(tmp_path / "table.parquet", ["x0", "y0", "y1"])
    with pytest.raises(ValueError, match="no columns 'weights'"):
        SampleSet.from_table(tmp_path / "table.parquet", NAMES, log_weights="weights")


def test_nulls_and_other_types_raise(tmp_path: Path) -> None:
    """A null in a selected column, and a text or boolean column, are errors."""
    values = draws(SAMPLES).tolist()
    with_null = pl.DataFrame({"x0": [None, *values[1:]], "x1": values})
    with_null.write_parquet(tmp_path / "null.parquet")
    with pytest.raises(ValueError, match="column 'x0' holds 1 null"):
        SampleSet.from_table(tmp_path / "null.parquet", ["x0", "x1"])
    # * A null in a column that no name selects does not matter.
    assert SampleSet.from_table(tmp_path / "null.parquet", ["x1"]).shape.param == 1
    text = pl.DataFrame({"x0": ["a"] * SAMPLES, "x1": values})
    text.write_parquet(tmp_path / "text.parquet")
    with pytest.raises(TypeError, match="column 'x0' must be integer or float, got"):
        SampleSet.from_table(tmp_path / "text.parquet", ["x0", "x1"])
    flags = pl.DataFrame({"x0": [True] * SAMPLES, "x1": values})
    flags.write_parquet(tmp_path / "flags.parquet")
    with pytest.raises(TypeError, match="column 'x0' must be integer or float"):
        SampleSet.from_table(tmp_path / "flags.parquet", ["x0", "x1"])


def test_narrow_columns_warn_once_before_stacking(tmp_path: Path) -> None:
    """A Float32 column raises the precision warning, naming it, and is stacked."""
    values = draws(SAMPLES, 2)
    frame = pl.DataFrame(
        {"x0": values[:, 0].astype(np.float32), "x1": values[:, 1].astype(np.float32)}
    )
    frame.write_parquet(tmp_path / "narrow.parquet")
    with pytest.warns(PrecisionWarning, match="column 'x0' .*column 'x1'") as warned:
        read = SampleSet.from_table(tmp_path / "narrow.parquet", ["x0", "x1"])
    assert len(warned) == 1
    assert read.samples.dtype == jnp.float64
    np.testing.assert_array_equal(
        read.samples[0, 0, :, 0], frame["x0"].to_numpy().astype(np.float64)
    )
    wide = frame.with_columns(pl.col("x1").cast(pl.Float64))
    wide.write_parquet(tmp_path / "mixed.parquet")
    with pytest.warns(PrecisionWarning, match="column 'x0'") as mixed:
        SampleSet.from_table(tmp_path / "mixed.parquet", ["x0", "x1"])
    assert "column 'x1'" not in str(mixed[0].message)


def test_suffixes_and_names_are_checked(tmp_path: Path) -> None:
    """Another suffix, and names that are empty, repeated or not strings, raise."""
    parameter_table().write_parquet(tmp_path / "table.parquet")
    (tmp_path / "table.txt").write_text("x0\n1\n")
    with pytest.raises(
        ValueError, match=r"must end in \.parquet or \.csv, got '\.txt'"
    ):
        SampleSet.from_table(tmp_path / "table.txt", ["x0"])
    path = tmp_path / "table.parquet"
    with pytest.raises(ValueError, match="names must not be empty"):
        SampleSet.from_table(path, [])
    with pytest.raises(ValueError, match="names must be unique"):
        SampleSet.from_table(path, ["x0", "x0"])
    with pytest.raises(ValueError, match="names must be a sequence of strings"):
        unchecked_table_reader(path, 3)
    with pytest.raises(ValueError, match="names must be a sequence of strings"):
        unchecked_table_reader(path, "x0")
    with pytest.raises(ValueError, match="names must hold strings"):
        unchecked_table_reader(path, ["x0", 1])
    with pytest.raises(OSError, match="nowhere"):
        SampleSet.from_table(tmp_path / "nowhere.parquet", ["x0"])
