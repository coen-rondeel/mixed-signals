"""Tests for the HDF5 reader and ``SampleSet.from_hdf5``."""

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from mixed_signals.core.samples import SampleSet
from mixed_signals.errors import MissingDependencyError, PrecisionWarning
from mixed_signals.io import hdf5

h5py = pytest.importorskip("h5py")

SAMPLES = 40
NAMES = ("x0", "x1", "x2")
BATCH = 2
MISSING_DATASETS = r"has no datasets 'y0', 'y1'; its datasets are 'other', 'x0'"


def draws(*shape: int, seed: int = 0) -> np.ndarray:
    """Return normal draws of a shape."""
    return np.random.default_rng(seed).normal(size=shape)


def write_group(
    path: Path, group: str = "posterior", *, batch: int | None = None, **extra: Any
) -> dict[str, np.ndarray]:
    """Write one dataset per parameter in a group, and return their values."""
    leading = () if batch is None else (batch,)
    values = {
        name: draws(*leading, SAMPLES, seed=position)
        for position, name in enumerate(NAMES)
    }
    with h5py.File(path, "w") as file:
        node = file.require_group(group) if group else file
        for name, data in {**values, **extra}.items():
            node.create_dataset(name, data=data)
    return values


def n_eff_of(sample_set: SampleSet) -> float:
    """Return the effective sample size of a set of one problem and posterior."""
    assert sample_set.n_eff is not None
    return float(sample_set.n_eff[0, 0])


def test_datasets_of_one_problem_give_one_set(tmp_path: Path) -> None:
    """Datasets shaped (N,) stack in the order of the names."""
    values = write_group(tmp_path / "run.h5", other=draws(SAMPLES, seed=9))
    read = SampleSet.from_hdf5(tmp_path / "run.h5", "posterior", NAMES)
    assert read.shape == (1, 1, SAMPLES, 3)
    for position, name in enumerate(NAMES):
        np.testing.assert_array_equal(read.samples[0, 0, :, position], values[name])
    assert read.n_eff is None
    np.testing.assert_array_equal(read.log_weights, 0.0)
    reordered = SampleSet.from_hdf5(tmp_path / "run.h5", "posterior", ["x2", "x0"])
    np.testing.assert_array_equal(reordered.samples[0, 0, :, 0], values["x2"])
    np.testing.assert_array_equal(reordered.samples[0, 0, :, 1], values["x0"])


def test_datasets_of_several_problems_give_one_problem_each(tmp_path: Path) -> None:
    """Datasets shaped (B, N) give B problems, with one n_eff for each."""
    values = write_group(tmp_path / "run.h5", batch=BATCH)
    read = SampleSet.from_hdf5(
        tmp_path / "run.h5", "posterior", NAMES, n_eff=[10.0, 12.0]
    )
    assert read.shape == (BATCH, 1, SAMPLES, 3)
    for position, name in enumerate(NAMES):
        np.testing.assert_array_equal(read.samples[:, 0, :, position], values[name])
    np.testing.assert_array_equal(read.n_eff, [[10.0], [12.0]])
    with pytest.raises(ValueError, match="n_eff must be one number for each of the 2"):
        SampleSet.from_hdf5(tmp_path / "run.h5", "posterior", NAMES, n_eff=10.0)
    write_group(tmp_path / "single.h5")
    single = SampleSet.from_hdf5(tmp_path / "single.h5", "posterior", NAMES, n_eff=9.0)
    assert n_eff_of(single) == 9.0


def test_groups_are_found_by_their_path(tmp_path: Path) -> None:
    """The root is the empty string or a slash, and groups may be nested."""
    values = write_group(tmp_path / "root.h5", group="")
    for root in ("", "/"):
        read = SampleSet.from_hdf5(tmp_path / "root.h5", root, NAMES)
        np.testing.assert_array_equal(read.samples[0, 0, :, 0], values["x0"])
    write_group(tmp_path / "nested.h5", group="results/chain_0")
    nested = SampleSet.from_hdf5(tmp_path / "nested.h5", "results/chain_0", NAMES)
    assert nested.shape.sample == SAMPLES


def test_log_weights_come_from_the_same_group(tmp_path: Path) -> None:
    """The weight dataset sits beside the parameters, in the same shape."""
    log_weights = draws(SAMPLES, seed=7)
    write_group(tmp_path / "run.h5", log_w=log_weights)
    read = SampleSet.from_hdf5(
        tmp_path / "run.h5", "posterior", NAMES, log_weights="log_w"
    )
    np.testing.assert_array_equal(read.log_weights[0, 0], log_weights)
    with pytest.raises(ValueError, match="log_weights names the dataset 'x0'"):
        SampleSet.from_hdf5(tmp_path / "run.h5", "posterior", NAMES, log_weights="x0")
    with pytest.raises(ValueError, match="has no datasets 'weights'"):
        SampleSet.from_hdf5(
            tmp_path / "run.h5", "posterior", NAMES, log_weights="weights"
        )
    write_group(tmp_path / "short.h5", log_w=log_weights[:-1])
    with pytest.raises(ValueError, match="must share one shape"):
        SampleSet.from_hdf5(
            tmp_path / "short.h5", "posterior", NAMES, log_weights="log_w"
        )


def test_missing_groups_and_datasets_are_listed(tmp_path: Path) -> None:
    """The message names the file, the group and what the group does hold."""
    write_group(tmp_path / "run.h5", other=draws(SAMPLES))
    with pytest.raises(ValueError, match=r"run\.h5 has no group 'results'"):
        SampleSet.from_hdf5(tmp_path / "run.h5", "results", NAMES)
    with pytest.raises(ValueError, match=MISSING_DATASETS):
        SampleSet.from_hdf5(tmp_path / "run.h5", "posterior", ["x0", "y0", "y1"])
    # * A dataset at the path of the group is not a group.
    with pytest.raises(ValueError, match="has no group 'posterior/x0'"):
        SampleSet.from_hdf5(tmp_path / "run.h5", "posterior/x0", NAMES)
    with pytest.raises(OSError, match="nowhere"):
        SampleSet.from_hdf5(tmp_path / "nowhere.h5", "posterior", NAMES)


def test_members_and_shapes_are_checked(tmp_path: Path) -> None:
    """Subgroups, text, scalars, three axes and unequal shapes raise."""
    path = tmp_path / "run.h5"
    write_group(path)
    with h5py.File(path, "a") as file:
        file["posterior"].create_group("sub")
        file["posterior"].create_dataset(
            "text", data=np.array(["a"] * SAMPLES, dtype="S1")
        )
        file["posterior"].create_dataset("flag", data=np.ones(SAMPLES, dtype=bool))
        file["posterior"].create_dataset("scalar", data=1.0)
        file["posterior"].create_dataset("cube", data=draws(2, 2, 2))
        file["posterior"].create_dataset("short", data=draws(SAMPLES - 1))
        file["posterior"].create_dataset("ints", data=np.arange(SAMPLES))
    with pytest.raises(ValueError, match="'sub' in group 'posterior' is not a dataset"):
        SampleSet.from_hdf5(path, "posterior", ["x0", "sub"])
    for name in ("text", "flag"):
        with pytest.raises(
            TypeError, match=f"dataset '{name}' must be integer or float"
        ):
            SampleSet.from_hdf5(path, "posterior", ["x0", name])
    for name in ("scalar", "cube"):
        with pytest.raises(ValueError, match=r"must be shaped \(N,\) or \(B, N\)"):
            SampleSet.from_hdf5(path, "posterior", [name])
    with pytest.raises(ValueError, match=r"must share one shape.*'short' \(39,\)"):
        SampleSet.from_hdf5(path, "posterior", ["x0", "short"])
    ints = SampleSet.from_hdf5(path, "posterior", ["ints"])
    np.testing.assert_array_equal(ints.samples[0, 0, :, 0], np.arange(SAMPLES))


def test_names_are_checked(tmp_path: Path) -> None:
    """Empty and repeated names raise before the file opens."""
    with pytest.raises(ValueError, match="names must not be empty"):
        SampleSet.from_hdf5(tmp_path / "nowhere.h5", "posterior", [])
    with pytest.raises(ValueError, match="names must be unique"):
        SampleSet.from_hdf5(tmp_path / "nowhere.h5", "posterior", ["x0", "x0"])


def test_narrow_datasets_warn(tmp_path: Path) -> None:
    """A float32 dataset raises the precision warning, naming it."""
    path = tmp_path / "run.h5"
    write_group(path)
    with h5py.File(path, "a") as file:
        file["posterior"].create_dataset(
            "narrow", data=draws(SAMPLES).astype(np.float32)
        )
    with pytest.warns(PrecisionWarning, match="dataset 'narrow'"):
        read = SampleSet.from_hdf5(path, "posterior", ["x0", "narrow"])
    assert read.samples.dtype == np.float64


def test_files_open_read_only(tmp_path: Path) -> None:
    """A file without write permission reads."""
    path = tmp_path / "run.h5"
    write_group(path)
    path.chmod(0o444)
    assert SampleSet.from_hdf5(path, "posterior", NAMES).shape.sample == SAMPLES


def test_a_missing_h5py_names_the_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without h5py, the reader raises with the install hint."""
    # * A None entry in sys.modules makes the import fail, as without the package.
    monkeypatch.setitem(sys.modules, "h5py", None)
    with pytest.raises(MissingDependencyError, match=r"mixed-signals\[hdf5\]"):
        hdf5.sample_set_from_hdf5("run.h5", "posterior", NAMES)
