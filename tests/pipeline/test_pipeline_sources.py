"""Tests for the import paths, spaces, arrays and posteriors of a pipeline run."""

import hashlib
import sys
import types
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pytest

from mixed_signals.core.chains import Chains
from mixed_signals.core.samples import SampleSet
from mixed_signals.core.space import ParamSpace
from mixed_signals.core.topology import Topology
from mixed_signals.errors import ConfigError, MissingDependencyError
from mixed_signals.pipeline._sources import (
    InputRecord,
    Sources,
    config_folder_on_path,
    file_sha256,
)
from mixed_signals.pipeline.config import PipelineConfig
from mixed_signals.priors import ErynPhysical

PREFIX = "pipeline_test_"
SPACE = ParamSpace.model_validate(
    {"parameters": [{"name": "x0"}, {"name": "x1"}, {"name": "x2"}]}
)


@pytest.fixture(autouse=True)
def forget_modules() -> Iterator[None]:
    """Drop the modules that a test imported from a temporary config folder."""
    before = set(sys.modules)
    yield
    for name in set(sys.modules) - before:
        if name.startswith(PREFIX):
            del sys.modules[name]


def n_eff_of(sample_set: SampleSet) -> float:
    """Return the effective sample size of a set of one problem and posterior."""
    assert sample_set.n_eff is not None
    return float(sample_set.n_eff[0, 0])


def sha256_of(path: Path) -> str:
    """Return the SHA-256 of a file, as hashlib gives it."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_module(folder: Path, name: str, text: str) -> Path:
    """Write a module of a config folder."""
    path = folder / f"{name}.py"
    path.write_text(text, encoding="utf-8")
    return path


def posterior(**fields: Any) -> Any:
    """Return a posterior entry from its fields."""
    config = PipelineConfig.model_validate(
        {
            "seed": 1,
            "output": "r",
            "space": {"parameters": [{"name": "x0"}]},
            "posteriors": [{"name": "run", **fields}],
            "diagnose": {},
        }
    )
    return config.posteriors[0]


# * The folder of the config on sys.path


def test_the_config_folder_comes_first_and_only_that_entry_leaves(
    tmp_path: Path,
) -> None:
    """The entry is added in front, and removed once, at the end of the block."""
    before = list(sys.path)
    with config_folder_on_path(tmp_path):
        assert sys.path[0] == str(tmp_path)
        assert sys.path[1:] == before
    assert sys.path == before
    # * A copy of the entry that was there before stays after the block.
    sys.path.append(str(tmp_path))
    try:
        with config_folder_on_path(tmp_path):
            assert sys.path.count(str(tmp_path)) == 2
        assert sys.path.count(str(tmp_path)) == 1
        assert sys.path[0] != str(tmp_path)
    finally:
        sys.path.remove(str(tmp_path))
    assert sys.path == before


def test_a_block_that_removed_its_entry_or_raised_still_restores_the_path(
    tmp_path: Path,
) -> None:
    """Neither a missing entry nor an exception leaves the path changed."""
    before = list(sys.path)
    with config_folder_on_path(tmp_path):
        sys.path.remove(str(tmp_path))
    assert sys.path == before
    with pytest.raises(RuntimeError, match="boom"), config_folder_on_path(tmp_path):
        raise RuntimeError("boom")
    assert sys.path == before


# * Import paths


def test_an_import_path_resolves_from_a_module_next_to_the_config(
    tmp_path: Path,
) -> None:
    """The module imports from the config folder, and the record hashes its file."""
    path = write_module(
        tmp_path, f"{PREFIX}models", "value = [1, 2]\nclass Box:\n    attr = 7\n"
    )
    sources = Sources(tmp_path)
    before = list(sys.path)
    assert sources.resolve(f"{PREFIX}models:value") == [1, 2]
    assert sources.resolve(f"{PREFIX}models:Box.attr") == 7
    assert sys.path == before
    assert f"{PREFIX}models" in sys.modules
    record = sources.modules[f"{PREFIX}models:value"]
    assert record.module == f"{PREFIX}models"
    assert record.file == str(path)
    assert record.sha256 == sha256_of(path)
    assert list(sources.modules) == [
        f"{PREFIX}models:value",
        f"{PREFIX}models:Box.attr",
    ]


def test_a_module_of_another_folder_with_the_same_name_is_refused(
    tmp_path: Path,
) -> None:
    """Python imports a module once, so a second folder cannot bring its own."""
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    second.mkdir()
    name = f"{PREFIX}clash"
    first_file = write_module(first, name, "value = 1\n")
    second_file = write_module(second, name, "value = 2\n")
    assert Sources(first).resolve(f"{name}:value") == 1
    refused = Sources(second)
    with pytest.raises(ConfigError, match="own process") as error:
        refused.resolve(f"{name}:value")
    message = str(error.value)
    assert name in message
    assert str(first_file) in message
    assert str(second_file) in message
    assert refused.modules == {}
    # * The same folder, resolving the module again, is no clash.
    again = Sources(first)
    assert again.resolve(f"{name}:value") == 1
    assert again.resolve(f"{name}:value") == 1


def test_a_package_of_another_folder_with_the_same_name_is_refused(
    tmp_path: Path,
) -> None:
    """The top-level name decides, also for a dotted module path of a package."""
    name = f"{PREFIX}package"
    folders = []
    for number in (1, 2):
        folder = tmp_path / f"folder_{number}"
        (folder / name).mkdir(parents=True)
        (folder / name / "__init__.py").write_text("", encoding="utf-8")
        write_module(folder / name, "inner", f"value = {number}\n")
        folders.append(folder)
    assert Sources(folders[0]).resolve(f"{name}.inner:value") == 1
    init_file = folders[1] / name / "__init__.py"
    with pytest.raises(ConfigError, match="own process") as error:
        Sources(folders[1]).resolve(f"{name}.inner:value")
    assert str(init_file) in str(error.value)
    assert Sources(folders[0]).resolve(f"{name}.inner:value") == 1


def test_a_folder_with_a_package_and_a_module_of_one_name_imports_the_package(
    tmp_path: Path,
) -> None:
    """The import system takes the package, so the check compares with its file."""
    name = f"{PREFIX}both"
    both = tmp_path / "both"
    (both / name).mkdir(parents=True)
    init_file = both / name / "__init__.py"
    init_file.write_text("value = 'package'\n", encoding="utf-8")
    write_module(both, name, "value = 'module'\n")
    sources = Sources(both)
    assert sources.resolve(f"{name}:value") == "package"
    assert sources.modules[f"{name}:value"].file == str(init_file)
    assert Sources(both).resolve(f"{name}:value") == "package"
    # * Another folder with the module file alone clashes with the imported package.
    only_module = tmp_path / "only_module"
    only_module.mkdir()
    module_file = write_module(only_module, name, "value = 'other'\n")
    with pytest.raises(ConfigError, match="own process") as error:
        Sources(only_module).resolve(f"{name}:value")
    assert f"imported from {init_file}" in str(error.value)
    assert str(module_file) in str(error.value)


def test_an_imported_module_without_a_file_clashes_with_a_file_of_the_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A module that has no file cannot be the one in the config folder."""
    name = f"{PREFIX}nofile_clash"
    monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    local = write_module(tmp_path, name, "value = 1\n")
    with pytest.raises(ConfigError, match="no file") as error:
        Sources(tmp_path).resolve(f"{name}:value")
    assert str(local) in str(error.value)


def test_a_missing_module_or_attribute_becomes_a_config_error(tmp_path: Path) -> None:
    """The message names the import path and the original error, chained."""
    write_module(tmp_path, f"{PREFIX}present", "value = 1\n")
    write_module(tmp_path, f"{PREFIX}broken", "import not_a_module_anywhere\n")
    sources = Sources(tmp_path)
    with pytest.raises(
        ConfigError, match=f"{PREFIX}absent:value.*No module named"
    ) as module_error:
        sources.resolve(f"{PREFIX}absent:value")
    assert isinstance(module_error.value.__cause__, ImportError)
    with pytest.raises(ConfigError, match=f"{PREFIX}present:other.*other"):
        sources.resolve(f"{PREFIX}present:other")
    with pytest.raises(ConfigError, match="not_a_module_anywhere"):
        sources.resolve(f"{PREFIX}broken:value")
    assert sources.modules == {}


def test_modules_without_a_file_or_with_a_missing_one_have_no_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Built-in and zipped modules have no readable file to hash."""
    no_file = types.ModuleType(f"{PREFIX}nofile")
    no_file.__dict__["thing"] = 5
    gone = types.ModuleType(f"{PREFIX}gone")
    gone.__dict__["thing"] = 6
    gone.__file__ = str(tmp_path / "gone.py")
    monkeypatch.setitem(sys.modules, no_file.__name__, no_file)
    monkeypatch.setitem(sys.modules, gone.__name__, gone)
    sources = Sources(tmp_path)
    assert sources.resolve(f"{PREFIX}nofile:thing") == 5
    assert sources.resolve(f"{PREFIX}gone:thing") == 6
    assert sources.modules[f"{PREFIX}nofile:thing"].file is None
    assert sources.modules[f"{PREFIX}nofile:thing"].sha256 is None
    assert sources.modules[f"{PREFIX}gone:thing"].file == str(tmp_path / "gone.py")
    assert sources.modules[f"{PREFIX}gone:thing"].sha256 is None


def test_files_are_hashed_in_blocks(tmp_path: Path) -> None:
    """A file of several blocks, and an empty file, give the digest of hashlib."""
    big = tmp_path / "big.bin"
    big.write_bytes(np.random.default_rng(0).bytes(2_500_000))
    assert file_sha256(big) == sha256_of(big)
    empty = tmp_path / "empty.bin"
    empty.write_bytes(b"")
    assert file_sha256(empty) == hashlib.sha256(b"").hexdigest()


# * Transforms


def test_transforms_resolve_by_import_path(pipeline_project: Path) -> None:
    """A Transform object passes, and other objects raise a config error."""
    sources = Sources(pipeline_project)
    assert sources.read_transform(None, None) is None
    transform = sources.read_transform("models:shift", None)
    assert transform is not None
    assert transform.description == "shift"
    with pytest.raises(
        ConfigError, match="'models:not_a_transform' names a int, not a Transform"
    ):
        sources.read_transform("models:not_a_transform", None)
    with pytest.raises(
        ConfigError, match="cannot resolve the import path 'models:nothing'"
    ):
        sources.read_transform("models:nothing", None)


def test_the_physical_transform_comes_from_the_eryn_priors(
    pipeline_project: Path,
) -> None:
    """Eryn priors give ErynPhysical; without priors the config cannot run."""
    sources = Sources(pipeline_project)
    _, priors = sources.read_space(
        PipelineConfig.model_validate(
            {
                "seed": 1,
                "output": "r",
                "space": {"eryn_priors": "models:priors"},
                "posteriors": [{"name": "a", "format": "table", "path": "p"}],
                "diagnose": {},
            }
        ).space
    )
    assert isinstance(sources.read_transform("physical", priors), ErynPhysical)
    with pytest.raises(
        ConfigError, match=r"cannot build the physical\(eryn\) transform"
    ):
        sources.read_transform("physical", None)


# * The space


def test_a_space_comes_from_a_file_an_inline_space_or_eryn_priors(
    pipeline_project: Path,
) -> None:
    """Each form gives a space, and only eryn priors give priors too."""
    folder = pipeline_project
    SPACE.to_yaml(folder / "space.yaml")
    sources = Sources(folder)
    from_file, priors = sources.read_space("space.yaml")
    assert from_file == SPACE
    assert priors is None
    assert sources.inputs[0].role == "space"
    assert sources.inputs[0].path == "space.yaml"
    assert sources.inputs[0].sha256 == sha256_of(folder / "space.yaml")
    inline, priors = sources.read_space(SPACE)
    assert inline is SPACE
    assert priors is None
    assert len(sources.inputs) == 1
    config = PipelineConfig.model_validate(
        {
            "seed": 1,
            "output": "r",
            "space": {
                "eryn_priors": "models:priors",
                "periodic": {"x2": 40.0},
                "spheres": [],
            },
            "posteriors": [{"name": "a", "format": "table", "path": "p"}],
            "diagnose": {},
        }
    )
    eryn_space, priors = sources.read_space(config.space)
    assert eryn_space.names == ("x0", "x1", "x2")
    assert priors is not None
    assert eryn_space["x2"].topology is Topology.CIRCLE
    assert eryn_space["x2"].period == 40.0
    assert eryn_space["x0"].topology is Topology.INTERVAL
    assert "models:priors" in sources.modules


def test_a_space_that_does_not_resolve_becomes_a_config_error(
    pipeline_project: Path,
) -> None:
    """A missing file, a bad file, bad priors and a bad period name the space."""
    folder = pipeline_project
    sources = Sources(folder)
    with pytest.raises(
        ConfigError, match=r"cannot read the space file 'nowhere\.yaml'"
    ):
        sources.read_space("nowhere.yaml")
    (folder / "bad_space.yaml").write_text("parameters: []\n", encoding="utf-8")
    with pytest.raises(
        ConfigError, match=r"cannot read the space file 'bad_space\.yaml'"
    ):
        sources.read_space("bad_space.yaml")
    (folder / "broken_space.yaml").write_text("parameters: [\n", encoding="utf-8")
    with pytest.raises(
        ConfigError, match=r"cannot read the space file 'broken_space\.yaml'"
    ):
        sources.read_space("broken_space.yaml")
    not_priors = PipelineConfig.model_validate(
        {
            "seed": 1,
            "output": "r",
            "space": {"eryn_priors": "models:not_a_transform"},
            "posteriors": [{"name": "a", "format": "table", "path": "p"}],
            "diagnose": {},
        }
    ).space
    with pytest.raises(
        ConfigError,
        match="cannot build the space from the eryn priors 'models:not_a_transform'",
    ):
        sources.read_space(not_priors)
    assert sources.inputs == []


# * Arrays


def test_arrays_are_lists_or_npy_files(tmp_path: Path) -> None:
    """A list becomes float64; a file loads without pickle and is recorded."""
    np.save(tmp_path / "fisher.npy", np.eye(2, dtype=np.float32))
    np.save(
        tmp_path / "objects.npy", np.array([{"a": 1}], dtype=object), allow_pickle=True
    )
    sources = Sources(tmp_path)
    listed = sources.read_array([[1, 2], [3, 4]], "compare fisher")
    assert listed.dtype == np.float64
    assert sources.inputs == []
    loaded = sources.read_array("fisher.npy", "compare fisher")
    assert loaded.dtype == np.float64
    np.testing.assert_array_equal(loaded, np.eye(2))
    (record,) = sources.inputs
    assert record.role == "compare fisher"
    assert record.path == "fisher.npy"
    assert record.size_bytes == (tmp_path / "fisher.npy").stat().st_size
    with pytest.raises(
        ConfigError, match=r"cannot read compare centre from 'nowhere\.npy'"
    ):
        sources.read_array("nowhere.npy", "compare centre")
    with pytest.raises(
        ConfigError, match=r"cannot read compare centre from 'objects\.npy'"
    ):
        sources.read_array("objects.npy", "compare centre")


# * Posteriors


def test_each_format_reads_into_chains_or_a_sample_set(
    pipeline_project: Path,
) -> None:
    """The five formats give chains and sets, with their records."""
    folder = pipeline_project
    sources = Sources(folder)
    config = PipelineConfig.model_validate(
        {
            "seed": 1,
            "output": "r",
            "space": {"eryn_priors": "models:priors"},
            "posteriors": [
                {"name": "mcmc", "format": "eryn", "path": "data/mcmc.h5"},
                {"name": "chains", "format": "chains", "path": "data/ar1.npz"},
                {"name": "samples", "format": "samples", "path": "data/draws.npz"},
                {"name": "table", "format": "table", "path": "data/table.parquet"},
                {
                    "name": "group",
                    "format": "hdf5",
                    "path": "data/peer.h5",
                    "group": "posterior",
                    "n_eff": 100,
                },
            ],
            "diagnose": {},
        }
    )
    space, priors = sources.read_space(config.space)
    loaded = [sources.read_posterior(each, space, priors) for each in config.posteriors]
    mcmc, chains, samples, table, group = loaded
    assert isinstance(mcmc, Chains)
    assert mcmc.shape == (120, 1, 8, 1, 3)
    assert isinstance(chains, Chains)
    assert chains.shape == (400, 1, 8, 1, 3)
    assert isinstance(samples, SampleSet)
    assert samples.shape == (1, 1, 300, 3)
    assert n_eff_of(samples) == 200.0
    assert isinstance(table, SampleSet)
    assert table.shape == (1, 1, 300, 3)
    assert isinstance(group, SampleSet)
    assert n_eff_of(group) == 100.0
    # * The table holds the columns in another order, which the names undo.
    by_name = pl.read_parquet(folder / "data" / "table.parquet").select(
        "x0", "x1", "x2"
    )
    np.testing.assert_array_equal(table.samples[0, 0], by_name.to_numpy())
    # * An .npz file has no names, so its last axis is read in order.
    in_order = np.load(folder / "data" / "draws.npz")["samples"]
    np.testing.assert_array_equal(samples.samples[0, 0], in_order)
    assert [record.role for record in sources.inputs] == [
        "posterior mcmc",
        "posterior chains",
        "posterior samples",
        "posterior table",
        "posterior group",
    ]
    record = sources.inputs[0]
    mcmc_file = folder / "data" / "mcmc.h5"
    assert record == InputRecord(
        "posterior mcmc",
        "data/mcmc.h5",
        str(mcmc_file.resolve()),
        mcmc_file.stat().st_size,
        sha256_of(mcmc_file),
    )


def test_the_eryn_reader_picks_columns_by_the_names_of_the_space(
    pipeline_project: Path,
) -> None:
    """A space with a subset of the sampled parameters reads that subset."""
    sources = Sources(pipeline_project)
    subset = ParamSpace.model_validate({"parameters": [{"name": "x2"}, {"name": "x0"}]})
    chains = sources.read_posterior(
        posterior(format="eryn", path="data/mcmc.h5"), subset, None
    )
    full = sources.read_posterior(
        posterior(format="eryn", path="data/mcmc.h5"), SPACE, None
    )
    assert isinstance(chains, Chains)
    assert isinstance(full, Chains)
    np.testing.assert_array_equal(chains.samples, full.samples[..., [2, 0]])


def test_reader_errors_become_config_errors_naming_the_posterior(
    pipeline_project: Path, tmp_path: Path
) -> None:
    """A missing file, name or size names the posterior and its path."""
    sources = Sources(pipeline_project)
    for fields in (
        {"format": "eryn", "path": "data/nowhere.h5"},
        {"format": "chains", "path": "data/nowhere.npz"},
        {"format": "samples", "path": "data/nowhere.npz"},
        {"format": "table", "path": "data/nowhere.parquet"},
        {"format": "hdf5", "path": "data/nowhere.h5", "group": "posterior"},
    ):
        with pytest.raises(
            ConfigError,
            match=f"cannot read the posterior 'run' from '{fields['path']}'",
        ):
            sources.read_posterior(posterior(**fields), SPACE, None)
    two = ParamSpace.model_validate({"parameters": [{"name": "x0"}, {"name": "x1"}]})
    for fields in (
        {"format": "chains", "path": "data/ar1.npz"},
        {"format": "samples", "path": "data/draws.npz"},
    ):
        with pytest.raises(ConfigError, match="has 3 parameters, but the space has 2"):
            sources.read_posterior(posterior(**fields), two, None)
    absent = ParamSpace.model_validate({"parameters": [{"name": "x0"}, {"name": "y"}]})
    with pytest.raises(ConfigError, match="no columns 'y'"):
        sources.read_posterior(
            posterior(format="table", path="data/table.parquet"), absent, None
        )
    with pytest.raises(ConfigError, match="has no group 'results'"):
        sources.read_posterior(
            posterior(format="hdf5", path="data/peer.h5", group="results"), SPACE, None
        )
    with pytest.raises(ConfigError, match="no parameters 'y'"):
        sources.read_posterior(
            posterior(format="eryn", path="data/mcmc.h5"), absent, None
        )
    leaves = tmp_path / "leaves.npz"
    np.savez(leaves, samples=np.zeros((10, 1, 4, 2, 3)))
    with pytest.raises(ConfigError, match="chains with 2 leaves are not supported"):
        sources.read_posterior(
            posterior(format="chains", path=str(leaves)), SPACE, None
        )
    assert sources.inputs == []


def test_a_missing_optional_dependency_passes_unchanged(
    pipeline_project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Installing the extra is the fix, so the error is not a config error."""
    pytest.importorskip("h5py")
    monkeypatch.setitem(sys.modules, "h5py", None)
    with pytest.raises(MissingDependencyError, match=r"mixed-signals\[hdf5\]"):
        Sources(pipeline_project).read_posterior(
            posterior(format="hdf5", path="data/peer.h5", group="posterior"),
            SPACE,
            None,
        )


def test_a_period_and_a_sphere_that_the_priors_reject_become_a_config_error(
    pipeline_project: Path,
) -> None:
    """The period 6 does not match the prior width, and the sphere needs a circle."""
    config = PipelineConfig.model_validate(
        {
            "seed": 1,
            "output": "r",
            "space": {
                "eryn_priors": "models:priors",
                "periodic": {"x0": 6.0},
                "spheres": [
                    {"azimuth": "x0", "polar": "x1", "polar_form": "sin_latitude"}
                ],
            },
            "posteriors": [{"name": "a", "format": "table", "path": "p"}],
            "diagnose": {},
        }
    )
    with pytest.raises(
        ConfigError, match="cannot build the space from the eryn priors"
    ):
        Sources(pipeline_project).read_space(config.space)
