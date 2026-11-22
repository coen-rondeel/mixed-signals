"""Tests for reading a run folder: parts, settings, problems, levels and plot keys."""

import hashlib
import json
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pytest
from pydantic import ValidationError

from mixed_signals import prng
from mixed_signals.core.result import Provenance, Result
from mixed_signals.core.samples import SampleSet
from mixed_signals.errors import ReportError
from mixed_signals.mcmc.reduce import ReductionRecord
from mixed_signals.report.folder import (
    DEFAULT_PROBLEMS,
    PosteriorSamples,
    ReportSettings,
    RunFolder,
    check_problems,
    open_run_folder,
    plot_key,
    read_run_folder,
    samples_for,
    select_problems,
    significance_level,
)
from mixed_signals.sigma.thresholds import SIGNIFICANCE_LEVEL

NAMES = ("mcmc", "table", "group")


def only_results(source: Path, target: Path) -> Path:
    """Copy the result table alone into a new folder, as an API user would write it."""
    target.mkdir()
    shutil.copy(source / "results.parquet", target / "results.parquet")
    return target


def with_settings(folder: RunFolder, **changes: Any) -> RunFolder:
    """Return the folder with other settings of the report."""
    settings = ReportSettings(**{**folder.settings.model_dump(), **changes})
    return folder._replace(settings=settings)


def flagged_result(
    result_of: Callable[..., Result], flags: dict[int, int], clean: int = 0
) -> Result:
    """Return a result with that many flagged rows in each problem, then clean rows."""
    rows: list[dict[str, Any]] = []
    for problem, count in flags.items():
        rows.extend(
            {"problem": problem, "threshold": 1.0, "flag": True} for _ in range(count)
        )
        rows.extend(
            {"problem": problem, "threshold": 1.0, "flag": False} for _ in range(clean)
        )
    return result_of(*rows)


@pytest.fixture(scope="module")
def full(report_run_folder: Path) -> RunFolder:
    """The full run folder, read once."""
    return read_run_folder(report_run_folder)


# * Reading the parts


def test_a_full_run_folder_reads_every_part(
    full: RunFolder, report_run_folder: Path
) -> None:
    """Names, space, config, hash, samples, reductions, cost and provenance."""
    assert full.path == report_run_folder
    assert full.names == NAMES
    assert full.space is not None
    assert full.space.names == ("x0", "x1", "x2")
    assert full.config is not None
    assert full.config["seed"] == 11
    config_bytes = (report_run_folder / "config.json").read_bytes()
    assert full.config_hash == hashlib.sha256(config_bytes).hexdigest()
    assert list(full.reductions) == ["mcmc"]
    written = json.loads((report_run_folder / "reductions.json").read_text())
    assert full.reductions["mcmc"] == ReductionRecord.model_validate(written["mcmc"])
    assert full.cost is not None
    assert full.cost.equals(pl.read_parquet(report_run_folder / "cost.parquet"))
    assert full.provenance is not None
    assert full.provenance["seed"] == 11
    assert full.settings == ReportSettings()
    assert full.settings.problems == DEFAULT_PROBLEMS
    assert full.settings.plot_cap is None
    table = Result.read_parquet(report_run_folder / "results.parquet").table
    assert full.result.table.equals(table)


def test_the_samples_are_the_archives_in_config_order(
    full: RunFolder, report_run_folder: Path
) -> None:
    """The input samples and the mapped copies of compare, with their spaces."""
    assert set(full.samples) == {"input", "compare"}
    for section, subfolder in (("input", "samples"), ("compare", "samples/compare")):
        posterior_samples = full.samples[section]
        assert isinstance(posterior_samples, PosteriorSamples)
        assert list(posterior_samples.sets) == list(NAMES)
        for position, name in enumerate(NAMES):
            archive = report_run_folder / subfolder / f"{position}.npz"
            expected = SampleSet.from_npz(archive)
            read = posterior_samples.sets[name]
            np.testing.assert_array_equal(read.samples, expected.samples)
            np.testing.assert_array_equal(read.log_weights, expected.log_weights)
    original, mapped = full.samples["input"].space, full.samples["compare"].space
    assert original == full.space
    for before, after in zip(original.parameters, mapped.parameters, strict=True):
        assert after.bounds == (before.bounds[0] + 1.0, before.bounds[1] + 1.0)
    shift = (
        full.samples["compare"].sets["table"].samples
        - full.samples["input"].sets["table"].samples
    )
    np.testing.assert_allclose(shift, 1.0)


def test_a_folder_with_only_results_reads_with_names_from_the_pairs(
    report_run_folder: Path, tmp_path: Path
) -> None:
    """Every other part is optional: no samples, space, config or provenance."""
    folder = read_run_folder(only_results(report_run_folder, tmp_path / "bare"))
    assert folder.names == NAMES
    assert folder.space is None
    assert folder.config is None
    assert folder.config_hash is None
    assert folder.samples == {}
    assert folder.reductions == {}
    assert folder.cost is None
    assert folder.provenance is None
    assert folder.settings == ReportSettings()


def test_samples_need_the_names_of_the_config(
    report_run_folder: Path, tmp_path: Path
) -> None:
    """Without config.json the archives stay unread, since their names are unknown."""
    copy = tmp_path / "copy"
    shutil.copytree(report_run_folder, copy)
    (copy / "config.json").unlink()
    assert read_run_folder(copy).samples == {}


def test_an_incomplete_sample_folder_leaves_the_section_out(
    report_run_folder: Path, tmp_path: Path
) -> None:
    """A missing archive, or a section without its space file, is not read."""
    copy = tmp_path / "copy"
    shutil.copytree(report_run_folder, copy)
    (copy / "samples" / "compare" / "space.yaml").unlink()
    assert set(read_run_folder(copy).samples) == {"input"}
    (copy / "samples" / "1.npz").unlink()
    assert read_run_folder(copy).samples == {}
    shutil.rmtree(copy / "samples")
    assert read_run_folder(copy).samples == {}


def test_a_folder_without_results_raises_and_opens_as_none(tmp_path: Path) -> None:
    """The error names the folder; the notebook's open returns None instead."""
    with pytest.raises(ReportError, match=str(tmp_path)):
        read_run_folder(tmp_path)
    assert open_run_folder(tmp_path) is None
    assert open_run_folder(tmp_path / "nowhere") is None


def test_open_returns_the_folder_that_read_gives(
    report_run_folder: Path, full: RunFolder
) -> None:
    """Opening a run folder is reading it."""
    opened = open_run_folder(report_run_folder)
    assert opened is not None
    assert opened.names == full.names
    assert opened.config_hash == full.config_hash


# * The settings


def test_settings_round_trip_through_json() -> None:
    """A count, all, a list and a plot cap survive report.json."""
    for problems in (3, "all", (4, 1)):
        settings = ReportSettings(problems=problems, plot_cap=500)
        text = settings.model_dump_json(indent=2)
        assert ReportSettings.model_validate_json(text) == settings
    assert ReportSettings(problems=[2, 0]).problems == (2, 0)


@pytest.mark.parametrize(
    "text",
    [
        '{"problems": 0}',
        '{"problems": []}',
        '{"problems": [1, 1]}',
        '{"problems": [-1]}',
        '{"problems": "some"}',
        '{"plot_cap": 0}',
        '{"colour": "red"}',
        "not json",
    ],
)
def test_an_invalid_report_json_raises_pydantics_error(
    report_run_folder: Path, tmp_path: Path, text: str
) -> None:
    """The settings are validated when the folder is read."""
    folder = only_results(report_run_folder, tmp_path / "bad")
    (folder / "report.json").write_text(text)
    with pytest.raises(ValidationError):
        read_run_folder(folder)


def test_a_listed_problem_outside_the_result_raises(
    report_run_folder: Path, tmp_path: Path
) -> None:
    """A list must name problems that the result holds, whether read or checked."""
    folder = only_results(report_run_folder, tmp_path / "listed")
    (folder / "report.json").write_text('{"problems": [0, 4, 7]}')
    with pytest.raises(ReportError, match=r"problems \[4, 7\] are not in the result"):
        read_run_folder(folder)
    table = Result.read_parquet(folder / "results.parquet").table
    check_problems(ReportSettings(problems=(0,)), table)
    check_problems(ReportSettings(problems=3), table)
    check_problems(ReportSettings(problems="all"), table)
    with pytest.raises(ReportError):
        check_problems(ReportSettings(problems=(1,)), table)


# * The samples of a section, and the level


def test_samples_for_prefers_the_mapped_copies(full: RunFolder) -> None:
    """A section with a transform reads its copies, the others the input samples."""
    assert samples_for(full, "compare") is full.samples["compare"]
    assert samples_for(full, "diagnose") is full.samples["input"]
    assert samples_for(full._replace(samples={}), "compare") is None
    assert samples_for(full._replace(samples={}), "diagnose") is None


def test_the_level_comes_from_the_section_of_the_config(full: RunFolder) -> None:
    """The fixture sets the level of compare and of compare_signals to 1."""
    assert significance_level(full, "compare") == 1.0
    assert significance_level(full, "compare_signals") == 1.0
    assert significance_level(full, "diagnose") == SIGNIFICANCE_LEVEL
    assert significance_level(full, "reweight") == SIGNIFICANCE_LEVEL
    assert (
        significance_level(full._replace(config=None), "compare") == SIGNIFICANCE_LEVEL
    )
    section = {"compare": {"draws": 20}}
    assert (
        significance_level(full._replace(config=section), "compare")
        == SIGNIFICANCE_LEVEL
    )


# * The problems to show


def test_a_count_ranks_problems_by_flagged_rows_with_ties_by_index(
    full: RunFolder, result_of: Callable[..., Result]
) -> None:
    """Most flagged first; equal counts go by index; a null flag does not count."""
    result = flagged_result(result_of, {0: 1, 1: 3, 2: 3, 3: 0, 4: 2}, clean=1)
    folder = full._replace(result=result)
    assert select_problems(with_settings(folder, problems=2)) == (1, 2)
    assert select_problems(with_settings(folder, problems=4)) == (1, 2, 4, 0)
    assert select_problems(with_settings(folder, problems=99)) == (1, 2, 4, 0, 3)
    unflagged = result_of(
        {"problem": 0}, {"problem": 1, "threshold": 1.0, "flag": True}
    )
    ranked = with_settings(full._replace(result=unflagged), problems=1)
    assert select_problems(ranked) == (1,)


def test_all_gives_every_problem_and_a_list_its_own_order(
    full: RunFolder, result_of: Callable[..., Result]
) -> None:
    """The listed problems keep their order, and all runs in increasing order."""
    result = flagged_result(result_of, {2: 1, 0: 0, 1: 4}, clean=1)
    folder = full._replace(result=result)
    assert select_problems(with_settings(folder, problems="all")) == (0, 1, 2)
    assert select_problems(with_settings(folder, problems=(2, 0))) == (2, 0)


# * The key of a plot subsample


def test_the_plot_key_derives_from_the_seed_of_the_run(full: RunFolder) -> None:
    """The path is report, plot, section, problem and name below the root key."""
    expected = prng.derive_key(
        prng.root_key(11), "report", "plot", "compare", 0, "mcmc"
    )
    key = plot_key(full, "compare", 0, "mcmc")
    assert prng.format_key(key) == prng.format_key(expected)
    others = [
        plot_key(full, "diagnose", 0, "mcmc"),
        plot_key(full, "compare", 1, "mcmc"),
        plot_key(full, "compare", 0, "table"),
    ]
    assert len({prng.format_key(other) for other in [key, *others]}) == 4


def key_text(folder: RunFolder) -> str:
    """Return the plot key of one posterior of a folder as text."""
    return prng.format_key(plot_key(folder, "compare", 0, "mcmc"))


def seeded_text(seed: int) -> str:
    """Return the plot key that a run of that seed must give, as text."""
    root = prng.root_key(seed)
    return prng.format_key(
        prng.derive_key(root, "report", "plot", "compare", 0, "mcmc")
    )


def test_the_seed_falls_back_to_the_rows_and_then_to_zero(
    full: RunFolder, result_of: Callable[..., Result]
) -> None:
    """Without provenance.json the first row gives the seed; without one, 0."""
    no_file = full._replace(provenance=None)
    assert key_text(no_file) == seeded_text(11)
    assert key_text(full._replace(provenance={"engines": []})) == seeded_text(11)
    unseeded = result_of({"provenance": Provenance.collect()})
    assert key_text(no_file._replace(result=unseeded)) == seeded_text(0)
    assert key_text(no_file._replace(result=Result.empty())) == seeded_text(0)
