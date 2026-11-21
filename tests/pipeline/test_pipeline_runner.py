"""Tests for the pipeline run: the estimate, the engines and the run folder."""

import datetime
import functools
import hashlib
import json
import math
import platform
import shutil
import warnings
from collections.abc import Callable, Iterator
from pathlib import Path
from types import MappingProxyType
from typing import Any, NamedTuple

import jax
import polars as pl
import pytest

import mixed_signals
from mixed_signals import _config
from mixed_signals.core.result import Result
from mixed_signals.engines.compare import compare
from mixed_signals.engines.cost import CostEstimate
from mixed_signals.engines.diagnose import diagnose
from mixed_signals.errors import ConfigError, CostWarning, ValidityWarning
from mixed_signals.mcmc.reduce import ReductionRecord
from mixed_signals.pipeline import (
    PipelineConfig,
    PipelineEstimate,
    PipelineRun,
    estimate_pipeline,
    load_config,
    run_pipeline,
    runner,
)

pytest.importorskip("eryn")
pytest.importorskip("h5py")

FILES = {
    "results.parquet",
    "config.json",
    "reductions.json",
    "cost.parquet",
    "provenance.json",
}
SHORT_RUN = r"fewer than 50 autocorrelation times"
HOLDS_RESULTS = r"already holds results\.parquet"
ENGINES = ("diagnose", "compare", "compare_signals")
# * Two flat posteriors and two cheap engines, which no chains reduce.
LIGHT = """
seed: 5
output: light_results
space: {eryn_priors: models:priors}
posteriors:
  - {name: table, format: table, path: data/table.parquet}
  - {name: group, format: hdf5, path: data/peer.h5, group: posterior}
"""
LIGHT_DIAGNOSE = "diagnose: {draws: 20, diagnostics: [railing], cap: 200}\n"
LIGHT_COMPARE = (
    "compare: {reference: table, metrics: [mean_shift], draws: 20, cap: 200}\n"
)
SIGNALS = "generator: 'models:generator', noise: 'models:noise'"


class FullRuns(NamedTuple):
    """Two runs of the full config into two folders, and what they left behind."""

    config_path: Path
    first: PipelineRun
    second: PipelineRun
    folder: Path
    other_folder: Path
    diagnoses: list[Any]
    refusal: str


def light(*sections: str, extra: str = "") -> str:
    """Return the light config with sections, and extra top-level lines."""
    return LIGHT + extra + "".join(sections)


def write_config(folder: Path, text: str, name: str = "config.yaml") -> Path:
    """Write a config next to the models of the project, so that they import."""
    path = folder / name
    path.write_text(text, encoding="utf-8")
    return path


def read_json(path: Path) -> Any:
    """Read a JSON file; NaN is accepted, as the run folder writes it."""
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_of(path: Path) -> str:
    """Return the SHA-256 of a file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def refuse_engine(calls: list[str], name: str, prepared: Any) -> Any:
    """Record that an engine ran, and fail."""
    calls.append(name)
    raise AssertionError(f"the {name} engine ran")


def fail_engine(prepared: Any) -> Any:
    """Stand in for an engine that fails while it runs."""
    raise RuntimeError("engine failed")


def record_budget(
    budgets: list[float], original: Callable[..., Any], *args: Any, **kwargs: Any
) -> Any:
    """Record the budget that an engine receives, then call it."""
    budgets.append(kwargs["budget"])
    return original(*args, **kwargs)


def keep_diagnosis(
    diagnoses: list[Any], original: Callable[..., Any], *args: Any, **kwargs: Any
) -> Any:
    """Call diagnose and keep what it returns."""
    diagnoses.append(original(*args, **kwargs))
    return diagnoses[-1]


@pytest.fixture
def no_engines(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[str]]:
    """Replace the engine runs by ones that record a call and fail."""
    calls: list[str] = []
    for name in ENGINES:
        monkeypatch.setitem(
            runner._RUNNERS, name, functools.partial(refuse_engine, calls, name)
        )
    yield calls
    assert calls == []


@pytest.fixture(scope="module")
def runs(
    pipeline_project: Path,
    pipeline_config: str,
    tmp_path_factory: pytest.TempPathFactory,
) -> FullRuns:
    """Run the full config into one folder, and again into a folder of stale files."""
    config_path = write_config(pipeline_project, pipeline_config, "full.yaml")
    folder = tmp_path_factory.mktemp("run_a")
    other_folder = tmp_path_factory.mktemp("run_b")
    diagnoses: list[Any] = []
    keep = functools.partial(keep_diagnosis, diagnoses, diagnose)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(runner, "diagnose", keep)
        with pytest.warns(ValidityWarning, match=SHORT_RUN):
            first = run_pipeline(config_path, output=folder)
        # * Stale files of an earlier run: the folder already holds a results table.
        for name in FILES:
            (other_folder / name).write_text("stale", encoding="utf-8")
        with pytest.raises(ConfigError) as refused:
            run_pipeline(config_path, output=other_folder)
        assert len(diagnoses) == 1
        with pytest.warns(ValidityWarning, match=SHORT_RUN):
            second = run_pipeline(config_path, output=other_folder, overwrite=True)
    return FullRuns(
        config_path, first, second, folder, other_folder, diagnoses, str(refused.value)
    )


# * The run folder


def test_the_run_folder_holds_five_files(runs: FullRuns) -> None:
    """Every file is written on every run, in the folder that was given."""
    for folder, run in ((runs.folder, runs.first), (runs.other_folder, runs.second)):
        assert {path.name for path in folder.iterdir()} == FILES
        assert run.output == folder.resolve()
    assert isinstance(runs.first.estimate, PipelineEstimate)


def test_results_read_back_with_the_rows_of_each_engine_in_run_order(
    runs: FullRuns,
) -> None:
    """The table holds diagnose, compare and compare_signals rows, in that order."""
    result = Result.read_parquet(runs.folder / "results.parquet")
    assert result.table.equals(runs.first.result.table)
    provenance = read_json(runs.folder / "provenance.json")
    assert [engine["name"] for engine in provenance["engines"]] == list(ENGINES)
    counts = [engine["rows"] for engine in provenance["engines"]]
    assert sum(counts) == len(result)
    table = result.table
    diagnose_rows = table.slice(0, counts[0])
    compare_rows = table.slice(counts[0], counts[1])
    signal_rows = table.slice(counts[0] + counts[1], counts[2])
    assert set(diagnose_rows["pair"].list.len()) == {1}
    assert {pair[0] for pair in compare_rows["pair"].to_list()} == {"mcmc"}
    assert set(compare_rows["metric"]) == {"mean_shift", "energy"}
    others = {"chains", "samples", "table", "group"}
    assert {pair[1] for pair in compare_rows["pair"].to_list()} == others
    assert signal_rows["space"].str.starts_with("data(").all()
    assert {pair[1] for pair in signal_rows["pair"].to_list()} == others


def test_the_hash_of_config_json_is_in_every_row(runs: FullRuns) -> None:
    """sha256sum config.json gives the hash, and every row carries it and the seed."""
    config_bytes = (runs.folder / "config.json").read_bytes()
    digest = hashlib.sha256(config_bytes).hexdigest()
    assert digest == runs.first.config_hash
    assert not config_bytes.endswith(b"\n")
    canonical = json.loads(config_bytes)
    assert set(canonical) == {"config", "space"}
    assert "output" not in canonical["config"]
    assert canonical["config"]["seed"] == 7
    names = [parameter["name"] for parameter in canonical["space"]["parameters"]]
    assert names == ["x0", "x1", "x2"]
    result = Result.read_parquet(runs.folder / "results.parquet")
    for provenance in result.table["provenance"]:
        assert provenance["config_hash"] == digest
        assert provenance["seed"] == 7
    versions = set(result.table["provenance"].struct.field("package_version"))
    assert versions == {mixed_signals.__version__}


def test_reductions_match_the_returned_ones_and_those_of_diagnose(
    runs: FullRuns,
) -> None:
    """The records are the pipeline's own, and equal the ones that diagnose made."""
    written = read_json(runs.folder / "reductions.json")
    assert list(written) == ["mcmc", "chains"]
    assert list(runs.first.reductions) == ["mcmc", "chains"]
    for name in ("mcmc", "chains"):
        record = ReductionRecord.model_validate(written[name])
        assert record == runs.first.reductions[name].record
        assert record == runs.diagnoses[0].reductions[name].record
    assert written["mcmc"]["burn_in"] == 20
    assert written["mcmc"]["thin"] == 4
    assert written["chains"]["burn_in"] == 10
    assert (runs.folder / "reductions.json").read_text().endswith("}\n")
    read_only: Any = runs.first.reductions
    with pytest.raises(TypeError):
        read_only["new"] = read_only["mcmc"]


def test_the_cost_table_is_the_estimate(runs: FullRuns) -> None:
    """cost.parquet holds the tables of the engines, stacked in run order."""
    estimate = runs.first.estimate
    assert list(estimate.estimates) == list(ENGINES)
    expected = pl.concat([each.table for each in estimate.estimates.values()])
    assert pl.read_parquet(runs.folder / "cost.parquet").equals(expected)
    assert estimate.seconds == sum(each.seconds for each in estimate.estimates.values())
    assert isinstance(estimate.estimates["compare"], CostEstimate)
    assert set(expected["engine"]) == set(ENGINES)


def test_the_provenance_records_the_run(runs: FullRuns, pipeline_project: Path) -> None:
    """The versions, the machine, the files that were read and the modules."""
    provenance = read_json(runs.folder / "provenance.json")
    assert (runs.folder / "provenance.json").read_text().endswith("}\n")
    assert provenance["config_hash"] == runs.first.config_hash
    assert provenance["seed"] == 7
    # * config.json leaves the budget out, so the provenance is where it is recorded.
    assert provenance["budget"] == _config.COST_BUDGET
    assert provenance["package_version"] == mixed_signals.__version__
    assert provenance["jax_version"] == jax.__version__
    assert provenance["python_version"] == platform.python_version()
    assert provenance["device_platform"] == jax.default_backend()
    assert provenance["machine"] == platform.platform()
    assert provenance["config_file"] == str(runs.config_path.resolve())
    assert provenance["output"] == str(runs.folder.resolve())
    started = datetime.datetime.fromisoformat(provenance["started"])
    assert started.utcoffset() == datetime.timedelta(0)
    assert started.microsecond == 0
    assert 0 < provenance["wall_seconds"] < 600
    assert provenance["estimated_seconds"] == runs.first.estimate.seconds
    for engine in provenance["engines"]:
        estimate = runs.first.estimate.estimates[engine["name"]]
        assert engine["estimated_seconds"] == estimate.seconds
        assert engine["wall_seconds"] > 0
    roles = [record["role"] for record in provenance["inputs"]]
    assert roles == [
        "posterior mcmc",
        "posterior chains",
        "posterior samples",
        "posterior table",
        "posterior group",
    ]
    for record in provenance["inputs"]:
        path = Path(record["absolute_path"])
        assert path == (pipeline_project / record["path"]).resolve()
        assert record["size_bytes"] == path.stat().st_size
        assert record["sha256"] == sha256_of(path)
    models = pipeline_project / "models.py"
    paths = [record["import_path"] for record in provenance["modules"]]
    assert paths == ["models:priors", "models:generator", "models:noise"]
    for record in provenance["modules"]:
        assert record["module"] == "models"
        assert record["file"] == str(models)
        assert record["sha256"] == sha256_of(models)


def test_two_runs_give_the_same_table_and_files(runs: FullRuns) -> None:
    """Same config, data and seed: identical results, config, reductions and costs."""
    assert runs.first.result.table.equals(runs.second.result.table)
    assert runs.first.config_hash == runs.second.config_hash
    for name in ("config.json", "reductions.json", "cost.parquet", "results.parquet"):
        first = (runs.folder / name).read_bytes()
        assert first == (runs.other_folder / name).read_bytes(), name
    # * The provenance holds the start time and the wall seconds, so it may differ.
    first_provenance = read_json(runs.folder / "provenance.json")
    second_provenance = read_json(runs.other_folder / "provenance.json")
    assert first_provenance["config_hash"] == second_provenance["config_hash"]
    assert first_provenance["output"] != second_provenance["output"]


def test_stale_files_stop_the_run_until_they_are_overwritten(runs: FullRuns) -> None:
    """A results table stops the run first, and overwrite replaces all five files."""
    assert "already holds results.parquet" in runs.refusal
    assert str(runs.other_folder.resolve()) in runs.refusal
    assert len(runs.diagnoses) == 2
    for name in FILES:
        assert (runs.other_folder / name).read_bytes() != b"stale"


# * Before the engines run


def test_an_existing_results_table_raises_before_any_work(
    pipeline_project: Path,
    tmp_path: Path,
    no_engines: list[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The check comes first, so no file of the data is read."""
    config_path = write_config(pipeline_project, light(LIGHT_DIAGNOSE), "existing.yaml")
    (tmp_path / "results.parquet").write_bytes(b"")
    reads: list[Any] = []
    monkeypatch.setattr(runner, "_prepare", reads.append)
    with pytest.raises(ConfigError, match=HOLDS_RESULTS):
        run_pipeline(config_path, output=tmp_path)
    assert reads == []
    (tmp_path / "results.parquet").unlink()
    (tmp_path / "file").write_text("x")
    with pytest.raises(ConfigError, match="exists and is not a folder"):
        run_pipeline(config_path, output=tmp_path / "file")


@pytest.mark.parametrize(
    ("section", "message"),
    [
        (
            "compare: {null: split_half, metrics: [mean_shift]}\n",
            r"the compare section.*split_half",
        ),
        ("compare: {metrics: [c2st]}\n", r"the compare section"),
        ("compare: {metrics: [nonsense]}\n", r"the compare section.*unknown metric"),
        (
            "compare: {whitening: fisher, fisher: [[1, 0], [0, 1]], centre: [0, 0]}\n",
            r"the compare section.*fisher must be shaped",
        ),
        (
            "compare: {whitening: fisher, centre: [0, 0, 0], "
            "fisher: [[1, 2, 0], [2, 1, 0], [0, 0, 1]]}\n",
            r"the compare section.*not positive definite",
        ),
        (
            "compare_signals: {generator: 'models:shift', noise: 'models:noise'}\n",
            r"the compare_signals section",
        ),
        (
            f"compare_signals: {{{SIGNALS}, null: split_half}}\n",
            r"the compare_signals section.*split_half",
        ),
        (
            "diagnose: {diagnostics: [shapes]}\n",
            r"the diagnose section.*unknown diagnostic families",
        ),
    ],
)
def test_an_argument_that_does_not_fit_names_its_section(
    pipeline_project: Path,
    tmp_path: Path,
    no_engines: list[str],
    section: str,
    message: str,
) -> None:
    """Every argument is checked before any engine runs."""
    config_path = write_config(pipeline_project, light(section), "bad.yaml")
    with pytest.raises(ConfigError, match=message):
        run_pipeline(config_path, output=tmp_path / "out")
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("transform", ["", "transform: 'models:shift', "])
def test_a_valid_fisher_whitening_passes_the_estimate(
    pipeline_project: Path, transform: str
) -> None:
    """The matrix builds its whitening in the row coordinates, mapped or not."""
    compare = (
        f"compare: {{reference: table, {transform}whitening: fisher, "
        "centre: [0, 0, 0], fisher: [[2, 0.3, 0], [0.3, 1, 0], [0, 0, 1]], "
        "metrics: [mean_shift], draws: 20, cap: 200}\n"
    )
    config_path = write_config(pipeline_project, light(compare), "fisher.yaml")
    estimate = estimate_pipeline(config_path)
    assert list(estimate.estimates) == ["compare"]


def test_a_run_of_chains_that_cannot_be_reduced_names_the_posterior(
    pipeline_project: Path, tmp_path: Path, no_engines: list[str]
) -> None:
    """A burn-in beyond the chains is a config error, before the engines."""
    entry = "  - {name: mcmc, format: eryn, path: data/mcmc.h5, burn_in: 500}\n"
    text = light(LIGHT_DIAGNOSE).replace("posteriors:\n", "posteriors:\n" + entry)
    config_path = write_config(pipeline_project, text, "burn.yaml")
    with pytest.raises(ConfigError, match="reduce the chains of the posterior 'mcmc'"):
        run_pipeline(config_path, output=tmp_path / "out")


def test_a_file_that_does_not_read_names_the_posterior(
    pipeline_project: Path, tmp_path: Path, no_engines: list[str]
) -> None:
    """A missing file is a config error, from the readers."""
    text = light(LIGHT_DIAGNOSE).replace("table.parquet", "nowhere.parquet")
    config_path = write_config(pipeline_project, text, "missing.yaml")
    with pytest.raises(
        ConfigError, match=r"posterior 'table' from 'data/nowhere\.parquet'"
    ):
        run_pipeline(config_path, output=tmp_path / "out")


# * The estimate and the budget


def test_estimating_writes_nothing(
    pipeline_project: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The estimate of each engine, in run order, from a path or a config."""
    text = light(LIGHT_DIAGNOSE, LIGHT_COMPARE)
    config_path = write_config(pipeline_project, text, "estimate.yaml")
    monkeypatch.chdir(tmp_path)
    estimate = estimate_pipeline(config_path)
    assert isinstance(estimate, PipelineEstimate)
    assert list(estimate.estimates) == ["diagnose", "compare"]
    assert isinstance(estimate.estimates, MappingProxyType)
    assert estimate.seconds == sum(each.seconds for each in estimate.estimates.values())
    assert estimate.seconds > 0
    assert not (pipeline_project / "light_results").exists()
    assert list(tmp_path.iterdir()) == []
    again = estimate_pipeline(load_config(config_path))
    assert again.seconds == estimate.seconds


def test_a_total_above_the_budget_raises_one_warning_and_goes_on(
    pipeline_project: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The pipeline warns once for all engines, which run with an infinite budget."""
    text = light(LIGHT_DIAGNOSE, LIGHT_COMPARE, extra="budget: 0.000001\n")
    config_path = write_config(pipeline_project, text, "budget.yaml")
    budgets: list[float] = []
    for name, original in (("diagnose", diagnose), ("compare", compare)):
        spy = functools.partial(record_budget, budgets, original)
        monkeypatch.setattr(runner, name, spy)
    with pytest.warns(CostWarning, match="the pipeline run is estimated at") as warned:
        run = run_pipeline(config_path, output=tmp_path / "out")
    assert [each.category for each in warned] == [CostWarning]
    assert budgets == [math.inf, math.inf]
    assert (tmp_path / "out" / "results.parquet").exists()
    assert len(run.result) > 0
    quiet = write_config(
        pipeline_project,
        light(LIGHT_DIAGNOSE, LIGHT_COMPARE, extra="budget: .inf\n"),
        "inf.yaml",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        run_pipeline(quiet, output=tmp_path / "quiet")
    # * JSON writes an infinite budget as Infinity, which Python reads back.
    assert read_json(tmp_path / "quiet" / "provenance.json")["budget"] == math.inf
    assert read_json(tmp_path / "out" / "provenance.json")["budget"] == 0.000_001


def test_progress_goes_to_the_logger(
    pipeline_project: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The estimate, each engine and the files are logged at INFO."""
    text = light(LIGHT_DIAGNOSE, LIGHT_COMPARE)
    config_path = write_config(pipeline_project, text, "log.yaml")
    with caplog.at_level("INFO", logger="mixed_signals.pipeline"):
        run_pipeline(config_path, output=tmp_path / "out")
    lines = [record.getMessage() for record in caplog.records]
    assert lines[0].startswith("Estimated run time on one CPU of the developers'")
    assert "against a budget of 3,600 s" in lines[0]
    assert lines[1].startswith("  diagnose: ")
    assert lines[2].startswith("  compare: ")
    assert lines[3].startswith("Costliest statistics: ")
    assert lines[4].startswith("diagnose finished: ")
    assert " rows in " in lines[4]
    assert lines[5].startswith("compare finished: ")
    assert lines[6] == (
        "Wrote results.parquet, config.json, reductions.json, cost.parquet and"
        f" provenance.json to {(tmp_path / 'out').resolve()}"
    )
    assert len(lines) == 7
    caplog.clear()
    unlimited = light(LIGHT_COMPARE, extra="budget: .inf\n")
    path = write_config(pipeline_project, unlimited, "log_inf.yaml")
    with caplog.at_level("INFO", logger="mixed_signals.pipeline"):
        estimate_pipeline(path)
    assert "against no budget" in caplog.records[0].getMessage()


# * Transforms and the folder of the config


def test_the_physical_transform_names_the_space_of_the_rows(
    pipeline_project: Path, tmp_path: Path
) -> None:
    """ErynPhysical of the priors, in the space column of the compare rows."""
    compare = LIGHT_COMPARE.replace("compare: {", "compare: {transform: physical, ")
    config_path = write_config(pipeline_project, light(compare), "physical.yaml")
    run = run_pipeline(config_path, output=tmp_path / "out")
    spaces = run.result.table["space"]
    assert len(spaces) > 0
    assert spaces.str.starts_with("physical(eryn)").all()


def test_a_transform_resolves_by_import_path(
    pipeline_project: Path, tmp_path: Path
) -> None:
    """A Transform object of models.py, in the space column of the rows."""
    prefix = "transform: 'models:shift', "
    compare = LIGHT_COMPARE.replace("compare: {", "compare: {" + prefix)
    diagnose = LIGHT_DIAGNOSE.replace("diagnose: {", "diagnose: {" + prefix)
    config_path = write_config(pipeline_project, light(diagnose, compare), "shift.yaml")
    run = run_pipeline(config_path, output=tmp_path / "out")
    table = run.result.table
    assert (
        table.filter(pl.col("pair").list.len() == 2)["space"]
        .str.starts_with("shift")
        .all()
    )
    assert table["space"].str.contains("shift").all()


def test_relative_paths_start_at_the_config_folder(
    pipeline_project: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    own_models: None,
) -> None:
    """The output of the config starts at its folder; an argument starts at the cwd."""
    copy = tmp_path / "copy"
    shutil.copytree(pipeline_project, copy)
    config_path = write_config(copy, light(LIGHT_COMPARE), "config.yaml")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    run = run_pipeline(config_path)
    assert run.output == (copy / "light_results").resolve()
    assert {path.name for path in (copy / "light_results").iterdir()} == FILES
    # * The copy imported its own models.py, not the cached one of the project.
    modules = read_json(run.output / "provenance.json")["modules"]
    assert {record["file"] for record in modules} == {
        str((copy / "models.py").resolve())
    }
    other = run_pipeline(config_path, output="given")
    assert other.output == (elsewhere / "given").resolve()
    assert other.result.table.equals(run.result.table)
    with pytest.raises(ConfigError, match=HOLDS_RESULTS):
        run_pipeline(config_path)
    assert run_pipeline(config_path, overwrite=True).output == run.output


def test_a_config_from_a_mapping_has_no_file(
    pipeline_project: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Relative paths start at the working folder, and the provenance has no file."""
    group = {"name": "group", "format": "hdf5", "path": "data/peer.h5"}
    # * The folder of the paths is the working folder when the mapping is validated.
    monkeypatch.chdir(pipeline_project)
    config = PipelineConfig.model_validate(
        {
            "seed": 5,
            "output": "out",
            "space": {"eryn_priors": "models:priors"},
            "posteriors": [
                {"name": "table", "format": "table", "path": "data/table.parquet"},
                {**group, "group": "posterior"},
            ],
            "compare": {"metrics": ["mean_shift"], "draws": 20, "cap": 200},
        }
    )
    monkeypatch.chdir(tmp_path)
    run = run_pipeline(config, output=tmp_path / "out")
    assert read_json(tmp_path / "out" / "provenance.json")["config_file"] is None
    assert run.reductions == {}
    assert read_json(tmp_path / "out" / "reductions.json") == {}


# * Failures


def test_a_failing_engine_leaves_no_folder(
    pipeline_project: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The folder is written at the end, so a failed run writes nothing."""
    config_path = write_config(pipeline_project, light(LIGHT_COMPARE), "fail.yaml")
    monkeypatch.setitem(runner._RUNNERS, "compare", fail_engine)
    out = tmp_path / "out"
    with pytest.raises(RuntimeError, match="engine failed"):
        run_pipeline(config_path, output=out)
    assert not out.exists()
