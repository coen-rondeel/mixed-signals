"""Tests for the ``mixed-signals`` command line."""

import logging
import re
import shutil
import sys
from importlib.metadata import entry_points
from pathlib import Path

import pytest

import mixed_signals
from mixed_signals import cli
from mixed_signals.errors import CostWarning
from mixed_signals.pipeline import runner

FILES = {
    "results.parquet",
    "config.json",
    "reductions.json",
    "cost.parquet",
    "provenance.json",
}
# * Two flat posteriors and the cheapest engines: the commands run in a few seconds.
CONFIG = """
seed: 5
output: results
space: {eryn_priors: models:priors}
posteriors:
  - {name: table, format: table, path: data/table.parquet}
  - {name: group, format: hdf5, path: data/peer.h5, group: posterior}
diagnose: {draws: 20, diagnostics: [railing], cap: 200}
compare: {reference: table, metrics: [mean_shift], draws: 20, cap: 200}
"""


def write_config(folder: Path, text: str = CONFIG) -> Path:
    """Write a config next to the models of the project, so that they import."""
    path = folder / "cli.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def fail_engine(prepared: object) -> object:
    """Stand in for an engine that fails while it runs."""
    raise RuntimeError("engine failed")


def test_version_flag_prints_version(capsys: pytest.CaptureFixture[str]) -> None:
    """``--version`` prints the package version and exits with code 0."""
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["--version"])
    assert exit_info.value.code == 0
    assert mixed_signals.__version__ in capsys.readouterr().out


def test_no_arguments_prints_usage(capsys: pytest.CaptureFixture[str]) -> None:
    """Without arguments the command prints its usage and succeeds."""
    assert cli.main([]) == 0
    assert "usage: mixed-signals" in capsys.readouterr().out


def test_console_script_points_at_main() -> None:
    """The installed ``mixed-signals`` script calls ``cli.main``."""
    (script,) = entry_points(group="console_scripts", name="mixed-signals")
    assert script.value == "mixed_signals.cli:main"


def test_the_help_lists_the_run_command(capsys: pytest.CaptureFixture[str]) -> None:
    """The usage names the subcommand, and its help names the flags."""
    assert cli.main([]) == 0
    assert "run" in capsys.readouterr().out
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["run", "--help"])
    assert exit_info.value.code == 0
    help_text = capsys.readouterr().out
    for flag in ("--output", "--overwrite", "--dry-run"):
        assert flag in help_text


def test_a_run_needs_a_config(capsys: pytest.CaptureFixture[str]) -> None:
    """The argument parser rejects a run without a config, with exit code 2."""
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["run"])
    assert exit_info.value.code == 2
    assert "config" in capsys.readouterr().err


def test_a_dry_run_prints_the_estimate_and_writes_nothing(
    pipeline_project: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The estimate of every engine on standard output, and no run folder."""
    config_path = write_config(pipeline_project)
    assert (
        cli.main(
            ["run", str(config_path), "--dry-run", "--output", str(tmp_path / "out")]
        )
        == 0
    )
    captured = capsys.readouterr()
    lines = captured.out.splitlines()
    assert lines[0].startswith("Estimated run time on one CPU of the developers'")
    assert [line.split(":")[0] for line in lines[1:3]] == ["  diagnose", "  compare"]
    assert lines[3].startswith("Costliest statistics: ")
    assert len(lines) == 4
    assert captured.err == ""
    assert not (tmp_path / "out").exists()
    assert not (pipeline_project / "results").exists()


def test_a_run_writes_the_folder_and_prints_the_progress(
    pipeline_project: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The estimate, each engine and the files, and exit code 0."""
    config_path = write_config(pipeline_project)
    out = tmp_path / "out"
    assert cli.main(["run", str(config_path), "--output", str(out)]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].startswith("Estimated run time")
    assert [line.split(" finished")[0] for line in lines[4:6]] == [
        "diagnose",
        "compare",
    ]
    assert lines[6] == (
        "Wrote results.parquet, config.json, reductions.json, cost.parquet and"
        f" provenance.json to {out.resolve()}"
    )
    assert {path.name for path in out.iterdir()} == FILES


def test_a_run_without_output_writes_the_folder_of_the_config(
    pipeline_project: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    own_models: None,
) -> None:
    """The run folder of the config starts at the folder of the config file."""
    copy = tmp_path / "copy"
    shutil.copytree(pipeline_project, copy)
    config_path = write_config(copy)
    assert cli.main(["run", str(config_path)]) == 0
    assert "Wrote results.parquet" in capsys.readouterr().out
    assert {path.name for path in (copy / "results").iterdir()} == FILES
    assert not (pipeline_project / "results").exists()


def test_an_existing_folder_needs_overwrite(
    pipeline_project: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Without --overwrite the run stops with code 2; with it, the run goes on."""
    config_path = write_config(pipeline_project)
    out = tmp_path / "out"
    assert cli.main(["run", str(config_path), "--output", str(out)]) == 0
    capsys.readouterr()
    assert cli.main(["run", str(config_path), "--output", str(out)]) == 2
    captured = capsys.readouterr()
    assert captured.err.startswith("mixed-signals: error: ")
    assert "already holds results.parquet" in captured.err
    assert "Traceback" not in captured.err
    assert captured.out == ""
    assert cli.main(["run", str(config_path), "--output", str(out), "--overwrite"]) == 0
    assert "Wrote results.parquet" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("seed: true\n", "validation errors? for PipelineConfig"),
        ("seed: 1\nseed: 2\n", "found the key 'seed' more than once"),
        ("- a\n- b\n", "must hold a mapping"),
        (CONFIG.replace("5", "5.5", 1), "seed"),
        (CONFIG.replace("[railing]", "[shapes]"), "unknown diagnostic families"),
        (
            CONFIG.replace("table.parquet", "nowhere.parquet"),
            "cannot read the posterior",
        ),
        (
            CONFIG.replace("models:priors", "models:missing"),
            "cannot resolve the import path",
        ),
    ],
)
def test_an_invalid_config_returns_two_with_the_message(
    pipeline_project: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    text: str,
    message: str,
) -> None:
    """The message goes to standard error, with no traceback."""
    config_path = write_config(pipeline_project, text)
    assert cli.main(["run", str(config_path), "--output", str(tmp_path / "out")]) == 2
    captured = capsys.readouterr()
    assert captured.err.startswith("mixed-signals: error: ")
    assert re.search(message, captured.err)
    assert "Traceback" not in captured.err
    assert not (tmp_path / "out").exists()


def test_a_missing_config_returns_two(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A config file that does not exist is an invalid argument."""
    assert cli.main(["run", str(tmp_path / "nowhere.yaml")]) == 2
    assert "cannot read the config" in capsys.readouterr().err


def test_a_missing_dependency_returns_two(
    pipeline_project: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The message names the extra to install."""
    config_path = write_config(pipeline_project)
    monkeypatch.setitem(sys.modules, "h5py", None)
    assert cli.main(["run", str(config_path), "--output", str(tmp_path / "out")]) == 2
    assert 'pip install "mixed-signals[hdf5]"' in capsys.readouterr().err


def test_an_engine_failure_returns_one_with_a_traceback(
    pipeline_project: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failure while running prints the traceback and exits with code 1."""
    config_path = write_config(pipeline_project)
    monkeypatch.setitem(runner._RUNNERS, "compare", fail_engine)
    assert cli.main(["run", str(config_path), "--output", str(tmp_path / "out")]) == 1
    captured = capsys.readouterr()
    assert "Traceback (most recent call last)" in captured.err
    assert "RuntimeError: engine failed" in captured.err
    assert not (tmp_path / "out").exists()


def test_the_cost_warning_is_a_python_warning(
    pipeline_project: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A total above the budget warns through Python, and the run goes on."""
    config_path = write_config(pipeline_project, CONFIG + "budget: 0.000001\n")
    with pytest.warns(CostWarning):
        assert (
            cli.main(["run", str(config_path), "--output", str(tmp_path / "out")]) == 0
        )
    assert "Wrote results.parquet" in capsys.readouterr().out


def test_repeated_calls_do_not_print_a_line_twice(
    pipeline_project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The handler lives for one call, so the logger returns to how it was."""
    config_path = write_config(pipeline_project)
    logger = logging.getLogger("mixed_signals.pipeline")
    handlers, level = list(logger.handlers), logger.level
    outputs = []
    for _ in range(3):
        assert cli.main(["run", str(config_path), "--dry-run"]) == 0
        outputs.append(capsys.readouterr().out)
        assert logger.handlers == handlers
        assert logger.level == level
    assert outputs[0] == outputs[1] == outputs[2]
    assert len(outputs[0].splitlines()) == 4
    # * A failing call restores the logger too.
    assert cli.main(["run", str(pipeline_project / "nowhere.yaml")]) == 2
    capsys.readouterr()
    assert logger.handlers == handlers
