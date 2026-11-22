"""Tests for the notebook: it runs on a run folder, and stops outside one."""

import shutil
import subprocess
import sys
from pathlib import Path

from mixed_signals.report import notebook
from mixed_signals.report.folder import RunFolder
from mixed_signals.report.writer import write_report

HEADINGS = (
    "mixed signals report",
    "Diagnostics",
    "Comparisons",
    "Warnings",
    "Inputs and reductions",
    "Run time and cost",
    "Configuration",
)


def test_the_notebook_runs_on_a_run_folder_and_shows_every_section(
    report_run_folder: Path,
) -> None:
    """The cells give the headings, a posterior and a pair; no cell raises."""
    outputs, definitions = notebook.app.run(defs={"run_folder": report_run_folder})
    page = "".join(output.text for output in outputs if output is not None)
    for heading in HEADINGS:
        assert heading in page, heading
    assert "mcmc" in page
    assert "mcmc vs table" in page
    assert "callout" not in page
    folder = definitions["folder"]
    assert isinstance(folder, RunFolder)
    assert folder.names == ("mcmc", "table", "group")
    assert definitions["run_folder"] == report_run_folder
    # * The batch section has nothing to say for one problem, so its cell shows nothing.
    assert sum(output is None for output in outputs) >= 3


def test_outside_a_run_folder_the_notebook_stops_with_its_callout(
    tmp_path: Path,
) -> None:
    """A folder without results gives a note, no section and no exception."""
    outputs, definitions = notebook.app.run(defs={"run_folder": tmp_path})
    shown = [output for output in outputs if output is not None]
    assert len(shown) == 1
    html = shown[0].text
    assert "reads the run folder that it sits in" in html
    assert "mixed-signals report RUN_FOLDER" in html
    assert "folder" not in definitions or definitions["folder"] is None


def test_a_copy_runs_as_a_script_in_its_run_folder(
    report_run_folder: Path, tmp_path: Path
) -> None:
    """Python runs report.py on the folder it sits in, through mo.notebook_dir."""
    copy = tmp_path / "run"
    shutil.copytree(report_run_folder, copy)
    write_report(copy, html=False)
    completed = subprocess.run(
        [sys.executable, str(copy / "report.py")],
        capture_output=True,
        text=True,
        check=False,
        cwd=tmp_path,
    )
    assert completed.returncode == 0, completed.stderr
    assert "Traceback" not in completed.stderr


def test_the_notebook_documents_how_to_open_and_export_it() -> None:
    """The module docstring names the commands, and the file ends as a script."""
    assert notebook.__doc__ is not None
    for command in (
        "marimo edit report.py",
        # * Without -o, marimo prints the HTML to standard output.
        "marimo export html report.py --no-include-code -o report.html",
    ):
        assert command in notebook.__doc__
    assert "standard output" in notebook.__doc__
    assert notebook.__generated_with == "0.25.1"
    source = Path(notebook.__file__).read_text(encoding="utf-8")
    assert source.rstrip().endswith('if __name__ == "__main__":\n    app.run()')
