"""Tests for writing a report: files, the overwrite rule, settings and the export."""

import shutil
import subprocess
import sys
from importlib import resources
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from mixed_signals.errors import ReportError
from mixed_signals.report.folder import ReportSettings
from mixed_signals.report.writer import ReportFiles, write_report

REPORT_FILES = ("report.py", "report.json", "report.html")


@pytest.fixture
def run_copy(report_run_folder: Path, tmp_path: Path) -> Path:
    """A private copy of the full run folder, which a test may write reports into."""
    copy = tmp_path / "run"
    shutil.copytree(report_run_folder, copy)
    return copy


def names_in(folder: Path) -> set[str]:
    """Return the names of the entries of a folder."""
    return {path.name for path in folder.iterdir()}


def failing_export(calls: list[list[str]], stdout: str = "", stderr: str = "") -> Any:
    """Return a stand-in for subprocess.run that records its command and fails."""

    def run(command: list[str], **keywords: Any) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        return subprocess.CompletedProcess(command, 1, stdout=stdout, stderr=stderr)

    return run


# * The files


def test_the_notebook_is_copied_byte_for_byte_beside_its_settings(
    run_copy: Path,
) -> None:
    """report.py is the package's notebook and report.json holds the settings."""
    files = write_report(run_copy, problems=[0], plot_cap=300, html=False)
    assert isinstance(files, ReportFiles)
    assert files.notebook == run_copy / "report.py"
    assert files.settings == run_copy / "report.json"
    assert files.html is None
    packaged = resources.files("mixed_signals.report").joinpath("notebook.py")
    assert files.notebook.read_bytes() == packaged.read_bytes()
    text = files.settings.read_text(encoding="utf-8")
    assert text.endswith("}\n")
    assert (
        text
        == ReportSettings(problems=(0,), plot_cap=300).model_dump_json(indent=2) + "\n"
    )
    assert ReportSettings.model_validate_json(text) == ReportSettings(
        problems=(0,), plot_cap=300
    )
    assert not (run_copy / "report.html").exists()
    assert not (run_copy / "__marimo__").exists()


def test_a_count_and_all_are_written_as_given(run_copy: Path) -> None:
    """The default is ten problems; all is the word."""
    write_report(run_copy, html=False)
    assert (
        ReportSettings.model_validate_json(
            (run_copy / "report.json").read_text()
        ).problems
        == 10
    )
    write_report(run_copy, problems="all", html=False, overwrite=True)
    settings = ReportSettings.model_validate_json(
        (run_copy / "report.json").read_text()
    )
    assert settings.problems == "all"
    assert settings.plot_cap is None


@pytest.mark.parametrize("existing", REPORT_FILES)
def test_an_existing_report_file_stops_the_call_unless_overwritten(
    run_copy: Path, existing: str
) -> None:
    """Any of the three files stops it, named; overwrite replaces all three."""
    (run_copy / existing).write_text("old", encoding="utf-8")
    before = names_in(run_copy)
    with pytest.raises(ReportError, match=existing.replace(".", r"\.")):
        write_report(run_copy, html=False)
    assert names_in(run_copy) == before
    assert (run_copy / existing).read_text(encoding="utf-8") == "old"
    write_report(run_copy, html=False, overwrite=True)
    assert (run_copy / "report.py").read_bytes() != b"old"
    assert (run_copy / "report.json").exists()
    assert not (run_copy / "report.html").exists()


def test_without_html_an_overwrite_leaves_no_stale_page(run_copy: Path) -> None:
    """All three files are removed first, so report.html cannot outlive its notebook."""
    for name in REPORT_FILES:
        (run_copy / name).write_text("old", encoding="utf-8")
    files = write_report(run_copy, html=False, overwrite=True)
    assert files.html is None
    assert not (run_copy / "report.html").exists()
    assert (run_copy / "report.py").read_bytes() != b"old"


def test_a_folder_without_results_cannot_be_reported(tmp_path: Path) -> None:
    """The error names the folder, and nothing is written."""
    with pytest.raises(ReportError, match=str(tmp_path)):
        write_report(tmp_path)
    assert names_in(tmp_path) == set()


@pytest.mark.parametrize(
    ("changes", "error"),
    [
        ({"problems": [0, 5]}, ReportError),
        ({"problems": 0}, ValidationError),
        ({"problems": []}, ValidationError),
        ({"problems": [0, 0]}, ValidationError),
        ({"problems": "some"}, ValidationError),
        ({"plot_cap": 0}, ValidationError),
    ],
)
def test_invalid_settings_stop_the_call_before_anything_changes(
    run_copy: Path, changes: dict[str, Any], error: type[Exception]
) -> None:
    """Indices outside the result and invalid values raise, and the old files stay."""
    (run_copy / "report.json").write_text('{"problems": 3}', encoding="utf-8")
    before = names_in(run_copy)
    with pytest.raises(error):
        write_report(run_copy, html=False, overwrite=True, **changes)
    assert names_in(run_copy) == before
    assert (run_copy / "report.json").read_text(encoding="utf-8") == '{"problems": 3}'


# * The export


def test_a_failing_export_raises_with_the_last_lines_of_marimo(
    run_copy: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A nonzero exit means a cell failed; the message holds the tail of the output."""
    calls: list[list[str]] = []
    output = "\n".join(f"line {number}" for number in range(40))
    monkeypatch.setattr(
        "mixed_signals.report.writer.subprocess.run",
        failing_export(calls, stdout=output, stderr="MarimoExceptionRaisedError: boom"),
    )
    with pytest.raises(RuntimeError, match="exit code 1") as raised:
        write_report(run_copy)
    message = str(raised.value)
    assert "MarimoExceptionRaisedError: boom" in message
    assert "line 39" in message
    assert "line 0\n" not in message
    assert str(run_copy / "report.py") in message
    assert calls == [
        [
            sys.executable,
            "-m",
            "marimo",
            "export",
            "html",
            str(run_copy / "report.py"),
            "--output",
            str(run_copy / "report.html"),
            "--no-include-code",
            "--force",
        ]
    ]
    assert (run_copy / "report.py").exists()
    assert (run_copy / "report.json").exists()


def test_a_real_export_holds_the_outputs_and_no_code(run_copy: Path) -> None:
    """Marimo exits with 0; the page has headings, a pair, an image and no code."""
    files = write_report(run_copy)
    assert files.html == run_copy / "report.html"
    html = files.html.read_text(encoding="utf-8")
    for heading in (
        "mixed signals report",
        "Diagnostics",
        "Comparisons",
        "Configuration",
    ):
        assert heading in html
    assert "mcmc vs table" in html
    assert "data:image/png" in html
    # * The figures and tables are embedded, but the front end comes from a CDN.
    assert 'src="https://cdn.jsdelivr.net/npm/@marimo-team/frontend@' in html
    assert "report.overview_section" not in html
    assert "MarimoExceptionRaisedError" not in html
    assert (run_copy / "__marimo__").is_dir()
    with pytest.raises(ReportError, match=r"report\.html"):
        write_report(run_copy)
