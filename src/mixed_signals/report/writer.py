"""Write the report of a run folder: the notebook, its settings and its HTML.

:func:`write_report` copies the notebook of the package into the run folder as
``report.py``, writes the settings as ``report.json`` and exports the notebook to
``report.html`` with marimo. The HTML holds the outputs only, with no code
(``marimo export html --no-include-code``), so it can go to a collaborator that
does not use marimo. The figures and tables are embedded in the page, but marimo
loads the front end that shows them from ``cdn.jsdelivr.net``, so the page renders
only with an internet connection. The notebook is copied byte for byte, so every
report of a run folder is the same program, and it reads the run folder next to
it.

Overwrite
    The three files belong together: ``report.py`` and ``report.json`` fit the
    ``report.html`` exported from them. A call stops when any of them exists,
    unless ``overwrite`` is true, which removes all three first, so that a call
    without HTML never leaves the HTML of an earlier report.
Export
    marimo runs the notebook and writes the HTML even when a cell raises, but
    exits with a nonzero code. The call then raises with the last lines of the
    output of marimo, and the HTML shows the error where the cell failed. marimo
    also leaves a cache folder ``__marimo__`` next to the notebook, which stays.
"""

import os
import subprocess
import sys
from collections.abc import Sequence
from importlib import resources
from pathlib import Path
from typing import Final, NamedTuple

import polars as pl

from mixed_signals.errors import ReportError
from mixed_signals.report.folder import (
    DEFAULT_PROBLEMS,
    RESULT_FILE,
    SETTINGS_FILE,
    ReportSettings,
    check_problems,
)

__all__ = ["ReportFiles", "write_report"]

NOTEBOOK_FILE: Final = "report.py"
HTML_FILE: Final = "report.html"
_TAIL_LINES: Final = 12


class ReportFiles(NamedTuple):
    """The files that :func:`write_report` wrote."""

    notebook: Path
    """The marimo notebook, ``report.py``."""
    settings: Path
    """The settings, ``report.json``."""
    html: Path | None
    """The exported page, ``report.html``; ``None`` when it was not written."""


def write_report(
    folder: str | os.PathLike[str],
    *,
    problems: int | str | Sequence[int] = DEFAULT_PROBLEMS,
    plot_cap: int | None = None,
    html: bool = True,
    overwrite: bool = False,
) -> ReportFiles:
    """Write ``report.py``, ``report.json`` and ``report.html`` into a run folder.

    Args:
        folder: The run folder, which holds a ``results.parquet``.
        problems: How many problems to show in full, those with the most flagged
            rows first; ``"all"``; or a list of problem indices.
        plot_cap: The most samples that a posterior contributes to a corner plot;
            ``None`` for no cap.
        html: Whether to export ``report.html``; the notebook and its settings
            are written either way.
        overwrite: Whether to replace report files that already exist.

    Returns:
        The paths that were written.

    Raises:
        ReportError: If the folder holds no ``results.parquet``, a report file
            exists and ``overwrite`` is false, or a listed problem is not in the
            result.
        pydantic.ValidationError: If ``problems`` or ``plot_cap`` is invalid.
        RuntimeError: If the export fails, which means that a cell of the
            notebook raised. The message holds the last lines of the output of
            marimo.
    """
    run_folder = Path(folder)
    if not (run_folder / RESULT_FILE).is_file():
        raise ReportError(
            f"the folder {run_folder} holds no {RESULT_FILE}; give a run folder"
            " that mixed-signals run wrote"
        )
    notebook = run_folder / NOTEBOOK_FILE
    settings_file = run_folder / SETTINGS_FILE
    html_file = run_folder / HTML_FILE
    existing = [
        path.name for path in (notebook, settings_file, html_file) if path.exists()
    ]
    if existing and not overwrite:
        raise ReportError(
            f"the folder {run_folder} already holds {', '.join(existing)}; remove"
            " them or overwrite them"
        )
    settings = ReportSettings(problems=problems, plot_cap=plot_cap)
    check_problems(
        settings, pl.read_parquet(run_folder / RESULT_FILE, columns=["problem"])
    )
    for path in (notebook, settings_file, html_file):
        path.unlink(missing_ok=True)
    settings_file.write_text(
        settings.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    source = resources.files("mixed_signals.report").joinpath("notebook.py")
    notebook.write_bytes(source.read_bytes())
    if html:
        _export(notebook, html_file)
    return ReportFiles(notebook, settings_file, html_file if html else None)


def _export(notebook: Path, html_file: Path) -> None:
    """Export the notebook to HTML without its code, or raise with marimo's output."""
    command = [
        sys.executable,
        "-m",
        "marimo",
        "export",
        "html",
        str(notebook),
        "--output",
        str(html_file),
        "--no-include-code",
        "--force",
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        tail = (completed.stdout + completed.stderr).strip().splitlines()[-_TAIL_LINES:]
        raise RuntimeError(
            f"marimo could not export {notebook} (exit code {completed.returncode});"
            " the last lines of its output are:\n" + "\n".join(tail)
        )
