"""Command-line interface for the ``mixed-signals`` command.

Usage::

    mixed-signals --version
    mixed-signals run CONFIG [--output DIR] [--overwrite] [--dry-run]
    mixed-signals report RUN_FOLDER [--problems VALUE] [--plot-cap N]
        [--no-html] [--overwrite]

``run`` reads the YAML config, estimates the cost of every engine that it names,
runs them and writes the run folder. The estimate and the progress go to
standard output; Python warnings, such as the
:class:`~mixed_signals.errors.CostWarning` of an estimate above the budget, and
errors go to standard error. It writes no report and works without the
``report`` extra.

``--output DIR``
    Writes the run folder to ``DIR`` instead of the ``output`` of the config.
``--overwrite``
    Replaces a run folder that already holds a ``results.parquet``, with its
    samples and its report; without it, such a folder stops the run before any
    work.
``--dry-run``
    Prints the estimate of every engine and stops. Nothing is written.

``report`` copies the report notebook into a run folder as ``report.py``, writes
``report.json`` and exports ``report.html``, and prints the files that it wrote
to standard output. It needs the ``report`` extra, and
:mod:`mixed_signals.report` describes the report.

``--problems VALUE``
    The problems that the report shows in full: ``all``, a count, which takes the
    problems with the most flagged rows, or problem indices separated by commas.
    A value with a comma is a list, so ``3,`` shows problem 3 alone and ``3``
    shows three problems. The default is 10.
``--plot-cap N``
    The most samples that a posterior contributes to a corner plot, a positive
    integer; no cap by default.
``--no-html``
    Writes ``report.py`` and ``report.json`` only.
``--overwrite``
    Replaces report files that already exist; without it they stop the command.

Exit codes
    0 when the command finished, 1 when a run failed while running, with the
    traceback on standard error, and 2 for an invalid config or invalid
    arguments, with the message on standard error. A config is invalid when its
    file does not parse or validate, a file or an import path that it names does
    not resolve, an argument of an engine does not fit its inputs, the run folder
    holds results already, or an optional dependency is missing. A report is
    invalid when the folder holds no ``results.parquet``, report files exist, a
    problem is not in the result, ``--problems`` does not parse or the ``report``
    extra is missing. When the export of the report fails, because a cell of the
    notebook raised, the code is 1, with the last lines of the output of marimo
    on standard error and no traceback.

The format of the config is described in :mod:`mixed_signals.pipeline.config`
and the files of the run folder in :mod:`mixed_signals.pipeline.runner`.
Without a subcommand, the command prints its usage and succeeds.
"""

import argparse
import logging
import sys
import traceback
from collections.abc import Sequence
from pathlib import Path

from pydantic import ValidationError

import mixed_signals
from mixed_signals.errors import ConfigError, MissingDependencyError, ReportError
from mixed_signals.pipeline.runner import LOGGER_NAME, estimate_pipeline, run_pipeline

_INVALID_EXIT_CODE = 2
_FAILED_EXIT_CODE = 1


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the ``mixed-signals`` command.

    Returns:
        The configured parser.
    """
    parser = argparse.ArgumentParser(
        prog="mixed-signals",
        description=(
            "Posterior quality checks and posterior-to-posterior comparisons."
        ),
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {mixed_signals.__version__}",
    )
    subcommands = parser.add_subparsers(dest="command")
    run = subcommands.add_parser(
        "run",
        help="run the engines of a YAML config and write the run folder",
        description=(
            "Estimate the cost of the engines of a config, run them and write the"
            " run folder: results.parquet, config.json, reductions.json,"
            " cost.parquet, provenance.json and the folder samples."
        ),
    )
    run.add_argument("config", type=Path, help="the YAML config file")
    run.add_argument(
        "--output",
        type=Path,
        default=None,
        help="write the run folder here instead of the output of the config",
    )
    run.add_argument(
        "--overwrite",
        action="store_true",
        help="replace a run folder that already holds results, with its report",
    )
    run.add_argument(
        "--dry-run",
        action="store_true",
        help="print the estimate of every engine and stop",
    )
    report = subcommands.add_parser(
        "report",
        help="write the report of a run folder: report.py, report.json, report.html",
        description=(
            "Copy the report notebook into a run folder as report.py, write"
            " report.json and export report.html. Needs the report extra:"
            ' pip install "mixed-signals[report]".'
        ),
    )
    report.add_argument(
        "run_folder", type=Path, help="the run folder of mixed-signals run"
    )
    report.add_argument(
        "--problems",
        default=None,
        metavar="VALUE",
        help=(
            "problems to show in full: all, a count (those with the most flagged"
            " rows), or indices separated by commas, where 3, is problem 3 alone"
            " (default: 10)"
        ),
    )
    report.add_argument(
        "--plot-cap",
        type=int,
        default=None,
        metavar="N",
        help="most samples of a posterior in a corner plot (default: no cap)",
    )
    report.add_argument(
        "--no-html",
        action="store_true",
        help="write report.py and report.json only",
    )
    report.add_argument(
        "--overwrite",
        action="store_true",
        help="replace report files that already exist",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command line.

    Args:
        argv: Command-line arguments without the program name. ``None`` reads
            them from ``sys.argv``.

    Returns:
        The process exit code.
    """
    parser = build_parser()
    arguments = parser.parse_args(argv)
    if arguments.command is None:
        parser.print_help()
        return 0
    logger = logging.getLogger(LOGGER_NAME)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))
    previous_level = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    try:
        return _report(arguments) if arguments.command == "report" else _run(arguments)
    finally:
        # * Repeated calls in one process must not print a line twice.
        logger.removeHandler(handler)
        logger.setLevel(previous_level)


def _run(arguments: argparse.Namespace) -> int:
    """Estimate or run the config of the arguments, and return the exit code."""
    try:
        if arguments.dry_run:
            estimate_pipeline(arguments.config)
        else:
            run_pipeline(
                arguments.config,
                output=arguments.output,
                overwrite=arguments.overwrite,
            )
    except (ValidationError, ConfigError, MissingDependencyError) as error:
        print(f"mixed-signals: error: {error}", file=sys.stderr)
        return _INVALID_EXIT_CODE
    except Exception:
        traceback.print_exc()
        return _FAILED_EXIT_CODE
    return 0


def _report(arguments: argparse.Namespace) -> int:
    """Write the report of the arguments' run folder, and return the exit code."""
    try:
        problems = (
            None if arguments.problems is None else _parse_problems(arguments.problems)
        )
    except ValueError as error:
        print(f"mixed-signals: error: {error}", file=sys.stderr)
        return _INVALID_EXIT_CODE
    try:
        # * Imported here, so that run works without the report extra.
        from mixed_signals.report import DEFAULT_PROBLEMS, write_report

        files = write_report(
            arguments.run_folder,
            problems=DEFAULT_PROBLEMS if problems is None else problems,
            plot_cap=arguments.plot_cap,
            html=not arguments.no_html,
            overwrite=arguments.overwrite,
        )
    except (ValidationError, ReportError, MissingDependencyError) as error:
        print(f"mixed-signals: error: {error}", file=sys.stderr)
        return _INVALID_EXIT_CODE
    except RuntimeError as error:
        # ? The writer raises RuntimeError only when the export of marimo fails.
        print(f"mixed-signals: error: {error}", file=sys.stderr)
        return _FAILED_EXIT_CODE
    except Exception:
        traceback.print_exc()
        return _FAILED_EXIT_CODE
    for path in files:
        if path is not None:
            print(f"Wrote {path}")
    return 0


def _parse_problems(text: str) -> int | str | tuple[int, ...]:
    """Read ``--problems``: ``all``, a count, or indices with at least one comma.

    Raises:
        ValueError: If the text is none of the three.
    """
    if text == "all":
        return text
    try:
        if "," in text:
            return tuple(int(entry) for entry in text.split(",") if entry.strip())
        return int(text)
    except ValueError:
        raise ValueError(
            f"--problems must be all, a count, or indices separated by commas,"
            f" got {text!r}"
        ) from None
