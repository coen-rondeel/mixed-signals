"""Command-line interface for the ``mixed-signals`` command.

Usage::

    mixed-signals --version
    mixed-signals run CONFIG [--output DIR] [--overwrite] [--dry-run]

``run`` reads the YAML config, estimates the cost of every engine that it names,
runs them and writes the run folder. The estimate and the progress go to
standard output; Python warnings, such as the
:class:`~mixed_signals.errors.CostWarning` of an estimate above the budget, and
errors go to standard error.

``--output DIR``
    Writes the run folder to ``DIR`` instead of the ``output`` of the config.
``--overwrite``
    Replaces a run folder that already holds a ``results.parquet``; without it,
    such a folder stops the run before any work.
``--dry-run``
    Prints the estimate of every engine and stops. Nothing is written.

Exit codes
    0 when the command finished, 1 when a run failed while running, with the
    traceback on standard error, and 2 for an invalid config or invalid
    arguments, with the message on standard error. A config is invalid when its
    file does not parse or validate, a file or an import path that it names does
    not resolve, an argument of an engine does not fit its inputs, the run folder
    holds results already, or an optional dependency is missing.

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
from mixed_signals.errors import ConfigError, MissingDependencyError
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
            " cost.parquet and provenance.json."
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
        help="replace a run folder that already holds results",
    )
    run.add_argument(
        "--dry-run",
        action="store_true",
        help="print the estimate of every engine and stop",
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
        return _run(arguments)
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
