"""Command-line interface for the ``mixed-signals`` command.

The ``run`` subcommand arrives with the pipeline; until then the command only
reports its version and usage.
"""

import argparse
from collections.abc import Sequence

import mixed_signals


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
    parser.parse_args(argv)
    parser.print_help()
    return 0
