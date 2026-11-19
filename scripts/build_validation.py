"""Rebuild the validation page from the zoo checks of the sessions.

The script reruns every check with fewer replicates than the original zoo
checks, with a standard error on every number, and writes ``docs/validation.md``,
its figures under ``docs/validation/figures/`` and the raw numbers in
``docs/validation/results.json``. All three are stamped with the package
version, the JAX version and the seed. The full build takes about an hour on the
12-core Apple M4 Pro of the developers; ``--quick`` takes about three minutes and is
what the test suite runs.

Usage:
    uv run python scripts/build_validation.py [--quick] [--seed SEED]
        [--only LABEL ...] [--output-dir DIRECTORY]

A partial run (``--only``) writes nothing unless ``--output-dir`` names a
directory, so that it never overwrites the committed page. The exit status is 1
if a claim that the run evaluates is not consistent, after everything is written.
"""

import argparse
import importlib
import time
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Final, cast

from zoo_validation.report import (
    Check,
    CheckResult,
    Context,
    Stamp,
    failed_claims,
    run_check,
    write_outputs,
)

REPOSITORY: Final = Path(__file__).resolve().parent.parent
DEFAULT_SEED: Final = 20_261_117
"""Seed of the committed page, stated in its stamp."""

TOPICS: Final = (
    ("nulls", "Nulls"),
    ("significance", "Significance"),
    ("effects", "Effect sizes"),
    ("transport", "Optimal transport"),
    ("divergences", "Divergences"),
    ("estimators", "Estimators"),
    ("diagnostics", "Diagnostics"),
    ("reweighting", "Reweighting"),
    ("costs", "Costs"),
)
"""The topic modules in page order, with the heading of each.

A module that does not exist yet is skipped.
"""

type Topics = Sequence[tuple[str, Sequence[Check]]]


def load_topics() -> list[tuple[str, tuple[Check, ...]]]:
    """Import the topic modules that exist, in page order, with their checks."""
    loaded = []
    for module_name, heading in TOPICS:
        qualified = f"zoo_validation.{module_name}"
        try:
            module = importlib.import_module(qualified)
        except ModuleNotFoundError as error:
            if error.name != qualified:
                raise
            continue
        loaded.append((heading, cast(tuple[Check, ...], module.CHECKS)))
    return loaded


def parse_arguments(argv: Sequence[str] | None) -> argparse.Namespace:
    """Read the command line."""
    parser = argparse.ArgumentParser(
        description="Rebuild the validation page from the zoo checks."
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="run the quick counts (about three minutes)",
    )
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED, help="seed of the root key"
    )
    parser.add_argument(
        "--only", nargs="+", metavar="LABEL", help="run only these checks"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="directory of validation.md; the docs directory by default",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None, *, topics: Topics | None = None) -> int:
    """Run the checks, write the page and return the exit status.

    Args:
        argv: Command-line arguments; ``sys.argv`` when ``None``.
        topics: Headings and checks in page order; the topic modules when
            ``None``. The tests pass their own.

    Returns:
        0 if every evaluated claim is consistent, 1 otherwise.
    """
    arguments = parse_arguments(argv)
    available = load_topics() if topics is None else topics
    labels = [check.label for _, checks in available for check in checks]
    if len(set(labels)) != len(labels):
        raise ValueError(f"check labels must be unique, got {sorted(labels)}")
    selected = set(arguments.only) if arguments.only else set(labels)
    unknown = selected - set(labels)
    if unknown:
        raise SystemExit(
            f"unknown check labels {sorted(unknown)}; the checks are {labels}"
        )
    context = Context.create(arguments.seed, quick=arguments.quick)
    results: list[CheckResult] = []
    started = time.perf_counter()
    with warnings.catch_warnings():
        # * No warning stops the build; each check records its own.
        warnings.simplefilter("always")
        for heading, checks in available:
            for check in checks:
                if check.label in selected:
                    result = run_check(check, heading, context)
                    results.append(result)
                    print(_summary_line(result, quick=arguments.quick), flush=True)
    total_seconds = time.perf_counter() - started
    print(f"{'total':<28} {total_seconds:7.1f} s")
    partial = arguments.only is not None
    if partial and arguments.output_dir is None:
        print("partial run: nothing written; give --output-dir to write the page")
    else:
        output_directory = arguments.output_dir or REPOSITORY / "docs"
        write_outputs(
            output_directory,
            Stamp.create(context),
            results,
            quick=arguments.quick,
            total_seconds=total_seconds,
        )
        print(f"written to {output_directory}")
    failed = failed_claims(results, quick=arguments.quick)
    for claim in failed:
        print(f"NOT CONSISTENT {claim}")
    return 1 if failed else 0


def _summary_line(result: CheckResult, *, quick: bool) -> str:
    claims = result.section.claims
    evaluated = [claim for claim in claims if not claim.failed(quick=quick)]
    return (
        f"{result.check.label:<28} {result.seconds:7.1f} s"
        f"  {len(evaluated)}/{len(claims)} claims ok"
    )


if __name__ == "__main__":
    raise SystemExit(main())
