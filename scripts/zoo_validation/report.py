"""Data types and rendering of the validation page.

A :class:`Check` reruns one zoo check of a session and returns a
:class:`Section`: its setup text, its tables and figures, and its claims. A
claim is a number with an expectation and a tolerance that the code states, and
the page prints its verdict, never a hand-written conclusion. :func:`run_check`
times a check and records the package warnings that it raised, and
:func:`write_outputs` writes the page, the figures and the raw numbers as JSON.
"""

import dataclasses
import datetime
import functools
import importlib.metadata
import io
import json
import math
import platform
import re
import subprocess
import sys
import time
import warnings
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Final, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import betainc, ndtr
from jax.typing import ArrayLike
from jaxtyping import PRNGKeyArray
from matplotlib.figure import Figure as MatplotlibFigure

from mixed_signals import errors, prng
from zoo_validation import figures

LABEL_PATTERN: Final = re.compile(r"[a-z0-9]+(-[a-z0-9]+)*")
"""A check label is kebab case: lower-case words and digits joined by hyphens."""

ROLE_PATTERN: Final = re.compile(r"\{py:(?P<kind>[a-z]+)\}`(?P<target>[^`]+)`")
"""A cross-reference role of the supports list, such as ``{py:func}`name```."""

BINOMIAL_ERRORS: Final = 3.0
"""Binomial errors that a rejection rate may lie from its level."""

STANDARD_ERRORS: Final = 3.0
"""Standard errors that a mean may lie from its expectation.

With few replicates the factor is the Student t quantile that keeps the tail
probability of three standard errors of a normal distribution; see
:func:`critical_value`.
"""

_BISECTION_STEPS: Final = 100
_SEARCH_LIMIT: Final = 1e8


@functools.cache
def critical_value(replicates: int | None) -> float:
    """Return the multiple of the standard error that a mean may lie from its truth.

    It is three for a normal distribution. For a mean of ``replicates`` values
    whose standard error comes from their scatter, it is the quantile of the
    Student t distribution with ``replicates - 1`` degrees of freedom that has
    the same upper tail probability as three standard errors of a normal
    distribution, so that a claim does not fail by chance when few replicates
    underestimate the scatter.

    Args:
        replicates: Number of replicates behind the mean, or ``None`` for a
            value with a known error.

    Returns:
        The factor, at least three.
    """
    if replicates is None or replicates < 2:
        return STANDARD_ERRORS
    degrees = float(replicates - 1)
    target = ndtr(-STANDARD_ERRORS)

    def survival(value: jax.Array) -> jax.Array:
        return 0.5 * betainc(degrees / 2, 0.5, degrees / (degrees + value**2))

    def step(
        _: int, bracket: tuple[jax.Array, jax.Array]
    ) -> tuple[jax.Array, jax.Array]:
        lower, upper = bracket
        middle = 0.5 * (lower + upper)
        too_small = survival(middle) > target
        return jnp.where(too_small, middle, lower), jnp.where(too_small, upper, middle)

    lower, upper = jax.lax.fori_loop(
        0, _BISECTION_STEPS, step, (jnp.asarray(0.0), jnp.asarray(_SEARCH_LIMIT))
    )
    return float(0.5 * (lower + upper))


_SCIENTIFIC_ABOVE: Final = 1e7
_MAXIMUM_DECIMALS: Final = 8
_ERROR_DIGITS: Final = 2
_QUICK_NOTE: Final = "not evaluated (full run only)"
_NEGLIGIBLE_ERROR: Final = 1e-12
"""Relative error below which a value counts as exact, such as a constant."""
_GROUPED_FROM: Final = 1_000

INTRODUCTION: Final = (
    "This page reruns the zoo checks behind the defaults of mixed signals: the"
    " calibration of the nulls, the recovery of known effect sizes, the optimal"
    " transport settings, the limits of the chordal metrics on circles and"
    " spheres, the ranks of the nearest-neighbour divergences, the estimators,"
    " the calibration and the flags of the diagnostics, the reweighting, the"
    " engines on weighted sets and on the truths of the zoo, and the costs of"
    " the metrics and of the dry run. Every number comes with a standard error,"
    " and every check lists its claims with an expectation and a tolerance that"
    " the script states. The verdict of a claim, consistent or not consistent,"
    " is computed from the numbers of the run; no verdict is written by hand. A"
    " tolerance of three standard errors uses the Student t quantile with the"
    " same tail probability when a mean rests on few replicates. The script"
    " `scripts/build_validation.py` writes this page, its figures and the raw"
    " numbers (`validation/results.json`). Rebuild them with"
    " `uv run python scripts/build_validation.py`; the option `--quick` runs"
    " fewer replicates in about four minutes, which the test suite uses."
)

type Cell = str | int | float | bool | Measurement | None
"""One cell of a table: text, a number, a yes or no, nothing or a measurement."""


def grouped(number: float) -> str:
    """Write an integer with a space between groups of three digits, as in 10 000."""
    return f"{int(number):,}".replace(",", " ")


def _plain(value: float) -> str:
    """Write a value without error: integers in groups, others to four digits."""
    nearest = round(value)
    if abs(value - nearest) <= _NEGLIGIBLE_ERROR * max(1.0, abs(value)) * 1e3:
        return grouped(nearest) if abs(nearest) >= _GROUPED_FROM else str(nearest)
    return f"{value:.4g}"


def role(kind: str, target: str, *, short: bool = True) -> str:
    """Return a cross-reference to a package name, shown by its last part if short."""
    return f"{{py:{kind}}}`{'~' if short else ''}{target}`"


@dataclasses.dataclass(frozen=True)
class Context:
    """The settings that every check of one build shares."""

    seed: int
    """Seed of the root key."""
    quick: bool
    """Whether the build runs the quick counts."""
    root: PRNGKeyArray
    """Root key; a check derives its keys from it with :meth:`key`."""

    @classmethod
    def create(cls, seed: int, *, quick: bool) -> "Context":
        """Return the context of a seed and a mode."""
        return cls(seed=seed, quick=quick, root=prng.root_key(seed))

    def count(self, full: int, quick: int) -> int:
        """Return the count of the full run, or of the quick run in quick mode."""
        return quick if self.quick else full

    def key(self, label: str, *path: str | int) -> PRNGKeyArray:
        """Derive a key from the check label and a path, independent of other checks."""
        return prng.derive_key(self.root, label, *path)


@dataclasses.dataclass(frozen=True)
class Measurement:
    """A value with its standard error, or the binomial error of a rate."""

    value: float
    error: float | None = None
    """Standard error; ``None`` for a value without error, such as a closed form."""
    replicates: int | None = None
    """Number of replicates behind a mean, which sets its :meth:`margin`."""

    @classmethod
    def from_replicates(cls, values: Sequence[float] | ArrayLike) -> "Measurement":
        """Return the mean of replicates with the standard error of the mean."""
        array = jnp.asarray(values, dtype=jnp.float64).reshape(-1)
        count = int(array.size)
        error = float(jnp.std(array, ddof=1)) / math.sqrt(count) if count >= 2 else None
        return cls(float(jnp.mean(array)), error, count)

    @classmethod
    def from_rate(
        cls, successes: int, trials: int, *, level: float | None = None
    ) -> "Measurement":
        """Return a rate with its binomial error.

        The error is that of the stated ``level`` when it is given, which is the
        error that a claim about a rejection rate uses, and otherwise that of
        the rate shifted towards one half by one success and one failure, so
        that a rate of zero still has an error.
        """
        rate = successes / trials
        reference = level if level is not None else (successes + 1) / (trials + 2)
        return cls(rate, math.sqrt(reference * (1 - reference) / trials))

    def render(self) -> str:
        """Write the value and its error, the error to two significant digits."""
        if not math.isfinite(self.value):
            return "NaN"
        if (
            self.error is None
            or not math.isfinite(self.error)
            or self.error <= _NEGLIGIBLE_ERROR * abs(self.value)
        ):
            return _plain(self.value)
        exponent = math.floor(math.log10(self.error))
        decimals = max(0, _ERROR_DIGITS - 1 - exponent)
        if decimals > _MAXIMUM_DECIMALS or abs(self.value) >= _SCIENTIFIC_ABOVE:
            value_exponent = (
                math.floor(math.log10(abs(self.value))) if self.value != 0 else exponent
            )
            digits = max(0, value_exponent - exponent + _ERROR_DIGITS - 1)
            return f"{self.value:.{digits}e} ± {self.error:.{_ERROR_DIGITS - 1}e}"
        value_text = f"{self.value:,.{decimals}f}".replace(",", " ")
        error_string = f"{self.error:,.{decimals}f}".replace(",", " ")
        return f"{value_text} ± {error_string}"

    def margin(self) -> float:
        """Return the distance from the truth that a claim allows.

        It is :func:`critical_value` times the error, zero without an error.
        """
        if (
            self.error is None
            or not math.isfinite(self.error)
            or self.error <= _NEGLIGIBLE_ERROR * abs(self.value)
        ):
            return 0.0
        return critical_value(self.replicates) * self.error

    def to_json(self) -> dict[str, float | None]:
        """Return the value and the error at full precision; NaN and inf become null."""
        entry: dict[str, float | None] = {
            "value": _finite_or_none(self.value),
            "error": _finite_or_none(self.error),
        }
        if self.replicates is not None:
            entry["replicates"] = float(self.replicates)
        return entry

    @classmethod
    def from_json(cls, entry: Mapping[str, float | None]) -> "Measurement":
        """Rebuild a measurement from :meth:`to_json`; a null value reads as NaN."""
        value = entry["value"]
        replicates = entry.get("replicates")
        return cls(
            math.nan if value is None else float(value),
            entry["error"],
            None if replicates is None else int(replicates),
        )


def _finite_or_none(number: float | None) -> float | None:
    return number if number is not None and math.isfinite(number) else None


def _render_cell(cell: Cell) -> str:
    if isinstance(cell, Measurement):
        return cell.render()
    if cell is None:
        return "-"
    if isinstance(cell, bool):
        return "yes" if cell else "no"
    if isinstance(cell, int):
        return grouped(cell) if abs(cell) >= _GROUPED_FROM else str(cell)
    if isinstance(cell, float):
        return "NaN" if not math.isfinite(cell) else f"{cell:.4g}"
    return cell.replace("|", r"\|")


def _cell_to_json(cell: Cell) -> object:
    if isinstance(cell, Measurement):
        return cell.to_json()
    if isinstance(cell, float):
        return _finite_or_none(cell)
    return cell


def _cell_from_json(entry: object) -> Cell:
    if isinstance(entry, dict):
        return Measurement.from_json(entry)
    if entry is None or isinstance(entry, str | int | float | bool):
        return entry
    raise TypeError(f"a table cell cannot be {type(entry).__name__}")


@dataclasses.dataclass(frozen=True)
class Table:
    """A caption, column headers and rows of text, numbers or measurements."""

    caption: str
    columns: tuple[str, ...]
    rows: tuple[tuple[Cell, ...], ...]

    def __post_init__(self) -> None:
        """Check that every row has one cell per column."""
        for row in self.rows:
            if len(row) != len(self.columns):
                raise ValueError(
                    f"a row of {len(row)} cells does not fit"
                    f" {len(self.columns)} columns"
                )

    @classmethod
    def of(
        cls, caption: str, columns: Sequence[str], rows: Sequence[Sequence[Cell]]
    ) -> "Table":
        """Return a table from sequences."""
        return cls(caption, tuple(columns), tuple(tuple(row) for row in rows))

    def render(self) -> str:
        """Write the table as a MyST pipe table inside a table directive."""
        lines = [f":::{{table}} {self.caption}", ""]
        lines.append("| " + " | ".join(self.columns) + " |")
        lines.append("| " + " | ".join("---" for _ in self.columns) + " |")
        lines.extend(
            "| " + " | ".join(_render_cell(cell) for cell in row) + " |"
            for row in self.rows
        )
        lines.append(":::")
        return "\n".join(lines)

    def to_json(self) -> dict[str, object]:
        """Return the table with every number at full precision."""
        return {
            "type": "table",
            "caption": self.caption,
            "columns": list(self.columns),
            "rows": [[_cell_to_json(cell) for cell in row] for row in self.rows],
        }

    @classmethod
    def from_json(cls, entry: Mapping[str, object]) -> "Table":
        """Rebuild a table from :meth:`to_json`."""
        rows = entry["rows"]
        columns = entry["columns"]
        if not isinstance(rows, list) or not isinstance(columns, list):
            raise TypeError("a table needs a list of columns and a list of rows")
        return cls.of(
            str(entry["caption"]),
            [str(column) for column in columns],
            [[_cell_from_json(cell) for cell in row] for row in rows],
        )


def jsonable(value: object) -> object:
    """Return a value as plain JSON types, with arrays as lists and NaN as null."""
    if isinstance(value, Measurement):
        return value.to_json()
    if isinstance(value, Mapping):
        return {str(name): jsonable(item) for name, item in value.items()}
    if isinstance(value, str | bool) or value is None:
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return _finite_or_none(value)
    if isinstance(value, Sequence):
        return [jsonable(item) for item in value]
    return jsonable(np.asarray(value).tolist())


@dataclasses.dataclass(frozen=True)
class Figure:
    """A figure drawn in both themes from the same data."""

    label: str
    """File stem: ``<label>-light.svg`` and ``<label>-dark.svg``."""
    caption: str
    alt: str
    """Alternative text of the image, for readers who do not see it."""
    draw: Callable[[figures.FigureStyle], MatplotlibFigure]
    """Draws the figure in a theme."""
    data: Mapping[str, object]
    """Every number of the figure, stored in the JSON."""

    def save(self, directory: Path) -> None:
        """Write the light and the dark file into ``directory``."""
        directory.mkdir(parents=True, exist_ok=True)
        with figures.style_context():
            for style in figures.STYLES:
                drawing = self.draw(style)
                buffer = io.BytesIO()
                drawing.savefig(
                    buffer,
                    format="svg",
                    facecolor=style.background,
                    metadata={"Date": None},
                )
                # * Matplotlib leaves spaces at line ends, which the commit hooks strip.
                lines = buffer.getvalue().decode("utf-8").splitlines()
                (directory / f"{self.label}-{style.name}.svg").write_text(
                    "\n".join(line.rstrip() for line in lines) + "\n",
                    encoding="utf-8",
                )

    def render(self) -> str:
        """Write both images with furo's classes, and the caption."""
        images = [
            f"```{{image}} validation/figures/{self.label}-{style.name}.svg\n"
            f":class: only-{style.name}\n"
            f":alt: {self.alt}\n"
            "```"
            for style in figures.STYLES
        ]
        return "\n\n".join([*images, f"**Figure.** {self.caption}"])

    def to_json(self) -> dict[str, object]:
        """Return the figure's label, texts and data."""
        return {
            "type": "figure",
            "label": self.label,
            "caption": self.caption,
            "alt": self.alt,
            "data": jsonable(self.data),
        }


@dataclasses.dataclass(frozen=True)
class Claim:
    """One claim of a check: an expectation, an observation and a verdict."""

    statement: str
    expected: str
    observed: str
    consistent: bool
    full_only: bool = False
    """Whether the quick replicates are too few for the tolerance."""
    error_part: float | None = None
    """Allowance from the standard error, of the case with the widest error."""
    tolerance_part: float | None = None
    """Allowance from the stated relative tolerance, of the same case."""

    def verdict(self, *, quick: bool) -> str:
        """Return the verdict, or say that the quick run does not evaluate the claim."""
        if self.full_only and quick:
            return _QUICK_NOTE
        return "consistent" if self.consistent else "not consistent"

    def failed(self, *, quick: bool) -> bool:
        """Return whether the claim is evaluated and not consistent."""
        return not self.consistent and not (self.full_only and quick)

    def to_json(self, *, quick: bool) -> dict[str, object]:
        """Return the claim with its verdict."""
        return {
            "statement": self.statement,
            "expected": self.expected,
            "observed": self.observed,
            "consistent": self.consistent,
            "full_only": self.full_only,
            "verdict": self.verdict(quick=quick),
            "error_part": _finite_or_none(self.error_part),
            "tolerance_part": _finite_or_none(self.tolerance_part),
        }


def error_text(replicates: int | None) -> str:
    """Name the multiple of the standard error that a claim allows, with its source."""
    factor = critical_value(replicates)
    if replicates is None or round(factor, 1) == STANDARD_ERRORS:
        return "three standard errors"
    return f"{factor:.1f} standard errors (Student t, {replicates} replicates)"


class Case(NamedTuple):
    """One case of a claim: a deviation from the truth and what the claim allows."""

    label: str
    deviation: float
    error_part: float
    """Allowance from the standard error, :meth:`Measurement.margin`."""
    tolerance_part: float
    """Allowance from the stated relative tolerance, in the unit of the deviation."""


def claim_cases(
    statement: str,
    cases: Sequence[Case],
    *,
    tolerance_text: str | None,
    replicates: int | None,
    full_only: bool = True,
) -> Claim:
    """State that every case lies within its tolerance plus its standard errors.

    Args:
        statement: The claim.
        cases: The cases, each with its deviation and the two parts of its
            allowance.
        tolerance_text: The stated tolerance, such as ``"3% of the truth"``;
            ``None`` for a claim that allows the standard errors only.
        replicates: Replicates behind the means, which set the factor of the
            standard error that the claim names.
        full_only: Whether the quick replicates are too few for the tolerance.

    Returns:
        The claim. Its observed text names the worst case, the one with the
        least room to its allowance, with its deviation and its allowance.
    """
    consistent = all(
        math.isfinite(case.deviation)
        and abs(case.deviation) <= case.error_part + case.tolerance_part
        for case in cases
    )
    worst = max(
        cases,
        key=lambda case: (
            math.inf
            if not math.isfinite(case.deviation)
            else abs(case.deviation) - case.error_part - case.tolerance_part
        ),
    )
    allowance = worst.error_part + worst.tolerance_part
    spread = max(cases, key=lambda case: case.error_part - case.tolerance_part)
    expected = (
        f"within {error_text(replicates)}"
        if tolerance_text is None
        else f"within {tolerance_text} plus {error_text(replicates)}"
    )
    return Claim(
        statement,
        expected,
        f"worst of {len(cases)}: {worst.label}: {worst.deviation:+.3g} against"
        f" ±{allowance:.2g}",
        consistent,
        full_only,
        spread.error_part,
        spread.tolerance_part,
    )


def claim_cases_above(
    statement: str,
    cases: Sequence[Case],
    *,
    tolerance_text: str | None,
    replicates: int | None,
    reference_text: str = "the truth",
    full_only: bool = True,
) -> Claim:
    """State that every case lies above its reference by more than its allowance.

    The claim is the opposite of :func:`claim_cases` for one side: a deviation
    must exceed the tolerance plus the standard errors, so it can only hold when
    the effect is larger than the scatter.

    Args:
        statement: The claim.
        cases: The cases, each with its deviation and the two parts of its
            allowance.
        tolerance_text: The stated tolerance, such as ``"10% of the truth"``;
            ``None`` for a claim that allows the standard errors only.
        replicates: Replicates behind the means, which set the factor of the
            standard error that the claim names.
        reference_text: What the deviation is measured from, such as
            ``"the truth"``.
        full_only: Whether the quick replicates are too few for the tolerance.

    Returns:
        The claim. Its observed text names the worst case, the one with the
        least room above its allowance.
    """
    consistent = all(
        math.isfinite(case.deviation)
        and case.deviation > case.error_part + case.tolerance_part
        for case in cases
    )
    worst = min(
        cases,
        key=lambda case: (
            -math.inf
            if not math.isfinite(case.deviation)
            else case.deviation - case.error_part - case.tolerance_part
        ),
    )
    allowance = worst.error_part + worst.tolerance_part
    spread = max(cases, key=lambda case: case.error_part - case.tolerance_part)
    margin_text = (
        error_text(replicates)
        if tolerance_text is None
        else f"{tolerance_text} plus {error_text(replicates)}"
    )
    return Claim(
        statement,
        f"above {reference_text} by more than {margin_text}",
        f"worst of {len(cases)}: {worst.label}: {worst.deviation:+.3g} against"
        f" more than {allowance:.2g}",
        consistent,
        full_only,
        spread.error_part,
        spread.tolerance_part,
    )


def claim_within(
    statement: str,
    observed: Measurement,
    expected: float,
    tolerance: float,
    *,
    tolerance_text: str,
    full_only: bool = False,
) -> Claim:
    """State that a measurement lies within a tolerance of its expected value."""
    consistent = (
        math.isfinite(observed.value) and abs(observed.value - expected) <= tolerance
    )
    return Claim(
        statement,
        f"{expected:.4g} within {tolerance:.2g} ({tolerance_text})",
        observed.render(),
        consistent,
        full_only,
    )


def claim_at_most(
    statement: str,
    observed: Measurement,
    limit: float,
    *,
    limit_text: str,
    full_only: bool = False,
) -> Claim:
    """State that a measurement does not exceed a limit."""
    consistent = math.isfinite(observed.value) and observed.value <= limit
    return Claim(
        statement,
        f"at most {limit:.4g} ({limit_text})",
        observed.render(),
        consistent,
        full_only,
    )


def claim_at_least(
    statement: str,
    observed: Measurement,
    limit: float,
    *,
    limit_text: str,
    full_only: bool = False,
) -> Claim:
    """State that a measurement reaches a lower limit."""
    consistent = math.isfinite(observed.value) and observed.value >= limit
    return Claim(
        statement,
        f"at least {limit:.4g} ({limit_text})",
        observed.render(),
        consistent,
        full_only,
    )


def claim_true(
    statement: str,
    expected: str,
    observed: str,
    consistent: bool,
    full_only: bool = False,
) -> Claim:
    """State a claim whose test the check evaluated itself."""
    return Claim(statement, expected, observed, consistent, full_only)


def claim_rates_within(
    statement: str,
    counts: Sequence[tuple[int, int]],
    levels: Sequence[float],
    *,
    full_only: bool = False,
) -> Claim:
    """State that each rejection rate lies within three binomial errors of its level.

    Args:
        statement: The claim.
        counts: The rejections and the trials of each level.
        levels: The levels, such as 0.01, 0.05 and 0.1.
        full_only: Whether the quick replicates are too few for the tolerance.

    Returns:
        The claim, true if every rate is within its tolerance.
    """
    observed, consistent = [], True
    for (successes, trials), level in zip(counts, levels, strict=True):
        measurement = Measurement.from_rate(successes, trials, level=level)
        tolerance = BINOMIAL_ERRORS * (measurement.error or 0.0)
        consistent &= abs(measurement.value - level) <= tolerance
        observed.append(measurement.render())
    expected = ", ".join(f"{level:.0%}" for level in levels)
    return Claim(
        statement,
        f"rates within three binomial errors of {expected}",
        "; ".join(observed),
        consistent,
        full_only,
    )


def claim_rates_at_most(
    statement: str,
    counts: Sequence[tuple[int, int]],
    levels: Sequence[float],
    *,
    allowance: float = 1.0,
    full_only: bool = False,
) -> Claim:
    """State that no rejection rate exceeds its level by three binomial errors.

    With an ``allowance`` above one, the bound is that multiple of the level, and
    the binomial error is the one at the bound, the boundary of the claim.
    """
    observed, consistent = [], True
    for (successes, trials), level in zip(counts, levels, strict=True):
        bound = allowance * level
        measurement = Measurement.from_rate(successes, trials, level=bound)
        tolerance = BINOMIAL_ERRORS * (measurement.error or 0.0)
        consistent &= measurement.value <= bound + tolerance
        observed.append(measurement.render())
    expected = ", ".join(f"{level:.0%}" for level in levels)
    if allowance == 1.0:
        expected_text = f"rates at most three binomial errors above {expected}"
    else:
        bounds = ", ".join(f"{allowance * level:.1%}" for level in levels)
        expected_text = (
            f"rates at most {allowance:g} times the levels {expected}, that is"
            f" {bounds}, plus three binomial errors at those bounds"
        )
    return Claim(statement, expected_text, "; ".join(observed), consistent, full_only)


def claim_rate_above(
    statement: str,
    successes: int,
    trials: int,
    level: float,
    *,
    full_only: bool = False,
) -> Claim:
    """State that a rejection rate is above its level by three binomial errors."""
    measurement = Measurement.from_rate(successes, trials, level=level)
    tolerance = BINOMIAL_ERRORS * (measurement.error or 0.0)
    return Claim(
        statement,
        f"rate above {level:.0%} by more than three binomial errors",
        measurement.render(),
        measurement.value > level + tolerance,
        full_only,
    )


@dataclasses.dataclass(frozen=True)
class Section:
    """The content of one check: setup, tables and figures, and claims."""

    setup: str
    """What was run: cases, sizes, replicates and draws, from the numbers of the run."""
    blocks: tuple[str | Table | Figure, ...] = ()
    """Paragraphs of MyST text, tables and figures, in page order."""
    claims: tuple[Claim, ...] = ()


@dataclasses.dataclass(frozen=True)
class Check:
    """One zoo check of a session, rerun with fewer replicates."""

    label: str
    """Kebab-case name, unique; the page target is ``validation-<label>``."""
    title: str
    session: str
    """The session whose check it reruns, such as ``12a``."""
    supports: tuple[str, ...]
    """Names or decisions that the check backs, as ``{py:data}``-style roles or text."""
    run: Callable[[Context], Section]

    def __post_init__(self) -> None:
        """Check that the label is kebab case."""
        if LABEL_PATTERN.fullmatch(self.label) is None:
            raise ValueError(f"the label {self.label!r} is not kebab case")


@dataclasses.dataclass(frozen=True)
class WarningCount:
    """A package warning that a check raised, with how often."""

    category: str
    message: str
    count: int


@dataclasses.dataclass(frozen=True)
class CheckResult:
    """A check that has run: its section, its wall time and its warnings."""

    check: Check
    topic: str
    section: Section
    seconds: float
    warnings: tuple[WarningCount, ...]


def run_check(check: Check, topic: str, context: Context) -> CheckResult:
    """Run a check, time it and record the warnings that it raised.

    The package warnings come from :func:`mixed_signals.errors.capture_warnings`;
    any other warning is recorded as well, so that none stops the build.
    """
    started = time.perf_counter()
    with warnings.catch_warnings(record=True) as other_warnings:
        warnings.simplefilter("always")
        with errors.capture_warnings() as package_records:
            section = check.run(context)
    seconds = time.perf_counter() - started
    counts = Counter(
        [(record.category, record.message) for record in package_records]
        + [(caught.category.__name__, str(caught.message)) for caught in other_warnings]
    )
    recorded = tuple(
        WarningCount(category, message, count)
        for (category, message), count in sorted(counts.items())
    )
    return CheckResult(check, topic, section, seconds, recorded)


def failed_claims(results: Sequence[CheckResult], *, quick: bool) -> list[str]:
    """List the claims that are evaluated and not consistent, with their checks."""
    return [
        f"{result.check.label}: {claim.statement} (expected {claim.expected};"
        f" observed {claim.observed})"
        for result in results
        for claim in result.section.claims
        if claim.failed(quick=quick)
    ]


@dataclasses.dataclass(frozen=True)
class Stamp:
    """Versions, machine, seed and mode of a build."""

    package_version: str
    jax_version: str
    jaxlib_version: str
    python_version: str
    platform_text: str
    processor: str
    cpu_name: str
    seed: int
    mode: str
    date_utc: str

    @classmethod
    def create(cls, context: Context) -> "Stamp":
        """Read the versions and the machine of the running build."""
        return cls(
            package_version=importlib.metadata.version("mixed-signals"),
            jax_version=jax.__version__,
            jaxlib_version=importlib.metadata.version("jaxlib"),
            python_version=sys.version.split()[0],
            platform_text=platform.platform(),
            processor=platform.processor() or "unknown",
            cpu_name=_cpu_name(),
            seed=context.seed,
            mode="quick" if context.quick else "full",
            date_utc=datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%d %H:%M UTC"),
        )

    def rows(self) -> list[tuple[str, str]]:
        """Return the stamp as pairs of an item and its value."""
        return [
            ("mixed-signals", self.package_version),
            ("JAX", self.jax_version),
            ("jaxlib", self.jaxlib_version),
            ("Python", self.python_version),
            ("Platform", self.platform_text),
            ("Processor", self.processor),
            ("CPU", self.cpu_name),
            ("Seed", str(self.seed)),
            ("Mode", self.mode),
            ("Date", self.date_utc),
        ]


def _cpu_name() -> str:
    """Return the CPU name that macOS reports, or ``unknown`` without ``sysctl``."""
    try:
        completed = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return "unknown"
    return completed.stdout.strip() or "unknown"


def _role_support(entry: str) -> str:
    return f"- {entry}"


def _rerun_sentence(session: str) -> str:
    """Name the session or the sessions whose check a section reruns."""
    if " and " in session:
        return f"Reruns the checks of sessions {session}."
    return f"Reruns the check of session {session}."


def _render_check(result: CheckResult, *, quick: bool) -> str:
    check, section = result.check, result.section
    parts = [
        f"(validation-{check.label})=\n### {check.title}",
        _rerun_sentence(check.session),
    ]
    if check.supports:
        parts.append(
            "Supports:\n\n" + "\n".join(_role_support(item) for item in check.supports)
        )
    parts.append(section.setup)
    for block in section.blocks:
        parts.append(block if isinstance(block, str) else block.render())
    if section.claims:
        parts.append(
            Table.of(
                "Claims of the check, each with its verdict.",
                ("Claim", "Expected", "Observed", "Verdict"),
                [
                    (
                        claim.statement,
                        claim.expected,
                        claim.observed,
                        claim.verdict(quick=quick),
                    )
                    for claim in section.claims
                ],
            ).render()
        )
    if result.warnings:
        per_category = Counter[str]()
        for entry in result.warnings:
            per_category[entry.category] += entry.count
        parts.append(
            Table.of(
                "Warnings that the check raised; the messages are in the JSON.",
                ("Category", "Count"),
                sorted(per_category.items()),
            ).render()
        )
    else:
        parts.append("The check raised no warning.")
    return "\n\n".join(parts)


def render_page(
    stamp: Stamp,
    results: Sequence[CheckResult],
    *,
    quick: bool,
    total_seconds: float,
) -> str:
    """Write the whole page: introduction, stamp and one subsection per check."""
    parts = ["# Validation", INTRODUCTION]
    parts.append(
        Table.of(
            "Version, machine, seed and mode of this build.",
            ("Item", "Value"),
            stamp.rows(),
        ).render()
    )
    parts.append(
        Table.of(
            "Wall time of each check, in seconds.",
            ("Check", "Wall time (s)"),
            [
                *[
                    (
                        f"{{ref}}`validation-{result.check.label}`",
                        f"{result.seconds:.1f}",
                    )
                    for result in results
                ],
                ("Total", f"{total_seconds:.1f}"),
            ],
        ).render()
    )
    topic = None
    for result in results:
        if result.topic != topic:
            topic = result.topic
            parts.append(f"## {topic}")
        parts.append(_render_check(result, quick=quick))
    return "\n\n".join(parts) + "\n"


def results_to_json(
    stamp: Stamp,
    results: Sequence[CheckResult],
    *,
    quick: bool,
    total_seconds: float,
) -> dict[str, object]:
    """Return the stamp, the times and every check's numbers as plain JSON types."""
    return {
        "stamp": dict(stamp.rows()),
        "total_seconds": total_seconds,
        "checks": [
            {
                "label": result.check.label,
                "title": result.check.title,
                "topic": result.topic,
                "session": result.check.session,
                "supports": list(result.check.supports),
                "seconds": result.seconds,
                "setup": result.section.setup,
                "blocks": [
                    block if isinstance(block, str) else block.to_json()
                    for block in result.section.blocks
                ],
                "claims": [
                    claim.to_json(quick=quick) for claim in result.section.claims
                ],
                "warnings": [dataclasses.asdict(entry) for entry in result.warnings],
            }
            for result in results
        ],
    }


def write_outputs(
    output_directory: Path,
    stamp: Stamp,
    results: Sequence[CheckResult],
    *,
    quick: bool,
    total_seconds: float,
) -> None:
    """Write ``validation.md``, the figures and ``validation/results.json``."""
    (output_directory / "validation").mkdir(parents=True, exist_ok=True)
    payload = results_to_json(stamp, results, quick=quick, total_seconds=total_seconds)
    # * The numbers go first, so that a failing figure never loses them.
    (output_directory / "validation" / "results.json").write_text(
        json.dumps(payload, indent=1, allow_nan=False) + "\n", encoding="utf-8"
    )
    figure_directory = output_directory / "validation" / "figures"
    for result in results:
        for block in result.section.blocks:
            if isinstance(block, Figure):
                block.save(figure_directory)
    page = render_page(stamp, results, quick=quick, total_seconds=total_seconds)
    (output_directory / "validation.md").write_text(page, encoding="utf-8")
