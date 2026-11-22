"""Read a run folder for the report: results, samples, reductions and settings.

:func:`read_run_folder` reads the files that
:func:`~mixed_signals.pipeline.runner.run_pipeline` writes and nothing else, so a
report needs no input file, no eryn and no user code. Only ``results.parquet``
is required, which lets a folder that an API user wrote by hand, with a result
table alone, be reported too; every other part is optional and ``None`` or empty
when its file is absent.

Posterior names
    The names come from the posteriors of ``config.json`` and otherwise from the
    pairs of the result, in order of first appearance. Posterior k of the run,
    in this order, always takes slot k of the series colours.
Samples
    The samples are read only when ``config.json`` gives the names, because
    ``samples/<k>.npz`` belongs to the k-th posterior of the config. The key
    ``"input"`` holds the samples in the space of the config. The keys
    ``"diagnose"`` and ``"compare"`` hold the mapped copies and their space for
    a section with a transform, in the coordinates of the rows of that section.
Settings
    ``report.json`` holds the :class:`ReportSettings`: how many problems the
    report shows in full, and the plot cap of the corner plots.
"""

import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final, Literal, NamedTuple

import polars as pl
from jaxtyping import PRNGKeyArray
from pydantic import (
    BaseModel,
    ConfigDict,
    NonNegativeInt,
    PositiveInt,
    field_validator,
)

from mixed_signals.core.result import Result
from mixed_signals.core.samples import SampleSet
from mixed_signals.core.space import ParamSpace
from mixed_signals.errors import ReportError
from mixed_signals.mcmc.reduce import ReductionRecord
from mixed_signals.prng import derive_key, root_key
from mixed_signals.sigma.thresholds import SIGNIFICANCE_LEVEL

__all__ = [
    "DEFAULT_PROBLEMS",
    "PosteriorSamples",
    "ReportSettings",
    "RunFolder",
    "check_problems",
    "open_run_folder",
    "plot_key",
    "read_run_folder",
    "samples_for",
    "select_problems",
    "significance_level",
]

DEFAULT_PROBLEMS: Final = 10
"""The number of problems that a report shows in full by default."""

RESULT_FILE: Final = "results.parquet"
SETTINGS_FILE: Final = "report.json"
# * The folder of the samples of each section: the input samples lie in its root.
_SECTION_FOLDERS: Final = {"input": "", "diagnose": "diagnose", "compare": "compare"}


class ReportSettings(BaseModel):
    """What a report shows, stored as ``report.json`` next to the notebook.

    Attributes:
        problems: How many problems the report shows in full: a count, those
            with the most flagged rows first; ``"all"``; or a list of problem
            indices, in the order to show them. A list is not empty and does not
            repeat an index.
        plot_cap: The most samples that a posterior contributes to a corner
            plot. Above it, the plot draws a uniform subsample without
            replacement at the bandwidth of the full set; ``None`` draws every
            sample.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    problems: PositiveInt | Literal["all"] | tuple[NonNegativeInt, ...] = (
        DEFAULT_PROBLEMS
    )
    plot_cap: PositiveInt | None = None

    @field_validator("problems")
    @classmethod
    def _check_indices(
        cls, problems: PositiveInt | Literal["all"] | tuple[NonNegativeInt, ...]
    ) -> PositiveInt | Literal["all"] | tuple[NonNegativeInt, ...]:
        if isinstance(problems, tuple):
            if not problems:
                raise ValueError("a list of problems must not be empty")
            if len(set(problems)) != len(problems):
                raise ValueError("a list of problems must not repeat an index")
        return problems


class PosteriorSamples(NamedTuple):
    """The samples of every posterior in one parameter space."""

    space: ParamSpace
    """The parameter space of the samples."""
    sets: Mapping[str, SampleSet]
    """The samples of each posterior by name, in config order."""


class RunFolder(NamedTuple):
    """Everything that a report reads from a run folder."""

    path: Path
    """The run folder."""
    result: Result
    """Every row of every engine."""
    names: tuple[str, ...]
    """The posterior names: the config order, or first appearance in the rows."""
    space: ParamSpace | None
    """The parameter space of the config; ``None`` without ``config.json``."""
    config: Mapping[str, Any] | None
    """The ``"config"`` mapping of ``config.json``; ``None`` without the file."""
    config_hash: str | None
    """The SHA-256 of the bytes of ``config.json``; ``None`` without the file."""
    samples: Mapping[str, PosteriorSamples]
    """The samples by key ``"input"``, ``"diagnose"`` and ``"compare"``, each
    present only when its files are."""
    reductions: Mapping[str, ReductionRecord]
    """The reduction of every run of chains by posterior name."""
    cost: pl.DataFrame | None
    """The table of the estimate; ``None`` without ``cost.parquet``."""
    provenance: Mapping[str, Any] | None
    """The content of ``provenance.json``; ``None`` without the file."""
    settings: ReportSettings
    """The settings of ``report.json``, or the defaults."""


def read_run_folder(path: str | os.PathLike[str]) -> RunFolder:
    """Read a run folder.

    The function reads files only: it never imports user code or eryn.

    Args:
        path: The run folder, which holds at least ``results.parquet``.

    Returns:
        The parts of the folder that are present.

    Raises:
        ReportError: If the folder holds no ``results.parquet``, or the
            problem indices of ``report.json`` name a problem that the result
            does not hold.
        pydantic.ValidationError: If ``report.json``, ``config.json`` or
            ``reductions.json`` is invalid.
        ValueError: If a sample archive does not fit the config.
    """
    folder = Path(path)
    if not (folder / RESULT_FILE).is_file():
        raise ReportError(
            f"the folder {folder} holds no {RESULT_FILE}; give a run folder that"
            " mixed-signals run wrote"
        )
    result = Result.read_parquet(folder / RESULT_FILE)
    config_bytes = _read_bytes(folder / "config.json")
    config_json = None if config_bytes is None else json.loads(config_bytes)
    config = None if config_json is None else config_json["config"]
    space = (
        None if config_json is None else ParamSpace.model_validate(config_json["space"])
    )
    names = (
        _names_from_rows(result.table)
        if config is None
        else tuple(entry["name"] for entry in config["posteriors"])
    )
    settings_bytes = _read_bytes(folder / SETTINGS_FILE)
    settings = (
        ReportSettings()
        if settings_bytes is None
        else ReportSettings.model_validate_json(settings_bytes)
    )
    check_problems(settings, result.table)
    return RunFolder(
        path=folder,
        result=result,
        names=names,
        space=space,
        config=config,
        config_hash=(
            None if config_bytes is None else hashlib.sha256(config_bytes).hexdigest()
        ),
        samples=_read_samples(folder, names, space),
        reductions=_read_reductions(folder / "reductions.json"),
        cost=(
            pl.read_parquet(folder / "cost.parquet")
            if (folder / "cost.parquet").is_file()
            else None
        ),
        provenance=_read_json(folder / "provenance.json"),
        settings=settings,
    )


def open_run_folder(path: str | os.PathLike[str]) -> RunFolder | None:
    """Read a run folder, or return ``None`` when the folder holds no result.

    The notebook opens the folder it sits in with this function, so that a copy
    outside a run folder stops with an explanation, not with an exception.

    Args:
        path: The folder to open.

    Returns:
        The run folder, or ``None`` when it holds no ``results.parquet``.

    Raises:
        ReportError: As :func:`read_run_folder`, for a folder that holds a result.
    """
    if not (Path(path) / RESULT_FILE).is_file():
        return None
    return read_run_folder(path)


def check_problems(settings: ReportSettings, table: pl.DataFrame) -> None:
    """Raise unless a list of problem indices names problems of the result.

    Args:
        settings: The settings of the report.
        table: The result table.

    Raises:
        ReportError: If a listed index is not a problem of the result.
    """
    if not isinstance(settings.problems, tuple):
        return
    held = set(table["problem"].to_list())
    missing = [problem for problem in settings.problems if problem not in held]
    if missing:
        raise ReportError(
            f"the problems {missing} are not in the result, which holds the"
            f" problems {sorted(held)}"
        )


def samples_for(folder: RunFolder, section: str) -> PosteriorSamples | None:
    """Return the samples in the coordinates of a section's rows.

    Args:
        folder: The run folder.
        section: ``"diagnose"`` or ``"compare"``.

    Returns:
        The mapped samples of the section when it has a transform, otherwise the
        samples of the config, otherwise ``None``.
    """
    return folder.samples.get(section) or folder.samples.get("input")


def significance_level(folder: RunFolder, engine: str) -> float:
    """Return the σ_sig that a row of an engine must reach to flag.

    Args:
        folder: The run folder.
        engine: ``"diagnose"``, ``"compare"`` or ``"compare_signals"``;
            reweighting rows have no level.

    Returns:
        The ``level`` of the engine's section in ``config.json``, otherwise
        :data:`~mixed_signals.sigma.thresholds.SIGNIFICANCE_LEVEL`.
    """
    section = None if folder.config is None else folder.config.get(engine)
    if section is None:
        return SIGNIFICANCE_LEVEL
    return float(section.get("level", SIGNIFICANCE_LEVEL))


def select_problems(folder: RunFolder) -> tuple[int, ...]:
    """Return the problems that the report shows in full, in display order.

    Args:
        folder: The run folder, whose settings choose the problems.

    Returns:
        Every problem for ``"all"``; for a count, that many problems ranked by
        the number of rows that flag, most first, ties by index; for a list, the
        problems of the list in its order.
    """
    problems = folder.settings.problems
    if isinstance(problems, tuple):
        return problems
    table = folder.result.table
    held = sorted(set(table["problem"].to_list()))
    if problems == "all":
        return tuple(held)
    flagged = table.filter(pl.col("flag").fill_null(value=False)).group_by("problem")
    counts = dict(flagged.len().iter_rows())
    ranked = sorted((-counts.get(problem, 0), problem) for problem in held)
    return tuple(problem for _, problem in ranked[:problems])


def plot_key(folder: RunFolder, section: str, problem: int, name: str) -> PRNGKeyArray:
    """Return the key of the plot subsample of one posterior of one problem.

    The key is ``derive_key(root_key(seed), "report", "plot", section, problem,
    name)``, so a report of the same run draws the same subsample.

    Args:
        folder: The run folder.
        section: ``"diagnose"`` or ``"compare"``.
        problem: The problem index.
        name: The posterior name.

    Returns:
        The key. The seed comes from ``provenance.json``, otherwise from the
        provenance of the first row, otherwise it is 0.
    """
    return derive_key(root_key(_seed(folder)), "report", "plot", section, problem, name)


def _seed(folder: RunFolder) -> int:
    """Return the seed of the run, or 0 when no record holds one."""
    if folder.provenance is not None and folder.provenance.get("seed") is not None:
        return int(folder.provenance["seed"])
    table = folder.result.table
    if table.height > 0:
        seed = table["provenance"].struct.field("seed")[0]
        if seed is not None:
            return int(seed)
    return 0


def _read_bytes(path: Path) -> bytes | None:
    """Return the bytes of a file, or ``None`` when it does not exist."""
    return path.read_bytes() if path.is_file() else None


def _read_json(path: Path) -> Mapping[str, Any] | None:
    """Return the content of a JSON file, which may hold NaN, or ``None``."""
    content = _read_bytes(path)
    return None if content is None else json.loads(content)


def _read_reductions(path: Path) -> Mapping[str, ReductionRecord]:
    """Return the reduction records by posterior name; none without the file."""
    records = _read_json(path) or {}
    return {
        name: ReductionRecord.model_validate(record) for name, record in records.items()
    }


def _names_from_rows(table: pl.DataFrame) -> tuple[str, ...]:
    """Return the names of the pairs of the rows, in order of first appearance."""
    names = table["pair"].explode().drop_nulls().unique(maintain_order=True)
    return tuple(names.to_list())


def _read_samples(
    folder: Path, names: tuple[str, ...], space: ParamSpace | None
) -> dict[str, PosteriorSamples]:
    """Read the samples of the config, and the mapped copies of each section."""
    if space is None:
        return {}
    samples: dict[str, PosteriorSamples] = {}
    for section, subfolder in _SECTION_FOLDERS.items():
        section_folder = folder / "samples" / subfolder
        archives = [
            section_folder / f"{position}.npz" for position in range(len(names))
        ]
        if not all(archive.is_file() for archive in archives):
            continue
        if section == "input":
            section_space = space
        elif (section_folder / "space.yaml").is_file():
            section_space = ParamSpace.from_yaml(section_folder / "space.yaml")
        else:
            continue
        samples[section] = PosteriorSamples(
            space=section_space,
            sets={
                name: SampleSet.from_npz(archive)
                for name, archive in zip(names, archives, strict=True)
            },
        )
    return samples
