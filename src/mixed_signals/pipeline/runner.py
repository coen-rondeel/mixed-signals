"""Estimate and run a pipeline, and write its run folder.

:func:`run_pipeline` reads a config
(:func:`~mixed_signals.pipeline.config.load_config`), prepares its inputs,
estimates the cost of every engine, runs the engines that the config names and
writes one folder. :func:`estimate_pipeline` stops after the estimate and writes
nothing.

Steps
    A run first reads the config and resolves the run folder, and stops with a
    :class:`~mixed_signals.errors.ConfigError` when the folder already holds a
    ``results.parquet`` and ``overwrite`` is false, before any work. It then
    prepares everything inside one block that holds the folder of the config on
    ``sys.path``: the parameter space, the transform of each section, the
    generator, the noise model and the arrays, the posteriors, and the reduction
    of every run of chains with :func:`~mixed_signals.mcmc.reduce.reduce_chains`,
    which gives the reduction that ``diagnose`` makes. It then estimates every
    engine with the arguments of its section, so every argument of every engine is
    checked before any engine runs; an argument that does not fit raises a
    :class:`~mixed_signals.errors.ConfigError` that names the section. The
    engines run in the order ``diagnose``, ``compare``, ``compare_signals``, each
    with the seed of the config as its key and the config hash in every row. The
    folder is written last.
Budget
    The pipeline logs the estimate, raises one
    :class:`~mixed_signals.errors.CostWarning` when the total of all engines is
    above the budget of the config and goes on. It calls the engines with an
    infinite budget, so they do not warn again.
Run folder
    The folder holds five files, all written on every run, so that ``overwrite``
    never leaves a stale file of an earlier run. ``results.parquet`` holds every
    row of every engine in run order. ``config.json`` holds the canonical JSON that
    the config hash covers, byte for byte, so ``sha256sum config.json`` gives the
    hash in the provenance of every row. It leaves out ``output`` and ``budget``,
    which change no row; ``provenance.json`` records the run folder and the
    budget, the two settings that ``config.json`` leaves out. ``reductions.json``
    holds the :class:`~mixed_signals.mcmc.reduce.ReductionRecord` of every run of
    chains by name, in the order of the config. ``cost.parquet`` holds the table
    of the estimate. ``provenance.json`` holds the config hash, the seed, the
    budget (JSON writes infinity as ``Infinity``), the versions of the package,
    JAX and Python, the device platform and the machine, the config file, the run
    folder, the start time in UTC, the wall and estimated seconds, the rows and
    seconds of each engine, the file, size and SHA-256 of every input and the file
    and SHA-256 of the module of every import path.
Why last
    The folder is written once, at the end, so a failed run writes nothing and a
    folder with a ``results.parquet`` is always a finished run.
Determinism
    ``results.parquet`` is the same, table for table, from the same config, data
    and seed: the rows hold no time, and every key derives from the seed.
    ``provenance.json`` holds the start time and the wall seconds, which differ.
Logging
    The progress goes to the logger ``mixed_signals.pipeline`` at INFO: the
    estimate, the rows and wall time of each engine and the files written. The
    library adds no handler; the command line does.

References:
    :cite:t:`nist2015secure`.
"""

import datetime
import json
import logging
import math
import os
import platform
import time
from collections.abc import Callable, Mapping
from functools import partial
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final, NamedTuple

import jax
import numpy as np
import polars as pl
from jaxtyping import Array

from mixed_signals import __version__
from mixed_signals.core.chains import Chains
from mixed_signals.core.result import Result
from mixed_signals.core.samples import SampleSet
from mixed_signals.core.space import ParamSpace
from mixed_signals.diagnostics import flags
from mixed_signals.engines._arguments import _read_fisher
from mixed_signals.engines.compare import compare
from mixed_signals.engines.cost import (
    CostEstimate,
    _costliest_statistics,
    _format_seconds,
    estimate_compare_cost,
    estimate_compare_signals_cost,
    estimate_diagnose_cost,
    warn_if_over_budget,
)
from mixed_signals.engines.dataspace import compare_signals
from mixed_signals.engines.diagnose import diagnose
from mixed_signals.errors import CapabilityError, ConfigError, capture_warnings
from mixed_signals.mcmc.reduce import Reduction, reduce_chains
from mixed_signals.pipeline._sources import Sources, config_folder_on_path
from mixed_signals.pipeline.config import (
    ChainsEntry,
    ErynEntry,
    PipelineConfig,
    _canonical_json,
    _config_hash,
    load_config,
)
from mixed_signals.transforms.base import Transform
from mixed_signals.transforms.whiten import Whitening

__all__ = [
    "LOGGER_NAME",
    "PipelineEstimate",
    "PipelineRun",
    "estimate_pipeline",
    "run_pipeline",
]

LOGGER_NAME: Final = "mixed_signals.pipeline"
"""Name of the logger that carries the progress of a run."""

_LOGGER: Final = logging.getLogger(LOGGER_NAME)

# * The engines in run order, which is also the order of their sections.
_ENGINES: Final = ("diagnose", "compare", "compare_signals")
_RESULT_FILE: Final = "results.parquet"
_FILES_WRITTEN: Final = (
    "results.parquet, config.json, reductions.json, cost.parquet and provenance.json"
)

# * Errors that an engine raises for an argument that does not fit its inputs.
_ENGINE_ERRORS: Final = (
    ValueError,
    TypeError,
    KeyError,
    NotImplementedError,
    CapabilityError,
)


class PipelineEstimate(NamedTuple):
    """The estimated cost of the engines of a config."""

    estimates: Mapping[str, CostEstimate]
    """The estimate of each engine that the config names, in run order."""
    seconds: float
    """The estimated seconds of all engines on one CPU of the developers' machine."""


class PipelineRun(NamedTuple):
    """What a finished run returns; the run folder holds the same."""

    result: Result
    """Every row of every engine, in run order."""
    reductions: Mapping[str, Reduction]
    """The reduction of every run of chains, by posterior name, in config order."""
    estimate: PipelineEstimate
    """The estimate that the run started from."""
    config_hash: str
    """The SHA-256 of the canonical config, as in the provenance of every row."""
    output: Path
    """The absolute path of the run folder."""


class _SectionObjects(NamedTuple):
    """The objects that a section of the config names."""

    transform: Transform | None
    fisher: np.ndarray | None
    centre: np.ndarray | None
    generator: Any
    noise: Any


class _Prepared(NamedTuple):
    """Everything an engine needs, read from the config and its files."""

    config: PipelineConfig
    space: ParamSpace
    names: tuple[str, ...]
    inputs: tuple[Chains | SampleSet, ...]
    """One input per posterior: chains, or a sample set."""
    sets: tuple[SampleSet, ...]
    """One flat set per posterior: the reduction of chains, or the loaded set."""
    reductions: Mapping[str, Reduction]
    burn_ins: tuple[int, ...]
    """The burn-in of each run of chains, in config order."""
    thins: tuple[int | str, ...]
    """The thinning of each run of chains, in config order."""
    objects: Mapping[str, _SectionObjects]
    """The objects of each section that the config names."""
    sources: Sources
    canonical: bytes
    config_hash: str


class _EngineTiming(NamedTuple):
    """The rows and the wall seconds of one engine run."""

    rows: int
    seconds: float


def estimate_pipeline(
    config: PipelineConfig | str | os.PathLike[str],
) -> PipelineEstimate:
    """Estimate the run time of every engine of a config, and write nothing.

    The estimate reads the data and resolves the code of the config, as a run does,
    but computes no metric. It is logged to ``mixed_signals.pipeline``.

    Args:
        config: A validated config, or the path of a config file.

    Returns:
        The estimate of each engine and their total.

    Raises:
        ConfigError: If the config does not resolve, a file does not fit, or an
            argument of an engine does not fit its inputs.
        pydantic.ValidationError: If a config file is invalid.
        MissingDependencyError: If a reader needs an optional dependency that is
            not installed.
    """
    loaded = _read_config(config)
    with config_folder_on_path(loaded.base_folder):
        estimate = _estimate_prepared(_prepare(loaded))
    _log_estimate(estimate, loaded.budget)
    return estimate


def run_pipeline(
    config: PipelineConfig | str | os.PathLike[str],
    *,
    output: str | os.PathLike[str] | None = None,
    overwrite: bool = False,
) -> PipelineRun:
    """Run the engines of a config and write the run folder.

    The module docstring describes the steps and the files. A failed run writes
    nothing.

    Args:
        config: A validated config, or the path of a config file.
        output: The run folder, instead of the ``output`` of the config; taken as
            given, so a relative path starts at the working folder.
        overwrite: Whether to replace a run folder that already holds results.

    Returns:
        The rows of every engine, the reductions, the estimate, the config hash
        and the run folder.

    Raises:
        ConfigError: If the run folder already holds results and ``overwrite`` is
            false, the config does not resolve, a file does not fit, or an
            argument of an engine does not fit its inputs.
        pydantic.ValidationError: If a config file is invalid.
        MissingDependencyError: If a reader needs an optional dependency that is
            not installed.

    Warns:
        CostWarning: Once, if the estimate of all engines exceeds the budget of
            the config. The run goes on.
    """
    started = datetime.datetime.now(datetime.UTC).replace(microsecond=0)
    clock = time.perf_counter()
    loaded = _read_config(config)
    folder = _run_folder(loaded, output, overwrite)
    with config_folder_on_path(loaded.base_folder):
        prepared = _prepare(loaded)
        estimate = _estimate_prepared(prepared)
        _log_estimate(estimate, loaded.budget)
        warn_if_over_budget("pipeline", _combine(estimate), loaded.budget)
        results, timings = _run_engines(prepared)
    result = Result.concat(results.values())
    wall_seconds = time.perf_counter() - clock
    _write_folder(
        folder, prepared, estimate, result, timings, started.isoformat(), wall_seconds
    )
    _LOGGER.info(f"Wrote {_FILES_WRITTEN} to {folder}")
    return PipelineRun(
        result=result,
        reductions=prepared.reductions,
        estimate=estimate,
        config_hash=prepared.config_hash,
        output=folder,
    )


def _read_config(config: PipelineConfig | str | os.PathLike[str]) -> PipelineConfig:
    """Return the config, reading the file when a path is given."""
    return config if isinstance(config, PipelineConfig) else load_config(config)


def _run_folder(
    config: PipelineConfig, output: str | os.PathLike[str] | None, overwrite: bool
) -> Path:
    """Return the absolute run folder, after checking that it can take a new run."""
    folder = (
        config.base_folder / config.output if output is None else Path(output)
    ).resolve()
    if folder.exists() and not folder.is_dir():
        raise ConfigError(f"the output {folder} exists and is not a folder")
    if (folder / _RESULT_FILE).exists() and not overwrite:
        raise ConfigError(
            f"the folder {folder} already holds {_RESULT_FILE}; choose another folder"
            " or overwrite it"
        )
    return folder


# * Preparing the run


def _prepare(config: PipelineConfig) -> _Prepared:
    """Read the space, the objects and the posteriors, and reduce the chains.

    The caller holds the folder of the config on ``sys.path``.
    """
    sources = Sources(config.base_folder)
    space, priors = sources.read_space(config.space)
    objects = {
        name: _read_objects(sources, name, getattr(config, name), priors)
        for name in _ENGINES
        if getattr(config, name) is not None
    }
    inputs = tuple(
        sources.read_posterior(entry, space, priors) for entry in config.posteriors
    )
    sets: list[SampleSet] = []
    reductions: dict[str, Reduction] = {}
    burn_ins: list[int] = []
    thins: list[int | str] = []
    for entry, loaded in zip(config.posteriors, inputs, strict=True):
        if isinstance(loaded, Chains):
            assert isinstance(entry, ErynEntry | ChainsEntry)
            reductions[entry.name] = _reduce(entry, loaded)
            burn_ins.append(entry.burn_in)
            thins.append(entry.thin)
            sets.append(reductions[entry.name].sample_set)
        else:
            sets.append(loaded)
    canonical = _canonical_json(config, space)
    return _Prepared(
        config=config,
        space=space,
        names=tuple(entry.name for entry in config.posteriors),
        inputs=inputs,
        sets=tuple(sets),
        reductions=MappingProxyType(reductions),
        burn_ins=tuple(burn_ins),
        thins=tuple(thins),
        objects=MappingProxyType(objects),
        sources=sources,
        canonical=canonical,
        config_hash=_config_hash(canonical),
    )


def _read_objects(
    sources: Sources, name: str, section: Any, priors: object | None
) -> _SectionObjects:
    """Resolve the transform, the arrays and the code of one section."""
    fisher = getattr(section, "fisher", None)
    centre = getattr(section, "centre", None)
    generator = getattr(section, "generator", None)
    noise = getattr(section, "noise", None)
    return _SectionObjects(
        transform=sources.read_transform(section.transform, priors),
        fisher=None if fisher is None else sources.read_array(fisher, f"{name} fisher"),
        centre=None if centre is None else sources.read_array(centre, f"{name} centre"),
        generator=None if generator is None else sources.resolve(generator),
        noise=None if noise is None else sources.resolve(noise),
    )


def _reduce(entry: ErynEntry | ChainsEntry, chains: Chains) -> Reduction:
    """Reduce one run of chains with the burn-in and thinning of its entry."""
    try:
        return reduce_chains(chains, burn_in=entry.burn_in, thin=entry.thin)
    except (ValueError, TypeError) as error:
        raise ConfigError(
            f"cannot reduce the chains of the posterior {entry.name!r}: {error}"
        ) from error


# * Estimating the engines


def _guarded[Value](section: str, call: Callable[[], Value]) -> Value:
    """Call an engine, and name its section when an argument does not fit."""
    try:
        return call()
    except _ENGINE_ERRORS as error:
        raise ConfigError(
            f"the {section} section does not fit its inputs: {error}"
        ) from error


def _estimate_prepared(prepared: _Prepared) -> PipelineEstimate:
    """Estimate every engine that the config names, in run order."""
    estimates = {
        name: _guarded(name, partial(_ESTIMATORS[name], prepared))
        for name in _ENGINES
        if getattr(prepared.config, name) is not None
    }
    return PipelineEstimate(
        estimates=MappingProxyType(estimates),
        seconds=sum(estimate.seconds for estimate in estimates.values()),
    )


def _estimate_diagnose(prepared: _Prepared) -> CostEstimate:
    section = prepared.config.diagnose
    assert section is not None
    return estimate_diagnose_cost(
        list(prepared.inputs),
        prepared.names,
        prepared.space,
        burn_in=list(prepared.burn_ins),
        thin=list(prepared.thins),
        transform=prepared.objects["diagnose"].transform,
        diagnostics=section.diagnostics,
        draws=section.draws,
        cap=section.cap,
        pair_cap=section.pair_cap,
    )


def _estimate_compare(prepared: _Prepared) -> CostEstimate:
    section = prepared.config.compare
    assert section is not None
    objects = prepared.objects["compare"]
    # * The estimate takes no Fisher arrays, so they are read and built here.
    fisher, centre = _read_fisher(
        objects.fisher,
        objects.centre,
        prepared.sets[0].shape.batch,
        len(prepared.space),
        objects.transform,
    )
    if section.whitening == "fisher":
        transform = objects.transform
        row_space = (
            prepared.space if transform is None else transform.map_space(prepared.space)
        )
        _check_fisher_whitenings(fisher, centre, row_space)
    return estimate_compare_cost(
        list(prepared.sets),
        prepared.names,
        prepared.space,
        reference=section.reference,
        whitening=section.whitening,
        transform=objects.transform,
        metrics=section.metrics,
        null=section.null,
        draws=section.draws,
        cap=section.cap,
        sinkhorn_epsilon_factors=section.sinkhorn_epsilon_factors,
        uncertainty=section.uncertainty,
        half_samples=section.half_samples,
    )


def _check_fisher_whitenings(
    fisher: Array | None, centre: Array | None, space: ParamSpace
) -> None:
    """Build the Fisher whitening of every problem, to reject a bad matrix early.

    The engine builds its whitenings only after ``diagnose`` has run, and the
    run folder is written last, so a matrix that is not finite, symmetric and
    positive definite would cost the whole run. The engine records the warnings
    of the whitening in its rows, so they are not raised here.
    """
    assert fisher is not None
    assert centre is not None
    with capture_warnings():
        for problem in range(fisher.shape[0]):
            Whitening.from_fisher(
                fisher[problem], centre[problem], space, decorrelate=True
            )


def _estimate_compare_signals(prepared: _Prepared) -> CostEstimate:
    section = prepared.config.compare_signals
    assert section is not None
    objects = prepared.objects["compare_signals"]
    return estimate_compare_signals_cost(
        list(prepared.sets),
        prepared.names,
        prepared.space,
        generator=objects.generator,
        noise=objects.noise,
        reference=section.reference,
        fisher=objects.fisher,
        centre=objects.centre,
        transform=objects.transform,
        null=section.null,
        draws=section.draws,
        cap=section.cap,
        uncertainty=section.uncertainty,
        half_samples=section.half_samples,
        backend=section.backend,
        chunk_size=section.chunk_size,
    )


_ESTIMATORS: Final[Mapping[str, Callable[[_Prepared], CostEstimate]]] = {
    "diagnose": _estimate_diagnose,
    "compare": _estimate_compare,
    "compare_signals": _estimate_compare_signals,
}


def _combine(estimate: PipelineEstimate) -> CostEstimate:
    """Return the estimate of all engines: their tables stacked and seconds summed."""
    table = pl.concat(
        [each.table for each in estimate.estimates.values()], how="vertical"
    )
    return CostEstimate(table=table, seconds=estimate.seconds)


def _log_estimate(estimate: PipelineEstimate, budget: float) -> None:
    """Log the seconds of each engine, the total, the budget and the costliest."""
    budget_text = (
        "no budget" if math.isinf(budget) else f"a budget of {_format_seconds(budget)}"
    )
    _LOGGER.info(
        "Estimated run time on one CPU of the developers' machine:"
        f" {_format_seconds(estimate.seconds)}, against {budget_text}"
    )
    for name, each in estimate.estimates.items():
        _LOGGER.info(f"  {name}: {_format_seconds(each.seconds)}")
    _LOGGER.info(
        f"Costliest statistics: {_costliest_statistics(_combine(estimate).table)}"
    )


# * Running the engines


def _run_diagnose(prepared: _Prepared) -> Result:
    section = prepared.config.diagnose
    assert section is not None
    return diagnose(
        list(prepared.inputs),
        prepared.names,
        prepared.space,
        key=prepared.config.seed,
        burn_in=list(prepared.burn_ins),
        thin=list(prepared.thins),
        transform=prepared.objects["diagnose"].transform,
        diagnostics=section.diagnostics,
        draws=section.draws,
        cap=section.cap,
        pair_cap=section.pair_cap,
        # * A threshold of the config keeps the side and tail of its default.
        thresholds={
            metric: flags.THRESHOLDS[metric]._replace(value=value)
            for metric, value in section.thresholds.items()
        },
        level=section.level,
        budget=math.inf,
        config_hash=prepared.config_hash,
    ).result


def _run_compare(prepared: _Prepared) -> Result:
    section = prepared.config.compare
    assert section is not None
    objects = prepared.objects["compare"]
    return compare(
        list(prepared.sets),
        prepared.names,
        prepared.space,
        key=prepared.config.seed,
        reference=section.reference,
        whitening=section.whitening,
        fisher=objects.fisher,
        centre=objects.centre,
        transform=objects.transform,
        metrics=section.metrics,
        null=section.null,
        draws=section.draws,
        cap=section.cap,
        sinkhorn_epsilon_factors=section.sinkhorn_epsilon_factors,
        uncertainty=section.uncertainty,
        half_samples=section.half_samples,
        thresholds=section.thresholds,
        level=section.level,
        budget=math.inf,
        config_hash=prepared.config_hash,
    )


def _run_compare_signals(prepared: _Prepared) -> Result:
    section = prepared.config.compare_signals
    assert section is not None
    objects = prepared.objects["compare_signals"]
    return compare_signals(
        list(prepared.sets),
        prepared.names,
        prepared.space,
        generator=objects.generator,
        noise=objects.noise,
        key=prepared.config.seed,
        reference=section.reference,
        fisher=objects.fisher,
        centre=objects.centre,
        transform=objects.transform,
        null=section.null,
        draws=section.draws,
        cap=section.cap,
        uncertainty=section.uncertainty,
        half_samples=section.half_samples,
        thresholds=section.thresholds,
        level=section.level,
        budget=math.inf,
        backend=section.backend,
        chunk_size=section.chunk_size,
        config_hash=prepared.config_hash,
    )


_RUNNERS: Final[Mapping[str, Callable[[_Prepared], Result]]] = {
    "diagnose": _run_diagnose,
    "compare": _run_compare,
    "compare_signals": _run_compare_signals,
}


def _run_engines(
    prepared: _Prepared,
) -> tuple[dict[str, Result], dict[str, _EngineTiming]]:
    """Run the engines that the config names, in order, and time each."""
    results: dict[str, Result] = {}
    timings: dict[str, _EngineTiming] = {}
    for name in _ENGINES:
        if getattr(prepared.config, name) is None:
            continue
        clock = time.perf_counter()
        results[name] = _RUNNERS[name](prepared)
        timings[name] = _EngineTiming(len(results[name]), time.perf_counter() - clock)
        _LOGGER.info(
            f"{name} finished: {timings[name].rows} rows in"
            f" {_format_seconds(timings[name].seconds)}"
        )
    return results, timings


# * Writing the run folder


def _write_folder(
    folder: Path,
    prepared: _Prepared,
    estimate: PipelineEstimate,
    result: Result,
    timings: Mapping[str, _EngineTiming],
    started: str,
    wall_seconds: float,
) -> None:
    """Write the five files of the run folder."""
    folder.mkdir(parents=True, exist_ok=True)
    result.write_parquet(folder / _RESULT_FILE)
    # * The exact bytes of the hash, with no trailing newline.
    (folder / "config.json").write_bytes(prepared.canonical)
    records = {
        name: reduction.record.model_dump(mode="json")
        for name, reduction in prepared.reductions.items()
    }
    (folder / "reductions.json").write_text(_json_text(records), encoding="utf-8")
    _combine(estimate).table.write_parquet(folder / "cost.parquet")
    provenance = {
        "config_hash": prepared.config_hash,
        "seed": prepared.config.seed,
        "budget": prepared.config.budget,
        "package_version": __version__,
        "jax_version": jax.__version__,
        "python_version": platform.python_version(),
        "device_platform": jax.default_backend(),
        "machine": platform.platform(),
        "config_file": (
            None
            if prepared.config.config_file is None
            else str(prepared.config.config_file)
        ),
        "output": str(folder),
        "started": started,
        "wall_seconds": wall_seconds,
        "estimated_seconds": estimate.seconds,
        "engines": [
            {
                "name": name,
                "rows": timing.rows,
                "estimated_seconds": estimate.estimates[name].seconds,
                "wall_seconds": timing.seconds,
            }
            for name, timing in timings.items()
        ],
        "inputs": [record._asdict() for record in prepared.sources.inputs],
        "modules": [record._asdict() for record in prepared.sources.modules.values()],
    }
    (folder / "provenance.json").write_text(_json_text(provenance), encoding="utf-8")


def _json_text(value: object) -> str:
    """Write JSON indented by two, with NaN and infinity as JSON constants."""
    return json.dumps(value, indent=2) + "\n"
