"""Tables of a result, for the report and for API users: polars only, no plotting.

Every function takes a result table, ``folder.result.table`` or any other result
table, and returns a new :class:`polars.DataFrame`, or ``None`` where its docstring
says so. Floats stay floats; the sections round them for display.

Engines
    The result table of a pipeline holds the rows of four engines in one table.
    :func:`row_engines` tells them apart by the metric and the space of a row:
    reweighting rows have two names like comparison rows, so their metrics come
    first, then the rows of data space, then the rows with one name.
Effects
    :func:`with_effects` adds the effect size that a threshold acts on, whether
    σ_sig reaches the significance level, and whether the p-value comes from the
    tail fit. The effect of a comparison row is the one of
    :func:`~mixed_signals.sigma.thresholds.verdict`, and the effect of a
    diagnostic row is the one of :func:`~mixed_signals.diagnostics.flags.evaluate`,
    so the tables can never disagree with the flags of the engines.
Pairs
    The two directions of the Kullback-Leibler rows belong to one pair, so a pair
    is unordered and keeps the orientation of its first row.

References:
    :cite:t:`vehtari2024pareto`; :cite:t:`kullback1951information`.
"""

import json
import math
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, Final

import jax.numpy as jnp
import polars as pl

from mixed_signals.report.folder import RunFolder
from mixed_signals.reweight.psis import pareto_k_threshold
from mixed_signals.sigma.maps import null_subtract
from mixed_signals.sigma.thresholds import (
    R_EFF_METRICS,
    SIGNIFICANCE_LEVEL,
    Verdict,
    verdict,
)
from mixed_signals.transforms.base import COMPOSITION_SEPARATOR

__all__ = [
    "DIAGNOSTIC_FAMILY_METRICS",
    "REWEIGHT_METRICS",
    "chain_health_table",
    "comparison_pairs",
    "comparison_table",
    "cost_table",
    "dataspace_table",
    "diagnostic_table",
    "flag_counts",
    "inputs_table",
    "pair_rows",
    "posterior_rows",
    "rate_table",
    "reweighting_table",
    "row_engines",
    "timings_table",
    "warnings_table",
    "with_effects",
]

REWEIGHT_METRICS: Final = ("pareto_k", "importance_ess", "log_evidence_ratio")
"""The metrics of the rows that ``reweight`` writes."""

DIAGNOSTIC_FAMILY_METRICS: Final[Mapping[str, tuple[str, ...]]] = MappingProxyType(
    {
        "chains": (
            "tau",
            "ess",
            "split_chain_ks",
            "split_chain_kuiper_ks",
            "rhat",
            "acceptance_rate",
            "swap_rate",
        ),
        "shape": (
            "skewness",
            "kurtosis",
            "circular_skewness",
            "circular_kurtosis",
            "anderson_darling",
            "mardia_skewness",
            "mardia_kurtosis",
        ),
        "dependence": (
            "pearson",
            "spearman",
            "distance_correlation",
            "mutual_information",
        ),
        "modes": ("gmm_bic", "mode_count", "mode_dip"),
        "railing": ("railing_lower", "railing_upper"),
    }
)
"""The metrics of each family of ``diagnose``, in the order of its families."""

_FAMILY_OF_METRIC: Final = {
    metric: family
    for family, metrics in DIAGNOSTIC_FAMILY_METRICS.items()
    for metric in metrics
}
_KL_METRICS: Final = ("kl", "kl_location", "kl_shape")
_SPLIT_METRICS: Final = ("split_chain_ks", "split_chain_kuiper_ks")
_CHAIN_METRICS: Final = ("tau", "ess", *_SPLIT_METRICS, "rhat")
_RATE_METRICS: Final = ("acceptance_rate", "swap_rate")
_COMPARISON_ENGINES: Final = ("compare", "compare_signals")
_DIRECTION_SEPARATOR: Final = "‖"  # a double vertical line, read "a given b"
_CHART_SPACE: Final = "chart(fisher)"
_DATA_SPACE: Final = "data"
_HASH_DIGITS: Final = 12


def row_engines(table: pl.DataFrame) -> pl.Series:
    """Return the engine of each row.

    The rules apply in this order, because reweighting rows have two names like
    comparison rows: a metric of :data:`REWEIGHT_METRICS` is ``"reweight"``; a
    space whose last segment, split on ``" | "``, is ``"data"``, starts with
    ``"data("`` or is ``"chart(fisher)"`` is ``"compare_signals"``; a pair of
    one name is ``"diagnose"``; anything else is ``"compare"``.

    Args:
        table: A result table.

    Returns:
        The engine names as strings, one per row.
    """
    last_segment = pl.col("space").str.split(COMPOSITION_SEPARATOR).list.last()
    in_data_space = (
        (last_segment == _DATA_SPACE)
        | last_segment.str.starts_with(f"{_DATA_SPACE}(")
        | (last_segment == _CHART_SPACE)
    )
    engine = (
        pl.when(pl.col("metric").is_in(REWEIGHT_METRICS))
        .then(pl.lit("reweight"))
        .when(in_data_space)
        .then(pl.lit("compare_signals"))
        .when(pl.col("pair").list.len() == 1)
        .then(pl.lit("diagnose"))
        .otherwise(pl.lit("compare"))
    )
    return table.select(engine.alias("engine")).to_series()


def comparison_pairs(table: pl.DataFrame, problem: int) -> tuple[tuple[str, str], ...]:
    """Return the pairs of posteriors that the comparison rows of a problem cover.

    A pair is unordered, so the rows of the reverse Kullback-Leibler direction
    belong to the pair of the forward rows.

    Args:
        table: A result table.
        problem: The problem index.

    Returns:
        The pairs of the ``compare`` and ``compare_signals`` rows in order of
        first appearance, each oriented as its first row.
    """
    rows = _engine_rows(table, problem, _COMPARISON_ENGINES)
    pairs: dict[frozenset[str], tuple[str, str]] = {}
    for pair in rows["pair"].unique(maintain_order=True).to_list():
        pairs.setdefault(frozenset(pair), (pair[0], pair[1]))
    return tuple(pairs.values())


def with_effects(table: pl.DataFrame, levels: Mapping[str, float]) -> pl.DataFrame:
    """Add the engine, the effect size and the verdict columns to a result table.

    The effect of a comparison row is
    :func:`~mixed_signals.sigma.thresholds.verdict` of its metric with the
    threshold of the row, its σ_eff or r_eff, its σ_sig and, for the shape terms,
    the null-subtracted value of
    :func:`~mixed_signals.sigma.maps.null_subtract`. The effect of a diagnostic row
    is its value minus the null mean, or the value itself for a row without a
    null, as :func:`~mixed_signals.diagnostics.flags.evaluate` defines it.

    Args:
        table: A result table.
        levels: The σ_sig that a row of each engine must reach, by engine name;
            :data:`~mixed_signals.sigma.thresholds.SIGNIFICANCE_LEVEL` for an
            engine that is not listed.

    Returns:
        The table with four more columns. ``engine`` is from :func:`row_engines`.
        ``effect`` is the effect size, and ``resolved`` is whether σ_sig reaches
        the level of the engine, which a diagnostic row without a test, such as
        ``ess``, always does. Both are null for a row without a threshold.
        ``extrapolated`` is whether the warnings of the row hold an
        ``ExtrapolationWarning``, for every row.
    """
    engines = row_engines(table)
    effects: list[float | None] = []
    resolved: list[bool | None] = []
    for row, engine in zip(table.iter_rows(named=True), engines, strict=True):
        level = levels.get(engine, SIGNIFICANCE_LEVEL)
        if row["threshold"] is None:
            effects.append(None)
            resolved.append(None)
        elif engine in _COMPARISON_ENGINES:
            outcome = _comparison_verdict(row, level)
            effects.append(outcome.effect)
            resolved.append(outcome.resolved)
        else:
            effect, is_resolved = _diagnostic_effect(row, level)
            effects.append(effect)
            resolved.append(is_resolved)
    extrapolated = pl.col("warnings").list.eval(
        pl.element().struct.field("category") == "ExtrapolationWarning"
    )
    return table.with_columns(
        engines.alias("engine"),
        pl.Series("effect", effects, dtype=pl.Float64),
        pl.Series("resolved", resolved, dtype=pl.Boolean),
        extrapolated.list.any().alias("extrapolated"),
    )


def comparison_table(
    table: pl.DataFrame, problem: int, pair: tuple[str, str], level: float
) -> pl.DataFrame:
    """Return the ``compare`` rows of one pair, in result order.

    Args:
        table: A result table.
        problem: The problem index.
        pair: The two posterior names, in either order.
        level: The σ_sig that a row must reach.

    Returns:
        One row per result row, with ``metric``, ``estimator``, ``direction``
        (``"a‖b"`` in the order of the pair of the row for the Kullback-Leibler
        family, null otherwise), ``parameters`` (joined with ``", "``),
        ``value``, ``null_mean``, ``sigma_eff``, ``r_eff``, ``sigma_sig``,
        ``p_value``, ``tail`` of the p-value, ``threshold``, ``effect``,
        ``resolved``, ``flag``, ``warnings`` (their count) and, for the
        ``sinkhorn`` rows, ``epsilon_factor``, ε over the squared median
        bandwidth, which tells the rows of one metric apart.
    """
    rows = pair_rows(table, problem, pair, ("compare",))
    rows = with_effects(rows, {"compare": level})
    first, second = pl.col("pair").list.get(0), pl.col("pair").list.get(1)
    direction = (
        pl.when(pl.col("metric").is_in(_KL_METRICS))
        .then(pl.concat_str([first, second], separator=_DIRECTION_SEPARATOR))
        .otherwise(None)
    )
    # * Sinkhorn rows differ only in ε/ℓ², which the heatmap needs to tell them apart.
    epsilon_factor = (
        pl.col("estimator_settings")
        .str.json_path_match("$.epsilon_factor")
        .cast(pl.Float64, strict=False)
    )
    return rows.select(
        "metric",
        "estimator",
        direction.alias("direction"),
        pl.col("parameters").list.join(", "),
        "value",
        "null_mean",
        "sigma_eff",
        "r_eff",
        "sigma_sig",
        "p_value",
        pl.col("null_recipe").struct.field("tail").alias("tail"),
        "threshold",
        "effect",
        "resolved",
        "flag",
        pl.col("warnings").list.len().alias("warnings"),
        epsilon_factor.alias("epsilon_factor"),
    )


def diagnostic_table(
    table: pl.DataFrame, problem: int, name: str, level: float
) -> pl.DataFrame:
    """Return the diagnostic rows of one posterior, in result order.

    Args:
        table: A result table.
        problem: The problem index.
        name: The posterior name.
        level: The σ_sig that a row with a null must reach.

    Returns:
        One row per result row, with ``family``, ``metric``, ``estimator``,
        ``parameters`` (joined with ``", "``), ``value``, ``null_mean``,
        ``effect``, ``threshold``, ``sigma_sig``, ``p_value``, ``tail``,
        ``flag`` and ``warnings`` (their count).
    """
    rows = posterior_rows(table, problem, name)
    rows = with_effects(rows, {"diagnose": level})
    return rows.select(
        pl.col("metric")
        .replace_strict(_FAMILY_OF_METRIC, default="other")
        .alias("family"),
        "metric",
        "estimator",
        pl.col("parameters").list.join(", "),
        "value",
        "null_mean",
        "effect",
        "threshold",
        "sigma_sig",
        "p_value",
        pl.col("null_recipe").struct.field("tail").alias("tail"),
        "flag",
        pl.col("warnings").list.len().alias("warnings"),
    )


def chain_health_table(
    table: pl.DataFrame, problem: int, name: str
) -> pl.DataFrame | None:
    """Return the chain health of a posterior with one row per parameter.

    Args:
        table: A result table.
        problem: The problem index.
        name: The posterior name.

    Returns:
        ``None`` without chain rows. Otherwise one row per parameter with
        ``parameter``, ``tau``, ``ess``, ``split_test`` (``"ks"`` or
        ``"kuiper"``), ``split_statistic``, ``split_p_value``, ``rhat`` and
        ``flagged``, whether any of its rows flags.
    """
    rows = posterior_rows(table, problem, name).filter(
        pl.col("metric").is_in(_CHAIN_METRICS)
    )
    if rows.is_empty():
        return None
    health: dict[str, dict[str, Any]] = {}
    for row in rows.iter_rows(named=True):
        parameter = row["parameters"][0]
        entry = health.setdefault(
            parameter, dict.fromkeys(_HEALTH_SCHEMA, None) | {"parameter": parameter}
        )
        if row["metric"] in _SPLIT_METRICS:
            entry["split_test"] = "kuiper" if "kuiper" in row["metric"] else "ks"
            entry["split_statistic"] = row["value"]
            entry["split_p_value"] = row["p_value"]
        else:
            entry[row["metric"]] = row["value"]
        entry["flagged"] = bool(entry["flagged"]) or bool(row["flag"])
    return pl.DataFrame(list(health.values()), schema=_HEALTH_SCHEMA)


_HEALTH_SCHEMA: Final = pl.Schema(
    {
        "parameter": pl.String(),
        "tau": pl.Float64(),
        "ess": pl.Float64(),
        "split_test": pl.String(),
        "split_statistic": pl.Float64(),
        "split_p_value": pl.Float64(),
        "rhat": pl.Float64(),
        "flagged": pl.Boolean(),
    }
)


def rate_table(table: pl.DataFrame, problem: int, name: str) -> pl.DataFrame | None:
    """Return the acceptance and swap rates of a posterior.

    Args:
        table: A result table.
        problem: The problem index.
        name: The posterior name.

    Returns:
        ``None`` without rate rows. Otherwise one row per rate with ``metric``,
        ``temperatures`` (the temperature of an acceptance rate, and the two
        temperatures of a swap rate, read from the settings of the row) and
        ``value``.
    """
    rows = posterior_rows(table, problem, name).filter(
        pl.col("metric").is_in(_RATE_METRICS)
    )
    if rows.is_empty():
        return None
    entries = []
    for row in rows.iter_rows(named=True):
        settings = json.loads(row["estimator_settings"])
        if row["metric"] == "acceptance_rate":
            temperatures = str(settings["temperature"])
        else:
            temperatures = " and ".join(map(str, settings["temperatures"]))
        entries.append(
            {
                "metric": row["metric"],
                "temperatures": temperatures,
                "value": row["value"],
            }
        )
    return pl.DataFrame(
        entries,
        schema={
            "metric": pl.String(),
            "temperatures": pl.String(),
            "value": pl.Float64(),
        },
    )


def dataspace_table(table: pl.DataFrame, problem: int) -> pl.DataFrame | None:
    """Return the data-space comparison of every pair, next to its chart row.

    The ratio of the data value to the chart value shows where the linear order
    of the Fisher chart fails.

    Args:
        table: A result table.
        problem: The problem index.

    Returns:
        ``None`` without data-space rows. Otherwise one row per pair with
        ``pair`` (``"a vs b"``), ``data_space`` (the ``data(...)`` segment),
        ``data_value``, ``chart_value``, ``ratio``, ``data_sigma_eff``,
        ``chart_sigma_eff``, ``data_sigma_sig``, ``chart_sigma_sig``,
        ``data_flag`` and ``chart_flag``; the chart columns are null for a pair
        without a chart row.
    """
    rows = _engine_rows(table, problem, ("compare_signals",))
    if rows.is_empty():
        return None
    last_segment = pl.col("space").str.split(COMPOSITION_SEPARATOR).list.last()
    entries = []
    for pair in comparison_pairs(table, problem):
        rows_of_pair = pair_rows(rows, problem, pair, ("compare_signals",))
        data_rows = rows_of_pair.filter(last_segment != _CHART_SPACE)
        if data_rows.is_empty():
            continue
        chart_rows = rows_of_pair.filter(last_segment == _CHART_SPACE)
        data = data_rows.row(0, named=True)
        chart = chart_rows.row(0, named=True) if chart_rows.height else None
        entries.append(
            {
                "pair": f"{pair[0]} vs {pair[1]}",
                "data_space": data["space"].split(COMPOSITION_SEPARATOR)[-1],
                "data_value": data["value"],
                "chart_value": None if chart is None else chart["value"],
                "ratio": None
                if chart is None
                else _ratio(data["value"], chart["value"]),
                "data_sigma_eff": data["sigma_eff"],
                "chart_sigma_eff": None if chart is None else chart["sigma_eff"],
                "data_sigma_sig": data["sigma_sig"],
                "chart_sigma_sig": None if chart is None else chart["sigma_sig"],
                "data_flag": data["flag"],
                "chart_flag": None if chart is None else chart["flag"],
            }
        )
    return pl.DataFrame(entries, schema=_DATASPACE_SCHEMA)


_DATASPACE_SCHEMA: Final = pl.Schema(
    {
        "pair": pl.String(),
        "data_space": pl.String(),
        "data_value": pl.Float64(),
        "chart_value": pl.Float64(),
        "ratio": pl.Float64(),
        "data_sigma_eff": pl.Float64(),
        "chart_sigma_eff": pl.Float64(),
        "data_sigma_sig": pl.Float64(),
        "chart_sigma_sig": pl.Float64(),
        "data_flag": pl.Boolean(),
        "chart_flag": pl.Boolean(),
    }
)


def reweighting_table(table: pl.DataFrame, problem: int) -> pl.DataFrame | None:
    """Return the reweighting rows with one row per weight set.

    The k̂ verdict is the warning of the ``pareto_k`` row, because the threshold
    of k̂ depends on the number of draws. The Kish size of the new weights is
    informational.

    Args:
        table: A result table.
        problem: The problem index.

    Returns:
        ``None`` without reweighting rows. Otherwise one row per weight set,
        with ``source`` and ``target`` (the pair), ``pareto_k``, ``threshold``
        (of k̂ for the draws of the row), ``flagged`` (an
        ``ImportanceWeightWarning`` among the warnings of the ``pareto_k`` row),
        ``draws``, ``tail_draws``, ``relative_efficiency``, ``importance_ess``,
        ``log_evidence_ratio`` and its ``uncertainty``.

    References:
        :cite:t:`vehtari2024pareto`.
    """
    rows = _engine_rows(table, problem, ("reweight",))
    if rows.is_empty():
        return None
    sets: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows.iter_rows(named=True):
        key = (row["pair"][0], row["pair"][1])
        entry = sets.setdefault(key, {"source": key[0], "target": key[1]})
        if row["metric"] == "pareto_k":
            settings = json.loads(row["estimator_settings"])
            draws = int(settings["draws"])
            entry |= {
                "pareto_k": row["value"],
                "threshold": pareto_k_threshold(draws) if draws >= 2 else math.nan,
                "flagged": any(
                    warning["category"] == "ImportanceWeightWarning"
                    for warning in row["warnings"]
                ),
                "draws": draws,
                "tail_draws": int(settings["tail_draws"]),
                "relative_efficiency": settings["relative_efficiency"],
            }
        elif row["metric"] == "importance_ess":
            entry["importance_ess"] = row["value"]
        else:
            entry["log_evidence_ratio"] = row["value"]
            entry["uncertainty"] = row["uncertainty"]
    return pl.DataFrame(
        [dict.fromkeys(_REWEIGHT_SCHEMA, None) | entry for entry in sets.values()],
        schema=_REWEIGHT_SCHEMA,
    )


_REWEIGHT_SCHEMA: Final = pl.Schema(
    {
        "source": pl.String(),
        "target": pl.String(),
        "pareto_k": pl.Float64(),
        "threshold": pl.Float64(),
        "flagged": pl.Boolean(),
        "draws": pl.Int64(),
        "tail_draws": pl.Int64(),
        "relative_efficiency": pl.Float64(),
        "importance_ess": pl.Float64(),
        "log_evidence_ratio": pl.Float64(),
        "uncertainty": pl.Float64(),
    }
)


def warnings_table(table: pl.DataFrame) -> pl.DataFrame | None:
    """Return every distinct warning of the rows, with how many rows raised it.

    Args:
        table: A result table.

    Returns:
        ``None`` without warnings. Otherwise one row per engine, category and
        message with ``engine``, ``category``, ``message``, ``rows`` (the count)
        and ``metrics`` (sorted, unique, joined with ``", "``), the most frequent
        first.
    """
    entries = (
        table.select(row_engines(table).alias("engine"), "metric", "warnings")
        .explode("warnings")
        .drop_nulls("warnings")
        .unnest("warnings")
    )
    if entries.is_empty():
        return None
    return (
        entries.group_by("engine", "category", "message")
        .agg(
            pl.len().alias("rows"),
            pl.col("metric").unique().sort().str.join(", ").alias("metrics"),
        )
        .sort(
            ["rows", "engine", "category", "message"],
            descending=[True, False, False, False],
        )
    )


def flag_counts(table: pl.DataFrame) -> pl.DataFrame:
    """Return how many rows each engine wrote for each problem, and how many flag.

    Args:
        table: A result table.

    Returns:
        One row per problem and engine, in order of first appearance, with
        ``problem``, ``engine``, ``rows``, ``with_threshold`` (rows that have a
        threshold) and ``flagged``.
    """
    return (
        table.select("problem", row_engines(table).alias("engine"), "threshold", "flag")
        .group_by("problem", "engine", maintain_order=True)
        .agg(
            pl.len().alias("rows"),
            pl.col("threshold").is_not_null().sum().alias("with_threshold"),
            pl.col("flag").fill_null(value=False).sum().alias("flagged"),
        )
    )


def inputs_table(folder: RunFolder) -> pl.DataFrame | None:
    """Return what each posterior was read from and how its samples look.

    Args:
        folder: The run folder.

    Returns:
        ``None`` without ``config.json``. Otherwise one row per posterior of the
        config with ``name``, ``format``, ``path``, ``size_bytes`` and ``sha256``
        (the first 12 hex digits) from ``provenance.json``, ``problems``,
        ``samples`` and ``n_eff`` (joined over the problems) from the sample
        archive, and, for a run of chains, ``burn_in``, ``thin``, ``kept_steps``,
        the largest ``tau`` and ``chain_shape`` from ``reductions.json``.
    """
    if folder.config is None:
        return None
    recorded = {
        record["role"]: record for record in (folder.provenance or {}).get("inputs", [])
    }
    input_samples = folder.samples.get("input")
    entries = []
    for entry in folder.config["posteriors"]:
        name = entry["name"]
        record = recorded.get(f"posterior {name}")
        row: dict[str, Any] = {
            "name": name,
            "format": entry["format"],
            "path": entry["path"],
            "size_bytes": None if record is None else record["size_bytes"],
            "sha256": None if record is None else record["sha256"][:_HASH_DIGITS],
        }
        if input_samples is not None:
            sample_set = input_samples.sets[name]
            sizes = sample_set.effective_sample_size()[:, 0]
            row |= {
                "problems": sample_set.shape.batch,
                "samples": sample_set.shape.sample,
                "n_eff": ", ".join(f"{float(size):.0f}" for size in sizes),
            }
        reduction = folder.reductions.get(name)
        if reduction is not None:
            finite_tau = [value for value in reduction.tau if not math.isnan(value)]
            row |= {
                "burn_in": reduction.burn_in,
                "thin": reduction.thin,
                "kept_steps": reduction.kept_steps,
                "tau": max(finite_tau) if finite_tau else math.nan,
                "chain_shape": " x ".join(map(str, reduction.chain_shape)),
            }
        entries.append(dict.fromkeys(_INPUTS_SCHEMA, None) | row)
    return pl.DataFrame(entries, schema=_INPUTS_SCHEMA)


_INPUTS_SCHEMA: Final = pl.Schema(
    {
        "name": pl.String(),
        "format": pl.String(),
        "path": pl.String(),
        "size_bytes": pl.Int64(),
        "sha256": pl.String(),
        "problems": pl.Int64(),
        "samples": pl.Int64(),
        "n_eff": pl.String(),
        "burn_in": pl.Int64(),
        "thin": pl.Int64(),
        "kept_steps": pl.Int64(),
        "tau": pl.Float64(),
        "chain_shape": pl.String(),
    }
)


def timings_table(folder: RunFolder) -> pl.DataFrame | None:
    """Return the estimated and the measured seconds of each engine.

    Args:
        folder: The run folder.

    Returns:
        ``None`` without the engines of ``provenance.json``. Otherwise one row per
        engine with ``engine``, ``rows``, ``estimated_seconds``, ``wall_seconds``
        and ``ratio`` (wall over estimated), then a row ``total``.
    """
    engines = (folder.provenance or {}).get("engines")
    if not engines:
        return None
    entries = [
        {
            "engine": engine["name"],
            "rows": engine["rows"],
            "estimated_seconds": engine["estimated_seconds"],
            "wall_seconds": engine["wall_seconds"],
        }
        for engine in engines
    ]
    total = {
        "engine": "total",
        "rows": sum(entry["rows"] for entry in entries),
        "estimated_seconds": sum(entry["estimated_seconds"] for entry in entries),
        "wall_seconds": sum(entry["wall_seconds"] for entry in entries),
    }
    return pl.DataFrame(
        [*entries, total],
        schema={
            "engine": pl.String(),
            "rows": pl.Int64(),
            "estimated_seconds": pl.Float64(),
            "wall_seconds": pl.Float64(),
        },
    ).with_columns(
        (pl.col("wall_seconds") / pl.col("estimated_seconds")).alias("ratio")
    )


def cost_table(folder: RunFolder) -> pl.DataFrame | None:
    """Return the estimate of the run, one row per statistic, costliest first.

    Args:
        folder: The run folder.

    Returns:
        ``None`` without ``cost.parquet``. Otherwise its table sorted by
        ``seconds``, descending.
    """
    if folder.cost is None:
        return None
    return folder.cost.sort("seconds", descending=True, nulls_last=True)


def _engine_rows(
    table: pl.DataFrame, problem: int, engines: tuple[str, ...]
) -> pl.DataFrame:
    """Return the rows of a problem that the given engines wrote."""
    mask = row_engines(table).is_in(engines) & (table["problem"] == problem)
    return table.filter(mask)


def pair_rows(
    table: pl.DataFrame,
    problem: int,
    pair: tuple[str, str],
    engines: tuple[str, ...],
) -> pl.DataFrame:
    """Return the rows of a problem and some engines whose pair is the given one.

    Args:
        table: A result table.
        problem: The problem index.
        pair: The two posterior names, in either order.
        engines: The engines whose rows to keep, from :func:`row_engines`; each
            writes rows with two names.

    Returns:
        The rows of both orientations of the pair, in result order.
    """
    rows = _engine_rows(table, problem, engines)
    first, second = pl.col("pair").list.get(0), pl.col("pair").list.get(1)
    forward = (first == pair[0]) & (second == pair[1])
    reverse = (first == pair[1]) & (second == pair[0])
    return rows.filter(forward | reverse)


def posterior_rows(table: pl.DataFrame, problem: int, name: str) -> pl.DataFrame:
    """Return the diagnostic rows of one posterior of a problem.

    Args:
        table: A result table.
        problem: The problem index.
        name: The posterior name.

    Returns:
        The rows of the ``diagnose`` engine whose pair is ``[name]``.
    """
    rows = _engine_rows(table, problem, ("diagnose",))
    return rows.filter(pl.col("pair").list.get(0) == name)


def _comparison_verdict(row: Mapping[str, Any], level: float) -> Verdict:
    """Return the verdict of one comparison row, from the package's own function."""
    metric = row["metric"]
    is_width = metric in R_EFF_METRICS
    draws = row["null_draws"]
    subtracted = None
    if metric in ("kl_shape", "w2_shape", "hellinger_shape"):
        # * Only the shape terms need the sign of the null-subtracted value.
        subtracted = (
            math.nan
            if draws is None
            else float(
                null_subtract(
                    metric, row["value"], jnp.asarray(draws, dtype=jnp.float64)
                )
            )
        )
    return verdict(
        metric,
        sigma_eff=None if is_width else _as_number(row["sigma_eff"]),
        r_eff=_as_number(row["r_eff"]) if is_width else None,
        subtracted=subtracted,
        sigma_sig=_as_number(row["sigma_sig"]),
        thresholds={metric: row["threshold"]},
        level=level,
    )


def _diagnostic_effect(row: Mapping[str, Any], level: float) -> tuple[float, bool]:
    """Return the effect and whether σ_sig reaches the level, as ``evaluate`` does."""
    effect = row["value"]
    if row["null_mean"] is not None:
        effect = row["value"] - row["null_mean"]
    if row["sigma_sig"] is None:
        return effect, True
    return effect, row["sigma_sig"] >= level


def _as_number(value: float | None) -> float:
    """Return a float, with NaN for a missing value."""
    return math.nan if value is None else float(value)


def _ratio(numerator: float, denominator: float) -> float:
    """Return the quotient, which is NaN for 0 over 0 and infinite over zero."""
    if denominator == 0:
        return math.nan if numerator == 0 else math.copysign(math.inf, numerator)
    return numerator / denominator
