"""Tests for the tables of the report: engines, pairs, effects and each summary."""

import json
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import polars as pl
import pytest

from mixed_signals import zoo
from mixed_signals.core.chains import Chains
from mixed_signals.core.result import Result
from mixed_signals.core.samples import SampleSet
from mixed_signals.core.space import ParamSpace
from mixed_signals.diagnostics import flags
from mixed_signals.engines.diagnose import DIAGNOSTIC_FAMILIES, diagnose
from mixed_signals.errors import capture_warnings
from mixed_signals.mcmc.reduce import ReductionRecord
from mixed_signals.report.folder import RunFolder, read_run_folder
from mixed_signals.report.tables import (
    DIAGNOSTIC_FAMILY_METRICS,
    REWEIGHT_METRICS,
    chain_health_table,
    comparison_pairs,
    comparison_table,
    cost_table,
    dataspace_table,
    diagnostic_table,
    flag_counts,
    inputs_table,
    pair_rows,
    posterior_rows,
    rate_table,
    reweighting_table,
    row_engines,
    timings_table,
    warnings_table,
    with_effects,
)
from mixed_signals.reweight.psis import pareto_k_threshold
from mixed_signals.sigma.maps import null_subtract
from mixed_signals.sigma.thresholds import SIGNIFICANCE_LEVEL

LEVELS = {"diagnose": SIGNIFICANCE_LEVEL, "compare": 1.0, "compare_signals": 1.0}
SHAPE_TERMS = ("kl_shape", "w2_shape", "hellinger_shape")
PAIRS = (("mcmc", "table"), ("mcmc", "group"))


@pytest.fixture(scope="module")
def folder(report_run_folder: Path) -> RunFolder:
    """The full run folder, read once."""
    return read_run_folder(report_run_folder)


@pytest.fixture(scope="module")
def table(folder: RunFolder) -> pl.DataFrame:
    """The result table of the full run folder."""
    return folder.result.table


def compare_row(**changes: Any) -> dict[str, Any]:
    """Return the fields of a comparison row of the pair a and b."""
    return {
        "pair": ["a", "b"],
        "metric": "mean_shift",
        "space": "zscore(reference)",
        "sigma_eff": 0.5,
        "sigma_sig": 2.5,
        "threshold": 0.2,
        "flag": True,
        "null_draws": [0.0, 0.1, -0.1],
        "null_mean": 0.0,
        "value": 0.5,
    } | changes


def independent_effect(row: dict[str, Any]) -> float:
    """Return the effect of a comparison row from the definition in the thresholds."""
    metric = row["metric"]
    if metric == "mean_shift":
        return float(abs(row["sigma_eff"]))
    if metric == "width_ratio":
        return abs(math.log(row["r_eff"]))
    if metric in SHAPE_TERMS:
        draws = jnp.asarray(row["null_draws"])
        subtracted = float(null_subtract(metric, row["value"], draws))
        return math.copysign(1.0, subtracted) * abs(math.log(row["r_eff"]))
    return float(row["sigma_eff"])


# * Engines


def test_every_engine_is_told_apart_by_metric_space_and_pair(
    result_of: Callable[..., Result],
) -> None:
    """Reweighting first, then data space, then one name, then anything else."""
    result = result_of(
        {"pair": ["a", "b"], "metric": "pareto_k"},
        {"pair": ["a", "b"], "metric": "importance_ess", "space": "data(white)"},
        {"pair": ["a", "b"], "metric": "log_evidence_ratio", "space": "shift"},
        {"pair": ["a", "b"], "metric": "energy", "space": "data(white)"},
        {"pair": ["a", "b"], "metric": "energy", "space": "shift | data(white)"},
        {"pair": ["a", "b"], "metric": "energy", "space": "data"},
        {"pair": ["a", "b"], "metric": "energy", "space": "chart(fisher)"},
        {"pair": ["a", "b"], "metric": "energy", "space": "shift | chart(fisher)"},
        {"pair": ["a"], "metric": "skewness"},
        {"pair": ["a"], "metric": "tau", "space": "data(white)"},
        {"pair": ["a", "b"], "metric": "kl", "space": "shift | whiten(reference)"},
        {"pair": ["a", "b"], "metric": "kl", "space": "data(white) | whiten(x)"},
    )
    engines = row_engines(result.table)
    assert isinstance(engines, pl.Series)
    assert engines.to_list() == [
        "reweight",
        "reweight",
        "reweight",
        "compare_signals",
        "compare_signals",
        "compare_signals",
        "compare_signals",
        "compare_signals",
        "diagnose",
        "compare_signals",
        "compare",
        "compare",
    ]
    assert set(REWEIGHT_METRICS) == {"pareto_k", "importance_ess", "log_evidence_ratio"}


def test_the_engines_of_a_run_match_the_rows_that_it_recorded(
    folder: RunFolder, table: pl.DataFrame
) -> None:
    """Counting the rows by engine gives the counts of provenance.json."""
    assert folder.provenance is not None
    counts = row_engines(table).value_counts()
    by_engine = dict(zip(counts["engine"], counts["count"], strict=True))
    recorded = {entry["name"]: entry["rows"] for entry in folder.provenance["engines"]}
    assert by_engine == recorded
    assert set(by_engine) == {"diagnose", "compare", "compare_signals"}


def test_every_diagnostic_metric_belongs_to_exactly_one_family() -> None:
    """The families cover what diagnose writes on chains, a circle and a bound."""
    assert list(DIAGNOSTIC_FAMILY_METRICS) == list(DIAGNOSTIC_FAMILIES)
    metrics: set[str] = set()
    with capture_warnings():
        ar1 = zoo.ar1_chains([0.3, 0.2])
        cold = ar1.sample(jax.random.key(0), 400, walker_count=6)
        hot = ar1.sample(jax.random.key(1), 400, walker_count=6)
        tempered = Chains(
            jnp.concatenate([cold.samples, hot.samples], axis=1),
            betas=jnp.tile(jnp.array([1.0, 0.5]), (400, 1)),
            accepted=jnp.full((2, 6), 30),
            proposed=jnp.full((2, 6), 100),
            swaps_accepted=jnp.array([20]),
            swaps_proposed=jnp.array([50]),
        )
        cases: list[tuple[Chains | SampleSet, list[str], ParamSpace]] = [
            (tempered, ["ar1"], ar1.space)
        ]
        circle = zoo.wrapped_gaussians(0.0, width=0.3, param_count=2)
        draws = circle.sample(jax.random.key(2), 300).select(posteriors=0)
        cases.append((draws, ["circle"], circle.space))
        # * Chains on a circle give Kuiper's split test, not the KS test.
        angles = np.asarray(draws.samples[0, 0]).reshape(75, 1, 4, 1, 2)
        cases.append((Chains(angles), ["circle chains"], circle.space))
        for side in ("lower", "upper"):
            railed = zoo.railed_gaussians([0.2], param_count=2, side=side)
            sample_set = railed.sample(jax.random.key(3), 300)
            cases.append((sample_set, ["railed"], railed.space))
        for posterior, names, space in cases:
            result = diagnose(posterior, names, space, key=1, draws=10, cap=100)
            metrics |= set(result.result.table["metric"])
    families = [
        [
            family
            for family, members in DIAGNOSTIC_FAMILY_METRICS.items()
            if metric in members
        ]
        for metric in sorted(metrics)
    ]
    assert all(len(family) == 1 for family in families)
    every_metric = {
        metric for members in DIAGNOSTIC_FAMILY_METRICS.values() for metric in members
    }
    assert metrics == every_metric


# * Pairs and effects


def test_pairs_are_unordered_and_keep_the_orientation_of_their_first_row(
    result_of: Callable[..., Result], table: pl.DataFrame
) -> None:
    """Both directions of the KL rows belong to one pair; diagnose rows to none."""
    assert comparison_pairs(table, 0) == PAIRS
    assert comparison_pairs(table, 7) == ()
    result = result_of(
        {"pair": ["b", "a"], "metric": "kl", "space": "whiten(x)"},
        {"pair": ["a", "b"], "metric": "kl", "space": "whiten(x)"},
        {"pair": ["a", "c"], "metric": "energy", "space": "data(white)"},
        {"pair": ["c", "a"], "metric": "energy", "space": "chart(fisher)"},
        {"pair": ["a"], "metric": "skewness"},
        {"pair": ["a", "b"], "metric": "pareto_k"},
        {"problem": 1, "pair": ["c", "b"], "metric": "kl", "space": "whiten(x)"},
    )
    assert comparison_pairs(result.table, 0) == (("b", "a"), ("a", "c"))
    assert comparison_pairs(result.table, 1) == (("c", "b"),)


def test_the_rows_of_a_pair_hold_both_orientations_in_result_order(
    table: pl.DataFrame,
) -> None:
    """A pair, an engine and a problem choose the rows; the order is the table's."""
    rows = pair_rows(table, 0, ("table", "mcmc"), ("compare",))
    assert set(map(tuple, rows["pair"].to_list())) == {
        ("mcmc", "table"),
        ("table", "mcmc"),
    }
    assert row_engines(rows).unique().to_list() == ["compare"]
    both = pair_rows(table, 0, ("mcmc", "table"), ("compare", "compare_signals"))
    assert set(row_engines(both)) == {"compare", "compare_signals"}
    assert pair_rows(table, 1, ("mcmc", "table"), ("compare",)).is_empty()
    assert posterior_rows(table, 0, "table")["pair"].list.get(0).unique().to_list() == [
        "table"
    ]
    assert posterior_rows(table, 0, "nobody").is_empty()


def test_the_effects_are_the_verdicts_that_set_the_flags(table: pl.DataFrame) -> None:
    """For every comparison row: the effect follows its definition, and so does flag."""
    effects = with_effects(table, LEVELS)
    assert effects.columns[-4:] == ["engine", "effect", "resolved", "extrapolated"]
    assert effects.height == table.height
    rows = effects.filter(pl.col("engine").is_in(["compare", "compare_signals"]))
    assert rows.height > 0
    expected = [independent_effect(row) for row in rows.iter_rows(named=True)]
    np.testing.assert_allclose(rows["effect"].to_numpy(), expected)
    for row in rows.iter_rows(named=True):
        level = LEVELS[row["engine"]]
        assert row["resolved"] == (row["sigma_sig"] >= level)
        assert row["flag"] == (row["resolved"] and row["effect"] >= row["threshold"])
    assert rows["flag"].any(), "the fixture must hold a flagged comparison row"
    assert not rows["flag"].all()


def test_the_effects_of_diagnostics_are_those_of_evaluate(table: pl.DataFrame) -> None:
    """Value minus the null mean, or the value alone, with the flag of the engine."""
    effects = with_effects(table, LEVELS)
    rows = effects.filter(
        (pl.col("engine") == "diagnose") & pl.col("threshold").is_not_null()
    )
    assert rows.height > 0
    thresholds = {
        metric: flags.THRESHOLDS[metric]._replace(value=threshold)
        for metric, threshold in rows.select("metric", "threshold").unique().iter_rows()
    }
    for row in rows.iter_rows(named=True):
        draws = None if row["null_draws"] is None else jnp.asarray(row["null_draws"])
        with capture_warnings():
            evaluation = flags.evaluate(
                row["metric"],
                row["value"],
                draws,
                p_value=row["p_value"],
                thresholds=thresholds,
                level=SIGNIFICANCE_LEVEL,
            )
        assert row["effect"] == pytest.approx(evaluation.effect, rel=1e-9, abs=1e-12)
        assert row["flag"] == evaluation.flagged
        if row["sigma_sig"] is not None:
            assert row["resolved"] == (row["sigma_sig"] >= SIGNIFICANCE_LEVEL)
    assert rows["flag"].any(), "the fixture must hold a flagged diagnostic row"


def test_rows_without_a_threshold_get_nulls_and_the_rest_follow_their_definition(
    result_of: Callable[..., Result],
) -> None:
    """Informational rows get no effect; a missing null leaves the value as effect."""
    warning = {"category": "ExtrapolationWarning", "message": "tail"}
    result = result_of(
        {"metric": "tau", "value": 12.0},
        {"metric": "ess", "value": 30.0, "threshold": 400.0, "flag": True},
        {
            "metric": "skewness",
            "value": 0.9,
            "threshold": 0.5,
            "null_mean": 0.1,
            "null_draws": [0.0, 0.2],
            "sigma_sig": 1.0,
            "flag": False,
            "warnings": [warning],
        },
        {
            "metric": "skewness",
            "value": 0.9,
            "threshold": 0.5,
            "null_mean": 0.1,
            "null_draws": [0.0, 0.2],
            "sigma_sig": 3.0,
            "flag": True,
        },
    )
    effects = with_effects(result.table, {"diagnose": 2.0})
    assert effects["effect"].to_list() == [
        None,
        30.0,
        pytest.approx(0.8),
        pytest.approx(0.8),
    ]
    assert effects["resolved"].to_list() == [None, True, False, True]
    assert effects["extrapolated"].to_list() == [False, False, True, False]
    default = with_effects(result.table, {})
    assert default["resolved"].to_list() == effects["resolved"].to_list()


def test_a_comparison_row_without_a_value_has_a_nan_effect(
    result_of: Callable[..., Result],
) -> None:
    """The rows that no null ran for never resolve and never fail."""
    nan = math.nan
    result = result_of(
        compare_row(
            metric="mean_shift",
            value=nan,
            sigma_eff=nan,
            sigma_sig=nan,
            null_draws=None,
        ),
        compare_row(
            metric="width_ratio",
            value=nan,
            r_eff=nan,
            sigma_eff=None,
            sigma_sig=nan,
            null_draws=None,
        ),
        compare_row(
            metric="kl_shape",
            value=nan,
            r_eff=nan,
            sigma_eff=None,
            sigma_sig=nan,
            null_draws=None,
        ),
        compare_row(
            metric="kl_shape",
            value=0.3,
            r_eff=1.2,
            sigma_eff=None,
            null_draws=[0.1, 0.2],
        ),
        compare_row(metric="mean_shift", sigma_eff=None, sigma_sig=None),
    )
    effects = with_effects(result.table, {"compare": 2.0})
    values = effects["effect"].to_list()
    assert all(math.isnan(value) for value in values[:3])
    assert values[3] == pytest.approx(math.log(1.2))
    assert math.isnan(values[4])
    assert effects["resolved"].to_list() == [False, False, False, True, False]


# * The comparison table


def test_the_comparison_table_lists_the_compare_rows_of_a_pair(
    table: pl.DataFrame,
) -> None:
    """Columns, order, directions, parameters, tails and the factor of Sinkhorn."""
    pair = ("mcmc", "table")
    rows = comparison_table(table, 0, pair, 1.0)
    expected_rows = pair_rows(table, 0, pair, ("compare",))
    assert rows.height == expected_rows.height
    assert rows.columns == [
        "metric",
        "estimator",
        "direction",
        "parameters",
        "value",
        "null_mean",
        "sigma_eff",
        "r_eff",
        "sigma_sig",
        "p_value",
        "tail",
        "threshold",
        "effect",
        "resolved",
        "flag",
        "warnings",
        "epsilon_factor",
    ]
    assert rows["metric"].to_list() == expected_rows["metric"].to_list()
    assert rows["value"].to_list() == expected_rows["value"].to_list()
    kl = rows.filter(pl.col("metric").is_in(["kl", "kl_location", "kl_shape"]))
    assert set(kl["direction"]) == {"mcmc‖table", "table‖mcmc"}
    assert (
        rows.filter(~pl.col("metric").is_in(["kl", "kl_location", "kl_shape"]))[
            "direction"
        ]
        .is_null()
        .all()
    )
    assert "x0, x1, x2" in set(rows["parameters"])
    assert "x1" in set(rows["parameters"])
    assert set(rows.filter(pl.col("metric") == "mean_shift")["tail"]) == {"two_sided"}
    assert set(rows.filter(pl.col("metric") == "ks")["tail"]) == {"upper"}
    sinkhorn = rows.filter(pl.col("metric") == "sinkhorn")
    assert sinkhorn["epsilon_factor"].to_list() == [0.25, 1.0]
    assert rows.filter(pl.col("metric") != "sinkhorn")["epsilon_factor"].is_null().all()
    assert (rows["warnings"] >= 0).all()
    assert comparison_table(table, 0, ("table", "mcmc"), 1.0).height == rows.height
    assert comparison_table(table, 0, ("table", "group"), 1.0).is_empty()


def test_a_level_changes_resolved_and_leaves_the_flag_of_the_engine(
    table: pl.DataFrame,
) -> None:
    """The table recomputes resolved at the level that it is given."""
    pair = ("mcmc", "table")
    high = comparison_table(table, 0, pair, 99.0)
    low = comparison_table(table, 0, pair, -math.inf)
    assert not high["resolved"].any()
    assert low["resolved"].all()
    assert high["flag"].to_list() == low["flag"].to_list()


# * The diagnostic tables


def test_the_diagnostic_table_gives_every_row_of_a_posterior_with_its_family(
    table: pl.DataFrame,
) -> None:
    """Result order, the family of each metric, and the effect of each row."""
    rows = diagnostic_table(table, 0, "mcmc", 2.0)
    expected = posterior_rows(table, 0, "mcmc")
    assert rows.height == expected.height
    assert rows["metric"].to_list() == expected["metric"].to_list()
    assert rows["family"].to_list() == [
        next(
            family
            for family, members in DIAGNOSTIC_FAMILY_METRICS.items()
            if metric in members
        )
        for metric in rows["metric"]
    ]
    assert list(dict.fromkeys(rows["family"])) == [
        "chains",
        "shape",
        "modes",
        "railing",
    ]
    assert rows.columns == [
        "family",
        "metric",
        "estimator",
        "parameters",
        "value",
        "null_mean",
        "effect",
        "threshold",
        "sigma_sig",
        "p_value",
        "tail",
        "flag",
        "warnings",
    ]
    assert (
        rows.filter(pl.col("flag").fill_null(value=False))["metric"].to_list()
        == ["ess"] * 3
    )
    assert diagnostic_table(table, 0, "nobody", 2.0).is_empty()


def test_an_unknown_metric_is_in_the_family_other(
    result_of: Callable[..., Result],
) -> None:
    """A metric that the families do not list still shows."""
    rows = diagnostic_table(result_of({"metric": "novelty"}).table, 0, "a", 2.0)
    assert rows["family"].to_list() == ["other"]


def test_chain_health_has_one_row_per_parameter(
    table: pl.DataFrame, result_of: Callable[..., Result]
) -> None:
    """The statistics of a run of chains, with the split test and a flag."""
    health = chain_health_table(table, 0, "mcmc")
    assert health is not None
    assert health.columns == [
        "parameter",
        "tau",
        "ess",
        "split_test",
        "split_statistic",
        "split_p_value",
        "rhat",
        "flagged",
    ]
    assert health["parameter"].to_list() == ["x0", "x1", "x2"]
    assert set(health["split_test"]) == {"ks"}
    assert health["flagged"].all()
    ess = posterior_rows(table, 0, "mcmc").filter(pl.col("metric") == "ess")
    assert health["ess"].to_list() == ess["value"].to_list()
    assert chain_health_table(table, 0, "table") is None
    rows = [
        {"metric": "tau", "value": 5.0},
        {"metric": "ess", "value": 900.0, "threshold": 400.0, "flag": False},
        {"metric": "split_chain_kuiper_ks", "value": 0.05, "p_value": 0.4},
        {"metric": "rhat", "value": 1.01},
    ]
    clean = chain_health_table(result_of(*rows).table, 0, "a")
    assert clean is not None
    assert clean["split_test"].to_list() == ["kuiper"]
    assert clean["split_p_value"].to_list() == [0.4]
    assert clean["flagged"].to_list() == [False]


def test_the_rate_table_reads_the_temperatures_from_the_settings(
    table: pl.DataFrame, result_of: Callable[..., Result]
) -> None:
    """One row per acceptance rate and swap rate; none without rate rows."""
    assert rate_table(table, 0, "mcmc") is None
    result = result_of(
        {
            "metric": "acceptance_rate",
            "estimator_settings": '{"burn_in":0,"temperature":0}',
            "value": 0.3,
        },
        {
            "metric": "acceptance_rate",
            "estimator_settings": '{"burn_in":0,"temperature":1}',
            "value": 0.2,
        },
        {
            "metric": "swap_rate",
            "estimator_settings": '{"burn_in":0,"temperatures":[0,1]}',
            "value": 0.1,
        },
    )
    rates = rate_table(result.table, 0, "a")
    assert rates is not None
    assert rates.to_dicts() == [
        {"metric": "acceptance_rate", "temperatures": "0", "value": 0.3},
        {"metric": "acceptance_rate", "temperatures": "1", "value": 0.2},
        {"metric": "swap_rate", "temperatures": "0 and 1", "value": 0.1},
    ]


# * Data space and reweighting


def test_the_dataspace_table_sets_the_data_row_beside_the_chart_row(
    table: pl.DataFrame,
) -> None:
    """One row per pair, with the ratio of data over chart."""
    rows = dataspace_table(table, 0)
    assert rows is not None
    assert rows["pair"].to_list() == ["mcmc vs table", "mcmc vs group"]
    assert set(rows["data_space"]) == {"data(white)"}
    signals = table.filter(row_engines(table) == "compare_signals")
    data = signals.filter(pl.col("space") == "data(white)")
    chart = signals.filter(pl.col("space") == "chart(fisher)")
    assert rows["data_value"].to_list() == data["value"].to_list()
    assert rows["chart_value"].to_list() == chart["value"].to_list()
    expected = np.asarray(data["value"]) / np.asarray(chart["value"])
    np.testing.assert_allclose(rows["ratio"].to_numpy(), expected)
    assert rows["data_flag"].to_list() == data["flag"].to_list()
    assert rows["chart_sigma_sig"].to_list() == chart["sigma_sig"].to_list()
    assert (
        dataspace_table(table.filter(row_engines(table) != "compare_signals"), 0)
        is None
    )
    assert dataspace_table(table, 3) is None


def test_a_pair_without_a_chart_row_leaves_the_chart_columns_null(
    result_of: Callable[..., Result],
) -> None:
    """The ratio is null, infinite or NaN where the chart value is null or zero."""
    base = compare_row(metric="energy", space="shift | data(white)")
    result = result_of(
        {**base, "pair": ["a", "b"], "value": 2.0},
        {**base, "pair": ["a", "c"], "value": 2.0},
        {**base, "pair": ["a", "c"], "value": 0.0, "space": "chart(fisher)"},
        {**base, "pair": ["a", "d"], "value": -2.0},
        {**base, "pair": ["a", "d"], "value": 0.0, "space": "chart(fisher)"},
        {**base, "pair": ["a", "e"], "value": 0.0},
        {**base, "pair": ["a", "e"], "value": 0.0, "space": "chart(fisher)"},
        {**base, "pair": ["a", "f"], "value": 3.0, "space": "chart(fisher)"},
    )
    rows = dataspace_table(result.table, 0)
    assert rows is not None
    assert rows["pair"].to_list() == ["a vs b", "a vs c", "a vs d", "a vs e"]
    assert rows["data_space"].to_list() == ["data(white)"] * 4
    assert rows["chart_value"].to_list()[0] is None
    assert rows["chart_flag"].to_list()[0] is None
    ratios = rows["ratio"].to_list()
    assert ratios[0] is None
    assert ratios[1] == math.inf
    assert ratios[2] == -math.inf
    assert math.isnan(ratios[3])


def test_the_reweighting_table_has_the_threshold_and_the_warning_of_each_set(
    reweighting_run_folder: Path,
) -> None:
    """One row per weight set; the verdict is the warning, not a comparison."""
    folder = read_run_folder(reweighting_run_folder)
    rows = reweighting_table(folder.result.table, 0)
    assert rows is not None
    assert rows["source"].to_list() == ["reference", "shifted"]
    assert rows["target"].to_list() == ["reference:reweighted", "shifted:reweighted"]
    assert rows["draws"].to_list() == [2_000, 2_000]
    np.testing.assert_allclose(rows["threshold"].to_numpy(), pareto_k_threshold(2_000))
    source = folder.result.table
    pareto = source.filter(pl.col("metric") == "pareto_k")
    assert rows["pareto_k"].to_list() == pareto["value"].to_list()
    warned = [
        any(entry["category"] == "ImportanceWeightWarning" for entry in warnings)
        for warnings in pareto["warnings"]
    ]
    assert rows["flagged"].to_list() == warned
    assert any(warned), "the fixture must hold a flagged weight set"
    assert not all(warned), "the fixture must hold a weight set that does not flag"
    ess = source.filter(pl.col("metric") == "importance_ess")
    assert rows["importance_ess"].to_list() == ess["value"].to_list()
    evidence = source.filter(pl.col("metric") == "log_evidence_ratio")
    assert rows["log_evidence_ratio"].to_list() == evidence["value"].to_list()
    assert rows["uncertainty"].to_list() == evidence["uncertainty"].to_list()
    settings = [json.loads(text) for text in pareto["estimator_settings"]]
    assert rows["tail_draws"].to_list() == [entry["tail_draws"] for entry in settings]
    assert rows["relative_efficiency"].to_list() == [
        entry["relative_efficiency"] for entry in settings
    ]
    assert reweighting_table(source.filter(pl.col("metric") == "none"), 0) is None
    assert reweighting_table(source, 5) is None


def test_a_weight_set_of_one_draw_has_no_threshold(
    result_of: Callable[..., Result],
) -> None:
    """The threshold needs two draws; with one the column holds NaN."""
    settings = json.dumps(
        {"draws": 1, "tail_draws": 0, "shape_prior": True, "relative_efficiency": 1.0}
    )
    result = result_of(
        {"pair": ["a", "a:r"], "metric": "pareto_k", "estimator_settings": settings}
    )
    rows = reweighting_table(result.table, 0)
    assert rows is not None
    assert math.isnan(rows["threshold"][0])
    assert rows["importance_ess"].to_list() == [None]


# * Warnings, flags, inputs and costs


def test_the_warnings_table_counts_the_rows_of_each_distinct_warning(
    table: pl.DataFrame, result_of: Callable[..., Result]
) -> None:
    """Engine, category and message, with the rows and metrics, the most first."""
    warnings = warnings_table(table)
    assert warnings is not None
    assert warnings.columns == ["engine", "category", "message", "rows", "metrics"]
    expected: dict[tuple[str, str, str], list[str]] = {}
    engines = row_engines(table)
    for engine, metric, entries in zip(
        engines, table["metric"], table["warnings"], strict=True
    ):
        for entry in entries:
            key = (engine, entry["category"], entry["message"])
            expected.setdefault(key, []).append(metric)
    assert warnings.height == len(expected)
    for row in warnings.iter_rows(named=True):
        metrics = expected[(row["engine"], row["category"], row["message"])]
        assert row["rows"] == len(metrics)
        assert row["metrics"] == ", ".join(sorted(set(metrics)))
    assert warnings["rows"].to_list() == sorted(
        warnings["rows"].to_list(), reverse=True
    )
    assert warnings_table(result_of({"metric": "skewness"}).table) is None


def test_flag_counts_group_by_problem_and_engine(
    table: pl.DataFrame, batch_run_folder: Path
) -> None:
    """Rows, rows with a threshold and flagged rows, in order of first appearance."""
    counts = flag_counts(table)
    assert counts.columns == ["problem", "engine", "rows", "with_threshold", "flagged"]
    assert counts["engine"].to_list() == ["diagnose", "compare", "compare_signals"]
    engines = row_engines(table)
    for row in counts.iter_rows(named=True):
        mine = table.filter(engines == row["engine"])
        assert row["rows"] == mine.height
        assert row["with_threshold"] == mine["threshold"].is_not_null().sum()
        assert row["flagged"] == mine["flag"].fill_null(value=False).sum()
    batch = flag_counts(read_run_folder(batch_run_folder).result.table)
    assert batch["problem"].to_list() == [0, 1, 2]
    assert set(batch["engine"]) == {"compare"}


def test_the_inputs_table_joins_the_config_the_provenance_and_the_reductions(
    folder: RunFolder,
) -> None:
    """One row per posterior of the config, with the reduction of a run of chains."""
    inputs = inputs_table(folder)
    assert inputs is not None
    assert inputs["name"].to_list() == ["mcmc", "table", "group"]
    assert inputs["format"].to_list() == ["eryn", "table", "hdf5"]
    assert inputs["path"].to_list() == [
        "data/mcmc.h5",
        "data/table.parquet",
        "data/peer.h5",
    ]
    assert folder.provenance is not None
    records = {record["role"]: record for record in folder.provenance["inputs"]}
    for row in inputs.iter_rows(named=True):
        record = records[f"posterior {row['name']}"]
        assert row["size_bytes"] == record["size_bytes"]
        assert row["sha256"] == record["sha256"][:12]
    assert inputs["problems"].to_list() == [1, 1, 1]
    assert inputs["samples"].to_list() == [200, 300, 300]
    assert inputs["n_eff"].to_list() == ["50", "300", "300"]
    record = folder.reductions["mcmc"]
    first = inputs.row(0, named=True)
    assert (first["burn_in"], first["thin"], first["kept_steps"]) == (
        20,
        4,
        record.kept_steps,
    )
    assert first["tau"] == max(record.tau)
    assert first["chain_shape"] == " x ".join(map(str, record.chain_shape))
    assert inputs.row(1, named=True)["burn_in"] is None
    assert inputs.row(2, named=True)["chain_shape"] is None


def test_the_inputs_table_needs_only_the_config(folder: RunFolder) -> None:
    """Without provenance, samples or reductions the columns are null."""
    assert inputs_table(folder._replace(config=None)) is None
    bare = inputs_table(folder._replace(provenance=None, samples={}, reductions={}))
    assert bare is not None
    assert bare["sha256"].is_null().all()
    assert bare["size_bytes"].is_null().all()
    assert bare["samples"].is_null().all()
    assert bare["burn_in"].is_null().all()
    still = inputs_table(folder._replace(provenance={"seed": 1}))
    assert still is not None
    assert still["sha256"].is_null().all()


def test_a_chain_that_never_moves_has_a_nan_tau_in_the_inputs(
    folder: RunFolder,
) -> None:
    """The largest tau skips NaN, and is NaN when every tau is."""
    record = folder.reductions["mcmc"]
    frozen = ReductionRecord.model_validate(
        record.model_dump() | {"tau": (math.nan,) * len(record.tau)}
    )
    inputs = inputs_table(folder._replace(reductions={"mcmc": frozen}))
    assert inputs is not None
    assert math.isnan(inputs["tau"][0])
    mixed = ReductionRecord.model_validate(
        record.model_dump() | {"tau": (math.nan, 3.0, 2.0)}
    )
    inputs = inputs_table(folder._replace(reductions={"mcmc": mixed}))
    assert inputs is not None
    assert inputs["tau"][0] == 3.0


def test_the_timings_table_sums_the_engines_and_compares_with_the_estimate(
    folder: RunFolder,
) -> None:
    """One row per engine and a total, with wall over estimated seconds."""
    timings = timings_table(folder)
    assert timings is not None
    assert timings["engine"].to_list() == [
        "diagnose",
        "compare",
        "compare_signals",
        "total",
    ]
    assert timings.columns == [
        "engine",
        "rows",
        "estimated_seconds",
        "wall_seconds",
        "ratio",
    ]
    parts = timings.head(3)
    total = timings.row(3, named=True)
    assert total["rows"] == parts["rows"].sum()
    assert total["wall_seconds"] == pytest.approx(parts["wall_seconds"].sum())
    assert total["estimated_seconds"] == pytest.approx(parts["estimated_seconds"].sum())
    np.testing.assert_allclose(
        timings["ratio"].to_numpy(),
        timings["wall_seconds"].to_numpy() / timings["estimated_seconds"].to_numpy(),
    )
    assert timings_table(folder._replace(provenance=None)) is None
    assert timings_table(folder._replace(provenance={"engines": []})) is None


def test_the_cost_table_lists_the_costliest_statistic_first(folder: RunFolder) -> None:
    """The estimate of the run, sorted by seconds, descending."""
    costs = cost_table(folder)
    assert costs is not None
    assert folder.cost is not None
    assert costs.height == folder.cost.height
    seconds = [value for value in costs["seconds"].to_list() if value is not None]
    assert seconds == sorted(seconds, reverse=True)
    # * A statistic without an estimate sorts last.
    nulls = costs["seconds"].null_count()
    assert costs["seconds"].tail(nulls).null_count() == nulls
    assert cost_table(folder._replace(cost=None)) is None
