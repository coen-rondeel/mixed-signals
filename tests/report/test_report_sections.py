"""Tests for the sections of the report: their sources, headings, tabs and captions."""

import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import jax
import marimo as mo
import polars as pl
import pytest
from matplotlib.patches import Rectangle

import mixed_signals
from mixed_signals import zoo
from mixed_signals.core.result import Result
from mixed_signals.core.space import ParamSpace
from mixed_signals.engines import compare
from mixed_signals.report import sections
from mixed_signals.report._style import SERIES_COLOURS
from mixed_signals.report.corner import overview_factor, silverman_factor
from mixed_signals.report.figures import effect_heatmap
from mixed_signals.report.folder import (
    PosteriorSamples,
    ReportSettings,
    RunFolder,
    read_run_folder,
)
from mixed_signals.report.sections import (
    batch_section,
    comparison_section,
    config_section,
    cost_section,
    diagnostics_section,
    inputs_section,
    overview_section,
    problem_sections,
    reweighting_section,
    warnings_section,
)
from mixed_signals.report.tables import comparison_table, row_engines

TABS = "marimo-tabs"


@pytest.fixture(scope="module")
def full(report_run_folder: Path) -> RunFolder:
    """The full run folder, read once."""
    return read_run_folder(report_run_folder)


@pytest.fixture(scope="module")
def diagnostics_html(full: RunFolder) -> str:
    """The diagnostics of problem 0 of the full run, built once."""
    section = diagnostics_section(full, 0)
    assert section is not None
    return section.text


@pytest.fixture(scope="module")
def comparison_html(full: RunFolder) -> str:
    """The comparisons of problem 0 of the full run, built once."""
    section = comparison_section(full, 0)
    assert section is not None
    return section.text


def keep(folder: RunFolder, rows: pl.Expr | pl.Series) -> RunFolder:
    """Return the folder with only the rows of the expression, the samples kept."""
    return folder._replace(result=Result(folder.result.table.filter(rows)))


def settings_of(folder: RunFolder, **changes: Any) -> RunFolder:
    """Return the folder with other settings of the report."""
    settings = ReportSettings(**{**folder.settings.model_dump(), **changes})
    return folder._replace(settings=settings)


def text_of(section: mo.Html | None) -> str:
    """Return the HTML of a section that must exist."""
    assert section is not None
    return section.text


@pytest.fixture(scope="module")
def four_posteriors(tmp_path_factory: pytest.TempPathFactory) -> RunFolder:
    """A folder of a result that compares four posteriors, without samples."""
    case = zoo.gaussians(
        [[0.0, 0.0], [0.3, 0.0], [0.0, 0.3], [0.3, 0.3]],
        [[[1.0, 0.0], [0.0, 1.0]]] * 4,
        names=["p0", "p1", "p2", "p3"],
    )
    draws = case.sample(jax.random.key(6), 300)
    result = compare(
        draws,
        case.names,
        case.space,
        key=1,
        reference="p0",
        metrics=["mean_shift"],
        draws=20,
        cap=100,
    )
    folder = tmp_path_factory.mktemp("four")
    result.write_parquet(folder / "results.parquet")
    return read_run_folder(folder)


# * The overview and the batch


def test_the_overview_defines_the_terms_and_gives_the_facts(full: RunFolder) -> None:
    """The title, σ_eff and the flag, the run facts and the counts of every engine."""
    html = text_of(overview_section(full))
    assert "mixed signals report" in html
    for term in ("σ_eff", "r_eff", "σ_sig", "threshold", "flags"):
        assert term in html
    assert full.config_hash is not None
    assert full.config_hash in html
    assert str(full.path) in html
    assert "Seed: 11" in html
    assert f"Report built with mixed-signals: {mixed_signals.__version__}" in html
    assert full.provenance is not None
    assert full.provenance["started"] in html
    for engine in ("diagnose", "compare", "compare_signals"):
        assert engine in html
    assert "not recorded" not in html
    # * σ_sig is Phi^-1(1 - p) for an upper tail and Phi^-1(1 - p/2) for two sides.
    assert "Φ⁻¹(1 − p) for an upper tail and Φ⁻¹(1 − p/2) for a two-sided one" in html
    assert "names the tail of each row" in html
    assert "one-sided" not in html


def test_the_facts_come_from_the_rows_without_provenance(full: RunFolder) -> None:
    """The package, seed, JAX and device come from the rows; the rest is missing."""
    html = text_of(overview_section(full._replace(provenance=None)))
    assert "Seed: 11" in html
    assert f"Package: {mixed_signals.__version__}" in html
    assert f"JAX: {jax.__version__}" in html
    assert "Started: not recorded" in html
    assert "Wall time: not recorded" in html
    assert "Machine: not recorded" in html
    bare = full._replace(provenance=None, config_hash=None, result=Result.empty())
    assert "Seed: not recorded" in text_of(overview_section(bare))


def test_a_result_of_one_problem_has_no_batch_section(full: RunFolder) -> None:
    """The table of flag counts is for B above one."""
    assert batch_section(full) is None


def test_a_batch_names_the_problems_shown_and_the_rule(batch_run_folder: Path) -> None:
    """The counts of every problem, and the rule of each way to choose."""
    folder = read_run_folder(batch_run_folder)
    html = text_of(batch_section(folder))
    assert "3 problems" in html
    assert "those with the most flagged rows, ties by index" in html
    assert "compare" in html
    everything = text_of(batch_section(settings_of(folder, problems="all")))
    assert "every problem" in everything
    assert "0, 1, 2" in everything
    listed = text_of(batch_section(settings_of(folder, problems=(2, 0))))
    assert "the problems listed in report.json, in its order" in listed
    assert "shown in full below are 2, 0" in listed


# * The problems


def test_one_problem_gives_the_stack_and_several_give_tabs(
    full: RunFolder, batch_run_folder: Path
) -> None:
    """Problem tabs appear only when several problems are shown."""
    single = text_of(problem_sections(settings_of(full, problems=1)))
    assert "Problem 0" not in single
    assert "Diagnostics" in single
    assert "Comparisons" in single
    assert "Reweighting" not in single
    batch = read_run_folder(batch_run_folder)
    several = text_of(problem_sections(batch))
    assert TABS in several
    for problem in range(3):
        assert f"Problem {problem}" in several
    assert "Diagnostics" not in several
    only = text_of(problem_sections(settings_of(batch, problems=(1,))))
    assert "Problem 1" not in only
    assert "Comparisons" in only


def test_a_problem_without_sections_is_an_empty_stack(full: RunFolder) -> None:
    """A listed problem with no rows shows nothing, and does not fail."""
    nothing = full._replace(
        result=Result(full.result.table.filter(pl.col("problem") == 9))
    )
    assert isinstance(problem_sections(nothing), mo.Html)


# * The diagnostics


def test_the_diagnostics_have_a_tab_per_posterior(
    full: RunFolder, diagnostics_html: str
) -> None:
    """Heading, names, the corner plot and its caption, tables and accordions."""
    assert "Diagnostics" in diagnostics_html
    assert TABS in diagnostics_html
    for name in ("mcmc", "table", "group"):
        assert name in diagnostics_html
    assert "of the" in diagnostics_html
    for part in (
        "All diagnostic rows",
        "Null distributions",
        "Flagged rows",
        "split_test",
    ):
        assert part in diagnostics_html
    assert "data:image/png" in diagnostics_html
    mcmc = full.samples["input"].sets["mcmc"].select(problems=0)
    factor = silverman_factor(mcmc.log_weights[0, 0])
    assert f"bandwidth factor {factor:.4g}" in diagnostics_html
    assert "which its mode and railing rows use" in diagnostics_html
    assert "Thin grey lines mark the finite bounds" in diagnostics_html
    assert "68% and 95%" in diagnostics_html
    assert "Gaussian surrogates with the same mean and covariance" in diagnostics_html
    assert "same width" not in diagnostics_html
    assert "<strong>Flagged rows</strong>" in diagnostics_html
    assert "<strong>Chain health</strong>" in diagnostics_html
    assert "Acceptance and swap rates" not in diagnostics_html


def test_the_rate_table_has_its_own_label(
    full: RunFolder, result_of: Callable[..., Result]
) -> None:
    """Rates of a tempered run come after the chain health, under their label."""
    rates = result_of(
        {
            "pair": ["mcmc"],
            "metric": "acceptance_rate",
            "estimator_settings": '{"burn_in":0,"temperature":0}',
            "value": 0.3,
        }
    )
    table = pl.concat([full.result.table, rates.table])
    html = text_of(diagnostics_section(full._replace(result=Result(table)), 0))
    assert "<strong>Acceptance and swap rates</strong>" in html
    assert html.index("<strong>Chain health</strong>") < html.index(
        "<strong>Acceptance and swap rates</strong>"
    )


def test_a_single_posterior_has_no_tabs_and_names_its_flagged_rows(
    full: RunFolder,
) -> None:
    """The count of flagged rows, and no tab bar."""
    chains = keep(full, pl.col("pair").list.get(0, null_on_oob=True) == "mcmc")
    mcmc_rows = chains.result.table.filter(pl.col("pair").list.len() == 1)
    one = chains._replace(result=Result(mcmc_rows))
    html = text_of(diagnostics_section(one, 0))
    assert TABS not in html
    assert '<h3 id="mcmc">mcmc</h3>' in html
    flagged = int(mcmc_rows["flag"].fill_null(value=False).sum())
    tested = int(mcmc_rows["threshold"].is_not_null().sum())
    assert f"{flagged} of the {tested} rows of mcmc that have a threshold flag" in html


def test_the_diagnostics_need_diagnostic_rows(
    full: RunFolder, batch_run_folder: Path
) -> None:
    """None for a problem without them, and for a result of comparisons."""
    assert diagnostics_section(full, 5) is None
    assert diagnostics_section(read_run_folder(batch_run_folder), 0) is None


def test_without_samples_the_diagnostics_say_why_there_is_no_plot(
    full: RunFolder,
) -> None:
    """The tables stay; the corner plot gives way to a sentence."""
    html = text_of(diagnostics_section(full._replace(samples={}), 0))
    assert "The samples are not in the run folder, so there is no corner plot." in html
    assert "All diagnostic rows" in html
    assert "bandwidth factor" not in html
    partial = full._replace(samples={"input": full.samples["input"]._replace(sets={})})
    assert "no corner plot" in text_of(diagnostics_section(partial, 0))


def test_the_plot_cap_is_named_when_it_acts(full: RunFolder) -> None:
    """A cap below the number of samples draws a subsample and says so."""
    capped = text_of(diagnostics_section(settings_of(full, plot_cap=50), 0))
    assert "uniform subsample of 50 samples" in capped
    assert "plot_cap in report.json" in capped
    loose = text_of(diagnostics_section(settings_of(full, plot_cap=10_000), 0))
    assert "uniform subsample" not in loose


def test_without_null_draws_there_is_no_null_accordion(full: RunFolder) -> None:
    """The accordion needs a row that stores finite draws."""
    table = full.result.table.with_columns(
        pl.lit(None, dtype=pl.List(pl.Float64)).alias("null_draws")
    )
    bare = full._replace(result=Result(table))
    assert "Null distributions" not in text_of(diagnostics_section(bare, 0))
    assert "Null distributions" not in text_of(comparison_section(bare, 0))


# * The comparisons


def test_the_comparisons_hold_the_overview_the_data_table_and_a_tab_per_pair(
    full: RunFolder, comparison_html: str
) -> None:
    """Headings, pair names, the null accordion and the factors of the captions."""
    assert "Comparisons" in comparison_html
    assert "Overview" in comparison_html
    assert "Data space" in comparison_html
    assert TABS in comparison_html
    for pair in ("mcmc vs table", "mcmc vs group"):
        assert pair in comparison_html
    assert "Null distributions" in comparison_html
    assert "dashed ellipses and intervals" in comparison_html
    compared = full.samples["compare"].sets
    sets = [compared[name].select(problems=0) for name in full.names]
    factor = overview_factor(sets)
    assert f"bandwidth factor {factor:.4g}" in comparison_html
    assert "the fewest of the posteriors" in comparison_html
    assert "the bandwidth of the JS and Hellinger rows of this pair" in comparison_html
    assert "Sinkhorn divergence against ε/ℓ²" in comparison_html
    assert "left panel shows S_ε of each row (blue points)" in comparison_html
    assert "median of its null (grey line)" in comparison_html
    assert "right panel shows σ_eff of each row, with the threshold" in comparison_html
    assert "data:image/png" in comparison_html


def test_the_comparisons_need_comparison_or_data_space_rows(
    full: RunFolder, reweighting_run_folder: Path
) -> None:
    """None for diagnostics alone, for reweighting rows and for another problem."""
    diagnostics = keep(full, row_engines(full.result.table) == "diagnose")
    assert comparison_section(diagnostics, 0) is None
    assert comparison_section(full, 4) is None
    assert comparison_section(read_run_folder(reweighting_run_folder), 0) is None


def test_a_single_pair_has_no_tabs_and_a_pair_without_rows_for_the_bandwidth_says_so(
    full: RunFolder,
) -> None:
    """One pair is shown under its name; without kde rows the shared factor applies."""
    engines = row_engines(full.result.table)
    one_pair = ~pl.col("pair").list.contains("group")
    pair_only = keep(full, one_pair & (engines != "diagnose"))
    html = text_of(comparison_section(pair_only, 0))
    assert TABS not in html
    assert '<h3 id="mcmc-vs-table">mcmc vs table</h3>' in html
    assert "mcmc vs group" not in html
    # * Two posteriors: the overview would repeat the plot of the pair, so it is gone.
    assert "Overview" not in html
    assert "left out" not in html
    assert "bandwidth factor" in html
    bare = text_of(comparison_section(pair_only._replace(samples={}), 0))
    assert "Overview" not in bare
    assert "left out" not in bare
    assert "so there is no corner plot" in bare
    no_kde = keep(
        pair_only,
        ~(pl.col("metric").is_in(["js", "hellinger"]) & (pl.col("estimator") == "kde")),
    )
    html = text_of(comparison_section(no_kde, 0))
    assert "which the JS and Hellinger rows of this pair would use" in html


def test_a_pair_with_data_space_rows_alone_has_no_heatmap(full: RunFolder) -> None:
    """The data-space table and the plot stay; the overview needs compare rows."""
    signals = keep(full, row_engines(full.result.table) == "compare_signals")
    html = text_of(comparison_section(signals, 0))
    assert "Data space" in html
    assert "Overview" not in html
    assert "all posteriors" not in html
    assert "mcmc vs table" in html


def test_the_overview_gives_way_to_a_sentence(
    full: RunFolder, four_posteriors: RunFolder
) -> None:
    """Above three posteriors, or without samples, the sentence says why."""
    many = text_of(comparison_section(four_posteriors, 0))
    assert "overview corner plot of all posteriors is left out" in many
    assert "the run compares 4 posteriors, more than the 3" in many
    assert "p0 vs p1" in many
    assert "no corner plot" in many
    bare = text_of(comparison_section(full._replace(samples={}), 0))
    assert "left out: the samples are not in the run folder" in bare
    assert "so there is no corner plot" in bare


# * Reweighting, warnings, inputs, costs and the config


def test_the_reweighting_section_shows_the_table_and_the_verdict(
    full: RunFolder, reweighting_run_folder: Path
) -> None:
    """The k̂ verdict is the warning; absent without reweighting rows."""
    assert reweighting_section(full, 0) is None
    folder = read_run_folder(reweighting_run_folder)
    html = text_of(reweighting_section(folder, 0))
    assert "Reweighting" in html
    assert "reference:reweighted" in html
    assert "k̂" in html
    assert "informational" in html
    assert reweighting_section(folder, 3) is None


def test_the_run_sections_are_none_exactly_when_their_source_is_absent(
    full: RunFolder, result_of: Callable[..., Result]
) -> None:
    """Warnings, inputs, cost and config each need their own file or rows."""
    assert "Warnings" in text_of(warnings_section(full))
    assert "SubsamplingWarning" in text_of(warnings_section(full))
    quiet = full._replace(result=result_of({"metric": "skewness"}))
    assert warnings_section(quiet) is None
    assert "Inputs and reductions" in text_of(inputs_section(full))
    assert "mcmc" in text_of(inputs_section(full))
    assert inputs_section(full._replace(config=None)) is None
    both = text_of(cost_section(full))
    assert "Run time and cost" in both
    assert "estimated_seconds" in both
    assert "evaluations" in both
    only_costs = text_of(cost_section(full._replace(provenance=None)))
    assert "evaluations" in only_costs
    assert "estimated_seconds" not in only_costs
    only_timings = text_of(cost_section(full._replace(cost=None)))
    assert "estimated_seconds" in only_timings
    assert "evaluations" not in only_timings
    assert cost_section(full._replace(cost=None, provenance=None)) is None
    html = text_of(config_section(full))
    assert "Configuration" in html
    assert full.config_hash is not None
    assert full.config_hash in html
    assert "config.json" in html
    assert "eryn_priors" in html
    assert config_section(full._replace(config=None)) is None


# * Colours, captions and rounding


def renamed_folder(folder: RunFolder, names: dict[str, str]) -> RunFolder:
    """Return the folder with compare samples and rows in renamed parameters."""
    space = folder.samples["compare"].space
    renamed = ParamSpace(
        parameters=tuple(
            parameter.model_copy(
                update={"name": names[parameter.name], "label": names[parameter.name]}
            )
            for parameter in space.parameters
        )
    )
    samples = dict(folder.samples)
    samples["compare"] = PosteriorSamples(renamed, folder.samples["compare"].sets)
    table = folder.result.table
    mapped = pl.col("parameters").list.eval(pl.element().replace(names))
    table = table.with_columns(
        pl.when(row_engines(table) == "compare")
        .then(mapped)
        .otherwise(pl.col("parameters"))
        .alias("parameters")
    )
    return folder._replace(samples=samples, result=Result(table))


def test_the_heatmap_columns_are_the_names_of_the_compare_space(
    full: RunFolder,
) -> None:
    """A transform that renames the parameters must not send every row to joint."""
    names = {"x0": "u0", "x1": "u1", "x2": "u2"}
    folder = renamed_folder(full, names)
    table = folder.result.table
    columns = sections._parameter_names(folder, table)
    assert columns == ("u0", "u1", "u2")
    assert full.space is not None
    assert columns != full.space.names
    comparison = comparison_table(table, 0, ("mcmc", "table"), 1.0)
    figure = effect_heatmap(comparison, columns, title="t")
    assert figure is not None
    axes = figure.axes[0]
    assert [label.get_text() for label in axes.get_xticklabels()] == [
        "u0",
        "u1",
        "u2",
        "joint",
    ]
    singles = comparison.filter(~pl.col("parameters").str.contains(", "))
    joints = comparison.height - singles.height
    assert singles.height > 0
    patches = [patch for patch in axes.patches if isinstance(patch, Rectangle)]
    patches = [patch for patch in patches if patch.get_fill()]
    in_joint = [patch for patch in patches if int(patch.get_x()) == 3]
    assert len(in_joint) == joints
    assert len(patches) - len(in_joint) == singles.height
    # * With the names of the config, every single-parameter row would be lost.
    wrong = effect_heatmap(comparison, full.space.names, title="t")
    assert wrong is not None
    assert [label.get_text() for label in wrong.axes[0].get_xticklabels()] == ["joint"]


def test_the_parameter_names_fall_back_to_the_config_and_then_to_the_rows(
    full: RunFolder,
) -> None:
    """Compare samples first, then the space of the config, then the rows."""
    table = full.result.table
    assert sections._parameter_names(full, table) == ("x0", "x1", "x2")
    no_samples = full._replace(samples={})
    assert sections._parameter_names(no_samples, table) == ("x0", "x1", "x2")
    bare = no_samples._replace(space=None)
    assert sections._parameter_names(bare, table) == ("x0", "x1", "x2")


def test_the_colours_follow_the_position_of_the_posterior(full: RunFolder) -> None:
    """Slot k for posterior k; slots 1 and 2 for a pair beyond eight posteriors."""
    assert [sections._colour(full, name) for name in full.names] == list(
        SERIES_COLOURS[:3]
    )
    many = full._replace(names=tuple(f"p{position}" for position in range(9)))
    assert sections._pair_colours(many, ("p5", "p6")) == list(SERIES_COLOURS[:2])
    assert sections._pair_colours(full, ("group", "mcmc")) == [
        SERIES_COLOURS[2],
        SERIES_COLOURS[0],
    ]
    assert sections._colour(many, "p8") == SERIES_COLOURS[0]


def test_a_caption_names_its_factor_masses_estimates_and_cap() -> None:
    """The four optional sentences appear exactly when they apply."""
    plain = sections._caption(0.2204, "a basis", (), None).text
    assert "bandwidth factor 0.2204, a basis" in plain
    assert "68% and 95%" in plain
    assert "bounded estimate" not in plain
    assert "subsample" not in plain
    full_text = sections._caption(0.2, "a basis", ("x0", "x2"), 400, " An extra.").text
    assert "The panels of x0, x2 come from the package's bounded estimate" in full_text
    assert "uniform subsample of 400 samples" in full_text
    assert "An extra." in full_text


def test_tables_are_rounded_for_display_only() -> None:
    """Four significant figures, three decimals for effects, lists joined."""
    frame = pl.DataFrame(
        {
            "value": [0.123_456_789, 12_345.678_9, 0.0, math.nan, math.inf, None],
            "sigma_eff": [1.234_56, -0.000_49, 2.0, math.nan, math.inf, None],
            "names": [["a", "b"]] * 6,
            "count": [1, 2, 3, 4, 5, 6],
        }
    )
    shown = sections._displayed(frame)
    assert shown["value"].to_list()[:3] == [0.1235, 12_350.0, 0.0]
    assert math.isnan(shown["value"][3])
    assert shown["value"][4] == math.inf
    assert shown["value"][5] is None
    assert shown["sigma_eff"].to_list()[:3] == [1.235, -0.0, 2.0]
    assert shown["names"].to_list() == ["a, b"] * 6
    assert shown["count"].to_list() == [1, 2, 3, 4, 5, 6]
    assert frame["value"][0] == 0.123_456_789
    assert sections._significant(None) is None
    assert sections._significant(1.0 / 3.0) == 0.3333
