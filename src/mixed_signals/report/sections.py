"""Sections of the report: marimo layouts of tables, figures and plain sentences.

Each function returns a :class:`marimo.Html` layout, or ``None`` when its source is
absent from the run folder. The layouts hold figures as PNG images and tables
with their full data, so a static export shows them, with tabs and collapsed
accordions, without a running kernel. Every section opens with a heading and a
few plain sentences that say what it shows and how to read it.

Rounding
    Tables round floats for display only: values, null means and p-values to
    four significant figures, and σ_eff, r_eff, σ_sig, effects and thresholds to
    three decimals. The numbers behind the tables keep their full precision.
Posteriors
    Posterior k of the run, in the order of ``folder.names``, has slot k of the
    series colours in every figure. With more than eight posteriors the slots
    wrap around, and a pair plot colours its two posteriors with slots 1 and 2.
"""

import json
import math
from collections.abc import Mapping, Sequence
from typing import Final

import marimo as mo
import polars as pl

import mixed_signals
from mixed_signals.core.samples import SampleSet, kish_effective_sample_size
from mixed_signals.report._style import SERIES_COLOURS
from mixed_signals.report.corner import (
    CONTOUR_LEVELS,
    OVERVIEW_LIMIT,
    bounded_parameters,
    corner_plot,
    overview_factor,
    pair_factor,
    plot_subsample,
    row_bandwidth,
    silverman_factor,
)
from mixed_signals.report.figures import effect_heatmap, epsilon_plot, null_histograms
from mixed_signals.report.folder import (
    PosteriorSamples,
    RunFolder,
    plot_key,
    samples_for,
    select_problems,
    significance_level,
)
from mixed_signals.report.tables import (
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

__all__ = [
    "batch_section",
    "comparison_section",
    "config_section",
    "cost_section",
    "diagnostics_section",
    "inputs_section",
    "overview_section",
    "problem_sections",
    "reweighting_section",
    "warnings_section",
]

_ENGINES: Final = ("diagnose", "compare", "compare_signals")
_ENGINE_ORDER: Final = ("reweight", *_ENGINES)
# * Columns shown to three decimals; the other floats show four significant figures.
_DECIMAL_COLUMNS: Final = (
    "sigma_eff",
    "r_eff",
    "sigma_sig",
    "effect",
    "threshold",
    "pareto_k",
    "data_sigma_eff",
    "chart_sigma_eff",
    "data_sigma_sig",
    "chart_sigma_sig",
)
_SIGNIFICANT_DIGITS: Final = 4
_DECIMALS: Final = 3
_PAGE_SIZE: Final = 15
_NOT_RECORDED: Final = "not recorded"


def overview_section(folder: RunFolder) -> mo.Html:
    """Build the title, the facts of the run and the flag counts of every engine.

    The text defines σ_eff, r_eff, σ_sig, the threshold and the flag once, for
    every later section.

    Args:
        folder: The run folder.

    Returns:
        The layout.
    """
    counts = (
        flag_counts(folder.result.table)
        .group_by("engine", maintain_order=True)
        .agg(
            pl.col("rows").sum(),
            pl.col("with_threshold").sum(),
            pl.col("flagged").sum(),
        )
    )
    counts = counts.sort(
        pl.col("engine").replace_strict(
            {name: position for position, name in enumerate(_ENGINE_ORDER)}
        )
    )
    return mo.vstack(
        [
            mo.md(_OVERVIEW_TEXT),
            mo.md(_run_facts(folder)),
            _table(counts),
        ]
    )


def batch_section(folder: RunFolder) -> mo.Html | None:
    """Build the flag counts of every problem, and say which problems follow.

    Args:
        folder: The run folder.

    Returns:
        ``None`` for a result of one problem. Otherwise the layout with one table,
        problems as rows and engines as columns, and the rule that chose the
        problems shown in full.
    """
    counts = flag_counts(folder.result.table)
    if counts["problem"].n_unique() < 2:
        return None
    wide = (
        counts.pivot(
            on="engine", index="problem", values="flagged", aggregate_function="sum"
        )
        .sort("problem")
        .fill_null(0)
    )
    engines = [name for name in _ENGINE_ORDER if name in wide.columns]
    wide = wide.select(
        "problem",
        *engines,
        pl.sum_horizontal(*engines).alias("flagged_in_total"),
    )
    settings = folder.settings.problems
    if settings == "all":
        rule = "every problem"
    elif isinstance(settings, tuple):
        rule = "the problems listed in report.json, in its order"
    else:
        rule = "those with the most flagged rows, ties by index"
    shown = ", ".join(str(problem) for problem in select_problems(folder))
    return mo.vstack(
        [
            mo.md(
                "## Problems\n\nThe result holds"
                f" {counts['problem'].n_unique()} problems. The table counts the rows"
                " of each engine that flag. The problems shown in full below are"
                f" {shown}: {rule}. Set `problems` in report.json to change them."
            ),
            _table(wide),
        ]
    )


def problem_sections(folder: RunFolder) -> mo.Html:
    """Build the diagnostics, comparisons and reweighting of the selected problems.

    Args:
        folder: The run folder, whose settings select the problems.

    Returns:
        The sections of one problem stacked, or one tab per problem.
    """
    problems = select_problems(folder)
    stacks = {}
    for problem in problems:
        sections = (
            diagnostics_section(folder, problem),
            comparison_section(folder, problem),
            reweighting_section(folder, problem),
        )
        stacks[problem] = mo.vstack(
            [section for section in sections if section is not None]
        )
    if len(problems) == 1:
        return stacks[problems[0]]
    tabs = {f"Problem {problem}": stack for problem, stack in stacks.items()}
    return mo.vstack([mo.ui.tabs(tabs)])


def diagnostics_section(folder: RunFolder, problem: int) -> mo.Html | None:
    """Build the diagnostics of every posterior of a problem.

    Each posterior has its own corner plot at its own bandwidth, its flagged rows,
    its chain health and the null of every row.

    Args:
        folder: The run folder.
        problem: The problem index.

    Returns:
        ``None`` without diagnostic rows. Otherwise the layout, with one tab per
        posterior when there are several.
    """
    table = folder.result.table
    engines = row_engines(table)
    rows = table.filter(engines == "diagnose", table["problem"] == problem)
    if rows.is_empty():
        return None
    held = set(rows["pair"].list.get(0).to_list())
    names = [name for name in folder.names if name in held]
    tabs = {name: _posterior_block(folder, problem, name) for name in names}
    return mo.vstack(
        [
            mo.md(
                "## Diagnostics\n\nDiagnostics check each posterior alone against"
                " Gaussian surrogates with the same mean and covariance. A row flags"
                " when its value, less the mean of its null, passes its threshold and"
                " σ_sig reaches the significance level. The corner plot shows the"
                " estimates that the mode and railing rows read."
            ),
            _tabs_or_block(tabs),
        ]
    )


def comparison_section(folder: RunFolder, problem: int) -> mo.Html | None:
    """Build the comparisons of every pair of posteriors of a problem.

    The overview corner plot comes first when the problem compares three
    posteriors, up to :data:`~mixed_signals.report.corner.OVERVIEW_LIMIT`, and
    gives way to a sentence for more posteriors or without samples. For two
    posteriors it is left out, since it would repeat the plot of their pair. Then
    come the data-space table and one tab per pair with its heatmap, table,
    corner plot, ε plot and nulls.

    Args:
        folder: The run folder.
        problem: The problem index.

    Returns:
        ``None`` without comparison or data-space rows. Otherwise the layout.
    """
    table = folder.result.table
    engines = row_engines(table)
    in_problem = table["problem"] == problem
    has_compare = (engines == "compare") & in_problem
    has_data = (engines == "compare_signals") & in_problem
    if not (has_compare.any() or has_data.any()):
        return None
    levels = _levels(folder)
    effects = with_effects(table.filter(in_problem), levels)
    pairs = comparison_pairs(table, problem)
    blocks: list[object] = [
        mo.md(
            "## Comparisons\n\nComparisons check each pair of posteriors against a"
            " permutation null. The heatmap shows the effect of each row over its"
            " threshold, so that a cell at 1 has reached it; bold text marks a row"
            " whose σ_sig reaches the level, and a frame marks a row that fails."
        )
    ]
    overview = _overview_block(folder, problem, effects) if has_compare.any() else None
    if overview is not None:
        blocks.append(overview)
    data_table = dataspace_table(table, problem)
    if data_table is not None:
        blocks.append(_dataspace_block(data_table))
    tabs = {
        f"{first} vs {second}": _pair_block(folder, problem, (first, second), effects)
        for first, second in pairs
    }
    blocks.append(_tabs_or_block(tabs))
    return mo.vstack(blocks)


def reweighting_section(folder: RunFolder, problem: int) -> mo.Html | None:
    """Build the reweighting table of a problem.

    Args:
        folder: The run folder.
        problem: The problem index.

    Returns:
        ``None`` without reweighting rows. Otherwise the layout.
    """
    weights = reweighting_table(folder.result.table, problem)
    if weights is None:
        return None
    return mo.vstack(
        [
            mo.md(
                "## Reweighting\n\nReweighting moves a posterior to a new prior or"
                " likelihood with importance weights. The weights are reliable when"
                " the Pareto shape k̂ of their tail stays below the threshold for"
                " their number of draws. The threshold depends on the draws, so"
                " the verdict is the engine's own warning, in the column flagged."
                " The Kish size of the new weights is informational: it does not"
                " flag."
            ),
            _table(weights),
        ]
    )


def warnings_section(folder: RunFolder) -> mo.Html | None:
    """Build the table of the distinct warnings of the rows.

    Args:
        folder: The run folder.

    Returns:
        ``None`` without warnings. Otherwise the layout.
    """
    warnings = warnings_table(folder.result.table)
    if warnings is None:
        return None
    return mo.vstack(
        [
            mo.md(
                "## Warnings\n\nThe engines record every package warning in the"
                " rows that raised it. Each line below is one distinct warning with"
                " the number of rows that raised it and their metrics, the most"
                " frequent first. Read them with the numbers that they qualify."
            ),
            _table(warnings),
        ]
    )


def inputs_section(folder: RunFolder) -> mo.Html | None:
    """Build the table of the posteriors, their files and their reductions.

    Args:
        folder: The run folder.

    Returns:
        ``None`` without ``config.json``. Otherwise the layout.
    """
    inputs = inputs_table(folder)
    if inputs is None:
        return None
    return mo.vstack(
        [
            mo.md(
                "## Inputs and reductions\n\nEach posterior is listed with the file"
                " it came from, its size and the first digits of its SHA-256, the"
                " samples and effective samples that the engines took, and, for a"
                " run of chains, the reduction that made them: the burn-in, the"
                " thinning, the steps kept and the largest autocorrelation time τ."
            ),
            _table(inputs),
        ]
    )


def cost_section(folder: RunFolder) -> mo.Html | None:
    """Build the run time of each engine and the estimated cost of each statistic.

    Args:
        folder: The run folder.

    Returns:
        ``None`` without ``provenance.json`` engines and ``cost.parquet``.
        Otherwise the layout with the timings table, then the cost table.
    """
    timings, costs = timings_table(folder), cost_table(folder)
    if timings is None and costs is None:
        return None
    blocks: list[object] = [
        mo.md(
            "## Run time and cost\n\nThe estimate is the dry run before the engines"
            " ran, in seconds on one CPU of the developers' machine; the ratio is"
            " the measured wall time over the estimate. The second table lists the"
            " statistics by estimated cost."
        )
    ]
    blocks.extend(_table(frame) for frame in (timings, costs) if frame is not None)
    return mo.vstack(blocks)


def config_section(folder: RunFolder) -> mo.Html | None:
    """Build the config hash and the config of the run, collapsed.

    Args:
        folder: The run folder.

    Returns:
        ``None`` without ``config.json``. Otherwise the layout.
    """
    if folder.config is None:
        return None
    text = json.dumps(json.loads((folder.path / "config.json").read_bytes()), indent=2)
    return mo.vstack(
        [
            mo.md(
                "## Configuration\n\nThe config hash is the SHA-256 of config.json,"
                " which every result row records:"
                f" `{folder.config_hash}`. The file holds the validated config with"
                " every default filled in, and the resolved parameter space."
            ),
            mo.accordion({"config.json": mo.md(f"```json\n{text}\n```")}),
        ]
    )


_OVERVIEW_TEXT: Final = """# mixed signals report

Each comparison row reports an effect size, σ_eff, or r_eff for a row that
measures only a width, and σ_sig. σ_eff is the shift between two unit Gaussians
that gives the same null-subtracted value, and r_eff is the width ratio that does.
σ_sig is the p-value of the value against a finite-sample null, written as a
Gaussian σ: Φ⁻¹(1 − p) for an upper tail and Φ⁻¹(1 − p/2) for a two-sided one. The
`tail` column of the tables names the tail of each row. A threshold acts on the
effect size, and a row flags when its effect size reaches its threshold and σ_sig
reaches the significance level of its engine. The table counts the rows of each
engine, the rows that have a threshold and the rows that flag.
"""


def _run_facts(folder: RunFolder) -> str:
    """Return the facts of the run as a markdown list.

    A fact comes from ``provenance.json`` and otherwise from the provenance column
    of the first row, which holds the package, the seed, JAX, the device and the
    config hash.
    """
    record = folder.provenance or {}
    column = folder.result.table["provenance"]
    row = column[0] if column.len() else {}
    facts = [
        ("Run folder", folder.path),
        ("Config hash", _fact(folder.config_hash, row.get("config_hash"))),
        ("Seed", _fact(record.get("seed"), row.get("seed"))),
        ("Started", _fact(record.get("started"))),
        ("Wall time", _seconds(_fact(record.get("wall_seconds")))),
        ("Package", _fact(record.get("package_version"), row.get("package_version"))),
        ("JAX", _fact(record.get("jax_version"), row.get("jax_version"))),
        ("Python", _fact(record.get("python_version"))),
        ("Device", _fact(record.get("device_platform"), row.get("device_platform"))),
        ("Machine", _fact(record.get("machine"))),
        ("Report built with mixed-signals", mixed_signals.__version__),
    ]
    return "\n".join(f"- {label}: {value}" for label, value in facts)


def _fact(*candidates: object) -> object:
    """Return the first candidate that is not None, or a note that none is recorded."""
    return next((value for value in candidates if value is not None), _NOT_RECORDED)


def _seconds(value: object) -> object:
    """Return a number of seconds as text with its unit, and anything else as is."""
    return f"{value:.1f} s" if isinstance(value, int | float) else value


def _levels(folder: RunFolder) -> dict[str, float]:
    """Return the significance level of each engine of the run."""
    return {name: significance_level(folder, name) for name in _ENGINES}


def _colour(folder: RunFolder, name: str) -> str:
    """Return the series colour of a posterior, by its position in the run."""
    return SERIES_COLOURS[folder.names.index(name) % len(SERIES_COLOURS)]


def _pair_colours(folder: RunFolder, pair: tuple[str, str]) -> list[str]:
    """Return the colours of a pair: their slots, or slots 1 and 2 beyond eight."""
    if len(folder.names) > len(SERIES_COLOURS):
        return list(SERIES_COLOURS[:2])
    return [_colour(folder, name) for name in pair]


def _parameter_names(folder: RunFolder, rows: pl.DataFrame) -> tuple[str, ...]:
    """Return the parameter names of the comparison rows, in space order.

    The rows of ``compare`` lie in the space of its transform, so the names come
    from the space of the compare samples when there is one, then from the space of
    the config, then from the rows in order of appearance.
    """
    samples = samples_for(folder, "compare")
    if samples is not None:
        return samples.space.names
    if folder.space is not None:
        return folder.space.names
    names = rows["parameters"].explode().drop_nulls().unique(maintain_order=True)
    return tuple(names.to_list())


def _tabs_or_block(blocks: Mapping[str, object]) -> mo.Html:
    """Return the only block, under its name, or one tab per block."""
    if len(blocks) == 1:
        ((name, block),) = blocks.items()
        return mo.vstack([mo.md(f"### {name}"), block])
    return mo.vstack([mo.ui.tabs(blocks)])


def _significant(value: float | None) -> float | None:
    """Return a number rounded to the significant figures that tables show."""
    if value is None or value == 0 or not math.isfinite(value):
        return value
    return float(f"{value:.{_SIGNIFICANT_DIGITS}g}")


def _displayed(frame: pl.DataFrame) -> pl.DataFrame:
    """Return a table rounded for display, with lists of names joined."""
    columns = []
    for name, dtype in frame.schema.items():
        if dtype == pl.Float64 and name in _DECIMAL_COLUMNS:
            columns.append(pl.col(name).round(_DECIMALS))
        elif dtype == pl.Float64:
            columns.append(
                pl.col(name).map_elements(_significant, return_dtype=pl.Float64)
            )
        elif dtype == pl.List(pl.String):
            columns.append(pl.col(name).list.join(", "))
    return frame.with_columns(columns)


def _table(frame: pl.DataFrame) -> object:
    """Return a table without selection, so that a static export shows all of it."""
    return mo.ui.table(
        _displayed(frame),
        selection=None,
        page_size=_PAGE_SIZE,
        show_column_summaries=False,
        show_download=False,
    )


def _figure(figure: object) -> mo.Html:
    """Return a figure as an image."""
    return mo.as_html(figure)


def _caption(
    factor: float,
    basis: str,
    bounded: Sequence[str],
    capped: int | None,
    extra: str = "",
) -> mo.Html:
    """Return the caption of a corner plot: its bandwidth, masses and estimates."""
    masses = " and ".join(f"{level:.0%}" for level in CONTOUR_LEVELS)
    text = (
        f"Panels show Gaussian kernel density estimates at the bandwidth factor"
        f" {factor:.4g}, {basis}. Solid contours enclose {masses} of the density;"
        " dashed ellipses and intervals show the moment-matched Gaussian of each"
        " posterior at the same masses."
    )
    if bounded:
        text += (
            f" The panels of {', '.join(bounded)} come from the package's bounded"
            " estimate, because of a circle or a bound within reach of the kernels;"
            " the others come from the exact estimate of cantuccio."
        )
    if capped is not None:
        text += (
            f" Each posterior is drawn from a uniform subsample of {capped}"
            " samples (plot_cap in report.json), at the bandwidth of the full set."
        )
    return mo.md(text + extra)


def _plot_sets(
    folder: RunFolder,
    samples: PosteriorSamples,
    section: str,
    problem: int,
    names: Sequence[str],
) -> tuple[list[tuple[str, SampleSet]], list[tuple[str, SampleSet]], int | None]:
    """Return the full sets of the posteriors, their plotted subsamples and the cap."""
    full = [(name, samples.sets[name].select(problems=problem)) for name in names]
    cap = folder.settings.plot_cap
    plotted = [
        (
            name,
            plot_subsample(sample_set, cap, plot_key(folder, section, problem, name))
            if cap is not None
            else sample_set,
        )
        for name, sample_set in full
    ]
    acted = cap is not None and any(
        sample_set.shape.sample > cap for _, sample_set in full
    )
    return full, plotted, cap if acted else None


def _posterior_block(folder: RunFolder, problem: int, name: str) -> mo.Html:
    """Return the diagnostics of one posterior: plot, tables and nulls."""
    level = significance_level(folder, "diagnose")
    table = folder.result.table
    rows = diagnostic_table(table, problem, name, level)
    tested = rows.filter(pl.col("threshold").is_not_null())
    flagged = rows.filter(pl.col("flag").fill_null(value=False))
    blocks: list[object] = [
        mo.md(
            f"{flagged.height} of the {tested.height} rows of {name} that have a"
            " threshold flag."
        )
    ]
    samples = samples_for(folder, "diagnose")
    if samples is None or name not in samples.sets:
        blocks.append(
            mo.md("The samples are not in the run folder, so there is no corner plot.")
        )
    else:
        blocks.append(_diagnostic_plot(folder, samples, problem, name))
    if flagged.height:
        blocks.extend([mo.md("**Flagged rows**"), _table(flagged)])
    for label, health in (
        ("**Chain health**", chain_health_table(table, problem, name)),
        ("**Acceptance and swap rates**", rate_table(table, problem, name)),
    ):
        if health is not None:
            blocks.extend([mo.md(label), _table(health)])
    blocks.append(mo.accordion({"All diagnostic rows": _table(rows)}))
    effects = with_effects(posterior_rows(table, problem, name), {"diagnose": level})
    nulls = null_histograms(effects, title=f"Null distributions of {name}")
    if nulls is not None:
        blocks.append(mo.accordion({"Null distributions": _figure(nulls)}))
    return mo.vstack(blocks)


def _diagnostic_plot(
    folder: RunFolder, samples: PosteriorSamples, problem: int, name: str
) -> mo.Html:
    """Return the corner plot of one posterior at its own factor, with its caption."""
    full, plotted, capped = _plot_sets(folder, samples, "diagnose", problem, [name])
    sample_set = full[0][1]
    factor = silverman_factor(sample_set.log_weights[0, 0])
    size = float(kish_effective_sample_size(sample_set.log_weights[0, 0]))
    figure = corner_plot(
        plotted,
        samples.space,
        factor=factor,
        colours=[_colour(folder, name)],
        title=f"{name}, problem {problem}",
        bounds=True,
    )
    basis = (
        f"Silverman's one-dimensional factor at the {size:.0f} effective samples of"
        " this posterior, which its mode and railing rows use"
    )
    return mo.vstack(
        [
            _figure(figure),
            _caption(
                factor,
                basis,
                bounded_parameters(full, samples.space, factor),
                capped,
                " Thin grey lines mark the finite bounds of the parameters.",
            ),
        ]
    )


def _overview_block(
    folder: RunFolder, problem: int, effects: pl.DataFrame
) -> mo.Html | None:
    """Return the overview corner plot of all compared posteriors.

    Returns:
        ``None`` for two posteriors, whose overview would be the plot of their
        pair; the plot with its caption for up to ``OVERVIEW_LIMIT`` posteriors;
        and a sentence that gives the reason for more of them, or without samples.
    """
    compared = comparison_pairs(effects.filter(pl.col("engine") == "compare"), problem)
    names = [name for name in folder.names if any(name in pair for pair in compared)]
    if len(names) <= 2:
        return None
    samples = samples_for(folder, "compare")
    reason = None
    if len(names) > OVERVIEW_LIMIT:
        reason = (
            f"the run compares {len(names)} posteriors, more than the"
            f" {OVERVIEW_LIMIT} that the palette keeps apart in one plot; the plots"
            " of the pairs carry the comparison"
        )
    elif samples is None or any(name not in samples.sets for name in names):
        reason = "the samples are not in the run folder"
    if reason is not None:
        return mo.md(
            f"The overview corner plot of all posteriors is left out: {reason}."
        )
    assert samples is not None
    full, plotted, capped = _plot_sets(folder, samples, "compare", problem, names)
    factor = overview_factor([sample_set for _, sample_set in full])
    size = min(
        float(kish_effective_sample_size(sample_set.log_weights[0, 0]))
        for _, sample_set in full
    )
    figure = corner_plot(
        plotted,
        samples.space,
        factor=factor,
        colours=[_colour(folder, name) for name in names],
        title=f"All posteriors, problem {problem}",
    )
    basis = (
        f"Silverman's one-dimensional factor at {size:.0f} effective samples, the"
        " fewest of the posteriors"
    )
    return mo.vstack(
        [
            mo.md("### Overview"),
            _figure(figure),
            _caption(
                factor, basis, bounded_parameters(full, samples.space, factor), capped
            ),
        ]
    )


def _dataspace_block(data_table: pl.DataFrame) -> mo.Html:
    """Return the data-space table with the sentence that explains it."""
    return mo.vstack(
        [
            mo.md(
                "### Data space\n\nData space compares the signals of the posteriors"
                " after whitening by the noise model. A chart row compares them in"
                " the local chart of the Fisher matrix. The ratio of the data value"
                " to the chart value shows where the linear order of the chart"
                " fails."
            ),
            _table(data_table),
        ]
    )


def _pair_block(
    folder: RunFolder,
    problem: int,
    pair: tuple[str, str],
    effects: pl.DataFrame,
) -> mo.Html:
    """Return the comparison of one pair: heatmap, table, plots and nulls."""
    level = significance_level(folder, "compare")
    table = folder.result.table
    blocks: list[object] = []
    comparison = comparison_table(table, problem, pair, level)
    if comparison.height:
        parameters = _parameter_names(folder, table)
        heatmap = effect_heatmap(
            comparison, parameters, title=f"{pair[0]} vs {pair[1]}, problem {problem}"
        )
        assert heatmap is not None
        blocks.extend([_figure(heatmap), _table(comparison)])
    blocks.append(_pair_plot(folder, problem, pair))
    pair_effects = pair_rows(effects, problem, pair, ("compare", "compare_signals"))
    sinkhorn = epsilon_plot(
        pair_effects.filter(pl.col("metric") == "sinkhorn"), level=level
    )
    if sinkhorn is not None:
        blocks.extend(
            [
                mo.md(
                    "The Sinkhorn divergence against ε/ℓ², with ℓ the median"
                    " bandwidth. The left panel shows S_ε of each row (blue points),"
                    " the median of its null (grey line) and the band of the null"
                    " between the Φ(−L) and Φ(L) quantiles for the significance"
                    " level L, so that a row reaches the level where its point"
                    " passes the upper edge of the band. The right panel shows σ_eff"
                    " of each row, with the threshold as a grey line."
                ),
                _figure(sinkhorn),
            ]
        )
    nulls = null_histograms(
        pair_effects, title=f"Null distributions of {pair[0]} vs {pair[1]}"
    )
    if nulls is not None:
        blocks.append(mo.accordion({"Null distributions": _figure(nulls)}))
    return mo.vstack(blocks)


def _pair_plot(folder: RunFolder, problem: int, pair: tuple[str, str]) -> mo.Html:
    """Return the corner plot of a pair at the factor of its rows, with a caption."""
    samples = samples_for(folder, "compare")
    if samples is None or any(name not in samples.sets for name in pair):
        return mo.md(
            "The samples are not in the run folder, so there is no corner plot."
        )
    full, plotted, capped = _plot_sets(folder, samples, "compare", problem, pair)
    table = folder.result.table
    factor = pair_factor(table, problem, pair, full[0][1], full[1][1])
    from_rows = row_bandwidth(table, problem, pair) is not None
    figure = corner_plot(
        plotted,
        samples.space,
        factor=factor,
        colours=_pair_colours(folder, pair),
        title=f"{pair[0]} vs {pair[1]}, problem {problem}",
    )
    basis = (
        "the bandwidth of the JS and Hellinger rows of this pair, so that each"
        " diagonal is the estimate behind their values"
        if from_rows
        else "Silverman's rule at the smaller effective sample size of the two"
        " posteriors, which the JS and Hellinger rows of this pair would use"
    )
    return mo.vstack(
        [
            _figure(figure),
            _caption(
                factor, basis, bounded_parameters(full, samples.space, factor), capped
            ),
        ]
    )
