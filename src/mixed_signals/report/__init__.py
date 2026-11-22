"""Reports: a notebook of tables and corner plots for a run folder.

A report answers three questions about the posteriors of a run: is each posterior
sound, do two posteriors differ, and what do the engines warn about. It holds an
overview of the run, the diagnostics of each posterior, the comparison of each pair
with a heatmap of the effects, a corner plot and the null of every row, the
reweighting of weight sets, the warnings, the inputs, the run time and the config.
The comparison rows decide every flag; the report only displays them, with the
effect size and the null beside the value.

Entry points
    ``mixed-signals report RUN_FOLDER`` and :func:`write_report` copy the notebook
    of the package into the run folder as ``report.py``, write ``report.json`` and
    export ``report.html``, a page of outputs without code. The figures and tables
    are embedded in the page, but marimo loads its front end from
    ``cdn.jsdelivr.net``, so the page renders only with an internet connection.
    ``mixed-signals run`` writes no report and works without the ``report``
    extra.
Settings
    ``report.json`` holds the :class:`ReportSettings`. ``problems`` is how many
    problems the report shows in full (the problems with the most flagged rows
    first, ten by default), ``"all"``, or a list of problem indices. ``plot_cap``
    is the most samples that a posterior contributes to a corner plot, none by
    default; above it, the plot draws a uniform subsample at the bandwidth of the
    full set, and its caption says so.
One notebook
    The notebook reads only its run folder: ``results.parquet`` and, when they are
    there, ``config.json``, ``samples/``, ``reductions.json``, ``cost.parquet``,
    ``provenance.json`` and ``report.json``. It needs no input file, no eryn and
    no user code, so a run folder can be reported on another machine.
Building blocks
    Every table, figure and section is a function of this package, so that API
    users can use them on their own results: :func:`corner_plot` for the posteriors
    of any :class:`~mixed_signals.core.samples.SampleSet`, :func:`null_histograms`,
    :func:`effect_heatmap` and :func:`epsilon_plot` for the rows of a result, and
    the tables of :mod:`mixed_signals.report.tables`.
Extra
    The report needs marimo and cantuccio, which the ``report`` extra installs:
    ``pip install "mixed-signals[report]"``. Importing this package without them
    raises :class:`~mixed_signals.errors.MissingDependencyError`, and
    ``import mixed_signals`` never imports it.

References:
    :cite:t:`santini2026cantuccio`.
"""

from mixed_signals.report import _requirements as _requirements
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
    DEFAULT_PROBLEMS,
    PosteriorSamples,
    ReportSettings,
    RunFolder,
    check_problems,
    open_run_folder,
    plot_key,
    read_run_folder,
    samples_for,
    select_problems,
    significance_level,
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
from mixed_signals.report.writer import ReportFiles, write_report

__all__ = [
    "CONTOUR_LEVELS",
    "DEFAULT_PROBLEMS",
    "DIAGNOSTIC_FAMILY_METRICS",
    "OVERVIEW_LIMIT",
    "REWEIGHT_METRICS",
    "PosteriorSamples",
    "ReportFiles",
    "ReportSettings",
    "RunFolder",
    "batch_section",
    "bounded_parameters",
    "chain_health_table",
    "check_problems",
    "comparison_pairs",
    "comparison_section",
    "comparison_table",
    "config_section",
    "corner_plot",
    "cost_section",
    "cost_table",
    "dataspace_table",
    "diagnostic_table",
    "diagnostics_section",
    "effect_heatmap",
    "epsilon_plot",
    "flag_counts",
    "inputs_section",
    "inputs_table",
    "null_histograms",
    "open_run_folder",
    "overview_factor",
    "overview_section",
    "pair_factor",
    "pair_rows",
    "plot_key",
    "plot_subsample",
    "posterior_rows",
    "problem_sections",
    "rate_table",
    "read_run_folder",
    "reweighting_section",
    "reweighting_table",
    "row_bandwidth",
    "row_engines",
    "samples_for",
    "select_problems",
    "significance_level",
    "silverman_factor",
    "timings_table",
    "warnings_section",
    "warnings_table",
    "with_effects",
    "write_report",
]
