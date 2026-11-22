"""The report notebook: a marimo app that reads the run folder it sits in.

:func:`~mixed_signals.report.writer.write_report` copies this file into a run
folder as ``report.py``, byte for byte, and writes ``report.json`` next to it. The
notebook opens the folder it sits in, reads its files and the settings of
``report.json``, and shows one section per question: the overview, the problems,
the warnings, the inputs, the run time and the config. Every table, figure and
section comes from :mod:`mixed_signals.report`; the notebook only calls it, so
the report is as tested as the package.

Open it with ``marimo edit report.py``, run it as a script with
``python report.py``, or export it again with
``marimo export html report.py --no-include-code -o report.html``, since without
``-o`` marimo prints the HTML to standard output. Copied outside a run folder,
it stops with a note that says how to make one.
"""

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    from mixed_signals import report

    return mo, report


@app.cell
def _(mo):
    # ? The tests replace this cell, with app.run(defs={"run_folder": folder}).
    run_folder = mo.notebook_dir()
    return (run_folder,)


@app.cell
def _(mo, report, run_folder):
    folder = None if run_folder is None else report.open_run_folder(run_folder)
    mo.stop(
        folder is None,
        mo.callout(
            mo.md(
                "This notebook reads the run folder that it sits in, and this folder"
                " holds no `results.parquet`. Make a report with"
                " `mixed-signals report RUN_FOLDER`, which copies the notebook into"
                " the run folder."
            ),
            kind="warn",
        ),
    )
    return (folder,)


@app.cell
def _(folder, report):
    report.overview_section(folder)
    return


@app.cell
def _(folder, report):
    report.batch_section(folder)
    return


@app.cell
def _(folder, report):
    report.problem_sections(folder)
    return


@app.cell
def _(folder, report):
    report.warnings_section(folder)
    return


@app.cell
def _(folder, report):
    report.inputs_section(folder)
    return


@app.cell
def _(folder, report):
    report.cost_section(folder)
    return


@app.cell
def _(folder, report):
    report.config_section(folder)
    return


if __name__ == "__main__":
    app.run()
