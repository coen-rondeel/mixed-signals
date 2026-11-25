---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Chains, the pipeline and the report

```{code-cell} ipython3
:tags: [remove-cell]

import warnings

import matplotlib.pyplot as plt
import polars as pl


def format_warning(message, category, filename, lineno, line=None):
    """Write a warning as its category and message."""
    return f"{category.__name__}: {message}\n"


# * A warning then shows without the path of the temporary kernel cell.
warnings.formatwarning = format_warning
# * Show whole tables in the page; polars cuts at eight rows and 30 characters.
pl.Config.set_tbl_rows(60)
pl.Config.set_tbl_cols(20)
pl.Config.set_fmt_str_lengths(200)
# * Sharper figures than the 72 dpi of the inline backend.
plt.rcParams["figure.dpi"] = 150
```

A Markov chain Monte Carlo (MCMC) sampler returns chains, not independent draws.
Successive steps are correlated, the first steps still remember where the chain
started, and the chains are worth fewer independent samples than they hold steps. This
tutorial starts from chains whose autocorrelation is known, checks their health,
reduces them to nearly independent samples and diagnoses those. It then runs the same
analysis from a configuration file, as a pipeline on a cluster would, and turns the run
into a report.

## Chains with a known autocorrelation

`zoo.ar1_chains` builds chains in which every parameter follows a first-order
autoregressive process, x_t = φ x_{t−1} + √(1 − φ²) ε_t with standard normal ε_t. Each
parameter then has the stationary distribution N(0, 1) and the integrated
autocorrelation time τ = (1 + φ)/(1 − φ) {cite:p}`thompson2010comparison`. The chains
below have two parameters, with φ = 0.5 (τ = 3) and φ = 0.9 (τ = 19), 32 walkers of
400 steps, and they start 30 standard deviations away from the stationary mean, so
their first steps are a burn-in.

```{code-cell} ipython3
import jax
import matplotlib.pyplot as plt
import polars as pl
from IPython.display import display

import mixed_signals as ms
from mixed_signals import zoo

case = zoo.ar1_chains([0.5, 0.9])
chains = case.sample(jax.random.key(0), 400, walker_count=32, start=30.0)

figure, axes = plt.subplots(1, 2, figsize=(7.5, 3.0), sharey=True, layout="constrained")
for parameter_index, axis in enumerate(axes):
    # * The chain array is (step, temperature, walker, leaf, parameter).
    axis.plot(chains.samples[:, 0, :4, 0, parameter_index], linewidth=0.8)
    name = case.space.names[parameter_index]
    axis.set_title(f"{name}, phi = {case.coefficients[parameter_index]}")
    axis.set_xlabel("step")
axes[0].set_ylabel("value")
plt.show()

display(
    case.reference.select(
        pl.col("pair").list.join(", "),
        "metric",
        pl.col("parameters").list.join(", "),
        "value",
        "basis",
    )
)
```

The traces fall from 30 towards zero and then wander about it. The parameter with
φ = 0.9 forgets its start more slowly and wanders more slowly: its steps are more
strongly correlated.

## The health of the chains

τ counts the steps that the chain needs to forget where it was. S steps of W walkers are
worth about S·W/τ independent samples, the effective sample size (ESS)
{cite:p}`sokal1997monte,goodman2010ensemble`. The package estimates τ from the
autocorrelation of the walkers {cite:p}`foremanmackey2013emcee,geyer1992practical`, and
it tests whether the chains have settled by comparing the first and the second half of
every walker with a Kolmogorov-Smirnov test {cite:p}`smirnov1948table`: a chain that
still drifts has halves that differ. `diagnose` reports both, with and without a
burn-in.

```{code-cell} ipython3
health_metrics = ["tau", "ess", "split_chain_ks", "split_chain_kuiper_ks", "rhat"]
health_rows = []
for burn_in in (0, 100):
    health = ms.diagnose(
        chains, ["ar1"], case.space, key=1, burn_in=burn_in, diagnostics=["chains"]
    )
    health_rows.append(
        health.result.table.filter(pl.col("metric").is_in(health_metrics)).with_columns(
            burn_in=pl.lit(burn_in)
        )
    )
health_table = pl.concat(health_rows)
display(
    health_table.select(
        "burn_in",
        "metric",
        pl.col("parameters").list.join(", "),
        pl.col("value", "sigma_sig", "threshold").round_sig_figs(3),
        "flag",
    )
)
print(case.reference.filter(pl.col("metric") == "tau").select("parameters", "value"))
```

Without a burn-in the split test flags for the parameter with φ = 0.9: the first
half of each walker still holds its slow fall from 30 standard deviations. The parameter
with φ = 0.5 forgets its start within a few steps, so its halves agree either way. With
100 steps dropped, the halves of both agree. τ of x0 lies near its true value, while τ
of x1 reads low: 300 steps hold fewer than 50 autocorrelation times of x1, the length
that a reliable estimate of τ needs {cite:p}`foremanmackey2017autocorrelation`, and the package
warns about it, as the pipeline cell below shows. A long run
hides a short transient: the same fall in a run of thousands of steps would fill too
small a share of each half for the split test to flag, so a burn-in is set by looking at
the traces as well as at the test. The ESS flags below 400 {cite:p}`vehtari2021rank`. R̂, the ratio of
the spread between walkers to the spread within them {cite:p}`gelman1992inference`, is
informational only, because the moves of an ensemble sampler couple its walkers
{cite:p}`goodman2010ensemble`.

## Reduce and diagnose

`diagnose` reduces each run of chains before its other checks: it drops the burn-in,
keeps about one step in τ, so that the kept samples are nearly independent, and records the
effective sample size of the result. The reduction is also available on its own as
`reduce_chains`.

```{code-cell} ipython3
diagnosis = ms.diagnose(chains, ["ar1"], case.space, key=2, burn_in=100, draws=100)
reduction = diagnosis.reductions["ar1"]
record = reduction.record
print(f"burn-in: {record.burn_in} steps, thinning factor: {record.thin}")
print(f"steps kept per walker: {record.kept_steps}, samples: {record.sample_count}")
print(f"effective sample size: {record.n_eff:.0f}")
print("tau after the burn-in:", [round(value, 2) for value in record.tau])
print("tau of the thinned steps:", [round(value, 2) for value in record.thinned_tau])
print("reduced sample set:", tuple(reduction.sample_set.shape))

table = diagnosis.result.table
flagged = table.filter(pl.col("flag"))
print(f"{flagged.height} of {table.height} rows flag")
display(
    flagged.select(
        "metric",
        pl.col("parameters").list.join(", "),
        pl.col("value", "sigma_sig").round_sig_figs(3),
        "flag",
    )
)
```

The reduced set feeds every other family of rows, and `compare` takes it like any
other `SampleSet`.

## The same steps from a configuration file

A pipeline runs the same engines from a YAML file, so that a run can be repeated and
checked. The file names the posteriors and their formats, the parameter space and the
engines; every random choice derives from its seed. The next cell writes the chains to
an archive, a second posterior of flat samples shifted by 0.3 widths, and a
configuration that diagnoses both and compares them.

```{code-cell} ipython3
import tempfile
from pathlib import Path

import numpy as np

scratch = tempfile.TemporaryDirectory()
run_root = Path(scratch.name)

np.savez(run_root / "chains.npz", samples=np.asarray(chains.samples))
shifted_case = zoo.shifted_gaussians(0.3, param_count=2)
shifted_set = shifted_case.sample(jax.random.key(1), 2_000)
shifted_set.select(posteriors=[1]).to_npz(run_root / "draws.npz")

config_text = """\
seed: 2026
output: run
space:
  parameters:
    - name: x0
    - name: x1
posteriors:
  - {name: chains, format: chains, path: chains.npz, burn_in: 100, thin: tau}
  - {name: draws, format: samples, path: draws.npz}
diagnose: {draws: 100}
compare: {metrics: [mean_shift, energy, "kl:gaussian"], draws: 100}
"""
config_path = run_root / "config.yaml"
config_path.write_text(config_text)
print(config_text)
```

`estimate_pipeline` reads the data and estimates the run time of every engine without
computing a statistic, the dry run. `run_pipeline` then runs the engines and writes the
run folder.

```{code-cell} ipython3
from mixed_signals.pipeline import estimate_pipeline, run_pipeline

estimate = estimate_pipeline(config_path)
print(f"estimated run time: {estimate.seconds:.1f} s")
display(
    pl.concat([part.table for part in estimate.estimates.values()]).select(
        "engine",
        pl.col("pair").list.join(" to "),
        "metric",
        "estimator",
        "samples",
        "evaluations",
        pl.col("seconds").round_sig_figs(3),
    )
)

run = run_pipeline(config_path)
display(
    pl.DataFrame(
        [
            {
                "file": str(path.relative_to(run.output)),
                "size_kB": round(path.stat().st_size / 1024, 1),
            }
            for path in sorted(run.output.rglob("*"))
            if path.is_file()
        ]
    )
)
display(
    pl.read_parquet(run.output / "results.parquet")
    .filter(pl.col("pair").list.len() == 2)
    .select(
        pl.col("pair").list.join(" to "),
        "metric",
        pl.col("parameters").list.join(", "),
        pl.col("sigma_eff", "sigma_sig").round_sig_figs(3),
        "flag",
    )
)
```

The run folder holds `results.parquet`, one row per statistic of every engine, the
validated configuration as `config.json`, whose SHA-256 hash every row records,
`reductions.json`, the dry-run table `cost.parquet`, `provenance.json` with the versions
and the hashes of every input, and the compared samples in `samples/`. From a shell, the
same run is

```bash
mixed-signals run config.yaml --dry-run
mixed-signals run config.yaml
```

## The report

`write_report` turns a run folder into a marimo notebook, `report.py`, that reads only
the folder it sits in: tables of the diagnostics and comparisons, a heatmap of the
effect sizes over their thresholds, corner plots, and the null of every row. With
`html=True`, the default, it also exports a page without code, `report.html`. That page
embeds every figure, but it loads marimo's front end from the internet, so it renders
only online. The functions behind the notebook can also be called directly:

```{code-cell} ipython3
from mixed_signals.report import (
    comparison_table,
    effect_heatmap,
    read_run_folder,
    significance_level,
    write_report,
)

report_files = write_report(run.output, html=False)
print([path.name for path in (report_files.notebook, report_files.settings)])

folder = read_run_folder(run.output)
level = significance_level(folder, "compare")
summary = comparison_table(folder.result.table, 0, ("chains", "draws"), level)
display(
    summary.select(
        "metric",
        "estimator",
        "direction",
        "parameters",
        pl.col("sigma_eff", "r_eff", "sigma_sig", "threshold", "effect").round_sig_figs(
            3
        ),
        "flag",
    )
)
display(effect_heatmap(summary, folder.space.names, title="chains vs draws"))
scratch.cleanup()
```

## Summary

- τ and the effective sample size measure how much a chain is worth; the split test
  shows whether it has settled.
- `diagnose` drops the burn-in and thins by τ before its other checks, and returns the
  reduction it made.
- A YAML configuration runs the same engines repeatably, and writes every number with
  its configuration, inputs and versions into one run folder.
- `write_report` turns the folder into a notebook of tables and figures.

The last tutorial, {doc}`lisa_sources`, applies all of this to the sources of the LISA
mission.
