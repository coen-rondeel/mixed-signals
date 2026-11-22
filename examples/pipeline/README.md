# Pipeline example

This example runs the three engines of mixed-signals from one YAML config on three
posteriors of one toy problem. The problem is the linear signal of the validation
zoo: a data vector that depends linearly on three correlated parameters, in white
noise, so that the true posterior is a Gaussian whose Fisher matrix is known.

The three posteriors come in three formats:

- `mcmc`, the reference, is a run of an eryn ensemble sampler stored in an HDF5
  file (`format: eryn`). The pipeline drops its burn-in and thins it by its
  autocorrelation time.
- `sbi` is a flat table of draws in a parquet file (`format: table`), from the same
  Gaussian shifted by 0.3 posterior widths, as a simulation-based inference method
  might return it.
- `peer` is a flat group of draws in an HDF5 file (`format: hdf5`), with one dataset
  per parameter, from the unshifted Gaussian.

`config.yaml` runs `diagnose` on all three posteriors, and `compare` and
`compare_signals` of `sbi` and `peer` against `mcmc`. The generator, the noise
model and the eryn priors that the config names live in `models.py`, next to the
config. Data space gives two rows per pair: the data row, and the Fisher chart
row, which compares the posteriors in the local chart of the Fisher matrix at
the centre `data/centre.npy`. The generator supplies the Fisher matrix.

## Run it

The example needs the development environment, which installs eryn and h5py (the
`eryn` and `hdf5` extras). From the repository root:

```bash
uv run python examples/pipeline/make_data.py
uv run mixed-signals run examples/pipeline/config.yaml --dry-run
uv run mixed-signals run examples/pipeline/config.yaml
```

The first command writes the data files (the three posteriors and the centre of
the Fisher chart) into `examples/pipeline/data/`. The
second prints the estimated run time of every engine and stops. The third runs the
engines, which takes a few minutes, and writes the run folder
`examples/pipeline/results/`. Neither folder is committed.

## The run folder

The run folder holds five files and the folder `samples/`:

- `results.parquet` has one row per statistic of every engine, in the order
  `diagnose`, `compare`, `compare_signals`.
- `config.json` is the validated config with every default filled in and the
  resolved parameter space, without the output folder and the budget, which change
  no row. Its SHA-256, which `sha256sum config.json` prints, is the config hash
  that every result row records.
- `reductions.json` records how each run of chains was reduced: burn-in, thinning,
  autocorrelation times and effective sample size.
- `cost.parquet` is the table of the dry-run estimate.
- `provenance.json` records the versions, the machine, the start time, the wall
  time, the file and SHA-256 of every input and the file and SHA-256 of every
  module that the config names.
- `samples/` holds the flat samples that the engines compared, one `.npz` archive
  per posterior in the order of the config (`0.npz` for `mcmc`, the reduction of
  its chains), which `SampleSet.from_npz` reads back. A section with a transform
  adds `samples/diagnose/` or `samples/compare/` with the mapped copies and their
  `space.yaml`; this example has no transform. The report needs nothing but this
  folder.

## The report

The report turns a run folder into a notebook of tables and plots: the diagnostics
of each posterior, the comparison of each pair with a heatmap of the effects, the
corner plots, the null of every row, the warnings and the inputs. It needs the
`report` extra (marimo and cantuccio), which the development environment installs.

```bash
uv run mixed-signals report examples/pipeline/results
```

The command writes three files into the run folder: `report.py`, a marimo notebook
that reads the folder it sits in, `report.json`, its settings, and `report.html`,
a page of its outputs without code. The figures and tables are embedded in the
page, but marimo loads the front end that shows them from `cdn.jsdelivr.net`, so
the page renders in a browser only with an internet connection. Open the notebook
with `uv run marimo edit examples/pipeline/results/report.py`. A second call stops
unless you add `--overwrite`, and `mixed-signals run --overwrite` removes the
report of the run that it replaces. The options `--problems` and `--plot-cap` set
how many problems and how many samples per posterior the report shows; see
`mixed-signals report --help`.

## The Sinkhorn curve

`epsilon.yaml` runs `compare` alone, with the Sinkhorn divergence at six values of
the entropic regularisation ε, to show how the divergence and its null depend on
it. Run it and report it:

```bash
uv run mixed-signals run examples/pipeline/epsilon.yaml
uv run mixed-signals report examples/pipeline/results_epsilon
```

The report then draws, for each pair, S_ε against ε/ℓ² on a logarithmic axis with
the band of its null, and σ_eff against ε/ℓ² with the threshold. Its run folder,
`examples/pipeline/results_epsilon/`, is not committed either.

## A config runs code

The names `models:priors`, `models:generator` and `models:noise` are import paths.
While the pipeline runs, the folder of the config comes first on `sys.path`, so
`models.py` imports like a script. A config therefore runs the code that it names:
run only configs that you trust. The config format is described in the API
reference of `mixed_signals.pipeline.config`.
