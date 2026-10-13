# mixed signals: Architecture Design

> Exported from the claude.ai design doc on 2026-10-13, after design round 8. This file is now the working copy: update it when a decision changes.

mixed signals checks the quality of single posteriors and compares posteriors from different analyses of the same data, reporting every number against a finite-sample null. This draft records design rounds 1–8: **Decided** items are agreed, **Proposed** items await confirmation, and open questions are collected near the end.

## At a glance

```mermaid
flowchart TD
    chains["Chains<br/>eryn axis convention<br/>cold chain, log L, log π"]
    flat["Flat samples<br/>(N, D) per sampler<br/>optional weights, log L, log π"]
    models["User models (optional)<br/>priors, generators, noise"]
    diagnose["diagnose<br/>τ, ESS, Gaussianity,<br/>bananas, prior railing"]
    sampleset["SampleSet in a space<br/>(B, K, N, D), log_weights<br/>ParamSpace, shared whitening"]
    subgraph engines["Comparison engines"]
        compare["compare<br/>modes A and B"]
        reweight["reweight<br/>prior or log L"]
        dataspace["data space<br/>generator, noise"]
    end
    result["One result table<br/>value, uncertainty, null, σ_sig, σ_eff, warnings, provenance<br/>one row per problem, pair, metric, estimator and space"]
    reports["Reports<br/>polars to parquet, provenance<br/>marimo notebook, cantuccio plots"]

    chains --> diagnose
    diagnose -- "burn-in, thin by τ" --> sampleset
    flat --> sampleset
    sampleset --> engines
    models -- "if supplied" --> engines
    diagnose -- "diagnostic rows" --> result
    engines --> result
    result --> reports
```

Diagnostics read raw chains and supply τ for thinning; every engine then writes rows of the same shape to one table.

## Purpose and scope

v1 covers fixed-dimensional posteriors in two comparison modes. The primary field is gravitational-wave data analysis, but the core stays domain-agnostic.

- **Mode A, approximation vs trusted reference** (e.g. SBI against MCMC). Asymmetric: the reference fixes the KL direction and the whitening.
- **Mode B, peers without truth** (e.g. two pipelines on the same LISA or LVK data). Symmetric.

In v1:

- Single-posterior quality: MCMC health, shape, dependence, prior railing.
- Posterior-to-posterior comparison in modes A and B.
- Prior and likelihood reweighting, when priors and likelihoods are supplied.
- Data-space comparison, when a signal generator and noise model are supplied.

## Decisions so far

Fifty-nine decisions are agreed after rounds 1–8 and the repository setup on 2026-10-13.

| Topic | Decision |
| --- | --- |
| Comparison modes | A (approximation vs reference) and B (peers); ensemble calibration later |
| Dimensionality | Fixed-dimensional only; trans-dimensional is future work |
| Inputs | General arrays: one posterior (N, D), K posteriors of the same data, or batches of either; eryn-style inputs are not assumed |
| Batch axis | Independent problems; identical parameter sets across all inputs |
| Multi-source runs | eryn leaves are not batch entries, because of label switching |
| Sample counts | Equal N first; unequal N anticipated in the design |
| Number of posteriors | Kept small; warnings on combinatorial cost |
| Replicate runs | Not supported; possibly never |
| Chain layout | eryn's axis convention; v1 reads the cold chain of a single branch |
| Flat samples | A separate flat container, not squeezed into the chain layout |
| Chain reduction | Built-in burn-in and τ-thinning tools, optional in pipelines and configs; direct metric calls assume cleaned samples |
| Per-sample data | Raw or cleaned chains with log L and log π; nested-sampling weights optional |
| Autocorrelation | Own τ estimator; eryn and emcee serve as reference examples |
| Backend | JAX is a hard dependency; converters to and from NumPy, CuPy and PyTorch |
| Precision | float64 by default, enabled at import |
| Randomness | Every stochastic function takes an explicit JAX PRNG key; pipelines derive all keys from one config seed |
| Generators | Batched, backend-agnostic interfaces wrapping user code; full contract benched |
| Gradients | JAX generators use `jax.grad` of `get_ll`; CuPy or PyTorch generators supply `get_ll_grad` |
| Fisher matrix | `information_matrix` returns (N, D, D) |
| Noise | Set at generator initialization, or passed as a kwarg |
| `get_swap_ll` | Dropped |
| Priors | Separate objects, using eryn's prior setup from the l2d-dev branch |
| eryn release | The l2d-dev prior setup is headed for release; until then, development pins a uv git source |
| ParamSpace | Set explicitly; a class method builds it from eryn priors |
| Parametrization | With eryn priors, comparing in physical parameters via eryn's transforms is optional |
| Fixed and discrete parameters | With eryn priors, `DeltaFunction` parameters are dropped and discrete priors stay out of v1; otherwise nothing is dropped |
| Periodic parameters | A non-Euclidean space inside optimal transport |
| Sky position | A coupled sphere topology with great-circle OT costs |
| Periodic whitening | For now, whiten the Euclidean block only and scale periodic axes by circular spread; optional |
| Nulls | Every comparison ships with a finite-sample null |
| Null construction | Configurable; each method has a standard default |
| Null defaults | 200 resamples, both sets subsampled to the smaller N_eff; more resamples for significance claims |
| O(N²) metrics | Subsample above a cap with a warning; a kwarg sets the cap |
| σ reporting | Report σ_eff and σ_sig; pass/fail on σ_eff, with σ_sig checking it is resolved |
| Whitening | An explicit choice, recorded with every result |
| Validation zoo | Shifted and rotated Gaussians, bananas, multimodal mixtures, periodic and sphere cases; used for tests and σ_eff calibration tables |
| Optimal transport | Method choices owned by Coen; ott-jax is the candidate backend |
| Data structures | Own design on pydantic, polars and JAX pytrees |
| Visualization | cantuccio for corner plots; marimo for interactive notebooks |
| Users | Notebooks, importable metrics, and config + CLI pipelines that write reports |
| Config format | YAML |
| CLI | argparse |
| Packaging | One package with extras; Python ≥ 3.12; Apache 2.0 license |
| Dependencies | Lean core: jax, ott-jax, equinox, jaxtyping, numpy, polars, pydantic, pyyaml; eryn is an extra, and the dev group installs every extra |
| Tooling | uv ≥ 0.13 manages the project; ruff, strict mypy, a style-rule script, pre-commit hooks and CPU-only GitHub Actions CI check every change; CI pins its actions by commit SHA |
| Package layout | The tree under Package layout, with `io/eryn_backend.py` and `priors/eryn_container.py` |
| MCMC tools | `mcmc/` (τ, ESS, reduction) stays separate from `diagnostics/` (posterior shape) |
| Kernels and engines | Kernels live in topic subpackages; SampleSet-level wrappers live in `engines/` |
| OT location | `metrics/ot/`, so users find every metric in one place |
| Pytrees | equinox modules; equinox already arrives through ott-jax and lineax |
| Shape annotations | jaxtyping on every kernel, with beartype runtime checks in tests |
| Shape names | jaxtyping dimensions are words: batch, posterior, sample and param for (B, K, N, D); chains add step, temp, walker and leaf |
| Docs | Sphinx with the furo theme and MyST pages; Google-style docstrings, rendered by napoleon |
| GPU tests | A marked suite, run by hand |
| Engine API | Plain functions that return results |
| Distribution name | `mixed-signals` on PyPI, free but not reserved on 2026-10-13; the GitHub repository takes the same name |
| Thresholds | Per-metric σ_eff defaults ship with the package, overridable in the YAML |
| GPUs | A single GPU in v1; sharding the batch axis comes later |
| Report contents | As proposed: σ_eff and σ_sig summary table, corner overlays, null histograms and diagnostics; additions maybe later |

## Design principles

Eight principles shape every module; the Status column marks which are agreed.

| Principle | What it means | Status |
| --- | --- | --- |
| Every number has a recipe | Computed by a (functional, estimator, space) triple, e.g. KL × kNN × pooled-whitened; all three are recorded | Proposed |
| Nulls are a core service | Each comparison ships with the metric's distribution when both inputs share one distribution at the same N_eff; MCMC samples are τ-thinned first | Decided |
| Two σ's | σ_eff is the size of a difference in posterior widths; σ_sig is how surely it is non-zero; thresholds use σ_eff | Decided |
| Whitening is a choice of space | Both inputs always share one whitening matrix, recorded with the result | Decided |
| Topology is first-class | Each parameter is real, interval, periodic or spherical; estimators, transforms and OT costs respect it | Proposed |
| Transforms track Jacobians | Bijectors carry log\|J\| for density metrics and reweighting, and full Jacobians to map Fisher matrices: Γ′ = JᵀΓJ | Proposed |
| Capability-based dispatch | Supplied priors and implemented generator methods decide which methods can run | Decided |
| Domain-agnostic core | GW conventions, readers and data-space helpers live in a gw subpackage | Proposed |

## Data model

Comparisons run on flat samples shaped (B, K, N, D); diagnostics also read chains, which follow eryn's axis convention. Inputs are general arrays; eryn objects are one adapter, not an assumption. B counts independent problems, K samplers, N samples and D parameters. In code, jaxtyping annotations spell these axes as `batch`, `posterior`, `sample` and `param`.

**Input shapes**

- One posterior: (N, D).
- K posteriors of the same data: stacked to (K, N, D).
- Batches: (B, K, N, D), with `log_weights` shaped (B, K, N).
- Unequal N (later): pad to a common N with zero-weight samples. Kernels honor weights from day one, so the API does not change.

**Chains (eryn layout)**

- Per branch: chain (nsteps, ntemps, nwalkers, nleaves, ndim); log_like and log_prior (nsteps, ntemps, nwalkers); betas (nsteps, ntemps).
- v1 reads the cold chain (β = 1) of a single branch. Runs with several leaves wait for the catalog work, because leaves are not batch entries. Diagnostics read chains directly. Comparisons use flat samples from an explicit, recorded reduction: cold chain, burn-in, thinning by τ. The package ships these tools; pipelines take reduction as an optional argument, and direct metric calls assume cleaned samples.
- Proposed: eryn walkers are coupled, so R̂ across walkers is not used. Decided: τ comes from our own estimator of walker-averaged autocorrelation functions, with eryn's and emcee's estimators as reference examples.

**Core objects (Proposed; a separate flat container is Decided)**

| Object | Holds |
| --- | --- |
| `ParamSpace` | Names, labels, units, topology and bounds, shared by every input to a comparison |
| `Chains` | eryn-layout arrays with log densities and betas |
| `SampleSet` | Flat samples, `log_weights`, optional per-sample `log_like`, `log_prior` and `log_q`; generic, so priors and approximations reuse it |
| `Transform` | Composable bijector: forward, inverse, log\|J\| and Jacobian |
| `Estimator` | Gaussian, bounded KDE, kNN, GMM, classifier density ratio, normalizing flow |
| `Result` | One row per problem, pair, metric, estimator, space and parameter subset |

**Structure layering (Proposed)**

- equinox modules (Decided) for anything crossing jit: samples, weights, transforms, estimator state.
- pydantic for configs, `ParamSpace` and provenance; validated, and it drives the CLI.
- polars for results: one long tidy table, written to parquet.
- Metadata stays outside pytrees, because changing static fields triggers recompilation.

## Engines

Four engines share the data model and declare what they need; the method menus below are proposals to prune.

| Engine | Needs | Proposed v1 methods |
| --- | --- | --- |
| `diagnose` | Chains (flat samples: shape checks only) | τ, ESS, split-chain KS, acceptance and swap rates; Gaussianity via Mahalanobis distances vs χ²_D, skewness, kurtosis, Anderson–Darling; Pearson, Spearman, distance correlation and mutual information vs Gaussian values (bananas); GMM-BIC multimodality; prior railing |
| `compare` | Two or more SampleSets in one space | Mean shift and covariance discrepancy; Gaussian KL and W2 split into location and shape; KS, k-sample Anderson–Darling, 1D W1, JS with bounded KDE; kNN KL, MMD, energy distance, C2ST; sliced W2, Sinkhorn divergence; parameter-shift probability |
| `reweight` | Stored log π plus a new prior, stored log L plus a new likelihood, or an approximation's density q | Importance weights; evidence ratio Z_new/Z_old = E_old[L_new/L_old]; ESS and PSIS k̂ on every weight set, flagged above 0.7 |
| Data space | A generator with `generate_template`, plus a noise model | Energy distance with the noise-weighted norm; Fisher-whitened parameter distances |

**Comparison topologies (Proposed)**

- Against a reference: mode A default, K − 1 comparisons.
- All pairs: mode B default, K(K − 1)/2 comparisons.
- Leave-one-out against the pooled rest: K comparisons; flags the odd one out.

A dry run counts metric evaluations × null resamples before anything runs, and warns above a budget. O(N²) metrics (MMD, energy distance, Sinkhorn, brute-force kNN) subsample above a cap with a warning, and a kwarg sets the cap.

**Data space**

```math
D_E = 2\,\mathbb{E}\lVert s_A - s_B\rVert - \mathbb{E}\lVert s_A - s_A'\rVert - \mathbb{E}\lVert s_B - s_B'\rVert, \qquad \lVert a\rVert^2 = \langle a \mid a \rangle
```

Distances are in noise-σ units, and ‖δh‖ ≲ 1 is indistinguishable in the data. At linear order ‖δh‖² ≈ ΔθᵀΓΔθ, so Fisher whitening gives parameter distances the same meaning. With trans-dimensional support, comparing summed catalog signals this way avoids cross-matching.

## Statistical reporting

Every result row carries a value, its uncertainty, its null, σ_sig and σ_eff; thresholds act on σ_eff alone, with per-metric defaults that the YAML config can override.

**Result columns:** problem, pair, metric, estimator, space (transform chain), parameters, value, uncertainty, null mean, null std, p-value, σ_sig, σ_eff, warnings, provenance (version, seed, input hashes).

**σ_eff, the Gaussian-equivalent shift (Proposed).** σ_eff is the mean shift between two unit Gaussians that gives the same null-subtracted metric value, in units of the whitening σ.

| Metric | σ_eff |
| --- | --- |
| Whitened W2 | The value itself, for a pure location shift |
| KL (nats) | √(2·KL) |
| C2ST accuracy | 2Φ⁻¹(acc); a lower bound, since real classifiers are sub-optimal |
| Hellinger | √(−8 ln(1 − H²)) |
| JS | Tabulated monotone curve, independent of D |
| MMD | Calibrated at run time; depends on D and kernel bandwidth |

**σ_sig.** The null's empirical p-value, converted to a one-sided σ. 1000 resamples resolve at most about 3.1σ; beyond that a fitted null tail is used and flagged as extrapolated.

**Null construction (Decided: configurable, with a standard default per method: 200 resamples, both sets subsampled to the smaller N_eff, more for significance claims)**

- Mode A default (Proposed): split the reference into halves at the approximation's N_eff.
- Mode B default (Proposed): permute τ-thinned pooled samples.

**Validity domains (Proposed).** Each metric declares where it is reliable: minimum N_eff relative to D, bandwidth and ε relative to the whitened scale, maximum D for KDEs. Violations land as structured warnings in the result table.

## Validation zoo

Distributions with known answers serve both as the test suite and as the source of the σ_eff calibration tables.

- **Shifted and rotated Gaussians:** closed-form KL, W2 and Mahalanobis shifts.
- **Bananas:** non-linear dependence with Gaussian-matched moments, which moment-based metrics miss.
- **Multimodal mixtures:** separated and overlapping modes, for multimodality checks and mode-dropping comparisons.
- **Periodic cases:** distributions straddling the period boundary.
- **Sphere cases:** sky-position distributions near the poles.

Proposed: the zoo is importable, so users can test their own metrics against it.

## Likelihood layer and generators

Generators are batched, backend-agnostic interfaces that users implement around their own code; the full contract is benched for now.

| Method | Returns |
| --- | --- |
| `generate_template(θ)` | Signals in the data domain, one per sample |
| `get_ll(θ)` | Log-likelihoods, shape (N,) |
| `get_ll_grad(θ)` | Gradients, shape (N, D); JAX generators default to `jax.grad` of `get_ll`, CuPy or PyTorch generators supply it |
| `information_matrix(θ)` | Fisher matrices, shape (N, D, D) |

- θ is batched, shape (N, D).
- Noise is set at initialization, or passed as a kwarg to the methods that need it.
- The pipeline never assumes user code is JAX: it calls generators at the boundary, in chunks, with zero-copy DLPack hand-off on GPU. jit covers the metric kernels.
- Which methods a generator implements decides which engines can run.
- Priors are separate objects; see Priors and parameter space.

## Priors and parameter space

Priors come from eryn's prior setup on the l2d-dev branch, and `ParamSpace` is built from them explicitly.

- **Priors:** an `eryn.priors.ProbDistContainer` keyed by parameter name, or by a tuple of names for joint priors. `logpdf` takes (N, D) and returns (N,); `rvs(size)` returns size + (D,). Conditional priors and CuPy are supported. The l2d-dev setup is headed for release.
- **ParamSpace:** set explicitly; a class method such as `ParamSpace.from_eryn_priors(priors)` reads names from `key_order` and each prior's label, unit, bounds and `boundary`.
- **Topology (Proposed mapping):** `periodic` becomes a circle with period maximum − minimum; `reflective` an interval with reflecting edges for bounded KDEs; no boundary an interval when both bounds are finite, else the real line.
- **Sky position (Decided):** coupled pairs such as (λ, β) are declared as a sphere, with great-circle OT costs.
- **Fixed and discrete parameters (Decided):** with eryn priors, `DeltaFunction` parameters are dropped automatically and discrete priors stay out of v1; without eryn priors, nothing is dropped.
- **Transforms:** eryn priors carry physical ↔ sampling transforms with log Jacobians, which wrap directly as `Transform` objects (Proposed). With eryn priors, comparing in physical parameters is optional (Decided).

**Periodic parameters in OT (Decided).** Periodic axes form a non-Euclidean space: the transport cost uses the wrapped difference min(|Δ|, P − |Δ|) on each one. Decided, as the default for now and optional: whitening acts on the Euclidean block only, and periodic axes are scaled by their circular spread.

## Backend, numerics and packaging

The pipeline runs in JAX with float64 switched on at import; arrays convert to and from NumPy, CuPy and PyTorch at the edges.

- float64 is needed because galactic-binary frequency posteriors have σ_f/f ~ 10⁻⁸, below one float32 step.
- `jax_enable_x64` is process-global, so importing mixed signals switches it on for the host pipeline too; the documentation must say so.
- Every stochastic function takes an explicit JAX PRNG `key`; pipelines derive all keys from one config seed.
- v1 runs on a single GPU; sharding the batch axis B across devices comes later.
- eryn supplies the chain axis convention and priors through the optional `eryn` extra; ott-jax supplies optimal transport.
- One package, `mixed-signals` on PyPI, with extras, licensed Apache 2.0, managed with uv, for Python ≥ 3.12.
- [JAX 0.11.0](https://docs.jax.dev/en/latest/changelog.html) dropped Python 3.11 in July 2026, so the 3.12 floor keeps current JAX available.

## Interfaces

Three layers serve three uses: kernels for code, SampleSet functions for notebooks, and a config-driven CLI for pipelines (Proposed).

1. **Kernels:** pure JAX functions on arrays, importable one by one.
2. **SampleSet API:** `diagnose`, `compare`, `reweight` and data-space functions that add nulls, both σ's, warnings and provenance. They are plain functions that return results (Decided).
3. **Pipeline:** a pydantic-validated YAML config plus an argparse CLI that writes `results.parquet`, provenance and a generated marimo notebook, which runs as a script and exports to HTML.

Plots use cantuccio, with contours drawn from the same estimator settings as the metrics, so plots and numbers agree. The report holds a σ_eff and σ_sig summary table, corner overlays, null histograms and diagnostics.

Illustrative only; every name is a placeholder:

```python
import jax
import mixed_signals as ms

key = jax.random.key(0)
space = ms.ParamSpace.from_file("params.yaml")
chains_a = ms.io.from_eryn("pipeline_a.h5")
chains_b = ms.io.from_eryn("pipeline_b.h5")

ms.diagnose(chains_a, space=space)

samples_a = ms.reduce(chains_a, burn="auto", thin="tau")
samples_b = ms.reduce(chains_b, burn="auto", thin="tau")
res = ms.compare(
    [samples_a, samples_b],
    space=space,
    whiten="pooled",
    metrics=["sinkhorn", "c2st", "kl_knn"],
    key=key,
)
res.table  # polars DataFrame
```

From the shell: `mixed-signals run config.yaml`.

## Package layout

The tree below is agreed.

```text
mixed-signals/
├── pyproject.toml          # uv_build, Python ≥ 3.12, Apache-2.0, extras
├── uv.lock
├── .pre-commit-config.yaml # ruff, style rules, lockfile check
├── .github/workflows/      # CI: lint, strict mypy, CPU tests, docs build
├── src/mixed_signals/
│   ├── __init__.py         # turns on x64, re-exports the public API
│   ├── _config.py          # x64 switch and check, package defaults
│   ├── errors.py           # warning and exception classes
│   ├── prng.py             # one seed → per-call JAX keys
│   ├── core/
│   │   ├── space.py        # ParamSpace
│   │   ├── topology.py     # Real, Interval, Circle, Sphere
│   │   ├── samples.py      # SampleSet: (B, K, N, D) + log_weights
│   │   ├── chains.py       # Chains: eryn axes, cold chain, one branch
│   │   └── result.py       # result schema and provenance
│   ├── io/
│   │   ├── arrays.py       # NumPy, CuPy, torch ↔ JAX via DLPack
│   │   ├── eryn_backend.py # eryn backends → Chains
│   │   └── weights.py      # nested-sampling weights → log_weights
│   ├── priors/
│   │   ├── protocol.py     # minimal prior interface: logpdf, rvs
│   │   └── eryn_container.py  # eryn priors → ParamSpace, Transforms
│   ├── transforms/
│   │   ├── base.py         # Transform: forward, inverse, log|J|, Jacobian
│   │   └── whiten.py       # reference, pooled, Fisher; Euclidean block
│   ├── estimators/         # Gaussian, bounded KDE, kNN, GMM, classifier, flow
│   ├── mcmc/
│   │   ├── autocorr.py     # own τ estimator
│   │   ├── ess.py          # ESS, split-chain KS
│   │   └── reduce.py       # burn-in, τ-thinning → SampleSet
│   ├── diagnostics/        # shape, dependence, modes, railing
│   ├── metrics/            # pure JAX kernels
│   │   ├── _cap.py         # O(N²) subsampling cap and warning
│   │   ├── moments.py      # mean shift, covariance, Gaussian KL and W2
│   │   ├── marginal.py     # KS, AD, 1D W1, JS
│   │   ├── divergence.py   # kNN KL and JS
│   │   ├── kernel.py       # MMD, energy distance
│   │   ├── classifier.py   # C2ST
│   │   ├── tension.py      # parameter-shift probability
│   │   └── ot/             # circle and sphere costs, Sinkhorn, sliced W2
│   ├── reweight/           # importance weights, evidence ratio, PSIS
│   ├── dataspace/          # noise-weighted energy distance, Fisher
│   ├── generators/         # benched: protocols, capability checks
│   ├── nulls/              # split-half, permutation, N_eff matching, defaults
│   ├── sigma/              # σ_sig, σ_eff maps, calibration tables
│   ├── engines/            # diagnose, compare, reweight, dataspace
│   ├── zoo/                # validation distributions with known answers
│   ├── pipeline/           # YAML config models, runner, dry-run cost
│   ├── report/             # marimo notebook, cantuccio plots
│   ├── cli.py              # argparse: mixed-signals run config.yaml
│   └── gw/                 # GW conventions, readers, waveform helpers
├── scripts/                # style-rule check; rebuild σ_eff tables from the zoo
├── tests/                  # mirrors src/, zoo validation, GPU suite run by hand
├── examples/               # marimo notebooks
└── docs/                   # Sphinx
```

Decided at setup: core dependencies are jax, ott-jax, equinox, jaxtyping, numpy, polars, pydantic and pyyaml. jaxtyping is listed directly because every kernel imports it. eryn is the `eryn` extra, because its runtime dependencies include matplotlib, pandas, seaborn and pytest, and only `io/eryn_backend.py` and `priors/eryn_container.py` use it; both import it lazily. The other extras so far are plot (cantuccio) and report (marimo, cantuccio); the gw, cupy and torch extras arrive with their modules. The dev dependency group installs every extra, beartype sits in the test group, and Sphinx in the docs group.

## Open questions

No questions remain after round 8; one task is open.

- [ ] Prune the proposed method menus in the Engines table.

## Deferred

Six items wait until the v1 core is stable; replicate runs may never come.

- **Trans-dimensional catalogs:** cross-matching and label switching, including multi-leaf eryn runs; comparing summed signals in data space avoids matching for the global check.
- **Injection-ensemble calibration:** PP plots, SBC, TARP; needs an ensemble container with true parameters.
- **Generator contract:** the full interface users implement around their waveform and likelihood code.
- **Unequal N:** zero-weight padding in the existing `log_weights` field, so the API does not change.
- **Multi-GPU:** sharding the batch axis B across devices.
- **Replicate runs:** not at this stage, possibly never; they would add a replicate axis and run-to-run nulls.
