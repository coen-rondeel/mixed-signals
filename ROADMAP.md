# mixed signals: Roadmap

The implementation is split into twenty focused sessions, numbered 0 to 19. They run one at a time, and each ends in one commit on `main` that Coen reviews. Session 0 settles the open design items first, so the later sessions build on decided ground. Sessions in the same wave do not depend on each other, so their order is free.

## Status

| # | Session | Wave | Needs | Status |
| --- | --- | --- | --- | --- |
| Setup | Repository scaffold, tooling and CI | | | Done (2026-10-13) |
| 0 | Design close-out | A | | Next |
| 1 | Foundations | A | 0 | Open |
| 2 | Sample containers | A | 1 | Open |
| 3 | Validation zoo | B | 2 | Open |
| 4 | Transforms and whitening | B | 2 | Open |
| 5 | MCMC tools | B | 2 | Open |
| 6 | Estimators | B | 2 | Open |
| 7 | eryn adapters and priors | C | 2, 4 | Open |
| 8 | Metrics I: moments and marginals | C | 3, 6 | Open |
| 9 | Metrics II: divergences, kernels and classifiers | C | 3, 6 | Open |
| 10 | Optimal transport | C | 3, 4 | Open |
| 11 | Diagnostics | C | 5, 6 | Open |
| 12 | Nulls and σ | D | 8 | Open |
| 13 | Reweighting | D | 7 | Open |
| 14 | Engines | E | 8 to 12 | Open |
| 15 | Data space | E | 4, 9 | Open |
| 16 | Pipeline and CLI | F | 14 | Open |
| 17 | Reports | F | 14 | Open |
| 18 | GW subpackage | F | 7 | Open |
| 19 | Release readiness | G | all | Open |

The critical path runs through sessions 0, 1, 2, 3 or 6, 8, 12, 14, 16 and 19. Session 8 needs both 3 and 6, so either one can delay it.

## How every session runs

1. Read `CLAUDE.md`, `DESIGN.md` and the session's brief below.
2. Confirm with Coen any Proposed item the session relies on. After session 0 there should be none.
3. Replace the stubs in scope with working code. Tests mirror `src/` and use the validation zoo once it exists.
4. Where a brief names catpylisa code, port it: adapt it to this package's data model, docstrings and style rules instead of copying it. The code sits next to this repository in Coen's `catpylisa_dev` workspace.
5. Run every check: `ruff check`, `ruff format --check`, the style script, `mypy`, `pytest` and the docs build.
6. Update `DESIGN.md` when a decision changes, and the status table above.
7. Commit once on `main` with the trailers from `CLAUDE.md`, then hand over to Coen for review.

## Sessions

### 0. Design close-out

Coen decides and Claude prepares the options with their trade-offs. There is no code; the session can also run on claude.ai, like the earlier design rounds.

- Prune the method menus of the four engines.
- Settle every item still marked Proposed in `DESIGN.md`: the four Proposed design principles (recipes, topology, Jacobians, a domain-agnostic core), the core objects and their fields, the structure layering, the comparison topologies, the σ_eff maps, the null defaults for modes A and B, validity domains, an importable zoo, the eryn topology mapping, eryn transforms as `Transform` objects, dropping R̂ across walkers, and the three interface layers.
- Update the briefs below where the pruning or a decision changes their scope.
- Done when `DESIGN.md` marks every item as Decided, changed or deferred.

### 1. Foundations

Scope: `errors.py`, `prng.py`, `_config.py` (package defaults), `core/topology.py`, `core/space.py`, `core/result.py`.

- Warning and exception classes, designed so that warnings land as structured entries in the result table.
- One config seed gives every stochastic call its own JAX key; propose the derivation scheme to Coen.
- Topologies: real line, interval, circle with its period, and sphere for coupled pairs such as sky position.
- `ParamSpace`: names, labels, units, topology and bounds, as a pydantic model that loads from YAML.
- The result schema in polars: problem, pair, metric, estimator, space, parameters, value, uncertainty, null mean, null standard deviation, p-value, σ_sig, σ_eff, warnings and provenance (package version, seed, input hashes).
- Done when a `ParamSpace` round-trips through YAML, invalid spaces fail with clear messages, and an empty result table has the agreed schema.

### 2. Sample containers

Scope: `core/samples.py`, `core/chains.py`, `io/arrays.py`, `io/weights.py`.

- `SampleSet`: an equinox module with samples shaped (batch, posterior, sample, param), `log_weights` shaped (batch, posterior, sample), and optional `log_like`, `log_prior` and `log_q`. It is built from one posterior (N, D), K posteriors (K, N, D) or batches (B, K, N, D).
- Metadata stays outside the pytree, because changing static fields triggers recompilation.
- `Chains`: eryn's axis layout (step, temp, walker, leaf, param) with log-likelihood, log-prior and inverse temperatures; it selects the cold chain of one branch and rejects several leaves for now.
- Conversion between JAX and NumPy, CuPy or PyTorch arrays through DLPack, zero-copy on GPU. CuPy and PyTorch tests skip when the package is missing.
- Nested-sampling weights become `log_weights`.
- Done when every input shape builds the same container, weights survive every operation, and conversions round-trip.

### 3. Validation zoo

Scope: `zoo/`.

- Shifted and rotated Gaussians with closed-form KL, W2 and Mahalanobis shifts.
- Bananas with Gaussian-matched moments, which moment-based metrics miss.
- Separated and overlapping multimodal mixtures.
- Periodic cases that straddle the period boundary, and sphere cases near the poles.
- Each case draws `SampleSet`s from an explicit key and carries its analytic reference values.
- Done when every case reproduces its reference values within sampling error.

### 4. Transforms and whitening

Scope: `transforms/base.py`, `transforms/whiten.py`.

- `Transform`: forward and inverse maps, the log-determinant of the Jacobian and the full Jacobian; transforms compose.
- Whitening from the reference, the pooled samples or a Fisher matrix. It acts on the Euclidean block, and periodic axes are only scaled by their circular spread. Both inputs of a comparison share one whitening, recorded with the result.
- Port from catpylisa: `transformations.py` (z-score, empirical whitening, Fisher projection), and the geometry schema in `containers.py` of the same module.
- Done when Jacobians match `jax.jacfwd`, inverses round-trip, Fisher matrices map as Γ′ = JᵀΓJ, and no periodic axis is mixed with another axis.

### 5. MCMC tools

Scope: `mcmc/autocorr.py`, `mcmc/ess.py`, `mcmc/reduce.py`.

- Our own integrated autocorrelation time τ from walker-averaged autocorrelation functions, checked against emcee's and eryn's estimators.
- Effective sample size and split-chain Kolmogorov-Smirnov tests.
- Reduction: burn-in removal and thinning by τ, producing a `SampleSet` and recording each step.
- Done when τ matches the analytic value for AR(1) chains, and a reduction can be repeated from its record.

### 6. Estimators

Scope: `estimators/`.

- Gaussian, bounded KDE (reflecting edges on intervals, wrapping on circles), kNN and GMM. Classifier and flow estimators only if session 0 keeps them.
- Estimator settings must be reusable by the plots, so that contours and metrics agree.
- Done when the estimators recover zoo densities within tolerance and respect each topology.

### 7. eryn adapters and priors

Scope: `io/eryn_backend.py`, `priors/protocol.py`, `priors/eryn_container.py`.

- Read eryn backends into `Chains`: the cold chain of one branch.
- The minimal prior protocol: `logpdf` maps (N, D) samples to (N,) values, and `rvs(size)` returns size + (D,).
- `ParamSpace.from_eryn_priors`: names from `key_order`, and each prior's label, unit, bounds and boundary. `DeltaFunction` parameters are dropped and discrete priors are rejected.
- Wrap eryn's physical-to-sampling transforms, with their log Jacobians, as `Transform` objects.
- Pitfalls: import eryn inside functions only, so `import mixed_signals` works without it; use `eryn.priors`, because `eryn.prior` is deprecated.
- Done when a short eryn run round-trips into `Chains` and a `ParamSpace`.

### 8. Metrics I: moments and marginals

Scope: `metrics/_cap.py`, `metrics/moments.py`, `metrics/marginal.py`.

- A subsampling cap for metrics that scale as N², with a warning and a keyword that sets the cap.
- Mean shift and covariance discrepancy; Gaussian KL and W2, split into location and shape terms.
- Marginals: Kolmogorov-Smirnov, k-sample Anderson-Darling, one-dimensional W1, and Jensen-Shannon with a bounded KDE.
- Every kernel is pure JAX, honours `log_weights` and carries jaxtyping annotations.
- Done when the zoo's closed-form values are recovered.

### 9. Metrics II: divergences, kernels and classifiers

Scope: `metrics/divergence.py`, `metrics/kernel.py`, `metrics/classifier.py`, `metrics/tension.py`, limited to the methods session 0 keeps.

- kNN estimates of KL and Jensen-Shannon divergences.
- Maximum mean discrepancy and energy distance, under the cap from session 8.
- The classifier two-sample test (C2ST) and the parameter-shift probability.
- Done when each metric vanishes on identical distributions, within its null, and orders the zoo's shifts correctly.

### 10. Optimal transport

Scope: `metrics/ot/`. Coen owns the method choices, so the session opens with options, not a pick.

- Costs: the wrapped difference min(|Δ|, P − |Δ|) on periodic axes, and great-circle costs on sky-position spheres.
- The Sinkhorn divergence through ott-jax, and sliced W2.
- Port from catpylisa: `geometries.py` (chordal and geodesic sphere costs, composite cost, coupling-scale calibration) and `divergence.py` (adaptive ε, Sinkhorn divergence). Read `plans/ot_convergence.md` in the parent workspace for the ε-scaling notes.
- Pitfall: ott-jax imports jaxopt, which raises a DeprecationWarning; tests turn warnings into errors, so add a targeted `filterwarnings` entry for it.
- Done when the costs respect each topology and the divergences vanish on identical inputs.

### 11. Diagnostics

Scope: `diagnostics/`.

- Gaussianity: Mahalanobis distances against χ²_D, skewness, kurtosis and Anderson-Darling.
- Dependence: Pearson, Spearman, distance correlation and mutual information against their Gaussian values, to catch bananas.
- Multimodality through GMM-BIC, and prior railing.
- Chain health from session 5 (τ, effective sample size, split-chain tests, acceptance and swap rates) as diagnostic rows.
- Done when the zoo's bananas, mixtures and railed cases are flagged and its Gaussians are not.

### 12. Nulls and σ

Scope: `nulls/`, `sigma/`, and a calibration script in `scripts/`.

- Nulls: split-half of the reference at the approximation's N_eff for mode A, permutation of τ-thinned pooled samples for mode B, and N_eff matching. 200 resamples by default, configurable.
- σ_sig: the null's empirical p-value as a one-sided σ, with a fitted tail beyond the resampling resolution, flagged as extrapolated.
- σ_eff maps per metric, and per-metric threshold defaults that the YAML config can override.
- A script that rebuilds the σ_eff calibration tables from the zoo.
- Done when nulls on identical distributions give uniform p-values and σ_eff recovers the zoo's known shifts.

### 13. Reweighting

Scope: `reweight/` and its engine function.

- Importance weights for a new prior, a new likelihood or an approximation's density q.
- The evidence ratio Z_new/Z_old = E_old[L_new/L_old].
- Effective sample size and PSIS k̂ for every weight set, flagged above 0.7.
- Done when reweighting between zoo Gaussians recovers the analytic target and k̂ flags heavy-tailed weights.

### 14. Engines

Scope: `engines/` (`compare` and `diagnose`) and the public API in `__init__.py`.

- `compare` in modes A and B, against a reference, over all pairs or leave-one-out, with one shared whitening per comparison. Every row carries its null, both σ's, validity warnings and provenance.
- A dry run that counts metric evaluations times null resamples and warns above a budget.
- `diagnose` on raw chains and on flat samples.
- Re-exports in `__init__.py` go below the float64 switch.
- Done when a zoo comparison produces the full result table end to end.

### 15. Data space

Scope: `generators/`, `dataspace/` and the data-space engine function.

- The minimal generator protocol that data space needs (`generate_template` and a noise model), with capability checks. The full generator contract stays deferred.
- User code is called at the boundary in chunks, with DLPack hand-off on GPU; it may be CuPy or PyTorch.
- Energy distance with the noise-weighted inner product, and Fisher-whitened parameter distances.
- Reference only: catpylisa's `estimate_distances.py` (waveform distances with JaxGB). The catpylisa team wrote it, so read it rather than port it.
- Done when a toy generator gives distances in noise-σ units that match the linear-order Fisher prediction.

### 16. Pipeline and CLI

Scope: `pipeline/`, `cli.py`.

- pydantic models for the YAML config, and a runner that writes `results.parquet` and its provenance. All keys derive from one config seed.
- `mixed-signals run config.yaml`, including the dry-run cost estimate.
- Done when an example config runs from the shell and gives the same table again from the same seed.

### 17. Reports

Scope: `report/`.

- A generated marimo notebook that runs as a script and exports to HTML.
- Contents: the σ_eff and σ_sig summary table, cantuccio corner overlays with contours from the metric estimators, null histograms and diagnostics.
- Done when the example pipeline produces a report that renders.

### 18. GW subpackage

Scope: `gw/`, with a `gw` extra if it needs new dependencies.

- GW parameter conventions (names, units, sky-position pairs), readers and waveform helpers, so the core stays domain-agnostic.
- Done when a LISA galactic-binary example loads through `gw` into the core containers.

### 19. Release readiness

- A Sphinx user guide and API pages, and marimo examples in `examples/`.
- The GPU suite on the cluster with a single GPU, and a performance check.
- Release blockers: reserve `mixed-signals` on PyPI, and wait for eryn to release its new prior setup, because a published package cannot depend on a git branch.
- Done when version 0.1.0 can be published.
