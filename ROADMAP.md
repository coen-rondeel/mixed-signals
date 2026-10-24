# mixed signals: Roadmap

The implementation is split into twenty focused sessions, numbered 0 to 19. They run one at a time, and each ends in one commit on `main` that a developer reviews. Session 0 settles the open design items first, so the later sessions build on decided ground. Sessions in the same wave do not depend on each other, so their order is free.

## Status

| # | Session | Wave | Needs | Status |
| --- | --- | --- | --- | --- |
| Setup | Repository scaffold, tooling and CI | | | Done (2026-10-13) |
| 0 | Design close-out | A | | Done (2026-10-13) |
| 1 | Foundations | A | 0 | Done (2026-10-13) |
| 2 | Sample containers | A | 1 | Done (2026-10-13) |
| 3 | Validation zoo | B | 2 | Done (2026-10-13) |
| 4 | Transforms and whitening | B | 2 | Done (2026-10-14) |
| 5 | MCMC tools | B | 2 | Done (2026-10-14) |
| 6 | Estimators | B | 2 | Done (2026-10-14) |
| 7 | eryn adapters and priors | C | 2, 4 | Done (2026-10-24) |
| 8 | Metrics I: moments and marginals | C | 3, 6 | Done (2026-10-24) |
| 9 | Metrics II: divergences, kernels and classifiers | C | 3, 6 | Open |
| 10 | Optimal transport | C | 3, 4 | Open |
| 11 | Diagnostics | C | 5, 6 | Open |
| 12 | Nulls and σ | D | 8 | Next |
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
2. Confirm with the developers any Proposed or Open item the session relies on. After session 7, only the eryn pin is open: `pyproject.toml` names eryn's `feat-jaxpriors` branch, which is not pushed yet.
3. Replace the stubs in scope with working code. Tests mirror `src/` and use the validation zoo once it exists.
4. Cite every method and assumption: a `References` section in each docstring, with keys in `docs/references.bib`, taken from the References section of `DESIGN.md`. A new method needs a verifiable source before it lands, and a new empirical finding goes on the validation page.
5. Where a brief names catpylisa code, port it: adapt it to this package's data model, docstrings and style rules instead of copying it.
6. Run every check: `ruff check`, `ruff format --check`, the style script, `mypy`, `pytest` and the docs build; until the eryn branch is pushed, through `uv run --no-sync`, as `CLAUDE.md` describes.
7. Update `DESIGN.md` when a decision changes, and the status table above.
8. Commit once on `main` with the trailers from `CLAUDE.md`, then hand over to a developer for review.

## Sessions

### 0. Design close-out

The developers decide and Claude prepares the options with their trade-offs. There is no code.

- Prune the method menus of the four engines.
- Settle every item still marked Proposed in `DESIGN.md`: the four Proposed design principles (recipes, topology, Jacobians, a domain-agnostic core), the core objects and their fields, the structure layering, the comparison topologies, the σ_eff maps, the null defaults for modes A and B, validity domains, an importable zoo, the eryn topology mapping, eryn transforms as `Transform` objects, dropping R̂ across walkers, and the three interface layers.
- Update the briefs below where the pruning or a decision changes their scope.
- Done when `DESIGN.md` marks every item as Decided, changed or deferred.

Outcome (2026-10-13): every item is Decided or Deferred, except the form of eryn transforms, which the developers left open for session 7. The close-out also added a citation rule for every method and assumption, the r_eff width scale and the validation page; the briefs below include these changes.

### 1. Foundations

Scope: `errors.py`, `prng.py`, `_config.py` (package defaults), `core/topology.py`, `core/space.py`, `core/result.py`, and the citation setup in `docs/`.

- Warning and exception classes, designed so that warnings land as structured entries in the result table.
- One config seed gives every stochastic call its own JAX key; propose the derivation scheme to the developers.
- Topologies: real line, interval (one bound may be infinite), circle with its period, and sphere for coupled pairs such as sky position. A sphere takes an azimuth axis and a polar axis given as latitude, colatitude, sin(latitude) or cos(colatitude).
- `ParamSpace`: names, labels, units, topology and bounds, as a pydantic model that loads from YAML.
- The result schema in polars: problem, pair, metric, estimator, space, parameters, value, uncertainty, null mean, null standard deviation, null draws, null recipe, p-value, σ_sig, σ_eff, r_eff, warnings and provenance (package version, seed, input hashes, JAX version, device platform, config hash).
- Citations: add sphinxcontrib-bibtex to the docs group, create `docs/references.bib` and a bibliography page, and cite the sources of this session's modules.
- Done when a `ParamSpace` round-trips through YAML, invalid spaces fail with clear messages, an empty result table has the agreed schema, and the docs build renders a citation.

Outcome (2026-10-13): the developers decided seven items, now in `DESIGN.md`: named key derivation, the warning design, package defaults as constants, the flat `ParamSpace` YAML, SHA-256 hashes, the `Result` object and the citation style. The result schema gained the estimator settings, the uncertainty method and the PRNG key in provenance. Later sessions raise the warning classes of `errors.py` outside jit, derive every key with `prng.derive_key` from a path of names, and build rows with `Result.from_rows`.

### 2. Sample containers

Scope: `core/samples.py`, `core/chains.py`, `io/arrays.py`, `io/weights.py` (benched in this session).

- `SampleSet`: an equinox module with samples shaped (batch, posterior, sample, param), `log_weights` shaped (batch, posterior, sample), an optional `n_eff` shaped (batch, posterior), and optional `log_like`, `log_prior` and `log_q`. It is built from one posterior (N, D), K posteriors (K, N, D) or batches (B, K, N, D). Without `n_eff`, the Kish effective sample size of the weights applies.
- Metadata stays outside the pytree, because changing static fields triggers recompilation.
- `Chains`: eryn's axis layout (step, temp, walker, leaf, param) with log-likelihood, log-prior and inverse temperatures, plus optional acceptance and swap counts; it selects the cold chain of one branch and rejects several leaves for now.
- Conversion between JAX and NumPy, CuPy or PyTorch arrays through DLPack, zero-copy on GPU. CuPy and PyTorch tests skip when the package is missing.
- Nested-sampling weights become `log_weights`.
- Done when every input shape builds the same container, weights survive every operation, and conversions round-trip.

Outcome (2026-10-13): the developers decided seven items, now in `DESIGN.md`, and deferred two. `SampleSet` holds arrays only, so engines take the posterior names and the `ParamSpace`. It selects problems, posteriors and parameters, takes samples (dropping `n_eff`) and converts to NumPy, JAX, torch or CuPy; stacking sets is left out of v1. A given `n_eff` lies between 1 and the number of positive-weight samples, and inputs narrower than float64 raise a `PrecisionWarning`. `Chains` takes the eryn layout or (step, walker, param), with optional log densities, and counts that carry their proposals. A `torch` extra and conflicting `cuda12` and `cuda13` extras arrived; the dev group installs CPU-only torch. Nested-sampling weights and eryn runs with several independent samplers are deferred. Later sessions build containers through the constructors, which check values only outside jit, and compute the Kish value with `core.samples.kish_effective_sample_size`.

### 3. Validation zoo

Scope: `zoo/`, public as `mixed_signals.zoo`.

- Shifted and rotated Gaussians with closed-form KL, W2 and Mahalanobis shifts.
- Scaled Gaussians with known width ratios.
- Bananas with Gaussian-matched moments, which moment-based metrics miss.
- Separated and overlapping multimodal mixtures.
- Railed cases: posteriors truncated at a prior bound.
- Periodic cases that straddle the period boundary, and sphere cases near the poles.
- Each case draws `SampleSet`s from an explicit key and carries its analytic reference values.
- Done when every case reproduces its reference values within sampling error.

Outcome (2026-10-13): the developers decided fourteen items, now in `DESIGN.md`, and deferred weighted draws. A `ZooCase` holds K named posteriors, their `ParamSpace`, their densities and a polars reference table with the result columns `pair`, `metric` and `parameters`, so results join their truth; `sample(key, N, batch_size=B)` draws a `SampleSet` (B, K, N, D), each posterior from `derive_key(key, name)`. Reference values are closed forms, plus one-dimensional quadrature for the marginal JS and AD of Gaussian pairs and for mixture KL, all in JAX, so they jit, vmap and differentiate. The quadrature uses Gauss-Legendre panels refined around every component and every crossing of two components, which also gets right the narrow components that QUADPACK steps over; concrete invalid inputs raise, and inside jit they give NaN. SciPy became a core dependency, which the tests use as an independent check. The families: shifted, scaled and rotated (±α/2) Gaussians with an optional base mean and covariance, any K Gaussians, the twisted Gaussian with its moment-matched Gaussian and closed-form KL, a mixture against its moment-matched Gaussian, a dropped mode and reweighted modes, railed normals, wrapped normals across a circle boundary, von Mises-Fisher distributions over a pole, and AR(1) chains with known τ for sessions 5, 11 and 12. The tests reproduce every reference value from draws within five standard errors and fail on any 10 per cent error. Later sessions take their test data from `mixed_signals.zoo` and their truths from `case.reference_value`.

### 4. Transforms and whitening

Scope: `transforms/base.py`, `transforms/whiten.py`.

- `Transform`: forward and inverse maps, the log-determinant of the Jacobian and the full Jacobian, which defaults to `jax.jacfwd`; transforms compose.
- Whitening from the reference, the pooled samples or a Fisher matrix. It acts on the Euclidean block, and periodic axes are only scaled by their circular spread. Both inputs of a comparison share one whitening, recorded with the result.
- Port from catpylisa's `common_tools.distances.optimal_transport` module: `transformations.py` (z-score, empirical whitening, Fisher projection) and the geometry schema in `containers.py`.
- Settle with the developers what pooled whitening uses. The covariance of the pooled samples includes the spread between the means, which shrinks a Mahalanobis shift d to d/√(1 + d²/4), below 2 for every d; the average of the within-posterior covariances, as in Cohen's d, does not. The zoo states its shifts against the first posterior's covariance.
- Done when Jacobians match `jax.jacfwd`, inverses round-trip, Fisher matrices map as Γ′ = JᵀΓJ, and no periodic axis is mixed with another axis.

Outcome (2026-10-14): the developers decided fourteen items, now in `DESIGN.md`, and deferred the rank-truncated Fisher projection. `ParamSpace.axis_blocks` groups the positions into the Euclidean block, the circles and the sphere pairs. A `Transform` describes one problem and maps points of any leading shape, with `jacobian` (`jax.jacfwd` by default), `map_fisher` (Γ′ = JᵀΓJ, now cited to Jeffreys 1946), `map_space`, `map_sample_set` (`log_prior` and `log_q` lose ln|det ∂f/∂x|) and a `description` for the space column; `Compose` acts from left to right and joins the descriptions with " | ". `Whitening` applies ZCA-cor to the Euclidean block, divides each circle by its circular standard deviation and leaves spheres alone. Reference whitening takes one posterior, pooled whitening the equal-weight mean of the two covariances of a pair, and Fisher whitening the marginal blocks of Γ⁻¹; `decorrelate=False` gives a z-score. Checks act on matrices scaled to unit diagonal: a singular sample covariance gives NaN and a `ValidityWarning`, an invalid Fisher matrix raises, and a condition number above 1/√ε warns. `core.samples` gained the package's widths: the unbiased weighted covariance, computed on samples shifted by the sample of largest weight, and the circular standard deviation. A separate commit fixed the session 3 zoo tests, which failed on Linux because a test helper evaluated x % inf. Later sessions build one whitening per problem with `jax.vmap`, pool the two posteriors of each pair, and take widths from `core.samples`.

### 5. MCMC tools

Scope: `mcmc/autocorr.py`, `mcmc/ess.py`, `mcmc/reduce.py`.

- Our own integrated autocorrelation time τ from walker-averaged autocorrelation functions, checked against emcee's and eryn's estimators.
- Effective sample size, split-chain Kolmogorov-Smirnov tests on τ-thinned halves, and rank-normalised split-R̂ across walkers as an informational statistic that always carries a warning, because eryn's walkers are coupled.
- Reduction: burn-in removal and thinning by τ, producing a `SampleSet` with its `n_eff` and recording each step. `n_eff` must lie between 1 and the number of samples, so τ-based values of anticorrelated chains are capped there.
- Test data: the zoo's `ar1_chains` gives τ in closed form, τ < 1 for negative φ, and a known transient through its `start` argument.
- Done when τ matches the analytic value for AR(1) chains, and a reduction can be repeated from its record.

Outcome (2026-10-14): the developers decided twelve items, now in `DESIGN.md`. Sokal's window, as emcee applies it, gave τ ≈ 0 or below for AR(1) chains with φ ≤ −0.5, so τ now truncates emcee's walker-averaged autocorrelation function with Geyer's initial monotone sequence and the antithetic correction of Vehtari et al. (2021), floored at 1/log₁₀ S; it recovers (1 + φ)/(1 − φ) from φ = 0.9 down to φ = −0.5. Kernels take arrays shaped (step, walker, param): `integrated_autocorrelation_time`, `autocorrelation_function`, `effective_sample_size` (the raw S·W/τ), `split_chain_ks` (the statistic and its asymptotic Kolmogorov p-value, on halves thinned per parameter and pooled over the walkers) and `rank_normalized_split_rhat` (the larger of bulk and folded R̂, always with a `CoupledChainsWarning`). `reduce_chains` takes `Chains`: cold chain, an explicit burn-in, thinning by the largest max(1, ⌈τ − 2ε⌉) over the parameters, with ε the error of τ of Madras and Sokal (1988), and `n_eff` = N/τ_k from every k-th lag, clipped to [1, N]; its `ReductionRecord` holds the chain shape, cold index, burn-in, factor, τ, τ_k, `n_eff` and the SHA-256 of the chains, and a call with the recorded burn-in and factor repeats it exactly. The weighted two-sample KS statistic, which session 8 also needs, now lives in `metrics/marginal.py`, and emcee joined the test group as a reference. Each walker is centred on its own mean, so its autocorrelations sum to ½; lag pairs that stay positive to the end of the chain therefore put τ at its floor, and the check that warns about them flags chains that are too short. `jnp.searchsorted` returns int32, which JAX divides in float32 even with x64 on, so integer positions are cast to float64 before any arithmetic. Later sessions take chain health from the `mcmc` kernels on the cold chain, reduce chains with `reduce_chains`, and size nulls and subsamples with its `n_eff`.

### 6. Estimators

Scope: `estimators/`.

- Gaussian, bounded KDE (reflecting edges at finite interval bounds, wrapping on circles), kNN and GMM. The classifier and normalizing-flow estimators are placeholders: final signatures that raise `NotImplementedError`.
- Estimator settings must be reusable by the plots, so that contours and metrics agree.
- Bounded KDEs run in the original or z-scored space: decorrelated whitened axes are real lines, because their bounds become oblique.
- Done when the estimators recover zoo densities within tolerance and respect each topology.

Outcome (2026-10-14): the developers decided sixteen items, now in `DESIGN.md`. Every estimator is an equinox module that a `fit` classmethod builds per problem and posterior, vmapped over (B, K), with frozen pydantic settings for the `estimator_settings` column and `resolved_settings` for the values that a rule chose; the `estimator` column reads `gaussian`, `kde`, `knn`, `gmm`, `flow` or `classifier`. `GaussianDensity` has the weighted mean and unbiased covariance and draws the Gaussian surrogates. `KernelDensity` wraps `jax.scipy.stats.gaussian_kde` with Silverman's factor on the Kish n_eff, and sums its kernels over windings and, in two or more dimensions, over mirror images at finite bounds. In one dimension an interval takes the non-negative correction of Jones and Foster (1996), which recovered the zoo's railing scores within a few per cent where reflection was high by 11 to 48 per cent; a KDE warns below Silverman's sample size for its dimension. Its estimates equal SciPy's and, with the resolved factor, cantuccio's. `NearestNeighbours` gives topology-aware k-th neighbour distances and digamma-corrected log densities, with the rank k as the required setting `neighbours`; zero weights leave its searches, while unequal weights, too few samples and coinciding samples give NaN. `MixtureDensity` runs weighted EM on z-scored axes from 18 weighted k-means++ starts with scikit-learn's stopping rule. A zoo check found that this rule ends 37 to 68 per cent of single starts near the one-Gaussian solution and leaves winners a few nats below the optimum, which the developers accepted, with more starts. The Gaussian and the GMM unwrap circles about their circular mean and warn when the Rayleigh test leaves the mean direction unresolved. `FlowDensity` and `ClassifierRatio` are placeholders. `wrap`, `sphere_coordinates` and `unit_vectors` moved from the zoo to `core/topology.py`, and `core/samples.py` gained `circular_mean`, `mean_resultant_length` and `unwrap_about`. A fit inside `jax.jit` counts its images from the widest kernel that the samples admit and gives NaN beyond. Later sessions build densities with `mixed_signals.estimators`, write `resolved_settings().to_json()` into their rows, and resample weighted sets before kNN metrics.

### 7. eryn adapters and priors

Scope: `io/eryn_backend.py`, `priors/protocol.py`, `priors/eryn_container.py`.

- Open first: eryn's transforms are NumPy or CuPy callables that JAX cannot jit or differentiate, so the session starts by settling with the developers how they become `Transform` objects, then builds that form. Whatever the form, it implements the session 4 `Transform` base, including `map_space` for physical bounds. The options are JAX re-implementations of the built-in families, `jax.pure_callback` wrappers, applying them at load time, or JAX-compatible eryn priors and transform containers upstream.
- Read eryn backends into `Chains`: one branch with all its temperatures, with acceptance and swap counts and the number of proposals behind each. eryn's backend does not store `num_repeats_in_model`, so the acceptance proposals need it from the caller or the sampler. Reject runs with several independent samplers (`nsamplers` > 1), which are deferred.
- The minimal prior protocol: `logpdf` maps (N, D) samples to (N,) values, and `rvs(size)` returns size + (D,).
- `ParamSpace.from_eryn_priors`: names from `key_order`, and each prior's label, unit, bounds and boundary. `periodic` gives a circle; every other parameter gives an interval from its bounds, where one bound may be infinite, and `reflective` maps like no flag. eryn stores bounds in the sampling space; physical bounds come from the inverse transform. `DeltaFunction` parameters are dropped and discrete priors are rejected.
- Pitfalls: import eryn inside functions only, so `import mixed_signals` works without it; use `eryn.priors`, because `eryn.prior` is deprecated.
- Done when a short eryn run round-trips into `Chains` and a `ParamSpace`.

Outcome (2026-10-24): the session paused on 2026-10-14 so that eryn's priors could become JAX-compatible upstream, on eryn's branch `feat-jaxpriors`; the developers then decided fourteen items, now in `DESIGN.md`. `ErynPhysical` wraps eryn's own JAX backend through `to_backend("jax")`: its forward map is eryn's `inverse_transform`, its log-determinant minus eryn's `log_jacobian`, its description `physical(eryn)`, and its `map_space` renames transformed axes to eryn's `name_phys` and turns SineUniform and CosineUniform sphere axes into latitude and colatitude. `Chains.from_eryn` reads a sampler, a backend or an HDF5 file, takes the acceptance proposals from `num_repeats_in_model` (from a sampler, otherwise optional) and the swap proposals from the walkers, and drops `DeltaFunction` columns when it knows the priors. `ParamSpace.from_eryn_priors` names parameters by key, or `x0`, `x1`, ... for integer keys, reads the sampler's `periodic` dict, because nothing in eryn reads `Prior.boundary`, takes `Sphere` objects, and rejects discrete priors, including `PoissonDistribution`, which has no `logpmf`. `PriorDistribution` is the minimal prior protocol, `logpdf` on (N, D) samples and `rvs(size, *, key)`, and `ErynPrior` adapts eryn containers to it in the sampling coordinates. `pyproject.toml` pins lisa-analysis-tools/Eryn at `feat-jaxpriors`, which is not pushed yet, so `uv.lock` still records l2d-dev and the checks run with `uv run --no-sync` until the push: the one Open item. Later sessions read eryn runs with `Chains.from_eryn` and their spaces with `ParamSpace.from_eryn_priors`, map them to physical parameters with `ErynPhysical`, and take priors as `PriorDistribution` objects.

### 8. Metrics I: moments and marginals

Scope: `metrics/_cap.py`, `metrics/moments.py`, `metrics/marginal.py`.

- A subsampling cap for metrics that scale as N², with a warning and a keyword that sets the cap.
- Per-parameter mean shift and width ratio; Gaussian KL in both directions, W2 and Hellinger, split into location and shape terms.
- Widths come from `core.samples.weighted_covariance`; the pooled σ of mode B is the scale of the pooled z-score (`Whitening.from_samples` with `decorrelate=False`), so per-parameter and joint units agree.
- Marginals: Kolmogorov-Smirnov, k-sample Anderson-Darling, one-dimensional W1, and Jensen-Shannon in nats and Hellinger with a bounded KDE. Session 5 built the weighted two-sample KS statistic, `metrics.marginal.ks_statistic`; this session adds its rows, its name and its σ_eff map.
- The bounded KDE is the one-dimensional `estimators.KernelDensity`, whose interval bounds take the Jones-Foster correction. Settle with the developers whether the two marginals of a pair share one bandwidth, fixed once per comparison as the MMD bandwidth is, and write `resolved_settings().to_json()` into each row.
- Every kernel is pure JAX, honours `log_weights`, carries jaxtyping annotations and cites its sources.
- Name the result metrics as the zoo's reference tables do (`kl`, `kl_location`, `w2`, `ks`, `js`, `ad` and so on), or rename the zoo's in the same change, so results join their truth.
- Done when the zoo's closed-form values are recovered.

Outcome (2026-10-24): the developers decided fourteen items, now in `DESIGN.md`. Kernels compute in the coordinates they receive, so engines z-score the per-parameter rows of a pair with its shared whitening and run joint rows in the decorrelated one. `metrics/moments.py` has `mean_shift` and `width_ratio` (estimator `empirical`, circular statistics on circles) and the Gaussian `gaussian_kl`, `gaussian_w2` and `gaussian_hellinger`, moved from the zoo, which return the total, the location and shape terms and ln(det Σ_q/det Σ_p) for the direction of r_eff, and take the shortest arc on circles. `metrics/marginal.py` has the weighted `kuiper_statistic`, `anderson_darling_distance` (each posterior pooled with half the mass, summed at the ends of runs of ties), `wasserstein_distance` and `circle_wasserstein_distance`. `marginal_distances` applies them per parameter: KS, AD and W1 on lines and intervals, and on circles Kuiper's V in rows named `kuiper_ks` (see `ks_metric_names`), W1 along the circle and AD cut opposite the pooled circular mean; sphere axes follow their own topologies. `kde_divergences` integrates the JS divergence and the Hellinger distance of two one-dimensional `KernelDensity` estimates on Gauss-Legendre panels of at most two bandwidths, and `marginal_kde_divergences` fits them per parameter with `shared_bandwidth_factor`, Silverman's factor at the smaller Kish n_eff. `metrics/_cap.py` gives the O(N²) metrics `cap_samples`, a uniform subsample without replacement above `SAMPLE_CAP` = 10 000 samples per posterior, set from a CPU timing. `sigma/maps.py` has the closed-form `sigma_eff` and `r_eff` maps. The zoo gained marginal truths for `wrapped_gaussians` and `railed_gaussians`, checked against SciPy; every kernel matched SciPy, an exact transport linear program or QUADPACK, and recovered the zoo within five standard errors, the KDE values against the zoo pair smoothed by the kernels. Later sessions call `cap_samples` before every O(N²) metric, fit marginal KDEs with one shared factor per comparison, and take σ_eff and r_eff from `sigma.maps`.

### 9. Metrics II: divergences, kernels and classifiers

Scope: `metrics/divergence.py`, `metrics/kernel.py`, `metrics/classifier.py`.

- kNN estimates of the KL divergence in both directions, of the Jensen-Shannon divergence from Kozachenko-Leonenko entropy estimates of the pooled samples and of each input, and of the Hellinger distance from the Rényi-α estimator at α = ½.
- Build them on `estimators.NearestNeighbours`: its k-th neighbour distances (`neighbour_distance`, `leave_one_out_distance`) and digamma-corrected log densities. Each metric sets its default k (`neighbours`). Unequal positive weights give NaN there, so weighted sets are resampled first.
- Maximum mean discrepancy with a Gaussian kernel, whose median-heuristic bandwidth is fixed once per comparison, and energy distance, both under the cap from session 8: `metrics._cap.cap_samples` with `SAMPLE_CAP`, on a key derived with `prng.derive_key` from the row's path ending in `"subsample"`.
- The classifier two-sample test (C2ST) as a placeholder: its final signature and docstring, raising `NotImplementedError`.
- Done when each metric vanishes on identical distributions, within its null, and orders the zoo's shifts correctly.

### 10. Optimal transport

Scope: `metrics/ot/`. The developers own the method choices, so the session opens with options, not a pick.

- Costs: the wrapped difference min(|Δ|, P − |Δ|) on periodic axes, and great-circle costs on sky-position spheres.
- Whitening divides each circle by its circular standard deviation, so circle costs use the period P/s of `Whitening.map_space`, and it leaves spheres in radians; how a sphere cost weighs against the whitened Euclidean block is this session's choice.
- Check that each cost keeps the debiased Sinkhorn divergence non-negative. Squared geodesic costs on circles and spheres do not make exp(−c/ε) a positive definite kernel, so compare chordal costs, unsquared geodesic costs and the current costs on the zoo's periodic and sphere cases before the developers choose.
- The Sinkhorn divergence through ott-jax, and sliced W2 on the Euclidean block.
- Port from catpylisa's `common_tools.distances.optimal_transport` module: `geometries.py` (chordal and geodesic sphere costs, composite cost, coupling-scale calibration) and `divergence.py` (adaptive ε, Sinkhorn divergence). Read the ε-scaling notes in `ot_convergence.md`.
- Pitfall: ott-jax imports jaxopt, which raises a DeprecationWarning; tests turn warnings into errors, so add a targeted `filterwarnings` entry for it.
- Done when the costs respect each topology and the divergences vanish on identical inputs.

### 11. Diagnostics

Scope: `diagnostics/`.

- Gaussianity: Mahalanobis distances against χ²_D, skewness, kurtosis and Anderson-Darling.
- Dependence: Pearson, Spearman, distance correlation and mutual information against their Gaussian values, to catch bananas. The KSG estimator of mutual information needs the max norm and neighbour counts, which this session adds to `estimators/knn.py`.
- Every shape and dependence statistic is compared with Gaussian surrogates at the posterior's N_eff, drawn with `estimators.GaussianDensity.sample`, so each row gets a null, a p-value and σ_sig.
- Multimodality through GMM-BIC with `estimators.MixtureDensity`: form BIC from its `mean_log_likelihood` and `parameter_count`, and settle with the developers the n of the penalty for weighted or correlated samples. Under scikit-learn's stopping rule the winning log-likelihood can lie a few nats below the optimum (the session 6 zoo check), which the comparison of component counts must allow for.
- Prior railing: the marginal posterior density at a finite bound over its peak density, from the one-dimensional `estimators.KernelDensity`, whose Jones-Foster correction recovered the zoo's railing scores within a few per cent, with a flag threshold calibrated on the zoo's railed cases.
- Chain health from session 5 (τ, effective sample size, split-chain tests, informational R̂, and acceptance and swap rates for eryn backends, from the counts and proposals in `Chains`) as diagnostic rows. The `mcmc` kernels take the cold chain shaped (step, walker, param); the split-chain test brings its own asymptotic p-value, and R̂ raises a `CoupledChainsWarning` that the rows capture.
- On circle parameters the split-chain test takes Kuiper's V (`metrics.marginal.kuiper_statistic`) with its limiting distribution (Kuiper 1960) instead of KS, whose value depends on where the circle is cut. The `mcmc` kernels take arrays without a space, so this session passes the topology.
- `Chains.from_eryn` gives acceptance counts only with `num_repeats_in_model`, so without it the chain-health rows show swap rates alone.
- Done when the zoo's bananas, mixtures and railed cases are flagged and its Gaussians are not.

### 12. Nulls and σ

Scope: `nulls/`, `sigma/`, and a calibration script in `scripts/`.

- Nulls: permutation of τ-thinned pooled samples in both modes by default, splitting the reference into halves at the approximation's N_eff as a mode A alternative, Gaussian surrogates for posterior-shape diagnostics (`estimators.GaussianDensity.sample`), and N_eff matching. 200 draws by default, configurable.
- σ_sig: the null's empirical p-value as a one-sided σ, with a fitted tail beyond the resampling resolution, flagged as extrapolated.
- σ_eff maps per metric, r_eff maps for shape-only rows, and per-metric threshold defaults on σ_eff and |ln r_eff| that the YAML config can override.
- A script that rebuilds the σ_eff and r_eff calibration tables and the validation page from the zoo. The zoo's `batch_size` draws independent problems for the null-uniformity checks, and its `mahalanobis_shift` and `width_ratio` rows are the true σ_eff and r_eff. The page also reruns the session 6 zoo checks, the GMM starts behind `DEFAULT_STARTS` and the railing scores of the reflected and corrected KDEs, and the session 8 timing behind `SAMPLE_CAP`.
- `sigma/maps.py` holds the closed-form maps of session 8; add the tabulated AD and JS curves next to them. The marginal JS and Hellinger come from KDEs, whose kernels shrink a shift by 1/√(1 + f²), 1.4% at N = 10⁴, so calibrate their curves with the KDE included.
- Settle with the developers how null-subtracted values below zero enter the maps, which give NaN outside the range of their metric.
- Done when nulls on identical distributions give uniform p-values, σ_eff recovers the zoo's known shifts, and r_eff its known width ratios.

### 13. Reweighting

Scope: `reweight/` and its engine function.

- Importance weights for a new prior, a new likelihood or an approximation's density q.
- The evidence ratio Z_new/Z_old = E_old[L_new/L_old].
- Effective sample size, the Kish value of `core.samples`, and PSIS k̂ for every weight set, flagged above min(1 − 1/log₁₀ S, 0.7). Weights stay raw; Pareto smoothing is opt-in.
- New priors are `PriorDistribution` objects; eryn containers enter through `ErynPrior`, whose densities are those of eryn's sampling coordinates, the coordinates of the stored `log_prior`. Add densities of the physical parameters, eryn's `logpdf_physical`, when a reweighting needs them.
- Done when reweighting between zoo Gaussians recovers the analytic target and k̂ flags heavy-tailed weights.

### 14. Engines

Scope: `engines/` (`compare` and `diagnose`) and the public API in `__init__.py`.

- `compare` in modes A and B, against a reference or over all pairs, with one shared whitening per comparison. Every row carries its null and null draws, σ_sig, σ_eff or r_eff, the null standard deviation as its uncertainty, validity warnings and provenance.
- A dry run that counts metric evaluations times null resamples and warns above a budget.
- kNN metrics need equally weighted samples: resample a weighted set in proportion to its weights first (Smith and Gelfand 1992), with a key derived from the row's path.
- `diagnose` on raw chains and on flat samples.
- Re-exports in `__init__.py` go below the float64 switch.
- One whitening per problem through `jax.vmap` of `Whitening.from_samples` (the reference posterior in mode A, the two posteriors of each pair in mode B) or `from_fisher`, with its `description` in the space column. Traced whitenings do not warn, so rebuild a NaN one concretely to capture its `ValidityWarning` in the rows of its problem.
- Per-parameter rows run on both inputs z-scored by their shared whitening (`decorrelate=False`), joint rows in the decorrelated whitening: the session 8 kernels compute in the coordinates they receive. Fit the marginal KDEs with one `shared_bandwidth_factor` per comparison, reuse it in every null draw, and write `KernelDensitySettings(bandwidth=factor).to_json()` into the rows; the factor is a static setting, so the null draws of one comparison map with `jax.vmap`, while problems with different factors run one by one. `ks_metric_names` names each KS or Kuiper row.
- Done when a zoo comparison produces the full result table end to end.

### 15. Data space

Scope: `generators/`, `dataspace/` and the data-space engine function.

- The minimal generator protocol that data space needs (`generate_template` and a noise model), with capability checks. The full generator contract stays deferred.
- User code is called at the boundary in chunks, with DLPack hand-off on GPU; it may be CuPy or PyTorch.
- Energy distance with a noise-weighted inner product supplied by the caller, and Fisher-whitened parameter distances when a Fisher matrix is available, through `Whitening.from_fisher`, which needs a positive-definite Γ (rank truncation is deferred); σ_eff from the energy-distance curve, valid at linear order.
- Reference only: `estimate_distances.py` in catpylisa's `common_tools.utils` module (waveform distances with JaxGB). The catpylisa team wrote it, so read it rather than port it.
- Done when a toy generator gives distances in noise-σ units that match the linear-order Fisher prediction.

### 16. Pipeline and CLI

Scope: `pipeline/`, `cli.py`.

- pydantic models for the YAML config, and a runner that writes `results.parquet` and its provenance, including the config hash. All keys derive from one config seed.
- Chain reduction through `mcmc.reduce_chains`, with the burn-in and thinning in the config, and each run's `ReductionRecord` written next to the results.
- eryn runs load with `Chains.from_eryn`, and their spaces with `ParamSpace.from_eryn_priors`.
- `mixed-signals run config.yaml`, including the dry-run cost estimate.
- Done when an example config runs from the shell and gives the same table again from the same seed.

### 17. Reports

Scope: `report/`.

- A generated marimo notebook that runs as a script and exports to HTML.
- Contents: the σ_eff, r_eff and σ_sig summary table, cantuccio corner overlays with contours from the metric estimators, null histograms from the stored null draws, and diagnostics.
- Pass `KernelDensity.resolved_settings().bandwidth` to cantuccio as its `bandwidth`, which reproduces the estimate away from bounds (tested in session 6). cantuccio neither reflects nor corrects at bounds, so near an interval bound draw the contours from `KernelDensity.log_density` on a grid.
- Done when the example pipeline produces a report that renders.

### 18. GW subpackage

Scope: `gw/`, with a `gw` extra if it needs new dependencies.

- GW parameter conventions (names, units, sky-position pairs), readers, waveform helpers and the PSD-weighted frequency-domain inner product, so the core stays domain-agnostic.
- Done when a LISA galactic-binary example loads through `gw` into the core containers.

### 19. Release readiness

- A Sphinx user guide and API pages, the validation page, and marimo examples in `examples/`.
- The GPU suite on the cluster with a single GPU, and a performance check.
- Release blockers: reserve `mixed-signals` on PyPI, and wait for eryn to release its JAX-compatible prior setup, now on its `feat-jaxpriors` branch, because a published package cannot depend on a git branch.
- Done when version 0.1.0 can be published.
