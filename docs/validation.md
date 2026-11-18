# Validation

This page reruns the zoo checks behind the defaults of mixed signals: the calibration of the nulls, the recovery of known effect sizes, the optimal transport settings, the limits of the chordal metrics on circles and spheres, the ranks of the nearest-neighbour divergences, the estimators, the calibration and the flags of the diagnostics, and the costs of the metrics. Every number comes with a standard error, and every check lists its claims with an expectation and a tolerance that the script states. The verdict of a claim, consistent or not consistent, is computed from the numbers of the run; no verdict is written by hand. A tolerance of three standard errors uses the Student t quantile with the same tail probability when a mean rests on few replicates. The script `scripts/build_validation.py` writes this page, its figures and the raw numbers (`validation/results.json`). Rebuild them with `uv run python scripts/build_validation.py`; the option `--quick` runs fewer replicates in about three minutes, which the test suite uses.

:::{table} Version, machine, seed and mode of this build.

| Item | Value |
| --- | --- |
| mixed-signals | 0.1.0.dev0 |
| JAX | 0.11.2 |
| jaxlib | 0.11.2 |
| Python | 3.12.12 |
| Platform | macOS-26.6.2-arm64-arm-64bit |
| Processor | arm |
| CPU | Apple M4 Pro |
| Seed | 20261117 |
| Mode | full |
| Date | 2026-11-18 11:10 UTC |
:::

:::{table} Wall time of each check, in seconds.

| Check | Wall time (s) |
| --- | --- |
| {ref}`validation-permutation-uniformity` | 105.1 |
| {ref}`validation-permutation-thinned-chains` | 204.4 |
| {ref}`validation-permutation-weights` | 61.8 |
| {ref}`validation-split-half` | 74.7 |
| {ref}`validation-tail-fit` | 3.9 |
| {ref}`validation-effect-recovery` | 188.6 |
| {ref}`validation-width-recovery` | 26.8 |
| {ref}`validation-signed-maps` | 36.3 |
| {ref}`validation-threshold-basis` | 0.0 |
| {ref}`validation-knn-kl-calibration` | 202.7 |
| {ref}`validation-sinkhorn-epsilon` | 173.8 |
| {ref}`validation-sliced-directions` | 294.2 |
| {ref}`validation-gaussian-sliced-w2` | 45.4 |
| {ref}`validation-chordal-widths` | 245.6 |
| {ref}`validation-sphere-width` | 1.0 |
| {ref}`validation-chords-and-arcs` | 7.8 |
| {ref}`validation-knn-ranks` | 459.7 |
| {ref}`validation-gmm-starts` | 97.2 |
| {ref}`validation-kde-railing` | 89.5 |
| {ref}`validation-surrogate-calibration` | 176.8 |
| {ref}`validation-circle-drift` | 83.3 |
| {ref}`validation-ksg-bananas` | 71.4 |
| {ref}`validation-bic-neff` | 26.3 |
| {ref}`validation-mode-counting` | 26.4 |
| {ref}`validation-split-chain-kuiper` | 1.2 |
| {ref}`validation-diagnostic-flags` | 244.4 |
| {ref}`validation-costs` | 293.7 |
| Total | 3242.0 |
:::

## Nulls

(validation-permutation-uniformity)=
### Uniform p-values under the permutation null

Reruns the check of session 12a.

Supports:

- {py:func}`mixed_signals.nulls.permutation.permutation_null`
- {py:func}`mixed_signals.sigma.significance.null_significance`
- the done criterion of session 12: uniform p-values for identical sets

Two sets of 1 000 samples are drawn from one zoo Gaussian ({py:func}`~mixed_signals.zoo.cases.shifted_gaussians` without a shift, 2 parameters), 120 times. For each replicate, {py:func}`~mixed_signals.nulls.permutation.permutation_null` splits the pool of both sets at random 199 times, {py:func}`~mixed_signals.nulls.plan.null_draws` evaluates a statistic on every draw, and {py:func}`~mixed_signals.sigma.significance.null_significance` reads the empirical p-value of the observed value, without the tail fit, on the tail of {py:func}`~mixed_signals.sigma.thresholds.comparison_tail`. The first parameter gives the mean shift, the width ratio and the KS statistic. The sliced W2 uses the 500 directions of its default and costs most per draw, so it runs on the first 10 replicates with 99 draws only. If the null is exact, the p-values are uniform and the rejection rate at a level equals the level; a number of draws one below a multiple of 100 makes the levels of 1%, 5% and 10% exact.

:::{table} Rejection rates of identical sets: the share of replicates whose p-value is at or below the level, with the binomial error at the level.

| Statistic | Replicates | Rate at 1% | Rate at 5% | Rate at 10% |
| --- | --- | --- | --- | --- |
| mean shift | 120 | 0.0167 ± 0.0091 | 0.042 ± 0.020 | 0.100 ± 0.027 |
| KS | 120 | 0.0083 ± 0.0091 | 0.017 ± 0.020 | 0.092 ± 0.027 |
| log width ratio | 120 | 0.0000 ± 0.0091 | 0.042 ± 0.020 | 0.075 ± 0.027 |
| energy distance | 120 | 0.0167 ± 0.0091 | 0.042 ± 0.020 | 0.058 ± 0.027 |
| sliced W2 | 10 | 0.000 ± 0.031 | 0.000 ± 0.069 | 0.000 ± 0.095 |
:::

```{image} validation/figures/permutation-uniformity-ecdf-light.svg
:class: only-light
:alt: Step curves of the p-value distribution functions of five statistics, each close to the diagonal.
```

```{image} validation/figures/permutation-uniformity-ecdf-dark.svg
:class: only-dark
:alt: Step curves of the p-value distribution functions of five statistics, each close to the diagonal.
```

**Figure.** Empirical distribution function of the p-values of identical sets; the line is the uniform distribution.

:::{table} Claims of the check, each with its verdict.

| Claim | Expected | Observed | Verdict |
| --- | --- | --- | --- |
| The mean shift rejects at its levels. | rates within three binomial errors of 1%, 5%, 10% | 0.0167 ± 0.0091; 0.042 ± 0.020; 0.100 ± 0.027 | consistent |
| The KS rejects at its levels. | rates within three binomial errors of 1%, 5%, 10% | 0.0083 ± 0.0091; 0.017 ± 0.020; 0.092 ± 0.027 | consistent |
| The log width ratio rejects at its levels. | rates within three binomial errors of 1%, 5%, 10% | 0.0000 ± 0.0091; 0.042 ± 0.020; 0.075 ± 0.027 | consistent |
| The energy distance rejects at its levels. | rates within three binomial errors of 1%, 5%, 10% | 0.0167 ± 0.0091; 0.042 ± 0.020; 0.058 ± 0.027 | consistent |
| The sliced W2 rejects at its levels. | rates within three binomial errors of 1%, 5%, 10% | 0.000 ± 0.031; 0.000 ± 0.069; 0.000 ± 0.095 | consistent |
:::

The check raised no warning.

(validation-permutation-thinned-chains)=
### Permutation null on thinned chains

Reruns the check of session 12a.

Supports:

- {py:func}`mixed_signals.nulls.permutation.permutation_side_size`
- the null sides at the effective sample size

Zoo chains {py:func}`~mixed_signals.zoo.cases.ar1_chains` with the coefficients 0.9 and 0.5, 32 walkers and 100 times the true autocorrelation time τ in steps, reduced by {py:func}`~mixed_signals.mcmc.reduce.reduce_chains` (thinning by the autocorrelation time). Each of the 300 replicates at coefficient 0.9 and the 120 at 0.5 compares two independent runs, with the statistics of the first parameter and 99 permutation draws. The default null gives each side the effective sample size of its set, through the arguments `first_n_eff` and `second_n_eff` of {py:func}`~mixed_signals.nulls.permutation.permutation_null`; the contrast permutes all samples with sides of the sample count. The thinned samples keep some autocorrelation, so a null of full sides is too narrow. The reduction estimates the autocorrelation time left after thinning at 1.3686 ± 0.0015, against 1.4292 ± 0.0014 that the thinning factor implies for an AR(1) chain, (1 + φᵏ)/(1 − φᵏ). Sides of N/τₖ samples therefore make the default null narrower than the scatter of the mean shift by their ratio, 1.044 in variance; the replicates at coefficient 0.9 give 1.217 ± 0.100 for the default null and 1.69 ± 0.14 for the sides of N, against one for a null of the right width. The developers allow the rejection rate of the mean shift at coefficient 0.9, the slowest chain, up to 1.5 times the level, plus three binomial errors at that bound. At the 5% level that lets σ_sig read up to 10% high: a rate of 7.5% is the rate of a null 1.10 times too narrow in standard deviation. The reason is that the τₖ of the reduction reads low. The other claims of the check stay strict.

:::{table} AR(1) runs after reduction: the true autocorrelation time of the case and the means over runs of the estimated one, the thinning factor k, the autocorrelation time left after thinning as the reduction estimates it, the same as the thinning factor implies for an AR(1) chain with coefficient φ, (1 + φᵏ)/(1 − φᵏ), the sample count and the effective sample size.

| Coefficient | True τ | Estimated τ | Thinning | Thinned τ | Implied thinned τ | N | N_eff |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.9 | 19 | 17.994 ± 0.032 | 16.483 ± 0.026 | 1.3686 ± 0.0015 | 1.4292 ± 0.0014 | 3 702.4 ± 5.9 | 2 707.3 ± 5.3 |
| 0.5 | 3 | 2.8563 ± 0.0078 | 3 | 1.2422 ± 0.0027 | 1.286 | 3 200 | 2 579.0 ± 5.4 |
:::

:::{table} Rejection rates of two independent runs of the same chain.

| Coefficient | Null | Statistic | Rate at 1% | Rate at 5% | Rate at 10% |
| --- | --- | --- | --- | --- | --- |
| 0.9 | sides at the effective sample size | mean shift | 0.0200 ± 0.0057 | 0.097 ± 0.013 | 0.163 ± 0.017 |
| 0.9 | sides at the effective sample size | KS | 0.0067 ± 0.0057 | 0.037 ± 0.013 | 0.080 ± 0.017 |
| 0.9 | sides at the effective sample size | log width ratio | 0.0000 ± 0.0057 | 0.017 ± 0.013 | 0.057 ± 0.017 |
| 0.9 | all samples, sides of N | mean shift | 0.0400 ± 0.0057 | 0.137 ± 0.013 | 0.203 ± 0.017 |
| 0.9 | all samples, sides of N | KS | 0.0333 ± 0.0057 | 0.113 ± 0.013 | 0.197 ± 0.017 |
| 0.9 | all samples, sides of N | log width ratio | 0.0067 ± 0.0057 | 0.060 ± 0.013 | 0.117 ± 0.017 |
| 0.5 | sides at the effective sample size | mean shift | 0.0083 ± 0.0091 | 0.042 ± 0.020 | 0.100 ± 0.027 |
| 0.5 | sides at the effective sample size | KS | 0.0083 ± 0.0091 | 0.025 ± 0.020 | 0.058 ± 0.027 |
| 0.5 | sides at the effective sample size | log width ratio | 0.0167 ± 0.0091 | 0.067 ± 0.020 | 0.100 ± 0.027 |
| 0.5 | all samples, sides of N | mean shift | 0.0083 ± 0.0091 | 0.092 ± 0.020 | 0.133 ± 0.027 |
| 0.5 | all samples, sides of N | KS | 0.0167 ± 0.0091 | 0.058 ± 0.020 | 0.108 ± 0.027 |
| 0.5 | all samples, sides of N | log width ratio | 0.0167 ± 0.0091 | 0.067 ± 0.020 | 0.117 ± 0.027 |
:::

:::{table} Variance over the replicates of the mean shift in units of its null: the value minus the null mean, over the null standard deviation. A null of the right width gives one; the error is that of a variance of normal scores.

| Coefficient | Replicates | Null | Variance |
| --- | --- | --- | --- |
| 0.9 | 300 | sides at the effective sample size | 1.217 ± 0.100 |
| 0.9 | 300 | all samples, sides of N | 1.69 ± 0.14 |
| 0.5 | 120 | sides at the effective sample size | 0.97 ± 0.13 |
| 0.5 | 120 | all samples, sides of N | 1.18 ± 0.15 |
:::

:::{table} Claims of the check, each with its verdict.

| Claim | Expected | Observed | Verdict |
| --- | --- | --- | --- |
| For coefficient 0.9, the default null over-rejects the mean shift by at most 50% of the level. | rates at most 1.5 times the levels 1%, 5%, 10%, that is 1.5%, 7.5%, 15.0%, plus three binomial errors at those bounds | 0.0200 ± 0.0070; 0.097 ± 0.015; 0.163 ± 0.021 | consistent |
| For coefficient 0.9, the default null does not over-reject the KS. | rates at most three binomial errors above 1%, 5%, 10% | 0.0067 ± 0.0057; 0.037 ± 0.013; 0.080 ± 0.017 | consistent |
| For coefficient 0.9, the default null does not over-reject the log width ratio. | rates at most three binomial errors above 1%, 5%, 10% | 0.0000 ± 0.0057; 0.017 ± 0.013; 0.057 ± 0.017 | consistent |
| For coefficient 0.5, the default null does not over-reject the mean shift. | rates at most three binomial errors above 1%, 5%, 10% | 0.0083 ± 0.0091; 0.042 ± 0.020; 0.100 ± 0.027 | consistent |
| For coefficient 0.5, the default null does not over-reject the KS. | rates at most three binomial errors above 1%, 5%, 10% | 0.0083 ± 0.0091; 0.025 ± 0.020; 0.058 ± 0.027 | consistent |
| For coefficient 0.5, the default null does not over-reject the log width ratio. | rates at most three binomial errors above 1%, 5%, 10% | 0.0167 ± 0.0091; 0.067 ± 0.020; 0.100 ± 0.027 | consistent |
:::

The check raised no warning.

(validation-permutation-weights)=
### Permutation null on weighted sets

Reruns the check of session 12a.

Supports:

- {py:func}`mixed_signals.nulls.permutation.permutation_null`
- the pool that keeps every sample with its weight

Zoo Gaussians in two parameters: the target N(0, I) and a proposal 1.5 times wider. The weighted set holds 4 000 proposal samples with the importance weights target over proposal (Kish size 2 766.3 ± 2.4), the other set 2 000 equal-weight draws of the target. 70 replicates use 99 permutation draws of {py:func}`~mixed_signals.nulls.permutation.permutation_null`, whose sides follow the Kish sizes. For contrast, the script resamples both sets to equal weights, with replacement, with `jax.random.choice` before the permutation, an alternative that the package does not offer. Then the width ratio of one weighted set against an equal-weight set is checked at 200, 800 and 3 200 samples per set with 100 replicates each.

:::{table} Rejection rates of an importance-weighted set against equal-weight draws of its target.

| Null | Statistic | Rate at 1% | Rate at 5% | Rate at 10% |
| --- | --- | --- | --- | --- |
| weights kept in the pool | mean shift | 0.000 ± 0.012 | 0.071 ± 0.026 | 0.100 ± 0.036 |
| weights kept in the pool | KS | 0.014 ± 0.012 | 0.043 ± 0.026 | 0.086 ± 0.036 |
| weights kept in the pool | log width ratio | 0.000 ± 0.012 | 0.057 ± 0.026 | 0.100 ± 0.036 |
| resampled to equal weights first | mean shift | 0.029 ± 0.012 | 0.186 ± 0.026 | 0.229 ± 0.036 |
| resampled to equal weights first | KS | 0.143 ± 0.012 | 0.371 ± 0.026 | 0.471 ± 0.036 |
| resampled to equal weights first | log width ratio | 0.043 ± 0.012 | 0.157 ± 0.026 | 0.243 ± 0.036 |
:::

:::{table} Rejection rates of the log width ratio of one weighted set against an equal-weight set, for growing sample counts per set.

| Samples per set | Rate at 1% | Rate at 5% | Rate at 10% |
| --- | --- | --- | --- |
| 200 | 0.0000 ± 0.0099 | 0.050 ± 0.022 | 0.130 ± 0.030 |
| 800 | 0.0000 ± 0.0099 | 0.040 ± 0.022 | 0.060 ± 0.030 |
| 3 200 | 0.0100 ± 0.0099 | 0.020 ± 0.022 | 0.070 ± 0.030 |
:::

:::{table} Claims of the check, each with its verdict.

| Claim | Expected | Observed | Verdict |
| --- | --- | --- | --- |
| With the weights kept, the mean shift rejects at its levels. | rates within three binomial errors of 1%, 5%, 10% | 0.000 ± 0.012; 0.071 ± 0.026; 0.100 ± 0.036 | consistent |
| With the weights kept, the KS rejects at its levels. | rates within three binomial errors of 1%, 5%, 10% | 0.014 ± 0.012; 0.043 ± 0.026; 0.086 ± 0.036 | consistent |
| With the weights kept, the log width ratio rejects at its levels. | rates within three binomial errors of 1%, 5%, 10% | 0.000 ± 0.012; 0.057 ± 0.026; 0.100 ± 0.036 | consistent |
| After resampling, the mean shift rejects too often at 5%. | rate above 5% by more than three binomial errors | 0.186 ± 0.026 | consistent |
| After resampling, the KS rejects too often at 5%. | rate above 5% by more than three binomial errors | 0.371 ± 0.026 | consistent |
| After resampling, the log width ratio rejects too often at 5%. | rate above 5% by more than three binomial errors | 0.157 ± 0.026 | consistent |
| At 200 samples per set, the log width ratio rejects at its levels. | rates within three binomial errors of 1%, 5%, 10% | 0.0000 ± 0.0099; 0.050 ± 0.022; 0.130 ± 0.030 | consistent |
| At 800 samples per set, the log width ratio rejects at its levels. | rates within three binomial errors of 1%, 5%, 10% | 0.0000 ± 0.0099; 0.040 ± 0.022; 0.060 ± 0.030 | consistent |
| At 3 200 samples per set, the log width ratio rejects at its levels. | rates within three binomial errors of 1%, 5%, 10% | 0.0100 ± 0.0099; 0.020 ± 0.022; 0.070 ± 0.030 | consistent |
:::

The check raised no warning.

(validation-split-half)=
### Split-half null

Reruns the check of session 12a.

Supports:

- {py:func}`mixed_signals.nulls.split_half.split_half_null`
- {py:func}`mixed_signals.nulls.split_half.split_half_side_size`

A reference and an approximation of equal weight are drawn from one zoo Gaussian ({py:func}`~mixed_signals.zoo.cases.shifted_gaussians` without a shift, one parameter), so that the null hypothesis holds. {py:func}`~mixed_signals.nulls.split_half.split_half_null` cuts the reference alone into two sides, 99 times per replicate, over 120 replicates. The default sides carry the variance of the comparison. For contrast, the script builds sides of the size of the approximation with {py:func}`~mixed_signals.nulls.plan.random_sides`. A reference of 4 000 against 2 000 cannot supply such sides: they are clipped at half the reference, and the null warns that it is wider than the comparison.

:::{table} Cases of the split-half null: sample counts and the side size of the default.

| Case | Reference N | Approximation N | Side size |
| --- | --- | --- | --- |
| large reference | 10 000 | 1 000 | 1 818 |
| small reference | 4 000 | 2 000 | 2 000 |
:::

:::{table} Rejection rates of an approximation drawn from the reference itself.

| Case | Sides | Statistic | Rate at 1% | Rate at 5% | Rate at 10% |
| --- | --- | --- | --- | --- | --- |
| large reference | default | mean shift | 0.0000 ± 0.0091 | 0.042 ± 0.020 | 0.075 ± 0.027 |
| large reference | default | KS | 0.0083 ± 0.0091 | 0.033 ± 0.020 | 0.067 ± 0.027 |
| large reference | default | log width ratio | 0.0083 ± 0.0091 | 0.042 ± 0.020 | 0.092 ± 0.027 |
| large reference | approximation size | mean shift | 0.0000 ± 0.0091 | 0.000 ± 0.020 | 0.008 ± 0.027 |
| large reference | approximation size | KS | 0.0000 ± 0.0091 | 0.000 ± 0.020 | 0.000 ± 0.027 |
| large reference | approximation size | log width ratio | 0.0000 ± 0.0091 | 0.008 ± 0.020 | 0.025 ± 0.027 |
| small reference | default | mean shift | 0.0083 ± 0.0091 | 0.050 ± 0.020 | 0.108 ± 0.027 |
| small reference | default | KS | 0.0167 ± 0.0091 | 0.042 ± 0.020 | 0.092 ± 0.027 |
| small reference | default | log width ratio | 0.0000 ± 0.0091 | 0.033 ± 0.020 | 0.050 ± 0.027 |
:::

:::{table} Claims of the check, each with its verdict.

| Claim | Expected | Observed | Verdict |
| --- | --- | --- | --- |
| With the default sides, the mean shift rejects at its levels. | rates within three binomial errors of 1%, 5%, 10% | 0.0000 ± 0.0091; 0.042 ± 0.020; 0.075 ± 0.027 | consistent |
| With the default sides, the KS rejects at its levels. | rates within three binomial errors of 1%, 5%, 10% | 0.0083 ± 0.0091; 0.033 ± 0.020; 0.067 ± 0.027 | consistent |
| With the default sides, the log width ratio rejects at its levels. | rates within three binomial errors of 1%, 5%, 10% | 0.0083 ± 0.0091; 0.042 ± 0.020; 0.092 ± 0.027 | consistent |
| With clipped sides, the mean shift is conservative. | rates at most three binomial errors above 1%, 5%, 10% | 0.0083 ± 0.0091; 0.050 ± 0.020; 0.108 ± 0.027 | consistent |
| With clipped sides, the KS is conservative. | rates at most three binomial errors above 1%, 5%, 10% | 0.0167 ± 0.0091; 0.042 ± 0.020; 0.092 ± 0.027 | consistent |
| With clipped sides, the log width ratio is conservative. | rates at most three binomial errors above 1%, 5%, 10% | 0.0000 ± 0.0091; 0.033 ± 0.020; 0.050 ± 0.027 | consistent |
:::

:::{table} Warnings that the check raised; the messages are in the JSON.

| Category | Count |
| --- | --- |
| ValidityWarning | 120 |
:::

## Significance

(validation-tail-fit)=
### Tail fit of σ_sig on known nulls

Reruns the check of session 12a.

Supports:

- {py:func}`mixed_signals.sigma.significance.null_significance`
- {py:data}`mixed_signals.sigma.significance.TAIL_FRACTION`
- {py:data}`mixed_signals.sigma.significance.SUPPORT_MARGIN`

500 replicates of each case. The nulls have a known survival function: the standard normal, the chi-squared distribution with 1 and 4 degrees of freedom and the largest of 8 absolute standard normals, all drawn with `jax.random`, with the true p-value from `jax.scipy.special.ndtr` or `gammaincc`. The value of each case has the p-value of a normal at 3σ, 4σ and 5σ, and {py:func}`~mixed_signals.sigma.significance.null_significance` reads its σ_sig against 200 or 1 000 draws. The record in the zoo check of the module gives medians for a normal, a Kolmogorov and a chi-squared null with 4 degrees of freedom; this page reruns the normal and the chi-squared null of the record and not the Kolmogorov distribution. A claim lets the median of each of those two nulls lie within the range of the record plus half a unit of its last digit, plus the standard errors of the median, and a second claim lets the median at 3σ of all four nulls lie within 0.15 of the truth plus the standard errors. The standard error of a median is 1.25 times the half-width of the 16th to 84th percentile range over the square root of the replicates. The fit that the developers rejected, the unconstrained maximum likelihood fit of {cite:t}`knijnenburg2009fewer`, is not rerun here; its failure (a p-value of zero in 18 to 79% of the cases beyond 3σ) is in the record of the module. The fit that the package uses keeps the support above the value ({cite:t}`peschel2026permapprox`) and uses the estimator of {cite:t}`zhang2009new`. The record gives single values, 4.9 and 7.7, at 10σ and 100σ against 200 normal draws; the claim is that each lies between the 16th and 84th percentiles of the replicates.

:::{table} σ_sig of a value at a true significance, against the draws of a known null: the median with its approximate standard error, the 16th and 84th percentiles over replicates, the rows that the tail fit extrapolates and the number of infinite values.

| Null | True σ | Draws | Median σ_sig | 16th percentile | 84th percentile | Extrapolated | Infinite |
| --- | --- | --- | --- | --- | --- | --- | --- |
| standard normal | 3 | 200 | 2.907 ± 0.015 | 2.636 | 3.155 | 500 of 500 | 0 |
| standard normal | 3 | 1 000 | 3.019 ± 0.013 | 2.847 | 3.305 | 500 of 500 | 0 |
| standard normal | 4 | 200 | 3.462 ± 0.015 | 3.205 | 3.723 | 500 of 500 | 0 |
| standard normal | 4 | 1 000 | 4.068 ± 0.015 | 3.774 | 4.293 | 500 of 500 | 0 |
| standard normal | 5 | 200 | 3.810 ± 0.015 | 3.532 | 4.08 | 500 of 500 | 0 |
| standard normal | 5 | 1 000 | 4.721 ± 0.014 | 4.445 | 4.934 | 500 of 500 | 0 |
| chi-squared (1 degree of freedom) | 3 | 200 | 2.887 ± 0.017 | 2.627 | 3.227 | 500 of 500 | 0 |
| chi-squared (1 degree of freedom) | 3 | 1 000 | 2.9716 ± 0.0095 | 2.814 | 3.155 | 500 of 500 | 0 |
| chi-squared (1 degree of freedom) | 4 | 200 | 3.559 ± 0.024 | 3.183 | 4.045 | 500 of 500 | 0 |
| chi-squared (1 degree of freedom) | 4 | 1 000 | 3.753 ± 0.019 | 3.48 | 4.158 | 500 of 500 | 0 |
| chi-squared (1 degree of freedom) | 5 | 200 | 3.991 ± 0.030 | 3.45 | 4.509 | 500 of 500 | 0 |
| chi-squared (1 degree of freedom) | 5 | 1 000 | 4.444 ± 0.028 | 3.996 | 4.98 | 500 of 500 | 0 |
| chi-squared (4 degrees of freedom) | 3 | 200 | 2.911 ± 0.018 | 2.614 | 3.247 | 500 of 500 | 0 |
| chi-squared (4 degrees of freedom) | 3 | 1 000 | 2.988 ± 0.012 | 2.833 | 3.258 | 500 of 500 | 0 |
| chi-squared (4 degrees of freedom) | 4 | 200 | 3.626 ± 0.020 | 3.276 | 3.984 | 500 of 500 | 0 |
| chi-squared (4 degrees of freedom) | 4 | 1 000 | 4.046 ± 0.023 | 3.657 | 4.463 | 500 of 500 | 0 |
| chi-squared (4 degrees of freedom) | 5 | 200 | 4.127 ± 0.024 | 3.679 | 4.529 | 500 of 500 | 0 |
| chi-squared (4 degrees of freedom) | 5 | 1 000 | 4.874 ± 0.028 | 4.356 | 5.371 | 500 of 500 | 0 |
| largest of 8 absolute normals | 3 | 200 | 2.903 ± 0.017 | 2.62 | 3.221 | 500 of 500 | 0 |
| largest of 8 absolute normals | 3 | 1 000 | 3.002 ± 0.012 | 2.828 | 3.241 | 500 of 500 | 0 |
| largest of 8 absolute normals | 4 | 200 | 3.528 ± 0.017 | 3.217 | 3.814 | 500 of 500 | 0 |
| largest of 8 absolute normals | 4 | 1 000 | 4.137 ± 0.020 | 3.706 | 4.428 | 500 of 500 | 0 |
| largest of 8 absolute normals | 5 | 200 | 3.916 ± 0.020 | 3.541 | 4.267 | 500 of 500 | 0 |
| largest of 8 absolute normals | 5 | 1 000 | 4.828 ± 0.019 | 4.436 | 5.119 | 500 of 500 | 0 |
:::

:::{table} σ_sig of values far in the tail against 200 normal draws.

| True σ | Draws | Median σ_sig | 16th percentile | 84th percentile |
| --- | --- | --- | --- | --- |
| 10 | 200 | 4.882 ± 0.020 | 4.483 | 5.188 |
| 100 | 200 | 7.955 ± 0.038 | 7.213 | 8.553 |
:::

:::{table} Claims of the check, each with its verdict.

| Claim | Expected | Observed | Verdict |
| --- | --- | --- | --- |
| No σ_sig is infinite. | 0 infinite values | 0 infinite values | consistent |
| The median σ_sig at 3σ is close to the truth. | within 0.15σ plus three standard errors | worst of 8: chi-squared (1 degree of freedom), 200 draws: -0.113 against ±0.2 | consistent |
| The median σ_sig of the standard normal reproduces the record of 12a. | within the range of the record plus half a unit of its last digit plus three standard errors | worst of 6: 1 000 draws, 4σ: +0.0178 against ±0.049 | consistent |
| The median σ_sig of the chi-squared (4 degrees of freedom) reproduces the record of 12a. | within the range of the record plus half a unit of its last digit plus three standard errors | worst of 6: 1 000 draws, 4σ: -0.00401 against ±0.073 | consistent |
| The record of 12a at 10σ and 100σ lies within the replicates. | each recorded value lies between the 16th and 84th percentiles | 4.9 in [4.48, 5.19]; 7.7 in [7.21, 8.55] | consistent |
:::

:::{table} Warnings that the check raised; the messages are in the JSON.

| Category | Count |
| --- | --- |
| ExtrapolationWarning | 26 |
:::

## Effect sizes

(validation-effect-recovery)=
### Recovery of known shifts

Reruns the check of session 12a.

Supports:

- {py:func}`mixed_signals.sigma.maps.sigma_eff`
- {py:func}`mixed_signals.sigma.maps.null_subtract`
- {py:data}`mixed_signals.sigma.maps.SUBTRACTION_SPACES`
- the done criterion of session 12: σ_eff recovers the zoo's shifts

Shifted zoo Gaussians ({py:func}`~mixed_signals.zoo.cases.shifted_gaussians`, 4 parameters) with true shifts 0, 0.2, 0.5, 1, 2, 1 000 samples per set and 12 replicates per shift. Each row uses the default permutation null, then {py:func}`~mixed_signals.sigma.maps.null_subtract` and {py:func}`~mixed_signals.sigma.maps.sigma_eff` with the settings of the row. The marginal metrics read the first parameter, which carries the shift. The JS divergence and the Hellinger distance come from kernel estimates with the bandwidth factor of {py:func}`~mixed_signals.metrics.marginal.shared_bandwidth_factor`, and the Gaussian KL, W2 and Hellinger distance from the weighted moments. MMD squared takes the median bandwidth of the comparison, fixed once, and the Sinkhorn divergence takes ε at 0.25 times its square. Both sliced W2 estimators share 500 directions. The null has 99 draws for the cheap metrics, 49 for the kernel estimates, 30 for the empirical sliced W2 and 20 for the Sinkhorn divergence, fewer than the default 200. The marginal KS, W1 and AD also run at 10 000 samples per set with 5 replicates. The sample means of a replicate differ by a shift of their own, which scatters by about 0.045 at 1 000 samples, and every metric follows it, so a mean σ_eff compared with the true shift carries that scatter. A claim therefore tests the mean of the paired difference, σ_eff minus the sample shift of the same replicate, which has the same expectation as σ_eff minus the truth and a far smaller scatter. The tolerance is 3% of the true shift (6% for the KS statistic, which keeps a small positive bias), plus the standard errors of the paired differences. The zero shift has no claim here; the check {ref}`validation-signed-maps` covers it with 40 replicates.

:::{table} Mean σ_eff over replicates of 1 000 samples per set, with the standard error of the mean, by true shift. The first row is the mean difference of the sample means along the shifted axis, the shift that the sets hold.

| Metric | Shift 0 | Shift 0.2 | Shift 0.5 | Shift 1 | Shift 2 |
| --- | --- | --- | --- | --- | --- |
| Sample shift | -0.005 ± 0.014 | 0.201 ± 0.012 | 0.491 ± 0.010 | 0.9798 ± 0.0090 | 2.011 ± 0.012 |
| KS | -0.002 ± 0.022 | 0.230 ± 0.020 | 0.522 ± 0.013 | 1.019 ± 0.011 | 2.037 ± 0.019 |
| W1 | -0.008 ± 0.013 | 0.192 ± 0.012 | 0.487 ± 0.010 | 0.9775 ± 0.0090 | 2.009 ± 0.012 |
| AD | -0.005 ± 0.015 | 0.197 ± 0.014 | 0.491 ± 0.011 | 0.9790 ± 0.0089 | 2.014 ± 0.018 |
| JS, KDE | -0.024 ± 0.014 | 0.192 ± 0.014 | 0.490 ± 0.012 | 0.9803 ± 0.0082 | 2.015 ± 0.018 |
| Hellinger, KDE | -0.025 ± 0.015 | 0.193 ± 0.014 | 0.492 ± 0.012 | 0.9861 ± 0.0082 | 2.029 ± 0.019 |
| KL, Gaussian | -0.014 ± 0.025 | 0.199 ± 0.012 | 0.493 ± 0.011 | 0.988 ± 0.011 | 2.018 ± 0.020 |
| W2, Gaussian | -0.018 ± 0.021 | 0.197 ± 0.012 | 0.488 ± 0.010 | 0.9793 ± 0.0085 | 2.010 ± 0.012 |
| Hellinger, Gaussian | -0.015 ± 0.025 | 0.197 ± 0.012 | 0.493 ± 0.012 | 0.9807 ± 0.0079 | 2.012 ± 0.017 |
| Energy distance | -0.020 ± 0.019 | 0.197 ± 0.012 | 0.488 ± 0.011 | 0.9802 ± 0.0097 | 2.012 ± 0.014 |
| MMD squared | -0.020 ± 0.019 | 0.197 ± 0.012 | 0.487 ± 0.011 | 0.9808 ± 0.0099 | 2.011 ± 0.014 |
| Sliced W2, Gaussian | -0.027 ± 0.017 | 0.196 ± 0.012 | 0.488 ± 0.010 | 0.9788 ± 0.0088 | 2.010 ± 0.012 |
| Sliced W2, empirical | -0.021 ± 0.019 | 0.193 ± 0.012 | 0.487 ± 0.010 | 0.9782 ± 0.0088 | 2.010 ± 0.012 |
| Sinkhorn divergence | -0.026 ± 0.022 | 0.193 ± 0.012 | 0.485 ± 0.011 | 0.9752 ± 0.0086 | 2.006 ± 0.012 |
:::

:::{table} Paired deviation, σ_eff minus the sample shift of its own replicate, at 1 000 samples per set, with the standard error of the paired differences. A dash marks the zero shift, which no claim covers.

| Metric | Shift 0 | Shift 0.2 | Shift 0.5 | Shift 1 | Shift 2 |
| --- | --- | --- | --- | --- | --- |
| KS | - | 0.029 ± 0.012 | 0.0317 ± 0.0081 | 0.0388 ± 0.0088 | 0.026 ± 0.011 |
| W1 | - | -0.00867 ± 0.00077 | -0.00368 ± 0.00012 | -0.002346 ± 0.000059 | -0.001791 ± 0.000056 |
| AD | - | -0.0044 ± 0.0032 | -0.0001 ± 0.0023 | -0.0009 ± 0.0043 | 0.0030 ± 0.0088 |
| JS, KDE | - | -0.0090 ± 0.0042 | -0.0006 ± 0.0031 | 0.0005 ± 0.0049 | 0.0045 ± 0.0090 |
| Hellinger, KDE | - | -0.0084 ± 0.0043 | 0.0016 ± 0.0032 | 0.0062 ± 0.0054 | 0.018 ± 0.011 |
| KL, Gaussian | - | -0.0017 ± 0.0078 | 0.0020 ± 0.0053 | 0.0084 ± 0.0053 | 0.007 ± 0.016 |
| W2, Gaussian | - | -0.0041 ± 0.0045 | -0.0026 ± 0.0020 | -0.00054 ± 0.00097 | -0.00051 ± 0.00033 |
| Hellinger, Gaussian | - | -0.0036 ± 0.0074 | 0.0024 ± 0.0050 | 0.0008 ± 0.0053 | 0.0010 ± 0.0077 |
| Energy distance | - | -0.0040 ± 0.0024 | -0.0031 ± 0.0022 | 0.0004 ± 0.0019 | 0.0010 ± 0.0028 |
| MMD squared | - | -0.0038 ± 0.0024 | -0.0037 ± 0.0028 | 0.0010 ± 0.0023 | 0.0006 ± 0.0035 |
| Sliced W2, Gaussian | - | -0.0049 ± 0.0028 | -0.0030 ± 0.0015 | -0.00105 ± 0.00088 | -0.00032 ± 0.00028 |
| Sliced W2, empirical | - | -0.0078 ± 0.0032 | -0.0036 ± 0.0017 | -0.00166 ± 0.00087 | -0.00113 ± 0.00035 |
| Sinkhorn divergence | - | -0.0081 ± 0.0048 | -0.0052 ± 0.0034 | -0.0046 ± 0.0014 | -0.00498 ± 0.00046 |
:::

No claim covers the shift of 0.2, which lies below the smallest claimed shift of 0.5 and is where the default threshold of the shift sits. There the paired deviation over the shift exceeds the tolerance of the claims for 6 of 13 metrics, largest first: KS (+14.7 ± 5.8%), JS, KDE (-4.5 ± 2.1%), W1 (-4.3 ± 0.4%), Hellinger, KDE (-4.2 ± 2.1%), Sinkhorn divergence (-4.0 ± 2.4%), Sliced W2, empirical (-3.9 ± 1.6%).

```{image} validation/figures/effect-recovery-shifts-light.svg
:class: only-light
:alt: Five panels of the paired relative deviation against the true shift, scattered around zero within a narrow band.
```

```{image} validation/figures/effect-recovery-shifts-dark.svg
:class: only-dark
:alt: Five panels of the paired relative deviation against the true shift, scattered around zero within a narrow band.
```

**Figure.** Paired deviation of σ_eff from the sample shift of its own replicate, over the true shift, against the true shift. The line is zero, the band is the 3% tolerance of the claims (6% for the KS statistic) and the bars are standard errors of the paired differences. The claims cover the shifts 0.5 to 2; the points at 0.2 carry no claim.

:::{table} Mean σ_eff of the marginal metrics at 10 000 samples per set, and the paired deviation from the sample shift in the second block, which has a dash at the zero shift.

| Metric | Shift 0 | Shift 0.2 | Shift 0.5 | Shift 1 | Shift 2 |
| --- | --- | --- | --- | --- | --- |
| Sample shift | -0.0037 ± 0.0079 | 0.2053 ± 0.0081 | 0.4953 ± 0.0045 | 0.9962 ± 0.0066 | 1.9987 ± 0.0038 |
| KS | 0.005 ± 0.011 | 0.2128 ± 0.0096 | 0.5062 ± 0.0034 | 1.004 ± 0.011 | 2.0160 ± 0.0100 |
| W1 | 0.0029 ± 0.0067 | 0.2044 ± 0.0081 | 0.4949 ± 0.0045 | 0.9959 ± 0.0066 | 1.9985 ± 0.0038 |
| AD | 0.0035 ± 0.0075 | 0.2044 ± 0.0078 | 0.4963 ± 0.0037 | 0.9962 ± 0.0091 | 2.0069 ± 0.0074 |
| KS, paired | - | 0.0075 ± 0.0074 | 0.0109 ± 0.0035 | 0.0080 ± 0.0057 | 0.0173 ± 0.0084 |
| W1, paired | - | -0.000849 ± 0.000035 | -0.000391 ± 0.000023 | -0.000230 ± 0.000012 | -0.0001707 ± 0.0000069 |
| AD, paired | - | -0.0009 ± 0.0011 | 0.0010 ± 0.0030 | 0.0001 ± 0.0032 | 0.0082 ± 0.0059 |
:::

:::{table} Mean σ_eff of the metrics that subtract in squares, when the null mean of the values is subtracted plainly instead: they read low.

| Metric | Shift 0 | Shift 0.2 | Shift 0.5 | Shift 1 | Shift 2 |
| --- | --- | --- | --- | --- | --- |
| KS | 0.0036 ± 0.0091 | 0.157 ± 0.017 | 0.435 ± 0.013 | 0.916 ± 0.010 | 1.885 ± 0.018 |
| W1 | 0.0012 ± 0.0055 | 0.145 ± 0.012 | 0.4335 ± 0.0099 | 0.9159 ± 0.0088 | 1.931 ± 0.012 |
| Hellinger, KDE | -0.0054 ± 0.0036 | 0.107 ± 0.011 | 0.383 ± 0.012 | 0.8638 ± 0.0079 | 1.875 ± 0.020 |
| W2, Gaussian | -0.0030 ± 0.0064 | 0.1084 ± 0.0094 | 0.374 ± 0.010 | 0.8536 ± 0.0079 | 1.872 ± 0.012 |
| Hellinger, Gaussian | -0.0011 ± 0.0079 | 0.0963 ± 0.0091 | 0.354 ± 0.011 | 0.8192 ± 0.0071 | 1.793 ± 0.015 |
:::

:::{table} Claims of the check, each with its verdict.

| Claim | Expected | Observed | Verdict |
| --- | --- | --- | --- |
| KS recovers the sample shift at 0.5, 1, 2. | within 6% of the truth plus 3.8 standard errors (Student t, 12 replicates) | worst of 3: shift 0.5: +0.0317 against ±0.061 | consistent |
| W1 recovers the sample shift at 0.5, 1, 2. | within 3% of the truth plus 3.8 standard errors (Student t, 12 replicates) | worst of 3: shift 0.5: -0.00368 against ±0.015 | consistent |
| AD recovers the sample shift at 0.5, 1, 2. | within 3% of the truth plus 3.8 standard errors (Student t, 12 replicates) | worst of 3: shift 0.5: -9.47e-05 against ±0.024 | consistent |
| JS, KDE recovers the sample shift at 0.5, 1, 2. | within 3% of the truth plus 3.8 standard errors (Student t, 12 replicates) | worst of 3: shift 0.5: -0.000626 against ±0.027 | consistent |
| Hellinger, KDE recovers the sample shift at 0.5, 1, 2. | within 3% of the truth plus 3.8 standard errors (Student t, 12 replicates) | worst of 3: shift 0.5: +0.00159 against ±0.027 | consistent |
| KL, Gaussian recovers the sample shift at 0.5, 1, 2. | within 3% of the truth plus 3.8 standard errors (Student t, 12 replicates) | worst of 3: shift 0.5: +0.00202 against ±0.036 | consistent |
| W2, Gaussian recovers the sample shift at 0.5, 1, 2. | within 3% of the truth plus 3.8 standard errors (Student t, 12 replicates) | worst of 3: shift 0.5: -0.00255 against ±0.023 | consistent |
| Hellinger, Gaussian recovers the sample shift at 0.5, 1, 2. | within 3% of the truth plus 3.8 standard errors (Student t, 12 replicates) | worst of 3: shift 0.5: +0.00242 against ±0.034 | consistent |
| Energy distance recovers the sample shift at 0.5, 1, 2. | within 3% of the truth plus 3.8 standard errors (Student t, 12 replicates) | worst of 3: shift 0.5: -0.00311 against ±0.023 | consistent |
| MMD squared recovers the sample shift at 0.5, 1, 2. | within 3% of the truth plus 3.8 standard errors (Student t, 12 replicates) | worst of 3: shift 0.5: -0.00373 against ±0.026 | consistent |
| Sliced W2, Gaussian recovers the sample shift at 0.5, 1, 2. | within 3% of the truth plus 3.8 standard errors (Student t, 12 replicates) | worst of 3: shift 0.5: -0.00303 against ±0.021 | consistent |
| Sliced W2, empirical recovers the sample shift at 0.5, 1, 2. | within 3% of the truth plus 3.8 standard errors (Student t, 12 replicates) | worst of 3: shift 0.5: -0.0036 against ±0.021 | consistent |
| Sinkhorn divergence recovers the sample shift at 0.5, 1, 2. | within 3% of the truth plus 3.8 standard errors (Student t, 12 replicates) | worst of 3: shift 0.5: -0.00523 against ±0.028 | consistent |
| KS recovers the sample shift at 10 000 samples per set. | within 6% of the truth plus 6.6 standard errors (Student t, 5 replicates) | worst of 3: shift 0.5: +0.0109 against ±0.053 | consistent |
| W1 recovers the sample shift at 10 000 samples per set. | within 3% of the truth plus 6.6 standard errors (Student t, 5 replicates) | worst of 3: shift 0.5: -0.000391 against ±0.015 | consistent |
| AD recovers the sample shift at 10 000 samples per set. | within 3% of the truth plus 6.6 standard errors (Student t, 5 replicates) | worst of 3: shift 0.5: +0.000995 against ±0.035 | consistent |
:::

The check raised no warning.

(validation-width-recovery)=
### Recovery of known width ratios

Reruns the check of session 12a.

Supports:

- {py:func}`mixed_signals.sigma.maps.r_eff`
- the done criterion of session 12: r_eff recovers the zoo's width ratios

Zoo Gaussians ({py:func}`~mixed_signals.zoo.cases.scaled_gaussians`, 4 parameters) with the first axis scaled by 0.8, 1.25 and 1.5, 1 000 samples per set, 60 replicates and 99 null draws per replicate. The width ratio of the first parameter is subtracted in logarithms, and the shape terms of the Gaussian KL, W2 and Hellinger distance in the space of their metric; the sign of the log-determinant ratio chooses the side of one that r_eff takes. A claim is consistent if the mean lies within 3% of the true ratio plus its standard errors.

:::{table} Mean r_eff over 60 replicates, with the standard error of the mean, by true width ratio.

| Metric | Ratio 0.8 | Ratio 1.25 | Ratio 1.5 |
| --- | --- | --- | --- |
| Width ratio | 0.8000 ± 0.0035 | 1.2424 ± 0.0052 | 1.5098 ± 0.0065 |
| KL, shape term | 0.7999 ± 0.0036 | 1.2406 ± 0.0060 | 1.5042 ± 0.0068 |
| W2, shape term | 0.8021 ± 0.0040 | 1.2416 ± 0.0052 | 1.5043 ± 0.0055 |
| Hellinger, shape term | 0.8020 ± 0.0037 | 1.2421 ± 0.0058 | 1.5067 ± 0.0067 |
:::

:::{table} Claims of the check, each with its verdict.

| Claim | Expected | Observed | Verdict |
| --- | --- | --- | --- |
| Width ratio recovers the width ratios 0.8, 1.25, 1.5. | within 3% of the truth plus 3.1 standard errors (Student t, 60 replicates) | worst of 3: ratio 0.8: -2.68e-05 against ±0.035 | consistent |
| KL, shape term recovers the width ratios 0.8, 1.25, 1.5. | within 3% of the truth plus 3.1 standard errors (Student t, 60 replicates) | worst of 3: ratio 0.8: -0.000114 against ±0.035 | consistent |
| W2, shape term recovers the width ratios 0.8, 1.25, 1.5. | within 3% of the truth plus 3.1 standard errors (Student t, 60 replicates) | worst of 3: ratio 0.8: +0.00211 against ±0.037 | consistent |
| Hellinger, shape term recovers the width ratios 0.8, 1.25, 1.5. | within 3% of the truth plus 3.1 standard errors (Student t, 60 replicates) | worst of 3: ratio 0.8: +0.00202 against ±0.036 | consistent |
:::

The check raised no warning.

(validation-signed-maps)=
### Signed maps on identical sets

Reruns the check of session 12a.

Supports:

- {py:func}`mixed_signals.sigma.maps.sigma_eff`
- the signed maps for values below the null mean

Two sets of 1 000 samples of one zoo Gaussian (4 parameters). The rows of the cheap group (KS, W1, AD, KL, Gaussian, W2, Gaussian, Hellinger, Gaussian, Energy distance, MMD squared, Sliced W2, Gaussian) take 40 replicates with 49 null draws. The other rows take 10 replicates: the kernel estimates (JS, KDE, Hellinger, KDE) with 29 draws, the empirical sliced W2 with 20 and the Sinkhorn divergence with 15, so the null draws run from 15 to 49. Each row is null-subtracted and mapped to σ_eff with the signed map of the package. For contrast the script clips the same σ_eff at zero, which a map that gives NaN or zero below the null would do. Clipping raises the mean by construction, because a clipped value is never below its signed value; the table shows by how much. The claims cover the signed σ_eff only.

:::{table} Identical sets: the share of negative null-subtracted values, and the mean and scatter of the signed σ_eff and of σ_eff clipped at zero.

| Metric | Replicates | Negative share | Signed mean | Signed scatter | Clipped mean | Clipped scatter |
| --- | --- | --- | --- | --- | --- | --- |
| KS | 40 | 0.700 ± 0.073 | -0.0166 ± 0.0099 | 0.0629 ± 0.0071 | 0.0212 ± 0.0059 | 0.0371 ± 0.0042 |
| W1 | 40 | 0.675 ± 0.075 | -0.0123 ± 0.0063 | 0.0401 ± 0.0045 | 0.0126 ± 0.0038 | 0.0238 ± 0.0027 |
| AD | 40 | 0.725 ± 0.071 | -0.0138 ± 0.0067 | 0.0422 ± 0.0048 | 0.0133 ± 0.0041 | 0.0258 ± 0.0029 |
| JS, KDE | 10 | 0.30 ± 0.15 | 0.019 ± 0.021 | 0.067 ± 0.016 | 0.042 ± 0.010 | 0.0319 ± 0.0075 |
| Hellinger, KDE | 10 | 0.30 ± 0.15 | 0.019 ± 0.021 | 0.067 ± 0.016 | 0.042 ± 0.010 | 0.0322 ± 0.0076 |
| KL, Gaussian | 40 | 0.525 ± 0.079 | 0.006 ± 0.017 | 0.105 ± 0.012 | 0.0519 ± 0.0098 | 0.0622 ± 0.0070 |
| W2, Gaussian | 40 | 0.500 ± 0.079 | 0.007 ± 0.014 | 0.0860 ± 0.0097 | 0.0431 ± 0.0080 | 0.0507 ± 0.0057 |
| Hellinger, Gaussian | 40 | 0.525 ± 0.079 | 0.007 ± 0.017 | 0.106 ± 0.012 | 0.0526 ± 0.0099 | 0.0628 ± 0.0071 |
| Energy distance | 40 | 0.575 ± 0.078 | -0.001 ± 0.012 | 0.0766 ± 0.0087 | 0.0347 ± 0.0073 | 0.0462 ± 0.0052 |
| MMD squared | 40 | 0.550 ± 0.079 | 0.000 ± 0.012 | 0.0781 ± 0.0088 | 0.0352 ± 0.0075 | 0.0475 ± 0.0054 |
| Sliced W2, Gaussian | 40 | 0.575 ± 0.078 | 0.002 ± 0.012 | 0.0774 ± 0.0088 | 0.0362 ± 0.0074 | 0.0470 ± 0.0053 |
| Sliced W2, empirical | 10 | 0.50 ± 0.16 | 0.025 ± 0.024 | 0.075 ± 0.018 | 0.045 ± 0.017 | 0.053 ± 0.012 |
| Sinkhorn divergence | 10 | 0.50 ± 0.16 | -0.004 ± 0.030 | 0.095 ± 0.022 | 0.040 ± 0.017 | 0.054 ± 0.013 |
:::

:::{table} Claims of the check, each with its verdict.

| Claim | Expected | Observed | Verdict |
| --- | --- | --- | --- |
| KS: signed σ_eff. | within 3.2 standard errors (Student t, 40 replicates) | worst of 1: mean: -0.0166 against ±0.032 | consistent |
| W1: signed σ_eff. | within 3.2 standard errors (Student t, 40 replicates) | worst of 1: mean: -0.0123 against ±0.02 | consistent |
| AD: signed σ_eff. | within 3.2 standard errors (Student t, 40 replicates) | worst of 1: mean: -0.0138 against ±0.021 | consistent |
| JS, KDE: signed σ_eff. | within 4.1 standard errors (Student t, 10 replicates) | worst of 1: mean: +0.0195 against ±0.087 | consistent |
| Hellinger, KDE: signed σ_eff. | within 4.1 standard errors (Student t, 10 replicates) | worst of 1: mean: +0.0193 against ±0.087 | consistent |
| KL, Gaussian: signed σ_eff. | within 3.2 standard errors (Student t, 40 replicates) | worst of 1: mean: +0.00648 against ±0.053 | consistent |
| W2, Gaussian: signed σ_eff. | within 3.2 standard errors (Student t, 40 replicates) | worst of 1: mean: +0.00716 against ±0.044 | consistent |
| Hellinger, Gaussian: signed σ_eff. | within 3.2 standard errors (Student t, 40 replicates) | worst of 1: mean: +0.00708 against ±0.054 | consistent |
| Energy distance: signed σ_eff. | within 3.2 standard errors (Student t, 40 replicates) | worst of 1: mean: -0.000542 against ±0.039 | consistent |
| MMD squared: signed σ_eff. | within 3.2 standard errors (Student t, 40 replicates) | worst of 1: mean: +0.000162 against ±0.04 | consistent |
| Sliced W2, Gaussian: signed σ_eff. | within 3.2 standard errors (Student t, 40 replicates) | worst of 1: mean: +0.00173 against ±0.039 | consistent |
| Sliced W2, empirical: signed σ_eff. | within 4.1 standard errors (Student t, 10 replicates) | worst of 1: mean: +0.0247 against ±0.098 | consistent |
| Sinkhorn divergence: signed σ_eff. | within 4.1 standard errors (Student t, 10 replicates) | worst of 1: mean: -0.00421 against ±0.12 | consistent |
:::

The check raised no warning.

(validation-threshold-basis)=
### Basis of the default thresholds

Reruns the check of session 12a.

Supports:

- {py:data}`mixed_signals.sigma.thresholds.SIGMA_EFF_THRESHOLD`
- {py:data}`mixed_signals.sigma.thresholds.LOG_R_EFF_THRESHOLD`

The GWTC-2 catalogue ({cite:t}`abbott2021gwtc`, Appendix A) deems a JS divergence of 0.007 bits between waveform models significant, which for a Gaussian corresponds to a 20% shift in the mean, measured in units of one standard deviation. The check is exact and has no replicates. The JS divergence of N(0, 1) and N(0.2, 1) comes from {py:func}`~mixed_signals.zoo.closed_forms.normal_js` in nats, and the width ratios on either side of one that give the same divergence come from a bisection on `normal_js(0, 1, 0, r)` (60 steps).

:::{table} Gaussians with the JS divergence of a 0.2σ shift: the shift, the width ratios on either side of one and their natural logarithms.

| Case | Shift or ratio | JS divergence (bits) | ln r |
| --- | --- | --- | --- |
| Shift of 0.2σ | 0.2 | 0.007178 | - |
| Width ratio above one | 1.153 | 0.007178 | 0.1423 |
| Width ratio below one | 0.8673 | 0.007178 | -0.1423 |
:::

:::{table} Claims of the check, each with its verdict.

| Claim | Expected | Observed | Verdict |
| --- | --- | --- | --- |
| A 0.2σ shift has the JS divergence that GWTC-2 deems significant. | 0.0072 bits, at least the 0.007 bits of the catalogue | 0.00718 bits | consistent |
| The shift threshold is 0.2. | 0.2 | 0.2 | consistent |
| The width ratios of the same JS divergence are 1.153 and 0.867. | 1.153 and 0.867 to three digits | 1.153 and 0.867 | consistent |
| The threshold of the logarithm of r_eff is 0.14. | 0.14 | 0.1423 and 0.1423 | consistent |
:::

The check raised no warning.

(validation-knn-kl-calibration)=
### Calibration of the kNN KL map

Reruns the check of session 12a.

Supports:

- {py:func}`mixed_signals.sigma.calibration.knn_kl_curve`
- {py:data}`mixed_signals.sigma.calibration.KNN_KL_SHIFTS`

Unit Gaussians, the second set shifted along one axis, with common random numbers across shifts as {py:func}`~mixed_signals.sigma.calibration.knn_kl_curve` draws them. The estimate is {py:func}`~mixed_signals.metrics.divergence.knn_kl_divergence` with k = 3, averaged over both directions, and its null is the mean estimate at zero shift of the same replicate. Replicates: 16 at N = 1 000, 4 at N = 3 000, 2 at N = 10 000. The recovery uses 16 comparisons per shift with 30 permutation draws, each with its own calibration curve. A claim tests the mean of the paired difference between σ_eff and the sample shift of the same replicate, the difference of the sample means along the shifted axis; the tolerance is 5% of the true shift plus the standard errors of the paired differences.

:::{table} The null-subtracted kNN KL (k = 3, both directions averaged) over the true KL of unit Gaussians, δ²/2, by sample count, dimension and shift.

| N | D | Replicates | Shift 0.5 | Shift 1 | Shift 2 | Shift 3 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 000 | 1 | 16 | 1.023 ± 0.049 | 0.932 ± 0.018 | 0.892 ± 0.011 | 0.7825 ± 0.0071 |
| 1 000 | 4 | 16 | 0.875 ± 0.065 | 0.846 ± 0.027 | 0.7906 ± 0.0093 | 0.6973 ± 0.0052 |
| 1 000 | 16 | 16 | 0.737 ± 0.035 | 0.703 ± 0.017 | 0.6633 ± 0.0064 | 0.6213 ± 0.0039 |
| 3 000 | 1 | 4 | 0.82 ± 0.10 | 0.946 ± 0.024 | 0.9286 ± 0.0083 | 0.8439 ± 0.0069 |
| 3 000 | 4 | 4 | 0.801 ± 0.070 | 0.872 ± 0.034 | 0.820 ± 0.012 | 0.7387 ± 0.0051 |
| 3 000 | 16 | 4 | 0.762 ± 0.025 | 0.716 ± 0.013 | 0.6811 ± 0.0060 | 0.6441 ± 0.0047 |
| 10 000 | 1 | 2 | 0.9868 ± 0.0061 | 1.0072 ± 0.0014 | 0.9909 ± 0.0076 | 0.92421 ± 0.00023 |
| 10 000 | 4 | 2 | 0.949 ± 0.055 | 0.925 ± 0.028 | 0.8805 ± 0.0061 | 0.799915 ± 0.000050 |
| 10 000 | 16 | 2 | 0.779 ± 0.011 | 0.74725 ± 0.00038 | 0.7106 ± 0.0026 | 0.6707 ± 0.0045 |
:::

```{image} validation/figures/knn-kl-calibration-ratio-light.svg
:class: only-light
:alt: Lines of the estimate over the truth against the shift, below one for four and sixteen dimensions and falling with the shift.
```

```{image} validation/figures/knn-kl-calibration-ratio-dark.svg
:class: only-dark
:alt: Lines of the estimate over the truth against the shift, below one for four and sixteen dimensions and falling with the shift.
```

**Figure.** Null-subtracted kNN KL over its truth against the shift, by sample count and dimension; the line is one.

:::{table} σ_eff of the kNN KL at D = 4 and N = 1 000: the run-time calibration curve against √(2 KL), each with its paired deviation from the sample shift.

| True shift | Sample shift | Calibrated σ_eff | Paired deviation | √(2 KL) | Paired deviation, √(2 KL) |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.9821 ± 0.0082 | 0.966 ± 0.012 | -0.016 ± 0.012 | 0.9029 ± 0.0093 | -0.0791 ± 0.0078 |
| 2 | 2.0216 ± 0.0065 | 2.013 ± 0.014 | -0.009 ± 0.012 | 1.774 ± 0.010 | -0.2480 ± 0.0085 |
:::

:::{table} Claims of the check, each with its verdict.

| Claim | Expected | Observed | Verdict |
| --- | --- | --- | --- |
| The calibrated σ_eff recovers the sample shift of 1. | within 5% of the truth plus 3.6 standard errors (Student t, 16 replicates) | worst of 1: shift 1: -0.0159 against ±0.094 | consistent |
| The map √(2 KL) reads the sample shift of 1 low. | paired deviation below zero by more than 3.6 standard errors (Student t, 16 replicates) | -0.0791 ± 0.0078 | consistent |
| The calibrated σ_eff recovers the sample shift of 2. | within 5% of the truth plus 3.6 standard errors (Student t, 16 replicates) | worst of 1: shift 2: -0.00898 against ±0.14 | consistent |
| The map √(2 KL) reads the sample shift of 2 low. | paired deviation below zero by more than 3.6 standard errors (Student t, 16 replicates) | -0.2480 ± 0.0085 | consistent |
:::

The check raised no warning.

## Optimal transport

(validation-sinkhorn-epsilon)=
### Sinkhorn regularisation and its null band

Reruns the check of session 10.

Supports:

- {py:data}`mixed_signals.metrics.ot.sinkhorn.SINKHORN_EPSILON_FACTOR`
- {py:func}`mixed_signals.metrics.ot.sinkhorn.sinkhorn_divergence`

Whitened zoo Gaussians (unit covariance) in 2, 8, 16 dimensions with 1 000 samples per set, a shift of 0.5σ and a width ratio of 1.3 on one axis. ε runs over 0.05, 0.1, 0.25, 0.5, 1, 2, 4 times ℓ², with ℓ the median bandwidth of the comparison, fixed once. One call of {py:func}`~mixed_signals.metrics.ot.sinkhorn.sinkhorn_divergence` with the vector of ε gives the curve, and {py:func}`~mixed_signals.nulls.plan.null_draws` gives the curve of each of 4 draws of the default permutation null for the shift, with 16 replicates, which the claim on the shift needs, and of 6 draws with 6 replicates for the width ratio. The default ε of the package is 0.25 ℓ². Iterations that do not converge give NaN; the table counts the replicates that converged and averages the others. A claim on the width compares the resolution at 0.25 ℓ² with the best one, allowing the standard errors of both. A claim on the shift tests the mean paired deviation of the recovered shift from the sample shift of the same replicate, which is within 3% of the true shift plus its margin of 3.6 standard errors (Student t, 16 replicates), at every ε of 0.25 ℓ² and above in each dimension. At D = 16 a further claim says that the root reads low at 0.05 and 0.1 ℓ², by more than its margin. No claim covers the smaller ε at D = 2 or D = 8; at D = 8 the paired deviation lies 4.1 to 4.8 standard errors below zero, too close to the margin of 3.6 standard errors for a claim, as the table shows.

:::{table} Resolution, in null widths, of a 0.5σ shift and of a width ratio of 1.3, the shift recovered as the root of the null-subtracted S_ε, the difference of the sample means along the shifted axis and the paired deviation of the two, by dimension and ε in units of ℓ², the squared median bandwidth.

| D | ε over ℓ² | Converged | Shift resolution | Width resolution | Recovered shift | Sample shift | Paired deviation |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 0.05 | 16 of 16 | 55.0 ± 7.0 | 18.1 ± 3.5 | 0.502 ± 0.013 | 0.505 ± 0.013 | -0.0029 ± 0.0015 |
| 2 | 0.1 | 16 of 16 | 55.5 ± 7.1 | 18.1 ± 3.6 | 0.502 ± 0.013 | 0.505 ± 0.013 | -0.0028 ± 0.0014 |
| 2 | 0.25 | 16 of 16 | 58.0 ± 7.6 | 17.9 ± 3.2 | 0.502 ± 0.013 | 0.505 ± 0.013 | -0.0028 ± 0.0013 |
| 2 | 0.5 | 16 of 16 | 62.0 ± 8.5 | 17.5 ± 2.8 | 0.502 ± 0.013 | 0.505 ± 0.013 | -0.0026 ± 0.0012 |
| 2 | 1 | 16 of 16 | 72 ± 12 | 16.5 ± 2.5 | 0.503 ± 0.013 | 0.505 ± 0.013 | -0.0021 ± 0.0011 |
| 2 | 2 | 16 of 16 | 84 ± 16 | 13.6 ± 2.3 | 0.503 ± 0.013 | 0.505 ± 0.013 | -0.0016 ± 0.0010 |
| 2 | 4 | 16 of 16 | 85 ± 13 | 9.0 ± 1.9 | 0.504 ± 0.013 | 0.505 ± 0.013 | -0.0014 ± 0.0010 |
| 8 | 0.05 | 16 of 16 | 14.9 ± 2.9 | 3.84 ± 0.75 | 0.457 ± 0.013 | 0.484 ± 0.011 | -0.0270 ± 0.0056 |
| 8 | 0.1 | 16 of 16 | 16.9 ± 2.3 | 4.37 ± 0.74 | 0.464 ± 0.012 | 0.484 ± 0.011 | -0.0198 ± 0.0048 |
| 8 | 0.25 | 16 of 16 | 21.5 ± 1.9 | 5.8 ± 1.1 | 0.477 ± 0.012 | 0.484 ± 0.011 | -0.0074 ± 0.0039 |
| 8 | 0.5 | 16 of 16 | 27.2 ± 2.5 | 5.5 ± 1.2 | 0.481 ± 0.012 | 0.484 ± 0.011 | -0.0028 ± 0.0033 |
| 8 | 1 | 16 of 16 | 32.4 ± 3.6 | 3.8 ± 1.0 | 0.483 ± 0.012 | 0.484 ± 0.011 | -0.0016 ± 0.0029 |
| 8 | 2 | 16 of 16 | 35.8 ± 4.7 | 2.29 ± 0.97 | 0.483 ± 0.012 | 0.484 ± 0.011 | -0.0013 ± 0.0028 |
| 8 | 4 | 16 of 16 | 37.7 ± 5.5 | 1.42 ± 0.95 | 0.483 ± 0.012 | 0.484 ± 0.011 | -0.0012 ± 0.0028 |
| 16 | 0.05 | 16 of 16 | 8.85 ± 0.67 | 2.11 ± 0.52 | 0.451 ± 0.015 | 0.497 ± 0.014 | -0.0463 ± 0.0087 |
| 16 | 0.1 | 16 of 16 | 12.7 ± 1.1 | 2.74 ± 0.48 | 0.459 ± 0.013 | 0.497 ± 0.014 | -0.0383 ± 0.0061 |
| 16 | 0.25 | 16 of 16 | 22.4 ± 2.8 | 3.96 ± 0.83 | 0.492 ± 0.013 | 0.497 ± 0.014 | -0.0051 ± 0.0045 |
| 16 | 0.5 | 16 of 16 | 22.5 ± 2.0 | 2.77 ± 0.81 | 0.497 ± 0.013 | 0.497 ± 0.014 | -0.0001 ± 0.0041 |
| 16 | 1 | 16 of 16 | 23.6 ± 2.8 | 1.52 ± 0.65 | 0.498 ± 0.013 | 0.497 ± 0.014 | 0.0005 ± 0.0038 |
| 16 | 2 | 16 of 16 | 24.7 ± 3.9 | 0.82 ± 0.53 | 0.498 ± 0.013 | 0.497 ± 0.014 | 0.0006 ± 0.0036 |
| 16 | 4 | 16 of 16 | 25.0 ± 4.3 | 0.47 ± 0.46 | 0.498 ± 0.013 | 0.497 ± 0.014 | 0.0006 ± 0.0035 |
:::

:::{table} Wall time of one call of the Sinkhorn divergence with the vector of ε.

| D | ε values | Seconds per call |
| --- | --- | --- |
| 2 | 7 | 0.434 ± 0.012 |
| 8 | 7 | 0.2964 ± 0.0052 |
| 16 | 7 | 0.2723 ± 0.0033 |
:::

```{image} validation/figures/sinkhorn-epsilon-curve-light.svg
:class: only-light
:alt: Three panels of S_ε against ε on logarithmic axes, the value above the null band and approaching the squared shift at large ε.
```

```{image} validation/figures/sinkhorn-epsilon-curve-dark.svg
:class: only-dark
:alt: Three panels of S_ε against ε on logarithmic axes, the value above the null band and approaching the squared shift at large ε.
```

**Figure.** The Sinkhorn divergence of a 0.5σ shift against ε, with the 16th to 84th percentile band of the permutation null; the line is the true value, the squared shift. Values are means over replicates.

:::{table} Claims of the check, each with its verdict.

| Claim | Expected | Observed | Verdict |
| --- | --- | --- | --- |
| D = 2: the default ε is near the best width resolution. | resolution at 0.25 over ℓ² within the standard errors of the best | best 18.1 ± 3.6 at 0.1; default 17.9 ± 3.2 | consistent |
| D = 2: the root of S_ε recovers the sample shift at every ε of 0.25 ℓ² and above. | within 3% of the truth plus 3.6 standard errors (Student t, 16 replicates) | worst of 5: ε = 0.5 ℓ²: -0.00258 against ±0.019 | consistent |
| D = 8: the default ε is near the best width resolution. | resolution at 0.25 over ℓ² within the standard errors of the best | best 5.8 ± 1.1 at 0.25; default 5.8 ± 1.1 | consistent |
| D = 8: the root of S_ε recovers the sample shift at every ε of 0.25 ℓ² and above. | within 3% of the truth plus 3.6 standard errors (Student t, 16 replicates) | worst of 5: ε = 0.25 ℓ²: -0.00741 against ±0.029 | consistent |
| D = 16: the default ε is near the best width resolution. | resolution at 0.25 over ℓ² within the standard errors of the best | best 3.96 ± 0.83 at 0.25; default 3.96 ± 0.83 | consistent |
| D = 16: the root of S_ε recovers the sample shift at every ε of 0.25 ℓ² and above. | within 3% of the truth plus 3.6 standard errors (Student t, 16 replicates) | worst of 5: ε = 0.25 ℓ²: -0.00514 against ±0.031 | consistent |
| D = 16: at ε of 0.05 ℓ² and 0.1 ℓ² the root of S_ε reads low. | paired deviation below zero by more than its margin at both ε | ε = 0.05 ℓ²: -0.0463 ± 0.0087 (margin 0.0311); ε = 0.1 ℓ²: -0.0383 ± 0.0061 (margin 0.0218) | consistent |
:::

The check raised no warning.

(validation-sliced-directions)=
### Directions of the sliced metrics

Reruns the check of session 12b.

Supports:

- {py:func}`mixed_signals.metrics.ot.sliced.sliced_directions`
- {py:data}`mixed_signals.metrics.ot.sliced.SLICED_DIRECTIONS`
- {py:data}`mixed_signals.metrics.ot.sliced.SLICED_SINKHORN_DIRECTIONS`

The Gaussian form {py:func}`~mixed_signals.metrics.ot.sliced.sliced_wasserstein_of_gaussians` gives the sliced W2 value of two Gaussians at no sampling cost, so the script draws the directions 200 times for each number of coordinates n and each number of directions, with {py:func}`~mixed_signals.metrics.ot.sliced.sliced_directions` as independent unit vectors and as complete orthonormal bases. The cases are a 1σ shift along one axis and one axis 1.5 times wider, in n real coordinates. The relative scatter is the standard deviation over the draws divided by the mean, with the standard error of a standard deviation. For independent directions the mean of L squared projections of a shift, each distributed as Beta(1/2, (n - 1)/2), has the relative scatter √(2(n - 1)/((n + 2)L)). The second part reruns the resolution of a permutation null with the empirical sliced W2, paired over the number of directions, and with the sliced Sinkhorn divergence at its default directions. The cost of the latter limits its replicates and draws, and it runs at a cap of 500 samples per set. A NaN draw is left out of the null mean and spread, and a replicate with a NaN value is left out of the resolution; the table counts both.

:::{table} Relative scatter, in per cent, of the sliced W2 value of two Gaussians over 200 draws of the directions, for a 1σ shift along one axis; the columns are the numbers of directions asked for. Orthonormal bases leave no scatter for a shift beyond round-off, which the first claim checks.

| n | Directions | 20 | 50 | 100 | 200 | 500 | 1000 | 2000 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4 | independent | 22.6 ± 1.1 | 12.96 ± 0.65 | 10.03 ± 0.50 | 7.19 ± 0.36 | 4.42 ± 0.22 | 3.03 ± 0.15 | 2.35 ± 0.12 |
| 8 | independent | 28.6 ± 1.4 | 16.76 ± 0.84 | 11.05 ± 0.55 | 8.77 ± 0.44 | 5.53 ± 0.28 | 3.49 ± 0.17 | 2.85 ± 0.14 |
| 16 | independent | 28.8 ± 1.4 | 19.30 ± 0.97 | 14.18 ± 0.71 | 9.10 ± 0.46 | 6.34 ± 0.32 | 4.19 ± 0.21 | 2.84 ± 0.14 |
| 32 | independent | 28.8 ± 1.4 | 18.11 ± 0.91 | 14.08 ± 0.71 | 9.49 ± 0.48 | 6.12 ± 0.31 | 4.65 ± 0.23 | 2.91 ± 0.15 |
:::

:::{table} Relative scatter, in per cent, of the sliced W2 value of two Gaussians over 200 draws of the directions, for one axis 1.5 times wider; the columns are the numbers of directions asked for.

| n | Directions | 20 | 50 | 100 | 200 | 500 | 1000 | 2000 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4 | independent | 33.0 ± 1.7 | 19.10 ± 0.96 | 14.68 ± 0.74 | 10.67 ± 0.53 | 6.49 ± 0.33 | 4.37 ± 0.22 | 3.56 ± 0.18 |
| 4 | orthonormal | 10.50 ± 0.53 | 6.62 ± 0.33 | 4.74 ± 0.24 | 3.45 ± 0.17 | 1.892 ± 0.095 | 1.558 ± 0.078 | 1.036 ± 0.052 |
| 8 | independent | 47.9 ± 2.4 | 26.7 ± 1.3 | 19.30 ± 0.97 | 14.73 ± 0.74 | 9.35 ± 0.47 | 5.95 ± 0.30 | 4.67 ± 0.23 |
| 8 | orthonormal | 14.87 ± 0.75 | 9.76 ± 0.49 | 7.25 ± 0.36 | 4.93 ± 0.25 | 3.44 ± 0.17 | 2.49 ± 0.12 | 1.664 ± 0.083 |
| 16 | independent | 55.0 ± 2.8 | 34.8 ± 1.7 | 26.5 ± 1.3 | 16.91 ± 0.85 | 11.52 ± 0.58 | 7.74 ± 0.39 | 5.28 ± 0.26 |
| 16 | orthonormal | 16.87 ± 0.85 | 11.88 ± 0.60 | 8.88 ± 0.44 | 6.75 ± 0.34 | 4.21 ± 0.21 | 3.11 ± 0.16 | 2.30 ± 0.12 |
| 32 | independent | 57.2 ± 2.9 | 35.9 ± 1.8 | 27.6 ± 1.4 | 19.46 ± 0.98 | 12.13 ± 0.61 | 9.30 ± 0.47 | 5.85 ± 0.29 |
| 32 | orthonormal | 21.7 ± 1.1 | 14.31 ± 0.72 | 11.13 ± 0.56 | 7.74 ± 0.39 | 5.23 ± 0.26 | 3.33 ± 0.17 | 2.77 ± 0.14 |
:::

```{image} validation/figures/sliced-directions-scatter-light.svg
:class: only-light
:alt: Four panels of relative scatter against the number of directions on logarithmic axes, falling as the inverse square root of the number of directions for independent directions and lower for orthonormal bases.
```

```{image} validation/figures/sliced-directions-scatter-dark.svg
:class: only-dark
:alt: Four panels of relative scatter against the number of directions on logarithmic axes, falling as the inverse square root of the number of directions for independent directions and lower for orthonormal bases.
```

**Figure.** Relative scatter of the sliced W2 value over draws of the directions against the number of directions, one panel per number of coordinates n; the line is the root of 2(n - 1)/((n + 2)L). The scatter of a shift with orthonormal bases is zero and is not drawn.

:::{table} Resolution, in null widths, of a 0.5σ shift and of a width ratio of 1.3 with 1 000 samples per set, by sliced W2 with growing numbers of directions and by the sliced Sinkhorn divergence at its default directions. For sliced W2 every number of directions uses the same samples and the same permutations, and the relative resolution is the ratio of each replicate's resolution to its own at 500 directions, with the standard error of the ratio. The seconds are the wall time of one call. The last two columns count the replicates with a NaN value and the NaN draws of the null, over all replicates.

| D | Case | Method | Directions | Resolution | Relative to the default | Seconds per call | NaN values | NaN null draws |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | shift | sliced W2 | 20 | 66.8 ± 4.4 | 0.9925 ± 0.0087 | 0.006514 ± 0.000054 | 0 of 4 | 0 of 40 |
| 2 | shift | sliced W2 | 100 | 67.9 ± 5.1 | 1.0062 ± 0.0070 | 0.01025 ± 0.00021 | 0 of 4 | 0 of 40 |
| 2 | shift | sliced W2 | 500 | 67.5 ± 5.0 | 1 | 0.04099 ± 0.00064 | 0 of 4 | 0 of 40 |
| 2 | shift | sliced W2 | 2 000 | 67.9 ± 5.2 | 1.0052 ± 0.0035 | 0.15974 ± 0.00079 | 0 of 4 | 0 of 40 |
| 2 | shift | sliced Sinkhorn | 20 | 33 ± 12 | - | 0.946 ± 0.018 | 0 of 3 | 0 of 15 |
| 2 | width | sliced W2 | 20 | 13.1 ± 1.3 | 1.015 ± 0.011 | 0.006635 ± 0.000098 | 0 of 4 | 0 of 40 |
| 2 | width | sliced W2 | 100 | 13.1 ± 1.5 | 1.014 ± 0.028 | 0.010492 ± 0.000033 | 0 of 4 | 0 of 40 |
| 2 | width | sliced W2 | 500 | 12.9 ± 1.4 | 1 | 0.04035 ± 0.00029 | 0 of 4 | 0 of 40 |
| 2 | width | sliced W2 | 2 000 | 13.0 ± 1.4 | 1.0059 ± 0.0036 | 0.16560 ± 0.00061 | 0 of 4 | 0 of 40 |
| 2 | width | sliced Sinkhorn | 20 | 11.3 ± 4.3 | - | 0.970 ± 0.087 | 0 of 3 | 0 of 15 |
| 8 | shift | sliced W2 | 20 | 37.4 ± 4.0 | 0.937 ± 0.064 | 0.007578 ± 0.000070 | 0 of 4 | 0 of 40 |
| 8 | shift | sliced W2 | 100 | 41.7 ± 5.6 | 1.040 ± 0.078 | 0.01100 ± 0.00025 | 0 of 4 | 0 of 40 |
| 8 | shift | sliced W2 | 500 | 39.8 ± 3.1 | 1 | 0.04232 ± 0.00027 | 0 of 4 | 0 of 40 |
| 8 | shift | sliced W2 | 2 000 | 40.5 ± 3.7 | 1.017 ± 0.035 | 0.16948 ± 0.00021 | 0 of 4 | 0 of 40 |
| 8 | shift | sliced Sinkhorn | 24 | 11.4 ± 1.4 | - | 1.25 ± 0.10 | 0 of 3 | 0 of 15 |
| 8 | width | sliced W2 | 20 | 3.40 ± 0.68 | 0.869 ± 0.082 | 0.00757 ± 0.00013 | 0 of 4 | 0 of 40 |
| 8 | width | sliced W2 | 100 | 3.98 ± 0.61 | 1.028 ± 0.014 | 0.0121 ± 0.0010 | 0 of 4 | 0 of 40 |
| 8 | width | sliced W2 | 500 | 3.87 ± 0.56 | 1 | 0.04269 ± 0.00056 | 0 of 4 | 0 of 40 |
| 8 | width | sliced W2 | 2 000 | 3.94 ± 0.73 | 1.004 ± 0.038 | 0.1738 ± 0.0021 | 0 of 4 | 0 of 40 |
| 8 | width | sliced Sinkhorn | 24 | 2.06 ± 0.87 | - | 1.24 ± 0.23 | 1 of 3 | 1 of 15 |
| 16 | shift | sliced W2 | 20 | 24.0 ± 4.1 | 0.873 ± 0.071 | 0.009446 ± 0.000084 | 0 of 4 | 0 of 40 |
| 16 | shift | sliced W2 | 100 | 29.8 ± 8.1 | 1.027 ± 0.057 | 0.01162 ± 0.00021 | 0 of 4 | 0 of 40 |
| 16 | shift | sliced W2 | 500 | 28.1 ± 5.9 | 1 | 0.04482 ± 0.00025 | 0 of 4 | 0 of 40 |
| 16 | shift | sliced W2 | 2 000 | 27.7 ± 5.2 | 0.994 ± 0.025 | 0.17021 ± 0.00034 | 0 of 4 | 0 of 40 |
| 16 | shift | sliced Sinkhorn | 32 | 12.2 ± 4.5 | - | 1.433 ± 0.011 | 0 of 3 | 0 of 15 |
| 16 | width | sliced W2 | 20 | 0.96 ± 0.21 | 1.10 ± 0.36 | 0.00955 ± 0.00012 | 0 of 4 | 0 of 40 |
| 16 | width | sliced W2 | 100 | 0.69 ± 0.11 | 0.75 ± 0.19 | 0.01199 ± 0.00029 | 0 of 4 | 0 of 40 |
| 16 | width | sliced W2 | 500 | 1.09 ± 0.23 | 1 | 0.04518 ± 0.00066 | 0 of 4 | 0 of 40 |
| 16 | width | sliced W2 | 2 000 | 1.32 ± 0.24 | 1.243 ± 0.086 | 0.1754 ± 0.0029 | 0 of 4 | 0 of 40 |
| 16 | width | sliced Sinkhorn | 32 | 0.54 ± 0.33 | - | 1.3613 ± 0.0025 | 0 of 3 | 1 of 15 |
:::

:::{table} Claims of the check, each with its verdict.

| Claim | Expected | Observed | Verdict |
| --- | --- | --- | --- |
| Complete orthonormal bases leave no scatter for a shift. | relative scatter at most 1e-09 for every n and count | worst of 28: n = 32, 20 directions: 2.7e-16 | consistent |
| Independent directions scatter as the Beta law predicts. | within 20% of the root of 2(n - 1)/((n + 2)L) | worst of 28: n = 16, 500 directions: +9.8% against ±20% | consistent |
:::

:::{table} Warnings that the check raised; the messages are in the JSON.

| Category | Count |
| --- | --- |
| SubsamplingWarning | 39 |
| ValidityWarning | 2 |
:::

(validation-gaussian-sliced-w2)=
### The Gaussian estimator of sliced W2

Reruns the check of session 12b.

Supports:

- {py:func}`mixed_signals.metrics.ot.sliced.gaussian_sliced_wasserstein_distance`
- {py:func}`mixed_signals.metrics.ot.sliced.sliced_wasserstein_of_gaussians`

10 replicates of 1 000 samples per set and 30 permutation draws. Each pair is whitened with its pooled covariance, the two estimators share 500 directions, and both are null-subtracted and mapped by the sliced W2 map with n coordinates. The zoo banana ({py:func}`~mixed_signals.zoo.cases.banana`) and the mixture ({py:func}`~mixed_signals.zoo.cases.gaussian_mixture`) share their mean and covariance with their Gaussians, so the Gaussian estimator, a lower bound of the sliced W2 for these moments ({cite:t}`gelbrich1990formula`), reads zero, while the empirical estimator sees the gap. The second table fits the Gaussian form to identical Gaussians without subtraction. 100 replicates give its bias, which the fitted moments raise by about 3/N.

:::{table} Mean σ_eff of the empirical and the Gaussian estimator of the sliced W2 row, null-subtracted, with the standard error of the mean.

| Case | Replicates | Empirical σ_eff | Gaussian σ_eff |
| --- | --- | --- | --- |
| banana against its Gaussian | 10 | 0.601 ± 0.011 | -0.013 ± 0.026 |
| mixture at separation 4 against its Gaussian | 10 | 0.1693 ± 0.0055 | 0.041 ± 0.020 |
| identical Gaussians | 10 | 0.006 ± 0.019 | 0.000 ± 0.020 |
:::

:::{table} Gaussian estimator before subtraction on identical 8-dimensional Gaussians: the mean value, the value 3/N and their ratio.

| N per set | Mean value | 3/N | Ratio |
| --- | --- | --- | --- |
| 1 000 | 0.00311 ± 0.00011 | 0.003 | 1.037 |
| 5 000 | 0.000615 ± 0.000020 | 0.0006 | 1.026 |
| 20 000 | 0.0001389 ± 0.0000048 | 0.00015 | 0.9259 |
:::

:::{table} Claims of the check, each with its verdict.

| Claim | Expected | Observed | Verdict |
| --- | --- | --- | --- |
| Identical Gaussians: the empirical estimator is at zero. | within 4.1 standard errors (Student t, 10 replicates) | worst of 1: mean σ_eff: +0.00571 against ±0.077 | consistent |
| Identical Gaussians: the Gaussian estimator is at zero. | within 4.1 standard errors (Student t, 10 replicates) | worst of 1: mean σ_eff: +0.000244 against ±0.08 | consistent |
| Banana: the Gaussian estimator is consistent with zero. | within 4.1 standard errors (Student t, 10 replicates) | worst of 1: mean σ_eff: -0.0129 against ±0.11 | consistent |
| Banana: the empirical estimator is not consistent with zero. | farther than 4.1 standard errors (Student t, 10 replicates) from 0 | 0.601 ± 0.011 | consistent |
| The Gaussian form before subtraction has a bias of 3/N. | within 5% of 3/N plus 3.1 standard errors (Student t, 100 replicates) | worst of 3: N = 20 000: -1.11e-05 against ±2.2e-05 | consistent |
:::

The check raised no warning.

(validation-chordal-widths)=
### Chordal metrics on wide circles and spheres

Reruns the check of session 12b.

Supports:

- {py:data}`mixed_signals.sigma.maps.CHORDAL_CIRCLE_WIDTH`
- {py:data}`mixed_signals.sigma.maps.CHORDAL_SPHERE_WIDTH`
- {py:func}`mixed_signals.sigma.maps.warn_wide_chordal_axes`

Zoo wrapped Gaussians ({py:func}`~mixed_signals.zoo.cases.wrapped_gaussians`) of width 0.1, 0.3, 0.5, 1 rad and von Mises-Fisher distributions ({py:func}`~mixed_signals.zoo.cases.von_mises_fisher`) of width 0.05, 0.2, 0.5 rad, shifted by 0.5, 1, 2 widths, with 1 000 samples per set and 6 replicates. The pooled whitening of {py:func}`~mixed_signals.transforms.whiten.Whitening.from_samples` averages the widths of the two posteriors, so it does not depend on the shift; the script fixes it once per width from a pilot pair of 10 000 samples, and the five chordal metrics run on the whitened samples with the default permutation null of 6 draws. The sliced Sinkhorn divergence, the costliest, runs at a shift of 1 with 2 replicates, 3 null draws and a cap of 500 samples per set; its numbers are in the tables and carry no claim, because two replicates cannot show a deviation of 5%. The limits are {py:data}`~mixed_signals.sigma.maps.CHORDAL_CIRCLE_WIDTH` and {py:data}`~mixed_signals.sigma.maps.CHORDAL_SPHERE_WIDTH`. The sets of a replicate hold a shift of their own, which differs from the true shift by sampling noise and which every metric follows, so a claim tests the mean paired deviation over the shift, σ_eff minus the sample shift of the same replicate: it is within 5% of the true shift, plus the standard errors of the paired differences, at and below the limit. A width at the limit falls on either side of it by sampling, so the warning claim asks the estimated width to match the limit within its standard errors there.

:::{table} σ_eff over the true shift, by geometry, width in radians, metric and shift in widths, with the standard error over replicates. The sliced Sinkhorn divergence ran at a shift of 1 only. A dash marks a value that was not computed or whose iterations did not converge.

| Geometry | Width (rad) | Metric | Shift 0.5 | Shift 1 | Shift 2 |
| --- | --- | --- | --- | --- | --- |
| circle | 0.1 | Energy distance | 1.014 ± 0.028 | 0.983 ± 0.016 | 0.9927 ± 0.0073 |
| circle | 0.1 | MMD squared | 1.016 ± 0.027 | 0.983 ± 0.016 | 0.9913 ± 0.0081 |
| circle | 0.1 | Sinkhorn divergence | 1.016 ± 0.030 | 0.987 ± 0.016 | 0.9983 ± 0.0066 |
| circle | 0.1 | Sliced Sinkhorn | - | 1.006 ± 0.015 | - |
| circle | 0.1 | Sliced W2 | 1.016 ± 0.030 | 0.986 ± 0.016 | 0.9981 ± 0.0066 |
| circle | 0.3 | Energy distance | 0.937 ± 0.042 | 0.987 ± 0.017 | 0.9674 ± 0.0061 |
| circle | 0.3 | MMD squared | 0.954 ± 0.044 | 1.013 ± 0.020 | 0.9907 ± 0.0080 |
| circle | 0.3 | Sinkhorn divergence | 0.956 ± 0.043 | 1.002 ± 0.016 | 0.9837 ± 0.0048 |
| circle | 0.3 | Sliced Sinkhorn | - | 0.9635 ± 0.0100 | - |
| circle | 0.3 | Sliced W2 | 0.946 ± 0.043 | 0.992 ± 0.016 | 0.9732 ± 0.0047 |
| circle | 0.5 | Energy distance | 0.960 ± 0.032 | 0.940 ± 0.013 | 0.9044 ± 0.0046 |
| circle | 0.5 | MMD squared | 1.008 ± 0.038 | 0.997 ± 0.012 | 0.9680 ± 0.0057 |
| circle | 0.5 | Sinkhorn divergence | 1.031 ± 0.034 | 0.990 ± 0.016 | 0.9302 ± 0.0052 |
| circle | 0.5 | Sliced Sinkhorn | - | 0.911 ± 0.042 | - |
| circle | 0.5 | Sliced W2 | 0.980 ± 0.030 | 0.946 ± 0.014 | 0.8917 ± 0.0051 |
| circle | 1 | Energy distance | 0.748 ± 0.042 | 0.7294 ± 0.0075 | 0.6588 ± 0.0058 |
| circle | 1 | MMD squared | 0.919 ± 0.053 | 0.8844 ± 0.0092 | 0.7751 ± 0.0066 |
| circle | 1 | Sinkhorn divergence | 0.806 ± 0.036 | 0.7465 ± 0.0078 | 0.5752 ± 0.0040 |
| circle | 1 | Sliced Sinkhorn | - | 0.633 ± 0.011 | - |
| circle | 1 | Sliced W2 | 0.694 ± 0.035 | 0.6739 ± 0.0072 | 0.5660 ± 0.0044 |
| sphere | 0.05 | Energy distance | 0.925 ± 0.042 | 1.011 ± 0.017 | 0.999 ± 0.014 |
| sphere | 0.05 | MMD squared | 0.911 ± 0.040 | 1.007 ± 0.018 | 0.998 ± 0.015 |
| sphere | 0.05 | Sinkhorn divergence | 0.944 ± 0.046 | 1.019 ± 0.017 | 1.001 ± 0.012 |
| sphere | 0.05 | Sliced Sinkhorn | - | 1.0501 ± 0.0055 | - |
| sphere | 0.05 | Sliced W2 | 0.943 ± 0.045 | 1.020 ± 0.016 | 1.001 ± 0.012 |
| sphere | 0.2 | Energy distance | 0.926 ± 0.031 | 0.940 ± 0.018 | 0.9711 ± 0.0066 |
| sphere | 0.2 | MMD squared | 0.938 ± 0.033 | 0.949 ± 0.020 | 0.9841 ± 0.0075 |
| sphere | 0.2 | Sinkhorn divergence | 0.921 ± 0.028 | 0.942 ± 0.015 | 0.9666 ± 0.0063 |
| sphere | 0.2 | Sliced Sinkhorn | - | 0.9132 ± 0.0066 | - |
| sphere | 0.2 | Sliced W2 | 0.919 ± 0.028 | 0.936 ± 0.015 | 0.9609 ± 0.0062 |
| sphere | 0.5 | Energy distance | 0.775 ± 0.034 | 0.791 ± 0.023 | 0.788 ± 0.012 |
| sphere | 0.5 | MMD squared | 0.819 ± 0.036 | 0.837 ± 0.025 | 0.832 ± 0.012 |
| sphere | 0.5 | Sinkhorn divergence | 0.790 ± 0.029 | 0.795 ± 0.022 | 0.783 ± 0.011 |
| sphere | 0.5 | Sliced Sinkhorn | - | 0.729 ± 0.087 | - |
| sphere | 0.5 | Sliced W2 | 0.748 ± 0.029 | 0.758 ± 0.021 | 0.754 ± 0.011 |
:::

:::{table} The shift that the sample sets hold, over the true shift: on a circle the difference of the circular means over the width of the whitening, on a sphere the difference of the angles of the mean directions in the plane of the true shift over the sphere width of the whitening.

| Geometry | Width (rad) | Shift 0.5 | Shift 1 | Shift 2 |
| --- | --- | --- | --- | --- |
| circle | 0.1 | 1.025 ± 0.029 | 0.989 ± 0.016 | 1.0013 ± 0.0067 |
| circle | 0.3 | 0.965 ± 0.043 | 1.012 ± 0.016 | 1.0048 ± 0.0051 |
| circle | 0.5 | 1.041 ± 0.032 | 1.014 ± 0.016 | 0.9988 ± 0.0068 |
| circle | 1 | 0.991 ± 0.049 | 1.020 ± 0.011 | 1.004 ± 0.010 |
| sphere | 0.05 | 0.950 ± 0.045 | 1.024 ± 0.017 | 1.004 ± 0.012 |
| sphere | 0.2 | 0.959 ± 0.029 | 0.970 ± 0.016 | 1.0001 ± 0.0067 |
| sphere | 0.5 | 0.963 ± 0.042 | 0.983 ± 0.025 | 1.013 ± 0.016 |
:::

:::{table} Paired deviation over the true shift: σ_eff minus the sample shift of the same replicate, over the true shift, with the standard error of the paired differences.

| Geometry | Width (rad) | Metric | Shift 0.5 | Shift 1 | Shift 2 |
| --- | --- | --- | --- | --- | --- |
| circle | 0.1 | Energy distance | -0.0105 ± 0.0044 | -0.00623 ± 0.00076 | -0.0086 ± 0.0012 |
| circle | 0.1 | MMD squared | -0.0085 ± 0.0088 | -0.0064 ± 0.0018 | -0.0100 ± 0.0023 |
| circle | 0.1 | Sinkhorn divergence | -0.0084 ± 0.0017 | -0.00294 ± 0.00035 | -0.00303 ± 0.00012 |
| circle | 0.1 | Sliced Sinkhorn | - | 0.009 ± 0.024 | - |
| circle | 0.1 | Sliced W2 | -0.0088 ± 0.0015 | -0.00303 ± 0.00036 | -0.00320 ± 0.00010 |
| circle | 0.3 | Energy distance | -0.0285 ± 0.0058 | -0.0242 ± 0.0025 | -0.0374 ± 0.0028 |
| circle | 0.3 | MMD squared | -0.011 ± 0.013 | 0.0015 ± 0.0052 | -0.0141 ± 0.0051 |
| circle | 0.3 | Sinkhorn divergence | -0.0095 ± 0.0016 | -0.00964 ± 0.00095 | -0.02112 ± 0.00048 |
| circle | 0.3 | Sliced Sinkhorn | - | -0.0802 ± 0.0092 | - |
| circle | 0.3 | Sliced W2 | -0.0193 ± 0.0015 | -0.01970 ± 0.00092 | -0.03159 ± 0.00090 |
| circle | 0.5 | Energy distance | -0.081 ± 0.012 | -0.0739 ± 0.0046 | -0.0944 ± 0.0046 |
| circle | 0.5 | MMD squared | -0.033 ± 0.023 | -0.0165 ± 0.0068 | -0.0308 ± 0.0068 |
| circle | 0.5 | Sinkhorn divergence | -0.0098 ± 0.0059 | -0.0236 ± 0.0010 | -0.0686 ± 0.0019 |
| circle | 0.5 | Sliced Sinkhorn | - | -0.084 ± 0.040 | - |
| circle | 0.5 | Sliced W2 | -0.0609 ± 0.0031 | -0.0683 ± 0.0021 | -0.1071 ± 0.0023 |
| circle | 1 | Energy distance | -0.243 ± 0.012 | -0.2910 ± 0.0056 | -0.3449 ± 0.0052 |
| circle | 1 | MMD squared | -0.072 ± 0.014 | -0.1360 ± 0.0065 | -0.2286 ± 0.0051 |
| circle | 1 | Sinkhorn divergence | -0.185 ± 0.019 | -0.2738 ± 0.0067 | -0.4284 ± 0.0065 |
| circle | 1 | Sliced Sinkhorn | - | -0.3670 ± 0.0019 | - |
| circle | 1 | Sliced W2 | -0.297 ± 0.017 | -0.3464 ± 0.0052 | -0.4376 ± 0.0060 |
| sphere | 0.05 | Energy distance | -0.0254 ± 0.0047 | -0.0124 ± 0.0028 | -0.0050 ± 0.0037 |
| sphere | 0.05 | MMD squared | -0.0394 ± 0.0071 | -0.0165 ± 0.0042 | -0.0060 ± 0.0060 |
| sphere | 0.05 | Sinkhorn divergence | -0.0061 ± 0.0031 | -0.0042 ± 0.0011 | -0.00299 ± 0.00022 |
| sphere | 0.05 | Sliced Sinkhorn | - | 0.043 ± 0.025 | - |
| sphere | 0.05 | Sliced W2 | -0.0064 ± 0.0013 | -0.00372 ± 0.00081 | -0.00308 ± 0.00017 |
| sphere | 0.2 | Energy distance | -0.0334 ± 0.0056 | -0.0309 ± 0.0031 | -0.0290 ± 0.0030 |
| sphere | 0.2 | MMD squared | -0.0213 ± 0.0093 | -0.0217 ± 0.0050 | -0.0160 ± 0.0045 |
| sphere | 0.2 | Sinkhorn divergence | -0.0380 ± 0.0032 | -0.0282 ± 0.0011 | -0.03349 ± 0.00056 |
| sphere | 0.2 | Sliced Sinkhorn | - | -0.038 ± 0.023 | - |
| sphere | 0.2 | Sliced W2 | -0.0406 ± 0.0022 | -0.03419 ± 0.00093 | -0.03917 ± 0.00065 |
| sphere | 0.5 | Energy distance | -0.1877 ± 0.0094 | -0.1924 ± 0.0061 | -0.2249 ± 0.0059 |
| sphere | 0.5 | MMD squared | -0.1442 ± 0.0081 | -0.1463 ± 0.0075 | -0.1808 ± 0.0058 |
| sphere | 0.5 | Sinkhorn divergence | -0.173 ± 0.014 | -0.1881 ± 0.0028 | -0.2301 ± 0.0060 |
| sphere | 0.5 | Sliced Sinkhorn | - | -0.2179 ± 0.0082 | - |
| sphere | 0.5 | Sliced W2 | -0.215 ± 0.014 | -0.2251 ± 0.0045 | -0.2586 ± 0.0057 |
:::

```{image} validation/figures/chordal-widths-ratio-light.svg
:class: only-light
:alt: Four panels of the paired deviation against the width, within the band up to the limit and falling below it above the limit.
```

```{image} validation/figures/chordal-widths-ratio-dark.svg
:class: only-dark
:alt: Four panels of the paired deviation against the width, within the band up to the limit and falling below it above the limit.
```

**Figure.** Paired deviation of σ_eff from the sample shift of its own replicate, over the true shift, against the width, at a shift of one width, for circles and spheres. The band is the 5% tolerance, the horizontal line zero and the vertical line the limit above which the package warns.

:::{table} Width estimated by the pooled whitening, in radians, and the comparisons for which `warn_wide_chordal_axes` fired.

| Geometry | Width (rad) | Limit (rad) | Estimated width | Warned |
| --- | --- | --- | --- | --- |
| circle | 0.1 | 0.3 | 0.10031 ± 0.00024 | 0 of 18 |
| circle | 0.3 | 0.3 | 0.3001 ± 0.0011 | 11 of 18 |
| circle | 0.5 | 0.3 | 0.4995 ± 0.0023 | 18 of 18 |
| circle | 1 | 0.3 | 1.0014 ± 0.0044 | 18 of 18 |
| sphere | 0.05 | 0.2 | 0.05017 ± 0.00018 | 0 of 18 |
| sphere | 0.2 | 0.2 | 0.20025 ± 0.00052 | 9 of 18 |
| sphere | 0.5 | 0.2 | 0.5014 ± 0.0011 | 18 of 18 |
:::

:::{table} Claims of the check, each with its verdict.

| Claim | Expected | Observed | Verdict |
| --- | --- | --- | --- |
| Energy distance on circles at and below the limit reads the sample shift. | within 5% of the truth plus 5.5 standard errors (Student t, 6 replicates) | worst of 6: circle 0.3 rad, shift 2: -0.0374 against ±0.065 | consistent |
| MMD squared on circles at and below the limit reads the sample shift. | within 5% of the truth plus 5.5 standard errors (Student t, 6 replicates) | worst of 6: circle 0.1 rad, shift 2: -0.01 against ±0.063 | consistent |
| Sliced W2 on circles at and below the limit reads the sample shift. | within 5% of the truth plus 5.5 standard errors (Student t, 6 replicates) | worst of 6: circle 0.3 rad, shift 2: -0.0316 against ±0.055 | consistent |
| Sinkhorn divergence on circles at and below the limit reads the sample shift. | within 5% of the truth plus 5.5 standard errors (Student t, 6 replicates) | worst of 6: circle 0.3 rad, shift 2: -0.0211 against ±0.053 | consistent |
| Energy distance on spheres at and below the limit reads the sample shift. | within 5% of the truth plus 5.5 standard errors (Student t, 6 replicates) | worst of 6: sphere 0.2 rad, shift 1: -0.0309 against ±0.067 | consistent |
| MMD squared on spheres at and below the limit reads the sample shift. | within 5% of the truth plus 5.5 standard errors (Student t, 6 replicates) | worst of 6: sphere 0.05 rad, shift 0.5: -0.0394 against ±0.089 | consistent |
| Sliced W2 on spheres at and below the limit reads the sample shift. | within 5% of the truth plus 5.5 standard errors (Student t, 6 replicates) | worst of 6: sphere 0.2 rad, shift 2: -0.0392 against ±0.054 | consistent |
| Sinkhorn divergence on spheres at and below the limit reads the sample shift. | within 5% of the truth plus 5.5 standard errors (Student t, 6 replicates) | worst of 6: sphere 0.2 rad, shift 2: -0.0335 against ±0.053 | consistent |
| A circle of width 0.1 rad below the limit stays silent. | the warning never fires | 0 of 18 | consistent |
| A circle of width 0.3 rad at the limit: the width is measured at the limit. | estimated width within 3.5 standard errors (Student t, 18 replicates) of 0.3; the warning fires on either side of the limit by sampling | width 0.3001 ± 0.0011; warned 11 of 18 | consistent |
| A circle of width 0.5 rad above the limit warns. | the warning fires in every comparison | 18 of 18 | consistent |
| A circle of width 1 rad above the limit warns. | the warning fires in every comparison | 18 of 18 | consistent |
| A sphere of width 0.05 rad below the limit stays silent. | the warning never fires | 0 of 18 | consistent |
| A sphere of width 0.2 rad at the limit: the width is measured at the limit. | estimated width within 3.5 standard errors (Student t, 18 replicates) of 0.2; the warning fires on either side of the limit by sampling | width 0.20025 ± 0.00052; warned 9 of 18 | consistent |
| A sphere of width 0.5 rad above the limit warns. | the warning fires in every comparison | 18 of 18 | consistent |
:::

:::{table} Warnings that the check raised; the messages are in the JSON.

| Category | Count |
| --- | --- |
| SubsamplingWarning | 21 |
:::

(validation-sphere-width)=
### Width of a sphere

Reruns the check of session 10.

Supports:

- {py:func}`mixed_signals.core.samples.spherical_width`

Zoo von Mises-Fisher distributions ({py:func}`~mixed_signals.zoo.cases.von_mises_fisher`) with κ from 1 to 400, 4 000 samples and 24 replicates each. The width is {py:func}`~mixed_signals.core.samples.spherical_width`, √(1 - R̄) for the mean resultant length R̄. A concentrated distribution has the width 1/√κ ({cite:t}`fisher1953dispersion`), and in general R̄ = coth κ - 1/κ, which {py:func}`~mixed_signals.zoo.closed_forms.von_mises_fisher_resultant_length` gives exactly.

:::{table} Sphere width √(1 - R) of von Mises-Fisher samples against the width 1/√κ and the exact width from the mean resultant length.

| κ | Measured width | 1/√κ | Exact width | Measured over 1/√κ |
| --- | --- | --- | --- | --- |
| 1 | 0.8297 ± 0.0011 | 1 | 0.8288 | 0.8297 |
| 2 | 0.67949 ± 0.00078 | 0.7071 | 0.6802 | 0.9609 |
| 5 | 0.44829 ± 0.00070 | 0.4472 | 0.4471 | 1.002 |
| 10 | 0.31571 ± 0.00049 | 0.3162 | 0.3162 | 0.9984 |
| 20 | 0.22407 ± 0.00030 | 0.2236 | 0.2236 | 1.002 |
| 50 | 0.14149 ± 0.00022 | 0.1414 | 0.1414 | 1.001 |
| 100 | 0.09982 ± 0.00015 | 0.1 | 0.1 | 0.9982 |
| 200 | 0.070513 ± 0.000096 | 0.07071 | 0.07071 | 0.9972 |
| 400 | 0.050046 ± 0.000080 | 0.05 | 0.05 | 1.001 |
:::

:::{table} Claims of the check, each with its verdict.

| Claim | Expected | Observed | Verdict |
| --- | --- | --- | --- |
| The measured width equals the exact one. | within 1% of the exact width plus 3.4 standard errors (Student t, 24 replicates) | worst of 9: κ = 400: +4.61e-05 against ±0.00077 | consistent |
| On wide spheres the width falls below 1/√κ. | below 1/√κ by more than 3.4 standard errors (Student t, 24 replicates) at κ = 1 and 2 | 2 of 2 | consistent |
:::

The check raised no warning.

(validation-chords-and-arcs)=
### Chords against arc lengths

Reruns the check of session 9.

Supports:

- {py:func}`mixed_signals.metrics.kernel.maximum_mean_discrepancy`
- {py:func}`mixed_signals.metrics.kernel.energy_distance`

500 samples per set and 6 replicates, with 20 permutation draws. The shifts are one width of a wrapped normal of 0.3 rad and of a von Mises-Fisher distribution of 0.1 rad. The bimodal circle compares an equal mixture of two wrapped normals of 0.3 rad with modes at 0 and π against modes at ±π/2, which differ in even harmonics only. Along chords the package computes the distances in the embedding; along arcs the script computes the shorter arc on the circle and the great-circle angle on the sphere, a rejected alternative that the package does not offer. Both use the median of the pooled pairwise distances as the MMD bandwidth. The arc of two points on a circle is not negative type in the strict sense ({cite:t}`lyons2020strong`), and a Gaussian kernel of the geodesic distance is not positive definite ({cite:t}`feragen2015geodesic`).

:::{table} Resolution, in null widths, of the energy distance and the MMD squared along chords (the package) and along arc lengths (the script).

| Case | Metric | Chords | Arcs |
| --- | --- | --- | --- |
| wrapped normals | Energy distance | 163 ± 37 | 167 ± 38 |
| wrapped normals | MMD squared | 128 ± 23 | 128 ± 22 |
| von Mises-Fisher | Energy distance | 120 ± 14 | 120 ± 14 |
| von Mises-Fisher | MMD squared | 110 ± 13 | 110 ± 13 |
| bimodal circle | Energy distance | 165 ± 21 | -0.06 ± 0.29 |
| bimodal circle | MMD squared | 107 ± 13 | -28.4 ± 3.3 |
:::

Along arcs the MMD squared of the bimodal circle falls far below its null, at -28.4 ± 3.3 null widths. The upper-tail test reads that as no difference. The sign comes from the Gaussian kernel of arc lengths, which is not positive definite.

:::{table} Claims of the check, each with its verdict.

| Claim | Expected | Observed | Verdict |
| --- | --- | --- | --- |
| The energy distance along chords resolves the bimodal circle. | resolution above 3 null widths | 165 ± 21 | consistent |
| The energy distance along arcs does not resolve the bimodal circle. | resolution below 3 null widths | -0.06 ± 0.29 | consistent |
| The MMD squared along chords resolves the bimodal circle. | resolution above 3 null widths | 107 ± 13 | consistent |
| The MMD squared along arcs does not resolve the bimodal circle. | resolution below 3 null widths | -28.4 ± 3.3 | consistent |
:::

The check raised no warning.

## Divergences

(validation-knn-ranks)=
### Ranks of the nearest-neighbour divergences

Reruns the check of session 9.

Supports:

- {py:data}`mixed_signals.metrics.divergence.KL_NEIGHBOURS`
- {py:data}`mixed_signals.metrics.divergence.JS_NEIGHBOURS`
- {py:data}`mixed_signals.metrics.divergence.HELLINGER_NEIGHBOURS`

Unit Gaussians in 1, 4, 16 parameters, the second set shifted along one axis ({py:func}`~mixed_signals.zoo.cases.shifted_gaussians`), with 1 000 samples per set and 16 replicates per dimension and shift. The estimates are {py:func}`~mixed_signals.metrics.divergence.knn_kl_divergence`, {py:func}`~mixed_signals.metrics.divergence.knn_js_divergence` and {py:func}`~mixed_signals.metrics.divergence.knn_hellinger_distance` at the ranks 1, 3, 5, 10, against the permutation null with 14 draws, fewer than the default 200 and the 40 of the original check; every rank of a replicate uses the same sets and the same permutations. The resolution at the 1σ shift is the mean of (value minus null mean) over the root of the mean null variance of the replicates, because every replicate draws the same null again; it is read relative to k = 3. The Hellinger resolution is read in H² = sign(H)·H², the space in which the package subtracts the null of that metric. In H, where the null draws lie below zero, the slope of the signed root narrows the null spread against the signal, and the resolution reads high. At D = 16, the null draws of the Hellinger estimate H at k = 5 have the mean -0.166, and 224 of 224 of them lie below zero. The bias rows subtract the null mean as the σ_eff maps do, in squares for the Hellinger distance, and compare with the truth at the shift that the two sample sets hold, the difference of their means along the shifted axis, which has the same expectation as the true shift and a far smaller scatter. The default ranks are KL 3, JS 10, Hellinger 5. The record, at 3 000 and 10 000 samples, gave a resolution against k = 3 of 0.54 and 1.62 at k = 1 and 10 for KL, 0.58 and 1.87 for JS, and 0.17, 1.41 and 2.08 at k = 1, 5 and 10 for Hellinger; at the defaults JS lay within −1 to +3% of the truth up to D = 8, H² within 0 to −9%, and KL read −9 to −17% low at D = 4. The nearest-neighbour searches cost O(N²), and the JS estimate costs most, so this check runs at a smaller N and fewer replicates than the record. No estimate or null draw was NaN or infinite.

:::{table} Resolution of a 1σ shift, (value minus null mean) over the null standard deviation, relative to k = 3: the ratio of the mean signal over the pooled null width of 16 replicates, which share their samples and permutations across ranks, with the leave-one-out error over the replicates. The third column is the resolution at k = 3, in null widths. The Hellinger rows read the signal and the null in H² = sign(H)·H², where the package subtracts the null of that metric.

| Metric | D | Resolution at k = 3 | k = 1 | k = 3 | k = 5 | k = 10 |
| --- | --- | --- | --- | --- | --- | --- |
| KL | 1 | 15.49 ± 0.95 | 0.527 ± 0.037 | 1 | 1.254 ± 0.084 | 1.67 ± 0.11 |
| KL | 4 | 12.64 ± 0.42 | 0.581 ± 0.035 | 1 | 1.183 ± 0.047 | 1.390 ± 0.079 |
| KL | 16 | 5.01 ± 0.36 | 0.831 ± 0.056 | 1 | 1.007 ± 0.023 | 1.002 ± 0.036 |
| JS | 1 | 8.62 ± 0.52 | 0.566 ± 0.039 | 1 | 1.341 ± 0.090 | 1.87 ± 0.17 |
| JS | 4 | 9.10 ± 0.45 | 0.501 ± 0.039 | 1 | 1.274 ± 0.072 | 1.83 ± 0.13 |
| JS | 16 | 8.71 ± 0.36 | 0.541 ± 0.035 | 1 | 1.384 ± 0.054 | 2.14 ± 0.15 |
| Hellinger | 1 | 12.08 ± 0.65 | 0.225 ± 0.045 | 1 | 1.49 ± 0.11 | 2.22 ± 0.20 |
| Hellinger | 4 | 10.95 ± 0.47 | 0.261 ± 0.038 | 1 | 1.270 ± 0.062 | 1.78 ± 0.13 |
| Hellinger | 16 | 8.66 ± 0.45 | 0.336 ± 0.041 | 1 | 1.359 ± 0.075 | 2.06 ± 0.17 |
:::

```{image} validation/figures/knn-ranks-resolution-light.svg
:class: only-light
:alt: Three panels of the resolution relative to k = 3 against the rank, rising with the rank and falling below one at k = 1.
```

```{image} validation/figures/knn-ranks-resolution-dark.svg
:class: only-dark
:alt: Three panels of the resolution relative to k = 3 against the rank, rising with the rank and falling below one at k = 1.
```

**Figure.** Resolution of a 1σ shift over its value at k = 3 against the rank k, one panel per metric and the dimension D as the series; the line is one and the bars are the errors of the ratio.

:::{table} Null-subtracted estimate at the default rank over the truth at the sample's own shift, minus one, with the standard error of the paired deviations: KL against δ²/2, JS against the closed form and H² against 1 − exp(−δ²/8).

| Metric | Rank | D | Shift 1 | Shift 2 |
| --- | --- | --- | --- | --- |
| KL | 3 | 1 | -0.012 ± 0.026 | -0.085 ± 0.011 |
| KL | 3 | 4 | -0.155 ± 0.017 | -0.2271 ± 0.0086 |
| KL | 3 | 16 | -0.360 ± 0.034 | -0.3717 ± 0.0087 |
| JS | 10 | 1 | -0.005 ± 0.019 | 0.0100 ± 0.0070 |
| JS | 10 | 4 | 0.004 ± 0.012 | 0.0059 ± 0.0051 |
| JS | 10 | 16 | 0.093 ± 0.010 | 0.1329 ± 0.0075 |
| Hellinger (H²) | 5 | 1 | 0.011 ± 0.019 | -0.0163 ± 0.0067 |
| Hellinger (H²) | 5 | 4 | -0.031 ± 0.020 | -0.0561 ± 0.0047 |
| Hellinger (H²) | 5 | 16 | -0.078 ± 0.016 | -0.0800 ± 0.0060 |
:::

:::{table} Claims of the check, each with its verdict.

| Claim | Expected | Observed | Verdict |
| --- | --- | --- | --- |
| The JS at its default rank k = 10 recovers the truth at the sample's own shift up to D = 4. | within 10% of the truth plus 3.6 standard errors (Student t, 16 replicates) | worst of 4: D = 4, shift 2: +0.00585 against ±0.12 | consistent |
| The H² at its default rank k = 5 recovers the truth at the sample's own shift up to D = 4. | within 10% of the truth plus 3.6 standard errors (Student t, 16 replicates) | worst of 4: D = 4, shift 2: -0.0561 against ±0.12 | consistent |
| The JS at its default rank k = 10 resolves a 1σ shift better than k = 3. | above the resolution at k = 3 by more than 3.6 standard errors (Student t, 16 replicates) | worst of 3: D = 1: +0.867 against more than 0.6 | consistent |
| The Hellinger at its default rank k = 5 resolves a 1σ shift better than k = 3. | above the resolution at k = 3 by more than 3.6 standard errors (Student t, 16 replicates) | worst of 3: D = 4: +0.27 against more than 0.22 | consistent |
:::

The check raised no warning.

## Estimators

(validation-gmm-starts)=
### Starts of the Gaussian mixture

Reruns the check of session 6.

Supports:

- {py:data}`mixed_signals.estimators.gmm.DEFAULT_STARTS`

Two-component fits of {py:class}`~mixed_signals.estimators.gmm.MixtureDensity` with {py:class}`~mixed_signals.estimators.gmm.MixtureSettings` (components 2, starts 1) to the posterior `mixture` of {py:func}`~mixed_signals.zoo.cases.gaussian_mixture` at the separations 2, 3, 4, 5, 6, with 4 000 samples and 40 replicates. Each replicate runs 100 single starts, each with its own key, once at the default tolerance and once at the tolerance of the contrast, and one fit of the default 18 starts. The optimum ℓ* of a replicate is the largest total log-likelihood of any of its fits, and ℓ₁ is that of one Gaussian. The record did not state its definition of the plateau, so this page defines it: a start ends on the plateau when its total log-likelihood lies below (ℓ₁ + ℓ*)/2, nearer the one-Gaussian solution than the optimum. The starts of one replicate share their samples, so the binomial error of the share counts them as independent and the scatter column shows the spread between replicates. Session 6 recorded 37 to 68% of single starts on the plateau against at most 19% at the tolerance of the contrast, and a median gap of 2.6 to 2.8 nats at 4 000 samples in the harder cases. The claims read the separation with the largest share on the plateau; the table gives the shares at every separation. The contrast keeps the default cap of 100 iterations, so its last column counts the starts that reached the tolerance before the cap.

:::{table} Fits of 2 components to 4 000 samples of the zoo mixture, by separation: the single starts of 40 replicates with 100 starts each. The share is the plateau starts over all starts, with its binomial error, and the scatter column is the standard deviation of the share between replicates. The gaps ℓ* − ℓ, in nats on the total log-likelihood, belong to the fits of 18 starts, one per replicate. The last columns are the contrast with a tolerance of 0.01 nats.

| Separation | Plateau starts | Starts | Share on the plateau | Scatter between replicates | Share to the power 18 | Fits of 18 starts on the plateau | Median gap | Largest gap | Share on the plateau, contrast | Share that met the tolerance, contrast | Non-finite fits |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 2 590 | 4 000 | 0.6475 ± 0.0076 | 0.06368 | 0.000400 ± 0.000084 | 0 of 40 | 3.482 | 16.29 | 0.2248 ± 0.0066 | 0.4763 ± 0.0079 | 0 |
| 3 | 1 207 | 4 000 | 0.3018 ± 0.0073 | 0.04534 | 4.3e-10 ± 1.9e-10 | 0 of 40 | 0.9956 | 13.76 | 0.1110 ± 0.0050 | 0.8702 ± 0.0053 | 0 |
| 4 | 950 | 4 000 | 0.2375 ± 0.0067 | 0.0455 | 5.8e-12 ± 2.9e-12 | 0 of 40 | 0.1765 | 1.876 | 0.1045 ± 0.0048 | 0.9585 ± 0.0032 | 0 |
| 5 | 887 | 4 000 | 0.2218 ± 0.0066 | 0.03419 | 1.68e-12 ± 9.0e-13 | 0 of 40 | 0.008369 | 0.05555 | 0.0990 ± 0.0047 | 0.9655 ± 0.0029 | 0 |
| 6 | 854 | 4 000 | 0.2135 ± 0.0065 | 0.02878 | 8.5e-13 ± 4.6e-13 | 0 of 40 | 4.069e-05 | 0.0004497 | 0.1095 ± 0.0049 | 0.9772 ± 0.0024 | 0 |
:::

:::{table} Claims of the check, each with its verdict.

| Claim | Expected | Observed | Verdict |
| --- | --- | --- | --- |
| The largest share of single starts on the plateau, at separation 2, does not exceed the record. | at most 0.6972 (the record's 0.675 plus three binomial errors at that level) | 0.6475 ± 0.0076 | consistent |
| The median gap of the fits of 18 starts reproduces the record, at separation 2. | between 1.3 and 5.6 nats (the record's 2.6 to 2.8 nats within a factor of 2) | 3.48 nats | consistent |
:::

The check raised no warning.

(validation-kde-railing)=
### Railing score of the corrected KDE

Reruns the checks of sessions 6 and 11.

Supports:

- {py:func}`mixed_signals.diagnostics.railing.railing_scores`
- {py:class}`mixed_signals.estimators.kde.KernelDensity`

Normals truncated at the bound zero, with their peak z standard deviations inside it: {py:func}`~mixed_signals.zoo.cases.railed_gaussians` with offsets 0, 1, 2, 2.5, 3, one parameter, 1 000, 4 000, 10 000 samples and 400 replicates per size. The corrected score is {py:func}`~mixed_signals.diagnostics.railing.railing_scores`, the density of the corrected KDE at the bound over its peak, and the truth is the reference row `railing_lower`, exp(−z²/2). The reflected score is computed in the script: the same estimate with the bandwidth of the corrected one and mirror images of every sample at the bound, whose density at the bound is read over the largest value on a grid of 2 048 points that reaches 3 bandwidths beyond the data. Session 6 recorded a reflected score high by +11% at N = 10 000 for z = 1, +31% at N = 10 000 for z = 2, +48% at N = 1 000 for z = 2, where the correction left 2%, 4% and −2%. A claim tests the mean score over the truth against one, with the tolerance and the standard errors. No score was NaN or infinite.

:::{table} Mean railing score over the truth exp(−z²/2) of 400 replicates, with the standard error of the mean, by estimator, offset z and sample count.

| Estimator | z | N = 1 000 | N = 4 000 | N = 10 000 |
| --- | --- | --- | --- | --- |
| corrected | 0 | 0.9661 ± 0.0027 | 0.9829 ± 0.0015 | 0.9886 ± 0.0010 |
| corrected | 1 | 1.0240 ± 0.0090 | 1.0184 ± 0.0050 | 1.0147 ± 0.0035 |
| corrected | 2 | 1.057 ± 0.016 | 1.0218 ± 0.0095 | 1.0119 ± 0.0068 |
| corrected | 2.5 | 1.132 ± 0.031 | 1.009 ± 0.016 | 1.024 ± 0.011 |
| corrected | 3 | 1.288 ± 0.060 | 1.097 ± 0.033 | 1.048 ± 0.022 |
| reflected | 0 | 0.9897 ± 0.0011 | 0.99502 ± 0.00056 | 0.99685 ± 0.00036 |
| reflected | 1 | 1.1658 ± 0.0059 | 1.1310 ± 0.0032 | 1.1097 ± 0.0024 |
| reflected | 2 | 1.520 ± 0.012 | 1.3673 ± 0.0072 | 1.2982 ± 0.0048 |
| reflected | 2.5 | 1.786 ± 0.025 | 1.515 ± 0.013 | 1.4184 ± 0.0081 |
| reflected | 3 | 2.134 ± 0.048 | 1.716 ± 0.026 | 1.555 ± 0.016 |
:::

:::{table} Scatter per replicate of the score at z = 2, with its standard error, and the scatter that session 6 recorded for the corrected estimate.

| N | Corrected | Reflected | Record, corrected |
| --- | --- | --- | --- |
| 1 000 | 0.0431 ± 0.0015 | 0.0330 ± 0.0012 | 0.04 |
| 4 000 | 0.02558 ± 0.00091 | 0.01953 ± 0.00069 | 0.022 |
| 10 000 | 0.01853 ± 0.00066 | 0.01304 ± 0.00046 | 0.015 |
:::

```{image} validation/figures/kde-railing-ratio-light.svg
:class: only-light
:alt: Three panels of the score over the truth against the offset: the corrected estimate stays near one, the reflected one lies above it and rises with the offset.
```

```{image} validation/figures/kde-railing-ratio-dark.svg
:class: only-dark
:alt: Three panels of the score over the truth against the offset: the corrected estimate stays near one, the reflected one lies above it and rises with the offset.
```

**Figure.** Mean railing score over its truth against the offset z of the peak from the bound, by sample count, for the corrected and the reflected estimate. The line is one and the bars are standard errors of the mean.

:::{table} Claims of the check, each with its verdict.

| Claim | Expected | Observed | Verdict |
| --- | --- | --- | --- |
| The corrected score recovers exp(−z²/2): z up to 2 at N = 1 000, z up to 2.5 at N = 4 000, z up to 2.5 at N = 10 000. | within 5% of the truth plus three standard errors | worst of 11: z = 0, N = 1 000: -0.0339 against ±0.058 | consistent |
| The reflected score reads high at z = 2, at every N. | above the truth by more than 10% of the truth plus three standard errors | worst of 3: N = 10 000: +0.298 against more than 0.11 | consistent |
:::

The check raised no warning.

## Diagnostics

(validation-surrogate-calibration)=
### Calibration of the Gaussian surrogate null

Reruns the check of session 11.

Supports:

- {py:func}`mixed_signals.nulls.surrogates.gaussian_surrogates`
- {py:func}`mixed_signals.diagnostics.dependence.dependence_statistics`

30 replicates of 2 000 samples from a zoo Gaussian ({py:func}`~mixed_signals.zoo.cases.gaussians`) with two parameters of correlation 0.6, so that the raw dependence statistics do not vanish. For each replicate, {py:func}`~mixed_signals.nulls.surrogates.gaussian_surrogates` draws 100 surrogate sets of the replicate's effective sample size, which is 2 000 for independent draws and at most the cap of 2 000 of the pairwise statistics. Every statistic of the replicate and of each surrogate set comes from the package: the marginal moments, the Anderson-Darling distance, Mardia's moments and the dependence rows. The dependence rows are returned as excesses over their Gaussian values at the set's own r, so a raw value adds that Gaussian value back. The spread of z over the replicates is a calibration: a calibrated null gives one. A spread from 30 replicates has a standard error of about 13%, which the table states for every row; the claims read each spread against its band as the record states it, without a margin for that error. No z was NaN or infinite.

:::{table} Calibration of the Gaussian surrogate null: the mean and the spread of z = (value minus surrogate mean) over the surrogate standard deviation, over 30 replicates of 2 000 samples with 100 surrogates each, with the standard errors, and the spread that session 11 recorded for the group.

| Statistic | Form | Mean z | Spread of z | Record, spread |
| --- | --- | --- | --- | --- |
| skewness x0 | shape | 0.08 ± 0.20 | 1.07 ± 0.14 | 0.8 to 1.2 |
| skewness x1 | shape | 0.06 ± 0.17 | 0.94 ± 0.12 | 0.8 to 1.2 |
| kurtosis x0 | shape | -0.01 ± 0.15 | 0.82 ± 0.11 | 0.8 to 1.2 |
| kurtosis x1 | shape | -0.03 ± 0.21 | 1.16 ± 0.15 | 0.8 to 1.2 |
| Anderson-Darling x0 | shape | -0.16 ± 0.14 | 0.751 ± 0.099 | 0.8 to 1.2 |
| Anderson-Darling x1 | shape | -0.33 ± 0.14 | 0.78 ± 0.10 | 0.8 to 1.2 |
| Mardia b₁ | shape | -0.18 ± 0.18 | 1.00 ± 0.13 | 0.8 to 1.2 |
| Mardia b₂ | shape | -0.01 ± 0.16 | 0.89 ± 0.12 | 0.8 to 1.2 |
| Pearson r x0-x1 | pearson | -0.010 ± 0.020 | 0.108 ± 0.014 | 0.12 |
| Spearman, raw x0-x1 | raw | 0.075 ± 0.070 | 0.382 ± 0.050 | 0.4 to 0.5 |
| distance correlation, raw x0-x1 | raw | 0.067 ± 0.063 | 0.347 ± 0.046 | 0.4 to 0.5 |
| mutual information, raw x0-x1 | raw | 0.09 ± 0.11 | 0.586 ± 0.077 | 0.4 to 0.5 |
| Spearman, excess x0-x1 | excess | 0.21 ± 0.16 | 0.89 ± 0.12 | 0.96 to 1.33 |
| distance correlation, excess x0-x1 | excess | 0.22 ± 0.16 | 0.88 ± 0.12 | 0.96 to 1.33 |
| mutual information, excess x0-x1 | excess | 0.22 ± 0.19 | 1.05 ± 0.14 | 0.96 to 1.33 |
:::

:::{table} Claims of the check, each with its verdict.

| Claim | Expected | Observed | Verdict |
| --- | --- | --- | --- |
| The spread of z of every shape statistic is that of a calibrated null. | between 0.7 and 1.4 | worst of 8: Anderson-Darling x0: 0.751 ± 0.099 | consistent |
| The spread of z of every excess dependence statistic is that of a calibrated null. | between 0.7 and 1.4 | worst of 3: distance correlation, excess x0-x1: 0.88 ± 0.12 | consistent |
| The raw Spearman, distance correlation and mutual information sit at the centre of their surrogate null. | at most 0.6 | worst of 3: mutual information, raw x0-x1: 0.586 ± 0.077 | consistent |
| The raw Pearson correlation sits at the centre of its surrogate null. | at most 0.3 | worst of 1: Pearson r x0-x1: 0.108 ± 0.014 | consistent |
:::

The check raised no warning.

(validation-circle-drift)=
### Drift of linear statistics on wide circles

Reruns the check of session 11.

Supports:

- {py:data}`mixed_signals.diagnostics._common.CIRCLE_WIDTH_LIMIT`

Wrapped normals with a real parameter ({py:func}`~mixed_signals.zoo.cases.wrapped_gaussians`, two parameters, correlation 0.5), circular standard deviations 0.25, 0.5, 0.75, 1, 1.25, 1.5 rad, 2 000 samples, 100 replicates per width and 100 Gaussian surrogates per replicate. The linear statistics are Pearson's r, the excess of Spearman's ρ_s, and the skewness and kurtosis, all of values unwrapped about the circular mean; the circular statistics are the circular skewness and kurtosis of the package. The observed set of each replicate runs outside jit, so that the package raises its warning above the limit of {py:data}`~mixed_signals.diagnostics._common.CIRCLE_WIDTH_LIMIT` rad; the table counts them. Session 11 recorded linear statistics that kept their calibration up to 1 rad and drifted from 1.25 rad, and circular moments that stayed calibrated at every width. The linear statistics that can flag are those with a threshold in {py:data}`~mixed_signals.diagnostics.flags.THRESHOLDS`: Spearman excess, skewness, unwrapped, kurtosis, unwrapped. Pearson's r has no threshold and the surrogates reproduce it by construction, so only its mean of z enters the claims, and only up to 0.75 rad: at 1 rad its mean of z is 0.875 ± 0.017, which biases the σ_sig of an informational row, and of no row that can flag. The developers kept the limit at 1 rad for that reason. The claims allow the standard errors of the means and of the spreads.

:::{table} Mean of z of each statistic by circular standard deviation, with the standard error of the mean.

| Statistic | 0.25 rad | 0.5 rad | 0.75 rad | 1 rad | 1.25 rad | 1.5 rad |
| --- | --- | --- | --- | --- | --- | --- |
| Pearson r | -0.0065 ± 0.0091 | 0.019 ± 0.011 | 0.0331 ± 0.0092 | 0.875 ± 0.017 | 2.728 ± 0.034 | 4.004 ± 0.046 |
| Spearman excess | -0.103 ± 0.100 | -0.104 ± 0.097 | 0.002 ± 0.096 | 0.10 ± 0.10 | 0.90 ± 0.13 | 2.72 ± 0.12 |
| skewness, unwrapped | -0.09 ± 0.10 | -0.016 ± 0.091 | -0.10 ± 0.10 | 0.022 ± 0.087 | -0.02 ± 0.10 | -0.004 ± 0.088 |
| kurtosis, unwrapped | -0.060 ± 0.092 | 0.12 ± 0.11 | -0.063 ± 0.100 | -0.22 ± 0.10 | -0.367 ± 0.082 | -1.977 ± 0.073 |
| circular skewness | 0.09 ± 0.10 | 0.031 ± 0.092 | -0.02 ± 0.11 | 0.038 ± 0.081 | 0.12 ± 0.10 | -0.117 ± 0.080 |
| circular kurtosis | -0.056 ± 0.092 | 0.11 ± 0.11 | -0.120 ± 0.097 | -0.161 ± 0.099 | 0.067 ± 0.091 | -0.145 ± 0.085 |
:::

:::{table} Spread of z of each statistic by circular standard deviation, with its standard error.

| Statistic | 0.25 rad | 0.5 rad | 0.75 rad | 1 rad | 1.25 rad | 1.5 rad |
| --- | --- | --- | --- | --- | --- | --- |
| Pearson r | 0.0912 ± 0.0065 | 0.1109 ± 0.0079 | 0.0925 ± 0.0066 | 0.174 ± 0.012 | 0.338 ± 0.024 | 0.459 ± 0.033 |
| Spearman excess | 0.998 ± 0.071 | 0.968 ± 0.069 | 0.964 ± 0.069 | 1.028 ± 0.073 | 1.271 ± 0.090 | 1.211 ± 0.086 |
| skewness, unwrapped | 1.035 ± 0.074 | 0.911 ± 0.065 | 1.030 ± 0.073 | 0.873 ± 0.062 | 1.029 ± 0.073 | 0.881 ± 0.063 |
| kurtosis, unwrapped | 0.919 ± 0.065 | 1.133 ± 0.081 | 0.995 ± 0.071 | 1.035 ± 0.074 | 0.817 ± 0.058 | 0.733 ± 0.052 |
| circular skewness | 1.041 ± 0.074 | 0.919 ± 0.065 | 1.069 ± 0.076 | 0.812 ± 0.058 | 1.019 ± 0.072 | 0.795 ± 0.056 |
| circular kurtosis | 0.924 ± 0.066 | 1.071 ± 0.076 | 0.965 ± 0.069 | 0.993 ± 0.071 | 0.909 ± 0.065 | 0.851 ± 0.060 |
:::

:::{table} Package warnings raised by the statistics of the observed sets, by circular standard deviation.

| Circular standard deviation | Warnings |
| --- | --- |
| 0.25 rad | 0 |
| 0.5 rad | 0 |
| 0.75 rad | 0 |
| 1 rad | 51 |
| 1.25 rad | 100 |
| 1.5 rad | 100 |
:::

:::{table} Claims of the check, each with its verdict.

| Claim | Expected | Observed | Verdict |
| --- | --- | --- | --- |
| At and below 1 rad, the mean of z of every linear statistic that can flag stays near zero. | within ±0.5 plus 3.1 standard errors (Student t, 100 replicates) | worst of 12: kurtosis, unwrapped at 1 rad: -0.216 against ±0.82 | consistent |
| Up to 0.75 rad, the mean of z of Pearson's r stays near zero. | within ±0.5 plus 3.1 standard errors (Student t, 100 replicates) | worst of 3: Pearson r at 0.75 rad: +0.0331 against ±0.53 | consistent |
| At and below 1 rad, the spread of z of every linear statistic that can flag lies in 0.7 to 1.4. | within the band 0.7 to 1.4 plus 3.1 standard errors (Student t, 100 replicates) | worst of 12: skewness, unwrapped at 1 rad: -0.177 against ±0.54 | consistent |
| At 1.5 rad, at least one linear statistic leaves the band: a mean of z beyond ±0.5 or a spread outside 0.7 to 1.4, by more than the standard errors. | at least one of 4 statistics outside the band by more than 3.1 standard errors (Student t, 100 replicates) | 3 of 4: Pearson r, Spearman excess, kurtosis, unwrapped | consistent |
| The mean of z of the circular moments stays near zero at every width. | within ±0.5 plus 3.1 standard errors (Student t, 100 replicates) | worst of 12: circular kurtosis at 1.5 rad: -0.145 against ±0.76 | consistent |
| The spread of z of the circular moments lies in 0.7 to 1.4 at every width. | within the band 0.7 to 1.4 plus 3.1 standard errors (Student t, 100 replicates) | worst of 12: circular skewness at 1.5 rad: -0.255 against ±0.52 | consistent |
:::

:::{table} Warnings that the check raised; the messages are in the JSON.

| Category | Count |
| --- | --- |
| ValidityWarning | 251 |
:::

(validation-ksg-bananas)=
### KSG mutual information on bananas

Reruns the check of session 11.

Supports:

- {py:data}`mixed_signals.diagnostics.dependence.MUTUAL_INFORMATION_NEIGHBOURS`
- {py:data}`mixed_signals.diagnostics.dependence.PAIR_CAP`

The zoo's twisted Gaussian ({py:func}`~mixed_signals.zoo.cases.banana`, variance 100) at the twists 0.001, 0.002, 0.003, 0.01, 0.03, 0.1. The zoo does not tabulate its mutual information, so the script computes it. With t the twist times the variance, the second parameter is a unit normal plus t(z² − 1) for z standard normal, so the mutual information is the entropy of that sum minus ½ ln(2πe). Its density comes from composite 8-point Gauss-Legendre panels in z, with edges uniform in z² up to z = 7 and a shift of at most 0.2 along y per panel, and its entropy from the trapezoid rule with a step of 0.05; halving both changes every value by at most 9.6e-12 nats. The estimate is the KSG mutual information of {py:func}`~mixed_signals.diagnostics.dependence.dependence_statistics`, which returns it as an excess over the Gaussian value at the set's Pearson r; the table adds that value back. Sample counts 2 000, 1 000, 30 replicates per cell, ranks 10 (the default, which {py:data}`~mixed_signals.diagnostics.dependence.MUTUAL_INFORMATION_NEIGHBOURS` sets) and 3. The claims cover the bananas of twist at least 0.003, the faint one included, whose value of 0.0799 nats is the record's banana of 0.08 nats. Session 11 recorded k = 10 within −9 to +4% at 2 000 samples and −26% at 1 000 samples for the faint banana, and a resolution 1.7 times that of k = 3.

:::{table} KSG estimate over the true mutual information, mean of 30 replicates with the standard error, by twist, sample count and rank k. The true value is the quadrature of the script.

| Twist b | True mutual information (nats) | N | k = 10 | k = 3 |
| --- | --- | --- | --- | --- |
| 0.001 | 0.009896 | 2 000 | 0.73 ± 0.13 | 0.63 ± 0.29 |
| 0.001 | 0.009896 | 1 000 | 0.95 ± 0.20 | 0.67 ± 0.41 |
| 0.002 | 0.03814 | 2 000 | 0.792 ± 0.037 | 0.893 ± 0.094 |
| 0.002 | 0.03814 | 1 000 | 0.606 ± 0.046 | 0.55 ± 0.12 |
| 0.003 | 0.07991 | 2 000 | 0.846 ± 0.019 | 1.023 ± 0.043 |
| 0.003 | 0.07991 | 1 000 | 0.727 ± 0.032 | 0.765 ± 0.053 |
| 0.01 | 0.4299 | 2 000 | 0.9895 ± 0.0087 | 1.0102 ± 0.0099 |
| 0.01 | 0.4299 | 1 000 | 0.953 ± 0.012 | 0.995 ± 0.013 |
| 0.03 | 1.068 | 2 000 | 0.9943 ± 0.0034 | 1.0047 ± 0.0038 |
| 0.03 | 1.068 | 1 000 | 0.9711 ± 0.0072 | 0.9999 ± 0.0082 |
| 0.1 | 1.994 | 2 000 | 0.9196 ± 0.0028 | 0.9755 ± 0.0035 |
| 0.1 | 1.994 | 1 000 | 0.8560 ± 0.0032 | 0.9540 ± 0.0037 |
:::

:::{table} Resolution of the banana of twist 0.003, whose mutual information is 0.0799 nats, at 2 000 samples: (value minus surrogate mean) over the surrogate width, pooled over 16 replicates with 30 surrogates each. The two ranks share their samples and surrogates.

| Rank k | Resolution (null widths) |
| --- | --- |
| 10 | 8.07 ± 0.41 |
| 3 | 4.99 ± 0.28 |
| Ratio of k = 10 to k = 3 | 1.62 ± 0.10 |
:::

:::{table} Claims of the check, each with its verdict.

| Claim | Expected | Observed | Verdict |
| --- | --- | --- | --- |
| The KSG estimate at k = 10 recovers the mutual information of every banana of twist at least 0.003 at 2 000 samples. | within 10% of the truth plus 3.3 standard errors (Student t, 30 replicates) | worst of 4: b = 0.003: -0.154 against ±0.16 | consistent |
| The KSG estimate at k = 10 resolves the faint banana better than at k = 3. | above the resolution at k = 3 by more than 3.6 standard errors (Student t, 16 replicates) | worst of 1: N = 2 000: +0.617 against more than 0.37 | consistent |
:::

The check raised no warning.

(validation-bic-neff)=
### BIC with the effective sample size

Reruns the check of session 11.

Supports:

- {py:func}`mixed_signals.diagnostics.modes.gmm_bic`

Independent draws of the zoo's Gaussian ({py:func}`~mixed_signals.zoo.cases.shifted_gaussians` without a shift, 2 parameters), banana and mixture at separation 4, with 1 000 samples and 20 replicates each, and the same draws repeated 5 times, which hold 5 000 samples and the effective sample size of the independent draws, 1 000. {py:func}`~mixed_signals.diagnostics.modes.gmm_bic` runs with n set to the repeated count N and to n_eff; the two share their fits, so only the n of BIC differs. Components range from 1 to 5. The record found that n = N inflated ΔBIC tenfold and chose too many components, while n_eff gave the result of the independent draws. The second claim does not apply to a case whose independent draws already choose 5 components.

:::{table} ΔBIC = BIC(1) minus the smallest BIC and the number of components that BIC chooses, over 20 replicates of each case and setting. The ratio divides the mean ΔBIC by that of the independent draws, and the last column counts the replicates whose chosen number equals that of the independent draws.

| Case | Setting | ΔBIC | ΔBIC over that of the independent draws | Components chosen (count: replicates) | Same count as independent draws |
| --- | --- | --- | --- | --- | --- |
| Gaussian | independent draws, n = n_eff | 0 | - | 1: 20 | 20 of 20 |
| Gaussian | repeated 5 times, n = N | 2.8 ± 2.7 | - | 1: 18, 3: 1, 4: 1 | 18 of 20 |
| Gaussian | repeated 5 times, n = n_eff | 0 | - | 1: 20 | 20 of 20 |
| Banana | independent draws, n = n_eff | 2 639 ± 20 | 1 | 5: 20 | 20 of 20 |
| Banana | repeated 5 times, n = N | 13 817 ± 100 | 5.236 | 5: 20 | 20 of 20 |
| Banana | repeated 5 times, n = n_eff | 2 638 ± 20 | 0.9999 | 5: 20 | 20 of 20 |
| Mixture, separation 4 | independent draws, n = n_eff | 317.8 ± 6.1 | 1 | 2: 20 | 20 of 20 |
| Mixture, separation 4 | repeated 5 times, n = N | 1 750 ± 31 | 5.508 | 2: 17, 3: 2, 4: 1 | 17 of 20 |
| Mixture, separation 4 | repeated 5 times, n = n_eff | 317.7 ± 6.1 | 0.9999 | 2: 20 | 20 of 20 |
:::

:::{table} Claims of the check, each with its verdict.

| Claim | Expected | Observed | Verdict |
| --- | --- | --- | --- |
| With n = n_eff, the chosen number of components equals that of the independent draws in every replicate. | every replicate of every case | 60 of 60 replicates | consistent |
| With n = N, the chosen number exceeds that of the independent draws in at least one replicate of every case where the independent draws choose fewer than 5. | at least one replicate per applicable case | Gaussian: 2 of 20; Banana: not applicable; Mixture, separation 4: 3 of 20 | consistent |
:::

The check raised no warning.

(validation-mode-counting)=
### Counting modes

Reruns the check of session 11.

Supports:

- {py:data}`mixed_signals.diagnostics.modes.MODE_FLOOR`
- {py:data}`mixed_signals.diagnostics.modes.MODE_SEPARATION`
- the choice of the modes of 1D kernel estimates over those of the fitted mixture

Cases: two bananas ({py:func}`~mixed_signals.zoo.cases.banana`), normals railed at the bound by 0σ and 1σ ({py:func}`~mixed_signals.zoo.cases.railed_gaussians`), a Student t distribution with 3 degrees of freedom in two independent parameters, drawn with `jax.random.t` because the zoo has none, mixtures at the separations 2, 2.5, 3 and 4 ({py:func}`~mixed_signals.zoo.cases.gaussian_mixture`) and one in 8 parameters at separation 3, with 2 000 samples and 12 replicates each. {py:func}`~mixed_signals.diagnostics.modes.marginal_modes` counts the modes along every parameter, and {py:func}`~mixed_signals.diagnostics.modes.projection_modes` along the discriminant directions of the mixture that {py:func}`~mixed_signals.diagnostics.modes.gmm_bic` chooses; the table gives the largest count of both. Two contrasts come from the script. The first counts the marginal modes with the package's own estimate, grid and valley rule but a peak floor of 1% instead of {py:data}`~mixed_signals.diagnostics.modes.MODE_FLOOR`; no private helper of the package takes the floor, so the counting is repeated for the contrast. The second finds the modes of the mixture that BIC chooses by the fixed-point iteration of Carreira-Perpiñán ({cite:t}`carreiraperpinan2000mode`) from every component mean, with 1 000 iterations, merging points closer than 0.01 overall standard deviations. The mixture at separation 2 is not claimed, because an equal mixture has two modes only above 2. The mixture at separation 2.5 stays in the tables without a claim: its true dip is 0.130, against 0.1 for {py:data}`~mixed_signals.diagnostics.modes.MODE_SEPARATION`, and the mean dip of its estimate is 0.065 ± 0.016 at 2 000 samples, because the bandwidth of the estimate fills the valley. Session 11 recorded two modes from separation 2.5; the developers moved the claim to separation 3. Session 11 recorded one mode for every banana and railed case and two from separation 2.5, with dips of 0.10 to 0.57 and including the 8-parameter mixture that BIC missed; a floor of 1% let single samples in heavy tails count as modes in 5 to 25% of marginals against none at 5%; and mixture modes numbered up to five for a unimodal banana.

:::{table} Components that BIC chooses and the modes of the 1D estimates, the largest count over the marginals and over the discriminant directions of the chosen mixture, over 12 replicates of 2 000 samples. The dip is the deepest valley; the true dip of an equal mixture is 1 − f(valley)/f(peak) of the zoo density, read on a grid of 100 001 points.

| Case | Parameters | BIC components | Replicates with one mode | with two modes | with more | Deepest dip | True dip |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Banana, b = 0.003 | 2 | 2.17 ± 0.11 | 12 | 0 | 0 | 0 | - |
| Banana, b = 0.03 | 2 | 5 | 12 | 0 | 0 | 0.00095 ± 0.00095 | - |
| Normal railed at 0σ | 2 | 4.00 ± 0.17 | 12 | 0 | 0 | 0.0034 ± 0.0033 | - |
| Normal railed at 1σ | 2 | 2.83 ± 0.17 | 12 | 0 | 0 | 0.0080 ± 0.0080 | - |
| Student t, 3 degrees of freedom | 2 | 4.00 ± 0.17 | 12 | 0 | 0 | 0 | - |
| Mixture, separation 2 | 2 | 1.33 ± 0.14 | 12 | 0 | 0 | 0.0058 ± 0.0040 | 0 |
| Mixture, separation 2.5 | 2 | 2 | 9 | 3 | 0 | 0.065 ± 0.016 | 0.1296 |
| Mixture, separation 3 | 2 | 2 | 0 | 12 | 0 | 0.236 ± 0.011 | 0.3582 |
| Mixture, separation 4 | 2 | 2 | 0 | 12 | 0 | 0.5675 ± 0.0063 | 0.7294 |
| Mixture, separation 3, 8 parameters | 8 | 1.083 ± 0.083 | 0 | 12 | 0 | 0.2352 ± 0.0096 | 0.3582 |
:::

:::{table} Contrasts. The replicates in which some marginal has a second mode with a peak floor of 1% and with the package's floor of 5%, and the modes of the mixture that BIC chooses, by the fixed-point iteration, with their largest count.

| Case | Second marginal mode, floor 1% | Second marginal mode, floor 5% | Modes of the BIC mixture | Largest count |
| --- | --- | --- | --- | --- |
| Banana, b = 0.003 | 1 of 12 | 0 of 12 | 1 | 1 |
| Banana, b = 0.03 | 2 of 12 | 0 of 12 | 4.83 ± 0.11 | 5 |
| Normal railed at 0σ | 0 of 12 | 0 of 12 | 2.33 ± 0.14 | 3 |
| Normal railed at 1σ | 0 of 12 | 0 of 12 | 1.67 ± 0.14 | 2 |
| Student t, 3 degrees of freedom | 2 of 12 | 0 of 12 | 1.25 ± 0.13 | 2 |
| Mixture, separation 2 | 0 of 12 | 0 of 12 | 1.25 ± 0.13 | 2 |
| Mixture, separation 2.5 | 3 of 12 | 3 of 12 | 2 | 2 |
| Mixture, separation 3 | 12 of 12 | 12 of 12 | 2 | 2 |
| Mixture, separation 4 | 12 of 12 | 12 of 12 | 2 | 2 |
| Mixture, separation 3, 8 parameters | 12 of 12 | 12 of 12 | 1.083 ± 0.083 | 2 |
:::

:::{table} Claims of the check, each with its verdict.

| Claim | Expected | Observed | Verdict |
| --- | --- | --- | --- |
| Every banana, railed and heavy-tailed replicate has one mode. | one mode in every replicate | 60 of 60 replicates | consistent |
| Every replicate of a mixture from separation 3, and of the 8-parameter mixture, has two modes. | two modes in every replicate | 36 of 36 replicates | consistent |
| With a peak floor of 1%, at least one unimodal replicate counts a second mode. | at least one replicate | 5 of 60 replicates | consistent |
| The modes of the mixture that BIC chooses exceed one for at least one banana replicate. | more than one mode in at least one replicate | largest count 5 in 24 replicates | consistent |
:::

The check raised no warning.

(validation-split-chain-kuiper)=
### Split-chain Kuiper test on stationary chains

Reruns the check of session 11.

Supports:

- {py:func}`mixed_signals.mcmc.ess.split_chain_kuiper`

300 runs of AR(1) chains with a coefficient of 0.5 ({py:func}`~mixed_signals.zoo.cases.ar1_chains`), 16 walkers of 1 000 steps, scaled by 2.5 and wrapped onto [−π, π), as the test of the package builds them. {py:func}`~mixed_signals.mcmc.ess.split_chain_kuiper` runs with the thinning factor 3, the autocorrelation time of the chains. The chains are stationary, so the p-values are uniform if the test is exact. Session 11 recorded 5.7 to 6.3% below 0.05 in 300 runs.

:::{table} Rejection rate of the split-chain Kuiper test on 300 stationary chains on a circle: the share of runs whose p-value is at or below the level, with the binomial error at the level, and the mean p-value.

| Runs | Rate at 1% | Rate at 5% | Rate at 10% | Mean p-value |
| --- | --- | --- | --- | --- |
| 300 | 0.0100 ± 0.0057 | 0.030 ± 0.013 | 0.073 ± 0.017 | 0.498 ± 0.016 |
:::

:::{table} Claims of the check, each with its verdict.

| Claim | Expected | Observed | Verdict |
| --- | --- | --- | --- |
| The split-chain Kuiper test rejects stationary circle chains at its levels. | rates within three binomial errors of 1%, 5%, 10% | 0.0100 ± 0.0057; 0.030 ± 0.013; 0.073 ± 0.017 | consistent |
:::

The check raised no warning.

(validation-diagnostic-flags)=
### Default flags on the zoo

Reruns the check of session 11.

Supports:

- {py:data}`mixed_signals.diagnostics.flags.THRESHOLDS`
- {py:func}`mixed_signals.diagnostics.flags.evaluate`
- the done criterion of session 11: the zoo's cases flag as recorded

Zoo cases with the diagnostics of session 11 assembled as the tests of the package assemble them: the marginal moments, the Anderson-Darling distance, Mardia's moments, the dependence rows and the mode dips, each against 100 Gaussian surrogates of the set's effective sample size, which is 2 000 for independent draws; GMM-BIC with its ΔBIC; the dips along the discriminant directions of the chosen mixture; and the railing scores, which have no surrogate null. The mode rows on intervals take surrogates truncated to the interval. {py:func}`~mixed_signals.diagnostics.flags.evaluate` flags each row against {py:data}`~mixed_signals.diagnostics.flags.THRESHOLDS`; 3 replicates per case. Session 11 recorded that the Gaussians, a narrow wrapped normal and a normal 4σ from its bound raised no flag, that bananas flagged their dependence, Mardia's skewness and GMM-BIC but no mode, that mixtures at separation 4 flagged their mode and kurtosis, and that normals railed at 0σ and 1σ flagged their bound and skewness. A claim names the rows of the record and holds only if they flag in every replicate; the table lists every row that flags, so a stray row is visible, and a stray row in a case that must raise no flag fails its claim.

:::{table} Rows that flag, by zoo case and replicate, with 100 Gaussian surrogates per set and 2 000 samples.

| Case | Parameters | Replicate | Rows that flag |
| --- | --- | --- | --- |
| Gaussian, independent, 2 parameters | 2 | 0 | none |
| Gaussian, independent, 2 parameters | 2 | 1 | none |
| Gaussian, independent, 2 parameters | 2 | 2 | none |
| Gaussian, correlated, 3 parameters | 3 | 0 | none |
| Gaussian, correlated, 3 parameters | 3 | 1 | none |
| Gaussian, correlated, 3 parameters | 3 | 2 | none |
| Wrapped normal, width 0.5 rad, 2 parameters | 2 | 0 | none |
| Wrapped normal, width 0.5 rad, 2 parameters | 2 | 1 | none |
| Wrapped normal, width 0.5 rad, 2 parameters | 2 | 2 | none |
| Normal 4σ from its bound, 2 parameters | 2 | 0 | none |
| Normal 4σ from its bound, 2 parameters | 2 | 1 | none |
| Normal 4σ from its bound, 2 parameters | 2 | 2 | none |
| Banana, 2 parameters | 2 | 0 | anderson_darling x1; distance_correlation x0-x1; gmm_bic all; kurtosis x1; mardia_kurtosis all; mardia_skewness all; mutual_information x0-x1; skewness x1 |
| Banana, 2 parameters | 2 | 1 | anderson_darling x1; distance_correlation x0-x1; gmm_bic all; kurtosis x1; mardia_kurtosis all; mardia_skewness all; mutual_information x0-x1; skewness x1 |
| Banana, 2 parameters | 2 | 2 | anderson_darling x1; distance_correlation x0-x1; gmm_bic all; kurtosis x1; mardia_kurtosis all; mardia_skewness all; mutual_information x0-x1; skewness x1 |
| Mixture, separation 4, 2 parameters | 2 | 0 | anderson_darling x0; gmm_bic all; kurtosis x0; mardia_kurtosis all; mode_dip direction 0; mode_dip x0 |
| Mixture, separation 4, 2 parameters | 2 | 1 | anderson_darling x0; gmm_bic all; kurtosis x0; mardia_kurtosis all; mode_dip direction 0; mode_dip x0 |
| Mixture, separation 4, 2 parameters | 2 | 2 | anderson_darling x0; gmm_bic all; kurtosis x0; mardia_kurtosis all; mode_dip direction 0; mode_dip x0 |
| Normal railed at 0σ, 2 parameters | 2 | 0 | anderson_darling x0; gmm_bic all; mardia_skewness all; railing_lower x0; skewness x0 |
| Normal railed at 0σ, 2 parameters | 2 | 1 | anderson_darling x0; gmm_bic all; mardia_skewness all; railing_lower x0; skewness x0 |
| Normal railed at 0σ, 2 parameters | 2 | 2 | anderson_darling x0; gmm_bic all; kurtosis x0; mardia_kurtosis all; mardia_skewness all; railing_lower x0; skewness x0 |
| Normal railed at 1σ, 2 parameters | 2 | 0 | anderson_darling x0; gmm_bic all; mardia_skewness all; railing_lower x0; skewness x0 |
| Normal railed at 1σ, 2 parameters | 2 | 1 | anderson_darling x0; gmm_bic all; mardia_skewness all; railing_lower x0; skewness x0 |
| Normal railed at 1σ, 2 parameters | 2 | 2 | anderson_darling x0; gmm_bic all; mardia_skewness all; railing_lower x0; skewness x0 |
:::

:::{table} Claims of the check, each with its verdict.

| Claim | Expected | Observed | Verdict |
| --- | --- | --- | --- |
| Gaussian, independent, 2 parameters: no row flags. | no row flags in any replicate | 3 of 3 replicates | consistent |
| Gaussian, correlated, 3 parameters: no row flags. | no row flags in any replicate | 3 of 3 replicates | consistent |
| Wrapped normal, width 0.5 rad, 2 parameters: no row flags. | no row flags in any replicate | 3 of 3 replicates | consistent |
| Normal 4σ from its bound, 2 parameters: no row flags. | no row flags in any replicate | 3 of 3 replicates | consistent |
| Banana, 2 parameters: the rows distance_correlation x0-x1, mutual_information x0-x1, mardia_skewness all, gmm_bic all flag. | every named row flags in every replicate, and no mode_dip row | 3 of 3 replicates | consistent |
| Mixture, separation 4, 2 parameters: the rows mode_dip x0, kurtosis x0 flag. | every named row flags in every replicate | 3 of 3 replicates | consistent |
| Normal railed at 0σ, 2 parameters: the rows railing_lower x0, skewness x0 flag. | every named row flags in every replicate | 3 of 3 replicates | consistent |
| Normal railed at 1σ, 2 parameters: the rows railing_lower x0, skewness x0 flag. | every named row flags in every replicate | 3 of 3 replicates | consistent |
:::

:::{table} Warnings that the check raised; the messages are in the JSON.

| Category | Count |
| --- | --- |
| ExtrapolationWarning | 79 |
:::

## Costs

(validation-costs)=
### Cost of the metrics and diagnostics

Reruns the checks of sessions 8, 9, 10 and 12.

Supports:

- {py:data}`mixed_signals._config.SAMPLE_CAP`
- {py:data}`mixed_signals.diagnostics.dependence.PAIR_CAP`
- {py:data}`mixed_signals.metrics.ot.sliced.SLICED_SINKHORN_CAP`
- the cost weights of the dry run in session 14

Zoo Gaussians in 16 parameters ({py:func}`~mixed_signals.zoo.cases.shifted_gaussians`, shift 1). Each row calls the function of the package once as an engine would, after one call that compiles it, and reports the median of three calls, or the one call when it takes more than 10 s. Before a timer starts, all earlier work has finished, and the timer stops when the result is ready, because JAX dispatches asynchronously. The energy distance runs at 5 000, 10 000, 20 000 samples per set, which sets {py:data}`~mixed_signals._config.SAMPLE_CAP`, and the other comparison rows at 10 000; the sliced Sinkhorn divergence runs at 2 000 ({py:data}`~mixed_signals.metrics.ot.sliced.SLICED_SINKHORN_CAP`), and the dependence statistics at 2 000 ({py:data}`~mixed_signals.diagnostics.dependence.PAIR_CAP`), with the default directions of each sliced metric. In 16 dimensions the orthonormal bases round the number of directions asked for up to whole bases, so the rows state how many were asked for and how many the array holds. The record of 6.4 s for the sliced Sinkhorn divergence was measured with 20 independent directions. The median bandwidth and the calibration curve of the kNN KL map run once per comparison, so they have no extrapolation. The times depend on the machine in the stamp of this page; the record column holds the times that the sessions measured on an Apple M4 Pro CPU, 20 minutes for the dependence statistics of 201 sets being about 6 s per set. No claim reads them.

:::{table} Wall time of one evaluation in 16 parameters, after one warm-up call that compiles, and the time of 201 calls, which is a comparison with 200 null draws and the observed value. A dash marks a row that a comparison evaluates once.

| Row | Samples per set | Seconds per call | Timing | 201 calls (minutes) | Record |
| --- | --- | --- | --- | --- | --- |
| energy distance | 5 000 | 0.06725 | median of 3 | 0.2253 | 0.06 s |
| energy distance | 10 000 | 0.2738 | median of 3 | 0.9174 | 0.48 s |
| energy distance | 20 000 | 1.26 | median of 3 | 4.222 | 2.0 s |
| median bandwidth | 10 000 | 0.3065 | median of 3 | - | 0.3 s |
| MMD squared | 10 000 | 0.3036 | median of 3 | 1.017 | 0.3 s |
| kNN KL | 10 000 | 2.499 | median of 3 | 8.37 | 2.7 s |
| kNN Hellinger | 10 000 | 5.002 | median of 3 | 16.76 | 5.3 s |
| kNN JS | 10 000 | 9.177 | median of 3 | 30.74 | 9.5 s |
| Sinkhorn divergence | 10 000 | 6.545 | median of 3 | 21.93 | 6.5 s |
| sliced W2, directions: 500 asked, 512 used | 10 000 | 0.5719 | median of 3 | 1.916 | 0.5 s |
| sliced Sinkhorn divergence, directions: 20 asked, 32 used | 2 000 | 10.26 | one call | 34.37 | 6.4 s |
| kNN KL calibration curve | 10 000 | 70.08 | one call | - | - |
| pairwise dependence statistics | 2 000 | 5.76 | median of 3 | 19.3 | about 20 min for 201 sets |
:::

The check raised no warning.
