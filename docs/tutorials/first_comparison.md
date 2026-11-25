---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# A first comparison

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

Two analyses of the same data, such as two samplers, two waveform models or two
pipelines, give two posteriors. Before you trust either of them, you want to know
two things: is there a difference at all, and if there is, is it large enough to
matter? mixed signals answers both questions for every statistic it computes. The
first answer is a significance, σ_sig: how surely the finite samples show a
difference. The second is an effect size, σ_eff: how large the difference is, in
units of the posterior width. This tutorial compares Gaussian posteriors whose
differences are known exactly, so you can check every number against the truth.

## Two posteriors with a known shift

The validation zoo of the package, `mixed_signals.zoo`, builds posteriors with known
answers. `shifted_gaussians` gives a reference Gaussian and a copy whose mean moves by
a chosen number of posterior widths. Here the shift is 0.3 widths, in three
parameters:

```{code-cell} ipython3
import jax
import matplotlib.pyplot as plt
import polars as pl
from IPython.display import display
from matplotlib.lines import Line2D

import mixed_signals as ms
from mixed_signals import zoo
from mixed_signals.prng import derive_key

case = zoo.shifted_gaussians(0.3, param_count=3)
sample_set = case.sample(jax.random.key(0), 2_000)
print(case.names)
print(sample_set.samples.shape)
```

The width is measured with the covariance of the posterior, as a Mahalanobis
distance {cite:p}`mahalanobis1936generalised`. A shift of 0.3 moves the mean by 0.3
standard deviations along the direction of the shift, whatever the correlations
between the parameters.

The samples come as a `SampleSet` shaped (B, K, N, D): B independent problems, K
posteriors of each problem, N samples per posterior and D parameters. Here there is
one problem with two posteriors of 2 000 samples in three parameters. A set also
carries a log weight per sample, zero here because the samples are equally weighted,
and an effective sample size, the number of independent samples that the set is
worth.

The case brings its parameter space, a `ParamSpace`: the names of the parameters,
their topology and their bounds. The space tells every statistic how to treat each
parameter. Here all three live on the real line; the next tutorial shows why the
topology matters for angles and for bounded parameters.

```{code-cell} ipython3
pl.DataFrame(
    {
        "name": [parameter.name for parameter in case.space.parameters],
        "topology": [parameter.topology.value for parameter in case.space.parameters],
        "lower_bound": [parameter.bounds[0] for parameter in case.space.parameters],
        "upper_bound": [parameter.bounds[1] for parameter in case.space.parameters],
    }
)
```

## Compare the two

`compare` evaluates a set of statistics on every pair of posteriors. Without a
reference it treats both posteriors alike, as two peers of which neither is the
truth; the package calls this mode B.

```{code-cell} ipython3
result = ms.compare(
    sample_set,
    case.names,
    case.space,
    key=1,
    metrics=["mean_shift", "ks", "kl:gaussian", "energy", "mmd"],
)
number_columns = ["value", "null_mean", "sigma_sig", "sigma_eff", "r_eff", "threshold"]
result.table.select(
    pl.col("pair").list.join(" to "),
    "metric",
    "estimator",
    pl.col("parameters").list.join(", "),
    pl.col(number_columns).round_sig_figs(3),
    "flag",
    # * The tail fit of an extrapolated significance is reported as a warning.
    extrapolated=pl.col("warnings")
    .list.eval(pl.element().struct.field("category") == "ExtrapolationWarning")
    .list.any(),
)
```

Each row is one statistic of the pair. Read it from left to right:

- `value` is the statistic of the two sample sets, for example the energy distance
  {cite:p}`szekely2013energy` or the squared maximum mean discrepancy (MMD)
  {cite:p}`gretton2012kernel`. Statistics of single parameters, such as the
  Kolmogorov-Smirnov distance {cite:p}`smirnov1948table`, give one row per parameter.
- `null_mean` is the mean of the same statistic when both sets come from one
  posterior. Finite samples never give zero: two sets of 2 000 samples from one
  distribution still differ by chance. The package builds this null by permutation
  {cite:p}`good2005permutation`. It pools the two sets, splits the pool at random
  into two new sets whose sizes follow the effective sample sizes, and evaluates the
  statistic again, 200 times by default.
- `sigma_sig` is the significance. The share of null draws at least as large as the
  observed value gives a p-value {cite:p}`phipson2010permutation`, and σ_sig is that
  p-value as a number of Gaussian standard deviations, Φ⁻¹(1 − p)
  {cite:p}`cowan2011asymptotic`. 200 draws resolve σ_sig up to 2.58, or up to 2.81
  for rows that count departures in both directions, such as a mean shift. Where fewer
  than ten draws reach the value, the package fits the tail of the null instead
  {cite:p}`knijnenburg2009fewer` and marks the row as extrapolated in its `warnings`
  column, shown here as `extrapolated`. Far in the tail the fit reads low, so a large
  σ_sig is a lower estimate.
- `sigma_eff` is the effect size: the shift between two Gaussians of unit width that
  gives the same value once the null mean is subtracted. Each metric has its own map
  from value to shift, from the closed form of the metric for two Gaussians, so rows
  of different metrics share one scale.
- `r_eff` replaces σ_eff in rows that measure only a difference in width, such as the
  width ratio: the ratio of the widths of two Gaussians that gives the same value.
- `threshold` is the effect size at which a row fails, 0.2 by default. The GWTC-2
  catalogue treats a Jensen-Shannon divergence of 0.007 bit between two posteriors
  as significant, the divergence of a shift of 0.2 standard deviations
  {cite:p}`abbott2021gwtc`, and 0.2 is a small effect in the classification of
  {cite:t}`cohen1988statistical`. Rows with an r_eff fail when the logarithm of r_eff
  reaches 0.14 in size, the width ratio whose Jensen-Shannon divergence equals that of
  a shift of 0.2.
- `flag` is true when the row fails: its effect size reaches the threshold and its
  σ_sig reaches 2, so the difference is both large enough and resolved.

The true shift is 0.3, along the first parameter, x0. Every row that sees it, the rows
of x0 and the joint rows, gives a σ_eff near 0.3 and flags, while the rows of x1 and x2
read near zero. The Kullback-Leibler divergence is not symmetric, so its rows come in
both directions, and it splits into a location part and a shape part. The shape part,
`kl_shape`, measures only a difference in covariance: it is a width row, and its r_eff
stays near 1, because both posteriors have the same width.

## Significance grows with the number of samples, the effect size does not

Why two numbers? Because a test alone answers the wrong question: with enough samples,
every difference becomes significant, however small. The next cell compares a shift
of 0.3 widths and a shift of 0.05 widths at three sample sizes. One pair of sample sets
is one random draw, and its σ_eff scatters by about √(2/N) around the truth, 0.09 at
N = 250, so the cell repeats every comparison five times with fresh samples and shows
the median and the range of the mean-shift row of x0.

```{code-cell} ipython3
root = ms.root_key(2026)
shifts = (0.05, 0.3)
counts = (250, 1_000, 4_000)
replicate_count = 5
rows = []
for shift in shifts:
    scan_case = zoo.shifted_gaussians(shift, param_count=3)
    for count in counts:
        for replicate in range(replicate_count):
            path = (str(shift), count, replicate)
            scan_set = scan_case.sample(derive_key(root, "draw", *path), count)
            scan_result = ms.compare(
                scan_set,
                scan_case.names,
                scan_case.space,
                key=derive_key(root, "compare", *path),
                metrics=["mean_shift"],
            )
            x0_row = scan_result.table.filter(pl.col("parameters").list.first() == "x0")
            rows.append(
                {
                    "shift": shift,
                    "sample_count": count,
                    "sigma_sig": x0_row["sigma_sig"][0],
                    "sigma_eff": x0_row["sigma_eff"][0],
                }
            )
growth = pl.DataFrame(rows)
summary = growth.group_by("shift", "sample_count", maintain_order=True).agg(
    pl.col("sigma_sig").median().alias("sigma_sig_median"),
    pl.col("sigma_sig").min().alias("sigma_sig_min"),
    pl.col("sigma_sig").max().alias("sigma_sig_max"),
    pl.col("sigma_eff").median().alias("sigma_eff_median"),
    pl.col("sigma_eff").min().alias("sigma_eff_min"),
    pl.col("sigma_eff").max().alias("sigma_eff_max"),
)
display(summary.with_columns(pl.exclude("shift", "sample_count").round_sig_figs(3)))

figure, (significance_axis, effect_axis) = plt.subplots(
    1, 2, figsize=(7.5, 3.4), layout="constrained"
)
colours = {0.05: "tab:blue", 0.3: "tab:orange"}
# * Each shift sits slightly to one side of its N, so that the bars do not overlap.
offsets = {0.05: 0.93, 0.3: 1.07}
for shift, colour in colours.items():
    subset = summary.filter(pl.col("shift") == shift)
    positions = subset["sample_count"] * offsets[shift]
    for axis, name in ((significance_axis, "sigma_sig"), (effect_axis, "sigma_eff")):
        median = subset[f"{name}_median"]
        axis.errorbar(
            positions,
            median,
            yerr=[median - subset[f"{name}_min"], subset[f"{name}_max"] - median],
            fmt="o-",
            color=colour,
            capsize=3,
        )
    effect_axis.axhline(shift, color=colour, linestyle="--")
significance_axis.axhline(2, color="gray", linestyle="-.")
effect_axis.axhline(0.2, color="gray", linestyle=":")
for axis, ylabel in (
    (significance_axis, r"$\sigma_\mathrm{sig}$"),
    (effect_axis, r"$\sigma_\mathrm{eff}$"),
):
    axis.set_xscale("log")
    axis.set_xticks(counts, labels=[str(count) for count in counts])
    axis.minorticks_off()
    axis.set_xlabel("samples per posterior N")
    axis.set_ylabel(ylabel)
handles = [
    Line2D([], [], color=colour, marker="o", label=f"shift {shift}")
    for shift, colour in colours.items()
]
handles += [
    Line2D([], [], color="gray", linestyle="--", label="true shift"),
    Line2D([], [], color="gray", linestyle=":", label="threshold 0.2"),
    Line2D([], [], color="gray", linestyle="-.", label="level 2"),
]
figure.legend(handles=handles, loc="outside lower center", ncol=5)
plt.show()
```

For the small shift, σ_sig grows roughly as the square root of N, as the standard error
√(2/N) of a difference of two means shrinks, and near N = 4 000 it starts to pass the
level of 2. For the large shift it soon passes the resolution of 200 draws, where the
tail fit gives a lower estimate. σ_eff instead does not grow with N. For the large
shift it stays near 0.3, and its range narrows as N grows. For the small shift it stays
far below the threshold, scattered around its truth by sampling noise as large as the
shift itself. As N grows, both shifts are resolved, and only the large one reaches the
threshold: the small shift does not fail however many samples you draw, because the
threshold acts on σ_eff, not on σ_sig.

## A reference changes the question

When one posterior is trusted, such as a long MCMC run that checks a fast
approximation, pass its name as `reference`; the package calls this mode A. The
comparison is then asymmetric. The reference fixes the whitening, the linear map that
turns its covariance into the identity, and every other posterior is measured against
it. In mode B both posteriors share a pooled whitening instead.

```{code-cell} ipython3
result_a = ms.compare(
    sample_set,
    case.names,
    case.space,
    key=1,
    reference="reference",
    metrics=["mean_shift", "energy"],
)
result_b = result.table.filter(pl.col("metric").is_in(["mean_shift", "energy"]))
pl.concat(
    [
        result_a.table.with_columns(mode=pl.lit("A, reference")),
        result_b.with_columns(mode=pl.lit("B, peers")),
    ]
).select(
    "mode",
    pl.col("pair").list.join(" to "),
    "metric",
    pl.col("parameters").list.join(", "),
    "space",
    pl.col("sigma_eff").round_sig_figs(3),
    "flag",
)
```

The `space` column records the whitening of every row, so a table always says in which
coordinates its numbers hold.

## Differences in shape

Mean and covariance are not the whole posterior. `zoo.banana` builds a twisted
Gaussian, the "banana" of {cite:t}`sejdinovic2014kernel`, and a Gaussian with exactly
its mean and covariance. Statistics that see only the first two moments cannot tell
them apart; statistics of the whole distribution can.

```{code-cell} ipython3
banana_case = zoo.banana()
banana_set = banana_case.sample(jax.random.key(2), 2_000)

figure, axes = plt.subplots(
    1, 2, figsize=(7.5, 3.4), sharex=True, sharey=True, layout="constrained"
)
for axis, index, colour in zip(axes, (0, 1), ("tab:blue", "tab:orange"), strict=True):
    points = banana_set.samples[0, index]
    axis.scatter(points[:, 0], points[:, 1], s=2, alpha=0.3, color=colour)
    axis.set_title(banana_case.names[index])
    axis.set_xlabel("x0")
axes[0].set_ylabel("x1")
plt.show()

banana_result = ms.compare(
    banana_set,
    banana_case.names,
    banana_case.space,
    key=3,
    metrics=[
        "mean_shift",
        "width_ratio",
        "kl:gaussian",
        "w2:gaussian",
        "ks",
        "energy",
        "mmd",
        "kl:knn",
    ],
    draws=100,
)
display(
    banana_result.table.select(
        pl.col("pair").list.join(" to "),
        "metric",
        "estimator",
        pl.col("parameters").list.join(", "),
        pl.col("value", "sigma_sig", "sigma_eff", "r_eff").round_sig_figs(3),
        "flag",
    )
)
true_forward = banana_case.reference_value("kl", ("banana", "gaussian"))
true_reverse = banana_case.reference_value("kl", ("gaussian", "banana"))
print(
    f"true KL(banana || gaussian) = {true_forward:.3f},",
    f"true KL(gaussian || banana) = {true_reverse:.3f}",
)
```

The moment rows read zero within their null: the mean shift, the width ratio and the
Kullback-Leibler divergence {cite:p}`kullback1951information` and 2-Wasserstein
distance {cite:p}`dowson1982frechet` between Gaussian fits. By construction the two
posteriors share both moments. The Kolmogorov-Smirnov row of the twisted parameter
sees the difference in its marginal, and the joint rows, the energy distance, the MMD
and the k-nearest-neighbour (kNN) estimate of the KL divergence
{cite:p}`wang2009divergence`, see the twist as a whole.

The kNN rows estimate the KL divergence itself, which the zoo gives in closed form for
both directions. From the banana to the Gaussian the estimate comes close to the truth,
about 1.5. The reverse divergence is about 16.5: the Gaussian puts mass where the banana
has almost none, on both sides of the parabola, and those regions dominate the integral.
Samples of the banana hardly reach them, so the reverse kNN row reads far less. An
estimate of a divergence is only as good as the samples in the region that dominates it,
which is one more reason to read several rows rather than one.

## Which statistics to run

Without `metrics`, `compare` runs a fast default set of fourteen statistics
(`mixed_signals.engines.compare.DEFAULT_METRICS`): the moment rows, the marginal rows,
the Gaussian fits, the kernel rows and sliced optimal transport. The slower kNN rows
and the Sinkhorn divergence {cite:p}`cuturi2013sinkhorn,feydy2019interpolating` see
differences in shape across all parameters at once; name them when the posteriors are
far from Gaussian. A difference that only some rows see tells you what kind of
difference it is.

## Summary

- σ_sig says whether a difference is resolved by the samples; it grows with N.
- σ_eff says how large the difference is, in posterior widths; it does not grow with N.
- A row fails when its effect size reaches its threshold and its σ_sig reaches 2.
- Mode B compares peers in a pooled whitening; mode A measures against a reference.
- Moment statistics see shifts and changes of width; statistics of the whole
  distribution also see changes of shape.

The next tutorial, {doc}`spaces_and_diagnostics`, treats angles, bounded parameters
and the quality checks of a single posterior.
