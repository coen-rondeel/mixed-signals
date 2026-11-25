---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Spaces and diagnostics

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

Statistics that treat every parameter as a number on the real line go wrong in two
common cases: an angle, which wraps around at 2π, and a parameter that presses against
a bound of its prior. And before you compare two posteriors, you want to know whether
each of them is sound on its own: one mode or several, Gaussian or skewed, free or
pressed against a bound. This tutorial covers both. It first describes the parameter
space and why its topology matters, then the quality checks of a single posterior,
which `diagnose` runs.

## Topologies and spheres

Every parameter of a `ParamSpace` has one of three topologies, and two parameters can
together form a sphere:

- **Real**: the real line, with no bound, such as a time.
- **Interval**: a finite bound on one side or on both, such as a mass, which is
  positive, or a spin between −1 and 1. Density estimates correct their bias near a
  finite bound {cite:p}`jones1993simple`, and the railing check below looks for
  posteriors that press against one.
- **Circle**: an angle with a period, such as a phase. It has no edge: 0.01 and
  2π − 0.01 are close neighbours. Means, widths and tests use circular statistics
  {cite:p}`mardia2000directional`.
- **Sphere**: not a topology of one parameter but a pair, declared under `spheres`: an
  azimuth on a circle and a polar angle on an interval, which together give a
  direction, such as a sky position. Distances run along the sphere, and the poles are
  no edges.

A space can be written in YAML. This one holds a frequency on (0, ∞), a phase on a
circle and a sky position on a sphere, whose azimuth and latitude are two parameters
of the space:

```{code-cell} ipython3
import math

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import polars as pl
from IPython.display import display

import mixed_signals as ms
from mixed_signals import zoo
from mixed_signals.core.space import Parameter, ParamSpace
from mixed_signals.core.topology import Topology

space_yaml = f"""\
parameters:
  - name: frequency
    unit: Hz
    topology: interval
    bounds: [0.0, .inf]
  - name: phase
    unit: rad
    topology: circle
    bounds: [0.0, {2 * math.pi}]
  - name: right_ascension
    unit: rad
    topology: circle
    bounds: [0.0, {2 * math.pi}]
  - name: declination
    unit: rad
    topology: interval
    bounds: [{-math.pi / 2}, {math.pi / 2}]
spheres:
  - azimuth: right_ascension
    polar: declination
    polar_form: latitude
"""
print(space_yaml)
sky_space = ParamSpace.from_yaml_text(space_yaml)
display(
    pl.DataFrame(
        {
            "name": [parameter.name for parameter in sky_space.parameters],
            "topology": [
                parameter.topology.value for parameter in sky_space.parameters
            ],
            "lower_bound": [parameter.bounds[0] for parameter in sky_space.parameters],
            "upper_bound": [parameter.bounds[1] for parameter in sky_space.parameters],
            "unit": [parameter.unit for parameter in sky_space.parameters],
        }
    )
)
print(sky_space.spheres)
```

## Why the circle matters

`zoo.wrapped_gaussians` builds a wrapped Gaussian centred on the cut of a circle, so
its samples fall on both sides of 0 and 2π, and a copy shifted by 0.3 of its width.
The next cell compares the two twice: once in the case's own space, where the first
parameter is a circle, and once in a space that wrongly declares it an interval from 0
to 2π.

```{code-cell} ipython3
case = zoo.wrapped_gaussians(0.3, width=0.5, param_count=2)
sample_set = case.sample(jax.random.key(0), 2_000)

# * The same parameters, but the first one declared as an interval, not a circle.
wrong_space = ParamSpace(
    parameters=(
        Parameter(
            name=case.space.names[0],
            topology=Topology.INTERVAL,
            bounds=case.space.parameters[0].bounds,
        ),
        *case.space.parameters[1:],
    )
)

figure, axis = plt.subplots(figsize=(5.5, 3.2), layout="constrained")
axis.hist(sample_set.samples[0, 0, :, 0], bins=60, range=(0, 2 * math.pi))
axis.set_xlabel("x0 (phase, radians)")
axis.set_ylabel("samples per bin")
axis.set_title("reference posterior")
plt.show()

rows = []
for label, used_space in (("circle", case.space), ("interval", wrong_space)):
    used_result = ms.compare(
        sample_set,
        case.names,
        used_space,
        key=7,
        metrics=["mean_shift", "energy"],
    )
    rows.append(used_result.table.with_columns(space_used=pl.lit(label)))
display(
    pl.concat(rows).select(
        "space_used",
        "metric",
        pl.col("parameters").list.join(", "),
        pl.col("sigma_eff").round_sig_figs(3),
        "flag",
    )
)
print("true shift:", case.reference_value("mean_shift", ("reference", "shifted"), "x0"))
# * The rows carry package warnings; the chordal one changes how to read energy.
display(
    pl.concat(rows)
    .select("space_used", "metric", "warnings")
    .explode("warnings")
    .unnest("warnings")
    .filter(pl.col("category") == "ValidityWarning")
)
```

On the circle the mean-shift row recovers the true shift of 0.3. Read as an interval,
the reference looks like two clumps at opposite ends of the range, and its mean falls
near π, between the clumps, where it has no samples at all. The mean shift then comes
out smaller, with the wrong sign, and no longer flags. The energy distance happens to
land not far from the truth in this case, but nothing guarantees that: a wrong topology
gives wrong numbers, though not always large ones. Declare angles as circles, and a sky
position as a sphere.

The energy row on the circle carries a warning in its `warnings` column. Kernel and
transport metrics measure a circle along its chords, which are shorter than its arcs,
so they read shifts low on circles wider than 0.3 rad; this one is about 0.5 rad wide,
which is why its energy row reads a little less than 0.3.

## Why the bound matters

A posterior rails when a bound of its prior cuts into it, for example a spin that
presses against 1 or a distance against the edge of its prior. The estimates of such a
posterior depend on the prior, and a comparison of two of them partly compares their
priors. `zoo.railed_gaussians` builds Gaussians truncated at a bound at zero, with their
peaks at chosen distances inside the bound.

```{code-cell} ipython3
offsets = [0.0, 1.0, 3.0]
railed_case = zoo.railed_gaussians(offsets)
railed_set = railed_case.sample(jax.random.key(3), 2_000)

figure, axis = plt.subplots(figsize=(5.5, 3.2), layout="constrained")
for index, name in enumerate(railed_case.names):
    axis.hist(
        railed_set.samples[0, index, :, 0],
        bins=50,
        range=(0, 6),
        histtype="step",
        label=f"{name} (peak {offsets[index]} widths inside)",
    )
axis.axvline(0.0, color="black", linestyle="--", label="bound")
axis.set_xlabel("x0")
axis.set_ylabel("samples per bin")
axis.legend()
plt.show()

railing = ms.diagnose(
    railed_set, railed_case.names, railed_case.space, key=4, diagnostics=["railing"]
)
expected = dict(
    zip(
        railed_case.names,
        (math.exp(-(offset**2) / 2) for offset in offsets),
        strict=True,
    )
)
railing.result.table.select(
    pl.col("pair").list.join(", ").alias("posterior"),
    "metric",
    pl.col("parameters").list.join(", "),
    pl.col("value", "threshold").round_sig_figs(3),
    "flag",
).with_columns(
    expected_score=pl.col("posterior").replace_strict(expected).round_sig_figs(3)
)
```

The railing score is the density of the posterior at the bound over its peak, from a
kernel density estimate with a boundary correction {cite:p}`jones1996simple`. A
Gaussian whose peak lies z widths inside the bound scores exp(−z²/2). Near the bound
the estimate follows that closed form within a few hundredths; three widths inside, the
density at the bound is tiny and the estimate less precise, here a few thousandths
against 0.011, but far below the threshold either way. The row flags
above 0.02, the edge-to-peak ratio at which the LVK tool pe-configurator reports
railing {cite:p}`estelles2023peconfigurator`, which a peak within about 2.8 widths of
the bound reaches. No paper defines a standard railing diagnostic, so the package
documents its own choice on the validation page.

## The shape of one posterior

`diagnose` checks single posteriors in five families: the health of MCMC chains (the
next tutorial), the shape of every parameter, the dependence between pairs of
parameters, the number of modes, and railing. To tell a real departure from sampling
noise, it compares each statistic with the same statistic of Gaussian surrogates: sets
drawn from the Gaussian with the posterior's mean and covariance and its effective
sample size, a parametric bootstrap {cite:p}`efron1993introduction`. A row flags when
the posterior departs from its Gaussian surrogates by more than the threshold of its
statistic and its σ_sig reaches 2.

The next cell diagnoses three posteriors: a Gaussian, the banana of the first tutorial
and a mixture of two well separated Gaussians.

```{code-cell} ipython3
gaussian_case = zoo.shifted_gaussians(0.0)
banana_case = zoo.banana()
mixture_case = zoo.gaussian_mixture(4.0)
assert gaussian_case.space == banana_case.space == mixture_case.space

# * One posterior of each case, stacked into one set of three posteriors.
posterior_names = ["gaussian", "banana", "mixture"]
three_set = ms.SampleSet(
    jnp.concatenate(
        [
            gaussian_case.sample(jax.random.key(5), 2_000).samples[:, 0:1],
            banana_case.sample(jax.random.key(6), 2_000).samples[:, 0:1],
            mixture_case.sample(jax.random.key(7), 2_000).samples[:, 0:1],
        ],
        axis=1,
    )
)

figure, axes = plt.subplots(1, 3, figsize=(7.5, 2.8), layout="constrained")
for axis, index, name in zip(axes, range(3), posterior_names, strict=True):
    points = three_set.samples[0, index]
    axis.scatter(points[:, 0], points[:, 1], s=2, alpha=0.3)
    axis.set_title(name)
    axis.set_xlabel("x0")
    axis.set_ylabel("x1")
plt.show()

shape_diagnosis = ms.diagnose(
    three_set,
    posterior_names,
    gaussian_case.space,
    key=5,
    diagnostics=["shape", "dependence", "modes"],
    draws=100,
)
rows = shape_diagnosis.result.table.select(
    pl.col("pair").list.get(0).alias("posterior"),
    "metric",
    pl.col("parameters").list.join(", "),
    "value",
    "threshold",
    "flag",
    "sigma_sig",
)
index_columns = ["metric", "parameters"]
# * Mode rows along several directions share one identity; the pivot keeps the largest.
flagging_rows = rows.filter(pl.col("threshold").is_not_null())
display(
    flagging_rows.pivot(
        on="posterior",
        index=index_columns,
        values="flag",
        aggregate_function="max",
        maintain_order=True,
    ).select(index_columns + posterior_names)
)
display(
    flagging_rows.pivot(
        on="posterior",
        index=index_columns,
        values="sigma_sig",
        aggregate_function="max",
        maintain_order=True,
    )
    .select(index_columns + posterior_names)
    .with_columns(pl.col(posterior_names).round_sig_figs(3))
)
dependence_metrics = [
    "pearson",
    "spearman",
    "distance_correlation",
    "mutual_information",
]
display(
    rows.filter(pl.col("metric").is_in(dependence_metrics))
    .pivot(
        on="posterior",
        index="metric",
        values="value",
        aggregate_function="first",
        maintain_order=True,
    )
    .select(["metric", *posterior_names])
    .with_columns(pl.col(posterior_names).round_sig_figs(3))
)
```

The Gaussian flags nothing: its statistics fall inside the spread of its own Gaussian
surrogates. The banana flags in shape and dependence. Its twisted coordinate is skewed,
and its parameters depend on each other in a way that no correlation coefficient
captures: the distance correlation {cite:p}`szekely2007measuring` and the mutual
information {cite:p}`kraskov2004mutual` exceed their Gaussian values, while Pearson's r
stays near zero. The banana also flags the mixture fit of the mode family: a mixture of
Gaussians needs several components to follow the twist, so the Bayesian information
criterion {cite:p}`schwarz1978dimension` prefers more than one. That row flags when the
preference exceeds 10, very strong evidence on the scale of {cite:t}`kass1995bayes`,
which {cite:t}`fraley2002model` use for mixtures. A flag of this row says that one
Gaussian does not describe the posterior, not that it has two separate modes.

The mixture flags in the mode family for both reasons: the information criterion prefers
two components, and the kernel density along the first parameter has two peaks. It also
flags in shape, because two separated modes make the marginal of the first parameter
flatter than a Gaussian, which its kurtosis and Anderson-Darling rows see. A σ_sig of −∞
in a mode row means that every surrogate was at least as extreme as the value, a
p-value of 1.

The statistics behind each family, with their sources, are listed in the API
reference of `mixed_signals.diagnostics`.

## Summary

- Declare every angle as a circle and every direction as a sphere; read on the real
  line, they give false differences.
- Bounds are part of the space: density estimates correct their bias at them, and the
  railing rows flag posteriors that press against them.
- `diagnose` measures shape, dependence, modes and railing, each against Gaussian
  surrogates of the same effective size, and flags departures that are both large and
  resolved.

The next tutorial, {doc}`chains_pipeline_report`, starts from MCMC chains and runs the
same steps from a configuration file.
