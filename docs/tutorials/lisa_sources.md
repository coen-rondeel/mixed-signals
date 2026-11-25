---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# LISA sources

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

The Laser Interferometer Space Antenna (LISA) will observe thousands of
gravitational-wave sources at once: binaries of white dwarfs in the Milky Way (Galactic
binaries), massive black-hole binaries (MBHBs), extreme mass-ratio inspirals and binaries
of stellar-mass black holes. A global fit analyses all of them together and returns one
posterior per detected source. Two global fits of the same data are compared source by
source: first the sources of one fit are matched to those of the other, then the
conventions of the two fits are aligned, and only then are the posteriors compared.
`mixed_signals.gw.lisa` does all three. This tutorial uses synthetic posteriors, drawn
in its first cell, so it runs without the files of a real global fit.

The frames, units and epochs follow the conventions document of the LISA Science Ground
Segment {cite:p}`baghi2026lisa`. Reading them needs the `gw` extra of the package,
`pip install "mixed-signals[gw]"`, for the LISA constants and the Planck15 cosmology
{cite:p}`ade2016planck,astropy2022astropy`.

## Synthetic posteriors in the layout of a global fit

The catalog pipeline of the LISA mission, catpylisa, gives the posteriors of a global fit
as a mapping from source id to a table of samples, a NumPy structured array with one
named field per parameter. The next cell draws such mappings for two fits of three
MBHBs. Each posterior is a Gaussian around the true parameters with a width typical of a
loud MBHB. Both fits agree on the first two sources; on the third, the second fit's
chirp mass sits half a posterior width higher.

```{code-cell} ipython3
import math

import numpy as np
import polars as pl
from IPython.display import display
from numpy.lib import recfunctions

from mixed_signals.errors import ConventionError
from mixed_signals.gw.lisa import (
    GalacticBinaries,
    MassiveBlackHoleBinaries,
    compare_sources,
    match_indices,
)

FIELD_NAMES = (
    "primary_mass_det_frame",
    "secondary_mass_det_frame",
    "luminosity_distance",
    "spin1",
    "spin2",
    "coalescence_time",
    "phase_at_reference_time",
    "right_ascension",
    "declination",
    "inclination",
    "polarization",
)
SAMPLE_COUNT = 4_000
TWO_PI = 2 * math.pi

# * Three true MBHBs: masses in solar masses, distances in Gpc, times in seconds.
true_sources = [
    {
        "primary_mass_det_frame": 1.1e6,
        "secondary_mass_det_frame": 2.0e5,
        "luminosity_distance": 10.0,
        "spin1": 0.3,
        "spin2": -0.2,
        "coalescence_time": 2.0e7,
        "phase_at_reference_time": 1.0,
        "right_ascension": 0.5,
        "declination": 0.3,
        "inclination": 0.8,
        "polarization": 0.4,
    },
    {
        "primary_mass_det_frame": 8.0e5,
        "secondary_mass_det_frame": 3.5e5,
        "luminosity_distance": 25.0,
        "spin1": 0.8,
        "spin2": 0.1,
        "coalescence_time": 4.5e7,
        "phase_at_reference_time": 4.0,
        "right_ascension": 3.0,
        "declination": -0.7,
        "inclination": 2.2,
        "polarization": 1.9,
    },
    {
        "primary_mass_det_frame": 1.5e6,
        "secondary_mass_det_frame": 2.0e5,
        "luminosity_distance": 50.0,
        "spin1": -0.5,
        "spin2": 0.5,
        "coalescence_time": 6.8e7,
        "phase_at_reference_time": 2.5,
        "right_ascension": 5.5,
        "declination": 1.0,
        "inclination": 1.3,
        "polarization": 5.0,
    },
]
SHIFTED_SOURCE = 2
# * Absolute widths of the posteriors of the remaining parameters.
PARAMETER_WIDTHS = {
    "spin1": 0.03,
    "spin2": 0.03,
    "coalescence_time": 100.0,
    "phase_at_reference_time": 0.3,
    "right_ascension": 0.02,
    "declination": 0.02,
    "inclination": 0.15,
    "polarization": 0.1,
}
CIRCLE_NAMES = ("phase_at_reference_time", "right_ascension", "polarization")


def draw_fit(seed, truth_of_id, chirp_shift_widths=0.0):
    """Draw the posteriors of one fit: source id to a structured array."""
    generator = np.random.default_rng(seed)
    posteriors = {}
    for source_id, truth_index in truth_of_id.items():
        truth = true_sources[truth_index]
        primary, secondary = (
            truth["primary_mass_det_frame"],
            truth["secondary_mass_det_frame"],
        )
        chirp_mass = (primary * secondary) ** 0.6 / (primary + secondary) ** 0.2
        mass_ratio = secondary / primary
        chirp_width = 0.005 * chirp_mass
        shift = chirp_shift_widths * chirp_width if truth_index == SHIFTED_SOURCE else 0
        chirp_samples = generator.normal(chirp_mass + shift, chirp_width, SAMPLE_COUNT)
        ratio_samples = generator.normal(mass_ratio, 0.03 * mass_ratio, SAMPLE_COUNT)
        primary_samples = (
            chirp_samples * (1 + ratio_samples) ** 0.2 / ratio_samples**0.6
        )
        table = np.zeros(SAMPLE_COUNT, dtype=[(name, "f8") for name in FIELD_NAMES])
        table["primary_mass_det_frame"] = primary_samples
        table["secondary_mass_det_frame"] = ratio_samples * primary_samples
        distance = truth["luminosity_distance"]
        table["luminosity_distance"] = generator.normal(
            distance, 0.08 * distance, SAMPLE_COUNT
        )
        for name, width in PARAMETER_WIDTHS.items():
            table[name] = generator.normal(truth[name], width, SAMPLE_COUNT)
        for name in CIRCLE_NAMES:
            table[name] = np.mod(table[name], TWO_PI)
        posteriors[source_id] = table
    return posteriors


posteriors_a = draw_fit(11, {"A-1": 0, "A-2": 1, "A-3": 2})
posteriors_b = draw_fit(12, {"B-7": 2, "B-3": 0, "B-5": 1}, chirp_shift_widths=0.5)
print(posteriors_a["A-1"].dtype.names)
print(posteriors_a["A-1"].shape)
```

## Holders

A holder keeps the B sources of one fit, N weighted samples each, with the name of the
fit and a record of its conventions. It checks every sample against the physical domain
of its parameter, pads sources with fewer samples with zero-weight copies, and keeps
parameters that are constant within a source, such as a reference frequency, as
settings of that source rather than as parameters.

```{code-cell} ipython3
fit_a = MassiveBlackHoleBinaries.from_posteriors(
    posteriors_a, name="fit_a", epoch=0.0, source_frame="LN"
)
fit_b = MassiveBlackHoleBinaries.from_posteriors(
    posteriors_b, name="fit_b", epoch=0.0, source_frame="LN"
)
print(repr(fit_a))
print(fit_a.parameter_names)
print(fit_a.conventions)
print(fit_a.sample_set.shape)
display(fit_a.settings)
display(
    pl.DataFrame(
        {
            "name": [parameter.name for parameter in fit_a.space.parameters],
            "topology": [
                parameter.topology.value for parameter in fit_a.space.parameters
            ],
            "lower_bound": [
                parameter.bounds[0] for parameter in fit_a.space.parameters
            ],
            "upper_bound": [
                parameter.bounds[1] for parameter in fit_a.space.parameters
            ],
        }
    )
)
print(fit_a.space.spheres)
```

The masses and distances lie on (0, ∞), the spins on [−1, 1], the inclination on [0, π],
the phase and the polarisation on circles, and the sky position on a sphere. The
reference frequency is missing from these files, as from those of some global fits, so
it takes the default of the catpylisa conventions, 0, which marks the reference that
the waveform code chooses.

## Conventions

Two fits may write the same source in different frames: the sky in equatorial (ICRS) or
ecliptic coordinates, the times from different epochs. A comparison refuses holders
whose conventions differ, and names the method that aligns them.

```{code-cell} ipython3
def sample_mean(holder, name, source=0):
    """Mean of one parameter of one source of a holder."""
    column = holder.space.index(name)
    return float(holder.sample_set.samples[source, 0, :, column].mean())


ecliptic_b = fit_b.to_ecliptic()
try:
    compare_sources(fit_a, ecliptic_b, key=0, metrics=["mean_shift"])
except ConventionError as error:
    print(error)

print([name for name in ecliptic_b.parameter_names if name.startswith("polarisation")])

aligned = ecliptic_b.to_icrs()
right_ascension_difference = abs(
    aligned.sample_set.samples[..., aligned.space.index("right_ascension")]
    - fit_b.sample_set.samples[..., fit_b.space.index("right_ascension")]
).max()
print(f"largest right-ascension difference: {right_ascension_difference:.1e} rad")

shifted_epoch = fit_b.at_epoch(1_000.0)
before = sample_mean(fit_b, "coalescence_time")
after = sample_mean(shifted_epoch, "coalescence_time")
print(f"mean coalescence time of the first source: {before:.1f} s, then {after:.1f} s")
print(f"difference: {after - before:.1f} s")
try:
    compare_sources(fit_a, shifted_epoch, key=0, metrics=["mean_shift"])
except ConventionError as error:
    print(error)

realigned = shifted_epoch.at_epoch(0.0)
print(realigned.conventions == fit_b.conventions)
```

The sky rotation between ICRS and the ecliptic also shifts the polarisation angle,
because the polarisation is measured against the frame's north (SGS Eq. 53); the holder
renames the angle `polarisation_ecliptic`, so no table can mix the two.

## Mass parameters and priors

The inspiral of a binary constrains its chirp mass 𝓜 = (m₁m₂)^{3/5}/(m₁ + m₂)^{1/5}
best {cite:p}`cutler1994gravitational`, so the posteriors of inspiral-dominated signals
are often closer to Gaussian in (𝓜, q) than in (m₁, m₂), with the mass ratio
q = m₂/m₁ ≤ 1. Priors are sometimes set in the total mass M = m₁ + m₂ and the inverted
ratio Q = m₁/m₂ ≥ 1. The holders map exactly between these forms and move the log-prior
by the Jacobian of each map {cite:p}`papamakarios2021normalizing`.

```{code-cell} ipython3
chirp_a = fit_a.to_chirp_mass_and_mass_ratio()
print(chirp_a.parameter_names)
print(
    f"first source: chirp mass {sample_mean(chirp_a, 'chirp_mass_det_frame'):.4g},"
    f" mass ratio {sample_mean(chirp_a, 'mass_ratio'):.4g}"
)

prior_form = fit_a.to_total_mass_and_mass_ratio(inverted=True).with_bounds(
    {"total_mass_det_frame": (1e5, 1e8), "inverted_mass_ratio": (1.0, 10.0)}
)
display(
    pl.DataFrame(
        [
            {
                "name": name,
                "topology": prior_form.space[name].topology.value,
                "lower_bound": prior_form.space[name].bounds[0],
                "upper_bound": prior_form.space[name].bounds[1],
            }
            for name in ("total_mass_det_frame", "inverted_mass_ratio")
        ]
    )
)

bounded_early = fit_a.with_bounds({"primary_mass_det_frame": (1e5, 1e8)})
try:
    bounded_early.to_total_mass_and_mass_ratio(inverted=True)
except ValueError as error:
    print(error)
```

A transform cannot carry a prior bound, because a box in one set of coordinates is no
box in another, so the order is: build the holder, transform it, then add the bounds
with `with_bounds`. Bounds matter for the density estimates and for the railing check
of {doc}`spaces_and_diagnostics`.

## Matching and comparing

The two fits name their sources differently and in another order. A match table pairs
them, as catpylisa's matching of detections gives it. `match_indices` turns the pairs of
source ids into pairs of positions, and `compare_sources` compares each pair as one
problem.

```{code-cell} ipython3
chirp_b = fit_b.to_chirp_mass_and_mass_ratio()
id_pairs = [("A-1", "B-3"), ("A-2", "B-5"), ("A-3", "B-7")]
matches = match_indices(chirp_a, chirp_b, id_pairs)
print(matches)

comparison = compare_sources(
    chirp_a,
    chirp_b,
    matches=matches,
    key=7,
    metrics=["mean_shift", "energy", "kl:gaussian"],
    draws=100,
)
display(comparison.pairs)
chirp_name = "chirp_mass_det_frame"
parameters = pl.col("parameters")
# * Rows over several parameters are joint rows, labelled by their number of parameters.
parameter_label = (
    pl.when(parameters.list.len() > 1)
    .then(
        pl.concat_str(pl.lit("joint, "), parameters.list.len(), pl.lit(" parameters"))
    )
    .otherwise(parameters.list.join(", "))
)
display(
    comparison.result.table.join(comparison.pairs, on="problem")
    .filter(
        (pl.col("metric") != "mean_shift") | (parameters.list.join(", ") == chirp_name)
    )
    .select(
        "first_source_id",
        "second_source_id",
        pl.col("pair").list.join(" to "),
        "metric",
        parameter_label.alias("parameters"),
        pl.col("sigma_eff", "sigma_sig").round_sig_figs(3),
        "flag",
    )
)
display(
    comparison.result.table.join(comparison.pairs, on="problem")
    .select("first_source_id", "second_source_id", "metric", "warnings")
    .explode("warnings")
    .unnest("warnings")
    .filter(pl.col("category") == "ValidityWarning")
)
```

The first two pairs pass. The third pair flags, with σ_eff near the 0.5 widths of its
chirp mass shift in the chirp-mass row; the joint rows read the same shift, measured in
the whitened space of all parameters. The energy rows of two pairs carry a warning in
their `warnings` column: the phase of those posteriors spreads over about 0.3 rad, where
the chordal distances of the kernel metrics start to read shifts low.

`compare_sources` also checks that the matched sources agree on their settings. A fit
that writes another reference frequency for a source cannot be compared on that source,
because its reference phase means something else:

```{code-cell} ipython3
posteriors_c = {
    source_id: recfunctions.append_fields(
        table, "frequency_at_reference_time", np.full(table.shape, 1e-3), usemask=False
    )
    for source_id, table in posteriors_b.items()
}
fit_c = MassiveBlackHoleBinaries.from_posteriors(
    posteriors_c, name="fit_c", epoch=0.0, source_frame="LN"
).to_chirp_mass_and_mass_ratio()
try:
    compare_sources(chirp_a, fit_c, matches=matches, key=7, metrics=["mean_shift"])
except ConventionError as error:
    print(error)
```

## Galactic binaries

The phase φ₀ and the polarisation ψ of a Galactic binary are degenerate: the waveform
depends on them only through the combinations θ± = φ₀ ± 2ψ, so four points of
(φ₀, ψ) give the same signal (SGS Eqs. 69, 146 and 147 {cite:p}`baghi2026lisa`). Two fits can
land on different copies of the same solution, and a comparison in (φ₀, ψ) would read
that as a difference. `to_phase_combinations` maps a Galactic-binary holder to (θ₊, θ₋),
where the copies coincide.

```{code-cell} ipython3
generator = np.random.default_rng(21)
GALACTIC_FIELDS = (
    "frequency",
    "frequency_dot",
    "amplitude",
    "right_ascension",
    "declination",
    "inclination",
    "polarization",
    "initial_phase",
)
GALACTIC_CENTRES = (3.0e-3, 1.0e-17, 1.0e-22, 2.0, -0.4, 1.1, 0.8, 1.0)
GALACTIC_WIDTHS = (1.0e-9, 1.0e-19, 1.0e-23, 0.05, 0.05, 0.2, 0.2, 0.3)
binary_table = np.zeros(2_000, dtype=[(name, "f8") for name in GALACTIC_FIELDS])
for name, centre, width in zip(
    GALACTIC_FIELDS, GALACTIC_CENTRES, GALACTIC_WIDTHS, strict=True
):
    binary_table[name] = generator.normal(centre, width, binary_table.shape)

# * The degenerate copy: (phi0 + pi, psi + pi/2) gives the same signal.
copy_table = binary_table.copy()
copy_table["initial_phase"] = np.mod(binary_table["initial_phase"] + math.pi, TWO_PI)
copy_table["polarization"] = np.mod(binary_table["polarization"] + math.pi / 2, TWO_PI)

binary_a = GalacticBinaries.from_posteriors(
    {"GB-1": binary_table}, name="fit_a", epoch=0.0
)
binary_b = GalacticBinaries.from_posteriors(
    {"GB-1": copy_table}, name="fit_b", epoch=0.0
)
rows = []
for form, first, second in (
    ("phi0 and psi", binary_a, binary_b),
    (
        "theta+ and theta-",
        binary_a.to_phase_combinations(),
        binary_b.to_phase_combinations(),
    ),
):
    form_result = compare_sources(
        first, second, key=8, metrics=["mean_shift"], draws=50
    ).result
    rows.append(form_result.table.with_columns(form=pl.lit(form)))
phase_names = [
    "initial_phase",
    "polarisation_icrs",
    "theta_plus_icrs",
    "theta_minus_icrs",
]
# * The other six parameters are identical in the two copies and read zero.
display(
    pl.concat(rows)
    .filter(pl.col("parameters").list.get(0).is_in(phase_names))
    .select(
        "form",
        pl.col("parameters").list.join(", "),
        pl.col("value", "sigma_eff", "sigma_sig").round_sig_figs(3),
        "flag",
    )
)
```

In (φ₀, ψ) the polarisation row reads a shift of several posterior widths and flags.
The phase row reads as large a shift, but it does not flag: pooled, the two copies put
their phases in two modes half a period apart, so the permutation null of that row is
wide, and the shift is not resolved. A degeneracy confuses the null as well as the
value. In (θ₊, θ₋) the two copies are the same posterior, and every row reads zero.

## Summary

- A holder keeps the sources of one global fit with their conventions, checks their
  values and keeps constant parameters as settings.
- Comparisons refuse holders whose frames, epochs or settings differ, and name the
  method that aligns them.
- Mass forms and phase combinations remove differences that only the coordinates make;
  prior bounds go on after the transform.
- `match_indices` and `compare_sources` compare two fits source by source, one problem
  per matched pair.
