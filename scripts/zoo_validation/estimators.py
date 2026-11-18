"""Checks of the estimators: mixture starts and the railing score of the KDE."""

import functools
import math
from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jaxtyping import Array
from matplotlib.figure import Figure as MatplotlibFigure

from mixed_signals import zoo
from mixed_signals.core.space import Parameter, ParamSpace
from mixed_signals.diagnostics.railing import railing_scores
from mixed_signals.estimators.gmm import DEFAULT_STARTS, MixtureDensity, MixtureSettings
from mixed_signals.estimators.kde import KernelDensity
from mixed_signals.prng import derive_key
from zoo_validation import figures
from zoo_validation.report import (
    BINOMIAL_ERRORS,
    Case,
    Cell,
    Check,
    Claim,
    Context,
    Figure,
    Measurement,
    Section,
    Table,
    claim_at_most,
    claim_cases,
    claim_cases_above,
    claim_true,
    grouped,
    role,
)

# * Check gmm-starts.

GMM_LABEL = "gmm-starts"
SEPARATIONS = (2.0, 3.0, 4.0, 5.0, 6.0)
GMM_SAMPLE_COUNT = 4_000
GMM_COMPONENTS = 2
PLATEAU_RECORD = 0.675
"""Largest share of single starts on the plateau that session 6 recorded."""
GAP_RECORD = (2.6, 2.8)
"""Median gap, in nats, of the fits of the default starts in session 6."""
GAP_FACTOR = 2.0
"""Factor within which the median gap must reproduce the record."""
CONTRAST_NATS = 0.01
"""Tolerance of the contrast fits, on the total log-likelihood."""
CONTRAST_RECORD = 0.19
"""Largest share on the plateau with the tolerance of the contrast, in session 6."""


@functools.cache
def _fit_totals(
    space: ParamSpace, settings: MixtureSettings
) -> Callable[[Array, Array], tuple[Array, Array]]:
    """Return the fit over a batch of keys: the total log-likelihood and the flag.

    The compiled program takes the samples and the keys as arguments, so every
    replicate and every separation reuses it.
    """

    def fit(samples: Array, key: Array) -> tuple[Array, Array]:
        mixture = MixtureDensity.fit(samples, None, space, settings, key=key)
        return samples.shape[0] * mixture.mean_log_likelihood, mixture.converged

    return jax.jit(jax.vmap(fit, in_axes=(None, 0)))


class _Starts(NamedTuple):
    """The total log-likelihoods of the fits of one separation, by replicate."""

    single: Array
    """Single starts at the default tolerance, shaped (replicates, starts)."""
    contrast: Array
    """The same keys at the tolerance of the contrast."""
    contrast_converged: Array
    """Whether each contrast start met its tolerance within the iteration cap."""
    default: Array
    """The fit of the default number of starts, shaped (replicates,)."""
    one_gaussian: Array
    """The fit of one component, shaped (replicates,)."""


def _mixture_starts(
    context: Context,
    separation_index: int,
    separation: float,
    replicates: int,
    starts: int,
) -> _Starts:
    """Fit every replicate of one separation with single starts and with 18 starts."""
    case = zoo.gaussian_mixture(separation)
    space = case.space
    sets = case.sample(
        context.key(GMM_LABEL, "sets", separation_index),
        GMM_SAMPLE_COUNT,
        batch_size=replicates,
    ).samples[:, case.names.index("mixture")]
    one_fit = _fit_totals(space, MixtureSettings(components=1, starts=1))
    single_fit = _fit_totals(
        space, MixtureSettings(components=GMM_COMPONENTS, starts=1)
    )
    contrast_fit = _fit_totals(
        space,
        MixtureSettings(
            components=GMM_COMPONENTS,
            starts=1,
            tolerance=CONTRAST_NATS / GMM_SAMPLE_COUNT,
        ),
    )
    default_fit = _fit_totals(space, MixtureSettings(components=GMM_COMPONENTS))
    single, contrast, converged, default, one = [], [], [], [], []
    for replicate in range(replicates):
        samples = sets[replicate]
        path = (separation_index, replicate)
        keys = jnp.stack(
            [
                derive_key(context.key(GMM_LABEL, "single", *path), "start", index)
                for index in range(starts)
            ]
        )
        single.append(single_fit(samples, keys)[0])
        contrast_total, contrast_flag = contrast_fit(samples, keys)
        contrast.append(contrast_total)
        converged.append(contrast_flag)
        default.append(
            default_fit(samples, context.key(GMM_LABEL, "default", *path)[None])[0][0]
        )
        one.append(one_fit(samples, context.key(GMM_LABEL, "one", *path)[None])[0][0])
    return _Starts(
        jnp.stack(single),
        jnp.stack(contrast),
        jnp.stack(converged),
        jnp.stack(default),
        jnp.stack(one),
    )


def _gmm_starts(context: Context) -> Section:
    replicates = context.count(40, 3)
    starts = context.count(100, 20)
    rows: list[list[Cell]] = []
    shares: dict[float, Measurement] = {}
    counts: dict[float, tuple[int, int]] = {}
    medians: dict[float, float] = {}
    for index, separation in enumerate(SEPARATIONS):
        fits = _mixture_starts(context, index, separation, replicates, starts)
        # * The optimum of a replicate is the best total of any of its fits.
        optimum = jnp.maximum(
            jnp.maximum(jnp.max(fits.single, axis=1), jnp.max(fits.contrast, axis=1)),
            fits.default,
        )
        midpoint = 0.5 * (fits.one_gaussian + optimum)
        on_plateau = fits.single < midpoint[:, None]
        contrast_plateau = fits.contrast < midpoint[:, None]
        trials = replicates * starts
        plateau_count = int(jnp.sum(on_plateau))
        contrast_count = int(jnp.sum(contrast_plateau))
        share = Measurement.from_rate(plateau_count, trials)
        contrast_share = Measurement.from_rate(contrast_count, trials)
        eighteenth = Measurement(
            share.value**DEFAULT_STARTS,
            DEFAULT_STARTS * share.value ** (DEFAULT_STARTS - 1) * (share.error or 0.0),
        )
        gaps = optimum - fits.default
        missed = int(jnp.sum(fits.default < midpoint))
        spread = float(jnp.std(jnp.mean(on_plateau, axis=1), ddof=1))
        nonfinite = int(jnp.sum(~jnp.isfinite(fits.single))) + int(
            jnp.sum(~jnp.isfinite(fits.contrast))
        )
        shares[separation] = share
        counts[separation] = (plateau_count, trials)
        medians[separation] = float(jnp.median(gaps))
        rows.append(
            [
                separation,
                plateau_count,
                trials,
                share,
                spread,
                eighteenth,
                f"{missed} of {replicates}",
                medians[separation],
                float(jnp.max(gaps)),
                contrast_share,
                Measurement.from_rate(int(jnp.sum(fits.contrast_converged)), trials),
                nonfinite,
            ]
        )
    table = Table.of(
        f"Fits of {GMM_COMPONENTS} components to {grouped(GMM_SAMPLE_COUNT)}"
        " samples of the zoo mixture, by separation: the single starts of"
        f" {replicates} replicates with {starts} starts each. The share is the"
        " plateau starts over all starts,"
        " with its binomial error, and the scatter column is the standard deviation of"
        f" the share between replicates. The gaps ℓ* − ℓ, in nats on the total"
        f" log-likelihood, belong to the fits of {DEFAULT_STARTS} starts, one per"
        " replicate. The last columns are the contrast with a tolerance of"
        f" {CONTRAST_NATS:g} nats.",
        (
            "Separation",
            "Plateau starts",
            "Starts",
            "Share on the plateau",
            "Scatter between replicates",
            f"Share to the power {DEFAULT_STARTS}",
            f"Fits of {DEFAULT_STARTS} starts on the plateau",
            "Median gap",
            "Largest gap",
            "Share on the plateau, contrast",
            "Share that met the tolerance, contrast",
            "Non-finite fits",
        ),
        rows,
    )
    hardest = max(shares, key=lambda separation: shares[separation].value)
    hardest_count, hardest_trials = counts[hardest]
    limit_measurement = Measurement.from_rate(
        hardest_count, hardest_trials, level=PLATEAU_RECORD
    )
    limit = PLATEAU_RECORD + BINOMIAL_ERRORS * (limit_measurement.error or 0.0)
    low, high = GAP_RECORD[0] / GAP_FACTOR, GAP_RECORD[1] * GAP_FACTOR
    claims = [
        claim_at_most(
            f"The largest share of single starts on the plateau, at separation"
            f" {hardest:g}, does not exceed the record.",
            shares[hardest],
            limit,
            limit_text=f"the record's {PLATEAU_RECORD} plus three binomial errors"
            " at that level",
            full_only=True,
        ),
        claim_true(
            f"The median gap of the fits of {DEFAULT_STARTS} starts reproduces the"
            f" record, at separation {hardest:g}.",
            f"between {low:.2g} and {high:.2g} nats (the record's"
            f" {GAP_RECORD[0]} to {GAP_RECORD[1]} nats within a factor of"
            f" {GAP_FACTOR:g})",
            f"{medians[hardest]:.3g} nats",
            low <= medians[hardest] <= high,
            full_only=True,
        ),
    ]
    setup = (
        "Two-component fits of"
        f" {role('class', 'mixed_signals.estimators.gmm.MixtureDensity')} with"
        f" {role('class', 'mixed_signals.estimators.gmm.MixtureSettings')}"
        f" (components {GMM_COMPONENTS}, starts 1) to the posterior `mixture` of"
        f" {role('func', 'mixed_signals.zoo.cases.gaussian_mixture')} at the"
        f" separations {', '.join(f'{separation:g}' for separation in SEPARATIONS)},"
        f" with {grouped(GMM_SAMPLE_COUNT)} samples and {replicates} replicates."
        f" Each replicate runs {starts} single starts, each with its own key, once at"
        " the default tolerance and once at the tolerance of the contrast, and one fit"
        f" of the default {DEFAULT_STARTS} starts. The optimum ℓ* of a replicate is the"
        " largest total log-likelihood of any of its fits, and ℓ₁ is that of one"
        " Gaussian. The record did not state its definition of the plateau, so this"
        " page defines it: a start ends on the plateau when its total log-likelihood"
        " lies below (ℓ₁ + ℓ*)/2, nearer the one-Gaussian solution than the optimum."
        " The starts of one replicate share their samples, so the binomial error of"
        " the share counts them as independent and the scatter column shows the"
        f" spread between replicates. Session 6 recorded 37 to 68% of single starts"
        f" on the plateau against at most {CONTRAST_RECORD:.0%} at the tolerance of the"
        f" contrast, and a median gap of {GAP_RECORD[0]} to {GAP_RECORD[1]} nats at"
        f" {grouped(GMM_SAMPLE_COUNT)} samples in the harder cases. The claims read"
        " the separation with the largest share on the plateau; the table gives the"
        " shares at every separation. The contrast keeps the default cap of"
        f" {MixtureSettings(components=1).max_iterations} iterations, so its last"
        " column counts the starts that reached the tolerance before the cap."
    )
    return Section(setup, (table,), tuple(claims))


# * Check kde-railing.

RAILING_LABEL = "kde-railing"
OFFSETS = (0.0, 1.0, 2.0, 2.5, 3.0)
RAILING_SIZES = (1_000, 4_000, 10_000)
RAILING_TOLERANCE = 0.05
REFLECTION_EXCESS = 0.10
RAILING_CLAIMED = {1_000: 2.0, 4_000: 2.5, 10_000: 2.5}
"""Largest offset z that the corrected score claims at each sample count."""
SCATTER_OFFSET = 2.0
SCATTER_RECORD = {1_000: 0.04, 4_000: 0.022, 10_000: 0.015}
"""Scatter per replicate of the corrected score at z = 2 in session 6."""
REFLECTED_RECORD = ((10_000, 1.0, 0.11), (10_000, 2.0, 0.31), (1_000, 2.0, 0.48))
"""Excess of the reflected score over the truth, as (N, z, excess), in session 6."""
CHUNK_REPLICATES = 50
OUTSIDE_LEGEND_HEIGHT = 1.0
"""Inches that the railing figure adds for the legends below its panels."""
GRID_POINTS = 2_048
"""Points of the grid on which the peak of the reflected estimate is sought."""
GRID_MARGIN = 3.0
"""Kernel widths that the grid reaches beyond the data."""


@functools.partial(jax.jit, static_argnames=("space",))
def _reflected_score(values: Array, space: ParamSpace) -> Array:
    """Return the density at the bound over the peak of the reflected estimate.

    The estimate has the bandwidth of the corrected one, and mirror images of
    every sample at the bound at zero.
    """
    kde = KernelDensity.fit(values[:, None], None, space)
    # * The covariance of JAX's estimator already holds the factor squared.
    width = jnp.sqrt(kde.kde.covariance[0, 0])

    def density(points: Array) -> Array:
        direct = (points[:, None] - values[None, :]) / width
        mirrored = (points[:, None] + values[None, :]) / width
        kernels = jnp.exp(-0.5 * direct**2) + jnp.exp(-0.5 * mirrored**2)
        return jnp.sum(kernels, axis=1) / (
            values.shape[0] * width * math.sqrt(2 * math.pi)
        )

    grid = jnp.linspace(
        jnp.maximum(jnp.min(values) - GRID_MARGIN * width, 0.0),
        jnp.max(values) + GRID_MARGIN * width,
        GRID_POINTS,
    )
    at_bound = density(jnp.zeros(1))[0]
    return at_bound / jnp.maximum(jnp.max(density(grid)), at_bound)


class _Scores(NamedTuple):
    """Railing scores by replicate and offset."""

    corrected: Array
    reflected: Array


def _railing_scores_of_size(context: Context, size: int, replicates: int) -> _Scores:
    """Return the corrected and the reflected score of every replicate and offset."""
    case = zoo.railed_gaussians(OFFSETS, param_count=1)
    space = case.space
    line = ParamSpace(
        parameters=(
            Parameter(
                name="value",
                topology=space.parameters[0].topology,
                bounds=space.parameters[0].bounds,
            ),
        )
    )
    corrected, reflected = [], []
    for chunk in range(-(-replicates // CHUNK_REPLICATES)):
        batch = min(CHUNK_REPLICATES, replicates - chunk * CHUNK_REPLICATES)
        sets = case.sample(
            context.key(RAILING_LABEL, "sets", size, chunk), size, batch_size=batch
        ).samples
        for replicate in range(batch):
            corrected.append(
                jnp.stack(
                    [
                        railing_scores(sets[replicate, position], None, space).lower[0]
                        for position in range(len(OFFSETS))
                    ]
                )
            )
            reflected.append(
                jnp.stack(
                    [
                        _reflected_score(sets[replicate, position, :, 0], line)
                        for position in range(len(OFFSETS))
                    ]
                )
            )
    return _Scores(jnp.stack(corrected), jnp.stack(reflected))


def _ratio_measurements(scores: Array, truths: Array) -> tuple[list[Measurement], int]:
    """Return the mean score over the truth per offset, and the non-finite count."""
    measurements, missing = [], 0
    for position in range(truths.shape[0]):
        column = scores[:, position]
        finite = jnp.isfinite(column)
        missing += int(jnp.sum(~finite))
        measurements.append(
            Measurement.from_replicates(column[finite] / truths[position])
        )
    return measurements, missing


def _scatter_measurement(scores: Array) -> Measurement:
    """Return the standard deviation of finite scores with its standard error."""
    finite = scores[jnp.isfinite(scores)]
    count = int(finite.shape[0])
    deviation = float(jnp.std(finite, ddof=1))
    return Measurement(deviation, deviation / math.sqrt(2 * (count - 1)), count)


def _kde_railing(context: Context) -> Section:
    replicates = context.count(400, 3)
    case = zoo.railed_gaussians(OFFSETS, param_count=1)
    truths = jnp.asarray(
        [case.reference_value("railing_lower", name, "x0") for name in case.names]
    )
    corrected: dict[int, list[Measurement]] = {}
    reflected: dict[int, list[Measurement]] = {}
    scatter: dict[int, tuple[Measurement, Measurement]] = {}
    missing = 0
    for size in RAILING_SIZES:
        scores = _railing_scores_of_size(context, size, replicates)
        corrected[size], missing_corrected = _ratio_measurements(
            scores.corrected, truths
        )
        reflected[size], missing_reflected = _ratio_measurements(
            scores.reflected, truths
        )
        missing += missing_corrected + missing_reflected
        position = OFFSETS.index(SCATTER_OFFSET)
        scatter[size] = (
            _scatter_measurement(scores.corrected[:, position]),
            _scatter_measurement(scores.reflected[:, position]),
        )
    rows: list[list[Cell]] = []
    for estimator, by_size in (("corrected", corrected), ("reflected", reflected)):
        for position, offset in enumerate(OFFSETS):
            rows.append(
                [
                    estimator,
                    offset,
                    *[by_size[size][position] for size in RAILING_SIZES],
                ]
            )
    ratio_table = Table.of(
        f"Mean railing score over the truth exp(−z²/2) of {replicates} replicates, with"
        " the standard error of the mean, by estimator, offset z and sample count.",
        ("Estimator", "z", *[f"N = {grouped(size)}" for size in RAILING_SIZES]),
        rows,
    )
    scatter_table = Table.of(
        f"Scatter per replicate of the score at z = {SCATTER_OFFSET:g}, with its"
        " standard error, and the scatter that session 6 recorded for the corrected"
        " estimate.",
        ("N", "Corrected", "Reflected", "Record, corrected"),
        [[size, *scatter[size], SCATTER_RECORD[size]] for size in RAILING_SIZES],
    )
    cases: list[Case] = []
    for size in RAILING_SIZES:
        for position, offset in enumerate(OFFSETS):
            if offset <= RAILING_CLAIMED[size]:
                entry = corrected[size][position]
                cases.append(
                    Case(
                        f"z = {offset:g}, N = {grouped(size)}",
                        entry.value - 1.0,
                        entry.margin(),
                        RAILING_TOLERANCE,
                    )
                )
    position = OFFSETS.index(SCATTER_OFFSET)
    excess_cases = [
        Case(
            f"N = {grouped(size)}",
            reflected[size][position].value - 1.0,
            reflected[size][position].margin(),
            REFLECTION_EXCESS,
        )
        for size in RAILING_SIZES
    ]
    claimed_text = ", ".join(
        f"z up to {offset:g} at N = {grouped(size)}"
        for size, offset in RAILING_CLAIMED.items()
    )
    claims: list[Claim] = [
        claim_cases(
            f"The corrected score recovers exp(−z²/2): {claimed_text}.",
            cases,
            tolerance_text=f"{RAILING_TOLERANCE:.0%} of the truth",
            replicates=replicates,
        ),
        claim_cases_above(
            f"The reflected score reads high at z = {SCATTER_OFFSET:g}, at every N.",
            excess_cases,
            tolerance_text=f"{REFLECTION_EXCESS:.0%} of the truth",
            replicates=replicates,
        ),
    ]
    data = {
        "offsets": OFFSETS,
        "ratios": {
            estimator: {
                str(size): [[entry.value, entry.error] for entry in by_size[size]]
                for size in RAILING_SIZES
            }
            for estimator, by_size in (
                ("corrected", corrected),
                ("reflected", reflected),
            )
        },
    }

    def draw(style: figures.FigureStyle) -> MatplotlibFigure:
        figure, panels = figures.new_figure(
            style,
            columns=len(RAILING_SIZES),
            sharey=True,
            extra_height=OUTSIDE_LEGEND_HEIGHT,
        )
        for index, (axes, size) in enumerate(zip(panels, RAILING_SIZES, strict=True)):
            for series, (name, by_size) in enumerate(
                (("corrected", corrected), ("reflected", reflected))
            ):
                figures.plot_series(
                    axes,
                    style,
                    series,
                    OFFSETS,
                    [entry.value for entry in by_size[size]],
                    label=name,
                    errors=[entry.error or 0.0 for entry in by_size[size]],
                )
            figures.plot_reference(axes, style, y_value=1.0)
            figures.finish_panel(
                axes,
                style,
                title=f"N = {grouped(size)}",
                x_label="offset z",
                y_label="score over truth" if index == 0 else None,
            )
            axes.set_xticks(OFFSETS)
            figures.add_legend(axes, style, outside=True)
        return figure

    figure = Figure(
        label=f"{RAILING_LABEL}-ratio",
        caption="Mean railing score over its truth against the offset z of the peak"
        " from the bound, by sample count, for the corrected and the reflected"
        " estimate. The line is one and the bars are standard errors of the mean.",
        alt="Three panels of the score over the truth against the offset: the"
        " corrected estimate stays near one, the reflected one lies above it and"
        " rises with the offset.",
        draw=draw,
        data=data,
    )
    reflected_record = ", ".join(
        f"+{excess:.0%} at N = {grouped(size)} for z = {offset:g}"
        for size, offset, excess in REFLECTED_RECORD
    )
    missing_text = (
        "No score was NaN or infinite."
        if missing == 0
        else f"{missing} scores were NaN or infinite and are left out of the means."
    )
    setup = (
        "Normals truncated at the bound zero, with their peak z standard deviations"
        " inside it:"
        f" {role('func', 'mixed_signals.zoo.cases.railed_gaussians')} with offsets"
        f" {', '.join(f'{offset:g}' for offset in OFFSETS)}, one parameter,"
        f" {', '.join(grouped(size) for size in RAILING_SIZES)} samples and"
        f" {replicates} replicates per size. The corrected score is"
        f" {role('func', 'mixed_signals.diagnostics.railing.railing_scores')}, the"
        " density of the corrected KDE at the bound over its peak, and the truth is"
        " the reference row `railing_lower`, exp(−z²/2). The reflected score is"
        " computed in the script: the same estimate with the bandwidth of the"
        " corrected one and mirror images of every sample at the bound, whose"
        f" density at the bound is read over the largest value on a grid of"
        f" {grouped(GRID_POINTS)} points that reaches {GRID_MARGIN:g} bandwidths"
        " beyond the data. Session 6 recorded a reflected score high by"
        f" {reflected_record}, where the correction left 2%, 4% and −2%. A claim"
        " tests the mean score over the truth against one, with the tolerance and"
        " the standard errors. "
        f"{missing_text}"
    )
    return Section(setup, (ratio_table, scatter_table, figure), tuple(claims))


CHECKS: tuple[Check, ...] = (
    Check(
        label=GMM_LABEL,
        title="Starts of the Gaussian mixture",
        session="6",
        supports=(
            role("data", "mixed_signals.estimators.gmm.DEFAULT_STARTS", short=False),
        ),
        run=_gmm_starts,
    ),
    Check(
        label=RAILING_LABEL,
        title="Railing score of the corrected KDE",
        session="6 and 11",
        supports=(
            role(
                "func", "mixed_signals.diagnostics.railing.railing_scores", short=False
            ),
            role("class", "mixed_signals.estimators.kde.KernelDensity", short=False),
        ),
        run=_kde_railing,
    ),
)
"""Checks of the estimators."""
