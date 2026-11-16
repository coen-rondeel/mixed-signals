"""Compare matched Galactic-binary posteriors with sliced optimal transport.

A marimo notebook that runs the comparison pipeline on synthetic posteriors
with known differences: pre-processing (effective sample size, the fold of
the (φ₀ + π, ψ + π/2) degeneracy, a local chart for the angles, Gaussianity
checks after a reference whitening), one pooled whitening per pair, the
empirical and Gaussian sliced W2 with self-term de-biasing and a permutation
null, and the per-parameter mean shift and width ratio. Figures follow each
step: cantuccio corner plots of the posteriors before and after the fold and
in the local chart, the Gaussianity effects, the de-biasing, the per-parameter
decomposition, and a distance map of the three posteriors. Run it as a script
with ``uv run --no-sync python examples/gb_sliced_w2.py``, or open it with
``uv run --no-sync marimo edit examples/gb_sliced_w2.py``.
"""

# ruff: noqa: N803, N806
# ? marimo passes every global between cells as a parameter, constants and classes too.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    import math

    import jax
    import jax.numpy as jnp
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import polars as pl
    from cantuccio import cornerplot, overlay_covariance

    from mixed_signals.core.result import Provenance, Result, hash_arrays
    from mixed_signals.core.samples import (
        SampleSet,
        circular_mean,
        circular_standard_deviation,
        unwrap_about,
    )
    from mixed_signals.core.space import ParamSpace
    from mixed_signals.core.topology import Topology, wrap
    from mixed_signals.diagnostics import CIRCLE_WIDTH_LIMIT, evaluate
    from mixed_signals.diagnostics.flags import THRESHOLDS
    from mixed_signals.diagnostics.modes import marginal_modes
    from mixed_signals.diagnostics.shape import (
        anderson_darling_normality,
        mardia_moments,
        marginal_moments,
    )
    from mixed_signals.errors import capture_warnings
    from mixed_signals.estimators import GaussianDensity
    from mixed_signals.metrics.moments import mean_shift, width_ratio
    from mixed_signals.metrics.ot.sliced import (
        sliced_directions,
        sliced_wasserstein_distance,
    )
    from mixed_signals.nulls.surrogates import gaussian_surrogates, surrogate_size
    from mixed_signals.prng import derive_key
    from mixed_signals.sigma.maps import r_eff, sigma_eff
    from mixed_signals.sigma.significance import empirical_p_value, sigma_sig
    from mixed_signals.transforms import Whitening

    return (
        CIRCLE_WIDTH_LIMIT,
        GaussianDensity,
        ParamSpace,
        Provenance,
        Result,
        SampleSet,
        THRESHOLDS,
        Topology,
        Whitening,
        anderson_darling_normality,
        capture_warnings,
        circular_mean,
        circular_standard_deviation,
        cornerplot,
        derive_key,
        empirical_p_value,
        evaluate,
        gaussian_surrogates,
        hash_arrays,
        jax,
        jnp,
        json,
        mardia_moments,
        marginal_modes,
        marginal_moments,
        math,
        mean_shift,
        mo,
        np,
        overlay_covariance,
        pl,
        plt,
        r_eff,
        sigma_eff,
        sigma_sig,
        sliced_directions,
        sliced_wasserstein_distance,
        surrogate_size,
        unwrap_about,
        width_ratio,
        wrap,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # Matched Galactic-binary posteriors under sliced optimal transport

    Three analyses, A, B and C, of the same Galactic binary give matched
    sample sets over $(A, f_0, \dot f, \phi_0, \iota, \psi, \alpha, \delta)$.
    The posteriors here are synthetic, with illustrative widths, so every
    number below has a known truth. B is a Gaussian shifted in $A$, $f_0$
    and $\alpha$, wider in $\dot f$ and narrower in $A$. C has the mean and
    covariance of A, except a wider $\iota$, but bends $\iota$ quadratically
    in $A$, a banana that no Gaussian fit describes. Part of each set sits in
    the degenerate image $(\phi_0 + \pi, \psi + \pi/2)$, which leaves the
    waveform unchanged.
    """)


@app.cell
def _(ParamSpace, math):
    SEED = 2026
    SAMPLE_COUNT = 5_000
    NULL_DRAWS = 200
    DIRECTION_COUNT = 500
    # * Halves carry four times the O(1/N) bias per side of the full set.
    SELF_TERM_FACTOR = 0.25
    SPLIT_COUNT = 4
    TRUTH_DIRECTION_COUNT = 100_000
    # * C bends iota by BANANA_STRENGTH · (Â² − 1) widths, with Â the standardised A.
    BANANA_STRENGTH = 0.4
    POSTERIOR_NAMES = ("analysis_a", "analysis_b", "analysis_c")
    # * Three categorical slots that colour-blind readers can still tell apart.
    POSTERIOR_COLOURS = ("#2a78d6", "#eb6834", "#1baf7a")
    PLOT_SAMPLE_COUNT = 2_000

    physical_space = ParamSpace(
        parameters=[
            {"name": "A", "label": "A", "unit": "1e-22"},
            {"name": "f0", "label": "f_0", "unit": "mHz"},
            {"name": "fdot", "label": r"\dot{f}", "unit": "1e-17 Hz/s"},
            {
                "name": "phi0",
                "label": r"\phi_0",
                "unit": "rad",
                "topology": "circle",
                "bounds": (0.0, 2 * math.pi),
            },
            {
                "name": "iota",
                "label": r"\iota",
                "unit": "rad",
                "topology": "interval",
                "bounds": (0.0, math.pi),
            },
            {
                "name": "psi",
                "label": r"\psi",
                "unit": "rad",
                "topology": "circle",
                "bounds": (0.0, math.pi),
            },
            {
                "name": "ra",
                "label": r"\alpha",
                "unit": "rad",
                "topology": "circle",
                "bounds": (0.0, 2 * math.pi),
            },
            {
                "name": "dec",
                "label": r"\delta",
                "unit": "rad",
                "topology": "interval",
                "bounds": (-math.pi / 2, math.pi / 2),
            },
        ],
        spheres=[{"azimuth": "ra", "polar": "dec", "polar_form": "latitude"}],
    )
    # * The waveform is unchanged under φ₀ + π together with ψ + π/2.
    DEGENERACY_SHIFTS = {"phi0": math.pi, "psi": math.pi / 2}
    return (
        BANANA_STRENGTH,
        DEGENERACY_SHIFTS,
        DIRECTION_COUNT,
        NULL_DRAWS,
        PLOT_SAMPLE_COUNT,
        POSTERIOR_COLOURS,
        POSTERIOR_NAMES,
        SAMPLE_COUNT,
        SEED,
        SELF_TERM_FACTOR,
        SPLIT_COUNT,
        TRUTH_DIRECTION_COUNT,
        physical_space,
    )


@app.cell
def _(
    BANANA_STRENGTH,
    DEGENERACY_SHIFTS,
    SAMPLE_COUNT,
    SEED,
    derive_key,
    jax,
    jnp,
    np,
    physical_space,
    wrap,
):
    root_key = jax.random.key(SEED)
    names = physical_space.names

    reference_mean = np.array([1.5, 3.0, 1.2, 6.1, 1.0, 0.08, 4.0, 0.3])
    reference_widths = np.array([0.15, 1e-6, 0.1, 0.25, 0.1, 0.12, 0.02, 0.02])
    _correlation = np.eye(len(names))
    for (_first, _second), _rho in {
        ("A", "iota"): -0.7,
        ("f0", "fdot"): 0.6,
        ("phi0", "psi"): 0.85,
        ("ra", "dec"): 0.3,
    }.items():
        _row, _column = names.index(_first), names.index(_second)
        _correlation[_row, _column] = _correlation[_column, _row] = _rho
    reference_covariance = np.outer(reference_widths, reference_widths) * _correlation

    # * Known differences of B, in units of the reference widths.
    shift_in_widths = np.zeros(len(names))
    shift_in_widths[[names.index("A"), names.index("f0"), names.index("ra")]] = (
        0.4,
        0.6,
        -0.3,
    )
    true_ratios = np.ones(len(names))
    true_ratios[[names.index("A"), names.index("fdot")]] = (0.85, 1.3)
    true_shift = shift_in_widths * reference_widths
    true_covariances = (
        reference_covariance,
        np.outer(true_ratios, true_ratios) * reference_covariance,
    )
    # * Var(b · (Â² − 1)) = 2b², so the bend widens iota by √(1 + 2b²), shifts nothing.
    true_banana_ratios = np.ones(len(names))
    true_banana_ratios[names.index("iota")] = np.sqrt(1 + 2 * BANANA_STRENGTH**2)
    _means = (reference_mean, reference_mean + true_shift, reference_mean)
    _covariances = (*true_covariances, reference_covariance)
    _image_fractions = (0.1, 0.35, 0.1)
    _bends = (0.0, 0.0, BANANA_STRENGTH)

    def draw_posterior(key, mean, covariance, image_fraction, bend):
        """Draw a posterior, bend iota in A, and move a fraction to its image."""
        normal_key, image_key = derive_key(key, "normal"), derive_key(key, "image")
        draws = jax.random.multivariate_normal(
            normal_key, jnp.asarray(mean), jnp.asarray(covariance), (SAMPLE_COUNT,)
        )
        amplitude, inclination = names.index("A"), names.index("iota")
        standardised = (draws[:, amplitude] - mean[amplitude]) / np.sqrt(
            covariance[amplitude, amplitude]
        )
        # * Â² − 1 has mean zero and no covariance with any Gaussian axis.
        draws = draws.at[:, inclination].add(
            bend * (standardised**2 - 1) * np.sqrt(covariance[inclination, inclination])
        )
        in_image = jax.random.uniform(image_key, (SAMPLE_COUNT,)) < image_fraction
        for _name, _offset in DEGENERACY_SHIFTS.items():
            _position = names.index(_name)
            draws = draws.at[:, _position].add(jnp.where(in_image, _offset, 0.0))
        for _position, _parameter in enumerate(physical_space.parameters):
            if _parameter.period is not None:
                draws = draws.at[:, _position].set(
                    wrap(draws[:, _position], _parameter.bounds[0], _parameter.period)
                )
        return draws

    raw_posteriors = tuple(
        draw_posterior(derive_key(root_key, "data", _index), *_inputs)
        for _index, _inputs in enumerate(
            zip(_means, _covariances, _image_fractions, _bends, strict=True)
        )
    )
    return (
        names,
        raw_posteriors,
        root_key,
        true_banana_ratios,
        true_covariances,
        true_ratios,
        true_shift,
    )


@app.cell
def _(PLOT_SAMPLE_COUNT, cornerplot, math, names, np, physical_space, plt):
    # * f0 is drawn as an offset in nHz, because its width is 1e-6 mHz.
    F0_REFERENCE = 3.0
    symbol_labels = {
        _parameter.name: rf"${_parameter.label}$"
        for _parameter in physical_space.parameters
    }
    axis_labels = {**symbol_labels, "f0": r"$f_0 - 3\,\mathrm{mHz}$ [nHz]"}
    # * Only φ₀ (near 2π) and ψ (near 0) wrap inside the range of their samples.
    physical_periodic = {
        axis_labels["phi0"]: (0.0, 2 * math.pi),
        axis_labels["psi"]: (0.0, math.pi),
    }
    legend_names = ("analysis A", "analysis B", "analysis C")
    INK = "#0b0b0b"
    MUTED_INK = "#898781"
    figure_style = {
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#e1e0d9",
        "grid.linewidth": 0.6,
        "legend.frameon": False,
    }

    def display_columns(samples, columns=None):
        """Return the plotted columns of a sample array, keyed by axis label."""
        # ? The draws are independent, so the first ones are a random subset.
        values = np.asarray(samples[:PLOT_SAMPLE_COUNT])
        columns = names if columns is None else columns
        displayed = {}
        for _name in columns:
            _column = values[:, names.index(_name)]
            if _name == "f0":
                _column = (_column - F0_REFERENCE) * 1e6
            displayed[axis_labels[_name]] = _column
        return displayed

    def posterior_corner(
        posteriors, legend, colours, title, columns=None, periodic=None
    ):
        """Return a corner plot of KDE contours at 68% and 90% of several posteriors."""
        figure, _axes = cornerplot(
            [display_columns(_samples, columns) for _samples in posteriors],
            labels=list(legend),
            colors=list(colours),
            offdiag_mode="kde",
            periodic=periodic,
            n_ticks=3,
            diagonal_ticks=True,
        )
        with plt.rc_context(figure_style):
            figure.suptitle(title, y=1.0)
        plt.close(figure)
        return figure

    return (
        INK,
        MUTED_INK,
        axis_labels,
        figure_style,
        legend_names,
        physical_periodic,
        posterior_corner,
        symbol_labels,
    )


@app.cell
def _(SampleSet, jnp, mo, raw_posteriors):
    # * Step 1: matched sample sets enter as one (K, N, D) container.
    sample_set = SampleSet(jnp.stack(raw_posteriors))
    effective_sizes = [float(_size) for _size in sample_set.effective_sample_size()[0]]
    mo.md(
        "**Step 1.** Kish effective sample sizes: "
        + ", ".join(f"{_size:.0f}" for _size in effective_sizes)
        + ". The draws are independent; MCMC chains would first be reduced with"
        " `mcmc.reduce_chains`, whose τ thinning sets `n_eff`."
    )
    return (effective_sizes,)


@app.cell
def _(
    POSTERIOR_COLOURS,
    legend_names,
    mo,
    physical_periodic,
    posterior_corner,
    raw_posteriors,
):
    raw_figure = posterior_corner(
        raw_posteriors,
        legend_names,
        POSTERIOR_COLOURS,
        "Posteriors as sampled, in physical coordinates",
        periodic=physical_periodic,
    )
    mo.vstack(
        [
            mo.md(
                "**Figure 1.** The three posteriors as they arrive, with KDE contours"
                " at 68% and 90%. The $(\\phi_0, \\psi)$ panel shows the second mode at"
                " $(\\phi_0 + \\pi, \\psi + \\pi/2)$, and the main mode of $\\phi_0$"
                " splits across $2\\pi$; the contours of $\\phi_0$ and $\\psi$ wrap"
                " around their periods."
            ),
            mo.as_html(raw_figure),
        ]
    )
    return (raw_figure,)


@app.cell
def _(
    DEGENERACY_SHIFTS,
    circular_mean,
    circular_standard_deviation,
    jnp,
    names,
    physical_space,
    raw_posteriors,
    unwrap_about,
    wrap,
):
    def circle_statistics(samples, statistic):
        """Return a circular statistic of every degenerate circle, in its units."""
        weights = jnp.zeros(samples.shape[0])
        values = []
        for _name in DEGENERACY_SHIFTS:
            _parameter = physical_space.parameters[names.index(_name)]
            _column = samples[:, names.index(_name)]
            if statistic is circular_mean:
                values.append(
                    circular_mean(
                        _column, weights, _parameter.period, _parameter.bounds[0]
                    )
                )
            else:
                values.append(statistic(_column, weights, _parameter.period))
        return jnp.stack(values)

    def fold_degeneracy(samples, centre, scales):
        """Map each sample to its degenerate image when that lies nearer the centre."""
        image = samples
        for _name, _offset in DEGENERACY_SHIFTS.items():
            _position = names.index(_name)
            _parameter = physical_space.parameters[_position]
            image = image.at[:, _position].set(
                wrap(
                    samples[:, _position] + _offset,
                    _parameter.bounds[0],
                    _parameter.period,
                )
            )

        def scaled_distance(points):
            # * Arcs in units of the reference widths, so φ₀ and ψ count alike.
            total = jnp.zeros(points.shape[0])
            for _index, _name in enumerate(DEGENERACY_SHIFTS):
                _parameter = physical_space.parameters[names.index(_name)]
                _arc = (
                    unwrap_about(
                        points[:, names.index(_name)],
                        centre[_index],
                        _parameter.period,
                    )
                    - centre[_index]
                )
                total = total + (_arc / scales[_index]) ** 2
            return total

        moved = scaled_distance(image) < scaled_distance(samples)
        return jnp.where(moved[:, None], image, samples), moved

    # ! The centre comes from analysis A, so every set folds onto the same mode.
    fold_centre = circle_statistics(raw_posteriors[0], circular_mean)
    fold_scales = circle_statistics(raw_posteriors[0], circular_standard_deviation)
    _folds = [
        fold_degeneracy(_samples, fold_centre, fold_scales)
        for _samples in raw_posteriors
    ]
    folded_posteriors = tuple(_fold[0] for _fold in _folds)
    moved_fractions = [float(jnp.mean(_fold[1])) for _fold in _folds]
    return fold_centre, folded_posteriors, moved_fractions


@app.cell
def _(
    POSTERIOR_NAMES,
    folded_posteriors,
    marginal_modes,
    mo,
    moved_fractions,
    names,
    physical_space,
    pl,
    raw_posteriors,
):
    # * Step 2b: modes along each marginal, before and after the fold.
    _rows = []
    for _name, _raw, _folded, _moved in zip(
        POSTERIOR_NAMES, raw_posteriors, folded_posteriors, moved_fractions, strict=True
    ):
        _before = marginal_modes(_raw, None, physical_space)
        _after = marginal_modes(_folded, None, physical_space)
        for _position, _parameter in enumerate(names):
            _rows.append(
                {
                    "posterior": _name,
                    "parameter": _parameter,
                    "modes_before": int(_before.modes[_position]),
                    "modes_after": int(_after.modes[_position]),
                    "dip_before": float(_before.dip[_position]),
                    "moved_fraction": _moved,
                }
            )
    mode_table = pl.DataFrame(_rows).filter(
        (pl.col("modes_before") > 1) | (pl.col("modes_after") > 1)
    )
    mo.vstack(
        [
            mo.md(
                "**Step 2b.** Marginals with more than one mode, before and after"
                " folding the $(\\phi_0 + \\pi, \\psi + \\pi/2)$ images onto the"
                " mode of analysis A:"
            ),
            mo.ui.table(mode_table),
        ]
    )
    return (mode_table,)


@app.cell
def _(
    DEGENERACY_SHIFTS,
    MUTED_INK,
    POSTERIOR_COLOURS,
    axis_labels,
    fold_centre,
    folded_posteriors,
    legend_names,
    mo,
    moved_fractions,
    names,
    physical_space,
    posterior_corner,
    raw_posteriors,
    unwrap_about,
):
    # * Axes centred halfway between the mode and its image, so that neither wraps.
    _windows = {
        _name: (
            names.index(_name),
            float(fold_centre[_index]) + _shift / 2,
            physical_space.parameters[names.index(_name)].period,
        )
        for _index, (_name, _shift) in enumerate(DEGENERACY_SHIFTS.items())
    }
    _periodic = {
        axis_labels[_name]: (_centre - _period / 2, _centre + _period / 2)
        for _name, (_position, _centre, _period) in _windows.items()
    }

    def _recentred(samples):
        for _position, _centre, _period in _windows.values():
            samples = samples.at[:, _position].set(
                unwrap_about(samples[:, _position], _centre, _period)
            )
        return samples

    fold_figures = [
        posterior_corner(
            [_recentred(_raw), _recentred(_folded)],
            ["as sampled", "folded"],
            [MUTED_INK, _colour],
            f"{_name}: {_moved:.1%} of the samples moved",
            columns=["phi0", "psi"],
            periodic=_periodic,
        )
        for _raw, _folded, _colour, _name, _moved in zip(
            raw_posteriors,
            folded_posteriors,
            POSTERIOR_COLOURS,
            legend_names,
            moved_fractions,
            strict=True,
        )
    ]
    mo.vstack(
        [
            mo.md(
                "**Figure 2.** The $(\\phi_0, \\psi)$ marginal before (grey) and after"
                " the fold. Each sample takes the image $(\\phi_0 + \\pi, \\psi +"
                " \\pi/2)$ when that lies nearer the circular mean of analysis A, so"
                " the second mode joins the first in all three posteriors. Each axis"
                " spans one period, centred halfway between the mode and its image,"
                " so neither wraps."
            ),
            mo.hstack([mo.as_html(_figure) for _figure in fold_figures]),
        ]
    )
    return (fold_figures,)


@app.cell
def _(
    CIRCLE_WIDTH_LIMIT,
    ParamSpace,
    Topology,
    circular_mean,
    circular_standard_deviation,
    folded_posteriors,
    jnp,
    math,
    mo,
    physical_space,
    unwrap_about,
):
    # * Local chart: every angle unwrapped about the circular mean of analysis A.
    _weights = jnp.zeros(folded_posteriors[0].shape[0])
    _chart_columns = tuple([] for _samples in folded_posteriors)
    wide_circles = []
    for _position, _parameter in enumerate(physical_space.parameters):
        if _parameter.topology is not Topology.CIRCLE:
            for _columns, _samples in zip(
                _chart_columns, folded_posteriors, strict=True
            ):
                _columns.append(_samples[:, _position])
            continue
        _centre = circular_mean(
            folded_posteriors[0][:, _position],
            _weights,
            _parameter.period,
            _parameter.bounds[0],
        )
        for _columns, _samples in zip(_chart_columns, folded_posteriors, strict=True):
            _width = circular_standard_deviation(
                _samples[:, _position], _weights, _parameter.period
            )
            if float(_width) * 2 * math.pi / _parameter.period > CIRCLE_WIDTH_LIMIT:
                wide_circles.append(_parameter.name)
            _columns.append(
                unwrap_about(_samples[:, _position], _centre, _parameter.period)
            )
    chart_posteriors = tuple(jnp.stack(_columns, axis=1) for _columns in _chart_columns)
    chart_space = ParamSpace(
        parameters=[
            {
                "name": _parameter.name,
                "label": _parameter.label,
                "unit": _parameter.unit,
            }
            for _parameter in physical_space.parameters
        ]
    )
    _status = (
        f"! wider than {CIRCLE_WIDTH_LIMIT} rad, the chart is not valid for: "
        + ", ".join(sorted(set(wide_circles)))
        if wide_circles
        else f"every circle is narrower than {CIRCLE_WIDTH_LIMIT} rad."
    )
    mo.md(
        "**Step 2a.** Local chart: φ₀, ψ and α unwrapped about the circular means"
        f" of analysis A, every axis a real line; {_status}"
    )
    return chart_posteriors, chart_space


@app.cell
def _(POSTERIOR_COLOURS, chart_posteriors, legend_names, mo, posterior_corner):
    chart_figure = posterior_corner(
        chart_posteriors,
        legend_names,
        POSTERIOR_COLOURS,
        "Folded posteriors in the local chart",
    )
    mo.vstack(
        [
            mo.md(
                "**Figure 3.** The posteriors that enter the comparison: folded, and"
                " with $\\phi_0$, $\\psi$ and $\\alpha$ unwrapped about the circular"
                " means of analysis A. B differs from A in $A$, $f_0$, $\\dot f$ and"
                " $\\alpha$; C bends $\\iota$ in $A$."
            ),
            mo.as_html(chart_figure),
        ]
    )
    return (chart_figure,)


@app.cell
def _(
    NULL_DRAWS,
    POSTERIOR_NAMES,
    Whitening,
    anderson_darling_normality,
    chart_posteriors,
    chart_space,
    derive_key,
    effective_sizes,
    evaluate,
    gaussian_surrogates,
    jax,
    mardia_moments,
    marginal_moments,
    mo,
    names,
    pl,
    root_key,
    surrogate_size,
):
    def gaussianity_rows(samples, posterior_name, effective_size):
        """Return the Gaussianity rows of one posterior, reference-whitened."""
        whitening = Whitening.from_samples(samples[None], None, chart_space)
        whitened = whitening.forward(samples)
        white_space = whitening.map_space(chart_space)
        key = derive_key(root_key, "gaussianity", posterior_name)
        surrogates = gaussian_surrogates(
            whitened,
            None,
            white_space,
            size=surrogate_size(effective_size),
            key=derive_key(key, "surrogates"),
            draws=NULL_DRAWS,
        )

        def statistics(points):
            moments = marginal_moments(points, None, white_space)
            mardia = mardia_moments(
                points, None, white_space, key=derive_key(key, "mardia", "subsample")
            )
            return {
                "skewness": moments.skewness,
                "kurtosis": moments.kurtosis,
                "anderson_darling": anderson_darling_normality(
                    points, None, white_space
                ),
                "mardia_skewness": mardia.skewness,
                "mardia_kurtosis": mardia.kurtosis,
            }

        observed = statistics(whitened)
        null = jax.lax.map(statistics, surrogates)
        rows = []
        for metric, values in observed.items():
            # * Mardia's moments are joint, the others one value per parameter.
            labels = names if values.ndim else ("joint",)
            for position, label in enumerate(labels):
                value = values[position] if values.ndim else values
                draws = null[metric][:, position] if values.ndim else null[metric]
                verdict = evaluate(metric, float(value), draws)
                rows.append(
                    {
                        "posterior": posterior_name,
                        "metric": metric,
                        "parameter": label,
                        "value": float(value),
                        "effect": verdict.effect,
                        "sigma_sig": verdict.sigma_sig,
                        "flagged": verdict.flagged,
                    }
                )
        return rows

    gaussianity_table = pl.DataFrame(
        [
            _row
            for _samples, _name, _size in zip(
                chart_posteriors, POSTERIOR_NAMES, effective_sizes, strict=True
            )
            for _row in gaussianity_rows(_samples, _name, _size)
        ]
    )
    _flag_counts = gaussianity_table.group_by("posterior", maintain_order=True).agg(
        pl.col("flagged").sum()
    )
    _summary = ", ".join(
        f"{_name} {_count}" for _name, _count in _flag_counts.iter_rows()
    )
    mo.vstack(
        [
            mo.md(
                "**Steps 2c and 2d.** Moments of each posterior after its own"
                f" reference whitening, against {NULL_DRAWS} Gaussian surrogates at"
                f" its n_eff. Flagged rows per posterior: {_summary}. A posterior"
                " with flags is not described by its Gaussian fit, so its analytic"
                " SW2² below is only a lower bound."
            ),
            mo.ui.table(gaussianity_table.filter(pl.col("flagged"))),
        ]
    )
    return (gaussianity_table,)


@app.cell
def _(
    GaussianDensity,
    INK,
    MUTED_INK,
    POSTERIOR_COLOURS,
    POSTERIOR_NAMES,
    THRESHOLDS,
    chart_posteriors,
    chart_space,
    figure_style,
    gaussianity_table,
    legend_names,
    mo,
    names,
    np,
    overlay_covariance,
    pl,
    plt,
    posterior_corner,
    symbol_labels,
):
    _metric_labels = {
        "skewness": "skewness",
        "kurtosis": "kurtosis",
        "anderson_darling": "Anderson-Darling",
        "mardia_skewness": r"Mardia $b_1$",
        "mardia_kurtosis": r"Mardia $b_2$",
    }
    _rows = (
        gaussianity_table.filter(pl.col("posterior") == POSTERIOR_NAMES[0])
        .select("metric", "parameter")
        .rows()
    )
    _positions = {_row: _index for _index, _row in enumerate(_rows)}
    with plt.rc_context(figure_style):
        gaussianity_figure, _axis = plt.subplots(figsize=(6.0, 0.24 * len(_rows) + 1.4))
        _axis.axvspan(-1, 1, color="#f0efec", zorder=0)
        for _offset, _name, _colour in zip(
            (-0.25, 0.0, 0.25), POSTERIOR_NAMES, POSTERIOR_COLOURS, strict=True
        ):
            _subset = gaussianity_table.filter(pl.col("posterior") == _name)
            _heights = _offset + np.array(
                [
                    _positions[_row]
                    for _row in _subset.select("metric", "parameter").rows()
                ]
            )
            # * One axis for every metric: the effect in units of its flag threshold.
            _ratios = np.array(
                [
                    _effect / THRESHOLDS[_metric].value
                    for _metric, _effect in _subset.select("metric", "effect").rows()
                ]
            )
            _flagged = _subset["flagged"].to_numpy()
            _axis.scatter(
                _ratios[~_flagged],
                _heights[~_flagged],
                facecolors="none",
                edgecolors=_colour,
                s=24,
                linewidths=1.2,
            )
            _axis.scatter(_ratios[_flagged], _heights[_flagged], color=_colour, s=34)
        for _side in (-1, 1):
            _axis.axvline(_side, color=MUTED_INK, linestyle="--", linewidth=1)
        # * Linear inside the thresholds and logarithmic beyond, to keep large effects.
        _axis.set_xscale("symlog", linthresh=1)
        _axis.margins(x=0.06)
        _ticks = [-10, -1, 0, 1, 10]
        _axis.set_xticks(_ticks, [str(_tick) for _tick in _ticks])
        _axis.set_yticks(
            range(len(_rows)),
            [
                _metric_labels[_metric]
                + ("" if _parameter == "joint" else f", {symbol_labels[_parameter]}")
                for _metric, _parameter in _rows
            ],
        )
        _axis.invert_yaxis()
        _axis.grid(axis="y", visible=False)
        _axis.set_xlabel("effect / flag threshold")
        _handles = [
            plt.Line2D(
                [], [], marker="o", linestyle="none", color=_colour, label=_label
            )
            for _colour, _label in zip(POSTERIOR_COLOURS, legend_names, strict=True)
        ] + [
            plt.Line2D(
                [],
                [],
                marker="o",
                linestyle="none",
                markerfacecolor="none",
                markeredgecolor=MUTED_INK,
                label="not flagged",
            ),
            plt.Line2D(
                [], [], marker="o", linestyle="none", color=MUTED_INK, label="flagged"
            ),
        ]
        _axis.legend(
            handles=_handles, loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=9
        )
        gaussianity_figure.tight_layout()
    plt.close(gaussianity_figure)

    # * C's Gaussian fit in the chart; the plotted axes leave out f0, so no unit change.
    _columns = ["A", "fdot", "iota"]
    _indices = [names.index(_column) for _column in _columns]
    _fit = GaussianDensity.fit(chart_posteriors[2], None, chart_space)
    banana_figure = posterior_corner(
        [chart_posteriors[2]],
        [legend_names[2]],
        [POSTERIOR_COLOURS[2]],
        "Analysis C against its Gaussian fit",
        columns=_columns,
    )
    overlay_covariance(
        banana_figure,
        np.asarray(_fit.covariance)[np.ix_(_indices, _indices)],
        means=np.asarray(_fit.mean)[_indices],
        levels=(0.68, 0.9),
        colors=INK,
        linestyles="--",
        linewidths=1.2,
        label="Gaussian fit",
    )
    mo.vstack(
        [
            mo.md(
                "**Figure 4.** Left: every Gaussianity row, its effect (value minus"
                " the surrogate null mean) in units of its flag threshold; the"
                " shaded band never flags, and a filled marker also reaches σ_sig ≥ 2."
                " Right: the bend of C in the $(A, \\iota)$ plane against the 68% and"
                " 90% contours of its Gaussian fit (dashed); $\\dot f$ is a control."
            ),
            mo.hstack(
                [mo.as_html(gaussianity_figure), mo.as_html(banana_figure)],
                align="center",
            ),
        ]
    )
    return banana_figure, gaussianity_figure


@app.cell
def _(
    DIRECTION_COUNT,
    GaussianDensity,
    NULL_DRAWS,
    SELF_TERM_FACTOR,
    SPLIT_COUNT,
    Whitening,
    capture_warnings,
    chart_space,
    derive_key,
    jax,
    jnp,
    mean_shift,
    np,
    root_key,
    sliced_directions,
    sliced_wasserstein_distance,
    width_ratio,
):
    def projected_gaussian_w2(
        mean_first, cov_first, mean_second, cov_second, unit_directions
    ):
        """Return SW2² between two Gaussians, exact per direction.

        Each direction θ projects the Gaussians to normals whose W2² is
        (θ·Δμ)² + (√(θᵀΣ₁θ) − √(θᵀΣ₂θ))² (Dowson and Landau 1982); the mean over
        directions is sliced W2 (Rabin et al. 2012). It bounds the SW2² of any
        two distributions with these moments from below (Gelbrich 1990).
        """
        shift = unit_directions @ (mean_second - mean_first)
        width_first = jnp.sqrt(
            jnp.einsum("ld,de,le->l", unit_directions, cov_first, unit_directions)
        )
        width_second = jnp.sqrt(
            jnp.einsum("ld,de,le->l", unit_directions, cov_second, unit_directions)
        )
        return jnp.mean(shift**2 + (width_first - width_second) ** 2)

    def sliced_estimators(white_space, directions):
        """Return the empirical and Gaussian SW2² on shared directions."""

        def empirical(left, right):
            return float(
                sliced_wasserstein_distance(
                    left, None, right, None, white_space, directions=directions
                )
            )

        def gaussian(left, right):
            left_fit = GaussianDensity.fit(left, None, white_space)
            right_fit = GaussianDensity.fit(right, None, white_space)
            return float(
                projected_gaussian_w2(
                    left_fit.mean,
                    left_fit.covariance,
                    right_fit.mean,
                    right_fit.covariance,
                    directions,
                )
            )

        return {"empirical": empirical, "gaussian": gaussian}

    def self_term_correction(statistic, white, key):
        """Return ¼ of the mean self-terms of both posteriors over random splits."""
        total = 0.0
        for split in range(SPLIT_COUNT):
            for label, samples in zip(("first", "second"), white, strict=True):
                order = jax.random.permutation(
                    derive_key(key, "self_term", label, split), samples.shape[0]
                )
                half = samples.shape[0] // 2
                total += statistic(samples[order[:half]], samples[order[half:]])
        return SELF_TERM_FACTOR * total / SPLIT_COUNT

    def compare_pair(first, second, pair):
        """Run steps 3 and 4 for one pair of posteriors in chart coordinates."""
        key = derive_key(root_key, "compare", *pair)
        with capture_warnings() as records:
            # * Step 3: one pooled whitening, the equal-weight mean of both covariances.
            stacked = jnp.stack([first, second])
            whitening = Whitening.from_samples(stacked, None, chart_space)
            zscore = Whitening.from_samples(
                stacked, None, chart_space, decorrelate=False
            )
            white_space = whitening.map_space(chart_space)
            zscore_space = zscore.map_space(chart_space)
            directions = sliced_directions(
                white_space, DIRECTION_COUNT, key=derive_key(key, "directions")
            )
            estimators = sliced_estimators(white_space, directions)
            white = (whitening.forward(first), whitening.forward(second))
            zscored = (zscore.forward(first), zscore.forward(second))

            # * Step 4: observed values, self-terms and one permutation null.
            raw = {name: statistic(*white) for name, statistic in estimators.items()}
            corrections = {
                name: self_term_correction(statistic, white, key)
                for name, statistic in estimators.items()
            }
            shifts = np.asarray(
                mean_shift(zscored[0], None, zscored[1], None, zscore_space)
            )
            ratios = np.asarray(width_ratio(first, None, second, None, chart_space))
            first_count = first.shape[0]
            pooled_white = jnp.concatenate(white)
            pooled_zscore = jnp.concatenate(zscored)
            pooled_chart = jnp.concatenate([first, second])
            null = {name: [] for name in estimators}
            null_shifts, null_ratios = [], []
            for draw in range(NULL_DRAWS):
                order = jax.random.permutation(
                    derive_key(key, "null", draw), pooled_white.shape[0]
                )
                left, right = order[:first_count], order[first_count:]
                for name, statistic in estimators.items():
                    null[name].append(
                        statistic(pooled_white[left], pooled_white[right])
                    )
                null_shifts.append(
                    mean_shift(
                        pooled_zscore[left],
                        None,
                        pooled_zscore[right],
                        None,
                        zscore_space,
                    )
                )
                null_ratios.append(
                    width_ratio(
                        pooled_chart[left],
                        None,
                        pooled_chart[right],
                        None,
                        chart_space,
                    )
                )
        return {
            "pair": pair,
            "key": key,
            "whitening": whitening,
            "zscore": zscore,
            "white_space": white_space,
            "directions": directions,
            "raw": raw,
            "corrections": corrections,
            "null": {name: np.asarray(draws) for name, draws in null.items()},
            "shifts": shifts,
            "ratios": ratios,
            "null_shifts": np.asarray(null_shifts),
            "null_ratios": np.asarray(null_ratios),
            "sample_size": first_count,
            "warnings": records,
        }

    return (
        compare_pair,
        projected_gaussian_w2,
        self_term_correction,
        sliced_estimators,
    )


@app.cell
def _(POSTERIOR_NAMES, chart_posteriors, compare_pair):
    # * A against B, the comparison with truths, and A against C, the banana.
    comparisons = [
        compare_pair(
            chart_posteriors[0],
            chart_posteriors[_index],
            (POSTERIOR_NAMES[0], POSTERIOR_NAMES[_index]),
        )
        for _index in (1, 2)
    ]
    return (comparisons,)


@app.cell
def _(
    DIRECTION_COUNT,
    NULL_DRAWS,
    POSTERIOR_NAMES,
    Provenance,
    Result,
    SEED,
    SELF_TERM_FACTOR,
    SPLIT_COUNT,
    chart_posteriors,
    comparisons,
    empirical_p_value,
    hash_arrays,
    json,
    names,
    np,
    r_eff,
    sigma_eff,
    sigma_sig,
):
    _hashes = dict(
        zip(
            POSTERIOR_NAMES,
            (hash_arrays(_samples) for _samples in chart_posteriors),
            strict=True,
        )
    )

    def comparison_row(
        problem,
        comparison,
        metric,
        estimator,
        settings,
        space,
        parameters,
        value,
        draws,
        tail,
    ):
        """Return one result row with its null, p-value and σ_sig."""
        p_value = float(empirical_p_value(value, draws, tail))
        return {
            "problem": problem,
            "pair": list(comparison["pair"]),
            "metric": metric,
            "estimator": estimator,
            "estimator_settings": json.dumps(settings),
            "space": space,
            "parameters": list(parameters),
            "value": float(value),
            "uncertainty": float(np.std(draws)),
            "uncertainty_method": "null_std",
            "null_mean": float(np.mean(draws)),
            "null_std": float(np.std(draws)),
            "null_draws": [float(_draw) for _draw in draws],
            "null_recipe": {
                "construction": "permutation",
                "draws": NULL_DRAWS,
                "sample_size": comparison["sample_size"],
            },
            "p_value": p_value,
            "sigma_sig": float(sigma_sig(p_value, tail)),
            "warnings": comparison["warnings"],
            "provenance": Provenance.collect(
                seed=SEED,
                input_hashes=[_hashes[_name] for _name in comparison["pair"]],
            ),
        }

    _rows = []
    for _problem, _comparison in enumerate(comparisons):
        for _estimator, _raw in _comparison["raw"].items():
            _correction = _comparison["corrections"][_estimator]
            # * Shifting the draws by the observed correction keeps the raw p-value.
            _value = _raw - _correction
            _row = comparison_row(
                _problem,
                _comparison,
                "sliced_w2",
                _estimator,
                {
                    "directions": DIRECTION_COUNT,
                    "self_term_factor": SELF_TERM_FACTOR,
                    "splits": SPLIT_COUNT,
                },
                _comparison["whitening"].description,
                names,
                _value,
                _comparison["null"][_estimator] - _correction,
                "upper",
            )
            # * All axes are real lines in the chart, so n = E + 2C + 3S = D.
            _row["sigma_eff"] = float(
                sigma_eff("sliced_w2", _value, dimension=len(names))
            )
            _rows.append(_row)
        for _position, _name in enumerate(names):
            _shift = _comparison["shifts"][_position]
            _row = comparison_row(
                _problem,
                _comparison,
                "mean_shift",
                "empirical",
                {},
                _comparison["zscore"].description,
                [_name],
                _shift,
                _comparison["null_shifts"][:, _position],
                "two_sided",
            )
            _row["sigma_eff"] = float(sigma_eff("mean_shift", _shift))
            _rows.append(_row)
            _ratio = _comparison["ratios"][_position]
            _row = comparison_row(
                _problem,
                _comparison,
                "width_ratio",
                "empirical",
                {},
                "local chart",
                [_name],
                _ratio,
                _comparison["null_ratios"][:, _position],
                "two_sided",
            )
            _row["r_eff"] = float(r_eff("width_ratio", _ratio))
            _rows.append(_row)
    comparison_result = Result.from_rows(_rows)
    return (comparison_result,)


@app.cell
def _(
    TRUTH_DIRECTION_COUNT,
    comparisons,
    derive_key,
    jnp,
    names,
    np,
    pl,
    projected_gaussian_w2,
    sigma_eff,
    sliced_directions,
    true_covariances,
    true_ratios,
    true_shift,
):
    # * Truth of A against B: ZCA-cor of the true pooled covariance.
    _comparison = comparisons[0]
    _pooled = (true_covariances[0] + true_covariances[1]) / 2
    _scales = np.sqrt(np.diag(_pooled))
    _eigenvalues, _eigenvectors = np.linalg.eigh(_pooled / np.outer(_scales, _scales))
    _matrix = (_eigenvectors / np.sqrt(_eigenvalues)) @ _eigenvectors.T / _scales
    _white_shift = jnp.asarray(_matrix @ true_shift)
    _white_covariances = [
        jnp.asarray(_matrix @ _cov @ _matrix.T) for _cov in true_covariances
    ]
    _zero = jnp.zeros(len(names))
    _dense = sliced_directions(
        _comparison["white_space"],
        TRUTH_DIRECTION_COUNT,
        key=derive_key(_comparison["key"], "truth"),
    )

    def truth_on(unit_directions):
        return float(
            projected_gaussian_w2(
                _zero,
                _white_covariances[0],
                _white_shift,
                _white_covariances[1],
                unit_directions,
            )
        )

    truth_sphere = truth_on(_dense)
    _raw, _corrections = _comparison["raw"], _comparison["corrections"]
    _values = {
        "truth": truth_on(_comparison["directions"]),
        "raw_empirical": _raw["empirical"],
        "debiased_empirical": _raw["empirical"] - _corrections["empirical"],
        "null_subtracted_empirical": _raw["empirical"]
        - float(np.mean(_comparison["null"]["empirical"])),
        "debiased_gaussian": _raw["gaussian"] - _corrections["gaussian"],
    }
    _sigmas = {
        _column: float(
            sigma_eff(
                "sliced_w2",
                truth_sphere if _column == "truth" else _value,
                dimension=len(names),
            )
        )
        for _column, _value in _values.items()
    }
    truth_table = pl.DataFrame(
        [
            {"quantity": "SW2², shared directions", **_values},
            {"quantity": "σ_eff from SW2²", **_sigmas},
        ]
    )
    true_parameter_table = pl.DataFrame(
        {
            "parameter": list(names),
            "true_mean_shift": true_shift / np.sqrt(np.diag(_pooled)),
            "true_width_ratio": true_ratios,
        }
    )
    return true_parameter_table, truth_sphere, truth_table


@app.cell
def _(INK, MUTED_INK, comparisons, figure_style, mo, np, plt, truth_table):
    _estimator_labels = ("empirical", "Gaussian")
    with plt.rc_context(figure_style):
        debias_figure, (_null_axis, *_value_axes) = plt.subplots(
            1, 3, figsize=(12.0, 3.8), gridspec_kw={"width_ratios": (1.2, 1.0, 1.0)}
        )
        # * Permuted A and B are identical in distribution, so their SW2² is pure bias.
        _draws = comparisons[0]["null"]["empirical"]
        _correction = comparisons[0]["corrections"]["empirical"]
        _bins = np.linspace((_draws - _correction).min(), _draws.max(), 40)
        _null_axis.hist(_draws, _bins, color=MUTED_INK, alpha=0.5)
        _null_axis.hist(_draws - _correction, _bins, histtype="step", color=INK)
        _null_axis.axvline(0.0, color=INK, linestyle="--", linewidth=1)
        _null_axis.set_xlabel(r"$\mathrm{SW}_2^2$ of permuted A and B")
        _null_axis.set_ylabel("null draws")
        _null_axis.set_title("(a) identical inputs, empirical", loc="left")
        # ! The null of A against B lies far below its values; only panel (c) draws it.
        for _axis, _comparison, _title, _show_null in zip(
            _value_axes,
            comparisons,
            ("(b) A against B", "(c) A against C"),
            (False, True),
            strict=True,
        ):
            for _row, _estimator in enumerate(("empirical", "gaussian")):
                _raw = _comparison["raw"][_estimator]
                _corrected = _raw - _comparison["corrections"][_estimator]
                if _show_null:
                    _null = _comparison["null"][_estimator]
                    _null = _null - _comparison["corrections"][_estimator]
                    _axis.plot(
                        np.quantile(_null, [0.025, 0.975]),
                        [_row, _row],
                        color=MUTED_INK,
                        alpha=0.45,
                        linewidth=9,
                        solid_capstyle="butt",
                    )
                _axis.annotate(
                    "",
                    xy=(_corrected, _row),
                    xytext=(_raw, _row),
                    arrowprops={"arrowstyle": "->", "color": INK, "shrinkB": 4},
                )
                _axis.scatter(_raw, _row, facecolors="none", edgecolors=INK, s=44)
                _axis.scatter(_corrected, _row, color=INK, s=44, zorder=3)
            _subtracted = _comparison["raw"]["empirical"] - np.mean(
                _comparison["null"]["empirical"]
            )
            _axis.scatter(_subtracted, 0, marker="d", color=MUTED_INK, s=44, zorder=4)
            if not _show_null:
                _axis.axvline(
                    truth_table["truth"][0], color=INK, linestyle="--", linewidth=1
                )
            _axis.set_yticks([0, 1], _estimator_labels)
            _axis.set_ylim(1.6, -0.6)
            _axis.grid(axis="y", visible=False)
            _axis.set_xlabel(r"$\mathrm{SW}_2^2$")
            _axis.set_title(_title, loc="left")
        for _axis in (_null_axis, *_value_axes):
            _axis.ticklabel_format(
                axis="x", style="sci", scilimits=(-3, -3), useMathText=True
            )
        _handles = [
            plt.Rectangle((0, 0), 1, 1, color=MUTED_INK, alpha=0.5, label="raw null"),
            plt.Line2D([], [], color=INK, label="de-biased null"),
            plt.Line2D(
                [],
                [],
                marker="o",
                linestyle="none",
                markerfacecolor="none",
                markeredgecolor=INK,
                label="raw",
            ),
            plt.Line2D(
                [],
                [],
                marker="o",
                linestyle="none",
                color=INK,
                label=r"de-biased, $\frac{1}{4}$ self-terms",
            ),
            plt.Line2D(
                [],
                [],
                marker="d",
                linestyle="none",
                color=MUTED_INK,
                label="raw minus null mean",
            ),
            plt.Line2D(
                [], [], color=MUTED_INK, alpha=0.45, linewidth=9, label="null, 95%"
            ),
            plt.Line2D([], [], color=INK, linestyle="--", label="truth"),
        ]
        debias_figure.legend(
            handles=_handles, loc="lower center", ncol=7, bbox_to_anchor=(0.5, -0.06)
        )
        debias_figure.tight_layout(rect=(0, 0.06, 1, 1))
    plt.close(debias_figure)
    mo.vstack(
        [
            mo.md(
                "**Figure 5.** De-biasing. (a) On permuted, hence identical, inputs"
                " the raw SW2² sits at its finite-sample bias; subtracting a quarter"
                " of the self-terms of the observed pair moves the null onto zero."
                " (b) and (c): raw (open) and de-biased (filled) values of both"
                " estimators. For A against B the de-biased values lie within the"
                " sampling scatter of the truth on the same directions; for A"
                " against C both lie well above the de-biased null, and the"
                " empirical estimator sees three times what the Gaussian one sees."
            ),
            mo.as_html(debias_figure),
        ]
    )
    return (debias_figure,)


@app.cell
def _(
    INK,
    MUTED_INK,
    POSTERIOR_COLOURS,
    comparisons,
    figure_style,
    mo,
    names,
    np,
    plt,
    symbol_labels,
    true_banana_ratios,
    true_parameter_table,
):
    _truths = (
        (
            true_parameter_table["true_mean_shift"].to_numpy(),
            true_parameter_table["true_width_ratio"].to_numpy(),
        ),
        (np.zeros(len(names)), true_banana_ratios),
    )
    _positions = np.arange(len(names))
    with plt.rc_context(figure_style):
        decomposition_figure, (_shift_axis, _ratio_axis) = plt.subplots(
            2, 1, figsize=(8.0, 6.0), sharex=True
        )
        for _offset, _comparison, _colour, _truth in zip(
            (-0.15, 0.15), comparisons, POSTERIOR_COLOURS[1:], _truths, strict=True
        ):
            for _axis, _observed, _null, _true in (
                (
                    _shift_axis,
                    _comparison["shifts"],
                    _comparison["null_shifts"],
                    _truth[0],
                ),
                (
                    _ratio_axis,
                    _comparison["ratios"],
                    _comparison["null_ratios"],
                    _truth[1],
                ),
            ):
                _low, _high = np.quantile(_null, [0.025, 0.975], axis=0)
                _axis.vlines(
                    _positions + _offset,
                    _low,
                    _high,
                    color=MUTED_INK,
                    alpha=0.45,
                    linewidth=7,
                )
                _axis.scatter(
                    _positions + _offset,
                    _true,
                    marker="_",
                    color=INK,
                    s=180,
                    linewidths=1.5,
                    zorder=3,
                )
                _axis.scatter(
                    _positions + _offset, _observed, color=_colour, s=30, zorder=4
                )
        _shift_axis.axhline(0.0, color=MUTED_INK, linewidth=0.8)
        _ratio_axis.axhline(1.0, color=MUTED_INK, linewidth=0.8)
        _shift_axis.set_ylabel(r"mean shift [pooled $\sigma$]")
        _ratio_axis.set_ylabel("width ratio")
        _ratio_axis.set_xticks(_positions, [symbol_labels[_name] for _name in names])
        _ratio_axis.grid(axis="x", visible=False)
        _shift_axis.grid(axis="x", visible=False)
        _handles = [
            plt.Line2D(
                [], [], marker="o", linestyle="none", color=_colour, label=_label
            )
            for _colour, _label in zip(
                POSTERIOR_COLOURS[1:], ("A against B", "A against C"), strict=True
            )
        ] + [
            plt.Line2D(
                [], [], color=MUTED_INK, alpha=0.45, linewidth=7, label="null, 95%"
            ),
            plt.Line2D(
                [],
                [],
                marker="_",
                linestyle="none",
                color=INK,
                markersize=12,
                label="truth",
            ),
        ]
        _shift_axis.legend(handles=_handles, loc="upper right", ncol=4, fontsize=9)
        decomposition_figure.tight_layout()
    plt.close(decomposition_figure)
    mo.vstack(
        [
            mo.md(
                "**Figure 6.** Per-parameter decomposition: the mean shift in pooled"
                " standard deviations and the width ratio of the second posterior to"
                " A, against their permutation nulls and truths. B recovers its"
                " shifts in $A$, $f_0$ and $\\alpha$ and its widths in $A$ and"
                " $\\dot f$; C differs only in the width of $\\iota$."
            ),
            mo.as_html(decomposition_figure),
        ]
    )
    return (decomposition_figure,)


@app.cell
def _(comparison_result, mo, pl, true_parameter_table, truth_sphere, truth_table):
    # * Step 5: summary of the comparison rows against their truths.
    _columns = [
        "problem",
        "metric",
        "estimator",
        "parameters",
        "value",
        "null_mean",
        "null_std",
        "sigma_sig",
        "sigma_eff",
        "r_eff",
    ]
    summary_table = comparison_result.table.select(_columns).with_columns(
        pl.col("parameters").list.join(", ")
    )
    _per_parameter = (
        summary_table.filter(
            (pl.col("problem") == 0) & (pl.col("metric") != "sliced_w2")
        )
        .pivot(on="metric", index="parameters", values="value")
        .rename({"parameters": "parameter"})
        .join(true_parameter_table, on="parameter")
    )
    banana_table = summary_table.filter(
        (pl.col("metric") == "sliced_w2")
        | ((pl.col("problem") == 1) & (pl.col("parameters") == "iota"))
    ).drop("parameters")
    print(summary_table)
    print(truth_table)
    print(_per_parameter)
    print(banana_table)
    mo.vstack(
        [
            mo.md(
                "**Step 5.** Comparison rows; problem 0 is A against B and problem"
                " 1 A against C. σ_sig saturates at 2.58 with 200 draws until the"
                " fitted tail of the null exists."
            ),
            mo.ui.table(summary_table),
            mo.md(
                "A against B: sliced W2 against its truth. The σ_eff truth uses the"
                f" whole sphere of directions (SW2² = {truth_sphere:.5f}), the SW2²"
                " truth the shared 500 directions."
            ),
            mo.ui.table(truth_table),
            mo.md("A against B: mean shift (pooled σ) and width ratio, with truths."),
            mo.ui.table(_per_parameter),
            mo.md(
                "Sliced W2 of both pairs, and the iota rows of A against C. For C the"
                " Gaussian form sees only the wider iota, while the empirical SW2²"
                " also sees the bend."
            ),
            mo.ui.table(banana_table),
        ]
    )
    return (summary_table,)


@app.cell
def _(
    DIRECTION_COUNT,
    POSTERIOR_NAMES,
    Whitening,
    chart_posteriors,
    chart_space,
    derive_key,
    jnp,
    names,
    root_key,
    self_term_correction,
    sigma_eff,
    sliced_directions,
    sliced_estimators,
):
    # * One whitening pooled over all three posteriors, so all distances share a metric.
    _key = derive_key(root_key, "distance_map")
    _whitening = Whitening.from_samples(jnp.stack(chart_posteriors), None, chart_space)
    _white_space = _whitening.map_space(chart_space)
    _estimators = sliced_estimators(
        _white_space,
        sliced_directions(
            _white_space, DIRECTION_COUNT, key=derive_key(_key, "directions")
        ),
    )
    map_posteriors = tuple(
        _whitening.forward(_samples) for _samples in chart_posteriors
    )
    map_sigmas = {}
    for _pair in ((0, 1), (0, 2), (1, 2)):
        _white = tuple(map_posteriors[_index] for _index in _pair)
        _pair_key = derive_key(_key, *(POSTERIOR_NAMES[_index] for _index in _pair))
        map_sigmas[_pair] = {
            _name: float(
                sigma_eff(
                    "sliced_w2",
                    _statistic(*_white)
                    - self_term_correction(_statistic, _white, _pair_key),
                    dimension=len(names),
                )
            )
            for _name, _statistic in _estimators.items()
        }
    return map_posteriors, map_sigmas


@app.cell
def _(
    INK,
    MUTED_INK,
    POSTERIOR_COLOURS,
    figure_style,
    legend_names,
    map_posteriors,
    map_sigmas,
    mo,
    names,
    np,
    plt,
):
    # * Three distances fix a triangle: A at 0, B on the x axis, C by the cosine rule.
    _first_side, _second_side, _third_side = (
        map_sigmas[_pair]["empirical"] for _pair in ((0, 1), (0, 2), (1, 2))
    )
    _along = (_first_side**2 + _second_side**2 - _third_side**2) / (2 * _first_side)
    # ! A triple that breaks the triangle inequality would put C on the line AB.
    _vertices = np.array(
        [
            [0.0, 0.0],
            [_first_side, 0.0],
            [_along, np.sqrt(max(_second_side**2 - _along**2, 0.0))],
        ]
    )
    _centroid = _vertices.mean(axis=0)
    _scale = max(_first_side, _second_side, _third_side)
    _inset_size = 0.45 * _scale
    _columns = [names.index("A"), names.index("iota")]
    _inset_points = np.concatenate(
        [np.asarray(_samples[:, _columns]) for _samples in map_posteriors]
    )
    _inset_limits = np.quantile(_inset_points, [0.002, 0.998], axis=0)
    with plt.rc_context(figure_style):
        distance_map_figure, _axis = plt.subplots(figsize=(9.0, 6.5))
        for _pair, _values in map_sigmas.items():
            _start, _end = _vertices[_pair[0]], _vertices[_pair[1]]
            _axis.annotate(
                "",
                xy=_end,
                xytext=_start,
                arrowprops={
                    "arrowstyle": "<->",
                    "color": INK,
                    "linewidth": 1.2,
                    "shrinkA": 7,
                    "shrinkB": 7,
                },
            )
            # * Each label sits beside its edge, on the side away from the third vertex.
            _middle = (_start + _end) / 2
            _normal = np.array([_start[1] - _end[1], _end[0] - _start[0]])
            _normal = _normal / np.linalg.norm(_normal)
            if _normal @ (_middle - _vertices[3 - sum(_pair)]) < 0:
                _normal = -_normal
            _axis.text(
                *(_middle + 0.04 * _scale * _normal),
                rf"$\sigma_\mathrm{{eff}}$ = {_values['empirical']:.2f}"
                f"\nGaussian {_values['gaussian']:.2f}",
                ha="right"
                if _normal[0] < -0.5
                else "left"
                if _normal[0] > 0.5
                else "center",
                va="top"
                if _normal[1] < -0.5
                else "bottom"
                if _normal[1] > 0.5
                else "center",
                fontsize=10,
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.5},
            )
        _corners = [_vertices]
        for _index, (_vertex, _samples) in enumerate(
            zip(_vertices, map_posteriors, strict=True)
        ):
            _outward = _vertex - _centroid
            _centre = _vertex + 0.8 * _scale * _outward / np.linalg.norm(_outward)
            _box = (*(_centre - _inset_size / 2), _inset_size, _inset_size)
            _corners.append(np.array([_box[:2], _centre + _inset_size / 2]))
            _axis.plot(*np.stack([_vertex, _centre]).T, color=MUTED_INK, linewidth=0.8)
            _axis.scatter(*_vertex, color=POSTERIOR_COLOURS[_index], s=70, zorder=4)
            _inset = _axis.inset_axes(_box, transform=_axis.transData)
            _inset.scatter(
                *np.asarray(_samples[:1_500, _columns]).T,
                s=1,
                alpha=0.3,
                color=POSTERIOR_COLOURS[_index],
                rasterized=True,
            )
            _inset.set_xlim(_inset_limits[:, 0])
            _inset.set_ylim(_inset_limits[:, 1])
            _inset.set_xticks([])
            _inset.set_yticks([])
            _inset.set_xlabel(r"$A$", fontsize=10, labelpad=2)
            _inset.set_ylabel(r"$\iota$", fontsize=10, labelpad=2)
            _inset.set_title(legend_names[_index], fontsize=10)
        _extent = np.concatenate(_corners)
        _padding = 0.08 * _scale
        # * A scale bar of 0.5 σ_eff below the map, since its axes carry no meaning.
        _bar_end = np.array([_extent[:, 0].max(), _extent[:, 1].min() - _padding])
        _axis.plot([_bar_end[0] - 0.5, _bar_end[0]], [_bar_end[1]] * 2, color=INK)
        _axis.text(
            _bar_end[0] - 0.25,
            _bar_end[1] - 0.02 * _scale,
            r"$0.5\,\sigma_\mathrm{eff}$",
            ha="center",
            va="top",
            fontsize=10,
        )
        _axis.set_xlim(_extent[:, 0].min() - _padding, _extent[:, 0].max() + _padding)
        _axis.set_ylim(_bar_end[1] - 2 * _padding, _extent[:, 1].max() + 2 * _padding)
        _axis.set_aspect("equal")
        _axis.axis("off")
    plt.close(distance_map_figure)
    mo.vstack(
        [
            mo.md(
                "**Figure 7.** Distance map of the three posteriors. Each edge is"
                " σ_eff = √(n · SW2²) of the de-biased empirical estimator, with"
                " n = 8, and the Gaussian estimator follows in brackets. All three"
                " come from one whitening pooled over A, B and C, so they differ"
                " slightly from the pairwise rows above. Three distances fix a"
                " triangle up to rotation and reflection, so the edge lengths are"
                " exact. Insets: $A$ against $\\iota$ in the common whitening."
            ),
            mo.as_html(distance_map_figure),
        ]
    )
    return (distance_map_figure,)


if __name__ == "__main__":
    app.run()
