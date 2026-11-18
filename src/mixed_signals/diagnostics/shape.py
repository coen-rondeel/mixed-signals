"""Gaussianity of one posterior: marginal moments, Anderson-Darling, Mardia's moments.

Every statistic here measures how far one posterior departs from a Gaussian,
and is compared with the same statistic of Gaussian surrogates
(:mod:`mixed_signals.nulls.surrogates`), so its null carries the
finite-sample bias of the estimator. The functions take the weighted samples
of one posterior and work under ``jax.jit`` and ``jax.lax.map`` over
surrogate sets.

Marginal moments
    :func:`marginal_moments` gives the skewness g₁ = m₃/m₂^(3/2) and the
    excess kurtosis g₂ = m₄/m₂² − 3 of each parameter, the moment ratios of
    the weighted empirical distribution, with mᵣ = Σ wᵢ(xᵢ − x̄)ʳ for weights
    summing to one (Rimoldini 2014); for equal weights they are the g forms
    of Joanes and Gill (1998), which a Gaussian sets to zero. On a circle it
    gives the circular skewness b̄₂/(1 − R̄)^(3/2) and kurtosis
    (ā₂ − R̄⁴)/(1 − R̄)², from the second trigonometric moment about the mean
    direction (Fisher 1993; Mardia and Jupp 2000), which a wrapped normal
    sets to zero; no rotation changes them. On a narrow circle they tend to
    −2√2·g₁ and 2·g₂ of the deviations from the mean direction. The kurtosis
    is written in s = sin(δ/2) of the deviations δ from the mean direction,
    (8Σws⁴ − 6V² + 4V³ − V⁴)/V² with V = 1 − R̄ = 2Σws², so a narrow circle
    keeps its digits. :func:`moment_metric_names` names the rows of each
    parameter.
Anderson-Darling
    :func:`anderson_darling_normality` gives A²/n, the Anderson-Darling
    distance between the weighted empirical distribution of each parameter
    and the normal with its weighted mean and unbiased variance (Anderson
    and Darling 1954; Stephens 1974, the case of an estimated mean and
    variance). The integral of (F_n − Φ)²/(Φ(1 − Φ)) over the steps of F_n
    is summed in closed form, with logarithms of Φ from both tails, so
    weights enter exactly and equal weights give the usual A²/n. A circle is
    cut opposite its circular mean, as the two-sample AD of
    :mod:`mixed_signals.metrics.marginal` cuts it.
Mardia's moments
    :func:`mardia_moments` gives the multivariate skewness
    b₁ = Σᵢⱼ wᵢwⱼ (zᵢᵀzⱼ)³ and kurtosis b₂ = Σᵢ wᵢ (zᵢᵀzᵢ)², with zᵢ the samples
    whitened by the weighted plug-in covariance (Mardia 1970). For a
    Gaussian b₁ tends to zero and b₂ to D(D + 2); for independent axes
    b₁ = Σ g₁² and b₂ − D(D + 2) = Σ g₂, so they read as the marginal
    moments do. b₂ is the mean of the squared Mahalanobis distances squared,
    which for a Gaussian follow χ²_D (Mahalanobis 1936; Johnson and Wichern
    2007). b₁ costs O(N²D), so both run on at most
    :data:`~mixed_signals._config.SAMPLE_CAP` samples. Circles are unwrapped
    about their circular means, and sphere axes follow their own topologies.
Wide circles
    Linear statistics of a circle unwrapped about its circular mean keep a
    calibrated surrogate null up to a circular standard deviation of
    :data:`~mixed_signals.diagnostics._common.CIRCLE_WIDTH_LIMIT` rad; wider
    circles raise a :class:`~mixed_signals.errors.ValidityWarning` for the
    Anderson-Darling and Mardia rows. The circular moments need no limit (see
    :ref:`validation-circle-drift`).
Validity
    A parameter without spread gives NaN, as does a singular covariance for
    Mardia's moments, with a :class:`~mixed_signals.errors.ValidityWarning`.

References:
    :cite:t:`joanes1998comparing`; :cite:t:`rimoldini2014weighted`;
    :cite:t:`mardia2000directional`; :cite:t:`fisher1993statistical`;
    :cite:t:`anderson1954test`; :cite:t:`stephens1974edf`;
    :cite:t:`mardia1970measures`; :cite:t:`mahalanobis1936generalised`;
    :cite:t:`johnson2007applied`; :cite:t:`efron1993introduction`.
"""

import functools
import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.scipy.special import log_ndtr
from jax.typing import ArrayLike
from jaxtyping import Array, Float, PRNGKeyArray

from mixed_signals._config import SAMPLE_CAP
from mixed_signals.core.samples import (
    _is_concrete,
    circular_mean,
    unwrap_about,
    weighted_covariance,
)
from mixed_signals.core.space import ParamSpace
from mixed_signals.core.topology import Topology
from mixed_signals.diagnostics._common import (
    _jit,
    _warn,
    circle_positions,
    period_of,
    warn_about_wide_circles,
    wide_circles,
)
from mixed_signals.estimators.base import (
    CHUNK_ENTRIES,
    _host_floats,
    _read_samples,
    _singularity,
    _unwrap_circles,
    factor_covariance,
)
from mixed_signals.metrics._cap import cap_samples


class MarginalMoments(NamedTuple):
    """Skewness and excess kurtosis of each parameter, circular on circles."""

    skewness: Float[Array, " param"]
    """g₁, or the circular skewness on a circle; zero for a Gaussian."""
    kurtosis: Float[Array, " param"]
    """g₂, or the circular kurtosis on a circle; zero for a Gaussian."""


class MardiaMoments(NamedTuple):
    """Mardia's multivariate skewness and kurtosis."""

    skewness: Float[Array, ""]
    """b₁, which tends to zero for a Gaussian."""
    kurtosis: Float[Array, ""]
    """b₂, which tends to D(D + 2) for a Gaussian."""


def moment_metric_names(space: ParamSpace) -> tuple[tuple[str, str], ...]:
    """Return the metric names of the skewness and kurtosis rows of each parameter.

    Args:
        space: Parameter space.

    Returns:
        ``("circular_skewness", "circular_kurtosis")`` for a circle, sphere
        azimuths included, and ``("skewness", "kurtosis")`` otherwise.
    """
    return tuple(
        ("circular_skewness", "circular_kurtosis")
        if parameter.topology is Topology.CIRCLE
        else ("skewness", "kurtosis")
        for parameter in space.parameters
    )


def marginal_moments(
    samples: ArrayLike, log_weights: ArrayLike | None, space: ParamSpace
) -> MarginalMoments:
    """Return the skewness and excess kurtosis of every parameter.

    Args:
        samples: Samples of one posterior, shaped (N, D).
        log_weights: Their log weights, shaped (N,); equal weights when
            ``None``.
        space: Parameter space of the samples.

    Returns:
        g₁ and g₂ of each parameter, or the circular skewness and kurtosis on
        circles; NaN for a parameter without spread.

    Raises:
        ValueError: If the shapes do not fit the space, or concrete values
            are invalid.
        TypeError: If an input is not real-valued.

    Warns:
        ValidityWarning: If a concrete parameter has no spread.
        PrecisionWarning: If an input is narrower than float64.

    References:
        :cite:t:`joanes1998comparing`; :cite:t:`rimoldini2014weighted`;
        :cite:t:`fisher1993statistical`; :cite:t:`mardia2000directional`.
    """
    samples_array, weights_array = _read_samples(samples, log_weights, space)
    moments = _marginal_moments(samples_array, weights_array, space=space)
    if _is_concrete(moments.skewness):
        _warn_about_flat_parameters(moments.skewness, space, "skewness and kurtosis")
    return moments


def anderson_darling_normality(
    samples: ArrayLike, log_weights: ArrayLike | None, space: ParamSpace
) -> Float[Array, " param"]:
    """Return the Anderson-Darling distance A²/n of every parameter from a normal.

    Args:
        samples: Samples of one posterior, shaped (N, D).
        log_weights: Their log weights, shaped (N,); equal weights when
            ``None``.
        space: Parameter space of the samples.

    Returns:
        A²/n of each parameter against the normal with its weighted mean and
        unbiased variance, a circle cut opposite its circular mean; NaN for a
        parameter without spread.

    Raises:
        ValueError: If the shapes do not fit the space, or concrete values
            are invalid.
        TypeError: If an input is not real-valued.

    Warns:
        ValidityWarning: If a concrete parameter has no spread, or a circle
            is wider than the limit of the surrogate null.
        PrecisionWarning: If an input is narrower than float64.

    References:
        :cite:t:`anderson1954test`; :cite:t:`stephens1974edf`.
    """
    samples_array, weights_array = _read_samples(samples, log_weights, space)
    unwrapped, _ = _unwrap_circles(samples_array, weights_array, space)
    distances = _anderson_darling(unwrapped, weights_array)
    if _is_concrete(distances):
        _warn_about_flat_parameters(distances, space, "the Anderson-Darling distance")
        warn_about_wide_circles(
            wide_circles(samples_array, weights_array, space),
            "the Anderson-Darling rows",
        )
    return distances


def mardia_moments(
    samples: ArrayLike,
    log_weights: ArrayLike | None,
    space: ParamSpace,
    *,
    key: PRNGKeyArray,
    cap: int = SAMPLE_CAP,
) -> MardiaMoments:
    """Return Mardia's multivariate skewness b₁ and kurtosis b₂.

    Args:
        samples: Samples of one posterior, shaped (N, D).
        log_weights: Their log weights, shaped (N,); equal weights when
            ``None``.
        space: Parameter space of the samples.
        key: JAX PRNG key for the subsample above the cap, derived from the
            row's path ending in ``"subsample"``.
        cap: Largest number of samples; above it a uniform subsample runs.

    Returns:
        b₁ and b₂ of the samples whitened by their weighted plug-in
        covariance, circles unwrapped about their circular means; NaN for a
        singular covariance.

    Raises:
        ValueError: If the shapes do not fit the space, concrete values are
            invalid, or ``cap`` is below one.
        TypeError: If an input is not real-valued.

    Warns:
        SubsamplingWarning: If N exceeds the cap.
        ValidityWarning: If the covariance of concrete samples is singular,
            or a circle is wider than the limit of the surrogate null.
        PrecisionWarning: If an input is narrower than float64.

    References:
        :cite:t:`mardia1970measures`; :cite:t:`mahalanobis1936generalised`.
    """
    samples_array, weights_array = _read_samples(samples, log_weights, space)
    samples_array, weights_array = cap_samples(
        samples_array, weights_array, key=key, metric="mardia", cap=cap
    )
    unwrapped, _ = _unwrap_circles(samples_array, weights_array, space)
    covariance = _plug_in_covariance(unwrapped, weights_array)
    factor = factor_covariance(covariance)
    moments = _mardia(unwrapped, weights_array, factor.cholesky)
    if _is_concrete(samples_array, weights_array):
        if (reason := _singularity(factor, space)) is not None:
            _warn(f"Mardia's moments are NaN: {reason}")
        warn_about_wide_circles(
            wide_circles(samples_array, weights_array, space), "Mardia's rows"
        )
    return moments


def _warn_about_flat_parameters(values: Array, space: ParamSpace, what: str) -> None:
    """Warn about the parameters whose statistic is NaN for lack of spread."""
    # ! Python floats only: NumPy flags invalid operations differently per platform.
    flat = [
        name
        for name, value in zip(space.names, _host_floats(values), strict=True)
        if math.isnan(value)
    ]
    if flat:
        _warn(f"{what} are NaN for {', '.join(map(repr, flat))}, which have no spread")


def _plug_in_covariance(
    samples: Float[Array, "sample param"], log_weights: Float[Array, " sample"]
) -> Float[Array, "param param"]:
    """Return Σ wᵢ(xᵢ − x̄)(xᵢ − x̄)ᵀ for normalised weights, the plug-in covariance."""
    _, covariance = weighted_covariance(samples, log_weights)
    weights = jax.nn.softmax(log_weights)
    return covariance * (1 - jnp.sum(weights**2))


def _linear_moments(
    values: Float[Array, " sample"], weights: Float[Array, " sample"]
) -> tuple[Array, Array]:
    """Return g₁ and g₂ of weighted values, measured from the most weighted value."""
    shifted = values - values[jnp.argmax(weights)]
    centred = shifted - jnp.sum(weights * shifted)
    second = jnp.sum(weights * centred**2)
    third = jnp.sum(weights * centred**3)
    fourth = jnp.sum(weights * centred**4)
    spread = second > 0
    safe = jnp.where(spread, second, 1.0)
    skewness = jnp.where(spread, third / safe**1.5, jnp.nan)
    kurtosis = jnp.where(spread, fourth / safe**2 - 3, jnp.nan)
    return skewness, kurtosis


def _circular_moments(
    values: Float[Array, " sample"],
    weights: Float[Array, " sample"],
    period: float,
    lower: float,
) -> tuple[Array, Array]:
    """Return the circular skewness and kurtosis of weighted values on a circle."""
    log_weights = jnp.log(weights)
    centre = circular_mean(values, log_weights, period, lower)
    angles = 2 * jnp.pi * (unwrap_about(values, centre, period) - centre) / period
    half_sines = jnp.sin(angles / 2)
    # * V = 1 − R̄ about the mean direction, from half-angle sines.
    deficit = 2 * jnp.sum(weights * half_sines**2)
    spread = deficit > 0
    safe = jnp.where(spread, deficit, 1.0)
    skewness = jnp.sum(weights * jnp.sin(2 * angles)) / safe**1.5
    numerator = (
        8 * jnp.sum(weights * half_sines**4) - 6 * safe**2 + 4 * safe**3 - safe**4
    )
    kurtosis = numerator / safe**2
    return jnp.where(spread, skewness, jnp.nan), jnp.where(spread, kurtosis, jnp.nan)


def _marginal_moments_kernel(
    samples: Float[Array, "sample param"],
    log_weights: Float[Array, " sample"],
    *,
    space: ParamSpace,
) -> MarginalMoments:
    """Return the marginal moments, circular on circles."""
    weights = jax.nn.softmax(log_weights)
    skewness, kurtosis = jax.vmap(_linear_moments, in_axes=(1, None))(samples, weights)
    for position in circle_positions(space):
        lower = space.parameters[position].bounds[0]
        circle_skewness, circle_kurtosis = _circular_moments(
            samples[:, position], weights, period_of(space, position), lower
        )
        skewness = skewness.at[position].set(circle_skewness)
        kurtosis = kurtosis.at[position].set(circle_kurtosis)
    return MarginalMoments(skewness=skewness, kurtosis=kurtosis)


def _anderson_darling_one(
    values: Float[Array, " sample"], log_weights: Float[Array, " sample"]
) -> Float[Array, ""]:
    """Return A²/n of weighted values against the normal of their moments."""
    weights = jax.nn.softmax(log_weights)
    mean, variance = weighted_covariance(values[:, None], log_weights)
    width = jnp.sqrt(variance[0, 0])
    scores = (values - mean[0]) / jnp.where(width > 0, width, 1.0)
    order = jnp.argsort(scores)
    sorted_scores = scores[order]
    # * F_n after each sorted value; the steps between values carry it.
    cumulative = jnp.cumsum(weights[order])
    log_lower = log_ndtr(sorted_scores)
    log_upper = log_ndtr(-sorted_scores)
    steps = cumulative[:-1] ** 2 * jnp.diff(log_lower) - (
        1 - cumulative[:-1]
    ) ** 2 * jnp.diff(log_upper)
    distance = -1 - log_upper[0] - log_lower[-1] + jnp.sum(steps)
    return jnp.where(width > 0, distance, jnp.nan)


def _anderson_darling_kernel(
    samples: Float[Array, "sample param"], log_weights: Float[Array, " sample"]
) -> Float[Array, " param"]:
    """Return A²/n of every parameter."""
    return jax.vmap(_anderson_darling_one, in_axes=(1, None))(samples, log_weights)


def _mardia_kernel(
    samples: Float[Array, "sample param"],
    log_weights: Float[Array, " sample"],
    cholesky: Float[Array, "param param"],
) -> MardiaMoments:
    """Return b₁ and b₂ of samples whitened by a plug-in covariance factor."""
    weights = jax.nn.softmax(log_weights)
    mean = jnp.sum(weights[:, None] * samples, axis=0)
    whitened = jax.scipy.linalg.solve_triangular(
        cholesky, (samples - mean).T, lower=True
    ).T
    kurtosis = jnp.sum(weights * jnp.sum(whitened**2, axis=1) ** 2)
    sample_count, size = whitened.shape
    chunk_size = max(1, min(sample_count, CHUNK_ENTRIES // max(sample_count, size)))
    chunk_count = -(-sample_count // chunk_size)
    padding = chunk_count * chunk_size - sample_count
    padded = jnp.concatenate([whitened, jnp.zeros((padding, size))])
    padded_weights = jnp.concatenate([weights, jnp.zeros(padding)])
    chunks = (
        padded.reshape(chunk_count, chunk_size, size),
        padded_weights.reshape(chunk_count, chunk_size),
    )
    partial_sums = jax.lax.map(
        functools.partial(_skewness_chunk, whitened, weights), chunks
    )
    return MardiaMoments(skewness=jnp.sum(partial_sums), kurtosis=kurtosis)


def _skewness_chunk(
    whitened: Float[Array, "sample param"],
    weights: Float[Array, " sample"],
    chunk: tuple[Array, Array],
) -> Float[Array, ""]:
    """Return Σᵢ Σⱼ wᵢwⱼ(zᵢᵀzⱼ)³ over the rows i of one chunk."""
    rows, row_weights = chunk
    products = rows @ whitened.T
    return jnp.sum(row_weights[:, None] * weights[None, :] * products**3)


_marginal_moments = _jit(_marginal_moments_kernel, "space")
_anderson_darling = _jit(_anderson_darling_kernel)
_mardia = _jit(_mardia_kernel)
