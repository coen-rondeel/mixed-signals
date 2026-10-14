"""Bounded kernel density estimator: wrapped on circles, corrected at interval bounds.

:class:`KernelDensity` wraps ``jax.scipy.stats.gaussian_kde``, the JAX port of
SciPy's estimator (Virtanen et al. 2020): Gaussian kernels whose covariance is
the bandwidth factor squared times the weighted covariance of the samples.

Bandwidth
    Silverman's factor (n (D + 2)/4)^(−1/(D+4)), with n the Kish effective
    sample size of the weights, minimises the asymptotic mean integrated
    squared error when the posterior is Gaussian (Silverman 1986). A float
    replaces the rule. :meth:`KernelDensity.resolved_settings` gives the
    factor as a number; plots pass it to cantuccio, whose estimates follow
    SciPy's convention, so contours and metrics agree away from bounds.
Circles
    The samples are unwrapped about their circular mean, which fixes the
    covariance and so the bandwidth, and the kernels are summed over
    windings, so the density does not depend on where the circle is cut.
Intervals in one dimension
    The non-negative boundary correction of Jones and Foster (1996),
    f̂ = f̄ exp(f̂_L/f̄ − 1). Here f̄ is the estimate divided by the kernel mass
    inside the support, and f̂_L the linear boundary correction of Jones
    (1993, eq. 3.4), the local linear fit at a bound, written for an interval
    with both bounds at once. Reflection (Schuster 1985) has a bias of order
    h at a bound wherever the slope there is not zero; this correction, of
    order h², recovers the zoo's railing scores within a few per cent where
    reflection is high by 10 to 30 per cent. The formula follows the
    ``jf96`` correction of evmix (Hu and Scarrott 2018). The estimate is
    renormalised to integrate to one: Gauss-Legendre quadrature (Davis and
    Rabinowitz 1984) covers :data:`~mixed_signals.estimators.base.IMAGE_REACH`
    bandwidths from each bound, and the normal distribution function the
    rest, where the correction vanishes.
Intervals in two or more dimensions
    Reflection at every finite bound (Schuster 1985): the unbounded estimate
    is summed over the mirror images of each point, so the density integrates
    to one over the support.
Spheres and whitened spaces
    A sphere's azimuth is a circle and its polar axis an interval, which is
    approximate near the poles. Decorrelated whitened axes are real lines
    without bounds, so bounded estimates run in the original space or in the
    z-scored one, where bounds stay bounds.
Validity
    A singular covariance gives NaN and a
    :class:`~mixed_signals.errors.ValidityWarning`. Below the sample size at
    which the relative mean squared error at the mode of a Gaussian reaches
    0.1 (Silverman 1986, Table 4.2), the estimate warns; see
    :data:`MINIMUM_SAMPLE_SIZES`.
Memory
    Points are evaluated in chunks of about 2²² kernel evaluations, so memory
    stays bounded for any number of points.

References:
    :cite:t:`virtanen2020scipy`; :cite:t:`jax2022gaussian`;
    :cite:t:`silverman1986density`; :cite:t:`schuster1985incorporating`;
    :cite:t:`jones1993simple`; :cite:t:`jones1996simple`;
    :cite:t:`hu2018evmix`; :cite:t:`davis1984methods`;
    :cite:t:`mardia2000directional`.
"""

import math
from typing import ClassVar, Final, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import logsumexp, ndtr
from jax.scipy.stats import gaussian_kde
from jax.typing import ArrayLike
from jaxtyping import Array, Bool, Float
from pydantic import PositiveFloat

from mixed_signals.core.samples import _is_concrete, unwrap_about
from mixed_signals.core.space import ParamSpace
from mixed_signals.core.topology import Topology
from mixed_signals.estimators.base import (
    CHUNK_ENTRIES,
    IMAGE_REACH,
    WIDTH_MARGIN,
    Estimator,
    EstimatorSettings,
    _circle_positions,
    _covers,
    _image_count,
    _image_table,
    _jit,
    _map_in_chunks,
    _period,
    _read_points,
    _read_samples,
    _singularity,
    _unwrap_circles,
    _warn,
    _winding_images,
    factor_covariance,
)

SILVERMAN: Final = "silverman"
"""Bandwidth rule (n (D + 2)/4)^(−1/(D+4)), the default."""

MINIMUM_SAMPLE_SIZES: Final = (
    4,
    19,
    67,
    223,
    768,
    2_788,
    10_730,
    43_684,
    187_275,
    841_566,
    3_948_098,
    19_269_575,
    97_555_966,
    511_006_511,
    2_763_308_000,
    15_396_451_411,
    88_238_189_914,
    519_367_649_968,
    3_135_321_120_093,
    19_388_443_096_830,
)
"""Smallest n_eff for a relative mean squared error below 0.1 at a Gaussian's mode.

Entry D − 1 holds the criterion of Silverman (1986, Table 4.2) for D
dimensions: a standard normal density estimated at its mode with a normal
kernel and the bandwidth that minimises the mean squared error there. The
values come from the exact bias and variance of that estimate and match the
table, which prints them to three digits, up to D = 10. A KDE warns below
them, and always above 20 dimensions.

References:
    :cite:t:`silverman1986density`.
"""

_HALF_LOG_TWO_PI: Final = 0.5 * math.log(2 * math.pi)
# * Panels per quadrature strip of IMAGE_REACH bandwidths: each spans half a bandwidth.
_STRIP_PANELS: Final = 24
_LEGENDRE_ORDER: Final = 8
# * Fixed node table: the panels move with the bandwidth, the rule does not.
_LEGENDRE_NODES, _LEGENDRE_WEIGHTS = np.polynomial.legendre.leggauss(_LEGENDRE_ORDER)


class KernelDensitySettings(EstimatorSettings):
    """Settings of the bounded kernel density estimator.

    Attributes:
        bandwidth: ``"silverman"`` for Silverman's rule, or the bandwidth
            factor itself, which multiplies the standard deviations of the
            samples to give the kernel widths.

    References:
        :cite:t:`silverman1986density`.
    """

    name: ClassVar[str] = "kde"

    bandwidth: Literal["silverman"] | PositiveFloat = SILVERMAN


class KernelDensity(Estimator):
    """Gaussian kernel density estimate of one posterior, respecting its topology.

    Build it with :meth:`fit`. Its settings keep the bandwidth rule, and
    :meth:`resolved_settings` gives the factor that the rule chose.

    Attributes:
        kde: JAX's estimator on the samples, with each circle unwrapped about
            its circular mean.
        factor: Bandwidth factor: the kernel covariance is its square times
            the weighted covariance of the samples.
        centres: Circular mean of each circle, about which it is unwrapped,
            shaped (C,).
        log_normaliser: Log of the integral of the uncorrected estimate over
            the support for a one-dimensional interval, zero otherwise; NaN
            when the estimate is invalid.
        space: Parameter space of the samples.
        image_signs: Sign of each axis in every image of a point, (I, D).
        image_shifts: Shift of each axis in every image of a point, (I, D).

    References:
        :cite:t:`jax2022gaussian`; :cite:t:`silverman1986density`.
    """

    kde: gaussian_kde
    factor: Float[Array, ""]
    centres: Float[Array, " circle"]
    log_normaliser: Float[Array, ""]
    space: ParamSpace = eqx.field(static=True)
    image_signs: tuple[tuple[float, ...], ...] = eqx.field(static=True)
    image_shifts: tuple[tuple[float, ...], ...] = eqx.field(static=True)
    settings: KernelDensitySettings = eqx.field(static=True)

    @classmethod
    def fit(
        cls,
        samples: ArrayLike,
        log_weights: ArrayLike | None,
        space: ParamSpace,
        settings: KernelDensitySettings | None = None,
    ) -> "KernelDensity":
        """Fit the kernel density estimate to the samples of one posterior.

        Args:
            samples: Samples in the parameter order of ``space``, shaped
                (N, D).
            log_weights: Log weights shaped (N,); equal weights when
                ``None``.
            space: Parameter space of the samples.
            settings: The settings; Silverman's rule when ``None``.

        Returns:
            The fitted estimate, NaN if the covariance of the samples is
            singular.

        Raises:
            ValueError: If the shapes do not fit the space, or concrete values
                are invalid.
            TypeError: If an input is not real-valued.

        Warns:
            ValidityWarning: If the covariance of concrete samples is
                singular, or their effective sample size lies below
                :data:`MINIMUM_SAMPLE_SIZES` for D dimensions.
            PrecisionWarning: If an input is narrower than float64.

        References:
            :cite:t:`silverman1986density`; :cite:t:`jones1996simple`.
        """
        settings = settings if settings is not None else KernelDensitySettings()
        samples_array, weights_array = _read_samples(samples, log_weights, space)
        unwrapped, centres = _unwrap_circles(samples_array, weights_array, space)
        size = len(space)
        # * JAX takes a fixed factor as a scalar array and a rule by its name.
        rule = (
            SILVERMAN
            if settings.bandwidth == SILVERMAN
            else jnp.asarray(float(settings.bandwidth))
        )
        kde = gaussian_kde(
            unwrapped.T,
            bw_method=rule,
            weights=jax.nn.softmax(weights_array),
        )
        factor = _bandwidth_factor(kde.neff, size, settings)
        covariance_factor = factor_covariance(kde.covariance / factor**2)
        if _is_concrete(samples_array, weights_array):
            if (reason := _singularity(covariance_factor, space)) is not None:
                _warn(f"the kernel density estimate is NaN: {reason}")
            _warn_about_sample_size(float(kde.neff), size)
        widths = jnp.sqrt(jnp.diagonal(kde.covariance))
        largest_factor = (
            _largest_silverman_factor(size)
            if settings.bandwidth == SILVERMAN
            else float(settings.bandwidth)
        )
        axis_images, covered = _axis_images(space, widths, largest_factor)
        signs, shifts = _image_table(axis_images)
        valid = jnp.isfinite(covariance_factor.log_determinant) & covered
        estimator = cls(
            kde=kde,
            factor=factor,
            centres=centres,
            log_normaliser=jnp.where(valid, 0.0, jnp.nan),
            space=space,
            image_signs=signs,
            image_shifts=shifts,
            settings=settings,
        )
        if _is_corrected(space):
            mass = _corrected_mass(estimator)
            estimator = eqx.tree_at(
                lambda fitted: fitted.log_normaliser,
                estimator,
                jnp.log(mass) + estimator.log_normaliser,
            )
        return estimator

    def log_density(
        self, points: Float[ArrayLike, "*leading param"]
    ) -> Float[Array, " *leading"]:
        """Return the log density estimate at points.

        Points outside an interval bound have log density −∞, and circle
        values may lie on any winding.

        Args:
            points: Points shaped (..., D), in the coordinates of the space.

        Returns:
            The log densities, shaped (...).

        Raises:
            ValueError: If the points do not end in the D parameters.
            TypeError: If the points are not real-valued.
        """
        values = _read_points(points, len(self.space))
        flat = values.reshape(-1, values.shape[-1])
        kernel = _corrected_log_density if _is_corrected(self.space) else _log_density
        return kernel(self, flat).reshape(values.shape[:-1])

    def resolved_settings(self) -> KernelDensitySettings:
        """Return the settings with the bandwidth factor as a number.

        Plots pass this factor to cantuccio as its ``bandwidth``, and fitting
        other samples with these settings uses the same factor. Call it on the
        estimator of one problem, outside ``jax.jit``.
        """
        return KernelDensitySettings(bandwidth=float(self.factor))


def _bandwidth_factor(
    effective_sample_size: Array, size: int, settings: KernelDensitySettings
) -> Float[Array, ""]:
    """Return the bandwidth factor that JAX's estimator applies for the settings."""
    if settings.bandwidth == SILVERMAN:
        return jnp.power(effective_sample_size * (size + 2) / 4.0, -1.0 / (size + 4))
    return jnp.asarray(float(settings.bandwidth))


def _largest_silverman_factor(size: int) -> float:
    """Return Silverman's factor for one effective sample, the largest it takes."""
    return float(((size + 2) / 4.0) ** (-1.0 / (size + 4)))


def _is_corrected(space: ParamSpace) -> bool:
    """Return whether the space is one interval, for the Jones-Foster estimate."""
    return len(space) == 1 and space.parameters[0].topology is Topology.INTERVAL


def _axis_images(
    space: ParamSpace, widths: Float[Array, " param"], largest_factor: float
) -> tuple[list[list[tuple[float, float]]], Bool[Array, ""]]:
    """Return the images of every axis and whether they cover the kernel reach.

    Circles take windings and, in two or more dimensions, intervals take
    mirror images at their finite bounds. Inside ``jax.jit`` the kernel
    widths are unknown, so the counts follow the widest kernel that the
    samples admit; the returned flag is false where a kernel is wider still.
    """
    corrected = _is_corrected(space)
    axis_images: list[list[tuple[float, float]]] = []
    covered = jnp.asarray(True)
    for position, parameter in enumerate(space.parameters):
        width = widths[position]
        lower, upper = parameter.bounds
        length = upper - lower
        if parameter.topology is Topology.CIRCLE:
            # * Unwrapped samples span one period, so their spread is at most half.
            bound = largest_factor * WIDTH_MARGIN * length / 2
            windings = _image_count(width, length, bound, 1.0)
            axis_images.append(_winding_images(length, windings))
            covered = covered & _covers(width, length, windings, 1.0)
        elif parameter.topology is Topology.REAL or corrected:
            axis_images.append([(1.0, 0.0)])
        elif math.isinf(upper):
            axis_images.append([(1.0, 0.0), (-1.0, 2 * lower)])
        elif math.isinf(lower):
            axis_images.append([(1.0, 0.0), (-1.0, 2 * upper)])
        else:
            bound = largest_factor * WIDTH_MARGIN * length / 2
            reflections = _image_count(width, length, bound, 2.0)
            axis_images.append(
                [
                    (1.0, 2 * step * length)
                    for step in range(-reflections, reflections + 1)
                ]
                + [
                    (-1.0, 2 * lower + 2 * step * length)
                    for step in range(-reflections, reflections + 1)
                ]
            )
            covered = covered & _covers(width, length, reflections, 2.0)
    return axis_images, covered


def _warn_about_sample_size(effective_sample_size: float, size: int) -> None:
    """Warn when n_eff lies below Silverman's sample size for D dimensions."""
    if size <= len(MINIMUM_SAMPLE_SIZES):
        needed = MINIMUM_SAMPLE_SIZES[size - 1]
        if effective_sample_size >= needed:
            return
        requirement = f"at least {needed} effective samples"
    else:
        requirement = f"over {MINIMUM_SAMPLE_SIZES[-1]:.2g} effective samples"
    _warn(
        f"a {size}-dimensional kernel density estimate from {effective_sample_size:.4g}"
        f" effective samples is unreliable: it needs {requirement} for a relative"
        " mean squared error below 0.1 at the mode of a Gaussian (Silverman 1986,"
        " Table 4.2)"
    )


def _reduce_circles(
    estimator: KernelDensity, points: Float[Array, "*leading param"]
) -> Float[Array, "*leading param"]:
    """Return points with each circle value unwrapped about its circular mean."""
    for index, position in enumerate(_circle_positions(estimator.space)):
        points = points.at[:, position].set(
            unwrap_about(
                points[:, position],
                estimator.centres[index],
                _period(estimator.space, position),
            )
        )
    return points


def _inside(
    space: ParamSpace, points: Float[Array, "*leading param"]
) -> Bool[Array, " *leading"]:
    """Return whether points lie within the bounds of every interval axis."""
    inside = jnp.ones(points.shape[0], dtype=bool)
    for position, parameter in enumerate(space.parameters):
        if parameter.topology is Topology.INTERVAL:
            lower, upper = parameter.bounds
            inside = (
                inside & (points[:, position] >= lower) & (points[:, position] <= upper)
            )
    return inside


def _chunk_size(estimator: KernelDensity) -> int:
    """Return the points per chunk that keep a chunk near the evaluation budget."""
    size, sample_count = estimator.kde.dataset.shape
    return max(1, CHUNK_ENTRIES // (int(sample_count) * int(size)))


def _log_density_kernel(
    estimator: KernelDensity, points: Float[Array, "*leading param"]
) -> Float[Array, " *leading"]:
    """Return the log density at points shaped (M, D), summed over their images."""
    reduced = _reduce_circles(estimator, points)
    images = reduced[:, None, :] * jnp.asarray(estimator.image_signs) + jnp.asarray(
        estimator.image_shifts
    )
    flat = images.reshape(-1, images.shape[-1])
    log_values = _map_in_chunks_of(estimator, flat)
    log_density = logsumexp(log_values.reshape(images.shape[:-1]), axis=-1)
    return jnp.where(
        _inside(estimator.space, points),
        log_density - estimator.log_normaliser,
        -jnp.inf,
    )


def _map_in_chunks_of(
    estimator: KernelDensity, points: Float[Array, "*leading param"]
) -> Float[Array, " *leading"]:
    """Return the unbounded log density of JAX's estimator at points, in chunks."""
    return _map_in_chunks(
        lambda chunk: estimator.kde.logpdf(chunk.T),  # type: ignore[no-untyped-call]
        points,
        _chunk_size(estimator),
    )


def _corrected_terms(
    estimator: KernelDensity, values: Float[Array, " *leading"]
) -> Float[Array, " *leading"]:
    """Return the log of the Jones-Foster estimate before renormalisation, at values."""
    lower, upper = estimator.space.parameters[0].bounds
    bandwidth = jnp.sqrt(estimator.kde.covariance[0, 0])
    data = estimator.kde.dataset[0]
    log_weights = jnp.log(estimator.kde.weights)
    offsets = (values[:, None] - data) / bandwidth
    log_terms = log_weights - 0.5 * offsets**2 - _HALF_LOG_TWO_PI
    log_sum = logsumexp(log_terms, axis=-1)
    # * The kernel-weighted mean offset is S₁/S₀ of the linear correction.
    mean_offset = jnp.sum(jax.nn.softmax(log_terms, axis=-1) * offsets, axis=-1)
    high = (values - lower) / bandwidth
    low = (values - upper) / bandwidth
    mass = ndtr(high) - ndtr(low)
    first = _normal_density(low) - _normal_density(high)
    second = mass + _times_density(low) - _times_density(high)
    correction = mass * (second - first * mean_offset) / (mass * second - first**2)
    log_estimate = log_sum - jnp.log(bandwidth) - jnp.log(mass) + correction - 1
    return jnp.where(jnp.isneginf(log_sum), -jnp.inf, log_estimate)


def _normal_density(values: Array) -> Array:
    """Return the standard normal density, zero at infinite arguments."""
    return jnp.exp(-0.5 * values**2) / math.sqrt(2 * math.pi)


def _times_density(values: Array) -> Array:
    """Return x φ(x), zero at infinite arguments where the product is 0 · ∞."""
    finite = jnp.isfinite(values)
    return jnp.where(
        finite, jnp.where(finite, values, 0.0) * _normal_density(values), 0.0
    )


def _corrected_log_density_kernel(
    estimator: KernelDensity, points: Float[Array, "*leading param"]
) -> Float[Array, " *leading"]:
    """Return the renormalised Jones-Foster log density at points shaped (M, 1)."""
    log_values = _map_in_chunks(
        lambda chunk: _corrected_terms(estimator, chunk[:, 0]),
        points,
        _chunk_size(estimator),
    )
    return jnp.where(
        _inside(estimator.space, points),
        log_values - estimator.log_normaliser,
        -jnp.inf,
    )


def _corrected_mass_kernel(estimator: KernelDensity) -> Float[Array, ""]:
    """Return the integral of the Jones-Foster estimate over its interval.

    Gauss-Legendre panels of half a bandwidth cover the strips within
    :data:`~mixed_signals.estimators.base.IMAGE_REACH` bandwidths of each
    finite bound, where the correction acts; between the strips the estimate
    equals the plain one, whose mass is a sum of normal distribution
    functions.
    """
    lower, upper = estimator.space.parameters[0].bounds
    bandwidth = jnp.sqrt(estimator.kde.covariance[0, 0])
    data = estimator.kde.dataset[0]
    weights = estimator.kde.weights
    reach = IMAGE_REACH * bandwidth
    # * With two bounds, the strips meet at the middle once the interval is narrow.
    middle = 0.5 * (lower + upper)
    inner_low = jnp.asarray(-jnp.inf)
    inner_high = jnp.asarray(jnp.inf)
    total = jnp.asarray(0.0)
    if math.isfinite(lower):
        inner_low = lower + reach
        if math.isfinite(upper):
            inner_low = jnp.minimum(inner_low, middle)
        total = total + _strip_integral(estimator, jnp.asarray(lower), inner_low)
    if math.isfinite(upper):
        inner_high = upper - reach
        if math.isfinite(lower):
            inner_high = jnp.maximum(inner_high, middle)
        total = total + _strip_integral(estimator, inner_high, jnp.asarray(upper))
    between = ndtr((inner_high - data) / bandwidth) - ndtr(
        (inner_low - data) / bandwidth
    )
    return total + jnp.sum(weights * between)


def _strip_integral(
    estimator: KernelDensity, start: Float[Array, ""], end: Float[Array, ""]
) -> Float[Array, ""]:
    """Integrate the unnormalised Jones-Foster estimate over [start, end]."""
    edges = start + (end - start) * jnp.linspace(0.0, 1.0, _STRIP_PANELS + 1)
    half_widths = 0.5 * jnp.diff(edges)
    midpoints = 0.5 * (edges[:-1] + edges[1:])
    nodes = midpoints[:, None] + half_widths[:, None] * _LEGENDRE_NODES
    values = jnp.exp(_corrected_terms(estimator, nodes.ravel())).reshape(nodes.shape)
    return jnp.sum(half_widths[:, None] * _LEGENDRE_WEIGHTS * values)


_log_density = _jit(_log_density_kernel)
_corrected_log_density = _jit(_corrected_log_density_kernel)
_corrected_mass = _jit(_corrected_mass_kernel)
