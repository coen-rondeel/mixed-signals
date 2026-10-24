"""Moment metrics: mean shift, width ratio, and Gaussian KL, W2 and Hellinger.

Coordinates
    Every metric is computed in the coordinates it receives. Engines z-score
    both inputs of a comparison with their shared
    :class:`~mixed_signals.transforms.whiten.Whitening`
    (``decorrelate=False``) for the per-parameter rows, so a mean shift is in
    units of the reference σ in mode A, as Glass's Δ (Glass, McGaw and Smith
    1981), and of the pooled σ in mode B, the root mean square of the two
    widths, as Cohen's d (Cohen 1988). The Gaussian divergences run in the
    decorrelated whitening, where the W2 location term is the Mahalanobis
    shift.
Per-parameter moments
    :func:`mean_shift` is μ_second − μ_first per parameter, from the weighted
    means of :func:`~mixed_signals.core.samples.weighted_covariance`, and on
    a circle the shortest signed arc between the two circular means.
    :func:`width_ratio` is σ_second/σ_first, from the unbiased weighted widths
    and on a circle from the circular standard deviations, the widths that
    the z-score divides by (Nakagawa et al. 2015 compare variability as such
    ratios). Their rows read the estimator ``empirical``.
Gaussian divergences
    :func:`gaussian_kl`, :func:`gaussian_w2` and :func:`gaussian_hellinger`
    split a divergence between two Gaussians into a location term, which
    only the means change, and a shape term, which only the covariances
    change. The KL divergence is their sum (Kullback and Leibler 1951;
    Rasmussen and Williams 2006). The squared W2 distance is the squared
    Euclidean distance between the means plus the squared Bures distance
    between the covariances (Dowson and Landau 1982; Givens and Shortt 1984).
    The Hellinger distance follows from the Bhattacharyya distance D_B as
    H² = 1 − exp(−D_B), and D_B is a location term plus a shape term
    (Hellinger 1909; Bhattacharyya 1943; Fukunaga 1990). Each result also
    carries ln(det Σ_q/det Σ_p), whose sign sets the direction of the r_eff of
    a shape term. Fed with two
    :class:`~mixed_signals.estimators.gaussian.GaussianDensity` fits, their
    rows read the estimator ``gaussian``.
Circles
    Given a space, the Gaussian divergences take the shortest signed arc as
    the mean difference on its circles. The
    :class:`~mixed_signals.estimators.gaussian.GaussianDensity` of each
    posterior unwraps every circle about its own circular mean, so its
    covariance is that of the unwrapped values. Sphere axes follow their own
    topologies, the azimuth a circle and the polar axis an interval.
Names
    The result rows carry the names of the zoo's reference tables:
    ``mean_shift``, ``width_ratio``, and ``kl``, ``w2`` and ``hellinger`` with
    the suffixes ``_location`` and ``_shape`` for their terms.

References:
    :cite:t:`glass1981meta`; :cite:t:`cohen1988statistical`;
    :cite:t:`nakagawa2015meta`; :cite:t:`harris2020array`;
    :cite:t:`mardia2000directional`; :cite:t:`kullback1951information`;
    :cite:t:`rasmussen2006gaussian`; :cite:t:`dowson1982frechet`;
    :cite:t:`givens1984class`; :cite:t:`hellinger1909neue`;
    :cite:t:`bhattacharyya1943measure`; :cite:t:`fukunaga1990introduction`.
"""

import math
import warnings
from collections.abc import Callable
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
from jax.scipy.linalg import solve_triangular
from jax.typing import ArrayLike
from jaxtyping import Array, Float

from mixed_signals.core.samples import (
    _is_concrete,
    circular_mean,
    circular_standard_deviation,
    unwrap_about,
    weighted_covariance,
)
from mixed_signals.core.space import ParamSpace
from mixed_signals.errors import ValidityWarning
from mixed_signals.estimators.base import (
    _PACKAGE_DIRECTORY,
    _circle_positions,
    _jit,
    _period,
    _read_samples,
    _warn_about_unresolved_circles,
)


def _jit_over_space[**Parameters, Result](
    function: Callable[Parameters, Result],
) -> Callable[Parameters, Result]:
    """Compile a kernel with :func:`jax.jit`, its ``space`` argument static."""
    return _jit(function, "space")


class Terms(NamedTuple):
    """A divergence between two Gaussians, with its location and shape terms."""

    total: Array
    """The divergence."""
    location: Array
    """The part that only the means change."""
    shape: Array
    """The part that only the covariances change."""
    log_det_ratio: Array
    """ln(det Σ_q/det Σ_p): positive when q is wider than p, the r_eff direction."""


@_jit_over_space
def gaussian_kl(
    mean_p: Float[Array, " param"],
    cov_p: Float[Array, "param param"],
    mean_q: Float[Array, " param"],
    cov_q: Float[Array, "param param"],
    space: ParamSpace | None = None,
) -> Terms:
    """Return the KL divergence KL(p ‖ q) between two Gaussians, in nats.

    The location term is ½ ΔμᵀΣ_q⁻¹Δμ and the shape term
    ½ [tr(Σ_q⁻¹Σ_p) − D + ln(det Σ_q / det Σ_p)]; the total is their sum.
    Equal covariances give a shape term of exactly zero. Inside
    ``jax.jit``, a covariance that is not positive definite gives NaN.

    Args:
        mean_p: Mean of p, shaped (D,).
        cov_p: Covariance of p, shaped (D, D).
        mean_q: Mean of q, shaped (D,).
        cov_q: Covariance of q, shaped (D, D).
        space: Space of the means; on its circles Δμ is the shortest signed
            arc. ``None`` treats every axis as a real line.

    Returns:
        The divergence, its location and shape terms and the log-determinant
        ratio.

    References:
        :cite:t:`kullback1951information`; :cite:t:`rasmussen2006gaussian`.
    """
    difference = _mean_difference(mean_p, mean_q, space)
    cholesky_p, cholesky_q = jnp.linalg.cholesky(cov_p), jnp.linalg.cholesky(cov_q)
    whitened_shift = solve_triangular(cholesky_q, difference, lower=True)
    location = 0.5 * whitened_shift @ whitened_shift
    whitened_root = solve_triangular(cholesky_q, cholesky_p, lower=True)
    log_det_ratio = _log_det(cholesky_q) - _log_det(cholesky_p)
    shape = 0.5 * (jnp.sum(whitened_root**2) - mean_p.shape[0] + log_det_ratio)
    # * Equal covariances give an exact zero, not a rounding residue.
    shape = jnp.where(jnp.all(cov_p == cov_q), 0.0, shape)
    return Terms(location + shape, location, shape, log_det_ratio)


@_jit_over_space
def gaussian_w2(
    mean_p: Float[Array, " param"],
    cov_p: Float[Array, "param param"],
    mean_q: Float[Array, " param"],
    cov_q: Float[Array, "param param"],
    space: ParamSpace | None = None,
) -> Terms:
    """Return the W2 distance between two Gaussians.

    The location term is the Euclidean distance between the means, and the
    shape term the Bures distance between the covariances,
    [tr Σ_p + tr Σ_q − 2 tr((Σ_p^½ Σ_q Σ_p^½)^½)]^½. The total is the root of
    the sum of their squares. The distance is in the units of the
    coordinates, so it changes when they are rescaled; whitened coordinates
    put it in units of σ. The matrix roots come from eigendecompositions,
    whose gradients are undefined where two eigenvalues coincide.

    Args:
        mean_p: Mean of p, shaped (D,).
        cov_p: Covariance of p, shaped (D, D).
        mean_q: Mean of q, shaped (D,).
        cov_q: Covariance of q, shaped (D, D).
        space: Space of the means; on its circles Δμ is the shortest signed
            arc. ``None`` treats every axis as a real line.

    Returns:
        The distance, its location and shape terms and the log-determinant
        ratio.

    References:
        :cite:t:`dowson1982frechet`; :cite:t:`givens1984class`.
    """
    location = jnp.linalg.norm(_mean_difference(mean_p, mean_q, space))
    eigenvalues, eigenvectors = jnp.linalg.eigh(cov_p)
    root_p = (eigenvectors * jnp.sqrt(jnp.clip(eigenvalues, 0.0))) @ eigenvectors.T
    middle = root_p @ cov_q @ root_p
    middle_eigenvalues = jnp.linalg.eigvalsh(0.5 * (middle + middle.T))
    cross_trace = jnp.sum(jnp.sqrt(jnp.clip(middle_eigenvalues, 0.0)))
    traces = jnp.trace(cov_p) + jnp.trace(cov_q)
    bures_squared = jnp.maximum(traces - 2 * cross_trace, 0.0)
    shape = jnp.where(jnp.all(cov_p == cov_q), 0.0, jnp.sqrt(bures_squared))
    log_det_ratio = _log_det(jnp.linalg.cholesky(cov_q)) - _log_det(
        jnp.linalg.cholesky(cov_p)
    )
    return Terms(jnp.hypot(location, shape), location, shape, log_det_ratio)


@_jit_over_space
def gaussian_hellinger(
    mean_p: Float[Array, " param"],
    cov_p: Float[Array, "param param"],
    mean_q: Float[Array, " param"],
    cov_q: Float[Array, "param param"],
    space: ParamSpace | None = None,
) -> Terms:
    """Return the Hellinger distance between two Gaussians.

    With Σ = (Σ_p + Σ_q)/2, the Bhattacharyya distance has the location term
    ⅛ ΔμᵀΣ⁻¹Δμ and the shape term ½ ln(det Σ / √(det Σ_p det Σ_q)). Each
    term converts to a distance as H = √(1 − exp(−D_B)), so that
    1 − H² of the total is the product of 1 − H² of the two terms. The
    distance does not change under an invertible linear map of the
    coordinates.

    Args:
        mean_p: Mean of p, shaped (D,).
        cov_p: Covariance of p, shaped (D, D).
        mean_q: Mean of q, shaped (D,).
        cov_q: Covariance of q, shaped (D, D).
        space: Space of the means; on its circles Δμ is the shortest signed
            arc. ``None`` treats every axis as a real line.

    Returns:
        The distance and its location and shape terms, each in [0, 1], and
        the log-determinant ratio.

    References:
        :cite:t:`hellinger1909neue`; :cite:t:`bhattacharyya1943measure`;
        :cite:t:`fukunaga1990introduction`.
    """
    cholesky_middle = jnp.linalg.cholesky(0.5 * (cov_p + cov_q))
    whitened_shift = solve_triangular(
        cholesky_middle, _mean_difference(mean_p, mean_q, space), lower=True
    )
    location_distance = 0.125 * whitened_shift @ whitened_shift
    log_det_p = _log_det(jnp.linalg.cholesky(cov_p))
    log_det_q = _log_det(jnp.linalg.cholesky(cov_q))
    shape_distance = jnp.maximum(
        0.5 * (_log_det(cholesky_middle) - 0.5 * (log_det_p + log_det_q)), 0.0
    )
    return Terms(
        _hellinger_from_bhattacharyya(location_distance + shape_distance),
        _hellinger_from_bhattacharyya(location_distance),
        _hellinger_from_bhattacharyya(shape_distance),
        log_det_q - log_det_p,
    )


def mean_shift(
    first: ArrayLike,
    first_log_weights: ArrayLike | None,
    second: ArrayLike,
    second_log_weights: ArrayLike | None,
    space: ParamSpace,
) -> Float[Array, " param"]:
    """Return the shift of each mean, from the first posterior to the second.

    The shift is μ_second − μ_first from the weighted means, and on a circle
    the shortest signed arc from the first circular mean to the second. It
    is in the units of the coordinates; z-scored coordinates make it the
    shift in units of the reference or pooled σ.

    Args:
        first: Samples of the first posterior, shaped (N, D), in the
            parameter order of ``space``.
        first_log_weights: Its log weights shaped (N,); equal weights when
            ``None``.
        second: Samples of the second posterior, shaped (M, D).
        second_log_weights: Its log weights shaped (M,); equal weights when
            ``None``.
        space: Space of the samples.

    Returns:
        The shift of each parameter, shaped (D,).

    Raises:
        ValueError: If the shapes do not fit the space, or concrete values are
            invalid.
        TypeError: If an input is not real-valued.

    Warns:
        ValidityWarning: If the Rayleigh test does not resolve the mean
            direction of a circle of concrete samples, whose circular mean,
            and shift, are then arbitrary.

    References:
        :cite:t:`glass1981meta`; :cite:t:`cohen1988statistical`;
        :cite:t:`mardia2000directional`.
    """
    first_array, first_weights = _read_samples(first, first_log_weights, space)
    second_array, second_weights = _read_samples(second, second_log_weights, space)
    if _is_concrete(first_array, first_weights, second_array, second_weights):
        for samples, log_weights in (
            (first_array, first_weights),
            (second_array, second_weights),
        ):
            _warn_about_unresolved_circles(
                samples,
                log_weights,
                space,
                consequence="its circular mean, and the mean shift, are arbitrary",
            )
    return _mean_shift(first_array, first_weights, second_array, second_weights, space)


def width_ratio(
    first: ArrayLike,
    first_log_weights: ArrayLike | None,
    second: ArrayLike,
    second_log_weights: ArrayLike | None,
    space: ParamSpace,
) -> Float[Array, " param"]:
    """Return the ratio of each width, the second posterior's over the first's.

    Widths are the unbiased weighted standard deviations of
    :func:`~mixed_signals.core.samples.weighted_covariance`, and on a circle
    the circular standard deviations, the widths that the z-score whitening
    divides by. A ratio below one means that the second posterior is
    narrower.

    Args:
        first: Samples of the first posterior, shaped (N, D), in the
            parameter order of ``space``.
        first_log_weights: Its log weights shaped (N,); equal weights when
            ``None``.
        second: Samples of the second posterior, shaped (M, D).
        second_log_weights: Its log weights shaped (M,); equal weights when
            ``None``.
        space: Space of the samples.

    Returns:
        The ratio of each parameter, shaped (D,); NaN where the first width
        is zero or a width is undefined.

    Raises:
        ValueError: If the shapes do not fit the space, or concrete values are
            invalid.
        TypeError: If an input is not real-valued.

    Warns:
        ValidityWarning: If a ratio of concrete samples is NaN.

    References:
        :cite:t:`nakagawa2015meta`; :cite:t:`harris2020array`;
        :cite:t:`mardia2000directional`.
    """
    first_array, first_weights = _read_samples(first, first_log_weights, space)
    second_array, second_weights = _read_samples(second, second_log_weights, space)
    ratios = _width_ratio(
        first_array, first_weights, second_array, second_weights, space
    )
    if _is_concrete(ratios):
        undefined = [
            name
            for name, ratio in zip(space.names, np.asarray(ratios), strict=True)
            if math.isnan(float(ratio))
        ]
        if undefined:
            warnings.warn(
                ValidityWarning(
                    f"the width ratio of {', '.join(map(repr, undefined))} is NaN:"
                    " the first posterior has no spread there, or a posterior has"
                    " fewer than two samples with positive weight"
                ),
                skip_file_prefixes=(_PACKAGE_DIRECTORY,),
            )
    return ratios


def _mean_shift_kernel(
    first: Float[Array, "sample param"],
    first_log_weights: Float[Array, " sample"],
    second: Float[Array, "other param"],
    second_log_weights: Float[Array, " other"],
    space: ParamSpace,
) -> Float[Array, " param"]:
    """Return the per-parameter mean shift, an arc between circular means on circles."""
    first_mean, _ = weighted_covariance(first, first_log_weights)
    second_mean, _ = weighted_covariance(second, second_log_weights)
    shift = second_mean - first_mean
    for position in _circle_positions(space):
        period = _period(space, position)
        lower = space.parameters[position].bounds[0]
        arc = circular_mean(
            second[:, position], second_log_weights, period, lower
        ) - circular_mean(first[:, position], first_log_weights, period, lower)
        shift = shift.at[position].set(unwrap_about(arc, 0.0, period))
    return shift


def _width_ratio_kernel(
    first: Float[Array, "sample param"],
    first_log_weights: Float[Array, " sample"],
    second: Float[Array, "other param"],
    second_log_weights: Float[Array, " other"],
    space: ParamSpace,
) -> Float[Array, " param"]:
    """Return the per-parameter width ratio, from circular widths on circles."""
    first_widths = _widths(first, first_log_weights, space)
    second_widths = _widths(second, second_log_weights, space)
    # ! A first width of zero has no ratio; it gives NaN, not infinity.
    return jnp.where(first_widths > 0, second_widths / first_widths, jnp.nan)


def _widths(
    samples: Float[Array, "sample param"],
    log_weights: Float[Array, " sample"],
    space: ParamSpace,
) -> Float[Array, " param"]:
    """Return the unbiased weighted widths, circular standard deviations on circles."""
    _, covariance = weighted_covariance(samples, log_weights)
    widths = jnp.sqrt(jnp.diagonal(covariance))
    for position in _circle_positions(space):
        widths = widths.at[position].set(
            circular_standard_deviation(
                samples[:, position], log_weights, _period(space, position)
            )
        )
    return widths


def _mean_difference(
    mean_p: Float[Array, " param"],
    mean_q: Float[Array, " param"],
    space: ParamSpace | None,
) -> Float[Array, " param"]:
    """Return μ_q − μ_p, as the shortest signed arc on the circles of a space."""
    difference = mean_q - mean_p
    if space is None:
        return difference
    for position in _circle_positions(space):
        difference = difference.at[position].set(
            unwrap_about(difference[position], 0.0, _period(space, position))
        )
    return difference


def _log_det(cholesky_factor: Float[Array, "param param"]) -> Float[Array, ""]:
    return 2.0 * jnp.sum(jnp.log(jnp.diag(cholesky_factor)))


def _hellinger_from_bhattacharyya(distance: Float[Array, ""]) -> Float[Array, ""]:
    return jnp.sqrt(-jnp.expm1(-distance))


_mean_shift = _jit_over_space(_mean_shift_kernel)
_width_ratio = _jit_over_space(_width_ratio_kernel)
