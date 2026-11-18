"""Kernel and distance metrics: maximum mean discrepancy and energy distance.

Both metrics compare two posteriors of one problem through distances between
all pairs of samples, in the coordinates they receive. Each function first
caps both posteriors at :data:`~mixed_signals._config.SAMPLE_CAP` samples
with :func:`~mixed_signals.metrics._cap.cap_pair`, because the sums over all
pairs cost O(N²). The rows read the metrics ``mmd`` and ``energy`` with the
estimator ``empirical``.

Maximum mean discrepancy
    MMD²(p, q) = E k(x, x') + E k(y, y') − 2 E k(x, y) for the Gaussian
    kernel k(x, y) = exp(−‖x − y‖²/(2ℓ²)) (Gretton et al. 2012).
    :func:`maximum_mean_discrepancy` returns the V-statistic of weighted
    samples, Σᵢⱼ wᵢwⱼ k(xᵢ, xⱼ) + Σᵢⱼ vᵢvⱼ k(yᵢ, yⱼ) − 2 Σᵢⱼ wᵢvⱼ k(xᵢ, yⱼ)
    with each posterior's weights normalised: the squared MMD between the two
    weighted empirical distributions, which is never negative. Gretton et
    al. (2012) call its root MMD_b. Its bias, of order 1/N, is shared by
    null draws at the same sample sizes, so it cancels when the null mean is
    subtracted. The row holds MMD², which the σ_eff map reads.
Bandwidth
    The median heuristic sets ℓ to the median distance between the pooled
    samples (Gretton et al. 2012), whose large-sample behaviour Garreau,
    Jitkrittum and Kanagawa (2017) analyse. :func:`median_bandwidth` takes the
    weighted median over all pairs of a uniform subsample of
    :data:`BANDWIDTH_SAMPLE_SIZE` samples per posterior, each posterior
    pooled with half the mass. Gretton et al. (2012, Appendix C) found a
    kernel chosen on a held-out tenth of the samples as good as one chosen on
    all of them. Fix ℓ once per comparison and reuse it in every null draw.
Energy distance
    E(p, q) = 2 E‖x − y‖ − E‖x − x'‖ − E‖y − y'‖ (Baringhaus and Franz 2004;
    Székely and Rizzo 2013). :func:`energy_distance` returns the V-statistic
    of weighted samples, as Székely and Rizzo (2013) use throughout, so it is
    never negative. It is the MMD of the kernel that the distance induces
    (Sejdinovic et al. 2013), and its root is a metric.
Circles and spheres
    Each circle of period P becomes a circle of circumference P in a plane,
    and each sphere a sphere of its radius in three dimensions, the unit
    sphere unless a whitening scaled it; distances are Euclidean in that
    embedding and equal the arc length locally. Euclidean
    spaces, and so the embedding, have strong negative type, which makes the
    energy distance zero only for equal distributions (Lyons 2013), and the
    Gaussian kernel is characteristic on them (Gretton et al. 2012). Arc
    lengths would lose both properties. Spheres and circles have negative
    type in their own metric but not strict negative type (Lyons 2020), so
    an energy distance of arc lengths cannot see differences in even
    harmonics on a circle, and a Gaussian kernel of the geodesic distance is
    not positive definite (Feragen, Lauze and Hauberg 2015). A zoo check
    found both choices equally sensitive to shifts of wrapped normals and of
    von Mises-Fisher distributions. On a circle with modes at 0 and π against
    modes at ±π/2, which differ only in even harmonics, arc lengths gave an
    energy distance within its null and an MMD² far below it, while the
    embedding resolved both. Chords are shorter than arcs, so the σ_eff of a
    2σ shift on a whitened circle read 2 to 9% low (see
    :ref:`validation-chords-and-arcs`). Whitening divides each
    circle by its circular standard deviation and gives each sphere the
    radius 1/s for its angular width s, which sets their weight against the
    Euclidean block.
Cost
    Three sums over all pairs of samples, evaluated in chunks of rows so
    that memory stays bounded.

References:
    :cite:t:`gretton2012kernel`; :cite:t:`garreau2017large`;
    :cite:t:`baringhaus2004new`; :cite:t:`szekely2013energy`;
    :cite:t:`sejdinovic2013equivalence`; :cite:t:`lyons2013distance`;
    :cite:t:`lyons2020strong`; :cite:t:`feragen2015geodesic`;
    :cite:t:`politis1999subsampling`.
"""

import functools
import math
from collections.abc import Callable
from typing import Final

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from jaxtyping import Array, Float, PRNGKeyArray

from mixed_signals._config import SAMPLE_CAP
from mixed_signals.core.samples import _is_concrete
from mixed_signals.core.space import ParamSpace
from mixed_signals.estimators.base import (
    CHUNK_ENTRIES,
    _jit,
    _map_in_chunks,
    _read_samples,
)
from mixed_signals.metrics._cap import _draw, read_and_cap_pair
from mixed_signals.metrics._embedding import embed
from mixed_signals.prng import derive_key

BANDWIDTH_SAMPLE_SIZE: Final = 1_000
"""Samples per posterior over whose pairs :func:`median_bandwidth` takes the median.

The bandwidth is fixed once per comparison and recorded, so its sampling
error changes which kernel is used, not the agreement between the value,
its null draws and its σ_eff map. Gretton et al. (2012, Appendix C) found a
kernel chosen on a held-out tenth of the samples as good as one chosen on all.

References:
    :cite:t:`gretton2012kernel`.
"""


def median_bandwidth(
    first: ArrayLike,
    first_log_weights: ArrayLike | None,
    second: ArrayLike,
    second_log_weights: ArrayLike | None,
    space: ParamSpace,
    *,
    key: PRNGKeyArray,
    sample_size: int = BANDWIDTH_SAMPLE_SIZE,
) -> Float[Array, ""]:
    """Return the median distance between the pooled samples of two posteriors.

    Each posterior above ``sample_size`` keeps a uniform subset of that many
    samples, drawn without replacement from ``derive_key(key, "first")`` or
    ``derive_key(key, "second")``. The median runs over every pair of the
    pooled subsets, each pair weighted by the product of its samples'
    weights, with each posterior pooled with half the mass; for equal
    weights and equal sample counts it is the plain median. Distances are
    those of :func:`maximum_mean_discrepancy`, chordal on circles and
    spheres.

    Args:
        first: Samples of the first posterior, shaped (N, D), in the
            parameter order of ``space``.
        first_log_weights: Its log weights shaped (N,); equal weights when
            ``None``.
        second: Samples of the second posterior, shaped (M, D).
        second_log_weights: Its log weights shaped (M,); equal weights when
            ``None``.
        space: Space of the samples.
        key: JAX PRNG key, derived with
            :func:`~mixed_signals.prng.derive_key` from the path of the
            comparison, ending in ``"bandwidth"``.
        sample_size: Largest number of samples per posterior.

    Returns:
        The bandwidth ℓ.

    Raises:
        ValueError: If the shapes do not fit the space, concrete values are
            invalid, or ``sample_size`` is below one.
        TypeError: If an input is not real-valued.

    References:
        :cite:t:`gretton2012kernel`; :cite:t:`garreau2017large`.
    """
    if sample_size < 1:
        raise ValueError(f"sample_size must be at least 1, got {sample_size}")
    subsets = []
    for name, samples, log_weights in (
        ("first", first, first_log_weights),
        ("second", second, second_log_weights),
    ):
        samples_array, weights_array = _read_samples(samples, log_weights, space)
        if samples_array.shape[0] > sample_size:
            samples_array, weights_array = _draw(
                samples_array, weights_array, derive_key(key, name), sample_size
            )
        subsets.extend([samples_array, weights_array])
    return _median_bandwidth(subsets[0], subsets[1], subsets[2], subsets[3], space)


def maximum_mean_discrepancy(
    first: ArrayLike,
    first_log_weights: ArrayLike | None,
    second: ArrayLike,
    second_log_weights: ArrayLike | None,
    space: ParamSpace,
    *,
    bandwidth: Float[ArrayLike, ""],
    key: PRNGKeyArray,
    cap: int = SAMPLE_CAP,
) -> Float[Array, ""]:
    """Return the squared MMD of a Gaussian kernel between two weighted samples.

    Args:
        first: Samples of the first posterior, shaped (N, D), in the
            parameter order of ``space``.
        first_log_weights: Its log weights shaped (N,); equal weights when
            ``None``.
        second: Samples of the second posterior, shaped (M, D).
        second_log_weights: Its log weights shaped (M,); equal weights when
            ``None``.
        space: Space of the samples.
        bandwidth: Kernel width ℓ, from :func:`median_bandwidth` once per
            comparison; positive.
        key: JAX PRNG key for the subsample above the cap, derived with
            :func:`~mixed_signals.prng.derive_key` from the path of the row,
            ending in ``"subsample"``.
        cap: Largest number of samples per posterior.

    Returns:
        The V-statistic MMD², non-negative up to rounding.

    Raises:
        ValueError: If the shapes do not fit the space, concrete values are
            invalid, a concrete bandwidth is not positive and finite, or
            ``cap`` is below one.
        TypeError: If an input is not real-valued.

    Warns:
        SubsamplingWarning: If a posterior has more samples than the cap.

    References:
        :cite:t:`gretton2012kernel`.
    """
    width = jnp.asarray(bandwidth, dtype=jnp.float64)
    if _is_concrete(width):
        # * Test the host value: a jnp test of a closed-over array is staged in a trace.
        host_width = float(width)
        if not (math.isfinite(host_width) and host_width > 0):
            raise ValueError(f"bandwidth must be positive and finite, got {host_width}")
    arrays = read_and_cap_pair(
        first,
        first_log_weights,
        second,
        second_log_weights,
        space,
        key=key,
        metric="mmd",
        cap=cap,
    )
    return _maximum_mean_discrepancy(*arrays, space, width)


def energy_distance(
    first: ArrayLike,
    first_log_weights: ArrayLike | None,
    second: ArrayLike,
    second_log_weights: ArrayLike | None,
    space: ParamSpace,
    *,
    key: PRNGKeyArray,
    cap: int = SAMPLE_CAP,
) -> Float[Array, ""]:
    """Return the energy distance between two weighted samples.

    Args:
        first: Samples of the first posterior, shaped (N, D), in the
            parameter order of ``space``.
        first_log_weights: Its log weights shaped (N,); equal weights when
            ``None``.
        second: Samples of the second posterior, shaped (M, D).
        second_log_weights: Its log weights shaped (M,); equal weights when
            ``None``.
        space: Space of the samples.
        key: JAX PRNG key for the subsample above the cap, derived with
            :func:`~mixed_signals.prng.derive_key` from the path of the row,
            ending in ``"subsample"``.
        cap: Largest number of samples per posterior.

    Returns:
        The V-statistic 2 E‖x − y‖ − E‖x − x'‖ − E‖y − y'‖, in the units of the
        coordinates, non-negative up to rounding.

    Raises:
        ValueError: If the shapes do not fit the space, concrete values are
            invalid, or ``cap`` is below one.
        TypeError: If an input is not real-valued.

    Warns:
        SubsamplingWarning: If a posterior has more samples than the cap.

    References:
        :cite:t:`baringhaus2004new`; :cite:t:`szekely2013energy`.
    """
    arrays = read_and_cap_pair(
        first,
        first_log_weights,
        second,
        second_log_weights,
        space,
        key=key,
        metric="energy",
        cap=cap,
    )
    return _energy_distance(*arrays, space)


def _gaussian(squared: Array, bandwidth: Array) -> Array:
    return jnp.exp(-squared / (2 * bandwidth**2))


def _distances_to(
    points: Float[Array, "sample _"], point: Float[Array, " _"]
) -> Float[Array, " sample"]:
    """Return the Euclidean distance from one point to every point."""
    return jnp.sqrt(jnp.sum((points - point) ** 2, axis=-1))


def _row_sums(
    others: Float[Array, "other _"],
    other_weights: Float[Array, " other"],
    transform: Callable[[Array], Array],
    chunk: Float[Array, "selected _"],
) -> Float[Array, " selected"]:
    """Return Σⱼ vⱼ f(‖xᵢ − yⱼ‖²) for each point xᵢ of a chunk."""
    squared = jnp.sum((chunk[:, None, :] - others[None, :, :]) ** 2, axis=-1)
    return transform(squared) @ other_weights


def _pair_sum(
    points: Float[Array, "sample _"],
    weights: Float[Array, " sample"],
    others: Float[Array, "other _"],
    other_weights: Float[Array, " other"],
    transform: Callable[[Array], Array],
) -> Float[Array, ""]:
    """Return Σᵢⱼ wᵢ vⱼ f(‖xᵢ − yⱼ‖²) over all pairs, in chunks of rows."""
    chunk_size = max(1, CHUNK_ENTRIES // (others.shape[0] * others.shape[1]))
    row_sums = _map_in_chunks(
        functools.partial(_row_sums, others, other_weights, transform),
        points,
        chunk_size,
    )
    return weights @ row_sums


def _maximum_mean_discrepancy_kernel(
    first: Float[Array, "sample param"],
    first_log_weights: Float[Array, " sample"],
    second: Float[Array, "other param"],
    second_log_weights: Float[Array, " other"],
    space: ParamSpace,
    bandwidth: Float[Array, ""],
) -> Float[Array, ""]:
    """Return the weighted V-statistic MMD² in the chordal embedding."""
    first_points, second_points = embed(first, space), embed(second, space)
    first_weights = jax.nn.softmax(first_log_weights)
    second_weights = jax.nn.softmax(second_log_weights)
    kernel = functools.partial(_gaussian, bandwidth=bandwidth)
    return (
        _pair_sum(first_points, first_weights, first_points, first_weights, kernel)
        + _pair_sum(
            second_points, second_weights, second_points, second_weights, kernel
        )
        - 2
        * _pair_sum(first_points, first_weights, second_points, second_weights, kernel)
    )


def _energy_distance_kernel(
    first: Float[Array, "sample param"],
    first_log_weights: Float[Array, " sample"],
    second: Float[Array, "other param"],
    second_log_weights: Float[Array, " other"],
    space: ParamSpace,
) -> Float[Array, ""]:
    """Return the weighted V-statistic energy distance in the chordal embedding."""
    first_points, second_points = embed(first, space), embed(second, space)
    first_weights = jax.nn.softmax(first_log_weights)
    second_weights = jax.nn.softmax(second_log_weights)
    return (
        2
        * _pair_sum(
            first_points, first_weights, second_points, second_weights, jnp.sqrt
        )
        - _pair_sum(first_points, first_weights, first_points, first_weights, jnp.sqrt)
        - _pair_sum(
            second_points, second_weights, second_points, second_weights, jnp.sqrt
        )
    )


def _median_bandwidth_kernel(
    first: Float[Array, "sample param"],
    first_log_weights: Float[Array, " sample"],
    second: Float[Array, "other param"],
    second_log_weights: Float[Array, " other"],
    space: ParamSpace,
) -> Float[Array, ""]:
    """Return the weighted median of the distances over all pooled pairs."""
    points = embed(jnp.concatenate([first, second]), space)
    weights = 0.5 * jnp.concatenate(
        [jax.nn.softmax(first_log_weights), jax.nn.softmax(second_log_weights)]
    )
    count, size = points.shape
    matrix: Array = jax.lax.map(
        functools.partial(_distances_to, points),
        points,
        batch_size=max(1, CHUNK_ENTRIES // (count * size)),
    )
    rows, columns = jnp.triu_indices(count, 1)
    return _weighted_median(matrix[rows, columns], weights[rows] * weights[columns])


def _weighted_median(
    values: Float[Array, " selected"], weights: Float[Array, " selected"]
) -> Float[Array, ""]:
    """Return the weighted median, the mean of the two central values for a tie.

    For equal weights it is the plain median, which takes the mean of the two
    central values for an even count.
    """
    order = jnp.argsort(values)
    # * Scaled by the largest, equal weights become exact ones, so sums count values.
    cumulative = jnp.cumsum(weights[order] / jnp.max(weights))
    half = 0.5 * cumulative[-1]
    sorted_values = values[order]
    lower = sorted_values[jnp.argmax(cumulative >= half)]
    upper = sorted_values[jnp.argmax(cumulative > half)]
    return 0.5 * (lower + upper)


_maximum_mean_discrepancy = _jit(_maximum_mean_discrepancy_kernel, "space")
_energy_distance = _jit(_energy_distance_kernel, "space")
_median_bandwidth = _jit(_median_bandwidth_kernel, "space")
