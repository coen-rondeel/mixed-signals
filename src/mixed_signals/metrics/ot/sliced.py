"""Sliced optimal transport: sliced W2 and the sliced Sinkhorn divergence.

Slicing
    A sliced divergence averages a one-dimensional divergence between the
    projections ⟨θ, x⟩ of two distributions over directions θ on the unit
    sphere (Rabin et al. 2012; Bonneel et al. 2015). Slicing keeps the metric
    axioms of the base divergence, and its sample complexity does not grow
    with the dimension (Nadjahi et al. 2020). Directions that are uniform on
    the sphere come in complete random orthonormal bases by default, and
    independent directions stay available.
Coordinates
    The samples are sliced in the chordal embedding of the space, in which
    each circle of period P is a circle of circumference P and each sphere a
    sphere of its radius, so circles and spheres enter as they do for the
    other distance-based metrics. Projections of the embedded points still
    determine their distribution (Cramér and Wold 1936). The directions
    therefore live in n = E + 2C + 3S dimensions, for E Euclidean axes, C
    circles and S spheres. :func:`sliced_directions` draws them once per
    comparison from a key whose path ends in ``"directions"``; every null draw
    reuses them. With the default ``orthonormal=True`` it returns the smallest
    number of complete bases that holds at least the number asked for, ⌈L/n⌉
    bases of n unit vectors each. Each basis is the Q factor of the QR
    decomposition of an n by n matrix of standard normals, with each column
    multiplied by the sign of the matching diagonal entry of R, which is
    uniform over the orthogonal group (Mezzadri 2007), so every direction is
    uniform on the sphere by itself, as a normal vector scaled to unit length
    is (Muller 1959). For a shift Δ the squared projections on an orthonormal
    basis sum to ‖Δ‖², so the mean over complete bases is ‖Δ‖²/n exactly,
    where independent directions reach it on average only. Rowland et al.
    (2019) study such orthogonal directions for Wasserstein distances and
    also report cases in which they raise the variance of a sliced estimate,
    which is why ``orthonormal=False`` returns independent directions, drawn
    as before from the same key.
Sliced W2
    :func:`sliced_wasserstein_distance` returns SW2², the mean over the
    directions of the squared W2 distance between the weighted projections,
    computed exactly from their quantile functions by ott-jax
    (Cuturi et al. 2022). Rows read the metric ``sliced_w2`` with the
    estimator ``empirical`` and hold SW2²: two samples of one distribution
    give a positive value, a finite-sample bias that the permutation null
    shares, so it cancels when the null mean is subtracted, as for MMD². A
    permutation test of sliced Wasserstein distances keeps its level at
    every sample size (Tran and Schreuder 2026). For a shift Δ of any
    distribution in the Euclidean block, SW2² = ‖Δ‖²/n over uniform
    directions, so σ_eff = √(n SW2²). In a zoo check, 100 and 1 000
    directions resolved shifts and width ratios alike in 2 to 16
    dimensions (see :ref:`validation-sliced-directions`);
    :data:`SLICED_DIRECTIONS` keeps a margin.
Gaussian estimator
    :func:`gaussian_sliced_wasserstein_distance` is a second estimator of the
    ``sliced_w2`` rows, named ``gaussian``. It fits a Gaussian to each set in
    the embedded coordinates, from the weighted mean and the unbiased
    weighted covariance, and averages over the shared directions the exact
    W2² of the projected normals, (θ·Δμ)² + (√(θᵀΣ₁θ) − √(θᵀΣ₂θ))²
    (Dowson and Landau 1982), which :func:`sliced_wasserstein_of_gaussians`
    computes. For any two distributions with these moments it is a lower
    bound of their sliced W2² (Gelbrich 1990), so the gap to the empirical
    value measures non-Gaussianity. Fitted moments bias it high, and the null
    subtraction removes the bias like any other.
Sliced Sinkhorn divergence
    :func:`sliced_sinkhorn_divergence` returns the mean over the directions
    of the Sinkhorn divergence S_ε of the projections, of the squared
    distance (Nadjahi et al. 2020); rows read ``sliced_sinkhorn``. A shift
    Δ again gives ‖Δ‖²/n, at every ε (Janati et al. 2020, Lemma 1). Its ε is
    :data:`~mixed_signals.metrics.ot.sinkhorn.SINKHORN_EPSILON_FACTOR` times
    the median projected squared distance between pooled samples, from
    :func:`sliced_sinkhorn_epsilon`, fixed once per comparison. Each
    direction costs a Sinkhorn problem over all N² pairs, so the divergence
    asks for :data:`SLICED_SINKHORN_DIRECTIONS` directions and has its own
    cap of :data:`SLICED_SINKHORN_CAP` samples per posterior: in 16
    dimensions, one call with 20 independent directions took 1.8 s at 1 000
    samples and 6.4 s at 2 000 on an Apple M4 Pro CPU, while sliced W2 with
    500 independent directions took 0.5 s at 10 000.
Zoo check
    With independent directions, the sliced value of an axis-aligned 1σ shift
    of Gaussians scattered by 4.8, 5.3 and 6.1% over 300 draws of about 500
    directions in 8, 16 and 32 dimensions, and by 26 to 30% at 20 directions
    (300 to 400 draws of the directions). The relative scatter of the value
    is the relative standard deviation of the mean of L squared projections
    of a uniform direction on the shift axis, each distributed as
    Beta(½, (n − 1)/2), which is √(2(n − 1)/((n + 2)L)): 5.3, 5.8 and 6.0% for
    L = 500 in 8, 16 and 32 dimensions. For one axis 1.5 times wider the
    measured scatter was 8 to 12% and 44 to 60%, at the two numbers of
    directions. With complete orthonormal bases the scatter was 0 for shifts
    in both cases, and 3 to 6% and 19 to 23% for the wider axis. The fitted
    moments of the Gaussian estimator bias it high by about 3/N in whitened
    units, +0.000 62 at N = 5 000 in 8 dimensions (see
    :ref:`validation-sliced-directions` and
    :ref:`validation-gaussian-sliced-w2`).

References:
    :cite:t:`rabin2012wasserstein`; :cite:t:`bonneel2015sliced`;
    :cite:t:`nadjahi2020statistical`; :cite:t:`cramer1936some`;
    :cite:t:`muller1959note`; :cite:t:`mezzadri2007random`;
    :cite:t:`rowland2019orthogonal`; :cite:t:`cuturi2022optimal`;
    :cite:t:`dowson1982frechet`; :cite:t:`gelbrich1990formula`;
    :cite:t:`tran2026minimax`; :cite:t:`janati2020entropic`.
"""

import functools
import math
from typing import Final

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from jaxtyping import Array, Bool, Float, PRNGKeyArray
from ott.geometry import costs, pointcloud
from ott.tools import sinkhorn_divergence as ott_divergence
from ott.tools import sliced

from mixed_signals.core.samples import _is_concrete, weighted_covariance
from mixed_signals.core.space import ParamSpace
from mixed_signals.estimators.base import CHUNK_ENTRIES, _jit, _read_samples, _warn
from mixed_signals.metrics._cap import _draw, read_and_cap_pair
from mixed_signals.metrics._embedding import embed, embedding_size
from mixed_signals.metrics.kernel import _weighted_median
from mixed_signals.metrics.ot.sinkhorn import (
    SINKHORN_EPSILON_FACTOR,
    SINKHORN_MAX_ITERATIONS,
    SINKHORN_THRESHOLD,
)
from mixed_signals.prng import derive_key

SLICED_DIRECTIONS: Final = 500
"""Directions of sliced W2 asked for once per comparison.

With the default orthonormal bases, sliced W2 takes ⌈500/n⌉ bases of n
directions, for example 512 directions in a 16- or 32-coordinate embedding.
In a zoo check with 1 000 samples per posterior, 100 and 1 000 independent
directions gave the same resolution in 2, 8 and 16 dimensions (see
:ref:`validation-sliced-directions`); how many directions a dimension
needs is left to a later benchmark.
"""

SLICED_SINKHORN_DIRECTIONS: Final = 20
"""Directions of the sliced Sinkhorn divergence, each a Sinkhorn problem over N² pairs.

With the default orthonormal bases, the divergence takes ⌈20/n⌉ bases of n
directions, for example 32 directions in a 16- or 32-coordinate embedding.
Nadjahi et al. (2020) used 10 directions on images and 100 on synthetic
data.
"""

SLICED_SINKHORN_CAP: Final = 2_000
"""Largest number of samples per posterior of the sliced Sinkhorn divergence.

With 20 independent directions in 16 dimensions, one call took 6.4 s at this
cap on an Apple M4 Pro CPU, so 200 null draws take about 21 minutes; the
common cap of 10 000 would take about 25 times longer (see
:ref:`validation-costs`). The time grows with
the number of directions, which the default bases round up to 32 in 16
dimensions.
"""

EPSILON_SAMPLE_SIZE: Final = 500
"""Samples per posterior whose pairs set the median of :func:`sliced_sinkhorn_epsilon`.

With 20 directions, 500 samples per posterior give about 10⁷ projected
squared distances, which bounds the time and memory of the median.
"""


def sliced_directions(
    space: ParamSpace,
    count: int,
    *,
    key: PRNGKeyArray,
    orthonormal: bool = True,
) -> Float[Array, "direction _"]:
    """Return unit directions of the embedded space, in complete orthonormal bases.

    With ``orthonormal=True`` the result holds ⌈count/n⌉ complete bases of
    n = E + 2C + 3S unit vectors each, the smallest number that holds at
    least ``count`` directions. Each basis is uniform over the orthogonal
    group: the Q factor of the QR decomposition of an n by n matrix of
    standard normals, with each column multiplied by the sign of the matching
    diagonal entry of R (Mezzadri 2007). Every direction is then uniform on
    the sphere by itself. For a vector v, the squared projections on a basis
    sum to the squared length of v, so the mean over the returned directions
    is that length squared over n exactly. With ``orthonormal=False`` the
    result holds ``count`` independent directions, normal vectors scaled to
    unit length (Muller 1959), which reach that mean on average only. The
    zoo check in the module description gives the scatter of both forms (see
    :ref:`validation-sliced-directions`).
    Rowland et al. (2019) study orthogonal directions and also report cases
    in which they raise the variance of a sliced estimate.

    Args:
        space: Space of the samples; the directions have one coordinate per
            embedded coordinate, E + 2C + 3S.
        count: Number of directions asked for, at least one.
        key: JAX PRNG key, derived with
            :func:`~mixed_signals.prng.derive_key` from the path of the
            comparison, ending in ``"directions"``.
        orthonormal: Whether to draw complete orthonormal bases, or
            ``count`` independent directions, drawn as they were before the
            bases were added, so that earlier results can be repeated.

    Returns:
        Unit vectors shaped (⌈count/n⌉ · n, n) for ``orthonormal=True``, in
        which each consecutive block of n rows is one basis, and (count, n)
        otherwise.

    Raises:
        ValueError: If ``count`` is below one.

    References:
        :cite:t:`mezzadri2007random`; :cite:t:`muller1959note`;
        :cite:t:`rowland2019orthogonal`.
    """
    if count < 1:
        raise ValueError(f"count must be at least 1, got {count}")
    size = embedding_size(space)
    if orthonormal:
        return _orthonormal_bases(key, math.ceil(count / size), size)
    normals = jax.random.normal(key, (count, size))
    lengths: Array = jnp.linalg.norm(normals, axis=1, keepdims=True)
    return normals / lengths


def _orthonormal_bases(
    key: PRNGKeyArray, bases: int, size: int
) -> Float[Array, "direction _"]:
    """Return ``bases`` random orthonormal bases of ``size`` unit rows, stacked."""
    normals = jax.random.normal(key, (bases, size, size))
    orthogonal, triangular = jnp.linalg.qr(normals)
    # * Fixing the signs of the diagonal of R makes Q uniform over the orthogonal group.
    diagonal = jnp.diagonal(triangular, axis1=-2, axis2=-1)
    signs = jnp.where(diagonal < 0, -1.0, 1.0)
    return (orthogonal * signs[:, None, :]).reshape(bases * size, size)


def sliced_wasserstein_distance(
    first: ArrayLike,
    first_log_weights: ArrayLike | None,
    second: ArrayLike,
    second_log_weights: ArrayLike | None,
    space: ParamSpace,
    *,
    directions: Float[ArrayLike, "direction _"],
) -> Float[Array, ""]:
    """Return SW2², the mean squared W2 distance between projected weighted samples.

    Args:
        first: Samples of the first posterior, shaped (N, D), in the
            parameter order of ``space``.
        first_log_weights: Its log weights shaped (N,); equal weights when
            ``None``.
        second: Samples of the second posterior, shaped (M, D).
        second_log_weights: Its log weights shaped (M,); equal weights when
            ``None``.
        space: Space of the samples.
        directions: Unit directions shaped (L, E + 2C + 3S), from
            :func:`sliced_directions` once per comparison.

    Returns:
        SW2², in the squared units of the coordinates.

    Raises:
        ValueError: If the shapes do not fit the space or the directions, or
            concrete values are invalid.
        TypeError: If an input is not real-valued.

    References:
        :cite:t:`rabin2012wasserstein`; :cite:t:`bonneel2015sliced`;
        :cite:t:`cuturi2022optimal`.
    """
    first_array, first_weights = _read_samples(first, first_log_weights, space)
    second_array, second_weights = _read_samples(second, second_log_weights, space)
    direction_array = _read_directions(directions, space)
    return _sliced_wasserstein(
        first_array, first_weights, second_array, second_weights, space, direction_array
    )


def sliced_wasserstein_of_gaussians(
    mean_first: Float[ArrayLike, " _"],
    covariance_first: Float[ArrayLike, "_ _"],
    mean_second: Float[ArrayLike, " _"],
    covariance_second: Float[ArrayLike, "_ _"],
    *,
    directions: Float[ArrayLike, "direction _"],
) -> Float[Array, ""]:
    """Return SW2² between two Gaussians, the mean exact W2² of their projections.

    A direction θ projects the Gaussians to the normals N(θ·μ₁, θᵀΣ₁θ) and
    N(θ·μ₂, θᵀΣ₂θ), whose squared W2 distance is
    (θ·Δμ)² + (√(θᵀΣ₁θ) − √(θᵀΣ₂θ))² (Dowson and Landau 1982), with
    Δμ = μ₁ − μ₂. The result is the mean of this value over the directions.
    For any two distributions with these moments, it is a lower bound of
    their sliced W2² (Gelbrich 1990). The function runs under
    :func:`jax.jit` and :func:`jax.vmap`.

    Args:
        mean_first: Mean of the first Gaussian, shaped (n,).
        covariance_first: Its covariance, shaped (n, n).
        mean_second: Mean of the second Gaussian, shaped (n,).
        covariance_second: Its covariance, shaped (n, n).
        directions: Unit directions shaped (L, n), in the coordinates of the
            means and covariances.

    Returns:
        SW2², in the squared units of the coordinates; NaN if a covariance
        gives a negative or undefined variance along a direction.

    Raises:
        ValueError: If the means, covariances and directions do not share
            one coordinate count n.

    References:
        :cite:t:`dowson1982frechet`; :cite:t:`gelbrich1990formula`;
        :cite:t:`rabin2012wasserstein`.
    """
    direction_array = jnp.asarray(directions, dtype=jnp.float64)
    if direction_array.ndim != 2 or direction_array.shape[0] == 0:
        raise ValueError(
            f"directions must be shaped (L, n) with L at least 1, got shape"
            f" {direction_array.shape}"
        )
    size = direction_array.shape[1]
    moments: dict[str, Array] = {}
    for name, moment, shape in (
        ("mean_first", mean_first, (size,)),
        ("covariance_first", covariance_first, (size, size)),
        ("mean_second", mean_second, (size,)),
        ("covariance_second", covariance_second, (size, size)),
    ):
        moments[name] = jnp.asarray(moment, dtype=jnp.float64)
        if moments[name].shape != shape:
            raise ValueError(
                f"{name} must be shaped {shape} for directions shaped (L, {size}),"
                f" got shape {moments[name].shape}"
            )
    return _sliced_gaussian(
        moments["mean_first"],
        moments["covariance_first"],
        moments["mean_second"],
        moments["covariance_second"],
        direction_array,
    )


def gaussian_sliced_wasserstein_distance(
    first: ArrayLike,
    first_log_weights: ArrayLike | None,
    second: ArrayLike,
    second_log_weights: ArrayLike | None,
    space: ParamSpace,
    *,
    directions: Float[ArrayLike, "direction _"],
) -> Float[Array, ""]:
    """Return SW2² between the moment-matched Gaussians of two weighted sample sets.

    Each set gives the weighted mean and the unbiased weighted covariance
    (:func:`~mixed_signals.core.samples.weighted_covariance`) of its embedded
    points, so that the Gaussians are compared in the coordinates that
    :func:`sliced_wasserstein_distance` slices; for a space of real axes the
    embedding is the identity. The value is that of
    :func:`sliced_wasserstein_of_gaussians` for them. Rows read the metric
    ``sliced_w2`` with the estimator ``gaussian``. For any two distributions
    with these moments the value is a lower bound of their sliced W2²
    (Gelbrich 1990), so the gap to the empirical value measures
    non-Gaussianity. The fitted moments bias it high, by about 3/N in
    whitened units (zoo check in the module description and
    :ref:`validation-gaussian-sliced-w2`), and the null
    subtraction removes the bias like any other. The function traces under
    :func:`~mixed_signals.nulls.plan.null_draws`.

    Args:
        first: Samples of the first posterior, shaped (N, D), in the
            parameter order of ``space``.
        first_log_weights: Its log weights shaped (N,); equal weights when
            ``None``.
        second: Samples of the second posterior, shaped (M, D).
        second_log_weights: Its log weights shaped (M,); equal weights when
            ``None``.
        space: Space of the samples.
        directions: Unit directions shaped (L, E + 2C + 3S), from
            :func:`sliced_directions` once per comparison, the same as those
            of the empirical row.

    Returns:
        SW2² of the Gaussian fits, in the squared units of the coordinates;
        NaN if a set has fewer than two samples with positive weight or a
        fitted covariance has a negative variance along a direction.

    Raises:
        ValueError: If the shapes do not fit the space or the directions, or
            concrete values are invalid.
        TypeError: If an input is not real-valued.

    References:
        :cite:t:`dowson1982frechet`; :cite:t:`gelbrich1990formula`;
        :cite:t:`rabin2012wasserstein`.
    """
    first_array, first_weights = _read_samples(first, first_log_weights, space)
    second_array, second_weights = _read_samples(second, second_log_weights, space)
    direction_array = _read_directions(directions, space)
    return _gaussian_sliced_wasserstein(
        first_array, first_weights, second_array, second_weights, space, direction_array
    )


def sliced_sinkhorn_divergence(
    first: ArrayLike,
    first_log_weights: ArrayLike | None,
    second: ArrayLike,
    second_log_weights: ArrayLike | None,
    space: ParamSpace,
    *,
    directions: Float[ArrayLike, "direction _"],
    epsilon: Float[ArrayLike, ""],
    key: PRNGKeyArray,
    cap: int = SLICED_SINKHORN_CAP,
) -> Float[Array, ""]:
    """Return the mean Sinkhorn divergence S_ε between projected weighted samples.

    Args:
        first: Samples of the first posterior, shaped (N, D), in the
            parameter order of ``space``.
        first_log_weights: Its log weights shaped (N,); equal weights when
            ``None``.
        second: Samples of the second posterior, shaped (M, D).
        second_log_weights: Its log weights shaped (M,); equal weights when
            ``None``.
        space: Space of the samples.
        directions: Unit directions shaped (L, E + 2C + 3S), from
            :func:`sliced_directions` once per comparison.
        epsilon: Entropic regularisation ε of every projection, positive,
            from :func:`sliced_sinkhorn_epsilon` once per comparison.
        key: JAX PRNG key for the subsample above the cap, derived with
            :func:`~mixed_signals.prng.derive_key` from the path of the row,
            ending in ``"subsample"``.
        cap: Largest number of samples per posterior.

    Returns:
        The mean of S_ε over the directions, in the squared units of the
        coordinates; NaN if the iterations did not converge for a direction.

    Raises:
        ValueError: If the shapes do not fit the space or the directions,
            concrete values are invalid, a concrete ε is not positive and
            finite, or ``cap`` is below one.
        TypeError: If an input is not real-valued.

    Warns:
        SubsamplingWarning: If a posterior has more samples than the cap.
        ValidityWarning: If concrete iterations did not converge.

    References:
        :cite:t:`nadjahi2020statistical`; :cite:t:`feydy2019interpolating`;
        :cite:t:`cuturi2022optimal`.
    """
    epsilon_array = jnp.asarray(epsilon, dtype=jnp.float64)
    if _is_concrete(epsilon_array):
        # * Test the host value: a jnp test of a closed-over array is staged in a trace.
        host_epsilon = float(epsilon_array)
        if not (math.isfinite(host_epsilon) and host_epsilon > 0):
            raise ValueError(f"epsilon must be positive and finite, got {host_epsilon}")
    direction_array = _read_directions(directions, space)
    arrays = read_and_cap_pair(
        first,
        first_log_weights,
        second,
        second_log_weights,
        space,
        key=key,
        metric="sliced_sinkhorn",
        cap=cap,
    )
    value, converged = _sliced_sinkhorn(*arrays, space, direction_array, epsilon_array)
    if _is_concrete(converged) and not bool(converged):
        _warn(
            "sliced_sinkhorn: the iterations did not reach the marginal error"
            f" {SINKHORN_THRESHOLD:g} within {SINKHORN_MAX_ITERATIONS} iterations"
            " for every direction; the divergence is NaN, and a larger epsilon"
            " converges faster"
        )
    return jnp.where(converged, value, jnp.nan)


def sliced_sinkhorn_epsilon(
    first: ArrayLike,
    first_log_weights: ArrayLike | None,
    second: ArrayLike,
    second_log_weights: ArrayLike | None,
    space: ParamSpace,
    *,
    directions: Float[ArrayLike, "direction _"],
    key: PRNGKeyArray,
    sample_size: int = EPSILON_SAMPLE_SIZE,
) -> Float[Array, ""]:
    """Return ε of the sliced Sinkhorn divergence from the projected pooled samples.

    ε is :data:`~mixed_signals.metrics.ot.sinkhorn.SINKHORN_EPSILON_FACTOR`
    times the weighted median of the squared differences of the projections
    over every pair of pooled samples and every direction, each posterior
    pooled with half the mass, as the median bandwidth pools them. Each
    posterior above ``sample_size`` keeps a uniform subset of that many
    samples, drawn without replacement from ``derive_key(key, "first")`` or
    ``derive_key(key, "second")``.

    Args:
        first: Samples of the first posterior, shaped (N, D), in the
            parameter order of ``space``.
        first_log_weights: Its log weights shaped (N,); equal weights when
            ``None``.
        second: Samples of the second posterior, shaped (M, D).
        second_log_weights: Its log weights shaped (M,); equal weights when
            ``None``.
        space: Space of the samples.
        directions: Unit directions shaped (L, E + 2C + 3S), those of the
            divergence.
        key: JAX PRNG key, derived with
            :func:`~mixed_signals.prng.derive_key` from the path of the
            comparison, ending in ``"epsilon"``.
        sample_size: Largest number of samples per posterior.

    Returns:
        ε, in the squared units of the coordinates.

    Raises:
        ValueError: If the shapes do not fit the space or the directions,
            concrete values are invalid, or ``sample_size`` is below one.
        TypeError: If an input is not real-valued.
    """
    if sample_size < 1:
        raise ValueError(f"sample_size must be at least 1, got {sample_size}")
    direction_array = _read_directions(directions, space)
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
    return SINKHORN_EPSILON_FACTOR * _projected_median(
        subsets[0], subsets[1], subsets[2], subsets[3], space, direction_array
    )


def _read_directions(directions: ArrayLike, space: ParamSpace) -> Array:
    """Return directions as float64 after checking that they fit the embedded space."""
    array = jnp.asarray(directions, dtype=jnp.float64)
    size = embedding_size(space)
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] != size:
        raise ValueError(
            f"directions must be shaped (L, {size}) for the {size} embedded"
            f" coordinates of the space, got shape {array.shape}"
        )
    return array


def _identity(_: PRNGKeyArray, points: Array) -> Array:
    """Return the points unchanged, which ott-jax then reads as projections."""
    return points


def _sliced_wasserstein_kernel(
    first: Float[Array, "sample param"],
    first_log_weights: Float[Array, " sample"],
    second: Float[Array, "other param"],
    second_log_weights: Float[Array, " other"],
    space: ParamSpace,
    directions: Float[Array, "direction _"],
) -> Float[Array, ""]:
    """Return the mean of the exact squared W2 distances between the projections."""
    value, _ = sliced.sliced_wasserstein(
        embed(first, space) @ directions.T,
        embed(second, space) @ directions.T,
        a=jax.nn.softmax(first_log_weights),
        b=jax.nn.softmax(second_log_weights),
        proj_fn=_identity,
    )
    return value


def _sliced_gaussian_kernel(
    mean_first: Float[Array, " _"],
    covariance_first: Float[Array, "_ _"],
    mean_second: Float[Array, " _"],
    covariance_second: Float[Array, "_ _"],
    directions: Float[Array, "direction _"],
) -> Float[Array, ""]:
    """Return the mean over the directions of the exact W2² of the projected normals."""
    location = directions @ (mean_first - mean_second)
    variance_first = jnp.sum((directions @ covariance_first) * directions, axis=1)
    variance_second = jnp.sum((directions @ covariance_second) * directions, axis=1)
    defined = jnp.all((variance_first >= 0) & (variance_second >= 0))
    # * Clipping keeps the gradient finite; the flag below returns NaN for the row.
    widths = jnp.sqrt(jnp.maximum(variance_first, 0.0)) - jnp.sqrt(
        jnp.maximum(variance_second, 0.0)
    )
    return jnp.where(defined, jnp.mean(location**2 + widths**2), jnp.nan)


def _gaussian_sliced_wasserstein_kernel(
    first: Float[Array, "sample param"],
    first_log_weights: Float[Array, " sample"],
    second: Float[Array, "other param"],
    second_log_weights: Float[Array, " other"],
    space: ParamSpace,
    directions: Float[Array, "direction _"],
) -> Float[Array, ""]:
    """Return SW2² of the moment-matched Gaussians of the embedded points."""
    mean_first, covariance_first = weighted_covariance(
        embed(first, space), first_log_weights
    )
    mean_second, covariance_second = weighted_covariance(
        embed(second, space), second_log_weights
    )
    return _sliced_gaussian_kernel(
        mean_first, covariance_first, mean_second, covariance_second, directions
    )


def _one_direction(
    first_weights: Float[Array, " sample"],
    second_weights: Float[Array, " other"],
    epsilon: Float[Array, ""],
    projections: tuple[Float[Array, " sample"], Float[Array, " other"]],
) -> tuple[Float[Array, ""], Bool[Array, ""]]:
    """Return S_ε of one pair of projections and whether its problems converged."""
    first_projection, second_projection = projections
    chunk = max(
        1, CHUNK_ENTRIES // max(first_projection.shape[0], second_projection.shape[0])
    )
    value, output = ott_divergence.sinkhorn_divergence(
        pointcloud.PointCloud,
        x=first_projection[:, None],
        y=second_projection[:, None],
        a=first_weights,
        b=second_weights,
        cost_fn=costs.SqEuclidean(),
        epsilon=epsilon,
        batch_size=chunk,
        solve_kwargs={
            "threshold": SINKHORN_THRESHOLD,
            "max_iterations": SINKHORN_MAX_ITERATIONS,
        },
    )
    return value, jnp.all(jnp.asarray(output.converged))


def _sliced_sinkhorn_kernel(
    first: Float[Array, "sample param"],
    first_log_weights: Float[Array, " sample"],
    second: Float[Array, "other param"],
    second_log_weights: Float[Array, " other"],
    space: ParamSpace,
    directions: Float[Array, "direction _"],
    epsilon: Float[Array, ""],
) -> tuple[Float[Array, ""], Bool[Array, ""]]:
    """Return the mean S_ε over the directions, one direction after the other."""
    projections = (
        (embed(first, space) @ directions.T).T,
        (embed(second, space) @ directions.T).T,
    )
    divergence = functools.partial(
        _one_direction,
        jax.nn.softmax(first_log_weights),
        jax.nn.softmax(second_log_weights),
        epsilon,
    )
    values, converged = jax.lax.map(divergence, projections)
    return jnp.mean(values), jnp.all(converged)


def _projected_median_kernel(
    first: Float[Array, "sample param"],
    first_log_weights: Float[Array, " sample"],
    second: Float[Array, "other param"],
    second_log_weights: Float[Array, " other"],
    space: ParamSpace,
    directions: Float[Array, "direction _"],
) -> Float[Array, ""]:
    """Return the weighted median of the projected squared distances of all pairs."""
    projections = embed(jnp.concatenate([first, second]), space) @ directions.T
    weights = 0.5 * jnp.concatenate(
        [jax.nn.softmax(first_log_weights), jax.nn.softmax(second_log_weights)]
    )
    rows, columns = jnp.triu_indices(projections.shape[0], 1)
    squared = (projections[rows] - projections[columns]) ** 2
    pair_weights = jnp.broadcast_to(
        (weights[rows] * weights[columns])[:, None], squared.shape
    )
    return _weighted_median(squared.reshape(-1), pair_weights.reshape(-1))


_sliced_wasserstein = _jit(_sliced_wasserstein_kernel, "space")
_sliced_gaussian = _jit(_sliced_gaussian_kernel)
_gaussian_sliced_wasserstein = _jit(_gaussian_sliced_wasserstein_kernel, "space")
_sliced_sinkhorn = _jit(_sliced_sinkhorn_kernel, "space")
_projected_median = _jit(_projected_median_kernel, "space")
