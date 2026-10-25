"""Sliced optimal transport: sliced W2 and the sliced Sinkhorn divergence.

Slicing
    A sliced divergence averages a one-dimensional divergence between the
    projections ⟨θ, x⟩ of two distributions over directions θ drawn
    uniformly on the unit sphere (Rabin et al. 2012; Bonneel et al. 2015).
    Slicing keeps the metric axioms of the base divergence, and its sample
    complexity does not grow with the dimension (Nadjahi et al. 2020).
Coordinates
    The samples are sliced in the chordal embedding of the space, in which
    each circle of period P is a circle of circumference P and each sphere a
    sphere of its radius, so circles and spheres enter as they do for the
    other distance-based metrics. Projections of the embedded points still
    determine their distribution (Cramér and Wold 1936). The directions
    therefore live in n = E + 2C + 3S dimensions, for E Euclidean axes, C
    circles and S spheres. :func:`sliced_directions` draws them, as normal
    vectors scaled to unit length (Muller 1959), once per comparison from a
    key whose path ends in ``"directions"``; every null draw reuses them.
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
    dimensions; :data:`SLICED_DIRECTIONS` keeps a margin.
Sliced Sinkhorn divergence
    :func:`sliced_sinkhorn_divergence` returns the mean over the directions
    of the Sinkhorn divergence S_ε of the projections, of the squared
    distance (Nadjahi et al. 2020); rows read ``sliced_sinkhorn``. A shift
    Δ again gives ‖Δ‖²/n, at every ε (Janati et al. 2020, Lemma 1). Its ε is
    :data:`~mixed_signals.metrics.ot.sinkhorn.SINKHORN_EPSILON_FACTOR` times
    the median projected squared distance between pooled samples, from
    :func:`sliced_sinkhorn_epsilon`, fixed once per comparison. Each
    direction costs a Sinkhorn problem over all N² pairs, so the divergence uses
    :data:`SLICED_SINKHORN_DIRECTIONS` directions and its own cap of
    :data:`SLICED_SINKHORN_CAP` samples per posterior: in 16 dimensions, one
    call took 1.8 s at 1 000 samples and 6.4 s at 2 000 on an Apple M4 Pro
    CPU, while sliced W2 with 500 directions took 0.5 s at 10 000.

References:
    :cite:t:`rabin2012wasserstein`; :cite:t:`bonneel2015sliced`;
    :cite:t:`nadjahi2020statistical`; :cite:t:`cramer1936some`;
    :cite:t:`muller1959note`; :cite:t:`cuturi2022optimal`;
    :cite:t:`tran2026minimax`; :cite:t:`janati2020entropic`.
"""

import functools
from typing import Final

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from jaxtyping import Array, Bool, Float, PRNGKeyArray
from ott.geometry import costs, pointcloud
from ott.tools import sinkhorn_divergence as ott_divergence
from ott.tools import sliced

from mixed_signals.core.samples import _is_concrete
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
"""Directions of sliced W2, drawn once per comparison.

In a zoo check with 1 000 samples per posterior, 100 and 1 000 directions
gave the same resolution in 2, 8 and 16 dimensions; how many directions a
dimension needs is left to a later benchmark.
"""

SLICED_SINKHORN_DIRECTIONS: Final = 20
"""Directions of the sliced Sinkhorn divergence, each a Sinkhorn problem over N² pairs.

Nadjahi et al. (2020) used 10 directions on images and 100 on synthetic
data.
"""

SLICED_SINKHORN_CAP: Final = 2_000
"""Largest number of samples per posterior of the sliced Sinkhorn divergence.

With 20 directions in 16 dimensions, one call took 6.4 s at this cap on an
Apple M4 Pro CPU, so 200 null draws take about 21 minutes; the common cap of
10 000 would take about 25 times longer.
"""

EPSILON_SAMPLE_SIZE: Final = 500
"""Samples per posterior whose pairs set the median of :func:`sliced_sinkhorn_epsilon`.

With 20 directions, 500 samples per posterior give about 10⁷ projected
squared distances, which bounds the time and memory of the median.
"""


def sliced_directions(
    space: ParamSpace, count: int, *, key: PRNGKeyArray
) -> Float[Array, "direction _"]:
    """Return directions drawn uniformly on the unit sphere of the embedded space.

    Args:
        space: Space of the samples; the directions have one coordinate per
            embedded coordinate, E + 2C + 3S.
        count: Number of directions, at least one.
        key: JAX PRNG key, derived with
            :func:`~mixed_signals.prng.derive_key` from the path of the
            comparison, ending in ``"directions"``.

    Returns:
        Unit vectors shaped (count, E + 2C + 3S).

    Raises:
        ValueError: If ``count`` is below one.

    References:
        :cite:t:`muller1959note`.
    """
    if count < 1:
        raise ValueError(f"count must be at least 1, got {count}")
    normals = jax.random.normal(key, (count, embedding_size(space)))
    lengths: Array = jnp.linalg.norm(normals, axis=1, keepdims=True)
    return normals / lengths


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
    if _is_concrete(epsilon_array) and not (
        bool(jnp.isfinite(epsilon_array)) and float(epsilon_array) > 0
    ):
        raise ValueError(
            f"epsilon must be positive and finite, got {float(epsilon_array)}"
        )
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
_sliced_sinkhorn = _jit(_sliced_sinkhorn_kernel, "space")
_projected_median = _jit(_projected_median_kernel, "space")
