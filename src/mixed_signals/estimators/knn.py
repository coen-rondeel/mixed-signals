"""k-nearest-neighbour estimator: topology-aware neighbour distances and log densities.

:class:`NearestNeighbours` holds the samples of one posterior and measures
distances in their space: Euclidean on the Euclidean block, the wrapped
difference along the shorter arc on circles and the great-circle angle on
spheres, combined as the root of the summed squares. The space is then
locally Euclidean with d = E + C + 2S dimensions, for E Euclidean axes, C
circles and S spheres.

Log density
    log p̂(x) = ψ(k) − ψ(n + 1) − ln V_d − d ln r_k(x), with r_k(x) the
    distance from x to the k-th nearest of n samples and
    V_d = π^(d/2)/Γ(d/2 + 1) the volume of the unit ball. The probability
    in the ball of radius r_k follows a Beta(k, n − k + 1) distribution,
    whose log has the mean ψ(k) − ψ(n + 1). The estimate is therefore the
    nearest-neighbour density of Loftsgaarden and Quesenberry (1965) with
    the correction that makes the mean of −log p̂ over the samples the
    entropy estimate of Kozachenko and Leonenko (1987), in the form of
    Kraskov, Stögbauer and Grassberger (2004). At a sample, the sample
    leaves its own search, so n is one fewer. On a sphere, the distances
    measure area, so the log of the area element of the polar form is added
    to refer the density to the coordinates (Mardia and Jupp 2000).
Weights
    The nearest-neighbour estimators assume equally weighted, independent
    draws (Wang, Kulkarni and Verdú 2009). Samples with zero weight leave
    every search, as the zero-weight padding of unequal sample counts needs.
    Unequal positive weights give NaN and a
    :class:`~mixed_signals.errors.ValidityWarning`; resample such a set in
    proportion to its weights first (Smith and Gelfand 1992).
Validity
    Fewer samples with positive weight than k, besides the point itself,
    give NaN. So does a log density whose k-th distance is zero, because k or
    more samples coincide with the point, as repeated states of a Markov
    chain do. Both raise a :class:`~mixed_signals.errors.ValidityWarning`
    for concrete inputs.
Cost
    The search is brute force, in chunks of about 2²² distances, so time
    grows as the product of the numbers of points and samples and memory
    stays bounded. Metrics subsample above their cap first.

References:
    :cite:t:`loftsgaarden1965nonparametric`; :cite:t:`kozachenko1987sample`;
    :cite:t:`kraskov2004mutual`; :cite:t:`wang2009divergence`;
    :cite:t:`smith1992bayesian`; :cite:t:`mardia2000directional`.
"""

import functools
import math
from typing import ClassVar, Final

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.scipy.special import digamma
from jax.typing import ArrayLike
from jaxtyping import Array, Bool, Float
from pydantic import PositiveInt

from mixed_signals.core.samples import _is_concrete, unwrap_about
from mixed_signals.core.space import ParamSpace
from mixed_signals.core.topology import unit_vectors
from mixed_signals.estimators.base import (
    CHUNK_ENTRIES,
    Estimator,
    EstimatorSettings,
    _jit,
    _map_in_chunks,
    _read_points,
    _read_samples,
    _warn,
)

WEIGHT_TOLERANCE: Final = 1e-9
"""Largest spread of the positive log weights that still counts as equal weights."""


class NearestNeighbourSettings(EstimatorSettings):
    """Settings of the k-nearest-neighbour estimator.

    Attributes:
        neighbours: The rank k of the neighbour whose distance enters the
            estimates. Each metric sets its own default.

    References:
        :cite:t:`loftsgaarden1965nonparametric`.
    """

    name: ClassVar[str] = "knn"

    neighbours: PositiveInt


class NearestNeighbours(Estimator):
    """Nearest-neighbour distances and log densities of one posterior.

    Build it with :meth:`fit`.

    Attributes:
        samples: The samples, shaped (N, D).
        present: Whether each sample has a positive weight and so takes part
            in the searches, shaped (N,).
        valid: Whether the positive weights are equal and outnumber k, as
            the estimates need.
        space: Parameter space of the samples.

    References:
        :cite:t:`loftsgaarden1965nonparametric`; :cite:t:`kozachenko1987sample`.
    """

    samples: Float[Array, "sample param"]
    present: Bool[Array, " sample"]
    valid: Bool[Array, ""]
    space: ParamSpace = eqx.field(static=True)
    settings: NearestNeighbourSettings = eqx.field(static=True)

    @classmethod
    def fit(
        cls,
        samples: ArrayLike,
        log_weights: ArrayLike | None,
        space: ParamSpace,
        settings: NearestNeighbourSettings,
    ) -> "NearestNeighbours":
        """Hold the samples of one posterior for nearest-neighbour searches.

        Args:
            samples: Samples in the parameter order of ``space``, shaped
                (N, D).
            log_weights: Log weights shaped (N,); equal weights when
                ``None``. Samples with zero weight leave every search.
            space: Parameter space of the samples.
            settings: The settings, with the rank k.

        Returns:
            The estimator; NaN in every estimate if the positive weights are
            unequal or do not outnumber k.

        Raises:
            ValueError: If the shapes do not fit the space, or concrete values
                are invalid.
            TypeError: If an input is not real-valued.

        Warns:
            ValidityWarning: If concrete positive weights are unequal, or no
                more than k samples have a positive weight.
            PrecisionWarning: If an input is narrower than float64.

        References:
            :cite:t:`wang2009divergence`; :cite:t:`smith1992bayesian`.
        """
        samples_array, weights_array = _read_samples(samples, log_weights, space)
        present = weights_array > -jnp.inf
        positive = jnp.where(present, weights_array, 0.0)
        spread = jnp.max(jnp.where(present, positive, -jnp.inf)) - jnp.min(
            jnp.where(present, positive, jnp.inf)
        )
        count = jnp.sum(present)
        equal = spread <= WEIGHT_TOLERANCE
        enough = count > settings.neighbours
        if _is_concrete(samples_array, weights_array):
            if not bool(equal):
                _warn(
                    "nearest-neighbour estimates need equally weighted samples, but"
                    f" the positive log weights differ by up to {float(spread):.3g};"
                    " resample the set in proportion to its weights first"
                )
            if not bool(enough):
                _warn(
                    "nearest-neighbour estimates with k ="
                    f" {settings.neighbours} need more than k samples with positive"
                    f" weight, got {int(count)}, so they are NaN"
                )
        return cls(
            samples=samples_array,
            present=present,
            valid=equal & enough,
            space=space,
            settings=settings,
        )

    @property
    def dimension(self) -> int:
        """Dimension d of the space: Euclidean axes, circles and twice the spheres."""
        blocks = self.space.axis_blocks
        return len(blocks.euclidean) + len(blocks.circles) + 2 * len(blocks.spheres)

    def neighbour_distance(
        self, points: Float[ArrayLike, "*leading param"]
    ) -> Float[Array, " *leading"]:
        """Return the distance from each point to its k-th nearest sample.

        Args:
            points: Points shaped (..., D), in the coordinates of the space.

        Returns:
            The distances r_k, shaped (...); NaN if the estimator is invalid.

        Raises:
            ValueError: If the points do not end in the D parameters.
            TypeError: If the points are not real-valued.
        """
        values = _read_points(points, len(self.space))
        flat = values.reshape(-1, values.shape[-1])
        return _distances(self, flat).reshape(values.shape[:-1])

    def leave_one_out_distance(self) -> Float[Array, " sample"]:
        """Return the distance from each sample to its k-th nearest other sample.

        Each sample leaves its own search, but copies of it stay.

        Returns:
            The distances, shaped (N,); NaN if the estimator is invalid.
        """
        return _leave_one_out_distances(self)

    def log_density(
        self, points: Float[ArrayLike, "*leading param"]
    ) -> Float[Array, " *leading"]:
        """Return the nearest-neighbour log density at points.

        Args:
            points: Points shaped (..., D), in the coordinates of the space.

        Returns:
            ψ(k) − ψ(n + 1) − ln V_d − d ln r_k, plus the log area element
            on spheres, shaped (...); NaN where r_k is zero or the estimator
            is invalid.

        Raises:
            ValueError: If the points do not end in the D parameters.
            TypeError: If the points are not real-valued.

        Warns:
            ValidityWarning: If r_k is zero at concrete points.

        References:
            :cite:t:`loftsgaarden1965nonparametric`;
            :cite:t:`kraskov2004mutual`.
        """
        values = _read_points(points, len(self.space))
        flat = values.reshape(-1, values.shape[-1])
        distances = _distances(self, flat)
        others = jnp.sum(self.present)
        log_values = _log_density_from(self, flat, distances, others)
        if _is_concrete(log_values):
            _warn_about_coincidences(distances, self.settings.neighbours, "points")
        return log_values.reshape(values.shape[:-1])

    def leave_one_out_log_density(self) -> Float[Array, " sample"]:
        """Return the log density at each sample, from the other samples.

        The mean of the negated values over the samples is the
        Kozachenko-Leonenko entropy estimate.

        Returns:
            The log densities, shaped (N,); NaN where the k-th distance is zero
            or the estimator is invalid.

        Warns:
            ValidityWarning: If a k-th distance is zero for concrete samples.

        References:
            :cite:t:`kozachenko1987sample`; :cite:t:`kraskov2004mutual`.
        """
        distances = _leave_one_out_distances(self)
        # * A sample with zero weight is outside the set, so it leaves no one out.
        others = jnp.sum(self.present) - self.present
        log_values = _log_density_from(self, self.samples, distances, others)
        if _is_concrete(log_values):
            _warn_about_coincidences(distances, self.settings.neighbours, "samples")
        return log_values


def _squared_distances(
    space: ParamSpace,
    queries: Float[Array, "*leading param"],
    samples: Float[Array, "sample param"],
) -> Float[Array, "*leading sample"]:
    """Return squared topology-aware distances between queries (Q, D) and samples."""
    blocks = space.axis_blocks
    total = jnp.zeros((queries.shape[0], samples.shape[0]))
    if blocks.euclidean:
        euclidean = list(blocks.euclidean)
        differences = queries[:, None, euclidean] - samples[None, :, euclidean]
        total = total + jnp.sum(differences**2, axis=-1)
    for position in blocks.circles:
        lower, upper = space.parameters[position].bounds
        differences = unwrap_about(
            queries[:, None, position] - samples[None, :, position], 0.0, upper - lower
        )
        total = total + differences**2
    for (azimuth, polar), sphere in zip(blocks.spheres, space.spheres, strict=True):
        query_units, _, _ = unit_vectors(
            queries[:, azimuth], queries[:, polar], sphere.polar_form
        )
        sample_units, _, _ = unit_vectors(
            samples[:, azimuth], samples[:, polar], sphere.polar_form
        )
        # * atan2 of the cross and dot products keeps small angles precise.
        cross = jnp.linalg.norm(
            jnp.cross(query_units[:, None, :], sample_units[None, :, :]), axis=-1
        )
        dot = jnp.sum(query_units[:, None, :] * sample_units[None, :, :], axis=-1)
        total = total + jnp.arctan2(cross, dot) ** 2
    return total


def _kth_distance(
    squared: Float[Array, "*leading sample"], neighbours: int
) -> Float[Array, " *leading"]:
    """Return the k-th smallest distance in each row of squared distances."""
    smallest, _ = jax.lax.top_k(-squared, neighbours)
    return jnp.sqrt(-smallest[..., neighbours - 1])


def _chunk_size(estimator: NearestNeighbours) -> int:
    """Return the points per chunk that keep a chunk near the distance budget."""
    sample_count, size = estimator.samples.shape
    return max(1, CHUNK_ENTRIES // (sample_count * size))


def _distances_kernel(
    estimator: NearestNeighbours, points: Float[Array, "*leading param"]
) -> Float[Array, " *leading"]:
    """Return the k-th neighbour distance of points shaped (M, D)."""
    if estimator.settings.neighbours > estimator.samples.shape[0]:
        return jnp.full(points.shape[0], jnp.nan)
    distances = _map_in_chunks(
        functools.partial(_chunk_distances, estimator), points, _chunk_size(estimator)
    )
    return jnp.where(estimator.valid, distances, jnp.nan)


def _chunk_distances(
    estimator: NearestNeighbours, chunk: Float[Array, "*leading param"]
) -> Float[Array, " *leading"]:
    """Return the k-th neighbour distance of one chunk of points."""
    squared = _squared_distances(estimator.space, chunk, estimator.samples)
    return _kth_distance(
        jnp.where(estimator.present, squared, jnp.inf), estimator.settings.neighbours
    )


def _leave_one_out_distances_kernel(
    estimator: NearestNeighbours,
) -> Float[Array, " sample"]:
    """Return the k-th neighbour distance of every sample, itself excluded."""
    sample_count = estimator.samples.shape[0]
    if estimator.settings.neighbours >= sample_count:
        return jnp.full(sample_count, jnp.nan)
    # * The index rides along as a last column, so a chunk knows which sample to skip.
    indexed = jnp.concatenate(
        [estimator.samples, jnp.arange(sample_count, dtype=jnp.float64)[:, None]],
        axis=1,
    )
    distances = _map_in_chunks(
        functools.partial(_chunk_leave_one_out, estimator),
        indexed,
        _chunk_size(estimator),
    )
    return jnp.where(estimator.valid, distances, jnp.nan)


def _chunk_leave_one_out(
    estimator: NearestNeighbours, chunk: Float[Array, "*leading param"]
) -> Float[Array, " *leading"]:
    """Return the k-th neighbour distance of indexed samples, each one skipped."""
    sample_count, size = estimator.samples.shape
    squared = _squared_distances(estimator.space, chunk[:, :size], estimator.samples)
    own = jnp.arange(sample_count)[None, :] == chunk[:, size:].astype(int)
    squared = jnp.where(estimator.present & ~own, squared, jnp.inf)
    return _kth_distance(squared, estimator.settings.neighbours)


def _log_density_from_kernel(
    estimator: NearestNeighbours,
    points: Float[Array, "*leading param"],
    distances: Float[Array, " *leading"],
    others: Array,
) -> Float[Array, " *leading"]:
    """Return ψ(k) − ψ(n + 1) − ln V_d − d ln r_k plus the log area elements."""
    dimension = estimator.dimension
    neighbours = estimator.settings.neighbours
    log_unit_ball = 0.5 * dimension * math.log(math.pi) - math.lgamma(
        0.5 * dimension + 1
    )
    others_float = jnp.asarray(others, dtype=jnp.float64)
    log_values = (
        digamma(float(neighbours))
        - digamma(others_float + 1)
        - log_unit_ball
        - dimension * jnp.log(distances)
    )
    blocks = estimator.space.axis_blocks
    for (azimuth, polar), sphere in zip(
        blocks.spheres, estimator.space.spheres, strict=True
    ):
        _, log_area, inside = unit_vectors(
            points[:, azimuth], points[:, polar], sphere.polar_form
        )
        log_values = jnp.where(inside, log_values + log_area, -jnp.inf)
    return jnp.where(distances > 0, log_values, jnp.nan)


def _warn_about_coincidences(distances: Array, neighbours: int, noun: str) -> None:
    """Warn when concrete k-th neighbour distances are zero."""
    zero_count = int(jnp.sum(distances == 0))
    if zero_count:
        _warn(
            f"the k-th neighbour distance is zero for {zero_count} {noun}: at least"
            f" k = {neighbours} samples coincide with each, as repeated states of a"
            " Markov"
            " chain do, so their log densities are NaN; thin the chains or remove"
            " repeated samples"
        )


_distances = _jit(_distances_kernel)
_leave_one_out_distances = _jit(_leave_one_out_distances_kernel)
_log_density_from = _jit(_log_density_from_kernel)
