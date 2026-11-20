"""k-nearest-neighbour estimator: topology-aware neighbour distances and log densities.

:class:`NearestNeighbours` holds the samples of one posterior and measures
distances in their space: Euclidean on the Euclidean block, the wrapped
difference along the shorter arc on circles and the great-circle arc on
spheres, the angle times the sphere's radius, combined as the root of the
summed squares. The space is then locally Euclidean with d = E + C + 2S
dimensions, for E Euclidean axes, C circles and S spheres.

Maximum norm
    With ``norm="max"`` the distance is instead the largest of the distances
    along each Euclidean axis, each circle and each sphere, the norm of the
    mutual-information estimator of Kraskov, Stögbauer and Grassberger
    (2004). Its unit ball is a product of intervals and discs, of volume
    2^(E+C) π^S, which the log density takes instead of the Euclidean ball.
Neighbour counts
    :meth:`NearestNeighbours.count_within` counts the samples strictly
    closer to each point than a radius of its own, and
    :meth:`NearestNeighbours.leave_one_out_count` does so for each sample,
    leaving the sample itself out. Both compare the same floating-point
    distances as the searches, so a neighbour that sets a radius in one
    space falls outside it, as the estimator of Kraskov, Stögbauer and
    Grassberger (2004) counts.

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
    measure area, so the log of the area element of the polar form, R² times
    that of the unit sphere for a radius R, is added to refer the density to
    the coordinates (Mardia and Jupp 2000).
Weights
    The nearest-neighbour estimators assume equally weighted, independent
    draws (Wang, Kulkarni and Verdú 2009). Samples with zero weight leave
    every search, as the zero-weight padding of unequal sample counts needs.
    Unequal positive weights give NaN and a
    :class:`~mixed_signals.errors.ValidityWarning`; convert such a set with
    :func:`~mixed_signals.core.samples.rejection_subset` first, which keeps
    each sample with probability wᵢ/max w and makes no copies, unlike
    resampling with replacement (Smith and Gelfand 1992; Ashton et al. 2019).
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
    :cite:t:`smith1992bayesian`; :cite:t:`ashton2019bilby`;
    :cite:t:`mardia2000directional`.
"""

import functools
import math
from typing import ClassVar, Final, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.scipy.special import digamma
from jax.typing import ArrayLike
from jaxtyping import Array, Bool, Float
from pydantic import PositiveInt

from mixed_signals.core.samples import _check_real, _is_concrete, unwrap_about
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

EUCLIDEAN_NORM: Final = "euclidean"
"""Norm that combines the distances along the axes as their root summed square."""

MAXIMUM_NORM: Final = "max"
"""Norm that takes the largest of the distances along the axes."""


class NearestNeighbourSettings(EstimatorSettings):
    """Settings of the k-nearest-neighbour estimator.

    Attributes:
        neighbours: The rank k of the neighbour whose distance enters the
            estimates. Each metric sets its own default.
        norm: ``"euclidean"`` combines the distances along the Euclidean
            axes, circles and spheres as the root of their summed squares;
            ``"max"`` takes the largest of them.

    References:
        :cite:t:`loftsgaarden1965nonparametric`; :cite:t:`kraskov2004mutual`.
    """

    name: ClassVar[str] = "knn"

    neighbours: PositiveInt
    norm: Literal["euclidean", "max"] = EUCLIDEAN_NORM


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
            :cite:t:`wang2009divergence`; :cite:t:`smith1992bayesian`;
            :cite:t:`ashton2019bilby`.
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
                    " convert the set with mixed_signals.core.samples.rejection_subset"
                    " first"
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

    def count_within(
        self,
        points: Float[ArrayLike, "*leading param"],
        radii: ArrayLike,
    ) -> Float[Array, " *leading"]:
        """Return the number of samples strictly within each point's radius.

        Args:
            points: Points shaped (..., D), in the coordinates of the space.
            radii: One radius per point, shaped (...), in the distance of
                the settings' norm.

        Returns:
            The counts as floats, shaped (...); NaN where the radius is NaN
            or the estimator is invalid.

        Raises:
            ValueError: If the points do not end in the D parameters, or the
                radii do not have the leading shape of the points.
            TypeError: If the points are not real-valued.

        References:
            :cite:t:`kraskov2004mutual`.
        """
        values = _read_points(points, len(self.space))
        radii_array = _read_radii(radii, values.shape[:-1])
        flat = values.reshape(-1, values.shape[-1])
        counts = _counts(self, flat, radii_array.reshape(-1))
        return counts.reshape(values.shape[:-1])

    def leave_one_out_count(self, radii: ArrayLike) -> Float[Array, " sample"]:
        """Return the number of other samples strictly within each sample's radius.

        Each sample leaves its own count, but copies of it stay.

        Args:
            radii: One radius per sample, shaped (N,), in the distance of the
                settings' norm.

        Returns:
            The counts as floats, shaped (N,); NaN where the radius is NaN or
            the estimator is invalid.

        Raises:
            ValueError: If the radii are not shaped (N,).

        References:
            :cite:t:`kraskov2004mutual`.
        """
        radii_array = _read_radii(radii, self.samples.shape[:1])
        return _leave_one_out_counts(self, radii_array)


def _comparable_distances(
    space: ParamSpace,
    norm: str,
    queries: Float[Array, "*leading param"],
    samples: Float[Array, "sample param"],
) -> Float[Array, "*leading sample"]:
    """Return values that order as the distances: squared ones for the Euclidean norm.

    For the maximum norm the values are the distances themselves, the largest
    distance along any Euclidean axis, circle or sphere.
    """
    euclidean = norm == EUCLIDEAN_NORM
    total = jnp.zeros((queries.shape[0], samples.shape[0]))
    for component in _axis_distances(space, queries, samples, squared=euclidean):
        total = total + component if euclidean else jnp.maximum(total, component)
    return total


def _axis_distances(
    space: ParamSpace,
    queries: Float[Array, "*leading param"],
    samples: Float[Array, "sample param"],
    *,
    squared: bool,
) -> list[Float[Array, "*leading sample"]]:
    """Return the distances along the Euclidean block, each circle and each sphere.

    Squared distances sum the whole Euclidean block into one term; plain ones
    keep each Euclidean axis apart, for the maximum norm.
    """
    blocks = space.axis_blocks
    components = []
    if blocks.euclidean:
        euclidean = list(blocks.euclidean)
        differences = queries[:, None, euclidean] - samples[None, :, euclidean]
        if squared:
            components.append(jnp.sum(differences**2, axis=-1))
        else:
            components.extend(jnp.moveaxis(jnp.abs(differences), -1, 0))
    for position in blocks.circles:
        lower, upper = space.parameters[position].bounds
        arcs = unwrap_about(
            queries[:, None, position] - samples[None, :, position], 0.0, upper - lower
        )
        components.append(arcs**2 if squared else jnp.abs(arcs))
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
        arcs = sphere.radius * jnp.arctan2(cross, dot)
        components.append(arcs**2 if squared else arcs)
    return components


def _comparable_radii(radii: Array, norm: str) -> Array:
    """Return radii in the form of :func:`_comparable_distances`."""
    return radii**2 if norm == EUCLIDEAN_NORM else radii


def _kth_distance(
    values: Float[Array, "*leading sample"], neighbours: int, norm: str
) -> Float[Array, " *leading"]:
    """Return the k-th smallest distance in each row of comparable distances."""
    smallest, _ = jax.lax.top_k(-values, neighbours)
    kth = -smallest[..., neighbours - 1]
    return jnp.sqrt(kth) if norm == EUCLIDEAN_NORM else kth


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
    norm = estimator.settings.norm
    values = _comparable_distances(estimator.space, norm, chunk, estimator.samples)
    return _kth_distance(
        jnp.where(estimator.present, values, jnp.inf),
        estimator.settings.neighbours,
        norm,
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
    norm = estimator.settings.norm
    values = _comparable_distances(
        estimator.space, norm, chunk[:, :size], estimator.samples
    )
    own = jnp.arange(sample_count)[None, :] == chunk[:, size:].astype(int)
    values = jnp.where(estimator.present & ~own, values, jnp.inf)
    return _kth_distance(values, estimator.settings.neighbours, norm)


def _counts_kernel(
    estimator: NearestNeighbours,
    points: Float[Array, "*leading param"],
    radii: Float[Array, " *leading"],
) -> Float[Array, " *leading"]:
    """Return the number of samples strictly within each point's radius."""
    # * The radius rides along as a last column of its point.
    indexed = jnp.concatenate([points, radii[:, None]], axis=1)
    counts = _map_in_chunks(
        functools.partial(_chunk_counts, estimator), indexed, _chunk_size(estimator)
    )
    return jnp.where(estimator.valid & ~jnp.isnan(radii), counts, jnp.nan)


def _chunk_counts(
    estimator: NearestNeighbours, chunk: Float[Array, "*leading param"]
) -> Float[Array, " *leading"]:
    """Return the counts of one chunk of points with their radii."""
    size = estimator.samples.shape[1]
    norm = estimator.settings.norm
    values = _comparable_distances(
        estimator.space, norm, chunk[:, :size], estimator.samples
    )
    inside = values < _comparable_radii(chunk[:, size], norm)[:, None]
    return jnp.sum(estimator.present & inside, axis=1).astype(jnp.float64)


def _leave_one_out_counts_kernel(
    estimator: NearestNeighbours, radii: Float[Array, " sample"]
) -> Float[Array, " sample"]:
    """Return the number of other samples strictly within each sample's radius."""
    sample_count = estimator.samples.shape[0]
    indexed = jnp.concatenate(
        [
            estimator.samples,
            jnp.arange(sample_count, dtype=jnp.float64)[:, None],
            radii[:, None],
        ],
        axis=1,
    )
    counts = _map_in_chunks(
        functools.partial(_chunk_leave_one_out_counts, estimator),
        indexed,
        _chunk_size(estimator),
    )
    return jnp.where(estimator.valid & ~jnp.isnan(radii), counts, jnp.nan)


def _chunk_leave_one_out_counts(
    estimator: NearestNeighbours, chunk: Float[Array, "*leading param"]
) -> Float[Array, " *leading"]:
    """Return the counts of indexed samples with their radii, each one skipped."""
    sample_count, size = estimator.samples.shape
    norm = estimator.settings.norm
    values = _comparable_distances(
        estimator.space, norm, chunk[:, :size], estimator.samples
    )
    own = jnp.arange(sample_count)[None, :] == chunk[:, size : size + 1].astype(int)
    inside = values < _comparable_radii(chunk[:, size + 1], norm)[:, None]
    return jnp.sum(estimator.present & ~own & inside, axis=1).astype(jnp.float64)


def _log_unit_ball(estimator: NearestNeighbours) -> float:
    """Return ln V of the unit ball: Euclidean, or of intervals and discs (max norm)."""
    dimension = estimator.dimension
    if estimator.settings.norm == EUCLIDEAN_NORM:
        return 0.5 * dimension * math.log(math.pi) - math.lgamma(0.5 * dimension + 1)
    blocks = estimator.space.axis_blocks
    lines = len(blocks.euclidean) + len(blocks.circles)
    return lines * math.log(2) + len(blocks.spheres) * math.log(math.pi)


def _log_density_from_kernel(
    estimator: NearestNeighbours,
    points: Float[Array, "*leading param"],
    distances: Float[Array, " *leading"],
    others: Array,
) -> Float[Array, " *leading"]:
    """Return ψ(k) − ψ(n + 1) − ln V_d − d ln r_k plus the log area elements."""
    dimension = estimator.dimension
    neighbours = estimator.settings.neighbours
    others_float = jnp.asarray(others, dtype=jnp.float64)
    log_values = (
        digamma(float(neighbours))
        - digamma(others_float + 1)
        - _log_unit_ball(estimator)
        - dimension * jnp.log(distances)
    )
    blocks = estimator.space.axis_blocks
    for (azimuth, polar), sphere in zip(
        blocks.spheres, estimator.space.spheres, strict=True
    ):
        _, log_area, inside = unit_vectors(
            points[:, azimuth], points[:, polar], sphere.polar_form
        )
        log_area = log_area + 2 * math.log(sphere.radius)
        log_values = jnp.where(inside, log_values + log_area, -jnp.inf)
    return jnp.where(distances > 0, log_values, jnp.nan)


def _read_radii(radii: ArrayLike, shape: tuple[int, ...]) -> Array:
    """Return radii as float64 after checking that they have the given shape."""
    array = jnp.asarray(radii)
    _check_real("radii", array)
    if array.shape != shape:
        raise ValueError(f"radii must be shaped {shape}, got shape {array.shape}")
    return array.astype(jnp.float64)


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
_counts = _jit(_counts_kernel)
_leave_one_out_counts = _jit(_leave_one_out_counts_kernel)
