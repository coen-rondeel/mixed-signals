"""Run-time calibration of the σ_eff map of kNN KL rows.

The nearest-neighbour estimate of the KL divergence (Wang, Kulkarni and Verdú
2009) reads low from four dimensions on in the zoo check below, by a ratio that
depends on the dimension, the sample counts and the shift. The developers chose
to measure the curve of a row at its own dimension, sample counts and neighbour
rank at run time, instead of storing a table over those three.

Curve
    :func:`knn_kl_curve` runs the estimator on unit Gaussians N(0, I) at the
    dimension, the sample counts and the neighbour rank of the row, for a few
    shifts δ along one axis, and gives the mean estimate at each shift minus
    its mean at δ = 0, the bias under the null. This is the null-subtracted
    value that a row compares with its own, so
    :func:`~mixed_signals.sigma.maps.sigma_eff` inverts the curve to find the
    shift that gives the value of the row.
Common random numbers
    Every replicate draws two sets of normal deviates, and every shift moves
    the same second set, so the curve is smooth in δ and its noise does not
    scatter the order of the points.
Monotone curve
    The mean estimate is not monotone in δ at small shifts, where noise
    outweighs the shift. The values take a running maximum over the grid, which
    starts at the point (0, 0), so that the curve can be inverted.
Dimension
    The dimension is E + C + 2S for E Euclidean axes, C circles and S spheres.
    The calibration runs on that many real parameters, so it does not carry
    the topology of the row's space.
Cost
    With the defaults, (shifts + 1) · replicates = 28 evaluations of the
    estimator, next to the 200 evaluations of the null draws.
Zoo check
    On unit Gaussians shifted by 0.5, 1, 2 and 3σ, with k = 3, the
    null-subtracted estimate over the truth δ²/2 was, in one dimension,
    1.01 / 0.99 / 0.93 / 0.81 at N = 1 000 and 1.01 / 0.99 / 0.97 / 0.90 at
    N = 10 000. In four dimensions it was 0.89 / 0.83 / 0.77 / 0.68 at
    N = 1 000, 0.95 / 0.90 / 0.83 / 0.74 at N = 3 000 and
    0.90 / 0.91 / 0.87 / 0.80 at N = 10 000. In 16 dimensions it was
    0.80 / 0.71 / 0.66 / 0.62 at N = 1 000, 0.64 / 0.68 / 0.66 / 0.63 at
    N = 3 000 and 0.72 / 0.71 / 0.69 / 0.66 at N = 10 000. So √(2 KL) reads 5
    to 20% low in σ_eff, and the ratio depends on the dimension, the sample
    count and the shift. One curve of four replicates at D = 4 and N = 1 000
    read 0.75 of the true KL at a shift of 1, where the average over 24
    replicates is 0.83. With such a curve, the calibrated σ_eff of six
    comparisons was 1.028 ± 0.044 and 2.071 ± 0.028 at shifts of 1 and 2,
    against 0.920 ± 0.039 and 1.772 ± 0.023 for √(2 KL) (see
    :ref:`validation-knn-kl-calibration`).

References:
    :cite:t:`wang2009divergence`.
"""

from collections.abc import Sequence
from typing import Final, NamedTuple

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from mixed_signals._config import SAMPLE_CAP
from mixed_signals.core.space import ParamSpace
from mixed_signals.metrics.divergence import KL_NEIGHBOURS, knn_kl_divergence
from mixed_signals.prng import derive_key

KNN_KL_SHIFTS: Final = (0.25, 0.5, 1.0, 2.0, 3.0, 4.0)
"""Shifts δ, in the σ of the unit Gaussians, at which the curve is measured.

The grid runs from near the default σ_eff threshold of 0.2 to 4σ, one point
beyond the largest shift of the zoo check of the module description; past the
last point :func:`~mixed_signals.sigma.maps.sigma_eff` carries the last ratio
of shift to √value on.
"""

KNN_KL_REPLICATES: Final = 4
"""Replicates of the unit Gaussians that each point of the curve averages.

The cost is (shifts + 1) · replicates = 28 evaluations of the estimator with
the default shifts, next to the 200 evaluations of the null draws.
"""


class CalibrationCurve(NamedTuple):
    """The null-subtracted estimate of a metric against the shift that it measures."""

    shifts: Float[Array, " grid"]
    """The shifts δ, increasing, starting at 0."""
    values: Float[Array, " grid"]
    """The mean estimate at each shift minus its mean at 0, non-decreasing, from 0."""


def knn_kl_curve(
    dimension: int,
    first_count: int,
    second_count: int,
    *,
    key: PRNGKeyArray,
    neighbours: int = KL_NEIGHBOURS,
    shifts: Sequence[float] = KNN_KL_SHIFTS,
    replicates: int = KNN_KL_REPLICATES,
    cap: int = SAMPLE_CAP,
) -> CalibrationCurve:
    """Measure the null-subtracted kNN KL estimate of unit Gaussians by shift.

    Replicate r draws ``first_count`` samples of N(0, I) from
    ``derive_key(key, "first", r)`` and ``second_count`` samples ``base`` from
    ``derive_key(key, "second", r)``. The value at the shift δ is the mean over
    the replicates of the estimate of KL(first ‖ base + δ e₁), with e₁ the first
    axis, minus the same mean at δ = 0. The values then take a running maximum
    and the point (0, 0) leads the grid. Call it outside :func:`jax.jit`.

    Args:
        dimension: Dimension E + C + 2S of the row's space, E Euclidean axes,
            C circles and S spheres; the estimator runs on that many real
            parameters.
        first_count: Number of samples of the first posterior of the row.
        second_count: Number of samples of the second posterior of the row.
        key: JAX PRNG key, derived with :func:`~mixed_signals.prng.derive_key`
            from the path of the row, ending in ``"calibration"``.
        neighbours: The rank k of the estimator of the row.
        shifts: Positive, strictly increasing shifts δ at which to measure.
        replicates: Replicates to average.
        cap: Largest number of samples per posterior of the estimator.

    Returns:
        The shifts and the values, each shaped (len(shifts) + 1,). A value is
        NaN where the estimator is undefined, for example for fewer samples
        than neighbours.

    Raises:
        TypeError: If the dimension, a count or the number of replicates is
            not an integer.
        ValueError: If the dimension, a count or the number of replicates is
            below one, or the shifts are not positive, finite and strictly
            increasing.

    Warns:
        SubsamplingWarning: If a count exceeds the cap, as the estimator
            subsamples it.

    References:
        :cite:t:`wang2009divergence`.
    """
    for name, count in (
        ("dimension", dimension),
        ("first_count", first_count),
        ("second_count", second_count),
        ("replicates", replicates),
    ):
        if isinstance(count, bool) or not isinstance(count, int):
            raise TypeError(f"{name} must be an integer, got {type(count).__name__}")
        if count < 1:
            raise ValueError(f"{name} must be at least 1, got {count}")
    grid = jnp.asarray(shifts, dtype=jnp.float64)
    if (
        grid.ndim != 1
        or grid.size == 0
        or not bool(jnp.all(jnp.isfinite(grid) & (grid > 0)))
        or not bool(jnp.all(jnp.diff(grid) > 0))
    ):
        raise ValueError(
            f"shifts must be positive, finite and strictly increasing, got {shifts!r}"
        )
    space = ParamSpace.model_validate(
        {"parameters": [{"name": f"x{index}"} for index in range(dimension)]}
    )
    full_grid = jnp.concatenate([jnp.zeros(1), grid])
    estimates = jnp.stack(
        [
            _estimates_of_one_replicate(
                space,
                full_grid,
                first_count,
                second_count,
                key=key,
                replicate=replicate,
                neighbours=neighbours,
                cap=cap,
            )
            for replicate in range(replicates)
        ]
    )
    means = jnp.mean(estimates, axis=0)
    # * The running maximum starts at the point (0, 0), so no value is negative.
    values = jax.lax.cummax(means - means[0], axis=0)
    return CalibrationCurve(shifts=full_grid, values=values)


def _estimates_of_one_replicate(
    space: ParamSpace,
    shifts: Float[Array, " grid"],
    first_count: int,
    second_count: int,
    *,
    key: PRNGKeyArray,
    replicate: int,
    neighbours: int,
    cap: int,
) -> Float[Array, " grid"]:
    """Return the estimate of one replicate at every shift, from common deviates."""
    dimension = len(space)
    first = jax.random.normal(
        derive_key(key, "first", replicate), (first_count, dimension)
    )
    base = jax.random.normal(
        derive_key(key, "second", replicate), (second_count, dimension)
    )
    subsample_key = derive_key(key, "subsample", replicate)
    return jnp.stack(
        [
            knn_kl_divergence(
                first,
                None,
                base.at[:, 0].add(shift),
                None,
                space,
                key=subsample_key,
                neighbours=neighbours,
                cap=cap,
            )
            for shift in shifts
        ]
    )
