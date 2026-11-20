"""Distances between whitened templates, from the Gram matrix of centred rows.

The Euclidean distance between two whitened templates is the noise-weighted
norm of the difference of the signals, in units of the noise standard deviation
(Finn 1992). :func:`distance_matrix` gives all pairs of a pool as one matrix,
which :func:`~mixed_signals.metrics.kernel.energy_distance_from_distances`
turns into the observed energy distance, every null draw and every half-sample
with matrix products. A matrix needs 8P² bytes for P pooled samples. The rows
are computed in blocks that are written into the matrix in place, so no
temporary array of the computation is larger than one block.

Centring
    The squared distance is ‖xᵢ‖² + ‖xⱼ‖² − 2 xᵢ·xⱼ. The norm of a template is
    its signal-to-noise ratio, large next to the distance between two
    templates of nearby parameters, so the three terms nearly cancel. Centring
    the rows on their mean removes the part that all templates share and
    leaves the norms of the deviations, which are of the size of the distances;
    sums of squares about a shifted origin lose far fewer digits (Chan, Golub
    and LeVeque 1983). A squared distance below zero by rounding becomes zero,
    and the diagonal is exactly zero.

References:
    :cite:t:`chan1983algorithms`; :cite:t:`finn1992detection`.
"""

from functools import partial

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from jaxtyping import Array, Float, Int

from mixed_signals.core.samples import _is_concrete
from mixed_signals.estimators.base import CHUNK_ENTRIES, _jit


def distance_matrix(
    points: Float[ArrayLike, "sample data"],
) -> Float[Array, "sample sample"]:
    """Return the Euclidean distances between all pairs of rows.

    The squared distances come from the Gram matrix of the rows centred on
    their mean. The rows are processed in blocks of at most
    :data:`~mixed_signals.estimators.base.CHUNK_ENTRIES` entries per block
    product, and each block is written into the result in place, so the peak
    memory stays near the 8P² bytes of the result.

    Args:
        points: The rows, shaped (P, M), with at least one row.

    Returns:
        The distances, shaped (P, P), symmetric up to rounding, with an exact
        zero diagonal.

    Raises:
        ValueError: If the points are not two-dimensional with at least one
            row, or concrete points are not finite.

    References:
        :cite:t:`chan1983algorithms`.
    """
    rows = jnp.asarray(points, dtype=jnp.float64)
    if rows.ndim != 2 or rows.shape[0] == 0:
        raise ValueError(
            "points must be shaped (P, M) with at least one row, got shape"
            f" {rows.shape}"
        )
    if _is_concrete(rows) and not bool(jnp.all(jnp.isfinite(rows))):
        raise ValueError("points must be finite")
    block_rows = max(1, min(rows.shape[0], CHUNK_ENTRIES // rows.shape[0]))
    return _distance_matrix(rows, block_rows)


def _block_distances(
    centred: Float[Array, "sample data"],
    squared_norms: Float[Array, " sample"],
    block: tuple[
        Float[Array, "selected data"], Float[Array, " selected"], Int[Array, ""]
    ],
) -> Float[Array, "selected sample"]:
    """Return the distances from the rows of one block to every row."""
    rows, block_norms, start = block
    positions = start + jnp.arange(rows.shape[0])
    squared = block_norms[:, None] + squared_norms[None, :] - 2 * rows @ centred.T
    distances = jnp.sqrt(jnp.maximum(squared, 0.0))
    # * The diagonal is exactly zero, whatever the rounding of the Gram matrix.
    diagonal = positions[:, None] == jnp.arange(centred.shape[0])[None, :]
    return jnp.where(diagonal, 0.0, distances)


def _fill_block(
    centred: Float[Array, "sample data"],
    squared_norms: Float[Array, " sample"],
    block_rows: int,
    index: Int[Array, ""],
    matrix: Float[Array, "sample sample"],
) -> Float[Array, "sample sample"]:
    """Write the distances of one block of rows into the matrix, in place."""
    count = centred.shape[0]
    # * The last block starts at count - block_rows: it overlaps, so no padding.
    start = jnp.minimum(index * block_rows, count - block_rows)
    block = (
        jax.lax.dynamic_slice_in_dim(centred, start, block_rows, axis=0),
        jax.lax.dynamic_slice_in_dim(squared_norms, start, block_rows, axis=0),
        start,
    )
    distances = _block_distances(centred, squared_norms, block)
    return jax.lax.dynamic_update_slice_in_dim(matrix, distances, start, axis=0)


def _distance_matrix_kernel(
    points: Float[Array, "sample data"], block_rows: int
) -> Float[Array, "sample sample"]:
    """Return the distance matrix, filled in blocks of ``block_rows`` rows."""
    count = points.shape[0]
    centred = points - jnp.mean(points, axis=0)
    squared_norms = jnp.sum(centred**2, axis=1)
    block_count = -(-count // block_rows)
    filled: Array = jax.lax.fori_loop(
        0,
        block_count,
        partial(_fill_block, centred, squared_norms, block_rows),
        jnp.zeros((count, count)),
    )
    return filled


_distance_matrix = _jit(_distance_matrix_kernel, "block_rows")
