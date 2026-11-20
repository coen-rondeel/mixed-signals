"""Tests for the distance matrix of whitened templates."""

import inspect

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import spatial

from mixed_signals.dataspace import distances
from mixed_signals.dataspace.distances import distance_matrix

unchecked_distance_matrix = inspect.unwrap(distance_matrix)


@pytest.fixture
def rng() -> np.random.Generator:
    """A seeded NumPy generator for test data."""
    return np.random.default_rng(2027)


def test_the_distances_equal_scipy_cdist(rng: np.random.Generator) -> None:
    """Every pair of rows, to 1e-10 relative."""
    rows = rng.normal(size=(120, 17))
    matrix = distance_matrix(rows)
    assert matrix.shape == (120, 120)
    assert matrix.dtype == jnp.float64
    np.testing.assert_allclose(
        matrix, spatial.distance.cdist(rows, rows), rtol=1e-10, atol=1e-12
    )


def test_the_diagonal_is_exactly_zero_and_the_matrix_symmetric(
    rng: np.random.Generator,
) -> None:
    """Rounding never puts a nonzero or negative value on the diagonal."""
    rows = rng.normal(size=(80, 40)) + 1e3
    matrix = np.asarray(distance_matrix(rows))
    assert np.all(np.diag(matrix) == 0.0)
    assert np.all(matrix >= 0.0)
    np.testing.assert_allclose(matrix, matrix.T, rtol=1e-12, atol=1e-12)


def test_a_large_common_offset_does_not_cancel_the_distances(
    rng: np.random.Generator,
) -> None:
    """Rows shifted by 1e6 per entry match cdist of the unshifted rows."""
    rows = rng.normal(size=(60, 30))
    shifted = distance_matrix(rows + 1e6)
    np.testing.assert_allclose(
        shifted, spatial.distance.cdist(rows, rows), rtol=1e-7, atol=1e-8
    )


def test_blocks_of_rows_give_the_same_matrix(
    rng: np.random.Generator, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A small block size, with padding in the last block, changes no value."""
    rows = rng.normal(size=(50, 9))
    whole = distance_matrix(rows)
    # * 50 rows in blocks of 8: the last block starts at row 42, so it overlaps
    # * the block before it instead of padding.
    monkeypatch.setattr(distances, "CHUNK_ENTRIES", 8 * 50)
    blocked = distance_matrix(rows)
    np.testing.assert_allclose(blocked, whole, rtol=1e-13, atol=1e-13)


def test_one_row_and_two_identical_rows() -> None:
    """One row gives [[0]], and identical rows have distance zero."""
    np.testing.assert_array_equal(distance_matrix(np.ones((1, 5))), [[0.0]])
    same = distance_matrix(np.ones((3, 4)))
    np.testing.assert_array_equal(same, np.zeros((3, 3)))


def test_the_function_runs_inside_a_trace(rng: np.random.Generator) -> None:
    """Under jax.jit the finite check does not run, and the values agree."""
    rows = jnp.asarray(rng.normal(size=(20, 6)))
    np.testing.assert_allclose(
        jax.jit(distance_matrix)(rows), distance_matrix(rows), rtol=1e-13, atol=1e-13
    )


def test_the_points_must_be_a_finite_matrix_with_a_row() -> None:
    """Wrong rank, no rows and non-finite entries raise ValueError."""
    with pytest.raises(ValueError, match=r"shaped \(P, M\).*got shape \(4,\)"):
        unchecked_distance_matrix(np.zeros(4))
    with pytest.raises(ValueError, match=r"shaped \(P, M\).*got shape \(0, 3\)"):
        unchecked_distance_matrix(np.zeros((0, 3)))
    with pytest.raises(ValueError, match=r"shaped \(P, M\).*got shape \(2, 2, 2\)"):
        unchecked_distance_matrix(np.zeros((2, 2, 2)))
    for bad_value in (np.nan, np.inf):
        rows = np.ones((4, 3))
        rows[2, 1] = bad_value
        with pytest.raises(ValueError, match="points must be finite"):
            distance_matrix(rows)
