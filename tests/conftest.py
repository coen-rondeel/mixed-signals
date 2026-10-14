"""Shared helpers for the tests of the zoo and the estimators."""

import functools
from collections.abc import Callable

import numpy as np
import pytest

type Density = Callable[[np.ndarray], np.ndarray]
"""Maps grid points shaped (..., D) to density values shaped (...)."""

type GridIntegral = Callable[..., float]
"""Integrates a density over a box given as (lower, upper, panels) per axis."""

GAUSS_LEGENDRE_ORDER = 8


def integrate_on_grid(density: Density, *axes: tuple[float, float, int]) -> float:
    """Integrate a density over a box with composite Gauss-Legendre rules.

    Each axis is split into equal panels with eight nodes each, which
    integrates smooth densities to near machine precision with one vectorised
    evaluation instead of thousands of scalar calls.

    Args:
        density: Function of points shaped (..., D), returning densities.
        *axes: One (lower, upper, panels) triple per axis.

    Returns:
        The integral over the box.
    """
    nodes, weights = np.polynomial.legendre.leggauss(GAUSS_LEGENDRE_ORDER)
    axis_nodes, axis_weights = [], []
    for lower, upper, panels in axes:
        edges = np.linspace(lower, upper, panels + 1)
        half_widths = 0.5 * np.diff(edges)
        centres = 0.5 * (edges[:-1] + edges[1:])
        axis_nodes.append((centres[:, None] + half_widths[:, None] * nodes).ravel())
        axis_weights.append((half_widths[:, None] * weights).ravel())
    points = np.stack(np.meshgrid(*axis_nodes, indexing="ij"), axis=-1)
    grid_weights = functools.reduce(np.multiply.outer, axis_weights)
    return float(np.sum(grid_weights * density(points)))


@pytest.fixture
def grid_integral() -> GridIntegral:
    """The composite Gauss-Legendre integrator, for tests that check normalisation."""
    return integrate_on_grid
