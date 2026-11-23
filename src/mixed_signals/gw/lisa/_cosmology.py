"""Redshift from luminosity distance in the Planck15 cosmology, outside JAX.

SGS Section 2.1 fixes the cosmology to Planck15 as astropy implements it
(:cite:t:`baghi2026lisa`; :cite:t:`ade2016planck`; :cite:t:`astropy2022astropy`).
The luminosity distance D_L(z) rises monotonically, so its inverse is a
monotone map. The module tabulates ln D_L and its derivative on a dense,
logarithmic redshift grid once, and inverts by cubic Hermite interpolation of
ln z against ln D_L, which needs no per-sample call of the cosmology. From
10⁻⁴ to 10⁴ Gpc, on 200 000 logarithmically spaced distances, the luminosity
distance of the interpolated redshift returns the input within a relative
2.3 × 10⁻¹². Six of those distances, at redshifts above 18, are the exception:
astropy's own luminosity distance differs there by up to 6 × 10⁻⁸ from a
quadrature of the comoving distance, which the interpolated redshift matches
within 2 × 10⁻¹⁴.
"""

import functools
import math
from typing import Final

import numpy as np

from mixed_signals.gw.lisa._dependencies import astropy_cosmology

__all__ = ["redshift_from_luminosity_distance"]

_REDSHIFT_RANGE: Final = (1e-8, 1e3)
_GRID_SIZE: Final = 4_001


@functools.cache
def _grid() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ln D_L, ln z and d ln z / d ln D_L on the grid, with D_L in Gpc."""
    cosmology = astropy_cosmology().Planck15
    redshift = np.geomspace(*_REDSHIFT_RANGE, _GRID_SIZE)
    distance = np.asarray(cosmology.luminosity_distance(redshift).to_value("Gpc"))
    hubble_distance = float(cosmology.hubble_distance.to_value("Gpc"))
    # * dD_L/dz = D_L/(1 + z) + (1 + z) c/(H0 E(z)) for a flat universe.
    slope = distance / (1 + redshift) + (1 + redshift) * hubble_distance / np.asarray(
        cosmology.efunc(redshift)
    )
    return np.log(distance), np.log(redshift), distance / (redshift * slope)


def redshift_from_luminosity_distance(distance: np.ndarray) -> np.ndarray:
    """Return the redshift of luminosity distances in Planck15.

    Args:
        distance: Luminosity distances in Gpc, positive and finite.

    Returns:
        The redshifts, shaped like ``distance``.

    Raises:
        ValueError: If a distance is not finite, not positive, or outside the
            tabulated range of redshifts from 10⁻⁸ to 10³.
        MissingDependencyError: If astropy is not installed.

    References:
        :cite:t:`ade2016planck`; :cite:t:`astropy2022astropy`.
    """
    log_distance, log_redshift, slope = _grid()
    query = np.asarray(distance, dtype=np.float64)
    if not np.all(np.isfinite(query) & (query > 0)):
        raise ValueError("luminosity distances must be positive and finite")
    log_query = np.log(query)
    if log_query.min() < log_distance[0] or log_query.max() > log_distance[-1]:
        raise ValueError(
            "luminosity distances must lie between"
            f" {math.exp(log_distance[0]):.3g} and"
            f" {math.exp(log_distance[-1]):.3g} Gpc, got {query.min():.3g} to"
            f" {query.max():.3g}"
        )
    node = np.clip(np.searchsorted(log_distance, log_query) - 1, 0, _GRID_SIZE - 2)
    width = log_distance[node + 1] - log_distance[node]
    position = (log_query - log_distance[node]) / width
    cube, square = position**3, position**2
    # * Cubic Hermite basis functions on the interval of each query.
    value = (
        (2 * cube - 3 * square + 1) * log_redshift[node]
        + (cube - 2 * square + position) * width * slope[node]
        + (-2 * cube + 3 * square) * log_redshift[node + 1]
        + (cube - square) * width * slope[node + 1]
    )
    return np.asarray(np.exp(value))
