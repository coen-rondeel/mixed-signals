"""Loaders of the optional ``gw`` extra: lisaconstants and astropy.

``import mixed_signals.gw.lisa`` works without the extra. Each loader imports
its package inside the function, so that only a call that needs a constant or
the cosmology raises :class:`~mixed_signals.errors.MissingDependencyError`,
which names the extra (``pip install "mixed-signals[gw]"``), as
:mod:`mixed_signals.io.hdf5` does for h5py.

Constants
    The values come from lisaconstants, the source of the LISA parameter data
    base (:cite:t:`bayle2026lisa`), which the conventions document names for
    the obliquity, the speed of light and the other constants
    (:cite:t:`baghi2026lisa`).
Import warning
    lisaconstants 2.0.2 warns at import under astropy 8, because its vacuum
    permeability differs from astropy's. The package uses no electromagnetic
    constant, so the loader filters exactly that warning.
"""

import importlib
import math
import warnings
from types import ModuleType
from typing import Final, NamedTuple

from mixed_signals.errors import MissingDependencyError

_INSTALL_HINT: Final = 'install it with the gw extra: pip install "mixed-signals[gw]"'
# * The text that lisaconstants 2.0.2 starts its import warning with.
_CONSTANTS_WARNING: Final = "The following constants differ between lisaconstants"


class LisaConstants(NamedTuple):
    """The lisaconstants values that the holders use, in SI units."""

    obliquity: float
    """Mean obliquity of the ecliptic at J2000, in radians."""
    speed_of_light: float
    """Speed of light in vacuum, in m/s."""
    gravitational_constant: float
    """Newton's constant, in m³ kg⁻¹ s⁻²."""
    solar_mass: float
    """Solar mass, in kg."""
    parsec: float
    """One parsec, in m."""


def lisa_constants() -> LisaConstants:
    """Return the constants of lisaconstants that the holders use.

    Returns:
        The obliquity in radians and the other constants in SI units.

    Raises:
        MissingDependencyError: If lisaconstants is not installed.

    References:
        :cite:t:`bayle2026lisa`.
    """
    module = _import("lisaconstants")
    # * lisaconstants gives the obliquity in degrees.
    return LisaConstants(
        obliquity=math.radians(float(module.OBLIQUITY)),
        speed_of_light=float(module.SPEED_OF_LIGHT),
        gravitational_constant=float(module.GRAVITATIONAL_CONSTANT),
        solar_mass=float(module.SOLAR_MASS),
        parsec=float(module.PARSEC),
    )


def astropy_cosmology() -> ModuleType:
    """Return the module ``astropy.cosmology``, which holds Planck15.

    Returns:
        The imported module.

    Raises:
        MissingDependencyError: If astropy is not installed.

    References:
        :cite:t:`astropy2022astropy`.
    """
    return _import("astropy.cosmology")


def _import(name: str) -> ModuleType:
    """Import a module of the ``gw`` extra, without the lisaconstants warning."""
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message=_CONSTANTS_WARNING, category=UserWarning
            )
            return importlib.import_module(name)
    except ImportError as error:
        raise MissingDependencyError(
            f"cannot import {name} ({error}); {_INSTALL_HINT}"
        ) from error
