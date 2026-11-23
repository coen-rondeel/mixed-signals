"""Stellar-origin binary black holes: inspirals that LISA sees years before merger.

A holder of stellar-origin binary black holes (SOBBHs) keeps the sixteen
parameters of the catpylisa conventions (SGS Sections 2.3 and 5.10.3,
:cite:t:`baghi2026lisa`): the detector-frame masses, the two spins, the
reference frequency, the inclination, the polarisation, the phase at the
reference time, the luminosity distance, the sky position, the eccentricity and
the directions of the two spins. Spin angles are taken in the sky frame for
now, and rotate with it; this is to be checked against the conventions of the
global fits. SOBBH holders cannot shift their epoch.

References:
    :cite:t:`baghi2026lisa`.
"""

from typing import ClassVar

from mixed_signals.gw.lisa.conventions import (
    SOBBH_ALIASES,
    SOBBH_SPECS,
    SOBBH_SPHERES,
)
from mixed_signals.gw.lisa.sources import BlackHoleBinarySources

__all__ = ["StellarOriginBinaryBlackHoles"]


class StellarOriginBinaryBlackHoles(BlackHoleBinarySources):
    """B stellar-origin binary black holes with N weighted samples each.

    The module docstring describes the parameters; see
    :class:`~mixed_signals.gw.lisa.sources.LisaSources` for the constructors,
    the space and the settings, and
    :class:`~mixed_signals.gw.lisa.sources.BlackHoleBinarySources` for the
    derived mass quantities.

    References:
        :cite:t:`baghi2026lisa`.
    """

    source_type: ClassVar[str] = "stellar_origin_binary_black_hole"
    parameter_specs = SOBBH_SPECS
    sphere_specs = SOBBH_SPHERES
    alias_table = SOBBH_ALIASES
