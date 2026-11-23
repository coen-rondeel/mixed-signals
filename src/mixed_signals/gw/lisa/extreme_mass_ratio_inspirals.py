"""Extreme mass-ratio inspirals: a compact object spiralling into a massive black hole.

A holder of extreme mass-ratio inspirals (EMRIs) keeps the fourteen parameters
of the catpylisa conventions (SGS Sections 5.6.7 and 5.10.1 and Appendix A,
:cite:t:`baghi2026lisa`): the detector-frame masses, the spin of the primary and
its direction, the semi-latus rectum, the eccentricity and the three phases at
the reference time, the luminosity distance, the sky position, and the orbital
inclination and polar phase at the reference time. The inclination is that of
the orbit at the reference time, not FEW's cosine, so the alias ``xI`` is not
accepted. The spin direction is measured in the sky frame and rotates with it.
EMRI holders cannot shift their epoch, because the parameters at the reference
time depend on the orbital evolution.

References:
    :cite:t:`baghi2026lisa`.
"""

from typing import ClassVar

from mixed_signals.gw.lisa.conventions import EMRI_ALIASES, EMRI_SPECS, EMRI_SPHERES
from mixed_signals.gw.lisa.sources import BlackHoleBinarySources

__all__ = ["ExtremeMassRatioInspirals"]


class ExtremeMassRatioInspirals(BlackHoleBinarySources):
    """B extreme mass-ratio inspirals with N weighted samples each.

    The module docstring describes the parameters; see
    :class:`~mixed_signals.gw.lisa.sources.LisaSources` for the constructors,
    the space and the settings, and
    :class:`~mixed_signals.gw.lisa.sources.BlackHoleBinarySources` for the
    derived mass quantities.

    References:
        :cite:t:`baghi2026lisa`.
    """

    source_type: ClassVar[str] = "extreme_mass_ratio_inspiral"
    parameter_specs = EMRI_SPECS
    sphere_specs = EMRI_SPHERES
    alias_table = EMRI_ALIASES
