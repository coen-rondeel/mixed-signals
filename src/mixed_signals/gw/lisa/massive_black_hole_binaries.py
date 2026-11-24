"""Massive black-hole binaries: the mergers that LISA sees at high signal-to-noise.

A holder of massive black-hole binaries (MBHBs) keeps the twelve parameters of
the catpylisa conventions (SGS Sections 2.3 and 5.10.2, :cite:t:`baghi2026lisa`):
the detector-frame masses, the luminosity distance in Gpc, the two spins along
the orbital angular momentum, the coalescence time, the phase and frequency at
the reference time, the sky position, the inclination and the polarisation. The
reference frequency defaults to 0 as in the catpylisa conventions file, where 0
marks the reference that the waveform code chooses, and some global fits leave
the column out; its domain is [0, ∞), and
:func:`~mixed_signals.gw.lisa.compare.compare_sources` checks per matched pair
that the settings of the two holders agree. The coalescence time counts seconds
from the epoch, so shifting the epoch shifts it. Masses, redshift and the
source-frame masses are in
:class:`~mixed_signals.gw.lisa.sources.BlackHoleBinarySources`.

References:
    :cite:t:`baghi2026lisa`; :cite:t:`ade2016planck`.
"""

from typing import ClassVar

from mixed_signals.gw.lisa.conventions import (
    MBHB_ALIASES,
    MBHB_SPECS,
    MBHB_SPHERES,
)
from mixed_signals.gw.lisa.sources import BlackHoleBinarySources
from mixed_signals.gw.lisa.transforms import CoalescenceTimeShift

__all__ = ["MassiveBlackHoleBinaries"]


class MassiveBlackHoleBinaries(BlackHoleBinarySources):
    """B massive black-hole binaries with N weighted samples each.

    The module docstring describes the parameters; see
    :class:`~mixed_signals.gw.lisa.sources.LisaSources` for the constructors,
    the space and the settings, and
    :class:`~mixed_signals.gw.lisa.sources.BlackHoleBinarySources` for the
    derived mass quantities.

    References:
        :cite:t:`baghi2026lisa`.
    """

    source_type: ClassVar[str] = "massive_black_hole_binary"
    parameter_specs = MBHB_SPECS
    sphere_specs = MBHB_SPHERES
    alias_table = MBHB_ALIASES
    can_shift_epoch: ClassVar[bool] = True

    def at_epoch[SourcesType: "MassiveBlackHoleBinaries"](
        self: SourcesType, epoch: float
    ) -> SourcesType:
        """Return the holder at another reference epoch.

        The coalescence time, sampled or constant, moves to t_c − Δ for
        Δ = epoch − the epoch of the holder, with a unit Jacobian.

        Args:
            epoch: The new epoch in TCB seconds since ``LISA_EPOCH_TCB``.

        Raises:
            ValueError: If the coalescence time has a prior bound.
        """
        delta = float(epoch) - self.conventions.epoch
        transform = CoalescenceTimeShift.from_space(self.full_space, delta=delta)
        return self._apply(
            transform,
            transform.touched_names(self.full_space),
            conventions=self.conventions.model_copy(update={"epoch": float(epoch)}),
        )
