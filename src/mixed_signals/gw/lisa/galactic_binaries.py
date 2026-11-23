"""Galactic binaries: white-dwarf binaries and verification binaries.

A holder of Galactic binaries keeps the eight parameters of SGS Eqs. 146 and 147
(:cite:t:`baghi2026lisa`): the GW frequency f, its derivative ḟ, the amplitude 𝒜,
the sky position, the polarisation ψ, the inclination ι and the initial phase
φ₀, with the waveform phase 2πft − φ₀. The frequency derivative may be negative,
because mass transfer drives binaries outward (SGS Section 5.10.4). The derived
quantities come from the leading-order chirp of
:cite:t:`peters1964gravitational`. The epoch is that of f, ḟ and φ₀; shifting it
moves f and φ₀ as JaxGB shifts its epoch (:cite:t:`bayle2026jaxgb`).

Opt-in forms of the parameters:

- the chirp mass 𝓜 and the distance D in place of ḟ and 𝒜, with ḟ ≤ 0 refused;
- the phase combinations θ± = φ₀ ± 2ψ, which the SSB strain depends on, with the
  four-to-one map and the log density that
  :class:`~mixed_signals.gw.lisa.transforms.PhaseCombinationTransform` derives.

References:
    :cite:t:`baghi2026lisa`; :cite:t:`peters1964gravitational`;
    :cite:t:`bayle2026jaxgb`; :cite:t:`bayle2026lisa`.
"""

from typing import ClassVar

import numpy as np
from jaxtyping import Array, Float

from mixed_signals.gw.lisa._dependencies import lisa_constants
from mixed_signals.gw.lisa.conventions import (
    GALACTIC_BINARY_ALIASES,
    GALACTIC_BINARY_SPECS,
    GALACTIC_BINARY_SPHERES,
)
from mixed_signals.gw.lisa.sources import LisaSources
from mixed_signals.gw.lisa.transforms import (
    ChirpMassAndDistanceTransform,
    GalacticBinaryEpochShift,
    PhaseCombinationTransform,
    chirp_mass_from_frequency_dot,
    distance_from_amplitude,
)

__all__ = ["GalacticBinaries"]


class GalacticBinaries(LisaSources):
    """B Galactic binaries with N weighted samples each.

    The module docstring describes the parameters and the opt-in forms; see
    :class:`~mixed_signals.gw.lisa.sources.LisaSources` for the constructors,
    the space and the settings. The derived properties return arrays shaped
    (B, N) over all samples.

    References:
        :cite:t:`baghi2026lisa`; :cite:t:`peters1964gravitational`.
    """

    source_type: ClassVar[str] = "galactic_binary"
    parameter_specs = GALACTIC_BINARY_SPECS
    sphere_specs = GALACTIC_BINARY_SPHERES
    alias_table = GALACTIC_BINARY_ALIASES
    spinning: ClassVar[bool] = False
    can_shift_epoch: ClassVar[bool] = True

    @property
    def chirp_mass(self) -> Float[Array, "batch sample"]:
        """The chirp mass in solar masses, NaN where ḟ ≤ 0.

        References:
            :cite:t:`peters1964gravitational`.
        """
        if "chirp_mass" in self.full_space.names:
            return self._column("chirp_mass")
        return chirp_mass_from_frequency_dot(
            self._column("frequency"), self._column("frequency_dot"), lisa_constants()
        )

    @property
    def distance(self) -> Float[Array, "batch sample"]:
        """The distance in kiloparsecs from f, ḟ and 𝒜, NaN where ḟ ≤ 0.

        References:
            :cite:t:`baghi2026lisa`; :cite:t:`peters1964gravitational`.
        """
        if "distance" in self.full_space.names:
            return self._column("distance")
        return distance_from_amplitude(
            self._column("frequency"),
            self._column("frequency_dot"),
            self._column("amplitude"),
            lisa_constants(),
        )

    @property
    def gw_period(self) -> Float[Array, "batch sample"]:
        """The gravitational-wave period 1/f in seconds."""
        return 1.0 / self._column("frequency")

    @property
    def orbital_period(self) -> Float[Array, "batch sample"]:
        """The orbital period 2/f in seconds, twice the GW period."""
        return 2.0 / self._column("frequency")

    @property
    def orbital_frequency(self) -> Float[Array, "batch sample"]:
        """The orbital frequency f/2 in hertz."""
        return self._column("frequency") / 2.0

    def to_chirp_mass_and_distance[SourcesType: "GalacticBinaries"](
        self: SourcesType,
    ) -> SourcesType:
        """Return the holder with the chirp mass and distance in place of ḟ and 𝒜.

        A holder that already has them is returned unchanged. The log-prior
        moves by the Jacobian (3𝓜/(5ḟ))(D/𝒜).

        Raises:
            ValueError: If a positive-weight sample has ḟ ≤ 0, with the count.
            MissingDependencyError: If lisaconstants is not installed.
        """
        if "chirp_mass" in self.full_space.names:
            return self
        derivative = np.asarray(self._column("frequency_dot"))
        positive = np.asarray(self.values.log_weights[:, 0]) > -np.inf
        offending = int(((derivative <= 0) & positive).sum())
        if offending:
            raise ValueError(
                f"frequency_dot must be positive for the chirp mass, but {offending}"
                f" of the {int(positive.sum())} positive-weight values are not"
            )
        transform = ChirpMassAndDistanceTransform.from_space(
            self.full_space, to_chirp_mass=True, constants=lisa_constants()
        )
        # * Padding with ḟ ≤ 0 would turn to NaN, so it takes a real sample's values.
        holder = self._overwrite_zero_weight_samples(derivative <= 0)
        return holder._apply(transform, transform.touched_names(holder.full_space))

    def to_frequency_dot_and_amplitude[SourcesType: "GalacticBinaries"](
        self: SourcesType,
    ) -> SourcesType:
        """Return the holder with ḟ and 𝒜, the canonical form, or itself if it has them.

        Raises:
            MissingDependencyError: If lisaconstants is not installed.
        """
        if "frequency_dot" in self.full_space.names:
            return self
        transform = ChirpMassAndDistanceTransform.from_space(
            self.full_space, to_chirp_mass=False, constants=lisa_constants()
        )
        return self._apply(transform, transform.touched_names(self.full_space))

    def to_phase_combinations[SourcesType: "GalacticBinaries"](
        self: SourcesType,
    ) -> SourcesType:
        """Return the holder with θ± = φ₀ ± 2ψ in place of φ₀ and ψ.

        The map is four-to-one and leaves the log density unchanged for densities
        that the symmetries of the waveform leave invariant; see
        :class:`~mixed_signals.gw.lisa.transforms.PhaseCombinationTransform`. A
        holder that already has θ± is returned unchanged.
        """
        if "initial_phase" not in self.full_space.names:
            return self
        transform = PhaseCombinationTransform.from_space(
            self.full_space, to_combinations=True
        )
        return self._apply(transform, transform.touched_names(self.full_space))

    def to_initial_phase_and_polarisation[SourcesType: "GalacticBinaries"](
        self: SourcesType,
    ) -> SourcesType:
        """Return the holder with φ₀ and ψ in place of θ±, taking ψ below π/2.

        A holder that has φ₀ and ψ is returned unchanged.
        """
        if "initial_phase" in self.full_space.names:
            return self
        transform = PhaseCombinationTransform.from_space(
            self.full_space, to_combinations=False
        )
        return self._apply(transform, transform.touched_names(self.full_space))

    def at_epoch[SourcesType: "GalacticBinaries"](
        self: SourcesType, epoch: float
    ) -> SourcesType:
        """Return the holder at another reference epoch.

        The frequency moves to f + ḟΔ and the phase to φ₀ − 2π(fΔ + ½ḟΔ²) for
        Δ = epoch − the epoch of the holder, with the unit Jacobian of
        :class:`~mixed_signals.gw.lisa.transforms.GalacticBinaryEpochShift`.
        Shifts compose.

        Args:
            epoch: The new epoch in TCB seconds since ``LISA_EPOCH_TCB``.

        Raises:
            ConventionError: If the holder carries the chirp mass in place of ḟ;
                convert it back first.
            ValueError: If the frequency or the phase has a prior bound.

        References:
            :cite:t:`bayle2026jaxgb`.
        """
        delta = float(epoch) - self.conventions.epoch
        transform = GalacticBinaryEpochShift.from_space(self.full_space, delta=delta)
        return self._apply(
            transform,
            transform.touched_names(self.full_space),
            conventions=self.conventions.model_copy(update={"epoch": float(epoch)}),
        )
