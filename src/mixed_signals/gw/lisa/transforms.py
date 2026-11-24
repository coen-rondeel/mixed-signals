r"""Transforms of the parameter vector of LISA sources, with exact Jacobians.

Every transform is a :class:`~mixed_signals.transforms.base.Transform` built for
the positions of its parameters in the full parameter vector of a holder, so
that it acts on the sampled and the constant parameters together. The holder
methods apply them with ``map_sample_set``, which moves ``log_prior`` by the
log-determinant (:cite:t:`papamakarios2021normalizing`), map the space and
rename the parameters. All run in JAX, so they jit and differentiate.

Sky frame
    :class:`SkyFrameTransform` rotates ICRS to the Barycentric Mean Ecliptic of
    J2000 and back (:cite:t:`baghi2026lisa`, SGS Section 5.3.2, Eqs. 51 and 52),
    about the x axis by the obliquity ε of lisaconstants
    (:cite:t:`bayle2026lisa`). It rotates the unit vectors of the sky position
    and of the spin directions measured in the sky frame, and it shifts the
    polarisation angle by δψ with cos δψ = (sin ε sin δ sin α + cos ε cos δ)/cos β
    and sin δψ = −sin ε cos α/cos β (SGS Eq. 53), ψ in ICRS being ψ in the
    ecliptic frame plus δψ. A rotation keeps the area element of a sphere, so
    the log-determinant from ICRS to ecliptic is ln cos δ − ln cos β for the sky
    and ln sin θ − ln sin θ′ for each spin direction, with θ before and θ′ after;
    the shift of ψ adds a unit factor. Where the holder carries the phase
    combinations θ± in place of ψ, both shift by ±2δψ.
Epoch shift
    :class:`GalacticBinaryEpochShift` moves the epoch of the frequency and the
    phase of a Galactic binary, and :class:`CoalescenceTimeShift` the time
    origin of a coalescence time. Both have a unit Jacobian determinant.
Phase combinations
    :class:`PhaseCombinationTransform` maps (φ₀, ψ) of a Galactic binary to
    θ± = φ₀ ± 2ψ. It is 4-to-1, and its docstring derives why the log density
    is unchanged.
Masses
    :class:`MassPairTransform` maps the detector-frame masses (m₁, m₂) to the
    total mass and the mass ratio q = m₂/m₁, or to the chirp mass and the mass
    ratio, and back (SGS Section 2.3). With ``inverted=True`` the ratio is
    Q = m₁/m₂.
Chirp mass and distance
    :class:`ChirpMassAndDistanceTransform` maps (ḟ, 𝒜) of a Galactic binary to
    the chirp mass and the distance with the leading-order chirp of
    :cite:t:`peters1964gravitational`.

References:
    :cite:t:`baghi2026lisa`; :cite:t:`bayle2026lisa`;
    :cite:t:`papamakarios2021normalizing`; :cite:t:`peters1964gravitational`;
    :cite:t:`bayle2026jaxgb`.
"""

import math
from collections.abc import Mapping, Sequence
from typing import Final

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float

from mixed_signals.core.space import Parameter, ParamSpace
from mixed_signals.core.topology import (
    AZIMUTH_PERIOD,
    PolarForm,
    Sphere,
    Topology,
    sphere_coordinates,
    unit_vectors,
    wrap,
)
from mixed_signals.errors import ConventionError
from mixed_signals.gw.lisa._dependencies import LisaConstants
from mixed_signals.gw.lisa.conventions import (
    GALACTIC_BINARY_SPECS,
    MBHB_SPECS,
    SkyFrame,
    frame_parameter,
    frame_sphere,
    rename_parameter,
)
from mixed_signals.transforms.base import Transform

__all__ = [
    "ChirpMassAndDistanceTransform",
    "CoalescenceTimeShift",
    "GalacticBinaryEpochShift",
    "MassPairTransform",
    "PhaseCombinationTransform",
    "SkyFrameTransform",
    "amplitude_from_distance",
    "chirp_mass_from_frequency_dot",
    "distance_from_amplitude",
    "frequency_dot_from_chirp_mass",
]

_ICRS_SUFFIX: Final = "_icrs"
_ECLIPTIC_SUFFIX: Final = "_ecliptic"
_GALACTIC_SPECS: Final = {spec.name: spec for spec in GALACTIC_BINARY_SPECS}
_MASS_SPECS: Final = {spec.name: spec for spec in MBHB_SPECS}
# * The parameters that MassPairTransform.map_space gives, by name; the domains of
# * with_bounds come from here.
_MASS_FORM_PARAMETERS: Final = {
    parameter.name: parameter
    for parameter in (
        Parameter(
            name="total_mass_det_frame",
            label=r"M_{\mathrm{det}}",
            unit="Msun",
            topology=Topology.INTERVAL,
            bounds=(0.0, math.inf),
        ),
        Parameter(
            name="chirp_mass_det_frame",
            label=r"\mathcal{M}_{\mathrm{det}}",
            unit="Msun",
            topology=Topology.INTERVAL,
            bounds=(0.0, math.inf),
        ),
        Parameter(
            name="mass_ratio",
            label="q",
            topology=Topology.INTERVAL,
            bounds=(0.0, 1.0),
        ),
        Parameter(
            name="inverted_mass_ratio",
            label="Q",
            topology=Topology.INTERVAL,
            bounds=(1.0, math.inf),
        ),
    )
}
# * A kiloparsec is a thousand parsecs; the Galactic-binary distance is in kpc.
_KILOPARSEC_PER_PARSEC: Final = 1_000.0


def _index_of(space: ParamSpace, name: str) -> int:
    """Return the position of a parameter, or raise a ConventionError that names it."""
    if name not in space.names:
        raise ConventionError(
            f"the parameter vector has no {name!r}; it holds {', '.join(space.names)}"
        )
    return space.index(name)


def _frame_of(name: str) -> SkyFrame:
    """Return the sky frame in whose spelling a polarisation-like name is written."""
    return SkyFrame.ECLIPTIC if name.endswith(_ECLIPTIC_SUFFIX) else SkyFrame.ICRS


def _replace_parameters(
    space: ParamSpace,
    replacements: Mapping[int, Parameter],
    spheres: Sequence[Sphere] | None = None,
) -> ParamSpace:
    """Return a space with some parameters, and optionally the spheres, replaced."""
    parameters = tuple(
        replacements.get(position, parameter)
        for position, parameter in enumerate(space.parameters)
    )
    return ParamSpace(
        parameters=parameters,
        spheres=space.spheres if spheres is None else tuple(spheres),
    )


class SkyFrameTransform(Transform):
    r"""Rotation of the sky frame between ICRS and the ecliptic, with its polarisation.

    The map takes the sky position, the polarisation angle (or the phase
    combinations θ±) and the spin directions measured in the sky frame from one
    frame to the other, and leaves every other parameter unchanged. The module
    docstring gives the formulas, the log-determinant and the sources.

    Attributes:
        to_ecliptic: ``True`` for ICRS to ecliptic, ``False`` for the inverse.
        obliquity: Obliquity ε of the ecliptic, in radians.
        sky: Positions of the azimuth (right ascension or longitude) and of the
            polar parameter (declination or latitude) in the vector.
        polarisation: Positions of ψ, or of θ₊ and θ₋.
        polarisation_weights: The weight of δψ in each of them: 1 for ψ, and
            2 and −2 for θ₊ and θ₋.
        spin_directions: Positions of the azimuth and the polar angle of each
            spin direction, in colatitude form.

    References:
        :cite:t:`baghi2026lisa`; :cite:t:`bayle2026lisa`.
    """

    to_ecliptic: bool = eqx.field(static=True)
    obliquity: float = eqx.field(static=True)
    sky: tuple[int, int] = eqx.field(static=True)
    polarisation: tuple[int, ...] = eqx.field(static=True)
    polarisation_weights: tuple[float, ...] = eqx.field(static=True)
    spin_directions: tuple[tuple[int, int], ...] = eqx.field(static=True)

    @classmethod
    def from_space(
        cls, space: ParamSpace, *, to_ecliptic: bool, obliquity: float
    ) -> "SkyFrameTransform":
        """Build the transform for the full parameter vector of a holder.

        Args:
            space: The space of the full vector, with the spheres of the sky
                position and of the spin directions.
            to_ecliptic: ``True`` for ICRS to ecliptic, ``False`` for the inverse.
            obliquity: Obliquity ε in radians.

        Returns:
            The transform.

        Raises:
            ConventionError: If the vector holds no sky position.
        """
        skies = [
            sphere
            for sphere in space.spheres
            if sphere.polar_form is PolarForm.LATITUDE
        ]
        if len(skies) != 1:
            raise ConventionError("the parameter vector holds no sky position")
        sky = (space.index(skies[0].azimuth), space.index(skies[0].polar))
        names = space.names
        # * EMRI vectors hold no polarisation, so the tuples may stay empty.
        polarisation: tuple[int, ...] = ()
        weights: tuple[float, ...] = ()
        for prefixes, signs in (
            (("polarisation_",), (1.0,)),
            (("theta_plus_", "theta_minus_"), (2.0, -2.0)),
        ):
            found = tuple(
                position
                for prefix in prefixes
                for position, name in enumerate(names)
                if name.startswith(prefix)
            )
            if len(found) == len(prefixes):
                polarisation, weights = found, signs
        spins = tuple(
            (space.index(sphere.azimuth), space.index(sphere.polar))
            for sphere in space.spheres
            if sphere.polar_form is PolarForm.COLATITUDE
        )
        return cls(
            to_ecliptic=to_ecliptic,
            obliquity=obliquity,
            sky=sky,
            polarisation=polarisation,
            polarisation_weights=weights,
            spin_directions=spins,
        )

    @property
    def _positions(self) -> tuple[int, ...]:
        """Positions of every parameter that the transform changes."""
        return (
            *self.sky,
            *self.polarisation,
            *(position for pair in self.spin_directions for position in pair),
        )

    def touched_names(self, space: ParamSpace) -> tuple[str, ...]:
        """Return the names of the parameters that the transform changes."""
        return tuple(space.names[position] for position in self._positions)

    def _map(
        self, points: Float[Array, "*leading param"]
    ) -> tuple[Float[Array, "*leading param"], Float[Array, " *leading"]]:
        """Return the mapped points and the log-determinant of the map at the points."""
        angle = self.obliquity if self.to_ecliptic else -self.obliquity
        azimuth_index, polar_index = self.sky
        mapped = points
        log_determinant = jnp.zeros(points.shape[:-1])
        unit, area_before, _ = unit_vectors(
            points[..., azimuth_index], points[..., polar_index], PolarForm.LATITUDE
        )
        azimuth, polar = sphere_coordinates(
            _rotate_about_x(unit, angle), PolarForm.LATITUDE
        )
        _, area_after, _ = unit_vectors(azimuth, polar, PolarForm.LATITUDE)
        mapped = (
            mapped.at[..., azimuth_index].set(azimuth).at[..., polar_index].set(polar)
        )
        log_determinant = log_determinant + area_before - area_after
        # * The shift of ψ follows from the ICRS position, the input or the output.
        icrs_azimuth, icrs_polar = (
            (points[..., azimuth_index], points[..., polar_index])
            if self.to_ecliptic
            else (azimuth, polar)
        )
        shift = _polarisation_shift(icrs_azimuth, icrs_polar, self.obliquity)
        direction = -1.0 if self.to_ecliptic else 1.0
        for position, weight in zip(
            self.polarisation, self.polarisation_weights, strict=True
        ):
            shifted = points[..., position] + direction * weight * shift
            mapped = mapped.at[..., position].set(wrap(shifted, 0.0, AZIMUTH_PERIOD))
        for spin_azimuth, spin_polar in self.spin_directions:
            unit, area_before, _ = unit_vectors(
                points[..., spin_azimuth], points[..., spin_polar], PolarForm.COLATITUDE
            )
            new_azimuth, new_polar = sphere_coordinates(
                _rotate_about_x(unit, angle), PolarForm.COLATITUDE
            )
            _, area_after, _ = unit_vectors(
                new_azimuth, new_polar, PolarForm.COLATITUDE
            )
            mapped = (
                mapped.at[..., spin_azimuth]
                .set(new_azimuth)
                .at[..., spin_polar]
                .set(new_polar)
            )
            log_determinant = log_determinant + area_before - area_after
        return mapped, log_determinant

    def forward(
        self, points: Float[Array, "*leading param"]
    ) -> Float[Array, "*leading param"]:
        """Return the points in the other frame."""
        return self._map(points)[0]

    def inverse(
        self, points: Float[Array, "*leading param"]
    ) -> Float[Array, "*leading param"]:
        """Return the points rotated back to the original frame."""
        return self._reversed()._map(points)[0]

    def log_abs_det_jacobian(
        self, points: Float[Array, "*leading param"]
    ) -> Float[Array, " *leading"]:
        """Return ln|det ∂f/∂x|: the change of the area elements of the spheres."""
        return self._map(points)[1]

    @property
    def description(self) -> str:
        """The direction of the rotation."""
        return "icrs_to_ecliptic" if self.to_ecliptic else "ecliptic_to_icrs"

    def map_space(self, space: ParamSpace) -> ParamSpace:
        """Return the space with the sky parameters named in the new frame."""
        frame = SkyFrame.ECLIPTIC if self.to_ecliptic else SkyFrame.ICRS
        replacements = {
            position: rename_parameter(space.parameters[position], frame)
            for position in self._positions
        }
        return _replace_parameters(
            space,
            replacements,
            [frame_sphere(sphere, frame) for sphere in space.spheres],
        )

    def _reversed(self) -> "SkyFrameTransform":
        """Return the rotation in the opposite direction, for the same positions."""
        return SkyFrameTransform(
            to_ecliptic=not self.to_ecliptic,
            obliquity=self.obliquity,
            sky=self.sky,
            polarisation=self.polarisation,
            polarisation_weights=self.polarisation_weights,
            spin_directions=self.spin_directions,
        )


def _rotate_about_x(
    unit: Float[Array, "*leading 3"], angle: float
) -> Float[Array, "*leading 3"]:
    """Rotate unit vectors about the x axis, as in SGS Section 5.3.1."""
    cosine, sine = math.cos(angle), math.sin(angle)
    return jnp.stack(
        [
            unit[..., 0],
            cosine * unit[..., 1] + sine * unit[..., 2],
            -sine * unit[..., 1] + cosine * unit[..., 2],
        ],
        axis=-1,
    )


def _polarisation_shift(
    right_ascension: Float[Array, " *leading"],
    declination: Float[Array, " *leading"],
    obliquity: float,
) -> Float[Array, " *leading"]:
    """Return δψ of SGS Eq. 53 from the ICRS sky position, in (−π, π]."""
    sine, cosine = math.sin(obliquity), math.cos(obliquity)
    # * Eq. 53 divides both terms by cos β > 0, which the arctangent does not need.
    return jnp.arctan2(
        -sine * jnp.cos(right_ascension),
        sine * jnp.sin(declination) * jnp.sin(right_ascension)
        + cosine * jnp.cos(declination),
    )


class GalacticBinaryEpochShift(Transform):
    r"""Shift of the reference epoch of a Galactic binary by Δ = t_new − t_old.

    The frequency drifts linearly, f → f + ḟΔ, and the waveform phase 2πft − φ₀
    keeps its value at the new epoch if φ₀ → φ₀ − 2π(fΔ + ½ḟΔ²), as the
    ``GBObject`` of JaxGB shifts its epoch (:cite:t:`bayle2026jaxgb`). The
    phase is wrapped into [0, 2π). With the phase combinations θ± in place of
    φ₀, both shift by the same amount. The matrix of the map is triangular with
    a unit diagonal, so its determinant is one, and shifts compose: Δ₁ then Δ₂
    equals Δ₁ + Δ₂.

    Attributes:
        delta: The shift Δ in seconds.
        frequency: Position of the frequency in the vector.
        frequency_dot: Position of the frequency derivative.
        phases: Positions of φ₀, or of θ₊ and θ₋.

    References:
        :cite:t:`bayle2026jaxgb`; :cite:t:`baghi2026lisa`.
    """

    delta: float = eqx.field(static=True)
    frequency: int = eqx.field(static=True)
    frequency_dot: int = eqx.field(static=True)
    phases: tuple[int, ...] = eqx.field(static=True)

    @classmethod
    def from_space(
        cls, space: ParamSpace, *, delta: float
    ) -> "GalacticBinaryEpochShift":
        """Build the shift for the full vector of a Galactic-binary holder.

        Raises:
            ConventionError: If the vector holds no frequency derivative, as when
                it carries the chirp mass: convert it back first.
        """
        if "frequency_dot" not in space.names:
            raise ConventionError(
                "the epoch shift needs frequency_dot; convert the holder back with"
                " to_frequency_dot_and_amplitude() first"
            )
        names = space.names
        phases = tuple(
            position
            for position, name in enumerate(names)
            if name == "initial_phase" or name.startswith("theta_")
        )
        return cls(
            delta=delta,
            frequency=space.index("frequency"),
            frequency_dot=space.index("frequency_dot"),
            phases=phases,
        )

    def touched_names(self, space: ParamSpace) -> tuple[str, ...]:
        """Return the names of the parameters that the shift changes."""
        positions = [self.frequency, *self.phases]
        return tuple(space.names[position] for position in positions)

    def _shift(
        self, points: Float[Array, "*leading param"], delta: float
    ) -> Float[Array, "*leading param"]:
        frequency = points[..., self.frequency]
        drift = points[..., self.frequency_dot]
        phase_change = 2 * jnp.pi * (frequency * delta + 0.5 * drift * delta**2)
        mapped = points.at[..., self.frequency].set(frequency + drift * delta)
        for position in self.phases:
            mapped = mapped.at[..., position].set(
                wrap(points[..., position] - phase_change, 0.0, AZIMUTH_PERIOD)
            )
        return mapped

    def forward(
        self, points: Float[Array, "*leading param"]
    ) -> Float[Array, "*leading param"]:
        """Return the parameters at the new epoch."""
        return self._shift(points, self.delta)

    def inverse(
        self, points: Float[Array, "*leading param"]
    ) -> Float[Array, "*leading param"]:
        """Return the parameters at the old epoch, a shift by −Δ."""
        return self._shift(points, -self.delta)

    def log_abs_det_jacobian(
        self, points: Float[Array, "*leading param"]
    ) -> Float[Array, " *leading"]:
        """Return zero: the Jacobian is triangular with a unit diagonal."""
        return jnp.zeros(points.shape[:-1])

    @property
    def description(self) -> str:
        """The shift in seconds."""
        return f"epoch_shift({self.delta:g} s)"

    def map_space(self, space: ParamSpace) -> ParamSpace:
        """Return the space unchanged: names, units and domains stay."""
        return space


class CoalescenceTimeShift(Transform):
    """Shift of the time origin of the coalescence time, t_c → t_c − Δ.

    The coalescence time counts seconds from the epoch, so moving the epoch
    forward by Δ = t_new − t_old moves t_c back by Δ. The Jacobian determinant
    is one.

    Attributes:
        delta: The shift Δ in seconds.
        time: Position of the coalescence time in the vector.

    References:
        :cite:t:`baghi2026lisa`.
    """

    delta: float = eqx.field(static=True)
    time: int = eqx.field(static=True)

    @classmethod
    def from_space(cls, space: ParamSpace, *, delta: float) -> "CoalescenceTimeShift":
        """Build the shift for the full vector of a massive black-hole binary holder."""
        return cls(delta=delta, time=_index_of(space, "coalescence_time"))

    def touched_names(self, space: ParamSpace) -> tuple[str, ...]:
        """Return the name of the coalescence time."""
        return (space.names[self.time],)

    def forward(
        self, points: Float[Array, "*leading param"]
    ) -> Float[Array, "*leading param"]:
        """Return the parameters with t_c measured from the new epoch."""
        return points.at[..., self.time].add(-self.delta)

    def inverse(
        self, points: Float[Array, "*leading param"]
    ) -> Float[Array, "*leading param"]:
        """Return the parameters with t_c measured from the old epoch."""
        return points.at[..., self.time].add(self.delta)

    def log_abs_det_jacobian(
        self, points: Float[Array, "*leading param"]
    ) -> Float[Array, " *leading"]:
        """Return zero: a translation has a unit Jacobian."""
        return jnp.zeros(points.shape[:-1])

    @property
    def description(self) -> str:
        """The shift in seconds."""
        return f"epoch_shift({self.delta:g} s)"

    def map_space(self, space: ParamSpace) -> ParamSpace:
        """Return the space unchanged: the coalescence time lies on the real line."""
        return space


class PhaseCombinationTransform(Transform):
    r"""The map from (φ₀, ψ) of a Galactic binary to θ± = φ₀ ± 2ψ, and back.

    Derivation
        With the plus and cross amplitudes of SGS Eqs. 146 and 147, h₊ = −𝒜
        ½(1 + cos²ι) cos Φ and h× = −𝒜 cos ι sin Φ for the phase Φ = 2πft − φ₀,
        the strain at the SSB of SGS Eq. 69 is h₊cos 2ψ − h×sin 2ψ and
        h₊sin 2ψ + h×cos 2ψ in the two polarisation tensors. Products of
        cosines and sines turn each into a combination of cos(Φ ∓ 2ψ) and
        sin(Φ ∓ 2ψ), so the strain depends on (φ₀, ψ) only through
        cos and sin of φ₀ + 2ψ and of φ₀ − 2ψ, that is, through θ₊ and θ₋.
    Symmetries
        The map is 4-to-1: it identifies (φ₀, ψ), (φ₀, ψ + π),
        (φ₀ + π, ψ + π/2) and (φ₀ + π, ψ + 3π/2), the symmetries of the
        waveform. Both θ are full circles of period 2π.
    Inverse
        d = (θ₊ − θ₋) mod 2π, ψ = d/4 in [0, π/2) and φ₀ = (θ₊ − 2ψ) mod 2π: the
        representative with ψ below π/2.
    Density
        The local Jacobian matrix [[1, 2], [1, −2]], with rows (θ₊, θ₋) and
        columns (φ₀, ψ), has determinant −4; the full vector map has a
        determinant of magnitude 4, whose sign depends on the positions.
        :meth:`jacobian` keeps the matrix for Fisher matrices. Four preimages share each
        image, so a density that the four symmetries leave invariant keeps its
        value: the factor 4 of the Jacobian cancels the four preimages. The
        likelihood is invariant, and so are priors uniform in φ₀ and ψ, and
        ``log_abs_det_jacobian`` returns zero. A density that breaks the
        symmetries has no such map. No published use of these variables was
        found.
    Positions
        θ₊ takes the lower and θ₋ the higher of the two positions of φ₀ and ψ,
        and the inverse puts ψ back at the lower and φ₀ at the higher, as in the
        canonical order of Galactic binaries.

    Attributes:
        phase: Position of φ₀ in the vector.
        polarisation: Position of ψ.
        plus: Position of θ₊.
        minus: Position of θ₋.
        to_combinations: ``True`` for (φ₀, ψ) to θ±, ``False`` for the inverse.

    References:
        :cite:t:`baghi2026lisa`; :cite:t:`bayle2026lisa`.
    """

    phase: int = eqx.field(static=True)
    polarisation: int = eqx.field(static=True)
    plus: int = eqx.field(static=True)
    minus: int = eqx.field(static=True)
    to_combinations: bool = eqx.field(static=True)

    @classmethod
    def from_space(
        cls, space: ParamSpace, *, to_combinations: bool
    ) -> "PhaseCombinationTransform":
        """Build the transform for the full vector of a Galactic-binary holder.

        Raises:
            ConventionError: If the vector lacks the parameters that the
                direction needs.
        """
        names = space.names
        if to_combinations:
            polarisation = next(
                (name for name in names if name.startswith("polarisation_")), None
            )
            if "initial_phase" not in names or polarisation is None:
                raise ConventionError(
                    "the phase combinations need initial_phase and a polarisation"
                )
            positions = (space.index("initial_phase"), space.index(polarisation))
        else:
            plus = next(
                (name for name in names if name.startswith("theta_plus_")), None
            )
            minus = next(
                (name for name in names if name.startswith("theta_minus_")), None
            )
            if plus is None or minus is None:
                raise ConventionError(
                    "the vector holds no phase combinations theta_plus and theta_minus"
                )
            positions = (space.index(plus), space.index(minus))
        lower, upper = min(positions), max(positions)
        if to_combinations:
            phase, polarisation_position = positions
        else:
            phase, polarisation_position = upper, lower
        return cls(
            phase=phase,
            polarisation=polarisation_position,
            plus=lower,
            minus=upper,
            to_combinations=to_combinations,
        )

    def touched_names(self, space: ParamSpace) -> tuple[str, ...]:
        """Return the names of the two parameters that the map changes."""
        return (space.names[self.plus], space.names[self.minus])

    def _combine(
        self, points: Float[Array, "*leading param"]
    ) -> Float[Array, "*leading param"]:
        phase = points[..., self.phase]
        polarisation = points[..., self.polarisation]
        plus = wrap(phase + 2 * polarisation, 0.0, AZIMUTH_PERIOD)
        minus = wrap(phase - 2 * polarisation, 0.0, AZIMUTH_PERIOD)
        return points.at[..., self.plus].set(plus).at[..., self.minus].set(minus)

    def _split(
        self, points: Float[Array, "*leading param"]
    ) -> Float[Array, "*leading param"]:
        plus = points[..., self.plus]
        minus = points[..., self.minus]
        polarisation = wrap(plus - minus, 0.0, AZIMUTH_PERIOD) / 4
        phase = wrap(plus - 2 * polarisation, 0.0, AZIMUTH_PERIOD)
        return (
            points.at[..., self.phase]
            .set(phase)
            .at[..., self.polarisation]
            .set(polarisation)
        )

    def forward(
        self, points: Float[Array, "*leading param"]
    ) -> Float[Array, "*leading param"]:
        """Return θ± of (φ₀, ψ), or the representative (φ₀, ψ) of θ±."""
        return self._combine(points) if self.to_combinations else self._split(points)

    def inverse(
        self, points: Float[Array, "*leading param"]
    ) -> Float[Array, "*leading param"]:
        """Return the representative of the preimage, or θ± of (φ₀, ψ)."""
        return self._split(points) if self.to_combinations else self._combine(points)

    def log_abs_det_jacobian(
        self, points: Float[Array, "*leading param"]
    ) -> Float[Array, " *leading"]:
        """Return zero: the log density keeps its value under the symmetries."""
        return jnp.zeros(points.shape[:-1])

    @property
    def description(self) -> str:
        """The direction of the map."""
        return (
            "phase_combinations"
            if self.to_combinations
            else "initial_phase_and_polarisation"
        )

    def map_space(self, space: ParamSpace) -> ParamSpace:
        """Return the space with θ± in place of (φ₀, ψ), or the reverse."""
        if self.to_combinations:
            frame = _frame_of(space.parameters[self.polarisation].name)
            suffix = _ECLIPTIC_SUFFIX if frame is SkyFrame.ECLIPTIC else _ICRS_SUFFIX
            plus = Parameter(
                name=f"theta_plus{suffix}",
                label=r"\theta_+",
                unit="rad",
                topology=Topology.CIRCLE,
                bounds=(0.0, AZIMUTH_PERIOD),
            )
            minus = plus.model_copy(
                update={"name": f"theta_minus{suffix}", "label": r"\theta_-"}
            )
            replacements = {self.plus: plus, self.minus: minus}
        else:
            frame = _frame_of(space.parameters[self.plus].name)
            replacements = {
                self.phase: frame_parameter(_GALACTIC_SPECS["initial_phase"], frame),
                self.polarisation: frame_parameter(
                    _GALACTIC_SPECS["polarisation_icrs"], frame
                ),
            }
        return _replace_parameters(space, replacements)


class MassPairTransform(Transform):
    r"""The map between detector-frame masses (m₁, m₂) and a mass and a mass ratio.

    Forward, (m₁, m₂) becomes (M, q) with the total mass M = m₁ + m₂ and
    q = m₂/m₁, or (𝓜, q) with the chirp mass 𝓜 = (m₁m₂)^{3/5}/M^{1/5}
    (SGS Section 2.3, with m₁ ≥ m₂ so that q lies in (0, 1]). The inverse is
    m₁ = M/(1 + q) or m₁ = 𝓜(1 + q)^{1/5}q^{−3/5}, and m₂ = qm₁. The Jacobian
    determinants of the forward maps are M/m₁² and 𝓜/m₁², whose logarithms
    :meth:`log_abs_det_jacobian` returns, and minus the same at the preimage for
    the maps in the other direction. The edge m₁ = m₂ becomes the bound q = 1
    (:cite:t:`papamakarios2021normalizing`).

    The inverted mass ratio Q = 1/q = m₁/m₂ ≥ 1 is the form in which some priors
    are set. With ``inverted=True`` the ratio is Q and the edge m₁ = m₂ becomes
    Q = 1. The inverse then solves for m₂ first: m₂ = M/(1 + Q) or
    m₂ = 𝓜(1 + Q)^{1/5}Q^{−3/5}, and m₁ = Qm₂. The Jacobian determinants of the
    forward maps are M/m₂² and 𝓜/m₂². The maps are defined for any positive
    masses; only the intervals of q and Q assume m₁ ≥ m₂.

    Derivation of the Jacobian
        M and 𝓜 are homogeneous of degree one in (m₁, m₂), so Euler's theorem
        gives m₁ ∂mass/∂m₁ + m₂ ∂mass/∂m₂ = mass. For q = m₂/m₁ the
        determinant of (m₁, m₂) → (mass, q) is
        (m₁ ∂mass/∂m₁ + m₂ ∂mass/∂m₂)/m₁² = mass/m₁². For Q = m₁/m₂ it is
        −(m₁ ∂mass/∂m₁ + m₂ ∂mass/∂m₂)/m₂² = −mass/m₂², and its absolute value
        is mass/m₂².

    Attributes:
        primary: Position of m₁ in the vector, which M or 𝓜 takes.
        secondary: Position of m₂, which the ratio takes.
        chirp: ``True`` for the chirp mass, ``False`` for the total mass.
        to_ratio: ``True`` for (m₁, m₂) to the mass and the ratio, ``False``
            for the inverse.
        inverted: ``True`` for Q = m₁/m₂, ``False`` for q = m₂/m₁.

    References:
        :cite:t:`baghi2026lisa`; :cite:t:`papamakarios2021normalizing`.
    """

    primary: int = eqx.field(static=True)
    secondary: int = eqx.field(static=True)
    chirp: bool = eqx.field(static=True)
    to_ratio: bool = eqx.field(static=True)
    inverted: bool = eqx.field(static=True, default=False)

    def touched_names(self, space: ParamSpace) -> tuple[str, ...]:
        """Return the names of the two parameters that the map changes."""
        return (space.names[self.primary], space.names[self.secondary])

    def _to_ratio(
        self, points: Float[Array, "*leading param"]
    ) -> Float[Array, "*leading param"]:
        primary = points[..., self.primary]
        secondary = points[..., self.secondary]
        total = primary + secondary
        mass = (
            (primary * secondary) ** (3 / 5) / total ** (1 / 5) if self.chirp else total
        )
        ratio = primary / secondary if self.inverted else secondary / primary
        return points.at[..., self.primary].set(mass).at[..., self.secondary].set(ratio)

    def _to_components(
        self, points: Float[Array, "*leading param"]
    ) -> Float[Array, "*leading param"]:
        mass = points[..., self.primary]
        ratio = points[..., self.secondary]
        # * The ratio is numerator over denominator mass: m₂ over m₁, or m₁ over m₂
        # * if inverted; the denominator mass solves first.
        denominator = (
            mass * (1 + ratio) ** (1 / 5) * ratio ** (-3 / 5)
            if self.chirp
            else mass / (1 + ratio)
        )
        numerator = ratio * denominator
        primary, secondary = (
            (numerator, denominator) if self.inverted else (denominator, numerator)
        )
        return (
            points.at[..., self.primary]
            .set(primary)
            .at[..., self.secondary]
            .set(secondary)
        )

    def _log_determinant_of_ratio_map(
        self, components: Float[Array, "*leading param"]
    ) -> Float[Array, " *leading"]:
        """Return ln(mass/m₁²), or ln(mass/m₂²) if inverted, at the components."""
        denominator = components[..., self.secondary if self.inverted else self.primary]
        mapped = self._to_ratio(components)[..., self.primary]
        return jnp.log(mapped) - 2 * jnp.log(denominator)

    def forward(
        self, points: Float[Array, "*leading param"]
    ) -> Float[Array, "*leading param"]:
        """Return the points in the new mass parameters."""
        return self._to_ratio(points) if self.to_ratio else self._to_components(points)

    def inverse(
        self, points: Float[Array, "*leading param"]
    ) -> Float[Array, "*leading param"]:
        """Return the points in the original mass parameters."""
        return self._to_components(points) if self.to_ratio else self._to_ratio(points)

    def log_abs_det_jacobian(
        self, points: Float[Array, "*leading param"]
    ) -> Float[Array, " *leading"]:
        """Return ln|det ∂f/∂x|, with the sign of the direction of the map."""
        if self.to_ratio:
            return self._log_determinant_of_ratio_map(points)
        return -self._log_determinant_of_ratio_map(self._to_components(points))

    @property
    def description(self) -> str:
        """The mass parameters that the map produces."""
        if not self.to_ratio:
            return "component_masses"
        mass = "chirp_mass" if self.chirp else "total_mass"
        ratio = "inverted_mass_ratio" if self.inverted else "mass_ratio"
        return f"{mass}_and_{ratio}"

    def map_space(self, space: ParamSpace) -> ParamSpace:
        """Return the space with the mass and the mass ratio in place of the masses."""
        if self.to_ratio:
            replacements = {
                self.primary: _MASS_FORM_PARAMETERS[
                    "chirp_mass_det_frame" if self.chirp else "total_mass_det_frame"
                ],
                self.secondary: _MASS_FORM_PARAMETERS[
                    "inverted_mass_ratio" if self.inverted else "mass_ratio"
                ],
            }
        else:
            replacements = {
                self.primary: frame_parameter(
                    _MASS_SPECS["primary_mass_det_frame"], SkyFrame.ICRS
                ),
                self.secondary: frame_parameter(
                    _MASS_SPECS["secondary_mass_det_frame"], SkyFrame.ICRS
                ),
            }
        return _replace_parameters(space, replacements)


def chirp_mass_from_frequency_dot(
    frequency: Float[Array, " *leading"],
    frequency_dot: Float[Array, " *leading"],
    constants: LisaConstants,
) -> Float[Array, " *leading"]:
    """Return the chirp mass in solar masses from f and ḟ, NaN where ḟ ≤ 0.

    The leading-order chirp ḟ = (96/5)π^{8/3}(G𝓜/c³)^{5/3}f^{11/3}
    (:cite:t:`peters1964gravitational`) gives
    𝓜 = (c³/G)(5ḟ/(96π^{8/3}f^{11/3}))^{3/5}.
    """
    positive = frequency_dot > 0
    safe = jnp.where(positive, frequency_dot, 1.0)
    mass_kg = (constants.speed_of_light**3 / constants.gravitational_constant) * (
        5 * safe / (96 * jnp.pi ** (8 / 3) * frequency ** (11 / 3))
    ) ** (3 / 5)
    return jnp.where(positive, mass_kg / constants.solar_mass, jnp.nan)


def distance_from_amplitude(
    frequency: Float[Array, " *leading"],
    frequency_dot: Float[Array, " *leading"],
    amplitude: Float[Array, " *leading"],
    constants: LisaConstants,
) -> Float[Array, " *leading"]:
    """Return the distance in kpc from f, ḟ and 𝒜, NaN where ḟ ≤ 0.

    With 𝒜 = 4(G𝓜)^{5/3}(πf)^{2/3}/(c⁴D) (SGS Eqs. 146 and 147,
    :cite:t:`baghi2026lisa`) and the chirp of
    :func:`chirp_mass_from_frequency_dot`, D = 5cḟ/(24π²f³𝒜).
    """
    positive = frequency_dot > 0
    safe = jnp.where(positive, frequency_dot, 1.0)
    distance_m = (
        5
        * constants.speed_of_light
        * safe
        / (24 * jnp.pi**2 * frequency**3 * amplitude)
    )
    kiloparsec = _KILOPARSEC_PER_PARSEC * constants.parsec
    return jnp.where(positive, distance_m / kiloparsec, jnp.nan)


def frequency_dot_from_chirp_mass(
    frequency: Float[Array, " *leading"],
    chirp_mass: Float[Array, " *leading"],
    constants: LisaConstants,
) -> Float[Array, " *leading"]:
    """Return ḟ in Hz/s from f and the chirp mass in solar masses."""
    mass_kg = chirp_mass * constants.solar_mass
    mass_time = constants.gravitational_constant * mass_kg / constants.speed_of_light**3
    frequency_dot: Array = (
        (96 / 5) * jnp.pi ** (8 / 3) * mass_time ** (5 / 3) * frequency ** (11 / 3)
    )
    return frequency_dot


def amplitude_from_distance(
    frequency: Float[Array, " *leading"],
    chirp_mass: Float[Array, " *leading"],
    distance: Float[Array, " *leading"],
    constants: LisaConstants,
) -> Float[Array, " *leading"]:
    """Return the amplitude 𝒜 from f, the chirp mass in solar masses and D in kpc."""
    mass_kg = chirp_mass * constants.solar_mass
    distance_m = distance * _KILOPARSEC_PER_PARSEC * constants.parsec
    return (
        4
        * (constants.gravitational_constant * mass_kg) ** (5 / 3)
        * (jnp.pi * frequency) ** (2 / 3)
        / (constants.speed_of_light**4 * distance_m)
    )


class ChirpMassAndDistanceTransform(Transform):
    r"""The map from (ḟ, 𝒜) of a Galactic binary to the chirp mass and the distance.

    With the leading-order chirp ḟ = (96/5)π^{8/3}(G𝓜/c³)^{5/3}f^{11/3}
    (:cite:t:`peters1964gravitational`) and the amplitude
    𝒜 = 4(G𝓜)^{5/3}(πf)^{2/3}/(c⁴D) of SGS Eqs. 146 and 147
    (:cite:t:`baghi2026lisa`), the forward map gives
    𝓜 = (c³/G)(5ḟ/(96π^{8/3}f^{11/3}))^{3/5} in solar masses and
    D = 5cḟ/(24π²f³𝒜) in kiloparsecs. Frequency is unchanged. The matrix is
    triangular, so the absolute determinant is (3𝓜/(5ḟ))(D/𝒜) in the units of
    the parameters. A sample with ḟ ≤ 0 has no chirp mass; its image is NaN.
    JaxGB's amplitude is half of 𝒜, a known difference
    (:cite:t:`bayle2026jaxgb`).

    Attributes:
        frequency: Position of f in the vector.
        frequency_dot: Position of ḟ, which 𝓜 takes.
        amplitude: Position of 𝒜, which D takes.
        to_chirp_mass: ``True`` for (ḟ, 𝒜) to (𝓜, D), ``False`` for the inverse.
        constants: The lisaconstants values of the speed of light, Newton's
            constant, the solar mass and the parsec.

    References:
        :cite:t:`peters1964gravitational`; :cite:t:`baghi2026lisa`;
        :cite:t:`bayle2026lisa`.
    """

    frequency: int = eqx.field(static=True)
    frequency_dot: int = eqx.field(static=True)
    amplitude: int = eqx.field(static=True)
    to_chirp_mass: bool = eqx.field(static=True)
    constants: LisaConstants = eqx.field(static=True)

    @classmethod
    def from_space(
        cls, space: ParamSpace, *, to_chirp_mass: bool, constants: LisaConstants
    ) -> "ChirpMassAndDistanceTransform":
        """Build the transform for the full vector of a Galactic-binary holder.

        Raises:
            ConventionError: If the vector lacks the parameters that the
                direction needs.
        """
        names = (
            ("frequency_dot", "amplitude")
            if to_chirp_mass
            else ("chirp_mass", "distance")
        )
        return cls(
            frequency=_index_of(space, "frequency"),
            frequency_dot=_index_of(space, names[0]),
            amplitude=_index_of(space, names[1]),
            to_chirp_mass=to_chirp_mass,
            constants=constants,
        )

    def touched_names(self, space: ParamSpace) -> tuple[str, ...]:
        """Return the names of the two parameters that the map changes."""
        return (space.names[self.frequency_dot], space.names[self.amplitude])

    def _to_chirp_mass(
        self, points: Float[Array, "*leading param"]
    ) -> Float[Array, "*leading param"]:
        frequency = points[..., self.frequency]
        frequency_dot = points[..., self.frequency_dot]
        amplitude = points[..., self.amplitude]
        return (
            points.at[..., self.frequency_dot]
            .set(
                chirp_mass_from_frequency_dot(frequency, frequency_dot, self.constants)
            )
            .at[..., self.amplitude]
            .set(
                distance_from_amplitude(
                    frequency, frequency_dot, amplitude, self.constants
                )
            )
        )

    def _to_frequency_dot(
        self, points: Float[Array, "*leading param"]
    ) -> Float[Array, "*leading param"]:
        frequency = points[..., self.frequency]
        chirp_mass = points[..., self.frequency_dot]
        distance = points[..., self.amplitude]
        return (
            points.at[..., self.frequency_dot]
            .set(frequency_dot_from_chirp_mass(frequency, chirp_mass, self.constants))
            .at[..., self.amplitude]
            .set(
                amplitude_from_distance(frequency, chirp_mass, distance, self.constants)
            )
        )

    def _log_determinant_of_chirp_map(
        self, original: Float[Array, "*leading param"]
    ) -> Float[Array, " *leading"]:
        """Return ln((3𝓜/(5ḟ))(D/𝒜)) at points that hold f, ḟ and 𝒜."""
        mapped = self._to_chirp_mass(original)
        return (
            math.log(3 / 5)
            + jnp.log(mapped[..., self.frequency_dot])
            - jnp.log(original[..., self.frequency_dot])
            + jnp.log(mapped[..., self.amplitude])
            - jnp.log(original[..., self.amplitude])
        )

    def forward(
        self, points: Float[Array, "*leading param"]
    ) -> Float[Array, "*leading param"]:
        """Return the points with the chirp mass and distance, or the inverse map."""
        return (
            self._to_chirp_mass(points)
            if self.to_chirp_mass
            else self._to_frequency_dot(points)
        )

    def inverse(
        self, points: Float[Array, "*leading param"]
    ) -> Float[Array, "*leading param"]:
        """Return the points of the other form."""
        return (
            self._to_frequency_dot(points)
            if self.to_chirp_mass
            else self._to_chirp_mass(points)
        )

    def log_abs_det_jacobian(
        self, points: Float[Array, "*leading param"]
    ) -> Float[Array, " *leading"]:
        """Return ln|det ∂f/∂x|, NaN where ḟ ≤ 0, with the sign of the direction."""
        if self.to_chirp_mass:
            return self._log_determinant_of_chirp_map(points)
        return -self._log_determinant_of_chirp_map(self._to_frequency_dot(points))

    @property
    def description(self) -> str:
        """The parameters that the map produces."""
        return (
            "chirp_mass_and_distance"
            if self.to_chirp_mass
            else "frequency_dot_and_amplitude"
        )

    def map_space(self, space: ParamSpace) -> ParamSpace:
        """Return the space with 𝓜 and D in place of ḟ and 𝒜, or the reverse."""
        if self.to_chirp_mass:
            chirp_mass = Parameter(
                name="chirp_mass",
                label=r"\mathcal{M}",
                unit="Msun",
                topology=Topology.INTERVAL,
                bounds=(0.0, math.inf),
            )
            distance = Parameter(
                name="distance",
                label="D",
                unit="kpc",
                topology=Topology.INTERVAL,
                bounds=(0.0, math.inf),
            )
            replacements = {self.frequency_dot: chirp_mass, self.amplitude: distance}
        else:
            replacements = {
                self.frequency_dot: frame_parameter(
                    _GALACTIC_SPECS["frequency_dot"], SkyFrame.ICRS
                ),
                self.amplitude: frame_parameter(
                    _GALACTIC_SPECS["amplitude"], SkyFrame.ICRS
                ),
            }
        return _replace_parameters(space, replacements)
