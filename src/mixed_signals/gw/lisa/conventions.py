r"""Conventions of LISA sources: frames, epoch, and the parameter tables of every type.

A holder of LISA sources records the conventions of its parameters in a
:class:`LisaConventions`. Frames follow the SGS conventions document
(:cite:t:`baghi2026lisa`), which is the ground truth for them; names, units and
defaults follow the catpylisa conventions file, with the frame in the name of
every parameter that depends on it.

Sky frame
    ICRS, the equatorial frame (SGS Section 5.2), or the Barycentric Mean
    Ecliptic of J2000 (SGS Section 5.3). The sky position reads
    ``right_ascension`` and ``declination`` in ICRS, and ``ecliptic_longitude``
    and ``ecliptic_latitude`` in the ecliptic frame. The polarisation angle,
    which differs between the two frames (SGS Eq. 53), is named
    ``polarisation_icrs`` or ``polarisation_ecliptic``, and so are the spin
    angles measured in the sky frame, ``polar_angle_spin1_icrs`` and the others,
    and the phase combinations ``theta_plus_icrs`` and ``theta_minus_icrs`` of
    Galactic binaries. The names that catpylisa uses, such as ``polarisation``
    and ``polar_angle_spin1``, are aliases that resolve to the names of the
    frame of the holder.
Epoch
    The reference epoch in TCB seconds since ``LISA_EPOCH_TCB`` (SGS Section
    7.1). It is the time origin of every time parameter, such as the
    coalescence time, and the epoch of the frequency and phase of Galactic
    binaries.
Source frame
    The frame of the spin vectors of the binary, with the SGS labels ``LN``,
    ``L``, ``J``, ``S1``, ``LAL`` and ``FEW`` (SGS Section 5.6). Binaries with
    spins record it; Galactic binaries have none.
Mass frame
    The detector frame, in which the masses are redshifted by 1 + z (SGS
    Section 2.3).

:class:`ParameterSpec` describes one parameter of a type: its name, LaTeX
label, unit, topology, physical domain and default. The module holds the
tables of the four types, in the canonical order of catpylisa and with ICRS
names, and their aliases. :func:`rename_to_frame` gives the name of a
frame-dependent parameter in the other frame, and
:func:`require_equal_conventions` is the check of a comparison.

References:
    :cite:t:`baghi2026lisa`; :cite:t:`bayle2026lisa`.
"""

import math
from collections.abc import Mapping
from enum import StrEnum
from types import MappingProxyType
from typing import Final

from pydantic import BaseModel, ConfigDict, Field

from mixed_signals.core.space import Parameter
from mixed_signals.core.topology import PolarForm, Sphere, Topology
from mixed_signals.errors import ConventionError

__all__ = [
    "EMRI_ALIASES",
    "EMRI_SPECS",
    "EMRI_SPHERES",
    "GALACTIC_BINARY_ALIASES",
    "GALACTIC_BINARY_SPECS",
    "GALACTIC_BINARY_SPHERES",
    "MBHB_ALIASES",
    "MBHB_SPECS",
    "MBHB_SPHERES",
    "SOBBH_ALIASES",
    "SOBBH_SPECS",
    "SOBBH_SPHERES",
    "LisaConventions",
    "MassFrame",
    "ParameterSpec",
    "SkyFrame",
    "SourceFrame",
    "frame_aliases",
    "frame_parameter",
    "frame_sphere",
    "rename_label",
    "rename_parameter",
    "rename_to_frame",
    "require_equal_conventions",
]

_TWO_PI: Final = 2 * math.pi


class SkyFrame(StrEnum):
    """Frame of the sky position, the polarisation and the spin directions."""

    ICRS = "icrs"
    """The equatorial frame (SGS Section 5.2)."""
    ECLIPTIC = "ecliptic"
    """The Barycentric Mean Ecliptic of J2000 (SGS Section 5.3)."""


class SourceFrame(StrEnum):
    """Frame of the spin vectors of a binary; the values are the SGS labels.

    SGS Section 5.6 defines the frames. The conventions document announces a
    preferred source frame but names none, so holders of spinning binaries
    require one.
    """

    KINEMATIC = "LN"
    """The kinematic frame, whose z axis is the orbital angular momentum direction."""
    ORBITAL_ANGULAR_MOMENTUM = "L"
    """The frame of the orbital angular momentum."""
    TOTAL_ANGULAR_MOMENTUM = "J"
    """The frame of the total angular momentum."""
    PRIMARY_SPIN = "S1"
    """The frame of the spin of the primary."""
    LAL = "LAL"
    """The frame of the LAL conventions."""
    FEW = "FEW"
    """The frame of FastEMRIWaveforms."""


class MassFrame(StrEnum):
    """Frame of the masses; only the detector frame exists so far."""

    DETECTOR = "detector"
    """Masses redshifted by 1 + z (SGS Section 2.3)."""


class LisaConventions(BaseModel):
    """The conventions that a holder of LISA sources records.

    Attributes:
        sky_frame: Frame of the sky position, the polarisation and the spin
            directions.
        epoch: Reference epoch in TCB seconds since ``LISA_EPOCH_TCB``, finite.
        source_frame: Frame of the spin vectors of spinning binaries; ``None``
            for Galactic binaries.
        mass_frame: Frame of the masses.

    References:
        :cite:t:`baghi2026lisa`.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    sky_frame: SkyFrame = SkyFrame.ICRS
    epoch: float = Field(allow_inf_nan=False)
    source_frame: SourceFrame | None = None
    mass_frame: MassFrame = MassFrame.DETECTOR

    def differences(self, other: "LisaConventions") -> tuple[str, ...]:
        """Return the names of the fields in which two records differ."""
        return tuple(
            name
            for name in type(self).model_fields
            if getattr(self, name) != getattr(other, name)
        )


class ParameterSpec(BaseModel):
    """One parameter of a source type, with its physical domain and its default.

    Attributes:
        name: Canonical name, with the ICRS spelling of frame-dependent names.
        label: LaTeX label without ``$`` and without the unit.
        unit: Unit string, or ``None`` for a dimensionless parameter.
        topology: Topology of the parameter axis.
        bounds: Physical domain of the parameter; the period of a circle is
            the difference of its bounds.
        default: Value that fills a missing parameter, or ``None`` when the
            parameter is required.
        closed: Whether the lower and the upper bound belong to the domain.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    label: str
    unit: str | None = None
    topology: Topology
    bounds: tuple[float, float]
    default: float | None = None
    closed: tuple[bool, bool] = (True, True)


def _scale(
    name: str, label: str, unit: str | None, default: float | None = None
) -> ParameterSpec:
    """Return a spec on (0, ∞), a frequency, mass, distance or amplitude."""
    return ParameterSpec(
        name=name,
        label=label,
        unit=unit,
        topology=Topology.INTERVAL,
        bounds=(0.0, math.inf),
        default=default,
        closed=(False, False),
    )


def _angle(name: str, label: str, default: float | None = None) -> ParameterSpec:
    """Return a spec of a phase or an azimuth, a circle of period 2π in radians."""
    return ParameterSpec(
        name=name,
        label=label,
        unit="rad",
        topology=Topology.CIRCLE,
        bounds=(0.0, _TWO_PI),
        default=default,
        closed=(True, False),
    )


def _polar(
    name: str, label: str, upper: float, default: float | None = None
) -> ParameterSpec:
    """Return a spec of an angle on an interval from 0 to ``upper``, in radians."""
    return ParameterSpec(
        name=name,
        label=label,
        unit="rad",
        topology=Topology.INTERVAL,
        bounds=(0.0, upper),
        default=default,
    )


def _spin(name: str, label: str, default: float | None = None) -> ParameterSpec:
    """Return a spec of a dimensionless spin magnitude on [−1, 1]."""
    return ParameterSpec(
        name=name,
        label=label,
        topology=Topology.INTERVAL,
        bounds=(-1.0, 1.0),
        default=default,
    )


def _eccentricity(name: str, default: float | None = None) -> ParameterSpec:
    """Return a spec of an eccentricity at the reference time, on [0, 1)."""
    return ParameterSpec(
        name=name,
        label=r"e(t_{\mathrm{ref}})",
        topology=Topology.INTERVAL,
        bounds=(0.0, 1.0),
        default=default,
        closed=(True, False),
    )


_DECLINATION: Final = ParameterSpec(
    name="declination",
    label=r"\delta",
    unit="rad",
    topology=Topology.INTERVAL,
    bounds=(-math.pi / 2, math.pi / 2),
)
_RIGHT_ASCENSION: Final = _angle("right_ascension", r"\alpha")
_POLARISATION: Final = _angle("polarisation_icrs", r"\psi")
_SKY_SPHERE: Final = Sphere(
    azimuth="right_ascension", polar="declination", polar_form=PolarForm.LATITUDE
)
_PRIMARY_MASS: Final = _scale("primary_mass_det_frame", r"M_{1,\mathrm{det}}", "Msun")
_SECONDARY_MASS: Final = _scale(
    "secondary_mass_det_frame", r"M_{2,\mathrm{det}}", "Msun"
)
_LUMINOSITY_DISTANCE: Final = _scale("luminosity_distance", "D_L", "Gpc")
_INCLINATION: Final = _polar("inclination", r"\iota", math.pi)

GALACTIC_BINARY_SPECS: Final = (
    _scale("frequency", "f", "Hz"),
    ParameterSpec(
        name="frequency_dot",
        label=r"\dot{f}",
        unit="Hz/s",
        topology=Topology.REAL,
        bounds=(-math.inf, math.inf),
        default=0.0,
        closed=(False, False),
    ),
    _scale("amplitude", r"\mathcal{A}", None),
    _RIGHT_ASCENSION,
    _DECLINATION,
    _POLARISATION,
    _INCLINATION,
    _angle("initial_phase", r"\phi_0", default=0.0),
)
"""Galactic binaries in the canonical order (SGS Eqs. 146 and 147)."""

GALACTIC_BINARY_SPHERES: Final = (_SKY_SPHERE,)
"""The sky position of a Galactic binary."""

MBHB_SPECS: Final = (
    _PRIMARY_MASS,
    _SECONDARY_MASS,
    _LUMINOSITY_DISTANCE,
    _spin("spin1", r"\chi_1", default=0.0),
    _spin("spin2", r"\chi_2", default=0.0),
    ParameterSpec(
        name="coalescence_time",
        label="t_c",
        unit="s",
        topology=Topology.REAL,
        bounds=(-math.inf, math.inf),
        closed=(False, False),
    ),
    _angle("phase_at_reference_time", r"\phi_{\mathrm{ref}}", default=0.0),
    _scale("frequency_at_reference_time", r"f_{\mathrm{ref}}", "Hz"),
    _RIGHT_ASCENSION,
    _DECLINATION,
    _INCLINATION,
    _POLARISATION,
)
"""Massive black-hole binaries, in the canonical order (SGS Sections 2.3 and 5.10.2)."""

MBHB_SPHERES: Final = (_SKY_SPHERE,)
"""The sky position of a massive black-hole binary."""

_SPIN_DIRECTION_SPHERES: Final = tuple(
    Sphere(
        azimuth=f"azimuthal_angle_spin{index}_icrs",
        polar=f"polar_angle_spin{index}_icrs",
        polar_form=PolarForm.COLATITUDE,
    )
    for index in (1, 2)
)

EMRI_SPECS: Final = (
    _PRIMARY_MASS,
    _SECONDARY_MASS,
    _spin("chi_1", r"\chi_1"),
    _polar("polar_angle_spin1_icrs", r"\theta_{S,1}", math.pi, default=0.0),
    _angle("azimuthal_angle_spin1_icrs", r"\phi_{S,1}", default=0.0),
    _scale("semi_latus_rectum_tref", r"p(t_{\mathrm{ref}})", None),
    _eccentricity("eccentricity_tref"),
    _angle("azimuthal_orbit_phase_tref", r"\Phi_\phi(t_{\mathrm{ref}})"),
    _angle("mean_anomaly_tref", r"\Phi_r(t_{\mathrm{ref}})"),
    _LUMINOSITY_DISTANCE,
    _RIGHT_ASCENSION,
    _DECLINATION,
    _polar("inclination_tref", r"\iota(t_{\mathrm{ref}})", math.pi, default=0.0),
    _angle("polar_orbit_phase_tref", r"\Phi_\theta(t_{\mathrm{ref}})", default=0.0),
)
"""Extreme mass-ratio inspirals (SGS Sections 5.6.7, 5.10.1 and Appendix A)."""

EMRI_SPHERES: Final = (_SKY_SPHERE, _SPIN_DIRECTION_SPHERES[0])
"""The sky position and the direction of the primary spin of an EMRI."""

SOBBH_SPECS: Final = (
    _PRIMARY_MASS,
    _SECONDARY_MASS,
    _spin("spin_1", r"\chi_1"),
    _spin("spin_2", r"\chi_2"),
    _scale("f_ref", r"f_{\mathrm{ref}}", "Hz"),
    _INCLINATION,
    _POLARISATION,
    _angle("phase_tref", r"\phi_{\mathrm{ref}}", default=0.0),
    _LUMINOSITY_DISTANCE,
    _RIGHT_ASCENSION,
    _DECLINATION,
    _eccentricity("eccentricity_tref", default=0.0),
    _polar("polar_angle_spin1_icrs", r"\theta_{S,1}", math.pi, default=0.0),
    _polar("polar_angle_spin2_icrs", r"\theta_{S,2}", math.pi, default=0.0),
    _angle("azimuthal_angle_spin1_icrs", r"\phi_{S,1}", default=0.0),
    _angle("azimuthal_angle_spin2_icrs", r"\phi_{S,2}", default=0.0),
)
"""Stellar-origin binary black holes (SGS Sections 2.3 and 5.10.3)."""

SOBBH_SPHERES: Final = (_SKY_SPHERE, *_SPIN_DIRECTION_SPHERES)
"""The sky position and the directions of the two spins of a SOBBH."""

# * Aliases of the catpylisa conventions: external spellings, so single letters occur.
_COMMON_ALIASES: Final = {
    "ra": "right_ascension",
    "dec": "declination",
    "inc": "inclination",
    "iota": "inclination",
    "lam": "right_ascension",
    "lambda": "right_ascension",
    "beta": "declination",
}
_POLARISATION_ALIASES: Final = {
    "psi": "polarisation_icrs",
    "polarisation": "polarisation_icrs",
    "polarization": "polarisation_icrs",
}

GALACTIC_BINARY_ALIASES: Final[Mapping[str, str]] = MappingProxyType(
    {
        "f": "frequency",
        "f0": "frequency",
        "gw_frequency": "frequency",
        "fdot": "frequency_dot",
        "df_dt": "frequency_dot",
        "A": "amplitude",
        "amp": "amplitude",
        "strain_amplitude": "amplitude",
        "phi0": "initial_phase",
        "phi": "initial_phase",
        "phase0": "initial_phase",
        "initial_phase_angle": "initial_phase",
        **_COMMON_ALIASES,
        **_POLARISATION_ALIASES,
    }
)
"""Aliases of Galactic binaries, mapped to the canonical names with ICRS spelling."""

_DISTANCE_ALIASES: Final = {
    "dl": "luminosity_distance",
    "dist": "luminosity_distance",
    "distance": "luminosity_distance",
    # ? The conventions file spells the SOBBH distance "luministy_distance".
    "luministy_distance": "luminosity_distance",
}

MBHB_ALIASES: Final[Mapping[str, str]] = MappingProxyType(
    {
        "m1": "primary_mass_det_frame",
        "mass1": "primary_mass_det_frame",
        "primary_mass": "primary_mass_det_frame",
        "m2": "secondary_mass_det_frame",
        "mass2": "secondary_mass_det_frame",
        "secondary_mass": "secondary_mass_det_frame",
        "chi1": "spin1",
        "chi_1": "spin1",
        "spin_1": "spin1",
        "chi2": "spin2",
        "chi_2": "spin2",
        "spin_2": "spin2",
        "tc": "coalescence_time",
        "t_c": "coalescence_time",
        "merger_time": "coalescence_time",
        "phi_ref": "phase_at_reference_time",
        "phase_ref": "phase_at_reference_time",
        "phi0": "phase_at_reference_time",
        "initial_phase": "phase_at_reference_time",
        "f_ref": "frequency_at_reference_time",
        "fref": "frequency_at_reference_time",
        "reference_frequency": "frequency_at_reference_time",
        **_DISTANCE_ALIASES,
        **_COMMON_ALIASES,
        **_POLARISATION_ALIASES,
    }
)
"""Aliases of massive black-hole binaries."""

# ? catpylisa maps "xI" to inclination_tref, but FEW's x_I is the inclination's cosine.
EMRI_ALIASES: Final[Mapping[str, str]] = MappingProxyType(
    {
        "m1": "primary_mass_det_frame",
        "mass1": "primary_mass_det_frame",
        "M": "primary_mass_det_frame",
        "primary_mass": "primary_mass_det_frame",
        "m2": "secondary_mass_det_frame",
        "mass2": "secondary_mass_det_frame",
        "mu": "secondary_mass_det_frame",
        "secondary_mass": "secondary_mass_det_frame",
        "a": "chi_1",
        "spin1": "chi_1",
        "spin_1": "chi_1",
        "primary_spin": "chi_1",
        "theta_s": "polar_angle_spin1_icrs",
        "polar_spin1": "polar_angle_spin1_icrs",
        "polar_angle_spin1": "polar_angle_spin1_icrs",
        "phi_s": "azimuthal_angle_spin1_icrs",
        "azimuthal_spin1": "azimuthal_angle_spin1_icrs",
        "azimuthal_angle_spin1": "azimuthal_angle_spin1_icrs",
        "p0": "semi_latus_rectum_tref",
        "p_tref": "semi_latus_rectum_tref",
        "p": "semi_latus_rectum_tref",
        "semi_latus_rectum": "semi_latus_rectum_tref",
        "e0": "eccentricity_tref",
        "e_tref": "eccentricity_tref",
        "e": "eccentricity_tref",
        "eccentricity": "eccentricity_tref",
        "phi_phi0": "azimuthal_orbit_phase_tref",
        "Phi_phi": "azimuthal_orbit_phase_tref",
        "azimuthal_phase": "azimuthal_orbit_phase_tref",
        "phi_r0": "mean_anomaly_tref",
        "Phi_r": "mean_anomaly_tref",
        "mean_anomaly": "mean_anomaly_tref",
        "phi_theta0": "polar_orbit_phase_tref",
        "Phi_theta": "polar_orbit_phase_tref",
        **_DISTANCE_ALIASES,
        **{
            alias: "inclination_tref" if target == "inclination" else target
            for alias, target in _COMMON_ALIASES.items()
        },
        "inclination": "inclination_tref",
    }
)
"""Aliases of extreme mass-ratio inspirals; the inclination is at the reference time."""

SOBBH_ALIASES: Final[Mapping[str, str]] = MappingProxyType(
    {
        "m1": "primary_mass_det_frame",
        "mass1": "primary_mass_det_frame",
        "primary_mass": "primary_mass_det_frame",
        "m2": "secondary_mass_det_frame",
        "mass2": "secondary_mass_det_frame",
        "secondary_mass": "secondary_mass_det_frame",
        "spin1": "spin_1",
        "chi1": "spin_1",
        "chi_1": "spin_1",
        "spin2": "spin_2",
        "chi2": "spin_2",
        "chi_2": "spin_2",
        "fref": "f_ref",
        "frequency_ref": "f_ref",
        "frequency_at_reference_time": "f_ref",
        "phase_ref": "phase_tref",
        "phi_ref": "phase_tref",
        "phi_tref": "phase_tref",
        "initial_phase": "phase_tref",
        "e0": "eccentricity_tref",
        "e_tref": "eccentricity_tref",
        "e": "eccentricity_tref",
        "eccentricity": "eccentricity_tref",
        "polar_angle_spin1": "polar_angle_spin1_icrs",
        "polar_angle_spin2": "polar_angle_spin2_icrs",
        "azimuthal_angle_spin1": "azimuthal_angle_spin1_icrs",
        "azimuthal_angle_spin2": "azimuthal_angle_spin2_icrs",
        **_DISTANCE_ALIASES,
        **_COMMON_ALIASES,
        **_POLARISATION_ALIASES,
    }
)
"""Aliases of stellar-origin binary black holes."""

_ECLIPTIC_NAMES: Final = {
    "right_ascension": "ecliptic_longitude",
    "declination": "ecliptic_latitude",
}
_ECLIPTIC_LABELS: Final = {r"\alpha": r"\lambda", r"\delta": r"\beta"}
_ICRS_LABELS: Final = {label: icrs for icrs, label in _ECLIPTIC_LABELS.items()}
_ICRS_SUFFIX: Final = "_icrs"
_ECLIPTIC_SUFFIX: Final = "_ecliptic"
# * Aliases that name one frame only; the other frame raises a ConventionError.
_ICRS_ONLY_ALIASES: Final = frozenset({"ra", "dec"})
_ECLIPTIC_ONLY_ALIASES: Final = frozenset({"lam", "lambda", "beta"})


def rename_to_frame(name: str, frame: SkyFrame) -> str:
    """Return the name of a parameter as it reads in a sky frame.

    A name that does not depend on the frame is returned unchanged, and a
    frame-dependent name is returned in the frame asked for, from either spelling.

    Args:
        name: A parameter name in ICRS or ecliptic spelling.
        frame: The frame whose spelling is wanted.

    Returns:
        The name in that frame, such as ``ecliptic_longitude`` or
        ``polarisation_icrs``.
    """
    if frame is SkyFrame.ECLIPTIC:
        if name in _ECLIPTIC_NAMES:
            return _ECLIPTIC_NAMES[name]
        return _swap_suffix(name, _ICRS_SUFFIX, _ECLIPTIC_SUFFIX)
    for icrs_name, ecliptic_name in _ECLIPTIC_NAMES.items():
        if name == ecliptic_name:
            return icrs_name
    return _swap_suffix(name, _ECLIPTIC_SUFFIX, _ICRS_SUFFIX)


def _swap_suffix(name: str, old: str, new: str) -> str:
    """Replace a trailing ``old`` of a name by ``new``, or return the name."""
    return name.removesuffix(old) + new if name.endswith(old) else name


def rename_label(label: str, frame: SkyFrame) -> str:
    """Return the LaTeX label of a parameter as it reads in a sky frame.

    Only the sky position changes: α and δ in ICRS read λ and β in the ecliptic
    frame. Other labels are returned unchanged.
    """
    if frame is SkyFrame.ECLIPTIC:
        return _ECLIPTIC_LABELS.get(label, label)
    return _ICRS_LABELS.get(label, label)


def rename_parameter(parameter: Parameter, frame: SkyFrame) -> Parameter:
    """Return a parameter with the name and label of a sky frame, other fields kept."""
    return parameter.model_copy(
        update={
            "name": rename_to_frame(parameter.name, frame),
            "label": rename_label(parameter.label, frame),
        }
    )


def frame_parameter(spec: ParameterSpec, frame: SkyFrame) -> Parameter:
    """Return the parameter of a spec as it reads in a sky frame.

    Args:
        spec: A spec with ICRS names.
        frame: The sky frame of the holder.

    Returns:
        The parameter with the name and label of that frame.
    """
    parameter = Parameter(
        name=spec.name,
        label=spec.label,
        unit=spec.unit,
        topology=spec.topology,
        bounds=spec.bounds,
    )
    return rename_parameter(parameter, frame)


def frame_sphere(sphere: Sphere, frame: SkyFrame) -> Sphere:
    """Return a sphere named with the spelling of a sky frame."""
    return sphere.model_copy(
        update={
            "azimuth": rename_to_frame(sphere.azimuth, frame),
            "polar": rename_to_frame(sphere.polar, frame),
        }
    )


def frame_aliases(
    aliases: Mapping[str, str], frame: SkyFrame
) -> tuple[dict[str, str], frozenset[str]]:
    """Return the aliases of a type as they resolve in a sky frame.

    Args:
        aliases: The alias table of a type, mapped to ICRS names.
        frame: The sky frame of the holder.

    Returns:
        The aliases that resolve in the frame, mapped to its names, and the
        aliases that name the other frame, which a holder refuses.
    """
    other_only = (
        _ECLIPTIC_ONLY_ALIASES if frame is SkyFrame.ICRS else _ICRS_ONLY_ALIASES
    )
    resolved = {
        alias: rename_to_frame(target, frame)
        for alias, target in aliases.items()
        if alias not in other_only
    }
    return resolved, frozenset(alias for alias in other_only if alias in aliases)


def require_equal_conventions(
    first: LisaConventions,
    second: LisaConventions,
    *,
    epoch_shift: bool,
) -> None:
    """Raise a ConventionError that names every difference of two records.

    Args:
        first: Conventions of the first holder.
        second: Conventions of the second holder.
        epoch_shift: Whether the holders can shift their epoch, which is the case
            for Galactic binaries and massive black-hole binaries.

    Raises:
        ConventionError: If the records differ. The message names each differing
            field with both values and the holder method that aligns the holders,
            or says that no transform exists yet.
    """
    differing = first.differences(second)
    if not differing:
        return
    reasons = []
    for field in differing:
        values = f"{getattr(first, field)} and {getattr(second, field)}"
        if field == "sky_frame":
            remedy = "align them with to_icrs() or to_ecliptic()"
        elif field == "epoch" and epoch_shift:
            remedy = "align them with at_epoch(epoch)"
        else:
            remedy = "no transform exists yet"
        reasons.append(f"{field} ({values}): {remedy}")
    raise ConventionError(
        "the holders differ in their conventions; " + "; ".join(reasons)
    )
