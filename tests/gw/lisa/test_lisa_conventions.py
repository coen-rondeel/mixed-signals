"""Tests for the conventions record, the frame names and the parameter tables."""

import math

import pytest
from pydantic import ValidationError

from mixed_signals.core.space import ParamSpace
from mixed_signals.core.topology import PolarForm, Topology
from mixed_signals.errors import ConventionError
from mixed_signals.gw.lisa.conventions import (
    EMRI_ALIASES,
    EMRI_SPECS,
    EMRI_SPHERES,
    GALACTIC_BINARY_ALIASES,
    GALACTIC_BINARY_SPECS,
    GALACTIC_BINARY_SPHERES,
    MBHB_ALIASES,
    MBHB_SPECS,
    MBHB_SPHERES,
    SOBBH_ALIASES,
    SOBBH_SPECS,
    SOBBH_SPHERES,
    LisaConventions,
    MassFrame,
    ParameterSpec,
    SkyFrame,
    SourceFrame,
    frame_aliases,
    frame_parameter,
    frame_sphere,
    rename_label,
    rename_parameter,
    rename_to_frame,
    require_equal_conventions,
)

TABLES = {
    "galactic_binary": (
        GALACTIC_BINARY_SPECS,
        GALACTIC_BINARY_SPHERES,
        GALACTIC_BINARY_ALIASES,
    ),
    "mbhb": (MBHB_SPECS, MBHB_SPHERES, MBHB_ALIASES),
    "emri": (EMRI_SPECS, EMRI_SPHERES, EMRI_ALIASES),
    "sobbh": (SOBBH_SPECS, SOBBH_SPHERES, SOBBH_ALIASES),
}
CANONICAL_ORDER = {
    "galactic_binary": (
        "frequency",
        "frequency_dot",
        "amplitude",
        "right_ascension",
        "declination",
        "polarisation_icrs",
        "inclination",
        "initial_phase",
    ),
    "mbhb": (
        "primary_mass_det_frame",
        "secondary_mass_det_frame",
        "luminosity_distance",
        "spin1",
        "spin2",
        "coalescence_time",
        "phase_at_reference_time",
        "frequency_at_reference_time",
        "right_ascension",
        "declination",
        "inclination",
        "polarisation_icrs",
    ),
    "emri": (
        "primary_mass_det_frame",
        "secondary_mass_det_frame",
        "chi_1",
        "polar_angle_spin1_icrs",
        "azimuthal_angle_spin1_icrs",
        "semi_latus_rectum_tref",
        "eccentricity_tref",
        "azimuthal_orbit_phase_tref",
        "mean_anomaly_tref",
        "luminosity_distance",
        "right_ascension",
        "declination",
        "inclination_tref",
        "polar_orbit_phase_tref",
    ),
    "sobbh": (
        "primary_mass_det_frame",
        "secondary_mass_det_frame",
        "spin_1",
        "spin_2",
        "f_ref",
        "inclination",
        "polarisation_icrs",
        "phase_tref",
        "luminosity_distance",
        "right_ascension",
        "declination",
        "eccentricity_tref",
        "polar_angle_spin1_icrs",
        "polar_angle_spin2_icrs",
        "azimuthal_angle_spin1_icrs",
        "azimuthal_angle_spin2_icrs",
    ),
}


# * The conventions record


def test_enums_carry_the_sgs_labels() -> None:
    """Source frames take the SGS labels as values, and the sky frames their names."""
    assert [frame.value for frame in SourceFrame] == [
        "LN",
        "L",
        "J",
        "S1",
        "LAL",
        "FEW",
    ]
    assert SkyFrame.ICRS.value == "icrs"
    assert SkyFrame.ECLIPTIC.value == "ecliptic"
    assert MassFrame.DETECTOR.value == "detector"


def test_the_record_defaults_to_icrs_and_the_detector_frame() -> None:
    """Only the epoch is required, and strings read as enum members."""
    record = LisaConventions(epoch=10.0)
    assert record.sky_frame is SkyFrame.ICRS
    assert record.mass_frame is MassFrame.DETECTOR
    assert record.source_frame is None
    assert (
        LisaConventions(epoch=0.0, source_frame="J").source_frame
        is SourceFrame.TOTAL_ANGULAR_MOMENTUM
    )


def test_the_record_is_frozen_and_rejects_extra_fields_and_bad_epochs() -> None:
    """A record cannot change, and an epoch must be a finite number."""
    record = LisaConventions(epoch=0.0)
    with pytest.raises(ValidationError):
        record.epoch = 1.0  # type: ignore[misc]
    with pytest.raises(ValidationError):
        LisaConventions(epoch=0.0, other=1)  # type: ignore[call-arg]
    for epoch in (math.nan, math.inf):
        with pytest.raises(ValidationError):
            LisaConventions(epoch=epoch)


def test_differences_list_the_differing_fields_in_order() -> None:
    """Equal records differ nowhere, and others list their fields in field order."""
    first = LisaConventions(epoch=0.0, source_frame="J")
    assert first.differences(first.model_copy()) == ()
    other = LisaConventions(sky_frame="ecliptic", epoch=5.0, source_frame="L")
    assert first.differences(other) == ("sky_frame", "epoch", "source_frame")


def test_equal_conventions_pass_the_check() -> None:
    """Equal records raise nothing."""
    record = LisaConventions(epoch=0.0)
    require_equal_conventions(record, record.model_copy(), epoch_shift=True)


def test_a_sky_frame_mismatch_names_the_aligning_methods() -> None:
    """The message names the field, both values and to_icrs() or to_ecliptic()."""
    first = LisaConventions(epoch=0.0)
    second = LisaConventions(sky_frame="ecliptic", epoch=0.0)
    with pytest.raises(
        ConventionError,
        match=r"sky_frame .*icrs.*ecliptic.*to_icrs\(\) or to_ecliptic\(\)",
    ):
        require_equal_conventions(first, second, epoch_shift=True)


def test_an_epoch_mismatch_names_at_epoch_where_it_exists() -> None:
    """Galactic binaries and MBHBs can shift the epoch; the other types cannot."""
    first, second = LisaConventions(epoch=0.0), LisaConventions(epoch=3.0)
    with pytest.raises(
        ConventionError, match=r"epoch .*0\.0 and 3\.0.*at_epoch\(epoch\)"
    ):
        require_equal_conventions(first, second, epoch_shift=True)
    with pytest.raises(ConventionError, match=r"epoch .*no transform exists yet"):
        require_equal_conventions(first, second, epoch_shift=False)


def test_a_source_frame_mismatch_says_that_no_transform_exists() -> None:
    """Source frames have no transform yet, and every difference is named."""
    first = LisaConventions(sky_frame="ecliptic", epoch=0.0, source_frame="J")
    second = LisaConventions(epoch=0.0, source_frame="L")
    with pytest.raises(ConventionError) as caught:
        require_equal_conventions(first, second, epoch_shift=True)
    message = str(caught.value)
    assert "sky_frame (ecliptic and icrs): align them with" in message
    assert "source_frame (J and L): no transform exists yet" in message
    assert isinstance(caught.value, ValueError)


# * Frame names


@pytest.mark.parametrize(
    ("icrs", "ecliptic"),
    [
        ("right_ascension", "ecliptic_longitude"),
        ("declination", "ecliptic_latitude"),
        ("polarisation_icrs", "polarisation_ecliptic"),
        ("polar_angle_spin1_icrs", "polar_angle_spin1_ecliptic"),
        ("azimuthal_angle_spin2_icrs", "azimuthal_angle_spin2_ecliptic"),
        ("theta_plus_icrs", "theta_plus_ecliptic"),
        ("theta_minus_icrs", "theta_minus_ecliptic"),
    ],
)
def test_frame_dependent_names_swap_in_both_directions(
    icrs: str, ecliptic: str
) -> None:
    """Each name reads in both frames, from either spelling."""
    assert rename_to_frame(icrs, SkyFrame.ECLIPTIC) == ecliptic
    assert rename_to_frame(ecliptic, SkyFrame.ICRS) == icrs
    assert rename_to_frame(icrs, SkyFrame.ICRS) == icrs
    assert rename_to_frame(ecliptic, SkyFrame.ECLIPTIC) == ecliptic


@pytest.mark.parametrize(
    "name", ["frequency", "inclination", "mass_ratio", "chirp_mass"]
)
def test_frame_independent_names_stay(name: str) -> None:
    """A name that does not depend on the sky frame is unchanged."""
    for frame in SkyFrame:
        assert rename_to_frame(name, frame) == name


def test_labels_follow_the_frame_of_the_sky_position() -> None:
    """α and δ read λ and β in the ecliptic frame, and back; others stay."""
    assert rename_label(r"\alpha", SkyFrame.ECLIPTIC) == r"\lambda"
    assert rename_label(r"\delta", SkyFrame.ECLIPTIC) == r"\beta"
    assert rename_label(r"\lambda", SkyFrame.ICRS) == r"\alpha"
    assert rename_label(r"\beta", SkyFrame.ICRS) == r"\delta"
    assert rename_label(r"\psi", SkyFrame.ECLIPTIC) == r"\psi"
    assert rename_label(r"\psi", SkyFrame.ICRS) == r"\psi"


def test_parameters_and_spheres_take_the_names_of_a_frame() -> None:
    """A spec becomes a parameter with the frame's name and label; spheres follow."""
    spec = next(
        spec for spec in GALACTIC_BINARY_SPECS if spec.name == "right_ascension"
    )
    icrs = frame_parameter(spec, SkyFrame.ICRS)
    ecliptic = frame_parameter(spec, SkyFrame.ECLIPTIC)
    assert (icrs.name, icrs.label) == ("right_ascension", r"\alpha")
    assert (ecliptic.name, ecliptic.label) == ("ecliptic_longitude", r"\lambda")
    assert ecliptic.unit == "rad"
    assert rename_parameter(icrs, SkyFrame.ECLIPTIC) == ecliptic
    sphere = frame_sphere(GALACTIC_BINARY_SPHERES[0], SkyFrame.ECLIPTIC)
    assert (sphere.azimuth, sphere.polar) == ("ecliptic_longitude", "ecliptic_latitude")
    assert sphere.polar_form is PolarForm.LATITUDE


def test_frame_aliases_resolve_to_the_frame_and_flag_the_other_frame() -> None:
    """The polarisation aliases resolve in both frames, ra and dec in ICRS only."""
    icrs, icrs_other = frame_aliases(GALACTIC_BINARY_ALIASES, SkyFrame.ICRS)
    assert icrs["psi"] == "polarisation_icrs"
    assert icrs["ra"] == "right_ascension"
    assert "lam" not in icrs
    assert icrs_other == {"lam", "lambda", "beta"}
    ecliptic, ecliptic_other = frame_aliases(GALACTIC_BINARY_ALIASES, SkyFrame.ECLIPTIC)
    assert ecliptic["psi"] == "polarisation_ecliptic"
    assert ecliptic["lam"] == "ecliptic_longitude"
    assert ecliptic["beta"] == "ecliptic_latitude"
    assert "ra" not in ecliptic
    assert ecliptic_other == {"ra", "dec"}


# * The tables


@pytest.mark.parametrize("kind", TABLES)
def test_tables_follow_the_canonical_order(kind: str) -> None:
    """The names and their order are those of catpylisa, with the frame suffixes."""
    specs, _, _ = TABLES[kind]
    assert tuple(spec.name for spec in specs) == CANONICAL_ORDER[kind]


@pytest.mark.parametrize("kind", TABLES)
def test_every_spec_is_a_valid_parameter_and_every_sphere_a_valid_sphere(
    kind: str,
) -> None:
    """The specs build parameters in both frames, and the spheres validate."""
    specs, spheres, _ = TABLES[kind]
    for frame in SkyFrame:
        parameters = tuple(frame_parameter(spec, frame) for spec in specs)
        space = ParamSpace(
            parameters=parameters,
            spheres=tuple(frame_sphere(sphere, frame) for sphere in spheres),
        )
        assert len(space) == len(specs)


@pytest.mark.parametrize("kind", TABLES)
def test_alias_targets_are_canonical_names(kind: str) -> None:
    """Every alias resolves to a name of the type, and none is the xI of FEW."""
    specs, _, aliases = TABLES[kind]
    names = {spec.name for spec in specs}
    assert set(aliases.values()) <= names
    assert "xI" not in aliases


def test_spec_bounds_and_topologies_follow_the_conventions() -> None:
    """Intervals, circles and the real line carry the physical domains."""
    by_name = {
        spec.name: spec for spec in GALACTIC_BINARY_SPECS + MBHB_SPECS + EMRI_SPECS
    }
    assert by_name["frequency"].bounds == (0.0, math.inf)
    assert by_name["frequency_dot"].topology is Topology.REAL
    assert by_name["coalescence_time"].topology is Topology.REAL
    assert by_name["declination"].bounds == (-math.pi / 2, math.pi / 2)
    assert by_name["inclination"].bounds == (0.0, math.pi)
    assert by_name["spin1"].bounds == (-1.0, 1.0)
    assert by_name["eccentricity_tref"].closed == (True, False)
    for name in ("right_ascension", "polarisation_icrs", "initial_phase"):
        assert by_name[name].topology is Topology.CIRCLE
        assert by_name[name].bounds == (0.0, 2 * math.pi)
    assert by_name["polar_angle_spin1_icrs"].bounds == (0.0, math.pi)


def test_defaults_follow_the_conventions_file() -> None:
    """Required parameters have no default; the MBHB reference frequency has 0."""
    defaults = {spec.name: spec.default for spec in GALACTIC_BINARY_SPECS}
    assert defaults["frequency"] is None
    assert defaults["frequency_dot"] == 0.0
    assert defaults["initial_phase"] == 0.0
    mbhb = {spec.name: spec.default for spec in MBHB_SPECS}
    assert mbhb["frequency_at_reference_time"] == 0.0
    assert mbhb["spin1"] == 0.0
    assert mbhb["phase_at_reference_time"] == 0.0
    emri = {spec.name: spec.default for spec in EMRI_SPECS}
    assert emri["polar_angle_spin1_icrs"] == 0.0
    assert emri["chi_1"] is None
    assert emri["inclination_tref"] == 0.0
    sobbh = {spec.name: spec.default for spec in SOBBH_SPECS}
    assert sobbh["spin_1"] is None
    assert sobbh["f_ref"] is None
    assert sobbh["eccentricity_tref"] == 0.0
    assert sobbh["phase_tref"] == 0.0


def test_the_mbhb_reference_frequency_is_closed_at_zero_and_open_above() -> None:
    """The default 0 of the conventions file lies in the domain [0, ∞) in Hz."""
    spec = {spec.name: spec for spec in MBHB_SPECS}["frequency_at_reference_time"]
    assert spec.default == 0.0
    assert spec.bounds == (0.0, math.inf)
    assert spec.closed == (True, False)
    assert spec.unit == "Hz"
    assert spec.topology is Topology.INTERVAL
    sobbh = {spec.name: spec for spec in SOBBH_SPECS}["f_ref"]
    assert sobbh.closed == (False, False)


def test_the_spheres_name_the_sky_and_the_spin_directions() -> None:
    """The sky is a latitude sphere and every spin direction a colatitude sphere."""
    assert [sphere.polar_form for sphere in GALACTIC_BINARY_SPHERES] == [
        PolarForm.LATITUDE
    ]
    assert [sphere.polar_form for sphere in EMRI_SPHERES] == [
        PolarForm.LATITUDE,
        PolarForm.COLATITUDE,
    ]
    assert [sphere.polar_form for sphere in SOBBH_SPHERES] == [
        PolarForm.LATITUDE,
        PolarForm.COLATITUDE,
        PolarForm.COLATITUDE,
    ]
    assert len(MBHB_SPHERES) == 1


def test_the_aliases_of_the_catpylisa_tables_are_present() -> None:
    """A sample of aliases of each type, with the spelling of the conventions file."""
    assert GALACTIC_BINARY_ALIASES["fdot"] == "frequency_dot"
    assert GALACTIC_BINARY_ALIASES["polarization"] == "polarisation_icrs"
    assert MBHB_ALIASES["tc"] == "coalescence_time"
    assert MBHB_ALIASES["luministy_distance"] == "luminosity_distance"
    assert EMRI_ALIASES["theta_s"] == "polar_angle_spin1_icrs"
    assert EMRI_ALIASES["polar_angle_spin1"] == "polar_angle_spin1_icrs"
    assert EMRI_ALIASES["inc"] == "inclination_tref"
    assert EMRI_ALIASES["inclination"] == "inclination_tref"
    assert SOBBH_ALIASES["frequency_at_reference_time"] == "f_ref"
    assert SOBBH_ALIASES["polar_angle_spin2"] == "polar_angle_spin2_icrs"


def test_a_parameter_spec_is_frozen() -> None:
    """Specs are immutable records."""
    spec = GALACTIC_BINARY_SPECS[0]
    assert isinstance(spec, ParameterSpec)
    with pytest.raises(ValidationError):
        spec.name = "other"  # type: ignore[misc]
