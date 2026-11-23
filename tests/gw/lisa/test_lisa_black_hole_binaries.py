"""Tests for the holders of binary black holes: masses, redshift, epochs and spins."""

import math
from collections.abc import Callable
from typing import Any

import astropy.units as astropy_units
import numpy as np
import pytest
from astropy.cosmology import Planck15, z_at_value

from mixed_signals.errors import ConventionError
from mixed_signals.gw.lisa import (
    BlackHoleBinarySources,
    ExtremeMassRatioInspirals,
    MassiveBlackHoleBinaries,
    SkyFrame,
    StellarOriginBinaryBlackHoles,
)

type Builder = Callable[..., dict[str, np.ndarray]]

TYPES: dict[str, tuple[type[BlackHoleBinarySources], str]] = {
    "massive_black_hole_binary": (MassiveBlackHoleBinaries, "J"),
    "extreme_mass_ratio_inspiral": (ExtremeMassRatioInspirals, "FEW"),
    "stellar_origin_binary_black_hole": (StellarOriginBinaryBlackHoles, "J"),
}
COUNTS = {"alpha": 16, "beta": 12, "gamma": 14}
TWO_PI = 2 * math.pi


def binaries(posteriors: Builder, kind: str, **keywords: Any) -> BlackHoleBinarySources:
    """Build binary black holes of a type with unequal numbers of samples."""
    cls, source_frame = TYPES[kind]
    data = posteriors(kind, COUNTS)
    return cls.from_posteriors(
        data, name="fit", epoch=0.0, source_frame=source_frame, **keywords
    )


def column(sources: BlackHoleBinarySources, name: str) -> np.ndarray:
    """Return one parameter of the full vector, shaped (B, N)."""
    return np.asarray(sources._column(name))


# * Mass quantities


@pytest.mark.parametrize("kind", TYPES)
def test_mass_quantities_follow_independent_formulas(
    lisa_posteriors: Builder, kind: str
) -> None:
    """M, q, Q, μ, η and 𝓜 from NumPy formulas of SGS Section 2.3."""
    sources = binaries(lisa_posteriors, kind)
    primary = column(sources, "primary_mass_det_frame")
    secondary = column(sources, "secondary_mass_det_frame")
    total = primary + secondary
    np.testing.assert_allclose(sources.total_mass, total, rtol=1e-14)
    np.testing.assert_allclose(sources.mass_ratio, secondary / primary, rtol=1e-14)
    np.testing.assert_allclose(
        sources.inverted_mass_ratio, primary / secondary, rtol=1e-14
    )
    np.testing.assert_allclose(
        sources.reduced_mass, primary * secondary / total, rtol=1e-14
    )
    np.testing.assert_allclose(
        sources.symmetric_mass_ratio, primary * secondary / total**2, rtol=1e-14
    )
    np.testing.assert_allclose(
        sources.chirp_mass, (primary * secondary) ** 0.6 / total**0.2, rtol=1e-13
    )
    assert (np.asarray(sources.mass_ratio) <= 1.0).all()
    np.testing.assert_allclose(
        np.asarray(sources.symmetric_mass_ratio),
        np.asarray(sources.reduced_mass / sources.total_mass),
        rtol=1e-13,
    )
    assert sources.total_mass.shape == (3, 16)


@pytest.mark.parametrize("kind", TYPES)
def test_redshift_and_source_frame_masses(lisa_posteriors: Builder, kind: str) -> None:
    """The redshift inverts D_L in Planck15, and source-frame masses divide by 1 + z."""
    sources = binaries(lisa_posteriors, kind)
    distance = column(sources, "luminosity_distance")
    redshift = np.asarray(sources.redshift)
    back = Planck15.luminosity_distance(redshift).to_value("Gpc")
    np.testing.assert_allclose(back, distance, rtol=1e-10)
    scale = 1 + redshift
    np.testing.assert_allclose(
        sources.source_frame_primary_mass,
        column(sources, "primary_mass_det_frame") / scale,
        rtol=1e-13,
    )
    np.testing.assert_allclose(
        sources.source_frame_secondary_mass,
        column(sources, "secondary_mass_det_frame") / scale,
        rtol=1e-13,
    )
    np.testing.assert_allclose(
        sources.source_frame_total_mass,
        np.asarray(sources.total_mass) / scale,
        rtol=1e-13,
    )
    np.testing.assert_allclose(
        sources.source_frame_chirp_mass,
        np.asarray(sources.chirp_mass) / scale,
        rtol=1e-13,
    )


def test_redshift_agrees_with_the_root_finder_of_astropy(
    lisa_posteriors: Builder,
) -> None:
    """From 10⁻³ to 20 Gpc, z agrees with z_at_value within a relative 10⁻⁸."""
    distances = np.geomspace(1e-3, 20.0, 40)
    data = lisa_posteriors("massive_black_hole_binary", {"a": 40})
    table = data["a"]
    table["luminosity_distance"] = distances
    sources = MassiveBlackHoleBinaries.from_posteriors(
        data, name="fit", epoch=0.0, source_frame="J"
    )
    redshift = np.asarray(sources.redshift)[0]
    reference = np.array(
        [
            float(
                z_at_value(
                    Planck15.luminosity_distance,
                    value * astropy_units.Gpc,
                    zmin=1e-5,
                    zmax=50.0,
                    ztol=1e-12,
                )
            )
            for value in distances
        ]
    )
    np.testing.assert_allclose(redshift, reference, rtol=1e-8)


# * The mass parametrisations


@pytest.mark.parametrize("kind", TYPES)
def test_mass_parametrisations_convert_and_round_trip(
    lisa_posteriors: Builder, kind: str
) -> None:
    """The total and chirp forms rename the masses, move log_prior and convert back."""
    sources = binaries(lisa_posteriors, kind)
    primary = column(sources, "primary_mass_det_frame")
    secondary = column(sources, "secondary_mass_det_frame")
    total = sources.to_total_mass_and_mass_ratio()
    assert total.parameter_names[:2] == ("total_mass_det_frame", "mass_ratio")
    np.testing.assert_allclose(
        column(total, "total_mass_det_frame"), primary + secondary, rtol=1e-14
    )
    np.testing.assert_allclose(
        column(total, "mass_ratio"), secondary / primary, rtol=1e-14
    )
    np.testing.assert_allclose(
        np.asarray(total.values.log_prior)[:, 0],
        np.asarray(sources.values.log_prior)[:, 0]
        - np.log((primary + secondary) / primary**2),
        atol=1e-10,
    )
    chirp = sources.to_chirp_mass_and_mass_ratio()
    assert chirp.parameter_names[:2] == ("chirp_mass_det_frame", "mass_ratio")
    np.testing.assert_allclose(
        column(chirp, "chirp_mass_det_frame"), sources.chirp_mass, rtol=1e-13
    )
    np.testing.assert_allclose(
        np.asarray(chirp.values.log_prior)[:, 0],
        np.asarray(sources.values.log_prior)[:, 0]
        - np.log(np.asarray(sources.chirp_mass) / primary**2),
        atol=1e-10,
    )
    assert total.to_total_mass_and_mass_ratio() is total
    assert chirp.to_chirp_mass_and_mass_ratio() is chirp
    assert sources.to_component_masses() is sources
    for form in (total, chirp):
        back = form.to_component_masses()
        np.testing.assert_allclose(
            back.values.samples, sources.values.samples, rtol=1e-12
        )
        assert back.space == sources.space
        np.testing.assert_allclose(back.total_mass, sources.total_mass, rtol=1e-12)
        np.testing.assert_allclose(back.chirp_mass, sources.chirp_mass, rtol=1e-12)
        np.testing.assert_allclose(form.mass_ratio, sources.mass_ratio, rtol=1e-12)
        np.testing.assert_allclose(form.reduced_mass, sources.reduced_mass, rtol=1e-12)
    across = total.to_chirp_mass_and_mass_ratio()
    np.testing.assert_allclose(across.values.samples, chirp.values.samples, rtol=1e-12)
    np.testing.assert_allclose(
        np.asarray(across.values.log_prior),
        np.asarray(chirp.values.log_prior),
        atol=1e-9,
    )
    other_way = chirp.to_total_mass_and_mass_ratio()
    np.testing.assert_allclose(
        other_way.values.samples, total.values.samples, rtol=1e-12
    )


def test_the_mass_ratio_space_has_the_edge_at_one(lisa_posteriors: Builder) -> None:
    """The mass ratio lies in (0, 1], and the masses keep their detector-frame units."""
    sources = binaries(lisa_posteriors, "massive_black_hole_binary")
    chirp = sources.to_chirp_mass_and_mass_ratio()
    assert chirp.space["mass_ratio"].bounds == (0.0, 1.0)
    assert chirp.space["chirp_mass_det_frame"].unit == "Msun"
    with pytest.raises(ValueError, match="carry prior bounds"):
        binaries(
            lisa_posteriors,
            "massive_black_hole_binary",
            bounds={"primary_mass_det_frame": (1e5, 1e7)},
        ).to_total_mass_and_mass_ratio()


def test_the_properties_read_the_masses_in_every_form(lisa_posteriors: Builder) -> None:
    """The mass properties give the same values from the total and the chirp form."""
    sources = binaries(lisa_posteriors, "stellar_origin_binary_black_hole")
    for form in (
        sources.to_total_mass_and_mass_ratio(),
        sources.to_chirp_mass_and_mass_ratio(),
    ):
        np.testing.assert_allclose(
            form.symmetric_mass_ratio, sources.symmetric_mass_ratio, rtol=1e-12
        )
        np.testing.assert_allclose(
            form.inverted_mass_ratio, sources.inverted_mass_ratio, rtol=1e-12
        )
        np.testing.assert_allclose(
            form.source_frame_chirp_mass, sources.source_frame_chirp_mass, rtol=1e-11
        )


# * Epochs


def test_the_mbhb_epoch_shift_moves_the_coalescence_time(
    lisa_posteriors: Builder,
) -> None:
    """t_c → t_c − Δ, sampled or constant, with the epoch of the record updated."""
    sources = binaries(lisa_posteriors, "massive_black_hole_binary")
    shifted = sources.at_epoch(4e5)
    assert shifted.conventions.epoch == 4e5
    np.testing.assert_allclose(
        column(shifted, "coalescence_time"),
        column(sources, "coalescence_time") - 4e5,
        rtol=1e-14,
    )
    np.testing.assert_array_equal(
        np.asarray(shifted.values.log_prior), np.asarray(sources.values.log_prior)
    )
    twice = sources.at_epoch(1e5).at_epoch(4e5)
    np.testing.assert_allclose(
        column(twice, "coalescence_time"),
        column(shifted, "coalescence_time"),
        rtol=1e-14,
    )
    data = lisa_posteriors(
        "massive_black_hole_binary", COUNTS, constant=("coalescence_time",)
    )
    fixed = MassiveBlackHoleBinaries.from_posteriors(
        data, name="fit", epoch=0.0, source_frame="J"
    )
    assert "coalescence_time" not in fixed.space.names
    moved = fixed.at_epoch(-2e5)
    assert "coalescence_time" not in moved.space.names
    np.testing.assert_allclose(
        moved.settings["coalescence_time"].to_numpy(),
        fixed.settings["coalescence_time"].to_numpy() + 2e5,
    )
    bounded = binaries(
        lisa_posteriors,
        "massive_black_hole_binary",
        bounds={"coalescence_time": (0.0, 1e8)},
    )
    with pytest.raises(ValueError, match="carry prior bounds"):
        bounded.at_epoch(1.0)


@pytest.mark.parametrize(
    "kind", ["extreme_mass_ratio_inspiral", "stellar_origin_binary_black_hole"]
)
def test_emri_and_sobbh_return_themselves_at_their_own_epoch_only(
    lisa_posteriors: Builder, kind: str
) -> None:
    """Without the dynamics of the binary only the own epoch is accepted."""
    sources = binaries(lisa_posteriors, kind)
    assert sources.at_epoch(0.0) is sources
    with pytest.raises(ConventionError, match="cannot shift its epoch"):
        sources.at_epoch(1.0)
    with pytest.raises(ConventionError, match="cannot shift its epoch"):
        sources.at_epoch(-0.5)


# * Spin directions and the sky frame


def test_the_emri_spin_direction_rotates_with_the_sky(lisa_posteriors: Builder) -> None:
    """The spin angles carry the frame suffix, rotate, and rotate back."""
    sources = binaries(lisa_posteriors, "extreme_mass_ratio_inspiral")
    ecliptic = sources.to_ecliptic()
    assert ecliptic.conventions.sky_frame is SkyFrame.ECLIPTIC
    assert "polar_angle_spin1_ecliptic" in ecliptic.parameter_names
    assert "azimuthal_angle_spin1_ecliptic" in ecliptic.parameter_names
    assert ecliptic.space.spheres[1].polar_form.value == "colatitude"
    theta, phi = (
        column(sources, "polar_angle_spin1_icrs"),
        column(sources, "azimuthal_angle_spin1_icrs"),
    )
    obliquity = 0.409_092_600_600_582_9
    vector = np.stack(
        [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)]
    )
    rotated = np.stack(
        [
            vector[0],
            math.cos(obliquity) * vector[1] + math.sin(obliquity) * vector[2],
            -math.sin(obliquity) * vector[1] + math.cos(obliquity) * vector[2],
        ]
    )
    np.testing.assert_allclose(
        column(ecliptic, "polar_angle_spin1_ecliptic"),
        np.arccos(rotated[2]),
        atol=1e-12,
    )
    back = ecliptic.to_icrs()
    np.testing.assert_allclose(
        column(back, "polar_angle_spin1_icrs"), theta, atol=1e-11
    )
    assert back.space == sources.space
    assert "polarisation_ecliptic" not in ecliptic.parameter_names


def test_the_sobbh_spin_directions_rotate_and_the_log_prior_follows(
    lisa_posteriors: Builder,
) -> None:
    """Both spin directions rotate, and ln|det| adds the area ratio of each sphere."""
    sources = binaries(lisa_posteriors, "stellar_origin_binary_black_hole")
    ecliptic = sources.to_ecliptic()
    expected = np.log(np.cos(column(sources, "declination"))) - np.log(
        np.cos(column(ecliptic, "ecliptic_latitude"))
    )
    for index in (1, 2):
        before = np.sin(column(sources, f"polar_angle_spin{index}_icrs"))
        after = np.sin(column(ecliptic, f"polar_angle_spin{index}_ecliptic"))
        expected = expected + np.log(before) - np.log(after)
    np.testing.assert_allclose(
        np.asarray(ecliptic.values.log_prior)[:, 0],
        np.asarray(sources.values.log_prior)[:, 0] - expected,
        atol=1e-9,
    )
    assert "polarisation_ecliptic" in ecliptic.parameter_names
    assert {sphere.azimuth for sphere in ecliptic.space.spheres} == {
        "ecliptic_longitude",
        "azimuthal_angle_spin1_ecliptic",
        "azimuthal_angle_spin2_ecliptic",
    }


def test_an_empty_mask_returns_the_holder_unchanged(
    lisa_posteriors: Builder,
) -> None:
    """With an all-false mask, the overwrite of zero-weight samples is skipped.

    It returns the holder itself.
    """
    sources = binaries(lisa_posteriors, "massive_black_hole_binary")
    assert (
        sources._overwrite_zero_weight_samples(np.zeros((3, 16), dtype=bool)) is sources
    )
