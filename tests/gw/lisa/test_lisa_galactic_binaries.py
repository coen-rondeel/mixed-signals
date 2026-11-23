"""Tests for the holder of Galactic binaries: derived quantities and transforms."""

import math
from collections.abc import Callable
from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest

from mixed_signals.errors import ConventionError
from mixed_signals.gw.lisa import GalacticBinaries, SkyFrame
from mixed_signals.gw.lisa._dependencies import lisa_constants

type Builder = Callable[..., dict[str, np.ndarray]]

COUNTS = {"alpha": 24, "beta": 18, "gamma": 21}
CONSTANTS = lisa_constants()
TWO_PI = 2 * math.pi


def binaries(posteriors: Builder, **keywords: Any) -> GalacticBinaries:
    """Build Galactic binaries with unequal numbers of samples."""
    data = posteriors("galactic_binary", COUNTS)
    return GalacticBinaries.from_posteriors(data, name="fit", epoch=0.0, **keywords)


def circle_distance(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Return the distance on a circle of period 2π."""
    difference = np.abs(first - second) % TWO_PI
    distance: np.ndarray = np.minimum(difference, TWO_PI - difference)
    return distance


def column(sources: GalacticBinaries, name: str) -> np.ndarray:
    """Return one parameter of the full vector, shaped (B, N)."""
    return np.asarray(sources._column(name))


# * Derived quantities


def test_the_chirp_mass_and_distance_follow_independent_formulas(
    lisa_posteriors: Builder,
) -> None:
    """𝓜 and D from NumPy formulas with the lisaconstants values, over all samples."""
    sources = binaries(lisa_posteriors)
    frequency = column(sources, "frequency")
    derivative = column(sources, "frequency_dot")
    amplitude = column(sources, "amplitude")
    light, newton = CONSTANTS.speed_of_light, CONSTANTS.gravitational_constant
    prefactor = (5.0 / (96.0 * np.pi ** (8.0 / 3.0))) ** 0.6
    mass = (
        (light**3 / newton) * prefactor * derivative**0.6 * frequency ** (-11.0 / 5.0)
    )
    np.testing.assert_allclose(
        sources.chirp_mass, mass / CONSTANTS.solar_mass, rtol=1e-12
    )
    distance = 5 * light / (24 * np.pi**2 * amplitude) * derivative / frequency**3
    np.testing.assert_allclose(
        sources.distance, distance / (1_000 * CONSTANTS.parsec), rtol=1e-12
    )
    assert 0.01 < float(sources.chirp_mass.mean()) < 10.0


def test_the_periods_and_the_orbital_frequency(lisa_posteriors: Builder) -> None:
    """The GW period is 1/f, the orbital period 2/f and the orbital frequency f/2."""
    sources = binaries(lisa_posteriors)
    frequency = column(sources, "frequency")
    np.testing.assert_allclose(sources.gw_period, 1.0 / frequency, rtol=1e-14)
    np.testing.assert_allclose(sources.orbital_period, 2.0 / frequency, rtol=1e-14)
    np.testing.assert_allclose(sources.orbital_frequency, frequency / 2.0, rtol=1e-14)
    assert sources.chirp_mass.shape == (3, 24)


def test_a_non_positive_derivative_gives_a_nan_chirp_mass(
    lisa_posteriors: Builder,
) -> None:
    """Outspiralling binaries, with ḟ ≤ 0, have no chirp mass and no distance."""
    data = lisa_posteriors("galactic_binary", COUNTS)
    data["alpha"]["frequency_dot"][:3] = -1e-17
    sources = GalacticBinaries.from_posteriors(data, name="fit", epoch=0.0)
    assert bool(jnp.isnan(sources.chirp_mass[0, :3]).all())
    assert bool(jnp.isnan(sources.distance[0, :3]).all())
    assert not bool(jnp.isnan(sources.chirp_mass[0, 3:]).any())


# * The chirp mass and the distance


def test_the_chirp_mass_form_replaces_the_derivative_and_the_amplitude(
    lisa_posteriors: Builder,
) -> None:
    """The holder gets 𝓜 in Msun and D in kpc, moves log_prior, and converts back."""
    sources = binaries(lisa_posteriors)
    chirp = sources.to_chirp_mass_and_distance()
    assert chirp.parameter_names[1:3] == ("chirp_mass", "distance")
    assert chirp.space["chirp_mass"].unit == "Msun"
    assert chirp.space["distance"].unit == "kpc"
    np.testing.assert_allclose(
        column(chirp, "chirp_mass"), sources.chirp_mass, rtol=1e-12
    )
    np.testing.assert_allclose(column(chirp, "distance"), sources.distance, rtol=1e-12)
    np.testing.assert_allclose(chirp.chirp_mass, sources.chirp_mass, rtol=1e-12)
    np.testing.assert_allclose(chirp.distance, sources.distance, rtol=1e-12)
    jacobian = 0.6 * column(chirp, "chirp_mass") / column(sources, "frequency_dot")
    jacobian *= column(chirp, "distance") / column(sources, "amplitude")
    np.testing.assert_allclose(
        np.asarray(chirp.values.log_prior)[:, 0],
        np.asarray(sources.values.log_prior)[:, 0] - np.log(jacobian),
        atol=1e-9,
    )
    assert chirp.to_chirp_mass_and_distance() is chirp
    back = chirp.to_frequency_dot_and_amplitude()
    np.testing.assert_allclose(back.values.samples, sources.values.samples, rtol=1e-11)
    assert sources.to_frequency_dot_and_amplitude() is sources
    assert back.parameter_names == sources.parameter_names
    assert back.space == sources.space


def test_the_chirp_mass_form_refuses_a_non_positive_derivative(
    lisa_posteriors: Builder,
) -> None:
    """Positive-weight samples with ḟ ≤ 0 raise with their count; padding does not."""
    data = lisa_posteriors("galactic_binary", COUNTS)
    data["alpha"]["frequency_dot"][:3] = 0.0
    data["beta"]["frequency_dot"][5] = -1e-17
    sources = GalacticBinaries.from_posteriors(data, name="fit", epoch=0.0)
    with pytest.raises(
        ValueError, match=r"frequency_dot must be positive.*4 of the 63"
    ):
        sources.to_chirp_mass_and_distance()
    log_weights = np.zeros((2, 6))
    log_weights[0, 4:] = -np.inf
    arrays = {
        "frequency": np.full((2, 6), 3e-3) + np.arange(12).reshape(2, 6) * 1e-9,
        "frequency_dot": np.full((2, 6), 1e-17),
        "amplitude": np.full((2, 6), 1e-22) * (1 + np.arange(6)),
        "right_ascension": np.full((2, 6), 1.0),
        "declination": np.full((2, 6), 0.3),
        "polarisation": np.full((2, 6), 0.5),
        "inclination": np.full((2, 6), 1.0),
    }
    arrays["frequency_dot"][0, 5] = -2e-17
    only_padding = GalacticBinaries.from_arrays(
        arrays, source_ids=["a", "b"], name="fit", epoch=0.0, log_weights=log_weights
    )
    converted = only_padding.to_chirp_mass_and_distance()
    assert bool(jnp.all(jnp.isfinite(converted.values.samples)))
    np.testing.assert_array_equal(
        np.asarray(converted.values.samples[0, 0, 5, 1]),
        np.asarray(converted.values.samples[0, 0, 0, 1]),
    )


# * The phase combinations


def test_phase_combinations_replace_the_phase_and_the_polarisation(
    lisa_posteriors: Builder,
) -> None:
    """θ± = φ₀ ± 2ψ, named after the frame, with the representative on return."""
    sources = binaries(lisa_posteriors)
    combined = sources.to_phase_combinations()
    names = combined.parameter_names
    assert names[5] == "theta_plus_icrs"
    assert names[7] == "theta_minus_icrs"
    assert combined.space["theta_plus_icrs"].topology.value == "circle"
    assert combined.space["theta_plus_icrs"].label == r"\theta_+"
    phase = column(sources, "initial_phase")
    polarisation = column(sources, "polarisation_icrs")
    assert (
        float(
            circle_distance(
                column(combined, "theta_plus_icrs"), phase + 2 * polarisation
            ).max()
        )
        < 1e-12
    )
    assert (
        float(
            circle_distance(
                column(combined, "theta_minus_icrs"), phase - 2 * polarisation
            ).max()
        )
        < 1e-12
    )
    assert combined.to_phase_combinations() is combined
    back = combined.to_initial_phase_and_polarisation()
    assert back.parameter_names == sources.parameter_names
    assert sources.to_initial_phase_and_polarisation() is sources
    again = back.to_phase_combinations()
    for name in ("theta_plus_icrs", "theta_minus_icrs"):
        assert (
            float(circle_distance(column(again, name), column(combined, name)).max())
            < 1e-12
        )
    np.testing.assert_array_equal(
        np.asarray(combined.values.log_prior), np.asarray(sources.values.log_prior)
    )


def test_phase_combinations_follow_the_sky_frame(lisa_posteriors: Builder) -> None:
    """The combinations take the frame suffix, and the sky rotation shifts both."""
    sources = binaries(lisa_posteriors)
    ecliptic = sources.to_phase_combinations().to_ecliptic()
    assert "theta_plus_ecliptic" in ecliptic.parameter_names
    via_polarisation = sources.to_ecliptic().to_phase_combinations()
    for name in ("theta_plus_ecliptic", "theta_minus_ecliptic"):
        assert (
            float(
                circle_distance(
                    column(ecliptic, name), column(via_polarisation, name)
                ).max()
            )
            < 1e-11
        )
    restored = ecliptic.to_icrs().to_initial_phase_and_polarisation()
    assert restored.parameter_names == sources.parameter_names
    for name in ("polarisation_icrs", "initial_phase"):
        assert (
            float(circle_distance(column(restored, name), column(sources, name)).max())
            < 1e-10
        )


# * The sky frame


def test_the_sky_frame_conversion_names_the_ecliptic_parameters(
    lisa_posteriors: Builder,
) -> None:
    """The ecliptic holder has longitude and latitude; the round trip restores ICRS."""
    sources = binaries(lisa_posteriors)
    ecliptic = sources.to_ecliptic()
    assert ecliptic.conventions.sky_frame is SkyFrame.ECLIPTIC
    assert ecliptic.conventions.epoch == 0.0
    assert ecliptic.space["ecliptic_longitude"].label == r"\lambda"
    assert ecliptic.space["ecliptic_latitude"].label == r"\beta"
    assert ecliptic.space.spheres[0].azimuth == "ecliptic_longitude"
    assert ecliptic.to_ecliptic() is ecliptic
    assert sources.to_icrs() is sources
    back = ecliptic.to_icrs()
    assert back.conventions == sources.conventions
    assert back.space == sources.space
    for name in ("right_ascension", "polarisation_icrs"):
        assert (
            float(circle_distance(column(back, name), column(sources, name)).max())
            < 1e-11
        )
    np.testing.assert_allclose(
        column(back, "declination"), column(sources, "declination"), rtol=1e-11
    )
    assert not np.allclose(
        column(ecliptic, "ecliptic_latitude"), column(sources, "declination")
    )
    log_det = np.log(np.cos(column(sources, "declination"))) - np.log(
        np.cos(column(ecliptic, "ecliptic_latitude"))
    )
    np.testing.assert_allclose(
        np.asarray(ecliptic.values.log_prior)[:, 0],
        np.asarray(sources.values.log_prior)[:, 0] - log_det,
        atol=1e-10,
    )


# * The epoch


def test_the_epoch_shift_moves_the_frequency_and_the_phase(
    lisa_posteriors: Builder,
) -> None:
    """The frequency moves to f + ḟΔ and the phase to φ₀ − 2π(fΔ + ½ḟΔ²).

    The epoch of the record is updated.
    """
    sources = binaries(lisa_posteriors)
    delta = 3e5
    shifted = sources.at_epoch(delta)
    assert shifted.conventions.epoch == delta
    frequency, derivative = (
        column(sources, "frequency"),
        column(sources, "frequency_dot"),
    )
    np.testing.assert_allclose(
        column(shifted, "frequency"), frequency + derivative * delta, rtol=1e-14
    )
    expected = (
        column(sources, "initial_phase")
        - TWO_PI * (frequency * delta + 0.5 * derivative * delta**2)
    ) % TWO_PI
    assert (
        float(circle_distance(column(shifted, "initial_phase"), expected).max()) < 1e-8
    )
    np.testing.assert_array_equal(
        np.asarray(shifted.values.log_prior), np.asarray(sources.values.log_prior)
    )
    twice = sources.at_epoch(1e5).at_epoch(delta)
    np.testing.assert_allclose(
        column(twice, "frequency"), column(shifted, "frequency"), rtol=1e-13
    )
    assert (
        float(
            circle_distance(
                column(twice, "initial_phase"), column(shifted, "initial_phase")
            ).max()
        )
        < 1e-8
    )
    assert twice.conventions.epoch == delta
    back = shifted.at_epoch(0.0)
    assert (
        float(
            circle_distance(
                column(back, "initial_phase"), column(sources, "initial_phase")
            ).max()
        )
        < 1e-8
    )
    same = sources.at_epoch(0.0)
    np.testing.assert_array_equal(
        np.asarray(same.values.samples), np.asarray(sources.values.samples)
    )


def test_a_constant_phase_varies_after_an_epoch_shift(lisa_posteriors: Builder) -> None:
    """Settings are recomputed: a constant φ₀ with a varying f becomes sampled."""
    data = lisa_posteriors("galactic_binary", COUNTS, constant=("initial_phase",))
    sources = GalacticBinaries.from_posteriors(data, name="fit", epoch=0.0)
    assert "initial_phase" not in sources.space.names
    shifted = sources.at_epoch(2e6)
    assert "initial_phase" in shifted.space.names
    assert "initial_phase" not in shifted.settings.columns


def test_the_epoch_shift_needs_the_derivative_and_refuses_bounds(
    lisa_posteriors: Builder,
) -> None:
    """The chirp-mass form must convert back first, and a bound on f does not map."""
    chirp = binaries(lisa_posteriors).to_chirp_mass_and_distance()
    with pytest.raises(ConventionError, match="to_frequency_dot_and_amplitude"):
        chirp.at_epoch(1.0)
    bounded = binaries(lisa_posteriors, bounds={"frequency": (2e-3, 4e-3)})
    with pytest.raises(ValueError, match="carry prior bounds"):
        bounded.at_epoch(1.0)
    assert bounded.to_phase_combinations().conventions == bounded.conventions


def test_the_epoch_shift_works_on_phase_combinations(lisa_posteriors: Builder) -> None:
    """Both combinations shift by the same phase as φ₀ does."""
    sources = binaries(lisa_posteriors)
    via_phase = sources.at_epoch(2e5).to_phase_combinations()
    via_combination = sources.to_phase_combinations().at_epoch(2e5)
    for name in ("theta_plus_icrs", "theta_minus_icrs"):
        assert (
            float(
                circle_distance(
                    column(via_phase, name), column(via_combination, name)
                ).max()
            )
            < 1e-8
        )
