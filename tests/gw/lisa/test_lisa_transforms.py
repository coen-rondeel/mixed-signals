"""Tests for the transforms of the parameter vector of LISA sources."""

import math
from collections.abc import Callable

import astropy.units as astropy_units
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from astropy.coordinates import BarycentricMeanEcliptic, SkyCoord

from mixed_signals.core.samples import SampleSet
from mixed_signals.core.space import Parameter, ParamSpace
from mixed_signals.errors import ConventionError
from mixed_signals.gw.lisa import (
    ExtremeMassRatioInspirals,
    GalacticBinaries,
    LisaSources,
    MassiveBlackHoleBinaries,
    StellarOriginBinaryBlackHoles,
)
from mixed_signals.gw.lisa._dependencies import lisa_constants
from mixed_signals.gw.lisa.conventions import (
    GALACTIC_BINARY_SPECS,
    GALACTIC_BINARY_SPHERES,
    SkyFrame,
    frame_parameter,
)
from mixed_signals.gw.lisa.transforms import (
    ChirpMassAndDistanceTransform,
    CoalescenceTimeShift,
    GalacticBinaryEpochShift,
    MassPairTransform,
    PhaseCombinationTransform,
    SkyFrameTransform,
    amplitude_from_distance,
    chirp_mass_from_frequency_dot,
    distance_from_amplitude,
    frequency_dot_from_chirp_mass,
)

type Builder = Callable[..., dict[str, np.ndarray]]

TWO_PI = 2 * math.pi
POINTS = 40
CONSTANTS = lisa_constants()
# * Ranges that keep random points away from the poles and the edges of the domains.
RANGES = {
    "frequency": (1e-3, 1e-2),
    "frequency_dot": (1e-18, 1e-16),
    "amplitude": (1e-23, 1e-21),
    "right_ascension": (0.0, TWO_PI),
    "declination": (-1.2, 1.2),
    "polarisation_icrs": (0.0, TWO_PI),
    "inclination": (0.2, 2.9),
    "initial_phase": (0.0, TWO_PI),
    "primary_mass_det_frame": (1e5, 1e7),
    "secondary_mass_det_frame": (1e4, 1e5),
    "polar_angle_spin1_icrs": (0.3, 2.8),
    "polar_angle_spin2_icrs": (0.3, 2.8),
    "azimuthal_angle_spin1_icrs": (0.0, TWO_PI),
    "azimuthal_angle_spin2_icrs": (0.0, TWO_PI),
    "coalescence_time": (1e6, 1e7),
    "luminosity_distance": (0.1, 50.0),
    "frequency_at_reference_time": (1e-4, 1e-3),
    "semi_latus_rectum_tref": (8.0, 15.0),
    "f_ref": (1e-3, 1e-1),
}
SPEC_SHIFTS = (1e5, -3e4)


def full_space_of(posteriors: Builder, kind: str = "galactic_binary") -> ParamSpace:
    """Return the full parameter space of a holder of the given type."""
    classes: dict[str, tuple[type[LisaSources], str | None]] = {
        "galactic_binary": (GalacticBinaries, None),
        "massive_black_hole_binary": (MassiveBlackHoleBinaries, "J"),
        "extreme_mass_ratio_inspiral": (ExtremeMassRatioInspirals, "FEW"),
        "stellar_origin_binary_black_hole": (StellarOriginBinaryBlackHoles, "J"),
    }
    cls, source_frame = classes[kind]
    data = posteriors(kind, {"a": 6, "b": 6})
    holder = cls.from_posteriors(data, name="fit", epoch=0.0, source_frame=source_frame)
    return holder.full_space


def galactic_binaries(posteriors: Builder) -> GalacticBinaries:
    """Build a holder of Galactic binaries from synthetic posteriors."""
    data = posteriors("galactic_binary", {"a": 6, "b": 6})
    return GalacticBinaries.from_posteriors(data, name="fit", epoch=0.0)


def vector_space() -> ParamSpace:
    """Return the Galactic-binary space of the canonical parameters in ICRS."""
    return ParamSpace(
        parameters=tuple(
            frame_parameter(spec, SkyFrame.ICRS) for spec in GALACTIC_BINARY_SPECS
        ),
        spheres=GALACTIC_BINARY_SPHERES,
    )


def draw_points(space: ParamSpace, seed: int = 0, count: int = POINTS) -> jax.Array:
    """Draw points of the full vector, uniform inside the ranges of each parameter."""
    keys = jax.random.split(jax.random.key(seed), len(space))
    columns = []
    for key, parameter in zip(keys, space.parameters, strict=True):
        lower, upper = RANGES.get(parameter.name, parameter.bounds)
        columns.append(jax.random.uniform(key, (count,), minval=lower, maxval=upper))
    return jnp.stack(columns, axis=-1)


def circle_difference(first: jax.Array, second: jax.Array) -> jax.Array:
    """Return the distance on a circle of period 2π, elementwise."""
    difference = jnp.abs(first - second) % TWO_PI
    return jnp.minimum(difference, TWO_PI - difference)


def log_determinant_by_autodiff(transform: object, points: jax.Array) -> jax.Array:
    """Return ln|det| of the Jacobian from jax.jacfwd, at each point."""
    jacobian = jax.vmap(jax.jacfwd(transform.forward))(points)  # type: ignore[attr-defined]
    log_determinant: jax.Array = jnp.linalg.slogdet(jacobian)[1]
    return log_determinant


def unit_vectors(longitude: np.ndarray, latitude: np.ndarray) -> np.ndarray:
    """Return the unit vectors of sky positions, shaped (N, 3)."""
    return np.stack(
        [
            np.cos(latitude) * np.cos(longitude),
            np.cos(latitude) * np.sin(longitude),
            np.sin(latitude),
        ],
        axis=-1,
    )


def sky_vector(right_ascension: float, declination: float) -> np.ndarray:
    """Return the unit vector of a sky position."""
    return np.array(
        [
            math.cos(declination) * math.cos(right_ascension),
            math.cos(declination) * math.sin(right_ascension),
            math.sin(declination),
        ]
    )


def reference_polarisation(
    direction: np.ndarray, pole: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return the reference polarisation vectors of a source direction.

    With k = −r̂, the first is ẑ × k/|ẑ × k| and the second k × the first, ẑ
    being the pole of the frame.
    """
    propagation = -direction
    first = np.cross(pole, propagation)
    first /= np.linalg.norm(first)
    return first, np.cross(propagation, first)


# * The sky frame


@pytest.mark.parametrize(
    "kind",
    [
        "galactic_binary",
        "massive_black_hole_binary",
        "extreme_mass_ratio_inspiral",
        "stellar_origin_binary_black_hole",
    ],
)
def test_sky_round_trip_and_log_determinant_for_every_type(
    lisa_posteriors: Builder, kind: str
) -> None:
    """The rotation inverts to 1e-12, and its log-determinant matches jax.jacfwd."""
    space = full_space_of(lisa_posteriors, kind)
    points = draw_points(space)
    for to_ecliptic in (True, False):
        transform = SkyFrameTransform.from_space(
            space, to_ecliptic=to_ecliptic, obliquity=CONSTANTS.obliquity
        )
        mapped = transform.forward(points)
        back = transform.inverse(mapped)
        circles = [
            position
            for position, parameter in enumerate(space.parameters)
            if parameter.period is not None
        ]
        for position in range(len(space)):
            if position in circles:
                assert (
                    float(
                        jnp.max(
                            circle_difference(back[:, position], points[:, position])
                        )
                    )
                    < 1e-12
                )
            else:
                np.testing.assert_allclose(
                    back[:, position], points[:, position], rtol=1e-12
                )
        exact = log_determinant_by_autodiff(transform, points)
        np.testing.assert_allclose(
            transform.log_abs_det_jacobian(points), exact, atol=1e-10
        )


def test_the_two_directions_have_opposite_log_determinants(
    lisa_posteriors: Builder,
) -> None:
    """ln|det| of the forward map at x plus that of the reverse map at f(x) is zero."""
    space = full_space_of(lisa_posteriors, "stellar_origin_binary_black_hole")
    points = draw_points(space)
    forward = SkyFrameTransform.from_space(
        space, to_ecliptic=True, obliquity=CONSTANTS.obliquity
    )
    backward = forward._reversed()
    total = forward.log_abs_det_jacobian(points) + backward.log_abs_det_jacobian(
        forward.forward(points)
    )
    np.testing.assert_allclose(total, 0.0, atol=1e-12)
    assert forward.description == "icrs_to_ecliptic"
    assert backward.description == "ecliptic_to_icrs"


def test_the_sky_position_agrees_with_astropy(lisa_posteriors: Builder) -> None:
    """Eq. 51 matches the Barycentric Mean Ecliptic of J2000 within 5e-7 rad."""
    space = full_space_of(lisa_posteriors, "galactic_binary")
    points = draw_points(space, seed=3, count=2_000)
    transform = SkyFrameTransform.from_space(
        space, to_ecliptic=True, obliquity=CONSTANTS.obliquity
    )
    mapped = np.asarray(transform.forward(points))
    coordinates = SkyCoord(
        ra=np.asarray(points[:, 3]) * astropy_units.rad,
        dec=np.asarray(points[:, 4]) * astropy_units.rad,
        frame="icrs",
    ).transform_to(BarycentricMeanEcliptic(equinox="J2000"))
    reference = unit_vectors(
        coordinates.lon.to_value("rad"), coordinates.lat.to_value("rad")
    )
    ours = unit_vectors(mapped[:, 3], mapped[:, 4])
    # * The great-circle separation, since longitudes blow up near the ecliptic poles.
    separation = np.arctan2(
        np.linalg.norm(np.cross(ours, reference), axis=-1),
        np.sum(ours * reference, axis=-1),
    )
    assert np.max(separation) < 5e-7


def test_the_polarisation_shift_matches_the_rotation_of_the_reference_vectors(
    lisa_posteriors: Builder,
) -> None:
    """ψ in ICRS minus ψ in the ecliptic is the angle between (u, v) and (u0, v0)."""
    space = full_space_of(lisa_posteriors, "galactic_binary")
    points = draw_points(space, seed=5, count=30)
    transform = SkyFrameTransform.from_space(
        space, to_ecliptic=True, obliquity=CONSTANTS.obliquity
    )
    mapped = np.asarray(transform.forward(points))
    obliquity = CONSTANTS.obliquity
    rotation = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, math.cos(obliquity), math.sin(obliquity)],
            [0.0, -math.sin(obliquity), math.cos(obliquity)],
        ]
    )
    for row, image in zip(np.asarray(points), mapped, strict=True):
        direction = sky_vector(row[3], row[4])
        pole = np.array([0.0, 0.0, 1.0])
        equatorial, equatorial_cross = reference_polarisation(direction, pole)
        ecliptic, _ = reference_polarisation(rotation @ direction, pole)
        # * Rotating the ecliptic basis back puts both bases in the equatorial frame.
        ecliptic = rotation.T @ ecliptic
        shift = math.atan2(ecliptic @ equatorial_cross, ecliptic @ equatorial)
        expected = (image[5] + shift) % TWO_PI
        assert abs(((row[5] - expected) + math.pi) % TWO_PI - math.pi) < 1e-12


def test_spin_directions_rotate_with_the_sky(lisa_posteriors: Builder) -> None:
    """The unit vector of a spin direction takes the rotation of the sky position."""
    space = full_space_of(lisa_posteriors, "extreme_mass_ratio_inspiral")
    points = draw_points(space, seed=7, count=10)
    transform = SkyFrameTransform.from_space(
        space, to_ecliptic=True, obliquity=CONSTANTS.obliquity
    )
    mapped = np.asarray(transform.forward(points))
    names = space.names
    polar, azimuth = (
        names.index("polar_angle_spin1_icrs"),
        names.index("azimuthal_angle_spin1_icrs"),
    )
    obliquity = CONSTANTS.obliquity
    for row, image in zip(np.asarray(points), mapped, strict=True):
        theta, phi = row[polar], row[azimuth]
        vector = np.array(
            [
                math.sin(theta) * math.cos(phi),
                math.sin(theta) * math.sin(phi),
                math.cos(theta),
            ]
        )
        rotated = np.array(
            [
                vector[0],
                math.cos(obliquity) * vector[1] + math.sin(obliquity) * vector[2],
                -math.sin(obliquity) * vector[1] + math.cos(obliquity) * vector[2],
            ]
        )
        new_theta, new_phi = image[polar], image[azimuth]
        got = np.array(
            [
                math.sin(new_theta) * math.cos(new_phi),
                math.sin(new_theta) * math.sin(new_phi),
                math.cos(new_theta),
            ]
        )
        np.testing.assert_allclose(got, rotated, atol=1e-12)


def test_phase_combinations_shift_by_plus_and_minus_twice_the_polarisation_shift(
    lisa_posteriors: Builder,
) -> None:
    """θ₊ and θ₋ move by ∓2δψ, where ψ moves by ∓δψ, and the round trip holds."""
    holder = galactic_binaries(lisa_posteriors)
    combined = holder.to_phase_combinations()
    plain = SkyFrameTransform.from_space(
        holder.full_space, to_ecliptic=True, obliquity=CONSTANTS.obliquity
    )
    with_theta = SkyFrameTransform.from_space(
        combined.full_space, to_ecliptic=True, obliquity=CONSTANTS.obliquity
    )
    assert with_theta.polarisation_weights == (2.0, -2.0)
    assert plain.polarisation_weights == (1.0,)
    points = draw_points(combined.full_space, seed=11)
    mapped = with_theta.forward(points)
    plus, minus = (
        combined.full_space.index("theta_plus_icrs"),
        combined.full_space.index("theta_minus_icrs"),
    )
    shift_plus = (points[:, plus] - mapped[:, plus]) % TWO_PI
    shift_minus = (mapped[:, minus] - points[:, minus]) % TWO_PI
    np.testing.assert_allclose(shift_plus, shift_minus, atol=1e-12)
    back = with_theta.inverse(mapped)
    assert float(jnp.max(circle_difference(back[:, plus], points[:, plus]))) < 1e-12
    np.testing.assert_allclose(
        with_theta.log_abs_det_jacobian(points),
        log_determinant_by_autodiff(with_theta, points),
        atol=1e-10,
    )


def test_the_sky_map_renames_the_space_and_restores_it(
    lisa_posteriors: Builder,
) -> None:
    """The mapped space has the ecliptic names, and mapping back restores it."""
    space = full_space_of(lisa_posteriors, "stellar_origin_binary_black_hole")
    forward = SkyFrameTransform.from_space(
        space, to_ecliptic=True, obliquity=CONSTANTS.obliquity
    )
    ecliptic = forward.map_space(space)
    assert "ecliptic_longitude" in ecliptic.names
    assert "polarisation_ecliptic" in ecliptic.names
    assert "polar_angle_spin2_ecliptic" in ecliptic.names
    assert ecliptic["ecliptic_latitude"].label == r"\beta"
    assert ecliptic.spheres[1].polar == "polar_angle_spin1_ecliptic"
    restored = forward._reversed().map_space(ecliptic)
    assert restored == space
    assert set(forward.touched_names(space)) == {
        "right_ascension",
        "declination",
        "polarisation_icrs",
        "polar_angle_spin1_icrs",
        "azimuthal_angle_spin1_icrs",
        "polar_angle_spin2_icrs",
        "azimuthal_angle_spin2_icrs",
    }


def test_the_sky_map_moves_log_prior_by_the_log_determinant(
    lisa_posteriors: Builder,
) -> None:
    """map_sample_set subtracts ln|det| from log_prior and leaves the likelihood."""
    space = full_space_of(lisa_posteriors, "galactic_binary")
    points = draw_points(space, count=12)[None, None]
    sample_set = SampleSet(
        points, log_like=jnp.ones((1, 1, 12)), log_prior=jnp.full((1, 1, 12), -2.0)
    )
    transform = SkyFrameTransform.from_space(
        space, to_ecliptic=True, obliquity=CONSTANTS.obliquity
    )
    mapped = transform.map_sample_set(sample_set)
    expected = -2.0 - transform.log_abs_det_jacobian(points[0, 0])
    np.testing.assert_allclose(np.asarray(mapped.log_prior)[0, 0], expected, atol=1e-12)
    np.testing.assert_allclose(np.asarray(mapped.log_like)[0, 0], 1.0)


def test_the_sky_map_needs_a_sky_position() -> None:
    """A vector without the sky sphere raises a ConventionError."""
    bare = ParamSpace(parameters=(Parameter(name="frequency"),))
    with pytest.raises(ConventionError, match="no sky position"):
        SkyFrameTransform.from_space(bare, to_ecliptic=True, obliquity=0.4)


def test_sky_maps_jit_and_batch(lisa_posteriors: Builder) -> None:
    """The map runs under jax.jit and jax.vmap and agrees with the eager result."""
    space = full_space_of(lisa_posteriors, "galactic_binary")
    points = draw_points(space, count=8)
    transform = SkyFrameTransform.from_space(
        space, to_ecliptic=True, obliquity=CONSTANTS.obliquity
    )
    jitted = jax.jit(transform.forward)(points)
    np.testing.assert_allclose(jitted, transform.forward(points), atol=1e-14)
    single = jax.vmap(transform.forward)(points)
    np.testing.assert_allclose(single, transform.forward(points), atol=1e-14)


# * The epoch shift


def test_the_galactic_binary_epoch_shift_follows_its_closed_form(
    lisa_posteriors: Builder,
) -> None:
    """The frequency moves to f + ḟΔ and the phase to φ₀ − 2π(fΔ + ½ḟΔ²).

    The phase is wrapped, and the determinant is one.
    """
    space = full_space_of(lisa_posteriors, "galactic_binary")
    points = draw_points(space)
    delta = 2.5e5
    transform = GalacticBinaryEpochShift.from_space(space, delta=delta)
    mapped = np.asarray(transform.forward(points))
    frequency, derivative = np.asarray(points[:, 0]), np.asarray(points[:, 1])
    np.testing.assert_allclose(mapped[:, 0], frequency + derivative * delta, rtol=1e-14)
    np.testing.assert_array_equal(mapped[:, 1], derivative)
    expected = (
        np.asarray(points[:, 7])
        - TWO_PI * (frequency * delta + 0.5 * derivative * delta**2)
    ) % TWO_PI
    difference = np.abs(mapped[:, 7] - expected) % TWO_PI
    assert np.max(np.minimum(difference, TWO_PI - difference)) < 1e-9
    for position in (2, 3, 4, 5, 6):
        np.testing.assert_array_equal(
            mapped[:, position], np.asarray(points[:, position])
        )
    determinant = jnp.linalg.det(jax.vmap(jax.jacfwd(transform.forward))(points))
    np.testing.assert_allclose(determinant, 1.0, atol=1e-9)
    np.testing.assert_array_equal(transform.log_abs_det_jacobian(points), 0.0)
    assert transform.description == "epoch_shift(250000 s)"
    assert transform.map_space(space) is space
    assert transform.touched_names(space) == ("frequency", "initial_phase")


def test_epoch_shifts_invert_and_compose(lisa_posteriors: Builder) -> None:
    """Shifting by Δ₁ and then Δ₂ equals Δ₁ + Δ₂, and the inverse undoes a shift."""
    space = full_space_of(lisa_posteriors, "galactic_binary")
    points = draw_points(space)
    first = GalacticBinaryEpochShift.from_space(space, delta=SPEC_SHIFTS[0])
    second = GalacticBinaryEpochShift.from_space(space, delta=SPEC_SHIFTS[1])
    together = GalacticBinaryEpochShift.from_space(space, delta=sum(SPEC_SHIFTS))
    composed = second.forward(first.forward(points))
    direct = together.forward(points)
    np.testing.assert_allclose(composed[:, :7], direct[:, :7], rtol=1e-13)
    assert float(jnp.max(circle_difference(composed[:, 7], direct[:, 7]))) < 1e-9
    back = first.inverse(first.forward(points))
    np.testing.assert_allclose(back[:, :7], points[:, :7], rtol=1e-13)
    assert float(jnp.max(circle_difference(back[:, 7], points[:, 7]))) < 1e-9


def test_the_epoch_shift_moves_both_phase_combinations_equally(
    lisa_posteriors: Builder,
) -> None:
    """With θ± in place of φ₀, both shift by the same phase."""
    combined = galactic_binaries(lisa_posteriors).to_phase_combinations()
    transform = GalacticBinaryEpochShift.from_space(combined.full_space, delta=1e5)
    assert len(transform.phases) == 2
    points = draw_points(combined.full_space)
    mapped = transform.forward(points)
    plus, minus = transform.phases
    first = (mapped[:, plus] - points[:, plus]) % TWO_PI
    second = (mapped[:, minus] - points[:, minus]) % TWO_PI
    assert float(jnp.max(circle_difference(first, second))) < 1e-9


def test_the_epoch_shift_needs_the_frequency_derivative(
    lisa_posteriors: Builder,
) -> None:
    """The chirp-mass form cannot shift its epoch until it is converted back."""
    chirp = galactic_binaries(lisa_posteriors).to_chirp_mass_and_distance()
    with pytest.raises(ConventionError, match="to_frequency_dot_and_amplitude"):
        GalacticBinaryEpochShift.from_space(chirp.full_space, delta=1.0)


def test_the_coalescence_time_shift(lisa_posteriors: Builder) -> None:
    """t_c → t_c − Δ, with composition, an inverse and a unit Jacobian."""
    space = full_space_of(lisa_posteriors, "massive_black_hole_binary")
    points = draw_points(space)
    position = space.index("coalescence_time")
    transform = CoalescenceTimeShift.from_space(space, delta=4e5)
    mapped = transform.forward(points)
    np.testing.assert_allclose(mapped[:, position], points[:, position] - 4e5)
    others = [index for index in range(len(space)) if index != position]
    np.testing.assert_array_equal(mapped[:, others], points[:, others])
    np.testing.assert_allclose(transform.inverse(mapped), points, rtol=1e-14)
    twice = CoalescenceTimeShift.from_space(space, delta=-1e5).forward(mapped)
    np.testing.assert_allclose(twice[:, position], points[:, position] - 3e5)
    determinant = jnp.linalg.det(jax.vmap(jax.jacfwd(transform.forward))(points))
    np.testing.assert_allclose(determinant, 1.0)
    np.testing.assert_array_equal(transform.log_abs_det_jacobian(points), 0.0)
    assert transform.touched_names(space) == ("coalescence_time",)
    assert transform.map_space(space) is space
    assert transform.description == "epoch_shift(400000 s)"
    with pytest.raises(ConventionError, match="no 'coalescence_time'"):
        CoalescenceTimeShift.from_space(full_space_of(lisa_posteriors), delta=1.0)


# * The phase combinations


def strain(
    frequency: float,
    amplitude: float,
    inclination: float,
    phase: jax.Array,
    polarisation: jax.Array,
    times: jax.Array,
) -> jax.Array:
    """Return the SSB strain of SGS Eqs. 146, 147 and 69 with Φ = 2πft − φ₀."""
    waveform_phase = 2 * jnp.pi * frequency * times - phase
    plus = -amplitude * 0.5 * (1 + jnp.cos(inclination) ** 2) * jnp.cos(waveform_phase)
    cross = -amplitude * jnp.cos(inclination) * jnp.sin(waveform_phase)
    first = plus * jnp.cos(2 * polarisation) - cross * jnp.sin(2 * polarisation)
    second = plus * jnp.sin(2 * polarisation) + cross * jnp.cos(2 * polarisation)
    return jnp.stack([first, second])


def test_the_strain_is_identical_for_the_four_symmetric_images() -> None:
    """(φ₀, ψ) and its three images, and two preimages of one θ±, give one waveform."""
    times = jnp.linspace(0.0, 3e5, 50)
    frequency, amplitude, inclination = 3e-3, 2e-22, 0.9
    phase, polarisation = jnp.asarray(1.3), jnp.asarray(0.4)
    reference = strain(frequency, amplitude, inclination, phase, polarisation, times)
    images = [
        (phase, polarisation + math.pi),
        (phase + math.pi, polarisation + math.pi / 2),
        (phase + math.pi, polarisation + 3 * math.pi / 2),
    ]
    scale = float(jnp.max(jnp.abs(reference)))
    for image_phase, image_polarisation in images:
        image = strain(
            frequency, amplitude, inclination, image_phase, image_polarisation, times
        )
        assert float(jnp.max(jnp.abs(image - reference))) < 1e-12 * scale
    space = vector_space()
    transform = PhaseCombinationTransform.from_space(space, to_combinations=True)
    point = (
        jnp.zeros(len(space))
        .at[transform.phase]
        .set(phase)
        .at[transform.polarisation]
        .set(polarisation)
    )
    representative = transform.inverse(transform.forward(point))
    other = strain(
        frequency,
        amplitude,
        inclination,
        representative[transform.phase],
        representative[transform.polarisation],
        times,
    )
    assert float(jnp.max(jnp.abs(other - reference))) < 1e-12 * scale
    different = strain(
        frequency, amplitude, inclination, phase + 0.3, polarisation, times
    )
    assert float(jnp.max(jnp.abs(different - reference))) > 1e-3 * scale


def test_phase_combinations_follow_their_closed_form_and_symmetries() -> None:
    """θ± = φ₀ ± 2ψ wrapped; four preimages share an image."""
    space = vector_space()
    transform = PhaseCombinationTransform.from_space(space, to_combinations=True)
    points = draw_points(space)
    mapped = transform.forward(points)
    phase, polarisation = points[:, 7], points[:, 5]
    np.testing.assert_allclose(
        mapped[:, transform.plus], (phase + 2 * polarisation) % TWO_PI, atol=1e-12
    )
    np.testing.assert_allclose(
        mapped[:, transform.minus], (phase - 2 * polarisation) % TWO_PI, atol=1e-12
    )
    assert transform.plus == 5
    assert transform.minus == 7
    back = transform.inverse(mapped)
    assert float(jnp.max(back[:, 5])) < math.pi / 2
    assert float(jnp.min(back[:, 5])) >= 0.0
    again = transform.forward(back)
    for position in (5, 7):
        assert (
            float(jnp.max(circle_difference(again[:, position], mapped[:, position])))
            < 1e-12
        )
    for shift_phase, shift_polarisation in (
        (0.0, math.pi),
        (math.pi, math.pi / 2),
        (math.pi, 3 * math.pi / 2),
    ):
        image = points.at[:, 7].add(shift_phase).at[:, 5].add(shift_polarisation)
        mapped_image = transform.forward(image)
        for position in (5, 7):
            assert (
                float(
                    jnp.max(
                        circle_difference(
                            mapped_image[:, position], mapped[:, position]
                        )
                    )
                )
                < 1e-12
            )
    assert transform.description == "phase_combinations"


def test_the_phase_combination_map_keeps_the_log_density_and_the_local_matrix() -> None:
    """The log-determinant is zero; the local matrix has determinant −4."""
    space = vector_space()
    transform = PhaseCombinationTransform.from_space(space, to_combinations=True)
    points = draw_points(space, count=10)
    np.testing.assert_array_equal(transform.log_abs_det_jacobian(points), 0.0)
    jacobian = transform.jacobian(points)
    np.testing.assert_allclose(jnp.abs(jnp.linalg.det(jacobian)), 4.0, atol=1e-9)
    # * Rows (θ₊, θ₋) against columns (φ₀, ψ): the matrix [[1, 2], [1, −2]].
    block = jacobian[:, [transform.plus, transform.minus]][
        :, :, [transform.phase, transform.polarisation]
    ]
    np.testing.assert_allclose(jnp.linalg.det(block), -4.0, atol=1e-9)
    fisher = jnp.broadcast_to(jnp.eye(len(space)), (10, len(space), len(space)))
    mapped = transform.map_fisher(fisher, points)
    np.testing.assert_allclose(jnp.linalg.det(mapped), 1.0 / 16.0, rtol=1e-9)
    reverse = PhaseCombinationTransform.from_space(
        transform.map_space(space), to_combinations=False
    )
    np.testing.assert_array_equal(reverse.log_abs_det_jacobian(points), 0.0)
    assert reverse.description == "initial_phase_and_polarisation"
    assert reverse.touched_names(transform.map_space(space)) == (
        "theta_plus_icrs",
        "theta_minus_icrs",
    )


def test_phase_combination_spaces_and_errors() -> None:
    """The mapped space has two circles named after the frame."""
    space = vector_space()
    transform = PhaseCombinationTransform.from_space(space, to_combinations=True)
    combined = transform.map_space(space)
    assert combined.names[5] == "theta_plus_icrs"
    assert combined.names[7] == "theta_minus_icrs"
    assert combined["theta_plus_icrs"].label == r"\theta_+"
    assert combined["theta_minus_icrs"].label == r"\theta_-"
    assert combined["theta_plus_icrs"].period == TWO_PI
    reverse = PhaseCombinationTransform.from_space(combined, to_combinations=False)
    assert reverse.map_space(combined) == space
    ecliptic = SkyFrameTransform.from_space(
        space, to_ecliptic=True, obliquity=0.4
    ).map_space(space)
    in_ecliptic = PhaseCombinationTransform.from_space(
        ecliptic, to_combinations=True
    ).map_space(ecliptic)
    assert in_ecliptic.names[5] == "theta_plus_ecliptic"
    assert (
        PhaseCombinationTransform.from_space(
            in_ecliptic, to_combinations=False
        ).map_space(in_ecliptic)
        == ecliptic
    )
    with pytest.raises(ConventionError, match="initial_phase and a polarisation"):
        PhaseCombinationTransform.from_space(combined, to_combinations=True)
    with pytest.raises(ConventionError, match="no phase combinations"):
        PhaseCombinationTransform.from_space(space, to_combinations=False)


# * The mass pairs


@pytest.mark.parametrize("chirp", [False, True])
def test_mass_pairs_follow_their_closed_forms(
    lisa_posteriors: Builder, chirp: bool
) -> None:
    """(M, q) and (𝓜, q) from (m₁, m₂), the inverse and both Jacobians."""
    space = full_space_of(lisa_posteriors, "massive_black_hole_binary")
    points = draw_points(space)
    forward = MassPairTransform(primary=0, secondary=1, chirp=chirp, to_ratio=True)
    backward = MassPairTransform(primary=0, secondary=1, chirp=chirp, to_ratio=False)
    mapped_jax = forward.forward(points)
    mapped = np.asarray(mapped_jax)
    first, second = np.asarray(points[:, 0]), np.asarray(points[:, 1])
    total = first + second
    mass = (first * second) ** 0.6 / total**0.2 if chirp else total
    np.testing.assert_allclose(mapped[:, 0], mass, rtol=1e-13)
    np.testing.assert_allclose(mapped[:, 1], second / first, rtol=1e-13)
    assert (mapped[:, 1] <= 1).all()
    assert (mapped[:, 1] > 0).all()
    np.testing.assert_allclose(forward.inverse(mapped_jax), points, rtol=1e-12)
    np.testing.assert_allclose(backward.forward(mapped_jax), points, rtol=1e-12)
    np.testing.assert_allclose(backward.inverse(points), mapped, rtol=1e-12)
    expected = np.log(mass / first**2)
    np.testing.assert_allclose(
        forward.log_abs_det_jacobian(points), expected, rtol=1e-12
    )
    np.testing.assert_allclose(
        forward.log_abs_det_jacobian(points),
        log_determinant_by_autodiff(forward, points),
        atol=1e-10,
    )
    np.testing.assert_allclose(
        backward.log_abs_det_jacobian(mapped_jax),
        log_determinant_by_autodiff(backward, mapped_jax),
        atol=1e-10,
    )
    np.testing.assert_allclose(
        backward.log_abs_det_jacobian(mapped_jax), -expected, rtol=1e-12
    )


def test_mass_pair_spaces_and_descriptions(lisa_posteriors: Builder) -> None:
    """The mapped space names the mass and the ratio, and maps back to the masses."""
    space = full_space_of(lisa_posteriors, "massive_black_hole_binary")
    total = MassPairTransform(primary=0, secondary=1, chirp=False, to_ratio=True)
    chirp = MassPairTransform(primary=0, secondary=1, chirp=True, to_ratio=True)
    mapped = total.map_space(space)
    assert mapped.names[:2] == ("total_mass_det_frame", "mass_ratio")
    assert mapped["total_mass_det_frame"].unit == "Msun"
    assert mapped["mass_ratio"].bounds == (0.0, 1.0)
    assert mapped["mass_ratio"].unit is None
    assert chirp.map_space(space).names[0] == "chirp_mass_det_frame"
    assert (
        chirp.map_space(space)["chirp_mass_det_frame"].label
        == r"\mathcal{M}_{\mathrm{det}}"
    )
    back = MassPairTransform(primary=0, secondary=1, chirp=False, to_ratio=False)
    assert back.map_space(mapped) == space
    assert total.description == "total_mass_and_mass_ratio"
    assert chirp.description == "chirp_mass_and_mass_ratio"
    assert back.description == "component_masses"
    assert total.touched_names(space) == (
        "primary_mass_det_frame",
        "secondary_mass_det_frame",
    )


# * The chirp mass and the distance


def test_the_chirp_mass_and_distance_follow_the_leading_order_formulas(
    lisa_posteriors: Builder,
) -> None:
    """𝓜 and D from independent NumPy formulas with the lisaconstants values."""
    space = full_space_of(lisa_posteriors, "galactic_binary")
    points = draw_points(space)
    transform = ChirpMassAndDistanceTransform.from_space(
        space, to_chirp_mass=True, constants=CONSTANTS
    )
    mapped_jax = transform.forward(points)
    mapped = np.asarray(mapped_jax)
    frequency, derivative, amplitude = (
        np.asarray(points[:, index]) for index in (0, 1, 2)
    )
    light, newton = CONSTANTS.speed_of_light, CONSTANTS.gravitational_constant
    mass_kg = (light**3 / newton) * (
        5 * derivative / (96 * np.pi ** (8 / 3) * frequency ** (11 / 3))
    ) ** 0.6
    np.testing.assert_allclose(mapped[:, 1], mass_kg / CONSTANTS.solar_mass, rtol=1e-12)
    distance = 5 * light * derivative / (24 * np.pi**2 * frequency**3 * amplitude)
    np.testing.assert_allclose(
        mapped[:, 2], distance / (1_000 * CONSTANTS.parsec), rtol=1e-12
    )
    np.testing.assert_array_equal(mapped[:, 0], np.asarray(points[:, 0]))
    # * Feeding the chirp mass back into the amplitude formula recovers 𝒜.
    amplitude_check = (
        4
        * (newton * mass_kg) ** (5 / 3)
        * (np.pi * frequency) ** (2 / 3)
        / (light**4 * distance)
    )
    np.testing.assert_allclose(amplitude_check, amplitude, rtol=1e-12)
    np.testing.assert_allclose(transform.inverse(mapped_jax), points, rtol=1e-12)
    np.testing.assert_allclose(
        chirp_mass_from_frequency_dot(points[:, 0], points[:, 1], CONSTANTS),
        mapped[:, 1],
        rtol=1e-14,
    )
    np.testing.assert_allclose(
        distance_from_amplitude(points[:, 0], points[:, 1], points[:, 2], CONSTANTS),
        mapped[:, 2],
        rtol=1e-14,
    )
    np.testing.assert_allclose(
        frequency_dot_from_chirp_mass(points[:, 0], mapped_jax[:, 1], CONSTANTS),
        derivative,
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        amplitude_from_distance(
            points[:, 0], mapped_jax[:, 1], mapped_jax[:, 2], CONSTANTS
        ),
        amplitude,
        rtol=1e-12,
    )


def test_the_chirp_mass_map_has_the_expected_jacobian() -> None:
    """|det| = (3𝓜/(5ḟ))(D/𝒜) in both directions, equal to ln|det jax.jacfwd|."""
    space = vector_space()
    points = draw_points(space)
    forward = ChirpMassAndDistanceTransform.from_space(
        space, to_chirp_mass=True, constants=CONSTANTS
    )
    mapped = forward.forward(points)
    expected = jnp.log(0.6 * mapped[:, 1] / points[:, 1] * mapped[:, 2] / points[:, 2])
    np.testing.assert_allclose(
        forward.log_abs_det_jacobian(points), expected, rtol=1e-11
    )
    exact = log_determinant_by_autodiff(forward, points)
    np.testing.assert_allclose(forward.log_abs_det_jacobian(points), exact, atol=1e-8)
    chirp_space = forward.map_space(space)
    backward = ChirpMassAndDistanceTransform.from_space(
        chirp_space, to_chirp_mass=False, constants=CONSTANTS
    )
    np.testing.assert_allclose(
        backward.log_abs_det_jacobian(mapped), -expected, rtol=1e-11
    )
    np.testing.assert_allclose(
        backward.log_abs_det_jacobian(mapped),
        log_determinant_by_autodiff(backward, mapped),
        atol=1e-8,
    )
    np.testing.assert_allclose(backward.forward(mapped), points, rtol=1e-12)


def test_a_non_positive_frequency_derivative_has_no_chirp_mass() -> None:
    """The map gives NaN for ḟ ≤ 0, inside jit too."""
    space = vector_space()
    points = draw_points(space, count=4).at[1, 1].set(-1e-17).at[2, 1].set(0.0)
    forward = ChirpMassAndDistanceTransform.from_space(
        space, to_chirp_mass=True, constants=CONSTANTS
    )
    mapped = jax.jit(forward.forward)(points)
    assert bool(jnp.isnan(mapped[1, 1]))
    assert bool(jnp.isnan(mapped[2, 2]))
    assert not bool(jnp.isnan(mapped[0, 1]))
    assert bool(jnp.isnan(forward.log_abs_det_jacobian(points)[1]))
    assert bool(
        jnp.isnan(chirp_mass_from_frequency_dot(points[1, 0], points[1, 1], CONSTANTS))
    )


def test_chirp_mass_spaces_and_errors() -> None:
    """The mapped space has the chirp mass in Msun and the distance in kpc."""
    space = vector_space()
    forward = ChirpMassAndDistanceTransform.from_space(
        space, to_chirp_mass=True, constants=CONSTANTS
    )
    mapped = forward.map_space(space)
    assert mapped["chirp_mass"].unit == "Msun"
    assert mapped["distance"].unit == "kpc"
    assert mapped["chirp_mass"].bounds == (0.0, math.inf)
    backward = ChirpMassAndDistanceTransform.from_space(
        mapped, to_chirp_mass=False, constants=CONSTANTS
    )
    assert backward.map_space(mapped) == space
    assert forward.description == "chirp_mass_and_distance"
    assert backward.description == "frequency_dot_and_amplitude"
    assert forward.touched_names(space) == ("frequency_dot", "amplitude")
    with pytest.raises(ConventionError, match="no 'chirp_mass'"):
        ChirpMassAndDistanceTransform.from_space(
            space, to_chirp_mass=False, constants=CONSTANTS
        )
