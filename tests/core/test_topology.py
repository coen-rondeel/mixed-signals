"""Tests for the parameter topologies and the sphere checks."""

import math

import pytest
from pydantic import ValidationError

from mixed_signals.core.topology import (
    AZIMUTH_PERIOD,
    PolarForm,
    Sphere,
    Topology,
    check_bounds,
    check_sphere_azimuth,
    check_sphere_polar,
)

INF = math.inf


@pytest.mark.parametrize(
    ("topology", "bounds"),
    [
        (Topology.REAL, (-INF, INF)),
        (Topology.INTERVAL, (0.0, 1.0)),
        (Topology.INTERVAL, (0.0, INF)),
        (Topology.INTERVAL, (-INF, -5.0)),
        (Topology.CIRCLE, (0.0, 2 * math.pi)),
        (Topology.CIRCLE, (-math.pi, math.pi)),
    ],
)
def test_valid_bounds_pass(topology: Topology, bounds: tuple[float, float]) -> None:
    """Each topology accepts the bounds it is defined by."""
    check_bounds(topology, bounds)


@pytest.mark.parametrize(
    ("topology", "bounds", "message"),
    [
        (Topology.INTERVAL, (math.nan, 1.0), "NaN"),
        (Topology.INTERVAL, (1.0, 1.0), "below the upper bound"),
        (Topology.INTERVAL, (2.0, 1.0), "below the upper bound"),
        (Topology.REAL, (0.0, INF), "use topology 'interval'"),
        (Topology.INTERVAL, (-INF, INF), "use topology 'real'"),
        (Topology.CIRCLE, (0.0, INF), "two finite bounds"),
    ],
)
def test_invalid_bounds_name_the_fix(
    topology: Topology, bounds: tuple[float, float], message: str
) -> None:
    """Bounds that do not fit the topology fail with a message that says why."""
    with pytest.raises(ValueError, match=message):
        check_bounds(topology, bounds)


def test_polar_forms_span_pole_to_pole() -> None:
    """Each polar form runs from one pole to the other."""
    assert PolarForm.LATITUDE.bounds == (-math.pi / 2, math.pi / 2)
    assert PolarForm.COLATITUDE.bounds == (0.0, math.pi)
    assert PolarForm.SIN_LATITUDE.bounds == (-1.0, 1.0)
    assert PolarForm.COS_COLATITUDE.bounds == (-1.0, 1.0)


def test_sphere_needs_two_different_parameters() -> None:
    """A sphere cannot use one parameter for both axes."""
    with pytest.raises(ValidationError, match="two different parameters"):
        Sphere(azimuth="ra", polar="ra", polar_form=PolarForm.LATITUDE)


def test_sphere_rejects_unknown_polar_forms() -> None:
    """Only the four polar forms are accepted."""
    with pytest.raises(ValidationError, match="sin_latitude"):
        Sphere.model_validate(
            {"azimuth": "ra", "polar": "dec", "polar_form": "declination"}
        )


@pytest.mark.parametrize("bounds", [(0.0, 2 * math.pi), (-math.pi, math.pi)])
def test_azimuth_accepts_full_turns(bounds: tuple[float, float]) -> None:
    """Any circle of period 2π can be an azimuth."""
    check_sphere_azimuth(Topology.CIRCLE, bounds)


def test_azimuth_accepts_last_digit_rounding() -> None:
    """A period of 2π rounded in its last digits still passes."""
    check_sphere_azimuth(Topology.CIRCLE, (0.0, 6.283_185_307_18))


@pytest.mark.parametrize(
    ("topology", "bounds", "message"),
    [
        (Topology.CIRCLE, (0.0, 6.2832), "period 2π"),
        (Topology.CIRCLE, (0.0, 360.0), "period 2π"),
        (Topology.INTERVAL, (0.0, AZIMUTH_PERIOD), "must be a circle"),
    ],
)
def test_azimuth_rejects_other_axes(
    topology: Topology, bounds: tuple[float, float], message: str
) -> None:
    """Degrees, rounded periods and intervals cannot be an azimuth."""
    with pytest.raises(ValueError, match=message):
        check_sphere_azimuth(topology, bounds)


@pytest.mark.parametrize("polar_form", list(PolarForm))
def test_polar_axis_accepts_its_full_range(polar_form: PolarForm) -> None:
    """Every polar form accepts bounds that span its whole range."""
    check_sphere_polar(polar_form, Topology.INTERVAL, polar_form.bounds)


def test_polar_axis_accepts_eryn_margins() -> None:
    """Bounds 1e-12 inside [-1, 1], where eryn keeps sin and cos axes, pass."""
    margin = 1e-12
    eryn_bounds = (-1.0 + margin, 1.0 - margin)
    check_sphere_polar(PolarForm.SIN_LATITUDE, Topology.INTERVAL, eryn_bounds)
    check_sphere_polar(PolarForm.COS_COLATITUDE, Topology.INTERVAL, eryn_bounds)


@pytest.mark.parametrize(
    ("polar_form", "topology", "bounds", "message"),
    [
        (PolarForm.LATITUDE, Topology.INTERVAL, (-2.0, 2.0), "must lie inside"),
        (PolarForm.COLATITUDE, Topology.INTERVAL, (-0.1, math.pi), "must lie inside"),
        (PolarForm.SIN_LATITUDE, Topology.INTERVAL, (-1.0, 1.5), "must lie inside"),
        (PolarForm.LATITUDE, Topology.CIRCLE, (-1.0, 1.0), "must be an interval"),
        (PolarForm.LATITUDE, Topology.REAL, (-INF, INF), "must be an interval"),
    ],
)
def test_polar_axis_rejects_bounds_beyond_the_poles(
    polar_form: PolarForm,
    topology: Topology,
    bounds: tuple[float, float],
    message: str,
) -> None:
    """A polar axis must be an interval that stays between the poles."""
    with pytest.raises(ValueError, match=message):
        check_sphere_polar(polar_form, topology, bounds)
