"""Parameter topologies: real line, interval, circle and sphere.

Each parameter of a :class:`~mixed_signals.core.space.ParamSpace` lies on the
real line, on an interval or on a circle, and two parameters may together form
a sphere, such as a sky position. Estimators, transforms and optimal-transport
costs follow these topologies.

Real line
    Both bounds are infinite.
Interval
    The lower bound lies below the upper bound, and at least one of them is
    finite. A finite bound is an edge of the support. Plain kernel density
    estimates are biased near such an edge (Jones 1993), so bounded kernel
    density estimates reflect the samples at it (Schuster 1985).
Circle
    Both bounds are finite, and the period is the upper minus the lower bound.
    Values that differ by a whole number of periods are the same point
    (Mardia and Jupp 2000).
Sphere
    An azimuth parameter, which is a circle of period 2π, and a polar
    parameter, which is an interval inside the range of its
    :class:`PolarForm`. Angles are in radians.

References:
    :cite:t:`mardia2000directional`; :cite:t:`schuster1985incorporating`;
    :cite:t:`jones1993simple`.
"""

import math
from collections.abc import Mapping
from enum import StrEnum
from types import MappingProxyType
from typing import Final

from pydantic import BaseModel, ConfigDict, model_validator

AZIMUTH_PERIOD: Final = 2 * math.pi
"""Period of the azimuth axis of every sphere, in radians."""

SPHERE_RELATIVE_TOLERANCE: Final = 1e-9
"""Relative tolerance of the sphere checks, which lets last-digit rounding pass."""


class Topology(StrEnum):
    """Topology of one parameter axis: ``real``, ``interval`` or ``circle``."""

    REAL = "real"
    INTERVAL = "interval"
    CIRCLE = "circle"


class PolarForm(StrEnum):
    """Coordinate of the polar axis of a sphere.

    ``latitude`` β runs from −π/2 at the south pole to π/2 at the north pole,
    and ``colatitude`` θ = π/2 − β runs from 0 at the north pole to π at the
    south pole. ``sin_latitude`` sin β and ``cos_colatitude`` cos θ are the same
    coordinate, the height above the equatorial plane, which is uniform on
    [−1, 1] when the points are uniform on the sphere (Mardia and Jupp 2000).
    The ``SineUniform`` and ``CosineUniform`` priors of eryn sample these two
    forms (Karnesis et al. 2023); both names exist, so that a space records
    which physical angle its polar axis stands for.

    References:
        :cite:t:`mardia2000directional`; :cite:t:`karnesis2023eryn`.
    """

    LATITUDE = "latitude"
    COLATITUDE = "colatitude"
    SIN_LATITUDE = "sin_latitude"
    COS_COLATITUDE = "cos_colatitude"

    @property
    def bounds(self) -> tuple[float, float]:
        """Range of the coordinate, from one pole to the other."""
        return _POLAR_BOUNDS[self]


_POLAR_BOUNDS: Final[Mapping[PolarForm, tuple[float, float]]] = MappingProxyType(
    {
        PolarForm.LATITUDE: (-math.pi / 2, math.pi / 2),
        PolarForm.COLATITUDE: (0.0, math.pi),
        PolarForm.SIN_LATITUDE: (-1.0, 1.0),
        PolarForm.COS_COLATITUDE: (-1.0, 1.0),
    }
)


class Sphere(BaseModel):
    """Two parameters that together form a sphere, such as a sky position.

    The space that holds the sphere checks that the azimuth is a circle of
    period 2π and that the polar bounds lie inside the range of the polar form.

    Attributes:
        azimuth: Name of the azimuth parameter.
        polar: Name of the polar parameter.
        polar_form: Coordinate of the polar axis.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    azimuth: str
    polar: str
    polar_form: PolarForm

    @model_validator(mode="after")
    def _check_distinct_axes(self) -> "Sphere":
        if self.azimuth == self.polar:
            raise ValueError(
                f"a sphere needs two different parameters, got {self.azimuth!r} twice"
            )
        return self


def check_bounds(topology: Topology, bounds: tuple[float, float]) -> None:
    """Check that bounds fit a topology.

    Args:
        topology: Topology of the parameter axis.
        bounds: Lower and upper bound.

    Raises:
        ValueError: If a bound is NaN, if the lower bound is not below the
            upper bound, or if the number of finite bounds does not fit the
            topology.
    """
    lower, upper = bounds
    if math.isnan(lower) or math.isnan(upper):
        raise ValueError(f"bounds must not be NaN, got {bounds}")
    if not lower < upper:
        raise ValueError(
            f"the lower bound must lie below the upper bound, got {bounds}"
        )
    finite_count = math.isfinite(lower) + math.isfinite(upper)
    if topology is Topology.REAL and finite_count > 0:
        raise ValueError(
            f"the real line has bounds (-inf, inf), got {bounds};"
            " use topology 'interval' for finite bounds"
        )
    if topology is Topology.INTERVAL and finite_count == 0:
        raise ValueError(
            "an interval needs at least one finite bound;"
            " use topology 'real' when both bounds are infinite"
        )
    if topology is Topology.CIRCLE and finite_count < 2:
        raise ValueError(
            "a circle needs two finite bounds, whose difference is its period,"
            f" got {bounds}"
        )


def check_sphere_azimuth(topology: Topology, bounds: tuple[float, float]) -> None:
    """Check that an axis can be the azimuth of a sphere.

    Args:
        topology: Topology of the azimuth parameter.
        bounds: Bounds of the azimuth parameter.

    Raises:
        ValueError: If the axis is not a circle of period 2π.
    """
    if topology is not Topology.CIRCLE:
        raise ValueError(f"a sphere azimuth must be a circle, got topology {topology}")
    period = bounds[1] - bounds[0]
    if not math.isclose(period, AZIMUTH_PERIOD, rel_tol=SPHERE_RELATIVE_TOLERANCE):
        raise ValueError(
            f"a sphere azimuth needs period 2π = {AZIMUTH_PERIOD!r} rad, got {period!r}"
        )


def check_sphere_polar(
    polar_form: PolarForm, topology: Topology, bounds: tuple[float, float]
) -> None:
    """Check that an axis can be the polar axis of a sphere.

    Args:
        polar_form: Coordinate of the polar axis.
        topology: Topology of the polar parameter.
        bounds: Bounds of the polar parameter.

    Raises:
        ValueError: If the axis is not an interval inside the range of its form.
    """
    if topology is not Topology.INTERVAL:
        raise ValueError(
            f"a sphere polar axis must be an interval, got topology {topology}"
        )
    range_lower, range_upper = polar_form.bounds
    margin = SPHERE_RELATIVE_TOLERANCE * max(abs(range_lower), abs(range_upper))
    if bounds[0] < range_lower - margin or bounds[1] > range_upper + margin:
        raise ValueError(
            f"a {polar_form} polar axis must lie inside"
            f" [{range_lower!r}, {range_upper!r}], got {bounds}"
        )
