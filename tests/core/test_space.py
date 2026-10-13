"""Tests for ``ParamSpace``: validation, accessors and the YAML round trip."""

import math
from pathlib import Path

import pytest
from pydantic import ValidationError

from mixed_signals.core.space import Parameter, ParamSpace
from mixed_signals.core.topology import PolarForm, Sphere, Topology

# * A galactic-binary-like space with every topology and a sky-position sphere.
SPACE_YAML = r"""
parameters:
  - name: frequency
    label: f
    unit: Hz
  - name: amplitude
    label: \mathcal{A}
    topology: interval
    bounds: [1.0e-24, .inf]
  - name: phase
    label: \phi_0
    unit: rad
    topology: circle
    bounds: [0.0, 6.283185307179586]
  - name: ecliptic_longitude
    label: \lambda
    unit: rad
    topology: circle
    bounds: [0.0, 6.283185307179586]
  - name: sin_ecliptic_latitude
    label: '\sin\beta'
    topology: interval
    bounds: [-1.0, 1.0]
spheres:
  - azimuth: ecliptic_longitude
    polar: sin_ecliptic_latitude
    polar_form: sin_latitude
"""


@pytest.fixture
def space() -> ParamSpace:
    """The example space of this module."""
    return ParamSpace.from_yaml_text(SPACE_YAML)


def test_yaml_text_round_trips(space: ParamSpace) -> None:
    """Writing a space to YAML and reading it back gives the same space."""
    assert ParamSpace.from_yaml_text(space.to_yaml_text()) == space


def test_yaml_file_round_trips(space: ParamSpace, tmp_path: Path) -> None:
    """A space written to a file reads back unchanged."""
    path = tmp_path / "space.yaml"
    space.to_yaml(path)
    assert ParamSpace.from_yaml(path) == space
    assert ParamSpace.from_yaml(str(path)) == space


def test_yaml_output_matches_the_input_layout(space: ParamSpace) -> None:
    """The output leaves out defaults and keeps bounds on one line."""
    text = space.to_yaml_text()
    assert "topology: real" not in text
    assert "bounds: [0.0, 6.283185307179586]" in text
    assert "bounds: [1.0e-24, .inf]" in text
    assert ParamSpace.from_yaml_text(SPACE_YAML).to_yaml_text() == text


def test_yaml_output_leaves_out_labels_equal_to_the_name() -> None:
    """A label that repeats the name is a default and is not written."""
    space = ParamSpace(parameters=(Parameter(name="chirp_mass"),))
    assert space.to_yaml_text() == "parameters:\n  - name: chirp_mass\n"


def test_yaml_reads_exponents_without_a_decimal_point() -> None:
    """PyYAML reads 1e-24 as text; validation still turns it into a number."""
    text = "parameters: [{name: amplitude, topology: interval, bounds: [1e-24, 1e-20]}]"
    assert ParamSpace.from_yaml_text(text)["amplitude"].bounds == (1e-24, 1e-20)


def test_fields_read_from_yaml(space: ParamSpace) -> None:
    """Every field of the YAML ends up in the parameters and spheres."""
    assert space["frequency"] == Parameter(name="frequency", label="f", unit="Hz")
    assert space["amplitude"].bounds == (1e-24, math.inf)
    assert space["phase"].topology is Topology.CIRCLE
    assert space["sin_ecliptic_latitude"].label == r"\sin\beta"
    assert space.spheres == (
        Sphere(
            azimuth="ecliptic_longitude",
            polar="sin_ecliptic_latitude",
            polar_form=PolarForm.SIN_LATITUDE,
        ),
    )


def test_parameter_defaults() -> None:
    """A bare parameter lies on the real line and takes its name as label."""
    parameter = Parameter(name="chirp_mass")
    assert parameter.label == "chirp_mass"
    assert parameter.unit is None
    assert parameter.topology is Topology.REAL
    assert parameter.bounds == (-math.inf, math.inf)
    assert parameter.period is None


def test_circle_period(space: ParamSpace) -> None:
    """The period of a circle is the difference of its bounds."""
    assert space["phase"].period == pytest.approx(2 * math.pi)


def test_accessors_follow_the_sample_axis(space: ParamSpace) -> None:
    """Names, length and positions follow the order of the last sample axis."""
    assert space.names == (
        "frequency",
        "amplitude",
        "phase",
        "ecliptic_longitude",
        "sin_ecliptic_latitude",
    )
    assert len(space) == 5
    assert space.index("phase") == 2
    assert space["phase"] is space.parameters[2]


def test_unknown_names_list_the_known_ones(space: ParamSpace) -> None:
    """Looking up a missing parameter names the parameters that exist."""
    with pytest.raises(KeyError, match=r"no parameter named 'beta'.*'frequency'"):
        space.index("beta")


def test_spaces_are_immutable(space: ParamSpace) -> None:
    """A space cannot change after validation."""
    with pytest.raises(ValidationError, match="frozen"):
        space.spheres = ()  # type: ignore[misc]


def test_space_builds_from_python_objects() -> None:
    """Spaces also build directly from parameter and sphere objects."""
    space = ParamSpace(
        parameters=(
            Parameter(name="ra", topology=Topology.CIRCLE, bounds=(0.0, 2 * math.pi)),
            Parameter(
                name="dec",
                topology=Topology.INTERVAL,
                bounds=(-math.pi / 2, math.pi / 2),
            ),
        ),
        spheres=(Sphere(azimuth="ra", polar="dec", polar_form=PolarForm.LATITUDE),),
    )
    assert space.names == ("ra", "dec")


CIRCLE = "{name: lam, topology: circle, bounds: [0.0, 6.283185307179586]}"
SIN_AXIS = "{name: sinb, topology: interval, bounds: [-1.0, 1.0]}"


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("parameters: []", "at least one parameter"),
        ("parameters: [{name: a}, {name: a}]", "repeated: 'a'"),
        ("parameters: [{name: ' a'}]", "without surrounding whitespace"),
        ("parameters: [{name: ''}]", "non-empty"),
        ("parameters: [{name: a, bounds: [0, 1]}]", "parameter 'a'.*'interval'"),
        ("parameters: [{name: a, topology: interval}]", "parameter 'a'.*'real'"),
        (
            "parameters: [{name: a, topology: circle, bounds: [0, .inf]}]",
            "parameter 'a'.*two finite bounds",
        ),
        (
            "parameters: [{name: a, topology: interval, bounds: [1, 0]}]",
            "below the upper bound",
        ),
        ("parameters: [{name: a, topology: torus}]", "'real', 'interval' or 'circle'"),
        ("parameters: [{name: a, bound: [0, 1]}]", "Extra inputs"),
        ("parameters: [{name: a, bounds: [0, 2pi]}]", "valid number"),
        ("- a\n- b", "valid dictionary"),
        (
            f"parameters: [{CIRCLE}]\n"
            "spheres: [{azimuth: lam, polar: b, polar_form: latitude}]",
            "polar 'b' is not a parameter",
        ),
        (
            f"parameters: [{CIRCLE}, {SIN_AXIS}, {{name: b}}]\n"
            "spheres: [{azimuth: lam, polar: sinb, polar_form: sin_latitude},"
            " {azimuth: lam, polar: b, polar_form: latitude}]",
            "'lam' is in spheres 0 and 1",
        ),
        (
            "parameters: [{name: lam, topology: circle, bounds: [0, 360]},"
            f" {SIN_AXIS}]\n"
            "spheres: [{azimuth: lam, polar: sinb, polar_form: sin_latitude}]",
            "azimuth 'lam'.*period 2π",
        ),
        (
            f"parameters: [{CIRCLE}, {{name: b, topology: circle, bounds: [0, 3]}}]\n"
            "spheres: [{azimuth: lam, polar: b, polar_form: latitude}]",
            "polar 'b'.*must be an interval",
        ),
        (
            f"parameters: [{CIRCLE},"
            " {name: b, topology: interval, bounds: [-2, 2]}]\n"
            "spheres: [{azimuth: lam, polar: b, polar_form: latitude}]",
            "polar 'b'.*must lie inside",
        ),
    ],
)
def test_invalid_spaces_fail_with_one_clear_message(text: str, message: str) -> None:
    """Each invalid space raises one error whose message names the fault."""
    with pytest.raises(ValidationError, match=message) as error_info:
        ParamSpace.from_yaml_text(text)
    assert error_info.value.error_count() == 1
