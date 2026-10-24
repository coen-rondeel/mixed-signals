r"""``ParamSpace``: parameter names, labels, units, topologies and bounds.

Every input to a comparison shares one ``ParamSpace``, whose parameters follow
the order of the last axis of the sample arrays. A space reads from and writes
to YAML, with one entry per parameter and a list of spheres::

    parameters:
      - name: frequency
        label: f
        unit: Hz
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
        label: \sin\beta
        topology: interval
        bounds: [-1.0, 1.0]
    spheres:
      - azimuth: ecliptic_longitude
        polar: sin_ecliptic_latitude
        polar_form: sin_latitude

A parameter without ``topology`` lies on the real line, and one without
``label`` takes its name as label. Write LaTeX labels plain or in single
quotes, because double-quoted YAML strings treat backslashes as escapes. An
invalid space raises :class:`pydantic.ValidationError` with a message that
names the parameter or sphere at fault.
"""

import math
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, NamedTuple

import yaml
from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from mixed_signals.core.topology import (
    Sphere,
    Topology,
    check_bounds,
    check_sphere_azimuth,
    check_sphere_polar,
)


class AxisBlocks(NamedTuple):
    """Positions of the parameters of a space, grouped by their geometry.

    Transforms, estimators and optimal-transport costs treat the groups
    differently: the Euclidean block may be rotated and scaled, a circle only
    scaled, and a sphere pair only handled as a whole.
    """

    euclidean: tuple[int, ...]
    """Positions of the real and interval parameters outside every sphere."""
    circles: tuple[int, ...]
    """Positions of the circle parameters outside every sphere."""
    spheres: tuple[tuple[int, int], ...]
    """Positions of the azimuth and the polar parameter of each sphere."""


class Parameter(BaseModel):
    """One parameter of a :class:`ParamSpace`.

    Attributes:
        name: Name, unique within the space, without surrounding whitespace.
        label: LaTeX label for plots; the name when not given.
        unit: Unit string, or ``None`` for a parameter without a unit.
        topology: Topology of the parameter axis.
        bounds: Lower and upper bound; ``(-inf, inf)`` on the real line.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", ser_json_inf_nan="constants")

    name: str
    label: str = ""
    unit: str | None = None
    topology: Topology = Topology.REAL
    bounds: tuple[float, float] = (-math.inf, math.inf)

    @model_validator(mode="before")
    @classmethod
    def _default_label(cls, data: Any) -> Any:
        # * Filling the label before validation keeps a bad name to a single error.
        if (
            isinstance(data, dict)
            and not data.get("label")
            and isinstance(data.get("name"), str)
        ):
            return {**data, "label": data["name"]}
        return data

    @field_validator("name")
    @classmethod
    def _check_name(cls, name: str) -> str:
        if not name or name != name.strip():
            raise ValueError(
                "a parameter name must be non-empty, without surrounding whitespace,"
                f" got {name!r}"
            )
        return name

    @model_validator(mode="after")
    def _check_topology(self) -> "Parameter":
        try:
            check_bounds(self.topology, self.bounds)
        except ValueError as error:
            raise ValueError(f"parameter {self.name!r}: {error}") from None
        return self

    @property
    def period(self) -> float | None:
        """Period of a circle, or ``None`` for the other topologies."""
        if self.topology is Topology.CIRCLE:
            return self.bounds[1] - self.bounds[0]
        return None


class ParamSpace(BaseModel):
    """Names, labels, units, topologies and bounds of the parameters of a problem.

    Attributes:
        parameters: The parameters, in the order of the last sample axis.
        spheres: Pairs of parameters that together form a sphere.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", ser_json_inf_nan="constants")

    parameters: tuple[Parameter, ...]
    spheres: tuple[Sphere, ...] = ()

    @model_validator(mode="after")
    def _check_space(self) -> "ParamSpace":
        # * Checked here, as min_length would also fail whenever an entry fails.
        if not self.parameters:
            raise ValueError("a space needs at least one parameter")
        names = [parameter.name for parameter in self.parameters]
        repeated = sorted({name for name in names if names.count(name) > 1})
        if repeated:
            repeated_text = ", ".join(map(repr, repeated))
            raise ValueError(
                f"parameter names must be unique, repeated: {repeated_text}"
            )
        by_name = {parameter.name: parameter for parameter in self.parameters}
        sphere_of_name: dict[str, int] = {}
        for sphere_index, sphere in enumerate(self.spheres):
            for role, name in (("azimuth", sphere.azimuth), ("polar", sphere.polar)):
                if name not in by_name:
                    raise ValueError(
                        f"sphere {sphere_index}: {role} {name!r} is not a parameter"
                        " of the space"
                    )
                if name in sphere_of_name:
                    raise ValueError(
                        f"parameter {name!r} is in spheres {sphere_of_name[name]} and"
                        f" {sphere_index}; a parameter belongs to one sphere at most"
                    )
                sphere_of_name[name] = sphere_index
            azimuth, polar = by_name[sphere.azimuth], by_name[sphere.polar]
            try:
                check_sphere_azimuth(azimuth.topology, azimuth.bounds)
            except ValueError as error:
                raise ValueError(
                    f"sphere {sphere_index}, azimuth {azimuth.name!r}: {error}"
                ) from None
            try:
                check_sphere_polar(sphere.polar_form, polar.topology, polar.bounds)
            except ValueError as error:
                raise ValueError(
                    f"sphere {sphere_index}, polar {polar.name!r}: {error}"
                ) from None
        return self

    @property
    def names(self) -> tuple[str, ...]:
        """Parameter names, in the order of the last sample axis."""
        return tuple(parameter.name for parameter in self.parameters)

    @property
    def axis_blocks(self) -> AxisBlocks:
        """Positions of the parameters in the Euclidean block, circles and spheres.

        A sphere's azimuth is a circle and its polar axis an interval, but both
        belong to the sphere only.
        """
        sphere_pairs = tuple(
            (self.index(sphere.azimuth), self.index(sphere.polar))
            for sphere in self.spheres
        )
        on_sphere = {position for pair in sphere_pairs for position in pair}
        free = [
            (position, parameter.topology is Topology.CIRCLE)
            for position, parameter in enumerate(self.parameters)
            if position not in on_sphere
        ]
        return AxisBlocks(
            euclidean=tuple(position for position, circle in free if not circle),
            circles=tuple(position for position, circle in free if circle),
            spheres=sphere_pairs,
        )

    def __len__(self) -> int:
        """Return the number of parameters, the size of the last sample axis."""
        return len(self.parameters)

    def __getitem__(self, name: str) -> Parameter:
        """Return the parameter with the given name."""
        return self.parameters[self.index(name)]

    def index(self, name: str) -> int:
        """Return the position of a parameter on the last sample axis.

        Args:
            name: Name of the parameter.

        Returns:
            The zero-based position.

        Raises:
            KeyError: If the space has no parameter of that name.
        """
        try:
            return self.names.index(name)
        except ValueError:
            raise KeyError(
                f"no parameter named {name!r}; the space has"
                f" {', '.join(map(repr, self.names))}"
            ) from None

    @classmethod
    def from_eryn_priors(
        cls,
        priors: object,
        *,
        periodic: (
            Mapping[str, float] | Mapping[int, float] | Mapping[str | int, float] | None
        ) = None,
        spheres: Sequence[Sphere] = (),
    ) -> "ParamSpace":
        """Build the space of eryn's sampling coordinates from a prior container.

        Names are the keys of the container, or ``x0``, ``x1``, ... for integer
        keys, and labels and units are eryn's. A parameter is a circle when its
        prior is periodic or ``periodic`` lists it, and otherwise an interval
        from the bounds of its prior or the real line. ``DeltaFunction``
        parameters are left out, and discrete priors raise. See
        :mod:`mixed_signals.priors.eryn_container` for the details.

        Args:
            priors: eryn ``ProbDistContainer`` of one branch.
            periodic: The branch's entry of the sampler's ``periodic`` dict:
                periods keyed by parameter name or position.
            spheres: Spheres of the space, named by the parameter names.

        Returns:
            The validated space.

        Raises:
            TypeError: If ``priors`` is not a ``ProbDistContainer``.
            ValueError: If a prior or a period does not fit the space.
            MissingDependencyError: If eryn is not installed.

        References:
            :cite:t:`karnesis2023eryn`.
        """
        from mixed_signals.priors.eryn_container import from_eryn_priors

        return from_eryn_priors(priors, periodic=periodic, spheres=spheres)

    @classmethod
    def from_yaml_text(cls, text: str) -> "ParamSpace":
        """Build a space from YAML text.

        Args:
            text: YAML document with ``parameters`` and optional ``spheres``.

        Returns:
            The validated space.
        """
        return cls.model_validate(yaml.safe_load(text))

    @classmethod
    def from_yaml(cls, path: str | os.PathLike[str]) -> "ParamSpace":
        """Read a space from a YAML file.

        Args:
            path: Path of the file, read as UTF-8.

        Returns:
            The validated space.
        """
        return cls.from_yaml_text(Path(path).read_text(encoding="utf-8"))

    def to_yaml_text(self) -> str:
        """Write the space as YAML text, leaving out fields at their defaults.

        Returns:
            The YAML document, which :meth:`from_yaml_text` reads back unchanged.
        """
        space_data = self.model_dump(mode="json", exclude_defaults=True)
        for entry in space_data["parameters"]:
            if entry.get("label") == entry["name"]:
                del entry["label"]
        return yaml.dump(
            space_data, Dumper=_SpaceDumper, sort_keys=False, allow_unicode=True
        )

    def to_yaml(self, path: str | os.PathLike[str]) -> None:
        """Write the space to a YAML file.

        Args:
            path: Path of the file, written as UTF-8.
        """
        Path(path).write_text(self.to_yaml_text(), encoding="utf-8")


class _SpaceDumper(yaml.SafeDumper):
    """YAML dumper that indents lists and writes lists of scalars on one line."""

    def increase_indent(self, flow: bool = False, indentless: bool = False) -> None:
        """Indent block lists under their key, as in the module example."""
        super().increase_indent(flow=flow, indentless=False)


def _represent_list(dumper: yaml.SafeDumper, items: list[Any]) -> yaml.SequenceNode:
    scalars_only = all(not isinstance(item, list | dict) for item in items)
    return dumper.represent_sequence(
        "tag:yaml.org,2002:seq", items, flow_style=scalars_only
    )


_SpaceDumper.add_representer(list, _represent_list)
