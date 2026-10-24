"""eryn priors as a ``ParamSpace``, a ``Transform`` and a ``PriorDistribution``.

eryn samples each parameter in a sampling coordinate u, which the transform of
its prior maps to the physical parameter x, such as u = ln x for ``LogUniform``
or u = sin β for ``SineUniform`` (Karnesis et al. 2023). The chains, the prior
bounds and the log-prior values of a run are all in the sampling coordinates.

Spaces
    :func:`from_eryn_priors`, which
    :meth:`~mixed_signals.core.space.ParamSpace.from_eryn_priors` calls,
    describes the sampling coordinates of a ``ProbDistContainer``. Names are
    the keys of the container, or ``x0``, ``x1``, ... for integer keys. Labels
    and units are eryn's, which describe the sampled coordinate; eryn labels a
    prior without a name ``"None"``, and such a parameter takes its name as
    label instead.
Topology
    A parameter is a circle when its prior has ``boundary="periodic"`` or the
    sampler's ``periodic`` dict lists it. The circle spans the bounds of the
    prior, which must differ by the period; without finite bounds it spans
    [0, P), the range into which eryn wraps it as x mod P. Every other
    parameter is an interval from the bounds of its prior, or the real line
    when both are infinite. eryn's ``reflective`` flag changes only its
    proposals, so it maps like no flag. Joint priors and priors that span
    several parameters have no bounds per parameter, so their axes are real
    lines unless periodic.
Fixed and discrete parameters
    A ``DeltaFunction`` fixes its parameter, whose zero spread would make every
    covariance and whitening singular. Spaces, chains and the classes below
    leave it out and put its value back only when they call eryn. Discrete
    priors, eryn's ``DiscreteUniform``, ``CategoricalDistribution`` and
    ``PoissonDistribution`` and any prior with ``logpmf``, raise, because v1
    compares continuous parameters only.
Physical parameters
    :class:`ErynPhysical` maps sampling coordinates to physical parameters with
    eryn's own transforms, on a JAX copy of the container, so that its maps,
    log-determinants and Jacobians run under ``jax.jit``, ``jax.vmap`` and
    ``jax.grad``. A density moves with ln|det ∂x/∂u| (Papamakarios et al.
    2021), and a Fisher matrix with the Jacobian of the inverse map (Jeffreys
    1946).
Prior densities
    :class:`ErynPrior` makes a container meet
    :class:`~mixed_signals.priors.protocol.PriorDistribution` in the sampling
    coordinates, those of eryn's stored ``log_prior``.

eryn is optional (the ``eryn`` extra), so this module imports it inside
functions.

References:
    :cite:t:`karnesis2023eryn`; :cite:t:`papamakarios2021normalizing`;
    :cite:t:`jeffreys1946invariant`.
"""

import importlib
import math
from collections.abc import Mapping, Sequence
from types import MappingProxyType, ModuleType
from typing import Any, Final, NamedTuple

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike
from jaxtyping import Array, Float, PRNGKeyArray

from mixed_signals.core.space import Parameter, ParamSpace
from mixed_signals.core.topology import PolarForm, Sphere, Topology
from mixed_signals.errors import MissingDependencyError
from mixed_signals.transforms.base import Transform

PERIOD_RELATIVE_TOLERANCE: Final = 1e-9
"""Relative tolerance between a sampler period and the width of its prior.

As for the sphere checks of :mod:`mixed_signals.core.topology`, it lets
rounding in the last digits pass, such as 2π written to fewer digits.
"""

PHYSICAL_DESCRIPTION: Final = "physical(eryn)"
"""Description of :class:`ErynPhysical` in the ``space`` column of result rows."""

_INSTALL_HINT: Final = (
    "install it with the eryn extra: pip install 'mixed-signals[eryn]'"
)
_REAL_BOUNDS: Final = (-math.inf, math.inf)
# * The angle that eryn's sine and cosine priors turn their sampled height into.
_PHYSICAL_POLAR_FORMS: Final[Mapping[PolarForm, PolarForm]] = MappingProxyType(
    {
        PolarForm.SIN_LATITUDE: PolarForm.LATITUDE,
        PolarForm.COS_COLATITUDE: PolarForm.COLATITUDE,
    }
)


class _Axis(NamedTuple):
    """What the priors say about one parameter, in its sampling coordinate."""

    name: str
    """Container key, or ``x{position}`` for integer keys."""
    position: int
    """Position in the parameter vector of the container."""
    prior: str
    """Class name of the prior, for messages."""
    label: str
    """eryn's label, or the name when eryn has none."""
    unit: str | None
    """eryn's unit."""
    bounds: tuple[float, float]
    """Bounds of the sampling coordinate; infinite when the prior has none."""
    periodic: bool
    """Whether the prior has ``boundary="periodic"``."""
    fixed: float | None
    """Value of a ``DeltaFunction`` parameter, ``None`` otherwise."""
    discrete: bool
    """Whether the prior is discrete."""
    transformed: bool
    """Whether eryn maps the coordinate to another physical parameter."""
    physical_name: str
    """eryn's ``name_phys``, or the name when eryn has none."""
    polar_form: PolarForm | None
    """Polar form that a ``SineUniform`` or ``CosineUniform`` prior samples."""


def from_eryn_priors(
    priors: object,
    *,
    periodic: (
        Mapping[str, float] | Mapping[int, float] | Mapping[str | int, float] | None
    ) = None,
    spheres: Sequence[Sphere] = (),
) -> ParamSpace:
    """Build the space of eryn's sampling coordinates from a prior container.

    :meth:`ParamSpace.from_eryn_priors
    <mixed_signals.core.space.ParamSpace.from_eryn_priors>` calls this
    function; the module docstring gives the rules.

    Args:
        priors: eryn ``ProbDistContainer`` of one branch.
        periodic: The branch's entry of the sampler's ``periodic`` dict:
            periods keyed by parameter name or position.
        spheres: Spheres of the space, named by the parameter names. A
            ``SineUniform`` polar axis needs the form ``sin_latitude``, and a
            ``CosineUniform`` one ``cos_colatitude``.

    Returns:
        The space of the parameters without a ``DeltaFunction`` prior.

    Raises:
        TypeError: If ``priors`` is not a ``ProbDistContainer``.
        ValueError: If a prior is discrete, a period does not fit its prior or
            its key, a periodic parameter has neither finite bounds nor a
            period, or a sphere does not fit its priors.
        pydantic.ValidationError: If the space is invalid, for example when a
            sphere names an unknown parameter.
        MissingDependencyError: If eryn is not installed.

    References:
        :cite:t:`karnesis2023eryn`.
    """
    axes = _describe(priors)
    _check_axes(axes)
    periods = _periods(axes, periodic)
    by_name = {axis.name: axis for axis in axes}
    for sphere in spheres:
        _check_sphere(sphere, by_name)
    parameters = tuple(
        _sampling_parameter(axis, periods.get(axis.position))
        for axis in axes
        if axis.fixed is None
    )
    return ParamSpace(parameters=parameters, spheres=tuple(spheres))


class ErynPhysical(Transform):
    """eryn's map from sampling coordinates u to physical parameters x = g(u).

    eryn's own ``forward_transform`` goes from physical to sampling
    coordinates, so :meth:`forward` is eryn's ``inverse_transform`` and
    :meth:`inverse` its ``forward_transform``. :meth:`log_abs_det_jacobian` is
    ln|det ∂x/∂u| = −ln|det ∂u/∂x|, the negative of eryn's ``log_jacobian``
    at x = g(u). Parameters without a transform pass unchanged. The points
    hold the parameters of the space, without those that a ``DeltaFunction``
    fixes; the transform puts the fixed values back only for eryn's maps.

    Args:
        priors: eryn ``ProbDistContainer`` of the run, on any backend; the
            transform keeps a JAX copy and leaves it unchanged.

    Attributes:
        priors: JAX copy of the container.
        free: Container positions of the parameters that the points hold.
        fixed: Container positions and values of the fixed parameters.

    Raises:
        TypeError: If ``priors`` is not a ``ProbDistContainer``.
        ValueError: If a prior is discrete or every parameter is fixed.
        MissingDependencyError: If eryn is not installed or has no JAX
            backend for its priors.

    References:
        :cite:t:`karnesis2023eryn`; :cite:t:`papamakarios2021normalizing`;
        :cite:t:`jeffreys1946invariant`.
    """

    priors: Any = eqx.field(static=True)
    free: tuple[int, ...] = eqx.field(static=True)
    fixed: tuple[tuple[int, float], ...] = eqx.field(static=True)
    _axes: tuple[_Axis, ...] = eqx.field(static=True)

    def __init__(self, priors: object) -> None:
        """Describe the container and keep its JAX copy."""
        self.priors, self.free, self.fixed, self._axes = _prepare(priors)

    def forward(
        self, points: Float[Array, "*leading param"]
    ) -> Float[Array, "*leading param"]:
        """Return the physical parameters x = g(u) at sampling coordinates u."""
        full = _with_fixed(points, self.free, self.fixed)
        return _free_columns(self.priors.inverse_transform(full), self.free, self.fixed)

    def inverse(
        self, points: Float[Array, "*leading param"]
    ) -> Float[Array, "*leading param"]:
        """Return the sampling coordinates u = g⁻¹(x) at physical parameters x."""
        full = _with_fixed(points, self.free, self.fixed)
        return _free_columns(self.priors.forward_transform(full), self.free, self.fixed)

    def log_abs_det_jacobian(
        self, points: Float[Array, "*leading param"]
    ) -> Float[Array, " *leading"]:
        """Return ln|det ∂x/∂u| at sampling coordinates u.

        Fixed parameters have no transform, so they add nothing.
        """
        physical = self.priors.inverse_transform(
            _with_fixed(points, self.free, self.fixed)
        )
        return -jnp.asarray(self.priors.log_jacobian(physical))

    @property
    def description(self) -> str:
        """``physical(eryn)``, for the ``space`` column of result rows."""
        return PHYSICAL_DESCRIPTION

    def map_space(self, space: ParamSpace) -> ParamSpace:
        """Return the space of the physical parameters.

        A transformed parameter takes eryn's ``name_phys`` as its name and
        label and has no unit, because eryn's label and unit describe the
        sampled coordinate. Its bounds are the inverse transforms of its
        sampling bounds, in increasing order, and it is the real line when
        both are infinite. On a sphere, the polar axis of a ``SineUniform``
        prior turns from ``sin_latitude`` into ``latitude``, and that of a
        ``CosineUniform`` prior from ``cos_colatitude`` into ``colatitude``.
        Parameters without a transform keep name, label, unit, topology and
        bounds. The method reads concrete values, so it runs outside
        ``jax.jit``.

        Args:
            space: Space of the sampling coordinates, with one parameter per
                free position of the priors, such as the space of
                :func:`from_eryn_priors`.

        Returns:
            The space of the physical parameters.

        Raises:
            ValueError: If the space has another number of parameters than
                the transform, a circle has a transform, a sphere axis has a
                transform other than the sine and cosine priors, or the
                inverse transform of a bound is NaN.
        """
        if len(space) != len(self.free):
            raise ValueError(
                f"the space has {len(space)} parameters, but the priors have"
                f" {len(self.free)} free parameters"
            )
        lower = jnp.asarray([parameter.bounds[0] for parameter in space.parameters])
        upper = jnp.asarray([parameter.bounds[1] for parameter in space.parameters])
        mapped = np.asarray(self.forward(jnp.stack([lower, upper])))
        sphere_of = {
            name: sphere
            for sphere in space.spheres
            for name in (sphere.azimuth, sphere.polar)
        }
        parameters = list(space.parameters)
        names: dict[str, str] = {}
        for index, axis in enumerate(self._axes):
            if not axis.transformed:
                continue
            original = parameters[index]
            if original.topology is Topology.CIRCLE:
                raise ValueError(
                    f"parameter {original.name!r} is a circle with a {axis.prior}"
                    " transform, which has no constant period"
                )
            sphere = sphere_of.get(original.name)
            if sphere is not None and (
                sphere.polar != original.name or axis.polar_form is None
            ):
                raise ValueError(
                    f"sphere axis {original.name!r} has a {axis.prior} transform;"
                    " only SineUniform and CosineUniform priors map a polar axis"
                )
            bounds = tuple(sorted(float(value) for value in mapped[:, index]))
            if any(math.isnan(bound) for bound in bounds):
                raise ValueError(
                    f"the inverse transform of the bounds {original.bounds} of"
                    f" {original.name!r} is NaN; build the physical space by hand"
                )
            parameters[index] = Parameter(
                name=axis.physical_name,
                label=axis.physical_name,
                topology=_open_topology(bounds),
                bounds=bounds,
            )
            names[original.name] = axis.physical_name
        spheres = tuple(
            _physical_sphere(sphere, names, self._axes, space)
            for sphere in space.spheres
        )
        return ParamSpace(parameters=tuple(parameters), spheres=spheres)


class ErynPrior(eqx.Module):
    """An eryn container as a ``PriorDistribution`` of the sampling coordinates.

    :meth:`logpdf` is eryn's ``logpdf``, the density of the sampled
    coordinates that eryn stores as ``log_prior``, and :meth:`rvs` passes the
    package's ``key`` on to eryn's ``rng_key``. As for :class:`ErynPhysical`,
    samples hold the parameters without a ``DeltaFunction`` prior, whose
    values return only for eryn's calls.

    Args:
        priors: eryn ``ProbDistContainer``, on any backend; the adapter keeps
            a JAX copy and leaves it unchanged.

    Attributes:
        priors: JAX copy of the container.
        free: Container positions of the parameters that samples hold.
        fixed: Container positions and values of the fixed parameters.

    Raises:
        TypeError: If ``priors`` is not a ``ProbDistContainer``.
        ValueError: If a prior is discrete or every parameter is fixed.
        MissingDependencyError: If eryn is not installed or has no JAX
            backend for its priors.

    References:
        :cite:t:`karnesis2023eryn`.
    """

    priors: Any = eqx.field(static=True)
    free: tuple[int, ...] = eqx.field(static=True)
    fixed: tuple[tuple[int, float], ...] = eqx.field(static=True)

    def __init__(self, priors: object) -> None:
        """Describe the container and keep its JAX copy."""
        self.priors, self.free, self.fixed, _ = _prepare(priors)

    def logpdf(self, samples: ArrayLike) -> Float[Array, " *leading"]:
        """Return the log prior density at samples shaped (..., D).

        Args:
            samples: Sampling coordinates of the free parameters.

        Returns:
            The log densities, shaped like ``samples`` without its last axis.
        """
        points = jnp.asarray(samples, dtype=jnp.float64)
        full = _with_fixed(points, self.free, self.fixed)
        values = self.priors.logpdf(full.reshape(-1, full.shape[-1]))
        return jnp.reshape(jnp.asarray(values), points.shape[:-1])

    def rvs(
        self, size: int | tuple[int, ...], *, key: PRNGKeyArray
    ) -> Float[Array, "*size param"]:
        """Draw sampling coordinates of the free parameters from the priors.

        Args:
            size: Shape of the draws without the parameter axis.
            key: JAX PRNG key.

        Returns:
            The draws, shaped ``size + (D,)``.
        """
        draws = jnp.asarray(self.priors.rvs(size=size, rng_key=key))
        return _free_columns(draws, self.free, self.fixed)


def _import_eryn(module_name: str) -> ModuleType:
    """Import an eryn module, which the optional eryn extra installs."""
    try:
        return importlib.import_module(module_name)
    except ImportError as error:
        raise MissingDependencyError(
            f"cannot import {module_name} ({error}); {_INSTALL_HINT}"
        ) from error


def _describe(priors: object) -> tuple[_Axis, ...]:
    """Describe every parameter of a prior container, in the order of its vector."""
    eryn_priors = _import_eryn("eryn.priors")
    if not isinstance(priors, eryn_priors.ProbDistContainer):
        raise TypeError(
            f"priors must be an eryn ProbDistContainer, got {type(priors).__name__}"
        )
    if priors.has_strings:
        names = [str(key) for key in priors.key_order]
    else:
        names = [f"x{position}" for position in range(priors.ndim)]
    return tuple(_container_axes(priors, names, list(range(priors.ndim)), eryn_priors))


def _container_axes(
    container: Any, names: list[str], positions: list[int], eryn_priors: ModuleType
) -> list[_Axis]:
    """Describe the parameters of a container, named and placed by the caller."""
    if container.ndim != len(names):
        raise ValueError(
            f"a nested container spans {container.ndim} parameters, but its key"
            f" names {len(names)}: {', '.join(map(repr, names))}"
        )
    found: dict[int, _Axis] = {}
    for local_positions, prior in container.priors:
        local = [int(position) for position in local_positions]
        block = _prior_axes(
            prior,
            [names[position] for position in local],
            [positions[position] for position in local],
            eryn_priors,
        )
        for position, axis in zip(local, block, strict=True):
            if position in found:
                raise ValueError(
                    f"parameter {names[position]!r} has more than one prior"
                )
            found[position] = axis
    return [found[position] for position in range(container.ndim)]


def _prior_axes(
    prior: Any, names: list[str], positions: list[int], eryn_priors: ModuleType
) -> list[_Axis]:
    """Describe the parameters that one entry of a container covers."""
    prior_name = type(prior).__name__
    discrete = isinstance(
        prior,
        (
            eryn_priors.DiscreteUniform,
            eryn_priors.CategoricalDistribution,
            eryn_priors.PoissonDistribution,
        ),
    ) or hasattr(prior, "logpmf")
    if isinstance(prior, eryn_priors.ProbDistContainer):
        return _container_axes(prior, names, positions, eryn_priors)
    if isinstance(prior, eryn_priors.JointPrior):
        if prior.num_vars != len(names):
            raise ValueError(
                f"the {prior_name} under {', '.join(map(repr, names))} spans"
                f" {prior.num_vars} parameters"
            )
        transformed = prior.forward_transform_nd is not None
        return [
            _Axis(
                name=name,
                position=position,
                prior=prior_name,
                label=label,
                unit=unit,
                bounds=_REAL_BOUNDS,
                periodic=boundary == "periodic",
                fixed=None,
                discrete=discrete,
                transformed=transformed,
                physical_name=physical_name,
                polar_form=None,
            )
            for name, position, label, unit, boundary, physical_name in zip(
                names,
                positions,
                prior.latex_label,
                prior.unit,
                prior.boundaries,
                prior.names_phys,
                strict=True,
            )
        ]
    if len(names) > 1:
        # * One label, unit and pair of bounds cannot describe several parameters.
        transformed = getattr(prior, "forward_transform", None) is not None
        return [
            _Axis(
                name=name,
                position=position,
                prior=prior_name,
                label=name,
                unit=None,
                bounds=_REAL_BOUNDS,
                periodic=False,
                fixed=None,
                discrete=discrete,
                transformed=transformed,
                physical_name=name,
                polar_form=None,
            )
            for name, position in zip(names, positions, strict=True)
        ]
    return [_single_axis(prior, names[0], positions[0], discrete, eryn_priors)]


def _single_axis(
    prior: Any, name: str, position: int, discrete: bool, eryn_priors: ModuleType
) -> _Axis:
    """Describe the parameter of a one-parameter prior."""
    label = getattr(prior, "latex_label", None)
    # * eryn labels a prior without a name "None", the text of its missing name.
    if not isinstance(label, str) or (
        getattr(prior, "name", None) is None and label == str(None)
    ):
        label = name
    physical_name = getattr(prior, "name_phys", None)
    if isinstance(prior, eryn_priors.SineUniform):
        polar_form: PolarForm | None = PolarForm.SIN_LATITUDE
    elif isinstance(prior, eryn_priors.CosineUniform):
        polar_form = PolarForm.COS_COLATITUDE
    else:
        polar_form = None
    return _Axis(
        name=name,
        position=position,
        prior=type(prior).__name__,
        label=label,
        unit=getattr(prior, "unit", None),
        bounds=(
            _bound(getattr(prior, "minimum", None), -math.inf),
            _bound(getattr(prior, "maximum", None), math.inf),
        ),
        periodic=getattr(prior, "boundary", None) == "periodic",
        fixed=(
            float(prior.peak) if isinstance(prior, eryn_priors.DeltaFunction) else None
        ),
        discrete=discrete,
        transformed=getattr(prior, "forward_transform", None) is not None,
        physical_name=(
            physical_name if isinstance(physical_name, str) and physical_name else name
        ),
        polar_form=polar_form,
    )


def _bound(value: Any, default: float) -> float:
    """Return a prior bound as a float, or the default when the prior has none."""
    return default if value is None else float(value)


def _check_axes(axes: Sequence[_Axis]) -> None:
    """Raise for discrete priors, and when every parameter is fixed."""
    for axis in axes:
        if axis.discrete:
            raise ValueError(
                f"parameter {axis.name!r} has the discrete prior {axis.prior};"
                " v1 compares continuous parameters only"
            )
    if all(axis.fixed is not None for axis in axes):
        raise ValueError("every parameter of the priors is fixed by a DeltaFunction")


def _periods(
    axes: Sequence[_Axis], periodic: Mapping[Any, float] | None
) -> dict[int, float]:
    """Return the periods of the sampler dict, keyed by container position."""
    if periodic is None:
        return {}
    position_of = {axis.name: axis.position for axis in axes}
    periods: dict[int, float] = {}
    for key, period in periodic.items():
        if isinstance(key, str):
            if key not in position_of:
                raise ValueError(
                    f"periodic names {key!r}, which is not a parameter of the priors"
                )
            position = position_of[key]
        elif 0 <= key < len(axes):
            position = key
        else:
            raise ValueError(
                f"periodic names position {key}, but the priors have {len(axes)}"
                " parameters"
            )
        value = float(period)
        if not (math.isfinite(value) and value > 0):
            raise ValueError(
                f"the period of {axes[position].name!r} must be positive and finite,"
                f" got {period!r}"
            )
        periods[position] = value
    return periods


def _sampling_parameter(axis: _Axis, period: float | None) -> Parameter:
    """Return the parameter of a sampling coordinate."""
    lower, upper = axis.bounds
    if not (axis.periodic or period is not None):
        return Parameter(
            name=axis.name,
            label=axis.label,
            unit=axis.unit,
            topology=_open_topology(axis.bounds),
            bounds=axis.bounds,
        )
    if math.isfinite(lower) and math.isfinite(upper):
        if period is not None and not math.isclose(
            period, upper - lower, rel_tol=PERIOD_RELATIVE_TOLERANCE
        ):
            raise ValueError(
                f"parameter {axis.name!r} has the period {period!r}, but its prior"
                f" spans [{lower!r}, {upper!r}]; the period must equal that width"
            )
        bounds = axis.bounds
    elif period is not None:
        # * eryn wraps a periodic coordinate as x mod P, into [0, P).
        bounds = (0.0, period)
    else:
        raise ValueError(
            f"parameter {axis.name!r} is periodic, but its prior has no finite"
            " bounds to give the period; give it in periodic="
        )
    return Parameter(
        name=axis.name,
        label=axis.label,
        unit=axis.unit,
        topology=Topology.CIRCLE,
        bounds=bounds,
    )


def _open_topology(bounds: tuple[float, ...]) -> Topology:
    """Return the real line for two infinite bounds, and an interval otherwise."""
    if any(math.isfinite(bound) for bound in bounds):
        return Topology.INTERVAL
    return Topology.REAL


def _check_sphere(sphere: Sphere, by_name: Mapping[str, _Axis]) -> None:
    """Check that the priors of a sphere's axes fit it."""
    for name in (sphere.azimuth, sphere.polar):
        axis = by_name.get(name)
        if axis is not None and axis.fixed is not None:
            raise ValueError(
                f"sphere axis {name!r} has a DeltaFunction prior, so the space"
                " leaves it out"
            )
    polar = by_name.get(sphere.polar)
    if (
        polar is not None
        and polar.polar_form is not None
        and sphere.polar_form is not polar.polar_form
    ):
        raise ValueError(
            f"the polar axis {sphere.polar!r} has a {polar.prior} prior, which"
            f" samples {polar.polar_form}, but the sphere gives {sphere.polar_form}"
        )


def _physical_sphere(
    sphere: Sphere,
    names: Mapping[str, str],
    axes: Sequence[_Axis],
    space: ParamSpace,
) -> Sphere:
    """Return a sphere with its physical names and polar form."""
    polar = axes[space.index(sphere.polar)]
    polar_form = sphere.polar_form
    if polar.transformed:
        if polar_form is not polar.polar_form:
            raise ValueError(
                f"the polar axis {sphere.polar!r} has a {polar.prior} prior, which"
                f" samples {polar.polar_form}, but the sphere gives {polar_form}"
            )
        polar_form = _PHYSICAL_POLAR_FORMS[polar_form]
    return Sphere(
        azimuth=names.get(sphere.azimuth, sphere.azimuth),
        polar=names.get(sphere.polar, sphere.polar),
        polar_form=polar_form,
    )


def _prepare(
    priors: object,
) -> tuple[Any, tuple[int, ...], tuple[tuple[int, float], ...], tuple[_Axis, ...]]:
    """Return a container's JAX copy, free and fixed positions, and free axes."""
    axes = _describe(priors)
    _check_axes(axes)
    to_backend = getattr(priors, "to_backend", None)
    if to_backend is None:
        raise MissingDependencyError(
            "this eryn has no JAX backend for its priors (ProbDistContainer has no"
            f" to_backend); {_INSTALL_HINT}"
        )
    free_axes = tuple(axis for axis in axes if axis.fixed is None)
    fixed = tuple(
        (axis.position, axis.fixed) for axis in axes if axis.fixed is not None
    )
    return (
        to_backend("jax"),
        tuple(axis.position for axis in free_axes),
        fixed,
        free_axes,
    )


def _free_positions(priors: object) -> tuple[int, ...]:
    """Return the container positions of the parameters that no DeltaFunction fixes.

    Discrete priors pass, because chains may hold them; spaces reject them.
    """
    return tuple(axis.position for axis in _describe(priors) if axis.fixed is None)


def _with_fixed(
    points: Float[Array, "*leading param"],
    free: tuple[int, ...],
    fixed: tuple[tuple[int, float], ...],
) -> Float[Array, "*leading _"]:
    """Return eryn's full parameter vectors, the points with the fixed values."""
    if points.shape[-1] != len(free):
        raise ValueError(
            f"points hold {points.shape[-1]} parameters, but the priors have"
            f" {len(free)} free parameters"
        )
    if not fixed:
        return points
    column_of = {position: points[..., column] for column, position in enumerate(free)}
    for position, value in fixed:
        column_of[position] = jnp.full(points.shape[:-1], value, dtype=points.dtype)
    # * One stack instead of in-place writes, which JAX arrays do not allow.
    return jnp.stack([column_of[position] for position in range(len(column_of))], -1)


def _free_columns(
    full: Float[Array, "*leading _"],
    free: tuple[int, ...],
    fixed: tuple[tuple[int, float], ...],
) -> Float[Array, "*leading param"]:
    """Return the columns of the free parameters from eryn's full vectors."""
    full = jnp.asarray(full)
    if not fixed:
        return full
    return full[..., np.asarray(free)]
