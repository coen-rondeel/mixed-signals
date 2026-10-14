"""``Transform``: forward map, inverse map, log-determinant and full Jacobian.

A transform is a bijection y = f(x) between the parameter vectors of one
problem. Its methods take points shaped (..., D), with any leading axes, and
run under ``jax.jit``, ``jax.vmap`` and ``jax.grad``. A batch of B problems
with one transform each uses ``jax.vmap`` over the transform and the points.

Jacobians
    :meth:`Transform.log_abs_det_jacobian` gives ln|det ∂f/∂x| at x, which
    density metrics and reweighting need, and :meth:`Transform.jacobian` the
    full matrix ∂f/∂x, from ``jax.jacfwd`` unless a transform supplies its own.
Densities
    A density of the original coordinates, divided by the Jacobian
    determinant, is the density of the transformed ones (Papamakarios et al.
    2021), so a log density loses ln|det ∂f/∂x|. Likelihood values and
    importance weights, which are ratios of densities in one space, do not
    change.
Fisher matrices
    The information form dθᵀ Γ dθ does not change under a change of
    parameters (Jeffreys 1946), so a Fisher matrix maps as Γ_y = Jᵀ Γ_x J,
    with J = ∂x/∂y the Jacobian of the inverse map at y.
Composition
    :class:`Compose` applies its transforms from left to right, and its
    description joins theirs with ``" | "`` in the order in which they act.
    Its log-determinant is the sum of those of its members, each at its own
    input (Papamakarios et al. 2021).

References:
    :cite:t:`papamakarios2021normalizing`; :cite:t:`jeffreys1946invariant`.
"""

import abc
from collections.abc import Sequence
from typing import Final

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from mixed_signals.core.samples import SampleSet
from mixed_signals.core.space import ParamSpace

COMPOSITION_SEPARATOR: Final = " | "
"""Text between the descriptions of composed transforms, in the order they act."""

_DENSITY_FIELDS: Final = ("log_prior", "log_q")
_JACOBIAN_SIGNATURE: Final = "(param)->(param,param)"


class Transform(eqx.Module):
    """A bijection y = f(x) between the parameter vectors of one problem.

    Subclasses define :meth:`forward`, :meth:`inverse`,
    :meth:`log_abs_det_jacobian` and :attr:`description`. They may supply an
    exact :meth:`jacobian` and define :meth:`map_space`.

    References:
        :cite:t:`papamakarios2021normalizing`; :cite:t:`jeffreys1946invariant`.
    """

    @abc.abstractmethod
    def forward(
        self, points: Float[Array, "*leading param"]
    ) -> Float[Array, "*leading param"]:
        """Return y = f(x) at points x."""

    @abc.abstractmethod
    def inverse(
        self, points: Float[Array, "*leading param"]
    ) -> Float[Array, "*leading param"]:
        """Return x = f⁻¹(y) at points y."""

    @abc.abstractmethod
    def log_abs_det_jacobian(
        self, points: Float[Array, "*leading param"]
    ) -> Float[Array, " *leading"]:
        """Return ln|det ∂f/∂x| at points x."""

    @property
    @abc.abstractmethod
    def description(self) -> str:
        """Short name of the transform for the ``space`` column of result rows."""

    def jacobian(
        self, points: Float[Array, "*leading param"]
    ) -> Float[Array, "*leading param param"]:
        """Return the Jacobian ∂f/∂x at points x.

        Row i holds the derivatives of yᵢ, and column j those with respect to
        xⱼ. The default differentiates :meth:`forward` with ``jax.jacfwd``.

        Args:
            points: Points x, shaped (..., D).

        Returns:
            The Jacobians, shaped (..., D, D).
        """
        jacobian: Array = jnp.vectorize(
            jax.jacfwd(self.forward), signature=_JACOBIAN_SIGNATURE
        )(points)
        return jacobian

    def map_fisher(
        self,
        fisher: Float[Array, "*leading param param"],
        points: Float[Array, "*leading param"],
    ) -> Float[Array, "*leading param param"]:
        """Return Fisher matrices in the transformed coordinates.

        The information form dθᵀ Γ dθ is invariant under a change of
        parameters (Jeffreys 1946), so Γ_y = Jᵀ Γ_x J, with J = ∂x/∂y the
        Jacobian of :meth:`inverse` at y = f(x).

        Args:
            fisher: Fisher matrices Γ_x of the original coordinates, shaped
                (..., D, D).
            points: Points x at which the matrices hold, shaped (..., D).

        Returns:
            The Fisher matrices Γ_y, shaped (..., D, D).

        References:
            :cite:t:`jeffreys1946invariant`.
        """
        inverse_jacobian = jnp.vectorize(
            jax.jacfwd(self.inverse), signature=_JACOBIAN_SIGNATURE
        )(self.forward(points))
        return jnp.einsum(
            "...ki,...kl,...lj->...ij", inverse_jacobian, fisher, inverse_jacobian
        )

    def map_space(self, space: ParamSpace) -> ParamSpace:
        """Return the parameter space of the transformed coordinates.

        Args:
            space: Space of the original coordinates.

        Returns:
            The space of the transformed coordinates.

        Raises:
            NotImplementedError: If the transform does not say how it maps a
                space.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not define how it maps a parameter space"
        )

    def map_sample_set(self, sample_set: SampleSet) -> SampleSet:
        """Return a sample set in the transformed coordinates.

        The samples move to y = f(x), and ``log_prior`` and ``log_q`` lose
        ln|det ∂f/∂x|, because they are densities (Papamakarios et al. 2021).
        ``log_like``, ``log_weights`` and ``n_eff`` stay as they are. The one
        transform applies to every problem of the set; for one transform per
        problem, map each problem with ``jax.vmap``.

        Args:
            sample_set: Samples in the original coordinates.

        Returns:
            The samples in the transformed coordinates.

        References:
            :cite:t:`papamakarios2021normalizing`.
        """
        fields = sample_set.to_arrays("jax")
        original = fields["samples"]
        fields["samples"] = self.forward(original)
        log_determinant = self.log_abs_det_jacobian(original)
        for name in _DENSITY_FIELDS:
            if name in fields:
                fields[name] = fields[name] - log_determinant
        return SampleSet(**fields)


class Compose(Transform):
    """Transforms applied in sequence, the first one first.

    ``Compose((first, second))`` maps x to ``second.forward(first.forward(x))``
    and is described as ``"<first> | <second>"``.

    Args:
        transforms: The transforms, in the order in which they act.

    Attributes:
        transforms: The transforms, in the order in which they act.

    Raises:
        ValueError: If no transform is given.
        TypeError: If a member is not a :class:`Transform`.

    References:
        :cite:t:`papamakarios2021normalizing`.
    """

    transforms: tuple[Transform, ...]

    def __init__(self, transforms: Sequence[object]) -> None:
        """Check and store the transforms."""
        members: list[Transform] = []
        for member in transforms:
            if not isinstance(member, Transform):
                raise TypeError(
                    f"every member must be a Transform, got {type(member).__name__}"
                )
            members.append(member)
        if not members:
            raise ValueError("compose at least one transform")
        self.transforms = tuple(members)

    def forward(
        self, points: Float[Array, "*leading param"]
    ) -> Float[Array, "*leading param"]:
        """Return y = f(x) at points x, applying the members in order."""
        for transform in self.transforms:
            points = transform.forward(points)
        return points

    def inverse(
        self, points: Float[Array, "*leading param"]
    ) -> Float[Array, "*leading param"]:
        """Return x = f⁻¹(y) at points y, inverting the members in reverse order."""
        for transform in reversed(self.transforms):
            points = transform.inverse(points)
        return points

    def log_abs_det_jacobian(
        self, points: Float[Array, "*leading param"]
    ) -> Float[Array, " *leading"]:
        """Return ln|det ∂f/∂x| at points x, the sum over the members."""
        total = jnp.zeros(points.shape[:-1])
        for transform in self.transforms:
            total = total + transform.log_abs_det_jacobian(points)
            points = transform.forward(points)
        return total

    @property
    def description(self) -> str:
        """The members' descriptions, joined in the order in which they act."""
        return COMPOSITION_SEPARATOR.join(
            transform.description for transform in self.transforms
        )

    def jacobian(
        self, points: Float[Array, "*leading param"]
    ) -> Float[Array, "*leading param param"]:
        """Return ∂f/∂x at points x, the product of the members' Jacobians."""
        size = points.shape[-1]
        product = jnp.broadcast_to(jnp.eye(size), (*points.shape, size))
        for transform in self.transforms:
            product = transform.jacobian(points) @ product
            points = transform.forward(points)
        return product

    def map_fisher(
        self,
        fisher: Float[Array, "*leading param param"],
        points: Float[Array, "*leading param"],
    ) -> Float[Array, "*leading param param"]:
        """Return Fisher matrices in the transformed coordinates, member by member."""
        for transform in self.transforms:
            fisher = transform.map_fisher(fisher, points)
            points = transform.forward(points)
        return fisher

    def map_space(self, space: ParamSpace) -> ParamSpace:
        """Return the parameter space after every member, mapped in order."""
        for transform in self.transforms:
            space = transform.map_space(space)
        return space
