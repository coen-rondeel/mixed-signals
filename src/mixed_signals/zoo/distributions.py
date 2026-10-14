"""Distributions of the validation zoo, as equinox modules.

Every distribution draws samples from an explicit JAX key and evaluates its log
density in float64, inside or outside :func:`jax.jit`. A density is taken with
respect to the Lebesgue measure of the coordinates that the samples come in, so
on a sphere in latitude it includes the factor cos β of the area element.
Build distributions outside ``jax.jit``: the constructors check their
parameters on the host, as the sample containers do.

:class:`Gaussian`
    The multivariate normal N(μ, Σ).
:class:`TwistedGaussian`
    The twisted Gaussian B(b, v) of Haario, Saksman and Tamminen (1999): a
    banana with non-linear dependence between its first two coordinates.
:class:`GaussianMixture`
    A finite mixture of multivariate normals.
:class:`TruncatedGaussian`
    Independent normals, each truncated to an interval, drawn by inversion of
    the distribution function (Devroye 1986) in the tail where it keeps its
    precision.
:class:`WrappedGaussian`
    A multivariate normal with one coordinate wrapped onto a circle, the
    wrapped normal of Mardia and Jupp (2000) on that axis.
:class:`VonMisesFisher`
    The von Mises-Fisher distribution on the 2-sphere (Fisher 1953), in
    azimuth and one of the four polar forms of
    :class:`~mixed_signals.core.topology.PolarForm`, drawn by inversion of
    the exponential density of its cosine (Fisher, Lewis and Embleton 1987),
    with optional independent standard normal axes.

References:
    :cite:t:`haario1999adaptive`; :cite:t:`devroye1986non`;
    :cite:t:`mardia2000directional`; :cite:t:`fisher1953dispersion`;
    :cite:t:`fisher1987statistical`.
"""

import abc
import math
import sys
from typing import Final

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.linalg import solve_triangular
from jax.scipy.special import logsumexp, ndtr, ndtri
from jax.typing import ArrayLike
from jaxtyping import Array, Float, PRNGKeyArray

from mixed_signals.core.topology import (
    AZIMUTH_PERIOD,
    PolarForm,
    sphere_coordinates,
    unit_vectors,
    wrap,
)
from mixed_signals.prng import derive_key
from mixed_signals.zoo import closed_forms
from mixed_signals.zoo._checks import (
    check_count,
    check_finite,
    check_positive,
    cholesky,
    matrix,
    vector,
)

WRAPPING_REACH: Final = 12.0
"""Standard deviations of the unwrapped normal that the winding sum must cover."""

_HALF_LOG_TWO_PI: Final = 0.5 * math.log(2 * math.pi)
_SMALLEST_UNIFORM: Final = sys.float_info.min


class ZooDistribution(eqx.Module):
    """A distribution with a JAX sampler and a JAX log density."""

    @property
    @abc.abstractmethod
    def param_count(self) -> int:
        """Number of parameters, D, of each sample."""

    @abc.abstractmethod
    def sample(
        self, key: PRNGKeyArray, sample_shape: tuple[int, ...] = ()
    ) -> Float[Array, "*leading param"]:
        """Draw independent samples.

        Args:
            key: JAX PRNG key.
            sample_shape: Leading shape of the draws.

        Returns:
            Samples shaped ``sample_shape + (D,)``.
        """

    @abc.abstractmethod
    def log_prob(
        self, points: Float[ArrayLike, "*leading param"]
    ) -> Float[Array, " *leading"]:
        """Return the log density at each point.

        Args:
            points: Points shaped (..., D).

        Returns:
            The log densities, shaped (...); −inf outside the support.
        """


class Gaussian(ZooDistribution):
    """The multivariate normal N(μ, Σ).

    Args:
        mean: Mean μ, shaped (D,).
        covariance: Covariance Σ, symmetric positive definite, shaped (D, D).

    Attributes:
        mean: Mean μ, shaped (D,).
        covariance: Covariance Σ, shaped (D, D).
        cholesky: Lower Cholesky factor L of Σ = LLᵀ.
        inverse_cholesky: L⁻¹, which whitens a point as L⁻¹(x − μ).

    Raises:
        ValueError: If the shapes do not fit, or Σ is not symmetric positive
            definite.
    """

    mean: Float[Array, " param"]
    covariance: Float[Array, "param param"]
    cholesky: Float[Array, "param param"]
    inverse_cholesky: Float[Array, "param param"]

    def __init__(self, mean: object, covariance: object) -> None:
        """Check the parameters and factor the covariance."""
        mean_array = jnp.asarray(vector("mean", mean))
        param_count = mean_array.shape[0]
        covariance_array = jnp.asarray(matrix("covariance", covariance, param_count))
        factor = cholesky(covariance_array, "covariance")
        self.mean = mean_array
        self.covariance = covariance_array
        self.cholesky = factor
        self.inverse_cholesky = solve_triangular(
            factor, jnp.eye(param_count), lower=True
        )

    @property
    def param_count(self) -> int:
        """Number of parameters, D."""
        return int(self.mean.shape[0])

    def sample(
        self, key: PRNGKeyArray, sample_shape: tuple[int, ...] = ()
    ) -> Float[Array, "*leading param"]:
        """Draw μ + Lz with z standard normal."""
        standard = jax.random.normal(key, (*sample_shape, self.param_count))
        return self.mean + standard @ self.cholesky.T

    def log_prob(
        self, points: Float[ArrayLike, "*leading param"]
    ) -> Float[Array, " *leading"]:
        """Return ln N(x; μ, Σ)."""
        whitened = (jnp.asarray(points) - self.mean) @ self.inverse_cholesky.T
        return (
            -0.5 * jnp.sum(whitened**2, axis=-1)
            - jnp.sum(jnp.log(jnp.diag(self.cholesky)))
            - self.param_count * _HALF_LOG_TWO_PI
        )


class TwistedGaussian(ZooDistribution):
    """The twisted Gaussian B(b, v), a banana-shaped distribution.

    It maps x ~ N(0, diag(v, 1, …, 1)) to y with y₂ = x₂ + b(x₁² − v) and
    every other coordinate unchanged (Haario, Saksman and Tamminen 1999). The
    map has a unit Jacobian, so ln B(y) is the log density of the base
    Gaussian at the untwisted point. Its mean is zero and its covariance
    diag(v, 1 + 2b²v², 1, …, 1), with no linear correlation between y₁ and
    y₂. Sejdinovic et al. (2014) use v = 100 with b = 0.03 (moderately
    twisted) and b = 0.1 (strongly twisted) in 8 dimensions.

    Args:
        twist: The twist b.
        variance: The variance v of the first coordinate, positive.
        param_count: Number of parameters D, at least 2.

    Attributes:
        twist: The twist b.
        scales: Standard deviations of the base Gaussian, (√v, 1, …, 1).

    Raises:
        ValueError: If a parameter is out of range.

    References:
        :cite:t:`haario1999adaptive`; :cite:t:`sejdinovic2014kernel`.
    """

    twist: Float[Array, ""]
    scales: Float[Array, " param"]

    def __init__(self, twist: float, variance: float, param_count: int) -> None:
        """Check the parameters and store the base scales."""
        check_finite("the twist", twist)
        check_positive("the variance", variance)
        check_count("param_count", param_count, minimum=2)
        self.twist = jnp.asarray(float(twist))
        self.scales = jnp.ones(param_count).at[0].set(math.sqrt(variance))

    @property
    def param_count(self) -> int:
        """Number of parameters, D."""
        return int(self.scales.shape[0])

    def sample(
        self, key: PRNGKeyArray, sample_shape: tuple[int, ...] = ()
    ) -> Float[Array, "*leading param"]:
        """Draw base Gaussian points and twist them."""
        base = self.scales * jax.random.normal(key, (*sample_shape, self.param_count))
        twist = self.twist * (base[..., 0] ** 2 - self.scales[0] ** 2)
        return base.at[..., 1].add(twist)

    def log_prob(
        self, points: Float[ArrayLike, "*leading param"]
    ) -> Float[Array, " *leading"]:
        """Return the base log density at the untwisted point."""
        twisted = jnp.asarray(points)
        base = twisted.at[..., 1].add(
            -self.twist * (twisted[..., 0] ** 2 - self.scales[0] ** 2)
        )
        standard = base / self.scales
        return (
            -0.5 * jnp.sum(standard**2, axis=-1)
            - jnp.sum(jnp.log(self.scales))
            - self.param_count * _HALF_LOG_TWO_PI
        )


class GaussianMixture(ZooDistribution):
    """A finite mixture of multivariate normals, Σ_c w_c N(μ_c, Σ_c).

    Args:
        weights: Component weights w_c, positive, shaped (C,); they are
            normalised to sum to one.
        means: Component means, shaped (C, D).
        covariances: Component covariances, each symmetric positive definite,
            shaped (C, D, D).

    Attributes:
        log_weights: Normalised log weights, shaped (C,).
        means: Component means, shaped (C, D).
        choleskys: Lower Cholesky factors of the covariances, shaped (C, D, D).
        inverse_choleskys: Their inverses, which whiten each component.

    Raises:
        ValueError: If the shapes do not fit, a weight is not positive, or a
            covariance is not symmetric positive definite.
    """

    log_weights: Float[Array, " component"]
    means: Float[Array, "component param"]
    choleskys: Float[Array, "component param param"]
    inverse_choleskys: Float[Array, "component param param"]

    def __init__(self, weights: object, means: object, covariances: object) -> None:
        """Check the parameters and factor every covariance."""
        weights_array = vector("weights", weights)
        if np.any(weights_array <= 0):
            raise ValueError("the mixture weights must be positive")
        means_array = np.asarray(means, dtype=np.float64)
        component_count = len(weights_array)
        if means_array.ndim != 2 or means_array.shape[0] != component_count:
            raise ValueError(
                f"means must be shaped ({component_count}, D), got shape"
                f" {means_array.shape}"
            )
        param_count = means_array.shape[1]
        covariances_array = np.asarray(covariances, dtype=np.float64)
        if covariances_array.shape != (component_count, param_count, param_count):
            raise ValueError(
                f"covariances must be shaped ({component_count}, {param_count},"
                f" {param_count}), got shape {covariances_array.shape}"
            )
        factors = jnp.stack(
            [
                cholesky(
                    jnp.asarray(matrix(f"covariance {index}", covariance, param_count)),
                    f"covariance {index}",
                )
                for index, covariance in enumerate(covariances_array)
            ]
        )
        means_checked = [
            vector(f"mean {index}", mean, param_count)
            for index, mean in enumerate(means_array)
        ]
        weights_jax = jnp.asarray(weights_array)
        self.log_weights = jnp.log(weights_jax / jnp.sum(weights_jax))
        self.means = jnp.asarray(np.stack(means_checked))
        self.choleskys = factors
        self.inverse_choleskys = jnp.stack(
            [
                solve_triangular(factor, jnp.eye(param_count), lower=True)
                for factor in factors
            ]
        )

    @property
    def param_count(self) -> int:
        """Number of parameters, D."""
        return int(self.means.shape[1])

    def sample(
        self, key: PRNGKeyArray, sample_shape: tuple[int, ...] = ()
    ) -> Float[Array, "*leading param"]:
        """Draw a component for each sample, then a point from that component."""
        component = jax.random.categorical(
            derive_key(key, "component"), self.log_weights, shape=sample_shape
        )
        standard = jax.random.normal(
            derive_key(key, "noise"), (*sample_shape, self.param_count)
        )
        return self.means[component] + jnp.einsum(
            "...ij,...j->...i", self.choleskys[component], standard
        )

    def log_prob(
        self, points: Float[ArrayLike, "*leading param"]
    ) -> Float[Array, " *leading"]:
        """Return ln Σ_c w_c N(x; μ_c, Σ_c), summed in log space."""
        offsets = jnp.asarray(points)[..., None, :] - self.means
        whitened = jnp.einsum("cij,...cj->...ci", self.inverse_choleskys, offsets)
        log_dets = jnp.sum(
            jnp.log(jnp.diagonal(self.choleskys, axis1=-2, axis2=-1)), axis=-1
        )
        component_log_probs = (
            -0.5 * jnp.sum(whitened**2, axis=-1)
            - log_dets
            - self.param_count * _HALF_LOG_TWO_PI
        )
        return logsumexp(self.log_weights + component_log_probs, axis=-1)


class TruncatedGaussian(ZooDistribution):
    """Independent normals, each truncated to an interval.

    Coordinate i follows N(μ_i, σ_i²) restricted to [lower_i, upper_i], where
    either bound may be infinite; a coordinate with two infinite bounds is an
    ordinary normal. Draws invert the distribution function (Devroye 1986).
    Where both bounds lie above the mean, the interval is mirrored into the
    lower tail first, where Φ keeps its relative precision, so that intervals
    far in a tail still give exact draws.

    Args:
        mean: Means μ before truncation, shaped (D,).
        std: Standard deviations σ before truncation, positive, shaped (D,).
        lower: Lower bounds, which may be −inf, shaped (D,).
        upper: Upper bounds, above the lower ones, which may be inf.

    Attributes:
        mean: Means before truncation.
        std: Standard deviations before truncation.
        lower: Lower bounds.
        upper: Upper bounds.
        log_mass: Log of the normal mass inside each interval.

    Raises:
        ValueError: If the shapes do not fit or a parameter is out of range.

    References:
        :cite:t:`devroye1986non`; :cite:t:`barr1999mean`.
    """

    mean: Float[Array, " param"]
    std: Float[Array, " param"]
    lower: Float[Array, " param"]
    upper: Float[Array, " param"]
    log_mass: Float[Array, " param"]

    def __init__(self, mean: object, std: object, lower: object, upper: object) -> None:
        """Check the parameters and compute the mass of each interval."""
        mean_array = vector("mean", mean)
        param_count = len(mean_array)
        std_array = vector("std", std, param_count)
        if np.any(std_array <= 0):
            raise ValueError("std must be positive")
        bounds = []
        for name, value in (("lower", lower), ("upper", upper)):
            bound = np.asarray(value, dtype=np.float64)
            if bound.shape != (param_count,) or np.any(np.isnan(bound)):
                raise ValueError(
                    f"{name} must be shaped ({param_count},) without NaN, got shape"
                    f" {bound.shape}"
                )
            bounds.append(bound)
        lower_array, upper_array = bounds
        if np.any(lower_array >= upper_array):
            raise ValueError("every lower bound must lie below its upper bound")
        self.mean = jnp.asarray(mean_array)
        self.std = jnp.asarray(std_array)
        self.lower = jnp.asarray(lower_array)
        self.upper = jnp.asarray(upper_array)
        self.log_mass = closed_forms.log_normal_mass(
            (self.lower - self.mean) / self.std, (self.upper - self.mean) / self.std
        )

    @property
    def param_count(self) -> int:
        """Number of parameters, D."""
        return int(self.mean.shape[0])

    def sample(
        self, key: PRNGKeyArray, sample_shape: tuple[int, ...] = ()
    ) -> Float[Array, "*leading param"]:
        """Draw by inversion of each truncated distribution function."""
        uniform = jax.random.uniform(
            key, (*sample_shape, self.param_count), minval=_SMALLEST_UNIFORM
        )
        low = (self.lower - self.mean) / self.std
        high = (self.upper - self.mean) / self.std
        # * Mirroring keeps the interval in the lower tail, where Φ is precise.
        mirror = low > 0
        mirrored_low = jnp.where(mirror, -high, low)
        mirrored_high = jnp.where(mirror, -low, high)
        cdf_low = ndtr(mirrored_low)
        mass = ndtr(mirrored_high) - cdf_low
        lower_tail = cdf_low + uniform * mass
        upper_tail = ndtr(-mirrored_high) + (1 - uniform) * mass
        # * Each probability is inverted on the side where it is below one half.
        standard = jnp.where(
            lower_tail < 0.5, ndtri(lower_tail), -ndtri(jnp.minimum(upper_tail, 0.5))
        )
        standard = jnp.clip(standard, mirrored_low, mirrored_high)
        standard = jnp.where(mirror, -standard, standard)
        return jnp.clip(self.mean + self.std * standard, self.lower, self.upper)

    def log_prob(
        self, points: Float[ArrayLike, "*leading param"]
    ) -> Float[Array, " *leading"]:
        """Return the summed log densities, −inf outside the intervals."""
        values = jnp.asarray(points)
        standard = (values - self.mean) / self.std
        per_axis = (
            -0.5 * standard**2 - _HALF_LOG_TWO_PI - jnp.log(self.std) - self.log_mass
        )
        inside = jnp.all((values >= self.lower) & (values <= self.upper), axis=-1)
        return jnp.where(inside, jnp.sum(per_axis, axis=-1), -jnp.inf)


class WrappedGaussian(ZooDistribution):
    """A multivariate normal with one coordinate wrapped onto a circle.

    The circle coordinate is reduced to [lower, lower + P) after drawing, so
    on that axis alone the distribution is the wrapped normal of Mardia and
    Jupp (2000). The density sums the unwrapped density over windings k P,
    for every k whose term matters: the sum reaches 12 standard deviations of
    the circle coordinate beyond the circle, where a term is below e^(−72) of
    the largest.

    Args:
        mean: Mean of the unwrapped normal, shaped (D,).
        covariance: Covariance of the unwrapped normal, shaped (D, D).
        circle: Index of the circle coordinate.
        lower: Lower end of the circle coordinate.
        period: Period P of the circle, positive.

    Attributes:
        base: The unwrapped normal.
        lower: Lower end of the circle coordinate.
        period: Period of the circle.
        circle: Index of the circle coordinate.
        windings: Number of windings summed on each side.

    Raises:
        ValueError: If the shapes do not fit or a parameter is out of range.

    References:
        :cite:t:`mardia2000directional`.
    """

    base: Gaussian
    lower: Float[Array, ""]
    period: Float[Array, ""]
    circle: int = eqx.field(static=True)
    windings: int = eqx.field(static=True)

    def __init__(
        self,
        mean: object,
        covariance: object,
        *,
        circle: int = 0,
        lower: float = 0.0,
        period: float = AZIMUTH_PERIOD,
    ) -> None:
        """Check the parameters and size the winding sum."""
        self.base = Gaussian(mean, covariance)
        check_count("circle", circle, minimum=0)
        if circle >= self.base.param_count:
            raise ValueError(
                f"circle {circle} is not an axis of {self.base.param_count} parameters"
            )
        check_finite("lower", lower)
        check_positive("period", period)
        circle_std = math.sqrt(float(self.base.covariance[circle, circle]))
        self.lower = jnp.asarray(float(lower))
        self.period = jnp.asarray(float(period))
        self.circle = circle
        self.windings = math.ceil(WRAPPING_REACH * circle_std / period) + 1

    @property
    def param_count(self) -> int:
        """Number of parameters, D."""
        return self.base.param_count

    def sample(
        self, key: PRNGKeyArray, sample_shape: tuple[int, ...] = ()
    ) -> Float[Array, "*leading param"]:
        """Draw from the unwrapped normal and wrap the circle coordinate."""
        draws = self.base.sample(key, sample_shape)
        wrapped = wrap(draws[..., self.circle], self.lower, self.period)
        return draws.at[..., self.circle].set(wrapped)

    def log_prob(
        self, points: Float[ArrayLike, "*leading param"]
    ) -> Float[Array, " *leading"]:
        """Return the log of the unwrapped density summed over windings."""
        values = jnp.asarray(points)
        steps = jnp.arange(-self.windings, self.windings + 1) * self.period
        shifts = jnp.zeros((len(steps), self.param_count)).at[:, self.circle].set(steps)
        return logsumexp(self.base.log_prob(values[..., None, :] + shifts), axis=-1)


class VonMisesFisher(ZooDistribution):
    """The von Mises-Fisher distribution on the 2-sphere, in angular coordinates.

    The density on the unit sphere is C(κ) exp(κ μᵀx) with
    C(κ) = κ/(4π sinh κ) (Fisher 1953). Samples come as an azimuth in
    [0, 2π) and a polar coordinate in the given form, followed by
    ``euclidean_count`` independent standard normal coordinates. The log
    density adds the log of the area element of the polar form: ln cos β for
    latitude, ln sin θ for colatitude, and zero for the sine and cosine forms.

    The cosine w = μᵀx has the density κ e^(κw) / (2 sinh κ) on [−1, 1] and
    the direction around μ is uniform, so a draw inverts the distribution
    function of w (Fisher, Lewis and Embleton 1987) as
    1 − w = −ln(e^(−2κ) + u(1 − e^(−2κ)))/κ, which keeps its precision near
    the mean direction.

    Args:
        direction: Mean direction μ in Cartesian coordinates, shaped (3,); it
            is normalised.
        concentration: Concentration κ, positive.
        polar_form: Coordinate of the polar axis.
        euclidean_count: Number of independent standard normal coordinates
            after the sphere.

    Attributes:
        direction: Unit mean direction μ.
        frame: Orthonormal frame whose last column is μ, shaped (3, 3).
        concentration: Concentration κ.
        log_normaliser: ln C(κ).
        polar_form: Coordinate of the polar axis.
        euclidean_count: Number of standard normal coordinates.

    Raises:
        ValueError: If a parameter is out of range.

    References:
        :cite:t:`fisher1953dispersion`; :cite:t:`fisher1987statistical`;
        :cite:t:`mardia2000directional`.
    """

    direction: Float[Array, " 3"]
    frame: Float[Array, "3 3"]
    concentration: Float[Array, ""]
    log_normaliser: Float[Array, ""]
    polar_form: PolarForm = eqx.field(static=True)
    euclidean_count: int = eqx.field(static=True)

    def __init__(
        self,
        direction: object,
        concentration: float,
        *,
        polar_form: PolarForm | str = PolarForm.LATITUDE,
        euclidean_count: int = 0,
    ) -> None:
        """Check the parameters and build the frame around the mean direction."""
        given = jnp.asarray(vector("direction", direction, 3))
        norm = jnp.linalg.norm(given)
        if float(norm) == 0.0:
            raise ValueError("direction must not be the zero vector")
        unit = given / norm
        check_positive("the concentration", concentration)
        check_count("euclidean_count", euclidean_count, minimum=0)
        # * The axis least aligned with μ gives a well-conditioned first tangent.
        helper = jnp.eye(3)[int(jnp.argmin(jnp.abs(unit)))]
        first = helper - (helper @ unit) * unit
        first = first / jnp.linalg.norm(first)
        self.direction = unit
        self.frame = jnp.stack([first, jnp.cross(unit, first), unit], axis=1)
        self.concentration = jnp.asarray(float(concentration))
        self.log_normaliser = closed_forms.von_mises_fisher_log_normaliser(
            float(concentration)
        )
        self.polar_form = PolarForm(polar_form)
        self.euclidean_count = euclidean_count

    @property
    def param_count(self) -> int:
        """Number of parameters: azimuth, polar and the standard normal axes."""
        return 2 + self.euclidean_count

    def sample(
        self, key: PRNGKeyArray, sample_shape: tuple[int, ...] = ()
    ) -> Float[Array, "*leading param"]:
        """Draw unit vectors by inversion, then convert them to coordinates."""
        uniform = jax.random.uniform(
            derive_key(key, "cosine"), sample_shape, minval=_SMALLEST_UNIFORM
        )
        concentration = self.concentration
        gap = (
            -jnp.logaddexp(
                -2 * concentration,
                jnp.log(uniform) + jnp.log(-jnp.expm1(-2 * concentration)),
            )
            / concentration
        )
        sine = jnp.sqrt(jnp.clip(gap * (2 - gap), 0.0, None))
        around = jax.random.uniform(derive_key(key, "angle"), sample_shape)
        angle = AZIMUTH_PERIOD * around
        local = jnp.stack([sine * jnp.cos(angle), sine * jnp.sin(angle), 1 - gap], -1)
        azimuth, polar = sphere_coordinates(local @ self.frame.T, self.polar_form)
        extra = jax.random.normal(
            derive_key(key, "euclidean"), (*sample_shape, self.euclidean_count)
        )
        return jnp.concatenate([azimuth[..., None], polar[..., None], extra], axis=-1)

    def log_prob(
        self, points: Float[ArrayLike, "*leading param"]
    ) -> Float[Array, " *leading"]:
        """Return the log density in the azimuth and polar coordinates."""
        values = jnp.asarray(points)
        unit, log_area, inside = unit_vectors(
            values[..., 0], values[..., 1], self.polar_form
        )
        sphere = self.log_normaliser + self.concentration * (unit @ self.direction)
        extra = values[..., 2:]
        euclidean = jnp.sum(-0.5 * extra**2 - _HALF_LOG_TWO_PI, axis=-1)
        return jnp.where(inside, sphere + log_area + euclidean, -jnp.inf)
