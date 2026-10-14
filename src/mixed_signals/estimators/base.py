"""Shared parts of the estimators: settings, the ``Estimator`` base and input checks.

Every estimator is an equinox module that a ``fit`` classmethod builds from
the samples of one posterior of one problem; ``jax.vmap`` gives one estimator
per problem and posterior, as for
:class:`~mixed_signals.transforms.whiten.Whitening`. Its settings are a frozen
pydantic model, whose JSON fills the ``estimator_settings`` column of the
result table and which plots reuse, so that contours and metrics agree.

Circles
    The Gaussian, the GMM and the bandwidth of the KDE need linear values, so
    each circle is unwrapped about its weighted circular mean: every value is
    replaced by its representative within half a period of that mean
    (Mardia and Jupp 2000). A density on the circle then sums over windings,
    which makes it periodic again. When the mean direction is not resolved,
    the cut opposite the mean may split a mode: the Gaussian and the GMM warn
    when the Rayleigh statistic 2 N_eff R̄² lies below the 95% point of χ²₂,
    its limiting distribution for a uniform circle (Mardia and Jupp 2000).
    Sphere axes follow their own topologies, the azimuth a circle and the
    polar axis an interval, which is approximate near the poles.
Images
    Wrapping and reflection evaluate an unbounded density at images of each
    point: shifts by whole periods on circles, and mirror images at finite
    interval bounds. Images whose distance to the data exceeds
    :data:`IMAGE_REACH` kernel or component widths are left out, because
    their terms lie below e^(−72) of the largest.
Covariances
    :func:`factor_covariance` factors covariances through their correlation
    matrices, so the factor keeps its precision at any parameter scale and
    the singularity check does not depend on units (van der Sluis 1969).
Checks
    ``fit`` checks shapes and types always, and the values of concrete
    inputs: finite samples, log weights without NaN or +∞, and a positive
    weight. Inside ``jax.jit`` and ``jax.vmap`` the values are unknown, so
    nothing warns and invalid inputs give NaN; a traced fit counts its
    images from the widest kernel the samples admit and gives NaN where a
    kernel is wider still.

References:
    :cite:t:`mardia2000directional`; :cite:t:`vandersluis1969condition`.
"""

import abc
import itertools
import math
import os
import warnings
from collections.abc import Callable, Sequence
from typing import ClassVar, Final, NamedTuple, cast

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike
from jaxtyping import Array, Bool, Float
from pydantic import BaseModel, ConfigDict

from mixed_signals.core.samples import (
    _check_real,
    _is_concrete,
    _warn_narrow_floats,
    circular_mean,
    kish_effective_sample_size,
    mean_resultant_length,
    unwrap_about,
)
from mixed_signals.core.space import ParamSpace
from mixed_signals.core.topology import Topology
from mixed_signals.errors import ValidityWarning
from mixed_signals.transforms.whiten import FLOAT64_EPSILON

IMAGE_REACH: Final = 12.0
"""Kernel or component widths that the images of a point must cover.

A Gaussian term beyond this reach lies below e^(−72) of the largest one, far
below the float64 resolution of a sum.
"""

RAYLEIGH_THRESHOLD: Final = -2 * math.log(0.05)
"""The 95% point of χ²₂, −2 ln 0.05 ≈ 5.99.

For n values spread uniformly around a circle, the Rayleigh statistic 2nR̄²
tends to χ²₂ (Mardia and Jupp 2000), so a smaller value leaves the mean
direction unresolved at 95%.

References:
    :cite:t:`mardia2000directional`.
"""

# * Unbiased widths of two or more effective samples exceed biased ones by at most √2.
WIDTH_MARGIN: Final = math.sqrt(2)
# * Kernel evaluations per chunk: 2²² float64 values take 32 MB per buffer.
CHUNK_ENTRIES: Final = 2**22
_PACKAGE_DIRECTORY: Final = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class EstimatorSettings(BaseModel):
    """Settings of an estimator, frozen and written as JSON in result rows.

    Subclasses add the fields of their estimator and set :attr:`name`, the
    value of the ``estimator`` column.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: ClassVar[str]
    """Name of the estimator in the ``estimator`` column of the result table."""

    def to_json(self) -> str:
        """Return the settings as JSON, for the ``estimator_settings`` column."""
        return self.model_dump_json()


class Estimator(eqx.Module):
    """A fitted estimator of the density of one posterior of one problem.

    Subclasses provide a ``fit`` classmethod and :meth:`log_density`.
    """

    settings: eqx.AbstractVar[EstimatorSettings]
    """The settings the estimator was fitted with."""

    @abc.abstractmethod
    def log_density(
        self, points: Float[ArrayLike, "*leading param"]
    ) -> Float[Array, " *leading"]:
        """Return the log density at points, with respect to their coordinates.

        Args:
            points: Points shaped (..., D), in the coordinates of the space.

        Returns:
            The log densities, shaped (...).
        """

    def resolved_settings(self) -> EstimatorSettings:
        """Return the settings with every value resolved, for result rows and plots.

        A setting that a rule chooses from the data, such as a bandwidth,
        becomes its number, so fitting other samples with the resolved
        settings uses the same value.
        """
        return self.settings


def _jit[**Parameters, Result](
    function: Callable[Parameters, Result], *static_names: str
) -> Callable[Parameters, Result]:
    """Compile a kernel with :func:`jax.jit` and keep its signature for type checks."""
    return cast(
        Callable[Parameters, Result],
        jax.jit(function, static_argnames=static_names),
    )


def _warn(message: str) -> None:
    """Raise a validity warning that points at the caller's line."""
    warnings.warn(ValidityWarning(message), skip_file_prefixes=(_PACKAGE_DIRECTORY,))


def _read_samples(
    samples: ArrayLike, log_weights: ArrayLike | None, space: ParamSpace
) -> tuple[Array, Array]:
    """Return the samples (N, D) and log weights (N,) of one posterior as float64.

    Args:
        samples: Samples in the parameter order of ``space``, shaped (N, D).
        log_weights: Log weights shaped (N,); equal weights when ``None``.
        space: Parameter space of the samples.

    Returns:
        The samples and the log weights.

    Raises:
        ValueError: If the shapes do not fit the space, or concrete values are
            invalid.
        TypeError: If an input is not real-valued.

    Warns:
        PrecisionWarning: If an input is narrower than float64.
    """
    size = len(space)
    samples_array = jnp.asarray(samples)
    _check_real("samples", samples_array)
    if samples_array.ndim != 2 or samples_array.shape[1] != size:
        raise ValueError(
            f"samples must be shaped (N, {size}) for a space of {size} parameters,"
            f" got shape {samples_array.shape}"
        )
    if samples_array.shape[0] == 0:
        raise ValueError("samples need at least one sample")
    arrays = {"samples": samples_array}
    if log_weights is not None:
        weights_array = jnp.asarray(log_weights)
        _check_real("log_weights", weights_array)
        if weights_array.shape != samples_array.shape[:1]:
            raise ValueError(
                f"log_weights must be shaped {samples_array.shape[:1]} for samples of"
                f" shape {samples_array.shape}, got shape {weights_array.shape}"
            )
        arrays["log_weights"] = weights_array
    _warn_narrow_floats(arrays)
    samples_array = samples_array.astype(jnp.float64)
    weights_array = (
        arrays["log_weights"].astype(jnp.float64)
        if "log_weights" in arrays
        else jnp.zeros(samples_array.shape[0])
    )
    if _is_concrete(samples_array, weights_array):
        if not bool(jnp.all(jnp.isfinite(samples_array))):
            raise ValueError("samples must be finite")
        if bool(jnp.any(jnp.isnan(weights_array) | (weights_array == jnp.inf))):
            raise ValueError("log_weights must not contain NaN or +inf")
        if not bool(jnp.any(weights_array > -jnp.inf)):
            raise ValueError(
                "at least one sample needs a positive weight; at least one log"
                " weight must exceed -inf"
            )
    return samples_array, weights_array


def _read_points(points: ArrayLike, size: int) -> Array:
    """Return points shaped (..., D) as float64, after checking their last axis.

    Raises:
        ValueError: If the last axis does not hold the D parameters.
        TypeError: If the points are not real-valued.

    Warns:
        PrecisionWarning: If the points are narrower than float64.
    """
    array = jnp.asarray(points)
    _check_real("points", array)
    if array.ndim == 0 or array.shape[-1] != size:
        raise ValueError(
            f"points must be shaped (..., {size}) for a space of {size} parameters,"
            f" got shape {array.shape}"
        )
    _warn_narrow_floats({"points": array})
    return array.astype(jnp.float64)


def _circle_positions(space: ParamSpace) -> tuple[int, ...]:
    """Return the positions of the circle parameters, sphere azimuths included."""
    return tuple(
        position
        for position, parameter in enumerate(space.parameters)
        if parameter.topology is Topology.CIRCLE
    )


def _period(space: ParamSpace, position: int) -> float:
    """Return the period of the circle parameter at a position."""
    lower, upper = space.parameters[position].bounds
    return upper - lower


def _unwrap_circles(
    samples: Float[Array, "sample param"],
    log_weights: Float[Array, " sample"],
    space: ParamSpace,
) -> tuple[Float[Array, "sample param"], Float[Array, " circle"]]:
    """Return the samples with each circle unwrapped about its circular mean.

    Returns:
        The samples with circle values replaced by their representatives
        within half a period of the mean, and the means, shaped (C,).
    """
    unwrapped = samples
    centres = []
    for position in _circle_positions(space):
        lower = space.parameters[position].bounds[0]
        period = _period(space, position)
        centre = circular_mean(samples[:, position], log_weights, period, lower)
        unwrapped = unwrapped.at[:, position].set(
            unwrap_about(samples[:, position], centre, period)
        )
        centres.append(centre)
    return unwrapped, jnp.asarray(centres, dtype=jnp.float64).reshape(-1)


def _warn_about_unresolved_circles(
    samples: Float[Array, "sample param"],
    log_weights: Float[Array, " sample"],
    space: ParamSpace,
) -> None:
    """Warn about circles whose mean direction the Rayleigh test does not resolve."""
    positions = _circle_positions(space)
    if not positions:
        return
    sample_size = float(kish_effective_sample_size(log_weights))
    unresolved = []
    for position in positions:
        resultant = float(
            mean_resultant_length(
                samples[:, position], log_weights, _period(space, position)
            )
        )
        statistic = 2 * sample_size * resultant**2
        if statistic < RAYLEIGH_THRESHOLD:
            unresolved.append(f"{space.names[position]!r} ({statistic:.3g})")
    if unresolved:
        _warn(
            f"the mean direction is not resolved for {', '.join(unresolved)}: the"
            f" Rayleigh statistic 2 N_eff R^2 lies below {RAYLEIGH_THRESHOLD:.3g},"
            " the 95% point of chi-squared with 2 degrees of freedom for a uniform"
            " circle, so the cut opposite the circular mean may split a mode"
        )


def _image_count(width: Array, length: float, bound: float, per_step: float) -> int:
    """Return how many images on each side cover :data:`IMAGE_REACH` widths.

    A point and the data share one period or interval, so the first image
    left out lies at least count · per_step · length from every datum.

    Args:
        width: Kernel or component width along the axis; traced inside
            ``jax.jit``, where ``bound`` replaces it.
        length: Period of a circle, or width of an interval.
        bound: Largest width expected on the axis, used when ``width`` is
            traced or not finite.
        per_step: Lengths that one more image moves away from the data: 1 for
            windings, 2 for reflections in an interval.

    Returns:
        The number of images on each side, at least 1.
    """
    largest = float(width) if _is_concrete(width) else bound
    if not math.isfinite(largest):
        largest = bound
    return max(1, math.ceil(IMAGE_REACH * largest / (per_step * length)))


def _covers(
    width: Array, length: float, count: int, per_step: float
) -> Bool[Array, " *leading"]:
    """Return whether ``count`` images on each side cover the reach of ``width``.

    Inside ``jax.jit`` the count comes from a bound; a wider kernel or
    component would need more images, so estimators give NaN there.
    """
    return IMAGE_REACH * width <= count * per_step * length


def _image_table(
    axis_images: Sequence[Sequence[tuple[float, float]]],
) -> tuple[tuple[tuple[float, ...], ...], tuple[tuple[float, ...], ...]]:
    """Return the signs and shifts of every image, from the images of each axis.

    An image maps a point x to s ⊙ x + t; the images of the whole space are
    every combination of the images of its axes.

    Args:
        axis_images: For each axis, its (sign, shift) pairs.

    Returns:
        The signs and the shifts, each shaped (I, D) as nested tuples, so they
        can be static fields.
    """
    combinations = list(itertools.product(*axis_images))
    signs = tuple(tuple(sign for sign, _ in images) for images in combinations)
    shifts = tuple(tuple(shift for _, shift in images) for images in combinations)
    return signs, shifts


def _winding_images(period: float, windings: int) -> list[tuple[float, float]]:
    """Return the images of a circle axis: shifts by −W to W periods."""
    return [(1.0, step * period) for step in range(-windings, windings + 1)]


def _map_in_chunks(
    function: Callable[[Array], Array],
    points: Float[Array, "*leading param"],
    chunk_size: int,
) -> Float[Array, " *leading"]:
    """Apply a function to chunks of points shaped (M, D), keeping memory bounded.

    Args:
        function: Maps points shaped (chunk, D) to values shaped (chunk,).
        points: Points shaped (M, D).
        chunk_size: Points per chunk, at least one.

    Returns:
        The values, shaped (M,).
    """
    count, size = points.shape
    if count == 0:
        return jnp.zeros(0)
    chunk_size = max(1, min(chunk_size, count))
    chunk_count = -(-count // chunk_size)
    padding = jnp.zeros((chunk_count * chunk_size - count, size))
    chunks = jnp.concatenate([points, padding]).reshape(chunk_count, chunk_size, size)
    values: Array = jax.lax.map(function, chunks)
    return values.reshape(-1)[:count]


class CovarianceFactor(NamedTuple):
    """Cholesky factor and log-determinant of a covariance, NaN when singular."""

    cholesky: Array
    """Lower Cholesky factor L of Σ = LLᵀ."""
    log_determinant: Array
    """ln det Σ."""
    variances: Array
    """Diagonal of Σ, kept unmasked for the checks."""
    smallest_eigenvalue: Array
    """Smallest eigenvalue of the correlation matrix, kept for the checks."""


def factor_covariance(
    covariance: Float[Array, "*leading param param"],
) -> CovarianceFactor:
    """Factor covariances through their correlation matrices.

    Scaling to unit diagonal keeps the factor exact at any parameter scale
    and makes the checks independent of units (van der Sluis 1969). A
    covariance is singular when a variance is not positive or the smallest
    eigenvalue of its correlation matrix lies below D·ε times the largest;
    its factor and log-determinant are then NaN.

    Args:
        covariance: Covariance matrices Σ, shaped (..., D, D).

    Returns:
        The factors, log-determinants and the quantities the checks read.

    References:
        :cite:t:`vandersluis1969condition`.
    """
    size = covariance.shape[-1]
    variances = jnp.diagonal(covariance, axis1=-2, axis2=-1)
    widths = jnp.sqrt(variances)
    correlation = covariance / (widths[..., :, None] * widths[..., None, :])
    # * eigh and cholesky read one triangle only, so average away the rounding.
    correlation = 0.5 * (correlation + jnp.swapaxes(correlation, -1, -2))
    eigenvalues = jnp.linalg.eigvalsh(correlation)
    valid = (
        jnp.all(variances > 0, axis=-1)
        & jnp.all(jnp.isfinite(covariance), axis=(-2, -1))
        & (eigenvalues[..., 0] > size * FLOAT64_EPSILON * eigenvalues[..., -1])
    )
    correlation_factor = jnp.linalg.cholesky(
        jnp.where(valid[..., None, None], correlation, jnp.eye(size))
    )
    cholesky = widths[..., :, None] * correlation_factor
    log_determinant = 2 * jnp.sum(jnp.log(widths), axis=-1) + 2 * jnp.sum(
        jnp.log(jnp.diagonal(correlation_factor, axis1=-2, axis2=-1)), axis=-1
    )
    return CovarianceFactor(
        cholesky=jnp.where(valid[..., None, None], cholesky, jnp.nan),
        log_determinant=jnp.where(valid, log_determinant, jnp.nan),
        variances=variances,
        smallest_eigenvalue=eigenvalues[..., 0] / eigenvalues[..., -1],
    )


def _singularity(factor: CovarianceFactor, space: ParamSpace) -> str | None:
    """Return why a concrete covariance is singular, or ``None`` if it is not."""
    if not math.isnan(float(factor.log_determinant)):
        return None
    variances = _host_floats(factor.variances)
    if any(math.isnan(variance) for variance in variances):
        return "fewer than two samples have a positive weight"
    flat = [
        name
        for name, variance in zip(space.names, variances, strict=True)
        if not variance > 0
    ]
    if flat:
        verb = "has" if len(flat) == 1 else "have"
        return f"{', '.join(map(repr, flat))} {verb} no spread"
    return (
        "the parameters are collinear: the correlation matrix has the eigenvalue"
        f" ratio {float(factor.smallest_eigenvalue):.3g}, below D*eps"
    )


def _host_floats(values: Array) -> list[float]:
    """Return array values as Python floats, for warning messages."""
    # ! Python floats only: NumPy flags invalid operations differently per platform.
    return [float(value) for value in np.asarray(values).ravel()]
