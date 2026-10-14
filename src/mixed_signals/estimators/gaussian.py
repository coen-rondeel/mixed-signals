"""Gaussian estimator: the moment-matched Gaussian of one posterior.

:class:`GaussianDensity` has the weighted mean and the unbiased weighted
covariance of the samples, the package's widths
(:func:`~mixed_signals.core.samples.weighted_covariance`; Harris et al. 2020).
It stands for a posterior in the Gaussian KL, W2 and Hellinger rows and in the
Mahalanobis diagnostic, and its draws are the Gaussian surrogates that the
shape and dependence diagnostics are compared with, a parametric bootstrap
(Efron and Tibshirani 1993).

Circles
    Each circle is unwrapped about its circular mean before the moments are
    taken, and the density sums the Gaussian over windings, which makes it a
    wrapped normal on that circle (Mardia and Jupp 2000); draws are wrapped
    back onto the circle. A circle whose mean direction the Rayleigh test
    does not resolve raises a :class:`~mixed_signals.errors.ValidityWarning`.
Intervals
    The Gaussian is the moment-matched Gaussian on the whole real line, so it
    puts mass beyond interval bounds, and so do its draws.
Validity
    A singular covariance, from fewer than two samples with positive weight,
    a parameter without spread or collinear parameters, gives NaN and a
    :class:`~mixed_signals.errors.ValidityWarning`.

References:
    :cite:t:`harris2020array`; :cite:t:`chan1983algorithms`;
    :cite:t:`mardia2000directional`; :cite:t:`efron1993introduction`.
"""

import math
from typing import ClassVar, Final

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.scipy.linalg import solve_triangular
from jax.scipy.special import logsumexp
from jax.typing import ArrayLike
from jaxtyping import Array, Float, PRNGKeyArray

from mixed_signals.core.samples import _is_concrete, unwrap_about, weighted_covariance
from mixed_signals.core.space import ParamSpace
from mixed_signals.core.topology import wrap
from mixed_signals.estimators.base import (
    WIDTH_MARGIN,
    Estimator,
    EstimatorSettings,
    _circle_positions,
    _covers,
    _image_count,
    _image_table,
    _jit,
    _period,
    _read_points,
    _read_samples,
    _singularity,
    _unwrap_circles,
    _warn,
    _warn_about_unresolved_circles,
    _winding_images,
    factor_covariance,
)

_HALF_LOG_TWO_PI: Final = 0.5 * math.log(2 * math.pi)


class GaussianSettings(EstimatorSettings):
    """Settings of the Gaussian estimator, which has nothing to choose."""

    name: ClassVar[str] = "gaussian"


class GaussianDensity(Estimator):
    """The moment-matched Gaussian of one posterior, wrapped on circles.

    Build it with :meth:`fit`.

    Attributes:
        mean: Weighted mean, shaped (D,); on a circle, the mean of the values
            unwrapped about the circular mean, wrapped back onto the circle.
        covariance: Unbiased weighted covariance, shaped (D, D).
        cholesky: Lower Cholesky factor L of the covariance.
        log_normaliser: −½ ln det(2πΣ), the log density at the mean before
            wrapping.
        space: Parameter space of the samples.
        windings: Windings summed on each side of every circle.

    References:
        :cite:t:`harris2020array`; :cite:t:`mardia2000directional`.
    """

    mean: Float[Array, " param"]
    covariance: Float[Array, "param param"]
    cholesky: Float[Array, "param param"]
    log_normaliser: Float[Array, ""]
    space: ParamSpace = eqx.field(static=True)
    windings: tuple[int, ...] = eqx.field(static=True)
    settings: GaussianSettings = eqx.field(static=True)

    @classmethod
    def fit(
        cls,
        samples: ArrayLike,
        log_weights: ArrayLike | None,
        space: ParamSpace,
        settings: GaussianSettings | None = None,
    ) -> "GaussianDensity":
        """Fit the moment-matched Gaussian to the samples of one posterior.

        Args:
            samples: Samples in the parameter order of ``space``, shaped
                (N, D).
            log_weights: Log weights shaped (N,); equal weights when
                ``None``.
            space: Parameter space of the samples.
            settings: The settings; the defaults when ``None``.

        Returns:
            The fitted Gaussian, NaN if its covariance is singular.

        Raises:
            ValueError: If the shapes do not fit the space, or concrete values
                are invalid.
            TypeError: If an input is not real-valued.

        Warns:
            ValidityWarning: If the covariance of concrete samples is
                singular, or the mean direction of a circle is not resolved.
            PrecisionWarning: If an input is narrower than float64.

        References:
            :cite:t:`harris2020array`; :cite:t:`mardia2000directional`.
        """
        samples_array, weights_array = _read_samples(samples, log_weights, space)
        unwrapped, _ = _unwrap_circles(samples_array, weights_array, space)
        mean, covariance = weighted_covariance(unwrapped, weights_array)
        factor = factor_covariance(covariance)
        if _is_concrete(samples_array, weights_array):
            _warn_about_unresolved_circles(samples_array, weights_array, space)
            if (reason := _singularity(factor, space)) is not None:
                _warn(f"the Gaussian is NaN: {reason}")
        windings = []
        covered = jnp.asarray(True)
        for position in _circle_positions(space):
            period = _period(space, position)
            lower = space.parameters[position].bounds[0]
            mean = mean.at[position].set(wrap(mean[position], lower, period))
            width = jnp.sqrt(covariance[position, position])
            # * Unwrapped values span one period, so their width is about half at most.
            count = _image_count(width, period, WIDTH_MARGIN * period / 2, 1.0)
            windings.append(count)
            covered = covered & _covers(width, period, count, 1.0)
        size = len(space)
        log_normaliser = -0.5 * factor.log_determinant - size * _HALF_LOG_TWO_PI
        return cls(
            mean=mean,
            covariance=covariance,
            cholesky=factor.cholesky,
            log_normaliser=jnp.where(covered, log_normaliser, jnp.nan),
            space=space,
            windings=tuple(windings),
            settings=settings if settings is not None else GaussianSettings(),
        )

    def log_density(
        self, points: Float[ArrayLike, "*leading param"]
    ) -> Float[Array, " *leading"]:
        """Return the log density of the wrapped Gaussian at points.

        Args:
            points: Points shaped (..., D), in the coordinates of the space.

        Returns:
            The log densities, shaped (...).

        Raises:
            ValueError: If the points do not end in the D parameters.
            TypeError: If the points are not real-valued.
        """
        values = _read_points(points, len(self.space))
        flat = values.reshape(-1, values.shape[-1])
        return _log_density(self, flat).reshape(values.shape[:-1])

    def sample(
        self, key: PRNGKeyArray, shape: tuple[int, ...] = ()
    ) -> Float[Array, "*leading param"]:
        """Draw from the Gaussian, wrapping each circle back onto its period.

        Args:
            key: JAX PRNG key.
            shape: Leading shape of the draws.

        Returns:
            The draws, shaped ``shape + (D,)``.

        References:
            :cite:t:`efron1993introduction`.
        """
        standard = jax.random.normal(key, (*shape, self.mean.shape[0]))
        draws = self.mean + standard @ self.cholesky.T
        for position in _circle_positions(self.space):
            lower = self.space.parameters[position].bounds[0]
            draws = draws.at[..., position].set(
                wrap(draws[..., position], lower, _period(self.space, position))
            )
        return draws


def _log_density_kernel(
    estimator: GaussianDensity, points: Float[Array, "*leading param"]
) -> Float[Array, " *leading"]:
    """Return the log density of the wrapped Gaussian at points shaped (M, D)."""
    offsets = points - estimator.mean
    axis_images: list[list[tuple[float, float]]] = [[(1.0, 0.0)]] * offsets.shape[-1]
    for position, windings in zip(
        _circle_positions(estimator.space), estimator.windings, strict=True
    ):
        period = _period(estimator.space, position)
        offsets = offsets.at[:, position].set(
            unwrap_about(offsets[:, position], 0.0, period)
        )
        axis_images[position] = _winding_images(period, windings)
    _, shifts = _image_table(axis_images)
    images = offsets[:, None, :] + jnp.asarray(shifts)
    whitened = solve_triangular(
        estimator.cholesky, images.reshape(-1, images.shape[-1]).T, lower=True
    )
    terms = -0.5 * jnp.sum(whitened**2, axis=0).reshape(images.shape[:-1])
    return logsumexp(terms, axis=-1) + estimator.log_normaliser


_log_density = _jit(_log_density_kernel)
