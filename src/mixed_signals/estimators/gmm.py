"""Gaussian mixture estimator: weighted EM from several k-means++ starts.

:class:`MixtureDensity` fits a mixture of C Gaussians with full covariances
to the weighted samples of one posterior by expectation maximisation
(Dempster, Laird and Rubin 1977; McLachlan and Peel 2000). The multimodality
diagnostic compares component counts by BIC (Schwarz 1978), formed from
:attr:`MixtureDensity.mean_log_likelihood` and
:attr:`MixtureDensity.parameter_count`.

Units
    EM runs on the axes z-scored by their weighted standard deviations, so
    the tolerance and the regularisation mean the same at every parameter
    scale, such as a frequency known to 1e-11 Hz; the fit maps back.
Starts
    Weighted k-means++ seeding (Arthur and Vassilvitskii 2007) draws the
    first centre in proportion to the weights, and every further one in
    proportion to the weight times the squared distance to the nearest centre
    so far. Each sample joins its nearest centre, and one M-step from these
    assignments gives the starting mixture, as the k-means++ initialisation
    of scikit-learn does (Pedregosa et al. 2011). Several starts run in
    parallel, and the one with the highest mean log-likelihood wins.
Stopping
    The rule of scikit-learn (Pedregosa et al. 2011): EM stops when the
    weighted mean log-likelihood per sample changes by less than the
    tolerance, 1e-3, or after 100 iterations, and 1e-6 is added to the
    diagonal of every z-scored covariance, which keeps a component from
    collapsing onto single samples. A winning start that hits the cap raises
    a :class:`~mixed_signals.errors.ConvergenceWarning`.
Zoo check
    Under this rule, EM often stops on the slow plateau near the
    one-Gaussian solution: two-component fits to the zoo's mixtures at
    separations 2 to 6 ended there from 37 to 68 per cent of single starts
    in the hardest cases, against 19 per cent at most with a tolerance of
    0.01 nats on the total log-likelihood. :data:`DEFAULT_STARTS` = 18
    starts keep the chance that all of them miss below 1e-3 for the worst
    case, 0.675¹⁸. The winning mean log-likelihood still lies below the
    optimum: by a median of 2.6 to 2.8 nats in total for 4 000 samples in
    the harder cases, and by up to 19 nats, so BIC differences of a few
    units do not resolve the number of components.
Circles
    Each circle is unwrapped about its circular mean before the fit, and the
    density sums every component over windings. A circle whose mean
    direction the Rayleigh test does not resolve raises a
    :class:`~mixed_signals.errors.ValidityWarning`, because the cut opposite
    the mean may split a mode.

References:
    :cite:t:`dempster1977maximum`; :cite:t:`mclachlan2000finite`;
    :cite:t:`schwarz1978dimension`; :cite:t:`arthur2007kmeans`;
    :cite:t:`pedregosa2011scikitlearn`; :cite:t:`mardia2000directional`.
"""

import functools
import math
import warnings
from typing import ClassVar, Final, NamedTuple, cast

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.scipy.linalg import solve_triangular
from jax.scipy.special import logsumexp
from jax.typing import ArrayLike
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray
from pydantic import NonNegativeFloat, PositiveFloat, PositiveInt

from mixed_signals.core.samples import _is_concrete, unwrap_about, weighted_covariance
from mixed_signals.core.space import ParamSpace
from mixed_signals.errors import ConvergenceWarning
from mixed_signals.estimators.base import (
    _PACKAGE_DIRECTORY,
    CHUNK_ENTRIES,
    WIDTH_MARGIN,
    Estimator,
    EstimatorSettings,
    _circle_positions,
    _covers,
    _image_count,
    _image_table,
    _jit,
    _map_in_chunks,
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
from mixed_signals.prng import derive_key
from mixed_signals.transforms.whiten import FLOAT64_EPSILON

DEFAULT_STARTS: Final = 18
"""Starts of EM per fit unless the settings choose another number.

With the stopping rule of scikit-learn, 67.5 per cent of single starts missed
a mode in the hardest zoo mixture checked, so 18 starts all miss with
probability 0.675¹⁸ < 1e-3; see the zoo check in the module description.
"""

_HALF_LOG_TWO_PI: Final = 0.5 * math.log(2 * math.pi)
# * As in scikit-learn, 10 ε on every component mass keeps an empty one finite.
_MASS_FLOOR: Final = 10 * FLOAT64_EPSILON


class MixtureSettings(EstimatorSettings):
    """Settings of the Gaussian mixture estimator.

    Attributes:
        components: Number of Gaussian components C.
        starts: Number of k-means++ starts; the best fit wins.
        max_iterations: Most EM iterations per start.
        tolerance: EM stops when the mean log-likelihood per sample changes
            by less than this.
        regularization: Added to the diagonal of every z-scored covariance.

    References:
        :cite:t:`pedregosa2011scikitlearn`; :cite:t:`arthur2007kmeans`.
    """

    name: ClassVar[str] = "gmm"

    components: PositiveInt
    starts: PositiveInt = DEFAULT_STARTS
    max_iterations: PositiveInt = 100
    tolerance: PositiveFloat = 1e-3
    regularization: NonNegativeFloat = 1e-6


class _Mixture(NamedTuple):
    """Weights, means and covariances of a mixture in z-scored units."""

    log_weights: Array
    means: Array
    covariances: Array


class _Run(NamedTuple):
    """One EM run: the mixture, its last mean log-likelihood and how it stopped."""

    mixture: _Mixture
    mean_log_likelihood: Array
    converged: Array
    iterations: Array


class MixtureDensity(Estimator):
    """A Gaussian mixture fitted to one posterior, wrapped on circles.

    Build it with :meth:`fit`.

    Attributes:
        log_weights: Log weights of the components, shaped (C,).
        means: Component means, shaped (C, D); on a circle, in the frame
            unwrapped about the circular mean.
        covariances: Component covariances, shaped (C, D, D).
        choleskys: Their lower Cholesky factors.
        log_normalisers: −½ ln det(2πΣ_c) of each component.
        mean_log_likelihood: Weighted mean log density of the fitted mixture
            over the samples, for BIC.
        converged: Whether the winning start met the tolerance.
        iterations: EM iterations of the winning start.
        space: Parameter space of the samples.
        windings: Windings summed on each side of every circle.

    References:
        :cite:t:`dempster1977maximum`; :cite:t:`mclachlan2000finite`.
    """

    log_weights: Float[Array, " component"]
    means: Float[Array, "component param"]
    covariances: Float[Array, "component param param"]
    choleskys: Float[Array, "component param param"]
    log_normalisers: Float[Array, " component"]
    mean_log_likelihood: Float[Array, ""]
    converged: Bool[Array, ""]
    iterations: Int[Array, ""]
    space: ParamSpace = eqx.field(static=True)
    windings: tuple[int, ...] = eqx.field(static=True)
    settings: MixtureSettings = eqx.field(static=True)

    @classmethod
    def fit(
        cls,
        samples: ArrayLike,
        log_weights: ArrayLike | None,
        space: ParamSpace,
        settings: MixtureSettings,
        *,
        key: PRNGKeyArray,
    ) -> "MixtureDensity":
        """Fit a Gaussian mixture to the samples of one posterior.

        Args:
            samples: Samples in the parameter order of ``space``, shaped
                (N, D).
            log_weights: Log weights shaped (N,); equal weights when
                ``None``.
            space: Parameter space of the samples.
            settings: The settings, with the number of components.
            key: JAX PRNG key; start i draws from ``derive_key(key, "start",
                i)``.

        Returns:
            The fitted mixture; NaN if a parameter has no spread.

        Raises:
            ValueError: If the shapes do not fit the space, or concrete values
                are invalid.
            TypeError: If an input is not real-valued.

        Warns:
            ConvergenceWarning: If the winning start of a concrete fit stops
                at ``max_iterations``.
            ValidityWarning: If a parameter of concrete samples has no spread,
                or the mean direction of a circle is not resolved.
            PrecisionWarning: If an input is narrower than float64.

        References:
            :cite:t:`dempster1977maximum`; :cite:t:`arthur2007kmeans`;
            :cite:t:`pedregosa2011scikitlearn`.
        """
        samples_array, weights_array = _read_samples(samples, log_weights, space)
        unwrapped, _ = _unwrap_circles(samples_array, weights_array, space)
        location, covariance = weighted_covariance(unwrapped, weights_array)
        spread = factor_covariance(jnp.diag(jnp.diagonal(covariance)))
        if _is_concrete(samples_array, weights_array):
            _warn_about_unresolved_circles(samples_array, weights_array, space)
            if (reason := _singularity(spread, space)) is not None:
                _warn(f"the Gaussian mixture is NaN: {reason}")
        scales = jnp.sqrt(jnp.diagonal(covariance))
        standardised = (unwrapped - location) / scales
        keys = jnp.stack(
            [derive_key(key, "start", index) for index in range(settings.starts)]
        )
        runs = jax.vmap(
            functools.partial(
                _run_em,
                standardised,
                jax.nn.log_softmax(weights_array),
                components=settings.components,
                max_iterations=settings.max_iterations,
                tolerance=settings.tolerance,
                regularization=settings.regularization,
            )
        )(keys)
        best = jnp.argmax(jnp.nan_to_num(runs.mean_log_likelihood, nan=-jnp.inf))
        mixture = jax.tree.map(lambda leaves: leaves[best], runs.mixture)
        means = location + scales * mixture.means
        covariances = mixture.covariances * scales[:, None] * scales[None, :]
        factor = factor_covariance(covariances)
        windings = []
        covered = jnp.asarray(True)
        for position in _circle_positions(space):
            period = _period(space, position)
            widest = jnp.max(jnp.sqrt(covariances[:, position, position]))
            count = _image_count(widest, period, WIDTH_MARGIN * period / 2, 1.0)
            windings.append(count)
            covered = covered & _covers(widest, period, count, 1.0)
        size = len(space)
        log_normalisers = -0.5 * factor.log_determinant - size * _HALF_LOG_TWO_PI
        estimator = cls(
            log_weights=mixture.log_weights,
            means=means,
            covariances=covariances,
            choleskys=factor.cholesky,
            log_normalisers=jnp.where(covered, log_normalisers, jnp.nan),
            mean_log_likelihood=jnp.asarray(jnp.nan),
            converged=runs.converged[best],
            iterations=runs.iterations[best],
            space=space,
            windings=tuple(windings),
            settings=settings,
        )
        mean_log_likelihood = jnp.sum(
            jax.nn.softmax(weights_array) * _log_density(estimator, samples_array)
        )
        estimator = cast(
            MixtureDensity,
            eqx.tree_at(
                lambda fitted: fitted.mean_log_likelihood,
                estimator,
                mean_log_likelihood,
            ),
        )
        # * The key may be traced while the samples are not, so test the outcome itself.
        if (
            _is_concrete(estimator.converged)
            and math.isfinite(float(estimator.mean_log_likelihood))
            and not bool(estimator.converged)
        ):
            warnings.warn(
                ConvergenceWarning(
                    f"EM stopped after {settings.max_iterations} iterations before the"
                    " mean log-likelihood per sample changed by less than"
                    f" {settings.tolerance:g}; raise max_iterations or check the"
                    " number of components"
                ),
                skip_file_prefixes=(_PACKAGE_DIRECTORY,),
            )
        return estimator

    @property
    def parameter_count(self) -> int:
        """Free parameters of the mixture, the p of BIC.

        C − 1 weights, C·D means and C·D(D + 1)/2 covariance entries.

        References:
            :cite:t:`schwarz1978dimension`; :cite:t:`mclachlan2000finite`.
        """
        components = self.settings.components
        size = len(self.space)
        return components - 1 + components * size + components * size * (size + 1) // 2

    def log_density(
        self, points: Float[ArrayLike, "*leading param"]
    ) -> Float[Array, " *leading"]:
        """Return the log density of the fitted mixture at points.

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


def _component_log_densities(
    means: Float[Array, "component param"],
    choleskys: Float[Array, "component param param"],
    points: Float[Array, "*leading param"],
) -> Float[Array, "*leading component"]:
    """Return ln N(x; μ_c, Σ_c) without normalisers, for unbounded z-scored values."""
    offsets = points[:, None, :] - means
    whitened = jax.vmap(
        lambda factor, offset: solve_triangular(factor, offset.T, lower=True).T,
        in_axes=(0, 1),
        out_axes=1,
    )(choleskys, offsets)
    log_determinants = jnp.sum(
        jnp.log(jnp.diagonal(choleskys, axis1=-2, axis2=-1)), axis=-1
    )
    size = points.shape[-1]
    return (
        -0.5 * jnp.sum(whitened**2, axis=-1)
        - log_determinants
        - size * _HALF_LOG_TWO_PI
    )


def _seed_means(
    points: Float[Array, "sample param"],
    log_weights: Float[Array, " sample"],
    key: PRNGKeyArray,
    components: int,
) -> Float[Array, "component param"]:
    """Return k-means++ centres, each drawn ∝ weight times squared distance."""
    first = jax.random.categorical(derive_key(key, "seed", 0), log_weights)
    centres = [points[first]]
    closest = jnp.sum((points - points[first]) ** 2, axis=-1)
    for index in range(1, components):
        logits = log_weights + jnp.log(closest)
        # * Fewer distinct samples than components: repeat a centre instead of failing.
        logits = jnp.where(jnp.any(jnp.isfinite(logits)), logits, log_weights)
        chosen = jax.random.categorical(derive_key(key, "seed", index), logits)
        centres.append(points[chosen])
        closest = jnp.minimum(closest, jnp.sum((points - points[chosen]) ** 2, axis=-1))
    return jnp.stack(centres)


def _maximise(
    points: Float[Array, "sample param"],
    log_weights: Float[Array, " sample"],
    log_responsibilities: Float[Array, "sample component"],
    regularization: float,
) -> _Mixture:
    """Return the weighted M-step: component weights, means and covariances."""
    shares = jnp.exp(log_weights[:, None] + log_responsibilities)
    masses = jnp.sum(shares, axis=0) + _MASS_FLOOR
    means = (shares.T @ points) / masses[:, None]
    offsets = points[:, None, :] - means
    covariances = jnp.einsum("sc,scp,scq->cpq", shares, offsets, offsets) / masses[
        :, None, None
    ] + regularization * jnp.eye(points.shape[-1])
    return _Mixture(jnp.log(masses / jnp.sum(masses)), means, covariances)


def _expect(
    points: Float[Array, "sample param"],
    log_weights: Float[Array, " sample"],
    mixture: _Mixture,
) -> tuple[Float[Array, ""], Float[Array, "sample component"]]:
    """Return the weighted mean log-likelihood and the log responsibilities."""
    choleskys = jnp.linalg.cholesky(mixture.covariances)
    joint = mixture.log_weights + _component_log_densities(
        mixture.means, choleskys, points
    )
    log_likelihoods = logsumexp(joint, axis=-1)
    mean_log_likelihood = jnp.sum(jnp.exp(log_weights) * log_likelihoods)
    return mean_log_likelihood, joint - log_likelihoods[:, None]


def _run_em(
    points: Float[Array, "sample param"],
    log_weights: Float[Array, " sample"],
    key: PRNGKeyArray,
    *,
    components: int,
    max_iterations: int,
    tolerance: float,
    regularization: float,
) -> _Run:
    """Run EM from one k-means++ start on z-scored points with normalised weights."""
    centres = _seed_means(points, log_weights, key, components)
    nearest = jnp.argmin(jnp.sum((points[:, None, :] - centres) ** 2, axis=-1), axis=-1)
    hard = jnp.where(jnp.arange(components) == nearest[:, None], 0.0, -jnp.inf)
    initial = _maximise(points, log_weights, hard, regularization)
    state = (initial, jnp.asarray(-jnp.inf), jnp.asarray(0), jnp.asarray(False))
    final = jax.lax.while_loop(
        functools.partial(_keep_going, max_iterations=max_iterations),
        functools.partial(
            _em_step,
            points,
            log_weights,
            tolerance=tolerance,
            regularization=regularization,
        ),
        state,
    )
    mixture, mean_log_likelihood, iterations, converged = final
    return _Run(mixture, mean_log_likelihood, converged, iterations)


def _keep_going(
    state: tuple[_Mixture, Array, Array, Array], *, max_iterations: int
) -> Array:
    """Return whether EM continues: below the iteration cap and not converged."""
    _, _, iterations, converged = state
    return (iterations < max_iterations) & ~converged


def _em_step(
    points: Float[Array, "sample param"],
    log_weights: Float[Array, " sample"],
    state: tuple[_Mixture, Array, Array, Array],
    *,
    tolerance: float,
    regularization: float,
) -> tuple[_Mixture, Array, Array, Array]:
    """Return the state after one E-step and M-step, as scikit-learn orders them."""
    mixture, previous, iterations, _ = state
    mean_log_likelihood, log_responsibilities = _expect(points, log_weights, mixture)
    updated = _maximise(points, log_weights, log_responsibilities, regularization)
    converged = jnp.abs(mean_log_likelihood - previous) < tolerance
    return updated, mean_log_likelihood, iterations + 1, converged


def _log_density_kernel(
    estimator: MixtureDensity, points: Float[Array, "*leading param"]
) -> Float[Array, " *leading"]:
    """Return the log density of the wrapped mixture at points shaped (M, D)."""
    axis_images: list[list[tuple[float, float]]] = [[(1.0, 0.0)]] * points.shape[-1]
    for position, windings in zip(
        _circle_positions(estimator.space), estimator.windings, strict=True
    ):
        axis_images[position] = _winding_images(
            _period(estimator.space, position), windings
        )
    _, shifts = _image_table(axis_images)
    image_count = len(shifts)
    components, size = estimator.means.shape
    chunk_size = max(1, CHUNK_ENTRIES // (components * image_count * size))
    return _map_in_chunks(
        functools.partial(_chunk_log_density, estimator, shifts), points, chunk_size
    )


def _chunk_log_density(
    estimator: MixtureDensity,
    shifts: tuple[tuple[float, ...], ...],
    points: Float[Array, "*leading param"],
) -> Float[Array, " *leading"]:
    """Return the log density of the wrapped mixture at one chunk of points."""
    offsets = points[:, None, :] - estimator.means
    for position in _circle_positions(estimator.space):
        offsets = offsets.at[..., position].set(
            unwrap_about(
                offsets[..., position], 0.0, _period(estimator.space, position)
            )
        )
    images = offsets[:, :, None, :] + jnp.asarray(shifts)
    whitened = jax.vmap(
        lambda factor, image: solve_triangular(
            factor, image.reshape(-1, image.shape[-1]).T, lower=True
        ).T.reshape(image.shape),
        in_axes=(0, 1),
        out_axes=1,
    )(estimator.choleskys, images)
    terms = -0.5 * jnp.sum(whitened**2, axis=-1) + estimator.log_normalisers[:, None]
    per_component = logsumexp(terms, axis=-1)
    return logsumexp(estimator.log_weights + per_component, axis=-1)


_log_density = _jit(_log_density_kernel)
