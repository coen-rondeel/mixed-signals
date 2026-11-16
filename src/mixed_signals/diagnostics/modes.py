"""Modes of one posterior: GMM-BIC structure and the modes of 1D kernel estimates.

GMM-BIC
    :func:`gmm_bic` fits Gaussian mixtures of 1 to :data:`MAX_COMPONENTS`
    components with :class:`~mixed_signals.estimators.gmm.MixtureDensity`
    and scores each by BIC = −2n·ℓ̄ + p ln n (Schwarz 1978), with ℓ̄ the
    weighted mean log-likelihood and p the free parameters. The n is the
    posterior's effective sample size, in the likelihood total and in the
    penalty: Kass and Raftery (1995) take n as the rate at which the
    information grows, and Jones (2011) replaces n by an effective sample
    size for correlated data. On draws repeated five times, n = N inflated
    ΔBIC tenfold and chose too many components, while n_eff gave the result
    of the independent draws (zoo check). ΔBIC = BIC(1) − min BIC measures
    how strongly a mixture beats one Gaussian, on the scale of Kass and
    Raftery (1995), which Fraley and Raftery (2002) use for mixtures. A
    mixture fits any non-Gaussian shape, a banana or a railed posterior as
    well as separate modes, so ΔBIC flags structure, not modes: on the zoo,
    the fixed-point iteration of Carreira-Perpiñán (2000) found five modes
    in the five-component mixtures that BIC chose for bananas, which have one.
Modes of 1D estimates
    :func:`marginal_modes` and :func:`projection_modes` count the modes of
    the bounded one-dimensional
    :class:`~mixed_signals.estimators.kde.KernelDensity` at Silverman's
    factor, as Silverman (1981) reads multimodality from the modes of a
    Gaussian kernel estimate: along every parameter, wrapped on circles and
    corrected at interval bounds, and along the discriminant coordinate of
    every pair of components of the chosen mixture, W⁻¹(μᵢ − μⱼ) with W the
    covariance of the two pooled with their weights, onto which Hennig (2010)
    projects a pair of components to test whether they form one mode. A mode
    that every marginal hides shows there. A curve cannot create a mode in
    one dimension, so bananas and railed posteriors stay unimodal (zoo
    check). A local maximum of the estimate on a grid counts as a peak when
    it reaches :data:`MODE_FLOOR` of the highest peak, which keeps single
    samples in long tails from counting. Scanning the grid, the depth of each
    valley, 1 − valley/lower peak, compares a peak with the mode before it;
    valleys shallower than :data:`MODE_SEPARATION` merge the two peaks into
    one mode. A circle is cut at its lowest density, so that two modes on it
    are separated by the shallower of their two valleys. The deepest valley
    is the effect size: zero for one peak, and compared with the same depth
    of Gaussian surrogates on the same directions. The floor, the merging of
    shallow valleys and the surrogate null are this package's choices.
Scope
    :func:`gmm_bic` reads concrete samples outside ``jax.jit``, because the
    chosen number of components sets the shapes of the mixture; the mode
    counts work under ``jax.jit`` and ``jax.lax.map`` over surrogate sets.

References:
    :cite:t:`schwarz1978dimension`; :cite:t:`kass1995bayes`;
    :cite:t:`fraley2002model`; :cite:t:`jones2011bayesian`;
    :cite:t:`mclachlan2000finite`; :cite:t:`carreiraperpinan2000mode`;
    :cite:t:`silverman1981using`; :cite:t:`silverman1986density`;
    :cite:t:`hennig2010methods`; :cite:t:`jones1996simple`;
    :cite:t:`mardia2000directional`.
"""

import math
import warnings
from typing import Final, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike
from jaxtyping import Array, Float, PRNGKeyArray

from mixed_signals.core.samples import unwrap_about, weighted_covariance
from mixed_signals.core.space import Parameter, ParamSpace
from mixed_signals.core.topology import Topology
from mixed_signals.diagnostics._common import (
    _PACKAGE_DIRECTORY,
    _jit,
    _warn,
    circle_positions,
    parameter_pairs,
    period_of,
)
from mixed_signals.errors import ConvergenceWarning
from mixed_signals.estimators.base import (
    _read_samples,
    _singularity,
    _unwrap_circles,
    _warn_about_unresolved_circles,
    factor_covariance,
)
from mixed_signals.estimators.gmm import MixtureDensity, MixtureSettings
from mixed_signals.estimators.kde import KernelDensity
from mixed_signals.prng import derive_key

MAX_COMPONENTS: Final = 5
"""Largest number of mixture components that BIC may choose."""

MODE_FLOOR: Final = 0.05
"""Fraction of the highest peak that a local maximum must reach to count as a mode.

With a floor of 1%, single samples in long tails counted as modes in 8 to
25% of Student-t and banana marginals and 5% of Gaussian ones, and true modes
at a separation of 3 resolved less often; with 5% no unimodal case counted a
second mode (zoo check).
"""

MODE_SEPARATION: Final = 0.1
"""Depth of valley, 1 − valley/lower peak, that separates two modes.

Shallower valleys merge their peaks into one mode for the mode count; the
deepest valley is reported as the dip either way. It equals the default flag
threshold of the dip: on the zoo, noise bumps stayed below 0.05, and mixtures
from a separation of 2.5 reached 0.10 to 0.17.
"""

GRID_POINTS: Final = 2_048
"""Grid points at which the 1D estimate is searched for modes."""

# * The grid reaches this many kernel widths beyond the data on a line.
_GRID_MARGIN: Final = 3.0


class GmmBic(NamedTuple):
    """BIC of every component count and the chosen mixture."""

    bic: tuple[float, ...]
    """BIC of 1, 2, ... components."""
    components: int
    """Number of components with the smallest BIC."""
    delta_bic: float
    """BIC(1) − min BIC, zero when one Gaussian wins."""
    mixture: MixtureDensity
    """The fitted mixture of the chosen number of components."""


class ModeCounts(NamedTuple):
    """Modes of 1D estimates and the depth of their deepest valley."""

    modes: Float[Array, " *leading"]
    """Number of modes, at least one."""
    dip: Float[Array, " *leading"]
    """1 − valley/lower peak of the deepest valley between neighbouring modes."""


def gmm_bic(
    samples: ArrayLike,
    log_weights: ArrayLike | None,
    space: ParamSpace,
    *,
    effective_sample_size: float,
    key: PRNGKeyArray,
    max_components: int = MAX_COMPONENTS,
    settings: MixtureSettings | None = None,
) -> GmmBic:
    """Fit mixtures of 1 to C components and choose by BIC with n = n_eff.

    Args:
        samples: Concrete samples of one posterior, shaped (N, D).
        log_weights: Their log weights, shaped (N,); equal weights when
            ``None``.
        space: Parameter space of the samples.
        effective_sample_size: The posterior's n_eff, the n of BIC.
        key: JAX PRNG key; the fit of c components draws its starts from
            ``derive_key(key, "components", c)``.
        max_components: Largest number of components.
        settings: Settings of the fits, whose number of components each fit
            replaces; the defaults of
            :class:`~mixed_signals.estimators.gmm.MixtureSettings` when
            ``None``.

    Returns:
        The BIC of every count, the chosen count, ΔBIC and the chosen mixture.

    Raises:
        ValueError: If the shapes do not fit the space, concrete values are
            invalid, ``max_components`` is below one, or the effective
            sample size is below two.
        TypeError: If an input is not real-valued.

    Warns:
        ConvergenceWarning: If EM of the chosen mixture stopped at its
            iteration cap.
        ValidityWarning: If a parameter has no spread or the mean direction
            of a circle is not resolved.
        PrecisionWarning: If an input is narrower than float64.

    References:
        :cite:t:`schwarz1978dimension`; :cite:t:`kass1995bayes`;
        :cite:t:`fraley2002model`; :cite:t:`jones2011bayesian`.
    """
    if max_components < 1:
        raise ValueError(f"max_components must be at least 1, got {max_components}")
    if not effective_sample_size >= 2:
        raise ValueError(
            "BIC needs an effective sample size of at least 2, got"
            f" {effective_sample_size}"
        )
    samples_array, weights_array = _read_samples(samples, log_weights, space)
    _warn_about_unresolved_circles(samples_array, weights_array, space)
    unwrapped, _ = _unwrap_circles(samples_array, weights_array, space)
    _, covariance = weighted_covariance(unwrapped, weights_array)
    # * As the mixtures check: only a parameter without spread makes them NaN.
    spread = factor_covariance(jnp.diag(jnp.diagonal(covariance)))
    if (reason := _singularity(spread, space)) is not None:
        _warn(f"the Gaussian mixtures are NaN: {reason}")
    fits = []
    scores = []
    for components in range(1, max_components + 1):
        fit_settings = (
            MixtureSettings(components=components)
            if settings is None
            else settings.model_copy(update={"components": components})
        )
        # * Fitting traced arrays keeps each fit from repeating the warnings above.
        fit = _fit_mixture(
            samples_array,
            weights_array,
            derive_key(key, "components", components),
            space=space,
            settings=fit_settings,
        )
        fits.append(fit)
        scores.append(
            -2 * effective_sample_size * float(fit.mean_log_likelihood)
            + fit.parameter_count * math.log(effective_sample_size)
        )
    finite = [score if math.isfinite(score) else math.inf for score in scores]
    chosen = int(np.argmin(finite))
    mixture = fits[chosen]
    if math.isfinite(finite[chosen]) and not bool(mixture.converged):
        warnings.warn(
            ConvergenceWarning(
                f"EM of the chosen mixture of {chosen + 1} components stopped at"
                f" {mixture.settings.max_iterations} iterations"
            ),
            skip_file_prefixes=(_PACKAGE_DIRECTORY,),
        )
    delta = scores[0] - finite[chosen] if math.isfinite(scores[0]) else math.nan
    return GmmBic(
        bic=tuple(scores),
        components=chosen + 1,
        delta_bic=delta,
        mixture=mixture,
    )


def discriminant_directions(mixture: MixtureDensity) -> Float[Array, "direction param"]:
    """Return the discriminant coordinate W⁻¹(μᵢ − μⱼ) of every pair of components.

    W = (πᵢΣᵢ + πⱼΣⱼ)/(πᵢ + πⱼ) pools the covariances of the two components
    with their weights π. The directions have unit length and refer to the
    frame of the mixture, with circles unwrapped about their circular means,
    in the order of :func:`parameter_pairs` over the components.

    Args:
        mixture: A fitted mixture of at least two components.

    Returns:
        The directions, shaped (C(C − 1)/2, D).

    References:
        :cite:t:`hennig2010methods`.
    """
    pairs = parameter_pairs(mixture.means.shape[0])
    if not pairs:
        return jnp.zeros((0, mixture.means.shape[1]))
    weights = jnp.exp(mixture.log_weights)
    directions = jnp.stack(
        [
            jnp.linalg.solve(
                weights[first] * mixture.covariances[first]
                + weights[second] * mixture.covariances[second],
                mixture.means[first] - mixture.means[second],
            )
            for first, second in pairs
        ]
    ).reshape(len(pairs), mixture.means.shape[1])
    normalised: Array = directions / jnp.linalg.norm(directions, axis=-1, keepdims=True)
    return normalised


def circle_centres(
    samples: ArrayLike, log_weights: ArrayLike | None, space: ParamSpace
) -> Float[Array, " circle"]:
    """Return the circular means about which the mixture unwraps each circle.

    Args:
        samples: Samples of one posterior, shaped (N, D).
        log_weights: Their log weights, shaped (N,); equal weights when
            ``None``.
        space: Parameter space of the samples.

    Returns:
        The circular mean of each circle, shaped (C,).
    """
    samples_array, weights_array = _read_samples(samples, log_weights, space)
    _, centres = _unwrap_circles(samples_array, weights_array, space)
    return centres


def marginal_modes(
    samples: ArrayLike, log_weights: ArrayLike | None, space: ParamSpace
) -> ModeCounts:
    """Return the modes of the bounded 1D estimate of every parameter.

    Args:
        samples: Samples of one posterior, shaped (N, D), inside the bounds
            of the space.
        log_weights: Their log weights, shaped (N,); equal weights when
            ``None``.
        space: Parameter space of the samples; each parameter keeps its
            topology, so circles wrap and interval bounds are corrected.

    Returns:
        The number of modes and the deepest dip of each parameter, shaped
        (D,).

    Raises:
        ValueError: If the shapes do not fit the space, or concrete values
            are invalid.
        TypeError: If an input is not real-valued.

    Warns:
        ValidityWarning: As
            :meth:`~mixed_signals.estimators.kde.KernelDensity.fit`.
        PrecisionWarning: If an input is narrower than float64.

    References:
        :cite:t:`silverman1981using`; :cite:t:`silverman1986density`;
        :cite:t:`jones1996simple`.
    """
    samples_array, weights_array = _read_samples(samples, log_weights, space)
    counts = [
        _line_modes(
            samples_array[:, position],
            weights_array,
            parameter=_fixed_name(space.parameters[position]),
        )
        for position in range(len(space))
    ]
    return ModeCounts(
        modes=jnp.stack([count.modes for count in counts]),
        dip=jnp.stack([count.dip for count in counts]),
    )


def projection_modes(
    samples: ArrayLike,
    log_weights: ArrayLike | None,
    space: ParamSpace,
    *,
    directions: Float[ArrayLike, "direction param"],
    centres: Float[ArrayLike, " circle"],
) -> ModeCounts:
    """Return the modes of the 1D estimate along each direction.

    Args:
        samples: Samples of one posterior, shaped (N, D).
        log_weights: Their log weights, shaped (N,); equal weights when
            ``None``.
        space: Parameter space of the samples.
        directions: Directions in the frame with circles unwrapped about
            ``centres``, shaped (P, D), such as
            :func:`discriminant_directions`.
        centres: Circular mean of each circle of the observed samples, from
            :func:`circle_centres`, so surrogate sets unwrap alike.

    Returns:
        The number of modes and the deepest dip along each direction, shaped
        (P,).

    Raises:
        ValueError: If the shapes do not fit the space, or concrete values
            are invalid.
        TypeError: If an input is not real-valued.

    References:
        :cite:t:`silverman1981using`; :cite:t:`hennig2010methods`.
    """
    samples_array, weights_array = _read_samples(samples, log_weights, space)
    direction_array = jnp.asarray(directions, dtype=jnp.float64)
    centre_array = jnp.asarray(centres, dtype=jnp.float64)
    for index, position in enumerate(circle_positions(space)):
        samples_array = samples_array.at[:, position].set(
            unwrap_about(
                samples_array[:, position],
                centre_array[index],
                period_of(space, position),
            )
        )
    projected = samples_array @ direction_array.T
    line = Parameter(name="value")
    counts = [
        _line_modes(projected[:, index], weights_array, parameter=line)
        for index in range(direction_array.shape[0])
    ]
    if not counts:
        return ModeCounts(modes=jnp.zeros(0), dip=jnp.zeros(0))
    return ModeCounts(
        modes=jnp.stack([count.modes for count in counts]),
        dip=jnp.stack([count.dip for count in counts]),
    )


def _fixed_name(parameter: Parameter) -> Parameter:
    """Return the parameter named alike, so equal topologies share one compilation."""
    return Parameter(name="value", topology=parameter.topology, bounds=parameter.bounds)


def _fit_mixture_kernel(
    samples: Float[Array, "sample param"],
    log_weights: Float[Array, " sample"],
    key: PRNGKeyArray,
    *,
    space: ParamSpace,
    settings: MixtureSettings,
) -> MixtureDensity:
    """Fit one mixture on traced inputs, which raise no warnings."""
    return MixtureDensity.fit(samples, log_weights, space, settings, key=key)


def _grid(
    values: Float[Array, " sample"], parameter: Parameter, width: Array
) -> Float[Array, " grid"]:
    """Return the grid on which the estimate of one parameter is searched."""
    lower, upper = parameter.bounds
    if parameter.topology is Topology.CIRCLE:
        return lower + (upper - lower) * jnp.arange(GRID_POINTS) / GRID_POINTS
    low = jnp.min(values) - _GRID_MARGIN * width
    high = jnp.max(values) + _GRID_MARGIN * width
    if math.isfinite(lower):
        low = jnp.maximum(low, lower)
    if math.isfinite(upper):
        high = jnp.minimum(high, upper)
    return jnp.linspace(low, high, GRID_POINTS)


def _line_modes_kernel(
    values: Float[Array, " sample"],
    log_weights: Float[Array, " sample"],
    *,
    parameter: Parameter,
) -> ModeCounts:
    """Return the modes and deepest dip of the bounded 1D estimate of values."""
    space = ParamSpace(parameters=(parameter,))
    kde = KernelDensity.fit(values[:, None], log_weights, space)
    # * The covariance of JAX's estimator already holds the factor squared.
    width = jnp.sqrt(kde.kde.covariance[0, 0])
    grid = _grid(values, parameter, width)
    density = jnp.exp(kde.log_density(grid[:, None]))
    if parameter.topology is Topology.CIRCLE:
        # * Cut at the lowest density, so the circle reads as a line between valleys.
        density = jnp.roll(density, -jnp.argmin(density))
    previous = jnp.concatenate([jnp.full(1, -jnp.inf), density[:-1]])
    following = jnp.concatenate([density[1:], jnp.full(1, -jnp.inf)])
    peaks = (density > previous) & (density >= following)
    counted = peaks & (density >= MODE_FLOOR * jnp.max(density))
    initial = (
        jnp.asarray(jnp.nan),
        jnp.asarray(jnp.inf),
        jnp.asarray(0.0),
        jnp.asarray(0.0),
    )
    (_, _, deepest, separations), _ = jax.lax.scan(
        _valley_step, initial, (density, counted)
    )
    valid = jnp.isfinite(kde.log_normaliser)
    return ModeCounts(
        modes=jnp.where(valid, 1 + separations, jnp.nan),
        dip=jnp.where(valid, deepest, jnp.nan),
    )


def _valley_step(
    state: tuple[Array, Array, Array, Array], point: tuple[Array, Array]
) -> tuple[tuple[Array, Array, Array, Array], None]:
    """Compare each counted peak with the current mode across the lowest density since.

    Peaks whose valley is shallower than :data:`MODE_SEPARATION` merge into the
    current mode, whose height is then the higher of the two.
    """
    current, lowest, deepest, separations = state
    density, counted = point
    lowest = jnp.minimum(lowest, density)
    compared = counted & jnp.isfinite(current)
    depth = 1 - lowest / jnp.minimum(current, density)
    deepest = jnp.where(compared, jnp.maximum(deepest, depth), deepest)
    separated = compared & (depth >= MODE_SEPARATION)
    separations = separations + separated
    merged_height = jnp.where(
        jnp.isfinite(current), jnp.maximum(current, density), density
    )
    current = jnp.where(counted, jnp.where(separated, density, merged_height), current)
    lowest = jnp.where(counted, density, lowest)
    return (current, lowest, deepest, separations), None


_fit_mixture = _jit(_fit_mixture_kernel, "space", "settings")
_line_modes = _jit(_line_modes_kernel, "parameter")
