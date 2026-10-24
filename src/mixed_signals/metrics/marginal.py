"""Marginal metrics: KS or Kuiper, Anderson-Darling, W1, JS and Hellinger.

Weighted distribution functions
    The function of a weighted sample puts the normalised weight of each
    sample at its value, so a weight of zero removes a sample, and equal
    weights give the empirical distribution function. Two functions are
    compared at the end of each run of equal values, where both are defined,
    so repeated values, such as the rejected proposals of a Markov chain,
    count once with their summed weight.
Kolmogorov-Smirnov and Kuiper statistics
    D is the largest absolute gap F(x) − G(x) (Smirnov 1948); equal weights
    give SciPy's ``ks_2samp`` (Virtanen et al. 2020). On a circle, D depends on
    where the circle is cut, so a circle marginal takes Kuiper's
    V = max(F − G) − min(F − G) instead (Kuiper 1960), in the two-sample form of
    Maag and Stephens (1968), which no rotation of the circle changes. Their
    rows read ``ks`` and ``kuiper_ks``. For two unit Gaussians shifted, F − G
    keeps one sign and V = D, so the two share a σ_eff map.
Anderson-Darling distance
    ∫ (F − G)²/(H(1 − H)) dH with the pooled function H = (F + G)/2, each
    posterior's weights normalised to sum to one. For equal weights and equal
    sample sizes it is the two-sample statistic of Pettitt (1976), the
    k-sample statistic of Scholz and Stephens (1987) for k = 2, divided by
    nm/(n + m): SciPy's ``anderson_ksamp`` with ``midrank=False``, whose
    sum over the ends of runs of ties is the one used here. Equal pooling
    makes its expectation the zoo's AD distance for any n and m. Linden (2026)
    weights the two-sample statistic by pooling the raw weights of both
    groups, which is this pooling when each group's weights sum to one. A
    circle marginal is cut opposite the circular mean of the two marginals
    pooled with equal weight; no verified two-sample Anderson-Darling
    statistic is invariant under rotations of the circle.
W1 distance
    The integral of the absolute gap between F and G (Vallender 1974),
    SciPy's ``wasserstein_distance`` with weights. On a circle with the arc
    length as cost, it is the smallest integral, over one period, of the
    absolute value of F − G − c for a constant c, reached at the
    length-weighted median of F − G (Rabin, Delon and Gousseau 2011); no cut
    changes it.
JS divergence and Hellinger distance
    Both come from two one-dimensional
    :class:`~mixed_signals.estimators.kde.KernelDensity` estimates: wrapped on
    circles, corrected at interval bounds as Jones and Foster (1996) do. The
    two share one bandwidth factor, Silverman's (Silverman 1986) at the
    smaller Kish effective sample size of the pair, fixed once per comparison
    and reused by every null draw (:func:`shared_bandwidth_factor`); each
    kernel is that factor times its own width, so smoothing keeps width
    ratios. The JS divergence, in nats (Lin 1991; Romero-Shaw et al. 2020),
    and H² = ½ ∫ (√p − √q)² dx are integrated with Gauss-Legendre rules
    (Davis and Rabinowitz 1984). Each estimate adds equal panels across its
    samples and :data:`~mixed_signals.estimators.base.IMAGE_REACH` bandwidths
    beyond them, clipped to an interval or to one period, each at most two
    bandwidths wide, where eight nodes integrate a sum of Gaussian kernels to
    1e-13. pesummary instead evaluates its estimates at 100 points between
    the pooled extremes (Hoy and Raymond 2021).
Per-parameter rows
    :func:`marginal_distances` and :func:`marginal_kde_divergences` compare
    every parameter of two posteriors with these metrics, sphere axes by
    their own topologies. Their rows read the estimator ``empirical`` and
    ``kde``, and :func:`ks_metric_names` names each parameter's KS or Kuiper
    row.

References:
    :cite:t:`smirnov1948table`; :cite:t:`virtanen2020scipy`;
    :cite:t:`kuiper1960tests`; :cite:t:`maag1968vnm`;
    :cite:t:`pettitt1976two`; :cite:t:`scholz1987k`;
    :cite:t:`linden2026weighted`; :cite:t:`vallender1974calculation`;
    :cite:t:`rabin2011transportation`; :cite:t:`jones1996simple`;
    :cite:t:`silverman1986density`; :cite:t:`lin1991divergence`;
    :cite:t:`romeroshaw2020bayesian`; :cite:t:`davis1984methods`;
    :cite:t:`hoy2021pesummary`; :cite:t:`mardia2000directional`.
"""

import math
from collections.abc import Callable
from typing import Final, NamedTuple, cast

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike
from jaxtyping import Array, Bool, Float

from mixed_signals.core.samples import (
    _is_concrete,
    circular_mean,
    kish_effective_sample_size,
    unwrap_about,
)
from mixed_signals.core.space import ParamSpace
from mixed_signals.core.topology import Topology, wrap
from mixed_signals.estimators.base import (
    IMAGE_REACH,
    _read_samples,
    _warn_about_unresolved_circles,
)
from mixed_signals.estimators.kde import KernelDensity, KernelDensitySettings

PANEL_COUNT: Final = 128
"""Equal panels per kernel density estimate inside ``jax.jit``.

Outside ``jax.jit`` the count grows until every panel is at most two
bandwidths wide; inside, a wider panel gives NaN.
"""

PANEL_BANDWIDTHS: Final = 2.0
"""Widest panel, in bandwidths of its estimate, on which eight Gauss-Legendre
nodes integrate a sum of Gaussian kernels to 1e-13."""

_LEGENDRE_ORDER: Final = 8
# * Fixed node table: the panels move with the estimates, the rule does not.
_LEGENDRE_NODES, _LEGENDRE_WEIGHTS = np.polynomial.legendre.leggauss(_LEGENDRE_ORDER)
_LOG_TWO: Final = math.log(2)


class MarginalDistances(NamedTuple):
    """Distances between the marginals of two posteriors, one entry per parameter."""

    ks: Array
    """KS statistic D, or Kuiper's V on a circle; see :func:`ks_metric_names`."""
    ad: Array
    """Anderson-Darling distance."""
    w1: Array
    """W1 distance, along the circle on a circle."""


class KdeDivergences(NamedTuple):
    """Divergences between two kernel density estimates, one entry per parameter."""

    js: Array
    """Jensen-Shannon divergence, in nats, in [0, ln 2]."""
    hellinger: Array
    """Hellinger distance, in [0, 1]."""


class _Steps(NamedTuple):
    """Sorted pooled values of two weighted samples and the steps of their functions."""

    values: Array
    """The pooled values, sorted."""
    gaps: Array
    """F − G after each value."""
    pooled: Array
    """H = (F + G)/2 after each value."""
    run_ends: Array
    """Whether a value ends a run of equal values, where F and G are defined."""


def _jit[**Parameters, Result](
    function: Callable[Parameters, Result], *static_names: str
) -> Callable[Parameters, Result]:
    """Compile a kernel with :func:`jax.jit` and keep its signature for type checks."""
    return cast(
        Callable[Parameters, Result], jax.jit(function, static_argnames=static_names)
    )


@_jit
def ks_statistic(
    first: Float[Array, "*leading sample"],
    first_log_weights: Float[Array, "*leading sample"],
    second: Float[Array, "*leading other"],
    second_log_weights: Float[Array, "*leading other"],
) -> Float[Array, " *leading"]:
    """Return the two-sample Kolmogorov-Smirnov statistic of weighted samples.

    The statistic is the largest gap between the weighted empirical
    distribution functions of the two samples, over the last axis. Each
    sample needs at least one weight above zero.

    Args:
        first: Values of the first sample, shaped (..., N).
        first_log_weights: Natural logarithms of its unnormalised weights,
            shaped (..., N); −∞ for zero.
        second: Values of the second sample, shaped (..., M).
        second_log_weights: Natural logarithms of its unnormalised weights,
            shaped (..., M); −∞ for zero.

    Returns:
        The statistic D in [0, 1], shaped (...).

    References:
        :cite:t:`smirnov1948table`; :cite:t:`virtanen2020scipy`.
    """
    steps = _distribution_steps(first, first_log_weights, second, second_log_weights)
    return _ks_from_steps(steps)


@_jit
def kuiper_statistic(
    first: Float[Array, "*leading sample"],
    first_log_weights: Float[Array, "*leading sample"],
    second: Float[Array, "*leading other"],
    second_log_weights: Float[Array, "*leading other"],
) -> Float[Array, " *leading"]:
    """Return Kuiper's two-sample statistic V of weighted samples.

    V = max(F − G) − min(F − G), over the last axis, with F − G = 0 before the
    first value. For values on a circle, reduced to one period, a different
    cut of the circle adds a constant to F − G, which V does not see. Each
    sample needs at least one weight above zero.

    Args:
        first: Values of the first sample, shaped (..., N); on a circle,
            within one period.
        first_log_weights: Natural logarithms of its unnormalised weights,
            shaped (..., N); −∞ for zero.
        second: Values of the second sample, shaped (..., M), within the same
            period.
        second_log_weights: Natural logarithms of its unnormalised weights,
            shaped (..., M); −∞ for zero.

    Returns:
        The statistic V in [0, 1], shaped (...).

    References:
        :cite:t:`kuiper1960tests`; :cite:t:`maag1968vnm`.
    """
    steps = _distribution_steps(first, first_log_weights, second, second_log_weights)
    return _kuiper_from_steps(steps)


@_jit
def anderson_darling_distance(
    first: Float[Array, "*leading sample"],
    first_log_weights: Float[Array, "*leading sample"],
    second: Float[Array, "*leading other"],
    second_log_weights: Float[Array, "*leading other"],
) -> Float[Array, " *leading"]:
    """Return the two-sample Anderson-Darling distance of weighted samples.

    The distance is Σ (F − G)² ΔH / (H(1 − H)) over the ends of the runs of
    equal values, with H = (F + G)/2 and ΔH the pooled mass of each run; the
    last run, where H = 1, adds nothing. For equal weights and n = m it is
    the statistic of Pettitt (1976) divided by nm/(n + m). Each sample needs
    at least one weight above zero.

    Args:
        first: Values of the first sample, shaped (..., N).
        first_log_weights: Natural logarithms of its unnormalised weights,
            shaped (..., N); −∞ for zero.
        second: Values of the second sample, shaped (..., M).
        second_log_weights: Natural logarithms of its unnormalised weights,
            shaped (..., M); −∞ for zero.

    Returns:
        The distance, non-negative, shaped (...).

    References:
        :cite:t:`pettitt1976two`; :cite:t:`scholz1987k`;
        :cite:t:`linden2026weighted`.
    """
    steps = _distribution_steps(first, first_log_weights, second, second_log_weights)
    return _anderson_darling_from_steps(steps)


@_jit
def wasserstein_distance(
    first: Float[Array, "*leading sample"],
    first_log_weights: Float[Array, "*leading sample"],
    second: Float[Array, "*leading other"],
    second_log_weights: Float[Array, "*leading other"],
) -> Float[Array, " *leading"]:
    """Return the W1 distance between weighted samples on the real line.

    The distance is the integral of the absolute gap between F and G: the
    sum of that gap times the spacing to the next pooled value, in the
    units of the values. Each sample needs at least one weight above zero.

    Args:
        first: Values of the first sample, shaped (..., N).
        first_log_weights: Natural logarithms of its unnormalised weights,
            shaped (..., N); −∞ for zero.
        second: Values of the second sample, shaped (..., M).
        second_log_weights: Natural logarithms of its unnormalised weights,
            shaped (..., M); −∞ for zero.

    Returns:
        The distance, non-negative, shaped (...).

    References:
        :cite:t:`vallender1974calculation`; :cite:t:`virtanen2020scipy`.
    """
    steps = _distribution_steps(first, first_log_weights, second, second_log_weights)
    return _wasserstein_from_steps(steps)


@_jit
def circle_wasserstein_distance(
    first: Float[Array, "*leading sample"],
    first_log_weights: Float[Array, "*leading sample"],
    second: Float[Array, "*leading other"],
    second_log_weights: Float[Array, "*leading other"],
    period: Float[ArrayLike, ""],
) -> Float[Array, " *leading"]:
    """Return the W1 distance between weighted samples on a circle.

    With the arc length as cost, the distance is the smallest integral, over
    one period, of the absolute value of F − G − c for a constant c. F − G is
    constant between neighbouring values, and on the arc from the last value
    back to the first it is zero, so the minimum lies at the median of those
    constants, each weighted by the length of its arc.

    Args:
        first: Values of the first sample, shaped (..., N), within one
            period.
        first_log_weights: Natural logarithms of its unnormalised weights,
            shaped (..., N); −∞ for zero.
        second: Values of the second sample, shaped (..., M), within the same
            period.
        second_log_weights: Natural logarithms of its unnormalised weights,
            shaped (..., M); −∞ for zero.
        period: Period P of the circle.

    Returns:
        The distance, in the units of the values, at most P/2, shaped (...).

    References:
        :cite:t:`rabin2011transportation`.
    """
    steps = _distribution_steps(first, first_log_weights, second, second_log_weights)
    return _circle_wasserstein_from_steps(steps, jnp.asarray(period))


def ks_metric_names(space: ParamSpace) -> tuple[str, ...]:
    """Return the metric name of each parameter's KS row: ``kuiper_ks`` on circles.

    Args:
        space: Space of the parameters, sphere azimuths included as circles.

    Returns:
        ``"ks"`` or ``"kuiper_ks"`` for each parameter, in order.
    """
    return tuple(
        "kuiper_ks" if parameter.topology is Topology.CIRCLE else "ks"
        for parameter in space.parameters
    )


def marginal_distances(
    first: ArrayLike,
    first_log_weights: ArrayLike | None,
    second: ArrayLike,
    second_log_weights: ArrayLike | None,
    space: ParamSpace,
) -> MarginalDistances:
    """Return the KS or Kuiper, Anderson-Darling and W1 distances of every marginal.

    Lines and intervals take the KS statistic, the Anderson-Darling distance
    and the W1 distance. A circle, sphere azimuths included, takes Kuiper's
    V, the W1 distance along the circle, and the Anderson-Darling distance
    with the circle cut opposite the circular mean of the two marginals
    pooled with equal weight. Distances are in the units of the coordinates;
    z-scored coordinates put W1 in units of σ.

    Args:
        first: Samples of the first posterior, shaped (N, D), in the
            parameter order of ``space``.
        first_log_weights: Its log weights shaped (N,); equal weights when
            ``None``.
        second: Samples of the second posterior, shaped (M, D).
        second_log_weights: Its log weights shaped (M,); equal weights when
            ``None``.
        space: Space of the samples.

    Returns:
        The distances, each shaped (D,).

    Raises:
        ValueError: If the shapes do not fit the space, or concrete values are
            invalid.
        TypeError: If an input is not real-valued.

    Warns:
        ValidityWarning: If the Rayleigh test does not resolve the pooled mean
            direction of a circle of concrete samples, where the
            Anderson-Darling cut may split a mode.

    References:
        :cite:t:`smirnov1948table`; :cite:t:`kuiper1960tests`;
        :cite:t:`maag1968vnm`; :cite:t:`pettitt1976two`;
        :cite:t:`scholz1987k`; :cite:t:`vallender1974calculation`;
        :cite:t:`rabin2011transportation`; :cite:t:`mardia2000directional`.
    """
    first_array, first_weights = _read_samples(first, first_log_weights, space)
    second_array, second_weights = _read_samples(second, second_log_weights, space)
    if _is_concrete(first_array, first_weights, second_array, second_weights):
        _warn_about_unresolved_circles(
            jnp.concatenate([first_array, second_array]),
            _pooled_log_weights(first_weights, second_weights),
            space,
            consequence=(
                "the Anderson-Darling cut opposite the pooled circular mean may"
                " split a mode"
            ),
        )
    return _marginal_distances(
        first_array, first_weights, second_array, second_weights, space
    )


def shared_bandwidth_factor(
    first_log_weights: Float[ArrayLike, "*leading sample"],
    second_log_weights: Float[ArrayLike, "*leading other"],
) -> Float[Array, " *leading"]:
    """Return the bandwidth factor that both marginal estimates of a pair share.

    The factor is Silverman's rule in one dimension, (3n/4)^(−1/5), at the
    smaller Kish effective sample size n of the two posteriors. Fix it once
    per comparison and reuse it in every null draw, through
    ``KernelDensitySettings(bandwidth=factor)``, whose JSON then fills the
    ``estimator_settings`` column of the rows.

    Args:
        first_log_weights: Log weights of the first posterior, shaped (..., N).
        second_log_weights: Log weights of the second posterior, shaped
            (..., M).

    Returns:
        The factor, shaped (...).

    References:
        :cite:t:`silverman1986density`; :cite:t:`kish1965survey`.
    """
    size = jnp.minimum(
        kish_effective_sample_size(jnp.asarray(first_log_weights)),
        kish_effective_sample_size(jnp.asarray(second_log_weights)),
    )
    return jnp.power(0.75 * size, -0.2)


def kde_divergences(
    first: KernelDensity, second: KernelDensity, *, panel_count: int | None = None
) -> KdeDivergences:
    """Return the JS divergence and the Hellinger distance between two estimates.

    Both estimates describe one parameter of the same one-dimensional space.
    The integrals use Gauss-Legendre panels over the reach of each estimate,
    as the module describes.

    Args:
        first: Estimate of the first posterior.
        second: Estimate of the second posterior.
        panel_count: Equal panels per estimate. ``None`` chooses enough for
            panels of at most two bandwidths when the estimates are concrete,
            and :data:`PANEL_COUNT` inside ``jax.jit``, where wider panels give
            NaN.

    Returns:
        The divergence in nats and the distance, each shaped ().

    Raises:
        ValueError: If the estimates are not one-dimensional in one space, or
            ``panel_count`` is not positive.

    References:
        :cite:t:`lin1991divergence`; :cite:t:`hellinger1909neue`;
        :cite:t:`davis1984methods`.
    """
    if len(first.space) != 1 or first.space != second.space:
        raise ValueError(
            "both estimates must describe one parameter of the same space, got"
            f" {first.space.names} and {second.space.names}"
        )
    if panel_count is None:
        panel_count = _panel_count(first, second)
    if panel_count < 1:
        raise ValueError(f"panel_count must be positive, got {panel_count}")
    return _kde_divergences(first, second, panel_count)


def marginal_kde_divergences(
    first: ArrayLike,
    first_log_weights: ArrayLike | None,
    second: ArrayLike,
    second_log_weights: ArrayLike | None,
    space: ParamSpace,
    *,
    factor: float | None = None,
) -> KdeDivergences:
    """Return the JS divergence and the Hellinger distance of every marginal.

    Each parameter gets two one-dimensional
    :class:`~mixed_signals.estimators.kde.KernelDensity` estimates with the
    shared bandwidth factor, on its own topology, and :func:`kde_divergences`
    compares them. Fit concrete samples, outside ``jax.jit``: the factor is
    a setting of the estimates.

    Args:
        first: Samples of the first posterior, shaped (N, D), in the
            parameter order of ``space``.
        first_log_weights: Its log weights shaped (N,); equal weights when
            ``None``.
        second: Samples of the second posterior, shaped (M, D).
        second_log_weights: Its log weights shaped (M,); equal weights when
            ``None``.
        space: Space of the samples.
        factor: Bandwidth factor of both estimates; :func:`shared_bandwidth_factor`
            of the samples when ``None``.

    Returns:
        The divergences, each shaped (D,).

    Raises:
        ValueError: If the shapes do not fit the space, concrete values are
            invalid, or the samples are traced and no factor is given.
        TypeError: If an input is not real-valued.

    Warns:
        ValidityWarning: If an estimate is NaN or has fewer effective samples
            than Silverman's criterion asks for.

    References:
        :cite:t:`silverman1986density`; :cite:t:`jones1996simple`;
        :cite:t:`lin1991divergence`; :cite:t:`hellinger1909neue`.
    """
    first_array, first_weights = _read_samples(first, first_log_weights, space)
    second_array, second_weights = _read_samples(second, second_log_weights, space)
    if factor is None:
        if not _is_concrete(first_weights, second_weights):
            raise ValueError("pass the bandwidth factor when the samples are traced")
        factor = float(shared_bandwidth_factor(first_weights, second_weights))
    settings = KernelDensitySettings(bandwidth=factor)
    divergences = []
    for position, parameter in enumerate(space.parameters):
        marginal_space = ParamSpace(parameters=(parameter,))
        estimates = [
            KernelDensity.fit(
                samples[:, position : position + 1], weights, marginal_space, settings
            )
            for samples, weights in (
                (first_array, first_weights),
                (second_array, second_weights),
            )
        ]
        divergences.append(kde_divergences(*estimates))
    return KdeDivergences(
        js=jnp.stack([values.js for values in divergences]),
        hellinger=jnp.stack([values.hellinger for values in divergences]),
    )


# * Kernels on the sorted steps of two weighted distribution functions.


def _distribution_steps(
    first: Float[Array, "*leading sample"],
    first_log_weights: Float[Array, "*leading sample"],
    second: Float[Array, "*leading other"],
    second_log_weights: Float[Array, "*leading other"],
) -> _Steps:
    """Return the pooled sorted values, F − G and H after each, and the run ends."""
    values = jnp.concatenate([first, second], axis=-1)
    first_weights = jax.nn.softmax(first_log_weights, axis=-1)
    second_weights = jax.nn.softmax(second_log_weights, axis=-1)
    signed = jnp.concatenate([first_weights, -second_weights], axis=-1)
    pooled = 0.5 * jnp.concatenate([first_weights, second_weights], axis=-1)
    order = jnp.argsort(values, axis=-1)
    sorted_values = jnp.take_along_axis(values, order, axis=-1)
    # * Inside a run of equal values the partial sums are not values of F − G.
    run_ends = jnp.concatenate(
        [
            sorted_values[..., 1:] != sorted_values[..., :-1],
            jnp.ones_like(sorted_values[..., :1], dtype=bool),
        ],
        axis=-1,
    )
    return _Steps(
        values=sorted_values,
        gaps=jnp.cumsum(jnp.take_along_axis(signed, order, axis=-1), axis=-1),
        pooled=jnp.cumsum(jnp.take_along_axis(pooled, order, axis=-1), axis=-1),
        run_ends=run_ends,
    )


def _ks_from_steps(steps: _Steps) -> Float[Array, " *leading"]:
    return jnp.max(jnp.where(steps.run_ends, jnp.abs(steps.gaps), 0.0), axis=-1)


def _kuiper_from_steps(steps: _Steps) -> Float[Array, " *leading"]:
    # * The fill value 0 is F − G before the first value, which V includes.
    gaps = jnp.where(steps.run_ends, steps.gaps, 0.0)
    return jnp.max(gaps, axis=-1) - jnp.min(gaps, axis=-1)


def _anderson_darling_from_steps(steps: _Steps) -> Float[Array, " *leading"]:
    # * H at the end of the previous run, so ΔH is the pooled mass of each run.
    at_ends = jnp.where(steps.run_ends, steps.pooled, 0.0)
    previous = jnp.concatenate(
        [
            jnp.zeros_like(at_ends[..., :1]),
            jax.lax.cummax(at_ends, axis=at_ends.ndim - 1)[..., :-1],
        ],
        axis=-1,
    )
    mass = steps.pooled - previous
    denominator = steps.pooled * (1 - steps.pooled)
    # ! The last run ends at H = 1, where rounding leaves 0/0.
    last = jnp.arange(steps.pooled.shape[-1]) == steps.pooled.shape[-1] - 1
    counted = steps.run_ends & ~last & (denominator > 0)
    terms = steps.gaps**2 * mass / jnp.where(counted, denominator, 1.0)
    return jnp.sum(jnp.where(counted, terms, 0.0), axis=-1)


def _wasserstein_from_steps(steps: _Steps) -> Float[Array, " *leading"]:
    widths = steps.values[..., 1:] - steps.values[..., :-1]
    return jnp.sum(jnp.abs(steps.gaps[..., :-1]) * widths, axis=-1)


def _circle_wasserstein_from_steps(
    steps: _Steps, period: Float[Array, ""]
) -> Float[Array, " *leading"]:
    # * The arc from the last value back to the first closes the circle, with F − G = 0.
    lengths = jnp.concatenate(
        [
            steps.values[..., 1:] - steps.values[..., :-1],
            steps.values[..., :1] + period - steps.values[..., -1:],
        ],
        axis=-1,
    )
    levels = jnp.concatenate(
        [steps.gaps[..., :-1], jnp.zeros_like(steps.gaps[..., :1])], axis=-1
    )
    order = jnp.argsort(levels, axis=-1)
    sorted_levels = jnp.take_along_axis(levels, order, axis=-1)
    covered = jnp.cumsum(jnp.take_along_axis(lengths, order, axis=-1), axis=-1)
    total = covered[..., -1:]
    median_index = jnp.argmax(covered >= 0.5 * total, axis=-1)
    median = jnp.take_along_axis(sorted_levels, median_index[..., None], axis=-1)
    return jnp.sum(lengths * jnp.abs(levels - median), axis=-1)


# * Per-parameter distances and kernel density divergences.


def _marginal_distances_kernel(
    first: Float[Array, "sample param"],
    first_log_weights: Float[Array, " sample"],
    second: Float[Array, "other param"],
    second_log_weights: Float[Array, " other"],
    space: ParamSpace,
) -> MarginalDistances:
    """Return the KS or Kuiper, AD and W1 distances of every parameter."""
    ks, ad, w1 = [], [], []
    for position, parameter in enumerate(space.parameters):
        first_values, second_values = first[:, position], second[:, position]
        if parameter.topology is Topology.CIRCLE:
            lower, period = (
                parameter.bounds[0],
                parameter.bounds[1] - parameter.bounds[0],
            )
            on_circle = _distribution_steps(
                wrap(first_values, lower, period),
                first_log_weights,
                wrap(second_values, lower, period),
                second_log_weights,
            )
            ks.append(_kuiper_from_steps(on_circle))
            w1.append(_circle_wasserstein_from_steps(on_circle, jnp.asarray(period)))
            centre = circular_mean(
                jnp.concatenate([first_values, second_values]),
                _pooled_log_weights(first_log_weights, second_log_weights),
                period,
                lower,
            )
            cut = _distribution_steps(
                unwrap_about(first_values, centre, period),
                first_log_weights,
                unwrap_about(second_values, centre, period),
                second_log_weights,
            )
            ad.append(_anderson_darling_from_steps(cut))
        else:
            steps = _distribution_steps(
                first_values, first_log_weights, second_values, second_log_weights
            )
            ks.append(_ks_from_steps(steps))
            ad.append(_anderson_darling_from_steps(steps))
            w1.append(_wasserstein_from_steps(steps))
    return MarginalDistances(ks=jnp.stack(ks), ad=jnp.stack(ad), w1=jnp.stack(w1))


def _pooled_log_weights(
    first_log_weights: Float[Array, " sample"],
    second_log_weights: Float[Array, " other"],
) -> Float[Array, " _"]:
    """Return log weights of the two samples pooled with half the mass each."""
    return jnp.concatenate(
        [
            jax.nn.log_softmax(first_log_weights) - _LOG_TWO,
            jax.nn.log_softmax(second_log_weights) - _LOG_TWO,
        ]
    )


class _Reach(NamedTuple):
    """Where an estimate puts its mass: an interval of the axis and its bandwidth."""

    lower: Array
    upper: Array
    bandwidth: Array


def _reach(estimator: KernelDensity) -> _Reach:
    """Return the samples' range plus IMAGE_REACH bandwidths, inside the support."""
    parameter = estimator.space.parameters[0]
    data = estimator.kde.dataset[0]
    positive = estimator.kde.weights > 0
    bandwidth = jnp.sqrt(estimator.kde.covariance[0, 0])
    lower = jnp.min(jnp.where(positive, data, jnp.inf)) - IMAGE_REACH * bandwidth
    upper = jnp.max(jnp.where(positive, data, -jnp.inf)) + IMAGE_REACH * bandwidth
    if parameter.topology is Topology.CIRCLE:
        period = parameter.bounds[1] - parameter.bounds[0]
        # * Unwrapped samples lie within half a period of the centre.
        wide = upper - lower > period
        centre = estimator.centres[0]
        lower = jnp.where(wide, centre - 0.5 * period, lower)
        upper = jnp.where(wide, centre + 0.5 * period, upper)
    elif parameter.topology is Topology.INTERVAL:
        lower = jnp.maximum(lower, parameter.bounds[0])
        upper = jnp.minimum(upper, parameter.bounds[1])
    return _Reach(lower=lower, upper=upper, bandwidth=bandwidth)


def _panel_count(first: KernelDensity, second: KernelDensity) -> int:
    """Return the panels per estimate that keep every panel within two bandwidths."""
    reaches = [_reach(first), _reach(second)]
    if not _is_concrete(*(reach.bandwidth for reach in reaches)):
        return PANEL_COUNT
    widths = [PANEL_BANDWIDTHS * float(reach.bandwidth) for reach in reaches]
    # ! An estimate without spread has no width; its divergences come out NaN anyway.
    if not all(width > 0 for width in widths):
        return PANEL_COUNT
    needed = [
        (float(reach.upper) - float(reach.lower)) / width
        for reach, width in zip(reaches, widths, strict=True)
    ]
    return max(PANEL_COUNT, math.ceil(max(needed)))


def _kde_divergences_kernel(
    first: KernelDensity, second: KernelDensity, panel_count: int
) -> KdeDivergences:
    """Integrate the JS and squared Hellinger integrands on the merged panels."""
    parameter = first.space.parameters[0]
    reaches = [_reach(first), _reach(second)]
    steps = jnp.linspace(0.0, 1.0, panel_count + 1)
    edges = jnp.concatenate(
        [reach.lower + (reach.upper - reach.lower) * steps for reach in reaches]
    )
    if parameter.topology is Topology.CIRCLE:
        lower, upper = parameter.bounds
        # * An estimate's panels may cross the cut; the merged ones cover one period.
        edges = jnp.concatenate(
            [wrap(edges, lower, upper - lower), jnp.asarray([lower, upper])]
        )
    edges = jnp.sort(edges)
    half_widths = 0.5 * jnp.diff(edges)
    midpoints = 0.5 * (edges[:-1] + edges[1:])
    nodes = (midpoints[:, None] + half_widths[:, None] * _LEGENDRE_NODES).reshape(-1, 1)
    node_weights = (half_widths[:, None] * _LEGENDRE_WEIGHTS).reshape(-1)
    log_first = first.log_density(nodes)
    log_second = second.log_density(nodes)
    js = jnp.sum(node_weights * _js_integrand(log_first, log_second))
    hellinger_squared = 0.5 * jnp.sum(
        node_weights * (jnp.exp(0.5 * log_first) - jnp.exp(0.5 * log_second)) ** 2
    )
    resolved = jnp.all(
        jnp.asarray(
            [
                (reach.upper - reach.lower) / panel_count
                <= PANEL_BANDWIDTHS * reach.bandwidth
                for reach in reaches
            ]
        )
    )
    return KdeDivergences(
        js=jnp.where(resolved, jnp.clip(js, 0.0, _LOG_TWO), jnp.nan),
        hellinger=jnp.where(
            resolved, jnp.sqrt(jnp.clip(hellinger_squared, 0.0, 1.0)), jnp.nan
        ),
    )


def _js_integrand(
    log_first: Float[Array, " _"], log_second: Float[Array, " _"]
) -> Float[Array, " _"]:
    """Return ½ p ln(2p/(p + q)) + ½ q ln(2q/(p + q)), zero where a density is."""
    log_sum = jnp.logaddexp(log_first, log_second)
    terms = []
    for log_density in (log_first, log_second):
        positive: Bool[Array, " _"] = log_density > -jnp.inf
        safe = jnp.where(positive, log_density, 0.0)
        terms.append(
            jnp.where(positive, 0.5 * jnp.exp(safe) * (_LOG_TWO + safe - log_sum), 0.0)
        )
    return terms[0] + terms[1]


_marginal_distances = _jit(_marginal_distances_kernel, "space")
_kde_divergences = _jit(_kde_divergences_kernel, "panel_count")
