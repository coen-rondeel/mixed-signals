"""Dependence between the parameters of one posterior, against its Gaussian value.

For every pair of parameters :func:`dependence_statistics` gives Pearson's r
and three statistics that a bivariate Gaussian fixes as functions of r:
Spearman's ρ_s, the distance correlation and the mutual information. Each is
returned as its excess over that Gaussian value at the r of the same
samples, so a banana, whose curved dependence Pearson misses, shows up as an
excess. The surrogate null of :mod:`mixed_signals.nulls.surrogates` then
carries the finite-sample bias of each estimator. A raw statistic would sit
at the centre of its surrogate null, because the surrogates are drawn at the
sample's own correlation; its excess is calibrated against them (zoo check,
see :ref:`validation-surrogate-calibration`).

Pearson's r
    The correlation of the weighted covariance, informational: the
    surrogates reproduce it by construction.
Spearman's ρ_s
    The weighted Pearson correlation of the mid-ranks of the weighted
    empirical distribution functions, F(x⁻) + w(x)/2, which for equal weights
    is Spearman's (1904) rank correlation with average ranks for ties. No
    weighted Spearman correlation is accepted in print, so the mid-ranks are
    this package's plug-in choice. Its Gaussian value is (6/π) arcsin(r/2)
    (Pearson 1907, equation xviii; Moran 1948). It catches monotone
    non-linear dependence, but not a symmetric banana.
Distance correlation
    The distance correlation of the weighted empirical distribution, from
    the distance covariance that Lyons (2013) defines for any probability
    measure, with absolute differences on lines and intervals and chords on
    circles; for equal weights it is the V-statistic of Székely, Rizzo and
    Bakirov (2007). Chords keep the strong negative type that makes the
    distance covariance characterise independence (Lyons 2013). Its
    Gaussian value is Theorem 7 of Székely, Rizzo and Bakirov (2007).
Mutual information
    The estimator I⁽¹⁾ of Kraskov, Stögbauer and Grassberger (2004),
    ψ(k) + ψ(N) − ⟨ψ(nₓ + 1) + ψ(n_y + 1)⟩, with the k-th neighbour of each
    sample in the maximum norm of the pair and the counts nₓ and n_y of
    samples strictly closer in each parameter, along arcs on circles, from
    :class:`~mixed_signals.estimators.knn.NearestNeighbours`. Its Gaussian
    value is −½ ln(1 − r²) (Cover and Thomas 2006). It needs equally weighted
    samples, so unequal weights give NaN, as for every nearest-neighbour
    estimate. Unlike the others, it does not change under a monotone map of
    either parameter, such as eryn's map from sampling to physical
    coordinates.
Rank and cap
    The default rank :data:`MUTUAL_INFORMATION_NEIGHBOURS` and the cap
    :data:`PAIR_CAP` come from a zoo check on bananas whose mutual
    information is known by quadrature; see the constants,
    :ref:`validation-ksg-bananas` and :ref:`validation-costs`.
Circles
    Pearson's r, Spearman's ρ_s and the r of the Gaussian values use values
    unwrapped about the circular mean. Pairs that involve a circle wider than
    :data:`~mixed_signals.diagnostics._common.CIRCLE_WIDTH_LIMIT` rad raise a
    :class:`~mixed_signals.errors.ValidityWarning`, because their surrogate
    null is approximate there. Sphere axes follow their own topologies.

References:
    :cite:t:`spearman1904proof`; :cite:t:`pearson1907mathematical`;
    :cite:t:`moran1948rank`; :cite:t:`szekely2007measuring`;
    :cite:t:`lyons2013distance`; :cite:t:`kraskov2004mutual`;
    :cite:t:`cover2006elements`; :cite:t:`efron1993introduction`.
"""

import functools
import math
from typing import Final, NamedTuple

import jax
import jax.numpy as jnp
from jax.scipy.special import digamma
from jax.typing import ArrayLike
from jaxtyping import Array, Float, PRNGKeyArray

from mixed_signals.core.samples import _is_concrete, weighted_covariance
from mixed_signals.core.space import Parameter, ParamSpace
from mixed_signals.diagnostics._common import (
    _jit,
    _warn,
    circle_positions,
    parameter_pairs,
    period_of,
    warn_about_wide_circles,
    wide_circles,
)
from mixed_signals.estimators.base import CHUNK_ENTRIES, _read_samples, _unwrap_circles
from mixed_signals.estimators.knn import (
    WEIGHT_TOLERANCE,
    NearestNeighbours,
    NearestNeighbourSettings,
)
from mixed_signals.metrics._cap import cap_samples

MUTUAL_INFORMATION_NEIGHBOURS: Final = 10
"""Default rank k of the mutual-information estimator.

On the zoo's bananas, whose mutual information is known by quadrature,
k = 10 resolved a banana of 0.08 nats with 1.7 times the null widths of
k = 3 and read within −9 to +4% of the truth at 2 000 samples (−26% at
1 000 for the faint banana, within ±5% for strong ones; see
:ref:`validation-ksg-bananas`).
"""

PAIR_CAP: Final = 2_000
"""Largest number of samples per posterior for the pairwise O(N²) statistics.

The distance correlation and the mutual information run over D(D − 1)/2
pairs and every surrogate set, so they subsample above this cap: with 201
sets, 16 parameters take about 20 minutes on one CPU at the cap, against
about an hour at 4 000 samples (see :ref:`validation-costs`).
"""

_DCOR_NORMALISER: Final = 1 + math.pi / 3 - math.sqrt(3)


class Dependence(NamedTuple):
    """Dependence of every parameter pair, in the order of :func:`parameter_pairs`."""

    pearson: Float[Array, " pair"]
    """Pearson's r of the weighted samples, informational."""
    spearman: Float[Array, " pair"]
    """Spearman's ρ_s minus its Gaussian value (6/π) arcsin(r/2)."""
    distance_correlation: Float[Array, " pair"]
    """Distance correlation minus its Gaussian value at r."""
    mutual_information: Float[Array, " pair"]
    """KSG mutual information in nats minus its Gaussian value −½ ln(1 − r²)."""


def gaussian_spearman(
    correlation: Float[ArrayLike, " *leading"],
) -> Float[Array, " *leading"]:
    """Return Spearman's ρ_s of a bivariate Gaussian, (6/π) arcsin(r/2).

    References:
        :cite:t:`pearson1907mathematical`; :cite:t:`moran1948rank`.
    """
    values = jnp.asarray(correlation, dtype=jnp.float64)
    return 6 / jnp.pi * jnp.arcsin(values / 2)


def gaussian_distance_correlation(
    correlation: Float[ArrayLike, " *leading"],
) -> Float[Array, " *leading"]:
    """Return the distance correlation of a bivariate Gaussian of correlation r.

    Theorem 7 of Székely, Rizzo and Bakirov (2007):
    R² = (r arcsin r + √(1 − r²) − r arcsin(r/2) − √(4 − r²) + 1)/(1 + π/3 − √3),
    returned as R.

    References:
        :cite:t:`szekely2007measuring`.
    """
    values = jnp.abs(jnp.asarray(correlation, dtype=jnp.float64))
    squared = (
        values * jnp.arcsin(values)
        + jnp.sqrt(1 - values**2)
        - values * jnp.arcsin(values / 2)
        - jnp.sqrt(4 - values**2)
        + 1
    ) / _DCOR_NORMALISER
    return jnp.sqrt(jnp.maximum(squared, 0.0))


def gaussian_mutual_information(
    correlation: Float[ArrayLike, " *leading"],
) -> Float[Array, " *leading"]:
    """Return the mutual information −½ ln(1 − r²) of a bivariate Gaussian, in nats.

    References:
        :cite:t:`cover2006elements`.
    """
    values = jnp.asarray(correlation, dtype=jnp.float64)
    return -0.5 * jnp.log1p(-(values**2))


def dependence_statistics(
    samples: ArrayLike,
    log_weights: ArrayLike | None,
    space: ParamSpace,
    *,
    key: PRNGKeyArray,
    cap: int = PAIR_CAP,
    neighbours: int = MUTUAL_INFORMATION_NEIGHBOURS,
) -> Dependence:
    """Return Pearson's r and the excess dependence of every parameter pair.

    Args:
        samples: Samples of one posterior, shaped (N, D), with D ≥ 2.
        log_weights: Their log weights, shaped (N,); equal weights when
            ``None``.
        space: Parameter space of the samples.
        key: JAX PRNG key for the subsample above the cap, derived from the
            row's path ending in ``"subsample"``.
        cap: Largest number of samples; above it a uniform subsample runs.
        neighbours: Rank k of the mutual-information estimator.

    Returns:
        Pearson's r and the excesses of Spearman's ρ_s, the distance
        correlation and the mutual information over their Gaussian values,
        each shaped (D(D − 1)/2,) in the order of :func:`parameter_pairs`;
        the mutual information is NaN for unequal weights.

    Raises:
        ValueError: If the shapes do not fit the space, the space has fewer
            than two parameters, concrete values are invalid, or ``cap`` is
            below one.
        TypeError: If an input is not real-valued.

    Warns:
        SubsamplingWarning: If N exceeds the cap.
        ValidityWarning: If the positive weights are unequal, too few samples
            have a positive weight for the rank, or a circle is wider than
            the limit of the surrogate null.
        PrecisionWarning: If an input is narrower than float64.

    References:
        :cite:t:`spearman1904proof`; :cite:t:`szekely2007measuring`;
        :cite:t:`kraskov2004mutual`.
    """
    if len(space) < 2:
        raise ValueError(
            f"dependence needs at least two parameters, the space has {len(space)}"
        )
    samples_array, weights_array = _read_samples(samples, log_weights, space)
    samples_array, weights_array = cap_samples(
        samples_array, weights_array, key=key, metric="dependence", cap=cap
    )
    unwrapped, _ = _unwrap_circles(samples_array, weights_array, space)
    pearson, spearman = _rank_statistics(unwrapped, weights_array)
    distance = _distance_correlation(samples_array, weights_array, space=space)
    information = jnp.stack(
        [
            _mutual_information(
                samples_array[:, jnp.array([first, second])],
                weights_array,
                pair_space=_pair_space(space, first, second),
                neighbours=neighbours,
            )
            for first, second in parameter_pairs(len(space))
        ]
    )
    if _is_concrete(samples_array, weights_array):
        _warn_about_weights(weights_array, neighbours)
        warn_about_wide_circles(
            wide_circles(samples_array, weights_array, space), "the dependence rows"
        )
    return Dependence(
        pearson=pearson,
        spearman=spearman - gaussian_spearman(pearson),
        distance_correlation=distance - gaussian_distance_correlation(pearson),
        mutual_information=information - gaussian_mutual_information(pearson),
    )


def _pair_space(space: ParamSpace, first: int, second: int) -> ParamSpace:
    """Return the space of two parameters, each in its own topology, with fixed names.

    Fixed names let every pair of the same topologies share one compiled search.
    """
    return ParamSpace(
        parameters=tuple(
            Parameter(
                name=name,
                topology=space.parameters[position].topology,
                bounds=space.parameters[position].bounds,
            )
            for name, position in (("first", first), ("second", second))
        )
    )


def _line_space(parameter: Parameter) -> ParamSpace:
    """Return the space of one parameter with a fixed name."""
    return ParamSpace(
        parameters=(
            Parameter(
                name="value", topology=parameter.topology, bounds=parameter.bounds
            ),
        )
    )


def _warn_about_weights(log_weights: Array, neighbours: int) -> None:
    """Warn once when the mutual information is NaN for its weights or sample count."""
    present = log_weights > -jnp.inf
    positive = log_weights[present]
    if float(jnp.max(positive) - jnp.min(positive)) > WEIGHT_TOLERANCE:
        _warn(
            "the mutual information needs equally weighted samples, so it is NaN;"
            " resample the set in proportion to its weights first"
        )
    elif int(jnp.sum(present)) <= neighbours:
        _warn(
            f"the mutual information with k = {neighbours} needs more than k samples"
            f" with positive weight, got {int(jnp.sum(present))}, so it is NaN"
        )


def _mutual_information_kernel(
    samples: Float[Array, "sample param"],
    log_weights: Float[Array, " sample"],
    *,
    pair_space: ParamSpace,
    neighbours: int,
) -> Float[Array, ""]:
    """Return the KSG estimate I⁽¹⁾ of one pair in nats; traced, so it never warns."""
    settings = NearestNeighbourSettings(neighbours=neighbours, norm="max")
    joint = NearestNeighbours.fit(samples, log_weights, pair_space, settings)
    radii = joint.leave_one_out_distance()
    counts = [
        NearestNeighbours.fit(
            samples[:, [position]],
            log_weights,
            _line_space(pair_space.parameters[position]),
            settings,
        ).leave_one_out_count(radii)
        for position in range(2)
    ]
    present = joint.present
    count = jnp.sum(present)
    terms = digamma(counts[0] + 1) + digamma(counts[1] + 1)
    mean_terms = jnp.sum(jnp.where(present, terms, 0.0)) / count
    return (
        digamma(float(neighbours))
        + digamma(jnp.asarray(count, dtype=jnp.float64))
        - mean_terms
    )


def _mid_ranks(
    values: Float[Array, " sample"], weights: Float[Array, " sample"]
) -> Float[Array, " sample"]:
    """Return F(x⁻) + w(x)/2 of the weighted empirical distribution at each value."""
    order = jnp.argsort(values)
    ordered = values[order]
    cumulative = jnp.concatenate([jnp.zeros(1), jnp.cumsum(weights[order])])
    # ! searchsorted gives int32; indexing with it is safe, arithmetic would not be.
    below = cumulative[jnp.searchsorted(ordered, values, side="left")]
    through = cumulative[jnp.searchsorted(ordered, values, side="right")]
    return 0.5 * (below + through)


def _correlations(
    values: Float[Array, "sample param"], log_weights: Float[Array, " sample"]
) -> Float[Array, " pair"]:
    """Return the weighted Pearson correlation of every pair of columns."""
    _, covariance = weighted_covariance(values, log_weights)
    widths = jnp.sqrt(jnp.diagonal(covariance))
    correlation = covariance / (widths[:, None] * widths[None, :])
    rows, columns = zip(*parameter_pairs(values.shape[1]), strict=True)
    return correlation[jnp.asarray(rows), jnp.asarray(columns)]


def _rank_statistics_kernel(
    unwrapped: Float[Array, "sample param"], log_weights: Float[Array, " sample"]
) -> tuple[Float[Array, " pair"], Float[Array, " pair"]]:
    """Return Pearson's r and Spearman's ρ_s of every pair."""
    weights = jax.nn.softmax(log_weights)
    ranks = jax.vmap(_mid_ranks, in_axes=(1, None), out_axes=1)(unwrapped, weights)
    return _correlations(unwrapped, log_weights), _correlations(ranks, log_weights)


def _parameter_distances(
    space: ParamSpace,
    rows: Float[Array, "*leading param"],
    samples: Float[Array, "sample param"],
) -> Float[Array, "param *leading sample"]:
    """Return the absolute difference of each parameter, or the chord on a circle.

    The chord of a difference Δ on a circle of period P is (P/π) sin(π|Δ|/P).
    """
    differences = rows[:, None, :] - samples[None, :, :]
    distances = jnp.abs(differences)
    for position in circle_positions(space):
        period = period_of(space, position)
        chords = (
            period
            / jnp.pi
            * jnp.abs(jnp.sin(jnp.pi * differences[..., position] / period))
        )
        distances = distances.at[..., position].set(chords)
    return jnp.moveaxis(distances, -1, 0)


def _distance_sums(
    space: ParamSpace,
    samples: Float[Array, "sample param"],
    weights: Float[Array, " sample"],
    chunk: tuple[Array, Array],
) -> tuple[Array, Array]:
    """Return the weighted row means and Σ wᵢwⱼ aᵢⱼbᵢⱼ of one chunk of rows."""
    rows, row_weights = chunk
    distances = _parameter_distances(space, rows, samples)
    row_means = jnp.einsum("prs,s->pr", distances, weights)
    products = jnp.einsum("prs,qrs,r,s->pq", distances, distances, row_weights, weights)
    return row_means, products


def _distance_correlation_kernel(
    samples: Float[Array, "sample param"],
    log_weights: Float[Array, " sample"],
    *,
    space: ParamSpace,
) -> Float[Array, " pair"]:
    """Return the weighted V-statistic distance correlation of every pair."""
    weights = jax.nn.softmax(log_weights)
    sample_count, size = samples.shape
    chunk_size = max(1, min(sample_count, CHUNK_ENTRIES // (size * sample_count)))
    chunk_count = -(-sample_count // chunk_size)
    padding = chunk_count * chunk_size - sample_count
    padded = jnp.concatenate([samples, jnp.broadcast_to(samples[:1], (padding, size))])
    padded_weights = jnp.concatenate([weights, jnp.zeros(padding)])
    row_means, products = jax.lax.map(
        functools.partial(_distance_sums, space, samples, weights),
        (
            padded.reshape(chunk_count, chunk_size, size),
            padded_weights.reshape(chunk_count, chunk_size),
        ),
    )
    means = jnp.moveaxis(row_means, 0, 1).reshape(size, -1)[:, :sample_count]
    grand = means @ weights
    # * With centred matrices, Σ wᵢwⱼAᵢⱼBᵢⱼ = Σ wᵢwⱼaᵢⱼbᵢⱼ − 2 Σ wᵢāᵢb̄ᵢ + āb̄.
    covariance = (
        jnp.sum(products, axis=0)
        - 2 * jnp.einsum("pi,qi,i->pq", means, means, weights)
        + grand[:, None] * grand[None, :]
    )
    variances = jnp.diagonal(covariance)
    rows, columns = zip(*parameter_pairs(size), strict=True)
    row_index, column_index = jnp.asarray(rows), jnp.asarray(columns)
    squared = covariance[row_index, column_index] / jnp.sqrt(
        variances[row_index] * variances[column_index]
    )
    return jnp.sqrt(jnp.maximum(squared, 0.0))


_rank_statistics = _jit(_rank_statistics_kernel)
_distance_correlation = _jit(_distance_correlation_kernel, "space")
_mutual_information = _jit(_mutual_information_kernel, "pair_space", "neighbours")
