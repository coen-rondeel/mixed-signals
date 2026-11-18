"""Nearest-neighbour estimates of the KL and JS divergences and the Hellinger distance.

Every estimate builds on
:class:`~mixed_signals.estimators.knn.NearestNeighbours`: its topology-aware
k-th neighbour distances and its digamma-corrected log densities, in the
d = E + C + 2S dimensions of a space with E Euclidean axes, C circles and S
spheres. Each function compares two posteriors of one problem and first caps
both at :data:`~mixed_signals._config.SAMPLE_CAP` samples with
:func:`~mixed_signals.metrics._cap.cap_pair`, because brute-force searches
cost O(N²). The rows read the metrics ``kl``, ``js`` and ``hellinger`` with the
estimator ``knn``.

Kullback-Leibler divergence
    :func:`knn_kl_divergence` is the estimator of Wang, Kulkarni and Verdú
    (2009), KL(p ‖ q) ≈ (d/n) Σᵢ ln(νₖ(i)/ρₖ(i)) + ln(m/(n − 1)), with ρₖ(i)
    the distance from the i-th of the n samples of p to its k-th nearest
    other sample of p, and νₖ(i) the distance to its k-th nearest of the m
    samples of q. Swapping the posteriors gives the other direction.
Jensen-Shannon divergence
    :func:`knn_js_divergence` combines entropy estimates,
    JS = H(m) − ½ H(p) − ½ H(q) in nats (Lin 1991), with the pooled samples
    standing for the mixture m = (p + q)/2. Each entropy is the mean of the
    negated leave-one-out log densities, the estimate of Kozachenko and
    Leonenko (1987) in the form of Kraskov, Stögbauer and Grassberger (2004).
    The pooled samples are a draw from m only when both posteriors have as
    many samples, so the larger one is first subsampled, uniformly without
    replacement, to the number of positive-weight samples of the smaller.
Hellinger distance
    Póczos and Schneider (2011) estimate D_α(p ‖ q) = ∫ p^α q^(1−α) as
    B_{k,α} (1/n) Σᵢ [(n − 1) ρₖ(i)^d / (m νₖ(i)^d)]^(1−α), with
    B_{k,α} = Γ(k)² / (Γ(k − α + 1) Γ(k + α − 1)). At α = ½ it is the
    Bhattacharyya coefficient BC, and the Rényi divergence of order ½ is
    −2 ln BC (Rényi 1961), so H² = 1 − BC. The estimator uses the samples of
    one posterior; :func:`knn_hellinger_distance` averages the estimates from
    the samples of p and of q, so the value does not depend on the order of
    the pair. Póczos and Schneider prove L2 consistency at α = ½ for k ≥ 3.
Values near zero
    Estimates are returned as they are. KL and JS fall below zero when
    noise outweighs a small divergence, and the Hellinger estimate is
    sign(1 − B̂C) √|1 − B̂C|, whose signed square keeps 1 − B̂C. Null draws
    therefore keep their full spread.
Zoo check
    On shifted unit Gaussians in 1 to 16 dimensions, with 3 000 and 10 000
    samples per posterior and 40 draws per case, the null-subtracted value of
    a shift of 1σ or 2σ against the truth, and the resolution of a 1σ shift,
    its true value over the null spread, gave the defaults
    :data:`KL_NEIGHBOURS`, :data:`JS_NEIGHBOURS` and
    :data:`HELLINGER_NEIGHBOURS`. With these defaults the JS estimate stayed
    within −1 to +3% of the truth up to D = 8 and +2 to +10% at D = 16, and
    the Hellinger estimate of H² within 0 to −9%. The KL estimate read low
    from four dimensions for every k: −9 to −17% at D = 4 and −29 to −32% at
    D = 16 with k = 3. The bias-reduced form of Wang, Kulkarni and Verdú (2009),
    with neighbour counts chosen per sample as kldest implements it
    (Hartung 2024), did not remove it, so the σ_eff of kNN KL rows needs a
    zoo calibration (see :ref:`validation-knn-ranks` and
    :ref:`validation-knn-kl-calibration`).
Weights
    The estimators assume equally weighted, independent draws (Wang,
    Kulkarni and Verdú 2009). Samples of zero weight leave every search;
    unequal positive weights give NaN with a
    :class:`~mixed_signals.errors.ValidityWarning`, so resample a weighted
    set in proportion to its weights first (Smith and Gelfand 1992).
Validity
    A posterior needs more samples of positive weight than k. A k-th
    neighbour distance of zero, from at least k coinciding samples such as
    the repeated states of a Markov chain, makes an estimate NaN, with a
    warning for concrete inputs.

References:
    :cite:t:`wang2009divergence`; :cite:t:`hartung2024kldest`;
    :cite:t:`lin1991divergence`; :cite:t:`kozachenko1987sample`;
    :cite:t:`kraskov2004mutual`; :cite:t:`poczos2011estimation`;
    :cite:t:`renyi1961measures`; :cite:t:`hellinger1909neue`;
    :cite:t:`bhattacharyya1943measure`; :cite:t:`smith1992bayesian`;
    :cite:t:`politis1999subsampling`.
"""

import math
from typing import Final

import jax.numpy as jnp
from jax.typing import ArrayLike
from jaxtyping import Array, Bool, Float, PRNGKeyArray

from mixed_signals._config import SAMPLE_CAP
from mixed_signals.core.samples import _is_concrete
from mixed_signals.core.space import ParamSpace
from mixed_signals.estimators.base import _jit, _read_samples, _warn
from mixed_signals.estimators.knn import NearestNeighbours, NearestNeighbourSettings
from mixed_signals.metrics._cap import _draw, cap_pair
from mixed_signals.prng import derive_key

KL_NEIGHBOURS: Final = 3
"""Default rank k of :func:`knn_kl_divergence`.

In the zoo check of the module description (see
:ref:`validation-knn-ranks`), k = 3 resolved a shift about twice as well as
k = 1, for 1 to 6 percentage points more bias.
"""

JS_NEIGHBOURS: Final = 10
"""Default rank k of :func:`knn_js_divergence`.

In the zoo check of the module description (see
:ref:`validation-knn-ranks`), k = 10 resolved a shift 1.9 times as well as
k = 3, for at most 3 percentage points more bias.
"""

HELLINGER_NEIGHBOURS: Final = 5
"""Default rank k of :func:`knn_hellinger_distance`.

In the zoo check of the module description (see
:ref:`validation-knn-ranks`), k = 5 resolved a shift 1.4 times
as well as k = 3, the smallest k with proven L2 consistency at α = ½
(Póczos and Schneider 2011), for at most 2 percentage points more bias;
k = 1 resolved it about six times worse.

References:
    :cite:t:`poczos2011estimation`.
"""


def knn_kl_divergence(
    first: ArrayLike,
    first_log_weights: ArrayLike | None,
    second: ArrayLike,
    second_log_weights: ArrayLike | None,
    space: ParamSpace,
    *,
    key: PRNGKeyArray,
    neighbours: int = KL_NEIGHBOURS,
    cap: int = SAMPLE_CAP,
) -> Float[Array, ""]:
    """Return the nearest-neighbour estimate of KL(first ‖ second), in nats.

    Args:
        first: Samples of the first posterior p, shaped (N, D), in the
            parameter order of ``space``.
        first_log_weights: Its log weights shaped (N,); equal weights when
            ``None``.
        second: Samples of the second posterior q, shaped (M, D).
        second_log_weights: Its log weights shaped (M,); equal weights when
            ``None``.
        space: Space of the samples.
        key: JAX PRNG key for the subsample above the cap, derived with
            :func:`~mixed_signals.prng.derive_key` from the path of the row,
            ending in ``"subsample"``.
        neighbours: The rank k of the neighbours.
        cap: Largest number of samples per posterior.

    Returns:
        The estimate, raw: it can fall below zero. NaN where the estimator
        is undefined.

    Raises:
        ValueError: If the shapes do not fit the space, concrete values are
            invalid, or ``neighbours`` or ``cap`` is below one.
        TypeError: If an input is not real-valued.

    Warns:
        ValidityWarning: If concrete positive weights are unequal, too few
            samples have a positive weight, or a k-th neighbour distance is
            zero.
        SubsamplingWarning: If a posterior has more samples than the cap.

    References:
        :cite:t:`wang2009divergence`.
    """
    first_fit, second_fit = _fitted_pair(
        first,
        first_log_weights,
        second,
        second_log_weights,
        space,
        key=key,
        metric="kl",
        neighbours=neighbours,
        cap=cap,
    )
    own = first_fit.leave_one_out_distance()
    cross = second_fit.neighbour_distance(first_fit.samples)
    _warn_about_zero_distances(neighbours, (first_fit.present, own, cross))
    return _kl_from_distances(
        own, cross, first_fit.present, jnp.sum(second_fit.present), first_fit.dimension
    )


def knn_js_divergence(
    first: ArrayLike,
    first_log_weights: ArrayLike | None,
    second: ArrayLike,
    second_log_weights: ArrayLike | None,
    space: ParamSpace,
    *,
    key: PRNGKeyArray,
    neighbours: int = JS_NEIGHBOURS,
    cap: int = SAMPLE_CAP,
) -> Float[Array, ""]:
    """Return the nearest-neighbour estimate of the JS divergence, in nats.

    Both posteriors enter with as many samples: above the cap, each draws
    its subset as :func:`~mixed_signals.metrics._cap.cap_pair` does, and the
    larger one then draws, from ``derive_key(key, "equal", "first")`` or
    ``"second"``, as many samples as the smaller has with positive weight.
    The counts are known only for concrete weights; inside ``jax.jit``
    they must already be equal, or the estimate is NaN.

    Args:
        first: Samples of the first posterior p, shaped (N, D), in the
            parameter order of ``space``.
        first_log_weights: Its log weights shaped (N,); equal weights when
            ``None``.
        second: Samples of the second posterior q, shaped (M, D).
        second_log_weights: Its log weights shaped (M,); equal weights when
            ``None``.
        space: Space of the samples.
        key: JAX PRNG key for the subsamples, derived with
            :func:`~mixed_signals.prng.derive_key` from the path of the row,
            ending in ``"subsample"``.
        neighbours: The rank k of the neighbours.
        cap: Largest number of samples per posterior.

    Returns:
        The estimate, raw: it can fall below zero. NaN where the estimator
        is undefined.

    Raises:
        ValueError: If the shapes do not fit the space, concrete values are
            invalid, or ``neighbours`` or ``cap`` is below one.
        TypeError: If an input is not real-valued.

    Warns:
        ValidityWarning: If concrete positive weights are unequal, too few
            samples have a positive weight, or a k-th neighbour distance is
            zero.
        SubsamplingWarning: If a posterior has more samples than the cap.

    References:
        :cite:t:`lin1991divergence`; :cite:t:`kozachenko1987sample`;
        :cite:t:`kraskov2004mutual`.
    """
    settings = NearestNeighbourSettings(neighbours=neighbours)
    first_array, first_weights, second_array, second_weights = _equal_counts(
        *_read_and_cap(
            first,
            first_log_weights,
            second,
            second_log_weights,
            space,
            key=key,
            metric="js",
            cap=cap,
        ),
        key=derive_key(key, "equal"),
    )
    first_fit = NearestNeighbours.fit(first_array, first_weights, space, settings)
    second_fit = NearestNeighbours.fit(second_array, second_weights, space, settings)
    pooled_present = jnp.concatenate([first_fit.present, second_fit.present])
    # * Within each posterior the present weights are equal, so pooled they all are.
    pooled_fit = NearestNeighbours.fit(
        jnp.concatenate([first_array, second_array]),
        jnp.where(pooled_present, 0.0, -jnp.inf),
        space,
        settings,
    )
    return _js_from_log_densities(
        first_fit.leave_one_out_log_density(),
        first_fit.present,
        second_fit.leave_one_out_log_density(),
        second_fit.present,
        pooled_fit.leave_one_out_log_density(),
    )


def knn_hellinger_distance(
    first: ArrayLike,
    first_log_weights: ArrayLike | None,
    second: ArrayLike,
    second_log_weights: ArrayLike | None,
    space: ParamSpace,
    *,
    key: PRNGKeyArray,
    neighbours: int = HELLINGER_NEIGHBOURS,
    cap: int = SAMPLE_CAP,
) -> Float[Array, ""]:
    """Return the nearest-neighbour estimate of the Hellinger distance.

    The Bhattacharyya coefficient is the mean of the estimates of Póczos
    and Schneider (2011) at α = ½ from the samples of each posterior, so
    swapping the posteriors leaves the value unchanged.

    Args:
        first: Samples of the first posterior, shaped (N, D), in the
            parameter order of ``space``.
        first_log_weights: Its log weights shaped (N,); equal weights when
            ``None``.
        second: Samples of the second posterior, shaped (M, D).
        second_log_weights: Its log weights shaped (M,); equal weights when
            ``None``.
        space: Space of the samples.
        key: JAX PRNG key for the subsample above the cap, derived with
            :func:`~mixed_signals.prng.derive_key` from the path of the row,
            ending in ``"subsample"``.
        neighbours: The rank k of the neighbours.
        cap: Largest number of samples per posterior.

    Returns:
        sign(1 − B̂C) √|1 − B̂C|, at most one and negative when the estimated
        coefficient exceeds one. NaN where the estimator is undefined.

    Raises:
        ValueError: If the shapes do not fit the space, concrete values are
            invalid, or ``neighbours`` or ``cap`` is below one.
        TypeError: If an input is not real-valued.

    Warns:
        ValidityWarning: If concrete positive weights are unequal, too few
            samples have a positive weight, or a k-th neighbour distance is
            zero.
        SubsamplingWarning: If a posterior has more samples than the cap.

    References:
        :cite:t:`poczos2011estimation`; :cite:t:`renyi1961measures`;
        :cite:t:`hellinger1909neue`; :cite:t:`bhattacharyya1943measure`.
    """
    first_fit, second_fit = _fitted_pair(
        first,
        first_log_weights,
        second,
        second_log_weights,
        space,
        key=key,
        metric="hellinger",
        neighbours=neighbours,
        cap=cap,
    )
    first_own = first_fit.leave_one_out_distance()
    first_cross = second_fit.neighbour_distance(first_fit.samples)
    second_own = second_fit.leave_one_out_distance()
    second_cross = first_fit.neighbour_distance(second_fit.samples)
    _warn_about_zero_distances(
        neighbours,
        (first_fit.present, first_own, first_cross),
        (second_fit.present, second_own, second_cross),
    )
    return _hellinger_from_distances(
        first_own,
        first_cross,
        first_fit.present,
        second_own,
        second_cross,
        second_fit.present,
        first_fit.dimension,
        neighbours,
    )


def _read_and_cap(
    first: ArrayLike,
    first_log_weights: ArrayLike | None,
    second: ArrayLike,
    second_log_weights: ArrayLike | None,
    space: ParamSpace,
    *,
    key: PRNGKeyArray,
    metric: str,
    cap: int,
) -> tuple[Array, Array, Array, Array]:
    """Return both posteriors as float64, each capped at ``cap`` samples."""
    first_array, first_weights = _read_samples(first, first_log_weights, space)
    second_array, second_weights = _read_samples(second, second_log_weights, space)
    return cap_pair(
        first_array,
        first_weights,
        second_array,
        second_weights,
        key=key,
        metric=metric,
        cap=cap,
    )


def _fitted_pair(
    first: ArrayLike,
    first_log_weights: ArrayLike | None,
    second: ArrayLike,
    second_log_weights: ArrayLike | None,
    space: ParamSpace,
    *,
    key: PRNGKeyArray,
    metric: str,
    neighbours: int,
    cap: int,
) -> tuple[NearestNeighbours, NearestNeighbours]:
    """Return the nearest-neighbour estimators of both capped posteriors."""
    settings = NearestNeighbourSettings(neighbours=neighbours)
    first_array, first_weights, second_array, second_weights = _read_and_cap(
        first,
        first_log_weights,
        second,
        second_log_weights,
        space,
        key=key,
        metric=metric,
        cap=cap,
    )
    return (
        NearestNeighbours.fit(first_array, first_weights, space, settings),
        NearestNeighbours.fit(second_array, second_weights, space, settings),
    )


def _equal_counts(
    first: Array,
    first_log_weights: Array,
    second: Array,
    second_log_weights: Array,
    *,
    key: PRNGKeyArray,
) -> tuple[Array, Array, Array, Array]:
    """Subsample the posterior with more positive weights down to the other's count."""
    if not _is_concrete(first_log_weights, second_log_weights):
        return first, first_log_weights, second, second_log_weights
    sets = [(first, first_log_weights), (second, second_log_weights)]
    counts = [int(jnp.sum(log_weights > -jnp.inf)) for _, log_weights in sets]
    size = min(counts)
    equalised = []
    for name, (samples, log_weights), count in zip(
        ("first", "second"), sets, counts, strict=True
    ):
        if count > size:
            # * Zero weights come last in the draw, so every kept sample is present.
            samples, log_weights = _draw(
                samples, log_weights, derive_key(key, name), size
            )
        equalised.extend([samples, log_weights])
    return equalised[0], equalised[1], equalised[2], equalised[3]


def _warn_about_zero_distances(
    neighbours: int,
    *sets: tuple[Array, Array, Array],
) -> None:
    """Warn when concrete k-th neighbour distances of present samples are zero."""
    arrays = [array for group in sets for array in group]
    if not _is_concrete(*arrays):
        return
    zero_count = sum(
        int(jnp.sum(present & ((own == 0) | (cross == 0))))
        for present, own, cross in sets
    )
    if zero_count:
        _warn(
            f"the k-th neighbour distance is zero for {zero_count} samples: at least"
            f" k = {neighbours} samples coincide, as repeated states of a Markov chain"
            " do, so the estimate is NaN; thin the chains or remove repeated samples"
        )


def _mean_over(
    values: Float[Array, " sample"], present: Bool[Array, " sample"]
) -> Float[Array, ""]:
    """Return the mean of the values of present samples; NaN in one stays NaN."""
    count = jnp.sum(present).astype(jnp.float64)
    return jnp.sum(jnp.where(present, values, 0.0)) / count


def _kl_from_distances_kernel(
    own: Float[Array, " sample"],
    cross: Float[Array, " sample"],
    present: Bool[Array, " sample"],
    other_count: Array,
    dimension: int,
) -> Float[Array, ""]:
    """Return (d/n) Σ ln(ν/ρ) + ln(m/(n − 1)) over the present samples."""
    count = jnp.sum(present).astype(jnp.float64)
    usable = (own > 0) & (cross > 0)
    log_ratio = jnp.log(jnp.where(usable, cross, 1.0)) - jnp.log(
        jnp.where(usable, own, 1.0)
    )
    # ! A NaN distance fails both tests, so an invalid estimator stays NaN.
    valid = jnp.all(jnp.where(present, usable, True))
    value = (
        dimension * _mean_over(log_ratio, present)
        + jnp.log(jnp.asarray(other_count, dtype=jnp.float64))
        - jnp.log(count - 1)
    )
    return jnp.where(valid, value, jnp.nan)


def _js_from_log_densities_kernel(
    first_log_density: Float[Array, " sample"],
    first_present: Bool[Array, " sample"],
    second_log_density: Float[Array, " other"],
    second_present: Bool[Array, " other"],
    pooled_log_density: Float[Array, " _"],
) -> Float[Array, ""]:
    """Return H(m) − ½H(p) − ½H(q) from leave-one-out log densities."""
    pooled_present = jnp.concatenate([first_present, second_present])
    pooled_entropy = -_mean_over(pooled_log_density, pooled_present)
    first_entropy = -_mean_over(first_log_density, first_present)
    second_entropy = -_mean_over(second_log_density, second_present)
    value = pooled_entropy - 0.5 * (first_entropy + second_entropy)
    # * The pooled samples stand for the equal mixture only for equal counts.
    equal = jnp.sum(first_present) == jnp.sum(second_present)
    return jnp.where(equal, value, jnp.nan)


def _bhattacharyya(
    own: Float[Array, " sample"],
    cross: Float[Array, " sample"],
    present: Bool[Array, " sample"],
    other_count: Array,
    dimension: int,
    log_constant: float,
) -> Float[Array, ""]:
    """Return B_{k,½} times the mean of √((n − 1)ρ^d/(mν^d)) over present samples."""
    count = jnp.sum(present).astype(jnp.float64)
    usable = (own > 0) & (cross > 0)
    log_terms = 0.5 * (
        jnp.log(count - 1)
        + dimension * jnp.log(jnp.where(usable, own, 1.0))
        - jnp.log(jnp.asarray(other_count, dtype=jnp.float64))
        - dimension * jnp.log(jnp.where(usable, cross, 1.0))
    )
    valid = jnp.all(jnp.where(present, usable, True))
    value = math.exp(log_constant) * _mean_over(jnp.exp(log_terms), present)
    return jnp.where(valid, value, jnp.nan)


def _hellinger_from_distances_kernel(
    first_own: Float[Array, " sample"],
    first_cross: Float[Array, " sample"],
    first_present: Bool[Array, " sample"],
    second_own: Float[Array, " other"],
    second_cross: Float[Array, " other"],
    second_present: Bool[Array, " other"],
    dimension: int,
    neighbours: int,
) -> Float[Array, ""]:
    """Return sign(1 − BC) √|1 − BC| with BC the mean of both directions."""
    # * B_{k,α} = Γ(k)²/(Γ(k − α + 1)Γ(k + α − 1)) at α = ½.
    log_constant = (
        2 * math.lgamma(neighbours)
        - math.lgamma(neighbours + 0.5)
        - math.lgamma(neighbours - 0.5)
    )
    coefficient = 0.5 * (
        _bhattacharyya(
            first_own,
            first_cross,
            first_present,
            jnp.sum(second_present),
            dimension,
            log_constant,
        )
        + _bhattacharyya(
            second_own,
            second_cross,
            second_present,
            jnp.sum(first_present),
            dimension,
            log_constant,
        )
    )
    squared = 1 - coefficient
    return jnp.sign(squared) * jnp.sqrt(jnp.abs(squared))


_kl_from_distances = _jit(_kl_from_distances_kernel, "dimension")
_js_from_log_densities = _jit(_js_from_log_densities_kernel)
_hellinger_from_distances = _jit(
    _hellinger_from_distances_kernel, "dimension", "neighbours"
)
