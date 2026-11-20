"""Half-sample standard errors: the subsampling bootstrap of two weighted sets.

The subsampling bootstrap evaluates a statistic on random subsets of the two
sets of a comparison, drawn without replacement, and reads the spread of the
values as the standard error of the statistic on the full sets (Politis, Romano
and Wolf 1999). Its subsets never hold a sample twice, so no estimator sees a
duplicate point, which resampling with replacement would give the
nearest-neighbour estimators.

Half-samples
    :func:`half_sample_draws` cuts each set into a random half and evaluates
    the statistic on the two halves, once per pair. A half holds
    ⌊N/2⌋ of the N samples of its set. Pair i uses
    ``derive_key(key, "pair", i)``, so a pair does not depend on how many
    pairs are drawn.
Standard error
    :func:`half_sample_error` turns the values of the pairs into one standard
    error per value of the statistic: their standard deviation about their
    mean, times a factor for the correlation of the samples, whose derivation
    is in its docstring.
Zoo finding
    The check ``half-sample-errors`` of ``scripts/zoo_validation/engines.py``,
    which joins the validation page at its next full build, compared the
    half-sample error with the true scatter of the full-sample value. It has
    two parameters, takes the true scatter over 200 replicates and 100
    half-sample pairs on 20 of them, and runs the mean shift, the log width
    ratio and KS of the first parameter, the energy distance, the Gaussian KL
    and sliced W2. The ratio of the half-sample error to the true scatter was
    1.02 to 1.12, each ± 0.06, for independent sets of 1 000 samples (shift
    0.5), and 0.99 to 1.05 with importance weights (4 000 draws of a proposal
    1.5 times wider against 2 000 equal-weight draws). On τ-thinned AR(1)
    chains (n_eff about 0.72 N) it was 0.86 to 0.91 without the correlation
    factor and 1.01 to 1.07 with it. The log width ratio, a function of squares
    whose autocorrelation is shorter than that of the slowest mean behind
    n_eff, is left out of the claim; it read 1.04 ± 0.05 with the factor. The
    subsampling scale √(1/2) read 0.70 to 0.79 on independent and weighted sets.

References:
    :cite:t:`politis1999subsampling`; :cite:t:`shao1989general`;
    :cite:t:`mcmurry2012subsampling`; :cite:t:`kish1965survey`.
"""

import math
from functools import partial
from typing import Final

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from jaxtyping import Array, Float, Int, PRNGKeyArray

from mixed_signals.nulls.plan import Statistic, _read_set, _set_sizes
from mixed_signals.prng import derive_key

HALF_SAMPLE_PAIRS: Final = 200
"""Number of half-sample pairs that the subsampling bootstrap draws by default.

The developers chose the number of the default null draws,
:data:`~mixed_signals._config.NULL_DRAWS`. A standard deviation from M values
has a relative standard error of 1/√(2(M − 1)), about 5% for 200 pairs.
"""


def half_sample_draws(
    statistic: Statistic,
    first: ArrayLike,
    first_log_weights: ArrayLike | None,
    second: ArrayLike,
    second_log_weights: ArrayLike | None,
    *,
    key: PRNGKeyArray,
    pairs: int = HALF_SAMPLE_PAIRS,
    batch_size: int | None = None,
) -> Float[Array, "... draw"]:
    """Evaluate a statistic on random half-samples of two weighted sets.

    Pair i takes the first ⌊N₁/2⌋ positions of a random permutation of the
    first set and the first ⌊N₂/2⌋ of a random permutation of the second, each
    from its own key, so no sample enters a half twice. The halves keep the
    normalised log weights of their samples. A half without a sample of
    positive weight gives NaN in a statistic that normalises its weights. The
    pairs run through :func:`jax.lax.map`, so the statistic must be traceable.

    Args:
        statistic: Function of (first samples, first log weights, second
            samples, second log weights) that returns an array of any shape;
            the two halves may differ in size.
        first: Samples of the first set, shaped (N₁, D).
        first_log_weights: Their log weights, shaped (N₁,); equal weights when
            ``None``.
        second: Samples of the second set, shaped (N₂, D).
        second_log_weights: Their log weights, shaped (N₂,); equal weights when
            ``None``.
        key: JAX PRNG key, derived with :func:`~mixed_signals.prng.derive_key`
            from the path of the row, ending in ``"half_samples"``.
        pairs: Number of half-sample pairs.
        batch_size: Pairs evaluated at once; one at a time when ``None``.

    Returns:
        The values of all pairs, shaped (..., pairs): the leading axes are
        those of the statistic's value and the pair axis comes last, as
        :func:`~mixed_signals.nulls.plan.null_draws` returns its draws.

    Raises:
        TypeError: If an input is not real-valued, or ``pairs`` is not an
            integer.
        ValueError: If ``pairs`` is below one, the shapes do not fit, the sets
            have different numbers of parameters or fewer than two samples,
            or concrete values are invalid.

    Warns:
        PrecisionWarning: If an input is narrower than float64.

    References:
        :cite:t:`politis1999subsampling`; :cite:t:`shao1989general`.
    """
    if isinstance(pairs, bool) or not isinstance(pairs, int):
        raise TypeError(f"pairs must be an integer, got {type(pairs).__name__}")
    if pairs < 1:
        raise ValueError(f"pairs must be at least 1, got {pairs}")
    first_samples, first_weights = _read_set("first", first, first_log_weights)
    second_samples, second_weights = _read_set("second", second, second_log_weights)
    if first_samples.shape[1] != second_samples.shape[1]:
        raise ValueError(
            "both sets need the same parameters, got"
            f" {first_samples.shape[1]} and {second_samples.shape[1]}"
        )
    for name, count in (
        ("first", first_samples.shape[0]),
        ("second", second_samples.shape[0]),
    ):
        if count < 2:
            raise ValueError(
                f"the {name} set needs at least 2 samples to form a half-sample,"
                f" got {count}"
            )
    halves = (
        _half_positions(key, "first", first_samples.shape[0], pairs),
        _half_positions(key, "second", second_samples.shape[0], pairs),
    )
    values: Array = jax.lax.map(
        partial(
            _statistic_on_halves,
            statistic=statistic,
            first_samples=first_samples,
            first_log_weights=first_weights,
            second_samples=second_samples,
            second_log_weights=second_weights,
        ),
        halves,
        batch_size=batch_size,
    )
    return jnp.moveaxis(values, 0, -1)


def half_sample_error(
    draws: Float[ArrayLike, "*leading draw"],
    first_log_weights: ArrayLike,
    second_log_weights: ArrayLike,
    *,
    first_n_eff: float | None = None,
    second_n_eff: float | None = None,
) -> Float[Array, " *leading"]:
    """Return the standard error of a statistic from its half-sample values.

    The error is the standard deviation (ddof = 1) of the finite draws along
    the last axis, times √((1/n₁ + 1/n₂)/(1/K₁ + 1/K₂)), where K_i is the Kish
    size of the weights of set i and n_i its effective sample size, K_i when
    not given. Draws that are not finite leave the standard deviation; the
    error is NaN where fewer than two draws are finite. The sizes are read
    from concrete weights, so call this function outside ``jax.jit``.

    The spread is the delete-d jackknife with d = n/2 (Shao and Wu 1989),
    whose factor (n − d)/d is one, so the spread of the half-sample values
    about their mean estimates the standard error of the full-sample value. It
    is consistent for sample quantiles when √n/d → 0 and n − d → ∞, which
    d = n/2 meets and the delete-one jackknife does not (Shao and Wu 1989); on
    the zoo it was also calibrated for the KS statistic, which is not smooth
    (see the zoo finding of this module). The subsampling scale √(b/n) of
    Politis, Romano and Wolf (1999) assumes b/n → 0. A mean of b samples drawn
    without replacement from n keeps the finite-population factor 1 − b/n of
    its variance (McMurry, Politis and Romano 2012), which is ½ for halves.
    Random halves see a thinned chain as exchangeable samples, so the factor
    √((1/n₁ + 1/n₂)/(1/K₁ + 1/K₂)), the variance ratio behind the side size of
    the permutation null (Kish 1965), restores the correlation. It is one for
    independent sets, weighted or not.

    Args:
        draws: Half-sample values of each statistic along the last axis,
            shaped (..., M), as :func:`half_sample_draws` returns them.
        first_log_weights: Log weights of the whole first set, shaped (N₁,).
        second_log_weights: Log weights of the whole second set, shaped (N₂,).
        first_n_eff: Effective sample size of the first set, for correlated
            samples; its Kish size when ``None``.
        second_n_eff: Effective sample size of the second set; its Kish size
            when ``None``.

    Returns:
        The standard error of each value of the statistic, shaped (...).

    Raises:
        ValueError: If a weight array is not one-dimensional, or a Kish or
            effective sample size is below one or not finite.

    References:
        :cite:t:`shao1989general`; :cite:t:`politis1999subsampling`;
        :cite:t:`mcmurry2012subsampling`; :cite:t:`kish1965survey`.
    """
    _, first_kish, first_effective = _set_sizes(
        "the first set", first_log_weights, first_n_eff
    )
    _, second_kish, second_effective = _set_sizes(
        "the second set", second_log_weights, second_n_eff
    )
    correlation_factor = math.sqrt(
        (1 / first_effective + 1 / second_effective)
        / (1 / first_kish + 1 / second_kish)
    )
    values = jnp.asarray(draws, dtype=jnp.float64)
    return correlation_factor * _finite_standard_deviation(values)


def _half_positions(
    key: PRNGKeyArray, side: str, count: int, pairs: int
) -> Int[Array, "draw selected"]:
    """Return the positions of the half of one set, for every pair."""
    keys = jnp.stack([derive_key(key, "pair", index, side) for index in range(pairs)])
    orders = jax.vmap(partial(jax.random.permutation, x=count))(keys)
    return orders[:, : count // 2]


def _statistic_on_halves(
    halves: tuple[Int[Array, " selected"], Int[Array, " other"]],
    *,
    statistic: Statistic,
    first_samples: Float[Array, "_ param"],
    first_log_weights: Float[Array, " _"],
    second_samples: Float[Array, "_ param"],
    second_log_weights: Float[Array, " _"],
) -> Array:
    """Run the statistic on the two halves of one pair."""
    first_half, second_half = halves
    return statistic(
        first_samples[first_half],
        first_log_weights[first_half],
        second_samples[second_half],
        second_log_weights[second_half],
    )


def _finite_standard_deviation(
    values: Float[Array, "*leading draw"],
) -> Float[Array, " *leading"]:
    """Return the ddof = 1 standard deviation of the finite draws, NaN below two."""
    finite = jnp.isfinite(values)
    count = jnp.sum(finite, axis=-1, dtype=jnp.float64)
    mean = jnp.sum(jnp.where(finite, values, 0.0), axis=-1) / jnp.maximum(count, 1.0)
    squares = jnp.sum(jnp.where(finite, (values - mean[..., None]) ** 2, 0.0), axis=-1)
    deviation = jnp.sqrt(squares / jnp.maximum(count - 1.0, 1.0))
    return jnp.where(count >= 2, deviation, jnp.nan)
