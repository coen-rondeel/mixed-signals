"""Integrated autocorrelation time from walker-averaged autocorrelation functions.

The integrated autocorrelation time τ = Σₜ ρ(t) = 1 + 2 Σₜ₌₁ ρ(t), summed over
all lags t, is the factor by which correlation inflates the variance of a chain
mean, so S steps hold S/τ effective samples (Goodman and Weare 2010); it is
twice the τ_int of Sokal (1997). The functions take samples shaped (step,
walker, param) after burn-in: the chains of emcee (Foreman-Mackey et al. 2013),
or the cold chain of :class:`~mixed_signals.core.chains.Chains`.

Autocorrelation function
    Each walker's function comes from its own mean and variance, through the
    FFT with the divisor S that Geyer (1992) recommends, and the functions are
    averaged over the walkers, as emcee does; the average has a lower variance
    than the function of the walker mean (Foreman-Mackey 2017). A walker that
    never moves in a parameter has no autocorrelation function and is left out
    of that parameter's average.
Truncation
    Noise grows with the lag, so the sum stops at a lag that the data choose.
    The lag pairs P_m = ρ(2m) + ρ(2m + 1) of a reversible Markov chain are
    positive and decrease (Geyer 1992). The initial monotone sequence
    estimator keeps the pairs up to the first one that is not positive, each
    reduced to the smallest pair before it, and τ = −1 + 2 Σ P_m. Adding the
    next even lag when it is positive averages the sums that end at an odd and
    at an even lag, which steadies antithetic chains (Vehtari et al. 2021).
    The automatic window of Sokal (1997), which emcee uses, closes at the
    first lag for anticorrelated chains and gives τ ≈ 0 where an AR(1) chain
    with φ = −0.5 has τ = 1/3; Geyer's pairs are right for every φ.
Floor
    τ never falls below 1/log₁₀ S for S draws over all walkers, so a chain
    holds at most S log₁₀ S effective samples, as in the software of Vehtari
    et al. (2021). Short, strongly antithetic chains could otherwise give
    τ ≤ 0.
Reliability
    An estimate from fewer than :data:`MINIMUM_LENGTH_IN_TAU` autocorrelation
    times warns (Foreman-Mackey 2017). So do lag pairs that stay positive to
    the end of the chain: each walker is centred on its own mean, so its
    autocorrelations sum to ½ over all lags, and τ then falls to about zero,
    its floor, a sign that the chains are too short.
Thinning
    Thinning keeps about one step per autocorrelation time, but noise in τ
    must not thin uncorrelated chains, whose τ lies near 1. The factor is
    k = max(1, ⌈τ − 2ε⌉), with ε = τ √(2(2M + 1)/(S W)) the a priori standard
    error of a sum over the lags up to M, here the last lag of the truncated
    sum, from S steps of W walkers (Madras and Sokal 1988; Wolff 2004).
Coupled walkers
    S/τ per walker counts the walkers as independent chains. Ensemble moves
    couple them, which is why Goodman and Weare (2010) define τ for the mean
    over the walkers.
Checks
    Shapes and types always; finite values and the warnings only for concrete
    inputs, since inside ``jax.jit`` the values are unknown.

References:
    :cite:t:`goodman2010ensemble`; :cite:t:`sokal1997monte`;
    :cite:t:`geyer1992practical`; :cite:t:`vehtari2021rank`;
    :cite:t:`madras1988pivot`; :cite:t:`wolff2004monte`;
    :cite:t:`foremanmackey2013emcee`; :cite:t:`foremanmackey2017autocorrelation`.
"""

import math
import os
import warnings
from collections.abc import Callable, Sequence
from typing import Final, NamedTuple, cast

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike
from jaxtyping import Array, Bool, Float, Int

from mixed_signals.core.samples import _check_real, _is_concrete, _warn_narrow_floats
from mixed_signals.errors import ValidityWarning

MINIMUM_LENGTH_IN_TAU: Final = 50
"""Steps per autocorrelation time below which an estimate of τ warns.

The emcee documentation advises against trusting τ from chains shorter than
about 50τ (Foreman-Mackey 2017).

References:
    :cite:t:`foremanmackey2017autocorrelation`.
"""

MINIMUM_STEPS: Final = 2
"""Fewest steps per walker from which τ is estimated: one lag pair needs two."""

_CHAIN_LAYOUT: Final = "(step, walker, param)"
_PACKAGE_DIRECTORY: Final = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# * Warnings name at most this many walkers or parameters, then count the rest.
_LISTED_INDICES: Final = 8


class _Autocorrelation(NamedTuple):
    """Walker-averaged autocorrelation function and the walkers that move."""

    function: Array
    moving: Array


class _Truncation(NamedTuple):
    """τ, whether a lag pair turned non-positive in time, and the last lag summed."""

    tau: Array
    finished: Array
    window: Array


class _Estimate(NamedTuple):
    """Autocorrelation function, moving walkers, τ and thinning factor of chains."""

    function: Array
    moving: Array
    tau: Array
    finished: Array
    thin: Array


def autocorrelation_function(samples: ArrayLike) -> Float[Array, "lag param"]:
    """Return the walker-averaged autocorrelation function of chains.

    Each walker's function is computed from its own mean and variance, with
    the divisor S at every lag, through an FFT padded to twice the next power
    of two, as in emcee; the functions are then averaged over the walkers
    that move (Foreman-Mackey 2017).

    Args:
        samples: Chains after burn-in, shaped (step, walker, param), with at
            least :data:`MINIMUM_STEPS` steps.

    Returns:
        ρ(t) at the lags t = 0, …, S − 1, shaped (S, param); NaN for a
        parameter in which no walker moves.

    Raises:
        ValueError: If the samples are not shaped (step, walker, param) with
            enough steps, or concrete samples are not finite.
        TypeError: If the samples are not real-valued.

    Warns:
        ValidityWarning: If a walker never moves in a parameter.
        PrecisionWarning: If the samples are narrower than float64.

    References:
        :cite:t:`foremanmackey2017autocorrelation`;
        :cite:t:`geyer1992practical`.
    """
    chain_samples = _read_chain_samples(samples, MINIMUM_STEPS)
    autocorrelation = _autocorrelation(chain_samples)
    if _is_concrete(chain_samples):
        _warn_about_walkers(autocorrelation.moving)
    return autocorrelation.function


def integrated_autocorrelation_time(samples: ArrayLike) -> Float[Array, " param"]:
    """Return the integrated autocorrelation time τ of each parameter.

    τ = −1 + 2 Σ P_m sums the initial monotone sequence of lag pairs of the
    walker-averaged autocorrelation function (Geyer 1992), plus the next even
    lag when it is positive, and is floored at 1/log₁₀ S for S draws over all
    walkers (Vehtari et al. 2021). τ is summed over all lags, as in emcee, so
    S/τ is the effective sample size of S steps.

    Args:
        samples: Chains after burn-in, shaped (step, walker, param), with at
            least :data:`MINIMUM_STEPS` steps.

    Returns:
        τ of each parameter in steps, shaped (param,); NaN for a parameter in
        which no walker moves.

    Raises:
        ValueError: If the samples are not shaped (step, walker, param) with
            enough steps, or concrete samples are not finite.
        TypeError: If the samples are not real-valued.

    Warns:
        ValidityWarning: If a walker never moves in a parameter, the chains
            are shorter than :data:`MINIMUM_LENGTH_IN_TAU` times τ, or the lag
            pairs stay positive to the end of the chains, which puts τ at its
            floor.
        PrecisionWarning: If the samples are narrower than float64.

    Example:
        >>> import jax
        >>> from mixed_signals import zoo
        >>> chains = zoo.ar1_chains([0.5]).sample(jax.random.key(0), 5_000)
        >>> tau = integrated_autocorrelation_time(chains.samples[:, 0, :, 0, :])
        >>> abs(float(tau[0]) - 3.0) < 0.2
        True

    References:
        :cite:t:`geyer1992practical`; :cite:t:`vehtari2021rank`;
        :cite:t:`goodman2010ensemble`; :cite:t:`foremanmackey2017autocorrelation`.
    """
    return _estimate_autocorrelation(_read_chain_samples(samples, MINIMUM_STEPS)).tau


def _read_chain_samples(samples: ArrayLike, minimum_steps: int) -> Array:
    """Return chains shaped (step, walker, param) as float64, after checking them.

    The MCMC tools share this check of their input.

    Args:
        samples: Chains shaped (step, walker, param).
        minimum_steps: Fewest steps per walker that the caller needs.

    Returns:
        The samples as a float64 JAX array.

    Raises:
        ValueError: If the shape does not fit, or concrete samples are not
            finite.
        TypeError: If the samples are not real-valued.

    Warns:
        PrecisionWarning: If the samples are narrower than float64.
    """
    array = jnp.asarray(samples)
    _check_real("samples", array)
    if array.ndim != 3 or 0 in array.shape:
        raise ValueError(
            f"samples must be shaped {_CHAIN_LAYOUT} with entries on every axis,"
            f" got shape {array.shape}"
        )
    if array.shape[0] < minimum_steps:
        raise ValueError(
            f"samples need at least {minimum_steps} steps per walker, got"
            f" {array.shape[0]}"
        )
    _warn_narrow_floats({"samples": array})
    array = array.astype(jnp.float64)
    if _is_concrete(array) and not bool(jnp.all(jnp.isfinite(array))):
        raise ValueError("samples must be finite")
    return array


def _estimate_autocorrelation(samples: Array) -> _Estimate:
    """Return the autocorrelation function and τ of checked chains, with warnings.

    Args:
        samples: Chains from :func:`_read_chain_samples`.

    Returns:
        The walker-averaged function, the walkers that move, τ, whether its
        lag pairs turned non-positive within the chains, and the thinning
        factor of each parameter.

    Warns:
        ValidityWarning: As :func:`integrated_autocorrelation_time`, for
            concrete samples.
    """
    step_count, walker_count, _ = samples.shape
    autocorrelation = _autocorrelation(samples)
    truncation = _truncation(
        autocorrelation.function, jnp.asarray(_tau_floor(step_count * walker_count))
    )
    thin = _thinning(
        truncation.tau,
        truncation.window,
        step_count * jnp.sum(autocorrelation.moving, axis=0),
    )
    if _is_concrete(samples):
        _warn_about_walkers(autocorrelation.moving)
        _warn_about_truncation(truncation, step_count)
    return _Estimate(
        function=autocorrelation.function,
        moving=autocorrelation.moving,
        tau=truncation.tau,
        finished=truncation.finished,
        thin=thin,
    )


def _tau_floor(draw_count: int) -> float:
    """Return the floor 1/log₁₀ S of τ for S draws, so that S/τ ≤ S log₁₀ S.

    Args:
        draw_count: Draws S over all walkers, at least 2.

    Returns:
        The floor of τ.

    References:
        :cite:t:`vehtari2021rank`.
    """
    return 1 / math.log10(draw_count)


def _truncated_tau(
    function: Float[Array, "lag param"], draw_count: int
) -> Float[Array, " param"]:
    """Return τ from a given autocorrelation function, without checks or warnings.

    Reductions read τ of a thinned chain from every k-th lag of the function
    of the full chain, in units of kept steps.

    Args:
        function: Autocorrelation function ρ(t) at lags 0, 1, …, shaped
            (lag, param), with ρ(0) = 1.
        draw_count: Draws over all walkers that the function describes, for
            the floor of τ.

    Returns:
        τ of each parameter, shaped (param,).
    """
    return _truncation(function, jnp.asarray(_tau_floor(draw_count))).tau


def _jit[**Parameters, Result](
    function: Callable[Parameters, Result],
) -> Callable[Parameters, Result]:
    """Compile a kernel with :func:`jax.jit` and keep its signature for type checks."""
    return cast(Callable[Parameters, Result], jax.jit(function))


def _next_power_of_two(size: int) -> int:
    """Return the smallest power of two that is at least ``size``."""
    return 1 << (size - 1).bit_length()


def _autocorrelation_kernel(
    samples: Float[Array, "step walker param"],
) -> _Autocorrelation:
    """Return the walker-averaged autocorrelation function and the moving walkers."""
    step_count = samples.shape[0]
    transform_size = 2 * _next_power_of_two(step_count)
    # * Measured from the first step, a walker that never moves gives exact zeros.
    shifted = samples - samples[:1]
    centred = shifted - jnp.mean(shifted, axis=0)
    power = jnp.abs(jnp.fft.rfft(centred, n=transform_size, axis=0)) ** 2
    autocovariance = jnp.fft.irfft(power, n=transform_size, axis=0)[:step_count]
    moving = jnp.any(shifted != 0, axis=0)
    variance = jnp.where(moving, autocovariance[0], 1.0)
    walker_functions = jnp.where(moving, autocovariance / variance, 0.0)
    return _Autocorrelation(
        function=jnp.sum(walker_functions, axis=1) / jnp.sum(moving, axis=0),
        moving=moving,
    )


def _truncation_kernel(
    function: Float[Array, "lag param"], floor: Float[Array, ""]
) -> _Truncation:
    """Return τ from the initial monotone sequence of lag pairs, and if it ended."""
    lag_count = function.shape[0]
    pair_count = lag_count // 2
    pairs = function[0 : 2 * pair_count : 2] + function[1 : 2 * pair_count : 2]
    # * Pair 0, 1 + ρ(1), always counts; a later one only if all before it are positive.
    keep = jnp.concatenate(
        [
            jnp.ones_like(pairs[:1], dtype=bool),
            jnp.cumsum(pairs[1:] <= 0, axis=0) == 0,
        ],
        axis=0,
    )
    kept_count = jnp.sum(keep, axis=0)
    total = jnp.sum(jnp.where(keep, jax.lax.cummin(pairs, axis=0), 0.0), axis=0)
    next_lag = 2 * kept_count
    next_value = jnp.take_along_axis(
        function, jnp.minimum(next_lag, lag_count - 1)[None], axis=0
    )[0]
    added = (next_lag < lag_count) & (next_value > 0)
    return _Truncation(
        tau=jnp.maximum(-1 + 2 * total + jnp.where(added, next_value, 0.0), floor),
        finished=kept_count < pair_count,
        window=jnp.where(added, next_lag, next_lag - 1),
    )


def _thinning_kernel(
    tau: Float[Array, " param"],
    window: Int[Array, " param"],
    draw_counts: Int[Array, " param"],
) -> Int[Array, " param"]:
    """Return max(1, ⌈τ − 2ε⌉) with the Madras-Sokal error ε of τ; 1 where τ is NaN."""
    error = tau * jnp.sqrt(2 * (2 * window + 1) / draw_counts)
    factor = jnp.ceil(tau - 2 * error)
    return jnp.where(jnp.isfinite(factor), jnp.maximum(factor, 1.0), 1.0).astype(
        jnp.int64
    )


_autocorrelation = _jit(_autocorrelation_kernel)
_truncation = _jit(_truncation_kernel)
_thinning = _jit(_thinning_kernel)


def _warn(message: str) -> None:
    """Raise a validity warning that points at the caller's line."""
    warnings.warn(ValidityWarning(message), skip_file_prefixes=(_PACKAGE_DIRECTORY,))


def _index_text(noun: str, indices: Sequence[int]) -> str:
    """Return indices after a noun, such as ``"walkers 0, 3"``, counting long tails."""
    listed = ", ".join(str(index) for index in indices[:_LISTED_INDICES])
    remaining = len(indices) - _LISTED_INDICES
    if remaining > 0:
        listed = f"{listed} and {remaining} more"
    return f"{noun if len(indices) == 1 else noun + 's'} {listed}"


def _warn_about_walkers(moving: Bool[Array, "walker param"]) -> None:
    """Warn about walkers that never move, and parameters in which none moves."""
    moving_walkers = np.asarray(moving)
    still: list[str] = []
    frozen: list[int] = []
    for position in range(moving_walkers.shape[1]):
        walkers = np.flatnonzero(~moving_walkers[:, position]).tolist()
        if len(walkers) == moving_walkers.shape[0]:
            frozen.append(position)
        elif walkers:
            still.append(f"{_index_text('walker', walkers)} of parameter {position}")
    if still:
        _warn(
            "walkers that never move are left out of the autocorrelation"
            f" function: {'; '.join(still)}; check their acceptance"
        )
    if frozen:
        _warn(
            f"no walker moves in {_index_text('parameter', frozen)}, so the"
            " autocorrelation time is NaN there"
        )


def _warn_about_truncation(truncation: _Truncation, step_count: int) -> None:
    """Warn when concrete chains are short for their τ, or τ is a lower bound."""
    # ! Python floats only: NumPy flags invalid operations differently per platform.
    taus = [float(tau) for tau in np.asarray(truncation.tau)]
    finished = [bool(ended) for ended in np.asarray(truncation.finished)]
    short = [
        position
        for position, tau in enumerate(taus)
        if math.isfinite(tau) and step_count < MINIMUM_LENGTH_IN_TAU * tau
    ]
    if short:
        largest = max(taus[position] for position in short)
        _warn(
            f"the chains have {step_count} steps, fewer than"
            f" {MINIMUM_LENGTH_IN_TAU} autocorrelation times of"
            f" {_index_text('parameter', short)} (tau up to {largest:.3g}), so tau"
            " is unreliable; run longer chains"
        )
    unfinished = [
        position
        for position, (tau, ended) in enumerate(zip(taus, finished, strict=True))
        if math.isfinite(tau) and not ended
    ]
    if unfinished:
        _warn(
            "the lag pairs of the autocorrelation function stay positive to the"
            f" end of the chains for {_index_text('parameter', unfinished)}, so"
            " tau falls to its floor: the chains are too short to estimate it"
        )
