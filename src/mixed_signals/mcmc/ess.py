"""Effective sample size, split-chain Kolmogorov-Smirnov tests and R̂.

The functions take samples shaped (step, walker, param) after burn-in, as
:mod:`mixed_signals.mcmc.autocorr` does, and return one value per parameter.

Effective sample size
    S·W/τ for S steps of W walkers, with τ from
    :func:`~mixed_signals.mcmc.autocorr.integrated_autocorrelation_time`: the
    number of independent draws that estimate a posterior mean as precisely
    (Geyer 1992; Vehtari et al. 2021). Antithetic chains, with τ < 1, hold
    more effective samples than draws, at most S·W·log₁₀(S·W) through the
    floor of τ. A reduced sample set caps its ``n_eff`` at its number of
    samples instead; see :mod:`mixed_signals.mcmc.reduce`.
Split-chain Kolmogorov-Smirnov test
    Compares the first and the second half of every walker's chain, pooled
    over the walkers, as a test of stationarity (Brooks, Giudici and Philippe
    2003; the ``KS.Diagnostic`` of LaplacesDemon). The test assumes
    independent draws, so each half is first thinned by the factor of its
    parameter, about τ, as :mod:`mixed_signals.mcmc.autocorr` sets it.
    The p-value comes from the limiting Kolmogorov distribution of
    √(nm/(n + m))·D (Smirnov 1948), which is conservative for small halves. A
    small p-value flags halves that differ, such as a transient that the
    burn-in has not removed. Coupled walkers make the pooled samples less than
    independent, so for ensemble samplers the p-value is approximate.
R̂
    The rank-normalised split-R̂ of Vehtari et al. (2021): the larger of the
    values for the draws and for the folded draws, their absolute deviations
    from the median, which also catch walkers that share a location but not a
    spread. Each walker's chain is split into halves, the middle step dropped
    for an odd number of steps. The draws become normal scores
    Φ⁻¹((r − 3/8)/(S + 1/4)) of their average ranks r over all S draws
    (Blom 1958), and R̂ = √(var⁺/W), with W the mean variance within the split
    chains, B the variance between their means times their length n, and
    var⁺ = (n − 1)W/n + B/n (Gelman and Rubin 1992). R̂ assumes independent
    chains, but the moves of ensemble samplers such as eryn couple the
    walkers, and the emcee documentation advises against Gelman-Rubin within
    one ensemble (Foreman-Mackey 2017). Every call therefore warns, and R̂
    decides no pass or fail. Vehtari et al. trust R̂ only when each split
    chain holds at least :data:`MINIMUM_SPLIT_EFFECTIVE_SAMPLE_SIZE` effective
    samples, from the autocorrelation time of the normal scores; fewer warn.
Checks
    As in :mod:`mixed_signals.mcmc.autocorr`: shapes and types always,
    finite values and the warnings only for concrete inputs.

References:
    :cite:t:`geyer1992practical`; :cite:t:`vehtari2021rank`;
    :cite:t:`brooks2003nonparametric`; :cite:t:`hall2026ks`;
    :cite:t:`smirnov1948table`; :cite:t:`blom1958statistical`;
    :cite:t:`gelman1992inference`; :cite:t:`foremanmackey2017autocorrelation`.
"""

import math
import warnings
from typing import Final, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import ndtri
from jax.typing import ArrayLike
from jaxtyping import Array, Float, Int

from mixed_signals.core.samples import _is_concrete
from mixed_signals.errors import CoupledChainsWarning
from mixed_signals.mcmc.autocorr import (
    _PACKAGE_DIRECTORY,
    MINIMUM_STEPS,
    _autocorrelation_kernel,
    _estimate_autocorrelation,
    _index_text,
    _jit,
    _read_chain_samples,
    _tau_floor,
    _truncation_kernel,
    _warn,
)
from mixed_signals.metrics.marginal import ks_statistic

MINIMUM_SPLIT_EFFECTIVE_SAMPLE_SIZE: Final = 50
"""Effective samples per split chain below which R̂ warns.

Vehtari et al. (2021) rely on R̂ only when each split chain has an
effective sample size of at least 50.

References:
    :cite:t:`vehtari2021rank`.
"""

MINIMUM_RHAT_STEPS: Final = 4
"""Fewest steps per walker for R̂: each half needs two draws for a variance."""

# * Normal scores Φ⁻¹((r − 3/8)/(S + 1/4)) of Blom (1958).
_BLOM_RANK_OFFSET: Final = 3 / 8
_BLOM_COUNT_OFFSET: Final = 1 / 4
# * Both series of the Kolmogorov distribution converge to float64 within 8 terms.
_KOLMOGOROV_TERMS: Final = 8
_KOLMOGOROV_CROSSOVER: Final = 1.0
# * Below this argument the survival function equals 1 in float64.
_KOLMOGOROV_SMALLEST: Final = 0.1


class SplitChainTest(NamedTuple):
    """Split-chain Kolmogorov-Smirnov test of each parameter."""

    statistic: Float[Array, " param"]
    """Largest gap D between the distribution functions of the two halves."""
    p_value: Float[Array, " param"]
    """Asymptotic p-value of D for independent draws."""
    thin: Int[Array, " param"]
    """Thinning factor max(1, ⌈τ − 2ε⌉) of each parameter."""
    half_size: Int[Array, " param"]
    """Kept draws of each half, pooled over the walkers."""


class _RankRhat(NamedTuple):
    """R̂ of each parameter and the effective samples per split chain."""

    rhat: Array
    split_ess: Array


def effective_sample_size(samples: ArrayLike) -> Float[Array, " param"]:
    """Return the effective sample size S·W/τ of each parameter.

    Args:
        samples: Chains after burn-in, shaped (step, walker, param), with at
            least :data:`~mixed_signals.mcmc.autocorr.MINIMUM_STEPS` steps.

    Returns:
        The effective sample size of each parameter, shaped (param,); above
        S·W for antithetic chains, and NaN for a parameter in which no walker
        moves.

    Raises:
        ValueError: If the samples are not shaped (step, walker, param) with
            enough steps, or concrete samples are not finite.
        TypeError: If the samples are not real-valued.

    Warns:
        ValidityWarning: As
            :func:`~mixed_signals.mcmc.autocorr.integrated_autocorrelation_time`.
        PrecisionWarning: If the samples are narrower than float64.

    References:
        :cite:t:`geyer1992practical`; :cite:t:`vehtari2021rank`.
    """
    chain_samples = _read_chain_samples(samples, MINIMUM_STEPS)
    step_count, walker_count, _ = chain_samples.shape
    tau = _estimate_autocorrelation(chain_samples).tau
    return step_count * walker_count / tau


def split_chain_ks(samples: ArrayLike) -> SplitChainTest:
    """Test whether the two halves of the chains share one distribution.

    Each walker's chain is split into its first and its second ⌊S/2⌋ steps,
    each half is thinned by max(1, ⌈τ − 2ε⌉) of the parameter, from its own
    first step, and the halves are pooled over the walkers. With n kept draws
    per half, the p-value is Q(√(n/2)·D) of the Kolmogorov distribution.

    Args:
        samples: Chains after burn-in, shaped (step, walker, param), with at
            least :data:`~mixed_signals.mcmc.autocorr.MINIMUM_STEPS` steps.

    Returns:
        The statistic, p-value, thinning factor and half size of each
        parameter.

    Raises:
        ValueError: If the samples are not shaped (step, walker, param) with
            enough steps, or concrete samples are not finite.
        TypeError: If the samples are not real-valued.

    Warns:
        ValidityWarning: As
            :func:`~mixed_signals.mcmc.autocorr.integrated_autocorrelation_time`,
            whose τ sets the thinning.
        PrecisionWarning: If the samples are narrower than float64.

    References:
        :cite:t:`brooks2003nonparametric`; :cite:t:`hall2026ks`;
        :cite:t:`smirnov1948table`.
    """
    chain_samples = _read_chain_samples(samples, MINIMUM_STEPS)
    thin = _estimate_autocorrelation(chain_samples).thin
    return _split_test(chain_samples, thin)


def rank_normalized_split_rhat(samples: ArrayLike) -> Float[Array, " param"]:
    """Return the rank-normalised split-R̂ of each parameter, informational only.

    The value is the larger of the rank-normalised split-R̂ of the draws and
    of the folded draws (Vehtari et al. 2021). It treats the walkers as
    independent chains, so for ensemble samplers it never decides pass or
    fail.

    Args:
        samples: Chains after burn-in, shaped (step, walker, param), with at
            least :data:`MINIMUM_RHAT_STEPS` steps.

    Returns:
        R̂ of each parameter, shaped (param,); 1 for chains that agree, and NaN
        for a parameter that never moves.

    Raises:
        ValueError: If the samples are not shaped (step, walker, param) with
            enough steps, or concrete samples are not finite.
        TypeError: If the samples are not real-valued.

    Warns:
        CoupledChainsWarning: Always, for concrete samples, because ensemble
            moves couple the walkers.
        ValidityWarning: If a split chain holds fewer than
            :data:`MINIMUM_SPLIT_EFFECTIVE_SAMPLE_SIZE` effective samples, or
            a parameter never moves.
        PrecisionWarning: If the samples are narrower than float64.

    References:
        :cite:t:`vehtari2021rank`; :cite:t:`gelman1992inference`;
        :cite:t:`blom1958statistical`; :cite:t:`foremanmackey2017autocorrelation`.
    """
    chain_samples = _read_chain_samples(samples, MINIMUM_RHAT_STEPS)
    step_count, walker_count, _ = chain_samples.shape
    split_draws = 2 * (step_count // 2) * walker_count
    result = _rank_rhat(chain_samples, jnp.asarray(_tau_floor(split_draws)))
    if _is_concrete(chain_samples):
        warnings.warn(
            CoupledChainsWarning(
                "R-hat treats the walkers as independent chains, but ensemble moves"
                " couple them, so the value is informational and decides no pass"
                " or fail"
            ),
            skip_file_prefixes=(_PACKAGE_DIRECTORY,),
        )
        _warn_about_rhat(result)
    return result.rhat


def _normal_scores(values: Float[Array, " sample"]) -> Float[Array, " sample"]:
    """Return the normal scores of values from their average ranks (Blom 1958)."""
    ordered = jnp.sort(values)
    # ! searchsorted gives int32, which JAX divides in float32 even with x64 on.
    below = jnp.searchsorted(ordered, values, side="left").astype(jnp.float64)
    through = jnp.searchsorted(ordered, values, side="right").astype(jnp.float64)
    # * A run of ties at 0-based positions below to through − 1 shares its mean rank.
    ranks = (below + through + 1) / 2
    return ndtri((ranks - _BLOM_RANK_OFFSET) / (values.shape[0] + _BLOM_COUNT_OFFSET))


def _scale_reduction(
    chains: Float[Array, "step walker param"],
) -> Float[Array, " param"]:
    """Return √(var⁺/W) over chains shaped (draw, chain, param)."""
    draw_count = chains.shape[0]
    within = jnp.mean(jnp.var(chains, axis=0, ddof=1), axis=0)
    between = draw_count * jnp.var(jnp.mean(chains, axis=0), axis=0, ddof=1)
    pooled = (draw_count - 1) / draw_count * within + between / draw_count
    return jnp.sqrt(pooled / within)


def _rank_scores(
    split: Float[Array, "step walker param"],
) -> Float[Array, "step walker param"]:
    """Return the normal scores of each parameter over all its split draws."""
    flat = split.reshape(-1, split.shape[-1])
    return jax.vmap(_normal_scores, in_axes=1, out_axes=1)(flat).reshape(split.shape)


def _rank_rhat_kernel(
    samples: Float[Array, "step walker param"], floor: Float[Array, ""]
) -> _RankRhat:
    """Return the larger of bulk and folded split-R̂, and the split-chain ESS."""
    half = samples.shape[0] // 2
    # * The second half starts at S − ⌊S/2⌋, so an odd S drops its middle step.
    split = jnp.concatenate(
        [samples[:half], samples[samples.shape[0] - half :]], axis=1
    )
    scores = _rank_scores(split)
    flat = split.reshape(-1, split.shape[-1])
    folded = jnp.abs(split - jnp.median(flat, axis=0))
    rhat = jnp.maximum(_scale_reduction(scores), _scale_reduction(_rank_scores(folded)))
    # ! Equal draws tie to one score, whose variance is rounding noise, not zero.
    spread = jnp.any(flat != flat[:1], axis=0)
    bulk_tau = _truncation_kernel(_autocorrelation_kernel(scores).function, floor).tau
    return _RankRhat(rhat=jnp.where(spread, rhat, jnp.nan), split_ess=half / bulk_tau)


def _kolmogorov_survival(
    scaled: Float[Array, " *leading"],
) -> Float[Array, " *leading"]:
    """Return Q(λ) = 2 Σⱼ (−1)ʲ⁻¹ exp(−2j²λ²), the Kolmogorov survival function."""
    terms = jnp.arange(1, _KOLMOGOROV_TERMS + 1, dtype=jnp.float64)
    signs = jnp.where(jnp.arange(_KOLMOGOROV_TERMS) % 2 == 0, 1.0, -1.0)
    large = jnp.maximum(scaled, _KOLMOGOROV_CROSSOVER)[..., None]
    alternating = 2 * jnp.sum(signs * jnp.exp(-2 * terms**2 * large**2), axis=-1)
    # * Below the crossover the theta-function form of the series converges faster.
    small = jnp.clip(scaled, _KOLMOGOROV_SMALLEST, _KOLMOGOROV_CROSSOVER)[..., None]
    theta = 1 - jnp.sqrt(2 * jnp.pi) / small[..., 0] * jnp.sum(
        jnp.exp(-((2 * terms - 1) ** 2) * jnp.pi**2 / (8 * small**2)), axis=-1
    )
    return jnp.where(scaled >= _KOLMOGOROV_CROSSOVER, alternating, theta)


def _split_test_kernel(
    samples: Float[Array, "step walker param"], thin: Int[Array, " param"]
) -> SplitChainTest:
    """Return the Kolmogorov-Smirnov test between the thinned halves of chains."""
    step_count, walker_count, param_count = samples.shape
    half = step_count // 2
    kept = jnp.arange(half)[:, None] % thin[None, :] == 0
    # * Dropped steps get zero weight, so the arrays keep fixed shapes under jit.
    log_weights = jnp.broadcast_to(
        jnp.where(kept, 0.0, -jnp.inf).T[:, :, None], (param_count, half, walker_count)
    ).reshape(param_count, -1)
    first = jnp.moveaxis(samples[:half], -1, 0).reshape(param_count, -1)
    second = jnp.moveaxis(samples[step_count - half :], -1, 0).reshape(param_count, -1)
    statistic = ks_statistic(first, log_weights, second, log_weights)
    half_size = walker_count * jnp.sum(kept, axis=0)
    return SplitChainTest(
        statistic=statistic,
        p_value=_kolmogorov_survival(jnp.sqrt(half_size / 2) * statistic),
        thin=thin,
        half_size=half_size,
    )


_rank_rhat = _jit(_rank_rhat_kernel)
_split_test = _jit(_split_test_kernel)


def _warn_about_rhat(result: _RankRhat) -> None:
    """Warn about parameters that never move or have short split chains."""
    # ! Python floats only: NumPy flags invalid operations differently per platform.
    rhats = [float(value) for value in np.asarray(result.rhat)]
    split_sizes = [float(value) for value in np.asarray(result.split_ess)]
    frozen = [position for position, rhat in enumerate(rhats) if math.isnan(rhat)]
    if frozen:
        frozen_text = _index_text("parameter", frozen)
        _warn(f"R-hat is NaN where the draws never move: {frozen_text}")
    short = [
        position
        for position, size in enumerate(split_sizes)
        if math.isfinite(size) and size < MINIMUM_SPLIT_EFFECTIVE_SAMPLE_SIZE
    ]
    if short:
        smallest = min(split_sizes[position] for position in short)
        _warn(
            f"the split chains of {_index_text('parameter', short)} hold as few as"
            f" {smallest:.3g} effective samples each, fewer than"
            f" {MINIMUM_SPLIT_EFFECTIVE_SAMPLE_SIZE}, so R-hat is unreliable"
        )
