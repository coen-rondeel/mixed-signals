"""Chain health: autocorrelation, effective sample size, split-chain tests, R̂ and rates.

:func:`chain_health` reads the cold chain of :class:`~mixed_signals.core.chains.Chains`
after a burn-in and applies the kernels of :mod:`mixed_signals.mcmc` with the
topology of each parameter, which those kernels do not know.

Lines and intervals
    τ, the effective sample size S·W/τ, the split-chain Kolmogorov-Smirnov
    test and the informational R̂ of the values.
Circles
    Raw angles jump by a period where a walker crosses the cut, so τ is the
    larger of the τ of the observables cos(θ − θ̄) and sin(θ − θ̄), with θ̄ the
    circular mean; τ is defined for any observable of a chain (Sokal 1997;
    Goodman and Weare 2010), and for a narrow circle sin(θ − θ̄) ≈ θ − θ̄ gives
    the τ of the angle. The split-chain test takes Kuiper's V, which no cut
    changes (Kuiper 1960), on values unwrapped about θ̄, thinned by the larger
    factor of the two observables; R̂ uses the unwrapped values. Sphere
    azimuths are circles.
Rates
    Acceptance rates per temperature, pooled over the walkers, and swap rates
    between neighbouring temperatures, from the counts and proposals that
    eryn records over the whole run; ``None`` when the chains carry no counts.
    They are informational, because good rates depend on the move and the
    ladder (Foreman-Mackey et al. 2013; Vousden, Farr and Mandel 2016).
Flags
    An effective sample size below 400, the minimum of Vehtari et al. (2021),
    and a split-chain statistic of at least 0.1, a shift of about 0.25σ
    between the halves by the σ_eff map of KS, with σ_sig ≥ 2 from its own
    p-value.

References:
    :cite:t:`goodman2010ensemble`; :cite:t:`sokal1997monte`;
    :cite:t:`kuiper1960tests`; :cite:t:`brooks2003nonparametric`;
    :cite:t:`vehtari2021rank`; :cite:t:`foremanmackey2013emcee`;
    :cite:t:`vousden2016dynamic`; :cite:t:`mardia2000directional`.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, Int

from mixed_signals.core.chains import Chains
from mixed_signals.core.samples import circular_mean, unwrap_about
from mixed_signals.core.space import ParamSpace
from mixed_signals.diagnostics._common import circle_positions, period_of
from mixed_signals.mcmc.autocorr import (
    MINIMUM_STEPS,
    _estimate_autocorrelation,
    _read_chain_samples,
)
from mixed_signals.mcmc.ess import (
    rank_normalized_split_rhat,
    split_chain_ks,
    split_chain_kuiper,
)


class ChainHealth(NamedTuple):
    """Health of the cold chain, one value per parameter, and the rates of the run."""

    tau: Float[Array, " param"]
    """Integrated autocorrelation time in steps; on circles, of cos and sin."""
    effective_sample_size: Float[Array, " param"]
    """S·W/τ for S steps of W walkers after the burn-in."""
    split_statistic: Float[Array, " param"]
    """KS statistic D of the split-chain test, Kuiper's V on circles."""
    split_p_value: Float[Array, " param"]
    """Asymptotic p-value of the split-chain statistic."""
    split_thin: Int[Array, " param"]
    """Thinning factor of the halves of each parameter."""
    rhat: Float[Array, " param"]
    """Rank-normalised split-R̂ across walkers, informational."""
    acceptance_rate: Float[Array, " temp"] | None
    """Accepted over proposed in-model moves at each temperature."""
    swap_rate: Float[Array, " temp-1"] | None
    """Accepted over proposed swaps between temperatures t and t + 1."""


def split_chain_metric_names(space: ParamSpace) -> tuple[str, ...]:
    """Return the metric name of the split-chain row of each parameter.

    Args:
        space: Parameter space.

    Returns:
        ``"split_chain_kuiper_ks"`` for a circle, sphere azimuths included,
        and ``"split_chain_ks"`` otherwise.
    """
    circles = set(circle_positions(space))
    return tuple(
        "split_chain_kuiper_ks" if position in circles else "split_chain_ks"
        for position in range(len(space))
    )


def chain_health(chains: Chains, space: ParamSpace, *, burn_in: int = 0) -> ChainHealth:
    """Return the health of the cold chain after a burn-in, and the rates of the run.

    Args:
        chains: Chains of one run, with the parameters of ``space``.
        space: Parameter space of the chains.
        burn_in: Steps to drop from the start of every walker.

    Returns:
        τ, the effective sample size, the split-chain test and R̂ of every
        parameter, and the acceptance and swap rates when the chains carry
        counts.

    Raises:
        ValueError: If the chains do not have the parameters of the space, no
            temperature stays at β = 1, the burn-in is negative or leaves
            fewer than four steps, or the samples are not finite.

    Warns:
        ValidityWarning: As the kernels of :mod:`mixed_signals.mcmc`, for τ
            of the values or of sin(θ − θ̄) on circles, and for R̂.
        CoupledChainsWarning: Always, from R̂, because ensemble moves couple
            the walkers.

    References:
        :cite:t:`goodman2010ensemble`; :cite:t:`kuiper1960tests`;
        :cite:t:`vehtari2021rank`.
    """
    if chains.shape.param != len(space):
        raise ValueError(
            f"the chains have {chains.shape.param} parameters, the space {len(space)}"
        )
    if burn_in < 0:
        raise ValueError(f"burn_in must be at least 0, got {burn_in}")
    cold = chains.cold()
    samples = _read_chain_samples(cold.samples[burn_in:, 0, :, 0, :], 2 * MINIMUM_STEPS)
    step_count, walker_count, _ = samples.shape
    circles = circle_positions(space)
    unwrapped = samples
    primary = samples
    sines = []
    cosines = []
    for position in circles:
        period = period_of(space, position)
        lower = space.parameters[position].bounds[0]
        values = samples[..., position]
        flat = values.reshape(-1)
        centre = circular_mean(flat, jnp.zeros(flat.shape[0]), period, lower)
        unwrapped = unwrapped.at[..., position].set(
            unwrap_about(values, centre, period)
        )
        angles = 2 * jnp.pi * (unwrapped[..., position] - centre) / period
        sines.append(jnp.sin(angles))
        cosines.append(jnp.cos(angles))
        primary = primary.at[..., position].set(sines[-1])
    estimate = _estimate_autocorrelation(primary)
    tau = estimate.tau
    thin = estimate.thin
    if circles:
        # * Traced, so τ of the quadratic observable adds no second set of warnings.
        cosine = jax.jit(_estimate_autocorrelation)(jnp.stack(cosines, axis=-1))
        positions = jnp.asarray(circles)
        tau = tau.at[positions].set(jnp.maximum(tau[positions], cosine.tau))
        thin = thin.at[positions].set(jnp.maximum(thin[positions], cosine.thin))
    statistic = jnp.full(len(space), jnp.nan)
    p_value = jnp.full(len(space), jnp.nan)
    lines = [position for position in range(len(space)) if position not in circles]
    for selected, test in (
        (lines, split_chain_ks),
        (list(circles), split_chain_kuiper),
    ):
        if selected:
            index = jnp.asarray(selected)
            result = test(unwrapped[..., index], thin=np.asarray(thin[index]))
            statistic = statistic.at[index].set(result.statistic)
            p_value = p_value.at[index].set(result.p_value)
    acceptance = None
    if chains.accepted is not None and chains.proposed is not None:
        acceptance = jnp.sum(chains.accepted, axis=1) / jnp.sum(chains.proposed, axis=1)
    swaps = None
    if chains.swaps_accepted is not None and chains.swaps_proposed is not None:
        swaps = chains.swaps_accepted / chains.swaps_proposed
    return ChainHealth(
        tau=tau,
        effective_sample_size=step_count * walker_count / tau,
        split_statistic=statistic,
        split_p_value=p_value,
        split_thin=thin,
        rhat=rank_normalized_split_rhat(unwrapped),
        acceptance_rate=acceptance,
        swap_rate=swaps,
    )
