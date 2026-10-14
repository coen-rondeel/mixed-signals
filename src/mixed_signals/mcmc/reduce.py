"""Chain reduction: burn-in removal and thinning by the autocorrelation time.

:func:`reduce_chains` turns :class:`~mixed_signals.core.chains.Chains` into the
flat :class:`~mixed_signals.core.samples.SampleSet` that comparisons take, and
records each step in a :class:`ReductionRecord`.

Steps
    The cold chain, at β = 1, samples the posterior (Vousden, Farr and Mandel
    2016). The burn-in drops the first ``burn_in`` steps, a number the caller
    chooses; the split-chain test of
    :func:`~mixed_signals.mcmc.ess.split_chain_ks` flags a transient that
    remains. Thinning keeps every k-th step after the burn-in; for
    ``thin="tau"``, k is the largest factor max(1, ⌈τ − 2ε⌉) over the
    parameters, with ε the a priori error of τ, so that noise in τ does not
    thin uncorrelated chains (Madras and Sokal 1988; Wolff 2004). The kept
    steps of all walkers are then flattened in step order, as the flat chains
    of emcee are (Foreman-Mackey et al. 2013), and ``log_like`` and
    ``log_prior`` follow their samples.
Effective sample size
    Thinning by k ≈ τ leaves correlation between kept steps: an AR(1) chain
    thinned at k = τ has τ_k = (1 + φᵏ)/(1 − φᵏ) ≈ 1.3 in kept steps. τ_k
    comes from every k-th lag of the autocorrelation function of the chain
    after burn-in, which uses every pair of steps, and n_eff = N/τ_k for the N
    kept samples and the slowest parameter, clipped to [1, N], the range that
    :class:`~mixed_signals.core.samples.SampleSet` admits. Antithetic chains,
    with τ_k < 1, therefore get n_eff = N. A parameter that never moves holds
    no information on n_eff; when none moves, all samples are one, n_eff = 1.
Repeating
    Every step is deterministic, so :func:`reduce_chains` with the ``burn_in``
    and ``thin`` of a record repeats the reduction exactly. The record also
    holds the SHA-256 digest of the input chains, which :func:`hash_chains`
    recomputes to check that a repeat reads the same chains.
Scope
    The reduction reads concrete chains outside ``jax.jit``: the cold chain
    and the thinning factor depend on their values.

References:
    :cite:t:`vousden2016dynamic`; :cite:t:`foremanmackey2013emcee`;
    :cite:t:`madras1988pivot`; :cite:t:`wolff2004monte`;
    :cite:t:`geyer1992practical`; :cite:t:`foremanmackey2017autocorrelation`;
    :cite:t:`nist2015secure`.
"""

import math
from typing import Final, NamedTuple

import jax.numpy as jnp
import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from mixed_signals.core.chains import Chains
from mixed_signals.core.result import SHA256_HEX_PATTERN, hash_arrays
from mixed_signals.core.samples import SampleSet
from mixed_signals.mcmc.autocorr import (
    MINIMUM_STEPS,
    _estimate_autocorrelation,
    _truncated_tau,
    _warn,
)

TAU_THINNING: Final = "tau"
"""Value of ``thin`` that thins by about τ of the slowest parameter."""

_DENSITY_FIELDS: Final = ("log_like", "log_prior")


class ReductionRecord(BaseModel):
    """Every step of one reduction, enough to repeat it and check the repeat.

    Values that are NaN, such as τ of a parameter that never moves, are
    written as ``NaN`` in JSON. A repeat of the reduction gives an equal
    record.

    Attributes:
        chain_shape: Axis sizes (step, temp, walker, leaf, param) of the input
            chains.
        temperature: Index of the cold chain on the temperature axis.
        burn_in: Steps dropped from the start of every walker.
        thin: Thinning factor k: every k-th step after the burn-in is kept.
        kept_steps: Steps kept per walker.
        sample_count: Samples of the reduced set, ``kept_steps`` times the
            walkers.
        tau: τ of each parameter after the burn-in, in steps; NaN for a
            parameter that never moves.
        thinned_tau: τ_k of each parameter in kept steps; NaN as for ``tau``.
        n_eff: Effective sample size of the reduced set.
        chains_hash: SHA-256 hex digest of the input chains, from
            :func:`~mixed_signals.core.result.hash_arrays` over ``samples``,
            ``betas``, ``log_like`` and ``log_prior``, those that are set, in
            this order.

    References:
        :cite:t:`geyer1992practical`; :cite:t:`nist2015secure`.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", ser_json_inf_nan="constants")

    chain_shape: tuple[int, int, int, int, int]
    temperature: int = Field(ge=0)
    burn_in: int = Field(ge=0)
    thin: int = Field(ge=1)
    kept_steps: int = Field(ge=MINIMUM_STEPS)
    sample_count: int = Field(ge=MINIMUM_STEPS)
    tau: tuple[float, ...]
    thinned_tau: tuple[float, ...]
    n_eff: float = Field(ge=1)
    chains_hash: str = Field(pattern=f"^{SHA256_HEX_PATTERN.pattern}$")


class Reduction(NamedTuple):
    """The reduced samples and the record of how they were reduced."""

    sample_set: SampleSet
    """Flat samples of the cold chain, one problem and one posterior, with n_eff."""
    record: ReductionRecord
    """Every step of the reduction."""


def reduce_chains(
    chains: Chains, *, burn_in: int = 0, thin: int | str = TAU_THINNING
) -> Reduction:
    """Reduce chains to flat samples: cold chain, burn-in, thinning, n_eff.

    Args:
        chains: Chains of one run.
        burn_in: Steps to drop from the start of every walker.
        thin: ``"tau"`` for the largest factor max(1, ⌈τ − 2ε⌉) over the
            parameters after the burn-in, or a thinning factor of at least 1.

    Returns:
        The sample set, shaped (1, 1, N, D) with its ``n_eff``, and the record
        of the reduction.

    Raises:
        TypeError: If ``burn_in`` or ``thin`` is not an integer.
        ValueError: If no temperature stays at β = 1, ``burn_in`` or ``thin``
            lies out of range, or fewer than
            :data:`~mixed_signals.mcmc.autocorr.MINIMUM_STEPS` steps per walker
            remain after the burn-in or the thinning.

    Warns:
        ValidityWarning: As
            :func:`~mixed_signals.mcmc.autocorr.integrated_autocorrelation_time`
            for the chain after the burn-in, and if no parameter moves.

    Example:
        >>> import math
        >>> import jax
        >>> from mixed_signals import zoo
        >>> chains = zoo.ar1_chains([0.9]).sample(jax.random.key(0), 5_000)
        >>> reduction = reduce_chains(chains, burn_in=100)
        >>> reduction.record.thin <= math.ceil(reduction.record.tau[0])
        True

    References:
        :cite:t:`vousden2016dynamic`; :cite:t:`foremanmackey2013emcee`;
        :cite:t:`geyer1992practical`.
    """
    _check_integer("burn_in", burn_in, minimum=0)
    if thin != TAU_THINNING:
        if isinstance(thin, str):
            raise ValueError(
                f"thin must be {TAU_THINNING!r} or an integer, got {thin!r}"
            )
        _check_integer("thin", thin, minimum=1)
    temperature = chains.cold_index()
    cold = chains.cold()
    step_count, _, walker_count, _, param_count = cold.shape
    remaining = step_count - burn_in
    if remaining < MINIMUM_STEPS:
        raise ValueError(
            f"burn_in = {burn_in} leaves {max(remaining, 0)} of {step_count} steps;"
            f" a reduction needs at least {MINIMUM_STEPS}"
        )
    after_burn_in = cold.samples[burn_in:, 0, :, 0, :]
    estimate = _estimate_autocorrelation(after_burn_in)
    # ! Python floats only: NumPy flags invalid operations differently per platform.
    taus = tuple(float(tau) for tau in np.asarray(estimate.tau))
    factor = int(np.max(estimate.thin)) if thin == TAU_THINNING else int(thin)
    kept_steps = -(-remaining // factor)
    if kept_steps < MINIMUM_STEPS:
        raise ValueError(
            f"thinning by {factor} keeps {kept_steps} of the {remaining} steps after"
            f" the burn-in; a reduction needs at least {MINIMUM_STEPS}, so the"
            " chains are too short for this factor"
        )
    sample_count = kept_steps * walker_count
    thinned_taus = tuple(
        float(tau)
        for tau in np.asarray(_truncated_tau(estimate.function[::factor], sample_count))
    )
    n_eff = _thinned_effective_sample_size(thinned_taus, sample_count)
    fields = {
        "samples": cold.samples[burn_in::factor, 0, :, 0, :].reshape(
            sample_count, param_count
        ),
        "n_eff": jnp.asarray(n_eff),
    }
    for name in _DENSITY_FIELDS:
        values = getattr(cold, name)
        if values is not None:
            fields[name] = values[burn_in::factor, 0].reshape(sample_count)
    record = ReductionRecord(
        chain_shape=tuple(chains.shape),
        temperature=temperature,
        burn_in=burn_in,
        thin=factor,
        kept_steps=kept_steps,
        sample_count=sample_count,
        tau=taus,
        thinned_tau=thinned_taus,
        n_eff=n_eff,
        chains_hash=hash_chains(chains),
    )
    return Reduction(sample_set=SampleSet(**fields), record=record)


def hash_chains(chains: Chains) -> str:
    """Return the SHA-256 hex digest that a :class:`ReductionRecord` stores.

    Args:
        chains: Chains of one run.

    Returns:
        The digest over ``samples``, ``betas``, ``log_like`` and
        ``log_prior``, those that are set, in this order.

    References:
        :cite:t:`nist2015secure`.
    """
    arrays = (chains.samples, chains.betas, chains.log_like, chains.log_prior)
    return hash_arrays(*(array for array in arrays if array is not None))


def _thinned_effective_sample_size(
    thinned_taus: tuple[float, ...], sample_count: int
) -> float:
    """Return N/τ_k of the slowest moving parameter, clipped to [1, N]."""
    moving = [tau for tau in thinned_taus if math.isfinite(tau)]
    if not moving:
        _warn("no parameter moves after the burn-in, so n_eff is 1")
        return 1.0
    return sample_count / min(max(max(moving), 1.0), sample_count)


def _check_integer(name: str, value: object, minimum: int) -> None:
    """Raise TypeError or ValueError unless the value is an integer >= minimum."""
    if isinstance(value, bool) or not isinstance(value, int | np.integer):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}, got {value}")
