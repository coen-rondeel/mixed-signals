"""Importance weights to a new prior, a new likelihood or an approximation q.

Samples xᵢ with normalised weights wᵢ represent a density, and weights vᵢ for
a new target density come from the ratio rᵢ of the new to the old unnormalised
density at xᵢ, so that vᵢ ∝ wᵢ rᵢ. The ratio rᵢ depends on the case:

Prior
    The samples represent the posterior L_old π_old. A new prior π_new
    gives rᵢ = π_new(xᵢ)/π_old(xᵢ), from the stored log-prior and the log
    density of the new prior in the same coordinates (Thrane and Talbot
    2019).
Likelihood
    A new likelihood L_new gives rᵢ = L_new(xᵢ)/L_old(xᵢ), as when a
    cheap waveform model is reweighted to an expensive one (Payne, Talbot and
    Thrane 2019). A prior and a likelihood change together as a product.
Approximation
    Samples drawn from a normalised approximation q of the posterior, such as
    a normalizing flow, have equal weights, and the target L π gives
    rᵢ = L(xᵢ) π(xᵢ)/q(xᵢ) (Dax et al. 2023).

Evidence ratio
    The weighted mean of the ratios, Σ wᵢ rᵢ, estimates the ratio Z_new/Z_old
    of the evidences, which is E_old[L_new π_new/(L_old π_old)] (Payne, Talbot
    and Thrane 2019). For an approximation q it is the evidence Z itself, because
    q is normalised (Dax et al. 2023). :func:`log_evidence_ratio` returns its
    logarithm, logsumexp of the new log weights less that of the input ones.
Uncertainty
    For the mean Z = Σ wᵢ rᵢ of the ratios, the squared standard error is
    Σ wᵢ² (rᵢ − Z)² for independent draws (Vehtari et al. 2024, Eq. 5). The
    delta method divides it by Z² for the logarithm, which gives
    Σ wᵢ² (rᵢ/Z − 1)², and with the normalised new weights vᵢ = wᵢ rᵢ/Z that
    is Σ (vᵢ − wᵢ)². Dividing by the relative efficiency R of correlated draws
    gives Eq. 6 of that paper. For equal input weights it equals (1 − ε)/(n ε)
    for ε = Kish/n, the error of Dax et al. (2023, Supplemental Material,
    Eq. 6). The error assumes that the weights have finite variance: the
    reweighting checks of the validation build
    (``scripts/zoo_validation/reweighting.py``) find it calibrated for k̂ up to
    about 0.5 and too small above.
Efficiency
    R is the effective sample size of the input set over its Kish value,
    which is 1 for independent draws, and the reweighted effective sample
    size is R times the Kish value of the new weights, clipped to the range
    that :class:`~mixed_signals.core.samples.SampleSet` admits, as in
    ``psis_n_eff`` of the loo package (Vehtari et al. 2024, Eq. 8; Vehtari
    et al. 2026). The size is the same for every function, so the variance of the
    reweighted mean can be larger or smaller than it implies: (1 + δ²) times
    larger for the mean of a shifted target and about half for the square of a
    narrower one (the reweighting checks of the validation build,
    ``scripts/zoo_validation/reweighting.py``). That is why the rows report it
    without a flag and k̂ carries the flag (Vehtari et al. 2024).

References:
    :cite:t:`payne2019higher`; :cite:t:`thrane2019introduction`;
    :cite:t:`dax2023neural`; :cite:t:`vehtari2024pareto`;
    :cite:t:`vehtari2026loo`; :cite:t:`kish1965survey`.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
from jax.typing import ArrayLike
from jaxtyping import Array, Float

from mixed_signals.core.samples import kish_effective_sample_size


class EvidenceRatio(NamedTuple):
    """The logarithm of an evidence ratio and its standard error."""

    value: Float[Array, " *leading"]
    """The log evidence ratio, ln Z_new/Z_old."""
    error: Float[Array, " *leading"]
    """The standard error of the value, from the delta method."""


def relative_efficiency(
    log_weights: Float[ArrayLike, "*leading sample"],
    n_eff: Float[ArrayLike, " *leading"] | None = None,
) -> Float[Array, " *leading"]:
    """Return the relative efficiency R of correlated draws, n_eff over the Kish value.

    R is 1 for independent draws, whose effective sample size is the Kish
    value of their weights, and below 1 for MCMC draws, whose ``n_eff`` is
    lower. It lengthens the tail of the Pareto fit and sets the error and the
    effective sample size of reweighted estimates (Vehtari et al. 2024,
    Appendix H).

    Args:
        log_weights: Natural logarithms of the weights of the input set;
            −∞ for zero.
        n_eff: The effective sample size of each set, shaped like the leading
            axes of ``log_weights``; ``None`` for independent draws.

    Returns:
        R for each set, shaped like the leading axes of ``log_weights``.

    References:
        :cite:t:`vehtari2024pareto`; :cite:t:`vehtari2026loo`;
        :cite:t:`kish1965survey`.
    """
    values = jnp.asarray(log_weights, dtype=jnp.float64)
    if n_eff is None:
        return jnp.ones(values.shape[:-1])
    return jnp.asarray(n_eff, dtype=jnp.float64) / kish_effective_sample_size(values)


def log_evidence_ratio(
    input_log_weights: Float[ArrayLike, "*leading sample"],
    new_log_weights: Float[ArrayLike, "*leading sample"],
    relative_efficiency: Float[ArrayLike, "..."] = 1.0,
) -> EvidenceRatio:
    """Return ln Z_new/Z_old and its standard error from old and new log weights.

    The value is the logsumexp of the new log weights less that of the input
    ones, which is ln Σ wᵢ rᵢ for the normalised input weights wᵢ and the
    ratios rᵢ. The squared error is Σ (vᵢ − wᵢ)²/R for the normalised new
    weights vᵢ and the relative efficiency R (Vehtari et al. 2024, Eq. 5 and 6,
    with the delta method for the logarithm). A sample with zero input weight
    must have zero new weight. The function acts on the last axis and runs
    under :func:`jax.jit`.

    Args:
        input_log_weights: Natural logarithms of the input weights, shaped
            (..., N); −∞ for zero.
        new_log_weights: Natural logarithms of the new weights, the input
            weights times the ratios, shaped like ``input_log_weights``.
        relative_efficiency: The relative efficiency R, positive, broadcast
            against the leading axes; 1 for independent draws.

    Returns:
        The log evidence ratio and its standard error, shaped like the leading
        axes of the weights.

    References:
        :cite:t:`vehtari2024pareto`; :cite:t:`dax2023neural`;
        :cite:t:`payne2019higher`.
    """
    input_values = jnp.asarray(input_log_weights, dtype=jnp.float64)
    new_values = jnp.asarray(new_log_weights, dtype=jnp.float64)
    value = logsumexp(new_values, axis=-1) - logsumexp(input_values, axis=-1)
    difference = jax.nn.softmax(new_values, axis=-1) - jax.nn.softmax(
        input_values, axis=-1
    )
    variance = jnp.sum(difference**2, axis=-1) / jnp.asarray(
        relative_efficiency, dtype=jnp.float64
    )
    return EvidenceRatio(value=value, error=jnp.sqrt(variance))


def reweighted_effective_sample_size(
    new_log_weights: Float[ArrayLike, "*leading sample"],
    relative_efficiency: Float[ArrayLike, "..."] = 1.0,
) -> Float[Array, " *leading"]:
    """Return the effective sample size of reweighted draws, R times the Kish value.

    The result is clipped to [1, number of positive weights], the range of the
    effective sample size that a :class:`~mixed_signals.core.samples.SampleSet`
    admits. For independent draws it is the Kish value of the new weights.

    Args:
        new_log_weights: Natural logarithms of the new weights, shaped
            (..., N); −∞ for zero.
        relative_efficiency: The relative efficiency R, positive, broadcast
            against the leading axes; 1 for independent draws.

    Returns:
        The effective sample size of each set.

    References:
        :cite:t:`vehtari2024pareto`; :cite:t:`vehtari2026loo`;
        :cite:t:`kish1965survey`.
    """
    values = jnp.asarray(new_log_weights, dtype=jnp.float64)
    positive_count = jnp.sum(values > -jnp.inf, axis=-1).astype(jnp.float64)
    efficiency = jnp.asarray(relative_efficiency, dtype=jnp.float64)
    return jnp.clip(
        efficiency * kish_effective_sample_size(values), 1.0, positive_count
    )
