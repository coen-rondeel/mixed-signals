"""Sinkhorn divergence: debiased entropic optimal transport with a squared chordal cost.

Entropic transport
    OT_ε(p, q) is the smallest ∫ c dπ + ε KL(π ‖ p ⊗ q) over the couplings π
    of p and q (Cuturi 2013). Sinkhorn's iterations solve it, but
    OT_ε(p, p) is not zero, so its value mixes the difference between p and q
    with their own spreads.
Debiasing
    The Sinkhorn divergence S_ε(p, q) = OT_ε(p, q) − ½ OT_ε(p, p) −
    ½ OT_ε(q, q) removes that bias, so S_ε(p, p) = 0 (Genevay, Peyré and
    Cuturi 2018). It interpolates between the squared W2 distance as ε → 0
    and half the squared MMD of the kernel −c as ε → ∞ (Feydy et al. 2019),
    which for the squared distance is the squared distance between the
    means. :func:`sinkhorn_divergence` returns S_ε of the weighted
    empirical distributions; rows read the metric ``sinkhorn`` with the
    estimator ``empirical``.
Cost
    The squared Euclidean distance of the chordal embedding, in which each
    circle of period P is a circle of circumference P and each sphere a
    sphere of its radius, the embedding of the maximum mean discrepancy and
    the energy distance in :mod:`mixed_signals.metrics.kernel`. Its Gibbs
    kernel exp(−c/ε) is a
    Gaussian kernel of the embedded points, positive definite and universal
    at every ε, so S_ε ≥ 0 and S_ε = 0 only for equal distributions
    (Feydy et al. 2019, Theorem 1). Squared arc lengths would not keep this:
    a geodesic Gaussian kernel is positive definite only on flat spaces
    (Feragen, Lauze and Hauberg 2015). In a zoo check, squared arcs gave
    S_ε of −0.14 to −0.23 at ε of 5 to 125 times the median pooled cost for
    a circle with modes at 0 and π against modes at ±π/2, 77 to 210 null
    widths below the null, and −0.57 to −0.93 for a bipolar against a girdle
    distribution on the sphere, while squared chords stayed non-negative.
    Chords are shorter than arcs, so shifts on wide circles and spheres
    read low.
Shifts
    For a shift Δ of any distribution in the Euclidean block,
    OT_ε(p, p + Δ) = OT_ε(p, p) + ‖Δ‖² (Janati et al. 2020, Lemma 1), so
    S_ε = ‖Δ‖² at every ε and √S_ε is σ_eff in whitened coordinates
    (Peyré and Cuturi 2019).
Regularisation
    ε is fixed once per comparison and reused in every null draw:
    :data:`SINKHORN_EPSILON_FACTOR` times ℓ², with ℓ the median bandwidth of
    :func:`~mixed_signals.metrics.kernel.median_bandwidth`, whose square is
    the median squared chord between pooled samples. ott-jax likewise scales
    its default ε with a statistic of the cost (Cuturi et al. 2022). A
    vector of ε gives the curve S_ε(ε) in one call.
Solver
    ott-jax (Cuturi et al. 2022) iterates in the log domain, with the
    symmetric updates of Feydy et al. (2019) for the two self terms, and
    recomputes the costs in chunks of about 2²² entries, so memory stays
    bounded. Iterations stop when the L1 error of the coupling's marginals
    falls below :data:`SINKHORN_THRESHOLD`, or after
    :data:`SINKHORN_MAX_ITERATIONS`; a divergence that has not converged by
    then is NaN, with a :class:`~mixed_signals.errors.ValidityWarning`.
Cost of a call
    Each iteration costs O(N²), so both posteriors are capped at
    :data:`~mixed_signals._config.SAMPLE_CAP` samples first. At the cap, in
    16 dimensions at ε = 0.25 ℓ², one divergence took 6.5 s and about 1 GB
    on an Apple M4 Pro CPU, against 3.7 s and about 5 GB with the cost
    matrices held in memory, so 200 null draws take about 22 minutes.

References:
    :cite:t:`cuturi2013sinkhorn`; :cite:t:`genevay2018learning`;
    :cite:t:`feydy2019interpolating`; :cite:t:`feragen2015geodesic`;
    :cite:t:`janati2020entropic`; :cite:t:`peyre2019computational`;
    :cite:t:`cuturi2022optimal`; :cite:t:`politis1999subsampling`.
"""

import functools
from typing import Final

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike
from jaxtyping import Array, Bool, Float, PRNGKeyArray
from ott.geometry import costs, pointcloud
from ott.tools import sinkhorn_divergence as ott_divergence

from mixed_signals._config import SAMPLE_CAP
from mixed_signals.core.samples import _is_concrete
from mixed_signals.core.space import ParamSpace
from mixed_signals.estimators.base import CHUNK_ENTRIES, _jit, _warn
from mixed_signals.metrics._cap import read_and_cap_pair
from mixed_signals.metrics._embedding import embed

SINKHORN_EPSILON_FACTOR: Final = 0.25
"""Factor of ℓ², the squared median bandwidth, that sets ε once per comparison.

In a zoo check on whitened Gaussians with 1 000 samples per posterior, 0.25
kept the most resolution of a width ratio of 1.3 on one axis in 2, 8 and
16 dimensions, and 72 to 85% of the best resolution of a 0.5σ shift in 8
and 16 dimensions, where a factor of 1 gained 20 to 30% on the shift but
lost most of the width and 0.05 lost half the shift at 16 dimensions with
6 to 20 times the iterations (see :ref:`validation-sinkhorn-epsilon`).
"""

SINKHORN_THRESHOLD: Final = 1e-6
"""L1 error of the coupling's marginals at which the Sinkhorn iterations stop.

At ε = 0.25 ℓ² the default of ott-jax, 1e-3, already moved S_ε by less than
a thousandth of its permutation null's spread at 10 000 samples in 16
dimensions; 1e-6 leaves a margin for smaller ε and few dimensions, at 1.7
times the time.
"""

SINKHORN_MAX_ITERATIONS: Final = 2_000
"""Largest number of Sinkhorn iterations, the default of ott-jax."""


def sinkhorn_divergence(
    first: ArrayLike,
    first_log_weights: ArrayLike | None,
    second: ArrayLike,
    second_log_weights: ArrayLike | None,
    space: ParamSpace,
    *,
    epsilon: Float[ArrayLike, " *leading"],
    key: PRNGKeyArray,
    cap: int = SAMPLE_CAP,
) -> Float[Array, " *leading"]:
    """Return the Sinkhorn divergence S_ε between two weighted samples.

    Args:
        first: Samples of the first posterior, shaped (N, D), in the
            parameter order of ``space``.
        first_log_weights: Its log weights shaped (N,); equal weights when
            ``None``.
        second: Samples of the second posterior, shaped (M, D).
        second_log_weights: Its log weights shaped (M,); equal weights when
            ``None``.
        space: Space of the samples.
        epsilon: Entropic regularisation ε, positive, in the squared units of
            the coordinates: :data:`SINKHORN_EPSILON_FACTOR` times the
            squared median bandwidth, fixed once per comparison. A vector of
            ε gives the divergence at each.
        key: JAX PRNG key for the subsample above the cap, derived with
            :func:`~mixed_signals.prng.derive_key` from the path of the row,
            ending in ``"subsample"``.
        cap: Largest number of samples per posterior.

    Returns:
        S_ε for each ε, shaped like ``epsilon``, in the squared units of the
        coordinates; NaN where the iterations did not converge.

    Raises:
        ValueError: If the shapes do not fit the space, concrete values are
            invalid, a concrete ε is not positive and finite, or ``cap`` is
            below one.
        TypeError: If an input is not real-valued.

    Warns:
        SubsamplingWarning: If a posterior has more samples than the cap.
        ValidityWarning: If concrete iterations did not converge for some ε.

    References:
        :cite:t:`genevay2018learning`; :cite:t:`feydy2019interpolating`;
        :cite:t:`cuturi2022optimal`.
    """
    epsilons = jnp.asarray(epsilon, dtype=jnp.float64)
    if _is_concrete(epsilons):
        # * Test host values: a jnp test of a closed-over array is staged in a trace.
        host_epsilons = np.asarray(epsilons)
        if not np.all(np.isfinite(host_epsilons) & (host_epsilons > 0)):
            raise ValueError(
                f"epsilon must be positive and finite, got {host_epsilons}"
            )
    arrays = read_and_cap_pair(
        first,
        first_log_weights,
        second,
        second_log_weights,
        space,
        key=key,
        metric="sinkhorn",
        cap=cap,
    )
    values, converged = _sinkhorn(*arrays, space, epsilons.reshape(-1))
    if _is_concrete(converged) and not bool(jnp.all(converged)):
        failed = [float(value) for value in epsilons.reshape(-1)[~converged]]
        _warn(
            "sinkhorn: the iterations did not reach the marginal error"
            f" {SINKHORN_THRESHOLD:g} within {SINKHORN_MAX_ITERATIONS} iterations"
            f" for epsilon = {', '.join(f'{value:.4g}' for value in failed)}; the"
            " divergence is NaN there, and a larger epsilon converges faster"
        )
    return jnp.where(converged, values, jnp.nan).reshape(epsilons.shape)


def _one_divergence(
    first_points: Float[Array, "sample _"],
    first_weights: Float[Array, " sample"],
    second_points: Float[Array, "other _"],
    second_weights: Float[Array, " other"],
    epsilon: Float[Array, ""],
) -> tuple[Float[Array, ""], Bool[Array, ""]]:
    """Return S_ε at one ε and whether its three problems converged."""
    chunk = max(1, CHUNK_ENTRIES // max(first_points.shape[0], second_points.shape[0]))
    value, output = ott_divergence.sinkhorn_divergence(
        pointcloud.PointCloud,
        x=first_points,
        y=second_points,
        a=first_weights,
        b=second_weights,
        cost_fn=costs.SqEuclidean(),
        epsilon=epsilon,
        batch_size=chunk,
        solve_kwargs={
            "threshold": SINKHORN_THRESHOLD,
            "max_iterations": SINKHORN_MAX_ITERATIONS,
        },
    )
    return value, jnp.all(jnp.asarray(output.converged))


def _sinkhorn_kernel(
    first: Float[Array, "sample param"],
    first_log_weights: Float[Array, " sample"],
    second: Float[Array, "other param"],
    second_log_weights: Float[Array, " other"],
    space: ParamSpace,
    epsilons: Float[Array, " selected"],
) -> tuple[Float[Array, " selected"], Bool[Array, " selected"]]:
    """Return S_ε in the chordal embedding at each ε, one ε after the other."""
    divergence = functools.partial(
        _one_divergence,
        embed(first, space),
        jax.nn.softmax(first_log_weights),
        embed(second, space),
        jax.nn.softmax(second_log_weights),
    )
    values, converged = jax.lax.map(divergence, epsilons)
    return values, converged


_sinkhorn = _jit(_sinkhorn_kernel, "space")
