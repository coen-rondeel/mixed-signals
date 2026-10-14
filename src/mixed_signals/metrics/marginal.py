"""One-dimensional marginal metrics: KS, Anderson-Darling, W1, JS and Hellinger.

The two-sample Kolmogorov-Smirnov statistic is implemented; the Anderson-Darling,
W1, Jensen-Shannon and Hellinger metrics are not implemented yet.

Kolmogorov-Smirnov statistic
    D is the largest absolute gap F₁(x) − F₂(x) between the empirical
    distribution functions of two samples (Smirnov 1948). The function of a
    weighted sample puts the normalised weight of each sample at its value, so
    a weight of zero removes a sample, and equal weights give the statistic of
    SciPy's ``ks_2samp`` (Virtanen et al. 2020). The functions are compared at
    the end of each run of equal values, where both are defined, so repeated
    values, such as the rejected proposals of a Markov chain, count once with
    their summed weight.

References:
    :cite:t:`smirnov1948table`; :cite:t:`virtanen2020scipy`.
"""

from collections.abc import Callable
from typing import cast

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


def _jit[**Parameters, Result](
    function: Callable[Parameters, Result],
) -> Callable[Parameters, Result]:
    """Compile a kernel with :func:`jax.jit` and keep its signature for type checks."""
    return cast(Callable[Parameters, Result], jax.jit(function))


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
    values = jnp.concatenate([first, second], axis=-1)
    signed_weights = jnp.concatenate(
        [
            jax.nn.softmax(first_log_weights, axis=-1),
            -jax.nn.softmax(second_log_weights, axis=-1),
        ],
        axis=-1,
    )
    order = jnp.argsort(values, axis=-1)
    sorted_values = jnp.take_along_axis(values, order, axis=-1)
    gaps = jnp.cumsum(jnp.take_along_axis(signed_weights, order, axis=-1), axis=-1)
    # * Inside a run of equal values the partial sums are not values of F₁ − F₂.
    run_ends = jnp.concatenate(
        [
            sorted_values[..., 1:] != sorted_values[..., :-1],
            jnp.ones_like(sorted_values[..., :1], dtype=bool),
        ],
        axis=-1,
    )
    return jnp.max(jnp.where(run_ends, jnp.abs(gaps), 0.0), axis=-1)
