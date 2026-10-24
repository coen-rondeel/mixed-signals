"""Subsampling cap for metrics whose cost grows as N², with a warning.

Metrics such as the maximum mean discrepancy, the energy distance, distance
correlation, the Sinkhorn divergence and brute-force nearest neighbours cost
O(N²) per evaluation. :func:`cap_samples` gives them at most ``cap`` samples
per posterior: above the cap it draws a uniform random subset without
replacement, as subsampling does (Politis, Romano and Wolf 1999), keeps the
log weights of the samples it keeps, and raises a
:class:`~mixed_signals.errors.SubsamplingWarning`. Each sample with a positive
weight is kept with the same probability, so weighted estimates stay
consistent; samples of zero weight are kept only when too few others remain.
The cost, not the effective sample size, sets the cap, so it compares the
number of samples. :data:`~mixed_signals._config.SAMPLE_CAP` is the default.

References:
    :cite:t:`politis1999subsampling`.
"""

import os
import warnings
from collections.abc import Callable
from typing import Final, cast

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from jaxtyping import Array, Float, PRNGKeyArray

from mixed_signals._config import SAMPLE_CAP
from mixed_signals.errors import SubsamplingWarning

_PACKAGE_DIRECTORY: Final = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def cap_samples(
    samples: ArrayLike,
    log_weights: ArrayLike,
    *,
    key: PRNGKeyArray,
    metric: str,
    cap: int = SAMPLE_CAP,
) -> tuple[Float[Array, "*leading selected param"], Float[Array, "*leading selected"]]:
    """Return at most ``cap`` samples per posterior, drawn uniformly, none twice.

    Below the cap the inputs come back unchanged. Above it, each leading
    index, such as each problem and posterior, draws its own subset from the
    key folded with its flat index. Call this function outside ``jax.jit``,
    where its warning is raised on every call.

    Args:
        samples: Samples shaped (..., N, D).
        log_weights: Their log weights shaped (..., N).
        key: JAX PRNG key, derived with
            :func:`~mixed_signals.prng.derive_key` from the path of the row,
            ending in ``"subsample"``.
        metric: Name of the metric, for the warning.
        cap: Largest number of samples per posterior, at least one.

    Returns:
        The samples, shaped (..., min(N, cap), D), and their log weights,
        shaped (..., min(N, cap)), in their original order.

    Raises:
        ValueError: If the shapes do not fit or ``cap`` is below one.

    Warns:
        SubsamplingWarning: If N exceeds the cap.

    References:
        :cite:t:`politis1999subsampling`.
    """
    if cap < 1:
        raise ValueError(f"cap must be at least 1, got {cap}")
    samples_array = jnp.asarray(samples)
    weights_array = jnp.asarray(log_weights)
    if samples_array.ndim < 2 or weights_array.shape != samples_array.shape[:-1]:
        raise ValueError(
            "samples must be shaped (..., N, D) and log_weights (..., N), got shapes"
            f" {samples_array.shape} and {weights_array.shape}"
        )
    sample_count = samples_array.shape[-2]
    if sample_count <= cap:
        return samples_array, weights_array
    warnings.warn(
        SubsamplingWarning(
            f"{metric}: {sample_count} samples per posterior exceed the cap of {cap};"
            f" the metric runs on {cap} samples drawn uniformly without replacement"
        ),
        skip_file_prefixes=(_PACKAGE_DIRECTORY,),
    )
    return _draw(samples_array, weights_array, key, cap)


def _draw_kernel(
    samples: Float[Array, "*leading sample param"],
    log_weights: Float[Array, "*leading sample"],
    key: PRNGKeyArray,
    cap: int,
) -> tuple[Float[Array, "*leading selected param"], Float[Array, "*leading selected"]]:
    """Keep the cap samples of smallest uniform key, zero weights last, in order."""
    leading = log_weights.shape[:-1]
    row_count = 1
    for size in leading:
        row_count *= size
    row_keys = jax.vmap(jax.random.fold_in, in_axes=(None, 0))(
        key, jnp.arange(row_count)
    )
    flat_weights = log_weights.reshape(row_count, -1)
    uniforms = jax.vmap(
        lambda row_key: jax.random.uniform(row_key, flat_weights.shape[1:])
    )(row_keys)
    # * A key above every uniform puts zero-weight samples last in the order.
    priorities = jnp.where(flat_weights > -jnp.inf, uniforms, 2.0)
    kept = jnp.sort(jnp.argsort(priorities, axis=-1)[:, :cap], axis=-1)
    kept = kept.reshape(*leading, cap)
    return (
        jnp.take_along_axis(samples, kept[..., None], axis=-2),
        jnp.take_along_axis(log_weights, kept, axis=-1),
    )


def _jit[**Parameters, Result](
    function: Callable[Parameters, Result], *static_names: str
) -> Callable[Parameters, Result]:
    """Compile a kernel with :func:`jax.jit` and keep its signature for type checks."""
    return cast(
        Callable[Parameters, Result], jax.jit(function, static_argnames=static_names)
    )


_draw = _jit(_draw_kernel, "cap")
