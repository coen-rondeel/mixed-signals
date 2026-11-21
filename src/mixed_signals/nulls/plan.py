"""Null plans: the draws of a null as positions in a pool of samples.

A permutation null and a split-half null both pool samples and cut the pool
into two sides, again and again. :class:`NullPlan` holds the pool, with its
log weights, and the positions of the two sides of every draw, so that a
statistic can run on the draws without copying the pool: :func:`null_draws`
evaluates it on the two sides of each draw. The constructions of
:mod:`mixed_signals.nulls.permutation` and
:mod:`mixed_signals.nulls.split_half` build plans, and :func:`random_sides`
draws the positions.

Sides
    Each draw shuffles the pool positions and takes the first
    ``side_size`` of them as side one and the next ``side_size`` as side two,
    so the two sides never share a sample. Under the null that the sides come
    from one distribution, a random split is as likely as the observed one
    (Good 2005).
Keys
    Draw i uses ``derive_key(key, "draw", i)``, so a draw does not depend on
    how many others are drawn.
Statistic
    A statistic takes the samples and log weights of side one and of side
    two, as the metrics of :mod:`mixed_signals.metrics` do, and may return an
    array of any shape, for example one divergence per parameter or per ε.
    :func:`null_draws` puts the draw axis last, so that its result fits
    :func:`~mixed_signals.sigma.significance.null_significance` and
    :func:`~mixed_signals.sigma.maps.null_subtract` as it is: a scalar
    statistic gives (M,), one shaped (D,) gives (D, M).
Recipe
    Rows name the construction, the number of draws and the sides' size in
    the ``null_recipe`` column of the result table, with the tail of the
    p-value; :meth:`NullPlan.recipe` gives them.

References:
    :cite:t:`good2005permutation`.
"""

import math
from collections.abc import Callable
from enum import StrEnum
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from jaxtyping import Array, Float, Int, PRNGKeyArray

from mixed_signals._config import NULL_DRAWS
from mixed_signals.core.samples import (
    _check_real,
    _is_concrete,
    _warn_narrow_floats,
    kish_effective_sample_size,
)
from mixed_signals.prng import derive_key
from mixed_signals.sigma.significance import Tail

type Statistic = Callable[[Array, Array, Array, Array], Array]
"""A statistic of two weighted sides: samples and log weights of each, in turn."""


class Construction(StrEnum):
    """The constructions of a null, as named in the ``null_recipe`` column.

    ``permutation``
        Pooled samples of both sets, split into two random sides.
    ``split_half``
        The reference alone, split into two random sides.
    ``gaussian_surrogate``
        Surrogate sets drawn from the moment-matched Gaussian of one
        posterior.
    """

    PERMUTATION = "permutation"
    SPLIT_HALF = "split_half"
    GAUSSIAN_SURROGATE = "gaussian_surrogate"


class NullPlan(NamedTuple):
    """The pool of a null and the two sides of each of its draws."""

    construction: str
    """The construction, a value of :class:`Construction`."""
    samples: Float[Array, "sample param"]
    """The pooled samples."""
    log_weights: Float[Array, " sample"]
    """Log weights of the pooled samples."""
    first: Int[Array, "draw selected"]
    """Positions in the pool of side one, for every draw."""
    second: Int[Array, "draw selected"]
    """Positions in the pool of side two, for every draw."""

    def recipe(self, tail: Tail | str | None = None) -> dict[str, str | int | None]:
        """Return the entry of the ``null_recipe`` column of a result row.

        Args:
            tail: The tail of the p-value that the row reads from the null,
                ``"upper"`` or ``"two_sided"``; ``None`` leaves it null.

        Returns:
            The construction, the number of draws, the sample size per side
            and the tail, under the field names of
            :data:`~mixed_signals.core.result.NULL_RECIPE_DTYPE`.

        Raises:
            ValueError: If the tail is not a value of
                :class:`~mixed_signals.sigma.significance.Tail`.
        """
        return {
            "construction": str(self.construction),
            "draws": int(self.first.shape[0]),
            "sample_size": int(self.first.shape[1]),
            "tail": None if tail is None else str(Tail(tail)),
        }


def random_sides(
    key: PRNGKeyArray,
    pool_size: int,
    side_size: int,
    *,
    draws: int = NULL_DRAWS,
) -> tuple[Int[Array, "draw selected"], Int[Array, "draw selected"]]:
    """Draw two disjoint random sides of a pool, for every draw.

    Args:
        key: JAX PRNG key; draw i uses ``derive_key(key, "draw", i)``.
        pool_size: Number of samples in the pool.
        side_size: Number of samples in each side.
        draws: Number of draws.

    Returns:
        The positions of side one and of side two, each shaped
        (draws, side_size).

    Raises:
        TypeError: If a count is not an integer.
        ValueError: If a count is below one, or the two sides do not fit in
            the pool.

    References:
        :cite:t:`good2005permutation`.
    """
    for name, count in (
        ("pool_size", pool_size),
        ("side_size", side_size),
        ("draws", draws),
    ):
        if isinstance(count, bool) or not isinstance(count, int):
            raise TypeError(f"{name} must be an integer, got {type(count).__name__}")
        if count < 1:
            raise ValueError(f"{name} must be at least 1, got {count}")
    if 2 * side_size > pool_size:
        raise ValueError(
            f"two sides of {side_size} samples need a pool of at least"
            f" {2 * side_size}, got {pool_size}"
        )
    keys = jnp.stack([derive_key(key, "draw", index) for index in range(draws)])
    orders = jax.vmap(partial(jax.random.permutation, x=pool_size))(keys)
    return orders[:, :side_size], orders[:, side_size : 2 * side_size]


def null_draws(
    statistic: Statistic,
    plan: NullPlan,
    *,
    batch_size: int | None = None,
) -> Float[Array, "... draw"]:
    """Evaluate a statistic on the two sides of every draw of a plan.

    The draws run through :func:`jax.lax.map`, so the statistic must be
    traceable, and the pool is not copied once per draw.

    Args:
        statistic: Function of (first samples, first log weights, second
            samples, second log weights) that returns an array of any shape.
        plan: The pool and the sides, for example from
            :func:`~mixed_signals.nulls.permutation.permutation_null`.
        batch_size: Draws evaluated at once; one at a time when ``None``.

    Returns:
        The values of all draws, shaped (..., draws). The leading axes are
        those of the statistic's value and the draw axis comes last, as the
        p-value and null subtraction functions read it.
    """
    values: Array = jax.lax.map(
        partial(
            _statistic_on_sides,
            statistic=statistic,
            samples=plan.samples,
            log_weights=plan.log_weights,
        ),
        (plan.first, plan.second),
        batch_size=batch_size,
    )
    return jnp.moveaxis(values, 0, -1)


def _statistic_on_sides(
    sides: tuple[Int[Array, " selected"], Int[Array, " selected"]],
    *,
    statistic: Statistic,
    samples: Float[Array, "sample param"],
    log_weights: Float[Array, " sample"],
) -> Array:
    """Run the statistic on the two sides of one draw."""
    first, second = sides
    return statistic(
        samples[first], log_weights[first], samples[second], log_weights[second]
    )


def _read_set(
    name: str, samples: ArrayLike, log_weights: ArrayLike | None
) -> tuple[Float[Array, "sample param"], Float[Array, " sample"]]:
    """Return the samples (N, D) and the normalised log weights (N,) of one set.

    Args:
        name: Name of the set, for the messages.
        samples: Samples shaped (N, D).
        log_weights: Log weights shaped (N,); equal weights when ``None``.

    Returns:
        The samples as float64 and ``log_softmax`` of the log weights, so that
        the weights of the set sum to one.

    Raises:
        TypeError: If an input is not real-valued.
        ValueError: If the shapes do not fit, or concrete values are invalid.
    """
    samples_array = jnp.asarray(samples)
    _check_real(f"{name} samples", samples_array)
    if samples_array.ndim != 2 or samples_array.shape[0] == 0:
        raise ValueError(
            f"{name} samples must be shaped (N, D) with at least one sample,"
            f" got shape {samples_array.shape}"
        )
    arrays = {f"{name} samples": samples_array}
    if log_weights is not None:
        weights_array = jnp.asarray(log_weights)
        _check_real(f"{name} log_weights", weights_array)
        if weights_array.shape != samples_array.shape[:1]:
            raise ValueError(
                f"{name} log_weights must be shaped {samples_array.shape[:1]} for"
                f" samples of shape {samples_array.shape}, got shape"
                f" {weights_array.shape}"
            )
        arrays[f"{name} log_weights"] = weights_array
    _warn_narrow_floats(arrays)
    samples_array = samples_array.astype(jnp.float64)
    weights_array = (
        arrays[f"{name} log_weights"].astype(jnp.float64)
        if log_weights is not None
        else jnp.zeros(samples_array.shape[0])
    )
    if _is_concrete(samples_array, weights_array):
        if not bool(jnp.all(jnp.isfinite(samples_array))):
            raise ValueError(f"{name} samples must be finite")
        if bool(jnp.any(jnp.isnan(weights_array) | (weights_array == jnp.inf))):
            raise ValueError(f"{name} log_weights must not contain NaN or +inf")
        if not bool(jnp.any(weights_array > -jnp.inf)):
            raise ValueError(f"{name} needs at least one sample with a positive weight")
    return samples_array, jax.nn.log_softmax(weights_array)


def _set_sizes(
    name: str, log_weights: ArrayLike, n_eff: float | None
) -> tuple[int, float, float]:
    """Return the sample count N, the Kish size K and the effective size n of a set.

    Args:
        name: Name of the set, for the messages.
        log_weights: Concrete log weights shaped (N,).
        n_eff: The effective sample size n of a correlated set; the Kish size
            when ``None``.

    Returns:
        N, K and n, where n equals K if no ``n_eff`` is given.

    Raises:
        ValueError: If the log weights are not one-dimensional, or a size is
            below one or not finite.
    """
    weights = jnp.asarray(log_weights, dtype=jnp.float64)
    if weights.ndim != 1:
        raise ValueError(
            f"the log weights of {name} must be shaped (N,), got shape {weights.shape}"
        )
    kish_size = float(kish_effective_sample_size(weights))
    effective_size = kish_size if n_eff is None else float(n_eff)
    _check_size("Kish", name, kish_size)
    _check_size("effective", name, effective_size)
    return weights.shape[0], kish_size, effective_size


def _check_size(label: str, name: str, size: float) -> None:
    """Raise ValueError unless a sample size of a set is finite and at least one."""
    if not (math.isfinite(size) and size >= 1):
        raise ValueError(
            f"the {label} sample size of {name} must be finite and at least 1,"
            f" got {size}"
        )
