"""Shared parts of the diagnostics: circle handling, the width limit and pairs."""

import itertools
import math
import os
import warnings
from collections.abc import Callable
from typing import Final, cast

import jax
from jaxtyping import Array, Float

from mixed_signals.core.samples import circular_standard_deviation
from mixed_signals.core.space import ParamSpace
from mixed_signals.core.topology import Topology
from mixed_signals.errors import ValidityWarning

CIRCLE_WIDTH_LIMIT: Final = 1.0
"""Circular standard deviation, in radians of the circle, above which the
surrogate null of a linear statistic on unwrapped values is approximate.

On wrapped normals correlated with a real parameter, linear statistics of
values unwrapped about the circular mean kept their calibration against the
Gaussian surrogates up to a circular standard deviation of 1 rad and drifted
from 1.25 rad, where wrapping changes the dependence that the surrogates
reproduce (zoo check of the validation page).
"""

_PACKAGE_DIRECTORY: Final = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _jit[**Parameters, Result](
    function: Callable[Parameters, Result], *static_names: str
) -> Callable[Parameters, Result]:
    """Compile a kernel with :func:`jax.jit` and keep its signature for type checks."""
    return cast(
        Callable[Parameters, Result], jax.jit(function, static_argnames=static_names)
    )


def _warn(message: str) -> None:
    """Raise a validity warning that points at the caller's line."""
    warnings.warn(ValidityWarning(message), skip_file_prefixes=(_PACKAGE_DIRECTORY,))


def circle_positions(space: ParamSpace) -> tuple[int, ...]:
    """Return the positions of the circle parameters, sphere azimuths included."""
    return tuple(
        position
        for position, parameter in enumerate(space.parameters)
        if parameter.topology is Topology.CIRCLE
    )


def period_of(space: ParamSpace, position: int) -> float:
    """Return the period of the circle parameter at a position."""
    lower, upper = space.parameters[position].bounds
    return upper - lower


def parameter_pairs(param_count: int) -> tuple[tuple[int, int], ...]:
    """Return every pair of parameter positions (i, j) with i < j, in order."""
    return tuple(itertools.combinations(range(param_count), 2))


def wide_circles(
    samples: Float[Array, "sample param"],
    log_weights: Float[Array, " sample"],
    space: ParamSpace,
) -> list[str]:
    """Return the names of circles wider than :data:`CIRCLE_WIDTH_LIMIT`.

    Args:
        samples: Concrete samples shaped (N, D).
        log_weights: Their log weights, shaped (N,).
        space: Parameter space of the samples.

    Returns:
        The names of the circles whose circular standard deviation, in
        radians of the circle, exceeds the limit.
    """
    wide = []
    for position in circle_positions(space):
        period = period_of(space, position)
        width = float(
            circular_standard_deviation(samples[:, position], log_weights, period)
        )
        if width * 2 * math.pi / period > CIRCLE_WIDTH_LIMIT:
            wide.append(space.names[position])
    return wide


def warn_about_wide_circles(names: list[str], rows: str) -> None:
    """Warn that rows involving wide circles have an approximate null."""
    if names:
        _warn(
            f"{rows} involve {', '.join(map(repr, names))}, wider than a circular"
            f" standard deviation of {CIRCLE_WIDTH_LIMIT:g} rad, where the Gaussian"
            " surrogates of values unwrapped about the circular mean no longer"
            " reproduce the wrapped dependence, so their null is approximate"
        )
