"""Helpers that several topics of the validation page share.

A replicate draws two sample sets from a zoo case, builds the null of the
comparison with the package, and evaluates a statistic on the full sets and on
every draw of the null. The helpers here only join those package calls.
"""

import functools
from collections.abc import Callable, Sequence
from typing import NamedTuple, cast

import jax
import jax.numpy as jnp
from jaxtyping import Array

from mixed_signals.core.space import ParamSpace
from mixed_signals.metrics.marginal import ks_statistic
from mixed_signals.metrics.moments import mean_shift, width_ratio
from mixed_signals.metrics.ot.sliced import sliced_wasserstein_distance
from mixed_signals.nulls import Construction, NullPlan, null_draws
from mixed_signals.nulls.plan import Statistic
from mixed_signals.sigma.significance import null_significance
from mixed_signals.sigma.thresholds import comparison_tail
from zoo_validation.report import Measurement

LEVELS = (0.01, 0.05, 0.10)
"""Significance levels of the rejection rates."""

FIRST_PARAMETER_METRICS = ("mean_shift", "ks", "width_ratio")
"""Metrics of the first parameter that the null checks read."""

DISPLAY_NAMES = {
    "mean_shift": "mean shift",
    "ks": "KS",
    "width_ratio": "log width ratio",
    "energy": "energy distance",
    "sliced_w2": "sliced W2",
}
"""Names of the statistics in the tables."""


class Comparison(NamedTuple):
    """The value of a statistic on the full sets and on every null draw."""

    value: Array
    draws: Array
    """Shaped like the value, with the draw axis last."""


def equal_log_weights(count: int) -> Array:
    """Return the log weights of ``count`` equally weighted samples."""
    return jnp.zeros(count)


def compare_on_plan(
    statistic: Statistic,
    plan: NullPlan,
    first: Array,
    first_log_weights: Array | None,
    second: Array,
    second_log_weights: Array | None,
) -> Comparison:
    """Evaluate a statistic on the two full sets and on every draw of a plan."""
    first_weights = (
        equal_log_weights(first.shape[0])
        if first_log_weights is None
        else first_log_weights
    )
    second_weights = (
        equal_log_weights(second.shape[0])
        if second_log_weights is None
        else second_log_weights
    )
    value = statistic(first, first_weights, second, second_weights)
    return Comparison(value, null_draws(statistic, plan))


type Builder = Callable[..., Statistic]
"""Maps the settings of one comparison, as arrays, to the statistic of its draws."""


def _null_draws_of_builder(
    builder: Builder,
    constants: tuple[Array, ...],
    pool: Array,
    log_weights: Array,
    first_positions: Array,
    second_positions: Array,
) -> Array:
    plan = NullPlan(
        Construction.PERMUTATION, pool, log_weights, first_positions, second_positions
    )
    return null_draws(builder(*constants), plan)


_JITTED_NULL_DRAWS = jax.jit(_null_draws_of_builder, static_argnames=("builder",))


def compare_with_constants(
    builder: Builder,
    constants: tuple[Array, ...],
    plan: NullPlan,
    first: Array,
    first_log_weights: Array | None,
    second: Array,
    second_log_weights: Array | None,
) -> Comparison:
    """Like :func:`compare_on_plan`, with the settings of the comparison as arguments.

    The draws run in one compiled program that takes the settings, such as a
    bandwidth or the directions, as arguments, so that a new comparison reuses
    the program of the last one. ``builder`` must be the same function object
    for every comparison of a check.
    """
    statistic = builder(*constants)
    first_weights = (
        equal_log_weights(first.shape[0])
        if first_log_weights is None
        else first_log_weights
    )
    second_weights = (
        equal_log_weights(second.shape[0])
        if second_log_weights is None
        else second_log_weights
    )
    value = statistic(first, first_weights, second, second_weights)
    draws = cast(
        Array,
        _JITTED_NULL_DRAWS(
            builder,
            constants,
            plan.samples,
            plan.log_weights,
            plan.first,
            plan.second,
        ),
    )
    # * JAX runs asynchronously; wait, so that a later timing starts on an idle device.
    draws.block_until_ready()
    return Comparison(value, draws)


@functools.cache
def sliced_w2_builder(space: ParamSpace) -> Builder:
    """Return the builder of the empirical sliced W2 statistic of the directions."""

    def build(directions: Array) -> Statistic:
        def statistic(
            first: Array, first_weights: Array, second: Array, second_weights: Array
        ) -> Array:
            return sliced_wasserstein_distance(
                first,
                first_weights,
                second,
                second_weights,
                space,
                directions=directions,
            )

        return statistic

    return build


def first_parameter_statistics(space: ParamSpace) -> dict[str, Statistic]:
    """Return the mean shift, KS and log width ratio of the first parameter."""

    def shift(
        first: Array, first_weights: Array, second: Array, second_weights: Array
    ) -> Array:
        return mean_shift(first, first_weights, second, second_weights, space)[0]

    def kolmogorov_smirnov(
        first: Array, first_weights: Array, second: Array, second_weights: Array
    ) -> Array:
        return ks_statistic(first[:, 0], first_weights, second[:, 0], second_weights)

    def log_width_ratio(
        first: Array, first_weights: Array, second: Array, second_weights: Array
    ) -> Array:
        return jnp.log(
            width_ratio(first, first_weights, second, second_weights, space)[0]
        )

    return {
        "mean_shift": shift,
        "ks": kolmogorov_smirnov,
        "width_ratio": log_width_ratio,
    }


def empirical_p_values(values: Array, draws: Array, metric: str) -> Array:
    """Return the empirical p-values of replicates, without the tail fit.

    Args:
        values: The statistic of each replicate, shaped (replicates,).
        draws: Its null draws, shaped (replicates, draws).
        metric: Name of the metric, which sets the tail.

    Returns:
        The p-values (1 + exceedances) / (1 + draws), whose levels are exact
        for a number of draws one below a multiple of 100.
    """
    return null_significance(
        values, draws, comparison_tail(metric), tail_fit=False
    ).p_value


def rejection_count(p_values: Array, level: float) -> int:
    """Count the replicates whose p-value is at or below a level."""
    return int(jnp.sum(p_values <= level + 1e-9))


def rate_measurements(
    p_values: Array, levels: Sequence[float] = LEVELS
) -> list[Measurement]:
    """Return the rejection rate at each level with its binomial error at the level."""
    trials = int(p_values.shape[0])
    return [
        Measurement.from_rate(rejection_count(p_values, level), trials, level=level)
        for level in levels
    ]


def rate_counts(
    p_values: Array, levels: Sequence[float] = LEVELS
) -> list[tuple[int, int]]:
    """Return the rejections and the trials at each level."""
    trials = int(p_values.shape[0])
    return [(rejection_count(p_values, level), trials) for level in levels]


def collect(
    function: Callable[[int], Comparison], replicates: int
) -> tuple[Array, Array]:
    """Run ``function`` for each replicate and stack the values and the draws."""
    comparisons = [function(replicate) for replicate in range(replicates)]
    return (
        jnp.stack([comparison.value for comparison in comparisons]),
        jnp.stack([comparison.draws for comparison in comparisons]),
    )


def finite_measurement(values: Array) -> Measurement | None:
    """Return the mean and standard error of the finite values, or ``None`` if none."""
    finite = values[jnp.isfinite(values)]
    if finite.shape[0] == 0:
        return None
    return Measurement.from_replicates(finite)


class ResolutionRuns(NamedTuple):
    """The signal and the null variance of each replicate of one statistic."""

    signals: Sequence[float]
    """Value minus null mean, per replicate."""
    variances: Sequence[float]
    """Variance of the null draws, per replicate."""


def _jackknife(statistic: Callable[[Array], Array], count: int) -> Measurement:
    """Return a statistic of the replicates with its leave-one-out standard error.

    ``statistic`` takes a mask over the replicates and returns the value from
    the replicates that the mask keeps.
    """
    keep = jnp.ones(count, dtype=bool)
    left_out = jax.vmap(statistic)(~jnp.eye(count, dtype=bool))
    spread = jnp.sum((left_out - jnp.mean(left_out)) ** 2)
    error = float(jnp.sqrt((count - 1) / count * spread))
    return Measurement(float(statistic(keep)), error, count)


def _finite_runs(*runs: ResolutionRuns) -> tuple[Array, ...]:
    """Return the signals and variances of every run on the finite replicates."""
    columns = [jnp.asarray(values) for entry in runs for values in entry]
    finite = jnp.all(jnp.stack([jnp.isfinite(column) for column in columns]), axis=0)
    return tuple(column[finite] for column in columns)


def pooled_resolution(runs: ResolutionRuns) -> Measurement:
    """Return the resolution: the mean signal over the pooled null width.

    The null width is the root of the mean null variance over the replicates,
    because every replicate draws the same null again, and the error is the
    leave-one-out error over the replicates.
    """
    signals, variances = _finite_runs(runs)

    def statistic(mask: Array) -> Array:
        return jnp.sum(jnp.where(mask, signals, 0.0)) / jnp.sqrt(
            jnp.sum(jnp.where(mask, variances, 0.0)) * jnp.sum(mask)
        )

    return _jackknife(statistic, int(signals.shape[0]))


def resolution_ratio(runs: ResolutionRuns, reference: ResolutionRuns) -> Measurement:
    """Return the resolution of ``runs`` over that of ``reference``, paired.

    Both come from the same replicates, so the leave-one-out error keeps the
    pairing. The ratio is that of the two pooled resolutions.
    """
    signals, variances, reference_signals, reference_variances = _finite_runs(
        runs, reference
    )

    def statistic(mask: Array) -> Array:
        def resolution(values: Array, spreads: Array) -> Array:
            return jnp.sum(jnp.where(mask, values, 0.0)) / jnp.sqrt(
                jnp.sum(jnp.where(mask, spreads, 0.0)) * jnp.sum(mask)
            )

        return resolution(signals, variances) / resolution(
            reference_signals, reference_variances
        )

    return _jackknife(statistic, int(signals.shape[0]))


def sample_shift(first: Array, second: Array) -> float:
    """Return the difference of the means along the first axis, in widths.

    It is the shift that two sample sets of unit width hold, and the truth of a
    paired comparison.
    """
    return float(jnp.mean(second[:, 0]) - jnp.mean(first[:, 0]))


def dimension_of(space: ParamSpace) -> int:
    """Return D = E + C + 2S, the dimension that the kernel maps read."""
    blocks = space.axis_blocks
    return len(blocks.euclidean) + len(blocks.circles) + 2 * len(blocks.spheres)


def embedding_dimension_of(space: ParamSpace) -> int:
    """Return n = E + 2C + 3S, the coordinates that the sliced directions span."""
    blocks = space.axis_blocks
    return len(blocks.euclidean) + 2 * len(blocks.circles) + 3 * len(blocks.spheres)


def real_space(count: int) -> ParamSpace:
    """Return a space of ``count`` real parameters."""
    return ParamSpace.model_validate(
        {"parameters": [{"name": f"x{index}"} for index in range(count)]}
    )
