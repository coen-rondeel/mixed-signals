"""Tests for the null plans: random sides, evaluating statistics and the recipe."""

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
import polars as pl
import pytest

from mixed_signals import zoo
from mixed_signals.core.result import NULL_RECIPE_DTYPE
from mixed_signals.metrics.divergence import knn_kl_divergence
from mixed_signals.metrics.kernel import energy_distance, maximum_mean_discrepancy
from mixed_signals.metrics.marginal import (
    marginal_distances,
    marginal_kde_divergences,
    shared_bandwidth_factor,
)
from mixed_signals.metrics.moments import mean_shift
from mixed_signals.metrics.ot.sinkhorn import sinkhorn_divergence
from mixed_signals.metrics.ot.sliced import (
    sliced_directions,
    sliced_sinkhorn_divergence,
    sliced_wasserstein_distance,
)
from mixed_signals.nulls import (
    CONSTRUCTION,
    Construction,
    NullPlan,
    null_draws,
    random_sides,
)
from mixed_signals.nulls.permutation import permutation_null
from mixed_signals.prng import derive_key
from mixed_signals.sigma.maps import null_subtract
from mixed_signals.sigma.significance import empirical_p_value, null_significance

CASE = zoo.shifted_gaussians(0.5, param_count=3)
SPACE = CASE.space
DRAWS = 4

type Statistic = Callable[[jax.Array, jax.Array, jax.Array, jax.Array], jax.Array]


def small_plan(draws: int = DRAWS) -> NullPlan:
    """Return a permutation plan of two zoo sets of 150 samples."""
    sets = CASE.sample(jax.random.key(0), 150).samples[0]
    return permutation_null(
        sets[0], None, sets[1], None, key=jax.random.key(1), draws=draws
    )


def evaluate_in_a_loop(statistic: Statistic, plan: NullPlan) -> jax.Array:
    """Evaluate the statistic on each draw of the plan in a Python loop, draws last."""
    return jnp.moveaxis(
        jnp.stack(
            [
                statistic(
                    plan.samples[first],
                    plan.log_weights[first],
                    plan.samples[second],
                    plan.log_weights[second],
                )
                for first, second in zip(plan.first, plan.second, strict=True)
            ]
        ),
        0,
        -1,
    )


# * Random sides


def test_sides_are_disjoint_and_inside_the_pool() -> None:
    """Each draw holds two sides of the asked size with no position in common."""
    first, second = random_sides(jax.random.key(0), 50, 20, draws=30)
    assert first.shape == second.shape == (30, 20)
    positions = np.concatenate([np.asarray(first), np.asarray(second)], axis=1)
    assert positions.min() >= 0
    assert positions.max() < 50
    for draw in positions:
        assert len(set(draw.tolist())) == 40


def test_sides_may_fill_the_pool() -> None:
    """Two sides of half the pool use every position once."""
    first, second = random_sides(jax.random.key(1), 12, 6, draws=5)
    for draw_first, draw_second in zip(first, second, strict=True):
        assert sorted(np.concatenate([draw_first, draw_second]).tolist()) == list(
            range(12)
        )


def test_each_draw_uses_its_own_key() -> None:
    """Draw i is a permutation from derive_key(key, "draw", i), whatever the number."""
    key = jax.random.key(2)
    few = random_sides(key, 40, 15, draws=2)
    many = random_sides(key, 40, 15, draws=6)
    np.testing.assert_array_equal(few[0], many[0][:2])
    np.testing.assert_array_equal(few[1], many[1][:2])
    order = jax.random.permutation(derive_key(key, "draw", 4), 40)
    np.testing.assert_array_equal(many[0][4], order[:15])
    np.testing.assert_array_equal(many[1][4], order[15:30])


def test_draws_differ_from_each_other() -> None:
    """Different draws give different splits."""
    first, _ = random_sides(jax.random.key(3), 100, 50, draws=3)
    assert not np.array_equal(first[0], first[1])


@pytest.mark.parametrize(
    ("pool_size", "side_size", "draws", "error"),
    [
        (10, 5, 0, ValueError),
        (10, 0, 2, ValueError),
        (0, 1, 2, ValueError),
        (10, 6, 2, ValueError),
        (10.0, 5, 2, TypeError),
        (10, 2.5, 2, TypeError),
        (10, 5, True, TypeError),
        (10, 5, "3", TypeError),
    ],
)
def test_counts_are_checked(
    pool_size: object, side_size: object, draws: object, error: type
) -> None:
    """The counts are integers of at least one, and two sides fit in the pool."""
    with pytest.raises(error):
        random_sides(
            jax.random.key(4),
            pool_size,  # type: ignore[arg-type]
            side_size,  # type: ignore[arg-type]
            draws=draws,  # type: ignore[arg-type]
        )


# * The plan and its recipe


def test_recipe_fits_the_result_schema() -> None:
    """The recipe names the construction, the draws and the side size."""
    plan = small_plan(draws=7)
    recipe = plan.recipe()
    assert recipe == {
        "construction": "permutation",
        "draws": 7,
        "sample_size": plan.first.shape[1],
    }
    row = pl.Series("null_recipe", [recipe], dtype=NULL_RECIPE_DTYPE)
    assert row.dtype == NULL_RECIPE_DTYPE
    assert row[0]["sample_size"] == 150


def test_construction_names_match_the_recipe_values() -> None:
    """The enum holds the three names, and the surrogates keep theirs."""
    assert [member.value for member in Construction] == [
        "permutation",
        "split_half",
        "gaussian_surrogate",
    ]
    assert CONSTRUCTION is Construction.GAUSSIAN_SURROGATE
    assert str(CONSTRUCTION) == "gaussian_surrogate"


# * Statistics on the sides


def weighted_mean_difference(
    first: jax.Array,
    first_log_weights: jax.Array,
    second: jax.Array,
    second_log_weights: jax.Array,
) -> jax.Array:
    """Return the difference of the weighted means of the first parameter."""
    first_mean = jnp.sum(jax.nn.softmax(first_log_weights) * first[:, 0])
    second_mean = jnp.sum(jax.nn.softmax(second_log_weights) * second[:, 0])
    return first_mean - second_mean


def weighted_means_and_widths(
    first: jax.Array,
    first_log_weights: jax.Array,
    second: jax.Array,
    second_log_weights: jax.Array,
) -> jax.Array:
    """Return the mean difference and the width ratio of every parameter."""
    first_weights = jax.nn.softmax(first_log_weights)[:, None]
    second_weights = jax.nn.softmax(second_log_weights)[:, None]
    first_mean = jnp.sum(first_weights * first, axis=0)
    second_mean = jnp.sum(second_weights * second, axis=0)
    first_width = jnp.sqrt(jnp.sum(first_weights * (first - first_mean) ** 2, axis=0))
    second_width = jnp.sqrt(
        jnp.sum(second_weights * (second - second_mean) ** 2, axis=0)
    )
    return jnp.stack([first_mean - second_mean, second_width / first_width])


def test_null_draws_equal_a_loop_for_a_scalar_statistic() -> None:
    """A scalar statistic gives one value per draw, as a Python loop does."""
    plan = small_plan()
    draws = null_draws(weighted_mean_difference, plan)
    assert draws.shape == (DRAWS,)
    np.testing.assert_allclose(
        draws, evaluate_in_a_loop(weighted_mean_difference, plan), rtol=1e-12
    )


def test_null_draws_equal_a_loop_for_a_vector_statistic() -> None:
    """A statistic with a matrix value keeps its shape before the draw axis."""
    plan = small_plan()
    draws = null_draws(weighted_means_and_widths, plan)
    assert draws.shape == (2, 3, DRAWS)
    np.testing.assert_allclose(
        draws, evaluate_in_a_loop(weighted_means_and_widths, plan), rtol=1e-12
    )


def test_draws_fit_the_p_value_and_subtraction_functions_as_they_are() -> None:
    """The draw axis comes last, so the p-value and subtraction read it as it is."""
    plan = small_plan(draws=8)
    draws = null_draws(weighted_means_and_widths, plan)
    value = jnp.full((2, 3), 0.05)
    assert empirical_p_value(value, draws, "upper").shape == (2, 3)
    assert null_significance(value, draws, "two_sided").p_value.shape == (2, 3)
    np.testing.assert_allclose(
        null_subtract("kl", value, draws), value - draws.mean(axis=-1), rtol=1e-12
    )


def test_batches_do_not_change_the_draws() -> None:
    """Evaluating several draws at once gives the same values."""
    plan = small_plan(draws=6)
    np.testing.assert_allclose(
        null_draws(weighted_mean_difference, plan, batch_size=3),
        null_draws(weighted_mean_difference, plan),
        rtol=1e-12,
    )


def test_each_draw_sees_the_sides_of_the_plan() -> None:
    """A statistic that reads the samples gets the rows that the plan names."""
    plan = small_plan()

    def first_row(
        first: jax.Array, _: jax.Array, second: jax.Array, __: jax.Array
    ) -> jax.Array:
        return jnp.stack([first[0], second[0]])

    draws = null_draws(first_row, plan)
    assert draws.shape == (2, 3, DRAWS)
    np.testing.assert_array_equal(
        draws[0].T, plan.samples[plan.first[:, 0]], strict=True
    )
    np.testing.assert_array_equal(
        draws[1].T, plan.samples[plan.second[:, 0]], strict=True
    )


# * Metrics that trace under the map

FACTOR = float(shared_bandwidth_factor(jnp.zeros(150), jnp.zeros(150)))
DIRECTIONS = sliced_directions(SPACE, 20, key=jax.random.key(5))
# * Concrete float64 arrays, as an engine closes over one bandwidth and ε per row.
BANDWIDTH = jnp.array(1.3, dtype=jnp.float64)
EPSILON = jnp.array(0.4, dtype=jnp.float64)
EPSILONS = jnp.array([0.3, 0.8], dtype=jnp.float64)

METRICS: list[tuple[str, Statistic]] = [
    (
        "mean_shift",
        lambda first, first_weights, second, second_weights: mean_shift(
            first, first_weights, second, second_weights, SPACE
        ),
    ),
    (
        "marginal_distances",
        lambda first, first_weights, second, second_weights: jnp.stack(
            marginal_distances(first, first_weights, second, second_weights, SPACE)
        ),
    ),
    (
        "marginal_kde_divergences",
        lambda first, first_weights, second, second_weights: jnp.stack(
            marginal_kde_divergences(
                first, first_weights, second, second_weights, SPACE, factor=FACTOR
            )
        ),
    ),
    (
        "energy_distance",
        lambda first, first_weights, second, second_weights: energy_distance(
            first, first_weights, second, second_weights, SPACE, key=jax.random.key(6)
        ),
    ),
    (
        "knn_kl_divergence",
        lambda first, first_weights, second, second_weights: knn_kl_divergence(
            first, first_weights, second, second_weights, SPACE, key=jax.random.key(7)
        ),
    ),
    (
        "sliced_wasserstein_distance",
        lambda first, first_weights, second, second_weights: (
            sliced_wasserstein_distance(
                first,
                first_weights,
                second,
                second_weights,
                SPACE,
                directions=DIRECTIONS,
            )
        ),
    ),
    (
        "maximum_mean_discrepancy_with_an_array_bandwidth",
        lambda first, first_weights, second, second_weights: maximum_mean_discrepancy(
            first,
            first_weights,
            second,
            second_weights,
            SPACE,
            bandwidth=BANDWIDTH,
            key=jax.random.key(8),
        ),
    ),
    (
        "sinkhorn_divergence_with_an_array_epsilon",
        lambda first, first_weights, second, second_weights: sinkhorn_divergence(
            first,
            first_weights,
            second,
            second_weights,
            SPACE,
            epsilon=EPSILON,
            key=jax.random.key(9),
        ),
    ),
    (
        "sinkhorn_divergence_with_a_vector_of_epsilon",
        lambda first, first_weights, second, second_weights: sinkhorn_divergence(
            first,
            first_weights,
            second,
            second_weights,
            SPACE,
            epsilon=EPSILONS,
            key=jax.random.key(9),
        ),
    ),
    (
        "sliced_sinkhorn_divergence_with_an_array_epsilon",
        lambda first, first_weights, second, second_weights: sliced_sinkhorn_divergence(
            first,
            first_weights,
            second,
            second_weights,
            SPACE,
            directions=DIRECTIONS,
            epsilon=EPSILON,
            key=jax.random.key(10),
        ),
    ),
]


@pytest.mark.parametrize(
    "statistic", [metric for _, metric in METRICS], ids=[name for name, _ in METRICS]
)
def test_package_metrics_trace_under_the_map(statistic: Statistic) -> None:
    """Metrics of the package run under the map and agree with a Python loop."""
    plan = small_plan(draws=3)
    np.testing.assert_allclose(
        null_draws(statistic, plan),
        evaluate_in_a_loop(statistic, plan),
        rtol=1e-9,
        atol=1e-12,
    )
