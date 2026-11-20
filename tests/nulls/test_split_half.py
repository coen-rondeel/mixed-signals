"""Tests for the split-half null: side sizes, the clip warning and calibration."""

import math
from collections.abc import Callable
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mixed_signals import zoo
from mixed_signals.errors import ValidityWarning
from mixed_signals.nulls import (
    Construction,
    NullPlan,
    null_draws,
    random_sides,
    split_half_null,
    split_half_side_size,
)
from mixed_signals.prng import derive_key
from mixed_signals.sigma.significance import empirical_p_value

STANDARD_ERRORS = 4.0
LEVEL = 0.1
REPLICATES = 1_000
PERMUTATIONS = 99
REFERENCE_SAMPLES = 1_000
APPROXIMATION_SAMPLES = 100

type Statistic = Callable[[jax.Array, jax.Array, jax.Array, jax.Array], jax.Array]


# * Side sizes


def test_equal_weights_give_the_harmonic_mean_of_the_effective_sizes() -> None:
    """A reference of 10 000 and an approximation of 1 000 samples give 1 818."""
    size = split_half_side_size(jnp.zeros(10_000), approximation_n_eff=1_000.0)
    assert size == round(2 / (1 / 10_000 + 1 / 1_000))
    assert size == 1_818


def test_the_kish_ratio_enters_the_size() -> None:
    """Weights that halve the effective size double the factor N_r/K_r."""
    log_weights = jnp.where(jnp.arange(1_000) < 500, 0.0, -jnp.inf)
    size = split_half_side_size(log_weights, approximation_n_eff=100.0)
    # * N_r/K_r = 2 and n_r = K_r = 500.
    assert size == round(2 * 2 / (1 / 500 + 1 / 100))
    assert size == 333


def test_a_given_reference_effective_size_replaces_the_kish_size() -> None:
    """Correlated reference samples carry their effective sample size n_r."""
    size = split_half_side_size(
        jnp.zeros(3_000), approximation_n_eff=500.0, reference_n_eff=1_000.0
    )
    assert size == round(2 / (1 / 1_000 + 1 / 500))
    assert size == 667


def test_the_size_is_at_least_two() -> None:
    """An approximation with one effective sample gives sides of two samples."""
    assert split_half_side_size(jnp.zeros(1_000), approximation_n_eff=1.0) == 2


def test_a_reference_three_times_the_approximation_just_fits() -> None:
    """At N_r = 3 n_a the size equals half the reference and nothing is clipped."""
    assert split_half_side_size(jnp.zeros(3_000), approximation_n_eff=1_000.0) == 1_500


def test_a_small_reference_is_clipped_with_a_warning() -> None:
    """Above three effective samples per approximation sample, the clip acts."""
    with pytest.warns(ValidityWarning, match="conservative") as caught:
        size = split_half_side_size(jnp.zeros(1_500), approximation_n_eff=1_000.0)
    assert size == 750
    assert "limited to 750" in str(caught[0].message)


def test_side_size_errors() -> None:
    """Sizes below one or not finite, bad shapes and tiny references raise."""
    ones = jnp.zeros(100)
    for invalid in (0.5, math.nan, math.inf):
        with pytest.raises(ValueError, match="approximation"):
            split_half_side_size(ones, approximation_n_eff=invalid)
        with pytest.raises(ValueError, match="reference"):
            split_half_side_size(
                ones, approximation_n_eff=10.0, reference_n_eff=invalid
            )
    with pytest.raises(ValueError, match="shaped"):
        split_half_side_size(jnp.zeros((4, 4)), approximation_n_eff=10.0)
    with pytest.raises(ValueError, match="reference of 4"):
        split_half_side_size(jnp.zeros(3), approximation_n_eff=10.0)


# * The plan


def make_plan(draws: int = 6) -> tuple[NullPlan, np.ndarray]:
    """Return a split-half plan of 300 zoo samples with weights, and the weights."""
    case = zoo.shifted_gaussians(0.0, param_count=2)
    reference = case.sample(jax.random.key(0), 300).samples[0, 0]
    log_weights = np.random.default_rng(1).normal(size=300) + 3.0
    plan = split_half_null(
        reference,
        log_weights,
        approximation_n_eff=20.0,
        key=jax.random.key(2),
        draws=draws,
    )
    return plan, log_weights


def test_the_pool_is_the_reference_with_normalised_weights() -> None:
    """The plan keeps the reference rows and its own weights, summing to one."""
    plan, log_weights = make_plan()
    assert plan.construction == Construction.SPLIT_HALF
    assert plan.samples.shape == (300, 2)
    weights = np.exp(np.asarray(plan.log_weights))
    assert weights.sum() == pytest.approx(1.0)
    expected = np.exp(log_weights) / np.exp(log_weights).sum()
    np.testing.assert_allclose(weights, expected)


def test_the_sides_are_disjoint_halves_of_the_reference() -> None:
    """Both sides of each draw lie in the reference and share no sample."""
    plan, log_weights = make_plan(draws=10)
    kish = float(
        np.exp(np.asarray(log_weights)).sum() ** 2
        / (np.exp(np.asarray(log_weights)) ** 2).sum()
    )
    expected = round(300 / kish * 2 / (1 / kish + 1 / 20))
    assert plan.first.shape == plan.second.shape == (10, expected)
    assert plan.recipe() == {
        "construction": "split_half",
        "draws": 10,
        "sample_size": expected,
        "tail": None,
    }
    assert plan.recipe("upper")["tail"] == "upper"
    for first, second in zip(plan.first, plan.second, strict=True):
        assert len(set(first.tolist()) | set(second.tolist())) == 2 * expected


def test_draw_i_does_not_depend_on_the_number_of_draws() -> None:
    """The plan takes its sides from random_sides with the same key."""
    reference = np.random.default_rng(3).normal(size=(200, 1))
    key = jax.random.key(4)
    few = split_half_null(reference, None, approximation_n_eff=50.0, key=key, draws=2)
    many = split_half_null(reference, None, approximation_n_eff=50.0, key=key, draws=5)
    np.testing.assert_array_equal(few.first, many.first[:2])
    sides = random_sides(key, 200, many.first.shape[1], draws=5)
    np.testing.assert_array_equal(many.second, sides[1])


def test_the_null_warns_when_the_sides_are_clipped() -> None:
    """The warning of the side size reaches the user of split_half_null."""
    reference = np.random.default_rng(5).normal(size=(100, 1))
    with pytest.warns(ValidityWarning, match="limited to 50"):
        plan = split_half_null(
            reference,
            None,
            approximation_n_eff=90.0,
            key=jax.random.key(6),
            draws=3,
        )
    assert plan.first.shape == (3, 50)


def test_inputs_are_checked() -> None:
    """Shapes, types and sizes are validated."""
    good = np.zeros((10, 2))
    key = jax.random.key(7)
    with pytest.raises(ValueError, match="shaped"):
        split_half_null(np.zeros(10), None, approximation_n_eff=5.0, key=key)
    with pytest.raises(ValueError, match="shaped"):
        split_half_null(good, np.zeros(9), approximation_n_eff=5.0, key=key)
    with pytest.raises(TypeError, match="real"):
        split_half_null(good + 1j, None, approximation_n_eff=5.0, key=key)
    with pytest.raises(ValueError, match="approximation"):
        split_half_null(good, None, approximation_n_eff=0.5, key=key)
    with pytest.raises(TypeError, match="draws"):
        split_half_null(
            good,
            None,
            approximation_n_eff=5.0,
            key=key,
            draws=2.5,  # type: ignore[arg-type]
        )


# * Calibration: an approximation of the reference's distribution is uniform


def mean_difference(
    first: jax.Array,
    first_log_weights: jax.Array,
    second: jax.Array,
    second_log_weights: jax.Array,
) -> jax.Array:
    """Return the absolute difference of the weighted means."""
    first_mean = jnp.sum(jax.nn.softmax(first_log_weights) * first[:, 0])
    second_mean = jnp.sum(jax.nn.softmax(second_log_weights) * second[:, 0])
    return jnp.abs(first_mean - second_mean)


def log_width_ratio(
    first: jax.Array,
    first_log_weights: jax.Array,
    second: jax.Array,
    second_log_weights: jax.Array,
) -> jax.Array:
    """Return the absolute log of the ratio of the weighted widths."""
    widths = []
    for samples, log_weights in (
        (first, first_log_weights),
        (second, second_log_weights),
    ):
        weights = jax.nn.softmax(log_weights)
        mean = jnp.sum(weights * samples[:, 0])
        widths.append(jnp.sqrt(jnp.sum(weights * (samples[:, 0] - mean) ** 2)))
    return jnp.abs(jnp.log(widths[1] / widths[0]))


def one_p_value(statistic: Statistic, replicate: tuple[jax.Array, ...]) -> jax.Array:
    """Return the p-value of one replicate: its observed value against its draws."""
    reference, first_side, second_side, approximation = replicate
    weights = jnp.zeros(reference.shape[0])
    draws = jax.lax.map(
        partial(
            side_statistic,
            statistic=statistic,
            reference=reference,
            weights=weights,
        ),
        (first_side, second_side),
    )
    observed = statistic(
        approximation, jnp.zeros(approximation.shape[0]), reference, weights
    )
    return empirical_p_value(observed, draws, "upper")


def side_statistic(
    positions: tuple[jax.Array, jax.Array],
    *,
    statistic: Statistic,
    reference: jax.Array,
    weights: jax.Array,
) -> jax.Array:
    """Return the statistic of two sides of the reference."""
    first, second = positions
    return statistic(
        reference[first], weights[first], reference[second], weights[second]
    )


@pytest.mark.parametrize("statistic", [mean_difference, log_width_ratio])
def test_an_approximation_of_the_reference_is_calibrated(
    statistic: Statistic,
) -> None:
    """Sides at the harmonic mean reject at 10% when the approximation is exact."""
    case = zoo.shifted_gaussians(0.0, param_count=1)
    references = case.sample(
        jax.random.key(10), REFERENCE_SAMPLES, batch_size=REPLICATES
    ).samples[:, 0]
    approximations = case.sample(
        jax.random.key(11), APPROXIMATION_SAMPLES, batch_size=REPLICATES
    ).samples[:, 1]
    size = split_half_side_size(
        jnp.zeros(REFERENCE_SAMPLES), approximation_n_eff=float(APPROXIMATION_SAMPLES)
    )
    assert size == 182
    keys = jnp.stack(
        [
            derive_key(jax.random.key(12), "replicate", index)
            for index in range(REPLICATES)
        ]
    )
    first_sides, second_sides = jax.vmap(
        partial(
            random_sides,
            pool_size=REFERENCE_SAMPLES,
            side_size=size,
            draws=PERMUTATIONS,
        )
    )(keys)
    p_values = jax.lax.map(
        partial(one_p_value, statistic),
        (references, first_sides, second_sides, approximations),
        batch_size=100,
    )
    rate = float(jnp.mean(p_values <= LEVEL))
    error = abs(rate - LEVEL) / math.sqrt(LEVEL * (1 - LEVEL) / REPLICATES)
    assert error < STANDARD_ERRORS


def test_the_public_path_reproduces_the_calibration_kernel() -> None:
    """split_half_null with null_draws gives the draws that the kernel evaluates."""
    case = zoo.shifted_gaussians(0.0, param_count=1)
    reference = case.sample(jax.random.key(13), 400).samples[0, 0]
    key = jax.random.key(14)
    plan = split_half_null(reference, None, approximation_n_eff=100.0, key=key, draws=5)
    draws = null_draws(mean_difference, plan)
    size = plan.first.shape[1]
    first_sides, second_sides = random_sides(key, 400, size, draws=5)
    weights = jnp.zeros(400)
    expected = jnp.stack(
        [
            side_statistic(
                (first, second),
                statistic=mean_difference,
                reference=reference,
                weights=weights,
            )
            for first, second in zip(first_sides, second_sides, strict=True)
        ]
    )
    np.testing.assert_allclose(draws, expected, rtol=1e-12)
