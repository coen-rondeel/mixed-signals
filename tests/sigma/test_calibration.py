"""Tests for the run-time calibration curve of the kNN KL σ_eff map."""

import functools
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mixed_signals import zoo
from mixed_signals.core.space import ParamSpace
from mixed_signals.errors import ValidityWarning
from mixed_signals.metrics.divergence import KL_NEIGHBOURS, knn_kl_divergence
from mixed_signals.nulls import null_draws, permutation_null
from mixed_signals.prng import derive_key
from mixed_signals.sigma.calibration import (
    KNN_KL_REPLICATES,
    KNN_KL_SHIFTS,
    CalibrationCurve,
    knn_kl_curve,
)
from mixed_signals.sigma.maps import null_subtract, sigma_eff

KEY = jax.random.key(11)
STANDARD_ERRORS = 3.0
SAMPLE_COUNT = 1_000
PARAM_COUNT = 4
REPLICATES = 6
DRAWS = 20


@functools.cache
def default_curve() -> CalibrationCurve:
    """Return the curve of 4 parameters and 1 000 samples per side, with defaults."""
    return knn_kl_curve(PARAM_COUNT, SAMPLE_COUNT, SAMPLE_COUNT, key=KEY)


def test_defaults_are_the_documented_ones() -> None:
    """Six shifts from a quarter of σ to 4σ, and four replicates."""
    assert KNN_KL_SHIFTS == (0.25, 0.5, 1.0, 2.0, 3.0, 4.0)
    assert KNN_KL_REPLICATES == 4


def test_curve_starts_at_zero_and_does_not_decrease() -> None:
    """The grid leads with δ = 0 and value 0, and the values are non-decreasing."""
    curve = default_curve()
    assert curve.shifts.shape == curve.values.shape == (len(KNN_KL_SHIFTS) + 1,)
    assert curve.shifts.dtype == curve.values.dtype == jnp.float64
    np.testing.assert_array_equal(curve.shifts, [0.0, *KNN_KL_SHIFTS])
    assert float(curve.values[0]) == 0.0
    assert bool(jnp.all(jnp.diff(curve.values) >= 0))
    assert bool(jnp.all(jnp.isfinite(curve.values)))
    # * The estimate rises as the shift squared over two, below it at D = 4.
    assert 0.5 * 4.0**2 > float(curve.values[-1]) > 0.5 * 1.0**2


def test_noise_at_small_shifts_cannot_make_the_curve_decrease() -> None:
    """With few samples the estimate is noise; the running maximum keeps it monotone."""
    curve = knn_kl_curve(
        2, 12, 12, key=KEY, shifts=(0.01, 0.02, 0.03, 0.04), replicates=2
    )
    assert float(curve.values[0]) == 0.0
    assert bool(jnp.all(jnp.diff(curve.values) >= 0))
    assert bool(jnp.all(curve.values >= 0))


def test_curve_follows_its_documented_construction() -> None:
    """Common deviates per replicate, the δ = 0 mean subtracted, a running maximum."""
    dimension, first_count, second_count, replicates = 3, 90, 70, 2
    shifts = (0.5, 2.0)
    curve = knn_kl_curve(
        dimension,
        first_count,
        second_count,
        key=KEY,
        shifts=shifts,
        replicates=replicates,
        neighbours=2,
    )
    space = ParamSpace.model_validate(
        {"parameters": [{"name": f"x{index}"} for index in range(dimension)]}
    )
    estimates = np.zeros((replicates, 3))
    for replicate in range(replicates):
        first = jax.random.normal(
            derive_key(KEY, "first", replicate), (first_count, dimension)
        )
        base = jax.random.normal(
            derive_key(KEY, "second", replicate), (second_count, dimension)
        )
        for position, shift in enumerate((0.0, *shifts)):
            moved = base.at[:, 0].add(shift)
            estimates[replicate, position] = float(
                knn_kl_divergence(
                    first,
                    None,
                    moved,
                    None,
                    space,
                    key=derive_key(KEY, "subsample", replicate),
                    neighbours=2,
                )
            )
    means = estimates.mean(axis=0)
    expected = np.maximum.accumulate(means - means[0])
    np.testing.assert_allclose(curve.values, expected, rtol=1e-12, atol=1e-14)
    np.testing.assert_array_equal(curve.shifts, [0.0, *shifts])


def test_curve_depends_only_on_its_inputs() -> None:
    """The same inputs give the same curve, whatever ran before; another key differs."""
    first = knn_kl_curve(2, 60, 60, key=KEY, replicates=2)
    knn_kl_curve(3, 40, 50, key=jax.random.key(99), replicates=2)
    again = knn_kl_curve(2, 60, 60, key=KEY, replicates=2)
    np.testing.assert_array_equal(first.values, again.values)
    other = knn_kl_curve(2, 60, 60, key=jax.random.key(12), replicates=2)
    assert not np.array_equal(first.values, other.values)
    more = knn_kl_curve(2, 60, 60, key=KEY, replicates=3)
    assert not np.array_equal(first.values, more.values)


def test_unequal_counts_and_other_ranks_are_used() -> None:
    """The counts and the rank k are those of the row: they change the curve."""
    base = knn_kl_curve(2, 80, 80, key=KEY, replicates=2)
    unequal = knn_kl_curve(2, 80, 160, key=KEY, replicates=2)
    other_rank = knn_kl_curve(2, 80, 80, key=KEY, replicates=2, neighbours=5)
    assert not np.array_equal(base.values, unequal.values)
    assert not np.array_equal(base.values, other_rank.values)
    assert KL_NEIGHBOURS == 3


def test_too_few_samples_give_nan_with_the_estimators_warning() -> None:
    """With no more samples than neighbours the estimator, and the curve, are NaN."""
    with pytest.warns(ValidityWarning, match="need more than k samples"):
        curve = knn_kl_curve(2, 3, 3, key=KEY, replicates=1)
    assert bool(jnp.all(jnp.isnan(curve.values[1:])))
    assert math.isnan(float(sigma_eff("kl", 0.1, estimator="knn", curve=curve)))


def call_with(arguments: dict[str, object]) -> CalibrationCurve:
    """Call ``knn_kl_curve`` with small valid settings, some of them replaced."""
    settings: dict[str, object] = {
        "dimension": 2,
        "first_count": 30,
        "second_count": 30,
        "replicates": 1,
        "shifts": (0.5, 1.0),
    }
    settings.update(arguments)
    return knn_kl_curve(
        settings["dimension"],  # type: ignore[arg-type]
        settings["first_count"],  # type: ignore[arg-type]
        settings["second_count"],  # type: ignore[arg-type]
        key=KEY,
        replicates=settings["replicates"],  # type: ignore[arg-type]
        shifts=settings["shifts"],  # type: ignore[arg-type]
    )


@pytest.mark.parametrize(
    "arguments",
    [
        {"dimension": 0},
        {"first_count": 0},
        {"second_count": -1},
        {"replicates": 0},
    ],
)
def test_counts_below_one_are_rejected(arguments: dict[str, object]) -> None:
    """The dimension, the counts and the replicates are at least one."""
    with pytest.raises(ValueError, match="must be at least 1"):
        call_with(arguments)


@pytest.mark.parametrize(
    "arguments",
    [
        {"dimension": 2.0},
        {"dimension": "2"},
        {"first_count": 30.0},
        {"second_count": 2.5},
        {"replicates": 1.0},
    ],
)
def test_counts_that_are_not_integers_are_rejected(
    arguments: dict[str, object],
) -> None:
    """A float or a string is not a count; the type check of the test hook may speak."""
    with pytest.raises(TypeError):
        call_with(arguments)


@pytest.mark.parametrize(
    "name", ["dimension", "first_count", "second_count", "replicates"]
)
def test_a_bool_is_not_a_count(name: str) -> None:
    """True is an int to Python, so the package's own check names the argument."""
    with pytest.raises(TypeError, match=f"{name} must be an integer, got bool"):
        call_with({name: True})


@pytest.mark.parametrize(
    "shifts",
    [
        (),
        (0.0, 1.0),
        (-0.5, 1.0),
        (1.0, 0.5),
        (0.5, 0.5),
        (0.5, math.nan),
        (0.5, math.inf),
    ],
)
def test_shifts_must_be_positive_finite_and_increasing(
    shifts: tuple[float, ...],
) -> None:
    """Shifts that cannot be inverted, or are not shifts, are rejected."""
    with pytest.raises(
        ValueError, match="shifts must be positive, finite and strictly"
    ):
        call_with({"shifts": shifts})


def mean_and_standard_error(effects: list[float]) -> tuple[float, float]:
    """Return the mean of the replicates and its standard error."""
    return float(np.mean(effects)), float(np.std(effects, ddof=1)) / math.sqrt(
        len(effects)
    )


@pytest.mark.parametrize("shift", [1.0, 2.0])
def test_calibrated_sigma_eff_recovers_the_shift_where_the_plain_root_reads_low(
    shift: float,
) -> None:
    """At D = 4 and N = 1 000, a shift of 1σ or 2σ comes back within 3 SE plus 5%."""
    curve = default_curve()
    case = zoo.shifted_gaussians(shift, param_count=PARAM_COUNT)
    sets = case.sample(
        derive_key(KEY, "sets", int(shift)), SAMPLE_COUNT, batch_size=REPLICATES
    ).samples

    def statistic(
        first: jax.Array,
        first_log_weights: jax.Array,
        second: jax.Array,
        second_log_weights: jax.Array,
    ) -> jax.Array:
        return knn_kl_divergence(
            first, first_log_weights, second, second_log_weights, case.space, key=KEY
        )

    calibrated, plain = [], []
    for replicate in range(REPLICATES):
        first, second = sets[replicate]
        plan = permutation_null(
            first,
            None,
            second,
            None,
            key=derive_key(KEY, "null", int(shift), replicate),
            draws=DRAWS,
        )
        zeros = (jnp.zeros(first.shape[0]), jnp.zeros(second.shape[0]))
        value = statistic(first, zeros[0], second, zeros[1])
        subtracted = null_subtract("kl", value, null_draws(statistic, plan))
        calibrated.append(
            float(sigma_eff("kl", subtracted, estimator="knn", curve=curve))
        )
        plain.append(float(sigma_eff("kl", subtracted)))
    mean, standard_error = mean_and_standard_error(calibrated)
    assert abs(mean - shift) <= STANDARD_ERRORS * standard_error + 0.05 * shift
    plain_mean, plain_error = mean_and_standard_error(plain)
    assert plain_mean < mean
    assert abs(plain_mean - shift) > abs(mean - shift)
    if shift == 2.0:
        assert plain_mean < shift - STANDARD_ERRORS * plain_error
