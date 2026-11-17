"""Tests for the empirical p-values and σ_sig."""

import math
import sys
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import stats

from mixed_signals import zoo
from mixed_signals.errors import ExtrapolationWarning
from mixed_signals.sigma.pareto import (
    generalized_pareto_fit,
    generalized_pareto_log_survival,
)
from mixed_signals.sigma.significance import (
    MINIMUM_EXCEEDANCES,
    SUPPORT_MARGIN,
    TAIL_FRACTION,
    Significance,
    Tail,
    empirical_p_value,
    null_significance,
    sigma_sig,
)

STANDARD_ERRORS = 5.0


def test_upper_p_value_counts_draws_at_or_above_the_value() -> None:
    """The p-value is (1 + draws ≥ value)/(1 + M) (Phipson and Smyth 2010)."""
    draws = jnp.array([1.0, 2.0, 3.0, 4.0])
    assert float(empirical_p_value(2.0, draws, "upper")) == pytest.approx(4 / 5)
    assert float(empirical_p_value(5.0, draws, Tail.UPPER)) == pytest.approx(1 / 5)
    assert float(empirical_p_value(0.0, draws, "upper")) == 1.0


def test_two_sided_p_value_counts_draws_as_far_from_their_mean() -> None:
    """Two sides compare distances from the mean of the draws."""
    draws = jnp.array([-2.0, -1.0, 1.0, 2.0])
    assert float(empirical_p_value(1.5, draws, "two_sided")) == pytest.approx(3 / 5)
    assert float(empirical_p_value(-1.5, draws, "two_sided")) == pytest.approx(3 / 5)
    assert float(empirical_p_value(3.0, draws, "two_sided")) == pytest.approx(1 / 5)


def test_p_values_broadcast_over_leading_axes() -> None:
    """Each leading index has its own value and draws."""
    draws = jnp.array([[0.0, 1.0, 2.0], [5.0, 6.0, 7.0]])
    values = jnp.array([1.5, 4.0])
    np.testing.assert_allclose(empirical_p_value(values, draws, "upper"), [0.5, 1.0])


def test_non_finite_draws_leave_the_count() -> None:
    """NaN draws are left out of M; a NaN value or no finite draw gives NaN."""
    draws = jnp.array([1.0, jnp.nan, 3.0, jnp.inf])
    assert float(empirical_p_value(2.0, draws, "upper")) == pytest.approx(2 / 3)
    assert math.isnan(float(empirical_p_value(jnp.nan, draws, "upper")))
    assert math.isnan(float(empirical_p_value(1.0, jnp.full(3, jnp.nan), "two_sided")))


def test_p_values_are_uniform_for_exchangeable_draws() -> None:
    """When the value is one more draw of the null, P(p ≤ α) = α to resolution."""
    rng = np.random.default_rng(0)
    trials = rng.normal(size=(20_000, 100))
    for tail in ("upper", "two_sided"):
        p_values = np.asarray(
            empirical_p_value(
                jnp.asarray(trials[:, 0]), jnp.asarray(trials[:, 1:]), tail
            )
        )
        for level in (0.05, 0.2, 0.5):
            error = math.sqrt(level * (1 - level) / len(p_values))
            assert abs(np.mean(p_values <= level) - level) < STANDARD_ERRORS * error


def test_sigma_sig_matches_the_normal_quantiles() -> None:
    """Upper: Φ⁻¹(1 − p); two sides: Φ⁻¹(1 − p/2)."""
    p_values = np.array([1e-12, 1e-4, 0.005, 0.05, 0.3, 0.999])
    np.testing.assert_allclose(sigma_sig(p_values, "upper"), stats.norm.isf(p_values))
    np.testing.assert_allclose(
        sigma_sig(p_values, "two_sided"), stats.norm.isf(p_values / 2)
    )
    assert float(sigma_sig(1.0, "upper")) == -math.inf
    assert float(sigma_sig(1.0, "two_sided")) == 0.0
    assert bool(jnp.all(jnp.isnan(sigma_sig(jnp.array([0.0, 1.5, jnp.nan]), "upper"))))


def test_two_sided_sigma_sig_reads_the_z_of_a_gaussian_null() -> None:
    """With many Gaussian draws, a value z from their mean gives σ_sig ≈ |z|."""
    draws = jnp.asarray(stats.norm.ppf((np.arange(100_000) + 0.5) / 100_000))
    for score in (-2.0, 1.0, 2.5):
        p_value = empirical_p_value(score, draws, "two_sided")
        assert float(sigma_sig(p_value, "two_sided")) == pytest.approx(
            abs(score), abs=1e-3
        )


def test_an_unknown_tail_raises() -> None:
    """Only the two tails of Tail are accepted."""
    with pytest.raises(ValueError, match="lower"):
        empirical_p_value(1.0, jnp.ones(3), "lower")
    with pytest.raises(ValueError, match="lower"):
        sigma_sig(0.1, "lower")


# * The tail fit beyond the resolution of the draws

DRAWS = 200
REPLICATES = 1_000


def normal_draws(replicates: int, draws: int, seed: int) -> jax.Array:
    """Return standard normal null draws of independent replicates, from the zoo."""
    case = zoo.shifted_gaussians(0.0, param_count=1)
    sets = case.sample(jax.random.key(seed), draws, batch_size=replicates)
    return sets.samples[:, 0, :, 0]


def extrapolated_significance(
    value: jax.Array | float, draws: jax.Array, tail: str
) -> Significance:
    """Return null_significance of a row that must raise the extrapolation warning."""
    with pytest.warns(ExtrapolationWarning):
        return null_significance(value, draws, tail)


def test_the_fit_settings_are_the_documented_ones() -> None:
    """The tail holds a quarter of the draws; ten exceedances end the fit."""
    assert TAIL_FRACTION == 0.25
    assert MINIMUM_EXCEEDANCES == 10
    assert SUPPORT_MARGIN == 1.05


def test_enough_exceedances_keep_the_empirical_p_value() -> None:
    """With at least ten draws at or above the value, p is the empirical one."""
    draws = normal_draws(1, DRAWS, seed=0)[0]
    for value in (-1.0, 0.5, 1.2):
        assert int(jnp.sum(draws >= value)) >= MINIMUM_EXCEEDANCES
        result = null_significance(value, draws, "upper")
        assert float(result.p_value) == float(empirical_p_value(value, draws, "upper"))
        assert int(result.exceedances) == int(jnp.sum(draws >= value))
        assert not bool(result.extrapolated)
        assert float(result.sigma_sig) == float(sigma_sig(result.p_value, Tail.UPPER))


def test_without_the_fit_the_p_value_is_empirical() -> None:
    """tail_fit=False returns the empirical p-value beyond the draws too."""
    draws = normal_draws(1, DRAWS, seed=1)[0]
    for tail in ("upper", "two_sided"):
        result = null_significance(6.0, draws, tail, tail_fit=False)
        assert float(result.p_value) == pytest.approx(1 / (DRAWS + 1))
        assert float(result.p_value) == float(empirical_p_value(6.0, draws, tail))
        assert int(result.exceedances) == 0
        assert not bool(result.extrapolated)


@pytest.mark.parametrize("tail", ["upper", "two_sided"])
def test_the_tail_fit_follows_its_construction(tail: str) -> None:
    """The p-value is (n_tail/M) S(d - t) of the fit with the support bound."""
    draws = np.asarray(normal_draws(1, DRAWS, seed=2)[0])
    value = 3.6
    centre = 0.0 if tail == "upper" else draws.mean()
    distances = draws if tail == "upper" else np.abs(draws - centre)
    distance = value if tail == "upper" else abs(value - centre)
    ranked = np.sort(distances)[::-1]
    tail_size = round(TAIL_FRACTION * DRAWS)
    threshold = 0.5 * (ranked[tail_size - 1] + ranked[tail_size])
    # * The array keeps its length M, so the grid of the fit is that of M draws.
    fit = generalized_pareto_fit(
        ranked - threshold,
        np.arange(DRAWS) < tail_size,
        bound=SUPPORT_MARGIN * (distance - threshold),
    )
    expected = (
        tail_size
        / DRAWS
        * math.exp(
            float(
                generalized_pareto_log_survival(
                    distance - threshold, fit.shape, fit.scale
                )
            )
        )
    )
    with pytest.warns(ExtrapolationWarning):
        result = null_significance(value, draws, tail)
    assert bool(result.extrapolated)
    assert float(result.p_value) == pytest.approx(expected, rel=1e-10)
    assert float(result.sigma_sig) == pytest.approx(
        float(sigma_sig(expected, tail)), rel=1e-10
    )
    # * The support constraint puts the end point of a bounded fit above the value.
    assert float(fit.shape) >= 0 or -float(fit.scale) / float(fit.shape) > (
        distance - threshold
    )


@pytest.mark.parametrize(
    ("true_sigma", "low", "high"), [(3.0, 2.6, 3.2), (4.0, 3.1, 3.9)]
)
def test_the_median_sigma_sig_of_a_normal_null_is_close_to_the_truth(
    true_sigma: float, low: float, high: float
) -> None:
    """With 200 normal draws, the median σ_sig at a true 3σ and 4σ is near it."""
    draws = normal_draws(REPLICATES, DRAWS, seed=3)
    values = jnp.full(REPLICATES, true_sigma)
    with pytest.warns(ExtrapolationWarning):
        result = null_significance(values, draws, "upper")
    sigmas = np.asarray(result.sigma_sig)
    assert result.sigma_sig.shape == (REPLICATES,)
    assert not np.any(np.isinf(sigmas))
    assert not np.any(np.isnan(sigmas))
    assert low <= np.median(sigmas) <= high
    assert np.mean(np.asarray(result.extrapolated)) > 0.95


@pytest.mark.parametrize("value", [-4.0, 4.0])
def test_two_sided_tails_fit_the_distance_from_the_null_mean(value: float) -> None:
    """A value 4σ from a normal null on either side gives a σ_sig near 4."""
    draws = normal_draws(REPLICATES, DRAWS, seed=4)
    with pytest.warns(ExtrapolationWarning):
        result = null_significance(jnp.full(REPLICATES, value), draws, "two_sided")
    sigmas = np.asarray(result.sigma_sig)
    assert not np.any(np.isinf(sigmas))
    assert 3.1 <= np.median(sigmas) <= 3.9


def test_leading_axes_match_entry_wise_calls() -> None:
    """Values shaped (2, 3) with draws shaped (2, 3, M) match entry-wise calls."""
    draws = normal_draws(6, DRAWS, seed=5).reshape(2, 3, DRAWS)
    values = jnp.array([[0.3, 3.9, 5.0], [4.4, 1.0, -4.5]])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ExtrapolationWarning)
        result = null_significance(values, draws, "two_sided")
        singles = [
            [
                null_significance(values[row, column], draws[row, column], "two_sided")
                for column in range(3)
            ]
            for row in range(2)
        ]
    assert isinstance(result, Significance)
    for field in result:
        assert field.shape == (2, 3)
    for row in range(2):
        for column in range(3):
            single = singles[row][column]
            assert float(single.p_value) == float(result.p_value[row, column])
            assert int(single.exceedances) == int(result.exceedances[row, column])
            assert bool(single.extrapolated) == bool(result.extrapolated[row, column])
    assert bool(result.extrapolated[0, 2])
    assert not bool(result.extrapolated[0, 0])


def test_non_finite_values_and_draws() -> None:
    """A NaN value or no finite draw gives NaN; NaN draws leave the count."""
    draws = normal_draws(1, DRAWS, seed=7)[0]
    nan_value = null_significance(jnp.nan, draws, "upper")
    assert math.isnan(float(nan_value.p_value))
    assert math.isnan(float(nan_value.sigma_sig))
    assert int(nan_value.exceedances) == 0
    assert not bool(nan_value.extrapolated)
    infinite = null_significance(jnp.inf, draws, "upper")
    assert math.isnan(float(infinite.p_value))
    nothing = null_significance(1.0, jnp.full(20, jnp.nan), "two_sided")
    assert math.isnan(float(nothing.p_value))
    assert not bool(nothing.extrapolated)
    empty = null_significance(1.0, jnp.zeros(0), "upper")
    assert math.isnan(float(empty.p_value))
    with_nan = jnp.concatenate([draws, jnp.full(20, jnp.nan)])
    cleaned = null_significance(0.2, with_nan, "upper")
    assert float(cleaned.p_value) == float(
        null_significance(0.2, draws, "upper").p_value
    )
    with pytest.warns(ExtrapolationWarning, match="200 draws"):
        fitted = null_significance(4.2, with_nan, "upper")
    reference = extrapolated_significance(4.2, draws, "upper")
    assert bool(fitted.extrapolated)
    assert float(fitted.p_value) == pytest.approx(float(reference.p_value), rel=1e-10)


def test_a_degenerate_tail_keeps_the_empirical_p_value() -> None:
    """Tied draws give a zero first quartile of excesses, so the fit is skipped."""
    result = null_significance(1.0, jnp.zeros(100), "upper")
    assert float(result.p_value) == pytest.approx(1 / 101)
    assert not bool(result.extrapolated)
    assert math.isfinite(float(result.sigma_sig))


def test_a_tail_of_fewer_than_ten_draws_keeps_the_empirical_p_value() -> None:
    """With 30 draws the tail holds 8 draws, too few for a fit."""
    draws = normal_draws(1, 30, seed=8)[0]
    result = null_significance(6.0, draws, "upper")
    assert float(result.p_value) == pytest.approx(1 / 31)
    assert not bool(result.extrapolated)


def test_a_very_large_value_gives_a_finite_sigma_sig() -> None:
    """The fitted p-value never falls below the smallest normal float."""
    draws = normal_draws(1, DRAWS, seed=9)[0]
    large = extrapolated_significance(1e6, draws, "upper")
    huge = extrapolated_significance(1e100, draws, "upper")
    assert 0 < float(large.p_value) < 1e-40
    assert 10 < float(large.sigma_sig) < 20
    assert float(huge.p_value) == sys.float_info.min
    assert float(huge.sigma_sig) == pytest.approx(37.52, abs=0.01)


def test_tail_must_be_known() -> None:
    """Only the two tails of Tail are accepted."""
    with pytest.raises(ValueError, match="lower"):
        null_significance(1.0, jnp.ones(3), "lower")


# * The extrapolation warning


def test_the_warning_names_the_entries_and_the_resolution() -> None:
    """One warning counts the extrapolated entries and gives Φ⁻¹(1 − 1/(M + 1))."""
    draws = normal_draws(3, DRAWS, seed=10)
    values = jnp.array([6.0, 0.0, 5.0])
    expected = float(stats.norm.isf(1 / (DRAWS + 1)))
    with pytest.warns(ExtrapolationWarning) as caught:
        null_significance(values, draws, "upper")
    assert len(caught) == 1
    message = str(caught[0].message)
    assert message.startswith("2 of 3 significances")
    assert f"σ_sig up to {expected:.2f}" in message
    assert "200 draws alone" in message
    assert "fewer than 10 of the draws" in message


def test_a_value_inside_the_resolution_of_the_draws_warns_for_the_same_reason() -> None:
    """Five draws reach the value, which lies in their range, yet the fit is used."""
    draws = normal_draws(1, DRAWS, seed=14)[0]
    value = float(jnp.sort(draws)[-5])
    with pytest.warns(ExtrapolationWarning) as caught:
        result = null_significance(value, draws, "upper")
    assert int(result.exceedances) == 5
    assert bool(result.extrapolated)
    message = str(caught[0].message)
    assert "fewer than 10 of the draws reach their values" in message
    assert "beyond" not in message


def test_two_sided_rows_report_the_two_sided_resolution() -> None:
    """For two sides the smallest p-value 1/(M + 1) resolves Φ⁻¹(1 − 1/(2(M + 1)))."""
    draws = normal_draws(1, DRAWS, seed=11)[0]
    expected = float(stats.norm.isf(1 / (2 * (DRAWS + 1))))
    with pytest.warns(ExtrapolationWarning) as caught:
        null_significance(6.0, draws, "two_sided")
    assert f"σ_sig up to {expected:.2f}" in str(caught[0].message)


def test_no_warning_without_extrapolation() -> None:
    """Warnings fail the test, so an empirical p-value must raise none."""
    draws = normal_draws(1, DRAWS, seed=12)[0]
    null_significance(1.0, draws, "upper")
    null_significance(6.0, draws, "upper", tail_fit=False)


def test_the_kernel_runs_under_jit_and_vmap_without_warning() -> None:
    """Traced inputs give the same result as concrete ones and raise no warning."""
    draws = normal_draws(4, DRAWS, seed=13)
    values = jnp.array([4.0, 3.0, 0.5, 5.5])

    def significance(value: jax.Array, null: jax.Array) -> jax.Array:
        return jnp.stack(
            [
                *(
                    field.astype(jnp.float64)
                    for field in null_significance(value, null, "upper")
                )
            ]
        )

    jitted = jax.jit(significance)(values[0], draws[0])
    mapped = jax.vmap(significance)(values, draws)
    with pytest.warns(ExtrapolationWarning):
        direct = null_significance(values, draws, "upper")
    for index, field in enumerate(direct):
        np.testing.assert_allclose(mapped[:, index], field, rtol=1e-12)
    np.testing.assert_allclose(jitted[0], direct.p_value[0], rtol=1e-12)
