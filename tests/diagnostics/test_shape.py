"""Tests for the Gaussianity statistics of one posterior."""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import integrate, special, stats

from mixed_signals import zoo
from mixed_signals.core.space import ParamSpace
from mixed_signals.diagnostics.shape import (
    anderson_darling_normality,
    mardia_moments,
    marginal_moments,
    moment_metric_names,
)
from mixed_signals.errors import SubsamplingWarning, ValidityWarning

TWO_PI = 2 * math.pi
CIRCLE = ParamSpace.model_validate(
    {"parameters": [{"name": "phase", "topology": "circle", "bounds": [0.0, TWO_PI]}]}
)
LINE = ParamSpace.model_validate({"parameters": [{"name": "x0"}]})


def banana_samples(count: int, seed: int) -> tuple[np.ndarray, ParamSpace]:
    """Return samples of the zoo's banana with b = 0.01 and its space."""
    case = zoo.banana(0.01)
    return np.asarray(
        case.sample(jax.random.key(seed), count).samples[0, 0]
    ), case.space


def reference_anderson_darling(values: np.ndarray) -> float:
    """Return A²/n against the normal of the sample mean and SD, by the textbook sum."""
    count = len(values)
    scores = np.sort((values - values.mean()) / values.std(ddof=1))
    index = np.arange(1, count + 1)
    # * Both tails from ln Φ, since ln(1 − Φ) of the far upper tail loses digits.
    logs = special.log_ndtr(scores) + special.log_ndtr(-scores[::-1])
    statistic = -count - np.sum((2 * index - 1) * logs) / count
    return float(statistic / count)


def reference_circular_moments(angles: np.ndarray) -> tuple[float, float]:
    """Return b̄₂/(1 − R̄)^(3/2) and (ā₂ − R̄⁴)/(1 − R̄)², the textbook forms."""
    first = np.mean(np.exp(1j * angles))
    length, direction = abs(first), np.angle(first)
    second = np.mean(np.exp(2j * (angles - direction)))
    return (
        float(second.imag / (1 - length) ** 1.5),
        float((second.real - length**4) / (1 - length) ** 2),
    )


# * Marginal moments


def test_marginal_moments_equal_scipy_for_equal_weights() -> None:
    """g₁ and g₂ are SciPy's biased skewness and Fisher kurtosis."""
    samples, space = banana_samples(3_000, 0)
    moments = marginal_moments(samples, None, space)
    np.testing.assert_allclose(moments.skewness, stats.skew(samples), rtol=1e-12)
    np.testing.assert_allclose(moments.kurtosis, stats.kurtosis(samples), rtol=1e-12)
    assert moment_metric_names(space) == (("skewness", "kurtosis"),) * 2


def test_integer_weights_equal_repeated_samples() -> None:
    """Weights act as repetitions in the moment ratios of the weighted distribution."""
    samples, space = banana_samples(500, 1)
    counts = np.random.default_rng(2).integers(1, 4, size=500)
    weighted = marginal_moments(samples, np.log(counts), space)
    repeated = np.repeat(samples, counts, axis=0)
    np.testing.assert_allclose(weighted.skewness, stats.skew(repeated), rtol=1e-11)
    np.testing.assert_allclose(weighted.kurtosis, stats.kurtosis(repeated), rtol=1e-11)


def test_circular_moments_equal_the_textbook_forms() -> None:
    """On a circle the rows hold the circular skewness and kurtosis."""
    angles = np.mod(1.0 + 0.6 * np.random.default_rng(3).gamma(3.0, size=5_000), TWO_PI)
    moments = marginal_moments(angles[:, None], None, CIRCLE)
    skewness, kurtosis = reference_circular_moments(angles)
    assert float(moments.skewness[0]) == pytest.approx(skewness, rel=1e-10)
    assert float(moments.kurtosis[0]) == pytest.approx(kurtosis, rel=1e-10)
    assert moment_metric_names(CIRCLE) == (("circular_skewness", "circular_kurtosis"),)


def test_narrow_circles_keep_their_digits() -> None:
    """At a width of 1e-4 rad the moments tend to −2√2·g₁ and 2·g₂ of the deviations."""
    deviations = np.random.default_rng(4).gamma(4.0, size=200_000)
    deviations = (deviations - deviations.mean()) / deviations.std()
    angles = 2.5 + 1e-4 * deviations
    moments = marginal_moments(angles[:, None], None, CIRCLE)
    assert float(moments.skewness[0]) == pytest.approx(
        -2 * math.sqrt(2) * stats.skew(deviations), rel=1e-3
    )
    assert float(moments.kurtosis[0]) == pytest.approx(
        2 * stats.kurtosis(deviations), rel=1e-3
    )


def test_a_rotation_leaves_the_circular_moments() -> None:
    """No rotation of the circle changes the circular skewness and kurtosis."""
    angles = np.mod(0.4 * np.random.default_rng(5).standard_t(5, size=3_000), TWO_PI)
    original = marginal_moments(angles[:, None], None, CIRCLE)
    turned = marginal_moments(np.mod(angles + 3.0, TWO_PI)[:, None], None, CIRCLE)
    np.testing.assert_allclose(turned.skewness, original.skewness, atol=1e-10)
    np.testing.assert_allclose(turned.kurtosis, original.kurtosis, atol=1e-10)


def test_a_parameter_without_spread_gives_nan_and_warns() -> None:
    """A constant parameter has no moment ratios."""
    samples, space = banana_samples(200, 6)
    samples = samples.copy()
    samples[:, 0] = 1.5
    with pytest.warns(ValidityWarning, match="'x0', which have no spread"):
        moments = marginal_moments(samples, None, space)
    assert math.isnan(float(moments.skewness[0]))
    assert math.isfinite(float(moments.kurtosis[1]))


def test_moments_compile_and_map_over_sets() -> None:
    """Under jax.jit and jax.lax.map the moments equal concrete calls."""
    samples, space = banana_samples(400, 7)
    sets = jnp.stack([jnp.asarray(samples), jnp.asarray(samples[::-1])])
    mapped = jax.lax.map(lambda values: marginal_moments(values, None, space), sets)
    direct = marginal_moments(samples, None, space)
    np.testing.assert_allclose(mapped.skewness[1], direct.skewness, rtol=1e-12)
    compiled = jax.jit(lambda values: marginal_moments(values, None, space))(samples)
    np.testing.assert_allclose(compiled.kurtosis, direct.kurtosis, rtol=1e-12)


# * Anderson-Darling


def test_anderson_darling_equals_scipy_for_equal_weights() -> None:
    """A²/n equals the textbook sum of Anderson and Darling over n."""
    samples, space = banana_samples(2_000, 8)
    distances = np.asarray(anderson_darling_normality(samples, None, space))
    for position in range(2):
        assert distances[position] == pytest.approx(
            reference_anderson_darling(samples[:, position]), rel=1e-9
        )


def test_weighted_anderson_darling_integrates_its_definition() -> None:
    """With weights, A²/n is ∫(F_w − Φ)²/(Φ(1 − Φ)) dΦ over the weighted steps."""
    rng = np.random.default_rng(9)
    values = rng.gamma(3.0, size=60)
    log_weights = rng.normal(size=60)
    distance = float(anderson_darling_normality(values[:, None], log_weights, LINE)[0])
    weights = np.exp(log_weights - log_weights.max())
    weights /= weights.sum()
    mean = np.sum(weights * values)
    variance = np.sum(weights * (values - mean) ** 2) / (1 - np.sum(weights**2))
    order = np.argsort(values)
    edges = stats.norm.cdf((values[order] - mean) / math.sqrt(variance))
    levels = np.concatenate([[0.0], np.cumsum(weights[order])])
    bounds = np.concatenate([[0.0], edges, [1.0]])
    expected = 0.0
    for level, low, high in zip(levels, bounds[:-1], bounds[1:], strict=True):
        if high > low:
            expected += integrate.quad(
                lambda value, step=level: (step - value) ** 2 / (value * (1 - value)),
                low,
                high,
            )[0]
    assert distance == pytest.approx(expected, rel=1e-8)


def test_circle_anderson_darling_is_cut_opposite_the_mean() -> None:
    """On a circle the distance is the linear one of values unwrapped about the mean."""
    angles = np.mod(0.4 * np.random.default_rng(10).gamma(2.0, size=2_000), TWO_PI)
    turned = np.mod(angles + 2.0, TWO_PI)
    distance = float(anderson_darling_normality(angles[:, None], None, CIRCLE)[0])
    rotated = float(anderson_darling_normality(turned[:, None], None, CIRCLE)[0])
    assert rotated == pytest.approx(distance, rel=1e-9)
    unwrapped = np.angle(np.exp(1j * (angles - np.angle(np.mean(np.exp(1j * angles))))))
    assert distance == pytest.approx(reference_anderson_darling(unwrapped), rel=1e-9)


def test_wide_circles_warn_for_anderson_darling() -> None:
    """A circle wider than 1 rad gives an approximate surrogate null."""
    case = zoo.wrapped_gaussians(0.0, width=1.5, param_count=1)
    samples = case.sample(jax.random.key(11), 1_000).samples[0, 0]
    with pytest.warns(ValidityWarning, match="'x0', wider than a circular"):
        anderson_darling_normality(samples, None, case.space)
    narrow = zoo.wrapped_gaussians(0.0, width=0.5, param_count=1)
    anderson_darling_normality(
        narrow.sample(jax.random.key(12), 1_000).samples[0, 0], None, narrow.space
    )


# * Mardia's moments


def reference_mardia(samples: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    """Return b₁ and b₂ with the weighted plug-in covariance, by dense NumPy algebra."""
    weights = weights / weights.sum()
    centred = samples - weights @ samples
    covariance = (weights[:, None] * centred).T @ centred
    whitened = np.linalg.solve(np.linalg.cholesky(covariance), centred.T).T
    gram = whitened @ whitened.T
    skewness = float(weights @ (gram**3) @ weights)
    kurtosis = float(weights @ np.sum(whitened**2, axis=1) ** 2)
    return skewness, kurtosis


def test_mardia_moments_match_dense_algebra() -> None:
    """b₁ and b₂ equal a dense computation, with and without weights."""
    samples, space = banana_samples(1_500, 13)
    log_weights = np.random.default_rng(14).normal(scale=0.5, size=1_500)
    for weights in (None, log_weights):
        moments = mardia_moments(samples, weights, space, key=jax.random.key(0))
        expected = reference_mardia(
            samples, np.ones(1_500) if weights is None else np.exp(weights)
        )
        assert float(moments.skewness) == pytest.approx(expected[0], rel=1e-10)
        assert float(moments.kurtosis) == pytest.approx(expected[1], rel=1e-10)


def test_mardia_moments_of_a_gaussian_take_their_normal_values() -> None:
    """For a Gaussian, E[n·b₁/6] = D(D + 1)(D + 2)/6 and b₂ tends to D(D + 2)."""
    case = zoo.shifted_gaussians(0.0, param_count=4)
    sets = case.sample(jax.random.key(15), 2_000, batch_size=20).samples[:, 0]
    moments = [
        mardia_moments(values, None, case.space, key=jax.random.key(0))
        for values in sets
    ]
    skewness = np.array([float(moment.skewness) for moment in moments])
    kurtosis = np.array([float(moment.kurtosis) for moment in moments])
    assert np.mean(2_000 * skewness / 6) == pytest.approx(4 * 5 * 6 / 6, rel=0.15)
    assert np.mean(kurtosis) == pytest.approx(4 * 6, rel=0.01)


def test_mardia_moments_subsample_above_the_cap() -> None:
    """Above the cap a uniform subsample runs, with a SubsamplingWarning."""
    samples, space = banana_samples(500, 16)
    with pytest.warns(SubsamplingWarning, match="mardia"):
        capped = mardia_moments(samples, None, space, key=jax.random.key(1), cap=300)
    assert math.isfinite(float(capped.skewness))


def test_a_singular_covariance_gives_nan_mardia_moments() -> None:
    """Collinear parameters leave no whitening."""
    samples, space = banana_samples(300, 17)
    samples = np.c_[samples[:, 0], 2 * samples[:, 0]]
    with pytest.warns(ValidityWarning, match="Mardia's moments are NaN"):
        moments = mardia_moments(samples, None, space, key=jax.random.key(2))
    assert math.isnan(float(moments.skewness))
    assert math.isnan(float(moments.kurtosis))
