"""Tests for the moment metrics: mean shift, width ratio and Gaussian divergences."""

import math
import warnings
from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mixed_signals import zoo
from mixed_signals.core.samples import SampleSet
from mixed_signals.core.space import ParamSpace
from mixed_signals.errors import ValidityWarning
from mixed_signals.estimators import GaussianDensity
from mixed_signals.metrics.moments import (
    Terms,
    gaussian_hellinger,
    gaussian_kl,
    gaussian_w2,
    mean_shift,
    width_ratio,
)
from mixed_signals.transforms import Whitening

STANDARD_ERRORS = 5.0
SAMPLE_COUNT = 40_000
BATCH_COUNT = 20
TWO_PI = 2 * math.pi


@pytest.fixture(scope="module")
def key() -> jax.Array:
    """A fixed JAX key for the draws of every case."""
    return jax.random.key(808)


def circle_space(*extra: str, period: float = TWO_PI) -> ParamSpace:
    """Return a space with a circle x0 of the given period and real axes after it."""
    return ParamSpace.model_validate(
        {
            "parameters": [
                {"name": "x0", "topology": "circle", "bounds": [0.0, period]},
                *({"name": name} for name in extra),
            ]
        }
    )


def batch_spread(
    statistic: Callable[[slice], np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Return a statistic of all draws and its standard error from disjoint batches."""
    size = SAMPLE_COUNT // BATCH_COUNT
    values = np.array(
        [
            statistic(slice(start, start + size))
            for start in range(0, SAMPLE_COUNT, size)
        ]
    )
    return statistic(slice(None)), values.std(axis=0, ddof=1) / math.sqrt(BATCH_COUNT)


def zscored(
    sample_set: SampleSet, space: ParamSpace, posteriors: list[int]
) -> tuple[np.ndarray, ParamSpace]:
    """Return problem 0 z-scored by the given posteriors, and the z-scored space."""
    whitening = Whitening.from_samples(
        sample_set.samples[0, np.asarray(posteriors)],
        sample_set.log_weights[0, np.asarray(posteriors)],
        space,
        decorrelate=False,
    )
    return np.asarray(whitening.forward(sample_set.samples[0])), whitening.map_space(
        space
    )


# * Gaussian divergences


def test_divergences_take_the_short_arc_on_circles() -> None:
    """Means on either side of the cut differ by their short arc, not by a period."""
    space = circle_space("x1")
    covariance = jnp.array([[0.04, 0.01], [0.01, 1.0]])
    near, far = jnp.array([0.1, 0.0]), jnp.array([TWO_PI - 0.1, 0.5])
    unwrapped = jnp.array([-0.1, 0.5])
    for function in (gaussian_kl, gaussian_w2, gaussian_hellinger):
        on_circle = function(near, covariance, far, 2 * covariance, space=space)
        on_line = function(near, covariance, unwrapped, 2 * covariance)
        np.testing.assert_allclose(on_circle, on_line, rtol=1e-12)


def test_log_determinant_ratio_points_to_the_wider_posterior() -> None:
    """ln(det Σ_q/det Σ_p) is shared by all three and changes sign with the order."""
    narrow, wide = jnp.diag(jnp.array([1.0, 0.5])), jnp.diag(jnp.array([1.0, 2.0]))
    for function in (gaussian_kl, gaussian_w2, gaussian_hellinger):
        forward = function(jnp.zeros(2), narrow, jnp.zeros(2), wide)
        backward = function(jnp.zeros(2), wide, jnp.zeros(2), narrow)
        assert isinstance(forward, Terms)
        assert float(forward.log_det_ratio) == pytest.approx(math.log(4.0), rel=1e-12)
        assert float(backward.log_det_ratio) == pytest.approx(-math.log(4.0), rel=1e-12)
        same = function(jnp.zeros(2), wide, jnp.ones(2), wide)
        assert float(same.log_det_ratio) == 0.0
        assert float(same.shape) == 0.0


def test_divergences_map_over_problems() -> None:
    """Mapping over a batch of mean pairs gives the looped values."""
    shifts = jnp.linspace(0.0, 2.0, 4)
    means = jnp.stack([shifts, jnp.zeros(4)], axis=1)
    covariance = jnp.eye(2)
    mapped = jax.vmap(gaussian_kl, in_axes=(None, None, 0, None))(
        jnp.zeros(2), covariance, means, covariance
    )
    for index, mean in enumerate(means):
        looped = gaussian_kl(jnp.zeros(2), covariance, mean, covariance)
        assert float(mapped.total[index]) == pytest.approx(float(looped.total))


@pytest.mark.parametrize(
    "build",
    [
        lambda: zoo.shifted_gaussians(1.2, param_count=3),
        lambda: zoo.scaled_gaussians(0.6, param_count=2),
        lambda: zoo.gaussians(
            np.array([[0.0, 0.0], [1.0, -0.5]]),
            np.array([np.eye(2), [[2.0, 0.6], [0.6, 1.0]]]),
        ),
    ],
)
def test_gaussian_fits_recover_the_zoo_divergences(
    build: Callable[[], zoo.ZooCase], key: jax.Array
) -> None:
    """KL in both directions, W2 and Hellinger of two Gaussian fits match the zoo."""
    case = build()
    draws = np.asarray(case.sample(key, SAMPLE_COUNT).samples[0])
    names = tuple(case.names)

    def terms(part: slice) -> np.ndarray:
        fits = [
            GaussianDensity.fit(draws[index, part], None, case.space)
            for index in (0, 1)
        ]
        values = []
        for function in (gaussian_kl, gaussian_w2, gaussian_hellinger):
            for first, second in ((0, 1), (1, 0)):
                result = function(
                    fits[first].mean,
                    fits[first].covariance,
                    fits[second].mean,
                    fits[second].covariance,
                    space=case.space,
                )
                values.extend([result.total, result.location, result.shape])
        return np.asarray(values, dtype=np.float64)

    estimate, error = batch_spread(terms)
    expected: list[float] = []
    for metric in ("kl", "w2", "hellinger"):
        for pair in (names, names[::-1]):
            # * W2 and Hellinger are symmetric, so the zoo keeps their case order.
            order = pair if metric == "kl" else names
            expected.extend(
                case.reference_value(name, order)
                for name in (metric, f"{metric}_location", f"{metric}_shape")
            )
    tolerance = STANDARD_ERRORS * error + 1e-12
    assert np.all(np.abs(estimate - np.asarray(expected)) <= tolerance)


# * Mean shift and width ratio


def assert_moments_recovered(
    draws: np.ndarray,
    space: ParamSpace,
    expected_shift: list[float],
    expected_ratio: list[float],
) -> None:
    """Check the shift and the ratio of two z-scored posteriors within 5 errors."""

    def moments(part: slice) -> np.ndarray:
        return np.stack(
            [
                np.asarray(function(draws[0][part], None, draws[1][part], None, space))
                for function in (mean_shift, width_ratio)
            ]
        )

    estimate, error = batch_spread(moments)
    expected = np.stack([np.asarray(expected_shift), np.asarray(expected_ratio)])
    assert np.all(np.abs(estimate - expected) <= STANDARD_ERRORS * error + 1e-12)


def test_mean_shift_and_width_ratio_recover_the_zoo_in_reference_units(
    key: jax.Array,
) -> None:
    """Z-scored by the reference, the shift is in its σ and the ratio is σ_b/σ_a."""
    case = zoo.gaussians(
        np.array([[0.0, 1.0, 5.0], [0.8, 1.0, 4.0]]),
        np.array([np.diag([1.0, 4.0, 0.25]), np.diag([2.25, 1.0, 0.25])]),
        names=["reference", "approximation"],
    )
    draws, space = zscored(case.sample(key, SAMPLE_COUNT), case.space, [0])
    pair = ("reference", "approximation")
    assert_moments_recovered(
        draws,
        space,
        [case.reference_value("mean_shift", pair, name) for name in space.names],
        [case.reference_value("width_ratio", pair, name) for name in space.names],
    )


def test_pooled_units_are_the_root_mean_square_width(key: jax.Array) -> None:
    """By the pair, the shift is in units of √((σ_a² + σ_b²)/2), Cohen's d."""
    case = zoo.gaussians(np.array([[0.0], [1.5]]), np.array([[[1.0]], [[4.0]]]))
    draws, space = zscored(case.sample(key, SAMPLE_COUNT), case.space, [0, 1])
    assert_moments_recovered(draws, space, [1.5 / math.sqrt(2.5)], [2.0])


def test_wrapped_shift_is_the_arc_in_circular_widths(key: jax.Array) -> None:
    """On a circle the shift is the short arc over the circular standard deviation."""
    case = zoo.wrapped_gaussians(-2.5, width=0.4, param_count=2, correlation=0.4)
    draws, space = zscored(case.sample(key, SAMPLE_COUNT), case.space, [0])
    pair = ("reference", "shifted")
    assert_moments_recovered(
        draws, space, [case.reference_value("mean_shift", pair, "x0"), 0.0], [1.0, 1.0]
    )


def test_circle_values_do_not_depend_on_the_cut() -> None:
    """Turning both samples around the circle changes neither shift nor ratio."""
    rng = np.random.default_rng(3)
    space = circle_space("x1")
    first = np.stack([rng.vonmises(0.0, 20.0, 500) % TWO_PI, rng.normal(size=500)], 1)
    second = np.stack([rng.vonmises(0.4, 5.0, 400) % TWO_PI, rng.normal(size=400)], 1)
    turned = [samples.copy() for samples in (first, second)]
    for samples in turned:
        samples[:, 0] = (samples[:, 0] + 3.0) % TWO_PI
    for function in (mean_shift, width_ratio):
        np.testing.assert_allclose(
            function(first, None, second, None, space),
            function(turned[0], None, turned[1], None, space),
            rtol=1e-10,
            atol=1e-12,
        )


def test_weights_follow_the_package_widths() -> None:
    """Weighted values equal NumPy's weighted mean and aweights covariance."""
    rng = np.random.default_rng(4)
    space = ParamSpace.model_validate({"parameters": [{"name": "x0"}, {"name": "x1"}]})
    first, second = rng.normal(size=(300, 2)), rng.normal(0.5, 2.0, size=(200, 2))
    weights_first, weights_second = rng.uniform(0.1, 1, 300), rng.uniform(0.1, 1, 200)
    shift = mean_shift(
        first, np.log(weights_first), second, np.log(weights_second), space
    )
    ratio = width_ratio(
        first, np.log(weights_first), second, np.log(weights_second), space
    )
    expected_shift = np.average(second, axis=0, weights=weights_second) - np.average(
        first, axis=0, weights=weights_first
    )
    widths = [
        np.sqrt(np.diag(np.cov(samples.T, aweights=weights)))
        for samples, weights in ((first, weights_first), (second, weights_second))
    ]
    np.testing.assert_allclose(shift, expected_shift, rtol=1e-12)
    np.testing.assert_allclose(ratio, widths[1] / widths[0], rtol=1e-12)
    removed = mean_shift(
        np.concatenate([first, first[:5] + 100.0]),
        np.concatenate([np.log(weights_first), np.full(5, -np.inf)]),
        second,
        np.log(weights_second),
        space,
    )
    np.testing.assert_allclose(removed, shift, rtol=1e-12)


def test_width_ratio_without_spread_is_nan_with_a_warning() -> None:
    """A first posterior without spread on an axis has no ratio there."""
    space = ParamSpace.model_validate({"parameters": [{"name": "x0"}, {"name": "x1"}]})
    first = np.stack([np.full(50, 2.0), np.linspace(0, 1, 50)], axis=1)
    second = np.stack([np.linspace(0, 1, 50), np.linspace(0, 2, 50)], axis=1)
    with pytest.warns(ValidityWarning, match="width ratio of 'x0' is NaN"):
        ratio = np.asarray(width_ratio(first, None, second, None, space))
    assert math.isnan(ratio[0])
    assert ratio[1] == pytest.approx(2.0, rel=1e-12)


def test_unresolved_circle_warns_for_the_mean_shift() -> None:
    """A circle spread evenly has no mean direction, so its shift is arbitrary."""
    space = circle_space()
    spread = np.linspace(0.0, TWO_PI, 60, endpoint=False)[:, None]
    narrow = np.full((40, 1), 1.0) + np.linspace(-0.01, 0.01, 40)[:, None]
    with pytest.warns(ValidityWarning, match="mean shift, are arbitrary"):
        mean_shift(spread, None, narrow, None, space)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        mean_shift(narrow, None, narrow + 0.1, None, space)


def test_moments_map_over_problems_without_warnings() -> None:
    """Inside vmap the values are traced: no warning, and the looped values."""
    rng = np.random.default_rng(5)
    space = ParamSpace.model_validate({"parameters": [{"name": "x0"}]})
    first, second = rng.normal(size=(3, 100, 1)), rng.normal(1.0, 2.0, size=(3, 80, 1))
    for function in (mean_shift, width_ratio):
        mapped = jax.vmap(
            lambda one, two, function=function: function(one, None, two, None, space)
        )(jnp.asarray(first), jnp.asarray(second))
        for index in range(3):
            looped = function(first[index], None, second[index], None, space)
            np.testing.assert_allclose(mapped[index], looped, rtol=1e-12)
