"""Tests for the maximum mean discrepancy, energy distance and median bandwidth."""

import math
from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.typing import ArrayLike
from scipy import spatial, stats

from mixed_signals import zoo
from mixed_signals._config import SAMPLE_CAP
from mixed_signals.core.space import ParamSpace
from mixed_signals.errors import SubsamplingWarning
from mixed_signals.metrics._cap import _draw, cap_pair
from mixed_signals.metrics.kernel import (
    BANDWIDTH_SAMPLE_SIZE,
    energy_distance,
    maximum_mean_discrepancy,
    median_bandwidth,
)
from mixed_signals.prng import derive_key

STANDARD_ERRORS = 5.0
TWO_PI = 2 * math.pi
KEY = jax.random.key(0)
PLANE = ParamSpace.model_validate({"parameters": [{"name": "x0"}, {"name": "x1"}]})
TRIPLE = ParamSpace.model_validate(
    {"parameters": [{"name": "x0"}, {"name": "x1"}, {"name": "x2"}]}
)
CIRCLE = ParamSpace.model_validate(
    {"parameters": [{"name": "phase", "topology": "circle", "bounds": [0, TWO_PI]}]}
)

type Metric = Callable[..., jax.Array]


def mmd(
    first: ArrayLike,
    first_log_weights: ArrayLike | None,
    second: ArrayLike,
    second_log_weights: ArrayLike | None,
    space: ParamSpace,
    *,
    key: jax.Array,
    cap: int = SAMPLE_CAP,
) -> jax.Array:
    """Return the MMD² at a fixed bandwidth of 1.3."""
    return maximum_mean_discrepancy(
        first,
        first_log_weights,
        second,
        second_log_weights,
        space,
        bandwidth=1.3,
        key=key,
        cap=cap,
    )


METRICS: tuple[Metric, ...] = (mmd, energy_distance)


@pytest.fixture
def rng() -> np.random.Generator:
    """A seeded NumPy generator for test data."""
    return np.random.default_rng(2026)


def weighted_statistics(
    first: np.ndarray,
    first_weights: np.ndarray,
    second: np.ndarray,
    second_weights: np.ndarray,
    bandwidth: float,
) -> tuple[float, float]:
    """Return the weighted V-statistic MMD² and energy distance with SciPy distances."""
    first_weights = first_weights / first_weights.sum()
    second_weights = second_weights / second_weights.sum()
    within_first = spatial.distance.cdist(first, first)
    within_second = spatial.distance.cdist(second, second)
    across = spatial.distance.cdist(first, second)

    def kernel(distances: np.ndarray) -> np.ndarray:
        return np.exp(-(distances**2) / (2 * bandwidth**2))

    squared = (
        first_weights @ kernel(within_first) @ first_weights
        + second_weights @ kernel(within_second) @ second_weights
        - 2 * first_weights @ kernel(across) @ second_weights
    )
    energy = (
        2 * first_weights @ across @ second_weights
        - first_weights @ within_first @ first_weights
        - second_weights @ within_second @ second_weights
    )
    return float(squared), float(energy)


# * The statistics


def test_weighted_v_statistics_match_scipy_distances(rng: np.random.Generator) -> None:
    """Both metrics are the V-statistics of the normalised weights over all pairs."""
    first = rng.normal(size=(500, 3))
    second = rng.normal(0.3, 1.2, size=(350, 3))
    first_weights = rng.uniform(0.1, 2.0, 500)
    second_weights = rng.uniform(0.1, 2.0, 350)
    expected_mmd, expected_energy = weighted_statistics(
        first, first_weights, second, second_weights, 1.3
    )
    arguments = (first, np.log(first_weights), second, np.log(second_weights), TRIPLE)
    assert float(mmd(*arguments, key=KEY)) == pytest.approx(expected_mmd, rel=1e-12)
    assert float(energy_distance(*arguments, key=KEY)) == pytest.approx(
        expected_energy, rel=1e-12
    )


def test_energy_in_one_dimension_is_scipys_squared(rng: np.random.Generator) -> None:
    """SciPy's energy_distance is the root of the weighted V-statistic."""
    line = ParamSpace.model_validate({"parameters": [{"name": "x0"}]})
    first, second = rng.normal(size=400), rng.normal(0.5, 0.8, size=300)
    first_weights, second_weights = rng.uniform(0.1, 1, 400), rng.uniform(0.1, 1, 300)
    value = energy_distance(
        first[:, None],
        np.log(first_weights),
        second[:, None],
        np.log(second_weights),
        line,
        key=KEY,
    )
    expected = stats.energy_distance(first, second, first_weights, second_weights)
    assert float(value) == pytest.approx(expected**2, rel=1e-12)


@pytest.mark.parametrize("metric", METRICS)
def test_weights_count_and_remove_samples(
    rng: np.random.Generator, metric: Metric
) -> None:
    """A weight of k counts like k copies, and a weight of zero removes a sample."""
    first = rng.normal(size=(60, 2))
    second = rng.normal(0.5, 1.0, size=(50, 2))
    copies = rng.integers(1, 4, size=60)
    weighted = metric(first, np.log(copies), second, None, PLANE, key=KEY)
    repeated = metric(
        np.repeat(first, copies, axis=0), None, second, None, PLANE, key=KEY
    )
    assert float(weighted) == pytest.approx(float(repeated), rel=1e-12)
    keep = np.arange(60) % 4 != 0
    with np.errstate(divide="ignore"):
        removed = metric(
            first, np.log(keep.astype(float)), second, None, PLANE, key=KEY
        )
    dropped = metric(first[keep], None, second, None, PLANE, key=KEY)
    assert float(removed) == pytest.approx(float(dropped), rel=1e-12)


@pytest.mark.parametrize("metric", METRICS)
def test_identical_inputs_give_exactly_zero(
    rng: np.random.Generator, metric: Metric
) -> None:
    """The cross term repeats the within terms, so nothing is left."""
    samples = rng.normal(size=(300, 3))
    assert float(metric(samples, None, samples, None, TRIPLE, key=KEY)) == 0.0


# * Circles and spheres


def plane_points(angles: np.ndarray) -> np.ndarray:
    """Return points on the unit circle in the plane."""
    return np.stack([np.cos(angles), np.sin(angles)], axis=1)


@pytest.mark.parametrize("metric", METRICS)
def test_circles_are_embedded_in_the_plane(
    rng: np.random.Generator, metric: Metric
) -> None:
    """A circle of period 2π acts as the unit circle in a plane, cut or turned."""
    first = rng.vonmises(0.0, 3.0, 300) % TWO_PI
    second = rng.vonmises(0.6, 2.0, 250) % TWO_PI
    on_circle = metric(first[:, None], None, second[:, None], None, CIRCLE, key=KEY)
    in_plane = metric(
        plane_points(first), None, plane_points(second), None, PLANE, key=KEY
    )
    assert float(on_circle) == pytest.approx(float(in_plane), rel=1e-12)
    turned = metric(
        ((first + 2.0) % TWO_PI)[:, None],
        None,
        ((second + 2.0) % TWO_PI)[:, None],
        None,
        CIRCLE,
        key=KEY,
    )
    assert float(turned) == pytest.approx(float(on_circle), rel=1e-10)


@pytest.mark.parametrize("metric", METRICS)
def test_spheres_are_the_unit_sphere(metric: Metric) -> None:
    """Sphere coordinates act as their unit vectors in three dimensions."""
    case = zoo.von_mises_fisher(1.0, width=0.3, polar_form="sin_latitude")
    draws = np.asarray(case.sample(jax.random.key(3), 300).samples[0])
    on_sphere = metric(draws[0], None, draws[1], None, case.space, key=KEY)

    def vectors(points: np.ndarray) -> np.ndarray:
        height = points[:, 1]
        horizontal = np.sqrt(1 - height**2)
        azimuth = points[:, 0]
        return np.stack(
            [horizontal * np.cos(azimuth), horizontal * np.sin(azimuth), height], axis=1
        )

    in_space = metric(vectors(draws[0]), None, vectors(draws[1]), None, TRIPLE, key=KEY)
    assert float(on_sphere) == pytest.approx(float(in_space), rel=1e-10)


@pytest.mark.parametrize("metric", METRICS)
def test_spheres_scale_with_their_radius(metric: Metric) -> None:
    """A sphere of radius R acts as R times its unit vectors."""
    case = zoo.von_mises_fisher(1.0, width=0.3, euclidean_count=1)
    draws = np.asarray(case.sample(jax.random.key(5), 300).samples[0])
    sphere = case.space.spheres[0].model_copy(update={"radius": 4.0})
    scaled_space = case.space.model_copy(update={"spheres": (sphere,)})
    on_sphere = metric(draws[0], None, draws[1], None, scaled_space, key=KEY)

    def embedded(points: np.ndarray) -> np.ndarray:
        azimuth, latitude = points[:, 0], points[:, 1]
        vectors = np.stack(
            [
                np.cos(latitude) * np.cos(azimuth),
                np.cos(latitude) * np.sin(azimuth),
                np.sin(latitude),
            ],
            axis=1,
        )
        return np.concatenate([points[:, 2:], 4.0 * vectors], axis=1)

    quadruple = ParamSpace.model_validate(
        {"parameters": [{"name": f"x{index}"} for index in range(4)]}
    )
    in_space = metric(
        embedded(draws[0]), None, embedded(draws[1]), None, quadruple, key=KEY
    )
    assert float(on_sphere) == pytest.approx(float(in_space), rel=1e-10)


@pytest.mark.parametrize("metric", METRICS)
def test_even_harmonics_are_resolved(metric: Metric) -> None:
    """Modes at 0 and π against ±π/2 differ only in even harmonics; chords see them."""
    rng = np.random.default_rng(5)

    def modes(centres: tuple[float, float]) -> np.ndarray:
        picks = np.asarray(centres)[rng.integers(0, 2, 300)]
        return ((picks + 0.3 * rng.normal(size=300)) % TWO_PI)[:, None]

    first, second = modes((0.0, math.pi)), modes((0.5 * math.pi, 1.5 * math.pi))
    value = float(metric(first, None, second, None, CIRCLE, key=KEY))
    pooled = np.concatenate([first, second])
    null = []
    for _ in range(20):
        order = rng.permutation(600)
        null.append(
            float(
                metric(
                    pooled[order[:300]],
                    None,
                    pooled[order[300:]],
                    None,
                    CIRCLE,
                    key=KEY,
                )
            )
        )
    assert value > np.mean(null) + 20 * np.std(null, ddof=1)


# * The bandwidth


def test_bandwidth_is_the_median_of_all_pooled_pairs(rng: np.random.Generator) -> None:
    """For equal weights and sizes below the subsample, the plain pooled median."""
    first, second = rng.normal(size=(300, 3)), rng.normal(0.4, 1.1, size=(300, 3))
    value = median_bandwidth(first, None, second, None, TRIPLE, key=KEY)
    expected = np.median(spatial.distance.pdist(np.concatenate([first, second])))
    assert float(value) == pytest.approx(expected, rel=1e-14)
    assert BANDWIDTH_SAMPLE_SIZE == 1_000


def test_bandwidth_subsamples_from_the_documented_keys(
    rng: np.random.Generator,
) -> None:
    """Above the subsample size each posterior draws from its own derived key."""
    first, second = rng.normal(size=(500, 2)), rng.normal(size=(150, 2))
    value = median_bandwidth(first, None, second, None, PLANE, key=KEY, sample_size=200)
    kept, _ = _draw(jnp.asarray(first), jnp.zeros(500), derive_key(KEY, "first"), 200)
    pooled = np.concatenate([np.asarray(kept), second])
    weights = np.concatenate([np.full(200, 1 / 400), np.full(150, 1 / 300)])
    rows, columns = np.triu_indices(350, 1)
    distances = np.linalg.norm(pooled[rows] - pooled[columns], axis=1)
    order = np.argsort(distances)
    cumulative = np.cumsum((weights[rows] * weights[columns])[order])
    half = 0.5 * cumulative[-1]
    expected = 0.5 * (
        distances[order][np.argmax(cumulative >= half)]
        + distances[order][np.argmax(cumulative > half)]
    )
    assert float(value) == pytest.approx(expected, rel=1e-12)


def test_bandwidth_follows_the_circle(rng: np.random.Generator) -> None:
    """On a circle the pairs are measured along chords, whatever the cut."""
    first = rng.vonmises(3.0, 2.0, 200) % TWO_PI
    second = rng.vonmises(-2.8, 2.0, 200) % TWO_PI
    value = median_bandwidth(
        first[:, None], None, second[:, None], None, CIRCLE, key=KEY
    )
    expected = np.median(
        spatial.distance.pdist(plane_points(np.concatenate([first, second])))
    )
    assert float(value) == pytest.approx(expected, rel=1e-12)


# * Population values and the zoo


def chi_mean(dimension: int, noncentrality: float) -> float:
    """Return E[χ_D(λ)] by SciPy's integration over the noncentral χ² law."""
    if noncentrality == 0:
        return float(stats.chi(dimension).mean())
    return float(stats.ncx2(dimension, noncentrality**2).expect(np.sqrt))


@pytest.mark.parametrize("metric_name", ["mmd", "energy"])
def test_shifted_unit_gaussians_match_the_closed_forms(metric_name: str) -> None:
    """The mean over problems equals the population value plus the V-statistic bias.

    The V-statistic keeps the diagonal pairs, so its expectation exceeds the
    population value by E k(x, x) − E k(x, x') per sample for MMD², and by
    E‖x − x'‖ per sample for the energy distance.
    """
    dimension, shift, count, bandwidth = 3, 1.0, 1_500, 2.0
    case = zoo.shifted_gaussians(shift, param_count=dimension)
    draws = np.asarray(case.sample(jax.random.key(8), count, batch_size=12).samples)
    if metric_name == "mmd":
        scale = (bandwidth**2 / (bandwidth**2 + 2)) ** (dimension / 2)
        truth = 2 * scale * (1 - math.exp(-(shift**2) / (2 * (bandwidth**2 + 2))))
        bias = 2 * (1 - scale) / count
        values = [
            float(
                maximum_mean_discrepancy(
                    problem[0],
                    None,
                    problem[1],
                    None,
                    case.space,
                    bandwidth=bandwidth,
                    key=KEY,
                )
            )
            for problem in draws
        ]
    else:
        base = chi_mean(dimension, 0.0)
        truth = 2 * math.sqrt(2) * (chi_mean(dimension, shift / math.sqrt(2)) - base)
        bias = 2 * math.sqrt(2) * base / count
        values = [
            float(
                energy_distance(problem[0], None, problem[1], None, case.space, key=KEY)
            )
            for problem in draws
        ]
    error = np.std(values, ddof=1) / math.sqrt(len(values))
    assert abs(np.mean(values) - truth - bias) < STANDARD_ERRORS * error


@pytest.mark.parametrize("metric", METRICS)
def test_larger_shifts_give_larger_values(metric: Metric) -> None:
    """The zoo's shifted Gaussians come out in order."""
    values = []
    for shift in (0.25, 0.5, 1.0, 2.0):
        case = zoo.shifted_gaussians(shift)
        draws = case.sample(jax.random.key(6), 1_000).samples[0]
        values.append(
            float(metric(draws[0], None, draws[1], None, case.space, key=KEY))
        )
    assert values == sorted(values)
    assert len(set(values)) == 4


@pytest.mark.parametrize("metric", METRICS)
def test_identical_distributions_lie_within_the_null(metric: Metric) -> None:
    """Two draws of one Gaussian give a value inside its permutation null."""
    rng = np.random.default_rng(12)
    first, second = rng.normal(size=(300, 2)), rng.normal(size=(300, 2))
    value = float(metric(first, None, second, None, PLANE, key=KEY))
    pooled = np.concatenate([first, second])
    null = []
    for _ in range(40):
        order = rng.permutation(600)
        null.append(
            float(
                metric(
                    pooled[order[:300]], None, pooled[order[300:]], None, PLANE, key=KEY
                )
            )
        )
    assert abs(value - np.mean(null)) < 4 * np.std(null, ddof=1)


# * Cap, checks and transformations


@pytest.mark.parametrize("metric", METRICS)
def test_the_cap_subsamples_each_posterior_once(
    rng: np.random.Generator, metric: Metric
) -> None:
    """Above the cap one warning names both counts, and the capped sets are used."""
    first = rng.normal(size=(500, 2))
    second = rng.normal(0.2, 1.0, size=(200, 2))
    with pytest.warns(SubsamplingWarning, match="500 and 200 samples exceed the cap"):
        value = metric(first, None, second, None, PLANE, key=KEY, cap=300)
    with pytest.warns(SubsamplingWarning):
        capped = cap_pair(
            jnp.asarray(first),
            jnp.zeros(500),
            jnp.asarray(second),
            jnp.zeros(200),
            key=KEY,
            metric="test",
            cap=300,
        )
    expected = metric(capped[0], None, capped[2], None, PLANE, key=KEY)
    assert float(value) == pytest.approx(float(expected), rel=1e-14)


def test_inputs_are_checked(rng: np.random.Generator) -> None:
    """Bandwidths must be positive and finite, subsamples hold at least one sample."""
    samples = rng.normal(size=(20, 2))
    for bandwidth in (0.0, -1.0, math.inf):
        with pytest.raises(ValueError, match="bandwidth must be positive and finite"):
            maximum_mean_discrepancy(
                samples, None, samples, None, PLANE, bandwidth=bandwidth, key=KEY
            )
    with pytest.raises(ValueError, match="sample_size must be at least 1"):
        median_bandwidth(samples, None, samples, None, PLANE, key=KEY, sample_size=0)
    with pytest.raises(ValueError, match=r"shaped \(N, 3\)"):
        energy_distance(samples, None, samples, None, TRIPLE, key=KEY)


def test_metrics_map_over_problems(rng: np.random.Generator) -> None:
    """Under jax.vmap each problem gets its own value, with a traced bandwidth."""
    first = jnp.asarray(rng.normal(size=(3, 200, 2)))
    second = jnp.asarray(rng.normal(0.3, 1.0, size=(3, 150, 2)))

    def both(one: jax.Array, two: jax.Array) -> tuple[jax.Array, jax.Array]:
        bandwidth = median_bandwidth(one, None, two, None, PLANE, key=KEY)
        return (
            maximum_mean_discrepancy(
                one, None, two, None, PLANE, bandwidth=bandwidth, key=KEY
            ),
            energy_distance(one, None, two, None, PLANE, key=KEY),
        )

    mapped = jax.vmap(both)(first, second)
    for index in range(3):
        single = both(first[index], second[index])
        assert float(mapped[0][index]) == pytest.approx(float(single[0]), rel=1e-12)
        assert float(mapped[1][index]) == pytest.approx(float(single[1]), rel=1e-12)
