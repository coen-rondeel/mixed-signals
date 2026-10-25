"""Tests for the nearest-neighbour KL, JS and Hellinger estimates."""

import math
from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import spatial, special

from mixed_signals import zoo
from mixed_signals.core.space import ParamSpace
from mixed_signals.errors import SubsamplingWarning, ValidityWarning
from mixed_signals.metrics._cap import _draw, cap_pair
from mixed_signals.metrics.divergence import (
    HELLINGER_NEIGHBOURS,
    JS_NEIGHBOURS,
    KL_NEIGHBOURS,
    knn_hellinger_distance,
    knn_js_divergence,
    knn_kl_divergence,
)
from mixed_signals.prng import derive_key

STANDARD_ERRORS = 5.0
TWO_PI = 2 * math.pi
KEY = jax.random.key(0)
PLANE = ParamSpace.model_validate({"parameters": [{"name": "x0"}, {"name": "x1"}]})
TRIPLE = ParamSpace.model_validate(
    {"parameters": [{"name": "x0"}, {"name": "x1"}, {"name": "x2"}]}
)

type Metric = Callable[..., jax.Array]
METRICS: tuple[Metric, ...] = (
    knn_kl_divergence,
    knn_js_divergence,
    knn_hellinger_distance,
)


def own_distances(points: np.ndarray, rank: int) -> np.ndarray:
    """Return the k-th neighbour distance of each point within its own set."""
    distances = spatial.cKDTree(points).query(points, k=rank + 1)[0]
    return np.asarray(distances, dtype=float)[:, rank]


def cross_distances(points: np.ndarray, others: np.ndarray, rank: int) -> np.ndarray:
    """Return the k-th neighbour distance of each point in another set."""
    queried = spatial.cKDTree(others).query(points, k=rank)[0]
    distances = np.asarray(queried, dtype=float)
    return distances if rank == 1 else distances[:, rank - 1]


def entropy(points: np.ndarray, rank: int) -> float:
    """Return the Kozachenko-Leonenko entropy ψ(n) − ψ(k) + ln V_d + d mean ln r_k."""
    count, dimension = points.shape
    log_ball = 0.5 * dimension * math.log(math.pi) - special.gammaln(
        0.5 * dimension + 1
    )
    return float(
        special.digamma(count)
        - special.digamma(rank)
        + log_ball
        + dimension * np.mean(np.log(own_distances(points, rank)))
    )


def bhattacharyya(first: np.ndarray, second: np.ndarray, rank: int) -> float:
    """Return the Póczos-Schneider estimate at α = ½ from the first set."""
    count, dimension = first.shape
    constant = math.exp(
        2 * special.gammaln(rank)
        - special.gammaln(rank + 0.5)
        - special.gammaln(rank - 0.5)
    )
    ratio = (count - 1) * own_distances(first, rank) ** dimension
    ratio /= len(second) * cross_distances(first, second, rank) ** dimension
    return float(constant * np.mean(np.sqrt(ratio)))


@pytest.fixture
def rng() -> np.random.Generator:
    """A seeded NumPy generator for test data."""
    return np.random.default_rng(2026)


# * The published formulas


@pytest.mark.parametrize("rank", [1, KL_NEIGHBOURS])
def test_kl_is_the_estimator_of_wang_kulkarni_and_verdu(
    rng: np.random.Generator, rank: int
) -> None:
    """(d/n) Σ ln(ν/ρ) + ln(m/(n − 1)), as SciPy's k-d tree measures the distances."""
    first = rng.normal(size=(800, 3))
    second = rng.normal(0.4, 1.2, size=(600, 3))
    value = knn_kl_divergence(
        first, None, second, None, TRIPLE, key=KEY, neighbours=rank
    )
    expected = 3 * np.mean(
        np.log(cross_distances(first, second, rank) / own_distances(first, rank))
    ) + math.log(600 / 799)
    assert float(value) == pytest.approx(expected, rel=1e-12)


def test_js_combines_kozachenko_leonenko_entropies(rng: np.random.Generator) -> None:
    """H(pooled) − ½H(p) − ½H(q), each entropy from its own samples."""
    first = rng.normal(size=(700, 3))
    second = rng.normal(0.5, 1.0, size=(700, 3))
    value = knn_js_divergence(first, None, second, None, TRIPLE, key=KEY)
    pooled = np.concatenate([first, second])
    expected = entropy(pooled, JS_NEIGHBOURS) - 0.5 * (
        entropy(first, JS_NEIGHBOURS) + entropy(second, JS_NEIGHBOURS)
    )
    assert float(value) == pytest.approx(expected, rel=1e-12)


def test_hellinger_averages_both_directions(rng: np.random.Generator) -> None:
    """BC is the mean of the two directional estimates; swapping changes nothing."""
    first = rng.normal(size=(700, 3))
    second = rng.normal(0.6, 0.9, size=(500, 3))
    value = knn_hellinger_distance(first, None, second, None, TRIPLE, key=KEY)
    coefficient = 0.5 * (
        bhattacharyya(first, second, HELLINGER_NEIGHBOURS)
        + bhattacharyya(second, first, HELLINGER_NEIGHBOURS)
    )
    assert float(value) == pytest.approx(math.sqrt(1 - coefficient), rel=1e-12)
    swapped = knn_hellinger_distance(second, None, first, None, TRIPLE, key=KEY)
    assert float(swapped) == pytest.approx(float(value), rel=1e-14)


def test_estimates_near_zero_stay_raw() -> None:
    """Negative KL estimates stay, and the Hellinger root keeps the sign of 1 − BC."""
    rng = np.random.default_rng(3)
    first, second = rng.normal(size=(300, 2)), rng.normal(size=(300, 2))
    value = float(knn_hellinger_distance(first, None, second, None, PLANE, key=KEY))
    coefficient = 0.5 * (
        bhattacharyya(first, second, HELLINGER_NEIGHBOURS)
        + bhattacharyya(second, first, HELLINGER_NEIGHBOURS)
    )
    assert coefficient > 1
    assert value == pytest.approx(-math.sqrt(coefficient - 1), rel=1e-12)
    rng = np.random.default_rng(1)
    first, second = rng.normal(size=(300, 2)), rng.normal(size=(300, 2))
    assert float(knn_kl_divergence(first, None, second, None, PLANE, key=KEY)) < 0


# * Sample counts, weights and the cap


def test_js_subsamples_the_larger_posterior(rng: np.random.Generator) -> None:
    """The larger set draws as many samples as the smaller from the documented key."""
    first = rng.normal(size=(300, 2))
    second = rng.normal(0.3, 1.0, size=(450, 2))
    value = knn_js_divergence(first, None, second, None, PLANE, key=KEY)
    kept, _ = _draw(
        jnp.asarray(second), jnp.zeros(450), derive_key(KEY, "equal", "second"), 300
    )
    equal = knn_js_divergence(first, None, kept, None, PLANE, key=KEY)
    assert float(value) == pytest.approx(float(equal), rel=1e-14)
    other = knn_js_divergence(first, None, second, None, PLANE, key=jax.random.key(9))
    assert float(other) != float(value)


def test_js_inside_jit_needs_equal_counts(rng: np.random.Generator) -> None:
    """Traced weights hide the counts: unequal ones give NaN, equal ones the value."""
    first = jnp.asarray(rng.normal(size=(200, 2)))
    second = jnp.asarray(rng.normal(size=(250, 2)))

    def compiled(one: jax.Array, two: jax.Array) -> jax.Array:
        return knn_js_divergence(one, None, two, None, PLANE, key=KEY)

    assert math.isnan(float(jax.jit(compiled)(first, second)))
    equal = jax.jit(compiled)(first, second[:200])
    assert float(equal) == pytest.approx(
        float(compiled(first, second[:200])), rel=1e-12
    )


@pytest.mark.parametrize("metric", METRICS)
def test_zero_weights_leave_every_estimate(
    rng: np.random.Generator, metric: Metric
) -> None:
    """Padding with zero-weight samples gives the estimate without them."""
    first = rng.normal(size=(400, 2))
    second = rng.normal(0.4, 1.0, size=(400, 2))
    keep = np.arange(400) % 5 != 0
    first_weights = np.where(keep, 0.0, -np.inf)
    second_weights = np.where(np.roll(keep, 1), 0.0, -np.inf)
    padded = metric(first, first_weights, second, second_weights, PLANE, key=KEY)
    plain = metric(first[keep], None, second[np.roll(keep, 1)], None, PLANE, key=KEY)
    assert float(padded) == pytest.approx(float(plain), rel=1e-12)


@pytest.mark.parametrize("metric", METRICS)
def test_unequal_weights_give_nan_and_warn(
    rng: np.random.Generator, metric: Metric
) -> None:
    """Weighted sets need resampling first."""
    first = rng.normal(size=(100, 2))
    second = rng.normal(size=(100, 2))
    with pytest.warns(ValidityWarning, match="equally weighted samples"):
        value = metric(first, np.linspace(0.0, 1.0, 100), second, None, PLANE, key=KEY)
    assert math.isnan(float(value))


@pytest.mark.parametrize("metric", [knn_kl_divergence, knn_hellinger_distance])
def test_coinciding_samples_give_nan_and_warn(
    rng: np.random.Generator, metric: Metric
) -> None:
    """A zero k-th neighbour distance has no ratio, as for repeated chain states."""
    first = np.concatenate(
        [np.repeat(rng.normal(size=(1, 2)), 6, axis=0), rng.normal(size=(100, 2))]
    )
    second = rng.normal(size=(100, 2))
    with pytest.warns(ValidityWarning, match="zero for 6 samples"):
        value = metric(first, None, second, None, PLANE, key=KEY)
    assert math.isnan(float(value))


@pytest.mark.parametrize("metric", METRICS)
def test_the_cap_subsamples_each_posterior_once(
    rng: np.random.Generator, metric: Metric
) -> None:
    """Above the cap one warning names both counts, and the capped sets are used."""
    first = rng.normal(size=(500, 2))
    second = rng.normal(0.2, 1.0, size=(400, 2))
    with pytest.warns(SubsamplingWarning, match="500 and 400 samples exceed the cap"):
        value = metric(first, None, second, None, PLANE, key=KEY, cap=300)
    with pytest.warns(SubsamplingWarning):
        capped = cap_pair(
            jnp.asarray(first),
            jnp.zeros(500),
            jnp.asarray(second),
            jnp.zeros(400),
            key=KEY,
            metric="test",
            cap=300,
        )
    expected = metric(capped[0], None, capped[2], None, PLANE, key=KEY)
    assert float(value) == pytest.approx(float(expected), rel=1e-14)


def test_settings_are_checked(rng: np.random.Generator) -> None:
    """The rank must be positive and the space must fit the samples."""
    samples = rng.normal(size=(50, 2))
    with pytest.raises(ValueError, match="greater than 0"):
        knn_kl_divergence(samples, None, samples, None, PLANE, key=KEY, neighbours=0)
    with pytest.raises(ValueError, match=r"shaped \(N, 3\)"):
        knn_js_divergence(samples, None, samples, None, TRIPLE, key=KEY)


# * Topologies


def turn_azimuth(draws: np.ndarray) -> np.ndarray:
    """Return sphere samples turned by one radian in azimuth."""
    turned = draws.copy()
    turned[..., 0] = (turned[..., 0] + 1.0) % TWO_PI
    return turned


@pytest.mark.parametrize("metric", METRICS)
@pytest.mark.parametrize(
    ("build", "turn"),
    [
        (
            lambda: zoo.wrapped_gaussians(1.0, width=0.5),
            lambda draws: (draws + 2.0) % TWO_PI,
        ),
        (lambda: zoo.von_mises_fisher(1.0, width=0.3), turn_azimuth),
    ],
)
def test_turning_circles_and_spheres_changes_nothing(
    metric: Metric,
    build: Callable[[], zoo.ZooCase],
    turn: Callable[[np.ndarray], np.ndarray],
) -> None:
    """Wrapped samples turned around the circle, or a sphere turned in azimuth."""
    case = build()
    draws = np.asarray(case.sample(jax.random.key(4), 600).samples[0])
    turned = turn(draws)
    value = metric(draws[0], None, draws[1], None, case.space, key=KEY)
    moved = metric(turned[0], None, turned[1], None, case.space, key=KEY)
    assert float(moved) == pytest.approx(float(value), rel=1e-9)


# * Statistical behaviour on the zoo


@pytest.mark.parametrize("metric", METRICS)
def test_identical_distributions_lie_within_the_null(metric: Metric) -> None:
    """Two draws of one Gaussian give a value inside its permutation null."""
    rng = np.random.default_rng(11)
    first, second = rng.normal(size=(400, 2)), rng.normal(size=(400, 2))
    value = float(metric(first, None, second, None, PLANE, key=KEY))
    pooled = np.concatenate([first, second])
    null = []
    for _ in range(30):
        order = rng.permutation(800)
        null.append(
            float(
                metric(
                    pooled[order[:400]], None, pooled[order[400:]], None, PLANE, key=KEY
                )
            )
        )
    assert abs(value - np.mean(null)) < 4 * np.std(null, ddof=1)


@pytest.mark.parametrize("metric", METRICS)
def test_larger_shifts_give_larger_values(metric: Metric) -> None:
    """The zoo's shifted Gaussians come out in order, in both KL directions."""
    values = []
    for shift in (0.5, 1.0, 2.0):
        case = zoo.shifted_gaussians(shift)
        draws = case.sample(jax.random.key(6), 2_000).samples[0]
        forward = metric(draws[0], None, draws[1], None, case.space, key=KEY)
        backward = metric(draws[1], None, draws[0], None, case.space, key=KEY)
        values.append((float(forward), float(backward)))
    for direction in (0, 1):
        ordered = [pair[direction] for pair in values]
        assert ordered == sorted(ordered)
        assert len(set(ordered)) == 3


def zoo_estimates(
    case: zoo.ZooCase, metric: Metric, pair: tuple[int, int], count: int
) -> np.ndarray:
    """Return the metric of one pair of the case over eight independent problems."""
    draws = np.asarray(case.sample(jax.random.key(7), count, batch_size=8).samples)
    return np.array(
        [
            float(
                metric(
                    problem[pair[0]], None, problem[pair[1]], None, case.space, key=KEY
                )
            )
            for problem in draws
        ]
    )


@pytest.mark.parametrize(
    ("build", "metric", "name", "pair"),
    [
        (
            lambda: zoo.scaled_gaussians(1.5, param_count=1),
            knn_kl_divergence,
            "kl",
            (0, 1),
        ),
        (
            lambda: zoo.scaled_gaussians(1.5, param_count=1),
            knn_kl_divergence,
            "kl",
            (1, 0),
        ),
        (lambda: zoo.von_mises_fisher(1.0, width=0.3), knn_kl_divergence, "kl", (0, 1)),
        (
            lambda: zoo.wrapped_gaussians(1.0, width=0.5),
            knn_js_divergence,
            "js",
            (0, 1),
        ),
        (
            lambda: zoo.railed_gaussians([1.0, 0.0], param_count=1),
            knn_js_divergence,
            "js",
            (0, 1),
        ),
        (
            lambda: zoo.wrapped_gaussians(1.0, width=0.5),
            knn_hellinger_distance,
            "hellinger",
            (0, 1),
        ),
        (
            lambda: zoo.von_mises_fisher(1.0, width=0.3),
            knn_hellinger_distance,
            "hellinger",
            (0, 1),
        ),
    ],
)
def test_low_dimensional_zoo_values_are_recovered(
    build: Callable[[], zoo.ZooCase],
    metric: Metric,
    name: str,
    pair: tuple[int, int],
) -> None:
    """In one or two dimensions the estimates lie within their errors and 10% bias.

    The zoo check behind the default ranks found biases below 10% in one and
    two dimensions for these sample sizes.
    """
    case = build()
    estimates = zoo_estimates(case, metric, pair, 3_000)
    truth = case.reference_value(name, [case.names[pair[0]], case.names[pair[1]]])
    error = estimates.std(ddof=1) / math.sqrt(len(estimates))
    assert abs(estimates.mean() - truth) <= STANDARD_ERRORS * error + 0.1 * truth


@pytest.mark.parametrize("metric", METRICS)
def test_estimates_compile_and_map_over_problems(
    rng: np.random.Generator, metric: Metric
) -> None:
    """Under jax.jit and jax.vmap each problem gets its concrete value."""
    first = jnp.asarray(rng.normal(size=(3, 200, 2)))
    second = jnp.asarray(rng.normal(0.3, 1.0, size=(3, 200, 2)))

    def estimate(one: jax.Array, two: jax.Array) -> jax.Array:
        return metric(one, None, two, None, PLANE, key=KEY)

    mapped = jax.jit(jax.vmap(estimate))(first, second)
    for index in range(3):
        single = estimate(first[index], second[index])
        assert float(mapped[index]) == pytest.approx(float(single), rel=1e-12)
