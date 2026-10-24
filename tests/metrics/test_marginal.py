"""Tests for the marginal metrics: KS, Kuiper, AD, W1, JS and Hellinger."""

import math
from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.integrate
import scipy.optimize
import scipy.stats

from mixed_signals import zoo
from mixed_signals.core.space import ParamSpace
from mixed_signals.errors import ValidityWarning
from mixed_signals.estimators import KernelDensity, KernelDensitySettings
from mixed_signals.metrics.marginal import (
    PANEL_COUNT,
    anderson_darling_distance,
    circle_wasserstein_distance,
    kde_divergences,
    ks_metric_names,
    ks_statistic,
    kuiper_statistic,
    marginal_distances,
    marginal_kde_divergences,
    shared_bandwidth_factor,
    wasserstein_distance,
)
from mixed_signals.zoo.distributions import TruncatedGaussian, WrappedGaussian


@pytest.fixture
def rng() -> np.random.Generator:
    """A seeded NumPy generator for test data."""
    return np.random.default_rng(2026)


def statistic(
    first: np.ndarray,
    second: np.ndarray,
    first_log_weights: np.ndarray | None = None,
    second_log_weights: np.ndarray | None = None,
) -> float:
    """Return the package's statistic of two one-dimensional samples."""
    return float(
        ks_statistic(
            jnp.asarray(first),
            jnp.zeros(first.shape)
            if first_log_weights is None
            else jnp.asarray(first_log_weights),
            jnp.asarray(second),
            jnp.zeros(second.shape)
            if second_log_weights is None
            else jnp.asarray(second_log_weights),
        )
    )


@pytest.mark.parametrize(
    ("first_size", "second_size", "decimals"),
    [(300, 200, None), (50, 500, 1), (1_000, 1_000, 0)],
)
def test_equal_weights_give_the_scipy_statistic(
    rng: np.random.Generator, first_size: int, second_size: int, decimals: int | None
) -> None:
    """With equal weights, D equals SciPy's ks_2samp, also for samples with ties."""
    first = rng.normal(size=first_size)
    second = rng.normal(0.3, 1.2, size=second_size)
    if decimals is not None:
        first, second = first.round(decimals), second.round(decimals)
    expected = scipy.stats.ks_2samp(first, second).statistic
    assert statistic(first, second) == pytest.approx(expected, abs=1e-12)


def test_integer_weights_count_as_repeated_samples(rng: np.random.Generator) -> None:
    """A weight of k counts like k copies of a sample; the scale does not matter."""
    first = rng.normal(size=40)
    copies = rng.integers(1, 5, size=40)
    second = rng.normal(0.5, 1.0, size=60)
    expected = scipy.stats.ks_2samp(np.repeat(first, copies), second).statistic
    log_weights = np.log(copies) + 7.0
    assert statistic(first, second, log_weights) == pytest.approx(expected, abs=1e-12)


def test_zero_weights_remove_samples(rng: np.random.Generator) -> None:
    """A log weight of −∞ removes a sample from its distribution function."""
    first = rng.normal(size=80)
    second = rng.normal(size=70)
    keep = np.arange(80) % 3 != 0
    expected = scipy.stats.ks_2samp(first[keep], second).statistic
    log_weights = np.where(keep, 0.0, -np.inf)
    assert statistic(first, second, log_weights) == pytest.approx(expected, abs=1e-12)


def test_leading_axes_give_one_statistic_per_row(rng: np.random.Generator) -> None:
    """Leading axes map: every row of (3, 2) pairs gets its own statistic."""
    first = rng.normal(size=(3, 2, 100))
    second = rng.normal(0.2, 1.0, size=(3, 2, 80))
    values = ks_statistic(
        jnp.asarray(first),
        jnp.zeros(first.shape),
        jnp.asarray(second),
        jnp.zeros(second.shape),
    )
    assert values.shape == (3, 2)
    for row in np.ndindex(3, 2):
        expected = scipy.stats.ks_2samp(first[row], second[row]).statistic
        assert float(values[row]) == pytest.approx(expected, abs=1e-12)


def test_identical_disjoint_and_swapped_samples(rng: np.random.Generator) -> None:
    """Identical samples give 0, separated ones 1, and swapping changes nothing."""
    first = rng.normal(size=200)
    second = rng.normal(size=150)
    assert statistic(first, first) == pytest.approx(0.0, abs=1e-12)
    assert statistic(first, first + 100.0) == pytest.approx(1.0)
    assert statistic(first, second) == pytest.approx(statistic(second, first))


def test_statistic_runs_under_vmap(rng: np.random.Generator) -> None:
    """The kernel maps over a batch axis like any JAX function."""
    first = jnp.asarray(rng.normal(size=(4, 50)))
    second = jnp.asarray(rng.normal(size=(4, 60)))
    mapped = jax.vmap(ks_statistic)(
        first, jnp.zeros((4, 50)), second, jnp.zeros((4, 60))
    )
    direct = ks_statistic(first, jnp.zeros((4, 50)), second, jnp.zeros((4, 60)))
    np.testing.assert_allclose(mapped, direct, rtol=0, atol=1e-15)


# * Kuiper, Anderson-Darling and W1 kernels


TWO_PI = 2 * math.pi
STANDARD_ERRORS = 5.0


def log_weights_of(weights: np.ndarray) -> jnp.ndarray:
    """Return log weights as a JAX array, −∞ for zero."""
    with np.errstate(divide="ignore"):
        return jnp.asarray(np.log(weights))


def distribution_gaps(
    first: np.ndarray,
    first_weights: np.ndarray,
    second: np.ndarray,
    second_weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return the distinct pooled values with F, G and the pooled mass at each."""
    points = np.unique(np.concatenate([first, second]))
    first_cdf = np.array([first_weights[first <= point].sum() for point in points])
    second_cdf = np.array([second_weights[second <= point].sum() for point in points])
    first_cdf /= first_weights.sum()
    second_cdf /= second_weights.sum()
    return points, first_cdf, second_cdf, 0.5 * (first_cdf + second_cdf)


def scholz_stephens_statistic(first: np.ndarray, second: np.ndarray) -> float:
    """Return A²_kN for k = 2 from SciPy's normalised statistic and its σ_N.

    SciPy returns (A²_kN − (k − 1))/σ_N, with the variance of Scholz and
    Stephens (1987, eq. 4), which depends only on the sample sizes.
    """
    result = scipy.stats.anderson_ksamp(
        [first, second],
        variant="right",
        method=scipy.stats.PermutationMethod(n_resamples=9, rng=0),
    )
    sizes = np.array([len(first), len(second)], dtype=float)
    total, groups, inverse = sizes.sum(), 2, (1 / sizes).sum()
    tail = (1.0 / np.arange(total - 1, 1, -1)).cumsum()
    harmonic = tail[-1] + 1
    weighted = (tail / np.arange(2, total)).sum()
    term_a = (4 * weighted - 6) * (groups - 1) + (10 - 6 * weighted) * inverse
    term_b = (
        (2 * weighted - 4) * groups**2
        + 8 * harmonic * groups
        + (2 * weighted - 14 * harmonic - 4) * inverse
        - 8 * harmonic
        + 4 * weighted
        - 6
    )
    term_c = (
        (6 * harmonic + 2 * weighted - 2) * groups**2
        + (4 * harmonic - 4 * weighted + 6) * groups
        + (2 * harmonic - 6) * inverse
        + 4 * harmonic
    )
    term_d = (2 * harmonic + 6) * groups**2 - 4 * harmonic * groups
    variance = (term_a * total**3 + term_b * total**2 + term_c * total + term_d) / (
        (total - 1) * (total - 2) * (total - 3)
    )
    return float(result.statistic * math.sqrt(variance) + (groups - 1))


@pytest.mark.parametrize("decimals", [None, 1])
def test_anderson_darling_equals_scipy_for_equal_sizes(
    rng: np.random.Generator, decimals: int | None
) -> None:
    """For n = m and equal weights it is SciPy's right-continuous A² over nm/(n + m)."""
    first, second = rng.normal(size=400), rng.normal(0.4, 1.3, size=400)
    if decimals is not None:
        first, second = first.round(decimals), second.round(decimals)
    value = anderson_darling_distance(
        jnp.asarray(first), jnp.zeros(400), jnp.asarray(second), jnp.zeros(400)
    )
    expected = scholz_stephens_statistic(first, second) * 800 / (400 * 400)
    assert float(value) == pytest.approx(expected, rel=1e-10)


def test_anderson_darling_pools_each_posterior_with_equal_mass(
    rng: np.random.Generator,
) -> None:
    """For unequal sizes and weights, H = (F + G)/2 with each posterior's mass one."""
    first, second = (
        rng.normal(size=150).round(1),
        rng.normal(0.3, 0.8, size=90).round(1),
    )
    first_weights, second_weights = rng.uniform(0.1, 2, 150), rng.uniform(0.1, 2, 90)
    points, first_cdf, second_cdf, pooled = distribution_gaps(
        first, first_weights, second, second_weights
    )
    mass = np.diff(np.concatenate([[0.0], pooled]))
    inside = slice(0, len(points) - 1)
    expected = np.sum(
        (first_cdf[inside] - second_cdf[inside]) ** 2
        * mass[inside]
        / (pooled[inside] * (1 - pooled[inside]))
    )
    value = anderson_darling_distance(
        jnp.asarray(first),
        log_weights_of(first_weights),
        jnp.asarray(second),
        log_weights_of(second_weights),
    )
    assert float(value) == pytest.approx(expected, rel=1e-10)


def test_integer_weights_and_zero_weights_act_as_counts(
    rng: np.random.Generator,
) -> None:
    """A weight of k counts like k copies, and a weight of zero removes a sample."""
    first, second = rng.normal(size=60), rng.normal(0.5, 1.0, size=40)
    copies = rng.integers(1, 4, size=60)
    repeated = np.repeat(first, copies)
    for kernel in (kuiper_statistic, anderson_darling_distance, wasserstein_distance):
        weighted = kernel(
            jnp.asarray(first),
            log_weights_of(copies),
            jnp.asarray(second),
            jnp.zeros(40),
        )
        direct = kernel(
            jnp.asarray(repeated),
            jnp.zeros(len(repeated)),
            jnp.asarray(second),
            jnp.zeros(40),
        )
        assert float(weighted) == pytest.approx(float(direct), rel=1e-12)
        kept = np.arange(60) % 4 != 0
        removed = kernel(
            jnp.asarray(first),
            log_weights_of(kept.astype(float)),
            jnp.asarray(second),
            jnp.zeros(40),
        )
        dropped = kernel(
            jnp.asarray(first[kept]),
            jnp.zeros(kept.sum()),
            jnp.asarray(second),
            jnp.zeros(40),
        )
        assert float(removed) == pytest.approx(float(dropped), rel=1e-12)


def test_kuiper_matches_a_direct_count(rng: np.random.Generator) -> None:
    """V is the largest gap upwards plus the largest gap downwards, with weights."""
    first, second = (
        rng.vonmises(0.0, 4.0, 60) % TWO_PI,
        rng.vonmises(2.5, 1.0, 50) % TWO_PI,
    )
    first_weights, second_weights = rng.uniform(0.2, 1, 60), rng.uniform(0.2, 1, 50)
    _, first_cdf, second_cdf, _ = distribution_gaps(
        first, first_weights, second, second_weights
    )
    gaps = first_cdf - second_cdf
    expected = max(gaps.max(), 0.0) - min(gaps.min(), 0.0)
    value = kuiper_statistic(
        jnp.asarray(first),
        log_weights_of(first_weights),
        jnp.asarray(second),
        log_weights_of(second_weights),
    )
    assert float(value) == pytest.approx(expected, rel=1e-12)


def test_kuiper_and_circle_w1_do_not_depend_on_the_cut(
    rng: np.random.Generator,
) -> None:
    """Turning both samples around the circle changes neither V nor the circle W1."""
    first, second = (
        rng.vonmises(0.3, 3.0, 80) % TWO_PI,
        rng.vonmises(1.2, 2.0, 70) % TWO_PI,
    )
    turned_first, turned_second = (first + 2.0) % TWO_PI, (second + 2.0) % TWO_PI

    def both(one: np.ndarray, two: np.ndarray) -> tuple[float, float]:
        arguments = (jnp.asarray(one), jnp.zeros(80), jnp.asarray(two), jnp.zeros(70))
        return (
            float(kuiper_statistic(*arguments)),
            float(circle_wasserstein_distance(*arguments, TWO_PI)),
        )

    assert both(turned_first, turned_second) == pytest.approx(
        both(first, second), rel=1e-10
    )


def test_kuiper_equals_ks_when_one_function_stays_above() -> None:
    """When F − G keeps one sign, as for a shift, V and D agree."""
    first = np.linspace(-2.0, 2.0, 50)
    second = first + 0.7
    arguments = (jnp.asarray(first), jnp.zeros(50), jnp.asarray(second), jnp.zeros(50))
    assert float(kuiper_statistic(*arguments)) == pytest.approx(
        float(ks_statistic(*arguments))
    )


def test_wasserstein_matches_scipy_with_weights(rng: np.random.Generator) -> None:
    """The line W1 equals SciPy's wasserstein_distance with weights."""
    first, second = rng.normal(size=300), rng.normal(0.5, 0.7, size=200)
    first_weights, second_weights = rng.uniform(0.1, 2, 300), rng.uniform(0.1, 2, 200)
    value = wasserstein_distance(
        jnp.asarray(first),
        log_weights_of(first_weights),
        jnp.asarray(second),
        log_weights_of(second_weights),
    )
    expected = scipy.stats.wasserstein_distance(
        first, second, first_weights, second_weights
    )
    assert float(value) == pytest.approx(expected, rel=1e-12)


def transport_cost(
    first: np.ndarray,
    first_weights: np.ndarray,
    second: np.ndarray,
    second_weights: np.ndarray,
) -> float:
    """Return the optimal-transport cost along the circle, by linear programming."""
    gaps = np.abs(first[:, None] - second[None, :])
    cost = np.minimum(gaps, TWO_PI - gaps)
    rows, columns = cost.shape
    row_constraints = np.kron(np.eye(rows), np.ones(columns))
    column_constraints = np.kron(np.ones(rows), np.eye(columns))
    result = scipy.optimize.linprog(
        cost.ravel(),
        A_eq=np.vstack([row_constraints, column_constraints]),
        b_eq=np.concatenate(
            [first_weights / first_weights.sum(), second_weights / second_weights.sum()]
        ),
        bounds=(0, None),
        method="highs",
    )
    assert result.fun is not None
    return float(result.fun)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_circle_wasserstein_solves_the_transport_problem(seed: int) -> None:
    """The median formula equals the transport cost with arc lengths as cost."""
    rng = np.random.default_rng(seed)
    first = rng.vonmises(rng.uniform(0, TWO_PI), 2.0, 25) % TWO_PI
    second = rng.vonmises(rng.uniform(0, TWO_PI), 0.5, 20) % TWO_PI
    first_weights, second_weights = rng.uniform(0.2, 1, 25), rng.uniform(0.2, 1, 20)
    value = circle_wasserstein_distance(
        jnp.asarray(first),
        log_weights_of(first_weights),
        jnp.asarray(second),
        log_weights_of(second_weights),
        TWO_PI,
    )
    expected = transport_cost(first, first_weights, second, second_weights)
    assert float(value) == pytest.approx(expected, rel=1e-9)
    assert float(value) <= math.pi


# * Distances of every marginal


def mixed_space() -> ParamSpace:
    """Return a space with a circle, a half-line interval and a real axis."""
    return ParamSpace.model_validate(
        {
            "parameters": [
                {"name": "phase", "topology": "circle", "bounds": [-math.pi, math.pi]},
                {"name": "mass", "topology": "interval", "bounds": [0.0, math.inf]},
                {"name": "x2"},
            ]
        }
    )


def test_marginal_distances_choose_the_form_of_each_topology(
    rng: np.random.Generator,
) -> None:
    """Circles take Kuiper, the circle W1 and AD after a cut; lines the line forms."""
    space = mixed_space()
    first = np.stack(
        [
            rng.vonmises(3.0, 8.0, 300),
            np.abs(rng.normal(0.5, 1.0, 300)),
            rng.normal(size=300),
        ],
        axis=1,
    )
    second = np.stack(
        [
            rng.vonmises(-2.9, 4.0, 200),
            np.abs(rng.normal(1.0, 1.0, 200)),
            rng.normal(0.4, 1.0, 200),
        ],
        axis=1,
    )
    values = marginal_distances(first, None, second, None, space)
    assert ks_metric_names(space) == ("kuiper_ks", "ks", "ks")
    zeros_first, zeros_second = jnp.zeros(300), jnp.zeros(200)
    angles = [
        jnp.asarray((samples[:, 0] + math.pi) % TWO_PI - math.pi)
        for samples in (first, second)
    ]
    assert float(values.ks[0]) == pytest.approx(
        float(kuiper_statistic(angles[0], zeros_first, angles[1], zeros_second))
    )
    assert float(values.w1[0]) == pytest.approx(
        float(
            circle_wasserstein_distance(
                angles[0], zeros_first, angles[1], zeros_second, TWO_PI
            )
        )
    )
    pooled = np.concatenate([first[:, 0], second[:, 0]])
    weights = np.concatenate([np.full(300, 1 / 600), np.full(200, 1 / 400)])
    centre = np.angle(np.sum(weights * np.exp(1j * pooled)))
    unwrapped = [
        jnp.asarray((samples[:, 0] - centre + math.pi) % TWO_PI + centre - math.pi)
        for samples in (first, second)
    ]
    assert float(values.ad[0]) == pytest.approx(
        float(
            anderson_darling_distance(
                unwrapped[0], zeros_first, unwrapped[1], zeros_second
            )
        ),
        rel=1e-10,
    )
    for column in (1, 2):
        arguments = (
            jnp.asarray(first[:, column]),
            zeros_first,
            jnp.asarray(second[:, column]),
            zeros_second,
        )
        assert float(values.ks[column]) == pytest.approx(
            float(ks_statistic(*arguments))
        )
        assert float(values.w1[column]) == pytest.approx(
            float(wasserstein_distance(*arguments))
        )
        assert float(values.ad[column]) == pytest.approx(
            float(anderson_darling_distance(*arguments))
        )


@pytest.mark.parametrize(
    "build",
    [
        lambda: zoo.shifted_gaussians(0.8, param_count=2),
        lambda: zoo.railed_gaussians([1.5, 0.0], param_count=2),
        lambda: zoo.wrapped_gaussians(1.0, width=0.5, param_count=2, correlation=0.3),
        lambda: zoo.wrapped_gaussians(-2.5, width=0.4, period=math.pi, lower=-1.0),
    ],
)
def test_marginal_distances_recover_the_zoo(build: Callable[[], zoo.ZooCase]) -> None:
    """Every KS or Kuiper, AD and W1 value of a pair is reproduced within its errors.

    The standard error comes from disjoint batches, and the bias bounds are
    those of the zoo's own checks, doubled for V, which adds two gaps.
    """
    case = build()
    sample_count, batch_count = 40_000, 20
    draws = np.asarray(case.sample(jax.random.key(5), sample_count).samples[0])
    pair = tuple(case.names[:2])
    names = ks_metric_names(case.space)

    def distances(part: slice) -> np.ndarray:
        values = marginal_distances(
            draws[0, part], None, draws[1, part], None, case.space
        )
        return np.stack([values.ks, values.ad, values.w1])

    size = sample_count // batch_count
    spread = np.std(
        [
            distances(slice(start, start + size))
            for start in range(0, sample_count, size)
        ],
        axis=0,
        ddof=1,
    )
    estimate, error = distances(slice(None)), spread / math.sqrt(batch_count)
    widths = draws[:2].std(axis=1).sum(axis=0)
    for column, parameter in enumerate(case.space.names):
        rows = [
            (
                names[column],
                0,
                (2 if names[column] == "kuiper_ks" else 1)
                * 2
                * math.sqrt(2 / sample_count),
            ),
            ("ad", 1, 4 / sample_count),
            ("w1", 2, 2 * widths[column] / math.sqrt(sample_count)),
        ]
        for metric, row, bias in rows:
            truth = case.reference_value(metric, pair, parameter)
            tolerance = STANDARD_ERRORS * error[row, column] + bias
            assert abs(estimate[row, column] - truth) <= tolerance, (metric, parameter)


def test_unresolved_pooled_circle_warns_about_the_cut() -> None:
    """Pooled values spread around the circle leave the AD cut arbitrary."""
    space = ParamSpace.model_validate(
        {"parameters": [{"name": "phase", "topology": "circle", "bounds": [0, TWO_PI]}]}
    )
    first = np.linspace(0.0, TWO_PI, 40, endpoint=False)[:, None]
    second = (first + math.pi / 40) % TWO_PI
    with pytest.warns(ValidityWarning, match="Anderson-Darling cut"):
        marginal_distances(first, None, second, None, space)


def test_sphere_axes_follow_their_own_topologies() -> None:
    """A sphere's azimuth is a circle and its polar axis an interval."""
    case = zoo.von_mises_fisher(1.0, width=0.3, colatitude=1.0, euclidean_count=1)
    draws = np.asarray(case.sample(jax.random.key(2), 2_000).samples[0])
    values = marginal_distances(draws[0], None, draws[1], None, case.space)
    assert ks_metric_names(case.space) == ("kuiper_ks", "ks", "ks")
    assert np.all(np.isfinite(np.asarray(values.ks)))
    assert float(values.ks[2]) < 0.1


# * Kernel density divergences


def fitted_pair(
    entry: dict[str, object], first: np.ndarray, second: np.ndarray
) -> tuple[KernelDensity, KernelDensity]:
    """Return the two estimates of one parameter with the shared factor."""
    space = ParamSpace.model_validate({"parameters": [entry]})
    factor = float(
        shared_bandwidth_factor(jnp.zeros(len(first)), jnp.zeros(len(second)))
    )
    settings = KernelDensitySettings(bandwidth=factor)
    return (
        KernelDensity.fit(first[:, None], None, space, settings),
        KernelDensity.fit(second[:, None], None, space, settings),
    )


def quadpack_divergences(
    first: KernelDensity, second: KernelDensity, lower: float, upper: float
) -> tuple[float, float]:
    """Return JS and Hellinger of two estimates by adaptive QUADPACK quadrature."""

    def densities(point: float) -> tuple[float, float]:
        values = np.array([[point]])
        return (
            float(np.exp(first.log_density(values))[0]),
            float(np.exp(second.log_density(values))[0]),
        )

    def js_integrand(point: float) -> float:
        density_p, density_q = densities(point)
        return sum(
            0.5 * own * math.log(2 * own / (density_p + density_q))
            for own in (density_p, density_q)
            if own > 0
        )

    def hellinger_integrand(point: float) -> float:
        density_p, density_q = densities(point)
        return 0.5 * (math.sqrt(density_p) - math.sqrt(density_q)) ** 2

    breaks = list(np.linspace(lower, upper, 60))
    js, squared = (
        scipy.integrate.quad(
            integrand,
            lower,
            upper,
            points=breaks,
            limit=2_000,
            epsabs=1e-12,
            epsrel=1e-10,
        )[0]
        for integrand in (js_integrand, hellinger_integrand)
    )
    return float(js), math.sqrt(squared)


@pytest.mark.parametrize(
    ("entry", "draw", "limits"),
    [
        (
            {"name": "x"},
            lambda rng: (rng.normal(0, 1, 3_000), rng.normal(0.8, 0.6, 2_000)),
            (-12.0, 12.0),
        ),
        (
            {"name": "x", "topology": "interval", "bounds": [0.0, math.inf]},
            lambda rng: (
                np.abs(rng.normal(0.3, 1, 3_000)),
                np.abs(rng.normal(1.0, 1, 2_000)),
            ),
            (0.0, 12.0),
        ),
        (
            {"name": "x", "topology": "circle", "bounds": [0.0, TWO_PI]},
            lambda rng: (
                rng.vonmises(0.0, 3.0, 3_000) % TWO_PI,
                rng.vonmises(0.7, 1.0, 2_000) % TWO_PI,
            ),
            (0.0, TWO_PI),
        ),
    ],
)
def test_kde_divergences_match_quadpack(
    rng: np.random.Generator,
    entry: dict[str, object],
    draw: Callable[[np.random.Generator], tuple[np.ndarray, np.ndarray]],
    limits: tuple[float, float],
) -> None:
    """On a line, an interval and a circle the panels give QUADPACK's integrals."""
    first, second = fitted_pair(entry, *draw(rng))
    values = kde_divergences(first, second)
    js, hellinger = quadpack_divergences(first, second, *limits)
    assert float(values.js) == pytest.approx(js, rel=1e-9)
    assert float(values.hellinger) == pytest.approx(hellinger, rel=1e-9)
    same = kde_divergences(first, first)
    assert float(same.js) == 0.0
    assert float(same.hellinger) == 0.0


def test_shared_factor_is_silverman_at_the_smaller_kish_size(
    rng: np.random.Generator,
) -> None:
    """The factor is (3n/4)^(−1/5) at the smaller Kish effective sample size."""
    log_weights = jnp.asarray(rng.normal(size=500))
    weights = np.exp(np.asarray(log_weights))
    kish = weights.sum() ** 2 / np.sum(weights**2)
    factor = shared_bandwidth_factor(log_weights, jnp.zeros(800))
    assert float(factor) == pytest.approx((0.75 * kish) ** -0.2, rel=1e-12)


def test_kde_divergences_check_their_inputs(rng: np.random.Generator) -> None:
    """Mismatched spaces and no panels are refused; coarse panels give NaN."""
    first, second = fitted_pair(
        {"name": "x"}, rng.normal(size=500), rng.normal(size=400)
    )
    other, _ = fitted_pair({"name": "y"}, rng.normal(size=500), rng.normal(size=400))
    with pytest.raises(ValueError, match="same space"):
        kde_divergences(first, other)
    with pytest.raises(ValueError, match="panel_count must be positive"):
        kde_divergences(first, second, panel_count=0)
    coarse = kde_divergences(first, second, panel_count=2)
    assert math.isnan(float(coarse.js))
    assert math.isnan(float(coarse.hellinger))


def test_kde_divergences_inside_jit_use_the_default_panels(
    rng: np.random.Generator,
) -> None:
    """Traced estimates get the default panel count, enough for these samples."""
    first, second = fitted_pair(
        {"name": "x"}, rng.normal(size=500), rng.normal(1, 1, 400)
    )
    compiled = jax.jit(kde_divergences)(first, second)
    eager = kde_divergences(first, second, panel_count=PANEL_COUNT)
    assert float(compiled.js) == pytest.approx(float(eager.js), rel=1e-12)


def smoothed_truth(case: zoo.ZooCase, metric: str, inflation: float) -> float:
    """Return the zoo value for the pair with each width widened by the kernel."""
    distribution, partner = case.posteriors[0], case.posteriors[1]
    if isinstance(distribution, WrappedGaussian):
        assert isinstance(partner, WrappedGaussian)
        function = getattr(zoo.closed_forms, f"wrapped_normal_{metric}")
        centres = [
            float(posterior.base.mean[0]) for posterior in (distribution, partner)
        ]
        width = math.sqrt(float(distribution.base.covariance[0, 0]))
        return float(function(*centres, width * inflation, float(distribution.period)))
    if isinstance(distribution, TruncatedGaussian):
        assert isinstance(partner, TruncatedGaussian)
        function = getattr(zoo.closed_forms, f"truncated_normal_{metric}")
        means = [float(posterior.mean[0]) for posterior in (distribution, partner)]
        return float(
            function(
                *means,
                inflation,
                float(distribution.lower[0]),
                float(distribution.upper[0]),
            )
        )
    function = getattr(zoo.closed_forms, f"normal_{metric}")
    pair = tuple(case.names[:2])
    moments = [
        (
            case.reference_value("mean", name, "x0"),
            case.reference_value("std", name, "x0"),
        )
        for name in pair
    ]
    return float(
        function(
            moments[0][0],
            moments[0][1] * inflation,
            moments[1][0],
            moments[1][1] * inflation,
        )
    )


@pytest.mark.parametrize(
    ("build", "exact"),
    [
        (lambda: zoo.shifted_gaussians(1.0, param_count=1), True),
        (lambda: zoo.scaled_gaussians(0.5, param_count=1), True),
        (lambda: zoo.wrapped_gaussians(1.0, width=0.5), True),
        (lambda: zoo.railed_gaussians([1.0, 0.0], param_count=1), False),
    ],
)
def test_marginal_kde_divergences_recover_the_zoo(
    build: Callable[[], zoo.ZooCase], exact: bool
) -> None:
    """JS and Hellinger reach the zoo values of the pair smoothed by the kernels.

    A Gaussian kernel of width f·σ widens a normal or a wrapped normal to
    width σ√(1 + f²), so those smoothed values are exact expectations up to the
    estimates' variance. A railed normal has no such form; there the bias of
    the corrected estimate, of order h² (Jones and Foster 1996), is bounded by
    the change that this smoothing makes.
    """
    case = build()
    problems, sample_count = 8, 20_000
    sample_set = case.sample(jax.random.key(13), sample_count, batch_size=problems)
    values = [
        marginal_kde_divergences(
            np.asarray(sample_set.samples[problem, 0]),
            None,
            np.asarray(sample_set.samples[problem, 1]),
            None,
            case.space,
        )
        for problem in range(problems)
    ]
    factor = float(
        shared_bandwidth_factor(jnp.zeros(sample_count), jnp.zeros(sample_count))
    )
    inflation = math.sqrt(1 + factor**2)
    pair = tuple(case.names[:2])
    for metric in ("js", "hellinger"):
        estimates = np.array([float(getattr(value, metric)[0]) for value in values])
        error = estimates.std(ddof=1) / math.sqrt(problems)
        truth = case.reference_value(metric, pair, "x0")
        smoothed = smoothed_truth(case, metric, inflation)
        target, bias = (smoothed, 0.0) if exact else (truth, abs(truth - smoothed))
        assert abs(estimates.mean() - target) <= STANDARD_ERRORS * error + bias, metric


def test_marginal_distances_map_over_problems(rng: np.random.Generator) -> None:
    """Inside vmap the samples are traced: no warning, and the looped values."""
    space = mixed_space()
    first = np.stack(
        [
            rng.vonmises(0.0, 4.0, (3, 200)),
            np.abs(rng.normal(size=(3, 200))),
            rng.normal(size=(3, 200)),
        ],
        axis=-1,
    )
    second = first[:, ::-1] * np.array([1.0, 1.5, 0.5]) + np.array([0.3, 0.0, 0.2])
    mapped = jax.vmap(lambda one, two: marginal_distances(one, None, two, None, space))(
        jnp.asarray(first), jnp.asarray(second)
    )
    for problem in range(3):
        looped = marginal_distances(first[problem], None, second[problem], None, space)
        for name in ("ks", "ad", "w1"):
            np.testing.assert_allclose(
                getattr(mapped, name)[problem], getattr(looped, name), rtol=1e-10
            )


def test_marginal_kde_divergences_take_a_given_factor(rng: np.random.Generator) -> None:
    """A given factor is used as it is; traced samples need one."""
    space = ParamSpace.model_validate({"parameters": [{"name": "x0"}]})
    first, second = rng.normal(size=(800, 1)), rng.normal(0.5, 1.0, size=(600, 1))
    factor = 0.3
    given = marginal_kde_divergences(first, None, second, None, space, factor=factor)
    settings = KernelDensitySettings(bandwidth=factor)
    expected = kde_divergences(
        KernelDensity.fit(first, None, space, settings),
        KernelDensity.fit(second, None, space, settings),
    )
    assert float(given.js[0]) == pytest.approx(float(expected.js), rel=1e-12)
    with pytest.raises(ValueError, match="pass the bandwidth factor"):
        jax.jit(lambda one, two: marginal_kde_divergences(one, None, two, None, space))(
            jnp.asarray(first), jnp.asarray(second)
        )


def test_kde_divergences_of_an_invalid_estimate_are_nan() -> None:
    """An estimate of samples without spread is NaN, and so are its divergences."""
    space = ParamSpace.model_validate({"parameters": [{"name": "x0"}]})
    settings = KernelDensitySettings(bandwidth=0.2)
    with pytest.warns(ValidityWarning, match="NaN"):
        flat = KernelDensity.fit(np.full((50, 1), 2.0), None, space, settings)
    spread = KernelDensity.fit(np.linspace(0, 1, 50)[:, None], None, space, settings)
    values = kde_divergences(flat, spread)
    assert math.isnan(float(values.js))
    assert math.isnan(float(values.hellinger))
