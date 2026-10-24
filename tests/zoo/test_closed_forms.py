"""Tests for the zoo's JAX closed forms against independent SciPy computations."""

import itertools
import math
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.typing import ArrayLike
from scipy import integrate, linalg, optimize, stats

from mixed_signals.zoo import closed_forms

type GridIntegral = Callable[..., float]

STANDARD_ERRORS = 5.0
NORMAL_PAIRS = [
    (0.0, 1.0, 1.5, 1.0),
    (0.3, 0.7, -0.4, 1.9),
    (2.0, 1.0, 2.0, 0.5),
    (-1.0, 0.2, 3.0, 0.25),
]


def normal_quad(integrand: Callable[[float], float], *centres: float) -> float:
    """Integrate a function of one variable over a wide interval around the centres."""
    return float(
        integrate.quad(
            integrand,
            min(centres) - 40,
            max(centres) + 40,
            points=sorted(set(centres)),
            limit=500,
            epsabs=1e-13,
            epsrel=1e-12,
        )[0]
    )


def random_covariance(rng: np.random.Generator, size: int) -> np.ndarray:
    """Return a well-conditioned random covariance."""
    factor = rng.normal(size=(size, size))
    return factor @ factor.T + size * np.eye(size)


# * Gaussian divergences


@pytest.mark.parametrize(("mean_p", "std_p", "mean_q", "std_q"), NORMAL_PAIRS)
def test_gaussian_kl_matches_quadrature_in_one_dimension(
    mean_p: float, std_p: float, mean_q: float, std_q: float
) -> None:
    """KL(p ‖ q) of two normals equals the integral of p ln(p/q)."""
    first, second = stats.norm(mean_p, std_p), stats.norm(mean_q, std_q)
    expected = normal_quad(
        lambda point: first.pdf(point) * (first.logpdf(point) - second.logpdf(point)),
        mean_p,
        mean_q,
    )
    terms = closed_forms.gaussian_kl(
        np.array([mean_p]),
        np.array([[std_p**2]]),
        np.array([mean_q]),
        np.array([[std_q**2]]),
    )
    assert terms.total == pytest.approx(expected, rel=1e-10, abs=1e-12)
    assert terms.total == pytest.approx(terms.location + terms.shape, rel=1e-14)


def test_gaussian_kl_is_invariant_under_affine_maps() -> None:
    """Mapping both Gaussians through x → Ax + b leaves the KL divergence unchanged."""
    rng = np.random.default_rng(1)
    cov_p, cov_q = random_covariance(rng, 3), random_covariance(rng, 3)
    mean_p, mean_q = rng.normal(size=3), rng.normal(size=3)
    transform, offset = rng.normal(size=(3, 3)), rng.normal(size=3)
    original = closed_forms.gaussian_kl(mean_p, cov_p, mean_q, cov_q)
    mapped = closed_forms.gaussian_kl(
        transform @ mean_p + offset,
        transform @ cov_p @ transform.T,
        transform @ mean_q + offset,
        transform @ cov_q @ transform.T,
    )
    np.testing.assert_allclose(mapped, original, rtol=1e-9)


def test_gaussian_kl_of_identical_covariances_has_an_exact_zero_shape_term() -> None:
    """A pure shift gives ½ Δμᵀ Σ⁻¹ Δμ and a shape term of exactly zero."""
    covariance = np.array([[2.0, 0.5], [0.5, 1.0]])
    shift = np.array([0.4, -1.2])
    terms = closed_forms.gaussian_kl(np.zeros(2), covariance, shift, covariance)
    assert terms.shape == 0.0
    assert terms.location == pytest.approx(
        0.5 * shift @ np.linalg.solve(covariance, shift), rel=1e-12
    )


def test_gaussian_w2_matches_known_cases() -> None:
    """W2 reduces to √(Δμ² + Δσ²) in one dimension and to √λ gaps for diagonals."""
    one_dimension = closed_forms.gaussian_w2(
        np.array([0.3]), np.array([[0.49]]), np.array([-0.4]), np.array([[3.61]])
    )
    assert one_dimension.total == pytest.approx(math.hypot(0.7, 1.2), rel=1e-12)
    diagonal = closed_forms.gaussian_w2(
        np.zeros(3), np.diag([1.0, 4.0, 9.0]), np.ones(3), np.diag([4.0, 1.0, 9.0])
    )
    assert diagonal.location == pytest.approx(math.sqrt(3), rel=1e-12)
    assert diagonal.shape == pytest.approx(math.sqrt(2), rel=1e-10)


def test_gaussian_w2_shape_matches_a_matrix_square_root() -> None:
    """The Bures term agrees with SciPy's matrix square root and is symmetric."""
    rng = np.random.default_rng(2)
    cov_p, cov_q = random_covariance(rng, 4), random_covariance(rng, 4)
    root = linalg.sqrtm(cov_p)
    cross = linalg.sqrtm(root @ cov_q @ root)
    expected = math.sqrt(np.trace(cov_p) + np.trace(cov_q) - 2 * np.trace(cross).real)
    forward = closed_forms.gaussian_w2(np.zeros(4), cov_p, np.ones(4), cov_q)
    backward = closed_forms.gaussian_w2(np.ones(4), cov_q, np.zeros(4), cov_p)
    assert forward.shape == pytest.approx(expected, rel=1e-9)
    assert backward.total == pytest.approx(forward.total, rel=1e-9)
    assert forward.total == pytest.approx(math.hypot(2.0, expected), rel=1e-9)


@pytest.mark.parametrize(("mean_p", "std_p", "mean_q", "std_q"), NORMAL_PAIRS)
def test_hellinger_distances_match_quadrature(
    mean_p: float, std_p: float, mean_q: float, std_q: float
) -> None:
    """H² = 1 − ∫ √(pq) for the Gaussian and the marginal forms."""
    first, second = stats.norm(mean_p, std_p), stats.norm(mean_q, std_q)
    coefficient = normal_quad(
        lambda point: math.sqrt(first.pdf(point) * second.pdf(point)), mean_p, mean_q
    )
    expected = math.sqrt(1 - coefficient)
    assert closed_forms.normal_hellinger(mean_p, std_p, mean_q, std_q) == pytest.approx(
        expected, rel=1e-9
    )


def test_hellinger_terms_combine_through_their_coefficients() -> None:
    """1 − H² of the total is the product of 1 − H² of the location and shape terms."""
    rng = np.random.default_rng(3)
    terms = closed_forms.gaussian_hellinger(
        rng.normal(size=3),
        random_covariance(rng, 3),
        rng.normal(size=3),
        random_covariance(rng, 3),
    )
    assert 1 - terms.total**2 == pytest.approx(
        (1 - terms.location**2) * (1 - terms.shape**2), rel=1e-12
    )
    assert 0 < terms.total < 1


def test_mahalanobis_shift_whitens_the_mean_difference() -> None:
    """The shift is the Euclidean norm for unit covariance and scales with the width."""
    assert closed_forms.mahalanobis_shift(
        np.zeros(2), np.eye(2), np.array([3.0, 4.0])
    ) == pytest.approx(5.0)
    assert closed_forms.mahalanobis_shift(
        np.zeros(2), 4 * np.eye(2), np.array([3.0, 4.0])
    ) == pytest.approx(2.5)


# * One-dimensional normals


@pytest.mark.parametrize(("mean_p", "std_p", "mean_q", "std_q"), NORMAL_PAIRS)
def test_normal_ks_matches_a_dense_grid(
    mean_p: float, std_p: float, mean_q: float, std_q: float
) -> None:
    """The supremum at the density crossings equals the maximum on a fine grid."""
    grid = np.linspace(-20, 20, 4_000_001)
    expected = np.max(
        np.abs(
            stats.norm.cdf(grid, mean_p, std_p) - stats.norm.cdf(grid, mean_q, std_q)
        )
    )
    value = closed_forms.normal_ks(mean_p, std_p, mean_q, std_q)
    assert value == pytest.approx(expected, abs=1e-9)
    assert closed_forms.normal_ks(mean_q, std_q, mean_p, std_p) == pytest.approx(value)


def test_normal_ks_of_equal_widths_is_the_midpoint_formula() -> None:
    """For equal widths the distance is 2Φ(|Δμ|/(2σ)) − 1, and zero for equal inputs."""
    assert closed_forms.normal_ks(0.0, 2.0, 3.0, 2.0) == pytest.approx(
        2 * stats.norm.cdf(0.75) - 1, rel=1e-12
    )
    assert closed_forms.normal_ks(1.0, 2.0, 1.0, 2.0) == 0.0


@pytest.mark.parametrize(("mean_p", "std_p", "mean_q", "std_q"), NORMAL_PAIRS)
def test_normal_w1_matches_the_integral_of_the_cdf_gap(
    mean_p: float, std_p: float, mean_q: float, std_q: float
) -> None:
    """W1 equals ∫ |F_p − F_q|, the area between the distribution functions."""
    expected = normal_quad(
        lambda point: abs(
            stats.norm.cdf(point, mean_p, std_p) - stats.norm.cdf(point, mean_q, std_q)
        ),
        mean_p,
        mean_q,
    )
    assert closed_forms.normal_w1(mean_p, std_p, mean_q, std_q) == pytest.approx(
        expected, rel=1e-8
    )


@pytest.mark.parametrize(("mean_p", "std_p", "mean_q", "std_q"), NORMAL_PAIRS)
def test_normal_js_matches_the_entropy_form(
    mean_p: float, std_p: float, mean_q: float, std_q: float
) -> None:
    """JS equals H(m) − ½H(p) − ½H(q), integrated through another route."""
    first, second = stats.norm(mean_p, std_p), stats.norm(mean_q, std_q)

    def mixture_entropy_density(point: float) -> float:
        density = 0.5 * (first.pdf(point) + second.pdf(point))
        return -density * math.log(density) if density > 0 else 0.0

    expected = (
        normal_quad(mixture_entropy_density, mean_p, mean_q)
        - 0.5 * first.entropy()
        - 0.5 * second.entropy()
    )
    value = closed_forms.normal_js(mean_p, std_p, mean_q, std_q)
    assert value == pytest.approx(float(expected), rel=1e-8)
    assert 0 < value <= math.log(2)


def test_normal_js_approaches_ln_2_for_separated_normals() -> None:
    """Normals without overlap have the largest divergence, ln 2 nats."""
    assert closed_forms.normal_js(0.0, 1.0, 80.0, 1.0) == pytest.approx(
        math.log(2), rel=1e-10
    )
    assert closed_forms.normal_js(0.5, 1.0, 0.5, 1.0) == 0.0


@pytest.mark.parametrize(("mean_p", "std_p", "mean_q", "std_q"), NORMAL_PAIRS[:3])
def test_normal_anderson_darling_matches_the_integral_over_h(
    mean_p: float, std_p: float, mean_q: float, std_q: float
) -> None:
    """The distance equals ∫₀¹ (F − G)²/(u(1 − u)) du with u = H(x)."""
    first, second = stats.norm(mean_p, std_p), stats.norm(mean_q, std_q)

    def integrand(level: float) -> float:
        point = optimize.brentq(
            lambda value: 0.5 * (first.cdf(value) + second.cdf(value)) - level,
            -80,
            80,
            xtol=1e-14,
        )
        return float(
            (first.cdf(point) - second.cdf(point)) ** 2 / (level * (1 - level))
        )

    expected = integrate.quad(integrand, 1e-12, 1 - 1e-12, limit=400)[0]
    assert closed_forms.normal_anderson_darling(
        mean_p, std_p, mean_q, std_q
    ) == pytest.approx(expected, rel=1e-7)


def test_normal_anderson_darling_vanishes_for_identical_normals() -> None:
    """Identical inputs give exactly zero without quadrature."""
    assert closed_forms.normal_anderson_darling(1.0, 2.0, 1.0, 2.0) == 0.0


def test_normal_mixture_kl_satisfies_the_dropped_mode_identity() -> None:
    """For equal weights, KL(m ‖ c₁) = δ²/4 − KL(c₁ ‖ m)."""
    separation = 3.0
    mixture = (np.array([0.5, 0.5]), np.array([-1.5, 1.5]), np.ones(2))
    component = (np.ones(1), np.array([-1.5]), np.ones(1))
    forward = closed_forms.normal_mixture_kl(*mixture, *component)
    backward = closed_forms.normal_mixture_kl(*component, *mixture)
    assert forward == pytest.approx(separation**2 / 4 - backward, rel=1e-10)
    assert 0 < backward < math.log(2)


def test_normal_mixture_kl_of_single_normals_is_the_gaussian_kl() -> None:
    """Mixtures of one component reduce to the closed-form Gaussian divergence."""
    first = (np.ones(1), np.array([0.3]), np.array([0.7]))
    second = (np.ones(1), np.array([-0.4]), np.array([1.9]))
    value = closed_forms.normal_mixture_kl(*first, *second)
    expected = closed_forms.gaussian_kl(
        np.array([0.3]), np.array([[0.49]]), np.array([-0.4]), np.array([[3.61]])
    ).total
    assert value == pytest.approx(expected, rel=1e-10)


def quadpack_with_breaks(
    integrand: Callable[[float], float], centres: list[float], widths: list[float]
) -> float:
    """Integrate with QUADPACK between break points at every centre ± k widths."""
    breaks = sorted(
        {
            centre + offset * width
            for centre, width in zip(centres, widths, strict=True)
            for offset in range(-12, 13)
        }
    )
    return math.fsum(
        integrate.quad(integrand, low, high, limit=200, epsabs=1e-15, epsrel=1e-13)[0]
        for low, high in itertools.pairwise(breaks)
    )


def test_quadrature_resolves_components_of_very_different_widths() -> None:
    """A narrow normal far from a wide one keeps its share of the integral.

    Without break points, QUADPACK steps over the narrow spike: it gave ½ ln 2
    for the first divergence below and 0.92 for the second.
    """
    first, second = stats.norm(0.0, 1.0), stats.norm(4.0, 1e-3)

    def js_integrand(point: float) -> float:
        log_p, log_q = first.logpdf(point), second.logpdf(point)
        log_mixture = np.logaddexp(log_p, log_q) - math.log(2)
        return float(
            0.5 * math.exp(log_p) * (log_p - log_mixture)
            + 0.5 * math.exp(log_q) * (log_q - log_mixture)
        )

    expected = quadpack_with_breaks(js_integrand, [0.0, 4.0], [1.0, 1e-3])
    assert closed_forms.normal_js(0.0, 1.0, 4.0, 1e-3) == pytest.approx(
        expected, rel=1e-10
    )
    # * Fully separated normals reach the exact limit of the Anderson-Darling integral.
    assert closed_forms.normal_anderson_darling(0.0, 1.0, 200.0, 1e-3) == pytest.approx(
        8 * math.log(2) - 4, rel=1e-10
    )


def test_quadrature_resolves_the_switch_between_separated_modes() -> None:
    """Panels at the crossing of two modes resolve where ln q changes component."""
    centres = np.array([-20.0, 20.0])
    weights = np.array([0.99, 0.01])
    matched_std = math.sqrt(1 + 400.0)

    def integrand(point: float) -> float:
        log_p = float(stats.norm.logpdf(point, scale=matched_std))
        log_q = float(
            np.logaddexp.reduce(np.log(weights) + stats.norm.logpdf(point, centres))
        )
        return math.exp(log_p) * (log_p - log_q)

    crossing = (math.log(0.01) - math.log(0.99)) / -40.0
    expected = quadpack_with_breaks(
        integrand, [0.0, -20.0, 20.0, crossing], [matched_std, 1.0, 1.0, 0.025]
    )
    value = closed_forms.normal_mixture_kl(
        np.ones(1), np.zeros(1), np.array([matched_std]), weights, centres, np.ones(2)
    )
    assert value == pytest.approx(expected, rel=1e-9)


# * JAX transformations


def test_closed_forms_compile_and_map() -> None:
    """Compiled closed forms give the eager values, and vmap maps them over a batch."""
    shifts = jnp.linspace(0.1, 3.0, 5)
    mapped = jax.vmap(closed_forms.normal_js, in_axes=(None, None, 0, None))(
        0.0, 1.0, shifts, 1.0
    )
    looped = [closed_forms.normal_js(0.0, 1.0, float(shift), 1.0) for shift in shifts]
    np.testing.assert_allclose(mapped, looped, rtol=1e-12)
    compiled = jax.jit(closed_forms.gaussian_hellinger)
    arguments = (np.zeros(2), np.eye(2), np.ones(2), 2 * np.eye(2))
    np.testing.assert_allclose(
        compiled(*arguments), closed_forms.gaussian_hellinger(*arguments), rtol=1e-12
    )


def test_closed_forms_differentiate() -> None:
    """Gradients match the analytic location gradient and finite differences."""
    covariance = np.array([[2.0, 0.3], [0.3, 1.0]])
    mean = np.array([0.5, -1.0])
    gradient = jax.grad(
        lambda mean_q: (
            closed_forms.gaussian_kl(
                np.zeros(2), covariance, mean_q, covariance
            ).location
        )
    )(mean)
    np.testing.assert_allclose(gradient, np.linalg.solve(covariance, mean), rtol=1e-12)
    step = 1e-5
    for function in (closed_forms.normal_js, closed_forms.normal_hellinger):
        slope = jax.grad(
            lambda shift, function=function: function(0.0, 1.0, shift, 1.3)
        )(1.5)
        difference = (
            function(0.0, 1.0, 1.5 + step, 1.3) - function(0.0, 1.0, 1.5 - step, 1.3)
        ) / (2 * step)
        assert float(slope) == pytest.approx(float(difference), rel=1e-6)


def test_invalid_inputs_give_nan_inside_jit() -> None:
    """Inside jit the values are unknown, so invalid inputs give NaN, not an error."""
    total = jax.jit(
        lambda covariance: (
            closed_forms.gaussian_kl(
                np.zeros(2), covariance, np.zeros(2), np.eye(2)
            ).total
        )
    )(-np.eye(2))
    assert math.isnan(float(total))


# * Twisted Gaussian


def test_twisted_gaussian_kl_matches_two_dimensional_quadrature(
    grid_integral: GridIntegral,
) -> None:
    """Both closed-form divergences agree with direct integration in 2D."""
    twist, variance = 0.5, 1.0
    _, covariance = closed_forms.twisted_gaussian_moments(twist, variance, 2)
    matched = stats.multivariate_normal(np.zeros(2), covariance)

    def log_banana(points: np.ndarray) -> np.ndarray:
        untwisted = points[..., 1] - twist * (points[..., 0] ** 2 - variance)
        return np.asarray(
            stats.norm.logpdf(points[..., 0], scale=math.sqrt(variance))
            + stats.norm.logpdf(untwisted)
        )

    def banana_term(points: np.ndarray) -> np.ndarray:
        log_density = log_banana(points)
        return np.asarray(np.exp(log_density) * (log_density - matched.logpdf(points)))

    def gaussian_term(points: np.ndarray) -> np.ndarray:
        log_density = matched.logpdf(points)
        return np.asarray(np.exp(log_density) * (log_density - log_banana(points)))

    forward = grid_integral(banana_term, (-9.0, 9.0, 72), (-12.0, 45.0, 228))
    backward = grid_integral(gaussian_term, (-9.0, 9.0, 72), (-12.0, 12.0, 96))
    expected_forward, expected_backward = closed_forms.twisted_gaussian_kl(
        twist, variance
    )
    assert expected_forward == pytest.approx(forward, rel=1e-10)
    assert expected_backward == pytest.approx(backward, rel=1e-10)


def test_twisted_gaussian_without_twist_is_its_own_gaussian() -> None:
    """A zero twist gives the base Gaussian, with zero divergence both ways."""
    forward, backward = closed_forms.twisted_gaussian_kl(0.0, 100.0)
    assert (float(forward), float(backward)) == (0.0, 0.0)
    mean, covariance = closed_forms.twisted_gaussian_moments(0.0, 4.0, 3)
    np.testing.assert_array_equal(mean, 0.0)
    np.testing.assert_array_equal(covariance, np.diag([4.0, 1.0, 1.0]))


def test_twisted_gaussian_defaults_match_the_moderately_twisted_target() -> None:
    """B(0.03, 100) has the variance 19 on its second coordinate and KL ½ ln 19."""
    _, covariance = closed_forms.twisted_gaussian_moments(0.03, 100.0, 8)
    assert covariance[1, 1] == pytest.approx(19.0)
    forward, backward = closed_forms.twisted_gaussian_kl(0.03, 100.0)
    assert forward == pytest.approx(0.5 * math.log(19))
    assert backward == pytest.approx(18 - 0.5 * math.log(19))


# * Truncated normals and railing


@pytest.mark.parametrize(
    ("mean", "std", "lower", "upper"),
    [
        (1.5, 1.0, 0.0, math.inf),
        (-2.0, 1.0, 0.0, math.inf),
        (0.0, 2.0, -math.inf, -9.0),
        (0.3, 0.5, -1.0, 0.2),
        (-12.0, 1.0, 0.0, math.inf),
    ],
)
def test_truncated_normal_moments_match_scipy(
    mean: float, std: float, lower: float, upper: float
) -> None:
    """Mean and width agree with scipy.stats.truncnorm, also far in a tail."""
    reference = stats.truncnorm(
        (lower - mean) / std, (upper - mean) / std, loc=mean, scale=std
    )
    truncated_mean, truncated_std = closed_forms.truncated_normal_moments(
        mean, std, lower, upper
    )
    assert truncated_mean == pytest.approx(reference.mean(), rel=1e-10)
    assert truncated_std == pytest.approx(reference.std(), rel=1e-7)


def test_truncated_normal_kl_matches_quadrature() -> None:
    """KL between two railed normals equals the integral of p ln(p/q) on the support."""
    first = stats.truncnorm(-1.5, np.inf, loc=1.5)
    second = stats.truncnorm(0.5, np.inf, loc=-0.5)
    expected = integrate.quad(
        lambda point: first.pdf(point) * (first.logpdf(point) - second.logpdf(point)),
        0,
        40,
        limit=400,
    )[0]
    assert closed_forms.truncated_normal_kl(
        1.5, -0.5, 1.0, 0.0, math.inf
    ) == pytest.approx(expected, rel=1e-9)
    assert closed_forms.truncated_normal_kl(1.5, 1.5, 1.0, 0.0, math.inf) == 0.0


TRUNCATED_PAIRS = [
    (3.0, 0.5, 0.0, math.inf),
    (0.0, -1.5, 0.0, math.inf),
    (-2.0, 0.5, -math.inf, 0.0),
    (1.0, 1.4, 0.0, 3.0),
]


def truncated_pair(
    mean_p: float, mean_q: float, lower: float, upper: float
) -> tuple[Any, Any, float, float]:
    """Return SciPy's two unit-width truncated normals and a finite span of both."""
    first = stats.truncnorm(lower - mean_p, upper - mean_p, loc=mean_p)
    second = stats.truncnorm(lower - mean_q, upper - mean_q, loc=mean_q)
    low = lower if math.isfinite(lower) else min(mean_p, mean_q) - 15
    high = upper if math.isfinite(upper) else max(mean_p, mean_q) + 15
    return first, second, low, high


@pytest.mark.parametrize(("mean_p", "mean_q", "lower", "upper"), TRUNCATED_PAIRS)
def test_truncated_normal_marginals_match_scipy(
    mean_p: float, mean_q: float, lower: float, upper: float
) -> None:
    """KS, W1, Hellinger, JS and AD of railed normals agree with SciPy integrals."""
    first, second, low, high = truncated_pair(mean_p, mean_q, lower, upper)
    breaks = sorted({mean for mean in (mean_p, mean_q) if low < mean < high})

    def integral(function: Callable[[float], float]) -> float:
        return float(
            integrate.quad(
                function, low, high, points=breaks or None, limit=500, epsabs=1e-13
            )[0]
        )

    def cdf_gap(point: float) -> float:
        return float(first.cdf(point) - second.cdf(point))

    widest = optimize.minimize_scalar(
        lambda point: -abs(cdf_gap(point)),
        bounds=(low, high),
        method="bounded",
        options={"xatol": 1e-12},
    )
    expected = {
        "ks": -widest.fun,
        "w1": integral(lambda point: abs(cdf_gap(point))),
        "hellinger": math.sqrt(
            1 - integral(lambda point: math.sqrt(first.pdf(point) * second.pdf(point)))
        ),
        "js": integral(
            lambda point: entropy_gap(float(first.pdf(point)), float(second.pdf(point)))
        ),
        "ad": integral(
            lambda point: anderson_darling_density(
                first.cdf(point), second.cdf(point), first.pdf(point), second.pdf(point)
            )
        ),
    }
    functions = {
        "ks": closed_forms.truncated_normal_ks,
        "w1": closed_forms.truncated_normal_w1,
        "hellinger": closed_forms.truncated_normal_hellinger,
        "js": closed_forms.truncated_normal_js,
        "ad": closed_forms.truncated_normal_anderson_darling,
    }
    for name, function in functions.items():
        value = function(mean_p, mean_q, 1.0, lower, upper)
        assert value == pytest.approx(expected[name], rel=1e-8), name


def test_truncated_normal_marginals_vanish_for_equal_means() -> None:
    """Two identical railed normals give exact zeros."""
    for function in (
        closed_forms.truncated_normal_ks,
        closed_forms.truncated_normal_w1,
        closed_forms.truncated_normal_js,
        closed_forms.truncated_normal_anderson_darling,
    ):
        assert function(0.7, 0.7, 1.0, 0.0, math.inf) == 0.0
    assert closed_forms.truncated_normal_hellinger(
        0.7, 0.7, 1.0, 0.0, math.inf
    ) == pytest.approx(0.0, abs=1e-7)


def entropy_gap(density_p: float, density_q: float) -> float:
    """Return ½ p ln(2p/(p + q)) + ½ q ln(2q/(p + q)), the JS integrand."""
    return sum(
        0.5 * own * math.log(2 * own / (density_p + density_q))
        for own in (density_p, density_q)
        if own > 0
    )


def anderson_darling_density(
    cdf_p: float, cdf_q: float, density_p: float, density_q: float
) -> float:
    """Return (F − G)²/(H(1 − H)) dH/dx with H = (F + G)/2, zero where H(1 − H) is."""
    pooled = 0.5 * (cdf_p + cdf_q)
    if not 0 < pooled < 1:
        return 0.0
    return float(
        (cdf_p - cdf_q) ** 2 / (pooled * (1 - pooled)) * 0.5 * (density_p + density_q)
    )


def test_log_normal_mass_keeps_its_precision_in_both_tails() -> None:
    """Interval masses agree with log Φ in each tail and with direct sums near zero."""
    assert closed_forms.log_normal_mass(-math.inf, -30.0) == pytest.approx(
        stats.norm.logcdf(-30.0), rel=1e-12
    )
    assert closed_forms.log_normal_mass(30.0, math.inf) == pytest.approx(
        stats.norm.logsf(30.0), rel=1e-12
    )
    assert closed_forms.log_normal_mass(-0.5, 1.0) == pytest.approx(
        math.log(stats.norm.cdf(1.0) - stats.norm.cdf(-0.5)), rel=1e-12
    )


def test_railing_score_is_the_density_ratio_at_the_bound() -> None:
    """The score is exp(−z²/2) for a peak inside the support and 1 at or beyond it."""
    assert closed_forms.railing_score(2.0) == pytest.approx(math.exp(-2.0))
    assert closed_forms.railing_score(0.0) == 1.0
    assert closed_forms.railing_score(-3.0) == 1.0
    with pytest.raises(ValueError, match="finite"):
        closed_forms.railing_score(math.inf)


# * Circles and spheres


def test_wrapped_normal_resultant_length_matches_integration() -> None:
    """R = E[cos(2π(θ − μ)/P)] for the wrapped density, with σ in units of P/(2π)."""
    std, period = 0.9, math.pi

    def wrapped_density(angle: float) -> float:
        return float(
            sum(
                stats.norm.pdf(angle + winding * period, scale=std)
                for winding in range(-8, 9)
            )
        )

    expected = integrate.quad(
        lambda angle: wrapped_density(angle) * math.cos(2 * math.pi * angle / period),
        -period / 2,
        period / 2,
        epsabs=1e-13,
    )[0]
    assert closed_forms.wrapped_normal_resultant_length(std, period) == pytest.approx(
        expected, rel=1e-10
    )


WRAPPED_PAIRS = [
    (0.0, 0.5, 0.5, 2 * math.pi),
    (-1.0, 0.142, 0.4, math.pi),
    (0.0, 2.0, 1.5, 2 * math.pi),
    (6.0, 0.3, 0.8, 2 * math.pi),
]


def wrapped_terms(
    mean: float, std: float, period: float, start: float
) -> tuple[Callable[[ArrayLike], np.ndarray], Callable[[ArrayLike], np.ndarray]]:
    """Return SciPy's wrapped density and distribution function from ``start``."""
    windings = np.arange(-20, 21) * period

    def density(points: ArrayLike) -> np.ndarray:
        values = np.atleast_1d(points)[:, None] - mean + windings
        return np.asarray(np.sum(stats.norm.pdf(values / std), axis=1) / std)

    def cdf(points: ArrayLike) -> np.ndarray:
        values = np.atleast_1d(points)[:, None] - mean + windings
        offsets = start - mean + windings
        return np.asarray(
            np.sum(stats.norm.cdf(values / std) - stats.norm.cdf(offsets / std), axis=1)
        )

    return density, cdf


@pytest.mark.parametrize(("mean_p", "mean_q", "std", "period"), WRAPPED_PAIRS)
def test_wrapped_normal_marginals_match_scipy(
    mean_p: float, mean_q: float, std: float, period: float
) -> None:
    """Kuiper's V, the circle W1, AD after the cut, JS and Hellinger agree with SciPy.

    V and W1 come from F − G on a dense grid of the circle, W1 as the integral of
    |F − G − c| at the median c, the others from adaptive quadrature over the
    period that the cut opposite the midpoint of the means leaves.
    """
    grid = np.linspace(0.0, period, 200_000, endpoint=False)
    _, cdf_p = wrapped_terms(mean_p, std, period, 0.0)
    _, cdf_q = wrapped_terms(mean_q, std, period, 0.0)
    gap = cdf_p(grid) - cdf_q(grid)
    arc = (mean_q - mean_p + 0.5 * period) % period - 0.5 * period
    middle = mean_p + 0.5 * arc
    start, end = middle - 0.5 * period, middle + 0.5 * period
    density_p, cut_cdf_p = wrapped_terms(mean_p, std, period, start)
    density_q, cut_cdf_q = wrapped_terms(mean_q, std, period, start)

    def integral(function: Callable[[float], float]) -> float:
        return float(
            integrate.quad(
                function,
                start,
                end,
                points=[middle - 0.5 * arc, middle + 0.5 * arc],
                limit=500,
                epsabs=1e-13,
            )[0]
        )

    def densities(point: float) -> tuple[float, float]:
        return float(density_p(point)[0]), float(density_q(point)[0])

    expected = {
        "kuiper": gap.max() - gap.min(),
        "w1": float(np.mean(np.abs(gap - np.median(gap))) * period),
        "ad": integral(
            lambda point: anderson_darling_density(
                float(cut_cdf_p(point)[0]),
                float(cut_cdf_q(point)[0]),
                *densities(point),
            )
        ),
        "js": integral(lambda point: entropy_gap(*densities(point))),
        "hellinger": math.sqrt(
            integral(
                lambda point: (
                    0.5
                    * (math.sqrt(densities(point)[0]) - math.sqrt(densities(point)[1]))
                    ** 2
                )
            )
        ),
    }
    functions = {
        "kuiper": closed_forms.wrapped_normal_kuiper,
        "w1": closed_forms.wrapped_normal_w1,
        "ad": closed_forms.wrapped_normal_anderson_darling,
        "js": closed_forms.wrapped_normal_js,
        "hellinger": closed_forms.wrapped_normal_hellinger,
    }
    for name, function in functions.items():
        value = function(mean_p, mean_q, std, period)
        assert value == pytest.approx(expected[name], rel=1e-8), name


def test_narrow_wrapped_normals_give_the_line_values() -> None:
    """Far narrower than the period, the circle forms equal those of two normals."""
    shift, std, period = 0.08, 0.05, 2 * math.pi
    assert closed_forms.wrapped_normal_kuiper(0.0, shift, std, period) == (
        pytest.approx(closed_forms.normal_ks(0.0, std, shift, std), rel=1e-12)
    )
    assert closed_forms.wrapped_normal_w1(0.0, shift, std, period) == pytest.approx(
        shift, rel=1e-12
    )
    for circle, line in (
        (
            closed_forms.wrapped_normal_anderson_darling,
            closed_forms.normal_anderson_darling,
        ),
        (closed_forms.wrapped_normal_js, closed_forms.normal_js),
        (closed_forms.wrapped_normal_hellinger, closed_forms.normal_hellinger),
    ):
        assert circle(0.0, shift, std, period) == pytest.approx(
            line(0.0, std, shift, std), rel=1e-10
        )


def test_wrapped_normal_marginals_ignore_the_cut_and_the_winding() -> None:
    """Turning both means, or moving one by a period, changes no value."""
    functions = (
        closed_forms.wrapped_normal_kuiper,
        closed_forms.wrapped_normal_w1,
        closed_forms.wrapped_normal_anderson_darling,
        closed_forms.wrapped_normal_js,
        closed_forms.wrapped_normal_hellinger,
    )
    for function in functions:
        value = function(0.2, 1.1, 0.6, 2 * math.pi)
        assert function(2.7, 3.6, 0.6, 2 * math.pi) == pytest.approx(value, rel=1e-9)
        assert function(0.2, 1.1 - 4 * math.pi, 0.6, 2 * math.pi) == pytest.approx(
            value, rel=1e-9
        )
        assert function(0.4, 0.4, 0.6, 2 * math.pi) == 0.0


def test_wrapped_normal_marginals_inside_jit() -> None:
    """Inside jit the windings are fixed: up to a period, wider widths give NaN."""
    compiled = jax.jit(closed_forms.wrapped_normal_w1)
    assert compiled(0.0, 1.0, 0.9, 1.0) == pytest.approx(
        closed_forms.wrapped_normal_w1(0.0, 1.0, 0.9, 1.0), rel=1e-12
    )
    assert math.isnan(float(compiled(0.0, 0.3, 2.5, 1.0)))
    assert not math.isnan(float(closed_forms.wrapped_normal_w1(0.0, 0.3, 2.5, 1.0)))


@pytest.mark.parametrize("concentration", [0.0, 1e-3, 0.009, 0.011, 2.0, 400.0])
def test_von_mises_fisher_resultant_length_matches_integration(
    concentration: float,
) -> None:
    """E[w] for the density κ e^(κw)/(2 sinh κ) on [−1, 1], across the series switch."""
    if concentration == 0.0:
        expected = 0.0
    else:
        log_normaliser = (
            math.log(concentration)
            - math.log(2)
            - (concentration + math.log1p(-math.exp(-2 * concentration)) - math.log(2))
        )
        expected = integrate.quad(
            lambda cosine: cosine * math.exp(log_normaliser + concentration * cosine),
            -1,
            1,
            epsabs=1e-14,
            limit=200,
        )[0]
    assert closed_forms.von_mises_fisher_resultant_length(
        concentration
    ) == pytest.approx(expected, rel=1e-9, abs=1e-15)


def test_von_mises_fisher_divergences_match_monte_carlo() -> None:
    """KL and Hellinger agree with Monte Carlo draws from SciPy's sampler."""
    direction_p = np.array([math.sin(0.1), 0.0, math.cos(0.1)])
    direction_q = np.array([math.sin(-0.15), 0.0, math.cos(-0.15)])
    first = stats.vonmises_fisher(direction_p, 80.0)
    second = stats.vonmises_fisher(direction_q, 50.0)
    draws = first.rvs(400_000, random_state=np.random.default_rng(4))
    log_ratio = first.logpdf(draws) - second.logpdf(draws)
    kl = closed_forms.von_mises_fisher_kl(direction_p, 80.0, direction_q, 50.0)
    standard_error = log_ratio.std() / math.sqrt(len(log_ratio))
    assert abs(kl - log_ratio.mean()) < STANDARD_ERRORS * standard_error
    root_ratio = np.exp(-0.5 * log_ratio)
    hellinger_squared = (
        closed_forms.von_mises_fisher_hellinger(direction_p, 80.0, direction_q, 50.0)
        ** 2
    )
    root_error = root_ratio.std() / math.sqrt(len(root_ratio))
    assert (
        abs(hellinger_squared - (1 - root_ratio.mean())) < STANDARD_ERRORS * root_error
    )


def test_von_mises_fisher_divergences_vanish_for_equal_inputs() -> None:
    """Identical distributions have zero KL and Hellinger distance."""
    direction = np.array([0.0, 0.6, 0.8])
    assert closed_forms.von_mises_fisher_kl(
        direction, 10.0, direction, 10.0
    ) == pytest.approx(0.0, abs=1e-12)
    assert closed_forms.von_mises_fisher_hellinger(
        direction, 10.0, direction, 10.0
    ) == pytest.approx(0.0, abs=1e-7)


def test_von_mises_fisher_hellinger_handles_opposite_directions() -> None:
    """Antipodal equal concentrations meet at the uniform normaliser, κ = 0."""
    up, down = np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, -1.0])
    log_normaliser = closed_forms.von_mises_fisher_log_normaliser(5.0)
    expected = math.sqrt(1 - math.exp(log_normaliser + math.log(4 * math.pi)))
    assert closed_forms.von_mises_fisher_hellinger(up, 5.0, down, 5.0) == pytest.approx(
        expected, rel=1e-12
    )


# * AR(1) chains


@pytest.mark.parametrize("coefficient", [0.0, 0.5, 0.98, -0.6])
def test_ar1_autocorrelation_time_sums_the_autocorrelations(coefficient: float) -> None:
    """τ = Σ_t φ^|t| over all lags, which Thompson (2010) gives as 99 at φ = 0.98."""
    lags = np.arange(1, 20_000)
    expected = 1 + 2 * np.sum(coefficient**lags)
    assert closed_forms.ar1_autocorrelation_time(coefficient) == pytest.approx(
        expected, rel=1e-12
    )
    if coefficient == 0.98:
        assert closed_forms.ar1_autocorrelation_time(coefficient) == pytest.approx(99.0)


@pytest.mark.parametrize("coefficient", [1.0, -1.0, 1.5])
def test_ar1_autocorrelation_time_rejects_non_stationary_coefficients(
    coefficient: float,
) -> None:
    """Coefficients outside (−1, 1) give no stationary chain."""
    with pytest.raises(ValueError, match=r"\(-1, 1\)"):
        closed_forms.ar1_autocorrelation_time(coefficient)


# * Input checks


def test_gaussian_inputs_are_checked() -> None:
    """Shapes, symmetry and positive definiteness fail with clear messages."""
    # * The jaxtyping shape check of the test hook catches mismatched shapes first.
    with pytest.raises(TypeError):
        closed_forms.gaussian_kl(np.zeros(2), np.eye(3), np.zeros(2), np.eye(2))
    with pytest.raises(ValueError, match="symmetric"):
        closed_forms.gaussian_w2(
            np.zeros(2), np.array([[1.0, 0.5], [0.0, 1.0]]), np.zeros(2), np.eye(2)
        )
    with pytest.raises(ValueError, match="positive definite"):
        closed_forms.gaussian_hellinger(
            np.zeros(2), np.zeros((2, 2)), np.zeros(2), np.eye(2)
        )
    with pytest.raises(ValueError, match="finite"):
        closed_forms.mahalanobis_shift(np.array([np.nan, 0.0]), np.eye(2), np.zeros(2))


def test_normal_inputs_are_checked() -> None:
    """Widths must be positive and means finite."""
    with pytest.raises(ValueError, match="std_q must be finite and positive"):
        closed_forms.normal_ks(0.0, 1.0, 0.0, -1.0)
    with pytest.raises(ValueError, match="mean_p must be finite"):
        closed_forms.normal_w1(math.inf, 1.0, 0.0, 1.0)
    with pytest.raises(ValueError, match="positive"):
        closed_forms.normal_mixture_kl(
            np.array([1.0, -1.0]),
            np.zeros(2),
            np.ones(2),
            np.ones(1),
            np.zeros(1),
            np.ones(1),
        )


def test_other_inputs_are_checked() -> None:
    """Twist, truncation, period, concentration and directions are validated."""
    with pytest.raises(ValueError, match="at least 2 parameters"):
        closed_forms.twisted_gaussian_moments(0.1, 1.0, 1)
    with pytest.raises(ValueError, match="twist must be finite"):
        closed_forms.twisted_gaussian_kl(math.nan, 1.0)
    with pytest.raises(ValueError, match="the mean must be finite"):
        closed_forms.truncated_normal_moments(math.inf, 1.0, 0.0, 1.0)
    with pytest.raises(ValueError, match="lower < upper"):
        closed_forms.truncated_normal_moments(0.0, 1.0, 1.0, 0.0)
    with pytest.raises(ValueError, match="period must be finite and positive"):
        closed_forms.wrapped_normal_resultant_length(1.0, 0.0)
    with pytest.raises(ValueError, match="non-negative"):
        closed_forms.von_mises_fisher_resultant_length(-1.0)
    with pytest.raises(ValueError, match="non-negative"):
        closed_forms.von_mises_fisher_log_normaliser(math.nan)
    with pytest.raises(ValueError, match="zero vector"):
        closed_forms.von_mises_fisher_kl(np.zeros(3), 1.0, np.ones(3), 1.0)
    with pytest.raises(ValueError, match="mean_q must be finite"):
        closed_forms.truncated_normal_ks(0.0, math.nan, 1.0, 0.0, math.inf)
    with pytest.raises(ValueError, match="std must be finite and positive"):
        closed_forms.wrapped_normal_kuiper(0.0, 1.0, 0.0, 1.0)
    with pytest.raises(ValueError, match="mean_p must be finite"):
        closed_forms.wrapped_normal_js(math.inf, 1.0, 0.5, 1.0)
