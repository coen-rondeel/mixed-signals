"""Tests for the bounded kernel density estimator."""

import math
import warnings
from collections.abc import Callable, Mapping

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import integrate, optimize, special, stats

from mixed_signals import zoo
from mixed_signals.core.space import ParamSpace
from mixed_signals.errors import ValidityWarning
from mixed_signals.estimators import (
    MINIMUM_SAMPLE_SIZES,
    KernelDensity,
    KernelDensitySettings,
)

type GridIntegral = Callable[..., float]

STANDARD_ERRORS = 5.0
TWO_PI = 2 * math.pi


def space_of(*entries: Mapping[str, object]) -> ParamSpace:
    """Return a space from parameter entries."""
    return ParamSpace.model_validate({"parameters": list(entries)})


def silverman_factor(effective_sample_size: float, size: int) -> float:
    """Return Silverman's bandwidth factor (n (D + 2)/4)^(−1/(D+4))."""
    return float((effective_sample_size * (size + 2) / 4) ** (-1 / (size + 4)))


def scipy_estimate(samples: np.ndarray, factor: float) -> stats.gaussian_kde:
    """Return SciPy's estimate of samples shaped (N, D) with a fixed factor."""
    return stats.gaussian_kde(samples.T, bw_method=factor)


def density_of(fitted: KernelDensity) -> Callable[[np.ndarray], np.ndarray]:
    """Return the density of a fitted estimate as a NumPy function."""
    return lambda points: np.exp(np.asarray(fitted.log_density(points)))


# * Unbounded estimates


@pytest.mark.parametrize("size", [1, 2])
def test_unbounded_estimate_matches_scipy(size: int) -> None:
    """Without bounds the estimate is SciPy's, with Silverman's rule and weights."""
    rng = np.random.default_rng(0)
    samples = rng.normal(size=(20_000, size)) @ np.triu(np.full((size, size), 0.6))
    samples += rng.normal(size=(20_000, size))
    log_weights = rng.normal(scale=0.3, size=20_000)
    space = space_of(*({"name": f"x{index}"} for index in range(size)))
    fitted = KernelDensity.fit(samples, log_weights, space)
    reference = stats.gaussian_kde(
        samples.T, bw_method="silverman", weights=np.exp(log_weights)
    )
    points = samples[:1_000] + 0.1
    np.testing.assert_allclose(
        fitted.log_density(points), reference.logpdf(points.T), rtol=1e-10
    )


def test_bandwidth_factor_follows_silverman_on_the_kish_size() -> None:
    """The factor is Silverman's at the Kish size, and its settings reproduce it."""
    rng = np.random.default_rng(1)
    samples = rng.normal(size=(500, 2))
    log_weights = rng.normal(size=500)
    space = space_of({"name": "x0"}, {"name": "x1"})
    fitted = KernelDensity.fit(samples, log_weights, space)
    weights = np.exp(log_weights) / np.exp(log_weights).sum()
    expected = silverman_factor(1 / np.sum(weights**2), 2)
    resolved = fitted.resolved_settings()
    assert resolved.bandwidth == pytest.approx(expected, rel=1e-12)
    refitted = KernelDensity.fit(samples, log_weights, space, resolved)
    np.testing.assert_allclose(
        refitted.log_density(samples[:5]), fitted.log_density(samples[:5]), rtol=1e-12
    )
    assert fitted.settings.to_json() == '{"bandwidth":"silverman"}'


def test_a_float_bandwidth_replaces_the_rule() -> None:
    """A float factor scales the covariance of the samples, as in SciPy."""
    samples = np.random.default_rng(2).normal(size=(400, 1))
    fitted = KernelDensity.fit(
        samples, None, space_of({"name": "x"}), KernelDensitySettings(bandwidth=0.3)
    )
    reference = scipy_estimate(samples, 0.3)
    np.testing.assert_allclose(
        fitted.log_density(samples[:5]), reference.logpdf(samples[:5].T), rtol=1e-12
    )
    assert float(fitted.factor) == 0.3


def test_estimate_is_unbiased_for_the_smoothed_density() -> None:
    """At fixed points the estimate scatters about the kernel-smoothed density."""
    case = zoo.gaussians([[0.7]], [[[2.25]]], names=["only"])
    sample_count = 20_000
    samples = case.sample(jax.random.key(3), sample_count).samples[0, 0]
    fitted = KernelDensity.fit(samples, None, case.space)
    bandwidth = float(jnp.sqrt(fitted.kde.covariance[0, 0]))
    points = np.array([0.7 - 3.0, 0.7, 0.7 + 2.2])
    expected = stats.norm.pdf(points, 0.7, math.sqrt(2.25 + bandwidth**2))
    second = stats.norm.pdf(points, 0.7, math.sqrt(2.25 + bandwidth**2 / 2)) / (
        2 * math.sqrt(math.pi) * bandwidth
    )
    error = np.sqrt((second - expected**2) / sample_count)
    estimate = np.exp(np.asarray(fitted.log_density(points[:, None])))
    assert np.all(np.abs(estimate - expected) < STANDARD_ERRORS * error)


# * Circles


def test_circle_estimate_sums_the_kernels_over_windings() -> None:
    """On a circle it is SciPy's estimate of the unwrapped samples, over windings."""
    case = zoo.wrapped_gaussians(0.0, width=0.7, param_count=2, correlation=0.4)
    samples = np.asarray(case.sample(jax.random.key(4), 3_000).samples[0, 0])
    fitted = KernelDensity.fit(samples, None, case.space)
    centre = stats.circmean(samples[:, 0])
    unwrapped = samples.copy()
    unwrapped[:, 0] = centre + (samples[:, 0] - centre + math.pi) % TWO_PI - math.pi
    reference = scipy_estimate(unwrapped, float(fitted.factor))
    points = samples[:6] + np.array([0.05, 0.1])
    images = points[None] + np.arange(-6, 7)[:, None, None] * [TWO_PI, 0.0]
    expected = np.log(np.sum([reference.pdf(image.T) for image in images], axis=0))
    np.testing.assert_allclose(fitted.log_density(points), expected, rtol=1e-10)


def test_circle_density_is_periodic_and_has_unit_mass(
    grid_integral: GridIntegral,
) -> None:
    """The density repeats with the period and integrates to one over it."""
    case = zoo.wrapped_gaussians(0.0, width=1.2, param_count=1)
    samples = case.sample(jax.random.key(5), 4_000).samples[0, 0]
    fitted = KernelDensity.fit(samples, None, case.space)
    points = np.array([[0.0], [1.0], [5.0]])
    np.testing.assert_allclose(
        fitted.log_density(points + TWO_PI), fitted.log_density(points), rtol=1e-12
    )
    assert grid_integral(density_of(fitted), (0.0, TWO_PI, 64)) == pytest.approx(
        1.0, abs=1e-10
    )


def test_circle_estimate_does_not_depend_on_where_the_circle_starts() -> None:
    """The same angles on [0, 2π) and on [−π, π) give the same density."""
    case = zoo.wrapped_gaussians(0.0, width=0.5, param_count=1)
    samples = np.asarray(case.sample(jax.random.key(6), 2_000).samples[0, 0])
    shifted_space = space_of(
        {"name": "x0", "topology": "circle", "bounds": [-math.pi, math.pi]}
    )
    shifted = (samples + math.pi) % TWO_PI - math.pi
    first = KernelDensity.fit(samples, None, case.space)
    second = KernelDensity.fit(shifted, None, shifted_space)
    points = np.array([[0.1], [3.0], [6.0]])
    np.testing.assert_allclose(
        first.log_density(points), second.log_density(points - TWO_PI), rtol=1e-10
    )


# * One-dimensional intervals: the Jones-Foster estimate


def jones_foster_on_half_line(
    points: np.ndarray, samples: np.ndarray, bandwidth: float
) -> np.ndarray:
    """The jf96 estimate of evmix on [0, ∞): renormalised, linear, then combined."""
    distance = points / bandwidth
    offsets = (points[:, None] - samples[None, :]) / bandwidth
    kernel = stats.norm.pdf(offsets)
    zeroth = stats.norm.cdf(distance)
    first = -stats.norm.pdf(distance)
    second = zeroth - distance * stats.norm.pdf(distance)
    denominator = second * zeroth - first**2
    linear = (
        np.mean(
            (
                second[:, None] / denominator[:, None]
                - first[:, None] / denominator[:, None] * offsets
            )
            * kernel,
            axis=1,
        )
        / bandwidth
    )
    renormalised = np.mean(kernel, axis=1) / (zeroth * bandwidth)
    # * As in evmix, a point beyond every kernel keeps the linear estimate, here zero.
    positive = renormalised > 0
    ratio = np.divide(linear, renormalised, out=np.ones_like(linear), where=positive)
    return np.where(positive, renormalised * np.exp(ratio - 1), linear)


def test_corrected_estimate_matches_the_evmix_formula() -> None:
    """On [0, ∞) the estimate is evmix's jf96 estimate, renormalised to unit mass."""
    case = zoo.railed_gaussians([1.0], param_count=1)
    samples = np.asarray(case.sample(jax.random.key(7), 2_000).samples[0, 0, :, 0])
    fitted = KernelDensity.fit(samples[:, None], None, case.space)
    bandwidth = float(jnp.sqrt(fitted.kde.covariance[0, 0]))

    def estimate(value: float) -> float:
        return float(
            jones_foster_on_half_line(np.array([value]), samples, bandwidth)[0]
        )

    near, _ = integrate.quad(estimate, 0.0, 8.0, points=[bandwidth, 1.0], limit=200)
    far, _ = integrate.quad(estimate, 8.0, np.inf, limit=200)
    mass = near + far
    points = np.array([0.0, 0.05, 0.3, 1.0, 2.5])
    expected = np.log(jones_foster_on_half_line(points, samples, bandwidth) / mass)
    np.testing.assert_allclose(fitted.log_density(points[:, None]), expected, rtol=1e-8)


@pytest.mark.parametrize(
    ("bounds", "draw"),
    [
        ((0.0, math.inf), lambda rng: np.abs(rng.normal(size=800))),
        ((-math.inf, 1.0), lambda rng: 1 - rng.exponential(size=800)),
        ((0.0, 1.0), lambda rng: rng.beta(2.0, 5.0, size=800)),
        ((0.0, 0.05), lambda rng: rng.uniform(0.0, 0.05, size=800)),
    ],
    ids=["lower", "upper", "both", "narrow"],
)
def test_corrected_estimate_integrates_to_one(
    bounds: tuple[float, float], draw: Callable[[np.random.Generator], np.ndarray]
) -> None:
    """Intervals with one bound, two, or narrower than a kernel have unit mass."""
    samples = draw(np.random.default_rng(8))
    space = space_of({"name": "x", "topology": "interval", "bounds": list(bounds)})
    fitted = KernelDensity.fit(samples[:, None], None, space)
    lower, upper = bounds
    finite_lower = lower if math.isfinite(lower) else float(samples.min()) - 20.0
    finite_upper = upper if math.isfinite(upper) else float(samples.max()) + 20.0
    mass, _ = integrate.quad(
        lambda value: math.exp(float(fitted.log_density(np.array([[value]]))[0])),
        finite_lower,
        finite_upper,
        limit=400,
    )
    assert mass == pytest.approx(1.0, abs=1e-6)
    outside = [[finite_lower - 1.0]] if math.isfinite(lower) else [[finite_upper + 1.0]]
    assert float(fitted.log_density(np.array(outside))[0]) == -math.inf


def test_corrected_estimate_is_the_plain_one_far_from_the_bound() -> None:
    """Beyond twelve bandwidths from the bound the correction vanishes."""
    samples = np.abs(np.random.default_rng(9).normal(size=1_000)) + 0.2
    space = space_of({"name": "x", "topology": "interval", "bounds": [0.0, math.inf]})
    fitted = KernelDensity.fit(samples[:, None], None, space)
    bandwidth = float(jnp.sqrt(fitted.kde.covariance[0, 0]))
    points = np.array([13 * bandwidth, 2.5])
    plain = scipy_estimate(samples[:, None], float(fitted.factor)).logpdf(points)
    np.testing.assert_allclose(
        fitted.log_density(points[:, None]) + fitted.log_normaliser, plain, rtol=1e-10
    )


def test_railing_score_is_recovered() -> None:
    """Density at the bound over the peak recovers exp(−z²/2), unlike reflection."""
    case = zoo.railed_gaussians([1.0], param_count=1)
    batch = case.sample(jax.random.key(10), 10_000, batch_size=20).samples[:, 0]
    grid = np.linspace(0.0, 4.0, 401)[:, None]
    scores = []
    for samples in batch:
        density = np.exp(
            np.asarray(KernelDensity.fit(samples, None, case.space).log_density(grid))
        )
        scores.append(density[0] / density.max())
    truth = case.reference_value("railing_score", "railed_0")
    # * Single scores scatter by 7 per cent, so the mean of 20 by 1.6 per cent.
    assert np.mean(scores) == pytest.approx(truth, rel=0.08)


# * Reflection in two or more dimensions


@pytest.mark.parametrize(("side", "sign"), [("lower", 1.0), ("upper", -1.0)])
def test_reflection_adds_the_mirror_images(side: str, sign: float) -> None:
    """At one bound, the estimate is SciPy's at the point plus at its mirror image."""
    case = zoo.railed_gaussians([0.5], param_count=2, side=side)
    samples = np.asarray(case.sample(jax.random.key(11), 3_000).samples[0, 0])
    fitted = KernelDensity.fit(samples, None, case.space)
    reference = scipy_estimate(samples, float(fitted.factor))
    points = samples[:6] + np.array([0.05 * sign, 0.0])
    mirrored = points * np.array([-1.0, 1.0])
    expected = np.logaddexp(reference.logpdf(points.T), reference.logpdf(mirrored.T))
    np.testing.assert_allclose(fitted.log_density(points), expected, rtol=1e-10)
    outside = np.array([-0.1 * sign, 0.0])
    assert float(fitted.log_density(outside)) == -math.inf


def test_two_bounds_reflect_into_every_image() -> None:
    """Between two bounds the images are shifts by 2L and mirrors at either bound."""
    rng = np.random.default_rng(12)
    samples = np.c_[rng.beta(0.8, 0.8, size=1_500), rng.normal(size=1_500)]
    space = space_of(
        {"name": "x0", "topology": "interval", "bounds": [0.0, 1.0]}, {"name": "x1"}
    )
    fitted = KernelDensity.fit(samples, None, space)
    reference = scipy_estimate(samples, float(fitted.factor))
    points = samples[:5]
    steps = np.arange(-10, 11)[:, None]
    shifts = [np.array([2.0 * step, 0.0]) for step in steps[:, 0]]
    mirror = np.array([-1.0, 1.0])
    images = [points + shift for shift in shifts] + [
        points * mirror + shift for shift in shifts
    ]
    expected = np.log(np.sum([reference.pdf(image.T) for image in images], axis=0))
    np.testing.assert_allclose(fitted.log_density(points), expected, rtol=1e-10)


def test_reflected_and_wrapped_densities_integrate_to_one(
    grid_integral: GridIntegral,
) -> None:
    """A railed pair and a sphere in latitude have unit mass over their supports."""
    railed = zoo.railed_gaussians([0.3], param_count=2)
    fitted = KernelDensity.fit(
        railed.sample(jax.random.key(13), 400).samples[0, 0], None, railed.space
    )
    # * Kernels about 0.3 wide span several panels, so Gauss-Legendre stays exact.
    assert grid_integral(
        density_of(fitted), (0.0, 6.0, 24), (-6.0, 6.0, 24)
    ) == pytest.approx(1.0, abs=1e-6)
    sphere = zoo.von_mises_fisher(0.0, width=0.4, colatitude=0.4)
    fitted = KernelDensity.fit(
        sphere.sample(jax.random.key(14), 400).samples[0, 0], None, sphere.space
    )
    assert grid_integral(
        density_of(fitted), (0.0, TWO_PI, 32), (-math.pi / 2, math.pi / 2, 16)
    ) == pytest.approx(1.0, abs=1e-6)


# * Validity


def test_singular_samples_give_nan_and_warn() -> None:
    """A parameter without spread leaves no covariance to scale the kernels."""
    samples = np.c_[np.zeros(30), np.arange(30.0)]
    with pytest.warns(ValidityWarning, match="'x0' has no spread"):
        fitted = KernelDensity.fit(
            samples, None, space_of({"name": "x0"}, {"name": "x1"})
        )
    assert math.isnan(float(fitted.log_density(np.zeros(2))))


def test_too_few_samples_for_the_dimension_warn() -> None:
    """Below Silverman's sample size for D dimensions the estimate warns."""
    space = space_of(*({"name": f"x{index}"} for index in range(3)))
    rng = np.random.default_rng(15)
    with pytest.warns(ValidityWarning, match="at least 67 effective samples"):
        KernelDensity.fit(rng.normal(size=(50, 3)), None, space)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        KernelDensity.fit(rng.normal(size=(100, 3)), None, space)
    wide = space_of(*({"name": f"x{index}"} for index in range(21)))
    with pytest.warns(ValidityWarning, match="over 1.9e"):
        KernelDensity.fit(rng.normal(size=(100, 21)), None, wide)


def test_minimum_sample_sizes_follow_silverman_criterion() -> None:
    """Each size is the smallest n with relative MSE below 0.1 at a Gaussian's mode."""

    def relative_error(log_width: float, count: int, size: int) -> float:
        width = math.exp(log_width)
        variance = (
            2 ** (-size / 2) * width ** (-size) * (1 + width**2 / 2) ** (-size / 2)
            - (1 + width**2) ** (-size)
        ) / count
        return float(variance + ((1 + width**2) ** (-size / 2) - 1) ** 2)

    def smallest_error(count: int, size: int) -> float:
        return float(
            optimize.minimize_scalar(
                relative_error,
                bounds=(-10.0, 3.0),
                args=(count, size),
                method="bounded",
                options={"xatol": 1e-12},
            ).fun
        )

    for size, needed in enumerate(MINIMUM_SAMPLE_SIZES, start=1):
        assert smallest_error(needed, size) < 0.1 <= smallest_error(needed - 1, size)
    # * Silverman (1986) prints the first ten to three digits.
    printed = (4, 19, 67, 223, 768, 2_790, 10_700, 43_700, 187_000, 842_000)
    for needed, value in zip(MINIMUM_SAMPLE_SIZES, printed, strict=False):
        assert needed == pytest.approx(value, rel=5e-3)
    assert special.gamma(1.0) == 1.0


# * Transformations and plots


def test_vmap_fits_match_concrete_fits() -> None:
    """Traced fits count windings from a bound and give the concrete densities."""
    case = zoo.wrapped_gaussians(0.5, width=0.4, param_count=2, correlation=0.3)
    batch = case.sample(jax.random.key(16), 2_000, batch_size=3).samples[:, 0]
    batched = jax.vmap(lambda values: KernelDensity.fit(values, None, case.space))(
        batch
    )
    concrete = KernelDensity.fit(batch[2], None, case.space)
    assert len(batched.image_shifts) > len(concrete.image_shifts)
    values = jax.vmap(lambda estimator, points: estimator.log_density(points))(
        batched, batch[:, :4]
    )
    np.testing.assert_allclose(
        values[2], concrete.log_density(batch[2, :4]), rtol=1e-10
    )
    compiled = jax.jit(lambda estimator, points: estimator.log_density(points))
    np.testing.assert_allclose(
        compiled(concrete, batch[2, :4]), concrete.log_density(batch[2, :4]), rtol=1e-12
    )


def test_cantuccio_contours_agree_with_the_estimate() -> None:
    """With the resolved factor, cantuccio's SciPy estimates equal ours."""
    cantuccio_kde = pytest.importorskip("cantuccio.kde")
    rng = np.random.default_rng(17)
    samples = rng.multivariate_normal([0.0, 1.0], [[1.0, 0.5], [0.5, 2.0]], size=1_500)
    fitted = KernelDensity.fit(samples, None, space_of({"name": "x"}, {"name": "y"}))
    factor = fitted.resolved_settings().bandwidth
    grid_x, grid_y, grid_density = cantuccio_kde.kde_2d(
        samples[:, 0], samples[:, 1], bw=factor, n=12, fast=False
    )
    ours = density_of(fitted)(np.stack([grid_x, grid_y], axis=-1))
    np.testing.assert_allclose(ours, grid_density, rtol=1e-10)
    circle = zoo.wrapped_gaussians(0.0, width=0.4, param_count=1)
    angles = np.asarray(circle.sample(jax.random.key(18), 2_000).samples[0, 0, :, 0])
    wrapped = KernelDensity.fit(angles[:, None], None, circle.space)
    grid, density = cantuccio_kde.kde_1d(
        angles,
        bw=wrapped.resolved_settings().bandwidth,
        n=50,
        fast=False,
        periodic=(0.0, TWO_PI),
    )
    np.testing.assert_allclose(density_of(wrapped)(grid[:, None]), density, rtol=1e-8)
