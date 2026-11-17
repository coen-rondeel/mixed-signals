"""Tests for the closed-form σ_eff and r_eff maps."""

import functools
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import integrate, special, stats

from mixed_signals.metrics.moments import gaussian_hellinger, gaussian_kl, gaussian_w2
from mixed_signals.sigma.calibration import CalibrationCurve
from mixed_signals.sigma.maps import (
    KERNEL_METRICS,
    QUADRATURE_METRICS,
    SIGMA_EFF_MAPS,
    SLICED_METRICS,
    SUBTRACTION_SPACES,
    SubtractionSpace,
    null_subtract,
    r_eff,
    sigma_eff,
)
from mixed_signals.zoo import closed_forms

SHIFTS = [0.0, 1e-6, 0.3, 1.0, 3.0]
RATIOS = [0.2, 0.7, 1.0, 1.5, 4.0]


def shifted_values(shift: float) -> dict[str, float]:
    """Return every metric of two unit Gaussians a given shift apart."""
    covariance = jnp.eye(2)
    mean = jnp.array([shift, 0.0])
    kl = gaussian_kl(jnp.zeros(2), covariance, mean, covariance)
    w2 = gaussian_w2(jnp.zeros(2), covariance, mean, covariance)
    hellinger = gaussian_hellinger(jnp.zeros(2), covariance, mean, covariance)
    return {
        "mean_shift": shift,
        "w1": float(closed_forms.normal_w1(0.0, 1.0, shift, 1.0)),
        "w2": float(w2.total),
        "w2_location": float(w2.location),
        "kl": float(kl.total),
        "kl_location": float(kl.location),
        "ks": float(closed_forms.normal_ks(0.0, 1.0, shift, 1.0)),
        "kuiper_ks": float(closed_forms.normal_ks(0.0, 1.0, shift, 1.0)),
        "hellinger": float(hellinger.total),
        "hellinger_location": float(hellinger.location),
        "sinkhorn": float(
            closed_forms.gaussian_sinkhorn(
                jnp.zeros(2), covariance, mean, covariance, 0.7
            ).total
        ),
    }


@pytest.mark.parametrize("shift", SHIFTS)
def test_sigma_eff_returns_the_shift_of_unit_gaussians(shift: float) -> None:
    """Each map inverts its metric for two unit Gaussians a shift apart."""
    values = shifted_values(shift)
    assert set(values) == set(SIGMA_EFF_MAPS)
    for metric, value in values.items():
        assert float(sigma_eff(metric, value)) == pytest.approx(
            shift, rel=1e-9, abs=1e-15
        ), metric


def test_sigma_eff_is_signed_for_every_map() -> None:
    """A subtracted value below zero gives minus the σ_eff of its magnitude."""
    assert float(sigma_eff("mean_shift", -0.7)) == -0.7
    for metric in SIGMA_EFF_MAPS:
        positive = float(sigma_eff(metric, 0.1))
        assert float(sigma_eff(metric, -0.1)) == -positive, metric
        assert float(sigma_eff(metric, 0.0)) == 0.0, metric
    for metric in ("sliced_w2", "sliced_sinkhorn"):
        assert float(sigma_eff(metric, -0.1, dimension=3)) == -float(
            sigma_eff(metric, 0.1, dimension=3)
        )
    assert float(sigma_eff("energy", -0.1, dimension=3)) == -float(
        sigma_eff("energy", 0.1, dimension=3)
    )
    assert float(sigma_eff("mmd", -0.01, bandwidth=1.0, dimension=3)) == -float(
        sigma_eff("mmd", 0.01, bandwidth=1.0, dimension=3)
    )
    for metric in QUADRATURE_METRICS:
        assert float(sigma_eff(metric, -0.1)) == -float(sigma_eff(metric, 0.1))


def test_sigma_eff_above_the_range_of_a_metric_is_nan() -> None:
    """The range limits of KS, Hellinger, MMD², JS and AD give NaN, with either sign."""
    for value in (1.2, -1.2):
        assert math.isnan(float(sigma_eff("ks", value)))
        assert math.isnan(float(sigma_eff("kuiper_ks", value)))
        assert math.isnan(float(sigma_eff("hellinger", value)))
    assert float(sigma_eff("ks", 1.0)) == math.inf
    assert float(sigma_eff("ks", -1.0)) == -math.inf
    for value in (math.log(2) * 1.001, -math.log(2) * 1.001):
        assert math.isnan(float(sigma_eff("js", value)))
    for value in ((8 * math.log(2) - 4) * 1.001, -(8 * math.log(2) - 4) * 1.001):
        assert math.isnan(float(sigma_eff("ad", value)))
    assert math.isnan(float(sigma_eff("js", math.nan)))


@pytest.mark.parametrize("ratio", RATIOS)
def test_r_eff_returns_the_width_ratio_of_unit_gaussians(ratio: float) -> None:
    """Each shape term of N(0, 1) and N(0, r²) maps back to r, on the right side."""
    kl = 0.5 * (1 / ratio**2 - 1 + math.log(ratio**2))
    values = {
        "kl_shape": kl,
        "w2_shape": abs(1 - ratio),
        "hellinger_shape": float(closed_forms.normal_hellinger(0.0, 1.0, 0.0, ratio)),
    }
    direction = math.log(ratio**2)
    for metric, value in values.items():
        assert float(r_eff(metric, value, direction)) == pytest.approx(
            ratio, rel=1e-8
        ), metric
    assert float(r_eff("width_ratio", ratio)) == ratio


def test_kl_shape_inversion_agrees_with_lambert_w() -> None:
    """1/r² = −W(−e^(−(1 + 2 KL))), on the principal branch for r ≥ 1, else on −1."""
    values = np.array([1e-8, 1e-3, 0.1, 1.0, 5.0])
    for wider, branch in ((True, 0), (False, -1)):
        expected = 1 / np.sqrt(
            -special.lambertw(-np.exp(-(1 + 2 * values)), branch).real
        )
        mapped = r_eff("kl_shape", values, np.where(wider, 1.0, -1.0) * np.ones(5))
        np.testing.assert_allclose(mapped, expected, rtol=1e-7)


def test_r_eff_of_a_value_below_zero_is_the_reciprocal_root() -> None:
    """A subtracted shape term below zero gives 1/r of the root r of its magnitude."""
    for metric in ("kl_shape", "w2_shape", "hellinger_shape"):
        for direction in (1.0, -1.0):
            positive = float(r_eff(metric, 0.1, direction))
            assert float(r_eff(metric, -0.1, direction)) == pytest.approx(
                1 / positive, rel=1e-14
            ), metric
            assert float(r_eff(metric, 0.0, direction)) == 1.0, metric
    assert float(r_eff("width_ratio", 0.8)) == 0.8


def test_r_eff_outside_the_range_is_nan() -> None:
    """A W2 shape term beyond one for the narrower side, H ≥ 1, and a ratio of zero."""
    assert math.isnan(float(r_eff("w2_shape", 1.2, -1.0)))
    assert math.isnan(float(r_eff("w2_shape", -1.2, -1.0)))
    assert math.isnan(float(r_eff("kl_shape", math.nan, 1.0)))
    assert float(r_eff("w2_shape", 1.2, 1.0)) == pytest.approx(2.2)
    assert math.isnan(float(r_eff("hellinger_shape", 1.0, 1.0)))
    assert math.isnan(float(r_eff("width_ratio", 0.0)))


def test_maps_check_their_names_and_directions() -> None:
    """Unknown metrics name the maps that exist; shape terms need a direction."""
    with pytest.raises(KeyError, match="no closed-form sigma_eff map for 'c2st'"):
        sigma_eff("c2st", 0.1)
    with pytest.raises(KeyError, match="no r_eff map for 'kl'"):
        r_eff("kl", 0.1, 1.0)
    with pytest.raises(ValueError, match="needs log_det_ratio"):
        r_eff("kl_shape", 0.1)


def test_maps_compile_and_map() -> None:
    """The maps run under jit and vmap, element by element."""
    values = jnp.array([0.1, 0.5, 2.0])
    compiled = jax.jit(lambda value: sigma_eff("kl", value))(values)
    np.testing.assert_allclose(compiled, np.sqrt(2 * np.asarray(values)), rtol=1e-12)
    ratios = jax.vmap(lambda value: r_eff("kl_shape", value, 1.0))(values)
    for index, value in enumerate(values):
        assert float(ratios[index]) == pytest.approx(
            float(r_eff("kl_shape", value, 1.0))
        )


# * Kernel metrics


def chi_mean(dimension: int, noncentrality: float) -> float:
    """Return E[χ_D(λ)]: folded normal and Rice closed forms, else SciPy integration."""
    if dimension == 1:
        return math.sqrt(2 / math.pi) * math.exp(
            -0.5 * noncentrality**2
        ) + noncentrality * math.erf(noncentrality / math.sqrt(2))
    if dimension == 2:
        half = 0.25 * noncentrality**2
        return math.sqrt(math.pi / 2) * float(
            (1 + 2 * half) * special.ive(0, half) + 2 * half * special.ive(1, half)
        )
    if noncentrality == 0:
        return float(stats.chi(dimension).mean())
    return float(stats.ncx2(dimension, noncentrality**2).expect(np.sqrt))


@pytest.mark.parametrize(
    ("dimension", "shift"),
    [
        *((dimension, shift) for dimension in (1, 2) for shift in SHIFTS),
        (1, 120.0),
        (2, 200.0),
        *(
            (dimension, shift)
            for dimension in (8, 16)
            for shift in (0.3, 1.0, 3.0, 9.0)
        ),
    ],
)
def test_energy_map_returns_the_shift_of_unit_gaussians(
    dimension: int, shift: float
) -> None:
    """2√2 (E[χ_D(Δ/√2)] − E[χ_D]) maps back to Δ, past the expansion threshold too."""
    value = (
        2
        * math.sqrt(2)
        * (chi_mean(dimension, shift / math.sqrt(2)) - chi_mean(dimension, 0.0))
    )
    mapped = float(sigma_eff("energy", value, dimension=dimension))
    assert mapped == pytest.approx(shift, rel=1e-7, abs=1e-6)


@pytest.mark.parametrize("dimension", [1, 3, 16])
@pytest.mark.parametrize("bandwidth", [0.5, 1.3, 4.0])
def test_mmd_map_returns_the_shift_of_unit_gaussians(
    dimension: int, bandwidth: float
) -> None:
    """2c(1 − exp(−Δ²/(2(ℓ² + 2)))) maps back to Δ for every bandwidth."""
    scale = (bandwidth**2 / (bandwidth**2 + 2)) ** (dimension / 2)
    for shift in SHIFTS:
        value = 2 * scale * -math.expm1(-(shift**2) / (2 * (bandwidth**2 + 2)))
        mapped = float(
            sigma_eff("mmd", value, bandwidth=bandwidth, dimension=dimension)
        )
        assert mapped == pytest.approx(shift, rel=1e-9, abs=1e-12)


def test_kernel_maps_outside_their_range_are_nan() -> None:
    """MMD² at or beyond 2c, either sign, has no shift; zero is no shift."""
    assert float(sigma_eff("energy", 0.0, dimension=2)) == 0.0
    scale = 1 / 3
    for sign in (1, -1):
        value = sign * 2 * scale
        assert math.isnan(float(sigma_eff("mmd", value, bandwidth=1.0, dimension=2)))
    values = jnp.array([-1.0, 0.0, 0.2, 5.0])
    mapped = sigma_eff("energy", values, dimension=3)
    for index, value in enumerate(values):
        single = sigma_eff("energy", value, dimension=3)
        np.testing.assert_allclose(mapped[index], single, rtol=1e-14)


def test_kernel_maps_check_their_arguments() -> None:
    """Kernel metrics need their dimension, MMD its bandwidth; names are listed."""
    with pytest.raises(ValueError, match="needs its bandwidth and dimension"):
        sigma_eff("mmd", 0.1, dimension=2)
    with pytest.raises(ValueError, match="needs its dimension"):
        sigma_eff("energy", 0.1)
    for dimension in (0, True):
        with pytest.raises(ValueError, match="dimension must be a positive integer"):
            sigma_eff("energy", 0.1, dimension=dimension)
    with pytest.raises(KeyError, match="ad, energy, hellinger"):
        sigma_eff("c2st", 0.1)
    assert KERNEL_METRICS == ("energy", "mmd")


# * Optimal transport


@pytest.mark.parametrize("shift", SHIFTS)
@pytest.mark.parametrize("dimension", [1, 3, 16])
def test_sliced_maps_return_the_shift_of_unit_gaussians(
    shift: float, dimension: int
) -> None:
    """A shift Δ gives ‖Δ‖²/n over uniform directions in n dimensions, at every ε."""
    for metric, value in (
        ("sliced_w2", closed_forms.gaussian_sliced_w2(shift, 1.0, 0, dimension)),
        (
            "sliced_sinkhorn",
            closed_forms.gaussian_sliced_sinkhorn(shift, 1.0, 0, dimension, 0.4),
        ),
    ):
        mapped = sigma_eff(metric, value, dimension=dimension)
        assert float(mapped) == pytest.approx(shift, rel=1e-12, abs=1e-15), metric


def test_transport_maps_check_their_arguments() -> None:
    """Sliced metrics need their dimension; values below zero keep their sign."""
    assert float(sigma_eff("sinkhorn", -1e-3)) == pytest.approx(-math.sqrt(1e-3))
    assert float(sigma_eff("sliced_w2", -1e-3, dimension=4)) == pytest.approx(
        -math.sqrt(4e-3)
    )
    with pytest.raises(ValueError, match="'sliced_sinkhorn' needs its dimension"):
        sigma_eff("sliced_sinkhorn", 0.1)
    for dimension in (0, False):
        with pytest.raises(ValueError, match="dimension must be a positive integer"):
            sigma_eff("sliced_w2", 0.1, dimension=dimension)
    assert SLICED_METRICS == ("sliced_sinkhorn", "sliced_w2")


# * JS and Anderson-Darling, by bisection on the zoo quadratures

QUADRATURE_SHIFTS = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 4.5, 6.0]


@pytest.mark.parametrize("shift", QUADRATURE_SHIFTS)
def test_js_and_ad_maps_invert_the_zoo_quadratures(shift: float) -> None:
    """The map of the zoo's JS or AD value of N(0, 1) and N(δ, 1) returns δ."""
    for metric, quadrature in (
        ("js", closed_forms.normal_js),
        ("ad", closed_forms.normal_anderson_darling),
    ):
        value = float(quadrature(0.0, 1.0, shift, 1.0))
        assert float(sigma_eff(metric, value)) == pytest.approx(shift, rel=1e-6), metric


def js_by_scipy(shift: float) -> float:
    """Return the JS divergence of N(0, 1) and N(δ, 1) by SciPy quadrature."""

    def integrand(point: float) -> float:
        first = stats.norm.pdf(point)
        second = stats.norm.pdf(point, shift)
        mixture = 0.5 * (first + second)
        return float(
            0.5 * (first * np.log(first / mixture) + second * np.log(second / mixture))
        )

    return float(
        integrate.quad(integrand, -12, shift + 12, points=[0.0, shift], limit=400)[0]
    )


def ad_by_scipy(shift: float) -> float:
    """Return the AD distance of N(0, 1) and N(δ, 1) by SciPy quadrature."""

    def integrand(point: float) -> float:
        first_cdf, second_cdf = stats.norm.cdf(point), stats.norm.cdf(point, shift)
        first_sf, second_sf = stats.norm.sf(point), stats.norm.sf(point, shift)
        mixture = 0.5 * (first_cdf + second_cdf)
        complement = 0.5 * (first_sf + second_sf)
        density = 0.5 * (stats.norm.pdf(point) + stats.norm.pdf(point, shift))
        return float((first_cdf - second_cdf) ** 2 / (mixture * complement) * density)

    return float(
        integrate.quad(integrand, -10, shift + 10, points=[0.0, shift], limit=400)[0]
    )


@pytest.mark.parametrize("shift", [0.5, 2.0])
def test_js_and_ad_maps_agree_with_scipy_quadrature(shift: float) -> None:
    """Values from SciPy's quadrature, independent of the zoo, map back to the shift."""
    assert float(sigma_eff("js", js_by_scipy(shift))) == pytest.approx(shift, rel=1e-6)
    assert float(sigma_eff("ad", ad_by_scipy(shift))) == pytest.approx(shift, rel=1e-6)


def test_js_and_ad_maps_saturate_beyond_ten_sigma() -> None:
    """At the limits of the metrics the map stops at the end of its interval."""
    for metric, limit in (("js", math.log(2)), ("ad", 8 * math.log(2) - 4)):
        beyond = float(sigma_eff(metric, limit))
        assert 12.0 < beyond <= 20.0, metric
        assert float(sigma_eff(metric, 0.0)) == 0.0


# * The KDE factor and the kNN curve


@pytest.mark.parametrize("metric", ["js", "hellinger"])
@pytest.mark.parametrize("factor", [0.1, 0.266, 0.5])
def test_kde_estimator_undoes_the_widening_of_the_kernel(
    metric: str, factor: float
) -> None:
    """A KDE of each unit Gaussian reads the shift δ/√(1 + f²); the map restores δ."""
    quadrature = (
        closed_forms.normal_js if metric == "js" else closed_forms.normal_hellinger
    )
    for shift in (0.2, 0.5, 1.0, 2.0):
        read = shift / math.sqrt(1 + factor**2)
        value = float(quadrature(0.0, 1.0, read, 1.0))
        mapped = float(sigma_eff(metric, value, estimator="kde", bandwidth=factor))
        assert mapped == pytest.approx(shift, rel=1e-6)
        assert mapped == pytest.approx(
            math.sqrt(1 + factor**2) * float(sigma_eff(metric, value)), rel=1e-14
        )
        negative = float(sigma_eff(metric, -value, estimator="kde", bandwidth=factor))
        assert negative == -mapped


def test_other_estimators_leave_the_maps_as_they_are() -> None:
    """None, "empirical" and "gaussian" change nothing, and "knn" only the KL map."""
    value = 0.07
    for estimator in (None, "empirical", "gaussian", "knn"):
        for metric in ("js", "hellinger", "w2", "ks"):
            assert float(sigma_eff(metric, value, estimator=estimator)) == float(
                sigma_eff(metric, value)
            )
    for estimator in (None, "empirical", "gaussian"):
        assert float(sigma_eff("kl", value, estimator=estimator)) == pytest.approx(
            math.sqrt(2 * value), rel=1e-14
        )


def test_estimator_arguments_are_checked() -> None:
    """The kde map needs js or hellinger and a factor; kNN KL needs its curve."""
    with pytest.raises(ValueError, match="hellinger, js only, got 'kl'"):
        sigma_eff("kl", 0.1, estimator="kde", bandwidth=0.3)
    with pytest.raises(ValueError, match="got 'mean_shift'"):
        sigma_eff("mean_shift", 0.1, estimator="kde", bandwidth=0.3)
    with pytest.raises(ValueError, match="'js' with the kde estimator needs its"):
        sigma_eff("js", 0.1, estimator="kde")
    with pytest.raises(ValueError, match="'kl' with the knn estimator needs its"):
        sigma_eff("kl", 0.1, estimator="knn")
    for shifts, values in (
        (jnp.zeros(3), jnp.zeros(2)),
        (jnp.zeros(1), jnp.zeros(1)),
        (jnp.zeros((2, 2)), jnp.zeros((2, 2))),
    ):
        with pytest.raises(ValueError, match="calibration curve needs"):
            sigma_eff(
                "kl", 0.1, estimator="knn", curve=CalibrationCurve(shifts, values)
            )


CURVE = CalibrationCurve(
    shifts=jnp.array([0.0, 1.0, 2.0, 3.0]), values=jnp.array([0.0, 0.4, 1.5, 3.0])
)


def test_knn_curve_is_interpolated_and_carried_on_beyond_its_end() -> None:
    """Inside the curve σ_eff interpolates linearly; beyond it δ_last √(v/v_last)."""
    cases = {0.0: 0.0, 0.4: 1.0, 0.95: 1.5, 1.5: 2.0, 3.0: 3.0, 12.0: 6.0}
    for value, expected in cases.items():
        mapped = float(sigma_eff("kl", value, estimator="knn", curve=CURVE))
        assert mapped == pytest.approx(expected, rel=1e-12), value
        negative = float(sigma_eff("kl", -value, estimator="knn", curve=CURVE))
        assert negative == pytest.approx(-expected, rel=1e-12), value
    assert math.isnan(float(sigma_eff("kl", math.nan, estimator="knn", curve=CURVE)))
    flat = CalibrationCurve(CURVE.shifts, jnp.zeros(4))
    assert math.isnan(float(sigma_eff("kl", 0.1, estimator="knn", curve=flat)))


def test_new_maps_compile_and_map() -> None:
    """The JS, AD, KDE and kNN maps run under jit and vmap, element by element."""
    values = jnp.array([-0.05, 0.0, 0.01, 0.2, 0.6])
    for metric in ("js", "ad"):
        jitted = jax.jit(lambda value, name=metric: sigma_eff(name, value))(values)
        mapped = jax.vmap(lambda value, name=metric: sigma_eff(name, value))(values)
        for index, value in enumerate(values):
            single = float(sigma_eff(metric, value))
            assert float(jitted[index]) == pytest.approx(single, rel=1e-12)
            assert float(mapped[index]) == pytest.approx(single, rel=1e-12)
    kde = jax.jit(
        lambda value, factor: sigma_eff("js", value, estimator="kde", bandwidth=factor)
    )(values, 0.3)
    np.testing.assert_allclose(
        kde, math.sqrt(1 + 0.3**2) * np.asarray(sigma_eff("js", values)), rtol=1e-12
    )
    curve_map = jax.jit(
        lambda value, curve: sigma_eff("kl", value, estimator="knn", curve=curve)
    )(values, CURVE)
    for index, value in enumerate(values):
        single = float(sigma_eff("kl", value, estimator="knn", curve=CURVE))
        assert float(curve_map[index]) == pytest.approx(single, rel=1e-12)
    assert sigma_eff("js", jnp.zeros((2, 3))).shape == (2, 3)


# * Null subtraction


def numpy_subtract(space: SubtractionSpace, value: float, draws: np.ndarray) -> float:
    """Subtract the mean of the finite draws in a space, in NumPy."""
    finite = draws[np.isfinite(draws)]
    if space is SubtractionSpace.LINEAR:
        return float(value - finite.mean())
    if space is SubtractionSpace.SQUARED:
        difference = value * abs(value) - (finite * np.abs(finite)).mean()
        return float(np.sign(difference) * np.sqrt(abs(difference)))
    positive = finite[finite > 0]
    return float(np.exp(np.log(value) - np.log(positive).mean()))


@pytest.mark.parametrize(
    ("metric", "space"),
    [
        ("kl", SubtractionSpace.LINEAR),
        ("ks", SubtractionSpace.SQUARED),
        ("width_ratio", SubtractionSpace.LOG),
    ],
)
def test_null_subtraction_follows_numpy_in_each_space(
    metric: str, space: SubtractionSpace
) -> None:
    """Linear, squared and log subtraction of a value from its draws, per value."""
    rng = np.random.default_rng(7)
    draws = np.abs(rng.normal(1.0, 0.3, size=(4, 50)))
    values = np.array([0.4, 1.1, 0.9, 2.5])
    subtracted = np.asarray(null_subtract(metric, values, draws))
    for index in range(4):
        assert subtracted[index] == pytest.approx(
            numpy_subtract(space, float(values[index]), draws[index]), rel=1e-12
        )


def test_squared_subtraction_keeps_values_below_the_null() -> None:
    """The signed square lets a value below the null mean give a negative result."""
    draws = np.full(10, 0.5)
    below = float(null_subtract("hellinger", 0.3, draws))
    assert below == pytest.approx(-math.sqrt(0.25 - 0.09))
    negative_value = float(null_subtract("hellinger_shape", -0.2, draws))
    assert negative_value == pytest.approx(-math.sqrt(0.25 + 0.04))
    assert float(null_subtract("hellinger", 0.5, draws)) == 0.0


def test_non_finite_draws_leave_the_mean() -> None:
    """NaN and infinite draws count in none of the spaces; no usable draw gives NaN."""
    clean = np.array([0.2, 0.4, 0.6, 0.8])
    dirty = np.array([0.2, np.nan, 0.4, np.inf, 0.6, -np.inf, 0.8])
    for metric in ("kl", "ks", "width_ratio"):
        assert float(null_subtract(metric, 0.5, dirty)) == pytest.approx(
            float(null_subtract(metric, 0.5, clean)), rel=1e-14
        )
        assert math.isnan(float(null_subtract(metric, 0.5, np.full(5, np.nan))))
        assert math.isnan(float(null_subtract(metric, 0.5, np.zeros(0))))
        assert math.isnan(float(null_subtract(metric, math.nan, clean)))
        assert math.isnan(float(null_subtract(metric, math.inf, clean)))


def test_log_subtraction_ignores_non_positive_draws_and_values() -> None:
    """A ratio needs a positive value; non-positive draws leave the mean of ln r."""
    draws = np.array([0.5, 1.0, 2.0, 0.0, -1.0])
    positive = draws[draws > 0]
    expected = 1.5 / math.exp(np.log(positive).mean())
    assert float(null_subtract("width_ratio", 1.5, draws)) == pytest.approx(
        expected, rel=1e-14
    )
    assert math.isnan(float(null_subtract("width_ratio", 0.0, draws)))
    assert math.isnan(float(null_subtract("width_ratio", -0.3, draws)))
    assert math.isnan(float(null_subtract("width_ratio", 1.0, np.array([0.0, -1.0]))))


def test_null_subtraction_keeps_leading_axes() -> None:
    """Values shaped (...) with draws shaped (..., M) give a result shaped (...)."""
    rng = np.random.default_rng(3)
    draws = rng.normal(size=(2, 3, 25))
    values = rng.normal(size=(2, 3))
    result = null_subtract("kl", values, draws)
    assert result.shape == (2, 3)
    np.testing.assert_allclose(result, values - draws.mean(axis=-1), rtol=1e-12)
    assert null_subtract("ks", 0.3, draws[0, 0]).shape == ()


def test_null_subtraction_compiles_and_maps() -> None:
    """The subtraction runs under jit and vmap, row by row."""
    rng = np.random.default_rng(5)
    draws = jnp.asarray(rng.normal(1.0, 0.2, size=(4, 30)))
    values = jnp.array([0.9, 1.2, 1.0, 0.7])
    for metric in ("kl", "ks", "width_ratio"):
        expected = null_subtract(metric, values, draws)
        subtract = functools.partial(null_subtract, metric)
        jitted = jax.jit(subtract)(values, draws)
        mapped = jax.vmap(subtract)(values, draws)
        np.testing.assert_allclose(jitted, expected, rtol=1e-12)
        np.testing.assert_allclose(mapped, expected, rtol=1e-12)


def test_null_subtraction_names_its_metrics() -> None:
    """An unknown metric lists the spaces, and the table is the agreed one."""
    with pytest.raises(KeyError, match=r"no null subtraction for 'c2st'.*width_ratio"):
        null_subtract("c2st", 0.1, np.zeros(3))
    linear = {
        "mean_shift", "kl", "kl_location", "kl_shape", "js", "ad", "energy", "mmd",
        "sinkhorn", "sliced_w2", "sliced_sinkhorn",
    }  # fmt: skip
    squared = {
        "ks", "kuiper_ks", "w1", "w2", "w2_location", "w2_shape", "hellinger",
        "hellinger_location", "hellinger_shape",
    }  # fmt: skip
    expected = {
        **dict.fromkeys(linear, SubtractionSpace.LINEAR),
        **dict.fromkeys(squared, SubtractionSpace.SQUARED),
        "width_ratio": SubtractionSpace.LOG,
    }
    assert dict(SUBTRACTION_SPACES) == expected
    mapped = {*SIGMA_EFF_MAPS, *KERNEL_METRICS, *SLICED_METRICS, *QUADRATURE_METRICS}
    assert mapped <= set(SUBTRACTION_SPACES)
