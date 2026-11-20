"""Tests for the closed-form σ_eff and r_eff maps."""

import functools
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import integrate, special, stats

from mixed_signals.core.space import ParamSpace
from mixed_signals.errors import ValidityWarning
from mixed_signals.metrics.moments import gaussian_hellinger, gaussian_kl, gaussian_w2
from mixed_signals.metrics.ot.sliced import (
    SLICED_DIRECTIONS,
    sliced_directions,
    sliced_wasserstein_of_gaussians,
)
from mixed_signals.prng import derive_key
from mixed_signals.sigma.calibration import CalibrationCurve
from mixed_signals.sigma.maps import (
    CHORDAL_CIRCLE_WIDTH,
    CHORDAL_METRICS,
    CHORDAL_SPHERE_WIDTH,
    KERNEL_METRICS,
    QUADRATURE_METRICS,
    SIGMA_EFF_MAPS,
    SLICED_METRICS,
    SUBTRACTION_SPACES,
    SubtractionSpace,
    null_subtract,
    r_eff,
    sigma_eff,
    warn_wide_chordal_axes,
)
from mixed_signals.transforms.whiten import Whitening
from mixed_signals.zoo import ZooCase, closed_forms, von_mises_fisher, wrapped_gaussians

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
    assert set(values) == {"mean_shift", *SIGMA_EFF_MAPS}
    for metric, value in values.items():
        assert float(sigma_eff(metric, value)) == pytest.approx(
            shift, rel=1e-9, abs=1e-15
        ), metric


def test_mean_shift_has_no_map_entry_and_returns_its_signed_value() -> None:
    """The mean shift is σ_eff already, so sigma_eff returns it before the maps."""
    assert "mean_shift" not in SIGMA_EFF_MAPS
    for value in (-0.7, 0.0, 0.25):
        assert float(sigma_eff("mean_shift", value)) == value
    values = jnp.array([-1.5, 0.0, 2.0])
    np.testing.assert_array_equal(sigma_eff("mean_shift", values), values)


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


POOLED_RATIOS = [0.5, 0.8, 1.25, 2.0, 5.0]


def pooled_w2_term(ratio: float) -> float:
    """Return the W2 shape term of a width ratio along one axis in a pooled frame."""
    return math.sqrt(2) * abs(ratio - 1) / math.sqrt(1 + ratio**2)


def pooled_pair(first: jax.Array, second: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Whiten two covariances by their mean, with the inverse Cholesky factor."""
    factor = jnp.linalg.inv(jnp.linalg.cholesky(0.5 * (first + second)))
    return factor @ first @ factor.T, factor @ second @ factor.T


@pytest.mark.parametrize("ratio", POOLED_RATIOS)
def test_pooled_w2_shape_inverts_its_relation_on_both_roots(ratio: float) -> None:
    """The pooled map returns r from g = √2 |r − 1|/√(1 + r²), wider or narrower."""
    value = pooled_w2_term(ratio)
    mapped = r_eff("w2_shape", value, math.log(ratio**2), pooled=True)
    assert float(mapped) == pytest.approx(ratio, rel=1e-10)
    # * A value below zero gives the reciprocal of the root, as in the other maps.
    for direction in (1.0, -1.0):
        positive = float(r_eff("w2_shape", value, direction, pooled=True))
        negative = float(r_eff("w2_shape", -value, direction, pooled=True))
        assert negative == pytest.approx(1 / positive, rel=1e-14)


def test_pooled_w2_shape_has_no_cancellation_and_a_pole_at_root_two() -> None:
    """Zero gives one, a tiny value its own size, and √2 and above give NaN."""
    assert float(r_eff("w2_shape", 0.0, 1.0, pooled=True)) == 1.0
    assert float(r_eff("w2_shape", 0.0, -1.0, pooled=True)) == 1.0
    # * Digits of r − 1 are limited by the rounding of r near one, about 1e-16/1e-8.
    wider = float(r_eff("w2_shape", 1e-8, 1.0, pooled=True))
    narrower = float(r_eff("w2_shape", 1e-8, -1.0, pooled=True))
    assert wider - 1 == pytest.approx(1e-8, rel=1e-6)
    assert 1 - narrower == pytest.approx(1e-8, rel=1e-6)
    for value in (math.sqrt(2), 1.5, -math.sqrt(2), math.nan):
        assert math.isnan(float(r_eff("w2_shape", value, 1.0, pooled=True)))
    near_pole = math.sqrt(2) * (1 - 1e-9)
    wide = float(r_eff("w2_shape", near_pole, 1.0, pooled=True))
    assert math.isfinite(wide)
    assert wide > 1e8
    # * The narrow root is 1/wide; the form 1 − spread would cancel here.
    narrow = float(r_eff("w2_shape", near_pole, -1.0, pooled=True))
    assert narrow == pytest.approx(1 / wide, rel=1e-6)


def test_pooled_keyword_leaves_the_other_maps_as_they_are() -> None:
    """Only the W2 shape term depends on the frame."""
    for metric in ("kl_shape", "hellinger_shape"):
        for direction in (1.0, -1.0):
            assert float(r_eff(metric, 0.1, direction, pooled=True)) == float(
                r_eff(metric, 0.1, direction)
            )
    assert float(r_eff("width_ratio", 0.8, pooled=True)) == 0.8
    # * With unit width of the first posterior, the reference map is the old one.
    assert float(r_eff("w2_shape", 0.1, 1.0)) == pytest.approx(1.1)
    assert float(r_eff("w2_shape", 0.1, 1.0, pooled=True)) != pytest.approx(1.1)


@pytest.mark.parametrize("ratio", [0.6, 1.25, 3.0])
def test_pooled_w2_shape_returns_the_ratio_of_pooled_whitened_gaussians(
    ratio: float,
) -> None:
    """gaussian_w2 of two covariances whitened by their mean maps back to r."""
    first = jnp.eye(2)
    second = jnp.diag(jnp.array([ratio**2, 1.0]))
    one, two = pooled_pair(first, second)
    np.testing.assert_allclose(one + two, 2 * jnp.eye(2), atol=1e-12)
    terms = gaussian_w2(jnp.zeros(2), one, jnp.zeros(2), two)
    assert float(terms.shape) == pytest.approx(pooled_w2_term(ratio), rel=1e-10)
    mapped = r_eff("w2_shape", terms.shape, terms.log_det_ratio, pooled=True)
    assert float(mapped) == pytest.approx(ratio, rel=1e-8)
    # * The reference map assumes a unit first width, so it reads the ratio wrongly.
    reference = r_eff("w2_shape", terms.shape, terms.log_det_ratio)
    assert abs(float(reference) - ratio) > 0.02


def test_pooled_w2_shape_separates_per_axis_and_ignores_the_whitening() -> None:
    """Two changed axes add in squares; another whitening factor gives the same term."""
    ratios = (1.5, 0.7)
    first = jnp.eye(2)
    second = jnp.diag(jnp.array(ratios) ** 2)
    one, two = pooled_pair(first, second)
    shape = float(gaussian_w2(jnp.zeros(2), one, jnp.zeros(2), two).shape)
    expected = math.sqrt(sum(pooled_w2_term(ratio) ** 2 for ratio in ratios))
    assert shape == pytest.approx(expected, rel=1e-10)
    # * The Bures form of commuting covariances with A + B = 2I, per eigenvalue a.
    eigenvalues = np.linalg.eigvalsh(np.asarray(one))
    bures = np.sqrt(np.sum((np.sqrt(eigenvalues) - np.sqrt(2 - eigenvalues)) ** 2))
    assert shape == pytest.approx(float(bures), rel=1e-10)
    # * Any whitening W with W (mean covariance) Wᵀ = I differs by a rotation.
    rotation = jnp.array([[0.6, -0.8], [0.8, 0.6]])
    factor = rotation @ jnp.linalg.inv(jnp.linalg.cholesky(0.5 * (first + second)))
    rotated = gaussian_w2(
        jnp.zeros(2),
        factor @ first @ factor.T,
        jnp.zeros(2),
        factor @ second @ factor.T,
    ).shape
    assert float(rotated) == pytest.approx(shape, rel=1e-10)


def test_pooled_w2_shape_of_correlated_covariances_obeys_its_relation() -> None:
    """For general covariances, the returned r reproduces the measured shape term."""
    first = jnp.array([[2.0, 0.5], [0.5, 1.0]])
    second = jnp.array([[3.0, 0.2], [0.2, 1.5]])
    one, two = pooled_pair(first, second)
    terms = gaussian_w2(jnp.zeros(2), one, jnp.zeros(2), two)
    mapped = float(r_eff("w2_shape", terms.shape, terms.log_det_ratio, pooled=True))
    assert pooled_w2_term(mapped) == pytest.approx(float(terms.shape), rel=1e-10)
    assert (mapped > 1) == (float(terms.log_det_ratio) > 0)


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


def test_default_bases_remove_the_direction_scatter_of_a_shift() -> None:
    """σ_eff of a 1σ shift is 1 on complete bases and scatters on independent ones."""
    size = 8
    space = ParamSpace.model_validate(
        {"parameters": [{"name": f"x{index}"} for index in range(size)]}
    )
    shift = np.zeros(size)
    shift[0] = 1.0
    root = jax.random.key(11)
    effects = {}
    for orthonormal in (True, False):
        values = [
            float(
                sliced_wasserstein_of_gaussians(
                    np.zeros(size),
                    np.eye(size),
                    shift,
                    np.eye(size),
                    directions=sliced_directions(
                        space,
                        SLICED_DIRECTIONS,
                        key=derive_key(root, "directions", index),
                        orthonormal=orthonormal,
                    ),
                )
            )
            for index in range(20)
        ]
        effects[orthonormal] = np.asarray(
            sigma_eff("sliced_w2", np.asarray(values), dimension=size)
        )
    np.testing.assert_allclose(effects[True], 1.0, atol=1e-9)
    # * Independent directions give a relative scatter of about 2.6% in σ_eff here.
    assert np.std(effects[False], ddof=1) > 0.01


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
    """At the limits of the metrics the map returns an unresolved shift beyond 10σ."""
    for metric, limit in (("js", math.log(2)), ("ad", 8 * math.log(2) - 4)):
        beyond = float(sigma_eff(metric, limit))
        # ? Float64 rounding noise sets where the bisection lands at the limit:
        # ? the AD value is 11.63 on the Linux runners and 18.78 on macOS arm64.
        assert 10.0 < beyond <= 20.0, metric
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


# * Wide circles and spheres

WIDTH_SAMPLE_COUNT = 20_000
WIDTH_KEY = jax.random.key(12)


def whitening_of(case: ZooCase) -> Whitening:
    """Return the pooled whitening of the two posteriors of a zoo case."""
    draws = case.sample(WIDTH_KEY, WIDTH_SAMPLE_COUNT)
    return Whitening.from_samples(draws.samples[0], None, case.space)


def test_chordal_metrics_and_limits_are_the_agreed_ones() -> None:
    """The five chordal metrics, 0.3 rad on a circle and 0.2 rad on a sphere."""
    assert CHORDAL_METRICS == (
        "energy",
        "mmd",
        "sinkhorn",
        "sliced_sinkhorn",
        "sliced_w2",
    )
    assert CHORDAL_CIRCLE_WIDTH == 0.3
    assert CHORDAL_SPHERE_WIDTH == 0.2


@pytest.mark.parametrize("metric", CHORDAL_METRICS)
def test_narrow_circles_and_spheres_give_no_width_warning(metric: str) -> None:
    """Widths of 0.1 rad on a circle and 0.05 rad on a sphere stay below the limits."""
    circle_case = wrapped_gaussians(1.0, width=0.1, param_count=2)
    sphere_case = von_mises_fisher(1.0, width=0.05, euclidean_count=1)
    for case in (circle_case, sphere_case):
        warn_wide_chordal_axes(metric, whitening_of(case), case.space)


@pytest.mark.parametrize(
    "metric", ["w1", "kl", "mean_shift", "ks", "kuiper_ks", "hellinger", "js", "ad"]
)
def test_other_metrics_give_no_width_warning(metric: str) -> None:
    """A circle of 1 rad and a sphere of 0.5 rad matter to the chordal metrics only."""
    circle_case = wrapped_gaussians(1.0, width=1.0)
    sphere_case = von_mises_fisher(1.0, width=0.5)
    for case in (circle_case, sphere_case):
        warn_wide_chordal_axes(metric, whitening_of(case), case.space)


@pytest.mark.parametrize("metric", CHORDAL_METRICS)
def test_a_wide_circle_gives_one_warning_that_names_it(metric: str) -> None:
    """A circular standard deviation of 0.5 rad is named with its width."""
    case = wrapped_gaussians(1.0, width=0.5, param_count=2)
    with pytest.warns(ValidityWarning) as record:
        warn_wide_chordal_axes(metric, whitening_of(case), case.space)
    assert len(record) == 1
    message = str(record[0].message)
    assert message.startswith(f"{metric}: ")
    assert "circle 'x0' (0.50 rad)" in message
    assert "x1" not in message


def test_a_wide_sphere_gives_one_warning_that_names_it() -> None:
    """An angular width of 0.3 rad on a sphere is named with its two axes."""
    case = von_mises_fisher(1.0, width=0.3, euclidean_count=1)
    with pytest.warns(ValidityWarning) as record:
        warn_wide_chordal_axes("sliced_w2", whitening_of(case), case.space)
    assert len(record) == 1
    assert "sphere ('azimuth', 'polar') (0.30 rad)" in str(record[0].message)


def test_wide_circles_and_spheres_share_one_warning() -> None:
    """A space with a wide circle, a wide sphere and a narrow circle warns once."""
    circle_case = wrapped_gaussians(1.0, width=0.5)
    sphere_case = von_mises_fisher(1.0, width=0.3)
    narrow_case = wrapped_gaussians(1.0, width=0.1)
    narrow_parameter = narrow_case.space.parameters[0].model_copy(
        update={"name": "phase"}
    )
    space = ParamSpace(
        parameters=(
            *circle_case.space.parameters,
            *sphere_case.space.parameters,
            narrow_parameter,
        ),
        spheres=sphere_case.space.spheres,
    )
    samples = jnp.concatenate(
        [
            case.sample(WIDTH_KEY, WIDTH_SAMPLE_COUNT).samples[0]
            for case in (circle_case, sphere_case, narrow_case)
        ],
        axis=-1,
    )
    whitening = Whitening.from_samples(samples, None, space)
    with pytest.warns(ValidityWarning) as record:
        warn_wide_chordal_axes("energy", whitening, space)
    assert len(record) == 1
    message = str(record[0].message)
    assert "circle 'x0' (0.50 rad)" in message
    assert "sphere ('azimuth', 'polar') (0.30 rad)" in message
    assert "phase" not in message


def test_a_circle_converts_its_width_to_radians_with_its_period() -> None:
    """A width of 0.2 in period 3 is 0.42 rad, and 0.4 in period 10 is 0.25 rad."""
    wide_in_radians = wrapped_gaussians(1.0, width=0.2, period=3.0, lower=-1.0)
    with pytest.warns(ValidityWarning, match=r"circle 'x0' \(0\.42 rad\)"):
        warn_wide_chordal_axes(
            "mmd", whitening_of(wide_in_radians), wide_in_radians.space
        )
    narrow_in_radians = wrapped_gaussians(1.0, width=0.4, period=10.0)
    warn_wide_chordal_axes(
        "mmd", whitening_of(narrow_in_radians), narrow_in_radians.space
    )


def test_a_fisher_whitening_gives_the_same_widths() -> None:
    """The scale of a circle is the root of its diagonal entry of the inverse Fisher."""
    space = ParamSpace.model_validate(
        {
            "parameters": [
                {"name": "phase", "topology": "circle", "bounds": [0.0, 2 * math.pi]},
                {"name": "amplitude"},
            ]
        }
    )
    centre = np.zeros(2)
    narrow = Whitening.from_fisher(np.diag([1 / 0.1**2, 1.0]), centre, space)
    warn_wide_chordal_axes("sliced_sinkhorn", narrow, space)
    wide = Whitening.from_fisher(np.diag([1 / 0.6**2, 1.0]), centre, space)
    with pytest.warns(ValidityWarning, match=r"circle 'phase' \(0\.60 rad\)"):
        warn_wide_chordal_axes("sliced_sinkhorn", wide, space)


def test_the_width_warning_needs_the_space_of_the_whitening() -> None:
    """A space with other axis blocks than the whitening raises ValueError."""
    circle_case = wrapped_gaussians(1.0, width=0.5)
    sphere_case = von_mises_fisher(1.0, width=0.1)
    with pytest.raises(ValueError, match="whitening was built for"):
        warn_wide_chordal_axes("energy", whitening_of(circle_case), sphere_case.space)
    # * Other metrics return before the check, as the row has nothing to warn about.
    warn_wide_chordal_axes("w1", whitening_of(circle_case), sphere_case.space)
