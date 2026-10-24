"""Tests for the closed-form σ_eff and r_eff maps."""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import special

from mixed_signals.metrics.moments import gaussian_hellinger, gaussian_kl, gaussian_w2
from mixed_signals.sigma.maps import SIGMA_EFF_MAPS, r_eff, sigma_eff
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


def test_sigma_eff_is_signed_for_the_mean_shift_only() -> None:
    """A negative shift keeps its sign; a negative divergence has no shift."""
    assert float(sigma_eff("mean_shift", -0.7)) == -0.7
    for metric in ("kl", "w1", "w2", "ks", "hellinger"):
        assert math.isnan(float(sigma_eff(metric, -0.1))), metric
    assert math.isnan(float(sigma_eff("ks", 1.2)))
    assert math.isnan(float(sigma_eff("hellinger", 1.2)))
    assert float(sigma_eff("ks", 1.0)) == math.inf


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


def test_r_eff_outside_the_range_is_nan() -> None:
    """Negative values, a W2 shape term beyond one for the narrower side, H ≥ 1."""
    assert math.isnan(float(r_eff("kl_shape", -0.1, 1.0)))
    assert math.isnan(float(r_eff("w2_shape", 1.2, -1.0)))
    assert float(r_eff("w2_shape", 1.2, 1.0)) == pytest.approx(2.2)
    assert math.isnan(float(r_eff("hellinger_shape", 1.0, 1.0)))
    assert math.isnan(float(r_eff("width_ratio", 0.0)))


def test_maps_check_their_names_and_directions() -> None:
    """Unknown metrics name the maps that exist; shape terms need a direction."""
    with pytest.raises(KeyError, match="no closed-form sigma_eff map for 'ad'"):
        sigma_eff("ad", 0.1)
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
