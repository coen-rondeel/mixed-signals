"""Tests for the shared estimator parts: settings, factors, checks and images."""

import math
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from pydantic import ValidationError

from mixed_signals.core.space import ParamSpace
from mixed_signals.errors import PrecisionWarning, ValidityWarning
from mixed_signals.estimators import (
    IMAGE_REACH,
    RAYLEIGH_THRESHOLD,
    Estimator,
    GaussianDensity,
    GaussianSettings,
    KernelDensitySettings,
    MixtureSettings,
    NearestNeighbourSettings,
    factor_covariance,
)
from mixed_signals.estimators.base import (
    _covers,
    _image_count,
    _image_table,
    _map_in_chunks,
    _read_points,
    _read_samples,
)

SPACE = ParamSpace.model_validate({"parameters": [{"name": "x0"}, {"name": "x1"}]})


# * Settings


def test_settings_name_their_estimator_and_write_json() -> None:
    """Each settings class names its estimator and serialises to a JSON object."""
    assert GaussianSettings.name == "gaussian"
    assert KernelDensitySettings.name == "kde"
    assert NearestNeighbourSettings.name == "knn"
    assert MixtureSettings.name == "gmm"
    assert KernelDensitySettings().to_json() == '{"bandwidth":"silverman"}'
    assert NearestNeighbourSettings(neighbours=5).to_json() == '{"neighbours":5}'


def test_settings_are_frozen_and_reject_unknown_fields() -> None:
    """Settings cannot change after construction or take fields they do not have."""
    settings = KernelDensitySettings(bandwidth=0.3)
    with pytest.raises(ValidationError):
        settings.bandwidth = 0.5  # type: ignore[misc]
    with pytest.raises(ValidationError):
        KernelDensitySettings(kernel="tophat")  # type: ignore[call-arg]
    with pytest.raises(ValidationError):
        KernelDensitySettings(bandwidth=-1.0)
    with pytest.raises(ValidationError):
        NearestNeighbourSettings(neighbours=0)


def test_the_estimator_base_is_abstract() -> None:
    """Estimator only describes the interface; fitted estimators subclass it."""
    with pytest.raises(TypeError):
        Estimator(settings=GaussianSettings())  # type: ignore[abstract]


def test_default_resolved_settings_are_the_settings() -> None:
    """Estimators without data-chosen settings resolve to their own settings."""
    fitted = GaussianDensity.fit(
        np.random.default_rng(0).normal(size=(50, 2)), None, SPACE
    )
    assert fitted.resolved_settings() == GaussianSettings()


# * Covariance factors


def test_factor_matches_cholesky_and_log_determinant() -> None:
    """The factor and log-determinant of a regular covariance are the usual ones."""
    covariance = jnp.array([[2.0, 0.6, 0.1], [0.6, 1.0, -0.3], [0.1, -0.3, 0.5]])
    factor = factor_covariance(covariance)
    np.testing.assert_allclose(
        factor.cholesky, np.linalg.cholesky(np.asarray(covariance)), rtol=1e-12
    )
    assert float(factor.log_determinant) == pytest.approx(
        float(np.linalg.slogdet(np.asarray(covariance))[1]), rel=1e-12
    )


def test_factor_keeps_gravitational_wave_scales() -> None:
    """Widths of 1e-11 and 1e-23 with correlation 0.9 factor exactly."""
    widths = jnp.array([1e-11, 1e-23])
    correlation = jnp.array([[1.0, 0.9], [0.9, 1.0]])
    covariance = correlation * jnp.outer(widths, widths)
    factor = factor_covariance(covariance)
    expected = 2 * math.log(1e-11) + 2 * math.log(1e-23) + math.log(1 - 0.81)
    assert float(factor.log_determinant) == pytest.approx(expected, rel=1e-14)
    np.testing.assert_allclose(
        factor.cholesky @ factor.cholesky.T, covariance, rtol=1e-12, atol=0.0
    )


@pytest.mark.parametrize(
    "covariance",
    [
        [[1.0, 0.0], [0.0, 0.0]],
        [[1.0, 1.0], [1.0, 1.0]],
        [[math.nan, 0.0], [0.0, 1.0]],
    ],
    ids=["no spread", "collinear", "nan"],
)
def test_singular_covariances_give_nan(covariance: list[list[float]]) -> None:
    """A singular covariance has a NaN factor and log-determinant."""
    factor = factor_covariance(jnp.asarray(covariance))
    assert math.isnan(float(factor.log_determinant))
    assert bool(jnp.all(jnp.isnan(factor.cholesky)))


def test_factor_maps_over_leading_axes() -> None:
    """Stacked covariances factor one by one."""
    stacked = jnp.stack([jnp.eye(2), 4 * jnp.eye(2)])
    factor = factor_covariance(stacked)
    np.testing.assert_allclose(factor.log_determinant, [0.0, 2 * math.log(4.0)])


# * Input checks


def test_read_samples_casts_and_defaults_the_weights() -> None:
    """Integer samples become float64, and missing weights become zeros."""
    samples, log_weights = _read_samples(np.array([[1, 2], [3, 4]]), None, SPACE)
    assert samples.dtype == jnp.float64
    np.testing.assert_array_equal(log_weights, [0.0, 0.0])


@pytest.mark.parametrize(
    ("samples", "log_weights", "message"),
    [
        (np.zeros((3, 3)), None, r"shaped \(N, 2\)"),
        (np.zeros(3), None, r"shaped \(N, 2\)"),
        (np.zeros((0, 2)), None, "at least one sample"),
        (np.zeros((3, 2)), np.zeros(4), r"log_weights must be shaped \(3,\)"),
        (np.array([[0.0, math.inf], [0.0, 0.0]]), None, "must be finite"),
        (np.zeros((2, 2)), np.array([0.0, math.nan]), "NaN or \\+inf"),
        (np.zeros((2, 2)), np.array([-math.inf, -math.inf]), "positive weight"),
    ],
)
def test_read_samples_rejects_invalid_inputs(
    samples: np.ndarray, log_weights: np.ndarray | None, message: str
) -> None:
    """Shapes always, and values of concrete inputs, are checked."""
    with pytest.raises(ValueError, match=message):
        _read_samples(samples, log_weights, SPACE)


def test_read_samples_rejects_complex_input() -> None:
    """Samples must be real numbers."""
    with pytest.raises(TypeError, match="real numbers"):
        _read_samples(np.zeros((2, 2), dtype=complex), None, SPACE)
    with pytest.raises(TypeError, match="real numbers"):
        _read_samples(np.zeros((2, 2)), np.zeros(2, dtype=complex), SPACE)


def test_read_samples_warns_about_narrow_floats() -> None:
    """float32 inputs are cast with a PrecisionWarning naming each field."""
    with pytest.warns(PrecisionWarning, match=r"samples \(float32\), log_weights"):
        _read_samples(
            np.zeros((2, 2), dtype=np.float32), np.zeros(2, dtype=np.float32), SPACE
        )


def test_read_samples_skips_value_checks_inside_jit() -> None:
    """Inside jit the values are unknown, so invalid ones pass through."""
    compiled = jax.jit(lambda values: _read_samples(values, None, SPACE)[0])
    result = compiled(jnp.array([[math.nan, 0.0], [0.0, 0.0]]))
    assert bool(jnp.isnan(result[0, 0]))


def test_read_points_checks_the_parameter_axis() -> None:
    """Points end in the D parameters and are real."""
    assert _read_points(np.array([[1, 2]]), 2).dtype == jnp.float64
    with pytest.raises(ValueError, match=r"\(\.\.\., 2\)"):
        _read_points(np.zeros(3), 2)
    with pytest.raises(ValueError, match=r"\(\.\.\., 2\)"):
        _read_points(np.float64(1.0), 2)
    with pytest.raises(TypeError, match="real numbers"):
        _read_points(np.zeros((1, 2), dtype=complex), 2)


# * Images and chunks


def test_image_count_covers_the_reach() -> None:
    """Concrete widths give the smallest count; traced or NaN widths the bound's."""
    assert _image_count(jnp.asarray(0.1), 2 * math.pi, 3.0, 1.0) == 1
    assert _image_count(jnp.asarray(2.0), 1.0, 3.0, 1.0) == 24
    assert _image_count(jnp.asarray(math.nan), 1.0, 0.5, 1.0) == 6
    assert _image_count(jnp.asarray(2.0), 1.0, 3.0, 2.0) == 12
    counts: list[int] = []

    def record(width: jax.Array) -> jax.Array:
        counts.append(_image_count(width, 1.0, 0.5, 1.0))
        return width

    jax.jit(record)(jnp.asarray(0.01))
    assert counts == [math.ceil(IMAGE_REACH * 0.5)]


def test_covers_compares_the_reach_with_the_images() -> None:
    """Images cover a width while the reach fits within count times the step."""
    assert bool(_covers(jnp.asarray(0.5), 1.0, 6, 1.0))
    assert not bool(_covers(jnp.asarray(0.51), 1.0, 6, 1.0))
    assert bool(_covers(jnp.asarray(1.0), 1.0, 6, 2.0))


def test_image_table_combines_the_images_of_each_axis() -> None:
    """The images of a space are every combination of the images of its axes."""
    signs, shifts = _image_table([[(1.0, 0.0), (-1.0, 2.0)], [(1.0, -3.0), (1.0, 3.0)]])
    assert signs == ((1.0, 1.0), (1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0))
    assert shifts == ((0.0, -3.0), (0.0, 3.0), (2.0, -3.0), (2.0, 3.0))


@pytest.mark.parametrize("chunk_size", [1, 3, 7, 50])
def test_map_in_chunks_equals_the_direct_evaluation(chunk_size: int) -> None:
    """Chunks, padding included, give the values of one direct call."""
    points = jnp.arange(14.0).reshape(7, 2)
    result = _map_in_chunks(
        lambda chunk: jnp.sum(chunk**2, axis=-1), points, chunk_size
    )
    np.testing.assert_array_equal(result, jnp.sum(points**2, axis=-1))


def test_map_in_chunks_of_no_points_is_empty() -> None:
    """No points give no values."""
    assert _map_in_chunks(lambda chunk: chunk[:, 0], jnp.zeros((0, 2)), 4).shape == (0,)


# * Circles


def test_unresolved_circles_warn_with_their_rayleigh_statistic() -> None:
    """Two opposite clusters leave the mean direction unresolved."""
    space = ParamSpace.model_validate(
        {"parameters": [{"name": "phase", "topology": "circle", "bounds": [0.0, 6.0]}]}
    )
    rng = np.random.default_rng(1)
    values = np.concatenate([rng.normal(1.0, 0.1, 500), rng.normal(4.0, 0.1, 500)])
    with pytest.warns(ValidityWarning, match=r"'phase' \(.*\).*Rayleigh"):
        GaussianDensity.fit(values[:, None], None, space)
    resolved = rng.normal(1.0, 0.5, 1_000)[:, None]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        GaussianDensity.fit(resolved, None, space)
    assert math.isclose(RAYLEIGH_THRESHOLD, 5.991_464_547, rel_tol=1e-9)
