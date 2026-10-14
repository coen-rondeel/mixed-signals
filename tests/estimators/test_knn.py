"""Tests for the k-nearest-neighbour estimator."""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import spatial, special

from mixed_signals import zoo
from mixed_signals.core.space import ParamSpace
from mixed_signals.errors import ValidityWarning
from mixed_signals.estimators import NearestNeighbours, NearestNeighbourSettings

STANDARD_ERRORS = 5.0
TWO_PI = 2 * math.pi
REAL_TRIPLE = ParamSpace.model_validate(
    {"parameters": [{"name": "x0"}, {"name": "x1"}, {"name": "x2"}]}
)


def settings(neighbours: int) -> NearestNeighbourSettings:
    """Return kNN settings with the rank k of the neighbour."""
    return NearestNeighbourSettings(neighbours=neighbours)


# * Distances


def test_distances_match_a_kd_tree() -> None:
    """Euclidean distances equal SciPy's k-d tree, each sample skipping itself."""
    samples = np.random.default_rng(0).normal(size=(3_000, 3))
    fitted = NearestNeighbours.fit(samples, None, REAL_TRIPLE, settings(4))
    tree = spatial.cKDTree(samples)
    queries = samples[:200] + 0.05
    expected, _ = tree.query(queries, k=4)
    np.testing.assert_allclose(fitted.neighbour_distance(queries), expected[:, 3])
    own, _ = tree.query(samples, k=5)
    np.testing.assert_allclose(fitted.leave_one_out_distance(), own[:, 4])
    assert fitted.neighbour_distance(queries[0]).shape == ()


def test_circle_distances_wrap_around() -> None:
    """Circle differences take the shorter arc, as a periodic k-d tree does."""
    case = zoo.wrapped_gaussians(0.0, width=1.0, param_count=2)
    samples = np.asarray(case.sample(jax.random.key(1), 2_000).samples[0, 0])
    fitted = NearestNeighbours.fit(samples, None, case.space, settings(3))
    # * The real axis gets a box far wider than its spread, so it never wraps.
    shifted = samples - np.array([0.0, samples[:, 1].min() - 1.0])
    tree = spatial.cKDTree(shifted, boxsize=np.array([TWO_PI, 1e9]))
    own, _ = tree.query(shifted, k=4)
    np.testing.assert_allclose(fitted.leave_one_out_distance(), own[:, 3], atol=1e-12)


def test_sphere_distances_are_great_circle_angles() -> None:
    """On a sphere in latitude, distances are the angles between the directions."""
    case = zoo.von_mises_fisher(0.0, width=0.3, colatitude=0.2)
    samples = np.asarray(case.sample(jax.random.key(2), 400).samples[0, 0])
    fitted = NearestNeighbours.fit(samples, None, case.space, settings(2))
    azimuth, latitude = samples[:, 0], samples[:, 1]
    vectors = np.c_[
        np.cos(latitude) * np.cos(azimuth),
        np.cos(latitude) * np.sin(azimuth),
        np.sin(latitude),
    ]
    angles = np.arccos(np.clip(vectors @ vectors.T, -1.0, 1.0))
    np.fill_diagonal(angles, np.inf)
    expected = np.sort(angles, axis=1)[:, 1]
    np.testing.assert_allclose(fitted.leave_one_out_distance(), expected, atol=1e-7)


def test_dimension_counts_each_sphere_twice() -> None:
    """D counts Euclidean axes and circles once and every sphere twice."""
    sphere = zoo.von_mises_fisher(0.0, euclidean_count=1)
    fitted = NearestNeighbours.fit(
        sphere.sample(jax.random.key(3), 50).samples[0, 0],
        None,
        sphere.space,
        settings(1),
    )
    assert fitted.dimension == 3
    circle = zoo.wrapped_gaussians(0.0, param_count=2)
    fitted = NearestNeighbours.fit(
        circle.sample(jax.random.key(3), 50).samples[0, 0],
        None,
        circle.space,
        settings(1),
    )
    assert fitted.dimension == 2


# * Log densities


def test_log_density_is_the_digamma_corrected_estimate() -> None:
    """Log p̂ = ψ(k) − ψ(n + 1) − ln V_d − d ln r_k, with n − 1 others at a sample."""
    samples = np.random.default_rng(4).normal(size=(500, 3))
    fitted = NearestNeighbours.fit(samples, None, REAL_TRIPLE, settings(5))
    query = np.array([[0.3, -0.2, 0.1]])
    distance = float(fitted.neighbour_distance(query)[0])
    log_unit_ball = 1.5 * math.log(math.pi) - special.gammaln(2.5)
    expected = (
        special.digamma(5)
        - special.digamma(501)
        - log_unit_ball
        - 3 * math.log(distance)
    )
    assert float(fitted.log_density(query)[0]) == pytest.approx(expected, rel=1e-12)
    own = np.asarray(fitted.leave_one_out_distance())
    expected_own = (
        special.digamma(5) - special.digamma(500) - log_unit_ball - 3 * np.log(own)
    )
    np.testing.assert_allclose(
        fitted.leave_one_out_log_density(), expected_own, rtol=1e-12
    )


def test_gaussian_entropy_is_recovered() -> None:
    """The mean negated leave-one-out log density is the Gaussian's entropy."""
    covariance = np.array([[2.0, 0.6], [0.6, 0.5]])
    case = zoo.gaussians([[0.0, 1.0]], [covariance], names=["only"])
    batch = case.sample(jax.random.key(5), 5_000, batch_size=8).samples[:, 0]
    entropies = np.asarray(
        jax.vmap(
            lambda values: (
                -jnp.mean(
                    NearestNeighbours.fit(
                        values, None, case.space, settings(4)
                    ).leave_one_out_log_density()
                )
            )
        )(batch)
    )
    truth = 0.5 * math.log(np.linalg.det(2 * math.pi * math.e * covariance))
    error = entropies.std(ddof=1) / math.sqrt(len(entropies))
    assert abs(entropies.mean() - truth) < STANDARD_ERRORS * error


@pytest.mark.parametrize("polar_form", ["latitude", "sin_latitude"])
def test_sphere_entropy_includes_the_area_element(polar_form: str) -> None:
    """In either polar form, the estimate refers to the coordinates."""
    case = zoo.von_mises_fisher(0.0, width=0.3, colatitude=0.5, polar_form=polar_form)
    samples = case.sample(jax.random.key(6), 10_000).samples[0, 0]
    fitted = NearestNeighbours.fit(samples, None, case.space, settings(4))
    estimate = -float(jnp.mean(fitted.leave_one_out_log_density()))
    true_log = np.asarray(case.log_prob(samples)[:, 0])
    truth = -true_log.mean()
    error = true_log.std() / math.sqrt(len(true_log))
    assert abs(estimate - truth) < 2 * STANDARD_ERRORS * error


def test_points_off_the_sphere_have_zero_density() -> None:
    """A latitude beyond the pole is not on the sphere."""
    case = zoo.von_mises_fisher(0.0, width=0.3)
    fitted = NearestNeighbours.fit(
        case.sample(jax.random.key(7), 200).samples[0, 0], None, case.space, settings(2)
    )
    assert float(fitted.log_density(np.array([0.1, 1.7]))) == -math.inf


# * Weights and validity


def test_zero_weights_leave_every_search() -> None:
    """Samples of zero weight count as absent, as for padded sets."""
    samples = np.random.default_rng(8).normal(size=(400, 3))
    log_weights = np.where(np.arange(400) % 4 == 0, -np.inf, 0.0)
    padded = NearestNeighbours.fit(samples, log_weights, REAL_TRIPLE, settings(3))
    kept = samples[log_weights == 0]
    plain = NearestNeighbours.fit(kept, None, REAL_TRIPLE, settings(3))
    queries = samples[:10] + 0.01
    np.testing.assert_allclose(padded.log_density(queries), plain.log_density(queries))
    np.testing.assert_allclose(
        np.asarray(padded.leave_one_out_log_density())[log_weights == 0],
        plain.leave_one_out_log_density(),
    )


def test_unequal_weights_give_nan_and_warn() -> None:
    """Unequal positive weights are a hard limit of the estimators."""
    samples = np.random.default_rng(9).normal(size=(50, 3))
    with pytest.warns(ValidityWarning, match="equally weighted samples"):
        fitted = NearestNeighbours.fit(
            samples, np.linspace(0.0, 1.0, 50), REAL_TRIPLE, settings(2)
        )
    assert bool(jnp.all(jnp.isnan(fitted.leave_one_out_distance())))
    assert math.isnan(float(fitted.neighbour_distance(np.zeros(3))))


def test_too_few_samples_give_nan_and_warn() -> None:
    """At least k + 1 samples need a positive weight."""
    samples = np.random.default_rng(10).normal(size=(6, 3))
    with pytest.warns(ValidityWarning, match="more than k samples"):
        fitted = NearestNeighbours.fit(
            samples,
            np.array([0.0, 0.0, 0.0, -np.inf, -np.inf, -np.inf]),
            REAL_TRIPLE,
            settings(3),
        )
    assert math.isnan(float(fitted.log_density(np.zeros(3))))
    with pytest.warns(ValidityWarning, match="more than k samples"):
        larger = NearestNeighbours.fit(samples, None, REAL_TRIPLE, settings(8))
    assert bool(jnp.all(jnp.isnan(larger.leave_one_out_distance())))
    assert math.isnan(float(larger.neighbour_distance(np.zeros(3))))


def test_coinciding_samples_give_nan_and_warn() -> None:
    """Repeated states leave a zero k-th distance and no log density."""
    rng = np.random.default_rng(11)
    samples = np.r_[
        np.repeat(rng.normal(size=(1, 3)), 3, axis=0), rng.normal(size=(40, 3))
    ]
    fitted = NearestNeighbours.fit(samples, None, REAL_TRIPLE, settings(2))
    assert float(fitted.leave_one_out_distance()[0]) == 0.0
    with pytest.warns(ValidityWarning, match="zero for 3 samples"):
        values = fitted.leave_one_out_log_density()
    assert bool(jnp.all(jnp.isnan(values[:3])))
    assert bool(jnp.all(jnp.isfinite(values[3:])))
    with pytest.warns(ValidityWarning, match="zero for 1 points"):
        assert math.isnan(float(fitted.log_density(samples[0])))


# * Transformations


def test_vmap_and_jit_give_the_concrete_results() -> None:
    """Batched fits and compiled evaluations match concrete calls."""
    batch = np.random.default_rng(12).normal(size=(3, 300, 3))
    batched = jax.vmap(
        lambda values: NearestNeighbours.fit(values, None, REAL_TRIPLE, settings(3))
    )(batch)
    concrete = NearestNeighbours.fit(batch[1], None, REAL_TRIPLE, settings(3))
    distances = jax.vmap(lambda estimator: estimator.leave_one_out_distance())(batched)
    np.testing.assert_allclose(distances[1], concrete.leave_one_out_distance())
    compiled = jax.jit(lambda estimator, points: estimator.log_density(points))
    np.testing.assert_allclose(
        compiled(concrete, batch[0, :5]), concrete.log_density(batch[0, :5])
    )
    assert concrete.settings.to_json() == '{"neighbours":3}'
