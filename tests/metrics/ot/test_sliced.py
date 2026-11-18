"""Tests for sliced W2, the sliced Sinkhorn divergence and their directions."""

import math
from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.typing import ArrayLike

from mixed_signals import zoo
from mixed_signals.core.space import ParamSpace
from mixed_signals.errors import SubsamplingWarning, ValidityWarning
from mixed_signals.metrics._cap import _draw
from mixed_signals.metrics.ot.sinkhorn import (
    SINKHORN_EPSILON_FACTOR,
    sinkhorn_divergence,
)
from mixed_signals.metrics.ot.sliced import (
    SLICED_DIRECTIONS,
    SLICED_SINKHORN_CAP,
    SLICED_SINKHORN_DIRECTIONS,
    gaussian_sliced_wasserstein_distance,
    sliced_directions,
    sliced_sinkhorn_divergence,
    sliced_sinkhorn_epsilon,
    sliced_wasserstein_distance,
    sliced_wasserstein_of_gaussians,
)
from mixed_signals.nulls.permutation import permutation_null
from mixed_signals.nulls.plan import null_draws
from mixed_signals.prng import derive_key
from mixed_signals.sigma.maps import null_subtract
from mixed_signals.zoo import closed_forms

STANDARD_ERRORS = 5.0
TWO_PI = 2 * math.pi
KEY = jax.random.key(0)
LINE = ParamSpace.model_validate({"parameters": [{"name": "x0"}]})
PLANE = ParamSpace.model_validate({"parameters": [{"name": "x0"}, {"name": "x1"}]})
TRIPLE = ParamSpace.model_validate(
    {"parameters": [{"name": "x0"}, {"name": "x1"}, {"name": "x2"}]}
)
# * A real axis, a circle of period 3 and a sky sphere of radius 2 in latitude.
MIXED = ParamSpace.model_validate(
    {
        "parameters": [
            {"name": "x0"},
            {"name": "phase", "topology": "circle", "bounds": [-1.0, 2.0]},
            {"name": "ra", "topology": "circle", "bounds": [0.0, TWO_PI]},
            {
                "name": "dec",
                "topology": "interval",
                "bounds": [-0.5 * math.pi, 0.5 * math.pi],
            },
        ],
        "spheres": [
            {"azimuth": "ra", "polar": "dec", "polar_form": "latitude", "radius": 2.0}
        ],
    }
)


@pytest.fixture
def rng() -> np.random.Generator:
    """A seeded NumPy generator for test data."""
    return np.random.default_rng(2026)


def mixed_samples(rng: np.random.Generator, count: int, offset: float) -> np.ndarray:
    """Return samples of the mixed space, shifted by ``offset`` on every axis."""
    return np.stack(
        [
            rng.normal(offset, 1.0, count),
            (rng.normal(offset, 0.4, count) + 1.0) % 3.0 - 1.0,
            (rng.normal(1.0 + offset, 0.3, count)) % TWO_PI,
            np.clip(rng.normal(0.2 + 0.1 * offset, 0.2, count), -1.5, 1.5),
        ],
        axis=1,
    )


def embedded(samples: np.ndarray) -> np.ndarray:
    """Return the chordal embedding of samples of the mixed space, independently."""
    angle = TWO_PI * (samples[:, 1] + 1.0) / 3.0
    radius = 3.0 / TWO_PI
    azimuth, latitude = samples[:, 2], samples[:, 3]
    return np.stack(
        [
            samples[:, 0],
            radius * np.cos(angle),
            radius * np.sin(angle),
            2.0 * np.cos(latitude) * np.cos(azimuth),
            2.0 * np.cos(latitude) * np.sin(azimuth),
            2.0 * np.sin(latitude),
        ],
        axis=1,
    )


def w2_squared(
    first: np.ndarray,
    first_weights: np.ndarray,
    second: np.ndarray,
    second_weights: np.ndarray,
) -> float:
    """Return W2² between weighted one-dimensional samples from their quantiles."""
    first_order, second_order = np.argsort(first), np.argsort(second)
    first_sorted = first[first_order]
    second_sorted = second[second_order]
    first_levels = np.cumsum(first_weights[first_order] / first_weights.sum())
    second_levels = np.cumsum(second_weights[second_order] / second_weights.sum())
    levels = np.unique(np.concatenate([first_levels, second_levels]))
    lower = np.concatenate([[0.0], levels[:-1]])
    middle = 0.5 * (lower + levels)
    first_index = np.minimum(np.searchsorted(first_levels, middle), len(first) - 1)
    second_index = np.minimum(np.searchsorted(second_levels, middle), len(second) - 1)
    gaps = first_sorted[first_index] - second_sorted[second_index]
    return float(np.sum((levels - lower) * gaps**2))


# * Directions


def space_of(size: int) -> ParamSpace:
    """Return a space of ``size`` real axes."""
    return ParamSpace.model_validate(
        {"parameters": [{"name": f"x{index}"} for index in range(size)]}
    )


def test_independent_directions_are_unit_vectors_of_the_embedded_space() -> None:
    """One coordinate per embedded coordinate, unit length, fixed by the key."""
    directions = sliced_directions(MIXED, 4_000, key=KEY, orthonormal=False)
    assert directions.shape == (4_000, 6)
    np.testing.assert_allclose(np.linalg.norm(directions, axis=1), 1.0, rtol=1e-12)
    np.testing.assert_array_equal(
        directions, sliced_directions(MIXED, 4_000, key=KEY, orthonormal=False)
    )
    other = sliced_directions(MIXED, 4_000, key=jax.random.key(1), orthonormal=False)
    assert not np.allclose(directions, other)
    # * Uniform directions have E[θθᵀ] = I/n; 4 000 draws resolve it to about 0.01.
    np.testing.assert_allclose(
        np.asarray(directions).T @ np.asarray(directions) / 4_000,
        np.eye(6) / 6,
        atol=0.02,
    )


def test_independent_directions_reproduce_the_earlier_draw_bit_for_bit() -> None:
    """orthonormal=False normalises standard normals of the key, as before."""
    normals = jax.random.normal(KEY, (37, 6))
    expected = normals / jnp.linalg.norm(normals, axis=1, keepdims=True)
    directions = sliced_directions(MIXED, 37, key=KEY, orthonormal=False)
    assert directions.shape == (37, 6)
    np.testing.assert_array_equal(directions, expected)


def test_default_directions_are_complete_orthonormal_bases() -> None:
    """⌈count/n⌉ bases of n unit rows, each block with a Gram matrix equal to I."""
    for count, bases in ((50, 9), (6, 1), (7, 2), (1, 1)):
        directions = np.asarray(sliced_directions(MIXED, count, key=KEY))
        assert directions.shape == (bases * 6, 6)
        for block in directions.reshape(bases, 6, 6):
            np.testing.assert_allclose(block @ block.T, np.eye(6), atol=1e-12)
            np.testing.assert_allclose(block.T @ block, np.eye(6), atol=1e-12)
    np.testing.assert_array_equal(
        sliced_directions(MIXED, 50, key=KEY), sliced_directions(MIXED, 50, key=KEY)
    )
    other = sliced_directions(MIXED, 50, key=jax.random.key(1))
    assert not np.allclose(sliced_directions(MIXED, 50, key=KEY), other)


def test_default_directions_use_the_embedded_size() -> None:
    """Circles count two coordinates and spheres three; a line has one per basis."""
    sphere_only = ParamSpace.model_validate(
        {
            "parameters": [
                {"name": "ra", "topology": "circle", "bounds": [0.0, TWO_PI]},
                {
                    "name": "dec",
                    "topology": "interval",
                    "bounds": [-0.5 * math.pi, 0.5 * math.pi],
                },
            ],
            "spheres": [{"azimuth": "ra", "polar": "dec", "polar_form": "latitude"}],
        }
    )
    assert sliced_directions(sphere_only, 4, key=KEY).shape == (6, 3)
    assert sliced_directions(MIXED, 4, key=KEY).shape == (6, 6)
    line = np.asarray(sliced_directions(LINE, 5, key=KEY))
    assert line.shape == (5, 1)
    np.testing.assert_allclose(np.abs(line), 1.0, rtol=1e-12)


def test_the_default_counts_round_up_to_complete_bases() -> None:
    """The documented counts: 512 directions of sliced W2, 32 of its Sinkhorn form."""
    for size in (16, 32):
        space = space_of(size)
        wide = sliced_directions(space, SLICED_DIRECTIONS, key=KEY)
        narrow = sliced_directions(space, SLICED_SINKHORN_DIRECTIONS, key=KEY)
        assert wide.shape == (512, size)
        assert narrow.shape == (32, size)


def test_complete_bases_project_a_vector_to_its_squared_length_over_n(
    rng: np.random.Generator,
) -> None:
    """The mean squared projection of any fixed vector is its squared length over n."""
    for space, size in ((MIXED, 6), (TRIPLE, 3), (space_of(8), 8)):
        directions = np.asarray(sliced_directions(space, 100, key=KEY))
        for _ in range(3):
            vector = rng.normal(size=size)
            mean_square = np.mean((directions @ vector) ** 2)
            assert mean_square == pytest.approx(vector @ vector / size, abs=1e-12)


def test_single_directions_of_the_bases_are_uniform_on_the_sphere() -> None:
    """The first and last row of many bases have the moments of a uniform direction."""
    size, count = 4, 20_000
    directions = np.asarray(sliced_directions(space_of(size), count * size, key=KEY))
    bases = directions.reshape(count, size, size)
    # * Each row is independent across bases; the standard errors below are 5 sigma.
    for rows in (bases[:, 0, :], bases[:, -1, :]):
        np.testing.assert_allclose(
            rows.mean(axis=0), 0.0, atol=5 * 0.5 / math.sqrt(count)
        )
        np.testing.assert_allclose(
            rows.T @ rows / count, np.eye(size) / size, atol=0.01
        )
        # * A uniform direction has E[θ_i⁴] = 3/(n(n + 2)).
        np.testing.assert_allclose(
            (rows**4).mean(axis=0), 3 / (size * (size + 2)), atol=0.008
        )


def test_sliced_directions_reject_a_count_below_one() -> None:
    """Both forms need at least one direction."""
    for orthonormal in (True, False):
        for count in (0, -3):
            with pytest.raises(ValueError, match="count must be at least 1"):
                sliced_directions(MIXED, count, key=KEY, orthonormal=orthonormal)


# * Sliced W2


def test_sliced_w2_matches_independent_quantile_functions(
    rng: np.random.Generator,
) -> None:
    """Weighted samples of unequal size in the chordal embedding of the space."""
    first, second = mixed_samples(rng, 300, 0.0), mixed_samples(rng, 220, 0.3)
    first_weights = rng.uniform(0.1, 2.0, 300)
    second_weights = rng.uniform(0.1, 2.0, 220)
    directions = sliced_directions(MIXED, 50, key=KEY)
    value = sliced_wasserstein_distance(
        first,
        np.log(first_weights),
        second,
        np.log(second_weights),
        MIXED,
        directions=directions,
    )
    projections = np.asarray(directions)
    expected = np.mean(
        [
            w2_squared(
                embedded(first) @ direction,
                first_weights,
                embedded(second) @ direction,
                second_weights,
            )
            for direction in projections
        ]
    )
    assert float(value) == pytest.approx(expected, rel=1e-12)


def test_a_shift_gives_the_mean_squared_projection(rng: np.random.Generator) -> None:
    """SW2²(p, p + Δ) is the mean of ⟨θ, Δ⟩² over the directions."""
    first = rng.normal(size=(400, 3))
    shift = np.array([0.5, -0.2, 0.1])
    directions = sliced_directions(TRIPLE, 500, key=KEY)
    value = sliced_wasserstein_distance(
        first, None, first + shift, None, TRIPLE, directions=directions
    )
    expected = np.mean((np.asarray(directions) @ shift) ** 2)
    assert float(value) == pytest.approx(expected, rel=1e-12)


def test_sliced_w2_weights_count_and_remove_samples(rng: np.random.Generator) -> None:
    """A weight of k counts like k copies, and a weight of zero removes a sample."""
    first, second = rng.normal(size=(60, 2)), rng.normal(0.5, 1.0, size=(50, 2))
    directions = sliced_directions(PLANE, 30, key=KEY)
    copies = rng.integers(1, 4, size=60)
    weighted = sliced_wasserstein_distance(
        first, np.log(copies), second, None, PLANE, directions=directions
    )
    repeated = sliced_wasserstein_distance(
        np.repeat(first, copies, axis=0),
        None,
        second,
        None,
        PLANE,
        directions=directions,
    )
    assert float(weighted) == pytest.approx(float(repeated), rel=1e-12)
    keep = np.arange(60) % 4 != 0
    with np.errstate(divide="ignore"):
        removed = sliced_wasserstein_distance(
            first,
            np.log(keep.astype(float)),
            second,
            None,
            PLANE,
            directions=directions,
        )
    dropped = sliced_wasserstein_distance(
        first[keep], None, second, None, PLANE, directions=directions
    )
    assert float(removed) == pytest.approx(float(dropped), rel=1e-12)


def test_scaled_gaussians_recover_the_zoo_value() -> None:
    """Null-corrected SW2² of a width ratio of 1.3 averages to the zoo's sliced_w2 row.

    In two dimensions, evenly spaced directions average over the circle of
    directions exactly, so no Monte Carlo error over directions remains.
    """
    case = zoo.scaled_gaussians(1.3)
    draws = np.asarray(case.sample(jax.random.key(5), 5_000, batch_size=8).samples)
    others = np.asarray(case.sample(jax.random.key(6), 5_000, batch_size=8).samples)
    angles = np.linspace(0.0, math.pi, 180, endpoint=False)
    directions = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    excess = [
        float(
            sliced_wasserstein_distance(
                problem[0], None, problem[1], None, case.space, directions=directions
            )
            - sliced_wasserstein_distance(
                problem[0], None, other[0], None, case.space, directions=directions
            )
        )
        for problem, other in zip(draws, others, strict=True)
    ]
    truth = case.reference_value("sliced_w2", ("reference", "scaled"))
    assert truth == pytest.approx(
        float(closed_forms.gaussian_sliced_w2(0.0, 1.3, 1, 2))
    )
    error = np.std(excess, ddof=1) / math.sqrt(len(excess))
    assert abs(np.mean(excess) - truth) < STANDARD_ERRORS * error


def test_sliced_w2_checks_its_directions(rng: np.random.Generator) -> None:
    """Directions need one coordinate per embedded coordinate of the space."""
    samples = rng.normal(size=(20, 2))
    with pytest.raises(ValueError, match=r"directions must be shaped \(L, 2\)"):
        sliced_wasserstein_distance(
            samples, None, samples, None, PLANE, directions=np.ones((5, 3))
        )


# * Sliced W2 between Gaussians


def random_covariance(rng: np.random.Generator, size: int) -> np.ndarray:
    """Return a random symmetric positive definite matrix of a given size."""
    factor = rng.normal(size=(size, size))
    return factor @ factor.T + 0.5 * size * np.eye(size)


def projected_w2_squared(
    mean_first: np.ndarray,
    covariance_first: np.ndarray,
    mean_second: np.ndarray,
    covariance_second: np.ndarray,
    direction: np.ndarray,
) -> float:
    """Return W2² of the normals that a direction projects two Gaussians to."""
    terms = closed_forms.gaussian_w2(
        np.array([direction @ mean_first]),
        np.array([[direction @ covariance_first @ direction]]),
        np.array([direction @ mean_second]),
        np.array([[direction @ covariance_second @ direction]]),
    )
    return float(terms.total) ** 2


def test_gaussian_sliced_w2_is_the_mean_w2_squared_of_the_projected_normals(
    rng: np.random.Generator,
) -> None:
    """Each direction contributes the zoo's one-dimensional Gaussian W2 squared."""
    size = 5
    mean_first, mean_second = rng.normal(size=size), rng.normal(size=size)
    covariance_first = random_covariance(rng, size)
    covariance_second = random_covariance(rng, size)
    for orthonormal in (True, False):
        directions = np.asarray(
            sliced_directions(space_of(size), 11, key=KEY, orthonormal=orthonormal)
        )
        expected = np.mean(
            [
                projected_w2_squared(
                    mean_first,
                    covariance_first,
                    mean_second,
                    covariance_second,
                    direction,
                )
                for direction in directions
            ]
        )
        value = sliced_wasserstein_of_gaussians(
            mean_first,
            covariance_first,
            mean_second,
            covariance_second,
            directions=directions,
        )
        assert float(value) == pytest.approx(expected, rel=1e-12)


def test_gaussian_sliced_w2_of_isotropic_gaussians_is_exact_on_complete_bases(
    rng: np.random.Generator,
) -> None:
    """A shift Δ and widths σ₁, σ₂ give ‖Δ‖²/n + (σ₁ − σ₂)² for any complete bases."""
    size = 5
    shift = rng.normal(size=size)
    first_width, second_width = 0.7, 1.9
    directions = sliced_directions(space_of(size), 40, key=KEY)
    value = sliced_wasserstein_of_gaussians(
        np.zeros(size),
        first_width**2 * np.eye(size),
        shift,
        second_width**2 * np.eye(size),
        directions=directions,
    )
    expected = shift @ shift / size + (first_width - second_width) ** 2
    assert float(value) == pytest.approx(expected, abs=1e-12)


def test_gaussian_sliced_w2_agrees_with_the_zoo_closed_form() -> None:
    """A unit Gaussian against a shifted copy with scaled axes, over many directions."""
    size, scaled, ratio = 4, 2, 1.5
    shift = np.array([0.5, -0.8, 0.3, 0.6])
    covariance_second = np.diag([ratio**2] * scaled + [1.0] * (size - scaled))
    truth = float(
        closed_forms.gaussian_sliced_w2(np.linalg.norm(shift), ratio, scaled, size)
    )
    space = space_of(size)
    independent = np.asarray(
        sliced_directions(space, 100_000, key=KEY, orthonormal=False)
    )
    per_direction = (independent @ shift) ** 2 + (
        1 - np.sqrt(np.sum((independent @ covariance_second) * independent, axis=1))
    ) ** 2
    error = np.std(per_direction, ddof=1) / math.sqrt(len(per_direction))
    for directions in (independent, sliced_directions(space, 100_000, key=KEY)):
        value = sliced_wasserstein_of_gaussians(
            np.zeros(size),
            np.eye(size),
            shift,
            covariance_second,
            directions=directions,
        )
        assert abs(float(value) - truth) < STANDARD_ERRORS * error


def test_gaussian_sliced_w2_runs_under_jit_and_vmap(rng: np.random.Generator) -> None:
    """The kernel compiles, and a batch of shifts maps to one value each."""
    size = 3
    directions = sliced_directions(TRIPLE, 9, key=KEY)
    covariance_first, covariance_second = random_covariance(rng, size), np.eye(size)
    shifts = jnp.asarray(rng.normal(size=(4, size)))

    def value_of(shift: jax.Array) -> jax.Array:
        return sliced_wasserstein_of_gaussians(
            jnp.zeros(size),
            covariance_first,
            shift,
            covariance_second,
            directions=directions,
        )

    mapped = jax.vmap(value_of)(shifts)
    compiled = jax.jit(value_of)
    for index in range(4):
        single = value_of(shifts[index])
        assert float(mapped[index]) == pytest.approx(float(single), rel=1e-12)
        assert float(compiled(shifts[index])) == pytest.approx(float(single), rel=1e-12)


def test_gaussian_sliced_w2_is_nan_for_a_negative_projected_variance() -> None:
    """An indefinite or undefined covariance gives NaN, a valid one a finite value."""
    directions = sliced_directions(PLANE, 2, key=KEY)
    zero = np.zeros(2)
    assert np.isfinite(
        float(
            sliced_wasserstein_of_gaussians(
                zero, np.eye(2), zero, 2 * np.eye(2), directions=directions
            )
        )
    )
    for bad in (np.diag([1.0, -1.0]), -np.eye(2), np.full((2, 2), np.nan)):
        for first, second in ((bad, np.eye(2)), (np.eye(2), bad)):
            value = sliced_wasserstein_of_gaussians(
                zero, first, zero, second, directions=directions
            )
            assert math.isnan(float(value))


def test_gaussian_sliced_w2_checks_its_shapes() -> None:
    """Means, covariances and directions share one coordinate count."""
    directions = np.ones((3, 2)) / math.sqrt(2)
    with pytest.raises(ValueError, match=r"mean_first must be shaped \(2,\)"):
        sliced_wasserstein_of_gaussians(
            np.zeros(3), np.eye(2), np.zeros(2), np.eye(2), directions=directions
        )
    with pytest.raises(ValueError, match=r"covariance_first must be shaped \(2, 2\)"):
        sliced_wasserstein_of_gaussians(
            np.zeros(2), np.eye(3), np.zeros(2), np.eye(2), directions=directions
        )
    with pytest.raises(ValueError, match=r"mean_second must be shaped \(2,\)"):
        sliced_wasserstein_of_gaussians(
            np.zeros(2), np.eye(2), np.zeros(4), np.eye(2), directions=directions
        )
    with pytest.raises(ValueError, match=r"covariance_second must be shaped"):
        sliced_wasserstein_of_gaussians(
            np.zeros(2), np.eye(2), np.zeros(2), np.eye(4), directions=directions
        )
    with pytest.raises(ValueError, match=r"directions must be shaped \(L, n\)"):
        sliced_wasserstein_of_gaussians(
            np.zeros(2), np.eye(2), np.zeros(2), np.eye(2), directions=np.ones((0, 2))
        )


def test_gaussian_distance_is_the_moment_form_on_the_weighted_moments(
    rng: np.random.Generator,
) -> None:
    """The weighted mean and the unbiased weighted covariance enter the moment form."""
    first, second = rng.normal(size=(80, 3)), rng.normal(0.4, 1.3, size=(60, 3))
    first_weights = rng.uniform(0.2, 2.0, 80)
    second_weights = rng.uniform(0.2, 2.0, 60)
    directions = sliced_directions(TRIPLE, 30, key=KEY)
    value = gaussian_sliced_wasserstein_distance(
        first,
        np.log(first_weights),
        second,
        np.log(second_weights),
        TRIPLE,
        directions=directions,
    )
    expected = sliced_wasserstein_of_gaussians(
        np.average(first, axis=0, weights=first_weights),
        np.cov(first.T, aweights=first_weights),
        np.average(second, axis=0, weights=second_weights),
        np.cov(second.T, aweights=second_weights),
        directions=directions,
    )
    assert float(value) == pytest.approx(float(expected), rel=1e-12)


def test_gaussian_distance_weights_enter_and_zero_weights_remove_samples(
    rng: np.random.Generator,
) -> None:
    """Uneven weights change the value, and a weight of zero drops the sample."""
    first, second = rng.normal(size=(60, 2)), rng.normal(0.5, 1.4, size=(50, 2))
    directions = sliced_directions(PLANE, 30, key=KEY)
    plain = gaussian_sliced_wasserstein_distance(
        first, None, second, None, PLANE, directions=directions
    )
    weighted = gaussian_sliced_wasserstein_distance(
        first,
        np.log(rng.uniform(0.1, 3.0, 60)),
        second,
        None,
        PLANE,
        directions=directions,
    )
    assert abs(float(weighted) - float(plain)) > 1e-3
    keep = np.arange(60) % 4 != 0
    with np.errstate(divide="ignore"):
        removed = gaussian_sliced_wasserstein_distance(
            first,
            np.log(keep.astype(float)),
            second,
            None,
            PLANE,
            directions=directions,
        )
    dropped = gaussian_sliced_wasserstein_distance(
        first[keep], None, second, None, PLANE, directions=directions
    )
    assert float(removed) == pytest.approx(float(dropped), rel=1e-12)


def test_gaussian_distance_uses_the_embedded_points(rng: np.random.Generator) -> None:
    """On a space with a circle and a sphere the moments are those of the embedding."""
    first, second = mixed_samples(rng, 300, 0.0), mixed_samples(rng, 220, 0.3)
    directions = sliced_directions(MIXED, 30, key=KEY)
    value = gaussian_sliced_wasserstein_distance(
        first, None, second, None, MIXED, directions=directions
    )
    first_points, second_points = embedded(first), embedded(second)
    expected = sliced_wasserstein_of_gaussians(
        first_points.mean(axis=0),
        np.cov(first_points.T),
        second_points.mean(axis=0),
        np.cov(second_points.T),
        directions=directions,
    )
    assert float(value) == pytest.approx(float(expected), rel=1e-9)


def test_gaussian_distance_agrees_with_the_empirical_value_for_gaussian_samples(
    rng: np.random.Generator,
) -> None:
    """On the same directions the estimators differ by their finite-sample biases."""
    size, count, replicates = 4, 4_000, 6
    space = space_of(size)
    directions = sliced_directions(space, 100, key=KEY)
    scales = np.array([1.5, 1.0, 1.0, 1.0])
    empirical: list[float] = []
    gaussian: list[float] = []
    for _ in range(replicates):
        first = rng.normal(size=(count, size))
        second = rng.normal(size=(count, size)) * scales + np.array([1.0, 0, 0, 0])
        for values, function in (
            (empirical, sliced_wasserstein_distance),
            (gaussian, gaussian_sliced_wasserstein_distance),
        ):
            values.append(
                float(function(first, None, second, None, space, directions=directions))
            )
    scatter = math.sqrt(
        (np.var(empirical, ddof=1) + np.var(gaussian, ddof=1)) / replicates
    )
    # * Each estimator is biased high by a few 1/N, so the allowance is 8/N.
    assert (
        abs(np.mean(empirical) - np.mean(gaussian))
        < STANDARD_ERRORS * scatter + 8 / count
    )
    assert np.mean(gaussian) > 0.2


def test_the_gelbrich_gap_separates_the_banana_from_its_gaussian() -> None:
    """The banana reads zero in the Gaussian form and not in the empirical one."""
    case = zoo.banana()
    samples = np.asarray(case.sample(jax.random.key(3), 2_000, batch_size=1).samples)[0]
    banana, gaussian = samples[0], samples[1]
    directions = sliced_directions(case.space, 100, key=jax.random.key(2))
    plan = permutation_null(
        banana, None, gaussian, None, key=jax.random.key(4), draws=100
    )
    results = {}
    for name, function in (
        ("gaussian", gaussian_sliced_wasserstein_distance),
        ("empirical", sliced_wasserstein_distance),
    ):

        def statistic(
            first: jax.Array,
            first_weights: jax.Array,
            second: jax.Array,
            second_weights: jax.Array,
            function: Callable[..., jax.Array] = function,
        ) -> jax.Array:
            return function(
                first,
                first_weights,
                second,
                second_weights,
                case.space,
                directions=directions,
            )

        draws = null_draws(statistic, plan)
        value = statistic(banana, jnp.zeros(2_000), gaussian, jnp.zeros(2_000))
        results[name] = (
            float(null_subtract("sliced_w2", value, draws)),
            float(jnp.std(draws, ddof=1)),
        )
    # * Equal moments leave only the bias, which the null subtraction removes.
    assert abs(results["gaussian"][0]) < 3 * results["gaussian"][1]
    assert results["empirical"][0] > 10 * results["empirical"][1]


# * Sliced Sinkhorn divergence


def test_sliced_sinkhorn_is_the_mean_one_dimensional_divergence(
    rng: np.random.Generator,
) -> None:
    """Each direction is a Sinkhorn divergence of the projected samples."""
    first, second = rng.normal(size=(200, 3)), rng.normal(0.3, 1.2, size=(150, 3))
    directions = np.asarray(sliced_directions(TRIPLE, 5, key=KEY))
    value = sliced_sinkhorn_divergence(
        first, None, second, None, TRIPLE, directions=directions, epsilon=0.2, key=KEY
    )
    expected = np.mean(
        [
            float(
                sinkhorn_divergence(
                    (first @ direction)[:, None],
                    None,
                    (second @ direction)[:, None],
                    None,
                    LINE,
                    epsilon=0.2,
                    key=KEY,
                )
            )
            for direction in directions
        ]
    )
    assert float(value) == pytest.approx(expected, rel=1e-10)


def test_sliced_sinkhorn_of_a_shift_is_the_mean_squared_projection(
    rng: np.random.Generator,
) -> None:
    """The translation property holds on every projection, whatever ε."""
    first = rng.normal(size=(300, 2))
    shift = np.array([0.6, -0.4])
    directions = sliced_directions(PLANE, 10, key=KEY)
    value = sliced_sinkhorn_divergence(
        first,
        None,
        first + shift,
        None,
        PLANE,
        directions=directions,
        epsilon=0.3,
        key=KEY,
    )
    expected = np.mean((np.asarray(directions) @ shift) ** 2)
    assert float(value) == pytest.approx(expected, rel=1e-6)


def test_sliced_sinkhorn_of_scaled_gaussians_matches_the_closed_forms() -> None:
    """Null-corrected values match the one-dimensional closed forms per direction."""
    case = zoo.scaled_gaussians(2.0)
    draws = np.asarray(case.sample(jax.random.key(7), 1_500, batch_size=4).samples)
    others = np.asarray(case.sample(jax.random.key(8), 1_500, batch_size=4).samples)
    directions = np.asarray(sliced_directions(case.space, 10, key=KEY))
    epsilon = 0.5
    excess = []
    for problem, other in zip(draws, others, strict=True):
        values = [
            sliced_sinkhorn_divergence(
                problem[0],
                None,
                second,
                None,
                case.space,
                directions=directions,
                epsilon=epsilon,
                key=KEY,
            )
            for second in (problem[1], other[0])
        ]
        excess.append(float(values[0] - values[1]))
    # * A direction θ sees the variances 1 and 1 + 3θ₀², whose S_ε is closed.
    truth = np.mean(
        [
            float(
                closed_forms.gaussian_sinkhorn(
                    np.zeros(1),
                    np.eye(1),
                    np.zeros(1),
                    np.array([[1 + 3 * direction[0] ** 2]]),
                    epsilon,
                ).total
            )
            for direction in directions
        ]
    )
    error = np.std(excess, ddof=1) / math.sqrt(len(excess))
    assert abs(np.mean(excess) - truth) < STANDARD_ERRORS * error + 0.03 * truth


def test_sliced_sinkhorn_caps_at_its_own_cap(rng: np.random.Generator) -> None:
    """Above 2 000 samples per posterior, both posteriors are subsampled once."""
    first = rng.normal(size=(SLICED_SINKHORN_CAP + 10, 1))
    second = rng.normal(size=(50, 1))
    directions = sliced_directions(LINE, 1, key=KEY)
    with pytest.warns(SubsamplingWarning, match="exceed the cap of 2000"):
        sliced_sinkhorn_divergence(
            first, None, second, None, LINE, directions=directions, epsilon=1.0, key=KEY
        )


def test_sliced_epsilon_is_the_factor_times_the_projected_median(
    rng: np.random.Generator,
) -> None:
    """Below the subsample size, the plain median over all pairs and directions."""
    first, second = rng.normal(size=(120, 2)), rng.normal(0.5, 1.5, size=(120, 2))
    directions = np.asarray(sliced_directions(PLANE, 7, key=KEY))
    value = sliced_sinkhorn_epsilon(
        first, None, second, None, PLANE, directions=directions, key=KEY
    )
    projections = np.concatenate([first, second]) @ directions.T
    rows, columns = np.triu_indices(240, 1)
    squared = (projections[rows] - projections[columns]) ** 2
    expected = SINKHORN_EPSILON_FACTOR * np.median(squared)
    assert float(value) == pytest.approx(expected, rel=1e-12)


def test_sliced_sinkhorn_without_convergence_gives_nan(
    rng: np.random.Generator,
) -> None:
    """A tiny ε needs more iterations than allowed, so the mean becomes NaN."""
    first, second = rng.normal(size=(200, 2)), rng.normal(1.0, 1.0, size=(200, 2))
    directions = sliced_directions(PLANE, 2, key=KEY)
    with pytest.warns(ValidityWarning, match="for every direction"):
        value = sliced_sinkhorn_divergence(
            first,
            None,
            second,
            None,
            PLANE,
            directions=directions,
            epsilon=1e-6,
            key=KEY,
        )
    assert math.isnan(float(value))


def test_sliced_epsilon_subsamples_from_the_documented_keys(
    rng: np.random.Generator,
) -> None:
    """Above the subsample size each posterior draws from its own derived key."""
    first, second = rng.normal(size=(300, 2)), rng.normal(size=(80, 2))
    directions = sliced_directions(PLANE, 4, key=KEY)
    value = sliced_sinkhorn_epsilon(
        first,
        None,
        second,
        None,
        PLANE,
        directions=directions,
        key=KEY,
        sample_size=100,
    )
    kept, _ = _draw(jnp.asarray(first), jnp.zeros(300), derive_key(KEY, "first"), 100)
    expected = sliced_sinkhorn_epsilon(
        np.asarray(kept), None, second, None, PLANE, directions=directions, key=KEY
    )
    assert float(value) == pytest.approx(float(expected), rel=1e-14)


def test_sliced_sinkhorn_checks_its_inputs(rng: np.random.Generator) -> None:
    """ε must be positive and finite, and the subsample size at least one."""
    samples = rng.normal(size=(20, 2))
    directions = sliced_directions(PLANE, 3, key=KEY)
    for epsilon in (0.0, -1.0, math.inf):
        with pytest.raises(ValueError, match="epsilon must be positive and finite"):
            sliced_sinkhorn_divergence(
                samples,
                None,
                samples,
                None,
                PLANE,
                directions=directions,
                epsilon=epsilon,
                key=KEY,
            )
    with pytest.raises(ValueError, match="sample_size must be at least 1"):
        sliced_sinkhorn_epsilon(
            samples,
            None,
            samples,
            None,
            PLANE,
            directions=directions,
            key=KEY,
            sample_size=0,
        )


def test_sliced_metrics_map_over_problems(rng: np.random.Generator) -> None:
    """Under jax.vmap each problem gets its own values."""
    first = jnp.asarray(rng.normal(size=(3, 120, 2)))
    second = jnp.asarray(rng.normal(0.3, 1.0, size=(3, 100, 2)))
    directions = sliced_directions(PLANE, 8, key=KEY)

    def both(left: ArrayLike, right: ArrayLike) -> tuple[jax.Array, jax.Array]:
        return (
            sliced_wasserstein_distance(
                left, None, right, None, PLANE, directions=directions
            ),
            sliced_sinkhorn_divergence(
                left,
                None,
                right,
                None,
                PLANE,
                directions=directions,
                epsilon=0.4,
                key=KEY,
            ),
        )

    mapped = jax.vmap(both)(first, second)
    for index in range(3):
        single = both(first[index], second[index])
        assert float(mapped[0][index]) == pytest.approx(float(single[0]), rel=1e-12)
        assert float(mapped[1][index]) == pytest.approx(float(single[1]), rel=1e-10)
