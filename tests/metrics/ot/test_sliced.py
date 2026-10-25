"""Tests for sliced W2, the sliced Sinkhorn divergence and their directions."""

import math

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
    SLICED_SINKHORN_CAP,
    sliced_directions,
    sliced_sinkhorn_divergence,
    sliced_sinkhorn_epsilon,
    sliced_wasserstein_distance,
)
from mixed_signals.prng import derive_key
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


def test_directions_are_unit_vectors_of_the_embedded_space() -> None:
    """One coordinate per embedded coordinate, unit length, fixed by the key."""
    directions = sliced_directions(MIXED, 4_000, key=KEY)
    assert directions.shape == (4_000, 6)
    np.testing.assert_allclose(np.linalg.norm(directions, axis=1), 1.0, rtol=1e-12)
    np.testing.assert_array_equal(directions, sliced_directions(MIXED, 4_000, key=KEY))
    other = sliced_directions(MIXED, 4_000, key=jax.random.key(1))
    assert not np.allclose(directions, other)
    # * Uniform directions have E[θθᵀ] = I/n; 4 000 draws resolve it to about 0.01.
    np.testing.assert_allclose(
        np.asarray(directions).T @ np.asarray(directions) / 4_000,
        np.eye(6) / 6,
        atol=0.02,
    )
    with pytest.raises(ValueError, match="count must be at least 1"):
        sliced_directions(MIXED, 0, key=KEY)


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
