"""Tests for the Sinkhorn divergence with the squared chordal cost."""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.typing import ArrayLike

from mixed_signals import zoo
from mixed_signals.core.space import ParamSpace
from mixed_signals.errors import SubsamplingWarning, ValidityWarning
from mixed_signals.metrics._cap import cap_pair
from mixed_signals.metrics.kernel import median_bandwidth
from mixed_signals.metrics.ot.sinkhorn import (
    SINKHORN_EPSILON_FACTOR,
    sinkhorn_divergence,
)
from mixed_signals.zoo import closed_forms

STANDARD_ERRORS = 5.0
TWO_PI = 2 * math.pi
KEY = jax.random.key(0)
PLANE = ParamSpace.model_validate({"parameters": [{"name": "x0"}, {"name": "x1"}]})
TRIPLE = ParamSpace.model_validate(
    {"parameters": [{"name": "x0"}, {"name": "x1"}, {"name": "x2"}]}
)
CIRCLE = ParamSpace.model_validate(
    {"parameters": [{"name": "phase", "topology": "circle", "bounds": [0, TWO_PI]}]}
)


@pytest.fixture
def rng() -> np.random.Generator:
    """A seeded NumPy generator for test data."""
    return np.random.default_rng(2026)


def default_epsilon(first: ArrayLike, second: ArrayLike, space: ParamSpace) -> float:
    """Return ε = 0.25 ℓ² for the median bandwidth ℓ of a pair."""
    bandwidth = median_bandwidth(first, None, second, None, space, key=KEY)
    return SINKHORN_EPSILON_FACTOR * float(bandwidth) ** 2


def divergence(
    first: ArrayLike,
    second: ArrayLike,
    space: ParamSpace,
    epsilon: float,
    first_log_weights: ArrayLike | None = None,
) -> float:
    """Return S_ε of two samples at one ε."""
    return float(
        sinkhorn_divergence(
            first, first_log_weights, second, None, space, epsilon=epsilon, key=KEY
        )
    )


# * Debiasing and shifts


def test_identical_inputs_give_zero(rng: np.random.Generator) -> None:
    """The self terms remove the entropic bias, so S_ε(p, p) vanishes."""
    samples = rng.normal(size=(400, 3))
    epsilon = default_epsilon(samples, samples, TRIPLE)
    assert abs(divergence(samples, samples, TRIPLE, epsilon)) < 1e-8


def test_a_shift_gives_its_squared_length_at_every_epsilon(
    rng: np.random.Generator,
) -> None:
    """S_ε(p, p + Δ) = ‖Δ‖² whatever ε, the translation property of the squared cost."""
    first = rng.normal(size=(500, 3)) @ np.diag([1.0, 2.0, 0.5])
    shift = np.array([0.4, -0.3, 0.2])
    epsilon = default_epsilon(first, first + shift, TRIPLE)
    values = sinkhorn_divergence(
        first,
        None,
        first + shift,
        None,
        TRIPLE,
        epsilon=epsilon * np.array([0.5, 1.0, 10.0]),
        key=KEY,
    )
    np.testing.assert_allclose(values, shift @ shift, rtol=1e-7)


def test_a_vector_of_epsilon_gives_the_curve(rng: np.random.Generator) -> None:
    """One call with several ε equals separate calls, in the shape of ε."""
    first, second = rng.normal(size=(300, 2)), rng.normal(0.3, 1.2, size=(250, 2))
    epsilons = np.array([[0.2, 0.5], [1.0, 3.0]])
    curve = sinkhorn_divergence(
        first, None, second, None, PLANE, epsilon=epsilons, key=KEY
    )
    assert curve.shape == (2, 2)
    for index in np.ndindex(epsilons.shape):
        single = divergence(first, second, PLANE, float(epsilons[index]))
        assert float(curve[index]) == pytest.approx(single, rel=1e-12)
    # * Larger ε smooths the transport, so a width difference shrinks.
    assert float(curve[1, 1]) < float(curve[0, 0])


def test_shifted_gaussians_recover_the_squared_shift() -> None:
    """Null-corrected values average to the zoo's ‖Δμ‖², the table's sinkhorn row."""
    case = zoo.shifted_gaussians(1.0, param_count=3)
    pair = ("reference", "shifted")
    draws = np.asarray(case.sample(jax.random.key(1), 1_500, batch_size=10).samples)
    others = np.asarray(case.sample(jax.random.key(2), 1_500, batch_size=10).samples)
    excess = []
    for problem, other in zip(draws, others, strict=True):
        epsilon = default_epsilon(problem[0], problem[1], case.space)
        # * A second draw of the reference carries the same finite-sample bias.
        excess.append(
            divergence(problem[0], problem[1], case.space, epsilon)
            - divergence(problem[0], other[0], case.space, epsilon)
        )
    error = np.std(excess, ddof=1) / math.sqrt(len(excess))
    truth = case.reference_value("sinkhorn", pair)
    assert truth == pytest.approx(1.0)
    assert abs(np.mean(excess) - truth) < STANDARD_ERRORS * error


def test_scaled_gaussians_match_the_closed_form() -> None:
    """A width ratio of 2 gives the closed form of Janati et al. within its bias."""
    case = zoo.scaled_gaussians(2.0)
    draws = np.asarray(case.sample(jax.random.key(3), 4_000, batch_size=4).samples)
    epsilon = 1.0
    values = [
        divergence(problem[0], problem[1], case.space, epsilon) for problem in draws
    ]
    truth = float(
        closed_forms.gaussian_sinkhorn(
            np.zeros(2), np.eye(2), np.zeros(2), np.diag([4.0, 1.0]), epsilon
        ).total
    )
    assert np.mean(values) == pytest.approx(truth, rel=0.03)


# * Weights


def test_weights_count_and_remove_samples(rng: np.random.Generator) -> None:
    """A weight of k counts like k copies, and a weight of zero removes a sample."""
    first = rng.normal(size=(60, 2))
    second = rng.normal(0.5, 1.0, size=(50, 2))
    copies = rng.integers(1, 4, size=60)
    weighted = divergence(first, second, PLANE, 0.7, np.log(copies))
    repeated = divergence(np.repeat(first, copies, axis=0), second, PLANE, 0.7)
    assert weighted == pytest.approx(repeated, rel=1e-7)
    keep = np.arange(60) % 4 != 0
    with np.errstate(divide="ignore"):
        removed = divergence(first, second, PLANE, 0.7, np.log(keep.astype(float)))
    assert removed == pytest.approx(
        divergence(first[keep], second, PLANE, 0.7), rel=1e-7
    )


# * Circles and spheres


def plane_points(angles: np.ndarray) -> np.ndarray:
    """Return points on the unit circle in the plane."""
    return np.stack([np.cos(angles), np.sin(angles)], axis=1)


def test_circles_are_embedded_in_the_plane(rng: np.random.Generator) -> None:
    """A circle of period 2π acts as the unit circle in a plane, cut or turned."""
    first = rng.vonmises(0.0, 3.0, 300) % TWO_PI
    second = rng.vonmises(0.6, 2.0, 250) % TWO_PI
    on_circle = divergence(first[:, None], second[:, None], CIRCLE, 0.3)
    in_plane = divergence(plane_points(first), plane_points(second), PLANE, 0.3)
    assert on_circle == pytest.approx(in_plane, rel=1e-10)
    turned = divergence(
        ((first + 2.0) % TWO_PI)[:, None],
        ((second + 2.0) % TWO_PI)[:, None],
        CIRCLE,
        0.3,
    )
    assert turned == pytest.approx(on_circle, rel=1e-8)


def test_spheres_scale_with_their_radius() -> None:
    """A sphere of radius R acts as R times its unit vectors in three dimensions."""
    case = zoo.von_mises_fisher(1.0, width=0.3, euclidean_count=1)
    draws = np.asarray(case.sample(jax.random.key(4), 300).samples[0])
    sphere = case.space.spheres[0].model_copy(update={"radius": 4.0})
    scaled_space = case.space.model_copy(update={"spheres": (sphere,)})
    on_sphere = divergence(draws[0], draws[1], scaled_space, 0.5)
    azimuth, latitude = draws[..., 0], draws[..., 1]
    vectors = np.stack(
        [
            np.cos(latitude) * np.cos(azimuth),
            np.cos(latitude) * np.sin(azimuth),
            np.sin(latitude),
        ],
        axis=-1,
    )
    embedded = np.concatenate([draws[..., 2:], 4.0 * vectors], axis=-1)
    quadruple = ParamSpace.model_validate(
        {"parameters": [{"name": f"x{index}"} for index in range(4)]}
    )
    in_space = divergence(embedded[0], embedded[1], quadruple, 0.5)
    assert on_sphere == pytest.approx(in_space, rel=1e-10)


def modes(rng: np.random.Generator, centres: tuple[float, float]) -> np.ndarray:
    """Return 300 draws of two equal modes of width 0.3 on a circle."""
    picks = np.asarray(centres)[rng.integers(0, 2, 300)]
    return ((picks + 0.3 * rng.normal(size=300)) % TWO_PI)[:, None]


def test_chords_keep_even_harmonics_and_stay_non_negative() -> None:
    """Modes at 0 and π against ±π/2 are resolved, and no ε makes S_ε negative.

    Squared arcs gave −0.14 to −0.23 for this pair at large ε; the squared
    chord makes exp(−c/ε) a positive definite kernel at every ε.
    """
    rng = np.random.default_rng(5)
    first, second = (
        modes(rng, (0.0, math.pi)),
        modes(rng, (0.5 * math.pi, 1.5 * math.pi)),
    )
    pooled = np.concatenate([first, second])
    epsilon = default_epsilon(first, second, CIRCLE)
    value = divergence(first, second, CIRCLE, epsilon)
    null = []
    for _ in range(20):
        order = rng.permutation(600)
        null.append(
            divergence(pooled[order[:300]], pooled[order[300:]], CIRCLE, epsilon)
        )
    assert value > np.mean(null) + 20 * np.std(null, ddof=1)
    large = sinkhorn_divergence(
        first,
        None,
        second,
        None,
        CIRCLE,
        epsilon=epsilon * np.array([5.0, 25.0, 125.0]),
        key=KEY,
    )
    assert bool(jnp.all(large > 0))


# * Validity, cap and transformations


def test_unconverged_iterations_give_nan_and_a_warning(
    rng: np.random.Generator,
) -> None:
    """A tiny ε needs more iterations than allowed; only its value becomes NaN."""
    first, second = rng.normal(size=(200, 2)), rng.normal(1.0, 1.0, size=(200, 2))
    with pytest.warns(ValidityWarning, match=r"did not reach the marginal error"):
        values = sinkhorn_divergence(
            first, None, second, None, PLANE, epsilon=np.array([1e-5, 1.0]), key=KEY
        )
    assert math.isnan(float(values[0]))
    assert math.isfinite(float(values[1]))


def test_the_cap_subsamples_each_posterior_once(rng: np.random.Generator) -> None:
    """Above the cap one warning names both counts, and the capped sets are used."""
    first = rng.normal(size=(500, 2))
    second = rng.normal(0.2, 1.0, size=(200, 2))
    with pytest.warns(SubsamplingWarning, match="500 and 200 samples exceed the cap"):
        value = sinkhorn_divergence(
            first, None, second, None, PLANE, epsilon=0.8, key=KEY, cap=300
        )
    with pytest.warns(SubsamplingWarning):
        capped = cap_pair(
            jnp.asarray(first),
            jnp.zeros(500),
            jnp.asarray(second),
            jnp.zeros(200),
            key=KEY,
            metric="test",
            cap=300,
        )
    expected = divergence(capped[0], capped[2], PLANE, 0.8)
    assert float(value) == pytest.approx(expected, rel=1e-12)


def test_inputs_are_checked(rng: np.random.Generator) -> None:
    """ε must be positive and finite, and the samples must fit the space."""
    samples = rng.normal(size=(20, 2))
    for epsilon in (0.0, -1.0, math.inf, math.nan):
        with pytest.raises(ValueError, match="epsilon must be positive and finite"):
            sinkhorn_divergence(
                samples, None, samples, None, PLANE, epsilon=epsilon, key=KEY
            )
    with pytest.raises(ValueError, match=r"shaped \(N, 3\)"):
        sinkhorn_divergence(samples, None, samples, None, TRIPLE, epsilon=1.0, key=KEY)


def test_the_divergence_maps_over_problems(rng: np.random.Generator) -> None:
    """Under jax.vmap each problem gets its own value, with a traced ε."""
    first = jnp.asarray(rng.normal(size=(3, 150, 2)))
    second = jnp.asarray(rng.normal(0.3, 1.0, size=(3, 120, 2)))
    epsilons = jnp.array([0.5, 1.0, 2.0])

    def one(left: jax.Array, right: jax.Array, epsilon: jax.Array) -> jax.Array:
        return sinkhorn_divergence(
            left, None, right, None, PLANE, epsilon=epsilon, key=KEY
        )

    mapped = jax.vmap(one)(first, second, epsilons)
    for index in range(3):
        single = one(first[index], second[index], epsilons[index])
        assert float(mapped[index]) == pytest.approx(float(single), rel=1e-10)
