"""Tests for the pairwise dependence statistics and their Gaussian values."""

import math
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import spatial, special, stats

from mixed_signals import zoo
from mixed_signals.core.space import ParamSpace
from mixed_signals.diagnostics._common import parameter_pairs
from mixed_signals.diagnostics.dependence import (
    dependence_statistics,
    gaussian_distance_correlation,
    gaussian_mutual_information,
    gaussian_spearman,
)
from mixed_signals.errors import SubsamplingWarning, ValidityWarning

TWO_PI = 2 * math.pi
CIRCLE_AND_LINE = ParamSpace.model_validate(
    {
        "parameters": [
            {"name": "phase", "topology": "circle", "bounds": [0.0, TWO_PI]},
            {"name": "x1"},
        ]
    }
)


def reference_dcor(first_distances: np.ndarray, second_distances: np.ndarray) -> float:
    """Return the V-statistic distance correlation of two distance matrices."""

    def centred(distances: np.ndarray) -> np.ndarray:
        return (
            distances
            - distances.mean(axis=0)
            - distances.mean(axis=1)[:, None]
            + distances.mean()
        )

    first, second = centred(first_distances), centred(second_distances)
    covariance = (first * second).mean()
    return float(np.sqrt(covariance / np.sqrt((first**2).mean() * (second**2).mean())))


def reference_mutual_information(
    first_distances: np.ndarray, second_distances: np.ndarray, neighbours: int
) -> float:
    """Return KSG's I⁽¹⁾ from the two distance matrices, by brute force."""
    joint = np.maximum(first_distances, second_distances)
    np.fill_diagonal(joint, np.inf)
    radii = np.sort(joint, axis=1)[:, neighbours - 1]
    count = len(radii)
    first_counts = np.sum(first_distances < radii[:, None], axis=1) - 1
    second_counts = np.sum(second_distances < radii[:, None], axis=1) - 1
    return float(
        special.digamma(neighbours)
        + special.digamma(count)
        - np.mean(
            special.digamma(first_counts + 1) + special.digamma(second_counts + 1)
        )
    )


# * Gaussian values


def test_gaussian_spearman_matches_a_large_gaussian_sample() -> None:
    """(6/π) arcsin(r/2) is Spearman's ρ_s of a bivariate Gaussian."""
    rng = np.random.default_rng(0)
    for correlation in (0.3, 0.8, -0.95):
        covariance = np.array([[1.0, correlation], [correlation, 1.0]])
        draws = rng.multivariate_normal(np.zeros(2), covariance, size=200_000)
        observed = stats.spearmanr(draws[:, 0], draws[:, 1]).statistic
        assert float(gaussian_spearman(correlation)) == pytest.approx(
            observed, abs=0.004
        )


def test_gaussian_distance_correlation_matches_its_limits_and_a_sample() -> None:
    """Theorem 7 gives 0 at r = 0, 1 at |r| = 1 and the dCor of large samples."""
    assert float(gaussian_distance_correlation(0.0)) == pytest.approx(0.0, abs=1e-15)
    assert float(gaussian_distance_correlation(1.0)) == pytest.approx(1.0, rel=1e-12)
    assert float(gaussian_distance_correlation(-0.6)) == pytest.approx(
        float(gaussian_distance_correlation(0.6)), rel=1e-15
    )
    rng = np.random.default_rng(1)
    for correlation in (0.5, 0.9):
        covariance = np.array([[1.0, correlation], [correlation, 1.0]])
        draws = rng.multivariate_normal(np.zeros(2), covariance, size=3_000)
        distances = [
            np.abs(draws[:, None, axis] - draws[None, :, axis]) for axis in (0, 1)
        ]
        observed = reference_dcor(*distances)
        assert float(gaussian_distance_correlation(correlation)) == pytest.approx(
            observed, abs=0.01
        )


def test_gaussian_mutual_information_is_the_gaussian_entropy_difference() -> None:
    """−½ ln(1 − r²) equals h(x) + h(y) − h(x, y) of a bivariate Gaussian."""
    for correlation in (0.0, 0.4, 0.99):
        joint = 0.5 * math.log((2 * math.pi * math.e) ** 2 * (1 - correlation**2))
        single = 0.5 * math.log(2 * math.pi * math.e)
        assert float(gaussian_mutual_information(correlation)) == pytest.approx(
            2 * single - joint, abs=1e-12
        )


# * Statistics


def test_statistics_match_independent_computations() -> None:
    """Pearson, Spearman, dCor and KSG equal SciPy and brute-force references."""
    case = zoo.banana(0.01, param_count=3)
    samples = np.asarray(case.sample(jax.random.key(2), 800).samples[0, 0])
    result = dependence_statistics(samples, None, case.space, key=jax.random.key(3))
    assert result.pearson.shape == (3,)
    for index, (first, second) in enumerate(parameter_pairs(3)):
        left, right = samples[:, first], samples[:, second]
        correlation = stats.pearsonr(left, right).statistic
        assert float(result.pearson[index]) == pytest.approx(correlation, abs=1e-12)
        assert float(result.spearman[index]) == pytest.approx(
            stats.spearmanr(left, right).statistic
            - float(gaussian_spearman(correlation)),
            abs=1e-12,
        )
        distances = [
            np.abs(left[:, None] - left[None, :]),
            np.abs(right[:, None] - right[None, :]),
        ]
        assert float(result.distance_correlation[index]) == pytest.approx(
            reference_dcor(*distances)
            - float(gaussian_distance_correlation(correlation)),
            abs=1e-10,
        )
        assert float(result.mutual_information[index]) == pytest.approx(
            reference_mutual_information(distances[0], distances[1], 10)
            - float(gaussian_mutual_information(correlation)),
            abs=1e-10,
        )


def test_ksg_matches_a_kd_tree_with_the_max_norm() -> None:
    """The joint radii and counts are those of SciPy's k-d tree with p = ∞."""
    case = zoo.shifted_gaussians(0.0, param_count=2)
    samples = np.asarray(case.sample(jax.random.key(4), 1_000).samples[0, 0])
    result = dependence_statistics(
        samples, None, case.space, key=jax.random.key(5), neighbours=4
    )
    tree = spatial.cKDTree(samples)
    radii = tree.query(samples, k=5, p=np.inf)[0][:, 4]
    counts = [
        np.array(
            [
                np.sum(np.abs(samples[:, axis] - samples[index, axis]) < radii[index])
                - 1
                for index in range(len(samples))
            ]
        )
        for axis in (0, 1)
    ]
    expected = (
        special.digamma(4)
        + special.digamma(len(samples))
        - np.mean(special.digamma(counts[0] + 1) + special.digamma(counts[1] + 1))
    )
    correlation = stats.pearsonr(samples[:, 0], samples[:, 1]).statistic
    assert float(result.mutual_information[0]) == pytest.approx(
        expected - float(gaussian_mutual_information(correlation)), abs=1e-10
    )


def test_integer_weights_equal_repeated_samples_for_ranks_and_dcor() -> None:
    """Weights act as repetitions in the mid-ranks and the V-statistic."""
    case = zoo.banana(0.01)
    samples = np.asarray(case.sample(jax.random.key(6), 300).samples[0, 0])
    counts = np.random.default_rng(7).integers(1, 4, size=300)
    with pytest.warns(ValidityWarning, match=r"equally weighted.*rejection_subset"):
        weighted = dependence_statistics(
            samples, np.log(counts), case.space, key=jax.random.key(8)
        )
    repeated = np.repeat(samples, counts, axis=0)
    left, right = repeated[:, 0], repeated[:, 1]
    correlation = stats.pearsonr(left, right).statistic
    assert float(weighted.pearson[0]) == pytest.approx(correlation, abs=1e-12)
    assert float(weighted.spearman[0]) == pytest.approx(
        stats.spearmanr(left, right).statistic - float(gaussian_spearman(correlation)),
        abs=1e-12,
    )
    distances = [
        np.abs(left[:, None] - left[None, :]),
        np.abs(right[:, None] - right[None, :]),
    ]
    assert float(weighted.distance_correlation[0]) == pytest.approx(
        reference_dcor(*distances) - float(gaussian_distance_correlation(correlation)),
        abs=1e-10,
    )
    assert math.isnan(float(weighted.mutual_information[0]))


def test_circles_use_chords_for_dcor_and_arcs_for_ksg() -> None:
    """On a circle dCor measures chords and KSG the shorter arc."""
    case = zoo.wrapped_gaussians(0.0, width=0.8, param_count=2, correlation=0.6)
    samples = np.asarray(case.sample(jax.random.key(9), 700).samples[0, 0])
    result = dependence_statistics(samples, None, case.space, key=jax.random.key(10))
    differences = samples[:, None, 0] - samples[None, :, 0]
    chords = TWO_PI / math.pi * np.abs(np.sin(math.pi * differences / TWO_PI))
    arcs = np.abs(np.mod(differences + math.pi, TWO_PI) - math.pi)
    lines = np.abs(samples[:, None, 1] - samples[None, :, 1])
    centre = np.angle(np.mean(np.exp(1j * samples[:, 0])))
    unwrapped = centre + np.angle(np.exp(1j * (samples[:, 0] - centre)))
    correlation = stats.pearsonr(unwrapped, samples[:, 1]).statistic
    assert float(result.pearson[0]) == pytest.approx(correlation, abs=1e-12)
    assert float(result.distance_correlation[0]) == pytest.approx(
        reference_dcor(chords, lines)
        - float(gaussian_distance_correlation(correlation)),
        abs=1e-10,
    )
    assert float(result.mutual_information[0]) == pytest.approx(
        reference_mutual_information(arcs, lines, 10)
        - float(gaussian_mutual_information(correlation)),
        abs=1e-10,
    )


def test_wide_circles_warn() -> None:
    """Pairs with a circle wider than 1 rad have an approximate null."""
    case = zoo.wrapped_gaussians(0.0, width=1.5, param_count=2, correlation=0.5)
    samples = case.sample(jax.random.key(11), 500).samples[0, 0]
    with pytest.warns(ValidityWarning, match="the dependence rows involve 'x0'"):
        dependence_statistics(samples, None, case.space, key=jax.random.key(12))


def test_samples_above_the_cap_are_subsampled() -> None:
    """Above the cap a uniform subsample runs, with a SubsamplingWarning."""
    case = zoo.banana(0.01)
    samples = case.sample(jax.random.key(13), 600).samples[0, 0]
    with pytest.warns(SubsamplingWarning, match="dependence"):
        result = dependence_statistics(
            samples, None, case.space, key=jax.random.key(14), cap=400
        )
    assert bool(jnp.all(jnp.isfinite(result.mutual_information)))


def test_one_parameter_has_no_pairs() -> None:
    """Dependence needs two parameters."""
    case = zoo.shifted_gaussians(0.0, param_count=1)
    with pytest.raises(ValueError, match="at least two parameters"):
        dependence_statistics(
            case.sample(jax.random.key(15), 50).samples[0, 0],
            None,
            case.space,
            key=jax.random.key(16),
        )


def test_statistics_map_over_surrogate_sets_without_warnings() -> None:
    """Under jax.lax.map the statistics equal concrete calls and never warn."""
    case = zoo.banana(0.01)
    sets = case.sample(jax.random.key(17), 300, batch_size=2).samples[:, 0]
    key = jax.random.key(18)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        mapped = jax.lax.map(
            lambda values: dependence_statistics(values, None, case.space, key=key),
            sets,
        )
    direct = dependence_statistics(sets[1], None, case.space, key=key)
    for field in ("pearson", "spearman", "distance_correlation", "mutual_information"):
        np.testing.assert_allclose(
            getattr(mapped, field)[1], getattr(direct, field), rtol=1e-10, atol=1e-12
        )


def test_too_few_samples_for_the_rank_warn() -> None:
    """The mutual information needs more samples than its rank."""
    case = zoo.banana(0.01)
    samples = case.sample(jax.random.key(19), 8).samples[0, 0]
    with pytest.warns(ValidityWarning, match="more than k samples"):
        result = dependence_statistics(
            samples, None, case.space, key=jax.random.key(20)
        )
    assert math.isnan(float(result.mutual_information[0]))
