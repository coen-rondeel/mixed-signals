"""Tests for the zoo cases: every reference value is reproduced within sampling error.

:func:`assert_reference_reproduced` estimates the value of every row of a
case's reference table from draws of the case, and fails on a metric it does
not know, so a new metric cannot pass unchecked. Averages carry their standard
errors; plug-in statistics take theirs from the spread over disjoint batches.
"""

import math
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import polars as pl
import pytest
from scipy import special, stats

from mixed_signals.core.chains import Chains
from mixed_signals.core.samples import SampleSet
from mixed_signals.core.space import ParamSpace
from mixed_signals.core.topology import (
    AZIMUTH_PERIOD,
    PolarForm,
    Topology,
    unit_vectors,
)
from mixed_signals.zoo import (
    REFERENCE_SCHEMA,
    ChainCase,
    ReferenceBasis,
    ZooCase,
    ar1_chains,
    banana,
    closed_forms,
    gaussian_mixture,
    gaussians,
    railed_gaussians,
    rotated_gaussians,
    scaled_gaussians,
    shifted_gaussians,
    von_mises_fisher,
    wrapped_gaussians,
)
from mixed_signals.zoo.distributions import Gaussian, TruncatedGaussian, WrappedGaussian

STANDARD_ERRORS = 5.0
SAMPLE_COUNT = 40_000
BATCH_COUNT = 20

type Estimate = tuple[float, float, float]
"""An estimate, its standard error and a bound on its bias."""


@pytest.fixture(scope="module")
def key() -> jax.Array:
    """A fixed JAX key for the draws of every case."""
    return jax.random.key(314)


# * Estimators of the reference values


def linearised(case: ZooCase, sample_set: SampleSet) -> np.ndarray:
    """Return the draws with every circle coordinate unwrapped around its mean.

    Each posterior's circle coordinate is replaced by its circular mean plus
    the shortest arc to each draw, so linear moments estimate the unwrapped
    normal when its width is small against the period.
    """
    draws = np.array(sample_set.samples[0])
    for index, parameter in enumerate(case.space.parameters):
        if parameter.topology is not Topology.CIRCLE:
            continue
        period = parameter.period
        assert period is not None
        for posterior in range(len(case.names)):
            angles = draws[posterior, :, index]
            centre = circular_mean(angles, period)
            draws[posterior, :, index] = centre + signed_arc(centre, angles, period)
    return draws


def circular_mean(angles: np.ndarray, period: float) -> float:
    """Return the circular mean of angles on a circle of the given period."""
    phase = np.mean(np.exp(2j * math.pi * angles / period))
    return float(np.angle(phase) * period / (2 * math.pi))


def signed_arc(origin: Any, target: Any, period: float) -> Any:
    """Return the shortest signed arc from origin to target."""
    # ! x % inf is an invalid operation: NumPy warns on Linux but not on macOS.
    assert math.isfinite(period), "signed_arc needs the finite period of a circle"
    return (target - origin + 0.5 * period) % period - 0.5 * period


def batch_estimate(statistic: Callable[[slice], float]) -> tuple[float, float]:
    """Return a plug-in statistic and its standard error from disjoint batches."""
    size = SAMPLE_COUNT // BATCH_COUNT
    values = [
        statistic(slice(start, start + size)) for start in range(0, SAMPLE_COUNT, size)
    ]
    return statistic(slice(None)), float(
        np.std(values, ddof=1) / math.sqrt(BATCH_COUNT)
    )


def mean_estimate(values: np.ndarray) -> tuple[float, float]:
    """Return a sample mean and its standard error."""
    return float(values.mean()), float(values.std(ddof=1) / math.sqrt(len(values)))


def anderson_darling_distance(first: np.ndarray, second: np.ndarray) -> float:
    """Return Pettitt's two-sample Anderson-Darling statistic divided by nm/(n + m)."""
    first_count, second_count = len(first), len(second)
    total_count = first_count + second_count
    order = np.argsort(np.concatenate([first, second]), kind="stable")
    first_seen = np.cumsum(order < first_count)[:-1]
    position = np.arange(1, total_count)
    statistic = np.sum(
        (total_count * first_seen - position * first_count) ** 2
        / (position * (total_count - position))
    ) / (first_count * second_count)
    return float(statistic * total_count / (first_count * second_count))


def distribution_gaps(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Return F − G after each pooled value of two equally weighted samples."""
    values = np.concatenate([first, second])
    steps = np.concatenate(
        [np.full(len(first), 1 / len(first)), np.full(len(second), -1 / len(second))]
    )
    return np.cumsum(steps[np.argsort(values, kind="stable")])


def kuiper_statistic(first: np.ndarray, second: np.ndarray) -> float:
    """Return Kuiper's two-sample V = max(F − G) − min(F − G), with F − G = 0 first."""
    gaps = distribution_gaps(first, second)
    return float(max(gaps.max(), 0.0) - min(gaps.min(), 0.0))


def circle_w1(first: np.ndarray, second: np.ndarray, period: float) -> float:
    """Return W1 along a circle: ∫ |F − G − c| at the length-weighted median c."""
    values = np.sort(np.concatenate([first, second]))
    lengths = np.append(np.diff(values), values[0] + period - values[-1])
    levels = np.append(distribution_gaps(first, second)[:-1], 0.0)
    order = np.argsort(levels)
    covered = np.cumsum(lengths[order])
    median = levels[order][np.searchsorted(covered, 0.5 * covered[-1])]
    return float(np.sum(lengths * np.abs(levels - median)))


def marginal_log_density(
    case: ZooCase, posterior: int, column: int
) -> Callable[[np.ndarray], np.ndarray]:
    """Return the log density of one marginal of a posterior, wrapped on a circle."""
    distribution = case.posteriors[posterior]
    if isinstance(distribution, TruncatedGaussian):
        mean = float(distribution.mean[column])
        std = float(distribution.std[column])
        lower = float(distribution.lower[column])
        upper = float(distribution.upper[column])
        truncated = stats.truncnorm(
            (lower - mean) / std, (upper - mean) / std, loc=mean, scale=std
        )
        return lambda values: np.asarray(truncated.logpdf(values))
    if isinstance(distribution, WrappedGaussian) and column == distribution.circle:
        mean = float(distribution.base.mean[column])
        std = math.sqrt(float(distribution.base.covariance[column, column]))
        windings = np.arange(-20, 21) * float(distribution.period)
        return lambda values: np.asarray(
            special.logsumexp(
                stats.norm.logpdf(np.asarray(values)[:, None] + windings, mean, std),
                axis=1,
            )
        )
    normal = reference_marginal(case, case.names[posterior], case.space.names[column])
    return lambda values: np.asarray(normal.logpdf(values))


def gaussian_moments(draws: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the sample mean and covariance of draws shaped (N, D)."""
    return draws.mean(axis=0), np.atleast_2d(np.cov(draws.T))


def reference_marginal(case: ZooCase, name: str, parameter: str) -> Any:
    """Return the normal marginal of a Gaussian posterior from its reference rows."""
    mean = case.reference_value("mean", name, parameter)
    std = case.reference_value("std", name, parameter)
    return stats.norm(mean, std)


def estimate_row(
    case: ZooCase,
    sample_set: SampleSet,
    draws: np.ndarray,
    log_densities: np.ndarray,
    row: dict[str, Any],
) -> Estimate:
    """Estimate one reference value from draws of the case.

    ``draws`` are the unwrapped draws of :func:`linearised`, shaped (K, N, D),
    and ``log_densities`` the log density of every posterior at the draws of
    every posterior, shaped (K, N, K).
    """
    metric, pair, parameters = row["metric"], row["pair"], row["parameters"]
    first = case.names.index(pair[0])
    second = case.names.index(pair[1]) if len(pair) == 2 else first
    columns = [case.space.index(parameter) for parameter in parameters]
    first_draws, second_draws = draws[first][:, columns], draws[second][:, columns]
    periods = [case.space.parameters[column].period for column in columns]
    period = periods[0] or math.inf

    def difference(slice_: slice) -> np.ndarray:
        means_first = first_draws[slice_].mean(axis=0)
        means_second = second_draws[slice_].mean(axis=0)
        # * Each circle wraps with its own period; the other axes take the plain gap.
        return np.array(
            [
                mean_second - mean_first
                if column_period is None
                else signed_arc(mean_first, mean_second, column_period)
                for mean_first, mean_second, column_period in zip(
                    means_first, means_second, periods, strict=True
                )
            ]
        )

    if metric == "mean":
        return (*mean_estimate(first_draws[:, 0]), 0.0)
    if metric == "std":
        return (*batch_estimate(lambda part: float(first_draws[part, 0].std())), 0.0)
    if metric in ("circular_mean", "mean_resultant_length") and len(columns) == 1:
        angles = np.asarray(sample_set.samples[0, first, :, columns[0]])
        lower = case.space.parameters[columns[0]].bounds[0]
        if metric == "circular_mean":
            target = row["value"]
            arc_estimate = batch_estimate(
                lambda part: float(
                    signed_arc(target, circular_mean(angles[part], period), period)
                )
            )
            assert lower <= target < lower + period
            return target + arc_estimate[0], arc_estimate[1], 0.0
        phases = np.cos(2 * math.pi * (angles - circular_mean(angles, period)) / period)
        return (*mean_estimate(phases), 0.0)
    if metric == "mean_resultant_length":
        vectors = np.asarray(sphere_vectors(case, sample_set, first))
        direction = vectors.mean(axis=0) / np.linalg.norm(vectors.mean(axis=0))
        return (*mean_estimate(vectors @ direction), 0.0)
    if metric == "mean_shift":
        return (
            *batch_estimate(
                lambda part: float(difference(part)[0] / first_draws[part, 0].std())
            ),
            0.0,
        )
    if metric == "width_ratio":
        return (
            *batch_estimate(
                lambda part: float(
                    second_draws[part, 0].std() / first_draws[part, 0].std()
                )
            ),
            0.0,
        )
    if metric == "mahalanobis_shift":

        def shift(part: slice) -> float:
            covariance = np.atleast_2d(np.cov(first_draws[part].T))
            gap = difference(part)
            return float(math.sqrt(gap @ np.linalg.solve(covariance, gap)))

        return (*batch_estimate(shift), 0.0)
    if metric == "angular_shift":

        def angle(part: slice) -> float:
            vectors_first = np.asarray(sphere_vectors(case, sample_set, first))[part]
            vectors_second = np.asarray(sphere_vectors(case, sample_set, second))[part]
            mean_first = vectors_first.mean(axis=0)
            mean_second = vectors_second.mean(axis=0)
            cosine = float(
                mean_first
                @ mean_second
                / (np.linalg.norm(mean_first) * np.linalg.norm(mean_second))
            )
            width = math.sqrt(1 - np.linalg.norm(mean_first))
            return math.acos(min(cosine, 1.0)) / width

        return (*batch_estimate(angle), 0.0)
    own_densities = log_densities[first]
    if metric == "kl":
        return (*mean_estimate(own_densities[:, first] - own_densities[:, second]), 0.0)
    if metric == "hellinger" and len(columns) == len(case.space):
        root_ratio = np.exp(0.5 * (own_densities[:, second] - own_densities[:, first]))
        squared, error = mean_estimate(1 - root_ratio)
        return (
            math.sqrt(max(squared, 0.0)),
            error / (2 * math.sqrt(max(squared, 1e-12))),
            0.0,
        )
    if metric in ("hellinger", "js") and len(columns) == 1:
        marginal_first = marginal_log_density(case, first, columns[0])
        marginal_second = marginal_log_density(case, second, columns[0])
        if metric == "hellinger":
            values = np.exp(
                0.5
                * (
                    marginal_second(first_draws[:, 0])
                    - marginal_first(first_draws[:, 0])
                )
            )
            squared, error = mean_estimate(1 - values)
            return (
                math.sqrt(max(squared, 0.0)),
                error / (2 * math.sqrt(max(squared, 1e-12))),
                0.0,
            )

        def half(draws_one: np.ndarray, own: Any, other: Any) -> np.ndarray:
            log_own, log_other = own(draws_one), other(draws_one)
            mixture = np.logaddexp(log_own, log_other)
            return np.asarray(0.5 * (math.log(2) + log_own - mixture))

        first_half = mean_estimate(
            half(first_draws[:, 0], marginal_first, marginal_second)
        )
        second_half = mean_estimate(
            half(second_draws[:, 0], marginal_second, marginal_first)
        )
        return (
            first_half[0] + second_half[0],
            math.hypot(first_half[1], second_half[1]),
            0.0,
        )
    if periods[0] is not None and metric in ("kuiper_ks", "w1", "ad"):
        return circle_marginal_estimate(
            case, sample_set, metric, first, second, columns
        )
    if metric == "ks":
        estimate = batch_estimate(
            lambda part: float(
                stats.ks_2samp(first_draws[part, 0], second_draws[part, 0]).statistic
            )
        )
        return (*estimate, 2 * math.sqrt(2 / SAMPLE_COUNT))
    if metric == "w1":
        estimate = batch_estimate(
            lambda part: float(
                stats.wasserstein_distance(first_draws[part, 0], second_draws[part, 0])
            )
        )
        widths = first_draws[:, 0].std() + second_draws[:, 0].std()
        return (*estimate, 2 * widths / math.sqrt(SAMPLE_COUNT))
    if metric == "ad":
        estimate = batch_estimate(
            lambda part: anderson_darling_distance(
                first_draws[part, 0], second_draws[part, 0]
            )
        )
        return (*estimate, 4 / SAMPLE_COUNT)
    if metric.startswith(("kl_", "w2", "hellinger_")):
        function = {
            "kl": closed_forms.gaussian_kl,
            "w2": closed_forms.gaussian_w2,
            "hellinger": closed_forms.gaussian_hellinger,
        }[metric.split("_")[0]]
        term = metric.split("_")[1] if "_" in metric else "total"
        return (
            *batch_estimate(
                lambda part: float(
                    getattr(
                        function(
                            *gaussian_moments(first_draws[part]),
                            *gaussian_moments(second_draws[part]),
                        ),
                        term,
                    )
                )
            ),
            0.0,
        )
    if metric == "railing_score":
        return railing_score_from_density(case, first, columns[0]), 0.0, 1e-7
    pytest.fail(f"no estimator for the metric {metric!r}")


def circle_marginal_estimate(
    case: ZooCase,
    sample_set: SampleSet,
    metric: str,
    first: int,
    second: int,
    columns: list[int],
) -> Estimate:
    """Estimate Kuiper's V, the circle W1 or AD after the pooled cut, from draws."""
    period = case.space.parameters[columns[0]].period
    assert period is not None
    first_angles = np.asarray(sample_set.samples[0, first, :, columns[0]])
    second_angles = np.asarray(sample_set.samples[0, second, :, columns[0]])
    if metric == "kuiper_ks":
        estimate = batch_estimate(
            lambda part: kuiper_statistic(first_angles[part], second_angles[part])
        )
        return (*estimate, 4 * math.sqrt(2 / SAMPLE_COUNT))
    if metric == "w1":
        estimate = batch_estimate(
            lambda part: circle_w1(first_angles[part], second_angles[part], period)
        )
        widths = sum(
            float(np.std(signed_arc(circular_mean(angles, period), angles, period)))
            for angles in (first_angles, second_angles)
        )
        return (*estimate, 2 * widths / math.sqrt(SAMPLE_COUNT))
    centre = circular_mean(np.concatenate([first_angles, second_angles]), period)

    def cut(part: slice) -> float:
        return anderson_darling_distance(
            signed_arc(centre, first_angles[part], period),
            signed_arc(centre, second_angles[part], period),
        )

    return (*batch_estimate(cut), 4 / SAMPLE_COUNT)


def sphere_vectors(case: ZooCase, sample_set: SampleSet, posterior: int) -> jax.Array:
    """Return the unit vectors of the sphere coordinates of one posterior."""
    sphere = case.space.spheres[0]
    azimuth = sample_set.samples[0, posterior, :, case.space.index(sphere.azimuth)]
    polar = sample_set.samples[0, posterior, :, case.space.index(sphere.polar)]
    vectors, _, inside = unit_vectors(azimuth, polar, sphere.polar_form)
    assert bool(jnp.all(inside))
    return vectors


def railing_score_from_density(case: ZooCase, posterior: int, column: int) -> float:
    """Return the density ratio between the bound and the peak along one axis."""
    bound = 0.0
    lower, upper = case.space.parameters[column].bounds
    side = 1.0 if math.isfinite(lower) else -1.0
    grid = side * np.linspace(0.0, 12.0, 120_001)
    points = np.zeros((len(grid), len(case.space)))
    points[:, column] = grid
    log_density = np.asarray(case.log_prob(points))[:, posterior]
    at_bound = log_density[np.argmin(np.abs(grid - bound))]
    assert np.all(np.isfinite(log_density))
    assert lower <= grid.min()
    assert grid.max() <= upper
    return float(math.exp(at_bound - log_density.max()))


def assert_reference_reproduced(case: ZooCase, sample_set: SampleSet) -> None:
    """Check every row of the reference table against its estimate from draws."""
    assert case.reference.schema == REFERENCE_SCHEMA
    assert len(case.reference) > 0
    keys = case.reference.select("pair", "metric", "parameters")
    assert keys.n_unique() == len(keys), (
        "every row needs its own pair, metric and subset"
    )
    draws = linearised(case, sample_set)
    log_densities = np.asarray(case.log_prob(sample_set.samples[0]))
    for row in case.reference.iter_rows(named=True):
        estimate, standard_error, bias = estimate_row(
            case, sample_set, draws, log_densities, row
        )
        tolerance = STANDARD_ERRORS * standard_error + bias
        assert abs(estimate - row["value"]) <= tolerance, (
            f"{case.name}: {row['metric']} of {row['pair']} over {row['parameters']}"
            f" is {row['value']!r}; draws give {estimate!r} ± {standard_error!r}"
        )


# * Every family reproduces its reference values


@pytest.mark.parametrize(
    "build",
    [
        lambda: shifted_gaussians(1.5, param_count=3),
        lambda: shifted_gaussians(
            0.8,
            direction=np.array([1.0, 1.0]),
            mean=np.array([3e-3, 1.0]),
            covariance=np.array([[4e-22, 1e-11], [1e-11, 1.0]]),
        ),
        lambda: scaled_gaussians(0.5),
        lambda: scaled_gaussians(
            1.7, param_count=3, axes=(0, 2), covariance=np.diag([1.0, 2.0, 0.5])
        ),
        lambda: rotated_gaussians(math.pi / 2),
        lambda: rotated_gaussians(
            0.6, param_count=3, plane=(2, 0), scales=np.array([2.0, 1.0, 0.1])
        ),
        lambda: gaussians(
            np.array([[0.0, 0.0], [1.0, -0.5], [0.3, 0.2]]),
            np.array([np.eye(2), [[2.0, 0.6], [0.6, 1.0]], [[0.5, -0.1], [-0.1, 0.3]]]),
        ),
        lambda: banana(),
        lambda: banana(0.1, param_count=4, variance=4.0),
        lambda: gaussian_mixture(4.0),
        lambda: gaussian_mixture(1.5, param_count=3, weights=(0.3, 0.7)),
        lambda: railed_gaussians([3.0, 0.5, 0.0, -1.5]),
        lambda: railed_gaussians([2.0, -0.5], param_count=1, side="upper"),
        lambda: wrapped_gaussians(1.0),
        lambda: wrapped_gaussians(
            -2.5, width=0.4, param_count=3, correlation=0.6, period=math.pi, lower=-1.0
        ),
        lambda: von_mises_fisher(2.0),
        lambda: von_mises_fisher(
            3.0, width=0.05, colatitude=0.1, polar_form="colatitude", euclidean_count=1
        ),
        lambda: von_mises_fisher(1.0, polar_form=PolarForm.SIN_LATITUDE),
        lambda: von_mises_fisher(0.5, colatitude=2.5, polar_form="cos_colatitude"),
    ],
)
def test_every_case_reproduces_its_reference_values(
    build: Callable[[], ZooCase], key: jax.Array
) -> None:
    """Draws reproduce every value of the reference table within sampling error."""
    case = build()
    assert_reference_reproduced(case, case.sample(key, SAMPLE_COUNT))


def test_location_and_shape_terms_combine_into_the_totals() -> None:
    """KL terms add, squared W2 terms add, and Hellinger coefficients multiply."""
    case = gaussians(
        np.array([[0.0, 1.0], [1.0, -0.5]]),
        np.array([[[1.0, 0.3], [0.3, 2.0]], [[2.0, 0.6], [0.6, 1.0]]]),
    )
    pair = ("posterior_0", "posterior_1")
    value = case.reference_value
    assert value("kl", pair) == pytest.approx(
        value("kl_location", pair) + value("kl_shape", pair)
    )
    assert value("w2", pair) ** 2 == pytest.approx(
        value("w2_location", pair) ** 2 + value("w2_shape", pair) ** 2
    )
    assert 1 - value("hellinger", pair) ** 2 == pytest.approx(
        (1 - value("hellinger_location", pair) ** 2)
        * (1 - value("hellinger_shape", pair) ** 2)
    )


# * Effect sizes of the canonical pairs


def test_shifted_gaussians_carry_the_shift_as_sigma_eff() -> None:
    """The Mahalanobis shift equals the requested shift, also for a correlated Σ."""
    covariance = np.array([[2.0, 0.8, 0.0], [0.8, 1.0, 0.2], [0.0, 0.2, 0.5]])
    case = shifted_gaussians(
        1.25, param_count=3, direction=np.array([1.0, -2.0, 0.5]), covariance=covariance
    )
    pair = ("reference", "shifted")
    assert case.reference_value("mahalanobis_shift", pair) == pytest.approx(1.25)
    assert case.reference_value("kl", pair) == pytest.approx(0.5 * 1.25**2)
    assert case.reference_value("kl_shape", pair) == 0.0
    assert case.reference_value("w2_shape", pair) == 0.0


def test_scaled_gaussians_carry_the_ratio_as_r_eff() -> None:
    """The scaled axis has the width ratio r and every shape term its 1D value."""
    case = scaled_gaussians(0.5, param_count=3)
    pair = ("reference", "scaled")
    assert case.reference_value("width_ratio", pair, "x0") == pytest.approx(0.5)
    assert case.reference_value("width_ratio", pair, "x1") == 1.0
    assert case.reference_value("kl_shape", pair) == pytest.approx(
        0.5 * (1 / 0.25 - 1 + math.log(0.25))
    )
    assert case.reference_value("w2_shape", pair) == pytest.approx(0.5)
    assert case.reference_value("kl_location", pair) == 0.0


def test_rotated_gaussians_differ_only_jointly() -> None:
    """Every marginal and per-parameter value is exactly zero; joint ones are not."""
    case = rotated_gaussians(math.pi / 2)
    pair = ("reference", "rotated")
    for metric in ("ks", "w1", "js", "ad", "hellinger", "mean_shift"):
        for parameter in ("x0", "x1"):
            assert case.reference_value(metric, pair, parameter) == 0.0
    assert case.reference_value("width_ratio", pair, "x0") == 1.0
    reference = case.posteriors[0]
    assert isinstance(reference, Gaussian)
    correlation = float(reference.covariance[0, 1] / reference.covariance[0, 0])
    assert correlation == pytest.approx(-(4 - 0.25) / (4 + 0.25))
    assert case.reference_value("kl", pair) == pytest.approx(
        2 * correlation**2 / (1 - correlation**2)
    )


def test_banana_defaults_are_the_moderately_twisted_target() -> None:
    """Moment metrics read zero; the KL divergences are ½ ln 19 and 18 − ½ ln 19."""
    case = banana()
    pair = ("banana", "gaussian")
    assert case.reference_value("kl", pair) == pytest.approx(0.5 * math.log(19))
    assert case.reference_value("kl", pair[::-1]) == pytest.approx(
        18 - 0.5 * math.log(19)
    )
    assert case.reference_value("mahalanobis_shift", pair) == 0.0
    assert case.reference_value("width_ratio", pair, "x1") == 1.0
    assert case.reference_value("std", "banana", "x1") == pytest.approx(math.sqrt(19))


def test_mixture_partners_carry_their_expected_relations() -> None:
    """The matched Gaussian shares the moments; the dropped mode obeys δ²/4 − KL."""
    separation = 3.0
    case = gaussian_mixture(separation)
    value = case.reference_value
    assert value("mahalanobis_shift", ("mixture", "gaussian")) == 0.0
    assert value("std", "gaussian", "x0") == pytest.approx(
        math.sqrt(1 + separation**2 / 4)
    )
    assert value("kl", ("mixture", "component")) == pytest.approx(
        separation**2 / 4 - value("kl", ("component", "mixture")), rel=1e-9
    )
    assert value("kl", ("component", "mixture")) < math.log(2)
    row = case.reference.filter(
        (pl.col("metric") == "kl") & (pl.col("pair") == ["mixture", "reweighted"])
    )
    assert row["basis"].to_list() == [ReferenceBasis.QUADRATURE]


def test_railed_scores_and_bounds() -> None:
    """Scores follow the offsets, and the space puts the finite bound at zero."""
    case = railed_gaussians([2.0, 0.0, -1.0], side="upper")
    scores = [case.reference_value("railing_score", name, "x0") for name in case.names]
    assert scores == pytest.approx([math.exp(-2.0), 1.0, 1.0])
    assert case.space["x0"].topology is Topology.INTERVAL
    assert case.space["x0"].bounds == (-math.inf, 0.0)
    assert case.space["x1"].topology is Topology.REAL


def test_wrapped_gaussians_straddle_the_boundary(key: jax.Array) -> None:
    """The reference centres on the cut, so draws fill both ends of the circle."""
    case = wrapped_gaussians(1.0, width=0.5)
    draws = np.asarray(case.sample(key, SAMPLE_COUNT).samples[0, 0, :, 0])
    assert np.all((draws >= 0) & (draws < AZIMUTH_PERIOD))
    assert np.mean(draws < 1) == pytest.approx(
        np.mean(draws > AZIMUTH_PERIOD - 1), abs=0.01
    )
    assert case.space["x0"].period == pytest.approx(AZIMUTH_PERIOD)
    assert case.reference_value(
        "mean_shift", ("reference", "shifted"), "x0"
    ) == pytest.approx(1.0)


def test_wrapped_shift_across_the_boundary_takes_the_short_arc() -> None:
    """A shift past the cut is measured along the shorter way round."""
    case = wrapped_gaussians(-2.0, width=0.5, lower=0.0)
    assert case.reference_value("circular_mean", "shifted", "x0") == pytest.approx(
        AZIMUTH_PERIOD - 1.0
    )
    assert case.reference_value(
        "mean_shift", ("reference", "shifted"), "x0"
    ) == pytest.approx(-2.0)


def test_von_mises_fisher_shift_crosses_the_pole(key: jax.Array) -> None:
    """A shift past the pole moves the centre to the opposite azimuth."""
    case = von_mises_fisher(2.0, width=0.1)
    shifted = case.sample(key, 1_000).samples[0, 1]
    vectors, _, _ = unit_vectors(shifted[:, 0], shifted[:, 1], PolarForm.LATITUDE)
    mean_vector = np.asarray(vectors.mean(axis=0))
    assert mean_vector[0] < 0
    assert case.reference_value("angular_shift", ("reference", "shifted")) == (
        pytest.approx(2.0)
    )
    assert case.space.spheres[0].polar_form is PolarForm.LATITUDE
    assert case.space["polar"].bounds == PolarForm.LATITUDE.bounds


# * AR(1) chains


def test_ar1_chains_reproduce_their_autocorrelation_time(key: jax.Array) -> None:
    """Lag-one correlation, stationary moments and batch-mean τ match the truth."""
    coefficients = (0.9, -0.5, 0.0)
    case = ar1_chains(coefficients)
    chains = case.sample(key, 20_000, walker_count=64)
    assert isinstance(chains, Chains)
    assert chains.shape == (20_000, 1, 64, 1, 3)
    samples = np.asarray(chains.samples[:, 0, :, 0, :])
    for index, coefficient in enumerate(coefficients):
        values = samples[..., index]
        tau = case.reference_value("tau", f"x{index}")
        step_count = values.shape[0]
        lag_one = np.sum(values[1:] * values[:-1]) / np.sum(values[:-1] ** 2)
        lag_error = math.sqrt((1 - coefficient**2) / values.size)
        assert abs(lag_one - coefficient) < STANDARD_ERRORS * lag_error
        walker_means = values.mean(axis=0)
        mean_error = math.sqrt(tau / values.size)
        assert abs(walker_means.mean()) < STANDARD_ERRORS * mean_error
        assert case.reference_value("std", f"x{index}") == 1.0
        assert values.std() == pytest.approx(1.0, abs=STANDARD_ERRORS * mean_error * 2)
        batch_tau = step_count * walker_means.var(ddof=1)
        tau_error = tau * math.sqrt(2 / (len(walker_means) - 1))
        assert abs(batch_tau - tau) < STANDARD_ERRORS * tau_error


def test_ar1_chains_relax_from_their_start(key: jax.Array) -> None:
    """A start far from zero decays as φ^t, the transient that burn-in removes."""
    case = ar1_chains([0.95])
    samples = np.asarray(case.sample(key, 60, walker_count=4_000, start=10.0).samples)
    walker_means = samples[:, 0, :, 0, 0].mean(axis=1)
    expected = 10.0 * 0.95 ** np.arange(60)
    spread = np.sqrt(1 - 0.95 ** (2 * np.arange(60)))
    assert np.all(
        np.abs(walker_means - expected) <= STANDARD_ERRORS * spread / math.sqrt(4_000)
    )
    assert samples[0, 0, 0, 0, 0] == 10.0


def test_ar1_draws_are_reproducible(key: jax.Array) -> None:
    """The same key gives the same chains; another key gives others."""
    case = ar1_chains([0.5])
    first = case.sample(key, 100, walker_count=3)
    again = case.sample(key, 100, walker_count=3)
    other = case.sample(jax.random.key(1), 100, walker_count=3)
    np.testing.assert_array_equal(first.samples, again.samples)
    assert not np.array_equal(first.samples, other.samples)


# * Draws and keys


def test_sample_gives_the_batch_layout(key: jax.Array) -> None:
    """Draws come as a SampleSet shaped (B, K, N, D) with equal weights."""
    case = gaussian_mixture(2.0, param_count=3)
    sample_set = case.sample(key, 500, batch_size=4)
    assert sample_set.shape == (4, 4, 500, 3)
    assert sample_set.samples.dtype == jnp.float64
    np.testing.assert_array_equal(sample_set.log_weights, 0.0)
    assert case.log_prob(sample_set.samples).shape == (4, 4, 500, 4)


def test_draws_depend_only_on_the_key_the_name_and_the_problem(key: jax.Array) -> None:
    """Same key, same draws; problem b is the same for any batch size."""
    case = shifted_gaussians(1.0)
    small = case.sample(key, 200, batch_size=2)
    large = case.sample(key, 200, batch_size=5)
    np.testing.assert_array_equal(large.samples[:2], small.samples)
    assert not np.array_equal(large.samples[0], large.samples[1])
    reference_draws = np.asarray(small.samples[0, 0])
    shifted_draws = np.asarray(small.samples[0, 1])
    assert abs(np.corrcoef(reference_draws[:, 1], shifted_draws[:, 1])[0, 1]) < 0.3


def test_posteriors_of_the_same_name_share_random_numbers(key: jax.Array) -> None:
    """Common random numbers: the reference of two shifted cases is drawn alike."""
    first = shifted_gaussians(1.0).sample(key, 100)
    second = shifted_gaussians(2.0).sample(key, 100)
    np.testing.assert_array_equal(first.samples[0, 0], second.samples[0, 0])
    np.testing.assert_allclose(
        second.samples[0, 1, :, 0] - first.samples[0, 1, :, 0], 1.0, rtol=1e-12
    )


def test_draws_respect_the_space(key: jax.Array) -> None:
    """Railed, wrapped and sphere draws lie inside their parameter bounds."""
    for case in (
        railed_gaussians([-2.0, 1.0], param_count=2),
        wrapped_gaussians(2.0, width=1.5, lower=-math.pi),
        von_mises_fisher(1.0, polar_form="sin_latitude", euclidean_count=1),
    ):
        draws = np.asarray(case.sample(key, 5_000, batch_size=2).samples)
        for index, parameter in enumerate(case.space.parameters):
            lower, upper = parameter.bounds
            values = draws[..., index]
            assert np.all(values >= lower)
            if parameter.topology is Topology.CIRCLE:
                assert np.all(values < upper)
            else:
                assert np.all(values <= upper)


# * Reference tables and lookups


def test_reference_value_names_the_missing_row() -> None:
    """An unknown row raises KeyError listing what the pair has."""
    case = shifted_gaussians(1.0)
    with pytest.raises(KeyError, match="'w3' for the pair") as error:
        case.reference_value("w3", ("reference", "shifted"))
    assert "w2 over ['x0', 'x1']" in str(error.value)
    assert case.reference_value("mean", "shifted", "x0") == 1.0
    assert case.reference_value(
        "mahalanobis_shift", ["reference", "shifted"], ["x0", "x1"]
    ) == pytest.approx(1.0)
    with pytest.raises(KeyError, match="that pair has: no rows"):
        case.reference_value("kl", ("shifted", "elsewhere"))


def test_symmetric_metrics_have_one_row_per_pair() -> None:
    """KL rows come in both orders; W2, Hellinger and marginal rows once."""
    covariances = np.stack([np.eye(2) * (1 + index) for index in range(3)])
    case = gaussians(np.zeros((3, 2)), covariances)
    table = case.reference
    assert table.filter(pl.col("metric") == "kl").height == 6
    assert table.filter(pl.col("metric") == "w2").height == 3
    assert table.filter(pl.col("metric") == "ks").height == 6
    assert set(table["basis"].to_list()) == {"closed form", "quadrature"}


# * Input checks


@pytest.mark.parametrize(
    ("build", "message"),
    [
        (lambda: shifted_gaussians(math.inf), "shift must be finite"),
        (lambda: shifted_gaussians(1.0, direction=np.zeros(2)), "zero vector"),
        (
            lambda: shifted_gaussians(1.0, mean=np.zeros(3)),
            r"mean must be shaped \(2,\)",
        ),
        (
            lambda: shifted_gaussians(1.0, param_count=0),
            "param_count must be at least 1",
        ),
        (lambda: scaled_gaussians(-1.0), "ratio must be finite and positive"),
        (lambda: scaled_gaussians(2.0, axes=(0, 0)), "distinct axis indices"),
        (lambda: scaled_gaussians(2.0, axes=(5,)), "outside the 2 parameters"),
        (
            lambda: scaled_gaussians(
                2.0, covariance=np.array([[1.0, 2.0], [2.0, 1.0]])
            ),
            "positive definite",
        ),
        (lambda: rotated_gaussians(1.0, param_count=1), "at least 2"),
        (lambda: rotated_gaussians(1.0, plane=(1, 1)), "distinct axis indices"),
        (lambda: rotated_gaussians(1.0, plane=(0, 1, 2)), "two entries each"),
        (lambda: rotated_gaussians(1.0, widths=(1.0,)), "two entries each"),
        (lambda: rotated_gaussians(1.0, widths=(1.0, 0.0)), "width must be finite"),
        (lambda: rotated_gaussians(1.0, scales=np.array([1.0, -1.0])), "positive"),
        (lambda: gaussians(np.zeros(2), np.eye(2)), r"means must be shaped \(K, D\)"),
        (lambda: gaussians(np.zeros((2, 2)), np.eye(2)), "covariances must be shaped"),
        (
            lambda: gaussians(np.zeros((2, 1)), np.ones((2, 1, 1)), names=["a", "a"]),
            "unique",
        ),
        (
            lambda: gaussians(np.zeros((2, 1)), np.ones((2, 1, 1)), names=["a"]),
            "1 names for 2 posteriors",
        ),
        (lambda: banana(0.1, param_count=1), "at least 2"),
        (lambda: gaussian_mixture(-1.0), "non-negative"),
        (lambda: gaussian_mixture(1.0, weights=(1.0, 0.0)), "weights must be positive"),
        (lambda: railed_gaussians([]), "at least one offset"),
        (lambda: railed_gaussians([math.nan]), "offset must be finite"),
        (lambda: railed_gaussians([1.0], side="left"), "'lower' or 'upper'"),
        (lambda: wrapped_gaussians(1.0, correlation=1.0), r"\(-1, 1\)"),
        (lambda: wrapped_gaussians(1.0, correlation=0.5), "at least 2 parameters"),
        (lambda: wrapped_gaussians(1.0, width=0.0), "width must be finite"),
        (lambda: von_mises_fisher(1.0, colatitude=4.0), r"colatitude must lie in \[0"),
        (lambda: von_mises_fisher(1.0, width=-0.1), "width must be finite"),
        (lambda: ar1_chains([]), "at least one coefficient"),
        (lambda: ar1_chains([1.0]), r"\(-1, 1\)"),
    ],
)
def test_builders_check_their_inputs(build: Callable[[], object], message: str) -> None:
    """Invalid builder inputs fail with a message that names the problem."""
    with pytest.raises(ValueError, match=message):
        build()


def test_case_containers_check_their_fields() -> None:
    """ZooCase and ChainCase reject names, posteriors or tables that do not fit."""
    case = shifted_gaussians(1.0)
    with pytest.raises(ValueError, match="2 names for 1 posteriors"):
        ZooCase(case.name, case.space, case.names, case.posteriors[:1], case.reference)
    with pytest.raises(ValueError, match="has 3 parameters, the space 2"):
        ZooCase(
            case.name,
            case.space,
            ("reference",),
            (Gaussian(np.zeros(3), np.eye(3)),),
            case.reference,
        )
    with pytest.raises(ValueError, match="schema"):
        ZooCase(
            case.name,
            case.space,
            case.names,
            case.posteriors,
            case.reference.drop("basis"),
        )
    with pytest.raises(ValueError, match="without surrounding whitespace"):
        ZooCase(
            case.name,
            case.space,
            ("reference", " shifted"),
            case.posteriors,
            case.reference,
        )
    with pytest.raises(ValueError, match="at least one posterior"):
        ZooCase(case.name, case.space, (), (), case.reference)
    chains = ar1_chains([0.5])
    with pytest.raises(ValueError, match="2 coefficients for 1 parameters"):
        ChainCase(chains.name, chains.space, (0.5, 0.2), chains.reference)
    with pytest.raises(ValueError, match="at least 1"):
        chains.sample(jax.random.key(0), 0)
    with pytest.raises(TypeError, match="must be an integer, got bool"):
        case.sample(jax.random.key(0), 10, batch_size=True)


def test_space_of_each_family() -> None:
    """Each builder declares the topology of its parameters."""
    sphere_case = von_mises_fisher(1.0, polar_form="cos_colatitude", euclidean_count=2)
    assert sphere_case.space.names == ("azimuth", "polar", "x2", "x3")
    assert sphere_case.space["azimuth"].topology is Topology.CIRCLE
    assert sphere_case.space["polar"].bounds == (-1.0, 1.0)
    wrapped_space = wrapped_gaussians(
        0.0, param_count=2, period=math.pi, lower=-1.0
    ).space
    assert wrapped_space["x0"].bounds == pytest.approx((-1.0, math.pi - 1.0))
    assert isinstance(banana().space, ParamSpace)
    assert banana(param_count=8).space.names == tuple(f"x{index}" for index in range(8))
