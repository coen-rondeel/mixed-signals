"""Tests for ``reweight``: recovery of zoo targets, k̂ flags, rows and errors."""

import inspect
import json
import math
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import stats

from mixed_signals import prng, zoo
from mixed_signals.core.result import RESULT_SCHEMA, Result, hash_arrays
from mixed_signals.core.samples import SampleSet, kish_effective_sample_size
from mixed_signals.core.space import Parameter, ParamSpace
from mixed_signals.engines import TARGET_SUFFIX, Proposal, Reweighting, reweight
from mixed_signals.errors import CapabilityError
from mixed_signals.mcmc import reduce_chains
from mixed_signals.priors import ErynPhysical, ErynPrior
from mixed_signals.reweight import pareto_k_threshold, pareto_tail_length

ROOT_KEY = prng.root_key(2027)
STANDARD_ERRORS = 5.0
unchecked_reweight = inspect.unwrap(reweight)


# * Helpers


def table_rows(result: Result, metric: str | None = None) -> list[dict[str, Any]]:
    """Return the rows of a result, those of one metric if named."""
    rows = result.table.to_dicts()
    return rows if metric is None else [row for row in rows if row["metric"] == metric]


def table_rows_of_problem(result: Result, problem: int) -> list[dict[str, Any]]:
    """Return the rows of one problem."""
    return [row for row in table_rows(result) if row["problem"] == problem]


def required(array: jax.Array | None) -> jax.Array:
    """Return a field that the test knows is set."""
    assert array is not None
    return array


def coordinate(axis: int) -> Callable[[Any], Any]:
    """Return the function that picks one coordinate of samples."""
    return lambda samples: samples[:, axis]


def squared_deviation(axis: int, centre: float) -> Callable[[Any], Any]:
    """Return the function that squares the distance of one coordinate to a centre."""
    return lambda samples: (samples[:, axis] - centre) ** 2


def self_normalised(
    samples: jax.Array, log_weights: jax.Array, function: Callable[[Any], Any]
) -> tuple[float, float]:
    """Return a weighted mean of a function and its standard error from Eq. 5.

    The squared error is the sum of w² (h − mean)² for normalised weights w and
    independent draws (Vehtari et al. 2024).
    """
    weights = jax.nn.softmax(log_weights)
    values = function(samples)
    mean = jnp.sum(weights * values)
    error = jnp.sqrt(jnp.sum(weights**2 * (values - mean) ** 2))
    return float(mean), float(error)


def approximation_set(
    ratio: float, axes: tuple[int, ...], sample_count: int, batch_size: int, tag: str
) -> SampleSet:
    """Return draws from the narrower or wider zoo posterior as an approximation q.

    The set stores log_q of the scaled posterior, and the density of the zoo
    reference as its log-likelihood with a flat log-prior, so that the
    reference is the target.
    """
    case = zoo.scaled_gaussians(ratio, param_count=2, axes=axes)
    draws = case.sample(
        prng.derive_key(ROOT_KEY, tag), sample_count, batch_size=batch_size
    )
    scaled, reference = case.names.index("scaled"), case.names.index("reference")
    points = draws.samples[:, scaled : scaled + 1]
    log_density = case.log_prob(points)
    return SampleSet(
        points,
        log_like=log_density[..., reference],
        log_prior=jnp.zeros(points.shape[:-1]),
        log_q=log_density[..., scaled],
    )


def gaussian_log_density(
    mean: np.ndarray, covariance: np.ndarray
) -> Callable[[np.ndarray], np.ndarray]:
    """Return a NumPy log density of a Gaussian, as user code would write it."""
    distribution = stats.multivariate_normal(mean=mean, cov=covariance)

    def log_density(points: np.ndarray) -> np.ndarray:
        return np.atleast_1d(distribution.logpdf(points))

    return log_density


class GaussianPrior:
    """A new prior from outside the package: logpdf and rvs of N(0, variance I)."""

    def __init__(self, variance: float, param_count: int = 2) -> None:
        """Store the variance."""
        self.variance = variance
        self.param_count = param_count

    def logpdf(self, samples: Any) -> np.ndarray:
        """Return the log density at samples shaped (N, D)."""
        return gaussian_log_density(
            np.zeros(self.param_count), self.variance * np.eye(self.param_count)
        )(np.asarray(samples))

    def rvs(self, size: int | tuple[int, ...], *, key: Any) -> Any:
        """Draw from the prior with the JAX key."""
        shape = (size,) if isinstance(size, int) else size
        return jnp.sqrt(self.variance) * jax.random.normal(
            key, (*shape, self.param_count)
        )


class ConstantPrior:
    """A prior whose density is one value everywhere, such as NaN or −∞."""

    def __init__(self, value: float) -> None:
        """Store the value of the log density."""
        self.value = value

    def logpdf(self, samples: Any) -> np.ndarray:
        """Return the value at every sample."""
        return np.full(len(samples), self.value)

    def rvs(self, size: int | tuple[int, ...], *, key: Any) -> Any:
        """Draw zeros; the engine never draws from the new prior."""
        shape = (size,) if isinstance(size, int) else size
        return jnp.zeros((*shape, 2))


class BoxPrior:
    """A uniform prior on [-half_width, half_width] in each of two dimensions."""

    def __init__(self, half_width: float) -> None:
        """Store the half width."""
        self.half_width = half_width

    def logpdf(self, samples: Any) -> np.ndarray:
        """Return the constant log density inside the box and −∞ outside."""
        inside = np.all(np.abs(np.asarray(samples)) <= self.half_width, axis=-1)
        return np.where(inside, -math.log((2 * self.half_width) ** 2), -np.inf)

    def rvs(self, size: int | tuple[int, ...], *, key: Any) -> Any:
        """Draw uniformly in the box."""
        shape = (size,) if isinstance(size, int) else size
        return jax.random.uniform(
            key, (*shape, 2), minval=-self.half_width, maxval=self.half_width
        )


def real_space(param_count: int) -> ParamSpace:
    """Return a space of real-line parameters x0, x1, ..."""
    return ParamSpace(
        parameters=tuple(Parameter(name=f"x{index}") for index in range(param_count))
    )


def box_set(sample_count: int, inside_count: int | None = None) -> SampleSet:
    """Return samples with a stored uniform log-prior on [-5, 5]².

    With ``inside_count`` the first samples lie in [-1, 1]² and the rest
    outside it, otherwise all samples are uniform in the large box.
    """
    key = prng.derive_key(ROOT_KEY, "box", sample_count)
    if inside_count is None:
        samples = jax.random.uniform(key, (sample_count, 2), minval=-5.0, maxval=5.0)
    else:
        inside = jax.random.uniform(key, (inside_count, 2), minval=-1.0, maxval=1.0)
        outside = jax.random.uniform(
            key, (sample_count - inside_count, 2), minval=3.0, maxval=4.0
        )
        samples = jnp.concatenate([inside, outside])
    return SampleSet(samples, log_prior=jnp.full(sample_count, -math.log(100.0)))


# * Recovery of the target


def test_an_approximation_is_reweighted_to_the_zoo_reference() -> None:
    """Draws from a wider q reweight to N(0, I): moments, evidence and no flag."""
    sample_set = approximation_set(1.3, (0,), 20_000, 1, "recovery")
    new_set, result = reweight(
        sample_set, ["q"], real_space(2), proposal=Proposal.APPROXIMATION
    )
    samples, log_weights = new_set.samples[0, 0], new_set.log_weights[0, 0]
    for axis in range(2):
        mean, error = self_normalised(samples, log_weights, coordinate(axis))
        assert abs(mean) < STANDARD_ERRORS * error
        second, error = self_normalised(
            samples, log_weights, squared_deviation(axis, 0)
        )
        assert abs(second - 1.0) < STANDARD_ERRORS * error
    cross, error = self_normalised(
        samples, log_weights, lambda points: points[:, 0] * points[:, 1]
    )
    assert abs(cross) < STANDARD_ERRORS * error
    (evidence,) = table_rows(result, "log_evidence_ratio")
    assert abs(evidence["value"]) < STANDARD_ERRORS * evidence["uncertainty"]
    (khat,) = table_rows(result, "pareto_k")
    assert khat["value"] < pareto_k_threshold(20_000)
    assert all(row["warnings"] == [] for row in table_rows(result))


def test_a_likelihood_change_recovers_the_known_evidence_ratio() -> None:
    """Shifting a Gaussian likelihood and its constants gives ln Z ratio = −3."""
    case = zoo.shifted_gaussians(0.5, param_count=3)
    draws = case.sample(prng.derive_key(ROOT_KEY, "likelihood"), 20_000)
    reference = case.names.index("reference")
    points = draws.samples[:, reference : reference + 1]
    stored = case.log_prob(points)[..., reference] + 2.0
    shifted_mean = np.array([0.5, 0.0, 0.0])
    shifted = gaussian_log_density(shifted_mean, np.eye(3))
    new_set, result = reweight(
        SampleSet(points, log_like=stored),
        ["reference"],
        case.space,
        log_like=lambda samples: shifted(samples) - 1.0,
    )
    (evidence,) = table_rows(result, "log_evidence_ratio")
    assert abs(evidence["value"] + 3.0) < STANDARD_ERRORS * evidence["uncertainty"]
    mean, error = self_normalised(
        new_set.samples[0, 0], new_set.log_weights[0, 0], coordinate(0)
    )
    assert abs(mean - 0.5) < STANDARD_ERRORS * error
    # * The set stores the new likelihood, so it can be reweighted again.
    expected = shifted(np.asarray(new_set.samples[0, 0])) - 1.0
    np.testing.assert_allclose(required(new_set.log_like)[0, 0], expected, rtol=1e-12)


def test_a_prior_change_recovers_the_conjugate_closed_form() -> None:
    """Gaussian likelihood and priors: the evidence ratio and moments are exact."""
    likelihood_mean = np.array([0.4, -0.2])
    likelihood_covariance = 0.3**2 * np.eye(2)
    old_variance, new_variance = 1.0, 2.0

    def posterior(prior_variance: float) -> tuple[np.ndarray, np.ndarray]:
        precision = np.linalg.inv(likelihood_covariance) + np.eye(2) / prior_variance
        covariance = np.linalg.inv(precision)
        return (
            covariance @ np.linalg.inv(likelihood_covariance) @ likelihood_mean,
            covariance,
        )

    (old_mean, old_covariance), (new_mean, new_covariance) = (
        posterior(old_variance),
        posterior(new_variance),
    )
    case = zoo.gaussians(
        [old_mean, new_mean], [old_covariance, new_covariance], names=["old", "new"]
    )
    draws = case.sample(prng.derive_key(ROOT_KEY, "conjugate"), 20_000)
    points = draws.samples[:, :1]
    stored_prior = GaussianPrior(old_variance).logpdf(points[0, 0])
    stored_like = gaussian_log_density(likelihood_mean, likelihood_covariance)(
        np.asarray(points[0, 0])
    )
    sample_set = SampleSet(points[0, 0], log_like=stored_like, log_prior=stored_prior)
    new_set, result = reweight(
        sample_set, ["old"], case.space, prior=GaussianPrior(new_variance)
    )
    evidence_ratio = stats.multivariate_normal(
        np.zeros(2), likelihood_covariance + new_variance * np.eye(2)
    ).logpdf(likelihood_mean) - stats.multivariate_normal(
        np.zeros(2), likelihood_covariance + old_variance * np.eye(2)
    ).logpdf(likelihood_mean)
    (evidence,) = table_rows(result, "log_evidence_ratio")
    assert abs(evidence["value"] - evidence_ratio) < (
        STANDARD_ERRORS * evidence["uncertainty"]
    )
    samples, log_weights = new_set.samples[0, 0], new_set.log_weights[0, 0]
    for axis in range(2):
        mean, error = self_normalised(samples, log_weights, coordinate(axis))
        assert abs(mean - new_mean[axis]) < STANDARD_ERRORS * error
        second, error = self_normalised(
            samples, log_weights, squared_deviation(axis, new_mean[axis])
        )
        assert abs(second - new_covariance[axis, axis]) < STANDARD_ERRORS * error
    np.testing.assert_allclose(
        required(new_set.log_prior)[0, 0],
        GaussianPrior(new_variance).logpdf(samples),
        rtol=1e-12,
    )


def test_a_prior_and_a_likelihood_change_together_as_a_product() -> None:
    """The log ratio of a joint change is the sum of the two separate ones."""
    sample_set = approximation_set(1.3, (0,), 2_000, 1, "joint")
    posterior = SampleSet(
        sample_set.samples,
        log_like=sample_set.log_like,
        log_prior=jnp.zeros_like(required(sample_set.log_like)),
    )
    prior, shifted = GaussianPrior(4.0), gaussian_log_density(np.ones(2), np.eye(2))
    only_prior = reweight(posterior, ["a"], real_space(2), prior=prior)
    only_like = reweight(posterior, ["a"], real_space(2), log_like=shifted)
    both = reweight(posterior, ["a"], real_space(2), prior=prior, log_like=shifted)
    np.testing.assert_allclose(
        both.sample_set.log_weights,
        only_prior.sample_set.log_weights + only_like.sample_set.log_weights,
        rtol=1e-12,
        atol=1e-12,
    )


# * k̂ on the zoo


def test_heavy_tailed_weights_are_flagged_and_light_ones_are_not() -> None:
    """A narrower q has k = 1 − r²: 0.84 flags most sets, and 0.19 flags none."""
    heavy = approximation_set(0.4, (0, 1), 10_000, 20, "heavy")
    new_set, result = reweight(
        heavy, ["q"], real_space(2), proposal=Proposal.APPROXIMATION
    )
    shapes = np.array([row["value"] for row in table_rows(result, "pareto_k")])
    assert shapes.shape == (20,)
    error = shapes.std(ddof=1) / math.sqrt(20)
    assert abs(shapes.mean() - 0.84) < 4 * error
    flagged = sum(
        all(len(row["warnings"]) == 1 for row in table_rows_of_problem(result, problem))
        for problem in range(20)
    )
    assert flagged >= 12
    for row in table_rows(result):
        for record in row["warnings"]:
            assert record["category"] == "ImportanceWeightWarning"
            assert "exceeds the threshold 0.7 for 10000 draws" in record["message"]
    light = approximation_set(0.9, (0, 1), 10_000, 20, "light")
    _, quiet = reweight(light, ["q"], real_space(2), proposal="approximation")
    assert all(row["warnings"] == [] for row in table_rows(quiet))
    assert new_set.shape == (20, 1, 10_000, 2)


def test_a_flat_tail_has_no_shape_and_no_warning() -> None:
    """A target equal to q gives weights of one value: k̂ = −∞, ESS = N, zero error."""
    case = zoo.gaussians(np.zeros((1, 2)), np.eye(2)[None])
    points = case.sample(prng.derive_key(ROOT_KEY, "flat"), 2_000).samples
    log_q = case.log_prob(points)[..., 0]
    sample_set = SampleSet(
        points, log_like=log_q, log_prior=jnp.zeros_like(log_q), log_q=log_q
    )
    new_set, result = reweight(
        sample_set, ["q"], case.space, proposal=Proposal.APPROXIMATION
    )
    np.testing.assert_array_equal(new_set.log_weights, 0.0)
    (khat,) = table_rows(result, "pareto_k")
    assert khat["value"] == -math.inf
    (size,) = table_rows(result, "importance_ess")
    assert size["value"] == pytest.approx(2_000.0, rel=1e-12)
    (evidence,) = table_rows(result, "log_evidence_ratio")
    assert evidence["value"] == 0.0
    assert evidence["uncertainty"] == 0.0
    assert all(row["warnings"] == [] for row in table_rows(result))


def test_a_uniform_prior_change_has_a_flat_tail_and_a_closed_form_evidence() -> None:
    """Uniform on [-5, 5]² to [-1, 1]²: k̂ = −∞ and ln(25 n_inside/N) exactly."""
    sample_set = box_set(5_000)
    new_set, result = reweight(sample_set, ["box"], real_space(2), prior=BoxPrior(1.0))
    inside = int(jnp.sum(jnp.all(jnp.abs(sample_set.samples[0, 0]) <= 1.0, axis=-1)))
    assert 100 < inside < 300
    (khat,) = table_rows(result, "pareto_k")
    assert khat["value"] == -math.inf
    assert khat["warnings"] == []
    (evidence,) = table_rows(result, "log_evidence_ratio")
    assert evidence["value"] == pytest.approx(math.log(25 * inside / 5_000), rel=1e-12)
    assert int(jnp.sum(new_set.log_weights > -jnp.inf)) == inside
    settings = json.loads(khat["estimator_settings"])
    assert settings["draws"] == inside


def test_too_few_kept_samples_leave_khat_undefined_and_warn() -> None:
    """With 10 samples kept, k̂ is NaN and every row of the set carries a warning."""
    sample_set = box_set(100, inside_count=10)
    new_set, result = reweight(sample_set, ["box"], real_space(2), prior=BoxPrior(1.0))
    (khat,) = table_rows(result, "pareto_k")
    assert math.isnan(khat["value"])
    rows = table_rows(result)
    assert len(rows) == 3
    for row in rows:
        (record,) = row["warnings"]
        assert record["category"] == "ImportanceWeightWarning"
        assert "at least 5 tail draws, so at least 25 samples" in record["message"]
        assert "but 10 keep one" in record["message"]
    assert int(jnp.sum(new_set.log_weights > -jnp.inf)) == 10


class OnePeakPrior:
    """A new prior whose log density is zero except at the first sample, 1e-6 there."""

    def logpdf(self, samples: Any) -> np.ndarray:
        """Return the log densities, with the peak at the first sample."""
        values = np.zeros(len(samples))
        values[0] = 1e-6
        return values

    def rvs(self, size: int | tuple[int, ...], *, key: Any) -> Any:
        """Draw zeros; the engine never draws from the new prior."""
        shape = (size,) if isinstance(size, int) else size
        return jnp.zeros((*shape, 2))


def test_an_undefined_fit_leaves_khat_nan_and_warns_in_every_row() -> None:
    """Many tied weights below one larger weight make the fit NaN, with a warning."""
    # * The top 95 log weights are one value of 1e-6 and 94 ties at 0, so the quartile
    # * excess is zero, while 1e-6 lies above the flat tolerance of 1e-12.
    sample_set = SampleSet(
        jnp.arange(2_000.0).reshape(1_000, 2) / 1_000,
        log_prior=jnp.zeros(1_000),
    )
    new_set, result = reweight(sample_set, ["a"], real_space(2), prior=OnePeakPrior())
    assert float(new_set.log_weights[0, 0, 0]) == 1e-6
    (khat,) = table_rows(result, "pareto_k")
    assert math.isnan(khat["value"])
    assert json.loads(khat["estimator_settings"])["tail_draws"] == 94
    rows = table_rows(result)
    assert len(rows) == 3
    for row in rows:
        (record,) = row["warnings"]
        assert record["category"] == "ImportanceWeightWarning"
        assert record["message"] == (
            "k-hat of posterior 'a' in problem 0 is undefined because the generalized"
            " Pareto fit of its tail failed, for example when many of the largest"
            " weights are equal, so the weights cannot be judged"
        )
    smooth_set, smooth_result = reweight(
        sample_set, ["a"], real_space(2), prior=OnePeakPrior(), smooth=True
    )
    np.testing.assert_array_equal(smooth_set.log_weights, new_set.log_weights)
    assert math.isnan(table_rows(smooth_result, "pareto_k")[0]["value"])
    for row in table_rows(smooth_result):
        (record,) = row["warnings"]
        assert record["message"].endswith("; smoothing leaves them unchanged")


def test_smoothing_replaces_only_the_tail_and_keeps_the_shape() -> None:
    """Smoothed weights differ from raw ones in the M largest, below the raw maximum."""
    heavy = approximation_set(0.4, (0, 1), 10_000, 1, "smoothing")
    raw_set, raw_result = reweight(
        heavy, ["q"], real_space(2), proposal=Proposal.APPROXIMATION
    )
    smooth_set, smooth_result = reweight(
        heavy, ["q"], real_space(2), proposal=Proposal.APPROXIMATION, smooth=True
    )
    raw, smoothed = raw_set.log_weights[0, 0], smooth_set.log_weights[0, 0]
    tail_draws = pareto_tail_length(10_000)
    order = np.argsort(np.asarray(raw))
    changed = np.flatnonzero(np.asarray(raw != smoothed))
    assert set(changed) <= set(order[-tail_draws:])
    assert len(changed) > 0.9 * tail_draws
    assert float(jnp.max(smoothed)) <= float(jnp.max(raw))
    assert table_rows(smooth_result, "log_evidence_ratio")[0]["estimator"] == "psis"
    assert table_rows(raw_result, "log_evidence_ratio")[0]["estimator"] == "importance"
    assert table_rows(smooth_result, "pareto_k")[0]["value"] == pytest.approx(
        table_rows(raw_result, "pareto_k")[0]["value"], rel=1e-12
    )
    expected_size = float(kish_effective_sample_size(smoothed))
    assert smooth_set.n_eff is not None
    np.testing.assert_allclose(smooth_set.n_eff[0, 0], expected_size, rtol=1e-12)
    assert table_rows(smooth_result, "importance_ess")[0]["value"] == pytest.approx(
        expected_size, rel=1e-12
    )


def test_smoothing_a_set_with_too_few_samples_leaves_it_raw() -> None:
    """Without a tail fit there is nothing to smooth."""
    sample_set = box_set(100, inside_count=10)
    raw_set, _ = reweight(sample_set, ["box"], real_space(2), prior=BoxPrior(1.0))
    smooth_set, result = reweight(
        sample_set, ["box"], real_space(2), prior=BoxPrior(1.0), smooth=True
    )
    np.testing.assert_array_equal(smooth_set.log_weights, raw_set.log_weights)
    assert table_rows(result, "log_evidence_ratio")[0]["estimator"] == "psis"


# * MCMC input


def test_mcmc_input_carries_its_relative_efficiency() -> None:
    """n_eff becomes R times the Kish value of the new weights; R divides the error."""
    chains = zoo.ar1_chains([0.9, 0.9]).sample(
        prng.derive_key(ROOT_KEY, "mcmc"), 1_900, walker_count=32
    )
    reduced = reduce_chains(chains).sample_set
    assert reduced.n_eff is not None
    reference = gaussian_log_density(np.zeros(2), np.eye(2))
    stored = reference(np.asarray(reduced.samples[0, 0]))
    sample_set = SampleSet(
        reduced.samples[0, 0], n_eff=float(reduced.n_eff[0, 0]), log_like=stored
    )
    shifted = gaussian_log_density(np.array([0.5, 0.0]), np.eye(2))
    new_set, result = reweight(sample_set, ["chain"], real_space(2), log_like=shifted)
    sample_count = sample_set.shape.sample
    efficiency = float(reduced.n_eff[0, 0]) / sample_count
    assert 0.5 < efficiency < 1.0
    log_ratio = shifted(np.asarray(sample_set.samples[0, 0])) - stored
    kish = float(kish_effective_sample_size(jnp.asarray(log_ratio)))
    (size,) = table_rows(result, "importance_ess")
    assert size["value"] == pytest.approx(efficiency * kish, rel=1e-10)
    assert new_set.n_eff is not None
    assert float(new_set.n_eff[0, 0]) == pytest.approx(size["value"], rel=1e-12)
    assert json.loads(size["estimator_settings"])["relative_efficiency"] == (
        pytest.approx(efficiency, rel=1e-12)
    )
    weights = np.exp(log_ratio - np.logaddexp.reduce(log_ratio))
    equal = np.full(sample_count, 1 / sample_count)
    expected_error = math.sqrt(np.sum((weights - equal) ** 2) / efficiency)
    (evidence,) = table_rows(result, "log_evidence_ratio")
    assert evidence["uncertainty"] == pytest.approx(expected_error, rel=1e-10)
    (khat,) = table_rows(result, "pareto_k")
    settings = json.loads(khat["estimator_settings"])
    assert settings["tail_draws"] == pareto_tail_length(sample_count, efficiency)
    assert settings["relative_efficiency"] == pytest.approx(efficiency, rel=1e-12)


# * Coordinates


@pytest.fixture(scope="module")
def amplitude_priors() -> tuple[Any, Any]:
    """Return eryn priors with a log-uniform and a uniform amplitude."""
    eryn_priors = pytest.importorskip("eryn.priors")
    old = eryn_priors.ProbDistContainer(
        {
            "ln_amplitude": eryn_priors.LogUniform(
                1e-23, 1e-21, name="ln_amplitude", name_phys="amplitude"
            ),
            "phase": eryn_priors.UniformDistribution(0.0, 2 * math.pi),
        }
    )
    new = eryn_priors.ProbDistContainer(
        {
            "amplitude": eryn_priors.UniformDistribution(1e-23, 1e-21),
            "phase": eryn_priors.UniformDistribution(0.0, 2 * math.pi),
        }
    )
    return old, new


def test_a_transform_moves_the_set_to_the_physical_coordinates(
    amplitude_priors: tuple[Any, Any],
) -> None:
    """Log-uniform to uniform amplitude: log ratio ln A + ln(ln b/a) − ln(b − a)."""
    old, new = amplitude_priors
    space = ParamSpace.from_eryn_priors(old, periodic={"phase": 2 * math.pi})
    sampling = ErynPrior(old).rvs(3_000, key=prng.derive_key(ROOT_KEY, "eryn"))
    sample_set = SampleSet(sampling, log_prior=ErynPrior(old).logpdf(sampling))
    new_set, result = reweight(
        sample_set,
        ["run"],
        space,
        prior=ErynPrior(new, coordinates="physical"),
        transform=ErynPhysical(old),
    )
    amplitude = np.exp(np.asarray(sampling[:, 0]))
    lower, upper = 1e-23, 1e-21
    expected = (
        np.log(amplitude) + math.log(math.log(upper / lower)) - math.log(upper - lower)
    )
    np.testing.assert_allclose(
        new_set.log_weights[0, 0], expected, rtol=1e-9, atol=1e-10
    )
    np.testing.assert_allclose(new_set.samples[0, 0, :, 0], amplitude, rtol=1e-12)
    for row in table_rows(result):
        assert row["space"] == "physical(eryn)"
        assert row["parameters"] == ["amplitude", "phase"]


def test_without_a_transform_the_space_column_reads_identity() -> None:
    """The set stays in its coordinates, and the rows say so."""
    _, result = reweight(box_set(500), ["box"], real_space(2), prior=BoxPrior(4.0))
    assert {row["space"] for row in table_rows(result)} == {"identity"}
    assert {tuple(row["parameters"]) for row in table_rows(result)} == {("x0", "x1")}


# * The boundary


def test_densities_are_called_in_chunks_of_numpy_arrays() -> None:
    """The prior and the likelihood receive NumPy arrays of at most chunk_size rows."""
    sample_set = approximation_set(1.2, (0,), 50, 1, "chunks")
    posterior = SampleSet(
        sample_set.samples,
        log_like=sample_set.log_like,
        log_prior=sample_set.log_prior,
    )
    sizes: list[tuple[type, int]] = []

    def log_like(points: Any) -> np.ndarray:
        sizes.append((type(points), len(points)))
        return np.zeros(len(points))

    class RecordingPrior(GaussianPrior):
        def logpdf(self, samples: Any) -> np.ndarray:
            sizes.append((type(samples), len(samples)))
            return super().logpdf(samples)

    reweight(
        posterior,
        ["a"],
        real_space(2),
        prior=RecordingPrior(2.0),
        log_like=log_like,
        chunk_size=20,
    )
    assert [size for _, size in sizes] == [20, 20, 10, 20, 20, 10]
    assert all(received is np.ndarray for received, _ in sizes)


def test_a_torch_backend_receives_tensors() -> None:
    """With backend="torch" the new likelihood gets a torch tensor."""
    torch = pytest.importorskip("torch")
    sample_set = approximation_set(1.2, (0,), 50, 1, "torch")
    posterior = SampleSet(sample_set.samples, log_like=sample_set.log_like)
    received: list[type] = []

    def log_like(points: Any) -> Any:
        received.append(type(points))
        return -0.5 * (points**2).sum(dim=-1)

    reweight(posterior, ["a"], real_space(2), log_like=log_like, backend="torch")
    assert received == [torch.Tensor]


def test_each_problem_gets_its_own_likelihood() -> None:
    """With B = 2, callable b is called only on the samples of problem b."""
    first = jnp.linspace(-1.0, 1.0, 60).reshape(30, 2)
    second = first + 10.0
    samples = jnp.stack([first, second])[:, None]
    stored = jnp.zeros((2, 1, 30))
    seen: dict[int, tuple[float, float]] = {}

    def make(problem: int) -> Callable[[np.ndarray], np.ndarray]:
        def log_like(points: np.ndarray) -> np.ndarray:
            seen[problem] = (float(points.min()), float(points.max()))
            return -0.5 * np.sum((points - 10.0 * problem) ** 2, axis=-1)

        return log_like

    new_set, result = reweight(
        SampleSet(samples, log_like=stored),
        ["a"],
        real_space(2),
        log_like=[make(0), make(1)],
    )
    assert seen[0] == pytest.approx((-1.0, 1.0))
    assert seen[1] == pytest.approx((9.0, 11.0))
    assert new_set.shape == (2, 1, 30, 2)
    assert {row["problem"] for row in table_rows(result)} == {0, 1}


def test_a_narrow_float_result_is_recorded_in_the_rows_of_its_problem() -> None:
    """A float32 likelihood puts one PrecisionWarning in every row, and raises none."""
    sample_set = approximation_set(1.2, (0,), 200, 1, "float32")
    posterior = SampleSet(sample_set.samples, log_like=sample_set.log_like)
    _, result = reweight(
        posterior,
        ["a"],
        real_space(2),
        log_like=lambda points: -0.5 * (points**2).sum(axis=-1).astype(np.float32),
    )
    for row in table_rows(result):
        categories = [record["category"] for record in row["warnings"]]
        assert categories == ["PrecisionWarning"]


def test_new_densities_at_zero_weight_samples_are_stored_as_minus_infinity() -> None:
    """Samples without weight no longer enter any estimate, so their density is −∞."""
    sample_set = SampleSet(
        jnp.arange(60.0).reshape(30, 2) / 30,
        log_like=jnp.zeros(30),
        log_weights=jnp.where(jnp.arange(30) < 28, 0.0, -jnp.inf),
    )

    def log_like(points: np.ndarray) -> np.ndarray:
        values = np.zeros(len(points))
        values[-2:] = [np.nan, np.inf]
        return values

    new_set, _ = reweight(sample_set, ["a"], real_space(2), log_like=log_like)
    new_log_like = required(new_set.log_like)
    assert bool(jnp.all(new_log_like[0, 0, 28:] == -jnp.inf))
    assert bool(jnp.all(new_log_like[0, 0, :28] == 0.0))
    assert bool(jnp.all(new_set.log_weights[0, 0, 28:] == -jnp.inf))


# * Rows


def test_every_weight_set_gets_three_rows_with_their_columns() -> None:
    """Rows are pareto_k, importance_ess and log_evidence_ratio, in this order."""
    sample_set = approximation_set(1.3, (0,), 2_000, 2, "rows")
    new_set, result = reweight(
        sample_set,
        ["q"],
        real_space(2),
        proposal="approximation",
        target_names=["target"],
    )
    assert isinstance(result, Result)
    assert result.table.schema == RESULT_SCHEMA
    rows = table_rows(result)
    assert len(rows) == 6
    assert [row["metric"] for row in rows[:3]] == [
        "pareto_k",
        "importance_ess",
        "log_evidence_ratio",
    ]
    assert [row["problem"] for row in rows] == [0, 0, 0, 1, 1, 1]
    assert [row["estimator"] for row in rows[:3]] == ["psis", "kish", "importance"]
    for row in rows:
        assert row["pair"] == ["q", "target"]
        assert row["space"] == "identity"
        assert row["parameters"] == ["x0", "x1"]
        for column in (
            "null_mean",
            "null_std",
            "null_draws",
            "null_recipe",
            "p_value",
            "sigma_sig",
            "sigma_eff",
            "r_eff",
        ):
            assert row[column] is None
    assert [row["uncertainty_method"] for row in rows[:3]] == [
        None,
        None,
        "delta_method",
    ]
    assert rows[0]["uncertainty"] is None
    assert rows[2]["uncertainty"] > 0.0
    assert json.loads(rows[0]["estimator_settings"]) == {
        "draws": 2_000,
        "tail_draws": pareto_tail_length(2_000),
        "shape_prior": True,
        "relative_efficiency": 1.0,
    }
    assert json.loads(rows[1]["estimator_settings"]) == {"relative_efficiency": 1.0}
    assert json.loads(rows[2]["estimator_settings"]) == {"relative_efficiency": 1.0}
    for index, problem in ((0, 0), (3, 1)):
        provenance = rows[index]["provenance"]
        assert provenance["seed"] is None
        assert provenance["prng_key"] is None
        assert provenance["input_hashes"] == [
            hash_arrays(
                sample_set.samples[problem, 0], sample_set.log_weights[problem, 0]
            ),
            hash_arrays(new_set.samples[problem, 0], new_set.log_weights[problem, 0]),
        ]
    Result(result.table)


def test_default_target_names_follow_the_source_names() -> None:
    """Without target_names, each target is the source name plus the suffix."""
    sample_set = approximation_set(1.3, (0,), 200, 1, "names")
    _, result = reweight(sample_set, ["q"], real_space(2), proposal="approximation")
    assert TARGET_SUFFIX == ":reweighted"
    assert {tuple(row["pair"]) for row in table_rows(result)} == {("q", "q:reweighted")}


def test_the_result_is_a_named_tuple() -> None:
    """The engine returns Reweighting(sample_set, result)."""
    sample_set = approximation_set(1.3, (0,), 200, 1, "tuple")
    outcome = reweight(sample_set, ["q"], real_space(2), proposal="approximation")
    assert isinstance(outcome, Reweighting)
    assert outcome[0] is outcome.sample_set
    assert outcome[1] is outcome.result
    np.testing.assert_array_equal(
        required(outcome.sample_set.log_q), required(sample_set.log_q)
    )


# * Errors


def test_names_and_spaces_must_fit_the_set() -> None:
    """Names are K unique strings; the space has D parameters."""
    sample_set = approximation_set(1.3, (0,), 200, 1, "errors-names")
    space = real_space(2)
    with pytest.raises(ValueError, match="names must name each of the 1 posteriors"):
        reweight(sample_set, ["a", "b"], space, proposal="approximation")
    with pytest.raises(ValueError, match="names must hold strings"):
        unchecked_reweight(sample_set, [3], space, proposal="approximation")
    with pytest.raises(
        ValueError, match="names must be a sequence of strings, got str"
    ):
        unchecked_reweight(sample_set, "a", space, proposal="approximation")
    with pytest.raises(ValueError, match="the space has 3 parameters"):
        reweight(sample_set, ["a"], real_space(3), proposal="approximation")
    two = SampleSet(jnp.concatenate([sample_set.samples] * 2, axis=1))
    with pytest.raises(ValueError, match="names must be unique"):
        reweight(two, ["a", "a"], space, prior=BoxPrior(1.0))


def test_target_names_must_be_unique_and_differ_from_the_names() -> None:
    """target_names follow the same rules, and may not repeat a source name."""
    sample_set = approximation_set(1.3, (0,), 200, 1, "errors-targets")
    kwargs: dict[str, Any] = {"proposal": "approximation"}
    with pytest.raises(ValueError, match="target_names must name each"):
        reweight(sample_set, ["a"], real_space(2), target_names=["x", "y"], **kwargs)
    with pytest.raises(ValueError, match=r"target_names must differ from names.*'a'"):
        reweight(sample_set, ["a"], real_space(2), target_names=["a"], **kwargs)
    two = SampleSet(
        jnp.concatenate([sample_set.samples] * 2, axis=1),
        log_like=jnp.concatenate([required(sample_set.log_like)] * 2, axis=1),
        log_prior=jnp.concatenate([required(sample_set.log_prior)] * 2, axis=1),
        log_q=jnp.concatenate([required(sample_set.log_q)] * 2, axis=1),
    )
    with pytest.raises(ValueError, match="target_names must be unique"):
        reweight(two, ["a", "b"], real_space(2), target_names=["t", "t"], **kwargs)


def test_the_proposal_prior_and_likelihood_arguments_are_checked() -> None:
    """Unknown proposals, wrong priors and wrong likelihoods raise."""
    sample_set = approximation_set(1.3, (0,), 200, 1, "errors-arguments")
    space = real_space(2)
    with pytest.raises(ValueError, match="'sampling' is not a valid Proposal"):
        reweight(sample_set, ["a"], space, proposal="sampling")
    with pytest.raises(TypeError, match="prior must be a PriorDistribution"):
        unchecked_reweight(sample_set, ["a"], space, prior=object())
    with pytest.raises(TypeError, match="log_like must be a callable or a sequence"):
        unchecked_reweight(sample_set, ["a"], space, log_like=3)
    with pytest.raises(TypeError, match="log_like must be a callable or a sequence"):
        unchecked_reweight(sample_set, ["a"], space, log_like="abc")
    with pytest.raises(TypeError, match="log_like entry 0 is not callable"):
        unchecked_reweight(sample_set, ["a"], space, log_like=[3])
    with pytest.raises(ValueError, match="holds 2 log-likelihoods for 1 problems"):
        reweight(sample_set, ["a"], space, log_like=[len, len])


def test_one_likelihood_cannot_serve_several_problems() -> None:
    """A single callable with B > 1 raises: independent problems have their own data."""
    sample_set = approximation_set(1.3, (0,), 200, 3, "errors-batch")
    with pytest.raises(ValueError, match="pass one log-likelihood per problem"):
        reweight(sample_set, ["a"], real_space(2), log_like=lambda points: points[:, 0])


def test_the_proposal_needs_the_stored_densities_it_uses() -> None:
    """Missing log-prior, log-likelihood or log_q raise CapabilityError."""
    space = real_space(2)
    plain = SampleSet(jnp.zeros((30, 2)) + jnp.arange(60.0).reshape(30, 2))
    with pytest.raises(ValueError, match="needs a new prior, a new log-likelihood"):
        reweight(plain, ["a"], space)
    with pytest.raises(CapabilityError, match="needs stored log-prior values"):
        reweight(plain, ["a"], space, prior=BoxPrior(1.0))
    with pytest.raises(CapabilityError, match="needs stored log-likelihood values"):
        reweight(plain, ["a"], space, log_like=lambda points: points[:, 0])
    with pytest.raises(CapabilityError, match="needs the stored log_q"):
        reweight(plain, ["a"], space, proposal="approximation")
    with_q = SampleSet(plain.samples[0, 0], log_q=jnp.zeros(30))
    with pytest.raises(CapabilityError, match="needs a target log-likelihood"):
        reweight(with_q, ["a"], space, proposal="approximation")
    with pytest.raises(CapabilityError, match="needs a target log-prior"):
        reweight(
            with_q,
            ["a"],
            space,
            proposal="approximation",
            log_like=lambda points: points[:, 0],
        )


def test_an_approximation_needs_equal_input_weights() -> None:
    """Draws from q are equally weighted, so unequal weights raise."""
    sample_set = approximation_set(1.3, (0,), 100, 1, "errors-weights")
    uneven = SampleSet(
        sample_set.samples,
        log_weights=jnp.linspace(0.0, 1.0, 100).reshape(1, 1, 100),
        log_like=sample_set.log_like,
        log_prior=sample_set.log_prior,
        log_q=sample_set.log_q,
    )
    with pytest.raises(ValueError, match=r"equal input weights.*'a' of problem 0"):
        reweight(uneven, ["a"], real_space(2), proposal="approximation")
    with_zeros = SampleSet(
        sample_set.samples,
        log_weights=jnp.where(jnp.arange(100) < 90, 0.0, -jnp.inf).reshape(1, 1, 100),
        log_like=sample_set.log_like,
        log_prior=sample_set.log_prior,
        log_q=sample_set.log_q,
    )
    reweight(with_zeros, ["a"], real_space(2), proposal="approximation")


def test_new_densities_must_be_defined_where_the_weight_is_positive() -> None:
    """NaN and +inf of a new density raise, naming density, problem and posterior."""
    sample_set = box_set(100)
    with pytest.raises(
        ValueError, match="new log-prior of posterior 'box' in problem 0"
    ):
        reweight(sample_set, ["box"], real_space(2), prior=ConstantPrior(np.nan))
    like = SampleSet(sample_set.samples[0, 0], log_like=jnp.zeros(100))
    with pytest.raises(ValueError, match="new log-likelihood of posterior 'box'"):
        reweight(
            like,
            ["box"],
            real_space(2),
            log_like=lambda points: np.full(len(points), np.inf),
        )


def test_stored_denominators_must_be_finite_where_the_weight_is_positive() -> None:
    """A stored density of −∞ at a positive weight cannot be a denominator."""
    samples = jnp.arange(60.0).reshape(30, 2) / 30
    stored = jnp.zeros(30).at[3].set(-jnp.inf)
    space = real_space(2)
    with pytest.raises(ValueError, match="stored log_prior of posterior 'a'"):
        reweight(
            SampleSet(samples, log_prior=stored), ["a"], space, prior=BoxPrior(9.0)
        )
    with pytest.raises(ValueError, match="stored log_like of posterior 'a'"):
        reweight(
            SampleSet(samples, log_like=stored),
            ["a"],
            space,
            log_like=lambda points: np.zeros(len(points)),
        )
    with pytest.raises(ValueError, match="stored log_q of posterior 'a'"):
        reweight(
            SampleSet(
                samples, log_like=jnp.zeros(30), log_prior=jnp.zeros(30), log_q=stored
            ),
            ["a"],
            space,
            proposal="approximation",
        )
    # * A stored −∞ at a zero-weight sample is no problem.
    weights = jnp.where(jnp.arange(30) == 3, -jnp.inf, 0.0)
    reweight(
        SampleSet(samples, log_prior=stored, log_weights=weights),
        ["a"],
        space,
        prior=BoxPrior(9.0),
    )


def test_a_posterior_without_any_kept_sample_raises() -> None:
    """If the new density is zero everywhere, nothing keeps a weight."""
    samples = box_set(100).samples[0, 0]
    with pytest.raises(ValueError, match="no sample of posterior 'a' in problem 0"):
        reweight(
            SampleSet(samples, log_prior=jnp.zeros(100)),
            ["a"],
            real_space(2),
            prior=ConstantPrior(-np.inf),
        )


def test_a_density_of_the_wrong_shape_raises() -> None:
    """The new density must return one value per sample."""
    sample_set = SampleSet(jnp.ones((30, 2)), log_like=jnp.zeros(30))
    with pytest.raises(ValueError, match="must return shape"):
        reweight(sample_set, ["a"], real_space(2), log_like=lambda points: np.zeros(3))
