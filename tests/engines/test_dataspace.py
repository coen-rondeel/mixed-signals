"""Tests for ``compare_signals``: modes, rows, Fisher chart, cap, nulls, arguments."""

import inspect
import math
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jaxtyping import Array, Float

from mixed_signals import prng, zoo
from mixed_signals.core.result import Result, hash_arrays
from mixed_signals.core.samples import (
    SampleSet,
    kish_effective_sample_size,
    subsample_effective_size,
)
from mixed_signals.core.space import ParamSpace
from mixed_signals.dataspace.templates import whitened_templates
from mixed_signals.engines.dataspace import compare_signals
from mixed_signals.errors import CapabilityError, CostWarning
from mixed_signals.metrics.kernel import energy_distance
from mixed_signals.nulls.permutation import permutation_null, permutation_side_size
from mixed_signals.nulls.plan import null_draws
from mixed_signals.sigma.thresholds import THRESHOLDS
from mixed_signals.transforms.base import Transform
from mixed_signals.zoo.signals import SignalCase

unchecked_compare_signals = inspect.unwrap(compare_signals)
KEY = 2028
DRAWS = 19
FISHER = np.array([[1.0, 0.5, 0.25], [0.5, 4.0, 1.0], [0.25, 1.0, 9.0]])
DIRECTION = [1.0, 0.5, 0.25]


# * Stubs: user code that records its calls, and a transform


class RecordingGenerator:
    """A generator that wraps another and records the parameters of its calls."""

    def __init__(self, inner: Any) -> None:
        """Wrap a generator and start without calls."""
        self.inner = inner
        self.calls: list[np.ndarray] = []

    def generate_template(self, params: Any) -> Any:
        """Record the parameters and return the templates of the wrapped generator."""
        self.calls.append(np.array(params))
        return self.inner.generate_template(params)

    def passed(self) -> np.ndarray:
        """Return every parameter vector that the generator received, stacked."""
        stacked: np.ndarray = np.concatenate(self.calls)
        return stacked


class FisherStub(RecordingGenerator):
    """A generator that also gives a constant Fisher matrix and records the noise."""

    def __init__(
        self, inner: Any, fisher: np.ndarray, shape_error: bool = False
    ) -> None:
        """Store the matrix; ``shape_error`` makes the shape wrong."""
        super().__init__(inner)
        self.fisher = fisher
        self.shape_error = shape_error
        self.noises: list[Any] = []
        self.fisher_params: list[np.ndarray] = []

    def information_matrix(self, params: Any, *, noise: Any) -> Any:
        """Record the call and return the matrix once per row."""
        self.noises.append(noise)
        self.fisher_params.append(np.array(params))
        count = len(params)
        if self.shape_error:
            return np.zeros((count, self.fisher.shape[0]))
        return np.broadcast_to(self.fisher, (count, *self.fisher.shape))


class PlainNoise:
    """A noise model without a description, which divides by one."""

    def whiten(self, signals: Any) -> Any:
        """Return the signals as vectors shaped (n, M)."""
        values = np.asarray(signals)
        return values.reshape(values.shape[0], -1)


class NamedNoise(PlainNoise):
    """A noise model with a description."""

    def __init__(self, description: str) -> None:
        """Store the description."""
        self.description = description


class Scaling(Transform):
    """The map y = f x + b with a diagonal matrix f, for the tests."""

    factors: Float[Array, " param"]
    offset: Float[Array, " param"]

    def forward(
        self, points: Float[Array, "*leading param"]
    ) -> Float[Array, "*leading param"]:
        """Return f x + b."""
        return points * self.factors + self.offset

    def inverse(
        self, points: Float[Array, "*leading param"]
    ) -> Float[Array, "*leading param"]:
        """Return (y − b)/f."""
        return (points - self.offset) / self.factors

    def log_abs_det_jacobian(
        self, points: Float[Array, "*leading param"]
    ) -> Float[Array, " *leading"]:
        """Return Σ ln abs f at every point."""
        return jnp.broadcast_to(
            jnp.sum(jnp.log(jnp.abs(self.factors))), points.shape[:-1]
        )

    @property
    def description(self) -> str:
        """Name of the test transform."""
        return "scale"

    def map_space(self, space: ParamSpace) -> ParamSpace:
        """Return the space unchanged: the map keeps every real line."""
        return space


# * Helpers


def linear_case(data_size: int = 32) -> SignalCase:
    """Return the linear signal whose Fisher matrix is :data:`FISHER`."""
    return zoo.linear_signal(FISHER, data_size=data_size)


def shifted_set(
    case: SignalCase,
    shift: float,
    count: int,
    batch: int = 1,
    seed: int = 0,
    posteriors: int = 2,
) -> SampleSet:
    """Return samples of the Gaussians of the Fisher matrix, shifted along DIRECTION."""
    gaussians = case.fisher_gaussians(shift, direction=DIRECTION)
    samples = gaussians.sample(jax.random.key(seed), count, batch_size=batch)
    if posteriors == 3:
        third = gaussians.sample(jax.random.key(seed + 1), count, batch_size=batch)
        return SampleSet(
            jnp.concatenate([samples.samples, third.samples[:, 1:] + 0.1], axis=1)
        )
    return samples


def run(
    case: SignalCase,
    samples: SampleSet,
    names: tuple[str, ...] = ("reference", "shifted"),
    **options: Any,
) -> Result:
    """Run ``compare_signals`` with the generator and noise of a case and few draws."""
    batch = samples.shape.batch
    function = (
        unchecked_compare_signals
        if options.pop("unchecked", False)
        else compare_signals
    )
    settings: dict[str, Any] = {
        "generator": case.generator,
        "noise": case.noise,
        "key": KEY,
        "draws": DRAWS,
        "reference": names[0],
    }
    if options.pop("with_fisher", False):
        settings["fisher"] = np.broadcast_to(case.fisher, (batch, *case.fisher.shape))
        settings["centre"] = np.broadcast_to(case.centre, (batch, *case.centre.shape))
    settings.update(options)
    return function(samples, list(names), case.space, **settings)


def rows_of(result: Result, **selection: Any) -> list[dict[str, Any]]:
    """Return the rows of a result that match the given columns, in order."""
    return [
        row
        for row in result.table.to_dicts()
        if all(row[column] == value for column, value in selection.items())
    ]


# * The done criterion: a linear signal


def test_a_linear_signal_gives_equal_data_and_chart_rows_that_recover_the_shift() -> (
    None
):
    """The data row equals the chart row, and sigma_eff of a shift of 1 reads 1."""
    case = linear_case()
    problems = 6
    samples = shifted_set(case, 1.0, 400, batch=problems)
    result = run(case, samples, with_fisher=True)
    assert len(result) == 2 * problems
    effects = []
    for problem in range(problems):
        data, chart = rows_of(result, problem=problem)
        assert data["space"] == "data(white)"
        assert chart["space"] == "chart(fisher)"
        assert data["value"] == pytest.approx(chart["value"], rel=1e-10)
        np.testing.assert_allclose(data["null_draws"], chart["null_draws"], rtol=1e-10)
        assert data["sigma_eff"] == pytest.approx(chart["sigma_eff"], rel=1e-9)
        effects.append(data["sigma_eff"])
    mean = float(np.mean(effects))
    error = float(np.std(effects, ddof=1)) / math.sqrt(problems)
    # * The mean of a σ_eff that rests on a V-statistic keeps a bias of order 1/N.
    assert abs(mean - 1.0) <= 4 * error + 0.05


def test_the_data_row_is_the_energy_distance_of_the_whitened_templates() -> None:
    """The value and the null draws equal the sample kernel on the templates."""
    case = linear_case(data_size=16)
    samples = shifted_set(case, 0.8, 120, seed=3)
    result = run(case, samples)
    (row,) = rows_of(result)
    reference, shifted = samples.samples[0]
    first = whitened_templates(case.generator, case.noise, reference)
    second = whitened_templates(case.generator, case.noise, shifted)
    data_space = ParamSpace.model_validate(
        {"parameters": [{"name": f"m{index}"} for index in range(16)]}
    )
    key = prng.derive_key(prng.root_key(KEY), "x")
    expected = energy_distance(first, None, second, None, data_space, key=key, cap=120)
    assert row["value"] == pytest.approx(float(expected), rel=1e-12)
    plan = permutation_null(
        reference,
        None,
        shifted,
        None,
        key=prng.derive_key(
            prng.root_key(KEY), "compare_signals", 0, "reference", "shifted", "null"
        ),
        first_n_eff=120.0,
        second_n_eff=120.0,
        draws=DRAWS,
    )
    pool = jnp.concatenate([first, second])

    def statistic(
        first_side: Any, first_weights: Any, second_side: Any, second_weights: Any
    ) -> Any:
        return energy_distance(
            first_side,
            first_weights,
            second_side,
            second_weights,
            data_space,
            key=key,
            cap=240,
        )

    expected_draws = null_draws(statistic, plan._replace(samples=pool))
    np.testing.assert_allclose(row["null_draws"], expected_draws, rtol=1e-10)


def test_the_columns_of_a_row_state_the_metric_the_settings_and_the_provenance() -> (
    None
):
    """Metric, estimator, parameters, threshold, settings, recipe and hashes."""
    case = linear_case(data_size=24)
    samples = shifted_set(case, 0.5, 80, seed=5)
    result = run(case, samples, with_fisher=True, config_hash=hash_arrays(np.arange(3)))
    data, chart = rows_of(result)
    for row in (data, chart):
        assert row["metric"] == "energy"
        assert row["estimator"] == "empirical"
        assert row["parameters"] == ["x0", "x1", "x2"]
        assert row["threshold"] == THRESHOLDS["energy"]
        assert row["pair"] == ["reference", "shifted"]
        assert row["uncertainty_method"] == "null_std"
        assert row["uncertainty"] == pytest.approx(row["null_std"], rel=1e-12)
        assert len(row["null_draws"]) == DRAWS
        assert row["null_recipe"]["construction"] == "permutation"
        assert row["null_recipe"]["draws"] == DRAWS
        assert row["null_recipe"]["tail"] == "upper"
        assert row["r_eff"] is None
        assert row["provenance"]["seed"] == KEY
        assert row["provenance"]["input_hashes"] == [
            hash_arrays(samples.samples[0, 0], samples.log_weights[0, 0]),
            hash_arrays(samples.samples[0, 1], samples.log_weights[0, 1]),
        ]
        assert row["provenance"]["config_hash"] == hash_arrays(np.arange(3))
        assert not row["warnings"]
    assert data["estimator_settings"] == '{"data_size":24}'
    assert chart["estimator_settings"] == '{"fisher":"array"}'


# * Modes and nulls


def test_mode_a_pairs_the_reference_and_mode_b_every_pair() -> None:
    """The pairs and their order, with the data row before the chart row."""
    case = linear_case(data_size=16)
    samples = shifted_set(case, 0.5, 60, posteriors=3)
    names = ("a", "b", "c")
    mode_a = run(case, samples, names, reference="b", with_fisher=True)
    pairs_a = [tuple(row["pair"]) for row in rows_of(mode_a)]
    assert pairs_a == [("b", "a")] * 2 + [("b", "c")] * 2
    assert [row["space"] for row in rows_of(mode_a)] == [
        "data(white)",
        "chart(fisher)",
    ] * 2
    mode_b = run(case, samples, names, reference=None)
    pairs_b = [tuple(row["pair"]) for row in rows_of(mode_b)]
    assert pairs_b == [("a", "b"), ("a", "c"), ("b", "c")]


def test_the_split_half_null_cuts_the_reference_and_needs_mode_a() -> None:
    """The recipe names the construction; mode B raises."""
    case = linear_case(data_size=16)
    samples = shifted_set(case, 0.5, 100, seed=2)
    result = run(case, samples, null="split_half", with_fisher=True)
    for row in rows_of(result):
        assert row["null_recipe"]["construction"] == "split_half"
        assert row["null_recipe"]["sample_size"] <= 50
        assert len(row["null_draws"]) == DRAWS
    with pytest.raises(ValueError, match="the split_half null needs reference"):
        run(case, samples, null="split_half", reference=None)


def test_the_subsampling_bootstrap_reports_the_half_sample_error() -> None:
    """The uncertainty is the half-sample error; the null spread keeps its column."""
    case = linear_case(data_size=16)
    samples = shifted_set(case, 0.5, 100, seed=6)
    plain = run(case, samples, with_fisher=True)
    boot = run(
        case,
        samples,
        with_fisher=True,
        uncertainty="subsampling_bootstrap",
        half_samples=10,
    )
    for row, other in zip(rows_of(boot), rows_of(plain), strict=True):
        assert row["uncertainty_method"] == "subsampling_bootstrap"
        assert math.isfinite(row["uncertainty"])
        assert row["uncertainty"] != pytest.approx(other["uncertainty"], rel=1e-6)
        assert row["null_std"] == pytest.approx(other["null_std"], rel=1e-12)
        # * More rows change the order of the sums of the products, in the last digits.
        assert row["value"] == pytest.approx(other["value"], rel=1e-12)
        np.testing.assert_allclose(row["null_draws"], other["null_draws"], rtol=1e-12)


# * Generators and noise models per problem, capabilities


def test_one_generator_serves_every_problem_and_a_sequence_gives_one_per_problem() -> (
    None
):
    """A sequence of B generators and noise models is used problem by problem."""
    case = linear_case(data_size=16)
    samples = shifted_set(case, 0.5, 60, batch=2, seed=7)
    shared = RecordingGenerator(case.generator)
    run(case, samples, generator=shared)
    # * Per problem, the dry run probes once and each of two posteriors calls once.
    assert len(shared.calls) == 2 * 3
    first = RecordingGenerator(case.generator)
    second = RecordingGenerator(case.generator)
    result = run(
        case,
        samples,
        generator=[first, second],
        noise=[NamedNoise("night"), NamedNoise("day")],
        unchecked=True,
    )
    assert len(first.calls) == len(second.calls) == 3
    np.testing.assert_allclose(
        np.concatenate(first.calls[1:]),
        np.concatenate([samples.samples[0, 0], samples.samples[0, 1]]),
        rtol=1e-14,
    )
    np.testing.assert_allclose(
        np.concatenate(second.calls[1:]),
        np.concatenate([samples.samples[1, 0], samples.samples[1, 1]]),
        rtol=1e-14,
    )
    assert [row["space"] for row in rows_of(result)] == ["data(night)", "data(day)"]


def test_a_sequence_of_the_wrong_length_raises() -> None:
    """The number of generators or noise models must be B."""
    case = linear_case(data_size=16)
    samples = shifted_set(case, 0.5, 40, batch=2)
    with pytest.raises(ValueError, match="generator holds 3 objects for 2 problems"):
        run(case, samples, generator=[case.generator] * 3)
    with pytest.raises(ValueError, match="noise holds 1 objects for 2 problems"):
        run(case, samples, noise=[case.noise])


def test_missing_capabilities_raise_before_any_call() -> None:
    """A generator without templates, noise without whiten, centre without Fisher.

    The test session checks the protocol annotations first, so the calls take
    the unwrapped function, as runs without runtime type checks do.
    """
    case = linear_case(data_size=16)
    samples = shifted_set(case, 0.5, 40)
    with pytest.raises(
        CapabilityError,
        match=r"compare_signals needs a generator with generate_template\(params\)"
        "; PlainNoise lacks it",
    ):
        run(case, samples, generator=PlainNoise(), unchecked=True)
    with pytest.raises(
        CapabilityError,
        match=r"compare_signals needs a noise model with whiten\(signals\)"
        "; LinearSignal lacks it",
    ):
        run(case, samples, noise=case.generator, unchecked=True)
    with pytest.raises(
        CapabilityError,
        match=r"needs a generator with information_matrix\(params, noise=noise\)"
        "; RecordingGenerator lacks it",
    ):
        run(
            case,
            samples,
            generator=RecordingGenerator(case.generator),
            centre=case.centre,
        )


# * The Fisher matrix: arrays, the generator, invalid values


def test_a_centre_alone_takes_the_fisher_matrix_from_the_generator() -> None:
    """information_matrix gets the centre and the noise; its rows equal the array's."""
    case = linear_case(data_size=16)
    samples = shifted_set(case, 0.7, 80, seed=8)
    stub = FisherStub(case.generator, FISHER)
    from_generator = run(case, samples, generator=stub, centre=case.centre)
    from_array = run(case, samples, with_fisher=True)
    assert len(stub.noises) == 1
    assert stub.noises[0] is case.noise
    np.testing.assert_allclose(stub.fisher_params[0], [np.zeros(3)])
    chart = rows_of(from_generator, space="chart(fisher)")[0]
    assert chart["estimator_settings"] == '{"fisher":"information_matrix"}'
    reference = rows_of(from_array, space="chart(fisher)")[0]
    assert chart["value"] == pytest.approx(reference["value"], rel=1e-12)
    assert len(rows_of(from_generator)) == 2


def test_a_wrong_fisher_shape_from_the_generator_or_the_arrays_raises() -> None:
    """The matrix must be (D, D) per centre; wrong shapes raise, not warn."""
    case = linear_case(data_size=16)
    samples = shifted_set(case, 0.5, 40)
    stub = FisherStub(case.generator, FISHER, shape_error=True)
    with pytest.raises(
        ValueError, match=r"returned shape \(1, 3\) .* shape \(1, 3, 3\)"
    ):
        run(case, samples, generator=stub, centre=case.centre)
    with pytest.raises(ValueError, match=r"fisher must be shaped \(1, 3, 3\)"):
        run(case, samples, fisher=np.eye(4), centre=case.centre)
    with pytest.raises(ValueError, match=r"centre must be shaped \(1, 3\)"):
        run(case, samples, fisher=FISHER, centre=np.zeros(4))
    with pytest.raises(ValueError, match=r"centre must be shaped \(1, 3\)"):
        run(
            case,
            samples,
            generator=FisherStub(case.generator, FISHER),
            centre=np.zeros(2),
        )
    with pytest.raises(ValueError, match="fisher needs centre="):
        run(case, samples, fisher=FISHER)


def test_an_invalid_fisher_matrix_gives_nan_fisher_rows_and_the_rest_goes_on() -> None:
    """A matrix that is not positive definite warns; other problems keep their rows."""
    case = linear_case(data_size=16)
    samples = shifted_set(case, 0.5, 60, batch=2, seed=9)
    broken = np.stack([np.diag([1.0, -1.0, 1.0]), FISHER])
    result = run(case, samples, fisher=broken, centre=np.zeros((2, 3)))
    data0, chart0, data1, chart1 = rows_of(result)
    assert math.isfinite(data0["value"])
    assert not data0["warnings"]
    assert math.isnan(chart0["value"])
    assert math.isnan(chart0["p_value"])
    assert math.isnan(chart0["sigma_sig"])
    assert math.isnan(chart0["sigma_eff"])
    assert chart0["flag"] is False
    assert chart0["threshold"] == THRESHOLDS["energy"]
    (record,) = chart0["warnings"]
    assert record["category"] == "ValidityWarning"
    assert record["message"].startswith("the Fisher matrix of problem 0 is not valid:")
    assert math.isfinite(data1["value"])
    assert math.isfinite(chart1["value"])
    assert chart1["space"] == "chart(fisher)"


def test_an_ill_conditioned_fisher_matrix_warns_in_its_rows_only() -> None:
    """The condition-number warning goes into the Fisher rows, which stay finite."""
    case = linear_case(data_size=16)
    samples = shifted_set(case, 0.5, 60, seed=10)
    correlation = 1 - 1e-9
    fisher = np.array(
        [[1.0, correlation, 0.0], [correlation, 1.0, 0.0], [0.0, 0.0, 1.0]]
    ) * np.outer([1e10, 1.0, 1.0], [1e10, 1.0, 1.0])
    result = run(case, samples, fisher=fisher, centre=np.zeros(3))
    data, chart = rows_of(result)
    assert not data["warnings"]
    assert math.isfinite(chart["value"])
    assert [record["category"] for record in chart["warnings"]] == ["ValidityWarning"]
    assert "condition number" in chart["warnings"][0]["message"]


def test_a_wide_circle_warns_in_the_fisher_row_only() -> None:
    """Samples spread over more than the cut of the chart warn in the chart row."""
    case = zoo.sinusoid_signal(10.0, data_size=100)
    space = ParamSpace.model_validate(
        {
            "parameters": [
                {"name": "amplitude"},
                {"name": "frequency"},
                {"name": "phase", "topology": "circle", "bounds": [0.0, 2 * math.pi]},
            ]
        }
    )
    generator = np.random.default_rng(4)
    spread = np.array([0.05, 0.05, 1.5])
    reference = np.asarray(case.centre) + generator.normal(size=(60, 3)) * spread
    other = np.asarray(case.centre) + generator.normal(size=(60, 3)) * spread
    samples = SampleSet(np.stack([reference, other])[None])
    result = compare_signals(
        samples,
        ["a", "b"],
        space,
        generator=case.generator,
        noise=case.noise,
        key=KEY,
        draws=DRAWS,
        fisher=case.fisher,
        centre=case.centre,
    )
    data, chart = rows_of(result)
    assert not data["warnings"]
    (record,) = chart["warnings"]
    assert record["category"] == "ValidityWarning"
    assert "'phase'" in record["message"]
    assert math.isfinite(chart["value"])


# * The cap, the effective size and zero weights


def test_the_cap_subsamples_once_and_every_row_records_the_warning() -> None:
    """Each posterior sends cap samples to the generator; both rows warn."""
    case = linear_case(data_size=16)
    samples = shifted_set(case, 0.5, 150, seed=11)
    recorder = RecordingGenerator(case.generator)
    result = run(case, samples, generator=recorder, cap=60, with_fisher=True)
    # * The dry run probes one sample, then each posterior sends its capped set.
    assert [len(call) for call in recorder.calls] == [1, 60, 60]
    for row in rows_of(result):
        # * Each of the two posteriors raises its own warning, and both enter the row.
        assert [record["category"] for record in row["warnings"]] == [
            "SubsamplingWarning"
        ] * 2
        for record in row["warnings"]:
            assert "150 samples per posterior exceed the cap of 60" in record["message"]
        assert row["null_recipe"]["sample_size"] <= 60
    # * The capped samples are a subset of the posterior, without repeats.
    sent = np.concatenate(recorder.calls[1:2])
    original = {tuple(row) for row in np.asarray(samples.samples[0, 0]).tolist()}
    assert {tuple(row) for row in sent.tolist()} <= original
    assert len({tuple(row) for row in sent.tolist()}) == 60


def test_a_capped_set_takes_the_design_effect_effective_size_in_its_null() -> None:
    """The sides of the null follow n_m of a correlated set; without a cap, n itself."""
    case = linear_case(data_size=16)
    base = shifted_set(case, 0.5, 300, seed=12)
    correlated = SampleSet(base.samples, n_eff=jnp.asarray([[150.0, 300.0]]))
    capped = run(case, correlated, cap=100)
    effective = subsample_effective_size(150.0, 300.0, 100.0)
    expected = permutation_side_size(
        jnp.zeros(100), jnp.zeros(100), first_n_eff=effective, second_n_eff=100.0
    )
    (row,) = rows_of(capped)
    assert row["null_recipe"]["sample_size"] == expected
    uncapped = run(case, correlated, cap=1_000)
    plain = permutation_side_size(
        jnp.zeros(300), jnp.zeros(300), first_n_eff=150.0, second_n_eff=300.0
    )
    (row,) = rows_of(uncapped)
    assert row["null_recipe"]["sample_size"] == plain
    assert plain != expected


def test_a_capped_set_without_its_heavy_sample_clips_the_kish_size() -> None:
    """The subset may have a larger Kish size than the set, which the engine clips."""
    case = linear_case(data_size=16)
    base = shifted_set(case, 0.5, 300, seed=21)
    log_weights = np.zeros((1, 2, 300))
    log_weights[0, 0, 0] = math.log(100.0)
    kish_size = float(kish_effective_sample_size(jnp.asarray(log_weights[0, 0])))
    assert kish_size == pytest.approx(15.46, abs=0.01)
    weighted = SampleSet(
        base.samples, log_weights=log_weights, n_eff=jnp.asarray([[8.0, 300.0]])
    )
    recorder = RecordingGenerator(case.generator)
    # * The key of this seed draws a subset without the heavy sample.
    result = run(case, weighted, generator=recorder, cap=100, key=1)
    sent = recorder.calls[1]
    assert len(sent) == 100
    assert not np.any(np.all(sent == np.asarray(base.samples[0, 0, 0]), axis=1))
    # * With the whole set as the subset, the design effect gives back n_eff.
    clipped = subsample_effective_size(8.0, kish_size, kish_size)
    assert clipped == pytest.approx(8.0, rel=1e-12)
    expected = permutation_side_size(
        jnp.zeros(100), jnp.zeros(100), first_n_eff=clipped, second_n_eff=100.0
    )
    (row,) = rows_of(result)
    assert row["null_recipe"]["sample_size"] == expected


def test_samples_of_zero_weight_never_reach_the_generator() -> None:
    """Only the positive-weight samples are sent, and they are the ones that count."""
    case = linear_case(data_size=16)
    base = shifted_set(case, 0.5, 100, seed=13)
    log_weights = np.zeros((1, 2, 100))
    log_weights[0, 0, 10:40] = -np.inf
    weighted = SampleSet(base.samples, log_weights=log_weights)
    recorder = RecordingGenerator(case.generator)
    run(case, weighted, generator=recorder)
    # * The probe takes the first positive sample, then the two posteriors follow.
    assert [len(call) for call in recorder.calls] == [1, 70, 100]
    reference = np.asarray(base.samples[0, 0])
    expected = np.concatenate([reference[:10], reference[40:]])
    np.testing.assert_allclose(recorder.calls[1], expected, rtol=1e-14)


# * Transform, noise description, backend and chunks


def test_a_transform_maps_the_samples_the_fisher_matrix_and_the_centre() -> None:
    """Rows on the input equal rows on the mapped set; the space names the map."""
    case = linear_case(data_size=16)
    mapped_samples = shifted_set(case, 0.8, 80, seed=14)
    factors = jnp.array([2.0, 0.5, 4.0])
    transform = Scaling(factors, jnp.array([0.3, -1.0, 0.0]))
    original = SampleSet(transform.inverse(mapped_samples.samples))
    scale = np.diag(np.asarray(factors))
    input_fisher = scale @ FISHER @ scale
    input_centre = np.asarray(transform.inverse(case.centre))
    direct = run(case, mapped_samples, with_fisher=True)
    through = run(
        case,
        original,
        transform=transform,
        fisher=input_fisher,
        centre=input_centre,
    )
    for row, other in zip(rows_of(direct), rows_of(through), strict=True):
        assert row["value"] == pytest.approx(other["value"], rel=1e-9)
        np.testing.assert_allclose(row["null_draws"], other["null_draws"], rtol=1e-9)
        assert other["space"] == f"scale | {row['space']}"
    assert [row["space"] for row in rows_of(through)] == [
        "scale | data(white)",
        "scale | chart(fisher)",
    ]
    hashes = rows_of(through)[0]["provenance"]["input_hashes"]
    assert hashes == [
        hash_arrays(original.samples[0, 0], original.log_weights[0, 0]),
        hash_arrays(original.samples[0, 1], original.log_weights[0, 1]),
    ]


def test_a_noise_model_without_a_description_gives_the_plain_data_text() -> None:
    """The space column reads ``data`` unless the noise model names itself."""
    case = linear_case(data_size=16)
    samples = shifted_set(case, 0.5, 40, seed=15)
    plain = run(case, samples, noise=PlainNoise(), unchecked=True)
    named = run(case, samples, noise=NamedNoise("detector"), unchecked=True)
    empty = run(case, samples, noise=NamedNoise(""), unchecked=True)
    assert rows_of(plain)[0]["space"] == "data"
    assert rows_of(named)[0]["space"] == "data(detector)"
    assert rows_of(empty)[0]["space"] == "data"


def test_the_jax_backend_and_chunks_give_the_same_rows() -> None:
    """Zoo generators run on JAX arrays, and chunks of samples change no value."""
    case = linear_case(data_size=16)
    samples = shifted_set(case, 0.6, 70, seed=16)
    reference = run(case, samples, with_fisher=True)
    on_jax = run(case, samples, with_fisher=True, backend="jax")
    chunked = run(case, samples, with_fisher=True, chunk_size=20)
    for other in (on_jax, chunked):
        np.testing.assert_allclose(
            other.table["value"].to_numpy(),
            reference.table["value"].to_numpy(),
            rtol=1e-12,
        )
        np.testing.assert_allclose(
            np.asarray(other.table["null_draws"].to_list()),
            np.asarray(reference.table["null_draws"].to_list()),
            rtol=1e-12,
        )


# * Templates that break the contract


def test_posteriors_of_a_problem_must_share_one_data_grid() -> None:
    """A generator that changes the number of data entries raises, naming both."""

    class SignGrid:
        """A generator whose number of data entries depends on the sign of x0."""

        def generate_template(self, params: Any) -> Any:
            """Return 4 entries for positive x0 and 5 otherwise."""
            size = 4 if float(np.asarray(params)[0, 0]) > 0 else 5
            return np.ones((len(params), size))

    reference = np.random.default_rng(0).normal(5.0, 0.1, size=(30, 3))
    other = np.random.default_rng(1).normal(-5.0, 0.1, size=(30, 3))
    samples = SampleSet(np.stack([reference, other])[None])
    space = linear_case().space
    with pytest.raises(
        ValueError,
        match="must share one data grid, but 'a' gives M = 4 data entries and 'b'"
        " gives M = 5",
    ):
        compare_signals(
            samples,
            ["a", "b"],
            space,
            generator=SignGrid(),
            noise=PlainNoise(),
            key=KEY,
            draws=DRAWS,
        )


def test_templates_that_are_not_finite_raise_with_the_number_of_rows() -> None:
    """A NaN in a template is an error, since no distance would mean anything."""

    class Poisoned:
        """A generator whose templates are NaN for large first parameters."""

        def generate_template(self, params: Any) -> Any:
            """Return ones, with NaN where the first parameter exceeds 1.5."""
            values = np.ones((len(params), 4))
            values[np.asarray(params)[:, 0] > 1.5] = np.nan
            return values

    case = linear_case()
    samples = shifted_set(case, 0.5, 100, seed=17)
    shifted = SampleSet(samples.samples + jnp.array([1.0, 0.0, 0.0]))
    with pytest.raises(
        ValueError, match=r"templates are not finite in \d+ of 100 rows"
    ):
        compare_signals(
            shifted,
            ["a", "b"],
            case.space,
            generator=Poisoned(),
            noise=PlainNoise(),
            key=KEY,
            draws=DRAWS,
        )


# * Keys, provenance and the dry run


def test_the_same_key_repeats_the_table_and_a_new_posterior_leaves_old_pairs() -> None:
    """Rows are a function of the key and the path of their pair."""
    case = linear_case(data_size=16)
    three = shifted_set(case, 0.5, 60, seed=18, posteriors=3)
    two = SampleSet(three.samples[:, :2])
    first = run(case, two, ("a", "b"), reference=None)
    again = run(case, two, ("a", "b"), reference=None)
    other_key = run(case, two, ("a", "b"), reference=None, key=KEY + 1)
    assert first.table.equals(again.table)
    assert first.table["value"].to_list() == other_key.table["value"].to_list()
    assert (
        first.table["null_draws"].to_list() != other_key.table["null_draws"].to_list()
    )
    larger = run(case, three, ("a", "b", "c"), reference=None)
    (old,) = rows_of(first)
    (same,) = rows_of(larger, pair=["a", "b"])
    assert same["value"] == old["value"]
    assert same["null_draws"] == old["null_draws"]
    assert len(larger) == 3


def test_a_small_budget_warns_once_with_the_data_space_estimate() -> None:
    """The dry run runs first, and its warning goes to the caller, not into rows."""
    case = linear_case(data_size=16)
    samples = shifted_set(case, 0.5, 40, seed=19)
    with pytest.warns(
        CostWarning, match="the compare_signals run is estimated at"
    ) as seen:
        result = run(case, samples, with_fisher=True, budget=1e-9)
    assert len(seen) == 1
    assert "energy (empirical)" in str(seen[0].message)
    # * User code has no estimate, so the warning does not name it.
    assert "generate_template" not in str(seen[0].message)
    assert all(not row["warnings"] for row in rows_of(result))


# * Argument checks


@pytest.mark.parametrize(
    ("options", "error", "message"),
    [
        ({"draws": 0}, ValueError, "draws must be at least 1"),
        ({"half_samples": 0}, ValueError, "half_samples must be at least 1"),
        ({"cap": 1}, ValueError, "cap must be at least 2"),
        ({"draws": 2.5}, TypeError, "draws must be an integer"),
        ({"uncertainty": "delta_method"}, ValueError, "uncertainty must be 'null_std'"),
        ({"uncertainty": 3}, TypeError, "uncertainty must be a string"),
        ({"null": "gaussian_surrogate"}, ValueError, "null must be 'permutation'"),
        ({"reference": "nobody"}, ValueError, "reference must be one of"),
        ({"thresholds": {}}, KeyError, "thresholds lack the metrics energy"),
        ({"thresholds": 3}, TypeError, "thresholds must be a mapping"),
        ({"level": math.nan}, ValueError, "level must be finite"),
        ({"budget": 0.0}, ValueError, "budget must be positive"),
        ({"key": 1.5}, TypeError, "key must be a JAX PRNG key or an integer seed"),
        ({"chunk_size": 0}, ValueError, "chunk_size must be at least 1"),
    ],
)
def test_the_arguments_are_checked(
    options: dict[str, Any], error: type[Exception], message: str
) -> None:
    """Every invalid argument raises the error of ``compare``, before any template.

    The test session checks the annotations first, so the calls take the
    unwrapped function, as runs without runtime type checks do.
    """
    case = linear_case(data_size=16)
    samples = shifted_set(case, 0.5, 40)
    with pytest.raises(error, match=message):
        run(case, samples, unchecked=True, **options)


def test_the_sets_and_names_are_checked_as_in_compare() -> None:
    """One posterior, repeated names and a wrong space raise."""
    case = linear_case(data_size=16)
    samples = shifted_set(case, 0.5, 40)
    with pytest.raises(ValueError, match="at least two posteriors, got 1"):
        run(case, samples.select(posteriors=[0]), ("a",), reference=None)
    with pytest.raises(ValueError, match="names must be unique"):
        run(case, samples, ("a", "a"), reference=None)
    with pytest.raises(ValueError, match="the space has 2 parameters"):
        compare_signals(
            samples,
            ["a", "b"],
            ParamSpace.model_validate({"parameters": [{"name": "u"}, {"name": "v"}]}),
            generator=case.generator,
            noise=case.noise,
            key=KEY,
        )


class SingleNoise(PlainNoise):
    """A noise model that returns float32 vectors."""

    def whiten(self, signals: Any) -> Any:
        """Return the signals as float32 vectors."""
        return np.asarray(super().whiten(signals), dtype=np.float32)


def test_a_float32_whitener_warns_in_the_rows_and_raises_nothing() -> None:
    """The precision warnings of the templates go into both rows of a pair.

    The dry run probes the whitener too, but drops its warnings, and the
    engine records those of the real calls, one for each posterior.
    """
    case = linear_case(data_size=16)
    samples = shifted_set(case, 0.5, 60, seed=20)
    result = run(case, samples, noise=SingleNoise(), with_fisher=True, unchecked=True)
    data, chart = rows_of(result)
    for row in (data, chart):
        assert [record["category"] for record in row["warnings"]] == [
            "PrecisionWarning"
        ] * 2
        assert "float32" in row["warnings"][0]["message"]
