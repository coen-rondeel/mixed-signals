"""Tests for ``compare``: modes, whitenings, recipes, rows, keys and argument checks."""

import inspect
import json
import math
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import polars as pl
import pytest
from scipy import stats

from mixed_signals import prng, zoo
from mixed_signals.core.result import RESULT_SCHEMA, Result, hash_arrays
from mixed_signals.core.samples import SampleSet, rejection_subset
from mixed_signals.core.space import ParamSpace
from mixed_signals.engines._arguments import select_recipes
from mixed_signals.engines.compare import (
    DEFAULT_METRICS,
    RECIPES,
    Recipe,
    Scope,
    compare,
)
from mixed_signals.errors import SubsamplingWarning
from mixed_signals.metrics.divergence import (
    JS_NEIGHBOURS,
    KL_NEIGHBOURS,
    knn_js_divergence,
)
from mixed_signals.metrics.kernel import median_bandwidth
from mixed_signals.metrics.marginal import marginal_distances, shared_bandwidth_factor
from mixed_signals.metrics.moments import mean_shift
from mixed_signals.nulls.permutation import permutation_null, permutation_side_size
from mixed_signals.nulls.plan import null_draws
from mixed_signals.sigma.calibration import knn_kl_curve
from mixed_signals.sigma.maps import sigma_eff
from mixed_signals.sigma.thresholds import THRESHOLDS
from mixed_signals.transforms.whiten import Whitening

unchecked_compare = inspect.unwrap(compare)
KEY = 2027
SMALL_DRAWS = 20
# * Whole-table tests use 60 draws, since 40 resolve σ_sig only up to 1.97.
TABLE_DRAWS = 60
TABLE_SIZE = 400
TABLE_PROBLEMS = 4


# * Helpers


def pair_rows(
    result: Result,
    metric: str | None = None,
    *,
    estimator: str | None = None,
    problem: int | None = None,
    pair: tuple[str, str] | None = None,
    parameters: tuple[str, ...] | None = None,
) -> list[dict[str, Any]]:
    """Return the rows of a result that match the given columns, in order."""
    rows = result.table.to_dicts()
    return [
        row
        for row in rows
        if (metric is None or row["metric"] == metric)
        and (estimator is None or row["estimator"] == estimator)
        and (problem is None or row["problem"] == problem)
        and (pair is None or tuple(row["pair"]) == pair)
        and (parameters is None or tuple(row["parameters"]) == parameters)
    ]


def one_row(result: Result, metric: str, **selection: Any) -> dict[str, Any]:
    """Return the only row of a metric, with other columns as given."""
    (row,) = pair_rows(result, metric, **selection)
    return row


def shift_set(
    shift: float = 1.0, count: int = TABLE_SIZE, batch: int = 1, seed: int = 0
) -> tuple[zoo.ZooCase, SampleSet]:
    """Return the shifted Gaussians of the zoo and a set of their samples."""
    case = zoo.shifted_gaussians(shift, param_count=2)
    return case, case.sample(jax.random.key(seed), count, batch_size=batch)


def three_posteriors(count: int = 80) -> tuple[ParamSpace, SampleSet]:
    """Return three Gaussians, the first two shifted from each other, as one set."""
    case = zoo.shifted_gaussians(0.5, param_count=2)
    base = case.sample(jax.random.key(1), count).samples
    third = case.sample(jax.random.key(2), count).samples[:, :1] + 0.2
    return case.space, SampleSet(jnp.concatenate([base, third], axis=1))


def whitened_zscore(
    samples: jax.Array, reference: jax.Array, space: ParamSpace
) -> np.ndarray:
    """Return samples z-scored by the unbiased width of a reference posterior."""
    scale = np.asarray(reference).std(axis=0, ddof=1)
    centre = np.asarray(reference).mean(axis=0)
    return (np.asarray(samples) - centre) / scale


# * Inputs and the order of the rows


def test_one_set_and_a_list_of_sets_with_different_sizes_run() -> None:
    """A set of two posteriors, and two sets of one with their own N, both compare."""
    case, samples = shift_set(count=120)
    single = compare(
        samples,
        ["ref", "other"],
        case.space,
        key=KEY,
        draws=SMALL_DRAWS,
        metrics=["mean_shift"],
    )
    assert len(single) == 2
    small = SampleSet(samples.samples[:, 0, :90])
    large = SampleSet(samples.samples[:, 1])
    joined = compare(
        [small, large],
        ["ref", "other"],
        case.space,
        key=KEY,
        draws=SMALL_DRAWS,
        metrics=["mean_shift"],
    )
    assert pair_rows(joined)[0]["pair"] == ["ref", "other"]
    assert len(joined) == 2


def test_input_checks_of_the_sets_and_names() -> None:
    """Mismatched problems or parameters, bad names and one posterior raise."""
    case, samples = shift_set(count=60, batch=2)
    other_batch = SampleSet(samples.samples[:1, :1])
    with pytest.raises(ValueError, match=r"\(B, D\) = \(2, 2\), but set 1 has"):
        compare([samples, other_batch], list("abc"), case.space, key=KEY)
    with pytest.raises(ValueError, match="the space has 3 parameters"):
        compare(
            samples,
            ["a", "b"],
            zoo.shifted_gaussians(1.0, param_count=3).space,
            key=KEY,
        )
    with pytest.raises(ValueError, match="names must name each of the 2 posteriors"):
        compare(samples, ["a"], case.space, key=KEY)
    with pytest.raises(ValueError, match="names must be unique"):
        compare(samples, ["a", "a"], case.space, key=KEY)
    with pytest.raises(ValueError, match="at least two posteriors, got 1"):
        compare(samples.select(posteriors=[0]), ["a"], case.space, key=KEY)
    with pytest.raises(ValueError, match="at least one SampleSet"):
        compare([], [], case.space, key=KEY)
    with pytest.raises(TypeError, match="SampleSet or a sequence"):
        unchecked_compare(np.zeros((2, 5, 2)), ["a", "b"], case.space, key=KEY)


def test_mode_a_pairs_the_reference_with_every_other_posterior() -> None:
    """Mode A compares the reference with the others in the order of the names."""
    space, samples = three_posteriors()
    result = compare(
        samples,
        ["a", "b", "c"],
        space,
        key=KEY,
        reference="b",
        draws=SMALL_DRAWS,
        metrics=["mean_shift"],
    )
    pairs = [tuple(row["pair"]) for row in pair_rows(result)]
    assert pairs == [("b", "a")] * 2 + [("b", "c")] * 2


def test_mode_b_pairs_every_posterior_with_each_later_one() -> None:
    """Mode B takes (names[i], names[j]) for i < j, in the order of the names."""
    space, samples = three_posteriors()
    result = compare(
        samples,
        ["a", "b", "c"],
        space,
        key=KEY,
        draws=SMALL_DRAWS,
        metrics=["mean_shift"],
    )
    pairs = [tuple(row["pair"]) for row in pair_rows(result)]
    assert pairs == [("a", "b")] * 2 + [("a", "c")] * 2 + [("b", "c")] * 2


def test_the_rows_follow_problems_pairs_recipes_and_the_table_of_recipes() -> None:
    """Problems, pairs, then recipes in table order; reverse KL after forward."""
    case, samples = shift_set(count=60, batch=2)
    metrics = ["energy", "kl:gaussian", "mean_shift", "w2"]
    result = compare(
        samples,
        ["ref", "other"],
        case.space,
        key=KEY,
        reference="ref",
        draws=SMALL_DRAWS,
        metrics=metrics,
    )
    sequence = [
        (row["problem"], row["metric"], tuple(row["pair"])) for row in pair_rows(result)
    ]
    forward, reverse = ("ref", "other"), ("other", "ref")
    per_problem = (
        [("mean_shift", forward)] * 2
        + [(metric, forward) for metric in ("kl", "kl_location", "kl_shape")]
        + [(metric, reverse) for metric in ("kl", "kl_location", "kl_shape")]
        + [(metric, forward) for metric in ("w2", "w2_location", "w2_shape")]
        + [("energy", forward)]
    )
    assert sequence == [
        (problem, metric, pair) for problem in (0, 1) for metric, pair in per_problem
    ]
    parameters = [tuple(row["parameters"]) for row in pair_rows(result, problem=0)]
    assert parameters[:2] == [("x0",), ("x1",)]
    assert parameters[2:] == [("x0", "x1")] * 10


def test_a_row_carries_every_column_of_the_schema() -> None:
    """Null, significance, effect size, threshold, flag, settings and provenance."""
    case, samples = shift_set(count=100)
    result = compare(
        samples,
        ["ref", "other"],
        case.space,
        key=KEY,
        reference="ref",
        draws=SMALL_DRAWS,
        metrics=["mean_shift", "kl:gaussian"],
    )
    shift = one_row(result, "mean_shift", parameters=("x0",))
    assert shift["estimator"] == "empirical"
    assert json.loads(shift["estimator_settings"]) == {}
    assert len(shift["null_draws"]) == SMALL_DRAWS
    assert shift["null_recipe"] == {
        "construction": "permutation",
        "draws": SMALL_DRAWS,
        "sample_size": 100,
        "tail": "two_sided",
    }
    assert shift["uncertainty_method"] == "null_std"
    assert shift["uncertainty"] == shift["null_std"]
    assert shift["threshold"] == THRESHOLDS["mean_shift"]
    assert shift["r_eff"] is None
    assert 0 < shift["p_value"] <= 1
    finite = np.asarray(shift["null_draws"])
    assert shift["null_mean"] == pytest.approx(finite.mean())
    assert shift["null_std"] == pytest.approx(finite.std(ddof=1))
    shape = one_row(result, "kl_shape", pair=("ref", "other"))
    assert shape["sigma_eff"] is None
    assert shape["r_eff"] is not None
    assert shape["threshold"] == THRESHOLDS["kl_shape"]
    assert shape["null_recipe"]["tail"] == "upper"
    assert shape["provenance"]["seed"] == KEY
    assert (
        json.loads(one_row(result, "kl", pair=("ref", "other"))["estimator_settings"])
        == {}
    )


# * Whitenings


def test_the_reference_whitening_z_scores_each_parameter_by_the_reference() -> None:
    """In mode A the mean shift is in units of the reference width, per parameter."""
    case, samples = shift_set(count=150)
    result = compare(
        samples,
        ["ref", "other"],
        case.space,
        key=KEY,
        reference="ref",
        draws=SMALL_DRAWS,
        metrics=["mean_shift", "kl:gaussian"],
    )
    reference, other = np.asarray(samples.samples[0])
    expected = (other.mean(axis=0) - reference.mean(axis=0)) / reference.std(
        axis=0, ddof=1
    )
    values = [row["value"] for row in pair_rows(result, "mean_shift")]
    np.testing.assert_allclose(values, expected, rtol=1e-9)
    assert {row["space"] for row in pair_rows(result, "mean_shift")} == {
        "zscore(reference)"
    }
    assert {row["space"] for row in pair_rows(result, "kl")} == {"whiten(reference)"}


def test_the_pooled_whitening_pads_the_smaller_set_with_zero_weights() -> None:
    """Mode B pools both widths; sets of different N use zero-weight padding."""
    case, samples = shift_set(count=200)
    first = SampleSet(samples.samples[:, 0, :120])
    second = SampleSet(samples.samples[:, 1])
    result = compare(
        [first, second],
        ["small", "large"],
        case.space,
        key=KEY,
        draws=SMALL_DRAWS,
        metrics=["mean_shift"],
    )
    small, large = np.asarray(first.samples[0, 0]), np.asarray(second.samples[0, 0])
    pooled = np.sqrt(0.5 * (small.var(axis=0, ddof=1) + large.var(axis=0, ddof=1)))
    expected = (large.mean(axis=0) - small.mean(axis=0)) / pooled
    values = [row["value"] for row in pair_rows(result)]
    np.testing.assert_allclose(values, expected, rtol=1e-9)
    assert {row["space"] for row in pair_rows(result)} == {"zscore(pooled)"}


def test_a_pooled_whitening_can_serve_mode_a() -> None:
    """whitening="pooled" with a reference whitens each pair, and names it."""
    case, samples = shift_set(count=100)
    result = compare(
        samples,
        ["ref", "other"],
        case.space,
        key=KEY,
        reference="ref",
        whitening="pooled",
        draws=SMALL_DRAWS,
        metrics=["mean_shift", "kl:gaussian"],
    )
    assert {row["space"] for row in pair_rows(result, "kl")} == {"whiten(pooled)"}


def test_a_fisher_whitening_scales_by_the_inverse_fisher_matrix() -> None:
    """The z-score width of a parameter is the root of the diagonal of Γ⁻¹."""
    case, samples = shift_set(count=100)
    fisher = np.diag([4.0, 9.0])
    result = compare(
        samples,
        ["ref", "other"],
        case.space,
        key=KEY,
        whitening="fisher",
        fisher=fisher,
        centre=np.zeros(2),
        draws=SMALL_DRAWS,
        metrics=["mean_shift", "kl:gaussian"],
    )
    reference, other = np.asarray(samples.samples[0])
    raw = other.mean(axis=0) - reference.mean(axis=0)
    values = [row["value"] for row in pair_rows(result, "mean_shift")]
    np.testing.assert_allclose(values, raw / np.array([0.5, 1 / 3]), rtol=1e-9)
    assert {row["space"] for row in pair_rows(result, "mean_shift")} == {
        "zscore(fisher)"
    }
    assert {row["space"] for row in pair_rows(result, "kl")} == {"whiten(fisher)"}


def test_a_transform_leads_the_space_and_maps_the_fisher_matrix_and_centre() -> None:
    """Fisher rows with a transform equal those on the mapped sets and matrices."""
    case, samples = shift_set(count=100, seed=4)
    transform = Whitening.from_samples(samples.samples[:, 1], None, case.space)
    fisher = jnp.array([[4.0, 0.5], [0.5, 2.0]])
    centre = jnp.array([0.3, -0.2])
    options: dict[str, Any] = {
        "key": KEY,
        "whitening": "fisher",
        "draws": SMALL_DRAWS,
        "metrics": ["mean_shift", "kl:gaussian"],
    }
    mapped = compare(
        transform.map_sample_set(samples),
        ["ref", "other"],
        transform.map_space(case.space),
        fisher=transform.map_fisher(fisher, centre),
        centre=transform.forward(centre),
        **options,
    )
    direct = compare(
        samples,
        ["ref", "other"],
        case.space,
        transform=transform,
        fisher=fisher,
        centre=centre,
        **options,
    )
    np.testing.assert_allclose(
        direct.table["value"].to_numpy(), mapped.table["value"].to_numpy(), rtol=1e-8
    )
    np.testing.assert_allclose(
        direct.table["sigma_eff"].to_numpy(),
        mapped.table["sigma_eff"].to_numpy(),
        rtol=1e-7,
    )
    prefix = transform.description + " | "
    assert all(row["space"].startswith(prefix) for row in pair_rows(direct))
    assert not any(" | " in row["space"] for row in pair_rows(mapped))
    # * The hashes read the untransformed input, not the mapped set.
    first_direct = pair_rows(direct)[0]["provenance"]["input_hashes"]
    assert first_direct == [
        hash_arrays(samples.samples[0, 0], samples.log_weights[0, 0]),
        hash_arrays(samples.samples[0, 1], samples.log_weights[0, 1]),
    ]
    assert first_direct != pair_rows(mapped)[0]["provenance"]["input_hashes"]


def test_a_singular_covariance_gives_nan_rows_and_the_other_problem_goes_on() -> None:
    """A parameter without spread leaves its problem's rows NaN with the warning."""
    case, samples = shift_set(count=80, batch=2)
    broken = samples.samples.at[1, :, :, 1].set(0.5)
    result = compare(
        SampleSet(broken),
        ["ref", "other"],
        case.space,
        key=KEY,
        reference="ref",
        draws=SMALL_DRAWS,
        metrics=["mean_shift", "kl:gaussian", "mmd"],
    )
    healthy = pair_rows(result, problem=0)
    assert all(math.isfinite(row["value"]) for row in healthy)
    assert all(not row["warnings"] for row in healthy)
    for row in pair_rows(result, problem=1):
        assert math.isnan(row["value"])
        assert math.isnan(row["p_value"])
        assert math.isnan(row["sigma_sig"])
        effect = "r_eff" if row["metric"] == "kl_shape" else "sigma_eff"
        assert math.isnan(row[effect])
        assert row["flag"] is False
        assert row["threshold"] == THRESHOLDS[row["metric"]]
        for column in ("null_mean", "null_std", "null_draws", "null_recipe"):
            assert row[column] is None
        assert row["uncertainty"] is None
        assert row["uncertainty_method"] is None
        assert {warning["category"] for warning in row["warnings"]} == {
            "ValidityWarning"
        }
        reference_hash = hash_arrays(broken[1, 0], jnp.zeros(80))
        hashes = row["provenance"]["input_hashes"]
        assert hashes[row["pair"].index("ref")] == reference_hash
    unavailable = one_row(result, "mmd", problem=1)
    assert json.loads(unavailable["estimator_settings"]) == {
        "bandwidth": None,
        "cap": 10_000,
    }


def test_a_degenerate_bandwidth_or_epsilon_leaves_its_rows_nan_with_a_warning() -> None:
    """Mostly equal samples have a median distance of zero: kernel rows are NaN."""
    rng = np.random.default_rng(0)
    sets = np.zeros((2, 100, 2))
    sets[:, :8] = rng.normal(size=(2, 8, 2))
    case, _ = shift_set(count=10)
    result = compare(
        SampleSet(sets),
        ["ref", "other"],
        case.space,
        key=KEY,
        reference="ref",
        draws=SMALL_DRAWS,
        metrics=["mean_shift", "mmd", "sinkhorn", "sliced_sinkhorn"],
    )
    assert all(math.isfinite(row["value"]) for row in pair_rows(result, "mean_shift"))
    for metric, text in (
        ("mmd", "median bandwidth"),
        ("sinkhorn", "median bandwidth"),
        ("sliced_sinkhorn", "sliced Sinkhorn divergence"),
    ):
        row = one_row(result, metric)
        assert math.isnan(row["value"])
        assert row["flag"] is False
        assert row["null_draws"] is None
        (warning,) = row["warnings"]
        assert warning["category"] == "ValidityWarning"
        assert text in warning["message"]
    unavailable = json.loads(one_row(result, "sinkhorn")["estimator_settings"])
    assert unavailable["bandwidth"] is None
    assert unavailable["epsilon"] is None


def test_sphere_axes_are_z_scored_as_a_circle_and_an_interval() -> None:
    """Per-parameter rows treat the azimuth as a circle and the polar axis as a line."""
    case = zoo.von_mises_fisher(0.5, width=0.05, euclidean_count=1)
    samples = case.sample(jax.random.key(3), 200)
    names = case.space.names
    result = compare(
        samples,
        ["ref", "other"],
        case.space,
        key=KEY,
        reference="ref",
        draws=SMALL_DRAWS,
        metrics=["ks", "w1"],
    )
    azimuth = case.space.spheres[0].azimuth
    metrics = {
        next(iter(row["parameters"])): row["metric"]
        for row in pair_rows(result, "ks") + pair_rows(result, "kuiper_ks")
    }
    assert metrics[azimuth] == "kuiper_ks"
    assert {metrics[name] for name in names if name != azimuth} == {"ks"}
    # * The same z-score, built by hand on the flat space, reproduces the W1 rows.
    flat = ParamSpace(parameters=case.space.parameters)
    reference = samples.samples[:, 0]
    whitening = Whitening.from_samples(reference, None, flat, decorrelate=False)
    mapped_space = whitening.map_space(flat)
    azimuth_position = names.index(azimuth)
    period = flat.parameters[azimuth_position].bounds
    scaled = mapped_space.parameters[azimuth_position].bounds
    scale = float(whitening.circle_scales[0])
    assert scaled[1] - scaled[0] == pytest.approx((period[1] - period[0]) / scale)
    distances = marginal_distances(
        whitening.forward(samples.samples[0, 0]),
        None,
        whitening.forward(samples.samples[0, 1]),
        None,
        mapped_space,
    )
    values = [row["value"] for row in pair_rows(result, "w1")]
    np.testing.assert_allclose(values, np.asarray(distances.w1), rtol=1e-9)
    assert {row["space"] for row in pair_rows(result, "w1")} == {"zscore(reference)"}


# * The selection of metrics


@pytest.mark.parametrize(
    ("entries", "expected"),
    [
        (["ks"], ("ks:empirical",)),
        (["kuiper_ks"], ("ks:empirical",)),
        (["kuiper_ks:empirical"], ("ks:empirical",)),
        (["kl"], ("kl:gaussian", "kl:knn")),
        (["js"], ("js:kde", "js:knn")),
        (["hellinger"], ("hellinger:kde", "hellinger:gaussian", "hellinger:knn")),
        (["sliced_w2"], ("sliced_w2:empirical", "sliced_w2:gaussian")),
        (["w1", "mean_shift", "w1"], ("mean_shift:empirical", "w1:empirical")),
        (["sinkhorn:empirical", "kl:knn"], ("kl:knn", "sinkhorn:empirical")),
        (["energy", "energy:empirical"], ("energy:empirical",)),
    ],
)
def test_selection_by_metric_or_recipe_follows_the_table_order(
    entries: list[str], expected: tuple[str, ...]
) -> None:
    """A metric selects every estimator, a recipe one; duplicates collapse."""
    assert select_recipes(entries) == expected


def test_the_default_selection_is_the_fast_set() -> None:
    """The fourteen recipes of the fast set, in the order of the table."""
    assert select_recipes(None) == DEFAULT_METRICS
    assert len(DEFAULT_METRICS) == 14
    assert list(DEFAULT_METRICS) == [key for key in RECIPES if key in DEFAULT_METRICS]
    assert not any(key.endswith(":knn") for key in DEFAULT_METRICS)
    for slow in ("sinkhorn:empirical", "sliced_sinkhorn:empirical"):
        assert slow not in DEFAULT_METRICS


def test_the_recipes_are_named_by_metric_and_estimator() -> None:
    """The table holds recipe records in the order of the rows, and a placeholder."""
    assert list(RECIPES)[:3] == [
        "mean_shift:empirical",
        "width_ratio:empirical",
        "ks:empirical",
    ]
    assert RECIPES["kl:knn"] == Recipe("kl", "knn", Scope.JOINT)
    assert RECIPES["js:kde"].scope is Scope.PARAMETER
    assert RECIPES["c2st:classifier"].estimator == "classifier"
    assert len(RECIPES) == 20


@pytest.mark.parametrize(
    "term", ["kl_location", "kl_shape", "w2_shape", "hellinger_location"]
)
def test_terms_are_rejected_with_a_hint_to_select_the_total(term: str) -> None:
    """A location or shape term comes with its total."""
    total = term.split("_")[0]
    with pytest.raises(ValueError, match=f"select '{total}'"):
        select_recipes([term])
    with pytest.raises(ValueError, match="is a term of"):
        select_recipes([f"{term}:gaussian"])


def test_unknown_names_list_every_valid_name() -> None:
    """The message names the metrics and the recipes."""
    with pytest.raises(ValueError, match="unknown metric 'mean_shfit'") as raised:
        select_recipes(["mean_shfit"])
    assert "kuiper_ks" in str(raised.value)
    assert "sliced_sinkhorn:empirical" in str(raised.value)
    with pytest.raises(ValueError, match="unknown metric 'kl:flow'"):
        select_recipes(["kl:flow"])
    with pytest.raises(ValueError, match="metrics must be a sequence of strings"):
        select_recipes("ks")
    with pytest.raises(ValueError, match="metrics must hold strings"):
        select_recipes(["ks", 3])
    with pytest.raises(ValueError, match="metrics must not be empty"):
        select_recipes([])


def test_the_classifier_test_is_a_placeholder() -> None:
    """Selecting c2st raises with the message of the placeholder."""
    for entry in ("c2st", "c2st:classifier"):
        with pytest.raises(NotImplementedError, match="classifier two-sample test"):
            select_recipes(["ks", entry])


def test_a_circle_selects_kuiper_for_its_ks_row() -> None:
    """The alias kuiper_ks selects the KS recipe, whose circle rows are Kuiper's V."""
    case = zoo.wrapped_gaussians(0.5, param_count=2, width=0.3)
    samples = case.sample(jax.random.key(5), 150)
    result = compare(
        samples,
        ["ref", "other"],
        case.space,
        key=KEY,
        reference="ref",
        draws=SMALL_DRAWS,
        metrics=["kuiper_ks"],
    )
    assert [row["metric"] for row in pair_rows(result)] == ["kuiper_ks", "ks"]
    assert {row["estimator"] for row in pair_rows(result)} == {"empirical"}
    with pytest.raises(KeyError, match="kuiper_ks"):
        compare(
            samples,
            ["ref", "other"],
            case.space,
            key=KEY,
            reference="ref",
            metrics=["ks"],
            thresholds={"ks": 0.2},
        )


# * End to end on the zoo: recovery of the shifts and width ratios

# * σ_eff of one problem scatters by 0.1 at N = 300 (12 replicates), so its mean over
# * 4 problems at N = 400 scatters by 0.05, and 0.2 is four times that.
SIGMA_TOLERANCE = 0.2
# * The log of a width ratio scatters by 1/sqrt(N − 1) = 0.05 per problem, 0.025 over
# * 4 problems, and 0.1 is four times that.
LOG_RATIO_TOLERANCE = 0.1


@pytest.fixture(scope="module")
def shift_tables() -> dict[str, Result]:
    """Run the default recipes on a 1σ shift in both modes."""
    case, samples = shift_set(1.0, TABLE_SIZE, TABLE_PROBLEMS, seed=6)
    common: dict[str, Any] = {"key": KEY, "draws": TABLE_DRAWS}
    return {
        "A": compare(
            samples,
            ["reference", "shifted"],
            case.space,
            reference="reference",
            **common,
        ),
        "B": compare(samples, ["reference", "shifted"], case.space, **common),
    }


@pytest.fixture(scope="module")
def scale_tables() -> dict[str, Result]:
    """Run the default recipes on a width ratio of 1.25 along x0, in both modes."""
    case = zoo.scaled_gaussians(1.25, param_count=2)
    samples = case.sample(jax.random.key(7), TABLE_SIZE, batch_size=TABLE_PROBLEMS)
    common: dict[str, Any] = {"key": KEY, "draws": TABLE_DRAWS}
    return {
        "A": compare(
            samples,
            ["reference", "scaled"],
            case.space,
            reference="reference",
            **common,
        ),
        "B": compare(samples, ["reference", "scaled"], case.space, **common),
    }


def mean_over_problems(
    result: Result, metric: str, effect: str, **selection: Any
) -> float:
    """Return the mean of an effect size over the problems of one row."""
    rows = pair_rows(result, metric, **selection)
    assert len(rows) == TABLE_PROBLEMS
    return float(np.mean([row[effect] for row in rows]))


@pytest.mark.parametrize("mode", ["A", "B"])
def test_the_shift_is_recovered_by_every_sigma_eff_row(
    shift_tables: dict[str, Result], mode: str
) -> None:
    """Rows on x0 and the joint rows read the shift of 1σ; rows on x1 read none."""
    case = zoo.shifted_gaussians(1.0, param_count=2)
    result = shift_tables[mode]
    pair = ("reference", "shifted")
    truth = {
        row["parameters"][0]: row["value"]
        for row in case.reference.to_dicts()
        if row["metric"] == "mean_shift" and row["pair"] == list(pair)
    }
    joint_truth = float(
        case.reference.filter(pl.col("metric") == "mahalanobis_shift")["value"][0]
    )
    checked = 0
    for metric in result.table["metric"].unique():
        if metric in ("width_ratio", "kl_shape", "w2_shape", "hellinger_shape"):
            continue
        for row in pair_rows(result, metric, problem=0, pair=pair):
            names = tuple(row["parameters"])
            expected = joint_truth if len(names) == 2 else truth[names[0]]
            estimate = mean_over_problems(
                result,
                metric,
                "sigma_eff",
                pair=pair,
                parameters=names,
                estimator=row["estimator"],
            )
            assert estimate == pytest.approx(expected, abs=SIGMA_TOLERANCE), (
                metric,
                row["estimator"],
                names,
            )
            checked += 1
    assert checked >= 20


@pytest.mark.parametrize("mode", ["A", "B"])
def test_the_width_ratio_is_recovered_by_every_r_eff_row(
    scale_tables: dict[str, Result], mode: str
) -> None:
    """The width ratio and every shape term read 1.25 on x0 and the location terms 0."""
    case = zoo.scaled_gaussians(1.25, param_count=2)
    result = scale_tables[mode]
    pair = ("reference", "scaled")
    ratio = {
        row["parameters"][0]: row["value"]
        for row in case.reference.to_dicts()
        if row["metric"] == "width_ratio" and row["pair"] == list(pair)
    }
    assert ratio == {"x0": 1.25, "x1": 1.0}
    for name, truth in ratio.items():
        estimate = mean_over_problems(
            result, "width_ratio", "r_eff", pair=pair, parameters=(name,)
        )
        assert math.log(estimate) == pytest.approx(
            math.log(truth), abs=LOG_RATIO_TOLERANCE
        )
    for metric in ("kl_shape", "w2_shape", "hellinger_shape"):
        estimate = mean_over_problems(result, metric, "r_eff", pair=pair)
        assert math.log(estimate) == pytest.approx(
            math.log(1.25), abs=LOG_RATIO_TOLERANCE
        ), metric
    for metric in ("kl_location", "w2_location", "hellinger_location"):
        estimate = mean_over_problems(result, metric, "sigma_eff", pair=pair)
        assert estimate == pytest.approx(0.0, abs=SIGMA_TOLERANCE), metric
    for metric in ("mean_shift", "ks", "w1", "ad", "js", "hellinger"):
        estimate = (
            mean_over_problems(
                result, metric, "sigma_eff", pair=pair, parameters=("x1",)
            )
            if metric != "hellinger"
            else mean_over_problems(
                result,
                metric,
                "sigma_eff",
                pair=pair,
                parameters=("x1",),
                estimator="kde",
            )
        )
        assert estimate == pytest.approx(0.0, abs=SIGMA_TOLERANCE), metric


# * A width ratio of 2 makes the frame of the W2 shape term matter: with the
# * unit-width map of a reference whitening, a pooled frame reads 1.63.
FRAME_RATIO = 2.0
FRAME_SIZE = 300
FRAME_PROBLEMS = TABLE_PROBLEMS
FRAME_DRAWS = 30
FRAME_METRICS = ["kl:gaussian", "w2:gaussian", "hellinger:gaussian"]
# * The three terms share the covariances of each problem and differ only by the
# * null subtraction and the map; 0.1 is a few times their spread at N = 300.
FRAME_TOLERANCE = 0.1


def shape_ratios(result: Result) -> dict[str, float]:
    """Return the mean r_eff over the problems of every shape term of one pair."""
    return {
        metric: mean_over_problems(
            result, metric, "r_eff", pair=("reference", "scaled")
        )
        for metric in ("kl_shape", "w2_shape", "hellinger_shape")
    }


@pytest.mark.parametrize(
    ("reference", "whitening"), [(None, None), ("reference", "pooled")]
)
def test_the_w2_shape_row_reads_the_pooled_frame(
    reference: str | None, whitening: str | None
) -> None:
    """Mode B and a pooled mode A map w2_shape as kl_shape and hellinger_shape do."""
    case = zoo.scaled_gaussians(FRAME_RATIO, param_count=2)
    samples = case.sample(jax.random.key(11), FRAME_SIZE, batch_size=FRAME_PROBLEMS)
    result = compare(
        samples,
        ["reference", "scaled"],
        case.space,
        key=KEY,
        reference=reference,
        whitening=whitening,
        metrics=FRAME_METRICS,
        draws=FRAME_DRAWS,
    )
    ratios = shape_ratios(result)
    assert ratios["kl_shape"] == pytest.approx(FRAME_RATIO, abs=FRAME_TOLERANCE * 2)
    for metric, ratio in ratios.items():
        assert ratio == pytest.approx(ratios["kl_shape"], abs=FRAME_TOLERANCE), metric


def test_the_w2_shape_row_keeps_the_reference_map_in_a_reference_frame() -> None:
    """A reference whitening gives the first posterior the unit width of the map."""
    case = zoo.scaled_gaussians(FRAME_RATIO, param_count=2)
    samples = case.sample(jax.random.key(11), FRAME_SIZE, batch_size=FRAME_PROBLEMS)
    result = compare(
        samples,
        ["reference", "scaled"],
        case.space,
        key=KEY,
        reference="reference",
        metrics=FRAME_METRICS,
        draws=FRAME_DRAWS,
    )
    ratios = shape_ratios(result)
    for metric, ratio in ratios.items():
        assert ratio == pytest.approx(ratios["kl_shape"], abs=FRAME_TOLERANCE), metric


@pytest.mark.parametrize("mode", ["A", "B"])
def test_a_one_sigma_shift_fails_the_shift_rows(
    shift_tables: dict[str, Result], mode: str
) -> None:
    """Every row that measures the shift flags in every problem."""
    result = shift_tables[mode]
    shape_metrics = ("width_ratio", "kl_shape", "w2_shape", "hellinger_shape")
    flagged = 0
    for row in pair_rows(result, pair=("reference", "shifted")):
        names = tuple(row["parameters"])
        if row["metric"] in shape_metrics or names == ("x1",):
            continue
        assert row["flag"], (row["problem"], row["metric"], row["estimator"], names)
        assert row["sigma_sig"] >= 2.0
        assert row["sigma_eff"] >= row["threshold"]
        flagged += 1
    # * Per problem: six rows on x0 and ten joint rows without the reverse KL.
    assert flagged == TABLE_PROBLEMS * 16


def test_identical_sets_flag_nothing() -> None:
    """Two copies of one posterior sit below their null: no row is significant."""
    case, samples = shift_set(0.0, 200)
    copies = SampleSet(jnp.stack([samples.samples[0, 0], samples.samples[0, 0]]))
    shape_metrics = ("kl_shape", "w2_shape", "hellinger_shape")
    for reference in ("a", None):
        result = compare(
            copies,
            ["a", "b"],
            case.space,
            key=KEY,
            draws=TABLE_DRAWS,
            reference=reference,
        )
        for row in pair_rows(result):
            assert not row["flag"], (row["metric"], row["estimator"])
            assert row["sigma_sig"] < 2.0, (row["metric"], row["estimator"])
            if row["metric"] not in (*shape_metrics, "mean_shift", "width_ratio"):
                assert row["sigma_eff"] <= 1e-9, (row["metric"], row["estimator"])


# * Weighted sets and different sample counts


def importance_weighted_set(
    seed: int, count: int, target_shift: float = 0.5
) -> tuple[ParamSpace, SampleSet, SampleSet]:
    """Return a set with importance weights and an equal-weight shifted one."""
    rng = np.random.default_rng(seed)
    proposal, target = stats.norm(scale=1.5), stats.norm()
    draws = proposal.rvs(size=(count, 2), random_state=rng)
    log_weights = (target.logpdf(draws) - proposal.logpdf(draws)).sum(axis=1)
    shifted = rng.normal(size=(count * 2 // 3, 2)) + np.array([target_shift, 0.0])
    space = zoo.shifted_gaussians(1.0, param_count=2).space
    return space, SampleSet(draws, log_weights=log_weights), SampleSet(shifted)


def test_a_weighted_set_runs_through_every_recipe() -> None:
    """Importance weights and different counts: every recipe gives finite kNN rows."""
    space, weighted, equal = importance_weighted_set(0, 150)
    recipes = [recipe for recipe in RECIPES if not recipe.startswith("c2st")]
    result = compare(
        [weighted, equal],
        ["weighted", "equal"],
        space,
        key=KEY,
        reference="equal",
        draws=SMALL_DRAWS,
        metrics=recipes,
    )
    produced = {(row["metric"], row["estimator"]) for row in pair_rows(result)}
    for recipe in recipes:
        metric, estimator, _ = RECIPES[recipe]
        assert (metric, estimator) in produced
    for row in pair_rows(result):
        effect = row["sigma_eff"] if row["r_eff"] is None else row["r_eff"]
        assert math.isfinite(row["value"]), (row["metric"], row["estimator"])
        assert math.isfinite(effect), (row["metric"], row["estimator"])
    for row in pair_rows(result, estimator="knn"):
        assert row["null_recipe"]["construction"] == "permutation"
    assert not any(
        warning["category"] == "SubsamplingWarning"
        for row in pair_rows(result)
        for warning in row["warnings"]
    )


# * Nearest-neighbour rows


def expected_cut(
    result_key: int,
    names: tuple[str, str],
    samples: SampleSet,
    whitening: Whitening,
) -> list[int]:
    """Return the sizes that rejection keeps for the two posteriors of problem 0."""
    root = prng.root_key(result_key)
    sizes = []
    for index, name in enumerate(names):
        whitened = whitening.forward(samples.samples[0, index])
        kept = rejection_subset(
            whitened,
            samples.log_weights[0, index],
            key=prng.derive_key(root, "compare", 0, *names, "knn", "prepare", name),
        )
        sizes.append(kept.shape[0])
    return sizes


def knn_pair(count: int = 120) -> tuple[ParamSpace, SampleSet]:
    """Return a weighted and an equal-weight posterior of two parameters, as one set."""
    space, weighted, equal = importance_weighted_set(3, count)
    size = min(weighted.shape.sample, equal.shape.sample)
    stacked = SampleSet(
        jnp.stack([weighted.samples[0, 0, :size], equal.samples[0, 0, :size]]),
        log_weights=jnp.stack([weighted.log_weights[0, 0, :size], jnp.zeros(size)]),
    )
    return space, stacked


def test_the_knn_rows_reject_cut_and_report_their_counts() -> None:
    """Rejection keeps M samples per set, the sets are cut to the smaller count."""
    space, samples = knn_pair()
    names = ("weighted", "equal")
    result = compare(
        samples,
        list(names),
        space,
        key=KEY,
        reference="equal",
        draws=SMALL_DRAWS,
        metrics=["js:knn", "hellinger:knn"],
    )
    # * Mode A has the reference "equal" as first posterior of the pair.
    assert [row["pair"] for row in pair_rows(result)] == [["equal", "weighted"]] * 2
    reference_whitening = Whitening.from_samples(samples.samples[0, 1:2], None, space)
    sizes = expected_cut(
        KEY,
        ("equal", "weighted"),
        SampleSet(
            samples.samples[:, [1, 0]], log_weights=samples.log_weights[:, [1, 0]]
        ),
        reference_whitening,
    )
    assert sizes[0] == 80  # every positive-weight sample of the equal-weight set
    expected = min(sizes)
    for row in pair_rows(result):
        assert row["null_recipe"]["sample_size"] == expected
        assert row["null_recipe"]["draws"] == SMALL_DRAWS
        assert row["warnings"] == []
    assert json.loads(pair_rows(result, "js")[0]["estimator_settings"]) == {
        "neighbours": 10,
        "norm": "euclidean",
    }
    assert (
        json.loads(pair_rows(result, "hellinger")[0]["estimator_settings"])[
            "neighbours"
        ]
        == 5
    )


def test_the_kl_calibration_curve_of_each_direction_inverts_its_row() -> None:
    """The σ_eff of a kNN KL row is that of the curve of its own direction."""
    case, samples = shift_set(1.0, 90, seed=9)
    result = compare(
        samples,
        ["ref", "other"],
        case.space,
        key=KEY,
        reference="ref",
        draws=SMALL_DRAWS,
        metrics=["kl:knn"],
    )
    rows = pair_rows(result, "kl")
    assert [row["pair"] for row in rows] == [["ref", "other"], ["other", "ref"]]
    root = prng.root_key(KEY)
    for row, (first, second) in zip(
        rows, (("ref", "other"), ("other", "ref")), strict=True
    ):
        curve = knn_kl_curve(
            2,
            90,
            90,
            key=prng.derive_key(
                root,
                "compare",
                0,
                "ref",
                "other",
                "kl",
                "knn",
                first,
                second,
                "calibration",
            ),
            neighbours=KL_NEIGHBOURS,
        )
        subtracted = row["value"] - row["null_mean"]
        expected = float(sigma_eff("kl", subtracted, estimator="knn", curve=curve))
        assert row["sigma_eff"] == pytest.approx(expected, rel=1e-9)
    forward, reverse = (float(row["sigma_eff"]) for row in rows)
    assert forward != reverse


def test_the_cap_cuts_the_knn_sets_and_warns() -> None:
    """Above the cap both sets are cut to it, with one SubsamplingWarning."""
    case, samples = shift_set(1.0, 90, seed=10)
    result = compare(
        samples,
        ["ref", "other"],
        case.space,
        key=KEY,
        reference="ref",
        draws=SMALL_DRAWS,
        cap=40,
        metrics=["hellinger:knn", "mean_shift"],
    )
    knn = one_row(result, "hellinger")
    assert knn["null_recipe"]["sample_size"] == 40
    (warning,) = knn["warnings"]
    assert warning["category"] == SubsamplingWarning.__name__
    assert "keeps 90 samples" in warning["message"]
    assert "cap of 40" in warning["message"]
    assert all(not row["warnings"] for row in pair_rows(result, "mean_shift"))


def test_the_split_half_null_cuts_the_reference_in_the_knn_rows() -> None:
    """In mode A the split-half sides come from the cut reference."""
    case, samples = shift_set(1.0, 100, seed=11)
    result = compare(
        samples,
        ["ref", "other"],
        case.space,
        key=KEY,
        reference="ref",
        null="split_half",
        draws=SMALL_DRAWS,
        metrics=["js:knn", "mean_shift"],
    )
    # * Equal effective sizes ask for sides of the whole reference, which clips to N/2.
    knn = one_row(result, "js")
    assert knn["null_recipe"] == {
        "construction": "split_half",
        "draws": SMALL_DRAWS,
        "sample_size": 50,
        "tail": "upper",
    }
    for row in pair_rows(result):
        assert row["null_recipe"]["construction"] == "split_half"
        assert row["null_recipe"]["sample_size"] == 50
        (warning,) = row["warnings"]
        assert warning["category"] == "ValidityWarning"
        assert "split-half null" in warning["message"]


def test_the_js_row_reads_equal_counts_of_the_cut_sets_and_the_half_samples() -> None:
    """JS takes the first min(N1, N2) samples: finite errors with unequal sets."""
    case, samples = shift_set(1.0, 100, seed=20)
    reference = SampleSet(samples.samples[:, 0])
    other = SampleSet(samples.samples[:, 1, :70])
    result = compare(
        [reference, other],
        ["ref", "other"],
        case.space,
        key=KEY,
        reference="ref",
        null="split_half",
        uncertainty="subsampling_bootstrap",
        half_samples=10,
        draws=SMALL_DRAWS,
        metrics=["js:knn"],
    )
    row = one_row(result, "js")
    assert math.isfinite(row["uncertainty"])
    assert row["uncertainty"] > 0
    assert row["uncertainty_method"] == "subsampling_bootstrap"
    assert math.isfinite(row["value"])
    whitening = Whitening.from_samples(reference.samples[0], None, case.space)
    root = prng.root_key(KEY)
    path = ("compare", 0, "ref", "other")
    cut = []
    for name, posterior in (("ref", reference), ("other", other)):
        whitened = whitening.forward(posterior.samples[0, 0])
        order = jax.random.permutation(
            prng.derive_key(root, *path, "knn", "equal_counts", name),
            whitened.shape[0],
        )
        cut.append(whitened[order])
    expected = knn_js_divergence(
        cut[0][:70],
        None,
        cut[1][:70],
        None,
        whitening.map_space(case.space),
        key=prng.derive_key(root, *path, "js", "knn", "subsample"),
        neighbours=JS_NEIGHBOURS,
    )
    assert row["value"] == pytest.approx(float(expected), rel=1e-12)


# * Keys, provenance and the shared null


def test_the_same_key_repeats_the_table_and_another_key_changes_the_nulls() -> None:
    """Rows are a function of the key; the observed values do not depend on it."""
    case, samples = shift_set(1.0, 100, seed=12)
    options: dict[str, Any] = {
        "reference": "ref",
        "draws": SMALL_DRAWS,
        "metrics": ["mean_shift", "energy"],
    }
    first = compare(samples, ["ref", "other"], case.space, key=KEY, **options)
    again = compare(samples, ["ref", "other"], case.space, key=KEY, **options)
    other = compare(samples, ["ref", "other"], case.space, key=KEY + 1, **options)
    assert first.table.equals(again.table)
    assert first.table["value"].to_list() == other.table["value"].to_list()
    assert first.table["null_draws"].to_list() != other.table["null_draws"].to_list()


def test_a_recipe_added_to_the_selection_leaves_the_other_nulls_unchanged() -> None:
    """The null draws of a row do not depend on which other recipes run."""
    case, samples = shift_set(1.0, 100, seed=13)
    options: dict[str, Any] = {"reference": "ref", "draws": SMALL_DRAWS, "key": KEY}
    small = compare(
        samples,
        ["ref", "other"],
        case.space,
        metrics=["mean_shift", "energy"],
        **options,
    )
    large = compare(
        samples,
        ["ref", "other"],
        case.space,
        metrics=["mean_shift", "energy", "mmd", "kl:gaussian"],
        **options,
    )
    for metric in ("mean_shift", "energy"):
        for row, bigger in zip(
            pair_rows(small, metric), pair_rows(large, metric), strict=True
        ):
            assert row["null_draws"] == bigger["null_draws"]
            assert row["value"] == bigger["value"]


def test_the_shared_null_is_the_permutation_plan_of_its_path() -> None:
    """The draws of a row are the statistic on the plan keyed by the path and "null"."""
    case, samples = shift_set(1.0, 100, seed=14)
    result = compare(
        samples,
        ["ref", "other"],
        case.space,
        key=KEY,
        reference="ref",
        draws=SMALL_DRAWS,
        metrics=["mean_shift"],
    )
    reference, other = samples.samples[0]
    root = prng.root_key(KEY)
    plan = permutation_null(
        reference,
        None,
        other,
        None,
        key=prng.derive_key(root, "compare", 0, "ref", "other", "null"),
        first_n_eff=100.0,
        second_n_eff=100.0,
        draws=SMALL_DRAWS,
    )
    whitening = Whitening.from_samples(
        reference[None],
        None,
        ParamSpace(parameters=case.space.parameters),
        decorrelate=False,
    )
    mapped = whitening.map_space(case.space)
    plan = plan._replace(samples=whitening.forward(plan.samples))

    def statistic(
        first: Any, first_weights: Any, second: Any, second_weights: Any
    ) -> Any:
        return mean_shift(first, first_weights, second, second_weights, mapped)

    draws = null_draws(statistic, plan)
    for position, row in enumerate(pair_rows(result)):
        np.testing.assert_allclose(row["null_draws"], draws[position], rtol=1e-12)
    assert permutation_side_size(jnp.zeros(100), jnp.zeros(100)) == 100


def test_an_integer_seed_fills_the_provenance_and_a_key_leaves_it_null() -> None:
    """Seed and key text in the provenance, with hashes in pair order."""
    case, samples = shift_set(1.0, 60, seed=15)
    options: dict[str, Any] = {
        "reference": "ref",
        "draws": SMALL_DRAWS,
        "metrics": ["mean_shift", "kl:gaussian"],
        "config_hash": hash_arrays(np.arange(3)),
    }
    seeded = compare(samples, ["ref", "other"], case.space, key=7, **options)
    keyed = compare(
        samples, ["ref", "other"], case.space, key=prng.root_key(7), **options
    )
    hashes = [
        hash_arrays(samples.samples[0, 0], samples.log_weights[0, 0]),
        hash_arrays(samples.samples[0, 1], samples.log_weights[0, 1]),
    ]
    for row in pair_rows(seeded):
        assert row["provenance"]["seed"] == 7
        assert row["provenance"]["prng_key"] == prng.format_key(prng.root_key(7))
        assert row["provenance"]["config_hash"] == options["config_hash"]
    for row in pair_rows(keyed):
        assert row["provenance"]["seed"] is None
        assert row["provenance"]["prng_key"] == prng.format_key(prng.root_key(7))
    assert seeded.table["value"].to_list() == keyed.table["value"].to_list()
    forward = one_row(seeded, "kl", pair=("ref", "other"))["provenance"]
    reverse = one_row(seeded, "kl", pair=("other", "ref"))["provenance"]
    assert forward["input_hashes"] == hashes
    assert reverse["input_hashes"] == hashes[::-1]


def test_the_settings_of_the_recipes_are_fixed_once_per_comparison() -> None:
    """Bandwidth, factor, ε, directions and caps are in the settings of the rows."""
    case, samples = shift_set(1.0, 100, seed=16)
    result = compare(
        samples,
        ["ref", "other"],
        case.space,
        key=KEY,
        reference="ref",
        draws=SMALL_DRAWS,
        cap=500,
        metrics=["js:kde", "mmd", "energy", "sinkhorn", "sliced_w2", "sliced_sinkhorn"],
    )
    reference, other = samples.samples[0]
    whitening = Whitening.from_samples(reference[None], None, case.space)
    joint = whitening.map_space(case.space)
    root = prng.root_key(KEY)
    bandwidth = float(
        median_bandwidth(
            whitening.forward(reference),
            None,
            whitening.forward(other),
            None,
            joint,
            key=prng.derive_key(root, "compare", 0, "ref", "other", "bandwidth"),
        )
    )
    settings = {
        (row["metric"], row["estimator"]): json.loads(row["estimator_settings"])
        for row in pair_rows(result)
    }
    assert settings[("js", "kde")]["bandwidth"] == pytest.approx(
        float(shared_bandwidth_factor(jnp.zeros(100), jnp.zeros(100)))
    )
    assert settings[("mmd", "empirical")] == {
        "bandwidth": pytest.approx(bandwidth),
        "cap": 500,
    }
    assert settings[("energy", "empirical")] == {"cap": 500}
    assert settings[("sinkhorn", "empirical")] == {
        "bandwidth": pytest.approx(bandwidth),
        "cap": 500,
        "epsilon": pytest.approx(0.25 * bandwidth**2),
        "epsilon_factor": 0.25,
    }
    assert settings[("sliced_w2", "empirical")] == {
        "directions": 500,
        "orthonormal": True,
    }
    sliced = settings[("sliced_sinkhorn", "empirical")]
    assert sliced["cap"] == 2_000
    assert sliced["directions"] == 20
    assert sliced["epsilon"] > 0
    assert sliced["orthonormal"] is True


# * The subsampling bootstrap


def test_the_half_sample_error_of_a_mean_shift_matches_its_analytic_error() -> None:
    """The error column holds the half-sample error; the null spread stays."""
    case, samples = shift_set(0.0, 400, seed=17)
    pairs = 300
    result = compare(
        samples,
        ["ref", "other"],
        case.space,
        key=KEY,
        reference="ref",
        draws=SMALL_DRAWS,
        uncertainty="subsampling_bootstrap",
        half_samples=pairs,
        metrics=["mean_shift"],
    )
    reference, other = np.asarray(samples.samples[0])
    # * Halves of fixed sets vary by the finite-population variance, s1²/N + s2²/N.
    analytic = np.sqrt(
        (reference.var(axis=0, ddof=1) + other.var(axis=0, ddof=1)) / 400
    ) / reference.std(axis=0, ddof=1)
    monte_carlo_error = 1 / math.sqrt(2 * (pairs - 1))
    for position, row in enumerate(pair_rows(result)):
        assert row["uncertainty_method"] == "subsampling_bootstrap"
        assert row["uncertainty"] == pytest.approx(
            analytic[position], rel=5 * monte_carlo_error
        )
        assert row["null_std"] != row["uncertainty"]
        assert row["null_std"] > 0


def test_the_bootstrap_gives_finite_errors_for_every_kind_of_recipe() -> None:
    """Moments, kernels, Gaussian fits and kNN rows all get a half-sample error."""
    space, samples = knn_pair(100)
    result = compare(
        samples,
        ["weighted", "equal"],
        space,
        key=KEY,
        reference="equal",
        draws=SMALL_DRAWS,
        uncertainty="subsampling_bootstrap",
        half_samples=8,
        metrics=["mean_shift", "ks", "kl:gaussian", "energy", "kl:knn"],
    )
    for row in pair_rows(result):
        assert row["uncertainty_method"] == "subsampling_bootstrap"
        assert math.isfinite(row["uncertainty"]), (row["metric"], row["estimator"])
        assert row["uncertainty"] >= 0


# * The ε option, thresholds and level


def test_one_sinkhorn_row_per_epsilon_factor() -> None:
    """Each factor gives a row with its ε in the settings."""
    case, samples = shift_set(1.0, 60, seed=18)
    result = compare(
        samples,
        ["ref", "other"],
        case.space,
        key=KEY,
        reference="ref",
        draws=SMALL_DRAWS,
        sinkhorn_epsilon_factors=(0.1, 0.25),
        metrics=["sinkhorn"],
    )
    rows = pair_rows(result)
    assert len(rows) == 2
    settings = [json.loads(row["estimator_settings"]) for row in rows]
    assert [entry["epsilon_factor"] for entry in settings] == [0.1, 0.25]
    bandwidth = settings[0]["bandwidth"]
    assert settings[0]["epsilon"] == pytest.approx(0.1 * bandwidth**2)
    assert settings[1]["epsilon"] == pytest.approx(0.25 * bandwidth**2)
    assert rows[0]["value"] != rows[1]["value"]


def test_thresholds_and_level_change_the_threshold_and_the_flag() -> None:
    """A stricter threshold or a lower level changes which rows fail."""
    case, samples = shift_set(1.0, 100, seed=19)
    options: dict[str, Any] = {
        "reference": "ref",
        "key": KEY,
        "draws": TABLE_DRAWS,
        "metrics": ["mean_shift", "kl:gaussian"],
    }
    default = compare(samples, ["ref", "other"], case.space, **options)
    assert one_row(default, "mean_shift", parameters=("x0",))["flag"]
    thresholds = {**THRESHOLDS, "mean_shift": 5.0, "kl": 0.05}
    strict = compare(
        samples, ["ref", "other"], case.space, thresholds=thresholds, **options
    )
    row = one_row(strict, "mean_shift", parameters=("x0",))
    assert row["threshold"] == 5.0
    assert not row["flag"]
    assert one_row(strict, "kl", pair=("ref", "other"))["threshold"] == 0.05
    unreachable = compare(samples, ["ref", "other"], case.space, level=50.0, **options)
    assert not any(row["flag"] for row in pair_rows(unreachable))
    assert not any(row["flag"] for row in pair_rows(unreachable, "kl"))
    with pytest.raises(KeyError, match="kl_location"):
        compare(
            samples,
            ["ref", "other"],
            case.space,
            thresholds={"mean_shift": 0.2, "kl": 0.2, "kl_shape": 0.14},
            **options,
        )


# * Argument errors


def test_the_reference_and_the_whitening_are_checked() -> None:
    """The reference is a name; whitening, Fisher matrices and centres fit the mode."""
    case, samples = shift_set(1.0, 40, batch=2)
    base: dict[str, Any] = {"key": KEY, "metrics": ["mean_shift"]}
    names = ["ref", "other"]
    with pytest.raises(
        ValueError, match=r"reference must be one of \['ref', 'other'\]"
    ):
        compare(samples, names, case.space, reference="third", **base)
    with pytest.raises(ValueError, match="reference whitening needs reference"):
        compare(samples, names, case.space, whitening="reference", **base)
    with pytest.raises(ValueError, match="is not a valid WhiteningSource"):
        compare(samples, names, case.space, whitening="pooled-ish", **base)
    with pytest.raises(TypeError, match="whitening must be a string or None, got int"):
        unchecked_compare(samples, names, case.space, whitening=3, **base)
    with pytest.raises(ValueError, match="needs fisher= and centre="):
        compare(samples, names, case.space, whitening="fisher", **base)
    with pytest.raises(ValueError, match="needs fisher= and centre="):
        compare(
            samples, names, case.space, whitening="fisher", fisher=np.eye(2), **base
        )
    with pytest.raises(ValueError, match="serve the fisher whitening"):
        compare(
            samples, names, case.space, fisher=np.eye(2), centre=np.zeros(2), **base
        )
    with pytest.raises(ValueError, match="serve the fisher whitening"):
        compare(
            samples, names, case.space, whitening="pooled", centre=np.zeros(2), **base
        )
    fisher = np.stack([np.eye(2)] * 2)
    centre = np.zeros((2, 2))
    with pytest.raises(ValueError, match=r"fisher must be shaped \(2, 2, 2\)"):
        compare(
            samples,
            names,
            case.space,
            whitening="fisher",
            fisher=np.eye(2),
            centre=centre,
            **base,
        )
    with pytest.raises(ValueError, match=r"centre must be shaped \(2, 2\)"):
        compare(
            samples,
            names,
            case.space,
            whitening="fisher",
            fisher=fisher,
            centre=np.zeros(2),
            **base,
        )
    with pytest.raises(ValueError, match="symmetric"):
        compare(
            samples,
            names,
            case.space,
            whitening="fisher",
            fisher=np.stack([np.array([[1.0, 0.5], [0.0, 1.0]])] * 2),
            centre=centre,
            **base,
        )


def test_a_fisher_matrix_of_one_problem_may_be_two_dimensional() -> None:
    """For B = 1 a (D, D) matrix and a (D,) centre stand for the batch of one."""
    case, samples = shift_set(1.0, 60)
    flat = compare(
        samples,
        ["ref", "other"],
        case.space,
        key=KEY,
        whitening="fisher",
        fisher=np.eye(2),
        centre=np.zeros(2),
        draws=SMALL_DRAWS,
        metrics=["mean_shift"],
    )
    batched = compare(
        samples,
        ["ref", "other"],
        case.space,
        key=KEY,
        whitening="fisher",
        fisher=np.eye(2)[None],
        centre=np.zeros((1, 2)),
        draws=SMALL_DRAWS,
        metrics=["mean_shift"],
    )
    assert flat.table.equals(batched.table)


def test_the_null_and_the_uncertainty_method_are_checked() -> None:
    """Only the permutation and split-half nulls, and two uncertainty methods."""
    case, samples = shift_set(1.0, 40)
    base: dict[str, Any] = {"key": KEY, "metrics": ["mean_shift"]}
    names = ["ref", "other"]
    with pytest.raises(
        ValueError, match="the Gaussian surrogate is the null of the diagnostics"
    ):
        compare(samples, names, case.space, null="gaussian_surrogate", **base)
    with pytest.raises(ValueError, match="split_half null needs reference"):
        compare(samples, names, case.space, null="split_half", **base)
    with pytest.raises(ValueError, match="is not a valid Construction"):
        compare(samples, names, case.space, null="bootstrap", **base)
    with pytest.raises(TypeError, match="null must be a string"):
        unchecked_compare(samples, names, case.space, null=1, **base)
    with pytest.raises(ValueError, match="delta method"):
        compare(samples, names, case.space, uncertainty="delta_method", **base)
    with pytest.raises(ValueError, match="is not a valid UncertaintyMethod"):
        compare(samples, names, case.space, uncertainty="jackknife", **base)
    with pytest.raises(TypeError, match="uncertainty must be a string"):
        unchecked_compare(samples, names, case.space, uncertainty=2, **base)


@pytest.mark.parametrize(
    ("option", "value", "error", "message"),
    [
        ("draws", 0, ValueError, "draws must be at least 1"),
        ("draws", 2.5, TypeError, "draws must be an integer"),
        ("draws", True, TypeError, "draws must be an integer, got bool"),
        ("half_samples", 0, ValueError, "half_samples must be at least 1"),
        ("half_samples", "8", TypeError, "half_samples must be an integer"),
        ("cap", 1, ValueError, "cap must be at least 2"),
        ("cap", 10.0, TypeError, "cap must be an integer"),
        ("sinkhorn_epsilon_factors", [], ValueError, "non-empty sequence"),
        ("sinkhorn_epsilon_factors", "0.25", ValueError, "non-empty sequence"),
        ("sinkhorn_epsilon_factors", [0.1, 0.1], ValueError, "distinct"),
        ("sinkhorn_epsilon_factors", [0.1, -0.2], ValueError, "positive and finite"),
        ("sinkhorn_epsilon_factors", [math.nan], ValueError, "positive and finite"),
        ("sinkhorn_epsilon_factors", [math.inf], ValueError, "positive and finite"),
        ("sinkhorn_epsilon_factors", ["a"], TypeError, "must be a number"),
        ("sinkhorn_epsilon_factors", [True], TypeError, "must be a number"),
        ("level", math.nan, ValueError, "level must be finite"),
        ("level", math.inf, ValueError, "level must be finite"),
        ("level", "2", TypeError, "level must be a number"),
        ("budget", 0.0, ValueError, "budget must be positive"),
        ("budget", -1.0, ValueError, "budget must be positive"),
        ("budget", math.nan, ValueError, "budget must be positive"),
        ("budget", None, TypeError, "budget must be a number"),
        ("thresholds", [0.2], TypeError, "thresholds must be a mapping"),
        ("key", 2.5, TypeError, "key must be a JAX PRNG key or an integer seed"),
        ("key", True, TypeError, "got bool"),
        ("key", -1, ValueError, "the seed must lie in"),
    ],
)
def test_counts_numbers_thresholds_and_the_key_are_checked(
    option: str, value: object, error: type[Exception], message: str
) -> None:
    """Each numeric argument and the key raise before any computation."""
    case, samples = shift_set(1.0, 40)
    options: dict[str, Any] = {"key": KEY, "metrics": ["mean_shift"], option: value}
    with pytest.raises(error, match=message):
        unchecked_compare(samples, ["ref", "other"], case.space, **options)


def test_an_infinite_budget_is_allowed() -> None:
    """A budget of infinity switches the dry-run check off."""
    case, samples = shift_set(1.0, 40)
    result = compare(
        samples,
        ["ref", "other"],
        case.space,
        key=KEY,
        budget=math.inf,
        draws=SMALL_DRAWS,
        metrics=["mean_shift"],
    )
    assert len(result) == 2


def test_the_result_is_a_result_table_with_the_schema() -> None:
    """The engine returns a Result whose table has the agreed columns."""
    case, samples = shift_set(1.0, 40)
    result = compare(
        samples,
        ["ref", "other"],
        case.space,
        key=KEY,
        draws=SMALL_DRAWS,
        metrics=["mean_shift"],
    )
    assert isinstance(result, Result)
    assert result.table.schema == RESULT_SCHEMA
