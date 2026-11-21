"""Tests for the dry run: the model, the groups of both engines and the warning."""

import inspect
import math
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import polars as pl
import pytest

from mixed_signals import _config, errors, zoo
from mixed_signals.core.samples import SampleSet
from mixed_signals.core.space import ParamSpace
from mixed_signals.engines import cost
from mixed_signals.engines.compare import compare
from mixed_signals.engines.cost import (
    COST_BUDGET,
    CostEstimate,
    estimate_compare_cost,
    estimate_compare_signals_cost,
    estimate_diagnose_cost,
    warn_if_over_budget,
)
from mixed_signals.engines.diagnose import diagnose
from mixed_signals.errors import CapabilityError, CostWarning
from mixed_signals.mcmc.reduce import reduce_chains
from mixed_signals.nulls.surrogates import surrogate_size
from mixed_signals.transforms.whiten import Whitening

unchecked_compare_cost = inspect.unwrap(estimate_compare_cost)
unchecked_diagnose_cost = inspect.unwrap(estimate_diagnose_cost)
unchecked_signals_cost = inspect.unwrap(estimate_compare_signals_cost)
DRAWS = 9
CAP = 100_000

# * The constants of the model, copied from the fit of 2026-11-19: (c, a, d0, cap kind).
TABLE: dict[str, tuple[float, float, float | None, str | None]] = {
    "moments": (4.5e-5, 0.62, None, None),
    "marginal": (0.0022, 1.13, None, None),
    "kde": (0.031, 1.01, None, None),
    "gaussian": (4.9e-4, 0.58, 9.4, None),
    "knn_kl": (2.54, 2.09, 43.0, "cap"),
    "knn_js": (8.9, 2.06, 40.0, "cap"),
    "knn_hellinger": (5.18, 2.10, 42.0, "cap"),
    "mmd": (0.27, 1.94, 10.5, "cap"),
    "energy": (0.26, 1.95, 7.0, "cap"),
    "sinkhorn": (6.7, 2.19, None, "cap"),
    "sliced_w2": (0.57, 1.10, None, None),
    "sliced_w2_gaussian": (5.2e-4, 0.65, 7.4, None),
    "sliced_sinkhorn": (5.34, 1.77, None, "sliced_sinkhorn_cap"),
    "median_bandwidth": (0.30, 0.0, None, None),
    "sliced_sinkhorn_epsilon": (0.08, 0.0, None, None),
    "marginal_moments": (3.9e-5, 0.72, None, None),
    "anderson_darling": (5.5e-4, 0.70, None, None),
    "mardia": (0.019, 1.62, 45.0, "cap"),
    "dependence": (3.6, 2.63, None, "pair_cap"),
    "marginal_modes": (0.0153, 0.98, None, None),
    "projection_modes": (0.0159, 0.96, 27.0, None),
    "gmm_bic": (0.69, 0.32, 0.0, None),
    "signal_matrix": (0.146, 2.10, 56.0, None),
    "signal_draw": (3.86e-3, 2.00, None, None),
}
MISFITS: dict[str, float] = {
    "moments": 2.87,
    "marginal": 1.18,
    "kde": 1.27,
    "gaussian": 1.13,
    "knn_kl": 1.05,
    "knn_js": 1.09,
    "knn_hellinger": 1.05,
    "mmd": 1.15,
    "energy": 1.13,
    "sinkhorn": 1.36,
    "sliced_w2": 1.02,
    "sliced_w2_gaussian": 1.07,
    "sliced_sinkhorn": 1.08,
    "median_bandwidth": 1.00,
    "sliced_sinkhorn_epsilon": 1.03,
    "marginal_moments": 2.14,
    "anderson_darling": 2.11,
    "mardia": 1.17,
    "dependence": 1.31,
    "marginal_modes": 1.24,
    "projection_modes": 1.34,
    "gmm_bic": 1.55,
    "signal_matrix": 1.32,
    "signal_draw": 1.03,
}


# * Helpers


def model(
    statistic: str,
    samples: int,
    dimension: int,
    units: float = 1.0,
    *,
    caps: dict[str, int] | None = None,
) -> float:
    """Return the seconds of one evaluation, as the model of the module states them."""
    coefficient, exponent, offset, cap_kind = TABLE[statistic]
    kept = samples
    if cap_kind is not None and caps is not None:
        kept = min(samples, caps[cap_kind])
    factor = 1.0 if offset is None else (dimension + offset) / (16 + offset)
    return coefficient * math.pow(kept / 10_000, exponent) * units * factor


def shift_set(count: int, batch: int = 1, posteriors: int = 2) -> SampleSet:
    """Return a set of Gaussians with the given sizes, in two parameters."""
    case = zoo.shifted_gaussians(0.5, param_count=2)
    samples = case.sample(jax.random.key(0), count, batch_size=batch)
    if posteriors == 3:
        return SampleSet(
            jnp.concatenate([samples.samples, samples.samples[:, :1]], axis=1)
        )
    return samples


SPACE = zoo.shifted_gaussians(0.5, param_count=2).space


def rows(estimate: CostEstimate, metric: str, **selection: Any) -> list[dict[str, Any]]:
    """Return the table rows of a metric, with other columns as given."""
    return [
        row
        for row in estimate.table.to_dicts()
        if row["metric"] == metric
        and all(row[column] == value for column, value in selection.items())
    ]


def compare_cost(
    count: int = 1000, metrics: list[str] | None = None, **options: Any
) -> CostEstimate:
    """Estimate a comparison of two Gaussians of `count` samples in mode A."""
    return estimate_compare_cost(
        shift_set(count),
        ["ref", "other"],
        SPACE,
        reference="ref",
        metrics=metrics,
        draws=options.pop("draws", DRAWS),
        **options,
    )


# * The model and its constants


def test_the_constants_are_those_of_the_fit() -> None:
    """Every constant of the table, its cap and its worst misfit, as fitted."""
    assert set(cost._FITS) == set(TABLE)
    for statistic, (coefficient, exponent, offset, cap_kind) in TABLE.items():
        fit = cost._FITS[statistic]
        assert fit.coefficient == coefficient, statistic
        assert fit.exponent == exponent, statistic
        assert fit.dimension_offset == offset, statistic
        assert fit.cap == cap_kind, statistic
        assert fit.misfit == MISFITS[statistic], statistic
    assert cost.COST_BUDGET is _config.COST_BUDGET
    assert COST_BUDGET == 3_600.0


def test_one_energy_evaluation_follows_the_model() -> None:
    """The observed evaluation and the draws of one statistic, from the formula."""
    estimate = compare_cost(1000, ["energy"])
    (row,) = rows(estimate, "energy")
    expected = (1 + DRAWS) * model("energy", 1000, 2)
    assert row["seconds"] == pytest.approx(expected, rel=1e-12)
    assert row["evaluations"] == 1 + DRAWS
    assert row["samples"] == 1000
    assert row["dimension"] == 2
    assert estimate.seconds == pytest.approx(expected, rel=1e-12)


def test_the_seconds_grow_with_the_draws_as_one_plus_the_draws() -> None:
    """The observed evaluation and each draw cost the same."""
    few = compare_cost(500, ["energy"], draws=4)
    many = compare_cost(500, ["energy"], draws=9)
    assert many.seconds / few.seconds == pytest.approx(10 / 5, rel=1e-12)


@pytest.mark.parametrize(
    ("recipe", "statistic", "exponent"),
    [("js:knn", "knn_js", 2.06), ("js:kde", "kde", 1.01), ("w1", "marginal", 1.13)],
)
def test_the_seconds_scale_with_the_samples_by_the_exponent_of_the_statistic(
    recipe: str, statistic: str, exponent: float
) -> None:
    """Quadratic for the kNN rows, linear for the kernel density estimates."""
    small = compare_cost(1000, [recipe])
    large = compare_cost(4000, [recipe])
    assert large.seconds / small.seconds == pytest.approx(4**exponent, rel=1e-9)
    assert exponent == cost._FITS[statistic].exponent


def test_a_cap_stops_the_growth_of_the_capped_statistics() -> None:
    """Above the cap the samples, and so the seconds, stay where they are."""
    small = compare_cost(1000, ["mmd"], cap=500)
    large = compare_cost(3000, ["mmd"], cap=500)
    assert small.seconds == pytest.approx(large.seconds, rel=1e-12)
    assert rows(large, "mmd")[0]["samples"] == 500
    uncapped = compare_cost(3000, ["mmd"], cap=CAP)
    assert rows(uncapped, "mmd")[0]["seconds"] > 10 * rows(large, "mmd")[0]["seconds"]


def test_pairs_and_problems_multiply_the_cost() -> None:
    """Mode B has one pair per two posteriors; each problem repeats the pairs."""
    one = estimate_compare_cost(
        shift_set(300), ["a", "b"], SPACE, metrics=["energy"], draws=DRAWS
    )
    three = estimate_compare_cost(
        shift_set(300, posteriors=3),
        list("abc"),
        SPACE,
        metrics=["energy"],
        draws=DRAWS,
    )
    assert three.seconds == pytest.approx(3 * one.seconds, rel=1e-12)
    assert three.table["pair"].to_list() == [["a", "b"], ["a", "c"], ["b", "c"]]
    reference = estimate_compare_cost(
        shift_set(300, posteriors=3),
        list("abc"),
        SPACE,
        reference="b",
        metrics=["energy"],
        draws=DRAWS,
    )
    assert reference.table["pair"].to_list() == [["b", "a"], ["b", "c"]]
    problems = estimate_compare_cost(
        shift_set(300, batch=4), ["a", "b"], SPACE, metrics=["energy"], draws=DRAWS
    )
    assert problems.seconds == pytest.approx(4 * one.seconds, rel=1e-12)
    assert problems.table["problem"].to_list() == [0, 1, 2, 3]


def test_the_bootstrap_adds_half_sample_evaluations_at_half_the_samples() -> None:
    """Each half-sample pair costs the model at ⌈N/2⌉ samples, 2^(−a) of the full."""
    plain = compare_cost(2000, ["energy"])
    boot = compare_cost(
        2000, ["energy"], uncertainty="subsampling_bootstrap", half_samples=7
    )
    extra = boot.seconds - plain.seconds
    assert extra == pytest.approx(7 * model("energy", 1000, 2), rel=1e-12)
    assert extra == pytest.approx(
        7 * 2 ** (-1.95) * model("energy", 2000, 2), rel=1e-12
    )
    assert rows(boot, "energy")[0]["evaluations"] == 1 + DRAWS + 7


def test_half_samples_above_the_cap_cost_as_much_as_the_full_evaluation() -> None:
    """A capped statistic subsamples halves above its cap, so they are not cheaper."""
    plain = compare_cost(4000, ["mmd"], cap=500)
    boot = compare_cost(
        4000, ["mmd"], cap=500, uncertainty="subsampling_bootstrap", half_samples=3
    )
    assert boot.seconds - plain.seconds == pytest.approx(
        3 * model("mmd", 500, 2), rel=1e-12
    )


def test_a_statistic_that_serves_several_recipes_counts_once() -> None:
    """KS, AD and W1 share one call; the Gaussian terms share one fit."""
    marginal = compare_cost(800, ["ks", "ad", "w1"])
    (row,) = rows(marginal, "ks/ad/w1")
    assert row["estimator"] == "empirical"
    assert marginal.seconds == pytest.approx((1 + DRAWS) * model("marginal", 800, 2, 2))
    gaussian = compare_cost(800, ["kl:gaussian", "w2", "hellinger:gaussian"])
    assert len(gaussian.table) == 1
    assert rows(gaussian, "kl/w2/hellinger")[0]["estimator"] == "gaussian"


def test_the_default_selection_is_the_fast_set() -> None:
    """The default metrics, with the median bandwidth; the slow rows run on request."""
    default = compare_cost(1000)
    metrics = default.table["metric"].to_list()
    assert metrics == [
        "mean_shift",
        "width_ratio",
        "ks/ad/w1",
        "js/hellinger",
        "kl/w2/hellinger",
        "mmd",
        "energy",
        "sliced_w2",
        "sliced_w2",
        "median_bandwidth",
    ]
    only = compare_cost(1000, ["mean_shift"])
    assert only.table["metric"].to_list() == ["mean_shift"]
    assert default.seconds > only.seconds
    assert rows(only, "mean_shift")[0]["seconds"] == pytest.approx(
        (1 + DRAWS) * model("moments", 1000, 2, 2)
    )
    median = rows(default, "median_bandwidth")[0]
    assert median["seconds"] == pytest.approx(model("median_bandwidth", 1000, 2))
    assert median["evaluations"] == 1


def test_the_slow_rows_run_on_request_with_their_extra_evaluations() -> None:
    """KNN rows, the Sinkhorn divergence per ε and the sliced Sinkhorn divergence."""
    estimate = compare_cost(
        400,
        ["kl:knn", "sinkhorn", "sliced_sinkhorn"],
        sinkhorn_epsilon_factors=(0.1, 0.25, 0.5),
    )
    kl = rows(estimate, "kl", estimator="knn")[0]
    assert kl["evaluations"] == 2 * (1 + DRAWS)
    assert kl["seconds"] == pytest.approx(2 * (1 + DRAWS) * model("knn_kl", 400, 2))
    curve = rows(estimate, "kl_calibration")[0]
    assert curve["evaluations"] == 2 * (6 + 1) * 4
    assert curve["seconds"] == pytest.approx(56 * model("knn_kl", 400, 2))
    sinkhorn = rows(estimate, "sinkhorn")[0]
    assert sinkhorn["evaluations"] == 3 * (1 + DRAWS)
    assert sinkhorn["seconds"] == pytest.approx(
        3 * (1 + DRAWS) * model("sinkhorn", 400, 2)
    )
    sliced = rows(estimate, "sliced_sinkhorn")[0]
    assert sliced["seconds"] == pytest.approx(
        (1 + DRAWS) * model("sliced_sinkhorn", 400, 2, 20)
    )
    epsilon = rows(estimate, "sliced_sinkhorn_epsilon")[0]
    assert epsilon["seconds"] == pytest.approx(
        model("sliced_sinkhorn_epsilon", 400, 2, 20)
    )
    assert len(rows(estimate, "median_bandwidth")) == 1


def test_the_sliced_directions_fill_whole_bases_of_the_embedded_space() -> None:
    """With a circle and a sphere the bases are E + 2C + 3S rows long."""
    case = zoo.von_mises_fisher(0.5, euclidean_count=1)
    samples = case.sample(jax.random.key(1), 100)
    estimate = estimate_compare_cost(
        samples, ["a", "b"], case.space, metrics=["sliced_sinkhorn"], draws=DRAWS
    )
    embedding = 1 + 3
    directions = math.ceil(20 / embedding) * embedding
    (row,) = rows(estimate, "sliced_sinkhorn")
    dimension = len(case.space)
    assert row["dimension"] == dimension
    assert row["seconds"] == pytest.approx(
        (1 + DRAWS) * model("sliced_sinkhorn", 100, dimension, directions)
    )


def test_the_knn_rows_run_on_the_positive_counts_of_the_smaller_set() -> None:
    """Zero weights leave the count; the larger set does not set the size."""
    case = zoo.shifted_gaussians(0.5, param_count=2)
    base = case.sample(jax.random.key(2), 600)
    weights = jnp.where(jnp.arange(600) % 2 == 0, -jnp.inf, 0.0)
    thinned = SampleSet(
        base.samples, log_weights=jnp.stack([jnp.zeros(600), weights])[None]
    )
    estimate = estimate_compare_cost(
        thinned, ["a", "b"], case.space, metrics=["hellinger:knn"], draws=DRAWS
    )
    (row,) = rows(estimate, "hellinger")
    assert row["samples"] == 300
    assert row["seconds"] == pytest.approx((1 + DRAWS) * model("knn_hellinger", 300, 2))


def test_the_table_has_the_columns_and_the_total_of_its_seconds() -> None:
    """Names, types, the engine and the sum of the seconds."""
    estimate = compare_cost(300)
    assert isinstance(estimate, CostEstimate)
    assert estimate.table.schema == pl.Schema(
        {
            "engine": pl.String(),
            "problem": pl.Int64(),
            "pair": pl.List(pl.String()),
            "metric": pl.String(),
            "estimator": pl.String(),
            "samples": pl.Int64(),
            "dimension": pl.Int64(),
            "evaluations": pl.Int64(),
            "seconds": pl.Float64(),
            "space": pl.String(),
            "memory": pl.Int64(),
        }
    )
    # * The columns of data space are null in the rows of compare.
    assert estimate.table["space"].null_count() == estimate.table.height
    assert estimate.table["memory"].null_count() == estimate.table.height
    assert set(estimate.table["engine"]) == {"compare"}
    assert estimate.seconds == pytest.approx(float(estimate.table["seconds"].sum()))
    assert all(seconds > 0 for seconds in estimate.table["seconds"])


def test_a_transform_changes_no_count_and_a_fisher_whitening_needs_no_matrix() -> None:
    """A transformed or Fisher-whitened comparison costs what the plain one does."""
    samples = shift_set(300)
    transform = Whitening.from_samples(samples.samples[:, 0], None, SPACE)
    plain = estimate_compare_cost(samples, ["a", "b"], SPACE, draws=DRAWS)
    mapped = estimate_compare_cost(
        samples, ["a", "b"], SPACE, draws=DRAWS, transform=transform
    )
    fisher = estimate_compare_cost(
        samples, ["a", "b"], SPACE, draws=DRAWS, whitening="fisher"
    )
    assert mapped.seconds == pytest.approx(plain.seconds, rel=1e-12)
    assert fisher.seconds == pytest.approx(plain.seconds, rel=1e-12)


def test_the_compare_estimate_shares_the_argument_errors_of_the_engine() -> None:
    """The same checks, from the same code, before any estimate."""
    samples = shift_set(60)
    names = ["a", "b"]
    with pytest.raises(ValueError, match="unknown metric 'nonsense'"):
        estimate_compare_cost(samples, names, SPACE, metrics=["nonsense"])
    with pytest.raises(ValueError, match="is a term of 'kl'"):
        estimate_compare_cost(samples, names, SPACE, metrics=["kl_shape"])
    with pytest.raises(NotImplementedError, match="classifier"):
        estimate_compare_cost(samples, names, SPACE, metrics=["c2st"])
    with pytest.raises(ValueError, match="reference must be one of"):
        estimate_compare_cost(samples, names, SPACE, reference="c")
    with pytest.raises(ValueError, match="reference whitening needs reference"):
        estimate_compare_cost(samples, names, SPACE, whitening="reference")
    with pytest.raises(ValueError, match="names must be unique"):
        estimate_compare_cost(samples, ["a", "a"], SPACE)
    with pytest.raises(ValueError, match="draws must be at least 1"):
        estimate_compare_cost(samples, names, SPACE, draws=0)
    with pytest.raises(ValueError, match="cap must be at least 2"):
        estimate_compare_cost(samples, names, SPACE, cap=1)
    with pytest.raises(ValueError, match="delta method"):
        estimate_compare_cost(samples, names, SPACE, uncertainty="delta_method")
    with pytest.raises(ValueError, match="split_half null needs reference"):
        estimate_compare_cost(samples, names, SPACE, null="split_half")
    with pytest.raises(TypeError, match="draws must be an integer"):
        unchecked_compare_cost(samples, names, SPACE, draws=2.5)
    with pytest.raises(ValueError, match="distinct"):
        estimate_compare_cost(
            samples, names, SPACE, sinkhorn_epsilon_factors=(0.2, 0.2)
        )


# * The estimate of the diagnostics


def diagnose_cost(
    count: int = 400, space: ParamSpace = SPACE, **options: Any
) -> CostEstimate:
    """Estimate the diagnostics of one posterior of Gaussian samples."""
    samples = zoo.shifted_gaussians(0.5, param_count=len(space)).sample(
        jax.random.key(3), count
    )
    return estimate_diagnose_cost(
        samples.select(posteriors=[0]),
        ["p"],
        space,
        draws=options.pop("draws", DRAWS),
        **options,
    )


def test_the_diagnostics_give_one_group_per_statistic_with_the_surrogate_sizes() -> (
    None
):
    """Shape, dependence and mode statistics: observed at N, draws at their size."""
    estimate = diagnose_cost(400, cap=150, pair_cap=250)
    assert estimate.table["metric"].to_list() == [
        "marginal_moments",
        "anderson_darling",
        "mardia",
        "dependence",
        "gmm_bic",
        "marginal_modes",
        "projection_modes",
    ]
    caps = {"cap": 150, "pair_cap": 250}
    shape_size = surrogate_size(400.0, 150)
    pair_size = surrogate_size(400.0, 250)
    (moments,) = rows(estimate, "marginal_moments")
    assert moments["samples"] == shape_size == 150
    assert moments["evaluations"] == 1 + DRAWS
    assert moments["seconds"] == pytest.approx(
        model("marginal_moments", 400, 2, 2)
        + DRAWS * model("marginal_moments", 150, 2, 2)
    )
    (mardia,) = rows(estimate, "mardia")
    assert mardia["seconds"] == pytest.approx(
        (1 + DRAWS) * model("mardia", 150, 2, caps=caps)
    )
    (dependence,) = rows(estimate, "dependence")
    assert dependence["samples"] == pair_size == 250
    assert dependence["seconds"] == pytest.approx(
        model("dependence", 250, 2, 1, caps=caps)
        + DRAWS * model("dependence", 250, 2, 1, caps=caps)
    )
    (bic,) = rows(estimate, "gmm_bic")
    assert bic["evaluations"] == 1
    assert bic["seconds"] == pytest.approx(model("gmm_bic", 400, 2))
    (projections,) = rows(estimate, "projection_modes")
    assert projections["seconds"] == pytest.approx(
        model("projection_modes", 400, 2, 10)
        + DRAWS * model("projection_modes", 150, 2, 10)
    )
    (modes,) = rows(estimate, "marginal_modes")
    assert modes["seconds"] == pytest.approx(
        model("marginal_modes", 400, 2, 2) + DRAWS * model("marginal_modes", 150, 2, 2)
    )
    assert set(estimate.table["engine"]) == {"diagnose"}
    assert estimate.table["pair"].to_list() == [["p"]] * 7


def test_one_parameter_has_no_dependence_group_and_families_are_selectable() -> None:
    """The family names select groups; chains and railing are not modelled."""
    one = ParamSpace.model_validate({"parameters": [{"name": "x0"}]})
    estimate = diagnose_cost(200, one)
    assert "dependence" not in estimate.table["metric"].to_list()
    shape = diagnose_cost(200, diagnostics=["shape"])
    assert shape.table["metric"].to_list() == [
        "marginal_moments",
        "anderson_darling",
        "mardia",
    ]
    nothing = diagnose_cost(200, diagnostics=["chains", "railing"])
    assert len(nothing.table) == 0
    assert nothing.seconds == 0.0
    assert nothing.table.schema == cost._TABLE_SCHEMA


def test_a_weighted_set_enters_with_its_effective_sample_size() -> None:
    """The surrogates have the Kish size of the weights, not the number of samples."""
    case = zoo.shifted_gaussians(0.5, param_count=2)
    base = case.sample(jax.random.key(4), 500).samples[:, 0]
    weights = 0.8 * jax.random.normal(jax.random.key(5), (1, 1, 500))
    weighted = SampleSet(base, log_weights=weights[0])
    n_eff = float(weighted.effective_sample_size()[0, 0])
    estimate = estimate_diagnose_cost(
        weighted, ["w"], SPACE, draws=DRAWS, diagnostics=["shape"]
    )
    (moments,) = rows(estimate, "marginal_moments")
    assert moments["samples"] == surrogate_size(n_eff, _config.SAMPLE_CAP) < 500


def test_chains_enter_before_their_reduction_with_the_upper_bound() -> None:
    """N = (steps − burn-in) · walkers and n_eff = N: the cost of the flat set of N."""
    chain_case = zoo.ar1_chains([0.9, 0.0])
    chains = chain_case.sample(jax.random.key(6), 400, walker_count=8)
    estimate = estimate_diagnose_cost(
        chains, ["run"], chain_case.space, burn_in=50, draws=DRAWS
    )
    count = (400 - 50) * 8
    flat = SampleSet(jnp.zeros((count, 2)).at[:, 0].set(jnp.arange(count) / count))
    reference = estimate_diagnose_cost(flat, ["run"], chain_case.space, draws=DRAWS)
    assert estimate.seconds == pytest.approx(reference.seconds, rel=1e-12)
    assert rows(estimate, "marginal_moments")[0]["samples"] == count
    shape_only = estimate_diagnose_cost(
        chains,
        ["run"],
        chain_case.space,
        burn_in=[350],
        draws=DRAWS,
        diagnostics=["shape"],
    )
    assert rows(shape_only, "marginal_moments")[0]["samples"] == 50 * 8


def test_an_integer_thinning_enters_with_the_samples_that_the_reduction_keeps() -> None:
    """A factor k keeps ceil((steps − burn-in)/k) steps per walker, with n_eff = N."""
    chain_case = zoo.ar1_chains([0.9, 0.0])
    chains = chain_case.sample(jax.random.key(6), 400, walker_count=8)
    estimate = estimate_diagnose_cost(
        chains, ["run"], chain_case.space, burn_in=50, thin=8, draws=DRAWS
    )
    # * 350 steps in factors of 8 leave 44 steps, a count that no division by 8 gives.
    count = math.ceil((400 - 50) / 8) * 8
    assert count == 352
    with errors.capture_warnings():
        reduction = reduce_chains(chains, burn_in=50, thin=8)
    assert reduction.sample_set.shape.sample == count
    assert rows(estimate, "marginal_moments")[0]["samples"] == count
    flat = SampleSet(jnp.zeros((count, 2)).at[:, 0].set(jnp.arange(count) / count))
    reference = estimate_diagnose_cost(flat, ["run"], chain_case.space, draws=DRAWS)
    assert estimate.seconds == pytest.approx(reference.seconds, rel=1e-12)
    # * The default and "tau" keep the upper bound, whatever the autocorrelation time.
    for thin in ("tau", ["tau"]):
        bound = estimate_diagnose_cost(
            chains, ["run"], chain_case.space, burn_in=50, thin=thin, draws=DRAWS
        )
        assert rows(bound, "marginal_moments")[0]["samples"] == (400 - 50) * 8
    one_per_run = estimate_diagnose_cost(
        chains, ["run"], chain_case.space, burn_in=50, thin=[8], draws=DRAWS
    )
    assert one_per_run.seconds == estimate.seconds


def test_the_diagnose_estimate_shares_the_argument_errors_of_the_engine() -> None:
    """The same checks, from the same code, before any estimate."""
    samples = shift_set(60).select(posteriors=[0])
    with pytest.raises(ValueError, match="unknown diagnostic families"):
        estimate_diagnose_cost(samples, ["a"], SPACE, diagnostics=["shapes"])
    with pytest.raises(ValueError, match="names must name each of the 1 posteriors"):
        estimate_diagnose_cost(samples, ["a", "b"], SPACE)
    with pytest.raises(ValueError, match="the space has 3 parameters"):
        estimate_diagnose_cost(
            samples, ["a"], zoo.shifted_gaussians(1.0, param_count=3).space
        )
    with pytest.raises(ValueError, match="burn_in must be at least 0"):
        estimate_diagnose_cost(samples, ["a"], SPACE, burn_in=-1)
    with pytest.raises(ValueError, match="one value for every Chains input"):
        estimate_diagnose_cost(samples, ["a"], SPACE, burn_in=[1, 2])
    with pytest.raises(ValueError, match="thin must be at least 1"):
        estimate_diagnose_cost(samples, ["a"], SPACE, thin=0)
    with pytest.raises(ValueError, match="thin must be 'tau' or an integer"):
        estimate_diagnose_cost(samples, ["a"], SPACE, thin="x")
    with pytest.raises(ValueError, match="one value for every Chains input"):
        estimate_diagnose_cost(samples, ["a"], SPACE, thin=[1, 2])
    with pytest.raises(ValueError, match="cap must be at least 2"):
        estimate_diagnose_cost(samples, ["a"], SPACE, cap=1)
    with pytest.raises(ValueError, match="pair_cap must be at least 2"):
        estimate_diagnose_cost(samples, ["a"], SPACE, pair_cap=1)
    with pytest.raises(TypeError, match="sources must be Chains"):
        unchecked_diagnose_cost(jnp.zeros((5, 2)), ["a"], SPACE)


# * The warning


def test_a_budget_below_the_estimate_raises_one_warning_naming_the_costliest() -> None:
    """The message names the estimate, the budget and the three costliest statistics."""
    case = shift_set(300)
    metrics = ["energy", "mmd", "kl:gaussian", "mean_shift"]
    estimate = estimate_compare_cost(
        case, ["a", "b"], SPACE, reference="a", metrics=metrics, draws=DRAWS
    )
    ranked = (
        estimate.table.group_by(["metric", "estimator"])
        .agg(pl.col("seconds").sum())
        .sort("seconds", descending=True)
    )
    top = ranked["metric"].to_list()[:3]
    with pytest.warns(CostWarning) as records:
        result = compare(
            case,
            ["a", "b"],
            SPACE,
            key=1,
            reference="a",
            metrics=metrics,
            draws=DRAWS,
            budget=0.001,
        )
    assert len(records) == 1
    message = str(records[0].message)
    assert "the compare run is estimated at" in message
    assert "above the budget of 0.001 s" in message
    positions = [message.index(f"{metric} (") for metric in top]
    assert positions == sorted(positions)
    assert message.count(" (") == 3
    assert all(not row["warnings"] for row in result.table.to_dicts())


def test_a_budget_above_the_estimate_or_infinite_raises_none() -> None:
    """The default hour, a generous budget and infinity stay silent."""
    samples = shift_set(100)
    for budget in (COST_BUDGET, 1e9, math.inf):
        result = compare(
            samples,
            ["a", "b"],
            SPACE,
            key=1,
            draws=DRAWS,
            metrics=["mean_shift"],
            budget=budget,
        )
        assert len(result) == 2


def test_diagnose_warns_before_it_starts_when_the_budget_is_small() -> None:
    """The diagnostics call the same estimate and warn once."""
    samples = shift_set(150).select(posteriors=[0])
    with pytest.warns(CostWarning, match="the diagnose run is estimated at") as records:
        diagnosis = diagnose(
            samples,
            ["p"],
            SPACE,
            key=1,
            draws=DRAWS,
            diagnostics=["shape"],
            budget=1e-4,
        )
    assert len(records) == 1
    assert "marginal_moments (empirical)" in str(records[0].message)
    assert len(diagnosis.result) > 0


def test_the_warning_function_checks_the_budget_and_writes_python_floats() -> None:
    """Equal estimate and budget stay silent; large seconds read as whole numbers."""
    table = pl.DataFrame(
        [
            {
                "engine": "compare",
                "problem": 0,
                "pair": ["a", "b"],
                "metric": name,
                "estimator": "empirical",
                "samples": 10,
                "dimension": 2,
                "evaluations": 2,
                "seconds": seconds,
            }
            for name, seconds in (
                ("a", 6000.0),
                ("b", 3000.0),
                ("c", 1500.0),
                ("d", 5.0),
            )
        ],
        schema=cost._TABLE_SCHEMA,
    )
    estimate = CostEstimate(table, 10_505.0)
    warn_if_over_budget("compare", estimate, 10_505.0)
    warn_if_over_budget("compare", estimate, math.inf)
    with pytest.warns(CostWarning) as records:
        warn_if_over_budget("compare", estimate, 3_600.0)
    message = str(records[0].message)
    assert "estimated at 10,505 s" in message
    assert "budget of 3,600 s" in message
    assert message.endswith(
        "a (empirical) 6,000 s, b (empirical) 3,000 s, c (empirical) 1,500 s"
    )
    assert "d (empirical)" not in message
    with pytest.warns(CostWarning, match="budget of 2.5 s"):
        warn_if_over_budget("diagnose", CostEstimate(table, 7.25), 2.5)


def test_an_empty_estimate_never_warns() -> None:
    """No groups cost nothing, so no budget is exceeded."""
    estimate = diagnose_cost(100, diagnostics=["chains"])
    warn_if_over_budget("diagnose", estimate, 0.0)


# * The data-space estimate


class SignalStub:
    """A generator with a fixed number of data entries, and no Fisher matrix."""

    def __init__(self, data_size: int) -> None:
        """Store the number of data entries."""
        self.data_size = data_size
        self.sizes: list[int] = []

    def generate_template(self, params: Any) -> Any:
        """Return ones shaped (n, M) and record the number of rows."""
        self.sizes.append(len(params))
        return np.ones((len(params), self.data_size))


class WhiteStub:
    """A noise model that returns its signals, with a description."""

    description = "stub"

    def whiten(self, signals: Any) -> Any:
        """Return the signals."""
        return signals


class FisherSignalStub(SignalStub):
    """A stub generator with a Fisher matrix."""

    def information_matrix(self, params: Any, *, noise: Any) -> Any:
        """Return the identity of the parameters, once per row."""
        return np.broadcast_to(np.eye(2), (len(params), 2, 2))


def signals_cost(
    count: int = 300,
    data_size: int = 24,
    posteriors: int = 2,
    batch: int = 1,
    **options: Any,
) -> CostEstimate:
    """Estimate a data-space comparison of Gaussian samples with a stub generator."""
    samples = shift_set(count, batch=batch, posteriors=3 if posteriors == 3 else 2)
    names = ["a", "b", "c"][: samples.shape.posterior]
    return estimate_compare_signals_cost(
        samples,
        names,
        SPACE,
        generator=options.pop("generator", SignalStub(data_size)),
        noise=options.pop("noise", WhiteStub()),
        draws=options.pop("draws", DRAWS),
        **options,
    )


def test_the_signal_estimate_has_generator_rows_and_energy_rows() -> None:
    """Per posterior a generator row without seconds, per pair a data row."""
    estimate = signals_cost(300, 24)
    table = estimate.table
    assert set(table["engine"]) == {"compare_signals"}
    generator_rows = rows(estimate, "generate_template")
    assert [row["pair"] for row in generator_rows] == [["a"], ["b"]]
    for row in generator_rows:
        assert row["estimator"] == "user"
        assert row["samples"] == 300
        assert row["dimension"] == 24
        assert row["evaluations"] == 1
        assert row["seconds"] is None
        assert row["space"] is None
        assert row["memory"] == 8 * 300 * 24
    (energy,) = rows(estimate, "energy")
    assert energy["estimator"] == "empirical"
    assert energy["pair"] == ["a", "b"]
    assert energy["space"] == "data(stub)"
    assert energy["samples"] == 300
    assert energy["dimension"] == 24
    assert energy["evaluations"] == 1 + DRAWS
    assert energy["memory"] == 8 * 600**2
    expected = model("signal_matrix", 300, 24) + (1 + DRAWS) * model(
        "signal_draw", 300, 0
    )
    assert energy["seconds"] == pytest.approx(expected, rel=1e-12)
    # * The total sums the seconds that exist, and the user code has none.
    assert estimate.seconds == pytest.approx(expected, rel=1e-12)


def test_the_fisher_row_costs_the_same_matrix_in_the_dimension_of_the_chart() -> None:
    """With a centre there is a chart row with D in place of M; without, none."""
    plain = signals_cost(200, 24)
    assert len(rows(plain, "energy")) == 1
    with_centre = signals_cost(
        200, 24, centre=np.zeros(2), fisher=np.eye(2), draws=DRAWS
    )
    data, chart = rows(with_centre, "energy")
    assert data["space"] == "data(stub)"
    assert chart["space"] == "chart(fisher)"
    assert chart["dimension"] == 2
    assert chart["memory"] == data["memory"] == 8 * 400**2
    expected = model("signal_matrix", 200, 2) + (1 + DRAWS) * model(
        "signal_draw", 200, 0
    )
    assert chart["seconds"] == pytest.approx(expected, rel=1e-12)
    assert chart["seconds"] < data["seconds"]
    only_centre = signals_cost(
        200, 24, centre=np.zeros(2), generator=FisherSignalStub(24)
    )
    assert len(rows(only_centre, "energy")) == 2


def test_the_estimate_counts_the_cap_the_zero_weights_and_the_calls() -> None:
    """Capped positive counts, one generator call per chunk, and the half-samples."""
    samples = shift_set(400)
    log_weights = np.zeros((1, 2, 400))
    log_weights[0, 1, :150] = -np.inf
    weighted = SampleSet(samples.samples, log_weights=log_weights)
    estimate = estimate_compare_signals_cost(
        weighted,
        ["a", "b"],
        SPACE,
        generator=SignalStub(8),
        noise=WhiteStub(),
        draws=DRAWS,
        cap=300,
        chunk_size=100,
        uncertainty="subsampling_bootstrap",
        half_samples=7,
    )
    first, second = rows(estimate, "generate_template")
    assert (first["samples"], second["samples"]) == (300, 250)
    assert (first["evaluations"], second["evaluations"]) == (3, 3)
    assert second["memory"] == 8 * 250 * 8
    (energy,) = rows(estimate, "energy")
    assert energy["samples"] == 300
    assert energy["evaluations"] == 1 + DRAWS + 7
    assert energy["memory"] == 8 * 550**2


def test_the_estimate_probes_one_sample_per_problem_for_the_data_size() -> None:
    """Each problem's generator is called once on one positive-weight sample."""
    samples = shift_set(100, batch=2)
    first, second = SignalStub(5), SignalStub(9)
    estimate = estimate_compare_signals_cost(
        samples,
        ["a", "b"],
        SPACE,
        generator=[first, second],
        noise=[WhiteStub(), WhiteStub()],
        draws=DRAWS,
    )
    assert first.sizes == second.sizes == [1]
    assert [row["dimension"] for row in rows(estimate, "energy")] == [5, 9]
    assert [row["problem"] for row in rows(estimate, "energy")] == [0, 1]
    # * The probe is the first sample with a positive weight.
    log_weights = np.zeros((1, 2, 100))
    log_weights[0, 0, :5] = -np.inf
    probe = SignalStub(4)
    estimate_compare_signals_cost(
        SampleSet(shift_set(100).samples, log_weights=log_weights),
        ["a", "b"],
        SPACE,
        generator=probe,
        noise=WhiteStub(),
        draws=DRAWS,
    )
    assert probe.sizes[0] == 1


def test_the_signal_estimate_shares_the_argument_errors_of_the_engine() -> None:
    """The same checks as compare_signals, from the same code, before any estimate.

    The test session checks the protocol annotations first, so the calls take
    the unwrapped function, as runs without runtime type checks do.
    """
    samples = shift_set(60)
    options: dict[str, Any] = {"generator": SignalStub(4), "noise": WhiteStub()}
    with pytest.raises(CapabilityError, match="needs a noise model with whiten"):
        unchecked_signals_cost(
            samples, ["a", "b"], SPACE, generator=SignalStub(4), noise=SignalStub(4)
        )
    with pytest.raises(
        CapabilityError, match="needs a generator with generate_template"
    ):
        unchecked_signals_cost(
            samples, ["a", "b"], SPACE, generator=WhiteStub(), noise=WhiteStub()
        )
    with pytest.raises(CapabilityError, match="information_matrix"):
        unchecked_signals_cost(
            samples, ["a", "b"], SPACE, centre=np.zeros(2), **options
        )
    with pytest.raises(ValueError, match="fisher needs centre="):
        unchecked_signals_cost(samples, ["a", "b"], SPACE, fisher=np.eye(2), **options)
    with pytest.raises(ValueError, match="draws must be at least 1"):
        unchecked_signals_cost(samples, ["a", "b"], SPACE, draws=0, **options)
    with pytest.raises(ValueError, match="the split_half null needs reference"):
        unchecked_signals_cost(samples, ["a", "b"], SPACE, null="split_half", **options)
    with pytest.raises(ValueError, match="chunk_size must be at least 1"):
        unchecked_signals_cost(samples, ["a", "b"], SPACE, chunk_size=0, **options)


def test_a_tiny_budget_warns_once_for_the_signal_estimate() -> None:
    """The warning names the costliest statistic that has seconds."""
    estimate = signals_cost(300, 24)
    with pytest.warns(
        CostWarning, match="the compare_signals run is estimated at"
    ) as seen:
        warn_if_over_budget("compare_signals", estimate, 1e-9)
    assert len(seen) == 1
    assert "energy (empirical)" in str(seen[0].message)
    # * User code has no estimate, so the warning does not name it.
    assert "generate_template" not in str(seen[0].message)


def test_the_estimates_of_compare_and_diagnose_keep_their_rows_with_null_columns() -> (
    None
):
    """The two new columns are null in every row of the older estimates."""
    for estimate in (compare_cost(300), diagnose_cost(300)):
        assert estimate.table.height > 0
        assert estimate.table["space"].null_count() == estimate.table.height
        assert estimate.table["memory"].null_count() == estimate.table.height
        assert estimate.table["seconds"].null_count() == 0


class SingleNoise:
    """A noise model that returns float32 vectors, which warns about precision."""

    def whiten(self, signals: Any) -> Any:
        """Return the signals as float32 vectors."""
        return np.asarray(signals, dtype=np.float32)


def test_the_probe_of_the_estimate_raises_none_of_the_template_warnings() -> None:
    """A float32 whitener warns in the rows of the engine, not in the estimate."""
    estimate = signals_cost(100, 6, noise=SingleNoise())
    assert len(rows(estimate, "energy")) == 1
