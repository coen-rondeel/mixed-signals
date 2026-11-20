"""Tests for ``diagnose``: chains, flat sets, families, surrogates, rows and checks."""

import inspect
import json
import math
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import stats

from mixed_signals import errors, prng, zoo
from mixed_signals.core.chains import Chains
from mixed_signals.core.result import RESULT_SCHEMA, Result, hash_arrays
from mixed_signals.core.samples import SampleSet, rejection_subset
from mixed_signals.core.space import ParamSpace
from mixed_signals.diagnostics import flags
from mixed_signals.diagnostics.chains import chain_health
from mixed_signals.diagnostics.dependence import (
    MUTUAL_INFORMATION_NEIGHBOURS,
    dependence_statistics,
)
from mixed_signals.diagnostics.flags import Side, Threshold
from mixed_signals.diagnostics.shape import marginal_moments
from mixed_signals.engines.diagnose import DIAGNOSTIC_FAMILIES, Diagnosis, diagnose
from mixed_signals.mcmc.reduce import hash_chains, reduce_chains
from mixed_signals.nulls.surrogates import gaussian_surrogates, surrogate_size
from mixed_signals.sigma.significance import Tail
from mixed_signals.transforms.whiten import Whitening

unchecked_diagnose = inspect.unwrap(diagnose)
KEY = 2027
DRAWS = 20
# * 60 draws resolve σ_sig up to 2.06, so a row with a null of ties can still flag.
FLAG_DRAWS = 60
FLAG_SIZE = 600
STEPS = 400
WALKERS = 8


# * Helpers


def table_rows(
    result: Result, metric: str | None = None, **selection: Any
) -> list[dict[str, Any]]:
    """Return the rows of a result that match a metric and other columns, in order."""
    rows = []
    for row in result.table.to_dicts():
        if metric is not None and row["metric"] != metric:
            continue
        if all(row[column] == value for column, value in selection.items()):
            rows.append(row)
    return rows


def categories(row: dict[str, Any]) -> list[str]:
    """Return the categories of the warnings of a row, in order."""
    return [warning["category"] for warning in row["warnings"]]


def ar1_chains(
    coefficients: tuple[float, ...] = (0.9, 0.0), steps: int = STEPS, seed: int = 0
) -> tuple[ParamSpace, Chains]:
    """Return AR(1) chains of the zoo with their space."""
    case = zoo.ar1_chains(coefficients)
    return case.space, case.sample(jax.random.key(seed), steps, walker_count=WALKERS)


def flat_case(
    case: zoo.ZooCase, posterior: str, count: int, seed: int
) -> tuple[ParamSpace, SampleSet]:
    """Return the samples of one posterior of a zoo case as a set of one posterior."""
    samples = case.sample(jax.random.key(seed), count)
    return case.space, samples.select(posteriors=[case.names.index(posterior)])


def quiet_health(chains: Chains, space: ParamSpace, burn_in: int) -> Any:
    """Return the chain health, with the coupled-walkers warning captured."""
    with errors.capture_warnings():
        return chain_health(chains, space, burn_in=burn_in)


def flagged(result: Result) -> set[tuple[str, str]]:
    """Return the (metric, parameters) of every row that flags."""
    return {
        (row["metric"], ",".join(row["parameters"]))
        for row in table_rows(result)
        if row["flag"]
    }


# * Chains


def test_chains_give_their_health_in_the_coordinates_of_the_chains() -> None:
    """Per parameter τ, ESS, the split-chain test and R̂, as the kernels give them."""
    space, chains = ar1_chains()
    diagnosis = diagnose(
        chains, ["run"], space, key=KEY, burn_in=50, diagnostics=["chains"]
    )
    assert isinstance(diagnosis, Diagnosis)
    health = quiet_health(chains, space, 50)
    rows = table_rows(diagnosis.result)
    assert [row["metric"] for row in rows] == [
        "tau",
        "ess",
        "split_chain_ks",
        "rhat",
        "tau",
        "ess",
        "split_chain_ks",
        "rhat",
    ]
    assert {row["space"] for row in rows} == {"identity"}
    assert {row["estimator"] for row in rows} == {"empirical"}
    assert {tuple(row["pair"]) for row in rows} == {("run",)}
    for position, name in enumerate(space.names):
        by_metric = {row["metric"]: row for row in rows if row["parameters"] == [name]}
        assert by_metric["tau"]["value"] == pytest.approx(float(health.tau[position]))
        assert by_metric["ess"]["value"] == pytest.approx(
            float(health.effective_sample_size[position])
        )
        assert by_metric["split_chain_ks"]["value"] == pytest.approx(
            float(health.split_statistic[position])
        )
        assert by_metric["split_chain_ks"]["p_value"] == pytest.approx(
            float(health.split_p_value[position])
        )
        assert by_metric["rhat"]["value"] == pytest.approx(float(health.rhat[position]))
        assert json.loads(by_metric["tau"]["estimator_settings"]) == {"burn_in": 50}
        assert json.loads(by_metric["split_chain_ks"]["estimator_settings"]) == {
            "burn_in": 50,
            "thin": int(health.split_thin[position]),
        }
        ess = by_metric["ess"]
        assert ess["threshold"] == 400.0
        assert ess["p_value"] is None
        assert ess["flag"] is (ess["value"] < 400.0)
        for informational in ("tau", "rhat"):
            assert by_metric[informational]["threshold"] is None
            assert by_metric[informational]["flag"] is None
        split = by_metric["split_chain_ks"]
        evaluation = flags.evaluate(
            "split_chain_ks", split["value"], None, p_value=split["p_value"]
        )
        assert split["flag"] is evaluation.flagged
        assert split["sigma_sig"] == pytest.approx(evaluation.sigma_sig)
    for row in rows:
        categories = {warning["category"] for warning in row["warnings"]}
        assert "CoupledChainsWarning" in categories


def test_tempered_chains_add_acceptance_and_swap_rows() -> None:
    """Acceptance per temperature and swaps per neighbouring pair, when counted."""
    rng = np.random.default_rng(3)
    samples = rng.normal(size=(200, 3, 4, 1, 2))
    chains = Chains(
        samples,
        betas=np.tile([1.0, 0.6, 0.3], (200, 1)),
        accepted=np.array([[10, 20, 30, 40], [5, 5, 5, 5], [1, 2, 3, 4]]),
        proposed=np.full((3, 4), 100),
        swaps_accepted=np.array([30, 40]),
        swaps_proposed=np.array([120, 100]),
    )
    space = ParamSpace.model_validate({"parameters": [{"name": "a"}, {"name": "b"}]})
    result = diagnose(
        chains, ["run"], space, key=KEY, burn_in=10, diagnostics=["chains"]
    ).result
    acceptance = table_rows(result, "acceptance_rate")
    assert [row["value"] for row in acceptance] == pytest.approx([0.25, 0.05, 0.025])
    assert [
        json.loads(row["estimator_settings"])["temperature"] for row in acceptance
    ] == [0, 1, 2]
    swaps = table_rows(result, "swap_rate")
    assert [row["value"] for row in swaps] == pytest.approx([0.25, 0.4])
    assert [json.loads(row["estimator_settings"])["temperatures"] for row in swaps] == [
        [0, 1],
        [1, 2],
    ]
    for row in acceptance + swaps:
        assert row["parameters"] == ["a", "b"]
        assert row["threshold"] is None
        assert row["space"] == "identity"


def test_each_run_is_reduced_after_its_own_burn_in() -> None:
    """The reduction of each Chains input is returned by name, with its record."""
    space, first = ar1_chains(seed=1)
    _, second = ar1_chains((0.5, 0.2), seed=2)
    diagnosis = diagnose(
        [first, second],
        ["slow", "fast"],
        space,
        key=KEY,
        burn_in=[20, 80],
        diagnostics=["chains"],
    )
    assert set(diagnosis.reductions) == {"slow", "fast"}
    for name, chains, burn_in in (("slow", first, 20), ("fast", second, 80)):
        reduction = diagnosis.reductions[name]
        with errors.capture_warnings():
            expected = reduce_chains(chains, burn_in=burn_in, thin="tau")
        assert reduction.record == expected.record
        np.testing.assert_array_equal(
            reduction.sample_set.samples, expected.sample_set.samples
        )
        rows = table_rows(diagnosis.result, "tau", pair=[name])
        assert {json.loads(row["estimator_settings"])["burn_in"] for row in rows} == {
            burn_in
        }
        for row in table_rows(diagnosis.result, pair=[name]):
            assert row["provenance"]["input_hashes"] == [hash_chains(chains)]
    shared = diagnose(
        [first, second],
        ["slow", "fast"],
        space,
        key=KEY,
        burn_in=30,
        diagnostics=["chains"],
    )
    assert {reduction.record.burn_in for reduction in shared.reductions.values()} == {
        30
    }


def test_chains_feed_the_other_families_through_their_reduced_set() -> None:
    """Shape rows use the reduced samples and n_eff, and hash the chains."""
    space, chains = ar1_chains(seed=3)
    diagnosis = diagnose(
        chains, ["run"], space, key=KEY, burn_in=40, draws=DRAWS, diagnostics=["shape"]
    )
    reduction = diagnosis.reductions["run"]
    moments = marginal_moments(reduction.sample_set.samples[0, 0], None, space)
    skewness = table_rows(diagnosis.result, "skewness", parameters=["x0"])[0]
    assert skewness["value"] == pytest.approx(float(moments.skewness[0]), rel=1e-12)
    assert skewness["null_recipe"]["sample_size"] == surrogate_size(
        reduction.record.n_eff
    )
    assert skewness["provenance"]["input_hashes"] == [hash_chains(chains)]
    assert skewness["space"] == "identity"


# * Flat sets, mixed inputs and the checks of the inputs


def test_flat_sets_give_rows_per_problem_and_posterior() -> None:
    """Every posterior of every problem is diagnosed with its own provenance."""
    case = zoo.shifted_gaussians(0.5, param_count=2)
    samples = case.sample(jax.random.key(0), 80, batch_size=2)
    result = diagnose(
        samples,
        ["ref", "other"],
        case.space,
        key=KEY,
        draws=DRAWS,
        diagnostics=["shape"],
    ).result
    order = [(row["problem"], row["pair"][0]) for row in table_rows(result, "skewness")]
    assert order == [
        (0, "ref"),
        (0, "ref"),
        (0, "other"),
        (0, "other"),
        (1, "ref"),
        (1, "ref"),
        (1, "other"),
        (1, "other"),
    ]
    for row in table_rows(result, "mardia_kurtosis"):
        posterior = ["ref", "other"].index(row["pair"][0])
        problem = row["problem"]
        assert row["provenance"]["input_hashes"] == [
            hash_arrays(
                samples.samples[problem, posterior],
                samples.log_weights[problem, posterior],
            )
        ]
        assert row["parameters"] == ["x0", "x1"]
        assert row["estimator_settings"] == '{"cap":10000}'


def test_chains_and_flat_sets_mix_when_they_share_one_problem() -> None:
    """A run of chains and a set of one problem diagnose together."""
    space, chains = ar1_chains(seed=4)
    flat = SampleSet(jax.random.normal(jax.random.key(5), (120, 2)))
    diagnosis = diagnose(
        [chains, flat],
        ["run", "flat"],
        space,
        key=KEY,
        burn_in=30,
        draws=DRAWS,
        diagnostics=["chains", "shape"],
    )
    assert set(diagnosis.reductions) == {"run"}
    assert {row["pair"][0] for row in table_rows(diagnosis.result, "tau")} == {"run"}
    assert {row["pair"][0] for row in table_rows(diagnosis.result, "skewness")} == {
        "run",
        "flat",
    }
    assert [
        row["pair"][0] for row in table_rows(diagnosis.result, "mardia_skewness")
    ] == ["run", "flat"]


def test_inputs_are_checked() -> None:
    """Names, problems, parameters and the type of the sources raise early."""
    space, chains = ar1_chains()
    flat = SampleSet(jax.random.normal(jax.random.key(0), (50, 2)))
    base: dict[str, Any] = {"key": KEY}
    with pytest.raises(ValueError, match="names must name each of the 2 posteriors"):
        diagnose([chains, flat], ["run"], space, **base)
    with pytest.raises(ValueError, match="names must be unique"):
        diagnose([chains, flat], ["same", "same"], space, **base)
    many = SampleSet(jax.random.normal(jax.random.key(1), (2, 1, 50, 2)))
    with pytest.raises(ValueError, match=r"same number of problems.*\[1, 2\]"):
        diagnose([chains, many], ["run", "flat"], space, **base)
    three = ParamSpace.model_validate(
        {"parameters": [{"name": label} for label in "abc"]}
    )
    with pytest.raises(
        ValueError, match="the space has 3 parameters, but source 0 has 2"
    ):
        diagnose(chains, ["run"], three, **base)
    wide = SampleSet(jax.random.normal(jax.random.key(3), (50, 3)))
    with pytest.raises(
        ValueError, match="the space has 2 parameters, but source 1 has 3"
    ):
        diagnose([flat, wide], ["a", "b"], space, **base)
    with pytest.raises(ValueError, match="at least one Chains or SampleSet"):
        diagnose([], [], space, **base)
    with pytest.raises(TypeError, match="sources must be Chains, a SampleSet"):
        unchecked_diagnose(np.zeros((50, 2)), ["a"], space, **base)
    with pytest.raises(TypeError, match="entry 1 is ndarray"):
        unchecked_diagnose([flat, np.zeros((5, 2))], ["a", "b"], space, **base)
    with pytest.raises(TypeError, match="sources must be Chains, a SampleSet"):
        unchecked_diagnose("chains", ["a"], space, **base)


def test_burn_in_is_checked() -> None:
    """One number or one per run; integers of at least zero that leave steps."""
    space, chains = ar1_chains()
    base: dict[str, Any] = {"key": KEY, "diagnostics": ["chains"]}
    with pytest.raises(
        ValueError, match="one value for every Chains input, got 2 values for 1"
    ):
        diagnose(chains, ["run"], space, burn_in=[10, 20], **base)
    with pytest.raises(ValueError, match="got 0 values for 1"):
        diagnose(chains, ["run"], space, burn_in=[], **base)
    with pytest.raises(ValueError, match="burn_in must be at least 0, got -1"):
        diagnose(chains, ["run"], space, burn_in=-1, **base)
    with pytest.raises(ValueError, match="burn_in must be at least 0, got -5"):
        diagnose(chains, ["run"], space, burn_in=[-5], **base)
    with pytest.raises(TypeError, match="burn_in must be an integer, got float"):
        unchecked_diagnose(chains, ["run"], space, burn_in=1.5, **base)
    with pytest.raises(TypeError, match="burn_in must be an integer, got bool"):
        unchecked_diagnose(chains, ["run"], space, burn_in=True, **base)
    with pytest.raises(ValueError, match="at least 4 steps"):
        diagnose(chains, ["run"], space, burn_in=STEPS - 2, **base)
    flat = SampleSet(jax.random.normal(jax.random.key(2), (50, 2)))
    assert (
        len(
            diagnose(
                flat,
                ["flat"],
                space,
                burn_in=[],
                key=KEY,
                draws=DRAWS,
                diagnostics=["railing"],
            ).result
        )
        == 0
    )
    assert (
        len(
            diagnose(
                flat,
                ["flat"],
                space,
                burn_in=7,
                key=KEY,
                draws=DRAWS,
                diagnostics=["railing"],
            ).result
        )
        == 0
    )


# * The families


def test_families_are_selected_by_name_and_follow_their_order() -> None:
    """Rows come family by family, whatever the order of the selection."""
    case = zoo.railed_gaussians([0.5])
    space, samples = flat_case(case, "railed_0", 200, 0)
    full = diagnose(samples, ["railed"], space, key=KEY, draws=DRAWS)
    metrics = [row["metric"] for row in table_rows(full.result)]
    first_seen: dict[str, int] = {}
    for position, metric in enumerate(metrics):
        first_seen.setdefault(metric, position)
    assert first_seen["skewness"] < first_seen["gmm_bic"] < first_seen["railing_lower"]
    assert "tau" not in metrics
    only = diagnose(
        samples,
        ["railed"],
        space,
        key=KEY,
        draws=DRAWS,
        diagnostics=["railing", "railing"],
    )
    assert [row["metric"] for row in table_rows(only.result)] == ["railing_lower"]
    reordered = diagnose(
        samples, ["railed"], space, key=KEY, draws=DRAWS, diagnostics=["modes", "shape"]
    )
    kinds = [row["metric"] for row in table_rows(reordered.result)]
    assert kinds.index("skewness") < kinds.index("gmm_bic")
    assert DIAGNOSTIC_FAMILIES == ("chains", "shape", "dependence", "modes", "railing")


def test_chains_without_a_chains_input_give_no_rows() -> None:
    """The chain family needs a run of chains; flat sets give none of its rows."""
    case = zoo.railed_gaussians([0.5])
    space, samples = flat_case(case, "railed_0", 100, 1)
    result = diagnose(
        samples, ["railed"], space, key=KEY, draws=DRAWS, diagnostics=["chains"]
    ).result
    assert len(result) == 0
    assert result.table.schema == RESULT_SCHEMA


def test_the_families_are_checked() -> None:
    """Unknown names, an empty selection and a bare string raise ValueError."""
    case = zoo.railed_gaussians([0.5])
    space, samples = flat_case(case, "railed_0", 60, 2)
    with pytest.raises(ValueError, match=r"unknown diagnostic families \['shapes'\]"):
        diagnose(samples, ["a"], space, key=KEY, diagnostics=["shapes"])
    with pytest.raises(ValueError, match="diagnostics must not be empty"):
        diagnose(samples, ["a"], space, key=KEY, diagnostics=[])
    with pytest.raises(
        ValueError, match="diagnostics must be a sequence of family names"
    ):
        unchecked_diagnose(samples, ["a"], space, key=KEY, diagnostics="shape")


# * The rows and their null


def test_a_shape_row_carries_its_surrogate_null_and_flag() -> None:
    """Null columns, recipe, p-value, threshold and flag of a row with draws."""
    case = zoo.railed_gaussians([0.0])
    space, samples = flat_case(case, "railed_0", 300, 3)
    result = diagnose(
        samples, ["railed"], space, key=KEY, draws=FLAG_DRAWS, diagnostics=["shape"]
    ).result
    row = table_rows(result, "skewness")[0]
    assert row["estimator"] == "empirical"
    assert row["pair"] == ["railed"]
    assert row["parameters"] == ["x0"]
    assert row["null_recipe"] == {
        "construction": "gaussian_surrogate",
        "draws": FLAG_DRAWS,
        "sample_size": 300,
        "tail": "two_sided",
    }
    assert len(row["null_draws"]) == FLAG_DRAWS
    assert row["uncertainty_method"] == "null_std"
    assert row["uncertainty"] == row["null_std"]
    assert row["sigma_eff"] is None
    assert row["r_eff"] is None
    with errors.capture_warnings() as records:
        evaluation = flags.evaluate(
            "skewness", row["value"], np.asarray(row["null_draws"])
        )
    assert categories(row) == [record.category for record in records]
    assert row["p_value"] == pytest.approx(evaluation.p_value)
    assert row["sigma_sig"] == pytest.approx(evaluation.sigma_sig)
    assert row["flag"] is evaluation.flagged
    assert row["threshold"] == 0.5
    assert row["flag"]
    anderson = table_rows(result, "anderson_darling")[0]
    assert anderson["null_recipe"]["tail"] == "upper"
    mardia = table_rows(result, "mardia_skewness")[0]
    assert mardia["parameters"] == ["x0", "x1"]
    assert json.loads(mardia["estimator_settings"]) == {"cap": 10_000}


def test_informational_rows_carry_the_value_only() -> None:
    """Pearson's r and the mode count carry the value only; BIC has no null."""
    case = zoo.banana()
    space, samples = flat_case(case, "banana", 150, 4)
    result = diagnose(
        samples,
        ["banana"],
        space,
        key=KEY,
        draws=DRAWS,
        diagnostics=["dependence", "modes"],
    ).result
    for metric in ("pearson", "mode_count"):
        for row in table_rows(result, metric):
            assert row["null_draws"] is None
            assert row["null_recipe"] is None
            assert row["uncertainty"] is None
            assert row["p_value"] is None
            assert row["threshold"] is None
            assert row["flag"] is None
            assert math.isfinite(row["value"])
    bic = table_rows(result, "gmm_bic")[0]
    assert bic["estimator"] == "gmm"
    assert bic["threshold"] == 10.0
    assert bic["null_draws"] is None
    assert bic["p_value"] is None
    settings = json.loads(bic["estimator_settings"])
    assert settings["components"] >= 1
    assert len(settings["bic"]) == 5
    assert bic["parameters"] == ["x0", "x1"]


def test_the_surrogate_null_follows_its_key_path() -> None:
    """The null draws are the statistic on the surrogates of the path's key."""
    case = zoo.railed_gaussians([0.5])
    space, samples = flat_case(case, "railed_0", 120, 5)
    result = diagnose(
        samples, ["p"], space, key=KEY, draws=DRAWS, diagnostics=["shape"]
    ).result
    surrogates = gaussian_surrogates(
        samples.samples[0, 0],
        samples.log_weights[0, 0],
        space,
        size=120,
        key=prng.derive_key(prng.root_key(KEY), "diagnose", 0, "p", "surrogates"),
        draws=DRAWS,
    )
    expected = jax.lax.map(
        lambda values: marginal_moments(values, None, space), surrogates
    )
    row = table_rows(result, "skewness")[0]
    np.testing.assert_allclose(row["null_draws"], expected.skewness[:, 0], rtol=1e-12)
    kurtosis = table_rows(result, "kurtosis")[0]
    np.testing.assert_allclose(
        kurtosis["null_draws"], expected.kurtosis[:, 0], rtol=1e-12
    )


def test_the_same_key_repeats_the_table() -> None:
    """Rows are a function of the key; the seed and key fill the provenance."""
    case = zoo.railed_gaussians([0.5])
    space, samples = flat_case(case, "railed_0", 80, 6)
    options: dict[str, Any] = {
        "draws": DRAWS,
        "diagnostics": ["shape"],
        "config_hash": hash_arrays(np.arange(2)),
    }
    first = diagnose(samples, ["p"], space, key=7, **options).result
    again = diagnose(samples, ["p"], space, key=7, **options).result
    keyed = diagnose(samples, ["p"], space, key=prng.root_key(7), **options).result
    other = diagnose(samples, ["p"], space, key=8, **options).result
    assert first.table.equals(again.table)
    assert first.table["null_draws"].to_list() != other.table["null_draws"].to_list()
    assert first.table["value"].to_list() == other.table["value"].to_list()
    for row in table_rows(first):
        assert row["provenance"]["seed"] == 7
        assert row["provenance"]["prng_key"] == prng.format_key(prng.root_key(7))
        assert row["provenance"]["config_hash"] == options["config_hash"]
    for row in table_rows(keyed):
        assert row["provenance"]["seed"] is None
        assert row["provenance"]["prng_key"] == prng.format_key(prng.root_key(7))
    assert first.table["null_draws"].to_list() == keyed.table["null_draws"].to_list()


# * The session 11 done criterion through the engine


def gaussian_cases() -> list[tuple[str, zoo.ZooCase, str]]:
    """Return Gaussian zoo cases: correlated, on a circle, and far from a bound."""
    covariance = np.array([[1.0, 0.8, 0.0], [0.8, 1.0, -0.3], [0.0, -0.3, 1.0]])
    return [
        (
            "correlated",
            zoo.gaussians(np.zeros((1, 3)), covariance[None], names=["gaussian"]),
            "gaussian",
        ),
        (
            "wrapped",
            zoo.wrapped_gaussians(0.0, width=0.5, param_count=2, correlation=0.5),
            "reference",
        ),
        ("railed far", zoo.railed_gaussians([4.0]), "railed_0"),
    ]


@pytest.mark.parametrize(
    ("label", "case", "posterior"),
    gaussian_cases(),
    ids=lambda value: value if isinstance(value, str) else None,
)
def test_gaussians_raise_no_flag(label: str, case: zoo.ZooCase, posterior: str) -> None:
    """Gaussian posteriors, also on a circle or far from a bound, flag nothing."""
    space, samples = flat_case(case, posterior, FLAG_SIZE, 0)
    result = diagnose(samples, [posterior], space, key=KEY + 1, draws=FLAG_DRAWS).result
    assert flagged(result) == set(), label


def test_the_banana_flags_its_curved_dependence_and_shape_not_its_modes() -> None:
    """Dependence, Mardia's skewness, x1 skewness and BIC flag; the modes do not."""
    space, samples = flat_case(zoo.banana(), "banana", FLAG_SIZE, 1)
    result = diagnose(samples, ["banana"], space, key=2, draws=FLAG_DRAWS).result
    found = flagged(result)
    for row in (
        ("distance_correlation", "x0,x1"),
        ("mutual_information", "x0,x1"),
        ("mardia_skewness", "x0,x1"),
        ("skewness", "x1"),
        ("gmm_bic", "x0,x1"),
    ):
        assert row in found
    assert not any(metric == "mode_dip" for metric, _ in found)
    assert ("spearman", "x0,x1") not in found


def test_a_separated_mixture_flags_its_modes_and_projection() -> None:
    """Two modes along x0, the discriminant projection, and no dependence."""
    space, samples = flat_case(zoo.gaussian_mixture(4.0), "mixture", FLAG_SIZE, 2)
    result = diagnose(samples, ["mixture"], space, key=3, draws=FLAG_DRAWS).result
    found = flagged(result)
    assert ("mode_dip", "x0") in found
    assert ("mode_dip", "x0,x1") in found
    assert ("kurtosis", "x0") in found
    assert ("gmm_bic", "x0,x1") in found
    assert not any(
        metric in ("distance_correlation", "mutual_information") for metric, _ in found
    )
    bic = table_rows(result, "gmm_bic")[0]
    components = json.loads(bic["estimator_settings"])["components"]
    assert components >= 2
    projections = [
        row
        for row in table_rows(result, "mode_dip")
        if row["parameters"] == ["x0", "x1"]
    ]
    assert len(projections) == components * (components - 1) // 2
    for position, row in enumerate(projections):
        settings = json.loads(row["estimator_settings"])
        assert settings["bandwidth"] == "silverman"
        assert len(settings["components"]) == 2
        assert math.sqrt(
            sum(value**2 for value in settings["direction"])
        ) == pytest.approx(1.0)
        assert row["null_recipe"]["sample_size"] == FLAG_SIZE
        count_row = table_rows(result, "mode_count", parameters=["x0", "x1"])[position]
        assert count_row["estimator_settings"] == row["estimator_settings"]
        assert count_row["null_draws"] is None


@pytest.mark.parametrize("offset", [0.0, 1.0])
def test_a_railed_gaussian_flags_railing_and_skewness_not_modes(offset: float) -> None:
    """A normal railed within 1σ of its bound flags its railing score and skewness."""
    space, samples = flat_case(zoo.railed_gaussians([offset]), "railed_0", FLAG_SIZE, 3)
    result = diagnose(samples, ["railed"], space, key=4, draws=FLAG_DRAWS).result
    found = flagged(result)
    assert ("railing_lower", "x0") in found
    assert ("skewness", "x0") in found
    assert not any(metric == "mode_dip" for metric, _ in found)
    railing = table_rows(result, "railing_lower")[0]
    assert railing["estimator"] == "kde"
    assert railing["null_draws"] is None
    assert railing["threshold"] == 0.02
    assert railing["flag"]
    assert table_rows(result, "railing_upper") == []


# * Weighted sets


def weighted_banana_set(count: int = 400) -> tuple[ParamSpace, SampleSet]:
    """Return two independent parameters with importance weights."""
    rng = np.random.default_rng(0)
    proposal, target = stats.norm(scale=1.5), stats.norm()
    draws = proposal.rvs(size=(count, 2), random_state=rng)
    log_weights = (target.logpdf(draws) - proposal.logpdf(draws)).sum(axis=1)
    space = ParamSpace.model_validate({"parameters": [{"name": "x0"}, {"name": "x1"}]})
    return space, SampleSet(draws, log_weights=log_weights)


def test_a_weighted_set_runs_its_dependence_rows_on_the_rejection_subset() -> None:
    """All four statistics read the subset; the surrogates have its effective size."""
    space, samples = weighted_banana_set()
    result = diagnose(
        samples, ["weighted"], space, key=KEY, draws=DRAWS, diagnostics=["dependence"]
    ).result
    root = prng.root_key(KEY)
    path = ("diagnose", 0, "weighted")
    subset = rejection_subset(
        samples.samples[0, 0],
        samples.log_weights[0, 0],
        key=prng.derive_key(root, *path, "dependence", "prepare"),
    )
    expected = dependence_statistics(
        subset, None, space, key=prng.derive_key(root, *path, "dependence", "subsample")
    )
    assert subset.shape[0] < 400
    for metric in ("pearson", "spearman", "distance_correlation", "mutual_information"):
        row = table_rows(result, metric)[0]
        assert row["value"] == pytest.approx(
            float(getattr(expected, metric)[0]), rel=1e-9
        )
        assert math.isfinite(row["value"])
    for metric in ("spearman", "distance_correlation", "mutual_information"):
        row = table_rows(result, metric)[0]
        assert row["null_recipe"]["sample_size"] == surrogate_size(
            float(subset.shape[0])
        )
        assert row["warnings"] == []
    mutual = table_rows(result, "mutual_information")[0]
    assert mutual["estimator"] == "knn"
    assert json.loads(mutual["estimator_settings"]) == {
        "cap": 2_000,
        "neighbours": MUTUAL_INFORMATION_NEIGHBOURS,
        "norm": "max",
    }
    assert table_rows(result, "spearman")[0]["estimator_settings"] == '{"cap":2000}'


def test_an_equal_weight_set_keeps_its_dependence_rows_unchanged() -> None:
    """Equal weights run on the set itself, with surrogates of its n_eff."""
    space, weighted = weighted_banana_set(200)
    equal = SampleSet(weighted.samples[0, 0], n_eff=jnp.asarray(150.0))
    result = diagnose(
        equal, ["equal"], space, key=KEY, draws=DRAWS, diagnostics=["dependence"]
    ).result
    expected = dependence_statistics(
        equal.samples[0, 0],
        None,
        space,
        key=prng.derive_key(
            prng.root_key(KEY), "diagnose", 0, "equal", "dependence", "subsample"
        ),
    )
    row = table_rows(result, "mutual_information")[0]
    assert row["value"] == pytest.approx(
        float(expected.mutual_information[0]), rel=1e-12
    )
    assert row["null_recipe"]["sample_size"] == 150


# * Caps, transform, thresholds


def test_the_caps_bound_the_surrogates_and_the_observed_subsamples() -> None:
    """A pair cap above the cap keeps its larger surrogates; both warn above them."""
    case = zoo.banana()
    space, samples = flat_case(case, "banana", 120, 7)
    result = diagnose(
        samples,
        ["banana"],
        space,
        key=KEY,
        draws=DRAWS,
        cap=40,
        pair_cap=80,
        diagnostics=["shape", "dependence"],
    ).result
    for metric in ("skewness", "anderson_darling", "mardia_kurtosis"):
        for row in table_rows(result, metric):
            assert row["null_recipe"]["sample_size"] == 40
    for metric in ("spearman", "distance_correlation", "mutual_information"):
        row = table_rows(result, metric)[0]
        assert row["null_recipe"]["sample_size"] == 80
        assert categories(row) == ["SubsamplingWarning"]
    mardia = table_rows(result, "mardia_skewness")[0]
    assert categories(mardia) == ["SubsamplingWarning"]
    assert json.loads(mardia["estimator_settings"]) == {"cap": 40}
    assert json.loads(table_rows(result, "spearman")[0]["estimator_settings"]) == {
        "cap": 80
    }
    # * A family's warnings go into each of its rows, so Mardia's reaches them all.
    for row in table_rows(result, "skewness"):
        assert categories(row) == ["SubsamplingWarning"]


def test_a_transform_maps_the_sets_and_names_itself_except_for_the_chains() -> None:
    """Shape rows read the mapped samples; chain rows stay in the chains' space."""
    space, chains = ar1_chains((0.5, 0.2), seed=8)
    transform = Whitening.from_samples(
        chains.samples[:, 0, :, 0, :].reshape(-1, 2)[None],
        None,
        space,
        decorrelate=False,
    )
    plain = diagnose(
        chains,
        ["run"],
        space,
        key=KEY,
        burn_in=20,
        draws=DRAWS,
        diagnostics=["chains", "shape"],
    )
    mapped = diagnose(
        chains,
        ["run"],
        space,
        key=KEY,
        burn_in=20,
        draws=DRAWS,
        transform=transform,
        diagnostics=["chains", "shape"],
    )
    assert {row["space"] for row in table_rows(mapped.result, "tau")} == {"identity"}
    assert {row["space"] for row in table_rows(mapped.result, "skewness")} == {
        transform.description
    }
    assert {row["space"] for row in table_rows(plain.result, "skewness")} == {
        "identity"
    }
    for metric in ("tau", "ess", "split_chain_ks"):
        assert [
            row["value"] for row in table_rows(mapped.result, metric)
        ] == pytest.approx([row["value"] for row in table_rows(plain.result, metric)])
    # * Skewness does not change under an affine map of each parameter.
    for old, new in zip(
        table_rows(plain.result, "skewness"),
        table_rows(mapped.result, "skewness"),
        strict=True,
    ):
        assert new["value"] == pytest.approx(old["value"], rel=1e-9, abs=1e-12)
    assert mapped.result.table["provenance"][0]["input_hashes"] == [hash_chains(chains)]


def test_a_transform_that_frees_a_bound_removes_its_railing_rows() -> None:
    """Whitening turns an interval into a line, so the railing rows disappear."""
    case = zoo.railed_gaussians([0.0])
    space, samples = flat_case(case, "railed_0", 150, 9)
    transform = Whitening.from_samples(samples.samples[0], None, space)
    result = diagnose(
        samples,
        ["railed"],
        space,
        key=KEY,
        draws=DRAWS,
        transform=transform,
        diagnostics=["railing"],
    ).result
    assert len(result) == 0
    assert (
        len(
            diagnose(
                samples, ["railed"], space, key=KEY, diagnostics=["railing"]
            ).result
        )
        == 1
    )


def test_thresholds_and_the_level_change_the_flags() -> None:
    """A caller's threshold replaces the default; a low level lets rows flag."""
    case = zoo.railed_gaussians([0.5])
    space, samples = flat_case(case, "railed_0", 200, 10)
    base: dict[str, Any] = {
        "key": KEY,
        "draws": FLAG_DRAWS,
        "diagnostics": ["shape", "modes"],
    }
    default = diagnose(samples, ["p"], space, **base).result
    row = table_rows(default, "kurtosis")[0]
    assert row["threshold"] == 1.0
    assert not row["flag"]
    thresholds = {
        **flags.THRESHOLDS,
        "kurtosis": Threshold(0.0, Side.BOTH, Tail.TWO_SIDED),
        "mode_count": Threshold(0.5, Side.ABOVE, None),
    }
    custom = diagnose(
        samples, ["p"], space, thresholds=thresholds, level=0.0, **base
    ).result
    kurtosis = table_rows(custom, "kurtosis")[0]
    assert kurtosis["threshold"] == 0.0
    assert kurtosis["flag"]
    count = table_rows(custom, "mode_count")[0]
    assert count["threshold"] == 0.5
    assert count["flag"] is True
    assert count["p_value"] is None
    stricter = diagnose(samples, ["p"], space, level=50.0, **base).result
    assert not any(row["flag"] for row in table_rows(stricter, "skewness"))
    missing = {
        key: value for key, value in flags.THRESHOLDS.items() if key != "gmm_bic"
    }
    with pytest.raises(KeyError, match="gmm_bic"):
        diagnose(samples, ["p"], space, thresholds=missing, **base)
    assert (
        len(
            diagnose(
                samples,
                ["p"],
                space,
                thresholds=missing,
                key=KEY,
                draws=DRAWS,
                diagnostics=["shape"],
            ).result
        )
        > 0
    )


def test_a_tiny_effective_sample_size_gives_a_nan_bic_with_a_warning() -> None:
    """BIC needs n_eff of at least 2; the row is NaN and the other rows go on."""
    space, weighted = weighted_banana_set(60)
    tiny = SampleSet(weighted.samples[0, 0], n_eff=jnp.asarray(1.5))
    result = diagnose(
        tiny, ["tiny"], space, key=KEY, draws=DRAWS, diagnostics=["modes"]
    ).result
    bic = table_rows(result, "gmm_bic")[0]
    assert math.isnan(bic["value"])
    assert bic["flag"] is False
    assert json.loads(bic["estimator_settings"]) == {"bic": [], "components": None}
    assert categories(bic) == ["ValidityWarning"]
    assert "below 2" in bic["warnings"][0]["message"]
    assert math.isfinite(table_rows(result, "mode_count")[0]["value"])


# * Argument errors


@pytest.mark.parametrize(
    ("option", "value", "error", "message"),
    [
        ("draws", 0, ValueError, "draws must be at least 1"),
        ("draws", 1.5, TypeError, "draws must be an integer"),
        ("draws", True, TypeError, "draws must be an integer, got bool"),
        ("cap", 1, ValueError, "cap must be at least 2"),
        ("cap", "10", TypeError, "cap must be an integer"),
        ("pair_cap", 1, ValueError, "pair_cap must be at least 2"),
        ("pair_cap", 2.0, TypeError, "pair_cap must be an integer"),
        ("level", math.nan, ValueError, "level must be finite"),
        ("level", "2", TypeError, "level must be a number"),
        ("budget", 0.0, ValueError, "budget must be positive"),
        ("budget", math.nan, ValueError, "budget must be positive"),
        ("budget", None, TypeError, "budget must be a number"),
        ("thresholds", [0.2], TypeError, "thresholds must be a mapping"),
        ("key", 1.5, TypeError, "key must be a JAX PRNG key or an integer seed"),
        ("key", -3, ValueError, "the seed must lie in"),
    ],
)
def test_counts_numbers_thresholds_and_the_key_are_checked(
    option: str, value: object, error: type[Exception], message: str
) -> None:
    """Each numeric argument and the key raise before any computation."""
    space = ParamSpace.model_validate({"parameters": [{"name": "x0"}, {"name": "x1"}]})
    flat = SampleSet(jax.random.normal(jax.random.key(0), (40, 2)))
    options: dict[str, Any] = {"key": KEY, "diagnostics": ["shape"], option: value}
    with pytest.raises(error, match=message):
        unchecked_diagnose(flat, ["flat"], space, **options)


def test_an_infinite_budget_is_allowed() -> None:
    """A budget of infinity switches the dry-run check off."""
    space = ParamSpace.model_validate({"parameters": [{"name": "x0"}]})
    flat = SampleSet(jax.random.normal(jax.random.key(0), (40, 1)))
    result = diagnose(
        flat,
        ["flat"],
        space,
        key=KEY,
        budget=math.inf,
        draws=DRAWS,
        diagnostics=["railing"],
    ).result
    assert len(result) == 0


def test_the_result_is_a_result_table_with_the_schema() -> None:
    """The table has the agreed columns, and one parameter has no dependence rows."""
    space = ParamSpace.model_validate({"parameters": [{"name": "x0"}]})
    flat = SampleSet(jax.random.normal(jax.random.key(0), (80, 1)))
    result = diagnose(flat, ["flat"], space, key=KEY, draws=DRAWS).result
    assert result.table.schema == RESULT_SCHEMA
    assert not any(row["metric"] == "spearman" for row in table_rows(result))
    assert len(table_rows(result, "skewness")) == 1
