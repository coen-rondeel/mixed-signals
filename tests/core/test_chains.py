"""Tests for ``Chains``: layouts, checks, counts and the cold chain."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mixed_signals.core.chains import Chains, ChainShape
from mixed_signals.errors import PrecisionWarning

STEP_COUNT = 30
WALKER_COUNT = 6
PARAM_COUNT = 2
LADDER = (1.0, 0.5, 0.2)


@pytest.fixture
def rng() -> np.random.Generator:
    """A seeded NumPy generator for test data."""
    return np.random.default_rng(2026)


def tempered_inputs(
    rng: np.random.Generator, temp_count: int = len(LADDER)
) -> dict[str, np.ndarray]:
    """Return eryn-layout inputs of a tempered run, with every optional field."""
    per_walker = (STEP_COUNT, temp_count, WALKER_COUNT)
    return {
        "samples": rng.normal(size=(*per_walker, 1, PARAM_COUNT)),
        "betas": np.tile(LADDER[:temp_count], (STEP_COUNT, 1)),
        "log_like": rng.normal(size=per_walker),
        "log_prior": rng.normal(size=per_walker),
        # * eryn stores its counts as floats.
        "accepted": np.full((temp_count, WALKER_COUNT), 12.0),
        "proposed": np.full((temp_count, WALKER_COUNT), STEP_COUNT),
        "swaps_accepted": np.arange(temp_count - 1),
        "swaps_proposed": np.full(temp_count - 1, STEP_COUNT * WALKER_COUNT),
    }


def build(inputs: dict[str, np.ndarray]) -> Chains:
    """Build chains from a mapping of inputs that holds the samples."""
    remaining = dict(inputs)
    return Chains(remaining.pop("samples"), **remaining)


# * Layouts


def test_eryn_layout_keeps_every_field(rng: np.random.Generator) -> None:
    """Inputs in the eryn layout keep their shapes; counts become integers."""
    inputs = tempered_inputs(rng)
    chains = build(inputs)
    assert chains.shape == ChainShape(STEP_COUNT, 3, WALKER_COUNT, 1, PARAM_COUNT)
    assert chains.shape == (STEP_COUNT, 3, WALKER_COUNT, 1, PARAM_COUNT)
    np.testing.assert_array_equal(chains.samples, inputs["samples"])
    np.testing.assert_array_equal(chains.betas, inputs["betas"])
    assert chains.accepted is not None
    assert chains.accepted.dtype == jnp.int64
    assert chains.swaps_accepted is not None
    assert chains.swaps_accepted.shape == (2,)


def test_single_temperature_layout_adds_temperature_and_leaf_axes(
    rng: np.random.Generator,
) -> None:
    """Chains shaped (step, walker, param) gain unit temperature and leaf axes."""
    samples = rng.normal(size=(STEP_COUNT, WALKER_COUNT, PARAM_COUNT))
    log_prior = rng.normal(size=(STEP_COUNT, WALKER_COUNT))
    chains = Chains(
        samples,
        log_prior=log_prior,
        accepted=np.arange(WALKER_COUNT),
        proposed=np.full(WALKER_COUNT, STEP_COUNT),
    )
    assert chains.shape == (STEP_COUNT, 1, WALKER_COUNT, 1, PARAM_COUNT)
    np.testing.assert_array_equal(chains.samples[:, 0, :, 0], samples)
    np.testing.assert_array_equal(chains.betas, np.ones((STEP_COUNT, 1)))
    assert chains.log_prior is not None
    np.testing.assert_array_equal(chains.log_prior[:, 0], log_prior)
    assert chains.accepted is not None
    assert chains.accepted.shape == (1, WALKER_COUNT)
    assert chains.log_like is None
    assert chains.swaps_accepted is None


def test_both_layouts_of_one_temperature_build_the_same_chains(
    rng: np.random.Generator,
) -> None:
    """A single-temperature run in either layout gives equal chains."""
    samples = rng.normal(size=(STEP_COUNT, WALKER_COUNT, PARAM_COUNT))
    log_like = rng.normal(size=(STEP_COUNT, WALKER_COUNT))
    compact = Chains(samples, log_like=log_like)
    eryn_layout = Chains(samples[:, None, :, None, :], log_like=log_like[:, None])
    assert eqx.tree_equal(compact, eryn_layout)


def test_log_densities_and_counts_are_optional(rng: np.random.Generator) -> None:
    """Samples alone build chains, as for samplers that store the log posterior."""
    chains = Chains(rng.normal(size=(STEP_COUNT, WALKER_COUNT, PARAM_COUNT)))
    for name in ("log_like", "log_prior", "accepted", "proposed", "swaps_accepted"):
        assert getattr(chains, name) is None


def test_narrow_floats_warn_and_counts_do_not(rng: np.random.Generator) -> None:
    """float32 samples warn; integer counts in any type never do."""
    with pytest.warns(PrecisionWarning, match=r"samples \(float32\)") as records:
        Chains(
            rng.normal(size=(STEP_COUNT, WALKER_COUNT, PARAM_COUNT)).astype(np.float32),
            accepted=np.ones(WALKER_COUNT, dtype=np.float32),
            proposed=np.full(WALKER_COUNT, 2, dtype=np.int32),
        )
    assert len(records) == 1


# * Checks of shapes and types


@pytest.mark.parametrize(
    "shape", [(STEP_COUNT, PARAM_COUNT), (STEP_COUNT, 3, WALKER_COUNT, PARAM_COUNT)]
)
def test_other_layouts_are_rejected(shape: tuple[int, ...]) -> None:
    """Only the eryn layout and (step, walker, param) are accepted."""
    with pytest.raises(ValueError, match="eryn layout"):
        Chains(np.zeros(shape))


def test_empty_axes_are_rejected() -> None:
    """Every axis needs at least one entry."""
    with pytest.raises(ValueError, match="entries on every axis"):
        Chains(np.zeros((0, WALKER_COUNT, PARAM_COUNT)))


def test_several_leaves_are_not_supported_yet(rng: np.random.Generator) -> None:
    """Runs with several leaves wait for the catalog work."""
    with pytest.raises(NotImplementedError, match="2 leaves"):
        Chains(rng.normal(size=(STEP_COUNT, 1, WALKER_COUNT, 2, PARAM_COUNT)))


@pytest.mark.parametrize("name", ["betas", "swaps_accepted"])
def test_ladder_fields_need_the_eryn_layout(
    name: str, rng: np.random.Generator
) -> None:
    """A run without a temperature axis has no ladder to describe."""
    with pytest.raises(ValueError, match=f"{name} needs samples in the eryn layout"):
        Chains(
            rng.normal(size=(STEP_COUNT, WALKER_COUNT, PARAM_COUNT)),
            **{name: np.ones(STEP_COUNT)},
        )


@pytest.mark.parametrize(
    ("name", "bad_shape"),
    [
        ("betas", (STEP_COUNT, 2)),
        ("log_like", (STEP_COUNT, 3)),
        ("log_prior", (STEP_COUNT, 3, WALKER_COUNT, 1)),
        ("accepted", (WALKER_COUNT,)),
        ("proposed", (3, WALKER_COUNT + 1)),
        ("swaps_accepted", (3,)),
        ("swaps_proposed", (1,)),
    ],
)
def test_field_shapes_must_fit_the_samples(
    name: str, bad_shape: tuple[int, ...], rng: np.random.Generator
) -> None:
    """A field whose shape does not fit the samples fails with both shapes."""
    inputs = tempered_inputs(rng)
    inputs[name] = np.ones(bad_shape)
    with pytest.raises(ValueError, match=f"{name} has shape"):
        build(inputs)


@pytest.mark.parametrize(
    ("given", "missing"),
    [("accepted", "proposed"), ("swaps_proposed", "swaps_accepted")],
)
def test_counts_come_with_their_proposals(
    given: str, missing: str, rng: np.random.Generator
) -> None:
    """A count without the proposals behind it, or the reverse, is rejected."""
    inputs = tempered_inputs(rng)
    del inputs[missing]
    with pytest.raises(ValueError, match="give both or neither"):
        build(inputs)
    assert given in inputs


def test_several_temperatures_need_betas(rng: np.random.Generator) -> None:
    """Without betas, only a single temperature can be assumed cold."""
    inputs = tempered_inputs(rng)
    del inputs["betas"]
    with pytest.raises(ValueError, match="betas are needed for chains with 3"):
        build(inputs)


@pytest.mark.parametrize("name", ["samples", "betas", "log_like"])
def test_complex_float_fields_are_rejected(name: str, rng: np.random.Generator) -> None:
    """Float fields must hold real numbers."""
    inputs = tempered_inputs(rng)
    inputs[name] = inputs[name].astype(complex)
    with pytest.raises(TypeError, match="real numbers"):
        build(inputs)


@pytest.mark.parametrize("dtype", [bool, complex])
def test_counts_must_be_numbers(dtype: type, rng: np.random.Generator) -> None:
    """Counts must be integers or floats, not bools or complex numbers."""
    inputs = tempered_inputs(rng)
    inputs["accepted"] = inputs["accepted"].astype(dtype)
    with pytest.raises(TypeError, match="accepted must hold whole numbers"):
        build(inputs)


# * Checks of values


@pytest.mark.parametrize("bad_value", [np.nan, np.inf])
def test_non_finite_samples_fail_with_their_location(
    bad_value: float, rng: np.random.Generator
) -> None:
    """NaN samples, as eryn stores for inactive leaves, fail with their place."""
    inputs = tempered_inputs(rng)
    inputs["samples"][4, 2, 1, 0, 1] = bad_value
    with pytest.raises(ValueError, match="step 4, temperature 2, walker 1"):
        build(inputs)


@pytest.mark.parametrize("bad_value", [-0.1, 1.5, np.nan])
def test_betas_lie_between_zero_and_one(
    bad_value: float, rng: np.random.Generator
) -> None:
    """Inverse temperatures outside [0, 1] are rejected."""
    inputs = tempered_inputs(rng)
    inputs["betas"][3, 1] = bad_value
    with pytest.raises(ValueError, match="step 3, temperature 1"):
        build(inputs)


def test_a_prior_chain_at_beta_zero_is_accepted(rng: np.random.Generator) -> None:
    """β = 0, a chain that samples the prior, is a valid inverse temperature."""
    inputs = tempered_inputs(rng)
    inputs["betas"][:, -1] = 0.0
    assert build(inputs).shape.temp == 3


@pytest.mark.parametrize("name", ["log_like", "log_prior"])
@pytest.mark.parametrize("bad_value", [np.nan, np.inf])
def test_log_densities_reject_nan_and_positive_infinity(
    name: str, bad_value: float, rng: np.random.Generator
) -> None:
    """Log densities may be -inf, but not NaN or +inf."""
    inputs = tempered_inputs(rng)
    inputs[name][0, 1, 2] = -np.inf
    build(inputs)
    inputs[name][5, 0, 3] = bad_value
    with pytest.raises(ValueError, match=f"{name} contains NaN or \\+inf at step 5"):
        build(inputs)


@pytest.mark.parametrize("bad_value", [-1.0, 2.5, np.nan])
def test_counts_are_non_negative_whole_numbers(
    bad_value: float, rng: np.random.Generator
) -> None:
    """Counts stored as floats must hold non-negative whole numbers."""
    inputs = tempered_inputs(rng)
    inputs["swaps_accepted"] = np.array([0.0, bad_value])
    with pytest.raises(ValueError, match="non-negative whole numbers"):
        build(inputs)


@pytest.mark.parametrize(
    ("count_name", "total_name"),
    [("accepted", "proposed"), ("swaps_accepted", "swaps_proposed")],
)
def test_counts_do_not_exceed_their_proposals(
    count_name: str, total_name: str, rng: np.random.Generator
) -> None:
    """More accepted than proposed moves is impossible."""
    inputs = tempered_inputs(rng)
    inputs[count_name] = inputs[total_name] + 1
    with pytest.raises(ValueError, match=f"{count_name} exceeds {total_name}"):
        build(inputs)


def test_values_are_not_checked_inside_jit(rng: np.random.Generator) -> None:
    """Under jit the values are unknown, so only shapes and types are checked."""
    inputs = tempered_inputs(rng)
    inputs["samples"][0, 0, 0, 0, 0] = np.nan
    inputs["accepted"] = inputs["accepted"] + 0.5
    arrays = {name: jnp.asarray(value) for name, value in inputs.items()}
    built = jax.jit(build)(arrays)
    assert built.shape.temp == 3
    assert built.accepted is not None
    assert built.accepted.dtype == jnp.int64


# * The cold chain and pytree behaviour


def test_cold_keeps_the_cold_temperature_and_its_counts(
    rng: np.random.Generator,
) -> None:
    """The cold chain keeps its row of every field and drops the swaps."""
    inputs = tempered_inputs(rng)
    chains = build(inputs)
    cold = chains.cold()
    assert chains.cold_index() == 0
    assert cold.shape == (STEP_COUNT, 1, WALKER_COUNT, 1, PARAM_COUNT)
    np.testing.assert_array_equal(cold.samples[:, 0], inputs["samples"][:, 0])
    np.testing.assert_array_equal(cold.betas, np.ones((STEP_COUNT, 1)))
    assert cold.log_like is not None
    np.testing.assert_array_equal(cold.log_like[:, 0], inputs["log_like"][:, 0])
    assert cold.accepted is not None
    np.testing.assert_array_equal(cold.accepted, inputs["accepted"][:1])
    assert cold.swaps_accepted is None
    assert cold.swaps_proposed is None


def test_cold_of_bare_chains_keeps_only_the_samples(
    rng: np.random.Generator,
) -> None:
    """Chains without log densities or counts give a bare cold chain."""
    inputs = tempered_inputs(rng)
    chains = Chains(inputs["samples"], betas=inputs["betas"])
    cold = chains.cold()
    np.testing.assert_array_equal(cold.samples[:, 0], inputs["samples"][:, 0])
    assert cold.log_like is None
    assert cold.accepted is None


def test_cold_index_finds_beta_one_anywhere_in_the_ladder(
    rng: np.random.Generator,
) -> None:
    """The cold chain need not come first in the ladder."""
    inputs = tempered_inputs(rng)
    inputs["betas"] = inputs["betas"][:, ::-1].copy()
    chains = build(inputs)
    assert chains.cold_index() == 2
    np.testing.assert_array_equal(chains.cold().samples[:, 0], inputs["samples"][:, 2])


def test_chains_without_a_cold_chain_fail(rng: np.random.Generator) -> None:
    """A ladder whose first β leaves 1 at some step has no cold chain."""
    inputs = tempered_inputs(rng)
    inputs["betas"][7, 0] = 0.999
    chains = build(inputs)
    with pytest.raises(ValueError, match="no cold chain"):
        chains.cold()


def test_chains_pass_through_jit_unchanged(rng: np.random.Generator) -> None:
    """Chains are a pytree: jit takes them as input and returns them intact."""
    chains = build(tempered_inputs(rng))
    assert eqx.tree_equal(jax.jit(lambda passed: passed)(chains), chains)
