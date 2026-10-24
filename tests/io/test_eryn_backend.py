"""Tests for reading eryn runs into ``Chains``, and the round trip to a space."""

import copy
import math
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import pytest

from mixed_signals.core.chains import Chains
from mixed_signals.core.space import ParamSpace
from mixed_signals.core.topology import Topology
from mixed_signals.errors import ValidityWarning
from mixed_signals.io.eryn_backend import from_eryn
from mixed_signals.mcmc import reduce_chains
from mixed_signals.priors import ErynPhysical, ErynPrior

eryn_priors = pytest.importorskip("eryn.priors")
ensemble = pytest.importorskip("eryn.ensemble")
backends = pytest.importorskip("eryn.backends")

TEMP_COUNT = 2
WALKER_COUNT = 8
STEP_COUNT = 20
THIN_BY = 2
REPEATS = 3
TWO_PI = 2 * math.pi
FIXED_VALUE = 0.5
BRANCH = "model_0"
FIELDS = (
    "samples",
    "betas",
    "log_like",
    "log_prior",
    "accepted",
    "proposed",
    "swaps_accepted",
    "swaps_proposed",
)


class Run(NamedTuple):
    """A finished eryn run and its priors."""

    sampler: Any
    priors: Any


def run_priors() -> Any:
    """Return priors with a transformed, a periodic and a fixed parameter."""
    return eryn_priors.ProbDistContainer(
        {
            "ln_amplitude": eryn_priors.LogUniform(
                1.0, 100.0, name="ln_amplitude", name_phys="amplitude"
            ),
            "phase": eryn_priors.UniformDistribution(0.0, TWO_PI),
            "fixed": eryn_priors.DeltaFunction(FIXED_VALUE),
        }
    )


def log_like(points: np.ndarray, *args: Any) -> np.ndarray:
    """Return a Gaussian log-likelihood in the sampling coordinates."""
    return -0.5 * (((points[..., 0] - 2.0) / 0.5) ** 2 + (points[..., 1] - 3.0) ** 2)


def run_sampler(temp_count: int, backend: Any = None) -> Run:
    """Run a short tempered eryn sampler that stores every second step."""
    priors = run_priors()
    sampler = ensemble.EnsembleSampler(
        WALKER_COUNT,
        3,
        log_like,
        priors,
        tempering_kwargs={"ntemps": temp_count},
        num_repeats_in_model=REPEATS,
        periodic={BRANCH: {1: TWO_PI}},
        backend=backend,
    )
    start = priors.rvs(size=(temp_count, WALKER_COUNT))
    sampler.run_mcmc(start, STEP_COUNT, thin_by=THIN_BY, progress=False)
    return Run(sampler, priors)


@pytest.fixture(scope="module")
def run() -> Run:
    """A tempered run with an in-memory backend."""
    return run_sampler(TEMP_COUNT)


@pytest.fixture(scope="module")
def hdf_run(tmp_path_factory: pytest.TempPathFactory) -> tuple[Run, Path]:
    """A tempered run stored in an HDF5 file, and the path of that file."""
    path = tmp_path_factory.mktemp("eryn") / "run.h5"
    return run_sampler(TEMP_COUNT, backends.HDFBackend(str(path))), path


def assert_same_chains(first: Chains, second: Chains) -> None:
    """Check that two chains hold the same arrays, or both lack a field."""
    for name in FIELDS:
        first_field, second_field = getattr(first, name), getattr(second, name)
        assert (first_field is None) == (second_field is None), name
        if first_field is not None:
            np.testing.assert_array_equal(first_field, second_field, err_msg=name)


# * Reading


def test_sampler_reads_every_temperature(run: Run) -> None:
    """Samples, log densities and betas are the backend's, without fixed columns."""
    chains = Chains.from_eryn(run.sampler)
    backend = run.sampler.backend
    assert chains.shape == (STEP_COUNT, TEMP_COUNT, WALKER_COUNT, 1, 2)
    np.testing.assert_array_equal(chains.samples, backend.get_chain()[BRANCH][..., :2])
    np.testing.assert_array_equal(chains.log_like, backend.get_log_like())
    np.testing.assert_array_equal(chains.log_prior, backend.get_log_prior())
    np.testing.assert_array_equal(chains.betas, backend.get_betas())


def test_counts_carry_their_proposals(run: Run) -> None:
    """Each stored step adds num_repeats_in_model proposals and a swap per walker."""
    chains = Chains.from_eryn(run.sampler)
    backend = run.sampler.backend
    assert chains.accepted is not None
    assert chains.proposed is not None
    assert chains.swaps_accepted is not None
    assert chains.swaps_proposed is not None
    np.testing.assert_array_equal(chains.accepted, backend.accepted)
    np.testing.assert_array_equal(chains.proposed, STEP_COUNT * REPEATS)
    np.testing.assert_array_equal(chains.swaps_accepted, backend.swaps_accepted)
    np.testing.assert_array_equal(chains.swaps_proposed, STEP_COUNT * WALKER_COUNT)
    assert chains.swaps_proposed.shape == (TEMP_COUNT - 1,)


def test_backend_with_its_settings_reads_like_the_sampler(run: Run) -> None:
    """A backend with num_repeats_in_model and priors gives the sampler's chains."""
    assert_same_chains(
        from_eryn(run.sampler.backend, num_repeats_in_model=REPEATS, priors=run.priors),
        from_eryn(run.sampler),
    )


def test_backend_alone_keeps_every_column(run: Run) -> None:
    """Without priors and repeats, every column stays and acceptance counts go."""
    chains = Chains.from_eryn(run.sampler.backend)
    assert chains.shape.param == 3
    np.testing.assert_array_equal(chains.samples[..., 2], FIXED_VALUE)
    assert chains.accepted is None
    assert chains.proposed is None
    assert chains.swaps_accepted is not None


def test_backends_without_a_parameter_order_still_drop_fixed_columns(
    run: Run,
) -> None:
    """A backend that stores no key order skips that check."""
    backend = copy.deepcopy(run.sampler.backend)
    backend.key_order = None
    chains = Chains.from_eryn(backend, priors=run.priors)
    assert chains.shape.param == 2


def test_hdf_files_read_like_their_backend(hdf_run: tuple[Run, Path]) -> None:
    """A path, as str or Path, reads the file that an HDFBackend wrote."""
    file_run, path = hdf_run
    expected = Chains.from_eryn(file_run.sampler)
    for source in (path, str(path)):
        assert_same_chains(
            Chains.from_eryn(
                source, num_repeats_in_model=REPEATS, priors=file_run.priors
            ),
            expected,
        )


def test_one_temperature_has_no_swap_counts() -> None:
    """A run without tempering has no neighbouring temperatures."""
    chains = Chains.from_eryn(run_sampler(1).sampler)
    assert chains.shape.temp == 1
    assert chains.swaps_accepted is None
    assert chains.swaps_proposed is None
    assert chains.accepted is not None


def test_samplers_with_other_priors_keep_every_column(
    run: Run, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Samplers also take plain logpdf objects, which name no fixed values."""
    monkeypatch.setattr(run.sampler, "_priors", {BRANCH: object()})
    assert Chains.from_eryn(run.sampler).shape.param == 3


def test_branches_are_chosen_by_name(run: Run) -> None:
    """A run with several branches needs branch=, and unknown names raise."""
    backend = copy.deepcopy(run.sampler.backend)
    backend.branch_names = [BRANCH, "other"]
    with pytest.raises(ValueError, match="choose one with branch="):
        Chains.from_eryn(backend)
    with pytest.raises(ValueError, match="has no branch 'missing'"):
        Chains.from_eryn(backend, branch="missing")
    assert Chains.from_eryn(backend, branch=BRANCH).shape.param == 3


# * Checks


def test_samplers_supply_repeats_and_priors(run: Run) -> None:
    """A sampler comes without num_repeats_in_model and priors."""
    with pytest.raises(ValueError, match="leave both out"):
        Chains.from_eryn(run.sampler, num_repeats_in_model=REPEATS)
    with pytest.raises(ValueError, match="leave both out"):
        Chains.from_eryn(run.sampler, priors=run.priors)


@pytest.mark.parametrize(
    ("repeats", "error", "message"),
    [(True, TypeError, "must be an integer"), (0, ValueError, "at least 1")],
)
def test_repeats_must_be_positive_integers(
    run: Run, repeats: Any, error: type[Exception], message: str
) -> None:
    """num_repeats_in_model counts proposals, so it is a positive integer."""
    with pytest.raises(error, match=message):
        Chains.from_eryn(run.sampler.backend, num_repeats_in_model=repeats)


def test_other_sources_raise() -> None:
    """Only samplers, backends and paths are sources."""
    with pytest.raises(TypeError, match="source must be an eryn EnsembleSampler"):
        Chains.from_eryn({"chain": np.zeros(3)})


def test_runs_must_hold_stored_steps(tmp_path: Path) -> None:
    """Backends without a run or without steps, and missing files, raise."""
    with pytest.raises(ValueError, match="holds no run"):
        Chains.from_eryn(backends.Backend())
    empty = backends.Backend()
    empty.reset(WALKER_COUNT, 2)
    with pytest.raises(ValueError, match="holds no stored step"):
        Chains.from_eryn(empty)
    with pytest.raises(ValueError, match="holds no run in"):
        Chains.from_eryn(tmp_path / "missing.h5")


@pytest.mark.parametrize(
    ("settings", "message"),
    [
        ({"nsamplers": 2}, "2 independent samplers"),
        ({"rj": True}, "reversible jumps"),
    ],
)
def test_deferred_run_types_raise(settings: dict[str, Any], message: str) -> None:
    """Runs with several samplers or reversible jumps wait for later versions."""
    backend = backends.Backend()
    backend.reset(WALKER_COUNT, 2, **settings)
    with pytest.raises(NotImplementedError, match=message):
        Chains.from_eryn(backend)


def test_priors_must_fit_the_stored_parameters(run: Run) -> None:
    """Priors of another size, order or type raise."""
    backend = run.sampler.backend
    renamed = eryn_priors.ProbDistContainer(
        {
            "phase": eryn_priors.UniformDistribution(0.0, TWO_PI),
            "ln_amplitude": eryn_priors.LogUniform(1.0, 100.0),
            "fixed": eryn_priors.DeltaFunction(FIXED_VALUE),
        }
    )
    with pytest.raises(ValueError, match="stores the parameters"):
        Chains.from_eryn(backend, priors=renamed)
    smaller = eryn_priors.ProbDistContainer(
        {"ln_amplitude": eryn_priors.LogUniform(1.0, 100.0)}
    )
    with pytest.raises(ValueError, match="the priors have 1 parameters"):
        Chains.from_eryn(backend, priors=smaller)
    with pytest.raises(TypeError, match="must be an eryn ProbDistContainer"):
        Chains.from_eryn(backend, priors={"ln_amplitude": smaller})


# * Round trip


def test_a_short_run_round_trips_into_chains_and_a_space(run: Run) -> None:
    """Chains, space, prior and physical transform of one run fit together."""
    chains = Chains.from_eryn(run.sampler)
    space = ParamSpace.from_eryn_priors(run.priors, periodic={1: TWO_PI})
    assert space.names == ("ln_amplitude", "phase")
    assert len(space) == chains.shape.param
    assert space["phase"].topology is Topology.CIRCLE
    # * eryn's stored log prior is the density of the sampling coordinates.
    assert chains.log_prior is not None
    np.testing.assert_allclose(
        ErynPrior(run.priors).logpdf(chains.samples[:, :, :, 0]),
        chains.log_prior,
        rtol=1e-12,
    )
    physical = ErynPhysical(run.priors)
    cold = chains.cold().samples[:, 0, :, 0]
    np.testing.assert_allclose(physical.forward(cold)[..., 0], np.exp(cold[..., 0]))
    assert physical.map_space(space).names == ("amplitude", "phase")
    # * A test run is far shorter than the 50 autocorrelation times that tau needs.
    with pytest.warns(ValidityWarning, match="fewer than 50 autocorrelation times"):
        reduction = reduce_chains(chains, burn_in=STEP_COUNT // 4, thin=1)
    assert reduction.sample_set.samples.shape[-1] == len(space)
