"""Shared helpers and fixtures: the zoo and estimator tests, and the pipeline data."""

import functools
import shutil
import sys
import types
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import polars as pl
import pytest

from mixed_signals import zoo
from mixed_signals.core.result import Provenance, Result
from mixed_signals.core.samples import SampleSet
from mixed_signals.engines import compare as run_compare
from mixed_signals.engines import reweight
from mixed_signals.errors import ValidityWarning
from mixed_signals.pipeline import run_pipeline

# * The validation build lives in scripts/, which is not an installed package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

type Density = Callable[[np.ndarray], np.ndarray]
"""Maps grid points shaped (..., D) to density values shaped (...)."""

type GridIntegral = Callable[..., float]
"""Integrates a density over a box given as (lower, upper, panels) per axis."""

GAUSS_LEGENDRE_ORDER = 8


def integrate_on_grid(density: Density, *axes: tuple[float, float, int]) -> float:
    """Integrate a density over a box with composite Gauss-Legendre rules.

    Each axis is split into equal panels with eight nodes each, which
    integrates smooth densities to near machine precision with one vectorised
    evaluation instead of thousands of scalar calls.

    Args:
        density: Function of points shaped (..., D), returning densities.
        *axes: One (lower, upper, panels) triple per axis.

    Returns:
        The integral over the box.
    """
    nodes, weights = np.polynomial.legendre.leggauss(GAUSS_LEGENDRE_ORDER)
    axis_nodes, axis_weights = [], []
    for lower, upper, panels in axes:
        edges = np.linspace(lower, upper, panels + 1)
        half_widths = 0.5 * np.diff(edges)
        centres = 0.5 * (edges[:-1] + edges[1:])
        axis_nodes.append((centres[:, None] + half_widths[:, None] * nodes).ravel())
        axis_weights.append((half_widths[:, None] * weights).ravel())
    points = np.stack(np.meshgrid(*axis_nodes, indexing="ij"), axis=-1)
    grid_weights = functools.reduce(np.multiply.outer, axis_weights)
    return float(np.sum(grid_weights * density(points)))


@pytest.fixture
def grid_integral() -> GridIntegral:
    """The composite Gauss-Legendre integrator, for tests that check normalisation."""
    return integrate_on_grid


# * Shared data of the pipeline tests: a short eryn run and a folder of config and data.

ERYN_PARAMS = 3
ERYN_WALKERS = 8
ERYN_STEPS = 120

PROJECT_MODELS = '''
import jax.numpy as jnp
import numpy as np
from eryn.priors import ProbDistContainer, UniformDistribution

from mixed_signals import zoo
from mixed_signals.core.space import ParamSpace
from mixed_signals.transforms.base import Transform

FISHER = np.array([[2.0, 0.3, 0.1], [0.3, 1.5, -0.2], [0.1, -0.2, 1.0]])
case = zoo.linear_signal(FISHER, data_size=16)
generator = case.generator
noise = case.noise
priors = ProbDistContainer(
    {position: UniformDistribution(-20.0, 20.0) for position in range(3)}
)


class Shift(Transform):
    """The map y = x + 1, which moves the bounds of every parameter."""

    def forward(self, points):
        return points + 1.0

    def inverse(self, points):
        return points - 1.0

    def log_abs_det_jacobian(self, points):
        return jnp.zeros(points.shape[:-1])

    @property
    def description(self):
        return "shift"

    def map_space(self, space):
        parameters = []
        for parameter in space.parameters:
            lower, upper = parameter.bounds
            moved = parameter.model_copy(update={"bounds": (lower + 1.0, upper + 1.0)})
            parameters.append(moved)
        return ParamSpace(parameters=tuple(parameters), spheres=space.spheres)


shift = Shift()
not_a_transform = 3
'''

PROJECT_CONFIG = """
seed: 7
output: results
space: {eryn_priors: models:priors}
posteriors:
  - {name: mcmc, format: eryn, path: data/mcmc.h5, burn_in: 20, thin: 4}
  - {name: chains, format: chains, path: data/ar1.npz, burn_in: 10}
  - {name: samples, format: samples, path: data/draws.npz}
  - {name: table, format: table, path: data/table.parquet}
  - {name: group, format: hdf5, path: data/peer.h5, group: posterior}
diagnose: {draws: 20, diagnostics: [chains, shape], cap: 200}
compare: {reference: mcmc, metrics: [mean_shift, energy], draws: 20, cap: 200}
compare_signals:
  generator: models:generator
  noise: models:noise
  reference: mcmc
  draws: 20
  cap: 200
"""


@pytest.fixture(scope="session")
def eryn_run_file(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """An HDF5 file of a short eryn run on a unit Gaussian with the keys 0, 1 and 2."""
    eryn_priors = pytest.importorskip("eryn.priors")
    ensemble = pytest.importorskip("eryn.ensemble")
    backends = pytest.importorskip("eryn.backends")
    path = tmp_path_factory.mktemp("eryn_run") / "run.h5"
    priors = eryn_priors.ProbDistContainer(
        {
            position: eryn_priors.UniformDistribution(-20.0, 20.0)
            for position in range(ERYN_PARAMS)
        }
    )
    # ! The sampler copies NumPy's global state when it is built, so seed before that.
    np.random.seed(0)  # noqa: NPY002
    sampler = ensemble.EnsembleSampler(
        ERYN_WALKERS,
        ERYN_PARAMS,
        lambda points, *args: -0.5 * np.sum(points**2, axis=-1),
        priors,
        tempering_kwargs={"ntemps": 1},
        backend=backends.HDFBackend(str(path)),
    )
    start = np.random.default_rng(0).normal(size=(1, ERYN_WALKERS, ERYN_PARAMS))
    sampler.run_mcmc(start, ERYN_STEPS, progress=False)
    return path


@pytest.fixture(scope="session")
def pipeline_project(
    tmp_path_factory: pytest.TempPathFactory, eryn_run_file: Path
) -> Path:
    """A config folder with ``models.py`` and one data file of each format."""
    h5py = pytest.importorskip("h5py")
    folder = tmp_path_factory.mktemp("pipeline_project")
    data = folder / "data"
    data.mkdir()
    (folder / "models.py").write_text(PROJECT_MODELS, encoding="utf-8")
    shutil.copy(eryn_run_file, data / "mcmc.h5")
    rng = np.random.default_rng(0)
    case = zoo.ar1_chains([0.3, 0.2, 0.1])
    chains = case.sample(jax.random.key(0), 400, walker_count=ERYN_WALKERS)
    np.savez(data / "ar1.npz", samples=np.asarray(chains.samples))
    values = rng.normal(0.3, 1.0, size=(300, ERYN_PARAMS))
    # * A shuffled order and an extra column: the table is read by name.
    pl.DataFrame(
        {
            "x2": values[:, 2],
            "x0": values[:, 0],
            "x1": values[:, 1],
            "other": values[:, 0],
        }
    ).write_parquet(data / "table.parquet")
    peer = rng.normal(size=(300, ERYN_PARAMS))
    with h5py.File(data / "peer.h5", "w") as file:
        group = file.create_group("posterior")
        for position, name in enumerate(("x0", "x1", "x2")):
            group.create_dataset(name, data=peer[:, position])
    np.savez(
        data / "draws.npz",
        samples=rng.normal(size=(300, ERYN_PARAMS)),
        n_eff=np.float64(200.0),
    )
    return folder


@pytest.fixture
def own_models(monkeypatch: pytest.MonkeyPatch) -> None:
    """Let a test import the ``models`` of its own config folder.

    Python imports a module once per process, so the pipeline refuses a folder whose
    ``models.py`` differs from the imported one. The cached module returns after the
    test.
    """
    monkeypatch.setitem(sys.modules, "models", types.ModuleType("models"))
    monkeypatch.delitem(sys.modules, "models")


@pytest.fixture(scope="session")
def pipeline_config() -> str:
    """The config of the five formats and three engines of ``pipeline_project``."""
    return PROJECT_CONFIG


# * Shared run folders of the report tests: a pipeline run, a reweighting and a batch.

REPORT_CONFIG = """
seed: 11
output: report_run
space: {eryn_priors: models:priors}
posteriors:
  - {name: mcmc, format: eryn, path: data/mcmc.h5, burn_in: 20, thin: 4}
  - {name: table, format: table, path: data/table.parquet}
  - {name: group, format: hdf5, path: data/peer.h5, group: posterior}
diagnose: {draws: 20, diagnostics: [chains, shape, modes, railing], cap: 200}
compare:
  reference: mcmc
  metrics: [mean_shift, width_ratio, ks, js, hellinger, kl, energy, sinkhorn]
  sinkhorn_epsilon_factors: [0.25, 1.0]
  draws: 20
  cap: 200
  level: 1.0
  transform: 'models:shift'
compare_signals:
  generator: models:generator
  noise: models:noise
  reference: mcmc
  centre: [0.0, 0.0, 0.0]
  draws: 20
  cap: 200
  level: 1.0
"""
BATCH_SIZE = 3


class GaussianPrior:
    """A new prior for the reweighting fixture: N(0, variance I) up to a constant."""

    def __init__(self, variance: float) -> None:
        """Store the variance."""
        self.variance = variance

    def logpdf(self, samples: object) -> np.ndarray:
        """Return the log density, up to a constant, at samples shaped (N, D)."""
        return -0.5 * np.sum(np.asarray(samples) ** 2, axis=-1) / self.variance

    def rvs(self, size: int | tuple[int, ...], *, key: jax.Array) -> jax.Array:
        """Draw from the prior with the JAX key."""
        shape = (size,) if isinstance(size, int) else size
        return jnp.sqrt(self.variance) * jax.random.normal(key, (*shape, 2))


@pytest.fixture(scope="session")
def report_run_folder(
    pipeline_project: Path, tmp_path_factory: pytest.TempPathFactory
) -> Path:
    """A run folder of the pipeline, with every engine and a transform of compare.

    It holds diagnostics of three posteriors, the comparisons of two pairs with
    the Sinkhorn divergence at two factors, the data-space rows with their chart
    rows, flags at the level 1, and the samples with the mapped copies of compare.
    """
    config_path = pipeline_project / "report.yaml"
    config_path.write_text(REPORT_CONFIG, encoding="utf-8")
    folder = tmp_path_factory.mktemp("report_run")
    with pytest.warns(ValidityWarning, match="fewer than 50 autocorrelation times"):
        run_pipeline(config_path, output=folder)
    return folder


@pytest.fixture(scope="session")
def reweighting_run_folder(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A folder of one result table of reweighting rows: one weight set flags."""
    case = zoo.shifted_gaussians(1.0, param_count=2)
    draws = case.sample(jax.random.key(3), 2_000)
    # * The stored prior is a unit Gaussian, so a wider new prior has heavy weights.
    sample_set = SampleSet(
        draws.samples, log_prior=-0.5 * jnp.sum(draws.samples**2, axis=-1)
    )
    reweighting = reweight(sample_set, case.names, case.space, prior=GaussianPrior(3.5))
    folder = tmp_path_factory.mktemp("reweighting_run")
    reweighting.result.write_parquet(folder / "results.parquet")
    return folder


@pytest.fixture(scope="session")
def batch_run_folder(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A folder of one result table of three problems from the compare engine."""
    case = zoo.shifted_gaussians(1.0, param_count=2)
    draws = case.sample(jax.random.key(4), 400, batch_size=BATCH_SIZE)
    result = run_compare(
        draws,
        case.names,
        case.space,
        key=5,
        metrics=["mean_shift", "width_ratio"],
        draws=20,
        cap=200,
        level=1.0,
    )
    folder = tmp_path_factory.mktemp("batch_run")
    result.write_parquet(folder / "results.parquet")
    return folder


# * Synthetic LISA posteriors in catpylisa's layout: a structured array per source.

LISA_CENTRES: dict[str, dict[str, tuple[float, float]]] = {
    "galactic_binary": {
        "frequency": (3e-3, 1e-9),
        "frequency_dot": (1e-17, 1e-18),
        "amplitude": (1e-22, 1e-23),
        "right_ascension": (1.0, 0.01),
        "declination": (0.3, 0.01),
        "polarization": (0.5, 0.05),
        "inclination": (1.0, 0.05),
        "initial_phase": (2.0, 0.1),
    },
    "massive_black_hole_binary": {
        "primary_mass_det_frame": (1e6, 1e4),
        "secondary_mass_det_frame": (5e5, 1e4),
        "luminosity_distance": (30.0, 0.5),
        "spin1": (0.5, 0.02),
        "spin2": (0.3, 0.02),
        "coalescence_time": (1e7, 100.0),
        "phase_at_reference_time": (1.0, 0.1),
        "frequency_at_reference_time": (1e-4, 1e-8),
        "right_ascension": (1.0, 0.01),
        "declination": (0.3, 0.01),
        "inclination": (1.0, 0.05),
        "polarization": (0.5, 0.05),
    },
    "extreme_mass_ratio_inspiral": {
        "primary_mass_det_frame": (1e6, 1e4),
        "secondary_mass_det_frame": (10.0, 0.1),
        "chi_1": (0.9, 0.01),
        "polar_angle_spin1": (1.0, 0.02),
        "azimuthal_angle_spin1": (2.0, 0.02),
        "semi_latus_rectum_tref": (12.0, 0.1),
        "eccentricity_tref": (0.3, 0.01),
        "azimuthal_orbit_phase_tref": (1.0, 0.1),
        "mean_anomaly_tref": (2.0, 0.1),
        "luminosity_distance": (2.0, 0.05),
        "right_ascension": (1.0, 0.01),
        "declination": (0.3, 0.01),
        "inclination_tref": (0.8, 0.02),
        "polar_orbit_phase_tref": (0.5, 0.1),
    },
    "stellar_origin_binary_black_hole": {
        "primary_mass_det_frame": (35.0, 0.5),
        "secondary_mass_det_frame": (30.0, 0.5),
        "spin_1": (0.2, 0.02),
        "spin_2": (0.1, 0.02),
        "f_ref": (0.01, 1e-6),
        "inclination": (1.0, 0.05),
        "polarization": (0.5, 0.05),
        "phase_tref": (1.0, 0.1),
        "luminosity_distance": (0.5, 0.01),
        "right_ascension": (1.0, 0.01),
        "declination": (0.3, 0.01),
        "eccentricity_tref": (0.05, 0.005),
        "polar_angle_spin1": (1.0, 0.02),
        "polar_angle_spin2": (1.5, 0.02),
        "azimuthal_angle_spin1": (2.0, 0.02),
        "azimuthal_angle_spin2": (3.0, 0.02),
    },
}


def build_lisa_posteriors(
    source_type: str,
    counts: Mapping[str, int],
    *,
    seed: int = 0,
    shift: float = 0.0,
    constant: tuple[str, ...] = (),
) -> dict[str, np.ndarray]:
    """Draw posteriors of LISA sources as structured arrays with the L2 field names.

    Each source gets its own number of samples, from Gaussians around plausible
    values. A ``shift`` moves every centre by that many widths, and the fields
    in ``constant`` take their unshifted centre value in every sample.
    """
    centres = LISA_CENTRES[source_type]
    rng = np.random.default_rng(seed)
    dtype = [(name, "f8") for name in (*centres, "loglikelihood", "logprior")]
    posteriors = {}
    for source_id, count in counts.items():
        table = np.zeros(count, dtype=dtype)
        for name, (centre, width) in centres.items():
            if name in constant:
                table[name] = centre
            else:
                table[name] = centre + (shift + rng.normal(size=count)) * width
        table["loglikelihood"] = rng.normal(size=count)
        table["logprior"] = rng.normal(size=count)
        posteriors[source_id] = table
    return posteriors


@pytest.fixture
def lisa_posteriors() -> Callable[..., dict[str, np.ndarray]]:
    """The builder of synthetic LISA posteriors in the layout of catpylisa."""
    return build_lisa_posteriors


def build_result(*rows: Mapping[str, Any]) -> Result:
    """Build a result table from rows that name only what a test needs.

    Every row is a diagnostic row of the posterior ``a`` unless it says otherwise.
    """
    defaults: dict[str, Any] = {
        "problem": 0,
        "pair": ["a"],
        "metric": "skewness",
        "estimator": "empirical",
        "estimator_settings": "{}",
        "space": "identity",
        "parameters": ["x0"],
        "value": 0.0,
        "warnings": [],
        "provenance": Provenance.collect(seed=1),
    }
    return Result.from_rows([{**defaults, **row} for row in rows])


@pytest.fixture
def result_of() -> Callable[..., Result]:
    """The builder of small result tables, for the tests of the report tables."""
    return build_result
