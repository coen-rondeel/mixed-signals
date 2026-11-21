"""Shared helpers and fixtures: the zoo and estimator tests, and the pipeline data."""

import functools
import shutil
import sys
import types
from collections.abc import Callable
from pathlib import Path

import jax
import numpy as np
import polars as pl
import pytest

from mixed_signals import zoo

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
