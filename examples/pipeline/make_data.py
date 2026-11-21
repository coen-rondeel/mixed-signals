"""Write the data of the pipeline example: three posteriors of one toy problem.

The problem is the linear signal of the validation zoo, whose posterior is a
Gaussian with Fisher matrix ``models.FISHER`` around ``models.CENTRE``. The
script writes the files that ``config.yaml`` reads into ``data/``:

``mcmc.h5``
    The reference run, from an eryn ensemble sampler with an HDF5 backend on the
    Gaussian log-likelihood -1/2 (theta - centre)^T Gamma (theta - centre) and
    the priors of ``models.py``.
``sbi.parquet``
    A flat table of draws from the same Gaussian shifted by 0.3 posterior
    widths, one column per parameter, as the output of a simulation-based
    inference method would be.
``peer.h5``
    Flat draws from the unshifted Gaussian, drawn with another key, in the group
    ``posterior`` with one dataset per parameter.
``centre.npy``
    The centre of the Gaussian, which the data-space comparison of the config
    reads, so that the numbers of the case live in ``models.py`` alone.

Run it from the repository root with ``uv run python
examples/pipeline/make_data.py``. The seed fixes the draws, so the files repeat:
eryn copies NumPy's global random state into its own generator when the sampler
is built, so the script seeds NumPy before that.
"""

from pathlib import Path

import h5py
import models
import numpy as np
import polars as pl
from eryn.backends import HDFBackend
from eryn.ensemble import EnsembleSampler

from mixed_signals import root_key
from mixed_signals.prng import derive_key

DATA_FOLDER = Path(__file__).resolve().parent / "data"
SEED = 2026
SHIFT = 0.3  # Mahalanobis length of the shift of the table, in posterior widths.
WALKER_COUNT = 32
STEP_COUNT = 3_000
SAMPLE_COUNT = 3_000  # Draws of the table and of the group.


def log_likelihood(points: np.ndarray, *args: object) -> np.ndarray:
    """Return the Gaussian log-likelihood of the offset from the centre."""
    offsets = points - models.CENTRE
    return -0.5 * np.einsum("...i,ij,...j->...", offsets, models.FISHER, offsets)


def write_mcmc(path: Path) -> None:
    """Run the eryn sampler and store its chains in an HDF5 file."""
    path.unlink(missing_ok=True)
    # ! The sampler copies the global state when it is built, so seed before that.
    np.random.seed(SEED)  # noqa: NPY002
    sampler = EnsembleSampler(
        WALKER_COUNT,
        len(models.CENTRE),
        log_likelihood,
        models.priors,
        tempering_kwargs={"ntemps": 1},
        backend=HDFBackend(str(path)),
    )
    start = models.priors.rvs(size=(1, WALKER_COUNT))
    sampler.run_mcmc(start, STEP_COUNT, progress=False)


def main() -> None:
    """Write the data files of the example into the data folder."""
    DATA_FOLDER.mkdir(exist_ok=True)
    write_mcmc(DATA_FOLDER / "mcmc.h5")
    case = models.case.fisher_gaussians(SHIFT)
    root = root_key(SEED)
    shifted = case.sample(derive_key(root, "sbi"), SAMPLE_COUNT)
    reference = case.sample(derive_key(root, "peer"), SAMPLE_COUNT)
    names = models.case.space.names
    shifted_draws = np.asarray(shifted.samples[0, 1])
    pl.DataFrame(
        {name: shifted_draws[:, position] for position, name in enumerate(names)}
    ).write_parquet(DATA_FOLDER / "sbi.parquet")
    reference_draws = np.asarray(reference.samples[0, 0])
    with h5py.File(DATA_FOLDER / "peer.h5", "w") as file:
        group = file.create_group("posterior")
        for position, name in enumerate(names):
            group.create_dataset(name, data=reference_draws[:, position])
    np.save(DATA_FOLDER / "centre.npy", models.CENTRE)
    print(f"Wrote mcmc.h5, sbi.parquet, peer.h5 and centre.npy to {DATA_FOLDER}")


if __name__ == "__main__":
    main()
