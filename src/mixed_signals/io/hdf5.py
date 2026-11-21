"""HDF5 groups of one dataset per parameter as sample sets.

:func:`sample_set_from_hdf5` backs
:meth:`SampleSet.from_hdf5 <mixed_signals.core.samples.SampleSet.from_hdf5>`.

Layout
    A group holds one dataset per parameter, named as the parameter, so that
    the path of a dataset is ``<group>/<parameter name>``, such as
    ``/posterior/frequency`` or ``/results/samples/chain_0/amplitude``. The
    path of the group is a function argument, so a file may hold several
    posteriors in different groups. Datasets shaped (N,) are one problem and
    datasets shaped (B, N) are B problems, all of one shape. The names that the
    caller passes, those of the parameter space, choose the datasets in their
    order, so the group may hold more datasets, such as the log-likelihood, and
    a space may name a subset of the parameters. An optional dataset of
    natural-log weights sits in the same group with the same shape. The result
    follows the one-posterior layout of :mod:`mixed_signals.io.files`.
Optional extra
    The reader needs h5py, which the ``hdf5`` extra installs
    (``pip install "mixed-signals[hdf5]"``). The module imports h5py inside the
    reader, so ``import mixed_signals`` works without it.
Access
    The file opens read-only, so a read never changes a file of results.
"""

import importlib
import os
from collections.abc import Sequence
from types import ModuleType
from typing import Final

import numpy as np

from mixed_signals.core.samples import SampleSet
from mixed_signals.errors import MissingDependencyError
from mixed_signals.io.files import _check_names, _flat_sample_set, _warn_narrow_arrays

__all__ = ["sample_set_from_hdf5"]

_INSTALL_HINT: Final = (
    'install it with the hdf5 extra: pip install "mixed-signals[hdf5]"'
)
_ROOT_GROUPS: Final = ("", "/")
_NUMERIC_KINDS: Final = "iuf"  # NumPy type codes: signed, unsigned and float.


def sample_set_from_hdf5(
    path: str | os.PathLike[str],
    group: str,
    names: Sequence[str],
    *,
    log_weights: str | None = None,
    n_eff: object = None,
) -> SampleSet:
    """Read a sample set of one posterior from the datasets of an HDF5 group.

    :meth:`SampleSet.from_hdf5 <mixed_signals.core.samples.SampleSet.from_hdf5>`
    calls this function; the module docstring describes the layout.

    Args:
        path: An HDF5 file, opened read-only.
        group: The path of the group inside the file; ``""`` or ``"/"`` for the
            root.
        names: The parameter datasets, in the order of the last sample axis,
            which are the names of the parameter space.
        log_weights: The dataset of natural-log weights in the same group, not
            among ``names``; ``None`` for equal weights.
        n_eff: The effective sample size, a number when the datasets are shaped
            (N,) or one number per problem when they are shaped (B, N); the
            Kish value of the weights when ``None``.

    Returns:
        The set, shaped (B, 1, N, D).

    Raises:
        ValueError: If the names are empty, repeated or not strings, the log
            weights are among the names, the group or a dataset is missing, a
            member is not a dataset, the shapes are not (N,) or (B, N) and
            equal, or the container rejects a size or a value.
        TypeError: If a dataset is not integer or float.
        OSError: If the file cannot be read.
        MissingDependencyError: If h5py is not installed.

    Warns:
        PrecisionWarning: If a dataset is narrower than float64.
    """
    h5py = _import_h5py()
    parameter_names = _check_names(names)
    if log_weights is not None and log_weights in parameter_names:
        raise ValueError(
            f"log_weights names the dataset {log_weights!r}, which is also a"
            " parameter name"
        )
    wanted = [*parameter_names, *([] if log_weights is None else [log_weights])]
    with h5py.File(path, "r") as file:
        node = file["/"] if group in _ROOT_GROUPS else file.get(group)
        if not isinstance(node, h5py.Group):
            raise ValueError(f"{os.fspath(path)} has no group {group!r}")
        members = {name: node.get(name) for name in wanted}
        datasets = [key for key in node if isinstance(node.get(key), h5py.Dataset)]
        missing = [name for name, member in members.items() if member is None]
        if missing:
            raise ValueError(
                f"group {group!r} of {os.fspath(path)} has no datasets"
                f" {', '.join(map(repr, missing))}; its datasets are"
                f" {', '.join(map(repr, datasets))}"
            )
        columns = {
            name: _read_dataset(h5py, group, name, member)
            for name, member in members.items()
        }
    if len({values.shape for values in columns.values()}) > 1:
        shape_text = ", ".join(
            f"{name!r} {values.shape}" for name, values in columns.items()
        )
        raise ValueError(
            f"the datasets of group {group!r} must share one shape, got {shape_text}"
        )
    _warn_narrow_arrays(columns, "dataset")
    arrays: dict[str, object] = {
        "samples": np.stack(
            [columns[name].astype(np.float64) for name in parameter_names], axis=-1
        ),
        "n_eff": n_eff,
    }
    if log_weights is not None:
        arrays["log_weights"] = columns[log_weights].astype(np.float64)
    return _flat_sample_set(arrays)


def _import_h5py() -> ModuleType:
    """Import h5py, which the optional ``hdf5`` extra installs."""
    try:
        return importlib.import_module("h5py")
    except ImportError as error:
        raise MissingDependencyError(
            f"cannot import h5py ({error}); {_INSTALL_HINT}"
        ) from error


def _read_dataset(
    h5py: ModuleType, group: str, name: str, member: object
) -> np.ndarray:
    """Return a numeric dataset of one or two axes as a NumPy array."""
    if not isinstance(member, h5py.Dataset):
        raise ValueError(f"{name!r} in group {group!r} is not a dataset")
    if member.dtype.kind not in _NUMERIC_KINDS:
        raise TypeError(
            f"dataset {name!r} must be integer or float, got dtype {member.dtype}"
        )
    if len(member.shape) not in (1, 2):
        raise ValueError(
            f"dataset {name!r} must be shaped (N,) or (B, N), got shape {member.shape}"
        )
    values: np.ndarray = member[()]
    return values
