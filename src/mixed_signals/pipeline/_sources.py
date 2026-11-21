"""The objects and inputs of a pipeline run: import paths, the space and posteriors.

The config of a pipeline names code by import path and data by file, and this
module turns both into objects and keeps a record of where each came from, for
``provenance.json``. It is private: its names may change with the runner.

Import paths
    :class:`Sources.resolve` resolves ``module:attribute`` with the standard
    library's ``pkgutil.resolve_name`` while the folder of the config comes first
    on ``sys.path`` (:func:`config_folder_on_path`), as ``python script.py`` puts
    the folder of a script there. It records the module of every resolved path:
    its name, its file and the SHA-256 of the bytes of that file (National
    Institute of Standards and Technology 2015). Modules stay in ``sys.modules``
    afterwards, and Python imports a module once per process. A module of the
    config folder whose top-level name an imported module already has, from
    another file or from none, would silently give way to that module, so
    :meth:`Sources.resolve` raises a :class:`~mixed_signals.errors.ConfigError`
    that names both files; the same folder resolving the same module again is
    fine. Run a config in its own process, as the ``mixed-signals`` command
    always does, or give the module a name that no imported module has. For the
    same reason, an edit of a module takes effect in a new process.
Inputs
    Each posterior entry reads into a
    :class:`~mixed_signals.core.chains.Chains` object (``eryn``, ``chains``) or a
    :class:`~mixed_signals.core.samples.SampleSet` (``samples``, ``table``,
    ``hdf5``). The reader of a format finds the columns by the names of the
    space where the file has names. Every file that a config reads, such as the
    posteriors, a space file and arrays, enters the input records with its path
    as written, its absolute path, its size and its SHA-256, which is computed in
    blocks because eryn files can be large.
Errors
    A file that cannot be read, and every error that a reader raises for a value
    of the file, become :class:`~mixed_signals.errors.ConfigError` with the
    posterior and its path named. A missing optional dependency passes
    unchanged, because installing it is the fix.

References:
    :cite:t:`nist2015secure`.
"""

import contextlib
import hashlib
import importlib
import pkgutil
import sys
from collections.abc import Generator
from pathlib import Path
from typing import Any, Final, NamedTuple

import numpy as np
import yaml

from mixed_signals.core.chains import Chains
from mixed_signals.core.samples import SampleSet
from mixed_signals.core.space import ParamSpace
from mixed_signals.errors import ConfigError
from mixed_signals.pipeline.config import (
    ChainsEntry,
    ErynEntry,
    PosteriorEntry,
    SamplesEntry,
    SpaceSpec,
    TableEntry,
)
from mixed_signals.priors.eryn_container import PHYSICAL_DESCRIPTION, ErynPhysical
from mixed_signals.transforms.base import Transform

__all__ = [
    "InputRecord",
    "ModuleRecord",
    "Sources",
    "config_folder_on_path",
]

_BLOCK_BYTES: Final = 1_048_576
"""Bytes read at a time when a file is hashed."""

# * Errors that a reader raises for a file or its values, which the config owns.
_READER_ERRORS: Final = (OSError, ValueError, TypeError, KeyError, NotImplementedError)


class ModuleRecord(NamedTuple):
    """The module behind one import path of the config."""

    import_path: str
    """The import path as written, ``module:attribute``."""
    module: str
    """The module name."""
    file: str | None
    """The ``__file__`` of the module; ``None`` for a module without one."""
    sha256: str | None
    """The SHA-256 of the bytes of that file; ``None`` without a readable file."""


class InputRecord(NamedTuple):
    """One file that the config reads."""

    role: str
    """What the file is: ``posterior <name>``, ``space`` or ``<section> <array>``."""
    path: str
    """The path as written in the config."""
    absolute_path: str
    """The resolved absolute path."""
    size_bytes: int
    """The size of the file in bytes."""
    sha256: str
    """The SHA-256 of the bytes of the file."""


@contextlib.contextmanager
def config_folder_on_path(folder: Path) -> Generator[None]:
    """Put a folder first on ``sys.path`` for a block, and remove that entry after.

    Args:
        folder: The folder of the config file.

    Yields:
        Nothing; modules imported inside the block stay in ``sys.modules``.
    """
    entry = str(folder)
    sys.path.insert(0, entry)
    # ? Without this, a module written after the folder was listed may not be found.
    importlib.invalidate_caches()
    try:
        yield
    finally:
        # * Only the entry that this block added: remove() takes the first match.
        with contextlib.suppress(ValueError):
            sys.path.remove(entry)


def file_sha256(path: Path) -> str:
    """Return the SHA-256 hex digest of a file, read in blocks.

    References:
        :cite:t:`nist2015secure`.
    """
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(_BLOCK_BYTES):
            digest.update(block)
    return digest.hexdigest()


class Sources:
    """Resolve the code and read the inputs of one config, recording each source.

    Args:
        base: The folder of the config file, where relative paths start.

    Attributes:
        base: The folder of the config file.
        modules: The module record of each resolved import path, in order.
        inputs: The record of each file that was read, in order.
    """

    def __init__(self, base: Path) -> None:
        """Start with no records."""
        self.base = base
        self.modules: dict[str, ModuleRecord] = {}
        self.inputs: list[InputRecord] = []

    def resolve(self, reference: str) -> object:
        """Resolve an import path ``module:attribute`` to its object.

        Args:
            reference: The import path.

        Returns:
            The object that the path names.

        Raises:
            ConfigError: If the module or the attribute cannot be found, or a
                module of the same name was imported from another file.
        """
        self._refuse_clash(reference)
        try:
            with config_folder_on_path(self.base):
                value = pkgutil.resolve_name(reference)
        except (ImportError, AttributeError, ValueError) as error:
            raise ConfigError(
                f"cannot resolve the import path {reference!r}: {error}"
            ) from error
        module_name = reference.partition(":")[0]
        module_file: str | None = getattr(sys.modules[module_name], "__file__", None)
        module_path = None if module_file is None else Path(module_file)
        self.modules[reference] = ModuleRecord(
            import_path=reference,
            module=module_name,
            file=module_file,
            sha256=(
                file_sha256(module_path)
                if module_path is not None and module_path.is_file()
                else None
            ),
        )
        return value

    def _refuse_clash(self, reference: str) -> None:
        """Raise if the process imported the top-level module from another file.

        Python imports a module once per process, so a module of the config
        folder would silently give way to an imported module of the same name.
        """
        top = reference.partition(":")[0].partition(".")[0]
        imported = sys.modules.get(top)
        if imported is None:
            return
        # * Import order: a package directory comes before a module file of its name.
        local = [
            candidate
            for candidate in (self.base / top / "__init__.py", self.base / f"{top}.py")
            if candidate.is_file()
        ]
        if not local:
            return
        imported_file: str | None = getattr(imported, "__file__", None)
        if imported_file is not None and Path(imported_file).resolve() == (
            local[0].resolve()
        ):
            return
        raise ConfigError(
            f"the module {top!r} of the import path {reference!r} was already"
            f" imported from {imported_file or 'no file'}, but the config folder"
            f" holds {local[0]}; Python imports a module once per process, so run"
            " the config in its own process, as the mixed-signals command always"
            " does, or give the module a name that no imported module has"
        )

    def read_space(self, spec: SpaceSpec) -> tuple[ParamSpace, object | None]:
        """Build the parameter space of the config.

        Args:
            spec: The path of a space file, an inline space or eryn priors.

        Returns:
            The space, and the eryn priors when the space comes from them.

        Raises:
            ConfigError: If the file, the priors or the space do not resolve.
        """
        if isinstance(spec, ParamSpace):
            return spec, None
        if isinstance(spec, str):
            path = self.base / spec
            try:
                space = ParamSpace.from_yaml(path)
            except (OSError, ValueError, yaml.YAMLError) as error:
                raise ConfigError(
                    f"cannot read the space file {spec!r}: {error}"
                ) from error
            self._record("space", spec, path)
            return space, None
        priors = self.resolve(spec.eryn_priors)
        try:
            space = ParamSpace.from_eryn_priors(
                priors, periodic=spec.periodic, spheres=spec.spheres
            )
        except (TypeError, ValueError) as error:
            raise ConfigError(
                f"cannot build the space from the eryn priors {spec.eryn_priors!r}:"
                f" {error}"
            ) from error
        return space, priors

    def read_transform(
        self, spec: str | None, priors: object | None
    ) -> Transform | None:
        """Build the transform of a section.

        Args:
            spec: ``None``, ``physical`` or the import path of a transform.
            priors: The eryn priors of the space, which ``physical`` needs.

        Returns:
            The transform, or ``None`` for no transform.

        Raises:
            ConfigError: If the import path does not name a transform, or the
                priors do not give one.
        """
        if spec is None:
            return None
        if spec == "physical":
            try:
                return ErynPhysical(priors)
            except (TypeError, ValueError) as error:
                raise ConfigError(
                    f"cannot build the {PHYSICAL_DESCRIPTION} transform: {error}"
                ) from error
        value = self.resolve(spec)
        if not isinstance(value, Transform):
            raise ConfigError(
                f"the import path {spec!r} names a {type(value).__name__}, not a"
                " Transform"
            )
        return value

    def read_array(self, spec: str | list[Any], role: str) -> np.ndarray:
        """Return an array of the config as float64.

        Args:
            spec: A nested list of numbers, or the path of a ``.npy`` file.
            role: What the array is, for the input record and messages.

        Returns:
            The array.

        Raises:
            ConfigError: If the file cannot be read.
        """
        if not isinstance(spec, str):
            return np.asarray(spec, dtype=np.float64)
        path = self.base / spec
        try:
            values = np.asarray(np.load(path, allow_pickle=False), dtype=np.float64)
        except (OSError, ValueError, TypeError) as error:
            raise ConfigError(f"cannot read {role} from {spec!r}: {error}") from error
        self._record(role, spec, path)
        return values

    def read_posterior(
        self, entry: PosteriorEntry, space: ParamSpace, priors: object | None
    ) -> Chains | SampleSet:
        """Read one posterior of the config.

        Args:
            entry: The posterior entry.
            space: The parameter space, whose names choose the columns.
            priors: The eryn priors of the space, if it comes from them.

        Returns:
            The chains of an ``eryn`` or ``chains`` entry, or the sample set of
            another entry.

        Raises:
            ConfigError: If the file is missing or a value of the file does not
                fit, with the posterior and its path named.
            MissingDependencyError: If a reader needs an optional dependency
                that is not installed.
        """
        path = self.base / entry.path
        try:
            value = _read_entry(entry, path, space, priors)
        except _READER_ERRORS as error:
            raise ConfigError(
                f"cannot read the posterior {entry.name!r} from {entry.path!r}: {error}"
            ) from error
        self._record(f"posterior {entry.name}", entry.path, path)
        return value

    def _record(self, role: str, written: str, path: Path) -> None:
        """Add the record of a file that was read."""
        self.inputs.append(
            InputRecord(
                role=role,
                path=written,
                absolute_path=str(path.resolve()),
                size_bytes=path.stat().st_size,
                sha256=file_sha256(path),
            )
        )


def _read_entry(
    entry: PosteriorEntry, path: Path, space: ParamSpace, priors: object | None
) -> Chains | SampleSet:
    """Read the file of a posterior entry with the reader of its format."""
    if isinstance(entry, ErynEntry):
        return Chains.from_eryn(
            path,
            branch=entry.branch,
            num_repeats_in_model=entry.num_repeats_in_model,
            priors=priors,
            parameters=space.names,
        )
    if isinstance(entry, ChainsEntry):
        chains = Chains.from_npz(path)
        _check_parameter_count(chains.shape.param, space)
        return chains
    if isinstance(entry, SamplesEntry):
        sample_set = SampleSet.from_npz(path, n_eff=entry.n_eff)
        _check_parameter_count(sample_set.shape.param, space)
        return sample_set
    if isinstance(entry, TableEntry):
        return SampleSet.from_table(
            path, space.names, log_weights=entry.log_weights, n_eff=entry.n_eff
        )
    return SampleSet.from_hdf5(
        path,
        entry.group,
        space.names,
        log_weights=entry.log_weights,
        n_eff=entry.n_eff,
    )


def _check_parameter_count(count: int, space: ParamSpace) -> None:
    """Raise unless a file without parameter names has as many as the space."""
    if count != len(space):
        raise ValueError(
            f"the file has {count} parameters, but the space has {len(space)}; files"
            " without names are read in order"
        )
