"""The YAML config of the pipeline: its models, the loader and the config hash.

A config names the posteriors to read, the parameter space and the engines to
run. :func:`load_config` reads and validates a file, and
:func:`~mixed_signals.pipeline.runner.run_pipeline` reads the data and runs the
engines. This page is the reference of the format. An annotated example with
every section::

    seed: 42                    # integer in [0, 2**63); every key derives from it
    output: results             # run folder, relative to the config folder
    budget: 3600                # estimated seconds above which one CostWarning follows
    space:                      # a ParamSpace file, an inline space or eryn priors
      eryn_priors: models:priors
      periodic: {phase: 6.283185307179586}
    posteriors:
      - {name: mcmc, format: eryn, path: data/mcmc.h5, burn_in: 200, thin: tau}
      - {name: chains, format: chains, path: data/ar1.npz, burn_in: 0, thin: 5}
      - {name: draws, format: samples, path: data/draws.npz, n_eff: 800}
      - {name: sbi, format: table, path: data/sbi.parquet, log_weights: log_w}
      - {name: peer, format: hdf5, path: data/peer.h5, group: posterior}
    diagnose:                   # an empty mapping runs the engine with its defaults
      diagnostics: [shape, dependence]
      draws: 100
      thresholds: {skewness: 0.4}
    compare:
      reference: mcmc           # mode A against this posterior; omit it for every pair
      metrics: [energy, "kl:gaussian"]
      transform: physical       # ErynPhysical of the eryn priors
      thresholds: {energy: 0.3}
    compare_signals:
      generator: models:generator
      noise: models:noise
      reference: mcmc
      centre: [1.0, 2.0, 3.0]   # a nested list, or the path of a .npy file

Sections
    ``diagnose``, ``compare`` and ``compare_signals`` run the engines of the
    same names, on every posterior of the config, and a config needs at least
    one of them. ``reweight`` stays in the Python API. Each section mirrors the
    keyword arguments of its engine by name, with the defaults of the engine,
    which the models read from the engine's constants: ``draws`` is
    :data:`~mixed_signals._config.NULL_DRAWS`, ``cap`` is
    :data:`~mixed_signals._config.SAMPLE_CAP`, and so on. An engine runs only
    when its section is present.
Posteriors
    Each entry has a unique ``name``, a ``format`` and a ``path``. One file holds
    one posterior. The formats are ``eryn`` (an eryn HDF5 file, with ``branch``,
    ``num_repeats_in_model``, ``burn_in`` and ``thin``), ``chains`` (an ``.npz``
    archive with the keyword names of :class:`~mixed_signals.core.chains.Chains`,
    with ``burn_in`` and ``thin``), ``samples`` (an ``.npz`` archive with the
    keyword names of :class:`~mixed_signals.core.samples.SampleSet`, or an
    ``.npy`` file of samples, with ``n_eff``), ``table`` (parquet or CSV, with
    ``log_weights`` and ``n_eff``) and ``hdf5`` (a group of one dataset per
    parameter, with ``group``, ``log_weights`` and ``n_eff``). A key of another
    format is invalid. A run of chains is reduced by
    :func:`~mixed_signals.mcmc.reduce.reduce_chains` with its ``burn_in``, which
    is 0 unless given, and its ``thin``, ``"tau"`` or an integer factor. Flat
    samples are (N, D) for one problem or (B, N, D) for B problems.
Reading by name
    Tables, HDF5 groups and eryn files name their columns, so the names of the
    space choose the columns in the order of the space, and a space may hold a
    subset of the sampled parameters. A missing name is an error. ``.npz`` and
    ``.npy`` files carry no names, so their last axis is read in order and must
    match the number of parameters. The readers are described in
    :mod:`mixed_signals.io.files`, :mod:`mixed_signals.io.hdf5` and
    :mod:`mixed_signals.io.eryn_backend`. With eryn priors, the columns of
    ``DeltaFunction`` parameters drop first, and a config has one set of
    sampling coordinates.
Space
    ``space`` is required and has three forms: the path of a
    :class:`~mixed_signals.core.space.ParamSpace` YAML file, an inline space
    (a mapping with ``parameters`` and optional ``spheres``), or a mapping with
    ``eryn_priors``, an import path of an eryn ``ProbDistContainer``, and
    optional ``periodic`` and ``spheres``, which
    :meth:`ParamSpace.from_eryn_priors
    <mixed_signals.core.space.ParamSpace.from_eryn_priors>` reads.
Transforms
    Each section has its own ``transform``: none, ``physical`` or an import path
    to a :class:`~mixed_signals.transforms.base.Transform` object.
    ``physical`` is :class:`~mixed_signals.priors.eryn_container.ErynPhysical`
    of the eryn priors, so it needs the ``eryn_priors`` form of the space.
Import paths
    A config names Python objects by import path ``module:attribute``, as the
    entry points of Python packages do, and the standard library's
    ``pkgutil.resolve_name`` resolves them. An import path names an attribute,
    not a call, so a factory with arguments needs a variable in a module of
    the user. While the pipeline resolves and runs, the folder of the config
    file comes first on ``sys.path``, as ``python script.py`` puts the folder of
    a script there, so a ``models.py`` next to the config imports as
    ``models``. A config therefore runs code, like a script: run configs that
    you trust. YAML is read with the safe loader, so a tag cannot build a
    Python object. The run folder records the file and the SHA-256 of the module
    of every import path. Python imports a module once per process, so a process
    that has imported a module of the same name from another file refuses the
    config, with both files named, instead of using the wrong module; run the
    config in its own process, as the ``mixed-signals`` command always does, or
    give the module a name that no imported module has. For the same reason, an
    edit of a module takes effect in a new process.
Paths
    Every relative path of a config, in the posteriors, the space file, the
    ``.npy`` arrays and ``output``, starts at the folder of the config file, not
    at the working folder of the shell. A config validated from a Python
    mapping starts them at the working folder. An array, such as ``fisher`` or
    ``centre``, is a nested YAML list of numbers or the path of a ``.npy`` file.
Strict keys
    An unknown key and a key that a mapping repeats are invalid, with the entry
    named. Counts and the seed are YAML integers, so a bool, a float or a string
    is invalid; a float field takes integers and floats, but not bools or
    strings. The ``null`` field of ``compare`` and ``compare_signals`` names the
    construction of the null; YAML reads an unquoted ``null`` as its null value,
    so the loader takes an unquoted ``null`` key as the field of that name. YAML
    merge keys (``<<: *anchor``) work, and an explicit key overrides the key
    that a merge brings in.
Thresholds
    A ``thresholds`` mapping gives a float for each metric to override, merged
    over the package defaults: :data:`mixed_signals.diagnostics.flags.THRESHOLDS`
    for ``diagnose``, where a metric keeps its default side and tail, and
    :data:`mixed_signals.sigma.thresholds.THRESHOLDS` for ``compare`` and
    ``compare_signals``. An unknown metric is invalid. After validation the
    field holds the full merged mapping, so a dump shows every threshold that
    runs.
Whitening
    ``whitening`` of ``compare`` is ``reference``, ``pooled``, ``fisher`` or
    ``null``, and a null takes the default of the engine: ``reference`` when
    ``reference`` is set and ``pooled`` otherwise. ``fisher`` and ``centre``
    belong to the ``fisher`` whitening of ``compare``.
Config hash
    The hash is the SHA-256 of the canonical JSON of the validated config with
    every default filled in and the paths as written, without ``output`` and
    ``budget``, and of the resolved parameter space, with sorted keys and
    compact separators. The run folder keeps those bytes as ``config.json``, so
    ``sha256sum config.json`` gives the hash that every result row records
    (National Institute of Standards and Technology 2015). The output folder
    and the budget stay out of the hash, because neither changes a row: one
    config can run into several folders, and a different budget only changes
    whether the run warns.
Run folder
    :mod:`mixed_signals.pipeline.runner` describes the run and its files.

References:
    :cite:t:`nist2015secure`.
"""

import hashlib
import json
import os
import re
from collections.abc import Hashable, Mapping
from functools import partial
from pathlib import Path
from types import MappingProxyType
from typing import Annotated, Any, Final, Literal

import numpy as np
import yaml
from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Discriminator,
    Field,
    PrivateAttr,
    Tag,
    model_validator,
)

from mixed_signals._config import COST_BUDGET, NULL_DRAWS, SAMPLE_CAP
from mixed_signals.core.result import UncertaintyMethod
from mixed_signals.core.space import ParamSpace
from mixed_signals.core.topology import Sphere
from mixed_signals.diagnostics import flags
from mixed_signals.diagnostics.dependence import PAIR_CAP
from mixed_signals.engines.compare import DEFAULT_METRICS
from mixed_signals.engines.diagnose import DIAGNOSTIC_FAMILIES
from mixed_signals.errors import ConfigError
from mixed_signals.mcmc.reduce import TAU_THINNING
from mixed_signals.metrics.ot.sinkhorn import SINKHORN_EPSILON_FACTOR
from mixed_signals.nulls.plan import Construction
from mixed_signals.prng import SEED_LIMIT
from mixed_signals.sigma import thresholds as sigma_thresholds
from mixed_signals.sigma.thresholds import SIGNIFICANCE_LEVEL
from mixed_signals.sigma.uncertainty import HALF_SAMPLE_PAIRS

__all__ = [
    "ChainsEntry",
    "CompareSection",
    "CompareSignalsSection",
    "DiagnoseSection",
    "ErynEntry",
    "ErynSpace",
    "Hdf5Entry",
    "PipelineConfig",
    "SamplesEntry",
    "TableEntry",
    "load_config",
]

_IDENTIFIER: Final = r"[^\W\d]\w*"
IMPORT_PATH_PATTERN: Final = re.compile(
    rf"{_IDENTIFIER}(?:\.{_IDENTIFIER})*:{_IDENTIFIER}(?:\.{_IDENTIFIER})*"
)
"""Pattern of an import path ``module:attribute``, each side dotted identifiers."""

_DIAGNOSE_THRESHOLDS: Final = MappingProxyType(
    {metric: threshold.value for metric, threshold in flags.THRESHOLDS.items()}
)
_COMPARE_THRESHOLDS: Final = MappingProxyType(dict(sigma_thresholds.THRESHOLDS))

# * Strict numbers: YAML integers and floats pass, bools and strings do not.
FiniteNumber = Annotated[float, Field(strict=True, allow_inf_nan=False)]


def _check_import_path(value: str) -> str:
    """Raise unless the text has the form ``module:attribute``."""
    if IMPORT_PATH_PATTERN.fullmatch(value) is None:
        raise ValueError(
            f"{value!r} is not an import path 'module:attribute'; each side is"
            " dotted Python identifiers, such as 'models:priors'"
        )
    return value


def _check_name(value: str) -> str:
    """Raise unless the name is non-empty and has no surrounding whitespace."""
    if not value or value != value.strip():
        raise ValueError(
            f"a name must be non-empty, without surrounding whitespace, got {value!r}"
        )
    return value


def _is_number_tree(value: object) -> bool:
    """Return whether a value is a number, or a non-empty list of such trees."""
    if isinstance(value, list):
        return bool(value) and all(_is_number_tree(entry) for entry in value)
    return isinstance(value, int | float) and not isinstance(value, bool)


def _check_array(value: str | list[Any]) -> str | list[Any]:
    """Raise unless the value is a ``.npy`` path or a rectangular list of numbers."""
    if isinstance(value, str):
        if not value.endswith(".npy"):
            raise ValueError(f"an array path must end in .npy, got {value!r}")
        return value
    if not _is_number_tree(value):
        raise ValueError(
            "an array is a nested list of numbers or the path of a .npy file"
        )
    try:
        values = np.asarray(value, dtype=np.float64)
    except ValueError as error:
        raise ValueError(f"the nested list is not rectangular: {error}") from error
    if not np.all(np.isfinite(values)):
        raise ValueError("an array must hold finite numbers")
    return value


def _merge_thresholds(
    defaults: Mapping[str, float], overrides: Mapping[str, float]
) -> dict[str, float]:
    """Return the defaults with the overrides on top; unknown metrics are invalid."""
    unknown = sorted(set(overrides) - set(defaults))
    if unknown:
        raise ValueError(
            f"unknown threshold metrics {', '.join(map(repr, unknown))}; the metrics"
            f" are {', '.join(defaults)}"
        )
    return {**defaults, **overrides}


ImportPath = Annotated[str, AfterValidator(_check_import_path)]
"""A text ``module:attribute``, resolved when the pipeline prepares a run."""

Name = Annotated[str, AfterValidator(_check_name)]
ArraySpec = Annotated[str | list[Any], AfterValidator(_check_array)]
"""A nested list of numbers, or the path of a ``.npy`` file."""

TransformSpec = Literal["physical"] | ImportPath | None
"""No transform, ``physical`` for eryn's map, or the import path of a transform."""

ThinSpec = Annotated[int, Field(strict=True, ge=1)] | Literal["tau"]
"""``"tau"`` or an integer thinning factor of at least 1."""

DiagnoseThresholds = Annotated[
    dict[str, FiniteNumber],
    AfterValidator(partial(_merge_thresholds, _DIAGNOSE_THRESHOLDS)),
]
CompareThresholds = Annotated[
    dict[str, FiniteNumber],
    AfterValidator(partial(_merge_thresholds, _COMPARE_THRESHOLDS)),
]


class _Model(BaseModel):
    """Base of the config models: frozen, and unknown keys are invalid."""

    model_config = ConfigDict(frozen=True, extra="forbid", ser_json_inf_nan="constants")


class ErynSpace(_Model):
    """A space from the sampling coordinates of eryn priors.

    Attributes:
        eryn_priors: Import path of an eryn ``ProbDistContainer``.
        periodic: Periods by parameter name or position, as the ``periodic``
            argument of :meth:`ParamSpace.from_eryn_priors
            <mixed_signals.core.space.ParamSpace.from_eryn_priors>` takes them.
        spheres: Spheres of the space, named by the parameter names.
    """

    eryn_priors: ImportPath
    periodic: dict[str | int, FiniteNumber] | None = None
    spheres: tuple[Sphere, ...] = ()


class _Entry(_Model):
    """Fields that every posterior entry has.

    Attributes:
        name: Unique name of the posterior.
        path: The file, relative to the config folder or absolute.
    """

    name: Name
    path: str = Field(min_length=1)


class ErynEntry(_Entry):
    """A posterior from an eryn HDF5 file.

    Attributes:
        format: ``eryn``.
        branch: Name of the branch; needed only when the run has several.
        num_repeats_in_model: In-model proposals per step, for acceptance counts.
        burn_in: Steps dropped from the start of every walker.
        thin: ``"tau"`` or a thinning factor.
    """

    format: Literal["eryn"]
    branch: str | None = None
    num_repeats_in_model: Annotated[int, Field(strict=True, ge=1)] | None = None
    burn_in: Annotated[int, Field(strict=True, ge=0)] = 0
    thin: ThinSpec = TAU_THINNING


class ChainsEntry(_Entry):
    """A posterior from an ``.npz`` archive of chains.

    Attributes:
        format: ``chains``.
        burn_in: Steps dropped from the start of every walker.
        thin: ``"tau"`` or a thinning factor.
    """

    format: Literal["chains"]
    burn_in: Annotated[int, Field(strict=True, ge=0)] = 0
    thin: ThinSpec = TAU_THINNING


class SamplesEntry(_Entry):
    """A posterior from an ``.npz`` archive or ``.npy`` file of samples.

    Attributes:
        format: ``samples``.
        n_eff: Effective sample size, a number or one number per problem.
    """

    format: Literal["samples"]
    n_eff: (
        Annotated[float, Field(strict=True, gt=0)]
        | list[Annotated[float, Field(strict=True, gt=0)]]
        | None
    ) = None


class TableEntry(_Entry):
    """A posterior from a parquet or CSV table.

    Attributes:
        format: ``table``.
        log_weights: Column of natural-log weights; equal weights when null.
        n_eff: Effective sample size.
    """

    format: Literal["table"]
    log_weights: str | None = None
    n_eff: Annotated[float, Field(strict=True, gt=0)] | None = None


class Hdf5Entry(_Entry):
    """A posterior from the datasets of an HDF5 group.

    Attributes:
        format: ``hdf5``.
        group: Path of the group inside the file.
        log_weights: Dataset of natural-log weights; equal weights when null.
        n_eff: Effective sample size, a number or one number per problem.
    """

    format: Literal["hdf5"]
    group: str
    log_weights: str | None = None
    n_eff: (
        Annotated[float, Field(strict=True, gt=0)]
        | list[Annotated[float, Field(strict=True, gt=0)]]
        | None
    ) = None


PosteriorEntry = Annotated[
    ErynEntry | ChainsEntry | SamplesEntry | TableEntry | Hdf5Entry,
    Field(discriminator="format"),
]
"""One posterior of the config, told apart by its ``format``."""


class DiagnoseSection(_Model):
    """The ``diagnose`` section: the arguments of the engine of that name.

    Attributes:
        diagnostics: The families to run.
        draws: Number of surrogate sets per posterior.
        cap: Largest number of samples of the shape and mode rows.
        pair_cap: Largest number of samples of the dependence rows.
        thresholds: Threshold of each metric, merged over the defaults.
        level: The σ_sig that a row must reach to flag.
        transform: Transform of the coordinates of the rows.
    """

    diagnostics: tuple[str, ...] = DIAGNOSTIC_FAMILIES
    draws: Annotated[int, Field(strict=True, ge=1)] = NULL_DRAWS
    cap: Annotated[int, Field(strict=True, ge=2)] = SAMPLE_CAP
    pair_cap: Annotated[int, Field(strict=True, ge=2)] = PAIR_CAP
    thresholds: DiagnoseThresholds = Field(
        default_factory=partial(dict, _DIAGNOSE_THRESHOLDS)
    )
    level: FiniteNumber = SIGNIFICANCE_LEVEL
    transform: TransformSpec = None


class CompareSection(_Model):
    """The ``compare`` section: the arguments of the engine of that name.

    Attributes:
        reference: Name of the reference posterior, for mode A.
        whitening: ``reference``, ``pooled`` or ``fisher``; after validation
            the default of the engine when null.
        fisher: Fisher matrices of the ``fisher`` whitening.
        centre: Points of the Fisher matrices.
        transform: Transform of the coordinates.
        metrics: Metric names and ``metric:estimator`` recipes.
        null: ``permutation`` or ``split_half``.
        draws: Number of null draws per row.
        cap: Largest number of samples of a metric with a cap.
        sinkhorn_epsilon_factors: The ε factors of the Sinkhorn divergence.
        uncertainty: ``null_std`` or ``subsampling_bootstrap``.
        half_samples: Number of half-sample pairs of the bootstrap.
        thresholds: Threshold of each metric, merged over the defaults.
        level: The σ_sig that a row must reach to fail.
    """

    reference: str | None = None
    whitening: Literal["reference", "pooled", "fisher"] | None = None
    fisher: ArraySpec | None = None
    centre: ArraySpec | None = None
    transform: TransformSpec = None
    metrics: tuple[str, ...] = DEFAULT_METRICS
    null: Literal["permutation", "split_half"] = Construction.PERMUTATION.value
    draws: Annotated[int, Field(strict=True, ge=1)] = NULL_DRAWS
    cap: Annotated[int, Field(strict=True, ge=2)] = SAMPLE_CAP
    sinkhorn_epsilon_factors: tuple[FiniteNumber, ...] = (SINKHORN_EPSILON_FACTOR,)
    uncertainty: Literal["null_std", "subsampling_bootstrap"] = (
        UncertaintyMethod.NULL_STD.value
    )
    half_samples: Annotated[int, Field(strict=True, ge=1)] = HALF_SAMPLE_PAIRS
    thresholds: CompareThresholds = Field(
        default_factory=partial(dict, _COMPARE_THRESHOLDS)
    )
    level: FiniteNumber = SIGNIFICANCE_LEVEL

    @model_validator(mode="before")
    @classmethod
    def _fill_whitening(cls, data: Any) -> Any:
        # * The default of the engine: the reference posterior, else the pooled sets.
        if isinstance(data, dict) and data.get("whitening") is None:
            default = "pooled" if data.get("reference") is None else "reference"
            return {**data, "whitening": default}
        return data

    @model_validator(mode="after")
    def _check_fisher(self) -> "CompareSection":
        has_arrays = self.fisher is not None or self.centre is not None
        if self.whitening == "fisher":
            if self.fisher is None or self.centre is None:
                raise ValueError("the fisher whitening needs fisher and centre")
        elif has_arrays:
            raise ValueError(
                f"fisher and centre serve the fisher whitening, not {self.whitening!r}"
            )
        return self


class CompareSignalsSection(_Model):
    """The ``compare_signals`` section: the arguments of the engine of that name.

    Attributes:
        generator: Import path of a signal generator.
        noise: Import path of a noise model.
        reference: Name of the reference posterior, for mode A.
        fisher: Fisher matrices; they need ``centre``.
        centre: Points of the Fisher matrices.
        transform: Transform of the coordinates the generator receives.
        null: ``permutation`` or ``split_half``.
        draws: Number of null draws per row.
        cap: Largest number of samples of a posterior of a problem.
        uncertainty: ``null_std`` or ``subsampling_bootstrap``.
        half_samples: Number of half-sample pairs of the bootstrap.
        thresholds: Threshold of each metric, merged over the defaults.
        level: The σ_sig that a row must reach to fail.
        backend: The array library of the arrays that the generator receives.
        chunk_size: The most samples in one call of the generator.
    """

    generator: ImportPath
    noise: ImportPath
    reference: str | None = None
    fisher: ArraySpec | None = None
    centre: ArraySpec | None = None
    transform: TransformSpec = None
    null: Literal["permutation", "split_half"] = Construction.PERMUTATION.value
    draws: Annotated[int, Field(strict=True, ge=1)] = NULL_DRAWS
    cap: Annotated[int, Field(strict=True, ge=2)] = SAMPLE_CAP
    uncertainty: Literal["null_std", "subsampling_bootstrap"] = (
        UncertaintyMethod.NULL_STD.value
    )
    half_samples: Annotated[int, Field(strict=True, ge=1)] = HALF_SAMPLE_PAIRS
    thresholds: CompareThresholds = Field(
        default_factory=partial(dict, _COMPARE_THRESHOLDS)
    )
    level: FiniteNumber = SIGNIFICANCE_LEVEL
    backend: Literal["numpy", "jax", "torch", "cupy"] = "numpy"
    chunk_size: Annotated[int, Field(strict=True, ge=1)] | None = None

    @model_validator(mode="after")
    def _check_fisher(self) -> "CompareSignalsSection":
        if self.fisher is not None and self.centre is None:
            raise ValueError("fisher needs centre")
        return self


def _space_form(value: object) -> str | None:
    """Tell the three forms of ``space`` apart, or return None for none of them."""
    if isinstance(value, str):
        return "file"
    if isinstance(value, ParamSpace):
        return "inline"
    if isinstance(value, ErynSpace):
        return "eryn"
    if isinstance(value, Mapping):
        if "eryn_priors" in value:
            return "eryn"
        if "parameters" in value:
            return "inline"
    return None


SpaceSpec = Annotated[
    Annotated[str, Tag("file")]
    | Annotated[ParamSpace, Tag("inline")]
    | Annotated[ErynSpace, Tag("eryn")],
    Discriminator(
        _space_form,
        custom_error_type="space_form",
        custom_error_message=(
            "space must be the path of a ParamSpace YAML file, a mapping with"
            " 'parameters' and optional 'spheres', or a mapping with 'eryn_priors'"
            " and optional 'periodic' and 'spheres'"
        ),
    ),
]
"""The path of a space file, an inline space, or eryn priors."""

_SECTIONS: Final = ("diagnose", "compare", "compare_signals")
_MERGE_TAG: Final = "tag:yaml.org,2002:merge"


class PipelineConfig(_Model):
    """A pipeline config: the posteriors, the space and the engines to run.

    Attributes:
        seed: Integer in ``[0, SEED_LIMIT)`` from which every key derives.
        output: The run folder, relative to the config folder or absolute.
        budget: The estimated seconds above which the run raises one
            :class:`~mixed_signals.errors.CostWarning`; infinity switches it off.
        space: The parameter space, as the path of a file, an inline space or
            eryn priors.
        posteriors: The posteriors, one entry per file, with unique names.
        diagnose: The ``diagnose`` section, or ``None`` to skip the engine.
        compare: The ``compare`` section, or ``None``.
        compare_signals: The ``compare_signals`` section, or ``None``.
    """

    seed: Annotated[int, Field(strict=True, ge=0, lt=SEED_LIMIT)]
    output: str = Field(min_length=1)
    budget: Annotated[float, Field(strict=True, gt=0)] = COST_BUDGET
    space: SpaceSpec
    posteriors: Annotated[tuple[PosteriorEntry, ...], Field(min_length=1)]
    diagnose: DiagnoseSection | None = None
    compare: CompareSection | None = None
    compare_signals: CompareSignalsSection | None = None

    _folder: Path = PrivateAttr(default_factory=Path.cwd)
    _file: Path | None = PrivateAttr(default=None)

    @model_validator(mode="after")
    def _check_config(self) -> "PipelineConfig":
        names = [entry.name for entry in self.posteriors]
        repeated = sorted({name for name in names if names.count(name) > 1})
        if repeated:
            raise ValueError(
                "posterior names must be unique, repeated:"
                f" {', '.join(map(repr, repeated))}"
            )
        sections = {name: getattr(self, name) for name in _SECTIONS}
        present = {
            name: section for name, section in sections.items() if section is not None
        }
        if not present:
            raise ValueError(
                f"a config needs at least one of the sections {', '.join(_SECTIONS)}"
            )
        for name, section in present.items():
            reference = getattr(section, "reference", None)
            if reference is not None and reference not in names:
                raise ValueError(
                    f"{name}: reference {reference!r} is not a posterior of the"
                    f" config; the posteriors are {', '.join(map(repr, names))}"
                )
            if section.transform == "physical" and not isinstance(
                self.space, ErynSpace
            ):
                raise ValueError(
                    f"{name}: transform 'physical' needs a space with eryn_priors"
                )
        return self

    @property
    def base_folder(self) -> Path:
        """The folder that relative paths start at: the folder of the config file."""
        return self._folder

    @property
    def config_file(self) -> Path | None:
        """The absolute path of the config file, or ``None`` for a Python mapping."""
        return self._file


class _StrictLoader(yaml.SafeLoader):
    """The safe YAML loader, which also rejects a key that a mapping repeats."""

    def construct_mapping(
        self, node: yaml.MappingNode, deep: bool = False
    ) -> dict[Hashable, Any]:
        """Build a mapping, raising for a repeated key with its line."""
        seen: set[Hashable] = set()
        for key_node, _ in node.value:
            # * A merge key brings in a mapping whose keys an explicit key may override.
            if key_node.tag == _MERGE_TAG:
                continue
            key = self.construct_object(key_node, deep=deep)
            # * An unhashable key is an error of the parent class, raised below.
            if not isinstance(key, Hashable):
                continue
            # * An unquoted null key is the field named null, not YAML's null value.
            key = "null" if key is None else key
            if key in seen:
                raise yaml.constructor.ConstructorError(
                    "while constructing a mapping",
                    node.start_mark,
                    f"found the key {key!r} more than once",
                    key_node.start_mark,
                )
            seen.add(key)
        self.flatten_mapping(node)
        mapping = super().construct_mapping(node, deep=deep)
        if None in mapping:
            mapping["null"] = mapping.pop(None)
        return mapping


def load_config(path: str | os.PathLike[str]) -> PipelineConfig:
    """Read and validate a pipeline config.

    The file is read as UTF-8 with the safe YAML loader. The function reads no
    data and imports no module of the user: the pipeline does that when it
    prepares a run. Relative paths of the config start at the folder of the file.

    Args:
        path: The YAML file.

    Returns:
        The validated config.

    Raises:
        ConfigError: If the file cannot be read, is not valid YAML, repeats a
            key, or does not hold a mapping.
        pydantic.ValidationError: If the config is invalid, with every problem
            named by its location.
    """
    config_file = Path(path).resolve()
    try:
        text = config_file.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise ConfigError(f"cannot read the config {config_file}: {error}") from error
    loader = _StrictLoader(text)
    loader.name = str(config_file)
    try:
        document = loader.get_single_data()
    except yaml.YAMLError as error:
        raise ConfigError(f"{config_file} is not valid YAML: {error}") from error
    finally:
        loader.dispose()
    if not isinstance(document, dict):
        raise ConfigError(
            f"{config_file} must hold a mapping of keys to values, got"
            f" {type(document).__name__}"
        )
    config = PipelineConfig.model_validate(document)
    config._folder = config_file.parent
    config._file = config_file
    return config


def _canonical_json(config: PipelineConfig, space: ParamSpace) -> bytes:
    """Return the canonical JSON of the config and the resolved space, as UTF-8.

    The config is validated, with every default filled in, and keeps its paths as
    written, without ``output`` and ``budget``, which change no row.
    """
    payload = {
        "config": config.model_dump(mode="json", exclude={"output", "budget"}),
        "space": space.model_dump(mode="json"),
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _config_hash(canonical_json: bytes) -> str:
    """Return the SHA-256 of the canonical JSON as 64 lowercase hex characters.

    References:
        :cite:t:`nist2015secure`.
    """
    return hashlib.sha256(canonical_json).hexdigest()
