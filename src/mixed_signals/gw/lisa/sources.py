r"""Holders of LISA sources: B sources with N weighted samples each.

A holder keeps the samples of every parameter of one source type, the names of
the sources, the conventions of its parameters
(:mod:`~mixed_signals.gw.lisa.conventions`) and optionally a table of detection
statistics. It gives the :class:`~mixed_signals.core.samples.SampleSet` shaped
(B, 1, N, D) of the sampled parameters and its
:class:`~mixed_signals.core.space.ParamSpace`, which the engines compare. A
catalog of injections or point estimates is a holder with N = 1. Holders are
plain immutable classes, because they never cross ``jax.jit``: their transforms
return new holders.

Layout
    The values keep every canonical parameter of the type, sampled and constant
    together, so that the transforms act on the full vector.
    :attr:`~LisaSources.space` and :attr:`~LisaSources.sample_set` hold the
    sampled parameters only, and :attr:`~LisaSources.settings` the constant
    ones, one value per source.
Settings
    A parameter is a setting when its positive-weight values are equal within
    every source that has at least two positive-weight samples, such as a
    default that nobody sampled. A holder whose sources all have one sample has
    no settings, because a single value does not tell a constant from a
    measurement. Settings are recomputed after every transform: an epoch shift
    can make a constant phase vary. A sphere stays in the space when either of
    its axes is sampled, because a direction is one object.
Padding
    Sources with fewer samples are padded to the largest N with zero-weight
    copies of their first sample. The engines drop zero-weight samples per
    problem, so padding never changes a row. :attr:`~LisaSources.sample_counts`
    gives the number of samples that carry a weight.
Checks
    The constructors check, on concrete inputs and over the positive-weight
    samples, that values are finite, scales are strictly positive, spins have a
    magnitude of at most one, eccentricities lie in [0, 1), inclinations in
    [0, π], declinations in [−π/2, π/2] and polar angles in [0, π], and that
    the primary mass is not below the secondary mass (SGS Section 2.3,
    :cite:t:`baghi2026lisa`). Circle parameters are wrapped into their period
    first (:cite:t:`mardia2000directional`). Each failure raises ``ValueError``
    with the parameter, the rule and the count of offending values.
Prior bounds
    ``bounds`` replaces the physical domain of an interval parameter by the
    support of its prior; it must lie inside the domain, and the positive-weight
    samples must lie inside it too, as a closed interval. Transforms refuse to
    change a parameter with such a bound, because the bound does not map.
    :meth:`~LisaSources.with_bounds` sets bounds on a holder after its
    transforms, on the parameters of the transformed vector, such as the total
    mass and the mass ratio.
Mass forms
    Binary black holes keep the detector-frame masses (m₁, m₂) or, as opt-in
    forms, the total mass M or the chirp mass 𝓜 with the mass ratio q = m₂/m₁
    or the inverted mass ratio Q = m₁/m₂ (SGS Section 2.3). A conversion
    passes through the components, and the log-prior moves by the Jacobians
    M/m₁² or M/m₂², with 𝓜 in place of M for the chirp mass.
Source ids
    Ids are strings. UTF-8 bytes decode, and integers become their decimal
    string. Whole-number floats of magnitude below 2⁵³ become the decimal
    string of their integer, because catpylisa writes the ids of its massive
    black-hole binary (MBHB) injection catalogs as float64; below 2⁵³ a float64
    holds every integer exactly (:cite:t:`ieee2019ieee`). Other floats raise
    ``ValueError``, and booleans ``TypeError``.

References:
    :cite:t:`baghi2026lisa`; :cite:t:`mardia2000directional`;
    :cite:t:`ade2016planck`; :cite:t:`astropy2022astropy`;
    :cite:t:`ieee2019ieee`.
"""

import dataclasses
import math
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar, Final

import jax.numpy as jnp
import numpy as np
import polars as pl
from jaxtyping import Array, Float

from mixed_signals.core.samples import SampleSet
from mixed_signals.core.space import Parameter, ParamSpace
from mixed_signals.core.topology import Sphere, Topology, wrap
from mixed_signals.errors import ConventionError
from mixed_signals.gw.lisa._cosmology import redshift_from_luminosity_distance
from mixed_signals.gw.lisa._dependencies import lisa_constants
from mixed_signals.gw.lisa.conventions import (
    LisaConventions,
    ParameterSpec,
    SkyFrame,
    SourceFrame,
    frame_aliases,
    frame_parameter,
    frame_sphere,
    rename_to_frame,
)
from mixed_signals.gw.lisa.transforms import (
    _MASS_FORM_PARAMETERS,
    MassPairTransform,
    SkyFrameTransform,
)
from mixed_signals.io.files import _warn_narrow_arrays
from mixed_signals.transforms.base import Transform

__all__ = ["BlackHoleBinarySources", "LisaSources"]

type Selection = int | np.integer[Any] | str | slice | Sequence[Any] | np.ndarray
"""Sources picked from a holder: an index, an id, a slice, or a sequence of them."""

# * Fields of a posterior table that are not parameters, and the names they take.
_LOG_FIELDS: Final = {"loglikelihood": "log_like", "logprior": "log_prior"}
_PRIMARY: Final = "primary_mass_det_frame"
_SECONDARY: Final = "secondary_mass_det_frame"
_TOTAL: Final = "total_mass_det_frame"
_CHIRP: Final = "chirp_mass_det_frame"
_RATIO: Final = "mass_ratio"
_INVERTED_RATIO: Final = "inverted_mass_ratio"
# * The five forms of the two mass parameters: the components, or a mass and a ratio.
_MASS_FORMS: Final = (
    (_PRIMARY, _SECONDARY),
    (_TOTAL, _RATIO),
    (_TOTAL, _INVERTED_RATIO),
    (_CHIRP, _RATIO),
    (_CHIRP, _INVERTED_RATIO),
)
# * A float64 holds every integer below 2**53, its 53-bit significand.
_FLOAT_ID_LIMIT: Final = 2**53


@dataclass(frozen=True, eq=False, repr=False)
class LisaSources:
    """B LISA sources of one type with N weighted samples each.

    Build holders with :meth:`from_arrays`, :meth:`from_posteriors` or
    :meth:`from_catalog` of a source type, such as
    :class:`~mixed_signals.gw.lisa.galactic_binaries.GalacticBinaries`; the
    constructor takes the finished pieces. The module docstring describes the
    layout, the settings and the checks.

    Attributes:
        name: The global fit that the samples come from; it names the posterior
            in comparisons.
        source_ids: The names of the B sources, unique.
        conventions: The conventions of the parameters.
        full_space: The space of the full parameter vector, sampled and constant
            parameters together, with the prior bounds applied.
        values: The samples of the full vector, shaped (B, 1, N, D_all), with
            their log weights, effective sample sizes, log-likelihoods and
            log-priors.
        detections: A table with one row per source in holder order, or ``None``;
            no comparison reads it.
        bounds: The prior bounds that replace the physical domains, by name.
    """

    name: str
    source_ids: tuple[str, ...]
    conventions: LisaConventions
    full_space: ParamSpace
    values: SampleSet
    detections: pl.DataFrame | None = None
    bounds: Mapping[str, tuple[float, float]] = field(default_factory=dict)
    _sampled: tuple[int, ...] = field(init=False, repr=False)

    source_type: ClassVar[str]
    """Name of the source type."""
    parameter_specs: ClassVar[tuple[ParameterSpec, ...]]
    """The parameters of the type in the canonical order, with ICRS names."""
    sphere_specs: ClassVar[tuple[Sphere, ...]]
    """The spheres of the type, with ICRS names."""
    alias_table: ClassVar[Mapping[str, str]]
    """Alias to canonical name, with ICRS spelling."""
    spinning: ClassVar[bool]
    """Whether the type has spins and so a source frame."""
    can_shift_epoch: ClassVar[bool] = False
    """Whether :meth:`at_epoch` can move the epoch of the type."""

    def __post_init__(self) -> None:
        """Check the pieces against each other and find the sampled parameters."""
        shape = self.values.shape
        if not self.source_ids:
            raise ValueError("a holder needs at least one source")
        if len(set(self.source_ids)) != len(self.source_ids):
            repeated = sorted(
                {name for name in self.source_ids if self.source_ids.count(name) > 1}
            )
            raise ValueError(f"source ids must be unique, repeated: {repeated}")
        if shape.batch != len(self.source_ids) or shape.posterior != 1:
            raise ValueError(
                f"the values are shaped (B, K, N, D) = {tuple(shape)}, but the holder"
                f" has {len(self.source_ids)} sources and needs K = 1"
            )
        if shape.param != len(self.full_space):
            raise ValueError(
                f"the space has {len(self.full_space)} parameters, but the values"
                f" have {shape.param}"
            )
        object.__setattr__(
            self, "_sampled", _sampled_positions(self.values, self.full_space)
        )

    def __repr__(self) -> str:
        """Name the type, the global fit and the sizes."""
        shape = self.values.shape
        return (
            f"{type(self).__name__}(name={self.name!r}, sources={shape.batch},"
            f" samples={shape.sample}, parameters={shape.param})"
        )

    def __len__(self) -> int:
        """Return the number of sources B."""
        return len(self.source_ids)

    def __getitem__[SourcesType: "LisaSources"](
        self: SourcesType, selection: Selection
    ) -> SourcesType:
        """Return the holder of some sources, picked by index, slice or source id.

        Args:
            selection: An index, a source id, a slice, or a sequence or array of
                indices or ids; the order and repeats of a sequence are kept, and
                repeated sources raise, since ids are unique. An integer selects
                by position, so a source with an integer id is selected through
                its decimal string, such as ``sources["12"]``.

        Returns:
            A holder of the same type with the selected sources, their
            detections and the same conventions.

        Raises:
            KeyError: If a source id is unknown.
            IndexError: If an index lies outside the sources.
            ValueError: If the selection is empty or repeats a source.
        """
        indices = self._indices(selection)
        return dataclasses.replace(
            self,
            source_ids=tuple(self.source_ids[position] for position in indices),
            values=self.values.select(problems=indices),
            detections=(
                None
                if self.detections is None
                else self.detections.select(pl.all().gather(indices.tolist()))
            ),
        )

    @property
    def parameter_names(self) -> tuple[str, ...]:
        """The names of every parameter of the full vector, sampled or constant."""
        return self.full_space.names

    @property
    def space(self) -> ParamSpace:
        """The space of the sampled parameters, with the spheres whose axes are sampled.

        Raises:
            ValueError: If no parameter varies, so that every one is a setting.
        """
        if not self._sampled:
            raise ValueError(
                "no parameter varies within any source, so every parameter is a"
                " setting; see settings"
            )
        sampled_names = {self.full_space.names[position] for position in self._sampled}
        return ParamSpace(
            parameters=tuple(
                self.full_space.parameters[position] for position in self._sampled
            ),
            spheres=tuple(
                sphere
                for sphere in self.full_space.spheres
                if {sphere.azimuth, sphere.polar} <= sampled_names
            ),
        )

    @property
    def sample_set(self) -> SampleSet:
        """The sampled parameters as a set shaped (B, 1, N, D), with their weights."""
        _ = self.space
        return self.values.select(params=list(self._sampled))

    @property
    def settings(self) -> pl.DataFrame:
        """The constant parameters: ``source_id`` and one column per setting.

        A value is that of the first sample with a positive weight. A holder
        without settings gives the column of ids alone.
        """
        samples = np.asarray(self.values.samples[:, 0])
        positive = np.asarray(self.values.log_weights[:, 0]) > -np.inf
        first = samples[np.arange(samples.shape[0]), np.argmax(positive, axis=1)]
        columns: dict[str, Any] = {"source_id": list(self.source_ids)}
        for position, name in enumerate(self.full_space.names):
            if position not in self._sampled:
                columns[name] = first[:, position].tolist()
        return pl.DataFrame(columns)

    @property
    def sample_counts(self) -> np.ndarray:
        """The number of samples with a positive weight of each source, shaped (B,)."""
        positive = np.asarray(self.values.log_weights[:, 0]) > -np.inf
        counts: np.ndarray = positive.sum(axis=1)
        return counts

    def to_icrs[SourcesType: "LisaSources"](self: SourcesType) -> SourcesType:
        """Return the holder in the equatorial ICRS frame, or itself if it is there.

        Rotates the sky position, shifts the polarisation angle and rotates the
        spin directions that are measured in the sky frame; see
        :class:`~mixed_signals.gw.lisa.transforms.SkyFrameTransform`.

        Raises:
            ValueError: If a sky parameter has a prior bound, which does not map.
            MissingDependencyError: If lisaconstants is not installed.
        """
        return self._in_frame(SkyFrame.ICRS)

    def to_ecliptic[SourcesType: "LisaSources"](self: SourcesType) -> SourcesType:
        """Return the holder in the Barycentric Mean Ecliptic frame, or itself.

        The inverse of :meth:`to_icrs`.

        Raises:
            ValueError: If a sky parameter has a prior bound, which does not map.
            MissingDependencyError: If lisaconstants is not installed.
        """
        return self._in_frame(SkyFrame.ECLIPTIC)

    def at_epoch[SourcesType: "LisaSources"](
        self: SourcesType, epoch: float
    ) -> SourcesType:
        """Return the holder at another reference epoch.

        Only Galactic binaries and massive black-hole binaries can shift their
        epoch. The parameters of the other types depend on the dynamics of the
        binary, which the holder does not model, so they return the holder
        itself at its own epoch, as :meth:`to_icrs` does in ICRS, and raise at
        any other.

        Args:
            epoch: The new epoch in TCB seconds since ``LISA_EPOCH_TCB``.

        Returns:
            The holder itself for the types that cannot shift their epoch, when
            ``epoch`` equals its own.

        Raises:
            ConventionError: For the types that cannot shift their epoch, at
                another epoch.
        """
        if float(epoch) == self.conventions.epoch:
            return self
        raise ConventionError(
            f"{type(self).__name__} cannot shift its epoch: no transform exists"
            " without the dynamics of the binary"
        )

    def with_bounds[SourcesType: "LisaSources"](
        self: SourcesType, bounds: Mapping[str, tuple[float, float]]
    ) -> SourcesType:
        """Return the holder with prior bounds set on parameters of its current vector.

        Transforms refuse parameters with prior bounds, because a bound does not
        map. A prior that is a rectangle in transformed parameters, such as the
        total mass and the mass ratio, is therefore set in the order build,
        transform, ``with_bounds``. The new bounds replace the physical domain
        of their parameters, or an earlier bound of the same name, and update
        :attr:`bounds` and the space.

        Args:
            bounds: Name of a parameter of the current full vector to its lower
                and upper bound, with lower below upper. Names of the type's
                parameters resolve through the aliases, as in the constructors;
                the names of the mass forms, ``total_mass_det_frame``,
                ``chirp_mass_det_frame``, ``mass_ratio`` and
                ``inverted_mass_ratio``, have no aliases. The bounds must lie
                inside the physical domain of the parameter: that of the type's
                table, or that of
                :class:`~mixed_signals.gw.lisa.transforms.MassPairTransform`
                for the mass forms.

        Returns:
            A holder of the same type whose ``full_space`` carries the new
            bounds: a bounded parameter becomes an interval, or the real line
            when both bounds are infinite.

        Raises:
            ConventionError: If a name belongs to the other sky frame.
            ValueError: If a name is not in the current vector or has no
                physical domain, a bound lies outside the domain or is not
                ordered, a parameter is a circle, or a positive-weight sample
                lies outside its bounds.

        Example:
            A prior on the total mass M in [10⁵, 10⁸] solar masses and on
            Q = m₁/m₂ in [1, 10], set after the transform to those parameters::

                sources = MassiveBlackHoleBinaries.from_posteriors(...)
                sources = sources.to_total_mass_and_mass_ratio(inverted=True)
                sources = sources.with_bounds(
                    {
                        "total_mass_det_frame": (1e5, 1e8),
                        "inverted_mass_ratio": (1.0, 10.0),
                    }
                )
        """
        names = self.full_space.names
        new_bounds = _read_bounds(
            type(self), bounds, self.conventions.sky_frame, vector_names=names
        )
        parameters = list(self.full_space.parameters)
        for name, bound in new_bounds.items():
            position = self.full_space.index(name)
            parameters[position] = _prior_parameter(
                parameters[position], _physical_domain(self, name), bound
            )
        host = np.asarray(self.values.samples[:, 0])
        positive = np.asarray(self.values.log_weights[:, 0]) > -np.inf
        _check_prior_bounds(host, names, new_bounds, positive, self.source_ids)
        return dataclasses.replace(
            self,
            full_space=ParamSpace(
                parameters=tuple(parameters), spheres=self.full_space.spheres
            ),
            bounds={**self.bounds, **new_bounds},
        )

    @classmethod
    def from_arrays[SourcesType: "LisaSources"](
        cls: type[SourcesType],
        values: Mapping[str, object],
        *,
        source_ids: Sequence[Any] | np.ndarray,
        name: str,
        epoch: float,
        sky_frame: SkyFrame | str = SkyFrame.ICRS,
        source_frame: SourceFrame | str | None = None,
        log_weights: object = None,
        n_eff: object = None,
        log_like: object = None,
        log_prior: object = None,
        bounds: Mapping[str, tuple[float, float]] | None = None,
        detections: object = None,
    ) -> SourcesType:
        """Build a holder from one array shaped (B, N) per parameter.

        Parameters may be named by their canonical names or their aliases;
        missing parameters take their defaults, and a missing parameter without
        a default raises. Circle parameters are wrapped into their period, and
        the values are checked as the module docstring says.

        Args:
            values: The samples of each parameter, shaped (B, N).
            source_ids: The B source ids: strings, UTF-8 bytes, integers or
                whole-number floats. Integers become their decimal string, and
                a float the decimal string of its integer, so 5.0 gives "5".
            name: The global fit that the samples come from.
            epoch: The reference epoch in TCB seconds since ``LISA_EPOCH_TCB``.
            sky_frame: The frame of the sky position, ICRS by default.
            source_frame: The frame of the spin vectors; required for the types
                with spins, and ``None`` for Galactic binaries.
            log_weights: Natural logarithms of the unnormalised weights, shaped
                (B, N); equal weights when ``None``.
            n_eff: The effective sample size, one number for every source or one
                per source; the Kish value of the weights when ``None``.
            log_like: Log-likelihoods, shaped (B, N).
            log_prior: Log-prior densities, shaped (B, N).
            bounds: Prior bounds that replace the physical domains of interval
                parameters, inside them.
            detections: A structured array or a polars table with a
                ``source_id`` column, whose ids read as those of the sources
                and are compared as strings; joined to the sources in holder
                order.

        Returns:
            The holder.

        Raises:
            ValueError: If a name is unknown or repeated, a required parameter
                is missing, a shape or a value does not fit, a source frame is
                missing or given for a type without spins, a bound lies outside
                its domain, or a positive-weight sample lies outside its prior
                bounds.
            ConventionError: If a name belongs to the other sky frame.
        """
        conventions = _read_conventions(
            cls, epoch=epoch, sky_frame=sky_frame, source_frame=source_frame
        )
        frame = conventions.sky_frame
        ids = _read_ids(source_ids)
        resolved = _resolve_names(cls, values, frame, context="values")
        columns = {resolved[key]: values[key] for key in resolved}
        return _build(
            cls,
            columns,
            ids=ids,
            name=name,
            conventions=conventions,
            extras={
                "log_weights": log_weights,
                "log_like": log_like,
                "log_prior": log_prior,
            },
            n_eff=n_eff,
            bounds=bounds,
            detections=detections,
        )

    @classmethod
    def from_posteriors[SourcesType: "LisaSources"](
        cls: type[SourcesType],
        posteriors: Mapping[Any, Any],
        *,
        name: str,
        epoch: float,
        sky_frame: SkyFrame | str = SkyFrame.ICRS,
        source_frame: SourceFrame | str | None = None,
        n_eff: object = None,
        bounds: Mapping[str, tuple[float, float]] | None = None,
        detections: object = None,
        ignore: Collection[str] = (),
    ) -> SourcesType:
        """Build a holder from the posteriors of catpylisa: one table per source.

        Each value of ``posteriors`` is a NumPy structured array, or a mapping of
        field to one-dimensional array, with the samples of one source. Fields
        resolve through the aliases; ``loglikelihood`` becomes ``log_like`` and
        ``logprior`` becomes ``log_prior``. Sources keep the order of the
        mapping. Shorter sources are padded to the largest N with zero-weight
        copies of their first sample, with a log weight of −∞ and copies of its
        log-likelihood and log-prior; real samples have log weight 0.

        Args:
            posteriors: Source id to the samples. An id is a string, UTF-8
                bytes, an integer or a whole-number float. An integer becomes
                its decimal string, and a float the decimal string of its
                integer, so 5.0 gives "5".
            name: The global fit that the samples come from.
            epoch: The reference epoch in TCB seconds since ``LISA_EPOCH_TCB``.
            sky_frame: The frame of the sky position, ICRS by default.
            source_frame: The frame of the spin vectors; required for the types
                with spins, and ``None`` for Galactic binaries.
            n_eff: ``None`` for the Kish value, one number for every source, or
                a mapping of source id or a sequence with one value per source.
            bounds: Prior bounds that replace the physical domains of interval
                parameters, inside them.
            detections: A structured array or a polars table with a
                ``source_id`` column, whose ids read as those of the sources
                and are compared as strings; joined to the sources in holder
                order.
            ignore: Fields to drop, such as columns that hold no parameter.

        Returns:
            The holder.

        Raises:
            ValueError: If a field is unknown, the sources hold different
                fields, a field is not one-dimensional or of different lengths,
                or a value or a bound is invalid; see :meth:`from_arrays`.
            TypeError: If a posterior is neither a structured array nor a mapping.
            ConventionError: If a field belongs to the other sky frame.

        Warns:
            PrecisionWarning: If a field is narrower than float64.
        """
        conventions = _read_conventions(
            cls, epoch=epoch, sky_frame=sky_frame, source_frame=source_frame
        )
        ids = _read_ids(list(posteriors))
        tables = [
            _read_fields(posterior, source)
            for source, posterior in zip(ids, posteriors.values(), strict=True)
        ]
        resolved_per_source = [
            _resolve_names(
                cls,
                table,
                conventions.sky_frame,
                special=_LOG_FIELDS,
                ignore=ignore,
                context="posterior",
            )
            for table in tables
        ]
        fields = set(resolved_per_source[0].values())
        for source, resolved in zip(ids, resolved_per_source, strict=True):
            if set(resolved.values()) != fields:
                raise ValueError(
                    f"source {source!r} holds the fields {sorted(resolved.values())},"
                    f" but the first source holds {sorted(fields)}"
                )
        _warn_narrow_arrays(
            {
                f"{source}.{key}": array
                for source, table in zip(ids, tables, strict=True)
                for key, array in table.items()
            },
            "field",
        )
        counts = [len(next(iter(table.values()))) for table in tables]
        largest = max(counts)
        arrays: dict[str, np.ndarray] = {}
        for field_name in fields:
            rows = []
            for table, resolved in zip(tables, resolved_per_source, strict=True):
                (key,) = [
                    key for key, target in resolved.items() if target == field_name
                ]
                row = table[key].astype(np.float64)
                rows.append(np.concatenate([row, np.full(largest - row.size, row[0])]))
            arrays[field_name] = np.stack(rows)
        padded = np.arange(largest)[None, :] >= np.asarray(counts)[:, None]
        log_weights = np.where(padded, -np.inf, 0.0)
        return _build(
            cls,
            {
                key: array
                for key, array in arrays.items()
                if key not in _LOG_FIELDS.values()
            },
            ids=ids,
            name=name,
            conventions=conventions,
            extras={
                "log_weights": log_weights,
                "log_like": arrays.get("log_like"),
                "log_prior": arrays.get("log_prior"),
            },
            n_eff=_read_source_n_eff(n_eff, ids),
            bounds=bounds,
            detections=detections,
        )

    @classmethod
    def from_catalog[SourcesType: "LisaSources"](
        cls: type[SourcesType],
        table: Any,
        *,
        name: str,
        epoch: float,
        sky_frame: SkyFrame | str = SkyFrame.ICRS,
        source_frame: SourceFrame | str | None = None,
        id_field: str = "source_id",
        bounds: Mapping[str, tuple[float, float]] | None = None,
        ignore: Collection[str] = (),
    ) -> SourcesType:
        """Build a holder of one sample per source from a catalog.

        A catalog of injections or point estimates has one row per source, a
        column per parameter, and the source id in ``id_field``.

        Args:
            table: A NumPy structured array or a polars table.
            name: The catalog, used as the name of the posterior in comparisons.
            epoch: The reference epoch in TCB seconds since ``LISA_EPOCH_TCB``.
            sky_frame: The frame of the sky position, ICRS by default.
            source_frame: The frame of the spin vectors; required for the types
                with spins, and ``None`` for Galactic binaries.
            id_field: The column of source ids: strings, UTF-8 bytes, integers
                or whole-number floats. Integers become their decimal string,
                and a float the decimal string of its integer, so 5.0 gives
                "5".
            bounds: Prior bounds that replace the physical domains of interval
                parameters, inside them.
            ignore: Columns to drop, such as columns that hold no parameter.

        Returns:
            The holder, with N = 1.

        Raises:
            ValueError: If the id column is missing, a column is unknown, or a
                value or a bound is invalid; see :meth:`from_arrays`.
            TypeError: If the table is neither a structured array nor a polars
                table.
            ConventionError: If a column belongs to the other sky frame.
        """
        if isinstance(table, pl.DataFrame):
            columns = {
                key: table[key].to_numpy() for key in table.columns if key != id_field
            }
            raw_ids = table[id_field].to_list() if id_field in table.columns else None
        elif isinstance(table, np.ndarray) and table.dtype.names is not None:
            columns = {key: table[key] for key in table.dtype.names if key != id_field}
            raw_ids = (
                table[id_field].tolist() if id_field in table.dtype.names else None
            )
        else:
            raise TypeError(
                "the catalog must be a NumPy structured array or a polars table,"
                f" got {type(table).__name__}"
            )
        if raw_ids is None:
            raise ValueError(f"the catalog has no id column {id_field!r}")
        columns = {key: array for key, array in columns.items() if key not in ignore}
        conventions = _read_conventions(
            cls, epoch=epoch, sky_frame=sky_frame, source_frame=source_frame
        )
        resolved = _resolve_names(
            cls, columns, conventions.sky_frame, context="catalog"
        )
        return _build(
            cls,
            {resolved[key]: np.asarray(columns[key])[:, None] for key in resolved},
            ids=_read_ids(raw_ids),
            name=name,
            conventions=conventions,
            extras={},
            n_eff=None,
            bounds=bounds,
            detections=None,
        )

    def _column(self, name: str) -> Float[Array, "batch sample"]:
        """Return one parameter of the full vector for every source and sample."""
        if name not in self.full_space.names:
            raise ValueError(
                f"the parameter vector has no {name!r}; it holds"
                f" {', '.join(self.full_space.names)}"
            )
        return self.values.samples[:, 0, :, self.full_space.index(name)]

    def _indices(self, selection: Selection) -> np.ndarray:
        """Return the positions of a selection, in order, as an integer array."""
        count = len(self.source_ids)
        positions: np.ndarray
        if isinstance(selection, slice):
            positions = np.arange(count)[selection]
        elif isinstance(selection, str):
            positions = np.asarray([self._position(selection)])
        elif isinstance(selection, int | np.integer):
            positions = np.asarray([selection])
        else:
            entries = list(selection)
            if entries and all(isinstance(entry, str) for entry in entries):
                positions = np.asarray([self._position(entry) for entry in entries])
            else:
                positions = np.asarray(entries)
        if positions.size == 0:
            raise ValueError("select at least one source")
        if not np.issubdtype(positions.dtype, np.integer):
            raise TypeError("sources are selected by integers or ids")
        if ((positions < -count) | (positions >= count)).any():
            outside = positions[(positions < -count) | (positions >= count)]
            raise IndexError(
                f"source index {outside[0]} is out of range for {count} sources"
            )
        positions = np.where(positions < 0, positions + count, positions)
        if len(set(positions.tolist())) != positions.size:
            raise ValueError("a selection must not repeat a source")
        return positions

    def _position(self, source_id: str) -> int:
        """Return the position of a source id."""
        try:
            return self.source_ids.index(source_id)
        except ValueError:
            raise KeyError(f"no source with the id {source_id!r}") from None

    def _overwrite_zero_weight_samples[SourcesType: "LisaSources"](
        self: SourcesType, mask: np.ndarray
    ) -> SourcesType:
        """Copy the first weighted sample of each source over masked samples.

        A transform that is undefined for some values, such as the chirp mass at
        ḟ ≤ 0, would turn the values of padding into NaN. The masked samples
        must carry no weight, so a copy of a real sample replaces them.
        """
        if not mask.any():
            return self
        fields = self.values.to_arrays("numpy", copy=True)
        positive = fields["log_weights"][:, 0] > -np.inf
        samples = fields["samples"]
        first = samples[np.arange(samples.shape[0]), 0, np.argmax(positive, axis=1)]
        replaced = np.where(mask[:, None, :, None], first[:, None, None, :], samples)
        return dataclasses.replace(
            self, values=SampleSet(**{**fields, "samples": replaced})
        )

    def _apply[SourcesType: "LisaSources"](
        self: SourcesType,
        transform: Transform,
        touched: Sequence[str],
        *,
        conventions: LisaConventions | None = None,
    ) -> SourcesType:
        """Return the holder after a transform of the full vector.

        The transform moves the samples and ``log_prior`` and maps the space; the
        settings follow from the new values.
        """
        clashes = [name for name in touched if name in self.bounds]
        if clashes:
            raise ValueError(
                f"{', '.join(map(repr, clashes))} carry prior bounds, which do not"
                " map through a transform; build the holder without those bounds,"
                " transform it, then add them with with_bounds"
            )
        return dataclasses.replace(
            self,
            conventions=self.conventions if conventions is None else conventions,
            full_space=transform.map_space(self.full_space),
            values=transform.map_sample_set(self.values),
        )

    def _in_frame[SourcesType: "LisaSources"](
        self: SourcesType, frame: SkyFrame
    ) -> SourcesType:
        """Return the holder in a sky frame, applying the rotation if needed."""
        if self.conventions.sky_frame is frame:
            return self
        transform = SkyFrameTransform.from_space(
            self.full_space,
            to_ecliptic=frame is SkyFrame.ECLIPTIC,
            obliquity=lisa_constants().obliquity,
        )
        return self._apply(
            transform,
            transform.touched_names(self.full_space),
            conventions=self.conventions.model_copy(update={"sky_frame": frame}),
        )


class BlackHoleBinarySources(LisaSources):
    """The sources whose parameters hold two masses in the detector frame.

    Massive black-hole binaries, extreme mass-ratio inspirals and stellar-origin
    binary black holes share the derived mass properties, the redshift and the
    transforms of the masses. The properties read the masses in whichever form
    the holder carries: components, total mass or chirp mass with the mass
    ratio q = m₂/m₁ or the inverted mass ratio Q = m₁/m₂. They return arrays
    shaped (B, N) over all samples, padding included, whose weights are zero.

    References:
        :cite:t:`baghi2026lisa`; :cite:t:`ade2016planck`;
        :cite:t:`astropy2022astropy`.
    """

    spinning: ClassVar[bool] = True

    @property
    def total_mass(self) -> Float[Array, "batch sample"]:
        """The detector-frame total mass M = m₁ + m₂ in solar masses."""
        primary, secondary = self._component_masses()
        return primary + secondary

    @property
    def mass_ratio(self) -> Float[Array, "batch sample"]:
        """The mass ratio q = m₂/m₁, at most one."""
        primary, secondary = self._component_masses()
        return secondary / primary

    @property
    def inverted_mass_ratio(self) -> Float[Array, "batch sample"]:
        """The inverted mass ratio Q = m₁/m₂, at least one."""
        primary, secondary = self._component_masses()
        return primary / secondary

    @property
    def reduced_mass(self) -> Float[Array, "batch sample"]:
        """The reduced mass μ = m₁m₂/M in solar masses."""
        primary, secondary = self._component_masses()
        return primary * secondary / (primary + secondary)

    @property
    def symmetric_mass_ratio(self) -> Float[Array, "batch sample"]:
        """The symmetric mass ratio η = m₁m₂/M²."""
        primary, secondary = self._component_masses()
        return primary * secondary / (primary + secondary) ** 2

    @property
    def chirp_mass(self) -> Float[Array, "batch sample"]:
        """The detector-frame chirp mass 𝓜 = (m₁m₂)^(3/5)/M^(1/5) in solar masses."""
        primary, secondary = self._component_masses()
        return (primary * secondary) ** (3 / 5) / (primary + secondary) ** (1 / 5)

    @property
    def redshift(self) -> Float[Array, "batch sample"]:
        """The redshift of the luminosity distance in Planck15, computed outside JAX.

        References:
            :cite:t:`ade2016planck`; :cite:t:`astropy2022astropy`.
        """
        distance = np.asarray(self._column("luminosity_distance"))
        return jnp.asarray(redshift_from_luminosity_distance(distance))

    @property
    def source_frame_primary_mass(self) -> Float[Array, "batch sample"]:
        """The primary mass in the source frame, m₁/(1 + z)."""
        return self._component_masses()[0] / (1 + self.redshift)

    @property
    def source_frame_secondary_mass(self) -> Float[Array, "batch sample"]:
        """The secondary mass in the source frame, m₂/(1 + z)."""
        return self._component_masses()[1] / (1 + self.redshift)

    @property
    def source_frame_total_mass(self) -> Float[Array, "batch sample"]:
        """The total mass in the source frame, M/(1 + z)."""
        return self.total_mass / (1 + self.redshift)

    @property
    def source_frame_chirp_mass(self) -> Float[Array, "batch sample"]:
        """The chirp mass in the source frame, 𝓜/(1 + z)."""
        return self.chirp_mass / (1 + self.redshift)

    def to_total_mass_and_mass_ratio[SourcesType: "BlackHoleBinarySources"](
        self: SourcesType, *, inverted: bool = False
    ) -> SourcesType:
        """Return the holder with the total mass and mass ratio in place of the masses.

        The ratio is q = m₂/m₁, or Q = m₁/m₂ with ``inverted=True``. The
        log-prior moves by the Jacobian M/m₁², or M/m₂² for Q; see
        :class:`~mixed_signals.gw.lisa.transforms.MassPairTransform`. A holder
        that already has the total mass and the same ratio is returned
        unchanged; any other form passes through the component masses, so the
        log-prior moves by both Jacobians.

        Args:
            inverted: ``True`` for Q = m₁/m₂ in place of q = m₂/m₁.

        Raises:
            ValueError: If a mass parameter has a prior bound, which does not
                map; build the holder without it and add it with
                :meth:`~LisaSources.with_bounds`.
        """
        return self._with_masses(chirp=False, inverted=inverted)

    def to_chirp_mass_and_mass_ratio[SourcesType: "BlackHoleBinarySources"](
        self: SourcesType, *, inverted: bool = False
    ) -> SourcesType:
        """Return the holder with the chirp mass and mass ratio in place of the masses.

        The ratio is q = m₂/m₁, or Q = m₁/m₂ with ``inverted=True``. The
        log-prior moves by the Jacobian 𝓜/m₁², or 𝓜/m₂² for Q. A holder that
        already has the chirp mass and the same ratio is returned unchanged;
        any other form passes through the component masses.

        Args:
            inverted: ``True`` for Q = m₁/m₂ in place of q = m₂/m₁.

        Raises:
            ValueError: If a mass parameter has a prior bound, which does not
                map.
        """
        return self._with_masses(chirp=True, inverted=inverted)

    def to_component_masses[SourcesType: "BlackHoleBinarySources"](
        self: SourcesType,
    ) -> SourcesType:
        """Return the holder with the component masses m₁ and m₂, the canonical form.

        A holder in any mass form converts back, with either mass ratio.
        """
        form = self._mass_form()
        if form == (_PRIMARY, _SECONDARY):
            return self
        mass, ratio = form
        names = self.full_space.names
        transform = MassPairTransform(
            primary=names.index(mass),
            secondary=names.index(ratio),
            chirp=mass == _CHIRP,
            to_ratio=False,
            inverted=ratio == _INVERTED_RATIO,
        )
        return self._apply(transform, transform.touched_names(self.full_space))

    def _mass_form(self) -> tuple[str, str]:
        """Return the names of the two mass parameters, in their current form."""
        names = self.full_space.names
        for form in _MASS_FORMS:
            if all(name in names for name in form):
                return form
        raise ValueError("the parameter vector holds no mass parameters")

    def _with_masses[SourcesType: "BlackHoleBinarySources"](
        self: SourcesType, *, chirp: bool, inverted: bool
    ) -> SourcesType:
        """Return the holder with the total or the chirp mass and the mass ratio."""
        target = (_CHIRP if chirp else _TOTAL, _INVERTED_RATIO if inverted else _RATIO)
        if self._mass_form() == target:
            return self
        components = self.to_component_masses()
        names = components.full_space.names
        transform = MassPairTransform(
            primary=names.index(_PRIMARY),
            secondary=names.index(_SECONDARY),
            chirp=chirp,
            to_ratio=True,
            inverted=inverted,
        )
        return components._apply(
            transform, transform.touched_names(components.full_space)
        )

    def _component_masses(
        self,
    ) -> tuple[Float[Array, "batch sample"], Float[Array, "batch sample"]]:
        """Return m₁ and m₂, whichever mass parameters the holder carries.

        The columns convert without a holder transform, so that a holder with
        prior bounds on its mass parameters keeps its properties.
        """
        mass, ratio = self._mass_form()
        if (mass, ratio) == (_PRIMARY, _SECONDARY):
            return self._column(_PRIMARY), self._column(_SECONDARY)
        transform = MassPairTransform(
            primary=0,
            secondary=1,
            chirp=mass == _CHIRP,
            to_ratio=False,
            inverted=ratio == _INVERTED_RATIO,
        )
        pair = transform.forward(
            jnp.stack([self._column(mass), self._column(ratio)], axis=-1)
        )
        return pair[..., 0], pair[..., 1]


def _read_conventions(
    cls: type[LisaSources],
    *,
    epoch: float,
    sky_frame: SkyFrame | str,
    source_frame: SourceFrame | str | None,
) -> LisaConventions:
    """Return the conventions of a new holder, with the rule on the source frame."""
    if cls.spinning and source_frame is None:
        raise ValueError(
            f"{cls.__name__} needs a source_frame, one of"
            f" {', '.join(repr(frame.value) for frame in SourceFrame)}"
        )
    if not cls.spinning and source_frame is not None:
        raise ValueError(f"{cls.__name__} has no source frame, got {source_frame!r}")
    return LisaConventions(sky_frame=sky_frame, epoch=epoch, source_frame=source_frame)


def _decode_id(value: object) -> str:
    """Return a source id as a string.

    Strings stay as they are, UTF-8 bytes decode, and integers, which catpylisa
    uses for source ids, become their decimal string. Whole-number floats of
    magnitude below 2⁵³ become the decimal string of their integer, so 5.0
    gives "5"; see the module docstring. Booleans are refused.

    Raises:
        TypeError: If the id is a boolean or of another type.
        ValueError: If the id is a float that is not whole, not finite or of
            magnitude 2⁵³ or more.
    """
    if isinstance(value, bool | np.bool_):
        raise TypeError("source ids must not be booleans")
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, str):
        return str(value)
    if isinstance(value, int | np.integer):
        return str(int(value))
    if isinstance(value, float | np.floating):
        number = float(value)
        # * float.is_integer is False for NaN, infinities and fractions.
        if not number.is_integer():
            raise ValueError(
                f"a float source id must be finite and whole, got {value!r}"
            )
        if abs(number) >= _FLOAT_ID_LIMIT:
            raise ValueError(
                "a float source id must have a magnitude below 2**53, where float64"
                f" holds every integer, got {value!r}"
            )
        return str(int(number))
    raise TypeError(
        "source ids must be strings, UTF-8 bytes, integers or whole-number floats,"
        " which become their decimal string or that of their integer, got"
        f" {type(value).__name__}"
    )


def _read_ids(source_ids: Iterable[object]) -> tuple[str, ...]:
    """Return the source ids as strings, of at least one source."""
    ids = tuple(_decode_id(value) for value in source_ids)
    if not ids:
        raise ValueError("a holder needs at least one source")
    return ids


def _resolve_names(
    cls: type[LisaSources],
    names: Iterable[str],
    frame: SkyFrame,
    *,
    special: Mapping[str, str] | None = None,
    ignore: Collection[str] = (),
    vector_names: Sequence[str] | None = None,
    context: str,
) -> dict[str, str]:
    """Map input names to the canonical names of a frame, through the aliases.

    Names in ``ignore`` are dropped, and names in ``special`` map to their
    targets. A name of the other sky frame raises a ConventionError, and any
    other unknown name a ValueError. With ``vector_names``, the names of a
    holder's current vector also resolve to themselves, and every name must
    resolve to one of them.
    """
    own = [rename_to_frame(spec.name, frame) for spec in cls.parameter_specs]
    other_frame = SkyFrame.ECLIPTIC if frame is SkyFrame.ICRS else SkyFrame.ICRS
    other = {rename_to_frame(spec.name, other_frame) for spec in cls.parameter_specs}
    aliases, other_aliases = frame_aliases(cls.alias_table, frame)
    resolved: dict[str, str] = {}
    unknown = []
    for key in names:
        if key in ignore:
            continue
        if special is not None and key in special:
            resolved[key] = special[key]
        elif key in own:
            resolved[key] = key
        elif key in (other - set(own)) or key in other_aliases:
            raise ConventionError(
                f"{key!r} names a parameter of the {other_frame.value} frame, but the"
                f" holder is in the {frame.value} frame; convert the samples first"
            )
        elif key in aliases:
            resolved[key] = aliases[key]
        elif vector_names is not None and key in vector_names:
            resolved[key] = key
        else:
            unknown.append(key)
    if vector_names is not None:
        unknown += [
            key for key, target in resolved.items() if target not in vector_names
        ]
    if unknown:
        listing = (
            f"the parameters of {cls.__name__} are {own}, and `ignore` drops fields"
            if vector_names is None
            else f"the holder carries {list(vector_names)}"
        )
        raise ValueError(f"unknown {context} fields {unknown}; {listing}")
    targets: dict[str, str] = {}
    for key, target in resolved.items():
        if target in targets:
            raise ValueError(
                f"{context} fields {targets[target]!r} and {key!r} are the same"
                f" parameter {target!r}"
            )
        targets[target] = key
    return resolved


def _read_fields(posterior: object, source: str) -> dict[str, np.ndarray]:
    """Return the one-dimensional fields of one source, from a table or a mapping."""
    if isinstance(posterior, np.ndarray) and posterior.dtype.names is not None:
        fields = {key: np.asarray(posterior[key]) for key in posterior.dtype.names}
    elif isinstance(posterior, Mapping):
        fields = {str(key): np.asarray(value) for key, value in posterior.items()}
    else:
        raise TypeError(
            f"the posterior of source {source!r} must be a structured array or a"
            f" mapping, got {type(posterior).__name__}"
        )
    if not fields:
        raise ValueError(f"the posterior of source {source!r} holds no fields")
    for key, array in fields.items():
        if array.ndim != 1 or array.size == 0:
            raise ValueError(
                f"field {key!r} of source {source!r} must be a one-dimensional array"
                f" with samples, got shape {array.shape}"
            )
        if array.dtype.kind not in "iuf":
            raise ValueError(
                f"field {key!r} of source {source!r} must be numeric, got dtype"
                f" {array.dtype}"
            )
    if len({array.size for array in fields.values()}) > 1:
        raise ValueError(f"the fields of source {source!r} differ in length")
    return fields


def _read_source_n_eff(n_eff: object, ids: tuple[str, ...]) -> object:
    """Return ``n_eff`` as one number per source, or ``None`` for the Kish value."""
    if not isinstance(n_eff, Mapping):
        return n_eff
    by_id = {_decode_id(key): value for key, value in n_eff.items()}
    missing = [source for source in ids if source not in by_id]
    if missing:
        raise ValueError(f"n_eff lacks the sources {missing}")
    return [by_id[source] for source in ids]


def _matrix(name: str, value: object, shape: tuple[int, int]) -> np.ndarray:
    """Return a per-sample input as a float64 array of the given shape."""
    array = np.asarray(value)
    if array.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be numeric, got dtype {array.dtype}")
    if array.shape != shape:
        raise ValueError(f"{name} must be shaped (B, N) = {shape}, got {array.shape}")
    return array.astype(np.float64)


def _domain_text(spec: ParameterSpec) -> str:
    """Describe the physical domain of a spec in words or interval notation."""
    lower, upper = spec.bounds
    if lower == 0.0 and math.isinf(upper) and not any(spec.closed):
        return "strictly positive"
    opening = "[" if spec.closed[0] else "("
    closing = "]" if spec.closed[1] else ")"
    return f"in {opening}{lower}, {upper}{closing}"


def _fail_where(
    name: str,
    rule: str,
    offending: np.ndarray,
    positive: np.ndarray,
    ids: Sequence[str],
) -> None:
    """Raise a ValueError that names the parameter, the rule and the offending count."""
    counted = offending & positive
    if counted.any():
        first = ids[int(np.argmax(counted.any(axis=1)))]
        raise ValueError(
            f"{name} must be {rule}, but {int(counted.sum())} of the"
            f" {int(positive.sum())} positive-weight values are not (first in"
            f" source {first!r})"
        )


def _check_domains(
    cls: type[LisaSources],
    host: np.ndarray,
    names: Sequence[str],
    positive: np.ndarray,
    ids: Sequence[str],
) -> None:
    """Check finiteness, domains and the mass order of the positive-weight samples."""
    for position, (spec, name) in enumerate(
        zip(cls.parameter_specs, names, strict=True)
    ):
        column = host[..., position]
        _fail_where(name, "finite", ~np.isfinite(column), positive, ids)
        if spec.topology is Topology.CIRCLE:
            continue
        lower, upper = spec.bounds
        below = column < lower if spec.closed[0] else column <= lower
        above = column > upper if spec.closed[1] else column >= upper
        _fail_where(name, _domain_text(spec), below | above, positive, ids)
    if _PRIMARY in names and _SECONDARY in names:
        primary = host[..., names.index(_PRIMARY)]
        secondary = host[..., names.index(_SECONDARY)]
        _fail_where(
            _PRIMARY,
            f"at least {_SECONDARY} (m1 >= m2, SGS Section 2.3)",
            primary < secondary,
            positive,
            ids,
        )


def _check_prior_bounds(
    host: np.ndarray,
    names: Sequence[str],
    prior_bounds: Mapping[str, tuple[float, float]],
    positive: np.ndarray,
    ids: Sequence[str],
) -> None:
    """Check that the positive-weight samples lie in their closed prior bounds.

    The bounded estimators and the railing scores assume that the samples lie
    inside the bounds of the space.
    """
    for key, (lower, upper) in prior_bounds.items():
        column = host[..., names.index(key)]
        _fail_where(
            key,
            f"inside its prior bounds [{lower!r}, {upper!r}]",
            (column < lower) | (column > upper),
            positive,
            ids,
        )


def _prior_parameter(
    parameter: Parameter,
    domain: ParameterSpec | Parameter,
    override: tuple[float, float] | None,
) -> Parameter:
    """Return a parameter whose bounds are a prior bound inside its physical domain.

    ``domain`` is the spec of the type, or the parameter of a mass form, that
    holds the topology and the bounds of the physical domain.
    """
    if override is None:
        return parameter
    lower, upper = (float(override[0]), float(override[1]))
    if domain.topology is Topology.CIRCLE:
        raise ValueError(
            f"{parameter.name!r} is a circle, whose period is its domain; it takes no"
            " prior bounds"
        )
    domain_lower, domain_upper = domain.bounds
    if not (domain_lower <= lower < upper <= domain_upper):
        raise ValueError(
            f"the prior bounds {(lower, upper)} of {parameter.name!r} must lie inside"
            f" its domain {(domain_lower, domain_upper)}"
        )
    topology = (
        Topology.REAL if math.isinf(lower) and math.isinf(upper) else Topology.INTERVAL
    )
    return Parameter(
        name=parameter.name,
        label=parameter.label,
        unit=parameter.unit,
        topology=topology,
        bounds=(lower, upper),
    )


def _physical_domain(sources: LisaSources, name: str) -> ParameterSpec | Parameter:
    """Return the physical domain of a parameter of the full vector of a holder.

    The domain comes from the table of the type, or from the mass forms of
    :class:`~mixed_signals.gw.lisa.transforms.MassPairTransform`.
    """
    frame = sources.conventions.sky_frame
    for spec in sources.parameter_specs:
        if rename_to_frame(spec.name, frame) == name:
            return spec
    if name in _MASS_FORM_PARAMETERS:
        return _MASS_FORM_PARAMETERS[name]
    raise ValueError(
        f"{name!r} has no physical domain to bound inside; bounds apply to the"
        f" parameters of {type(sources).__name__} and to the mass forms"
    )


def _build[SourcesType: "LisaSources"](
    cls: type[SourcesType],
    columns: Mapping[str, object],
    *,
    ids: tuple[str, ...],
    name: str,
    conventions: LisaConventions,
    extras: Mapping[str, object],
    n_eff: object,
    bounds: Mapping[str, tuple[float, float]] | None,
    detections: object,
) -> SourcesType:
    """Assemble a holder from resolved columns shaped (B, N), after the checks."""
    frame = conventions.sky_frame
    names = [rename_to_frame(spec.name, frame) for spec in cls.parameter_specs]
    count = len(ids)
    arrays = {key: np.asarray(value) for key, value in columns.items()}
    if not arrays:
        raise ValueError("give at least one parameter")
    shapes = {array.shape for array in arrays.values()}
    shape = next(iter(shapes))
    if len(shapes) > 1 or len(shape) != 2 or shape[0] != count:
        raise ValueError(
            f"every parameter must be shaped (B, N) with B = {count} sources, got"
            f" {sorted(shapes)}"
        )
    _warn_narrow_arrays(arrays, "parameter")
    missing = [
        key
        for spec, key in zip(cls.parameter_specs, names, strict=True)
        if key not in arrays and spec.default is None
    ]
    if missing:
        raise ValueError(
            f"missing required parameters {missing} of {cls.__name__}; defaults fill"
            " only the optional ones"
        )
    for key, array in arrays.items():
        if array.dtype.kind not in "iuf":
            raise ValueError(
                f"parameter {key!r} must be numeric, got dtype {array.dtype}"
            )
    stacked = np.stack(
        [
            arrays[key].astype(np.float64)
            if key in arrays
            else np.full(shape, spec.default, dtype=np.float64)
            for spec, key in zip(cls.parameter_specs, names, strict=True)
        ],
        axis=-1,
    )
    log_weights = (
        None
        if extras.get("log_weights") is None
        else _matrix("log_weights", extras["log_weights"], shape)
    )
    positive = (
        np.ones(shape, dtype=bool) if log_weights is None else log_weights > -np.inf
    )
    values = jnp.asarray(stacked)
    for position, spec in enumerate(cls.parameter_specs):
        if spec.topology is Topology.CIRCLE:
            values = values.at[..., position].set(
                wrap(
                    values[..., position],
                    spec.bounds[0],
                    spec.bounds[1] - spec.bounds[0],
                )
            )
    host = np.asarray(values)
    _check_domains(cls, host, names, positive, ids)
    prior_bounds = _read_bounds(cls, bounds, frame)
    parameters = tuple(
        _prior_parameter(frame_parameter(spec, frame), spec, prior_bounds.get(key))
        for spec, key in zip(cls.parameter_specs, names, strict=True)
    )
    _check_prior_bounds(host, names, prior_bounds, positive, ids)
    full_space = ParamSpace(
        parameters=parameters,
        spheres=tuple(frame_sphere(sphere, frame) for sphere in cls.sphere_specs),
    )
    sample_set = SampleSet(
        values[:, None],
        log_weights=None if log_weights is None else log_weights[:, None],
        n_eff=_read_n_eff(n_eff, count),
        log_like=_optional_matrix("log_like", extras.get("log_like"), shape),
        log_prior=_optional_matrix("log_prior", extras.get("log_prior"), shape),
    )
    return cls(
        name=name,
        source_ids=ids,
        conventions=conventions,
        full_space=full_space,
        values=sample_set,
        detections=_join_detections(detections, ids),
        bounds=prior_bounds,
    )


def _optional_matrix(
    name: str, value: object, shape: tuple[int, int]
) -> np.ndarray | None:
    """Return a per-sample input shaped (B, 1, N), or ``None`` when it is not given."""
    if value is None:
        return None
    return _matrix(name, value, shape)[:, None]


def _read_n_eff(n_eff: object, count: int) -> np.ndarray | None:
    """Return the effective sample sizes shaped (B, 1), or ``None`` for Kish."""
    if n_eff is None:
        return None
    array = np.asarray(n_eff, dtype=np.float64)
    if array.ndim == 0:
        array = np.full(count, float(array))
    if array.shape != (count,):
        raise ValueError(
            f"n_eff must be one number or one per source, {count}, got shape"
            f" {array.shape}"
        )
    return array[:, None]


def _read_bounds(
    cls: type[LisaSources],
    bounds: Mapping[str, tuple[float, float]] | None,
    frame: SkyFrame,
    *,
    vector_names: Sequence[str] | None = None,
) -> dict[str, tuple[float, float]]:
    """Return the prior bounds by canonical name, resolving aliases."""
    if not bounds:
        return {}
    resolved = _resolve_names(
        cls, bounds, frame, vector_names=vector_names, context="bounds"
    )
    return {
        resolved[key]: (float(value[0]), float(value[1]))
        for key, value in bounds.items()
    }


def _join_detections(detections: object, ids: tuple[str, ...]) -> pl.DataFrame | None:
    """Join a table of detections to the sources in holder order, or return ``None``."""
    if detections is None:
        return None
    if isinstance(detections, pl.DataFrame):
        columns = {key: detections[key].to_list() for key in detections.columns}
    elif isinstance(detections, np.ndarray) and detections.dtype.names is not None:
        columns = {
            key: (
                np.char.decode(detections[key], "utf-8").tolist()
                if detections[key].dtype.kind == "S"
                else detections[key].tolist()
            )
            for key in detections.dtype.names
        }
    else:
        raise TypeError(
            "detections must be a NumPy structured array or a polars table, got"
            f" {type(detections).__name__}"
        )
    if "source_id" not in columns:
        raise ValueError("the detections need a 'source_id' column")
    columns["source_id"] = [_decode_id(value) for value in columns["source_id"]]
    table = pl.DataFrame(columns)
    if table["source_id"].n_unique() != table.height:
        raise ValueError("the detections hold a source id more than once")
    return pl.DataFrame({"source_id": list(ids)}).join(
        table, on="source_id", how="left", maintain_order="left"
    )


def _sampled_positions(values: SampleSet, space: ParamSpace) -> tuple[int, ...]:
    """Return the positions of the parameters that vary within a source.

    A parameter varies when its positive-weight values differ within some
    source with at least two of them. Where no source has two, every parameter
    counts as sampled. A sphere with one sampled axis keeps both.
    """
    samples = np.asarray(values.samples[:, 0])
    positive = np.asarray(values.log_weights[:, 0]) > -np.inf
    several = positive.sum(axis=1) >= 2
    if several.any():
        mask = positive[several][..., None]
        high = np.where(mask, samples[several], -np.inf).max(axis=1)
        low = np.where(mask, samples[several], np.inf).min(axis=1)
        varying = (high != low).any(axis=0)
    else:
        varying = np.ones(samples.shape[-1], dtype=bool)
    for sphere in space.spheres:
        positions = (space.index(sphere.azimuth), space.index(sphere.polar))
        if varying[list(positions)].any():
            varying[list(positions)] = True
    return tuple(int(position) for position in np.flatnonzero(varying))
