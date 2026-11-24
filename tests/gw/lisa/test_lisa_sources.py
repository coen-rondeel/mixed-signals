"""Tests for the holders of LISA sources: constructors, checks, padding, settings."""

import dataclasses
import math
import re
from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.lib.recfunctions as recfunctions
import polars as pl
import pytest

from mixed_signals.core.space import ParamSpace
from mixed_signals.core.topology import PolarForm, Topology
from mixed_signals.errors import ConventionError, PrecisionWarning
from mixed_signals.gw.lisa import (
    ExtremeMassRatioInspirals,
    GalacticBinaries,
    LisaSources,
    MassiveBlackHoleBinaries,
    SkyFrame,
    StellarOriginBinaryBlackHoles,
)

type Builder = Callable[..., dict[str, np.ndarray]]

# * Each type with its source frame, which the spinning types require.
TYPES: dict[str, tuple[type[LisaSources], str | None]] = {
    "galactic_binary": (GalacticBinaries, None),
    "massive_black_hole_binary": (MassiveBlackHoleBinaries, "J"),
    "extreme_mass_ratio_inspiral": (ExtremeMassRatioInspirals, "FEW"),
    "stellar_origin_binary_black_hole": (StellarOriginBinaryBlackHoles, "J"),
}
COUNTS = {"alpha": 30, "beta": 22, "gamma": 26}
LISA_EMRI_FIELDS = (
    "primary_mass_det_frame",
    "secondary_mass_det_frame",
    "chi_1",
    "polar_angle_spin1",
    "azimuthal_angle_spin1",
    "semi_latus_rectum_tref",
    "eccentricity_tref",
    "azimuthal_orbit_phase_tref",
    "mean_anomaly_tref",
    "luminosity_distance",
    "right_ascension",
    "declination",
    "inclination_tref",
    "polar_orbit_phase_tref",
)


def holder(
    posteriors: Builder,
    kind: str,
    counts: dict[str, int] | None = None,
    **keywords: Any,
) -> LisaSources:
    """Build the holder of a source type from synthetic posteriors."""
    cls, source_frame = TYPES[kind]
    data = posteriors(kind, counts or COUNTS)
    options: dict[str, Any] = {
        "name": "fit",
        "epoch": 0.0,
        "source_frame": source_frame,
    } | keywords
    return cls.from_posteriors(data, **options)


def arrays_of(
    posteriors: Builder, kind: str, counts: dict[str, int] | None = None
) -> dict[str, np.ndarray]:
    """Return the parameters of synthetic posteriors as arrays shaped (B, N)."""
    data = posteriors(kind, counts or {"alpha": 20, "beta": 20})
    return {
        name: np.stack([table[name] for table in data.values()])
        for name in data["alpha"].dtype.names or ()
        if name not in ("loglikelihood", "logprior")
    }


def gb_arrays(posteriors: Builder, **overrides: Any) -> dict[str, np.ndarray]:
    """Return Galactic-binary arrays shaped (2, 20), with some entries replaced."""
    return arrays_of(posteriors, "galactic_binary") | overrides


# * Construction from posteriors


@pytest.mark.parametrize("kind", TYPES)
def test_from_posteriors_builds_every_type(lisa_posteriors: Builder, kind: str) -> None:
    """Unequal sources pad to the largest N, and the pieces fit together."""
    cls, source_frame = TYPES[kind]
    sources = holder(lisa_posteriors, kind)
    assert isinstance(sources, cls)
    assert len(sources) == 3
    assert sources.source_ids == ("alpha", "beta", "gamma")
    assert sources.values.shape.sample == 30
    assert sources.values.shape.batch == 3
    assert sources.values.shape.posterior == 1
    assert sources.sample_counts.tolist() == [30, 22, 26]
    assert sources.conventions.source_frame == source_frame
    assert sources.name == "fit"
    assert "alpha" not in repr(sources)
    assert repr(sources).startswith(cls.__name__)
    assert sources.sample_set.shape.param == len(sources.space)
    assert sources.parameter_names == sources.full_space.names


def test_padding_copies_the_first_sample_with_a_zero_weight(
    lisa_posteriors: Builder,
) -> None:
    """Shorter sources repeat their first sample and log densities at weight zero."""
    data = lisa_posteriors("galactic_binary", COUNTS)
    sources = GalacticBinaries.from_posteriors(data, name="fit", epoch=0.0)
    log_weights = np.asarray(sources.values.log_weights[:, 0])
    assert (log_weights[0] == 0.0).all()
    assert (log_weights[1, :22] == 0.0).all()
    assert np.isneginf(log_weights[1, 22:]).all()
    samples = np.asarray(sources.values.samples[1, 0])
    np.testing.assert_array_equal(samples[22:], np.repeat(samples[:1], 8, axis=0))
    log_like = np.asarray(sources.values.log_like)[1, 0]
    assert log_like[0] == data["beta"]["loglikelihood"][0]
    assert (log_like[22:] == log_like[0]).all()
    log_prior = np.asarray(sources.values.log_prior)[2, 0]
    assert log_prior[0] == data["gamma"]["logprior"][0]
    assert (log_prior[26:] == log_prior[0]).all()


def test_from_posteriors_decodes_bytes_and_resolves_aliases(
    lisa_posteriors: Builder,
) -> None:
    """Bytes ids decode as UTF-8, and the L2 spelling polarization is an alias."""
    data = lisa_posteriors("galactic_binary", {"alpha": 12, "beta": 12})
    with_bytes: dict[str | bytes, Any] = {b"alpha": data["alpha"], "beta": data["beta"]}
    sources = GalacticBinaries.from_posteriors(with_bytes, name="fit", epoch=0.0)
    assert sources.source_ids == ("alpha", "beta")
    assert "polarisation_icrs" in sources.parameter_names
    renamed = {
        name: data["alpha"][name]
        for name in data["alpha"].dtype.names or ()
        if name != "polarization"
    }
    renamed["psi"] = data["alpha"]["polarization"]
    again = GalacticBinaries.from_posteriors({"x": renamed}, name="fit", epoch=0.0)
    assert again.parameter_names == sources.parameter_names


def test_from_posteriors_reads_mappings_of_fields(lisa_posteriors: Builder) -> None:
    """A mapping of field to one-dimensional array is a posterior too."""
    table = lisa_posteriors("galactic_binary", {"alpha": 15})["alpha"]
    mapping = {name: table[name] for name in table.dtype.names or ()}
    sources = GalacticBinaries.from_posteriors(
        {"alpha": mapping}, name="fit", epoch=0.0
    )
    assert sources.values.shape.sample == 15


def test_log_fields_become_log_densities_and_ignore_drops_fields(
    lisa_posteriors: Builder,
) -> None:
    """Loglikelihood and logprior fill log_like and log_prior; ignore drops fields."""
    data = lisa_posteriors("galactic_binary", {"alpha": 10, "beta": 10})
    sources = GalacticBinaries.from_posteriors(data, name="fit", epoch=0.0)
    np.testing.assert_array_equal(
        np.asarray(sources.values.log_like)[:, 0],
        np.stack([data["alpha"]["loglikelihood"], data["beta"]["loglikelihood"]]),
    )
    assert sources.sample_set.log_prior is not None
    without = {
        key: recfunctions.drop_fields(table, ["loglikelihood", "logprior"])
        for key, table in data.items()
    }
    assert (
        GalacticBinaries.from_posteriors(without, name="fit", epoch=0.0).values.log_like
        is None
    )
    extra = {
        key: recfunctions.append_fields(table, "chain", np.zeros(10), usemask=False)
        for key, table in data.items()
    }
    with pytest.raises(ValueError, match=r"unknown posterior fields \['chain'\]"):
        GalacticBinaries.from_posteriors(extra, name="fit", epoch=0.0)
    kept = GalacticBinaries.from_posteriors(
        extra, name="fit", epoch=0.0, ignore=("chain",)
    )
    assert kept.parameter_names == sources.parameter_names


def test_emri_rejects_the_inclination_cosine_alias(lisa_posteriors: Builder) -> None:
    """FEW's x_I is a cosine, so the alias xI of catpylisa is not accepted."""
    data = lisa_posteriors("extreme_mass_ratio_inspiral", {"alpha": 8})["alpha"]
    fields = {name: data[name] for name in data.dtype.names or ()}
    fields["xI"] = fields.pop("inclination_tref")
    with pytest.raises(ValueError, match=r"unknown posterior fields \['xI'\]"):
        ExtremeMassRatioInspirals.from_posteriors(
            {"alpha": fields}, name="fit", epoch=0.0, source_frame="FEW"
        )


def test_polarisation_aliases_do_not_exist_for_emri(lisa_posteriors: Builder) -> None:
    """EMRIs hold no polarisation, so psi is an unknown field there."""
    fields = {
        name: lisa_posteriors("extreme_mass_ratio_inspiral", {"alpha": 8})["alpha"][
            name
        ]
        for name in LISA_EMRI_FIELDS
    }
    fields["psi"] = np.zeros(8)
    with pytest.raises(ValueError, match="unknown posterior fields"):
        ExtremeMassRatioInspirals.from_posteriors(
            {"alpha": fields}, name="fit", epoch=0.0, source_frame="FEW"
        )


def test_from_posteriors_input_errors(lisa_posteriors: Builder) -> None:
    """Bad posteriors, fields of differing sources, shapes and dtypes raise."""
    table = lisa_posteriors("galactic_binary", {"alpha": 10})["alpha"]
    fields = {name: table[name] for name in table.dtype.names or ()}

    def build(posteriors: dict[str, Any]) -> GalacticBinaries:
        return GalacticBinaries.from_posteriors(posteriors, name="fit", epoch=0.0)

    with pytest.raises(TypeError, match="structured array or a mapping"):
        build({"alpha": [1.0, 2.0]})
    with pytest.raises(ValueError, match="holds no fields"):
        build({"alpha": {}})
    with pytest.raises(ValueError, match="one-dimensional array"):
        build({"alpha": fields | {"frequency": np.ones((10, 2))}})
    with pytest.raises(ValueError, match="one-dimensional array"):
        build({"alpha": {name: values[:0] for name, values in fields.items()}})
    with pytest.raises(ValueError, match="must be numeric"):
        build({"alpha": fields | {"frequency": np.array(["a"] * 10)}})
    with pytest.raises(ValueError, match="differ in length"):
        build({"alpha": fields | {"frequency": fields["frequency"][:5]}})
    fewer = {name: values for name, values in fields.items() if name != "loglikelihood"}
    with pytest.raises(ValueError, match="source 'beta' holds the fields"):
        build({"alpha": fields, "beta": fewer})


def test_from_posteriors_n_eff_variants(lisa_posteriors: Builder) -> None:
    """n_eff may be one number, a mapping of ids or a sequence."""
    data = lisa_posteriors("galactic_binary", {"alpha": 20, "beta": 16})
    kwargs: dict[str, Any] = {"name": "fit", "epoch": 0.0}
    number = GalacticBinaries.from_posteriors(data, n_eff=10.0, **kwargs)
    assert number.values.n_eff is not None
    np.testing.assert_allclose(np.asarray(number.values.n_eff[:, 0]), [10.0, 10.0])
    with_bytes: dict[str | bytes, Any] = {b"alpha": data["alpha"], "beta": data["beta"]}
    mapping = GalacticBinaries.from_posteriors(
        with_bytes,
        n_eff={"alpha": 8.0, b"beta": 12.0},
        **kwargs,
    )
    assert mapping.values.n_eff is not None
    np.testing.assert_allclose(np.asarray(mapping.values.n_eff[:, 0]), [8.0, 12.0])
    sequence = GalacticBinaries.from_posteriors(data, n_eff=[5.0, 6.0], **kwargs)
    assert sequence.values.n_eff is not None
    np.testing.assert_allclose(np.asarray(sequence.values.n_eff[:, 0]), [5.0, 6.0])
    assert GalacticBinaries.from_posteriors(data, **kwargs).values.n_eff is None
    with pytest.raises(ValueError, match="n_eff lacks the sources"):
        GalacticBinaries.from_posteriors(data, n_eff={"alpha": 5.0}, **kwargs)
    with pytest.raises(ValueError, match="n_eff must be one number or one per source"):
        GalacticBinaries.from_posteriors(data, n_eff=[5.0, 6.0, 7.0], **kwargs)
    with pytest.raises(ValueError, match="n_eff of problem"):
        GalacticBinaries.from_posteriors(data, n_eff=[5.0, 30.0], **kwargs)


def test_narrow_float_fields_raise_the_precision_warning(
    lisa_posteriors: Builder,
) -> None:
    """A float32 field is cast, and the cast is announced."""
    table = lisa_posteriors("galactic_binary", {"alpha": 10})["alpha"]
    fields = {name: table[name] for name in table.dtype.names or ()}
    fields["amplitude"] = fields["amplitude"].astype(np.float32)
    with pytest.warns(PrecisionWarning, match="float32"):
        GalacticBinaries.from_posteriors({"alpha": fields}, name="fit", epoch=0.0)


# * Construction from arrays and catalogs


@pytest.mark.parametrize("kind", TYPES)
def test_from_arrays_resolves_aliases_and_fills_defaults(
    lisa_posteriors: Builder, kind: str
) -> None:
    """Canonical names and aliases both work, and optional parameters take defaults."""
    cls, source_frame = TYPES[kind]
    values = arrays_of(lisa_posteriors, kind)
    sources = cls.from_arrays(
        values, source_ids=["a", b"b"], name="fit", epoch=1.5, source_frame=source_frame
    )
    assert sources.source_ids == ("a", "b")
    assert sources.conventions.epoch == 1.5
    assert len(sources.parameter_names) == len(cls.parameter_specs)


def test_defaults_fill_missing_optional_parameters(lisa_posteriors: Builder) -> None:
    """A missing phase and frequency derivative take their defaults, as settings."""
    values = gb_arrays(lisa_posteriors)
    del values["initial_phase"], values["frequency_dot"]
    sources = GalacticBinaries.from_arrays(
        values, source_ids=["a", "b"], name="fit", epoch=0.0
    )
    settings = sources.settings
    assert settings.columns == ["source_id", "frequency_dot", "initial_phase"]
    assert settings["initial_phase"].to_list() == [0.0, 0.0]
    assert "initial_phase" not in sources.space.names


def test_missing_required_parameters_raise(lisa_posteriors: Builder) -> None:
    """A parameter without a default must be given, the SOBBH f_ref too."""
    values = gb_arrays(lisa_posteriors)
    del values["amplitude"]
    with pytest.raises(
        ValueError, match=r"missing required parameters \['amplitude'\]"
    ):
        GalacticBinaries.from_arrays(
            values, source_ids=["a", "b"], name="fit", epoch=0.0
        )
    sobbh = arrays_of(lisa_posteriors, "stellar_origin_binary_black_hole")
    del sobbh["f_ref"]
    with pytest.raises(ValueError, match=r"missing required parameters \['f_ref'\]"):
        StellarOriginBinaryBlackHoles.from_arrays(
            sobbh, source_ids=["a", "b"], name="fit", epoch=0.0, source_frame="J"
        )


# * The MBHB reference frequency, which defaults to 0

REFERENCE_FREQUENCY = "frequency_at_reference_time"


def test_a_missing_reference_frequency_defaults_to_a_zero_setting(
    lisa_posteriors: Builder,
) -> None:
    """Both constructors fill 0 for every source, and the parameter is not sampled."""
    kwargs: dict[str, Any] = {"name": "fit", "epoch": 0.0, "source_frame": "J"}
    values = arrays_of(lisa_posteriors, "massive_black_hole_binary")
    del values[REFERENCE_FREQUENCY]
    from_arrays = MassiveBlackHoleBinaries.from_arrays(
        values, source_ids=["a", "b"], **kwargs
    )
    posteriors = {
        source: {name: table[name] for name in table.dtype.names or ()}
        for source, table in lisa_posteriors(
            "massive_black_hole_binary", COUNTS
        ).items()
    }
    for fields in posteriors.values():
        del fields[REFERENCE_FREQUENCY]
    from_posteriors = MassiveBlackHoleBinaries.from_posteriors(posteriors, **kwargs)
    for sources, expected in ((from_arrays, [0.0] * 2), (from_posteriors, [0.0] * 3)):
        assert sources.settings[REFERENCE_FREQUENCY].to_list() == expected
        assert REFERENCE_FREQUENCY not in sources.space.names
        assert sources.full_space[REFERENCE_FREQUENCY].bounds == (0.0, math.inf)
        assert sources.full_space[REFERENCE_FREQUENCY].unit == "Hz"


def test_an_explicit_zero_reference_frequency_builds(lisa_posteriors: Builder) -> None:
    """The domain [0, ∞) is closed at 0, so the default value is also accepted."""
    values = arrays_of(lisa_posteriors, "massive_black_hole_binary")
    values[REFERENCE_FREQUENCY] = np.zeros((2, 20))
    sources = MassiveBlackHoleBinaries.from_arrays(
        values, source_ids=["a", "b"], name="fit", epoch=0.0, source_frame="J"
    )
    assert sources.settings[REFERENCE_FREQUENCY].to_list() == [0.0, 0.0]
    assert REFERENCE_FREQUENCY not in sources.space.names


@pytest.mark.parametrize(
    ("value", "rule"),
    [(-1e-3, r"in \[0.0, inf\)"), (math.nan, "finite")],
)
def test_a_negative_or_missing_reference_frequency_value_is_refused(
    lisa_posteriors: Builder, value: float, rule: str
) -> None:
    """Negative values and NaN fail with the domain message, naming the count."""
    values = arrays_of(lisa_posteriors, "massive_black_hole_binary")
    values[REFERENCE_FREQUENCY] = values[REFERENCE_FREQUENCY].copy()
    values[REFERENCE_FREQUENCY][1, :3] = value
    with pytest.raises(
        ValueError,
        match=rf"{REFERENCE_FREQUENCY} must be {rule}.*3 of the 40.*source 'b'",
    ):
        MassiveBlackHoleBinaries.from_arrays(
            values, source_ids=["a", "b"], name="fit", epoch=0.0, source_frame="J"
        )


def test_a_catalog_with_a_nan_reference_frequency_needs_ignore(
    lisa_posteriors: Builder,
) -> None:
    """The MBHB catalog of catpylisa holds NaN there; ignoring it fills the default."""
    catalog = table_of(lisa_posteriors, "massive_black_hole_binary")
    catalog[REFERENCE_FREQUENCY] = np.nan
    kwargs: dict[str, Any] = {"name": "injections", "epoch": 0.0, "source_frame": "J"}
    with pytest.raises(ValueError, match=rf"{REFERENCE_FREQUENCY} must be finite"):
        MassiveBlackHoleBinaries.from_catalog(catalog, **kwargs)
    filled = MassiveBlackHoleBinaries.from_catalog(
        catalog, ignore=(REFERENCE_FREQUENCY,), **kwargs
    )
    assert filled.sample_counts.tolist() == [1, 1, 1]
    np.testing.assert_array_equal(filled._column(REFERENCE_FREQUENCY), 0.0)


def test_source_frame_is_required_for_spinning_types_only(
    lisa_posteriors: Builder,
) -> None:
    """Spinning types need a source frame, and Galactic binaries refuse one."""
    values = arrays_of(lisa_posteriors, "massive_black_hole_binary")
    with pytest.raises(ValueError, match="needs a source_frame"):
        MassiveBlackHoleBinaries.from_arrays(
            values, source_ids=["a", "b"], name="fit", epoch=0.0
        )
    with pytest.raises(ValueError, match="has no source frame"):
        GalacticBinaries.from_arrays(
            gb_arrays(lisa_posteriors),
            source_ids=["a", "b"],
            name="fit",
            epoch=0.0,
            source_frame="J",
        )
    with pytest.raises(ValueError, match="source_frame"):
        MassiveBlackHoleBinaries.from_arrays(
            values, source_ids=["a", "b"], name="fit", epoch=0.0, source_frame="XYZ"
        )


def test_names_of_the_other_sky_frame_raise_a_convention_error(
    lisa_posteriors: Builder,
) -> None:
    """ICRS names fail in an ecliptic holder, and ecliptic names in an ICRS holder."""
    values = gb_arrays(lisa_posteriors)
    kwargs: dict[str, Any] = {"source_ids": ["a", "b"], "name": "fit", "epoch": 0.0}
    with pytest.raises(
        ConventionError, match="ecliptic frame, but the holder is in the icrs"
    ):
        GalacticBinaries.from_arrays(
            values | {"ecliptic_longitude": values.pop("right_ascension")}, **kwargs
        )
    values = gb_arrays(lisa_posteriors)
    with pytest.raises(
        ConventionError, match="icrs frame, but the holder is in the ecliptic"
    ):
        GalacticBinaries.from_arrays(values, sky_frame="ecliptic", **kwargs)
    ecliptic = gb_arrays(lisa_posteriors)
    ecliptic["ecliptic_longitude"] = ecliptic.pop("right_ascension")
    ecliptic["ecliptic_latitude"] = ecliptic.pop("declination")
    ecliptic["polarisation_ecliptic"] = ecliptic.pop("polarization")
    holder_in_ecliptic = GalacticBinaries.from_arrays(
        ecliptic, sky_frame="ecliptic", **kwargs
    )
    assert "polarisation_ecliptic" in holder_in_ecliptic.parameter_names
    assert holder_in_ecliptic.space["ecliptic_longitude"].label == r"\lambda"
    aliased = gb_arrays(lisa_posteriors)
    aliased["lam"] = aliased.pop("right_ascension")
    with pytest.raises(ConventionError, match="ecliptic frame"):
        GalacticBinaries.from_arrays(aliased, **kwargs)
    aliased = gb_arrays(lisa_posteriors)
    aliased["polarisation_ecliptic"] = aliased.pop("polarization")
    with pytest.raises(ConventionError, match="ecliptic frame"):
        GalacticBinaries.from_arrays(aliased, **kwargs)


def test_ecliptic_aliases_resolve_in_ecliptic_holders(lisa_posteriors: Builder) -> None:
    """lam, lambda and beta name the sky position of an ecliptic holder."""
    values = gb_arrays(lisa_posteriors)
    values["lam"] = values.pop("right_ascension")
    values["beta"] = values.pop("declination")
    sources = GalacticBinaries.from_arrays(
        values,
        source_ids=["a", "b"],
        name="fit",
        epoch=0.0,
        sky_frame=SkyFrame.ECLIPTIC,
    )
    assert {"ecliptic_longitude", "ecliptic_latitude"} <= set(sources.parameter_names)


def test_unknown_and_repeated_names_raise(lisa_posteriors: Builder) -> None:
    """An unknown name, and two spellings of one parameter, raise ValueError."""
    kwargs: dict[str, Any] = {"source_ids": ["a", "b"], "name": "fit", "epoch": 0.0}
    with pytest.raises(ValueError, match=r"unknown values fields \['nothing'\]"):
        GalacticBinaries.from_arrays(
            gb_arrays(lisa_posteriors, nothing=np.zeros((2, 20))), **kwargs
        )
    with pytest.raises(ValueError, match="are the same parameter 'frequency'"):
        GalacticBinaries.from_arrays(
            gb_arrays(lisa_posteriors, f=np.ones((2, 20))), **kwargs
        )


def test_array_shapes_and_types_are_checked(lisa_posteriors: Builder) -> None:
    """Shapes other than (B, N), mismatches and non-numeric inputs raise."""
    kwargs: dict[str, Any] = {"source_ids": ["a", "b"], "name": "fit", "epoch": 0.0}
    with pytest.raises(ValueError, match="give at least one parameter"):
        GalacticBinaries.from_arrays({}, **kwargs)
    with pytest.raises(ValueError, match="shaped"):
        GalacticBinaries.from_arrays(
            gb_arrays(lisa_posteriors, frequency=np.ones(20)), **kwargs
        )
    with pytest.raises(ValueError, match="shaped"):
        GalacticBinaries.from_arrays(
            gb_arrays(lisa_posteriors, frequency=np.ones((2, 19))), **kwargs
        )
    with pytest.raises(ValueError, match="shaped"):
        GalacticBinaries.from_arrays(
            gb_arrays(lisa_posteriors),
            source_ids=["a", "b", "c"],
            name="fit",
            epoch=0.0,
        )
    with pytest.raises(ValueError, match="must be numeric"):
        GalacticBinaries.from_arrays(
            gb_arrays(lisa_posteriors, frequency=np.full((2, 20), "x")), **kwargs
        )
    with pytest.raises(ValueError, match="log_weights must be shaped"):
        GalacticBinaries.from_arrays(
            gb_arrays(lisa_posteriors), log_weights=np.zeros((2, 5)), **kwargs
        )
    with pytest.raises(ValueError, match="log_like must be numeric"):
        GalacticBinaries.from_arrays(
            gb_arrays(lisa_posteriors), log_like=np.full((2, 20), "x"), **kwargs
        )
    with pytest.raises(ValueError, match="epoch"):
        GalacticBinaries.from_arrays(
            gb_arrays(lisa_posteriors), **(kwargs | {"epoch": math.nan})
        )


def test_source_ids_must_be_unique_and_present(lisa_posteriors: Builder) -> None:
    """Duplicate ids raise, and so does an empty holder."""
    with pytest.raises(
        ValueError, match=r"source ids must be unique, repeated: \['a'\]"
    ):
        GalacticBinaries.from_arrays(
            gb_arrays(lisa_posteriors), source_ids=["a", "a"], name="fit", epoch=0.0
        )
    with pytest.raises(ValueError, match="at least one source"):
        GalacticBinaries.from_posteriors({}, name="fit", epoch=0.0)
    with pytest.raises(ValueError, match="at least one source"):
        GalacticBinaries.from_arrays({}, source_ids=[], name="fit", epoch=0.0)
    sources = holder(lisa_posteriors, "galactic_binary")
    with pytest.raises(ValueError, match="at least one source"):
        dataclasses.replace(sources, source_ids=())
    with pytest.raises(ValueError, match="holder has 2 sources"):
        dataclasses.replace(sources, source_ids=("alpha", "beta"))
    smaller = ParamSpace(parameters=sources.full_space.parameters[:3])
    with pytest.raises(ValueError, match="the space has 3 parameters"):
        dataclasses.replace(sources, full_space=smaller)


def test_from_arrays_keeps_weights_densities_and_sizes(
    lisa_posteriors: Builder,
) -> None:
    """log_weights, n_eff, log_like and log_prior reach the sample set."""
    values = gb_arrays(lisa_posteriors)
    log_weights = np.zeros((2, 20))
    log_weights[0, 15:] = -np.inf
    sources = GalacticBinaries.from_arrays(
        values,
        source_ids=["a", "b"],
        name="fit",
        epoch=0.0,
        log_weights=log_weights,
        n_eff=12.0,
        log_like=np.ones((2, 20)),
        log_prior=-np.ones((2, 20)),
    )
    assert sources.sample_counts.tolist() == [15, 20]
    sample_set = sources.sample_set
    assert sample_set.log_like is not None
    assert sample_set.log_prior is not None
    assert float(sample_set.log_prior[0, 0, 0]) == -1.0
    assert sample_set.n_eff is not None
    np.testing.assert_allclose(np.asarray(sample_set.n_eff[:, 0]), 12.0)


def table_of(posteriors: Builder, kind: str = "galactic_binary") -> np.ndarray:
    """Return a catalog of three sources as a structured array with byte ids."""
    data = posteriors(kind, {"alpha": 1, "beta": 1, "gamma": 1})
    names = [
        name
        for name in data["alpha"].dtype.names or ()
        if name not in ("loglikelihood", "logprior")
    ]
    catalog = np.zeros(
        3, dtype=[("source_id", "S8"), *[(name, "f8") for name in names]]
    )
    catalog["source_id"] = [source.encode() for source in data]
    for position, table in enumerate(data.values()):
        for name in names:
            catalog[name][position] = table[name][0]
    return catalog


def test_from_catalog_reads_structured_arrays_and_tables(
    lisa_posteriors: Builder,
) -> None:
    """A catalog has one sample per source, byte ids decode, every parameter is kept."""
    catalog = table_of(lisa_posteriors)
    sources = GalacticBinaries.from_catalog(catalog, name="injections", epoch=0.0)
    assert sources.source_ids == ("alpha", "beta", "gamma")
    assert sources.values.shape.sample == 1
    assert sources.settings.columns == ["source_id"]
    assert len(sources.space) == 8
    columns = {
        name: catalog[name] for name in catalog.dtype.names or () if name != "source_id"
    }
    for ids in (["alpha", "beta", "gamma"], [b"alpha", b"beta", b"gamma"]):
        frame = pl.DataFrame({"source_id": ids, **columns})
        from_frame = GalacticBinaries.from_catalog(frame, name="injections", epoch=0.0)
        assert from_frame.source_ids == sources.source_ids
        np.testing.assert_array_equal(
            np.asarray(from_frame.values.samples), np.asarray(sources.values.samples)
        )


@pytest.mark.parametrize("kind", TYPES)
def test_from_catalog_builds_every_type(lisa_posteriors: Builder, kind: str) -> None:
    """A catalog of any type has N = 1, no settings and every parameter in the space."""
    cls, source_frame = TYPES[kind]
    catalog = table_of(lisa_posteriors, kind)
    sources = cls.from_catalog(
        catalog, name="injections", epoch=2.0, source_frame=source_frame
    )
    assert isinstance(sources, cls)
    assert sources.values.shape.sample == 1
    assert sources.sample_counts.tolist() == [1, 1, 1]
    assert sources.settings.columns == ["source_id"]
    assert len(sources.space) == len(cls.parameter_specs)
    assert sources.conventions.epoch == 2.0


def test_from_catalog_errors_and_options(lisa_posteriors: Builder) -> None:
    """A missing id column, unknown columns, ignore and bad tables are handled."""
    catalog = table_of(lisa_posteriors)
    kwargs: dict[str, Any] = {"name": "injections", "epoch": 0.0}
    with pytest.raises(TypeError, match="structured array or a polars table"):
        GalacticBinaries.from_catalog([1, 2], **kwargs)
    with pytest.raises(ValueError, match="no id column 'name'"):
        GalacticBinaries.from_catalog(catalog, id_field="name", **kwargs)
    extra = recfunctions.append_fields(catalog, "snr", np.ones(3), usemask=False)
    with pytest.raises(ValueError, match=r"unknown catalog fields \['snr'\]"):
        GalacticBinaries.from_catalog(extra, **kwargs)
    ignored = GalacticBinaries.from_catalog(extra, ignore=("snr",), **kwargs)
    assert len(ignored) == 3
    renamed = recfunctions.rename_fields(catalog, {"source_id": "name"})
    named = GalacticBinaries.from_catalog(renamed, id_field="name", **kwargs)
    assert named.source_ids == ("alpha", "beta", "gamma")
    frame = pl.DataFrame({"source_id": ["x", "y"], "frequency": [1e-3, 2e-3]})
    with pytest.raises(ValueError, match="missing required parameters"):
        GalacticBinaries.from_catalog(frame, **kwargs)


def test_detections_join_in_holder_order(lisa_posteriors: Builder) -> None:
    """Extra rows drop, missing sources give nulls, and bytes ids compare as strings."""
    table = pl.DataFrame(
        {"source_id": ["gamma", "alpha", "other"], "snr": [9.0, 7.0, 1.0]}
    )
    sources = holder(lisa_posteriors, "galactic_binary", detections=table)
    assert sources.detections is not None
    assert sources.detections["source_id"].to_list() == ["alpha", "beta", "gamma"]
    assert sources.detections["snr"].to_list() == [7.0, None, 9.0]
    structured = np.zeros(2, dtype=[("source_id", "S8"), ("snr", "f8"), ("tag", "S4")])
    structured["source_id"] = [b"beta", b"alpha"]
    structured["snr"] = [3.0, 4.0]
    structured["tag"] = [b"x", b"y"]
    again = holder(lisa_posteriors, "galactic_binary", detections=structured)
    assert again.detections is not None
    assert again.detections["snr"].to_list() == [4.0, 3.0, None]
    assert again.detections["tag"].to_list() == ["y", "x", None]
    assert holder(lisa_posteriors, "galactic_binary").detections is None
    ints = np.zeros(1, dtype=[("source_id", "U8"), ("count", "i8")])
    ints["source_id"] = ["alpha"]
    assert (
        holder(lisa_posteriors, "galactic_binary", detections=ints).detections
        is not None
    )


def test_detections_errors(lisa_posteriors: Builder) -> None:
    """The table needs unique source ids and the right type."""
    with pytest.raises(TypeError, match="detections must be"):
        holder(lisa_posteriors, "galactic_binary", detections=[1, 2])
    with pytest.raises(ValueError, match="'source_id' column"):
        holder(
            lisa_posteriors, "galactic_binary", detections=pl.DataFrame({"snr": [1.0]})
        )
    duplicated = pl.DataFrame({"source_id": ["alpha", "alpha"], "snr": [1.0, 2.0]})
    with pytest.raises(ValueError, match="more than once"):
        holder(lisa_posteriors, "galactic_binary", detections=duplicated)


# * Integer source ids


def integer_keyed(posteriors: Builder, counts: dict[str, int]) -> dict[Any, Any]:
    """Return posteriors keyed by an int, an np.int64 and an np.int32."""
    data = posteriors("galactic_binary", counts)
    first, second, third = data.values()
    return {1: first, np.int64(2): second, np.int32(10): third}


def test_integer_posterior_keys_read_as_decimal_strings(
    lisa_posteriors: Builder,
) -> None:
    """The keys of catpylisa, ints and NumPy integers, become their decimal strings."""
    posteriors = integer_keyed(lisa_posteriors, {"a": 12, "b": 12, "c": 12})
    sources = GalacticBinaries.from_posteriors(posteriors, name="fit", epoch=0.0)
    assert sources.source_ids == ("1", "2", "10")
    only_numpy = {np.int64(5): posteriors[1], np.int64(6): posteriors[10]}
    assert GalacticBinaries.from_posteriors(
        only_numpy, name="fit", epoch=0.0
    ).source_ids == ("5", "6")


def test_an_integer_selects_by_position_and_an_integer_id_by_its_string(
    lisa_posteriors: Builder,
) -> None:
    """Sources with integer ids are selected by position or through their strings."""
    posteriors = integer_keyed(lisa_posteriors, {"a": 12, "b": 12, "c": 12})
    sources = GalacticBinaries.from_posteriors(posteriors, name="fit", epoch=0.0)
    assert sources[1].source_ids == ("2",)
    assert sources["10"].source_ids == ("10",)
    assert sources[["10", "1"]].source_ids == ("10", "1")
    with pytest.raises(KeyError, match="no source with the id '3'"):
        sources["3"]


def test_integer_ids_work_in_every_reader(lisa_posteriors: Builder) -> None:
    """from_arrays, the n_eff mapping and the catalog columns read integer ids."""
    values = gb_arrays(lisa_posteriors)
    kwargs: dict[str, Any] = {"name": "fit", "epoch": 0.0}
    mixed = GalacticBinaries.from_arrays(values, source_ids=[3, np.int64(4)], **kwargs)
    assert mixed.source_ids == ("3", "4")
    array = GalacticBinaries.from_arrays(values, source_ids=np.array([5, 6]), **kwargs)
    assert array.source_ids == ("5", "6")
    posteriors = integer_keyed(lisa_posteriors, {"a": 12, "b": 12, "c": 12})
    sized = GalacticBinaries.from_posteriors(
        posteriors, n_eff={1: 5.0, np.int64(2): 6.0, 10: 7.0}, **kwargs
    )
    assert sized.values.n_eff is not None
    np.testing.assert_allclose(np.asarray(sized.values.n_eff[:, 0]), [5.0, 6.0, 7.0])
    catalog = table_of(lisa_posteriors)
    columns = {
        name: catalog[name] for name in catalog.dtype.names or () if name != "source_id"
    }
    structured = recfunctions.append_fields(
        recfunctions.drop_fields(catalog, "source_id"),
        "source_id",
        np.array([11, 12, 13]),
        usemask=False,
    )
    from_structured = GalacticBinaries.from_catalog(structured, **kwargs)
    assert from_structured.source_ids == ("11", "12", "13")
    frame = pl.DataFrame({"source_id": [11, 12, 13], **columns})
    from_frame = GalacticBinaries.from_catalog(frame, **kwargs)
    assert from_frame.source_ids == ("11", "12", "13")
    np.testing.assert_array_equal(
        np.asarray(from_frame.values.samples),
        np.asarray(from_structured.values.samples),
    )


def test_detections_with_integer_ids_join_to_the_string_ids(
    lisa_posteriors: Builder,
) -> None:
    """An integer source_id column, polars or structured, compares as strings."""
    posteriors = integer_keyed(lisa_posteriors, {"a": 12, "b": 12, "c": 12})
    table = pl.DataFrame({"source_id": [10, 1, 99], "snr": [3.0, 1.0, 9.0]})
    sources = GalacticBinaries.from_posteriors(
        posteriors, name="fit", epoch=0.0, detections=table
    )
    assert sources.detections is not None
    assert sources.detections["source_id"].to_list() == ["1", "2", "10"]
    assert sources.detections["snr"].to_list() == [1.0, None, 3.0]
    structured = np.zeros(2, dtype=[("source_id", "i8"), ("snr", "f8")])
    structured["source_id"] = [2, 10]
    structured["snr"] = [4.0, 5.0]
    again = GalacticBinaries.from_posteriors(
        posteriors, name="fit", epoch=0.0, detections=structured
    )
    assert again.detections is not None
    assert again.detections["snr"].to_list() == [None, 4.0, 5.0]
    as_text = pl.DataFrame({"source_id": ["2"], "snr": [8.0]})
    text_join = GalacticBinaries.from_posteriors(
        posteriors, name="fit", epoch=0.0, detections=as_text
    )
    assert text_join.detections is not None
    assert text_join.detections["snr"].to_list() == [None, 8.0, None]


def test_booleans_and_other_types_are_not_source_ids(lisa_posteriors: Builder) -> None:
    """Booleans raise, and so do other types, with the list of the accepted types."""
    values = gb_arrays(lisa_posteriors)
    kwargs: dict[str, Any] = {"name": "fit", "epoch": 0.0}
    with pytest.raises(TypeError, match="must not be booleans"):
        GalacticBinaries.from_arrays(values, source_ids=[True, False], **kwargs)
    with pytest.raises(TypeError, match="must not be booleans"):
        GalacticBinaries.from_arrays(
            values, source_ids=[np.bool_(True), np.bool_(False)], **kwargs
        )
    with pytest.raises(
        TypeError, match="strings, UTF-8 bytes, integers or whole-number floats"
    ):
        GalacticBinaries.from_arrays(values, source_ids=[1 + 2j, 3 + 4j], **kwargs)
    data = lisa_posteriors("galactic_binary", {"a": 8, "b": 8})
    with pytest.raises(TypeError, match="must not be booleans"):
        GalacticBinaries.from_posteriors({True: data["a"], "b": data["b"]}, **kwargs)


# * Whole-number float source ids


def float_catalog(
    posteriors: Builder, ids: list[float], kind: str = "galactic_binary"
) -> np.ndarray:
    """Return a catalog of three sources whose source_id column is float64."""
    return recfunctions.append_fields(
        recfunctions.drop_fields(
            table_of(posteriors, kind), "source_id", usemask=False
        ),
        "source_id",
        np.asarray(ids, dtype=np.float64),
        usemask=False,
    )


def test_a_float64_id_column_reads_as_integer_strings(lisa_posteriors: Builder) -> None:
    """The float64 ids 0.0, 1.0 and 5.0 of catpylisa become '0', '1' and '5'."""
    kwargs: dict[str, Any] = {"name": "injections", "epoch": 0.0}
    catalog = float_catalog(lisa_posteriors, [0.0, 1.0, 5.0])
    assert catalog["source_id"].dtype == np.float64
    sources = GalacticBinaries.from_catalog(catalog, **kwargs)
    assert sources.source_ids == ("0", "1", "5")
    columns = {
        name: catalog[name] for name in catalog.dtype.names or () if name != "source_id"
    }
    frame = pl.DataFrame({"source_id": [0.0, 1.0, 5.0], **columns})
    assert frame["source_id"].dtype == pl.Float64
    assert GalacticBinaries.from_catalog(frame, **kwargs).source_ids == ("0", "1", "5")
    mbhb = float_catalog(lisa_posteriors, [0.0, 1.0, 5.0], "massive_black_hole_binary")
    mbhb[REFERENCE_FREQUENCY] = np.nan
    injections = MassiveBlackHoleBinaries.from_catalog(
        mbhb, ignore=(REFERENCE_FREQUENCY,), source_frame="J", **kwargs
    )
    assert injections.source_ids == ("0", "1", "5")
    assert injections["5"].source_ids == ("5",)


def test_whole_floats_of_every_kind_read_as_integer_strings(
    lisa_posteriors: Builder,
) -> None:
    """Python and NumPy floats, also negative, zero and 2**53 - 1, give integers."""
    values = gb_arrays(lisa_posteriors)
    kwargs: dict[str, Any] = {"name": "fit", "epoch": 0.0}
    for ids, expected in (
        ([np.float32(12.0), 7.0], ("12", "7")),
        ([np.float16(3.0), np.float64(4.0)], ("3", "4")),
        ([float(2**53 - 1), -3.0], ("9007199254740991", "-3")),
        ([-0.0, 1.0], ("0", "1")),
        (np.array([5.0, 6.0]), ("5", "6")),
    ):
        sources = GalacticBinaries.from_arrays(values, source_ids=ids, **kwargs)
        assert sources.source_ids == expected
    posteriors = integer_keyed(lisa_posteriors, {"a": 12, "b": 12, "c": 12})
    first, second, third = posteriors.values()
    keyed = {1.0: first, np.float64(2.0): second, np.float32(10.0): third}
    from_posteriors = GalacticBinaries.from_posteriors(keyed, **kwargs)
    assert from_posteriors.source_ids == ("1", "2", "10")
    sized = GalacticBinaries.from_posteriors(
        keyed, n_eff={1.0: 5.0, 2.0: 6.0, 10.0: 7.0}, **kwargs
    )
    assert sized.values.n_eff is not None
    np.testing.assert_allclose(np.asarray(sized.values.n_eff[:, 0]), [5.0, 6.0, 7.0])


@pytest.mark.parametrize(
    ("bad", "rule"),
    [
        (5.5, "finite and whole"),
        (np.float32(2.5), "finite and whole"),
        (math.nan, "finite and whole"),
        (math.inf, "finite and whole"),
        (-math.inf, "finite and whole"),
        (2.0**53, "magnitude below 2"),
        (-(2.0**53), "magnitude below 2"),
        (1e300, "magnitude below 2"),
    ],
)
def test_other_floats_are_not_source_ids(
    lisa_posteriors: Builder, bad: float, rule: str
) -> None:
    """A fraction, NaN, an infinity or a float beyond 2**53 raises, naming the value."""
    values = gb_arrays(lisa_posteriors)
    with pytest.raises(ValueError, match=rf"{rule}.*{re.escape(repr(bad))}"):
        GalacticBinaries.from_arrays(
            values, source_ids=[1.0, bad], name="fit", epoch=0.0
        )


def test_booleans_stay_refused_next_to_floats(lisa_posteriors: Builder) -> None:
    """True is not the id 1.0: the boolean check comes first."""
    values = gb_arrays(lisa_posteriors)
    with pytest.raises(TypeError, match="must not be booleans"):
        GalacticBinaries.from_arrays(
            values, source_ids=[1.0, True], name="fit", epoch=0.0
        )


def test_detections_with_float_ids_join_to_the_string_ids(
    lisa_posteriors: Builder,
) -> None:
    """A float64 source_id column, polars or structured, compares as strings."""
    posteriors = integer_keyed(lisa_posteriors, {"a": 12, "b": 12, "c": 12})
    table = pl.DataFrame({"source_id": [10.0, 1.0, 99.0], "snr": [3.0, 1.0, 9.0]})
    assert table["source_id"].dtype == pl.Float64
    sources = GalacticBinaries.from_posteriors(
        posteriors, name="fit", epoch=0.0, detections=table
    )
    assert sources.detections is not None
    assert sources.detections["source_id"].to_list() == ["1", "2", "10"]
    assert sources.detections["snr"].to_list() == [1.0, None, 3.0]
    structured = np.zeros(2, dtype=[("source_id", "f8"), ("snr", "f8")])
    structured["source_id"] = [2.0, 10.0]
    structured["snr"] = [4.0, 5.0]
    again = GalacticBinaries.from_posteriors(
        posteriors, name="fit", epoch=0.0, detections=structured
    )
    assert again.detections is not None
    assert again.detections["snr"].to_list() == [None, 4.0, 5.0]
    fractional = pl.DataFrame({"source_id": [1.5], "snr": [1.0]})
    with pytest.raises(ValueError, match="finite and whole"):
        GalacticBinaries.from_posteriors(
            posteriors, name="fit", epoch=0.0, detections=fractional
        )


# * Selection


def test_selection_by_index_slice_ids_and_arrays(lisa_posteriors: Builder) -> None:
    """Selections keep the order, the conventions and the detections."""
    table = pl.DataFrame(
        {"source_id": ["alpha", "beta", "gamma"], "snr": [1.0, 2.0, 3.0]}
    )
    sources = holder(lisa_posteriors, "galactic_binary", detections=table)
    assert sources[0].source_ids == ("alpha",)
    assert sources[np.int64(2)].source_ids == ("gamma",)
    assert sources[-1].source_ids == ("gamma",)
    assert sources[1:].source_ids == ("beta", "gamma")
    assert sources["beta"].source_ids == ("beta",)
    picked = sources[["gamma", "alpha"]]
    assert picked.source_ids == ("gamma", "alpha")
    assert picked.detections is not None
    assert picked.detections["snr"].to_list() == [3.0, 1.0]
    assert sources[np.array([2, 0])].source_ids == ("gamma", "alpha")
    assert sources[[1, 2]].values.shape.batch == 2
    assert sources[0].conventions == sources.conventions
    np.testing.assert_array_equal(
        np.asarray(sources[1].values.samples[0]), np.asarray(sources.values.samples[1])
    )
    no_table = holder(lisa_posteriors, "galactic_binary")
    assert no_table[1:].detections is None


def test_selection_errors(lisa_posteriors: Builder) -> None:
    """Unknown ids, bad indices, empty and repeated selections raise."""
    sources = holder(lisa_posteriors, "galactic_binary")
    with pytest.raises(KeyError, match="no source with the id 'nope'"):
        sources["nope"]
    with pytest.raises(IndexError, match="out of range for 3 sources"):
        sources[3]
    with pytest.raises(IndexError, match="out of range"):
        sources[[0, -4]]
    with pytest.raises(ValueError, match="at least one source"):
        sources[[]]
    with pytest.raises(ValueError, match="at least one source"):
        sources[2:1]
    with pytest.raises(ValueError, match="must not repeat"):
        sources[[0, 0]]
    with pytest.raises(TypeError, match="integers or ids"):
        sources[[0.5, 1.5]]


# * Spaces, bounds and settings


@pytest.mark.parametrize("kind", TYPES)
def test_space_follows_the_conventions(lisa_posteriors: Builder, kind: str) -> None:
    """Topologies, bounds, units and spheres follow the tables of the type."""
    sources = holder(lisa_posteriors, kind)
    space = sources.space
    cls = type(sources)
    by_name = {spec.name: spec for spec in cls.parameter_specs}
    for parameter in space.parameters:
        spec = by_name[parameter.name]
        assert parameter.topology is spec.topology
        assert parameter.bounds == spec.bounds
        assert parameter.unit == spec.unit
    assert space.spheres[0].polar_form is PolarForm.LATITUDE
    assert space.spheres[0].azimuth == "right_ascension"


def test_every_parameter_with_a_spread_is_sampled(lisa_posteriors: Builder) -> None:
    """Spheres stay whole, and the sample set follows the sampled order."""
    sources = holder(lisa_posteriors, "stellar_origin_binary_black_hole")
    assert len(sources.space) == 16
    assert [sphere.azimuth for sphere in sources.space.spheres] == [
        "right_ascension",
        "azimuthal_angle_spin1_icrs",
        "azimuthal_angle_spin2_icrs",
    ]
    assert sources.sample_set.samples.shape[-1] == 16


def test_constant_parameters_become_settings(lisa_posteriors: Builder) -> None:
    """Parameters equal within every source leave the space, one value per source."""
    data = lisa_posteriors("galactic_binary", COUNTS, constant=("initial_phase",))
    for position, table in enumerate(data.values()):
        table["initial_phase"] = 0.5 * (position + 1)
    sources = GalacticBinaries.from_posteriors(data, name="fit", epoch=0.0)
    assert "initial_phase" not in sources.space.names
    assert sources.settings["initial_phase"].to_list() == [0.5, 1.0, 1.5]
    assert sources.settings["source_id"].to_list() == ["alpha", "beta", "gamma"]
    assert sources.sample_set.shape.param == 7


def test_a_sphere_keeps_both_axes_when_one_varies(lisa_posteriors: Builder) -> None:
    """A constant declination stays in the space next to a varying right ascension."""
    data = lisa_posteriors("galactic_binary", COUNTS, constant=("declination",))
    sources = GalacticBinaries.from_posteriors(data, name="fit", epoch=0.0)
    assert "declination" in sources.space.names
    assert len(sources.space.spheres) == 1
    both_constant = lisa_posteriors(
        "galactic_binary", COUNTS, constant=("declination", "right_ascension")
    )
    without_sky = GalacticBinaries.from_posteriors(both_constant, name="fit", epoch=0.0)
    assert "declination" not in without_sky.space.names
    assert without_sky.space.spheres == ()
    assert {"declination", "right_ascension"} <= set(without_sky.settings.columns)


def test_a_catalog_and_single_sample_sources_keep_every_parameter(
    lisa_posteriors: Builder,
) -> None:
    """With one positive sample per source nothing is constant."""
    sources = holder(lisa_posteriors, "galactic_binary", counts={"a": 1, "b": 1})
    assert len(sources.space) == 8
    assert sources.settings.columns == ["source_id"]
    mixed = holder(lisa_posteriors, "galactic_binary", counts={"a": 1, "b": 12})
    assert len(mixed.space) == 8


def test_a_holder_without_variation_has_no_space(lisa_posteriors: Builder) -> None:
    """When every parameter is a setting, the space and the sample set raise."""
    values = {
        name: np.repeat(column[:, :1], 5, axis=1)
        for name, column in gb_arrays(lisa_posteriors).items()
    }
    sources = GalacticBinaries.from_arrays(
        values, source_ids=["a", "b"], name="fit", epoch=0.0
    )
    assert len(sources.settings.columns) == 9
    with pytest.raises(ValueError, match="no parameter varies"):
        sources.space  # noqa: B018
    with pytest.raises(ValueError, match="no parameter varies"):
        sources.sample_set  # noqa: B018


def test_zero_weight_samples_do_not_make_a_parameter_vary(
    lisa_posteriors: Builder,
) -> None:
    """Padding with other values does not turn a constant into a sampled parameter."""
    values = gb_arrays(lisa_posteriors)
    values["initial_phase"] = np.full((2, 20), 1.0)
    values["initial_phase"][:, 15:] = 2.0
    log_weights = np.zeros((2, 20))
    log_weights[:, 15:] = -np.inf
    sources = GalacticBinaries.from_arrays(
        values, source_ids=["a", "b"], name="fit", epoch=0.0, log_weights=log_weights
    )
    assert "initial_phase" not in sources.space.names
    assert sources.settings["initial_phase"].to_list() == [1.0, 1.0]


def test_prior_bounds_replace_the_domain_inside_it(lisa_posteriors: Builder) -> None:
    """Bounds narrow an interval, may use aliases and bound the real line."""
    sources = holder(
        lisa_posteriors,
        "galactic_binary",
        bounds={
            "f": (2e-3, 4e-3),
            "declination": (-1.0, 1.0),
            "frequency_dot": (0.0, 1e-16),
        },
    )
    assert sources.space["frequency"].bounds == (2e-3, 4e-3)
    assert sources.space["declination"].bounds == (-1.0, 1.0)
    assert sources.space["frequency_dot"].topology is Topology.INTERVAL
    assert sources.space["frequency_dot"].bounds == (0.0, 1e-16)
    assert sources.bounds["frequency"] == (2e-3, 4e-3)
    wide = holder(
        lisa_posteriors,
        "galactic_binary",
        bounds={"frequency_dot": (-math.inf, math.inf)},
    )
    assert wide.space["frequency_dot"].topology is Topology.REAL


def test_prior_bounds_outside_the_domain_or_on_circles_raise(
    lisa_posteriors: Builder,
) -> None:
    """Bounds must lie inside the physical domain, and circles take none."""
    with pytest.raises(ValueError, match="must lie inside its domain"):
        holder(lisa_posteriors, "galactic_binary", bounds={"declination": (-2.0, 1.0)})
    with pytest.raises(ValueError, match="must lie inside its domain"):
        holder(lisa_posteriors, "galactic_binary", bounds={"frequency": (-1.0, 1.0)})
    with pytest.raises(ValueError, match="must lie inside its domain"):
        holder(lisa_posteriors, "galactic_binary", bounds={"frequency": (2.0, 1.0)})
    with pytest.raises(ValueError, match="is a circle"):
        holder(lisa_posteriors, "galactic_binary", bounds={"initial_phase": (0.0, 3.0)})
    with pytest.raises(ConventionError):
        holder(
            lisa_posteriors, "galactic_binary", bounds={"ecliptic_latitude": (0.0, 1.0)}
        )


def test_samples_must_lie_inside_the_prior_bounds(lisa_posteriors: Builder) -> None:
    """A positive-weight sample outside a bound raises with the count and the source."""
    data = lisa_posteriors("galactic_binary", COUNTS)
    data["alpha"]["frequency"][0] = 1e-3
    data["beta"]["frequency"][:2] = 4e-3
    bounds = {"frequency": (2.5e-3, 3.5e-3)}
    with pytest.raises(
        ValueError,
        match=(
            r"frequency must be inside its prior bounds \[0.0025, 0.0035\], but 3 of"
            r" the 78 positive-weight values are not \(first in source 'alpha'\)"
        ),
    ):
        GalacticBinaries.from_posteriors(data, name="fit", epoch=0.0, bounds=bounds)


def test_values_on_a_prior_bound_pass_and_zero_weights_are_ignored(
    lisa_posteriors: Builder,
) -> None:
    """The bounds are a closed interval, and a sample without weight may lie outside."""
    values = gb_arrays(lisa_posteriors)
    frequency = values["frequency"].copy()
    frequency[0, 0], frequency[1, 0] = 2.5e-3, 3.5e-3
    frequency[0, 1] = 5e-3
    log_weights = np.zeros((2, 20))
    log_weights[0, 1] = -np.inf
    sources = GalacticBinaries.from_arrays(
        values | {"frequency": frequency},
        source_ids=["a", "b"],
        name="fit",
        epoch=0.0,
        log_weights=log_weights,
        bounds={"frequency": (2.5e-3, 3.5e-3)},
    )
    assert sources.sample_counts.tolist() == [19, 20]
    assert sources.space["frequency"].bounds == (2.5e-3, 3.5e-3)
    frequency[0, 1] = 3.5e-3 * (1 + 1e-6)
    with pytest.raises(ValueError, match=r"frequency must be inside its prior bounds"):
        GalacticBinaries.from_arrays(
            values | {"frequency": frequency},
            source_ids=["a", "b"],
            name="fit",
            epoch=0.0,
            bounds={"frequency": (2.5e-3, 3.5e-3)},
        )


# * Domain checks


@pytest.mark.parametrize(
    ("name", "value", "rule"),
    [
        ("frequency", -1.0, "strictly positive"),
        ("frequency", 0.0, "strictly positive"),
        ("amplitude", -1e-22, "strictly positive"),
        ("inclination", 3.5, r"in \[0.0, 3.14"),
        ("inclination", -0.1, r"in \[0.0, 3.14"),
        ("declination", 2.0, r"in \[-1.57"),
        ("frequency", math.nan, "finite"),
        ("frequency", math.inf, "finite"),
    ],
)
def test_domain_checks_name_the_parameter_the_rule_and_the_count(
    lisa_posteriors: Builder, name: str, value: float, rule: str
) -> None:
    """A violation raises with the parameter, the rule and how many values break it."""
    values = gb_arrays(lisa_posteriors)
    values[name] = values[name].copy()
    values[name][1, :3] = value
    with pytest.raises(
        ValueError, match=rf"{name} must be {rule}.*3 of the 40.*source 'b'"
    ):
        GalacticBinaries.from_arrays(
            values, source_ids=["a", "b"], name="fit", epoch=0.0
        )


def test_spins_and_eccentricities_have_their_domains(lisa_posteriors: Builder) -> None:
    """Spins lie in [−1, 1], and eccentricities in [0, 1)."""
    mbhb = arrays_of(lisa_posteriors, "massive_black_hole_binary")
    mbhb["spin1"] = mbhb["spin1"].copy()
    mbhb["spin1"][0, 0] = 1.5
    with pytest.raises(
        ValueError, match=r"spin1 must be in \[-1.0, 1.0\].*1 of the 40"
    ):
        MassiveBlackHoleBinaries.from_arrays(
            mbhb, source_ids=["a", "b"], name="fit", epoch=0.0, source_frame="J"
        )
    emri = arrays_of(lisa_posteriors, "extreme_mass_ratio_inspiral")
    emri["eccentricity_tref"] = emri["eccentricity_tref"].copy()
    emri["eccentricity_tref"][0, :2] = 1.0
    with pytest.raises(
        ValueError, match=r"eccentricity_tref must be in \[0.0, 1.0\).*2 of"
    ):
        ExtremeMassRatioInspirals.from_arrays(
            emri, source_ids=["a", "b"], name="fit", epoch=0.0, source_frame="FEW"
        )
    sobbh = arrays_of(lisa_posteriors, "stellar_origin_binary_black_hole")
    sobbh["polar_angle_spin2"] = sobbh["polar_angle_spin2"].copy()
    sobbh["polar_angle_spin2"][1, 0] = 4.0
    with pytest.raises(
        ValueError, match=r"polar_angle_spin2_icrs must be in \[0.0, 3.14"
    ):
        StellarOriginBinaryBlackHoles.from_arrays(
            sobbh, source_ids=["a", "b"], name="fit", epoch=0.0, source_frame="J"
        )


@pytest.mark.parametrize(
    "kind",
    [
        "massive_black_hole_binary",
        "extreme_mass_ratio_inspiral",
        "stellar_origin_binary_black_hole",
    ],
)
def test_the_primary_mass_must_not_be_below_the_secondary(
    lisa_posteriors: Builder, kind: str
) -> None:
    """m₁ < m₂ raises with the count, and equal masses pass."""
    cls, source_frame = TYPES[kind]
    kwargs: dict[str, Any] = {
        "source_ids": ["a", "b"],
        "name": "fit",
        "epoch": 0.0,
        "source_frame": source_frame,
    }
    values = arrays_of(lisa_posteriors, kind)
    secondary = values["secondary_mass_det_frame"]
    primary = values["primary_mass_det_frame"].copy()
    primary[1, :4] = secondary[1, :4] * 0.5
    with pytest.raises(
        ValueError, match=r"primary_mass_det_frame must be at least.*4 of"
    ):
        cls.from_arrays(values | {"primary_mass_det_frame": primary}, **kwargs)
    equal = values | {"primary_mass_det_frame": secondary.copy()}
    cls.from_arrays(equal, **kwargs)


def test_checks_ignore_zero_weight_samples(lisa_posteriors: Builder) -> None:
    """A violation in a sample with no weight does not raise."""
    values = gb_arrays(lisa_posteriors)
    values["frequency"] = values["frequency"].copy()
    values["frequency"][0, 19] = -5.0
    log_weights = np.zeros((2, 20))
    log_weights[0, 19] = -np.inf
    sources = GalacticBinaries.from_arrays(
        values, source_ids=["a", "b"], name="fit", epoch=0.0, log_weights=log_weights
    )
    assert sources.sample_counts.tolist() == [19, 20]


def test_circle_parameters_wrap_into_their_period(lisa_posteriors: Builder) -> None:
    """Phases and angles outside [0, 2π) wrap before the checks."""
    values = gb_arrays(lisa_posteriors)
    values["initial_phase"] = values["initial_phase"] + 2 * math.pi * 3
    values["right_ascension"] = values["right_ascension"] - 2 * math.pi
    sources = GalacticBinaries.from_arrays(
        values, source_ids=["a", "b"], name="fit", epoch=0.0
    )
    column = np.asarray(
        sources.values.samples[..., sources.full_space.index("initial_phase")]
    )
    assert ((column >= 0) & (column < 2 * math.pi)).all()
    np.testing.assert_allclose(
        column[0, 0, :5],
        np.asarray(gb_arrays(lisa_posteriors)["initial_phase"])[0, :5],
        rtol=1e-12,
    )
    right_ascension = np.asarray(sources.values.samples[..., 3])
    assert ((right_ascension >= 0) & (right_ascension < 2 * math.pi)).all()


# * Internal consistency


def test_missing_parameters_and_transform_clashes_raise_clear_errors(
    lisa_posteriors: Builder,
) -> None:
    """A derived quantity needs its parameter, and a bound blocks a transform."""
    sources = holder(lisa_posteriors, "galactic_binary")
    with pytest.raises(ValueError, match="the parameter vector has no 'nothing'"):
        sources._column("nothing")
    bounded = holder(
        lisa_posteriors, "galactic_binary", bounds={"declination": (-1.0, 1.0)}
    )
    with pytest.raises(ValueError, match="carry prior bounds"):
        bounded.to_ecliptic()
    assert bounded.to_icrs() is bounded


def test_a_vector_without_masses_has_no_mass_form(lisa_posteriors: Builder) -> None:
    """The mass properties need the masses, in some form."""
    sources = MassiveBlackHoleBinaries.from_posteriors(
        lisa_posteriors("massive_black_hole_binary", {"a": 8}),
        name="fit",
        epoch=0.0,
        source_frame="J",
    )
    renamed = ParamSpace(
        parameters=tuple(
            parameter.model_copy(update={"name": f"x{position}"})
            for position, parameter in enumerate(sources.full_space.parameters)
        )
    )
    without = dataclasses.replace(sources, full_space=renamed)
    with pytest.raises(ValueError, match="no mass parameters"):
        without.total_mass  # noqa: B018


# * Prior bounds set after the transforms

MASS_BOUNDS = {"total_mass_det_frame": (1e5, 1e8), "inverted_mass_ratio": (1.0, 10.0)}


def binary_in_total_mass_and_inverted_ratio(posteriors: Builder) -> LisaSources:
    """Return massive black-hole binaries in the form (M, Q), without bounds."""
    sources = holder(posteriors, "massive_black_hole_binary")
    assert isinstance(sources, MassiveBlackHoleBinaries)
    return sources.to_total_mass_and_mass_ratio(inverted=True)


def test_bounds_on_the_total_mass_and_the_inverted_ratio_after_the_transform(
    lisa_posteriors: Builder,
) -> None:
    """The space carries the bounds as intervals, and ``bounds`` holds both names."""
    mapped = binary_in_total_mass_and_inverted_ratio(lisa_posteriors)
    bounded = mapped.with_bounds(MASS_BOUNDS)
    assert type(bounded) is MassiveBlackHoleBinaries
    for name, bound in MASS_BOUNDS.items():
        assert bounded.space[name].bounds == bound
        assert bounded.space[name].topology is Topology.INTERVAL
        assert bounded.full_space[name].bounds == bound
    assert bounded.space["total_mass_det_frame"].unit == "Msun"
    assert bounded.space["inverted_mass_ratio"].label == "Q"
    assert bounded.bounds == MASS_BOUNDS
    assert bounded.space.names == mapped.space.names
    assert bounded.space.spheres == mapped.space.spheres
    assert bounded.name == mapped.name
    assert bounded.source_ids == mapped.source_ids
    assert bounded.conventions == mapped.conventions
    np.testing.assert_array_equal(
        np.asarray(bounded.values.samples), np.asarray(mapped.values.samples)
    )
    assert mapped.bounds == {}
    assert mapped.space["inverted_mass_ratio"].bounds == (1.0, math.inf)


def test_unbounded_and_real_line_bounds_set_the_topology(
    lisa_posteriors: Builder,
) -> None:
    """Two infinite bounds give the real line, and one finite bound an interval."""
    sources = holder(lisa_posteriors, "galactic_binary")
    wide = sources.with_bounds({"frequency_dot": (-math.inf, math.inf)})
    assert wide.space["frequency_dot"].topology is Topology.REAL
    half = sources.with_bounds({"frequency_dot": (0.0, math.inf)})
    assert half.space["frequency_dot"].topology is Topology.INTERVAL
    assert half.space["frequency_dot"].bounds == (0.0, math.inf)


def test_bounds_follow_an_epoch_shift_but_do_not_precede_one(
    lisa_posteriors: Builder,
) -> None:
    """with_bounds after at_epoch works; at_epoch after with_bounds names the remedy."""
    sources = holder(lisa_posteriors, "massive_black_hole_binary")
    shifted = sources.at_epoch(4e5)
    bounded = shifted.with_bounds({"coalescence_time": (9e6, 1e7)})
    assert bounded.space["coalescence_time"].bounds == (9e6, 1e7)
    assert bounded.conventions.epoch == 4e5
    with pytest.raises(
        ValueError,
        match=(
            r"'coalescence_time' carry prior bounds.*build the holder without those"
            " bounds, transform it, then add them with with_bounds"
        ),
    ):
        sources.with_bounds({"coalescence_time": (9e6, 1.1e7)}).at_epoch(4e5)


def test_transforms_keep_refusing_bounded_parameters(lisa_posteriors: Builder) -> None:
    """A mass form with bounds does not convert, except to itself."""
    bounded = binary_in_total_mass_and_inverted_ratio(lisa_posteriors).with_bounds(
        MASS_BOUNDS
    )
    assert isinstance(bounded, MassiveBlackHoleBinaries)
    assert bounded.to_total_mass_and_mass_ratio(inverted=True) is bounded
    for conversion in (
        bounded.to_component_masses,
        bounded.to_total_mass_and_mass_ratio,
        bounded.to_chirp_mass_and_mass_ratio,
    ):
        with pytest.raises(ValueError, match="then add them with with_bounds"):
            conversion()
    bounded_sky = holder(lisa_posteriors, "galactic_binary").with_bounds(
        {"declination": (-1.0, 1.0)}
    )
    with pytest.raises(ValueError, match="then add them with with_bounds"):
        bounded_sky.to_ecliptic()


def test_with_bounds_resolves_aliases_and_the_sky_frame(
    lisa_posteriors: Builder,
) -> None:
    """Aliases resolve as in the constructors; the other sky frame raises."""
    sources = holder(lisa_posteriors, "massive_black_hole_binary")
    aliased = sources.with_bounds({"tc": (9e6, 1.1e7), "dec": (-1.0, 1.0)})
    assert aliased.bounds == {
        "coalescence_time": (9e6, 1.1e7),
        "declination": (-1.0, 1.0),
    }
    with pytest.raises(ConventionError, match="ecliptic frame, but the holder is in"):
        sources.with_bounds({"ecliptic_latitude": (-1.0, 1.0)})
    with pytest.raises(ConventionError, match="ecliptic frame, but the holder is in"):
        sources.with_bounds({"beta": (-1.0, 1.0)})
    ecliptic = sources.to_ecliptic()
    named = ecliptic.with_bounds({"beta": (-1.0, 1.0)})
    assert named.space["ecliptic_latitude"].bounds == (-1.0, 1.0)
    assert named.bounds == {"ecliptic_latitude": (-1.0, 1.0)}
    with pytest.raises(ConventionError, match="icrs frame, but the holder is in"):
        ecliptic.with_bounds({"dec": (-1.0, 1.0)})
    with pytest.raises(ConventionError, match="icrs frame, but the holder is in"):
        ecliptic.with_bounds({"declination": (-1.0, 1.0)})


def test_with_bounds_names_unknown_parameters_and_lists_the_vector(
    lisa_posteriors: Builder,
) -> None:
    """Unknown names, and names of the table that the transform removed, raise."""
    sources = holder(lisa_posteriors, "massive_black_hole_binary")
    with pytest.raises(
        ValueError, match=r"unknown bounds fields \['nothing'\].*holder carries \['pri"
    ):
        sources.with_bounds({"nothing": (0.0, 1.0)})
    with pytest.raises(ValueError, match=r"unknown bounds fields \['total_mass_det_fr"):
        sources.with_bounds({"total_mass_det_frame": (1e5, 1e8)})
    mapped = binary_in_total_mass_and_inverted_ratio(lisa_posteriors)
    for removed in ("primary_mass_det_frame", "m1", "mass_ratio"):
        with pytest.raises(ValueError, match="holder carries"):
            mapped.with_bounds({removed: (1.0, 10.0)})
    with pytest.raises(ValueError, match="are the same parameter"):
        sources.with_bounds({"tc": (9e6, 1.1e7), "coalescence_time": (9e6, 1.1e7)})


def test_with_bounds_has_no_domain_for_the_chirp_mass_of_galactic_binaries(
    lisa_posteriors: Builder,
) -> None:
    """Only the type's table and the mass forms of binary black holes have domains."""
    binaries = GalacticBinaries.from_posteriors(
        lisa_posteriors("galactic_binary", COUNTS), name="fit", epoch=0.0
    )
    chirp = binaries.to_chirp_mass_and_distance()
    with pytest.raises(ValueError, match="'chirp_mass' has no physical domain"):
        chirp.with_bounds({"chirp_mass": (0.1, 1.0)})
    phases = binaries.to_phase_combinations()
    with pytest.raises(ValueError, match="'theta_plus_icrs' has no physical domain"):
        phases.with_bounds({"theta_plus_icrs": (0.0, 3.0)})


@pytest.mark.parametrize(
    ("bounds", "message"),
    [
        ({"mass_ratio": (0.1, 1.5)}, "must lie inside its domain"),
        ({"mass_ratio": (-0.1, 0.9)}, "must lie inside its domain"),
        ({"spin1": (-2.0, 1.0)}, "must lie inside its domain"),
        ({"spin1": (0.5, 0.5)}, "must lie inside its domain"),
        ({"spin1": (0.9, 0.1)}, "must lie inside its domain"),
        ({"luminosity_distance": (-1.0, 50.0)}, "must lie inside its domain"),
        ({"phase_at_reference_time": (0.0, 3.0)}, "is a circle"),
        ({"polarisation_icrs": (0.0, 3.0)}, "is a circle"),
    ],
)
def test_with_bounds_refuses_bounds_outside_the_domain_and_circles(
    lisa_posteriors: Builder, bounds: dict[str, tuple[float, float]], message: str
) -> None:
    """The physical domain of the table or of the mass form limits the new bounds."""
    sources = holder(lisa_posteriors, "massive_black_hole_binary")
    assert isinstance(sources, MassiveBlackHoleBinaries)
    form = sources.to_total_mass_and_mass_ratio()
    with pytest.raises(ValueError, match=message):
        form.with_bounds(bounds)


def test_the_inverted_mass_ratio_domain_starts_at_one(lisa_posteriors: Builder) -> None:
    """Q lies in [1, ∞) and the total mass in (0, ∞), whatever bound came before."""
    mapped = binary_in_total_mass_and_inverted_ratio(lisa_posteriors)
    with pytest.raises(ValueError, match="must lie inside its domain"):
        mapped.with_bounds({"inverted_mass_ratio": (0.5, 10.0)})
    with pytest.raises(ValueError, match="must lie inside its domain"):
        mapped.with_bounds({"total_mass_det_frame": (-1.0, 1e8)})
    edge = mapped.with_bounds({"inverted_mass_ratio": (1.0, math.inf)})
    assert edge.space["inverted_mass_ratio"].bounds == (1.0, math.inf)


def test_with_bounds_checks_the_positive_weight_samples_only(
    lisa_posteriors: Builder,
) -> None:
    """A sample outside raises with its source; zero-weight padding is free."""
    values = gb_arrays(lisa_posteriors)
    frequency = values["frequency"].copy()
    frequency[0, 1] = 5e-3
    log_weights = np.zeros((2, 20))
    log_weights[0, 1] = -np.inf
    kwargs: dict[str, Any] = {
        "source_ids": ["a", "b"],
        "name": "fit",
        "epoch": 0.0,
        "log_weights": log_weights,
    }
    sources = GalacticBinaries.from_arrays(values | {"frequency": frequency}, **kwargs)
    bounded = sources.with_bounds({"frequency": (2.5e-3, 3.5e-3)})
    assert bounded.sample_counts.tolist() == [19, 20]
    assert bounded.space["frequency"].bounds == (2.5e-3, 3.5e-3)
    frequency[1, 3] = 5e-3
    frequency[1, 4] = 1e-3
    outside = GalacticBinaries.from_arrays(values | {"frequency": frequency}, **kwargs)
    with pytest.raises(
        ValueError,
        match=(
            r"frequency must be inside its prior bounds \[0.0025, 0.0035\], but 2 of"
            r" the 39 positive-weight values are not \(first in source 'b'\)"
        ),
    ):
        outside.with_bounds({"frequency": (2.5e-3, 3.5e-3)})
    low, high = float(values["frequency"].min()), float(values["frequency"].max())
    edge = GalacticBinaries.from_arrays(values, **kwargs).with_bounds(
        {"frequency": (low, high)}
    )
    assert edge.space["frequency"].bounds == (low, high)


def test_a_second_with_bounds_replaces_the_first_of_the_same_name(
    lisa_posteriors: Builder,
) -> None:
    """The physical domain limits a new bound, not the earlier one; other names stay."""
    mapped = binary_in_total_mass_and_inverted_ratio(lisa_posteriors)
    first = mapped.with_bounds(MASS_BOUNDS)
    second = first.with_bounds({"inverted_mass_ratio": (1.5, 100.0)})
    assert second.space["inverted_mass_ratio"].bounds == (1.5, 100.0)
    assert second.space["total_mass_det_frame"].bounds == (1e5, 1e8)
    assert second.bounds == {
        "total_mass_det_frame": (1e5, 1e8),
        "inverted_mass_ratio": (1.5, 100.0),
    }
    assert first.bounds == MASS_BOUNDS
    narrower = second.with_bounds({"inverted_mass_ratio": (1.5, 5.0)})
    assert narrower.bounds["inverted_mass_ratio"] == (1.5, 5.0)
