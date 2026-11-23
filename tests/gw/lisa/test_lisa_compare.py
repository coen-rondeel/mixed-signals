"""Tests for the comparison of two holders of LISA sources."""

import dataclasses
from collections.abc import Callable
from typing import Any

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from mixed_signals.core.samples import SampleSet
from mixed_signals.core.space import ParamSpace
from mixed_signals.engines import compare
from mixed_signals.errors import ConventionError
from mixed_signals.gw.lisa import (
    GalacticBinaries,
    MassiveBlackHoleBinaries,
    SourceComparison,
    StellarOriginBinaryBlackHoles,
    compare_sources,
    match_indices,
)
from mixed_signals.gw.lisa.compare import _compared_space

type Builder = Callable[..., dict[str, np.ndarray]]

IDS = ("s0", "s1", "s2", "s3")
FIRST_COUNTS = {"s0": 40, "s1": 36, "s2": 40, "s3": 32}
SECOND_COUNTS = {"s0": 34, "s1": 40, "s2": 30, "s3": 38}
KEYWORDS: dict[str, Any] = {
    "key": 5,
    "draws": 20,
    "cap": 60,
    "metrics": ["mean_shift", "energy"],
}


def fits(
    posteriors: Builder, *, constant: tuple[str, ...] = (), **keywords: Any
) -> tuple[GalacticBinaries, GalacticBinaries]:
    """Build two global fits of the same four Galactic binaries."""
    first = GalacticBinaries.from_posteriors(
        posteriors("galactic_binary", FIRST_COUNTS, seed=1, constant=constant),
        name="erebor",
        epoch=0.0,
        **keywords,
    )
    second = GalacticBinaries.from_posteriors(
        posteriors(
            "galactic_binary", SECOND_COUNTS, seed=2, shift=0.3, constant=constant
        ),
        name="geemoo",
        epoch=0.0,
        **keywords,
    )
    return first, second


# * Pairing and results


def test_holders_pair_in_order_by_default(lisa_posteriors: Builder) -> None:
    """One problem per pair, with the pairs table and the rows of ``compare``."""
    first, second = fits(lisa_posteriors)
    comparison = compare_sources(first, second, **KEYWORDS)
    assert isinstance(comparison, SourceComparison)
    pairs = comparison.pairs
    assert pairs.columns == [
        "problem",
        "first_index",
        "first_source_id",
        "second_index",
        "second_source_id",
    ]
    assert pairs["problem"].to_list() == [0, 1, 2, 3]
    assert pairs["first_source_id"].to_list() == list(IDS)
    assert pairs["second_source_id"].to_list() == list(IDS)
    assert (
        pairs["first_index"].to_list()
        == pairs["second_index"].to_list()
        == [0, 1, 2, 3]
    )
    assert pairs.schema == pl.Schema(
        {
            "problem": pl.Int64(),
            "first_index": pl.Int64(),
            "first_source_id": pl.String(),
            "second_index": pl.Int64(),
            "second_source_id": pl.String(),
        }
    )
    table = comparison.result.table
    assert sorted(table["problem"].unique().to_list()) == [0, 1, 2, 3]
    assert table["pair"].list.get(0).unique().to_list() == ["erebor"]
    assert table["pair"].list.get(1).unique().to_list() == ["geemoo"]
    sets = [holder.sample_set for holder in (first, second)]
    direct = compare(
        sets,
        ["erebor", "geemoo"],
        first.space,
        key=5,
        draws=20,
        cap=60,
        metrics=["mean_shift", "energy"],
    )
    assert_frame_equal(table, direct.table, check_exact=True)


def test_matches_as_pairs_or_one_entry_per_source(lisa_posteriors: Builder) -> None:
    """An (M, 2) array pairs sources, repeats allowed; (B,) with −1 skips some."""
    first, second = fits(lisa_posteriors)
    pairs = np.array([[3, 0], [0, 2], [0, 3]])
    comparison = compare_sources(first, second, matches=pairs, **KEYWORDS)
    assert comparison.pairs["first_source_id"].to_list() == ["s3", "s0", "s0"]
    assert comparison.pairs["second_source_id"].to_list() == ["s0", "s2", "s3"]
    assert comparison.pairs["problem"].to_list() == [0, 1, 2]
    per_source = compare_sources(first, second, matches=[2, -1, 0, 3], **KEYWORDS)
    assert per_source.pairs["first_index"].to_list() == [0, 2, 3]
    assert per_source.pairs["second_index"].to_list() == [2, 0, 3]
    assert per_source.result.table["problem"].n_unique() == 3
    listed = compare_sources(first, second, matches=[[1, 1], [2, 2]], **KEYWORDS)
    assert listed.pairs["problem"].to_list() == [0, 1]


def test_a_shuffled_match_equals_the_manually_reordered_sets(
    lisa_posteriors: Builder,
) -> None:
    """Pairing (i, π(i)) gives the rows of the second holder reordered by π."""
    first, second = fits(lisa_posteriors)
    order = [2, 0, 3, 1]
    shuffled = compare_sources(
        first,
        second,
        matches=np.array([[index, order[index]] for index in range(4)]),
        **KEYWORDS,
    )
    reordered = compare_sources(first, second[order], **KEYWORDS)
    assert_frame_equal(shuffled.result.table, reordered.result.table, check_exact=True)
    assert shuffled.pairs["second_source_id"].to_list() == [
        IDS[index] for index in order
    ]


def test_mode_a_takes_the_reference_by_name(lisa_posteriors: Builder) -> None:
    """``reference=`` names one of the holders, and other keywords pass through."""
    first, second = fits(lisa_posteriors)
    comparison = compare_sources(first, second, reference="erebor", **KEYWORDS)
    pairs = comparison.result.table["pair"].to_list()
    assert all(pair == ["erebor", "geemoo"] for pair in pairs)
    assert comparison.result.table["metric"].n_unique() >= 2


def test_match_indices_from_pairs_of_ids(lisa_posteriors: Builder) -> None:
    """Pairs of ids, strings or bytes, become an (M, 2) integer array."""
    first, second = fits(lisa_posteriors)
    pairs = match_indices(first, second, [("s2", "s0"), (b"s1", b"s3")])
    assert pairs.tolist() == [[2, 0], [1, 3]]
    assert pairs.dtype == np.int64
    comparison = compare_sources(first, second, matches=pairs, **KEYWORDS)
    assert comparison.pairs["first_source_id"].to_list() == ["s2", "s1"]
    with pytest.raises(
        ValueError, match="unknown source ids: \\['x'\\] in the first holder"
    ):
        match_indices(first, second, [("x", "s0")])
    with pytest.raises(ValueError, match="and \\['y'\\] in the second"):
        match_indices(first, second, [("s0", "y")])
    with pytest.raises(ValueError, match="at least one pair"):
        match_indices(first, second, [])


def test_match_indices_reads_integer_ids(lisa_posteriors: Builder) -> None:
    """Integer ids, as in the match tables of catpylisa, match the holders' strings."""
    counts = {"1": 30, "2": 30, "3": 30}
    first = GalacticBinaries.from_posteriors(
        lisa_posteriors("galactic_binary", counts, seed=1), name="erebor", epoch=0.0
    )
    keyed = {
        int(key): table
        for key, table in lisa_posteriors("galactic_binary", counts, seed=2).items()
    }
    second = GalacticBinaries.from_posteriors(keyed, name="geemoo", epoch=0.0)
    pairs = match_indices(
        first, second, [(1, 3), (np.int64(2), "1"), (b"3", np.int32(2))]
    )
    assert pairs.tolist() == [[0, 2], [1, 0], [2, 1]]
    with pytest.raises(ValueError, match=r"unknown source ids: \['7'\] in the first"):
        match_indices(first, second, [(7, 1)])
    with pytest.raises(TypeError, match="must not be booleans"):
        match_indices(first, second, [(True, 1)])


def test_the_matches_are_checked(lisa_posteriors: Builder) -> None:
    """Lengths, types, shapes, ranges and empty matches raise."""
    first, second = fits(lisa_posteriors)
    short = second[:3]
    with pytest.raises(ValueError, match="pair in order, but they hold 4 and 3"):
        compare_sources(first, short, **KEYWORDS)
    with pytest.raises(ValueError, match="matches must hold integers"):
        compare_sources(first, second, matches=np.zeros((2, 2)), **KEYWORDS)
    with pytest.raises(ValueError, match="one match per source of the first holder, 4"):
        compare_sources(first, second, matches=[0, 1], **KEYWORDS)
    with pytest.raises(ValueError, match="marked by -1, not lower"):
        compare_sources(first, second, matches=[0, 1, -2, 3], **KEYWORDS)
    with pytest.raises(ValueError, match=r"shaped \(M, 2\) or \(B,\)"):
        compare_sources(first, second, matches=np.zeros((2, 3), dtype=int), **KEYWORDS)
    with pytest.raises(ValueError, match="pair no source"):
        compare_sources(first, second, matches=[-1, -1, -1, -1], **KEYWORDS)
    with pytest.raises(
        ValueError, match="match index 4 lies outside the 4 sources of holder 'geemoo'"
    ):
        compare_sources(first, second, matches=[[0, 4]], **KEYWORDS)
    with pytest.raises(ValueError, match="match index -1 lies outside"):
        compare_sources(first, second, matches=[[-1, 0]], **KEYWORDS)
    with pytest.raises(
        ValueError, match="match index 7 lies outside the 4 sources of holder 'erebor'"
    ):
        compare_sources(first, second, matches=[[7, 0]], **KEYWORDS)


# * Conventions, names and types


def test_holders_of_different_types_or_with_the_same_name_are_refused(
    lisa_posteriors: Builder,
) -> None:
    """The type must agree, the names must differ."""
    first, second = fits(lisa_posteriors)
    other = StellarOriginBinaryBlackHoles.from_posteriors(
        lisa_posteriors("stellar_origin_binary_black_hole", {"s0": 10}),
        name="other",
        epoch=0.0,
        source_frame="J",
    )
    with pytest.raises(
        ConventionError, match="different types, GalacticBinaries and Stellar"
    ):
        compare_sources(first, other, **KEYWORDS)
    twin = dataclasses.replace(second, name="erebor")
    with pytest.raises(ValueError, match="need different names, both are 'erebor'"):
        compare_sources(first, twin, **KEYWORDS)


def test_differing_conventions_name_the_transform_that_aligns_them(
    lisa_posteriors: Builder,
) -> None:
    """A sky frame, an epoch or a source frame mismatch raises a ConventionError."""
    first, second = fits(lisa_posteriors)
    with pytest.raises(
        ConventionError, match=r"sky_frame .*to_icrs\(\) or to_ecliptic\(\)"
    ):
        compare_sources(first, second.to_ecliptic(), **KEYWORDS)
    with pytest.raises(ConventionError, match=r"epoch .*at_epoch\(epoch\)"):
        compare_sources(first, second.at_epoch(100.0), **KEYWORDS)
    mbhb = [
        MassiveBlackHoleBinaries.from_posteriors(
            lisa_posteriors("massive_black_hole_binary", {"a": 10}, seed=seed),
            name=name,
            epoch=epoch,
            source_frame=frame,
        )
        for seed, name, epoch, frame in ((1, "one", 0.0, "J"), (2, "two", 5.0, "L"))
    ]
    with pytest.raises(
        ConventionError, match=r"epoch .*at_epoch\(epoch\).*source_frame .*no transform"
    ):
        compare_sources(mbhb[0], mbhb[1], **KEYWORDS)
    compare_sources(first, second.to_ecliptic().to_icrs(), **KEYWORDS)


def test_emri_epoch_mismatches_have_no_transform(lisa_posteriors: Builder) -> None:
    """EMRI and SOBBH holders at different epochs are refused with no remedy."""
    holders = [
        StellarOriginBinaryBlackHoles.from_posteriors(
            lisa_posteriors("stellar_origin_binary_black_hole", {"a": 10}, seed=seed),
            name=name,
            epoch=epoch,
            source_frame="J",
        )
        for seed, name, epoch in ((1, "one", 0.0), (2, "two", 3.0))
    ]
    with pytest.raises(
        ConventionError, match=r"epoch \(0.0 and 3.0\): no transform exists yet"
    ):
        compare_sources(holders[0], holders[1], **KEYWORDS)


# * Parameters and settings


def test_sampled_parameters_must_agree_or_be_listed(lisa_posteriors: Builder) -> None:
    """A parameter sampled in one holder and constant in the other must be left out."""
    first = GalacticBinaries.from_posteriors(
        lisa_posteriors("galactic_binary", FIRST_COUNTS, seed=1),
        name="erebor",
        epoch=0.0,
    )
    second = GalacticBinaries.from_posteriors(
        lisa_posteriors(
            "galactic_binary", SECOND_COUNTS, seed=2, constant=("initial_phase",)
        ),
        name="geemoo",
        epoch=0.0,
    )
    with pytest.raises(
        ConventionError,
        match="'initial_phase' is sampled in 'erebor' and a setting in 'geemoo'",
    ):
        compare_sources(first, second, **KEYWORDS)
    with pytest.raises(ConventionError, match="'initial_phase' is sampled in 'erebor'"):
        compare_sources(second, first, **KEYWORDS)
    names = [name for name in first.space.names if name != "initial_phase"]
    comparison = compare_sources(first, second, parameters=names, **KEYWORDS)
    assert "initial_phase" not in {
        parameter
        for row in comparison.result.table["parameters"].to_list()
        for parameter in row
    }
    combined = first.to_phase_combinations()
    other = dataclasses.replace(combined, name="other")
    with pytest.raises(ConventionError, match=r"theta_plus_icrs.*absent in 'erebor'"):
        compare_sources(first, other, **KEYWORDS)


def test_the_parameters_keyword_is_checked(lisa_posteriors: Builder) -> None:
    """Names must exist in both holders, be sampled in both, and be unique."""
    first, second = fits(lisa_posteriors, constant=("initial_phase",))
    with pytest.raises(ValueError, match="'nothing' is not in both holders"):
        compare_sources(first, second, parameters=["nothing"], **KEYWORDS)
    with pytest.raises(
        ValueError, match="'initial_phase' is a setting in both holders"
    ):
        compare_sources(first, second, parameters=["initial_phase"], **KEYWORDS)
    with pytest.raises(ValueError, match="non-empty list of unique names"):
        compare_sources(first, second, parameters=[], **KEYWORDS)
    with pytest.raises(ValueError, match="non-empty list of unique names"):
        compare_sources(
            first, second, parameters=["frequency", "frequency"], **KEYWORDS
        )
    mixed_first, mixed_second = fits(lisa_posteriors)
    constant = GalacticBinaries.from_posteriors(
        lisa_posteriors(
            "galactic_binary", SECOND_COUNTS, seed=2, constant=("inclination",)
        ),
        name="geemoo",
        epoch=0.0,
    )
    with pytest.raises(
        ConventionError, match="'inclination' is sampled in one holder and a setting"
    ):
        compare_sources(
            mixed_first, constant, parameters=["inclination", "frequency"], **KEYWORDS
        )
    subset = compare_sources(
        mixed_first, mixed_second, parameters=["frequency", "amplitude"], **KEYWORDS
    )
    assert {tuple(row) for row in subset.result.table["parameters"].to_list()} >= {
        ("frequency",),
        ("amplitude",),
    }


def test_matched_sources_must_share_their_settings(lisa_posteriors: Builder) -> None:
    """Settings agree within a relative 10⁻¹²; a difference names the pair."""
    first, second = fits(lisa_posteriors, constant=("initial_phase",))
    compare_sources(first, second, **KEYWORDS)
    fields = first.values.to_arrays("numpy", copy=True)
    index = first.full_space.index("initial_phase")
    changed = fields["samples"].copy()
    changed[1, :, :, index] += 1e-9
    other = dataclasses.replace(
        first, values=SampleSet(**{**fields, "samples": changed})
    )
    with pytest.raises(
        ConventionError,
        match="matched sources 's1' and 's1' differ in the setting 'initial_phase'",
    ):
        compare_sources(other, second, **KEYWORDS)
    tiny = fields["samples"].copy()
    tiny[1, :, :, index] *= 1 + 1e-14
    close = dataclasses.replace(first, values=SampleSet(**{**fields, "samples": tiny}))
    compare_sources(close, second, **KEYWORDS)
    assert (
        compare_sources(other, second, matches=[[0, 0]], **KEYWORDS).pairs.height == 1
    )


# * The compared space


def test_prior_bounds_may_differ_and_the_space_takes_their_union(
    lisa_posteriors: Builder,
) -> None:
    """Overlapping priors, neither inside the other, compare.

    The space of the comparison takes their union.
    """
    first, second = fits(lisa_posteriors)
    low_prior = GalacticBinaries.from_posteriors(
        lisa_posteriors("galactic_binary", FIRST_COUNTS, seed=1),
        name="erebor",
        epoch=0.0,
        bounds={"frequency": (2e-3, 3.5e-3), "frequency_dot": (0.0, 1e-16)},
    )
    high_prior = GalacticBinaries.from_posteriors(
        lisa_posteriors("galactic_binary", SECOND_COUNTS, seed=2),
        name="geemoo",
        epoch=0.0,
        bounds={"frequency": (2.5e-3, 4e-3)},
    )
    names = list(low_prior.space.names)
    space = _compared_space(low_prior, high_prior, names)
    assert space["frequency"].bounds == (2e-3, 4e-3)
    assert _compared_space(high_prior, low_prior, names)["frequency"].bounds == (
        2e-3,
        4e-3,
    )
    assert space["frequency_dot"].topology.value == "real"
    assert space["frequency_dot"].bounds == (float("-inf"), float("inf"))
    compare_sources(low_prior, high_prior, **KEYWORDS)
    both_finite = _compared_space(low_prior, low_prior, names)
    assert both_finite["frequency_dot"].topology.value == "interval"
    assert both_finite["frequency_dot"].bounds == (0.0, 1e-16)
    assert _compared_space(first, second, list(first.space.names)) == first.space


def test_parameters_defined_differently_are_refused(lisa_posteriors: Builder) -> None:
    """A different label, unit or sphere layout raises a ConventionError."""
    first, second = fits(lisa_posteriors)
    renamed = ParamSpace(
        parameters=tuple(
            parameter.model_copy(update={"label": "x"})
            if parameter.name == "frequency"
            else parameter
            for parameter in second.full_space.parameters
        ),
        spheres=second.full_space.spheres,
    )
    with pytest.raises(
        ConventionError, match="parameter 'frequency' is defined differently"
    ):
        compare_sources(
            first, dataclasses.replace(second, full_space=renamed), **KEYWORDS
        )
    flat = ParamSpace(parameters=second.full_space.parameters)
    with pytest.raises(
        ConventionError, match="spheres of the compared parameters differ"
    ):
        compare_sources(first, dataclasses.replace(second, full_space=flat), **KEYWORDS)
    circles = ParamSpace(
        parameters=tuple(
            parameter.model_copy(update={"bounds": (0.0, 3.0)})
            if parameter.name == "initial_phase"
            else parameter
            for parameter in second.full_space.parameters
        ),
        spheres=second.full_space.spheres,
    )
    with pytest.raises(
        ConventionError, match="parameter 'initial_phase' is defined differently"
    ):
        compare_sources(
            first, dataclasses.replace(second, full_space=circles), **KEYWORDS
        )
