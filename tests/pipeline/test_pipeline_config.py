"""Tests for the config models, ``load_config`` and the config hash."""

import copy
import json
import math
import re
import textwrap
from pathlib import Path
from typing import Any, get_args

import pytest
from pydantic import ValidationError

from mixed_signals import _config
from mixed_signals.core.space import ParamSpace
from mixed_signals.core.topology import PolarForm
from mixed_signals.diagnostics import flags
from mixed_signals.diagnostics.dependence import PAIR_CAP
from mixed_signals.engines.compare import DEFAULT_METRICS
from mixed_signals.engines.diagnose import DIAGNOSTIC_FAMILIES
from mixed_signals.errors import ConfigError
from mixed_signals.io.arrays import ArrayBackend
from mixed_signals.metrics.ot.sinkhorn import SINKHORN_EPSILON_FACTOR
from mixed_signals.pipeline import config as pipeline_config
from mixed_signals.pipeline.config import (
    IMPORT_PATH_PATTERN,
    ChainsEntry,
    CompareSection,
    CompareSignalsSection,
    DiagnoseSection,
    ErynEntry,
    ErynSpace,
    Hdf5Entry,
    PipelineConfig,
    SamplesEntry,
    TableEntry,
    _canonical_json,
    _config_hash,
    load_config,
)
from mixed_signals.prng import SEED_LIMIT
from mixed_signals.sigma.thresholds import SIGNIFICANCE_LEVEL, THRESHOLDS
from mixed_signals.sigma.uncertainty import HALF_SAMPLE_PAIRS

EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "pipeline" / "config.yaml"
EPSILON_EXAMPLE = EXAMPLE.with_name("epsilon.yaml")
INLINE_SPACE: dict[str, Any] = {
    "parameters": [{"name": "x0"}, {"name": "x1"}],
}
ERYN_SPACE: dict[str, Any] = {"eryn_priors": "models:priors"}
TABLE = {"name": "a", "format": "table", "path": "a.parquet"}
OTHER_TABLE = {"name": "b", "format": "table", "path": "b.parquet"}
SIGNALS = {"generator": "models:generator", "noise": "models:noise"}


def assign(target: Any, name: str, value: Any) -> None:
    """Set an attribute by name."""
    setattr(target, name, value)


def mapping(**changes: Any) -> dict[str, Any]:
    """Return a minimal valid config as a Python mapping, with changes on top."""
    base: dict[str, Any] = {
        "seed": 1,
        "output": "results",
        "space": copy.deepcopy(INLINE_SPACE),
        "posteriors": [dict(TABLE), dict(OTHER_TABLE)],
        "diagnose": {},
    }
    return {**base, **changes}


def invalid(config: dict[str, Any]) -> list[tuple[tuple[int | str, ...], str]]:
    """Validate a failing mapping; return the (location, message) of each error."""
    with pytest.raises(ValidationError) as info:
        PipelineConfig.model_validate(config)
    return [(error["loc"], error["msg"]) for error in info.value.errors()]


def locations(config: dict[str, Any]) -> set[tuple[int | str, ...]]:
    """Return the locations of the errors of an invalid mapping."""
    return {location for location, _ in invalid(config)}


def write(tmp_path: Path, text: str) -> Path:
    """Write a config file and return its path."""
    path = tmp_path / "config.yaml"
    path.write_text(text, encoding="utf-8")
    return path


# * A valid config and its defaults


def test_a_minimal_config_is_valid(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Four keys and one section; everything else takes the defaults."""
    monkeypatch.chdir(tmp_path)
    config = PipelineConfig.model_validate(mapping())
    assert config.seed == 1
    assert config.budget == _config.COST_BUDGET
    assert isinstance(config.space, ParamSpace)
    assert [entry.name for entry in config.posteriors] == ["a", "b"]
    assert config.compare is None
    assert config.compare_signals is None
    # * A mapping has no file, so its relative paths start at the working folder.
    assert config.base_folder == Path.cwd()
    assert config.config_file is None
    with pytest.raises(ValidationError, match="frozen"):
        assign(config, "seed", 2)


def test_every_default_comes_from_the_engines_constants() -> None:
    """The filled-in sections hold the constants of the engines, never a copy."""
    config = PipelineConfig.model_validate(mapping(compare={}, compare_signals=SIGNALS))
    assert config.diagnose == DiagnoseSection(
        diagnostics=DIAGNOSTIC_FAMILIES,
        draws=_config.NULL_DRAWS,
        cap=_config.SAMPLE_CAP,
        pair_cap=PAIR_CAP,
        thresholds={
            metric: threshold.value for metric, threshold in flags.THRESHOLDS.items()
        },
        level=SIGNIFICANCE_LEVEL,
        transform=None,
    )
    assert config.compare is not None
    assert config.compare.metrics == DEFAULT_METRICS
    assert config.compare.draws == _config.NULL_DRAWS
    assert config.compare.cap == _config.SAMPLE_CAP
    assert config.compare.sinkhorn_epsilon_factors == (SINKHORN_EPSILON_FACTOR,)
    assert config.compare.half_samples == HALF_SAMPLE_PAIRS
    assert config.compare.null == "permutation"
    assert config.compare.uncertainty == "null_std"
    assert config.compare.level == SIGNIFICANCE_LEVEL
    assert config.compare.thresholds == dict(THRESHOLDS)
    assert config.compare_signals is not None
    assert config.compare_signals.thresholds == dict(THRESHOLDS)
    assert config.compare_signals.backend == "numpy"
    assert config.compare_signals.chunk_size is None


def test_the_backends_are_those_of_the_array_module() -> None:
    """The section accepts exactly the values of ``ArrayBackend``."""
    backends = get_args(ArrayBackend.__value__)
    for backend in backends:
        section = CompareSignalsSection(
            generator="models:generator", noise="models:noise", backend=backend
        )
        assert section.backend == backend
    with pytest.raises(ValidationError, match="backend"):
        CompareSignalsSection.model_validate({**SIGNALS, "backend": "tpu"})


def test_the_dump_shows_every_default_and_the_merged_thresholds() -> None:
    """A dump of the config holds every field that runs, thresholds included."""
    config = PipelineConfig.model_validate(
        mapping(
            compare={"thresholds": {"energy": 0.3}},
            diagnose={"thresholds": {"skewness": 0.4}},
        )
    )
    dump = config.model_dump(mode="json")
    assert dump["budget"] == _config.COST_BUDGET
    assert dump["diagnose"]["thresholds"]["skewness"] == 0.4
    assert (
        dump["diagnose"]["thresholds"]["kurtosis"] == flags.THRESHOLDS["kurtosis"].value
    )
    assert dump["compare"]["thresholds"]["energy"] == 0.3
    assert dump["compare"]["thresholds"]["mmd"] == THRESHOLDS["mmd"]
    assert set(dump["compare"]["thresholds"]) == set(THRESHOLDS)
    assert dump["compare"]["metrics"] == list(DEFAULT_METRICS)
    assert dump["compare_signals"] is None
    assert dump["posteriors"][0] == {
        "name": "a",
        "path": "a.parquet",
        "format": "table",
        "log_weights": None,
        "n_eff": None,
    }


def test_the_whitening_takes_the_default_of_the_engine() -> None:
    """Null becomes reference with a reference posterior, and pooled without."""
    pooled = PipelineConfig.model_validate(mapping(compare={}))
    assert pooled.compare is not None
    assert pooled.compare.whitening == "pooled"
    reference = PipelineConfig.model_validate(mapping(compare={"reference": "a"}))
    assert reference.compare is not None
    assert reference.compare.whitening == "reference"
    explicit = PipelineConfig.model_validate(
        mapping(compare={"reference": "a", "whitening": "pooled"})
    )
    assert explicit.compare is not None
    assert explicit.compare.whitening == "pooled"


def test_fisher_arrays_belong_to_the_fisher_whitening() -> None:
    """Both arrays with the fisher whitening, and neither without it."""
    arrays = {"fisher": [[2.0, 0.0], [0.0, 1.0]], "centre": [0.0, 1.0]}
    ok = PipelineConfig.model_validate(
        mapping(compare={"whitening": "fisher", **arrays})
    )
    assert ok.compare is not None
    assert ok.compare.fisher == arrays["fisher"]
    for section in (
        {"whitening": "fisher"},
        {"whitening": "fisher", "centre": [0.0, 1.0]},
        {"whitening": "fisher", "fisher": "fisher.npy"},
    ):
        with pytest.raises(
            ValidationError, match="fisher whitening needs fisher and centre"
        ):
            PipelineConfig.model_validate(mapping(compare=section))
    without_whitening: list[dict[str, Any]] = [
        arrays,
        {"centre": [0.0, 1.0]},
        {"whitening": "pooled", **arrays},
    ]
    for section in without_whitening:
        with pytest.raises(
            ValidationError, match="serve the fisher whitening, not 'pooled'"
        ):
            PipelineConfig.model_validate(mapping(compare=section))
    signals = PipelineConfig.model_validate(
        mapping(compare_signals={**SIGNALS, "centre": [0.0, 1.0]})
    )
    assert signals.compare_signals is not None
    with pytest.raises(ValidationError, match="fisher needs centre"):
        PipelineConfig.model_validate(
            mapping(compare_signals={**SIGNALS, "fisher": [[1.0]]})
        )


def test_arrays_are_nested_lists_or_npy_paths() -> None:
    """Both forms pass; other texts, ragged lists and non-numbers do not."""
    for value in ([1.0, 2.0], [[1, 2], [3, 4.5]], "fisher.npy", "dir/fisher.npy"):
        PipelineConfig.model_validate(
            mapping(compare={"whitening": "fisher", "fisher": value, "centre": value})
        )
    bad: list[Any] = [
        "fisher.txt",
        [],
        [[1, 2], [3]],
        [True, False],
        [1, "a"],
        [[1, 2], 3],
        {"a": 1},
        [math.nan],
    ]
    for value in bad:
        with pytest.raises(ValidationError):
            PipelineConfig.model_validate(
                mapping(
                    compare={"whitening": "fisher", "fisher": value, "centre": [1.0]}
                )
            )


# * Thresholds


def test_thresholds_override_the_defaults_by_metric() -> None:
    """An override replaces its metric alone; the others keep their defaults."""
    config = PipelineConfig.model_validate(
        mapping(
            compare={"thresholds": {"energy": 0.5, "width_ratio": 0.1}},
            compare_signals={**SIGNALS, "thresholds": {"energy": 0.7}},
        )
    )
    assert config.compare is not None
    assert config.compare_signals is not None
    assert config.compare.thresholds["energy"] == 0.5
    assert config.compare.thresholds["width_ratio"] == 0.1
    assert config.compare.thresholds["mmd"] == THRESHOLDS["mmd"]
    assert config.compare_signals.thresholds["energy"] == 0.7
    assert config.compare_signals.thresholds["width_ratio"] == THRESHOLDS["width_ratio"]
    assert config.compare_signals.thresholds["mmd"] == THRESHOLDS["mmd"]


def test_unknown_threshold_metrics_are_invalid() -> None:
    """Each section knows the metrics of its own defaults."""
    for section, bad in (
        ("diagnose", {"energy": 1.0}),
        ("compare", {"skewness": 1.0}),
        ("compare_signals", {"energi": 1.0}),
    ):
        content = (
            {**SIGNALS, "thresholds": bad}
            if section == "compare_signals"
            else {"thresholds": bad}
        )
        found = invalid(mapping(**{section: content}))
        assert (section, "thresholds") == found[0][0]
        assert "unknown threshold metrics" in found[0][1]
    with pytest.raises(ValidationError, match=r"finite|valid number"):
        PipelineConfig.model_validate(
            mapping(compare={"thresholds": {"energy": math.inf}})
        )
    with pytest.raises(ValidationError, match="valid number"):
        PipelineConfig.model_validate(
            mapping(compare={"thresholds": {"energy": "0.2"}})
        )


# * The space


def test_the_space_has_three_forms() -> None:
    """A file path, an inline space and eryn priors with periods and spheres."""
    path = PipelineConfig.model_validate(mapping(space="space.yaml"))
    assert path.space == "space.yaml"
    inline = PipelineConfig.model_validate(
        mapping(
            space={
                "parameters": [
                    {"name": "lon", "topology": "circle", "bounds": [0.0, 2 * math.pi]},
                    {"name": "slat", "topology": "interval", "bounds": [-1.0, 1.0]},
                ],
                "spheres": [
                    {"azimuth": "lon", "polar": "slat", "polar_form": "sin_latitude"}
                ],
            }
        )
    )
    assert isinstance(inline.space, ParamSpace)
    assert inline.space.spheres[0].polar_form is PolarForm.SIN_LATITUDE
    eryn = PipelineConfig.model_validate(
        mapping(
            space={
                "eryn_priors": "models:priors",
                "periodic": {"phase": 6.28, 1: 3.14},
                "spheres": [
                    {"azimuth": "lon", "polar": "slat", "polar_form": "sin_latitude"}
                ],
            }
        )
    )
    assert isinstance(eryn.space, ErynSpace)
    assert eryn.space.periodic == {"phase": 6.28, 1: 3.14}
    assert eryn.space.spheres[0].azimuth == "lon"
    bare = PipelineConfig.model_validate(mapping(space=ERYN_SPACE))
    assert isinstance(bare.space, ErynSpace)
    assert bare.space.periodic is None
    assert bare.space.spheres == ()


def test_a_space_that_fits_no_form_says_so() -> None:
    """A mapping without parameters or eryn_priors, or another type, is invalid."""
    for space in ({"names": ["x0"]}, ["x0"], 3, None, {}):
        found = invalid(mapping(space=space))
        assert found[0][0] == ("space",)
        assert "path of a ParamSpace YAML file" in found[0][1]
    # * A mapping with both keys is eryn priors, whose model rejects the extra key.
    both = invalid(mapping(space={**ERYN_SPACE, **INLINE_SPACE}))
    assert (("space", "eryn", "parameters"), "Extra inputs are not permitted") in both
    bad_inline = invalid(mapping(space={"parameters": [{"name": " x"}]}))
    assert bad_inline[0][0][:2] == ("space", "inline")
    bad_eryn = invalid(mapping(space={"eryn_priors": "models"}))
    assert bad_eryn[0][0] == ("space", "eryn", "eryn_priors")


# * Posteriors


def test_each_format_has_its_own_keys() -> None:
    """The five formats validate with their keys, and fill the rest."""
    entries = [
        {
            "name": "e",
            "format": "eryn",
            "path": "run.h5",
            "branch": "model_0",
            "num_repeats_in_model": 2,
            "burn_in": 10,
            "thin": "tau",
        },
        {"name": "c", "format": "chains", "path": "c.npz", "burn_in": 5, "thin": 3},
        {"name": "s", "format": "samples", "path": "s.npz", "n_eff": 100},
        {"name": "s2", "format": "samples", "path": "s.npy", "n_eff": [10, 20.5]},
        {
            "name": "t",
            "format": "table",
            "path": "t.csv",
            "log_weights": "w",
            "n_eff": 7.5,
        },
        {
            "name": "h",
            "format": "hdf5",
            "path": "h.h5",
            "group": "posterior",
            "log_weights": "w",
            "n_eff": [3, 4],
        },
    ]
    config = PipelineConfig.model_validate(mapping(posteriors=entries))
    eryn, chains, samples, listed, table, hdf5 = config.posteriors
    assert isinstance(eryn, ErynEntry)
    assert (eryn.branch, eryn.num_repeats_in_model) == ("model_0", 2)
    assert isinstance(chains, ChainsEntry)
    assert (chains.burn_in, chains.thin) == (5, 3)
    assert isinstance(samples, SamplesEntry)
    assert samples.n_eff == 100.0
    assert isinstance(listed, SamplesEntry)
    assert listed.n_eff == [10.0, 20.5]
    assert isinstance(table, TableEntry)
    assert (table.log_weights, table.n_eff) == ("w", 7.5)
    assert isinstance(hdf5, Hdf5Entry)
    assert (hdf5.group, hdf5.n_eff) == ("posterior", [3.0, 4.0])
    defaults = PipelineConfig.model_validate(
        mapping(
            posteriors=[
                {"name": "e", "format": "eryn", "path": "run.h5"},
                {"name": "c", "format": "chains", "path": "c.npz"},
            ]
        )
    )
    for entry in defaults.posteriors:
        assert isinstance(entry, ErynEntry | ChainsEntry)
        assert (entry.burn_in, entry.thin) == (0, "tau")


def test_a_key_of_another_format_is_rejected() -> None:
    """Each key belongs to its format, and the entry is named in the location."""
    cases: list[tuple[dict[str, Any], str]] = [
        ({"format": "table", "burn_in": 3}, "burn_in"),
        ({"format": "table", "group": "posterior"}, "group"),
        ({"format": "samples", "thin": 2}, "thin"),
        ({"format": "chains", "n_eff": 5}, "n_eff"),
        ({"format": "eryn", "log_weights": "w"}, "log_weights"),
        ({"format": "hdf5", "branch": "b", "group": "g"}, "branch"),
    ]
    for entry, key in cases:
        found = invalid(mapping(posteriors=[{"name": "a", "path": "p", **entry}]))
        assert ("posteriors", 0, entry["format"], key) in {
            location for location, _ in found
        }
    missing_group = invalid(
        mapping(posteriors=[{"name": "a", "path": "p", "format": "hdf5"}])
    )
    assert missing_group[0][0] == ("posteriors", 0, "hdf5", "group")


def test_the_format_must_be_known() -> None:
    """A missing or unknown format cannot pick an entry model."""
    for entry in (
        {"name": "a", "path": "p"},
        {"name": "a", "path": "p", "format": "fits"},
    ):
        found = invalid(mapping(posteriors=[entry]))
        assert found[0][0] == ("posteriors", 0)


def test_names_are_unique_and_clean() -> None:
    """Repeated, empty and padded names are invalid, and so is an empty list."""
    twice = invalid(mapping(posteriors=[dict(TABLE), dict(TABLE)]))
    assert "posterior names must be unique, repeated: 'a'" in twice[0][1]
    for name in ("", " a", "a "):
        found = invalid(mapping(posteriors=[{**TABLE, "name": name}]))
        assert found[0][0] == ("posteriors", 0, "table", "name")
    assert invalid(mapping(posteriors=[]))[0][0] == ("posteriors",)
    assert invalid(mapping(posteriors=[{**TABLE, "path": ""}]))[0][0][-1] == "path"


def test_counts_and_numbers_are_strict() -> None:
    """Bools, floats and strings are not integers; bools and strings are not floats."""
    for seed in (True, 1.0, "1", -1, SEED_LIMIT):
        assert ("seed",) in locations(mapping(seed=seed))
    PipelineConfig.model_validate(mapping(seed=SEED_LIMIT - 1))
    for field in ("draws", "cap", "pair_cap"):
        for value in (True, 5.0, "5", 0 if field == "draws" else 1):
            assert ("diagnose", field) in locations(mapping(diagnose={field: value}))
    for field in ("draws", "cap", "half_samples"):
        for value in (True, 5.0, "5"):
            assert ("compare", field) in locations(mapping(compare={field: value}))
    for value in (True, 2.0, 0):
        assert ("compare_signals", "chunk_size") in locations(
            mapping(compare_signals={**SIGNALS, "chunk_size": value})
        )
    for value in (True, 1.5, -1, "2"):
        found = locations(
            mapping(
                posteriors=[
                    {"name": "a", "path": "p", "format": "chains", "burn_in": value}
                ]
            )
        )
        assert ("posteriors", 0, "chains", "burn_in") in found
    for value in (True, 1.5, 0, "x", "Tau"):
        found = locations(
            mapping(
                posteriors=[
                    {"name": "a", "path": "p", "format": "chains", "thin": value}
                ]
            )
        )
        assert ("posteriors", 0, "chains", "thin") in {
            location[:4] for location in found
        }
    for field in ("level",):
        for value in (True, "2", math.nan):
            assert ("diagnose", field) in locations(mapping(diagnose={field: value}))
    for budget in (True, "10", 0, -1.0):
        assert ("budget",) in locations(mapping(budget=budget))
    for n_eff in (True, "5", 0, -1.0, [1.0, True]):
        found = locations(
            mapping(
                posteriors=[
                    {"name": "a", "path": "p", "format": "samples", "n_eff": n_eff}
                ]
            )
        )
        assert any(location[:3] == ("posteriors", 0, "samples") for location in found)
    assert PipelineConfig.model_validate(mapping(budget=5)).budget == 5.0


def test_an_infinite_budget_is_valid_and_stays_out_of_the_canonical_bytes() -> None:
    """The model holds infinity, and the canonical JSON does not record the budget."""
    config = PipelineConfig.model_validate(mapping(budget=math.inf))
    assert config.budget == math.inf
    assert config.model_dump(mode="json")["budget"] == math.inf
    # * The budget changes no row, so it stays out of the canonical bytes.
    canonical = _canonical_json(config, ParamSpace.model_validate(INLINE_SPACE))
    assert b"budget" not in canonical


# * Cross-field checks


def test_a_config_needs_a_section_and_its_required_keys() -> None:
    """Missing top-level keys and an empty list of sections are invalid."""
    without_section = mapping()
    del without_section["diagnose"]
    found = invalid(without_section)
    assert (
        "needs at least one of the sections diagnose, compare, compare_signals"
        in found[0][1]
    )
    assert invalid(mapping(diagnose=None))[0][0] == ()
    for key in ("seed", "output", "space", "posteriors"):
        incomplete = mapping()
        del incomplete[key]
        assert ((key,), "Field required") in invalid(incomplete)
    assert ("compare_signals", "generator") in locations(
        mapping(compare_signals={"noise": "a:b"})
    )
    assert ("compare_signals", "noise") in locations(
        mapping(compare_signals={"generator": "a:b"})
    )


def test_references_name_posteriors_of_the_config() -> None:
    """A reference that no posterior carries is invalid in both comparison sections."""
    for section, content in (("compare", {}), ("compare_signals", SIGNALS)):
        PipelineConfig.model_validate(
            mapping(**{section: {**content, "reference": "b"}})
        )
        found = invalid(mapping(**{section: {**content, "reference": "c"}}))
        assert (
            f"{section}: reference 'c' is not a posterior of the config" in found[0][1]
        )


def test_the_physical_transform_needs_eryn_priors() -> None:
    """In any section, ``physical`` is valid only with the eryn form of the space."""
    for section, content in (
        ("diagnose", {}),
        ("compare", {}),
        ("compare_signals", SIGNALS),
    ):
        section_mapping = {**content, "transform": "physical"}
        found = invalid(mapping(**{section: section_mapping}))
        assert (
            f"{section}: transform 'physical' needs a space with eryn_priors"
            in found[0][1]
        )
        config = PipelineConfig.model_validate(
            mapping(space=ERYN_SPACE, **{section: section_mapping})
        )
        assert getattr(config, section).transform == "physical"
    named = PipelineConfig.model_validate(
        mapping(compare={"transform": "models:shift"})
    )
    assert named.compare is not None
    assert named.compare.transform == "models:shift"


# * Import paths


@pytest.mark.parametrize(
    "reference",
    ["models:priors", "pkg.mod:Class.attribute", "_a:_b", "a1.b2:c3", "módulo:objeto"],
)
def test_import_paths_have_dotted_identifiers_on_both_sides(reference: str) -> None:
    """A path is ``module:attribute``, each side dotted identifiers."""
    assert IMPORT_PATH_PATTERN.fullmatch(reference)
    config = PipelineConfig.model_validate(mapping(space={"eryn_priors": reference}))
    assert isinstance(config.space, ErynSpace)


@pytest.mark.parametrize(
    "reference",
    [
        "models",
        "models:",
        ":priors",
        "a b:c",
        "a:b:c",
        "1a:b",
        "a..b:c",
        "a:b-c",
        "a.:b",
        "",
    ],
)
def test_other_texts_are_not_import_paths(reference: str) -> None:
    """The check is the shape alone; resolving waits for the run."""
    assert not IMPORT_PATH_PATTERN.fullmatch(reference)
    found = invalid(mapping(space={"eryn_priors": reference}))
    assert "is not an import path 'module:attribute'" in found[0][1]
    found_transform = {
        location[:2]
        for location in locations(mapping(compare={"transform": reference}))
    }
    assert ("compare", "transform") in found_transform


# * Loading files


def test_a_config_file_loads_with_its_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Relative paths start at the folder of the file, whatever the working folder."""
    folder = tmp_path / "project"
    folder.mkdir()
    path = write(
        folder,
        """
seed: 3
output: results
space: space.yaml
posteriors:
  - {name: a, format: table, path: data/a.parquet}
diagnose: {}
""",
    )
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    config = load_config(path)
    assert config.base_folder == folder.resolve()
    assert config.config_file == path.resolve()
    assert config.seed == 3
    assert config.diagnose is not None
    monkeypatch.chdir(tmp_path)
    relative = load_config("project/config.yaml")
    assert relative.base_folder == folder.resolve()


def test_the_annotated_example_of_the_module_validates(tmp_path: Path) -> None:
    """The YAML of the module docstring is a valid config with every section."""
    documentation = pipeline_config.__doc__
    assert documentation is not None
    block = documentation.split("every section::\n\n", 1)[1].split("\n\nSections", 1)[0]
    config = load_config(write(tmp_path, textwrap.dedent(block)))
    assert config.diagnose is not None
    assert config.compare is not None
    assert config.compare_signals is not None
    assert [entry.format for entry in config.posteriors] == [
        "eryn",
        "chains",
        "samples",
        "table",
        "hdf5",
    ]
    assert config.compare.transform == "physical"
    assert config.diagnose.thresholds["skewness"] == 0.4


def test_the_example_config_validates() -> None:
    """The config of the pipeline example needs no data to validate."""
    config = load_config(EXAMPLE)
    assert config.base_folder == EXAMPLE.parent
    assert [entry.name for entry in config.posteriors] == ["mcmc", "sbi", "peer"]
    assert config.diagnose is not None
    assert config.compare is not None
    assert config.compare.reference == "mcmc"
    assert config.compare_signals is not None
    assert isinstance(config.space, ErynSpace)


def test_the_epsilon_example_config_validates() -> None:
    """The Sinkhorn curve of the example compares alone, at six factors of epsilon."""
    config = load_config(EPSILON_EXAMPLE)
    assert config.base_folder == EPSILON_EXAMPLE.parent
    assert config.seed == 2026
    assert config.output == "results_epsilon"
    assert [entry.name for entry in config.posteriors] == ["mcmc", "sbi", "peer"]
    assert config.diagnose is None
    assert config.compare_signals is None
    assert config.compare is not None
    assert config.compare.reference == "mcmc"
    assert config.compare.metrics == ("sinkhorn",)
    assert config.compare.sinkhorn_epsilon_factors == (0.05, 0.1, 0.25, 0.5, 1.0, 2.0)
    assert config.compare.draws == 100
    assert config.compare.cap == 1_000
    main = load_config(EXAMPLE)
    assert config.posteriors == main.posteriors
    assert config.space == main.space


def test_unknown_keys_are_invalid_with_their_location() -> None:
    """A misspelt key names the entry that holds it."""
    top = invalid(mapping(sead=1))
    assert (("sead",), "Extra inputs are not permitted") in top
    nested = invalid(mapping(diagnose={"draw": 5}))
    assert (("diagnose", "draw"), "Extra inputs are not permitted") in nested
    deep = invalid(mapping(posteriors=[{**TABLE, "colour": 1}]))
    assert (
        ("posteriors", 0, "table", "colour"),
        "Extra inputs are not permitted",
    ) in deep
    space = invalid(mapping(space={**ERYN_SPACE, "period": {}}))
    assert (("space", "eryn", "period"), "Extra inputs are not permitted") in space


def test_a_repeated_key_names_the_key_and_its_line(tmp_path: Path) -> None:
    """The strict loader rejects what plain YAML overwrites silently."""
    top = write(tmp_path, "seed: 1\noutput: a\nseed: 2\n")
    with pytest.raises(
        ConfigError, match=r"(?s)found the key 'seed' more than once.*line 3"
    ):
        load_config(top)
    nested = write(tmp_path, "seed: 1\ndiagnose:\n  draws: 5\n  cap: 10\n  draws: 6\n")
    with pytest.raises(
        ConfigError, match=r"(?s)found the key 'draws' more than once.*line 5"
    ):
        load_config(nested)
    with pytest.raises(ConfigError, match=r"config\.yaml"):
        load_config(nested)
    # * An alias is no repeated key; the anchored key still meets the validation.
    aliased = write(
        tmp_path,
        """
base: &base {name: a, format: table, path: a.parquet}
seed: 1
output: r
space: {parameters: [{name: x0}]}
posteriors: [*base]
diagnose: {}
""",
    )
    with pytest.raises(ValidationError, match="base"):
        load_config(aliased)


MERGED = """
seed: 1
output: r
space: {parameters: [{name: x0}]}
posteriors:
  - &base {name: a, format: table, path: a.parquet}
  - <<: *base
    name: b
    path: b.parquet
diagnose: {}
"""


def test_a_merge_key_may_be_overridden_by_explicit_keys(tmp_path: Path) -> None:
    """An entry that merges an anchored one keeps its own name and path."""
    config = load_config(write(tmp_path, MERGED))
    first, second = config.posteriors
    assert (first.name, first.path) == ("a", "a.parquet")
    assert (second.name, second.path) == ("b", "b.parquet")
    assert second.format == "table"
    # * A list of merged mappings works too; the explicit name still wins.
    listed = MERGED.replace(
        "  - <<: *base\n",
        "  - <<: [*base, {name: z, log_weights: w}]\n",
    )
    third = load_config(write(tmp_path, listed)).posteriors[1]
    assert isinstance(third, TableEntry)
    assert (third.name, third.log_weights) == ("b", "w")


def test_an_explicit_key_repeated_beside_a_merge_key_still_raises(
    tmp_path: Path,
) -> None:
    """Only the explicit keys count, so a repeat among them is an error."""
    own_keys = "    path: b.parquet\n"
    repeated = MERGED.replace(own_keys, own_keys + "    name: c\n")
    with pytest.raises(ConfigError, match=r"(?s)found the key 'name' more than once"):
        load_config(write(tmp_path, repeated))


def test_an_unquoted_null_key_is_the_field_named_null(tmp_path: Path) -> None:
    """YAML reads null: as a null key; the loader takes it for the null construction."""
    template = (
        "seed: 1\noutput: r\nspace: {parameters: [{name: x0}]}\n"
        "posteriors: [{name: a, format: table, path: a.parquet}]\n"
        "compare: {reference: a, %s: split_half}\n"
    )
    for key in ("null", '"null"', "Null", "NULL", "~"):
        config = load_config(write(tmp_path, template % key))
        assert config.compare is not None
        assert config.compare.null == "split_half"
    twice = template % 'null: split_half, "null"'
    with pytest.raises(ConfigError, match="found the key 'null' more than once"):
        load_config(write(tmp_path, twice))


def test_documents_that_cannot_be_read_raise_a_config_error(tmp_path: Path) -> None:
    """A missing file, bad bytes, bad YAML and other documents name the file."""
    with pytest.raises(ConfigError, match=r"cannot read the config .*missing\.yaml"):
        load_config(tmp_path / "missing.yaml")
    bad_bytes = tmp_path / "bytes.yaml"
    bad_bytes.write_bytes(b"seed: \xff\xfe\n")
    with pytest.raises(ConfigError, match=r"cannot read the config .*bytes\.yaml"):
        load_config(bad_bytes)
    with pytest.raises(ConfigError, match=r"config\.yaml is not valid YAML"):
        load_config(write(tmp_path, "seed: [1, 2\n"))
    with pytest.raises(ConfigError, match="is not valid YAML"):
        load_config(write(tmp_path, "seed: 1\n---\nseed: 2\n"))
    for text, kind in (("", "NoneType"), ("- 1\n- 2\n", "list"), ("3\n", "int")):
        with pytest.raises(ConfigError, match=f"must hold a mapping.*got {kind}"):
            load_config(write(tmp_path, text))
    # * Tags cannot build Python objects: the safe loader refuses them.
    with pytest.raises(ConfigError, match="could not determine a constructor"):
        load_config(write(tmp_path, "seed: !!python/object/apply:os.getcwd []\n"))
    with pytest.raises(ConfigError, match="found unhashable key"):
        load_config(write(tmp_path, "? [1, 2]\n: 3\n"))


def test_an_invalid_file_raises_the_validation_error(tmp_path: Path) -> None:
    """Pydantic's error, with every problem and its location."""
    path = write(tmp_path, "seed: true\noutput: r\n")
    with pytest.raises(ValidationError) as info:
        load_config(path)
    found = {error["loc"] for error in info.value.errors()}
    assert {("seed",), ("space",), ("posteriors",)} <= found


# * The config hash


def test_the_hash_covers_the_config_and_the_space_but_not_output_or_budget() -> None:
    """The canonical JSON is sorted and compact; the folder and the budget stay out."""
    space = ParamSpace.model_validate(INLINE_SPACE)
    config = PipelineConfig.model_validate(mapping())
    canonical = _canonical_json(config, space)
    text = canonical.decode("utf-8")
    assert re.fullmatch(r"[0-9a-f]{64}", _config_hash(canonical))
    assert ": " not in text
    assert ", " not in text
    assert not text.endswith("\n")
    parsed = json.loads(text)
    assert set(parsed) == {"config", "space"}
    assert "output" not in parsed["config"]
    assert "budget" not in parsed["config"]
    assert parsed["space"] == space.model_dump(mode="json")
    assert list(parsed["config"]) == sorted(parsed["config"])
    other_folder = PipelineConfig.model_validate(mapping(output="elsewhere"))
    assert _canonical_json(other_folder, space) == canonical
    other_budget = PipelineConfig.model_validate(mapping(budget=10.0))
    assert _canonical_json(other_budget, space) == canonical
    for change in ({"seed": 2}, {"diagnose": {"draws": 5}}):
        changed = PipelineConfig.model_validate(mapping(**change))
        assert _config_hash(_canonical_json(changed, space)) != _config_hash(canonical)
    other_space = ParamSpace.model_validate({"parameters": [{"name": "y"}]})
    assert _canonical_json(config, other_space) != canonical


def test_a_space_with_infinite_bounds_writes_the_same_bytes_every_time() -> None:
    """The real line writes Infinity constants, deterministically."""
    space = ParamSpace.model_validate(INLINE_SPACE)
    config = PipelineConfig.model_validate(mapping(space=INLINE_SPACE))
    first = _canonical_json(config, space)
    assert b'"bounds":[-Infinity,Infinity]' in first
    assert _canonical_json(config, ParamSpace.model_validate(INLINE_SPACE)) == first
    assert first.count(b'"bounds":[-Infinity,Infinity]') == 4


def test_the_hash_does_not_depend_on_the_key_order_of_the_file(tmp_path: Path) -> None:
    """Sorted keys make equal configs hash equal, however the file orders them."""
    space = ParamSpace.model_validate(INLINE_SPACE)
    first = write(
        tmp_path,
        "seed: 1\noutput: a\nspace: {parameters: [{name: x0}, {name: x1}]}\n"
        "posteriors: [{name: a, format: table, path: a.parquet}]\n"
        "diagnose: {draws: 5, cap: 50}\n",
    )
    second = tmp_path / "second.yaml"
    second.write_text(
        "diagnose: {cap: 50, draws: 5}\n"
        "posteriors: [{path: a.parquet, format: table, name: a}]\n"
        "space: {parameters: [{name: x0}, {name: x1}]}\noutput: b\nseed: 1\n",
        encoding="utf-8",
    )
    assert _canonical_json(load_config(first), space) == _canonical_json(
        load_config(second), space
    )


def test_module_constants_name_the_pipeline_defaults() -> None:
    """The section models are public in the module, with the config at the top."""
    assert pipeline_config.PipelineConfig is PipelineConfig
    assert CompareSection().draws == _config.NULL_DRAWS
