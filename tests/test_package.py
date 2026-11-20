"""Smoke tests for the package import, version and module layout."""

import importlib
import pkgutil
import subprocess
import sys
from importlib.metadata import version

import pytest
from jaxtyping import TypeCheckError

import mixed_signals
from mixed_signals import cli, engines
from mixed_signals._config import COST_BUDGET
from mixed_signals.core.chains import Chains
from mixed_signals.core.result import Result
from mixed_signals.core.samples import SampleSet
from mixed_signals.core.space import ParamSpace
from mixed_signals.engines import compare, compare_signals, diagnose, reweight
from mixed_signals.engines.reweight import Reweighting
from mixed_signals.mcmc.reduce import reduce_chains
from mixed_signals.prng import root_key


def test_version_matches_distribution_metadata() -> None:
    """The package version is read from the installed distribution."""
    assert mixed_signals.__version__ == version("mixed-signals")


def test_every_module_imports() -> None:
    """Every module in the package imports without optional dependencies."""
    module_names = [
        module_info.name
        for module_info in pkgutil.walk_packages(
            mixed_signals.__path__, prefix="mixed_signals."
        )
    ]
    assert "mixed_signals.metrics.ot" in module_names
    for module_name in module_names:
        importlib.import_module(module_name)


def test_eryn_is_imported_lazily() -> None:
    """The eryn adapters import eryn only when called, so the extra stays optional."""
    code = (
        "import sys, mixed_signals.io.eryn_backend, mixed_signals.priors;"
        " sys.exit('eryn' in sys.modules)"
    )
    # * A fresh interpreter, because the test session itself imports eryn.
    assert subprocess.run([sys.executable, "-c", code], check=False).returncode == 0


def test_annotations_are_checked_at_runtime() -> None:
    """The test session runs jaxtyping and beartype on every package function."""
    with pytest.raises(TypeCheckError):
        cli.main(42)  # type: ignore[arg-type]


def test_the_main_objects_are_exported_at_the_top_level() -> None:
    """The documented entry points import from the package root, and list themselves."""
    expected = {
        "ParamSpace": ParamSpace,
        "SampleSet": SampleSet,
        "Chains": Chains,
        "Result": Result,
        "reduce_chains": reduce_chains,
        "diagnose": diagnose,
        "compare": compare,
        "compare_signals": compare_signals,
        "reweight": reweight,
        "Reweighting": Reweighting,
        "root_key": root_key,
    }
    for name, member in expected.items():
        assert getattr(mixed_signals, name) is member, name
        assert name in mixed_signals.__all__, name
    assert "__version__" in mixed_signals.__all__
    assert set(mixed_signals.__all__) == {*expected, "__version__"}


def test_the_engines_package_exports_the_engines_and_their_constants() -> None:
    """compare, compare_signals, diagnose, reweight, their constants and the dry run."""
    names = {
        "compare",
        "compare_signals",
        "DEFAULT_METRICS",
        "RECIPES",
        "diagnose",
        "Diagnosis",
        "DIAGNOSTIC_FAMILIES",
        "estimate_compare_cost",
        "estimate_compare_signals_cost",
        "estimate_diagnose_cost",
        "CostEstimate",
        "COST_BUDGET",
        "reweight",
        "Reweighting",
        "Proposal",
        "TARGET_SUFFIX",
    }
    assert set(engines.__all__) == names
    for name in names:
        assert hasattr(engines, name), name
    assert engines.COST_BUDGET is COST_BUDGET
    assert engines.compare is mixed_signals.compare
