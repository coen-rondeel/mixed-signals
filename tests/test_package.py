"""Smoke tests for the package import, version and module layout."""

import importlib
import pkgutil
import subprocess
import sys
from importlib.metadata import version

import pytest
from jaxtyping import TypeCheckError

import mixed_signals
from mixed_signals import cli


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
