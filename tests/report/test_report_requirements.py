"""Tests for the check of the report extra, and for the import of the package."""

import importlib
import subprocess
import sys

import pytest

from mixed_signals.errors import MissingDependencyError
from mixed_signals.report import _requirements

# * The report and its modules must not load when only the package is imported.
IMPORT_PROBE = """
import sys
import mixed_signals
import mixed_signals.cli
loaded = [
    name
    for name in sys.modules
    if name == "mixed_signals.report"
    or name.startswith("mixed_signals.report.")
    or name in ("marimo", "cantuccio")
]
assert not loaded, loaded
"""


def test_a_missing_module_names_the_extra() -> None:
    """The error says which module failed and how to install the extra."""
    with pytest.raises(MissingDependencyError) as raised:
        _requirements.require_module("a_module_that_does_not_exist")
    message = str(raised.value)
    assert "a_module_that_does_not_exist" in message
    assert 'pip install "mixed-signals[report]"' in message
    assert isinstance(raised.value.__cause__, ImportError)


def test_an_installed_module_is_returned() -> None:
    """The helper hands back the module that it imported."""
    assert _requirements.require_module("json") is sys.modules["json"]


@pytest.mark.parametrize("name", _requirements.REQUIRED_MODULES)
def test_the_report_needs_both_modules_of_the_extra(
    monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    """Importing the report without marimo or without cantuccio raises at once."""
    monkeypatch.setitem(sys.modules, name, None)
    with pytest.raises(MissingDependencyError, match=rf"cannot import {name}"):
        importlib.reload(_requirements)


def test_the_package_and_the_command_line_do_not_import_the_report() -> None:
    """A user without the extra imports mixed_signals and its command line freely."""
    completed = subprocess.run(
        [sys.executable, "-c", IMPORT_PROBE],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
