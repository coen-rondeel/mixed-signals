"""Tests for the ``mixed-signals`` command line."""

from importlib.metadata import entry_points

import pytest

import mixed_signals
from mixed_signals import cli


def test_version_flag_prints_version(capsys: pytest.CaptureFixture[str]) -> None:
    """``--version`` prints the package version and exits with code 0."""
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["--version"])
    assert exit_info.value.code == 0
    assert mixed_signals.__version__ in capsys.readouterr().out


def test_no_arguments_prints_usage(capsys: pytest.CaptureFixture[str]) -> None:
    """Without arguments the command prints its usage and succeeds."""
    assert cli.main([]) == 0
    assert "usage: mixed-signals" in capsys.readouterr().out


def test_console_script_points_at_main() -> None:
    """The installed ``mixed-signals`` script calls ``cli.main``."""
    (script,) = entry_points(group="console_scripts", name="mixed-signals")
    assert script.value == "mixed_signals.cli:main"
