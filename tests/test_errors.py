"""Tests for the warning and exception classes and the warning capture."""

import dataclasses
import warnings

import pytest

from mixed_signals import errors
from mixed_signals.core.result import WARNING_DTYPE

WARNING_CLASSES = (
    errors.ValidityWarning,
    errors.SubsamplingWarning,
    errors.CostWarning,
    errors.CoupledChainsWarning,
    errors.ExtrapolationWarning,
    errors.ImportanceWeightWarning,
    errors.ConvergenceWarning,
    errors.PrecisionWarning,
)


@pytest.mark.parametrize("warning_class", WARNING_CLASSES)
def test_warning_classes_share_the_package_base(
    warning_class: type[Warning],
) -> None:
    """Every package warning is a user warning, so default filters show it."""
    assert issubclass(warning_class, errors.MixedSignalsWarning)
    assert issubclass(warning_class, UserWarning)


def test_errors_keep_the_builtin_meaning() -> None:
    """Each package error is also the built-in exception that callers expect."""
    assert issubclass(errors.SchemaError, ValueError)
    assert issubclass(errors.MissingDependencyError, ImportError)
    for error_class in (
        errors.SchemaError,
        errors.MissingDependencyError,
        errors.CapabilityError,
    ):
        assert issubclass(error_class, errors.MixedSignalsError)


def test_capture_records_package_warnings_in_order() -> None:
    """Package warnings become records with their class name and text."""
    with errors.capture_warnings() as records:
        warnings.warn(
            "cap of 5000 samples applied", errors.SubsamplingWarning, stacklevel=2
        )
        warnings.warn("N_eff below 10 D", errors.ValidityWarning, stacklevel=2)
    assert records == [
        errors.WarningRecord("SubsamplingWarning", "cap of 5000 samples applied"),
        errors.WarningRecord("ValidityWarning", "N_eff below 10 D"),
    ]


def test_capture_records_repeated_and_filtered_warnings() -> None:
    """A repeated warning, or one that a filter hides, is still recorded."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", errors.SubsamplingWarning)
        with errors.capture_warnings() as records:
            for _ in range(3):
                warnings.warn("cap reached", errors.SubsamplingWarning, stacklevel=2)
    assert len(records) == 3


def test_capture_lets_other_warnings_through() -> None:
    """Warnings from outside the package reach the active filters unchanged."""
    with (
        pytest.warns(DeprecationWarning, match="old keyword"),
        errors.capture_warnings() as records,
    ):
        warnings.warn("old keyword", DeprecationWarning, stacklevel=2)
    assert records == []


def warn_then_fail() -> None:
    """Raise a package warning, then an exception."""
    warnings.warn("solver stopped", errors.ConvergenceWarning, stacklevel=2)
    raise RuntimeError("solver failed")


def test_capture_fills_the_records_when_the_block_raises() -> None:
    """Records survive an exception that leaves the block."""
    with pytest.raises(RuntimeError), errors.capture_warnings() as records:
        warn_then_fail()
    assert records == [errors.WarningRecord("ConvergenceWarning", "solver stopped")]


def test_record_fields_match_the_warnings_column() -> None:
    """A record converts to exactly the struct of the ``warnings`` column."""
    record = errors.WarningRecord(category="CostWarning", message="above budget")
    record_fields = [field.name for field in dataclasses.fields(record)]
    assert record_fields == [field.name for field in WARNING_DTYPE.fields]
    assert record.to_dict() == {"category": "CostWarning", "message": "above budget"}
