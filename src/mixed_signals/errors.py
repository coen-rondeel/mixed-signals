"""Warning and exception classes for the package.

Package warnings are ordinary Python warnings, so notebook users see them as
usual. Engines run the computation of each result row inside
:func:`capture_warnings`, which turns the package warnings raised there into
:class:`WarningRecord` entries for the ``warnings`` column of the result table.

Raise package warnings from Python code outside ``jax.jit``. Inside a jitted
function, Python code runs only while JAX traces it, so a cached call would not
warn again.

Exceptions that the package raises on purpose derive from
:class:`MixedSignalsError`. Invalid pydantic models, such as a
:class:`~mixed_signals.core.space.ParamSpace` with contradictory bounds, raise
:class:`pydantic.ValidationError`, which is a :class:`ValueError`.
"""

import contextlib
import dataclasses
import warnings
from collections.abc import Iterator


class MixedSignalsError(Exception):
    """Base class of the exceptions that the package raises on purpose."""


class SchemaError(MixedSignalsError, ValueError):
    """A table does not match the result schema."""


class MissingDependencyError(MixedSignalsError, ImportError):
    """An optional dependency is not installed.

    The message names the extra that provides it, for example
    ``pip install "mixed-signals[eryn]"``.
    """


class CapabilityError(MixedSignalsError):
    """An engine needs a capability that its inputs do not provide.

    For example, data-space comparisons need a generator that implements
    ``generate_template``, and prior reweighting needs stored log-prior values.
    """


class MixedSignalsWarning(UserWarning):
    """Base class of the warnings that the package raises."""


class ValidityWarning(MixedSignalsWarning):
    """A method ran outside its validity domain.

    At a soft limit the row keeps its value. At a hard limit, where the method
    is undefined, the value is NaN.
    """


class SubsamplingWarning(MixedSignalsWarning):
    """A metric whose cost grows as N² ran on a random subsample above its cap."""


class CostWarning(MixedSignalsWarning):
    """A planned run exceeds its evaluation budget."""


class CoupledChainsWarning(MixedSignalsWarning):
    """A statistic that assumes independent chains ran on coupled walkers."""


class ExtrapolationWarning(MixedSignalsWarning):
    """A significance lies beyond the resolution of the null draws.

    The value comes from a fit to the tail of the null distribution, not from
    the draws themselves.
    """


class ImportanceWeightWarning(MixedSignalsWarning):
    """Importance weights are too uneven for reliable estimates."""


class ConvergenceWarning(MixedSignalsWarning):
    """An iterative solver stopped before it converged."""


class PrecisionWarning(MixedSignalsWarning):
    """An input narrower than float64 was cast to float64.

    The cast cannot restore precision lost before it. Neighbouring float32
    values differ by 6e-8 to 1.2e-7 of their magnitude (IEEE 2019), so a
    float32 posterior narrower than about 1e-7 of its values is quantised.

    References:
        :cite:t:`ieee2019ieee`.
    """


@dataclasses.dataclass(frozen=True)
class WarningRecord:
    """One entry of the ``warnings`` column of the result table.

    Attributes:
        category: Name of the warning class, such as ``"SubsamplingWarning"``.
        message: Text of the warning.
    """

    category: str
    message: str

    def to_dict(self) -> dict[str, str]:
        """Return the record as a mapping with the fields of the table entry."""
        return dataclasses.asdict(self)


@contextlib.contextmanager
def capture_warnings() -> Iterator[list[WarningRecord]]:
    """Record the package warnings raised inside a block.

    Every package warning raised inside the block becomes a
    :class:`WarningRecord`, also when a warning filter would hide it or has
    already shown it once, so the result table holds all of them. The records
    are not shown again. Other warnings pass through to the active filters.

    The yielded list fills when the block exits, also when the block raises.
    Python's warning filters are global to the process, so the capture is not
    thread-safe.

    Example:
        >>> import warnings
        >>> with capture_warnings() as records:
        ...     warnings.warn("cap reached", SubsamplingWarning)
        >>> records
        [WarningRecord(category='SubsamplingWarning', message='cap reached')]

    Yields:
        The list that receives the records, in the order of the warnings.
    """
    records: list[WarningRecord] = []
    caught: list[warnings.WarningMessage] = []
    try:
        with warnings.catch_warnings(record=True) as caught_inside:
            caught = caught_inside
            # * Without "always", a filter could hide a warning or show it only once.
            warnings.simplefilter("always", MixedSignalsWarning)
            yield records
    finally:
        for caught_warning in caught:
            if issubclass(caught_warning.category, MixedSignalsWarning):
                records.append(
                    WarningRecord(
                        category=caught_warning.category.__name__,
                        message=str(caught_warning.message),
                    )
                )
            else:
                warnings.warn_explicit(
                    caught_warning.message,
                    caught_warning.category,
                    caught_warning.filename,
                    caught_warning.lineno,
                    source=caught_warning.source,
                )
