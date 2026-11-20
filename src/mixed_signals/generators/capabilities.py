"""Capabilities of generators and noise models: which methods exist, and the checks.

The methods that a generator implements decide which engines can run, so an
engine asks for the capabilities that it needs before it calls any user code and
raises :class:`~mixed_signals.errors.CapabilityError`, naming the missing
methods, when an object lacks one. A capability is a method that exists and is
callable. The three capabilities are those of
:mod:`mixed_signals.generators.protocol`.
"""

from collections.abc import Iterable
from enum import StrEnum
from typing import Final

from mixed_signals.errors import CapabilityError


class Capability(StrEnum):
    """A method of user code that an engine may need, named by the method."""

    TEMPLATES = "generate_template"
    """``generate_template(params)`` of a signal generator."""
    FISHER = "information_matrix"
    """``information_matrix(params, noise=noise)`` of a generator."""
    WHITEN = "whiten"
    """``whiten(signals)`` of a noise model."""


_SIGNATURES: Final = {
    Capability.TEMPLATES: "generate_template(params)",
    Capability.FISHER: "information_matrix(params, noise=noise)",
    Capability.WHITEN: "whiten(signals)",
}
"""The call of each capability, as the error message writes it."""


def capabilities(model: object) -> frozenset[Capability]:
    """Return the capabilities of an object: its methods that exist and are callable.

    Args:
        model: A generator, a noise model or any other object.

    Returns:
        Every :class:`Capability` whose method the object has and can call.
    """
    return frozenset(
        capability
        for capability in Capability
        if callable(getattr(model, capability.value, None))
    )


def require_capabilities(
    model: object, needed: Iterable[Capability | str], *, engine: str, role: str
) -> None:
    """Raise unless an object has the capabilities that an engine needs.

    Args:
        model: The generator or noise model.
        needed: The capabilities, or their method names; one name alone is
            read as one capability.
        engine: Name of the engine or function, for the message.
        role: What the object is for, ``"generator"`` or ``"noise model"``, for
            the message.

    Raises:
        CapabilityError: If a needed method is missing or not callable; the
            message names the engine, the role, the missing methods and the
            type of the object.
        ValueError: If a name is not a capability.
    """
    wanted = [needed] if isinstance(needed, str) else needed
    required = list(dict.fromkeys(Capability(entry) for entry in wanted))
    present = capabilities(model)
    missing = [capability for capability in required if capability not in present]
    if missing:
        methods = ", ".join(_SIGNATURES[capability] for capability in missing)
        pronoun = "it" if len(missing) == 1 else "them"
        raise CapabilityError(
            f"{engine} needs a {role} with {methods}; {type(model).__name__} lacks"
            f" {pronoun}"
        )
