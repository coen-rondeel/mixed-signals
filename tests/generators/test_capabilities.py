"""Tests for the capability checks of generators and noise models."""

from typing import Any

import pytest

from mixed_signals.errors import CapabilityError
from mixed_signals.generators import (
    Capability,
    capabilities,
    require_capabilities,
)


class TemplatesOnly:
    """A generator with templates and nothing else."""

    def generate_template(self, params: Any) -> Any:
        """Return the parameters as the signals."""
        return params


class Complete:
    """An object with every method that data space can ask for."""

    def generate_template(self, params: Any) -> Any:
        """Return the parameters as the signals."""
        return params

    def information_matrix(self, params: Any, *, noise: Any) -> Any:
        """Return nothing useful."""
        return params

    def whiten(self, signals: Any) -> Any:
        """Return the signals."""
        return signals


class NotCallable:
    """An object whose methods are attributes that cannot be called."""

    generate_template = 3
    whiten = None


def test_the_capabilities_are_named_by_their_methods() -> None:
    """The enum values are the method names, and the enum is a string."""
    assert Capability.TEMPLATES.value == "generate_template"
    assert Capability.FISHER.value == "information_matrix"
    assert Capability.WHITEN.value == "whiten"
    assert all(isinstance(capability, str) for capability in Capability)
    assert {capability.value for capability in Capability} == {
        "generate_template",
        "information_matrix",
        "whiten",
    }


def test_capabilities_lists_the_methods_that_exist() -> None:
    """Each object reports the capabilities of its own methods."""
    assert capabilities(TemplatesOnly()) == {Capability.TEMPLATES}
    assert capabilities(Complete()) == set(Capability)
    assert capabilities(object()) == frozenset()
    assert isinstance(capabilities(Complete()), frozenset)


def test_an_attribute_that_cannot_be_called_is_no_capability() -> None:
    """A method must be callable, not only present."""
    assert capabilities(NotCallable()) == frozenset()


def test_the_requirements_pass_when_every_method_exists() -> None:
    """Names, enum members and repeats all pass for a complete object."""
    require_capabilities(
        Complete(),
        [Capability.TEMPLATES, "whiten", Capability.FISHER, "whiten"],
        engine="compare_signals",
        role="generator",
    )
    require_capabilities(Complete(), [], engine="compare_signals", role="generator")


def test_a_missing_method_names_the_engine_the_role_and_the_type() -> None:
    """The message reads as the specification states it."""
    with pytest.raises(
        CapabilityError,
        match=r"compare_signals needs a noise model with whiten\(signals\); "
        "TemplatesOnly lacks it",
    ):
        require_capabilities(
            TemplatesOnly(),
            [Capability.WHITEN],
            engine="compare_signals",
            role="noise model",
        )


def test_several_missing_methods_are_all_named() -> None:
    """Every missing method appears once, in the order of the request."""
    with pytest.raises(CapabilityError) as caught:
        require_capabilities(
            object(),
            [Capability.FISHER, Capability.TEMPLATES, Capability.FISHER],
            engine="compare_signals",
            role="generator",
        )
    message = str(caught.value)
    assert message == (
        "compare_signals needs a generator with information_matrix(params,"
        " noise=noise), generate_template(params); object lacks them"
    )


def test_a_method_that_cannot_be_called_counts_as_missing() -> None:
    """An attribute that is not callable fails the requirement."""
    with pytest.raises(CapabilityError, match="NotCallable lacks them"):
        require_capabilities(
            NotCallable(),
            ["generate_template", "whiten"],
            engine="whitened_templates",
            role="generator",
        )


def test_a_single_name_is_one_capability_not_a_set_of_letters() -> None:
    """A bare string, or a bare member, is read as one capability."""
    require_capabilities(
        Complete(), "whiten", engine="compare_signals", role="noise model"
    )
    require_capabilities(
        Complete(),
        Capability.FISHER,
        engine="compare_signals",
        role="generator",
    )
    with pytest.raises(CapabilityError, match="lacks it"):
        require_capabilities(
            TemplatesOnly(), "whiten", engine="compare_signals", role="noise model"
        )


def test_an_unknown_capability_is_rejected() -> None:
    """A name that is no capability raises ValueError."""
    with pytest.raises(ValueError, match="'likelihood' is not a valid Capability"):
        require_capabilities(
            Complete(), ["likelihood"], engine="compare_signals", role="generator"
        )
