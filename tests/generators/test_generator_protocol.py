"""Tests for the generator and noise-model protocols of data space."""

from typing import Any

import jax.numpy as jnp
import numpy as np

from mixed_signals.generators import FisherGenerator, NoiseModel, SignalGenerator


class ScaledSignal:
    """A generator from outside the package: a scaled copy of the parameters."""

    def generate_template(self, params: Any) -> Any:
        """Return twice the parameters as the signals."""
        return 2.0 * np.asarray(params)


class ScaledSignalWithFisher(ScaledSignal):
    """A generator that also gives Fisher matrices."""

    def information_matrix(self, params: Any, *, noise: NoiseModel) -> Any:
        """Return the identity matrix of the dimension, once per row."""
        values = np.asarray(params)
        dimension = values.shape[1]
        return np.broadcast_to(np.eye(dimension), (len(values), dimension, dimension))


class FlatNoise:
    """A noise model from outside the package, with a description."""

    description = "flat"

    def whiten(self, signals: Any) -> Any:
        """Return the signals as real vectors."""
        values = jnp.asarray(signals)
        return values.reshape(values.shape[0], -1)


class SignalsOnly:
    """An object with neither a whitening nor Fisher matrices."""

    def generate_template(self, params: Any) -> Any:
        """Return the parameters as the signals."""
        return params


class Unrelated:
    """An object with none of the methods."""


def test_objects_with_the_methods_meet_the_protocols() -> None:
    """Any object with generate_template, whiten or information_matrix qualifies."""
    generator: SignalGenerator = ScaledSignal()
    assert isinstance(generator, SignalGenerator)
    assert isinstance(ScaledSignalWithFisher(), FisherGenerator)
    assert isinstance(FlatNoise(), NoiseModel)
    signals = generator.generate_template(np.ones((3, 2)))
    assert signals.shape == (3, 2)


def test_objects_without_the_methods_do_not_meet_the_protocols() -> None:
    """A protocol needs its method, and each protocol needs only its own."""
    assert not isinstance(SignalsOnly(), NoiseModel)
    assert not isinstance(SignalsOnly(), FisherGenerator)
    assert not isinstance(FlatNoise(), SignalGenerator)
    assert not isinstance(Unrelated(), SignalGenerator)
    assert not isinstance(ScaledSignal(), FisherGenerator)


def test_the_description_is_not_a_member_of_the_noise_protocol() -> None:
    """A noise model without a description is still a NoiseModel."""

    class QuietNoise:
        def whiten(self, signals: Any) -> Any:
            return signals

    assert isinstance(QuietNoise(), NoiseModel)
    assert not hasattr(QuietNoise(), "description")
    assert FlatNoise().description == "flat"


def test_the_protocols_check_the_methods_not_their_signatures() -> None:
    """A method with another signature still passes isinstance, as documented."""

    class WrongSignature:
        def generate_template(self) -> Any:
            return None

    assert isinstance(WrongSignature(), SignalGenerator)
