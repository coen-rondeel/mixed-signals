"""Generator protocols and capability checks for the user code of data space.

A signal generator maps parameters to signals in the data domain, a noise
model whitens signals to real vectors whose dot products are the noise-weighted
inner products, and a generator may also give Fisher matrices for a noise
model. :mod:`mixed_signals.generators.protocol` states the three protocols, and
:mod:`mixed_signals.generators.capabilities` lists which methods an object has
and raises :class:`~mixed_signals.errors.CapabilityError` when an engine needs a
method that it lacks. The methods that a generator implements decide which
engines can run.
"""

from mixed_signals.generators.capabilities import (
    Capability,
    capabilities,
    require_capabilities,
)
from mixed_signals.generators.protocol import (
    FisherGenerator,
    NoiseModel,
    SignalGenerator,
)

__all__ = [
    "Capability",
    "FisherGenerator",
    "NoiseModel",
    "SignalGenerator",
    "capabilities",
    "require_capabilities",
]
