"""Composable bijections between parameter spaces, and whitening.

:class:`~mixed_signals.transforms.base.Transform` maps the parameter vectors of
one problem and tracks its Jacobian, :class:`~mixed_signals.transforms.base.Compose`
chains transforms, and :class:`~mixed_signals.transforms.whiten.Whitening`
expresses a space in units of a posterior width::

    import jax
    from mixed_signals import zoo
    from mixed_signals.transforms import Whitening

    case = zoo.shifted_gaussians(3.0)
    sample_set = case.sample(jax.random.key(0), 10_000)
    whitening = Whitening.from_samples(
        sample_set.samples[0], sample_set.log_weights[0], case.space
    )
    whitening.description  # 'whiten(pooled)'
    whitened = whitening.map_sample_set(sample_set)
"""

from mixed_signals.transforms.base import COMPOSITION_SEPARATOR, Compose, Transform
from mixed_signals.transforms.whiten import (
    CONDITION_NUMBER_LIMIT,
    Whitening,
    WhiteningSource,
)

__all__ = [
    "COMPOSITION_SEPARATOR",
    "CONDITION_NUMBER_LIMIT",
    "Compose",
    "Transform",
    "Whitening",
    "WhiteningSource",
]
