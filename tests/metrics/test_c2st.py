"""Tests for the classifier two-sample test placeholder."""

import jax
import numpy as np
import pytest

from mixed_signals.core.space import ParamSpace
from mixed_signals.estimators import ClassifierSettings
from mixed_signals.metrics.classifier import classifier_two_sample_test

SPACE = ParamSpace.model_validate({"parameters": [{"name": "x0"}, {"name": "x1"}]})


def test_the_test_raises_until_it_is_implemented() -> None:
    """The final signature takes two sets and the classifier settings."""
    with pytest.raises(NotImplementedError, match="deferred"):
        classifier_two_sample_test(
            np.zeros((10, 2)),
            None,
            np.ones((12, 2)),
            None,
            SPACE,
            ClassifierSettings(),
            key=jax.random.key(0),
        )
