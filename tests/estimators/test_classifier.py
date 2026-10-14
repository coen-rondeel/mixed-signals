"""Tests for the classifier placeholder."""

import jax
import numpy as np
import pytest

from mixed_signals.core.space import ParamSpace
from mixed_signals.estimators import ClassifierRatio, ClassifierSettings, Estimator

SPACE = ParamSpace.model_validate({"parameters": [{"name": "x0"}, {"name": "x1"}]})


def test_fit_raises_until_the_classifier_is_implemented() -> None:
    """The final signature takes two sets and says that the classifier is deferred."""
    with pytest.raises(NotImplementedError, match="deferred"):
        ClassifierRatio.fit(
            np.zeros((10, 2)),
            None,
            np.ones((12, 2)),
            None,
            SPACE,
            key=jax.random.key(0),
        )


def test_log_ratio_raises_until_the_classifier_is_implemented() -> None:
    """A classifier gives a ratio of two densities, so it is not an Estimator."""
    placeholder = ClassifierRatio(settings=ClassifierSettings())
    assert Estimator not in type(placeholder).__mro__
    with pytest.raises(NotImplementedError, match="deferred"):
        placeholder.log_ratio(np.zeros((3, 2)))
    assert ClassifierSettings.name == "classifier"
