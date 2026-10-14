"""Tests for the normalizing-flow placeholder."""

import jax
import numpy as np
import pytest

from mixed_signals.core.space import ParamSpace
from mixed_signals.estimators import Estimator, FlowDensity, FlowSettings

SPACE = ParamSpace.model_validate({"parameters": [{"name": "x0"}, {"name": "x1"}]})


def test_fit_raises_until_the_flow_is_implemented() -> None:
    """The final signature exists and says that the flow is deferred."""
    with pytest.raises(NotImplementedError, match="deferred"):
        FlowDensity.fit(np.zeros((10, 2)), None, SPACE, key=jax.random.key(0))


def test_log_density_raises_until_the_flow_is_implemented() -> None:
    """A placeholder flow has no density to evaluate."""
    placeholder = FlowDensity(settings=FlowSettings())
    assert isinstance(placeholder, Estimator)
    with pytest.raises(NotImplementedError, match="deferred"):
        placeholder.log_density(np.zeros((3, 2)))
    assert FlowSettings.name == "flow"
    assert FlowSettings().to_json() == "{}"
