"""Tests for the prior-railing scores."""

import math

import jax
import numpy as np
import pytest

from mixed_signals import zoo
from mixed_signals.core.space import ParamSpace
from mixed_signals.diagnostics.railing import railed_parameters, railing_scores


def test_scores_recover_the_railed_gaussians() -> None:
    """The mean score of railed normals is exp(−z²/2) within its scatter."""
    offsets = [0.0, 1.0, 2.0]
    case = zoo.railed_gaussians(offsets)
    batch = case.sample(jax.random.key(0), 4_000, batch_size=10).samples
    for index, name in enumerate(case.names):
        scores = [
            float(railing_scores(batch[problem, index], None, case.space).lower[0])
            for problem in range(10)
        ]
        truth = case.reference_value("railing_lower", name, "x0")
        # * At N = 4 000 single scores scatter by at most 0.06 (zoo check).
        assert np.mean(scores) == pytest.approx(truth, abs=0.05)


def test_upper_bounds_get_upper_scores() -> None:
    """A bound above the posterior gives the upper score; the lower one is NaN."""
    case = zoo.railed_gaussians([0.0], side="upper")
    scores = railing_scores(
        case.sample(jax.random.key(1), 4_000).samples[0, 0], None, case.space
    )
    assert float(scores.upper[0]) == pytest.approx(1.0, abs=0.06)
    assert math.isnan(float(scores.lower[0]))
    assert bool(np.all(np.isnan(np.asarray(scores.lower[1:]))))


def test_both_bounds_of_an_interval_are_scored() -> None:
    """A posterior filling a narrow prior rails at both ends."""
    space = ParamSpace.model_validate(
        {"parameters": [{"name": "a", "topology": "interval", "bounds": [-1.0, 1.0]}]}
    )
    values = np.random.default_rng(2).uniform(-1.0, 1.0, size=(5_000, 1))
    scores = railing_scores(values, None, space)
    assert float(scores.lower[0]) > 0.8
    assert float(scores.upper[0]) > 0.8


def test_a_narrow_posterior_far_inside_a_wide_prior_does_not_rail() -> None:
    """The grid follows the data, so a narrow peak is found and the bounds read zero."""
    space = ParamSpace.model_validate(
        {"parameters": [{"name": "a", "topology": "interval", "bounds": [0.0, 1e4]}]}
    )
    values = 5_000.0 + np.random.default_rng(3).normal(size=(5_000, 1))
    scores = railing_scores(values, None, space)
    assert float(scores.lower[0]) == pytest.approx(0.0, abs=1e-12)
    assert float(scores.upper[0]) == pytest.approx(0.0, abs=1e-12)


def test_polar_axes_and_circles_get_no_rows() -> None:
    """Only intervals outside the polar axes of spheres rail."""
    sphere = zoo.von_mises_fisher(0.0, euclidean_count=1)
    assert railed_parameters(sphere.space) == ()
    scores = railing_scores(
        sphere.sample(jax.random.key(4), 500).samples[0, 0], None, sphere.space
    )
    assert bool(np.all(np.isnan(np.asarray(scores.lower))))
    railed = zoo.railed_gaussians([1.0], param_count=3)
    assert railed_parameters(railed.space) == (0,)
