"""Tests for the flag rule, and the zoo cases that the diagnostics must flag."""

import math
import warnings
from collections.abc import Callable
from typing import cast

import jax
import numpy as np
import pytest

from mixed_signals import zoo
from mixed_signals.core.space import ParamSpace
from mixed_signals.diagnostics._common import parameter_pairs
from mixed_signals.diagnostics.dependence import dependence_statistics
from mixed_signals.diagnostics.flags import THRESHOLDS, evaluate
from mixed_signals.diagnostics.modes import (
    circle_centres,
    discriminant_directions,
    gmm_bic,
    marginal_modes,
    projection_modes,
)
from mixed_signals.diagnostics.railing import railed_parameters, railing_scores
from mixed_signals.diagnostics.shape import (
    anderson_darling_normality,
    mardia_moments,
    marginal_moments,
    moment_metric_names,
)
from mixed_signals.errors import MixedSignalsWarning
from mixed_signals.nulls.surrogates import gaussian_surrogates

DRAWS = 80
SAMPLES = 2_000


# * The flag rule


def test_effects_pass_their_side_of_the_threshold() -> None:
    """Rows flag above, in absolute value, or below their threshold."""
    assert evaluate("gmm_bic", 12.0).flagged
    assert not evaluate("gmm_bic", 8.0).flagged
    assert evaluate("ess", 120.0).flagged
    assert not evaluate("ess", 4_000.0).flagged
    assert not evaluate("railing_lower", math.nan).flagged


def test_null_rows_subtract_the_null_mean_and_need_sigma_sig() -> None:
    """The effect is the value minus the null mean; σ_sig must reach the level."""
    draws = np.linspace(-0.05, 0.05, 199)
    flagged = evaluate("skewness", -0.8, draws)
    assert flagged.effect == pytest.approx(-0.8)
    assert flagged.p_value == pytest.approx(1 / 200)
    assert flagged.flagged
    wide = np.linspace(-2.0, 2.0, 199)
    unresolved = evaluate("skewness", 0.8, wide)
    assert abs(unresolved.effect) >= THRESHOLDS["skewness"].value
    assert not unresolved.flagged


def test_rows_with_their_own_p_value_use_it() -> None:
    """The split-chain tests bring an asymptotic p-value."""
    assert evaluate("split_chain_ks", 0.2, p_value=1e-4).flagged
    assert not evaluate("split_chain_ks", 0.2, p_value=0.3).flagged
    assert not evaluate("split_chain_ks", 0.05, p_value=1e-9).flagged


def test_informational_rows_have_no_threshold() -> None:
    """Pearson's r and the rates never flag."""
    for metric in ("pearson", "mode_count", "tau", "rhat", "swap_rate"):
        assert metric not in THRESHOLDS
        with pytest.raises(KeyError):
            evaluate(metric, 1.0)


# * Zoo cases


def null_of[Result](
    statistic: Callable[[jax.Array], Result],
    samples: jax.Array,
    space: ParamSpace,
    key: jax.Array,
    *,
    truncate: bool = False,
) -> Result:
    """Return the statistic of Gaussian surrogate sets of the samples' size."""
    surrogates = gaussian_surrogates(
        samples,
        None,
        space,
        size=samples.shape[0],
        draws=DRAWS,
        key=key,
        truncate=truncate,
    )
    return cast(Result, jax.lax.map(statistic, surrogates))


def flags_of(case: zoo.ZooCase, posterior: str, seed: int) -> set[tuple[str, str]]:
    """Run every diagnostic of one zoo posterior and return the rows that flag."""
    space = case.space
    names = space.names
    samples = case.sample(jax.random.key(seed), SAMPLES).samples[
        0, case.names.index(posterior)
    ]
    key = jax.random.key(seed + 1)
    flagged: set[tuple[str, str]] = set()

    def record(
        metric: str,
        label: str,
        value: jax.Array | float,
        draws: jax.Array | None = None,
    ) -> None:
        if evaluate(metric, float(value), draws).flagged:
            flagged.add((metric, label))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MixedSignalsWarning)
        moments = marginal_moments(samples, None, space)
        moment_null = null_of(
            lambda values: marginal_moments(values, None, space), samples, space, key
        )
        distances = anderson_darling_normality(samples, None, space)
        distance_null = null_of(
            lambda values: anderson_darling_normality(values, None, space),
            samples,
            space,
            key,
        )
        for position, (skew_name, kurt_name) in enumerate(moment_metric_names(space)):
            record(
                skew_name,
                names[position],
                moments.skewness[position],
                moment_null.skewness[:, position],
            )
            record(
                kurt_name,
                names[position],
                moments.kurtosis[position],
                moment_null.kurtosis[:, position],
            )
            record(
                "anderson_darling",
                names[position],
                distances[position],
                distance_null[:, position],
            )
        mardia = mardia_moments(samples, None, space, key=key)
        mardia_null = null_of(
            lambda values: mardia_moments(values, None, space, key=key),
            samples,
            space,
            key,
        )
        record("mardia_skewness", "all", mardia.skewness, mardia_null.skewness)
        record("mardia_kurtosis", "all", mardia.kurtosis, mardia_null.kurtosis)
        dependence = dependence_statistics(samples, None, space, key=key)
        dependence_null = null_of(
            lambda values: dependence_statistics(values, None, space, key=key),
            samples,
            space,
            key,
        )
        for index, (first, second) in enumerate(parameter_pairs(len(space))):
            label = f"{names[first]}-{names[second]}"
            for metric in ("spearman", "distance_correlation", "mutual_information"):
                record(
                    metric,
                    label,
                    getattr(dependence, metric)[index],
                    getattr(dependence_null, metric)[:, index],
                )
        modes = marginal_modes(samples, None, space)
        modes_null = null_of(
            lambda values: marginal_modes(values, None, space),
            samples,
            space,
            key,
            truncate=True,
        )
        for position in range(len(space)):
            record(
                "mode_dip",
                names[position],
                modes.dip[position],
                modes_null.dip[:, position],
            )
        structure = gmm_bic(
            samples, None, space, effective_sample_size=float(SAMPLES), key=key
        )
        record("gmm_bic", "all", structure.delta_bic)
        directions = discriminant_directions(structure.mixture)
        centres = circle_centres(samples, None, space)
        if directions.shape[0]:
            projected = projection_modes(
                samples, None, space, directions=directions, centres=centres
            )
            projected_null = null_of(
                lambda values: projection_modes(
                    values, None, space, directions=directions, centres=centres
                ),
                samples,
                space,
                key,
            )
            for index in range(directions.shape[0]):
                record(
                    "mode_dip",
                    f"direction {index}",
                    projected.dip[index],
                    projected_null.dip[:, index],
                )
        scores = railing_scores(samples, None, space)
        for position in railed_parameters(space):
            for metric, values in (
                ("railing_lower", scores.lower),
                ("railing_upper", scores.upper),
            ):
                if not math.isnan(float(values[position])):
                    record(metric, names[position], values[position])
    return flagged


def correlated_gaussian() -> zoo.ZooCase:
    """Return a correlated Gaussian in three parameters."""
    covariance = np.array([[1.0, 0.8, 0.0], [0.8, 1.0, -0.3], [0.0, -0.3, 1.0]])
    return zoo.gaussians(np.zeros((1, 3)), covariance[None], names=["gaussian"])


@pytest.mark.parametrize(
    ("case", "posterior"),
    [
        (correlated_gaussian(), "gaussian"),
        (
            zoo.wrapped_gaussians(0.0, width=0.5, param_count=2, correlation=0.5),
            "reference",
        ),
        (zoo.railed_gaussians([4.0]), "railed_0"),
    ],
)
def test_gaussians_are_not_flagged(case: zoo.ZooCase, posterior: str) -> None:
    """Gaussian posteriors, also on a circle or far from a bound, raise no flag."""
    assert flags_of(case, posterior, 0) == set()


def test_bananas_are_flagged_for_dependence_and_shape_not_modes() -> None:
    """The zoo's banana flags its curved dependence and shape, and keeps one mode."""
    flagged = flags_of(zoo.banana(), "banana", 1)
    for row in (
        ("distance_correlation", "x0-x1"),
        ("mutual_information", "x0-x1"),
        ("mardia_skewness", "all"),
        ("skewness", "x1"),
        ("gmm_bic", "all"),
    ):
        assert row in flagged
    assert not any(metric == "mode_dip" for metric, _ in flagged)
    assert ("spearman", "x0-x1") not in flagged


def test_separated_mixtures_are_flagged_for_their_modes() -> None:
    """A mixture at separation 4 flags two modes along its first parameter."""
    flagged = flags_of(zoo.gaussian_mixture(4.0), "mixture", 2)
    assert ("mode_dip", "x0") in flagged
    assert ("kurtosis", "x0") in flagged
    assert ("gmm_bic", "all") in flagged
    assert not any(
        metric in ("distance_correlation", "mutual_information")
        for metric, _ in flagged
    )


@pytest.mark.parametrize("offset", [0.0, 1.0])
def test_railed_gaussians_are_flagged_at_their_bound(offset: float) -> None:
    """A normal railed within 1σ of its bound flags railing and skewness."""
    flagged = flags_of(zoo.railed_gaussians([offset]), "railed_0", 3)
    assert ("railing_lower", "x0") in flagged
    assert ("skewness", "x0") in flagged
    assert not any(metric == "mode_dip" for metric, _ in flagged)
