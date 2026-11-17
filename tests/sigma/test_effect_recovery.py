"""End to end: nulls, null subtraction and the maps recover the zoo's effect sizes.

For each metric the permutation null of every replicate gives its draws through
``null_draws``, ``null_subtract`` removes their mean from the observed value,
and the σ_eff or r_eff map turns the result into an effect size, which the zoo
case knows. Replicates differ by their draws of the case. A replicate keeps its
bandwidth and its directions in every null draw, as an engine does.
"""

import functools
import math
from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp
import pytest
from jaxtyping import PRNGKeyArray

from mixed_signals import zoo
from mixed_signals.core.samples import weighted_covariance
from mixed_signals.metrics.kernel import (
    energy_distance,
    maximum_mean_discrepancy,
    median_bandwidth,
)
from mixed_signals.metrics.marginal import (
    marginal_distances,
    marginal_kde_divergences,
    shared_bandwidth_factor,
)
from mixed_signals.metrics.moments import (
    Terms,
    gaussian_hellinger,
    gaussian_kl,
    gaussian_w2,
    width_ratio,
)
from mixed_signals.metrics.ot.sliced import (
    sliced_directions,
    sliced_wasserstein_distance,
)
from mixed_signals.nulls import null_draws, permutation_null
from mixed_signals.prng import derive_key
from mixed_signals.sigma.maps import null_subtract, r_eff, sigma_eff
from mixed_signals.zoo import ZooCase

SAMPLE_COUNT = 1_000
PARAM_COUNT = 2
REPLICATES = 8
DRAWS = 30
DIRECTION_COUNT = 64
STANDARD_ERRORS = 3.0
ROOT = jax.random.key(2026)
SPACE = zoo.shifted_gaussians(0.0, param_count=PARAM_COUNT).space
# * Silverman's factor for 1 000 samples per set, the factor of the KDE rows.
KDE_FACTOR = float(
    shared_bandwidth_factor(jnp.zeros(SAMPLE_COUNT), jnp.zeros(SAMPLE_COUNT))
)

SHIFT_CASES: dict[str, float] = {"identical": 0.0, "shift_0.5": 0.5, "shift_1.0": 1.0}
SCALE_CASES: dict[str, float] = {"wider": 1.25, "narrower": 0.8}

type Statistic = Callable[[jax.Array, jax.Array, jax.Array, jax.Array], jax.Array]
type Effect = Callable[[jax.Array], jax.Array]
type Prepare = Callable[[jax.Array, jax.Array, PRNGKeyArray], tuple[Statistic, Effect]]


class Row(NamedTuple):
    """A comparison row: its metric, and how to build its statistic and its map."""

    metric: str
    prepare: Prepare


class Outcome(NamedTuple):
    """The mean effect size of the replicates, its standard error and the truth."""

    mean: float
    standard_error: float
    truth: float


def terms_of_moments(
    function: Callable[..., Terms],
    first: jax.Array,
    first_log_weights: jax.Array,
    second: jax.Array,
    second_log_weights: jax.Array,
) -> Terms:
    """Return a Gaussian divergence between the moments of two weighted sample sets."""
    first_mean, first_covariance = weighted_covariance(first, first_log_weights)
    second_mean, second_covariance = weighted_covariance(second, second_log_weights)
    return function(first_mean, first_covariance, second_mean, second_covariance)


def gaussian_statistic(function: Callable[..., Terms], term: str) -> Statistic:
    """Return the statistic that reads one term of a Gaussian divergence."""

    def statistic(
        first: jax.Array,
        first_log_weights: jax.Array,
        second: jax.Array,
        second_log_weights: jax.Array,
    ) -> jax.Array:
        terms = terms_of_moments(
            function, first, first_log_weights, second, second_log_weights
        )
        return jnp.asarray(getattr(terms, term))

    return statistic


def effect_of_sigma(metric: str, **settings: object) -> Effect:
    """Return the σ_eff map of a metric with fixed settings."""

    def effect(subtracted: jax.Array) -> jax.Array:
        return sigma_eff(metric, subtracted, **settings)  # type: ignore[arg-type]

    return effect


def sigma_row(
    metric: str,
    statistic: Statistic,
    **settings: object,
) -> Row:
    """Return a σ_eff row whose statistic and settings do not depend on the data."""
    return Row(metric, lambda *_: (statistic, effect_of_sigma(metric, **settings)))


def marginal_statistic(field: str) -> Statistic:
    """Return the statistic of one field of the marginal distances, first parameter."""

    def statistic(
        first: jax.Array,
        first_log_weights: jax.Array,
        second: jax.Array,
        second_log_weights: jax.Array,
    ) -> jax.Array:
        distances = marginal_distances(
            first, first_log_weights, second, second_log_weights, SPACE
        )
        return jnp.asarray(getattr(distances, field))[0]

    return statistic


def kde_statistic(field: str) -> Statistic:
    """Return the statistic of one field of the KDE divergences, first parameter."""

    def statistic(
        first: jax.Array,
        first_log_weights: jax.Array,
        second: jax.Array,
        second_log_weights: jax.Array,
    ) -> jax.Array:
        divergences = marginal_kde_divergences(
            first,
            first_log_weights,
            second,
            second_log_weights,
            SPACE,
            factor=KDE_FACTOR,
        )
        return jnp.asarray(getattr(divergences, field))[0]

    return statistic


def energy_statistic(
    first: jax.Array,
    first_log_weights: jax.Array,
    second: jax.Array,
    second_log_weights: jax.Array,
) -> jax.Array:
    """Return the energy distance between two weighted sample sets."""
    return energy_distance(
        first, first_log_weights, second, second_log_weights, SPACE, key=ROOT
    )


def prepare_mmd(
    first: jax.Array, second: jax.Array, key: PRNGKeyArray
) -> tuple[Statistic, Effect]:
    """Fix the median bandwidth of the replicate, as an engine does once per row."""
    bandwidth = median_bandwidth(
        first, None, second, None, SPACE, key=derive_key(key, "bandwidth")
    )

    def statistic(
        first_sides: jax.Array,
        first_log_weights: jax.Array,
        second_sides: jax.Array,
        second_log_weights: jax.Array,
    ) -> jax.Array:
        return maximum_mean_discrepancy(
            first_sides,
            first_log_weights,
            second_sides,
            second_log_weights,
            SPACE,
            bandwidth=bandwidth,
            key=ROOT,
        )

    return statistic, effect_of_sigma("mmd", bandwidth=bandwidth, dimension=PARAM_COUNT)


def prepare_sliced(
    first: jax.Array, second: jax.Array, key: PRNGKeyArray
) -> tuple[Statistic, Effect]:
    """Fix the directions of the replicate, as an engine does once per row."""
    directions = sliced_directions(
        SPACE, DIRECTION_COUNT, key=derive_key(key, "directions")
    )

    def statistic(
        first_sides: jax.Array,
        first_log_weights: jax.Array,
        second_sides: jax.Array,
        second_log_weights: jax.Array,
    ) -> jax.Array:
        return sliced_wasserstein_distance(
            first_sides,
            first_log_weights,
            second_sides,
            second_log_weights,
            SPACE,
            directions=directions,
        )

    return statistic, effect_of_sigma("sliced_w2", dimension=PARAM_COUNT)


def width_ratio_statistic(
    first: jax.Array,
    first_log_weights: jax.Array,
    second: jax.Array,
    second_log_weights: jax.Array,
) -> jax.Array:
    """Return the width ratio of the first parameter, second over first."""
    return width_ratio(first, first_log_weights, second, second_log_weights, SPACE)[0]


def prepare_shape_term(metric: str, function: Callable[..., Terms]) -> Prepare:
    """Return the preparation of a shape term, whose direction the data choose."""

    def prepare(
        first: jax.Array, second: jax.Array, key: PRNGKeyArray
    ) -> tuple[Statistic, Effect]:
        zeros = (jnp.zeros(first.shape[0]), jnp.zeros(second.shape[0]))
        log_det_ratio = terms_of_moments(
            gaussian_kl, first, zeros[0], second, zeros[1]
        ).log_det_ratio

        def effect(subtracted: jax.Array) -> jax.Array:
            return r_eff(metric, subtracted, log_det_ratio)

        return gaussian_statistic(function, "shape"), effect

    return prepare


SIGMA_ROWS: dict[str, Row] = {
    "ks": sigma_row("ks", marginal_statistic("ks")),
    "w1": sigma_row("w1", marginal_statistic("w1")),
    "ad": sigma_row("ad", marginal_statistic("ad")),
    "js_kde": sigma_row(
        "js", kde_statistic("js"), estimator="kde", bandwidth=KDE_FACTOR
    ),
    "hellinger_kde": sigma_row(
        "hellinger", kde_statistic("hellinger"), estimator="kde", bandwidth=KDE_FACTOR
    ),
    "kl_gaussian": sigma_row("kl", gaussian_statistic(gaussian_kl, "total")),
    "w2_gaussian": sigma_row("w2", gaussian_statistic(gaussian_w2, "total")),
    "hellinger_gaussian": sigma_row(
        "hellinger", gaussian_statistic(gaussian_hellinger, "total")
    ),
    "energy": sigma_row("energy", energy_statistic, dimension=PARAM_COUNT),
    "mmd": Row("mmd", prepare_mmd),
    "sliced_w2": Row("sliced_w2", prepare_sliced),
}

R_ROWS: dict[str, Row] = {
    "width_ratio": Row(
        "width_ratio",
        lambda *_: (
            width_ratio_statistic,
            lambda subtracted: r_eff("width_ratio", subtracted),
        ),
    ),
    "kl_shape": Row("kl_shape", prepare_shape_term("kl_shape", gaussian_kl)),
    "w2_shape": Row("w2_shape", prepare_shape_term("w2_shape", gaussian_w2)),
    "hellinger_shape": Row(
        "hellinger_shape", prepare_shape_term("hellinger_shape", gaussian_hellinger)
    ),
}


def build_case(name: str) -> ZooCase:
    """Return the zoo case of a name in ``SHIFT_CASES`` or ``SCALE_CASES``."""
    if name in SHIFT_CASES:
        return zoo.shifted_gaussians(SHIFT_CASES[name], param_count=PARAM_COUNT)
    return zoo.scaled_gaussians(SCALE_CASES[name], param_count=PARAM_COUNT, axes=(0,))


@functools.cache
def replicate_sets(name: str) -> jax.Array:
    """Return the samples of every replicate of a case, shaped (B, 2, N, D)."""
    sets = build_case(name).sample(
        derive_key(ROOT, name), SAMPLE_COUNT, batch_size=REPLICATES
    )
    return sets.samples


def effect_of_replicate(row: Row, name: str, replicate: int) -> float:
    """Return the effect size of one replicate, from its value and its null draws."""
    first, second = replicate_sets(name)[replicate]
    key = derive_key(ROOT, name, replicate)
    statistic, effect = row.prepare(first, second, key)
    plan = permutation_null(
        first, None, second, None, key=derive_key(key, "null"), draws=DRAWS
    )
    value = statistic(
        first, jnp.zeros(first.shape[0]), second, jnp.zeros(second.shape[0])
    )
    draws = null_draws(statistic, plan)
    return float(effect(null_subtract(row.metric, value, draws)))


@functools.cache
def outcome(row_name: str, name: str) -> Outcome:
    """Return the recovered effect size of a row on a case, over its replicates."""
    rows = {**SIGMA_ROWS, **R_ROWS}
    effects = jnp.array(
        [
            effect_of_replicate(rows[row_name], name, index)
            for index in range(REPLICATES)
        ]
    )
    assert bool(jnp.all(jnp.isfinite(effects))), (row_name, name, effects)
    truth = SHIFT_CASES.get(name, SCALE_CASES.get(name))
    assert truth is not None
    return Outcome(
        mean=float(jnp.mean(effects)),
        standard_error=float(jnp.std(effects, ddof=1)) / math.sqrt(REPLICATES),
        truth=truth,
    )


def check_recovery(result: Outcome, relative_tolerance: float) -> None:
    """Assert that the mean lies within 3 standard errors plus a share of the truth."""
    allowed = STANDARD_ERRORS * result.standard_error + relative_tolerance * abs(
        result.truth
    )
    assert abs(result.mean - result.truth) <= allowed, result


@pytest.mark.parametrize("row_name", list(SIGMA_ROWS))
def test_identical_gaussians_give_a_signed_effect_consistent_with_zero(
    row_name: str,
) -> None:
    """The subtracted values of identical sets scatter on both sides of zero."""
    result = outcome(row_name, "identical")
    assert abs(result.mean) <= STANDARD_ERRORS * result.standard_error, result


@pytest.mark.parametrize("case_name", ["shift_0.5", "shift_1.0"])
@pytest.mark.parametrize("row_name", list(SIGMA_ROWS))
def test_shifted_gaussians_recover_their_shift(row_name: str, case_name: str) -> None:
    """σ_eff of a 0.5σ or 1σ shift is within 3 standard errors plus 3% (6% for KS)."""
    check_recovery(outcome(row_name, case_name), 0.06 if row_name == "ks" else 0.03)


@pytest.mark.parametrize("case_name", ["wider", "narrower"])
@pytest.mark.parametrize("row_name", list(R_ROWS))
def test_scaled_gaussians_recover_their_width_ratio(
    row_name: str, case_name: str
) -> None:
    """r_eff of one axis 1.25 times wider or 0.8 times narrower is recovered."""
    check_recovery(outcome(row_name, case_name), 0.03)
