"""Check of σ_sig beyond the resolution of the draws: the tail fit on known nulls."""

import math
from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.scipy.special import gammaincc, ndtr
from jaxtyping import Array

from mixed_signals.sigma.significance import null_significance
from zoo_validation.report import (
    Case,
    Cell,
    Check,
    Claim,
    Context,
    Measurement,
    Section,
    Table,
    claim_cases,
    claim_true,
    grouped,
    role,
)

TAIL_FIT_LABEL = "tail-fit"
TRUE_SIGMAS = (3.0, 4.0, 5.0)
FAR_TAIL_SIGMAS = (10.0, 100.0)
DRAW_COUNTS = (200, 1_000)
FAR_TAIL_DRAWS = 200
MEDIAN_TOLERANCE = 0.15
"""Distance of the median σ_sig from a true 3σ that a claim allows."""
RECORD: dict[int, dict[float, tuple[float, float, float]]] = {
    200: {3.0: (2.9, 2.9, 0.05), 4.0: (3.4, 3.6, 0.05), 5.0: (3.8, 4.1, 0.05)},
    1_000: {3.0: (3.0, 3.0, 0.05), 4.0: (4.05, 4.05, 0.005), 5.0: (4.7, 4.9, 0.05)},
}
"""Medians of the zoo check in ``sigma/significance.py``, by draws and σ.

The lowest and the highest value of the normal and χ² nulls, and half a unit of
the last digit that the record gives.
"""
FAR_TAIL_RECORD = (4.9, 7.7)
"""Single values of the record at 10σ and 100σ against 200 normal draws."""
RECORD_NULLS = (0, 2)
"""Positions in ``NULLS`` of the nulls that the record shares: normal and χ² with 4."""
BISECTION_STEPS = 100
SEARCH_UPPER_BOUND = 400.0
MAXIMUM_ABS_COUNT = 8
"""Absolute standard normals of which the largest value is a null."""


class KnownNull(NamedTuple):
    """A null whose survival function is known, with a sampler in JAX."""

    name: str
    sample: Callable[[Array, tuple[int, ...]], Array]
    survival: Callable[[Array], Array]


def _normal_null() -> KnownNull:
    return KnownNull(
        "standard normal",
        lambda key, shape: jax.random.normal(key, shape),
        lambda value: ndtr(-value),
    )


def _chi_squared_null(degrees: int) -> KnownNull:
    return KnownNull(
        f"chi-squared ({degrees} degree{'s' if degrees > 1 else ''} of freedom)",
        lambda key, shape: jax.random.chisquare(key, float(degrees), shape),
        lambda value: gammaincc(degrees / 2, value / 2),
    )


def _largest_absolute_null() -> KnownNull:
    return KnownNull(
        f"largest of {MAXIMUM_ABS_COUNT} absolute normals",
        lambda key, shape: jnp.max(
            jnp.abs(jax.random.normal(key, (*shape, MAXIMUM_ABS_COUNT))), axis=-1
        ),
        lambda value: 1 - (1 - 2 * ndtr(-value)) ** MAXIMUM_ABS_COUNT,
    )


NULLS = (
    _normal_null(),
    _chi_squared_null(1),
    _chi_squared_null(4),
    _largest_absolute_null(),
)


def value_at_sigma(null: KnownNull, sigma: float) -> float:
    """Return the value with the upper-tail p-value of a normal at ``sigma``."""
    target = ndtr(-jnp.asarray(sigma))

    def step(_: int, bracket: tuple[Array, Array]) -> tuple[Array, Array]:
        lower, upper = bracket
        middle = 0.5 * (lower + upper)
        too_small = null.survival(middle) > target
        return (
            jnp.where(too_small, middle, lower),
            jnp.where(too_small, upper, middle),
        )

    lower, upper = jax.lax.fori_loop(
        0, BISECTION_STEPS, step, (jnp.asarray(0.0), jnp.asarray(SEARCH_UPPER_BOUND))
    )
    return float(0.5 * (lower + upper))


class TailResult(NamedTuple):
    """σ_sig of one value against many replicate nulls."""

    median: Measurement
    lower: float
    upper: float
    extrapolated: int
    infinite: int
    replicates: int


def _tail_result(
    null: KnownNull,
    value: float,
    *,
    draws: int,
    replicates: int,
    key: Array,
) -> TailResult:
    null_draws = null.sample(key, (replicates, draws))
    significance = null_significance(jnp.full(replicates, value), null_draws, "upper")
    sigma = significance.sigma_sig
    finite = jnp.isfinite(sigma)
    lower, middle, upper = (
        float(entry) for entry in jnp.percentile(sigma[finite], jnp.array([16, 50, 84]))
    )
    # * Standard error of a median from the scatter, as a normal sample has it.
    error = 1.2533 * 0.5 * (upper - lower) / math.sqrt(replicates)
    return TailResult(
        Measurement(middle, error, replicates),
        lower,
        upper,
        int(jnp.sum(significance.extrapolated)),
        int(jnp.sum(~finite)),
        replicates,
    )


def _tail_fit(context: Context) -> Section:
    label = TAIL_FIT_LABEL
    replicates = context.count(500, 200)
    rows: list[list[Cell]] = []
    infinite_total = 0
    medians: dict[tuple[int, float, int], Measurement] = {}
    for null_index, null in enumerate(NULLS):
        for sigma in TRUE_SIGMAS:
            value = value_at_sigma(null, sigma)
            for draws in DRAW_COUNTS:
                result = _tail_result(
                    null,
                    value,
                    draws=draws,
                    replicates=replicates,
                    key=context.key(label, "draws", null_index, int(sigma), draws),
                )
                infinite_total += result.infinite
                medians[(null_index, sigma, draws)] = result.median
                rows.append(
                    [
                        null.name,
                        sigma,
                        draws,
                        result.median,
                        result.lower,
                        result.upper,
                        f"{result.extrapolated} of {replicates}",
                        result.infinite,
                    ]
                )
    table = Table.of(
        "σ_sig of a value at a true significance, against the draws of a known"
        " null: the median with its approximate standard error, the 16th and 84th"
        " percentiles over replicates, the rows that the tail fit extrapolates and"
        " the number of infinite values.",
        (
            "Null",
            "True σ",
            "Draws",
            "Median σ_sig",
            "16th percentile",
            "84th percentile",
            "Extrapolated",
            "Infinite",
        ),
        rows,
    )
    claims: list[Claim] = [
        claim_true(
            "No σ_sig is infinite.",
            "0 infinite values",
            f"{infinite_total} infinite values",
            infinite_total == 0,
        ),
        claim_cases(
            "The median σ_sig at 3σ is close to the truth.",
            [
                Case(
                    f"{null.name}, {grouped(draws)} draws",
                    medians[(null_index, 3.0, draws)].value - 3.0,
                    medians[(null_index, 3.0, draws)].margin(),
                    MEDIAN_TOLERANCE,
                )
                for null_index, null in enumerate(NULLS)
                for draws in DRAW_COUNTS
            ],
            tolerance_text=f"{MEDIAN_TOLERANCE}σ",
            replicates=replicates,
            full_only=False,
        ),
    ]
    for null_index in RECORD_NULLS:
        cases = []
        for draws, by_sigma in RECORD.items():
            for sigma, (lowest, highest, rounding) in by_sigma.items():
                median = medians[(null_index, sigma, draws)]
                cases.append(
                    Case(
                        f"{grouped(draws)} draws, {sigma:g}σ",
                        median.value - 0.5 * (lowest + highest),
                        median.margin(),
                        0.5 * (highest - lowest) + rounding,
                    )
                )
        claims.append(
            claim_cases(
                f"The median σ_sig of the {NULLS[null_index].name} reproduces the"
                " record of 12a.",
                cases,
                tolerance_text="the range of the record plus half a unit of its"
                " last digit",
                replicates=replicates,
                full_only=False,
            )
        )
    normal = NULLS[0]
    far_rows: list[list[Cell]] = []
    far_results = []
    for sigma in FAR_TAIL_SIGMAS:
        result = _tail_result(
            normal,
            sigma,
            draws=FAR_TAIL_DRAWS,
            replicates=replicates,
            key=context.key(label, "far", int(sigma)),
        )
        far_results.append(result)
        far_rows.append(
            [sigma, FAR_TAIL_DRAWS, result.median, result.lower, result.upper]
        )
    far_table = Table.of(
        f"σ_sig of values far in the tail against {FAR_TAIL_DRAWS} normal draws.",
        ("True σ", "Draws", "Median σ_sig", "16th percentile", "84th percentile"),
        far_rows,
    )
    claims.append(
        claim_true(
            "The record of 12a at 10σ and 100σ lies within the replicates.",
            "each recorded value lies between the 16th and 84th percentiles",
            "; ".join(
                f"{recorded:g} in [{result.lower:.2f}, {result.upper:.2f}]"
                for recorded, result in zip(FAR_TAIL_RECORD, far_results, strict=True)
            ),
            all(
                result.lower <= recorded <= result.upper
                for recorded, result in zip(FAR_TAIL_RECORD, far_results, strict=True)
            ),
        )
    )
    setup = (
        f"{replicates} replicates of each case. The nulls have a known survival"
        " function: the standard normal, the chi-squared distribution with 1 and"
        " 4 degrees of freedom and the largest of"
        f" {MAXIMUM_ABS_COUNT} absolute standard normals, all drawn with"
        " `jax.random`, with the true p-value from `jax.scipy.special.ndtr` or"
        " `gammaincc`. The value of each case has the p-value of a normal at 3σ, 4σ"
        " and 5σ, and"
        f" {role('func', 'mixed_signals.sigma.significance.null_significance')}"
        f" reads its σ_sig against {DRAW_COUNTS[0]} or {grouped(DRAW_COUNTS[1])}"
        " draws. The record in the zoo check of the module gives medians for a"
        " normal, a Kolmogorov and a chi-squared null with 4 degrees of freedom;"
        " this page reruns the normal and the chi-squared null of the record and"
        " not the Kolmogorov distribution. A claim lets the median of each of"
        " those two nulls lie within the range of the record plus half a unit of"
        " its last digit, plus the standard errors of the median, and a second"
        f" claim lets the median at 3σ of all four nulls lie within"
        f" {MEDIAN_TOLERANCE} of the truth plus the standard errors. The standard"
        " error of a median is 1.25 times the half-width of the 16th to 84th"
        " percentile range over the square root of the replicates. The fit that the"
        " developers rejected, the unconstrained maximum likelihood fit of"
        " {cite:t}`knijnenburg2009fewer`, is not rerun here; its failure (a p-value"
        " of zero in 18 to 79% of the cases beyond 3σ) is in the record of"
        " the module. The fit that the package uses keeps the support above the"
        " value ({cite:t}`peschel2026permapprox`) and uses the estimator of"
        " {cite:t}`zhang2009new`. The record gives single values, 4.9 and 7.7, at"
        " 10σ and 100σ against 200 normal draws; the claim is that each lies"
        " between the 16th and 84th percentiles of the replicates."
    )
    return Section(setup, (table, far_table), tuple(claims))


CHECKS: tuple[Check, ...] = (
    Check(
        label=TAIL_FIT_LABEL,
        title="Tail fit of σ_sig on known nulls",
        session="12a",
        supports=(
            role(
                "func",
                "mixed_signals.sigma.significance.null_significance",
                short=False,
            ),
            role("data", "mixed_signals.sigma.significance.TAIL_FRACTION", short=False),
            role(
                "data", "mixed_signals.sigma.significance.SUPPORT_MARGIN", short=False
            ),
        ),
        run=_tail_fit,
    ),
)
