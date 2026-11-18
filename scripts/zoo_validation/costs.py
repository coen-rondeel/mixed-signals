"""Check of the costs: seconds per evaluation of the metrics and the diagnostics."""

import functools
import statistics
import time
from collections.abc import Callable
from typing import NamedTuple

import jax

from mixed_signals import zoo
from mixed_signals._config import SAMPLE_CAP
from mixed_signals.diagnostics import PAIR_CAP, dependence_statistics
from mixed_signals.metrics.divergence import (
    knn_hellinger_distance,
    knn_js_divergence,
    knn_kl_divergence,
)
from mixed_signals.metrics.kernel import (
    energy_distance,
    maximum_mean_discrepancy,
    median_bandwidth,
)
from mixed_signals.metrics.ot.sinkhorn import (
    SINKHORN_EPSILON_FACTOR,
    sinkhorn_divergence,
)
from mixed_signals.metrics.ot.sliced import (
    SLICED_DIRECTIONS,
    SLICED_SINKHORN_CAP,
    SLICED_SINKHORN_DIRECTIONS,
    sliced_directions,
    sliced_sinkhorn_divergence,
    sliced_sinkhorn_epsilon,
    sliced_wasserstein_distance,
)
from mixed_signals.sigma.calibration import (
    KNN_KL_REPLICATES,
    KNN_KL_SHIFTS,
    knn_kl_curve,
)
from zoo_validation.report import (
    Cell,
    Check,
    Context,
    Section,
    Table,
    grouped,
    role,
)

COSTS_LABEL = "costs"
PARAM_COUNT = 16
ENERGY_SIZES = (5_000, 10_000, 20_000)
SLOW_SECONDS = 10.0
"""A call slower than this is timed once."""
TIMED_CALLS = 3
NULL_DRAWS = 200
"""Draws of the null that a comparison row runs, by default."""


class _Sizes(NamedTuple):
    """The sample counts of the rows: the caps of the package in a full run."""

    energy: tuple[int, ...]
    cap: int
    sliced_sinkhorn: int
    pair: int


class _Timing(NamedTuple):
    """The time of one evaluation and how it was measured."""

    seconds: float
    calls: int
    """Calls that the time rests on: the median of three, or one slow call."""


def _wait(result: object) -> None:
    """Wait until every array of a result is ready."""
    for leaf in jax.tree.leaves(result):
        if isinstance(leaf, jax.Array):
            leaf.block_until_ready()


def _time_call(function: Callable[[], object]) -> _Timing:
    """Time one evaluation after a warm-up call that compiles.

    A call that takes more than ten seconds stands alone; otherwise the time is
    the median of three calls. JAX dispatches asynchronously, so each timer
    starts after all earlier work has finished and stops after the result has.
    """
    _wait(function())
    durations = []
    for _ in range(TIMED_CALLS):
        started = time.perf_counter()
        _wait(function())
        durations.append(time.perf_counter() - started)
        if durations[0] > SLOW_SECONDS:
            return _Timing(durations[0], 1)
    return _Timing(statistics.median(durations), TIMED_CALLS)


class _Row(NamedTuple):
    """A row of the table: the call, its size, its record and whether it is a draw."""

    name: str
    samples: int
    call: Callable[[], object]
    record: str
    per_draw: bool
    """Whether a comparison repeats the call for every null draw."""


def _costs(context: Context) -> Section:
    label = COSTS_LABEL
    sizes = (
        _Sizes(ENERGY_SIZES, SAMPLE_CAP, SLICED_SINKHORN_CAP, PAIR_CAP)
        if not context.quick
        else _Sizes((500, 1_000, 2_000), 1_000, 500, 500)
    )
    largest = max(*sizes.energy, sizes.cap)
    case = zoo.shifted_gaussians(1.0, param_count=PARAM_COUNT)
    space = case.space
    sets = case.sample(context.key(label, "sets"), largest).samples
    first, second = sets[0, 0], sets[0, 1]
    _wait((first, second))
    key = context.key(label, "subsample")
    capped = (first[: sizes.cap], second[: sizes.cap])
    bandwidth = median_bandwidth(capped[0], None, capped[1], None, space, key=key)
    directions = sliced_directions(
        space, SLICED_DIRECTIONS, key=context.key(label, "directions")
    )
    small = (first[: sizes.sliced_sinkhorn], second[: sizes.sliced_sinkhorn])
    small_directions = sliced_directions(
        space, SLICED_SINKHORN_DIRECTIONS, key=context.key(label, "small-directions")
    )
    epsilon = sliced_sinkhorn_epsilon(
        small[0], None, small[1], None, space, directions=small_directions, key=key
    )
    _wait((bandwidth, directions, small_directions, epsilon))
    rows: list[_Row] = []
    record_energy = {5_000: "0.06 s", 10_000: "0.48 s", 20_000: "2.0 s"}
    for count in sizes.energy:
        rows.append(
            _Row(
                "energy distance",
                count,
                functools.partial(
                    energy_distance,
                    first[:count],
                    None,
                    second[:count],
                    None,
                    space,
                    key=key,
                    cap=count,
                ),
                record_energy.get(count, "-"),
                True,
            )
        )
    rows.append(
        _Row(
            "median bandwidth",
            sizes.cap,
            functools.partial(
                median_bandwidth, capped[0], None, capped[1], None, space, key=key
            ),
            "0.3 s",
            False,
        )
    )
    rows.append(
        _Row(
            "MMD squared",
            sizes.cap,
            functools.partial(
                maximum_mean_discrepancy,
                capped[0],
                None,
                capped[1],
                None,
                space,
                bandwidth=bandwidth,
                key=key,
                cap=sizes.cap,
            ),
            "0.3 s",
            True,
        )
    )
    for name, divergence, record in (
        ("kNN KL", knn_kl_divergence, "2.7 s"),
        ("kNN Hellinger", knn_hellinger_distance, "5.3 s"),
        ("kNN JS", knn_js_divergence, "9.5 s"),
    ):
        rows.append(
            _Row(
                name,
                sizes.cap,
                functools.partial(
                    divergence,
                    capped[0],
                    None,
                    capped[1],
                    None,
                    space,
                    key=key,
                    cap=sizes.cap,
                ),
                record,
                True,
            )
        )
    rows.append(
        _Row(
            "Sinkhorn divergence",
            sizes.cap,
            functools.partial(
                sinkhorn_divergence,
                capped[0],
                None,
                capped[1],
                None,
                space,
                epsilon=SINKHORN_EPSILON_FACTOR * bandwidth**2,
                key=key,
                cap=sizes.cap,
            ),
            "6.5 s",
            True,
        )
    )
    rows.append(
        _Row(
            f"sliced W2, directions: {SLICED_DIRECTIONS} asked,"
            f" {directions.shape[0]} used",
            sizes.cap,
            functools.partial(
                sliced_wasserstein_distance,
                capped[0],
                None,
                capped[1],
                None,
                space,
                directions=directions,
            ),
            "0.5 s",
            True,
        )
    )
    rows.append(
        _Row(
            "sliced Sinkhorn divergence, directions:"
            f" {SLICED_SINKHORN_DIRECTIONS} asked, {small_directions.shape[0]} used",
            sizes.sliced_sinkhorn,
            functools.partial(
                sliced_sinkhorn_divergence,
                small[0],
                None,
                small[1],
                None,
                space,
                directions=small_directions,
                epsilon=epsilon,
                key=key,
                cap=sizes.sliced_sinkhorn,
            ),
            "6.4 s",
            True,
        )
    )
    rows.append(
        _Row(
            "kNN KL calibration curve",
            sizes.cap,
            functools.partial(
                knn_kl_curve,
                PARAM_COUNT,
                sizes.cap,
                sizes.cap,
                key=key,
                cap=sizes.cap,
                shifts=KNN_KL_SHIFTS if not context.quick else (0.5, 1.0),
                replicates=KNN_KL_REPLICATES if not context.quick else 1,
            ),
            "-",
            False,
        )
    )
    rows.append(
        _Row(
            "pairwise dependence statistics",
            sizes.pair,
            functools.partial(
                dependence_statistics,
                first[: sizes.pair],
                None,
                space,
                key=key,
                cap=sizes.pair,
            ),
            "about 20 min for 201 sets",
            True,
        )
    )
    table_rows: list[list[Cell]] = []
    for row in rows:
        timing = _time_call(row.call)
        extrapolated = (
            (NULL_DRAWS + 1) * timing.seconds / 60.0 if row.per_draw else None
        )
        table_rows.append(
            [
                row.name,
                row.samples,
                timing.seconds,
                "median of 3" if timing.calls == TIMED_CALLS else "one call",
                extrapolated,
                row.record,
            ]
        )
    table = Table.of(
        f"Wall time of one evaluation in {PARAM_COUNT} parameters, after one warm-up"
        " call that compiles, and the time of 201 calls, which is a comparison"
        f" with {NULL_DRAWS} null draws and the observed value. A dash marks a row"
        " that a comparison evaluates once.",
        (
            "Row",
            "Samples per set",
            "Seconds per call",
            "Timing",
            "201 calls (minutes)",
            "Record",
        ),
        table_rows,
    )
    shifted_role = role("func", "mixed_signals.zoo.cases.shifted_gaussians")
    setup = (
        "Zoo Gaussians in"
        f" {PARAM_COUNT} parameters ({shifted_role}, shift 1). Each row calls the"
        " function of the package once as an engine would, after one call that"
        " compiles it, and reports the median of three calls, or the one call when"
        f" it takes more than {SLOW_SECONDS:g} s."
        " Before a timer starts, all earlier work has finished, and the timer stops"
        " when the result is ready, because JAX dispatches asynchronously. The"
        f" energy distance runs at {', '.join(grouped(size) for size in sizes.energy)}"
        " samples per set, which sets"
        f" {role('data', 'mixed_signals._config.SAMPLE_CAP')}, and the other"
        f" comparison rows at {grouped(sizes.cap)}; the sliced Sinkhorn divergence"
        f" runs at {grouped(sizes.sliced_sinkhorn)}"
        f" ({role('data', 'mixed_signals.metrics.ot.sliced.SLICED_SINKHORN_CAP')}),"
        f" and the dependence statistics at {grouped(sizes.pair)}"
        f" ({role('data', 'mixed_signals.diagnostics.dependence.PAIR_CAP')}), with the"
        " default directions of each sliced metric. In"
        f" {PARAM_COUNT} dimensions the orthonormal bases round the number of"
        " directions asked for up to whole bases, so the rows state how many were"
        " asked for and how many the array holds. The record of"
        " 6.4 s for the sliced Sinkhorn divergence was measured with 20"
        " independent directions. The median bandwidth and the"
        " calibration curve of the kNN KL map run once per comparison, so they have"
        " no extrapolation. The times depend on the machine in the stamp of this"
        " page; the record column holds the times that the sessions measured on an"
        " Apple M4 Pro CPU, 20 minutes for the dependence statistics of 201 sets"
        " being about 6 s per set. No claim reads them."
    )
    return Section(setup, (table,), ())


CHECKS: tuple[Check, ...] = (
    Check(
        label=COSTS_LABEL,
        title="Cost of the metrics and diagnostics",
        session="8, 9, 10 and 12",
        supports=(
            role("data", "mixed_signals._config.SAMPLE_CAP", short=False),
            role("data", "mixed_signals.diagnostics.dependence.PAIR_CAP", short=False),
            role(
                "data",
                "mixed_signals.metrics.ot.sliced.SLICED_SINKHORN_CAP",
                short=False,
            ),
            "the cost weights of the dry run in session 14",
        ),
        run=_costs,
    ),
)
"""Checks of the costs."""
