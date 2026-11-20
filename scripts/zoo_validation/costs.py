"""Check of the costs: seconds per evaluation of the metrics and the diagnostics."""

import functools
import statistics
import time
from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray

from mixed_signals import zoo
from mixed_signals._config import SAMPLE_CAP
from mixed_signals.core.space import ParamSpace
from mixed_signals.diagnostics import (
    PAIR_CAP,
    anderson_darling_normality,
    dependence_statistics,
    gmm_bic,
    mardia_moments,
    marginal_modes,
    marginal_moments,
    projection_modes,
)
from mixed_signals.engines.cost import _FITS, _evaluation_seconds
from mixed_signals.estimators.gaussian import GaussianDensity
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
from mixed_signals.metrics.marginal import marginal_distances, marginal_kde_divergences
from mixed_signals.metrics.moments import (
    gaussian_hellinger,
    gaussian_kl,
    gaussian_w2,
    mean_shift,
    width_ratio,
)
from mixed_signals.metrics.ot.sinkhorn import (
    SINKHORN_EPSILON_FACTOR,
    sinkhorn_divergence,
)
from mixed_signals.metrics.ot.sliced import (
    SLICED_DIRECTIONS,
    SLICED_SINKHORN_CAP,
    SLICED_SINKHORN_DIRECTIONS,
    gaussian_sliced_wasserstein_distance,
    sliced_directions,
    sliced_sinkhorn_divergence,
    sliced_sinkhorn_epsilon,
    sliced_wasserstein_distance,
)
from mixed_signals.prng import derive_key
from mixed_signals.sigma.calibration import (
    KNN_KL_REPLICATES,
    KNN_KL_SHIFTS,
    knn_kl_curve,
)
from zoo_validation.report import (
    Cell,
    Check,
    Claim,
    Context,
    Section,
    Table,
    claim_true,
    grouped,
    role,
)

COSTS_LABEL = "costs"
PARAM_COUNT = 16
ENERGY_SIZES = (5_000, 10_000, 20_000)
SLOW_SECONDS = 10.0
"""A call slower than this is timed once."""
TIMED_CALLS = 3
GRID_SLOW_SECONDS = 3.0
"""A call of the grid slower than this is timed once, to keep the build short."""
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


def _time_call(
    function: Callable[[], object], slow_seconds: float = SLOW_SECONDS
) -> _Timing:
    """Time one evaluation after a warm-up call that compiles.

    A call that takes more than ``slow_seconds`` stands alone; otherwise the
    time is the median of three calls. JAX dispatches asynchronously, so each
    timer starts after all earlier work has finished and stops after the result
    has.
    """
    _wait(function())
    durations = []
    for _ in range(TIMED_CALLS):
        started = time.perf_counter()
        _wait(function())
        durations.append(time.perf_counter() - started)
        if durations[0] > slow_seconds:
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


# * The grid on which the cost model of the engines is checked.

GRID_POINTS = ((3_000, 16), (10_000, 2), (10_000, 8), (10_000, 16))
"""Samples per set and number of parameters of the full grid."""
QUICK_GRID = ((500, 4),)
QUICK_STATISTICS = (
    "moments",
    "marginal",
    "gaussian",
    "energy",
    "sliced_w2",
    "marginal_moments",
)
"""The statistics of the quick run, which skips the slow ones."""
MODEL_FACTOR = 1.5
"""The factor within which the model must predict each checked timing."""
COSTLY_SECONDS = 0.01
"""A timing of at least this, of a per-draw or per-surrogate statistic, is checked."""
ONCE_STATISTICS = ("median_bandwidth", "sliced_sinkhorn_epsilon", "gmm_bic")
"""Statistics that run once per comparison or posterior, not per draw or surrogate."""
EXISTING_ROWS = (
    ("energy distance", "energy"),
    ("median bandwidth", "median_bandwidth"),
    ("MMD squared", "mmd"),
    ("kNN KL", "knn_kl"),
    ("kNN Hellinger", "knn_hellinger"),
    ("kNN JS", "knn_js"),
    ("Sinkhorn divergence", "sinkhorn"),
    ("sliced W2", "sliced_w2"),
    ("sliced Sinkhorn divergence", "sliced_sinkhorn"),
    ("pairwise dependence statistics", "dependence"),
)
"""Rows of the table above, by name, that time a statistic of the model."""
KDE_FACTOR = 0.2
"""The bandwidth factor of the KDE statistics in the timings."""
PROJECTION_DIRECTIONS = 3
"""Directions of the mode projections in the timings; the model counts them as units."""


class _Timed(NamedTuple):
    """A statistic of the cost model, ready to call, with its work units."""

    statistic: str
    """Key of the constants in the model."""
    call: Callable[[], object]
    units: float


def _gaussian_terms(
    first: Array,
    first_weights: Array,
    second: Array,
    second_weights: Array,
    *,
    space: ParamSpace,
) -> tuple[object, ...]:
    """Return the Gaussian terms: both KL directions, W2 and the Hellinger distance."""
    one = GaussianDensity.fit(first, first_weights, space)
    two = GaussianDensity.fit(second, second_weights, space)
    return (
        gaussian_kl(one.mean, one.covariance, two.mean, two.covariance, space),
        gaussian_kl(two.mean, two.covariance, one.mean, one.covariance, space),
        gaussian_w2(one.mean, one.covariance, two.mean, two.covariance, space),
        gaussian_hellinger(one.mean, one.covariance, two.mean, two.covariance, space),
    )


def _moments(
    first: Array,
    first_weights: Array,
    second: Array,
    second_weights: Array,
    *,
    space: ParamSpace,
) -> tuple[Array, Array]:
    """Return the mean shift and the width ratio of every parameter."""
    return (
        mean_shift(first, first_weights, second, second_weights, space),
        width_ratio(first, first_weights, second, second_weights, space),
    )


def _grid_calls(
    first: Array, second: Array, space: ParamSpace, key: PRNGKeyArray
) -> list[_Timed]:
    """Return every statistic of the model as one call on a pair of sets.

    The per-draw statistics are jitted, as they are inside the loop of the null
    draws; the GMM-BIC fit runs once per posterior and is not jitted.
    """
    count, dimension = int(first.shape[0]), int(first.shape[1])
    weights = jnp.zeros(count)
    bandwidth = median_bandwidth(first, None, second, None, space, key=key)
    directions = sliced_directions(
        space, SLICED_DIRECTIONS, key=derive_key(key, "directions")
    )
    small_directions = sliced_directions(
        space, SLICED_SINKHORN_DIRECTIONS, key=derive_key(key, "small-directions")
    )
    epsilon = sliced_sinkhorn_epsilon(
        first, None, second, None, space, directions=small_directions, key=key
    )
    mode_directions = jnp.eye(dimension)[: min(PROJECTION_DIRECTIONS, dimension)]
    pair_statistics: dict[str, tuple[Callable[..., object], float]] = {
        "moments": (functools.partial(_moments, space=space), float(dimension)),
        "marginal": (
            functools.partial(marginal_distances, space=space),
            float(dimension),
        ),
        "kde": (
            functools.partial(marginal_kde_divergences, space=space, factor=KDE_FACTOR),
            float(dimension),
        ),
        "gaussian": (functools.partial(_gaussian_terms, space=space), 1.0),
        "knn_kl": (functools.partial(knn_kl_divergence, space=space, key=key), 1.0),
        "knn_js": (functools.partial(knn_js_divergence, space=space, key=key), 1.0),
        "knn_hellinger": (
            functools.partial(knn_hellinger_distance, space=space, key=key),
            1.0,
        ),
        "mmd": (
            functools.partial(
                maximum_mean_discrepancy, space=space, bandwidth=bandwidth, key=key
            ),
            1.0,
        ),
        "energy": (functools.partial(energy_distance, space=space, key=key), 1.0),
        "sinkhorn": (
            functools.partial(
                sinkhorn_divergence,
                space=space,
                epsilon=SINKHORN_EPSILON_FACTOR * bandwidth**2,
                key=key,
            ),
            1.0,
        ),
        "sliced_w2": (
            functools.partial(
                sliced_wasserstein_distance, space=space, directions=directions
            ),
            1.0,
        ),
        "sliced_w2_gaussian": (
            functools.partial(
                gaussian_sliced_wasserstein_distance,
                space=space,
                directions=directions,
            ),
            1.0,
        ),
        "sliced_sinkhorn": (
            functools.partial(
                sliced_sinkhorn_divergence,
                space=space,
                directions=small_directions,
                epsilon=epsilon,
                key=key,
            ),
            float(small_directions.shape[0]),
        ),
    }
    posterior_statistics: dict[str, tuple[Callable[..., object], float]] = {
        "marginal_moments": (
            functools.partial(marginal_moments, space=space),
            float(dimension),
        ),
        "anderson_darling": (
            functools.partial(anderson_darling_normality, space=space),
            float(dimension),
        ),
        "mardia": (functools.partial(mardia_moments, space=space, key=key), 1.0),
        "dependence": (
            functools.partial(dependence_statistics, space=space, key=key),
            float(dimension * (dimension - 1) // 2),
        ),
        "marginal_modes": (
            functools.partial(marginal_modes, space=space),
            float(dimension),
        ),
        "projection_modes": (
            functools.partial(
                projection_modes,
                space=space,
                directions=mode_directions,
                centres=jnp.zeros(0),
            ),
            float(mode_directions.shape[0]),
        ),
    }
    timed = [
        _Timed(
            name,
            functools.partial(jax.jit(function), first, weights, second, weights),
            units,
        )
        for name, (function, units) in pair_statistics.items()
    ]
    timed.extend(
        _Timed(name, functools.partial(jax.jit(function), first, weights), units)
        for name, (function, units) in posterior_statistics.items()
    )
    timed.append(
        _Timed(
            "median_bandwidth",
            functools.partial(
                jax.jit(functools.partial(median_bandwidth, space=space, key=key)),
                first,
                None,
                second,
                None,
            ),
            1.0,
        )
    )
    timed.append(
        _Timed(
            "sliced_sinkhorn_epsilon",
            functools.partial(
                jax.jit(
                    functools.partial(
                        sliced_sinkhorn_epsilon,
                        space=space,
                        directions=small_directions,
                        key=key,
                    )
                ),
                first,
                None,
                second,
                None,
            ),
            float(small_directions.shape[0]),
        )
    )
    timed.append(
        _Timed(
            "gmm_bic",
            functools.partial(
                gmm_bic,
                first,
                None,
                space,
                effective_sample_size=float(count),
                key=key,
            ),
            1.0,
        )
    )
    return timed


class _Prediction(NamedTuple):
    """The model and the timing of one statistic at one grid point."""

    statistic: str
    samples: int
    dimension: int
    units: float
    measured: float
    modelled: float

    @property
    def factor(self) -> float:
        """The larger of model over timing and timing over model."""
        ratio = self.modelled / self.measured
        return max(ratio, 1 / ratio)

    @property
    def checked(self) -> bool:
        """Whether the statistic runs per draw and the timing is costly."""
        return self.statistic not in ONCE_STATISTICS and (
            self.measured >= COSTLY_SECONDS
        )


def _grid_predictions(
    context: Context, existing: dict[str, float], statistics_of_quick: tuple[str, ...]
) -> list[_Prediction]:
    """Time the statistics on the grid and predict each with the model."""
    caps = {
        "cap": SAMPLE_CAP,
        "pair_cap": PAIR_CAP,
        "sliced_sinkhorn_cap": SLICED_SINKHORN_CAP,
    }
    points = QUICK_GRID if context.quick else GRID_POINTS
    predictions = []
    for count, dimension in points:
        case = zoo.shifted_gaussians(1.0, param_count=dimension)
        key = context.key(COSTS_LABEL, "grid", count, dimension)
        sets = case.sample(key, count).samples
        first, second = sets[0, 0], sets[0, 1]
        _wait((first, second))
        # * Statistics already timed at the largest point need no second timing.
        reused = existing if (count, dimension) == GRID_POINTS[-1] else {}
        for timed in _grid_calls(first, second, case.space, key):
            if context.quick and timed.statistic not in statistics_of_quick:
                continue
            seconds = reused.get(timed.statistic)
            if seconds is None:
                seconds = _time_call(timed.call, GRID_SLOW_SECONDS).seconds
            predictions.append(
                _Prediction(
                    timed.statistic,
                    count,
                    dimension,
                    timed.units,
                    seconds,
                    _evaluation_seconds(
                        _FITS[timed.statistic], count, dimension, timed.units, caps
                    ),
                )
            )
    return predictions


def _grid_section(predictions: list[_Prediction]) -> tuple[Table, Claim]:
    """Return the table of the grid and the claim that the model predicts it."""
    rows: list[list[Cell]] = [
        [
            prediction.statistic,
            prediction.samples,
            prediction.dimension,
            prediction.units,
            prediction.measured,
            prediction.modelled,
            prediction.factor,
            prediction.checked,
        ]
        for prediction in predictions
    ]
    table = Table.of(
        "The model of the dry run against the timings of every statistic of the model:"
        " the work units, the seconds of one jitted evaluation, the model, and the"
        " factor between them.",
        (
            "Statistic",
            "Samples per set",
            "Parameters",
            "Work units",
            "Measured (s)",
            "Model (s)",
            "Factor",
            "Checked",
        ),
        rows,
    )
    checked = [prediction for prediction in predictions if prediction.checked]
    if checked:
        worst = max(checked, key=lambda prediction: prediction.factor)
        observed = (
            f"worst of {len(checked)} checked rows of {len(predictions)}:"
            f" {worst.statistic} at {grouped(worst.samples)} samples and"
            f" {worst.dimension} parameters, factor {worst.factor:.2f}"
            f" (model {worst.modelled:.3g} s, measured {worst.measured:.3g} s)"
        )
    else:
        observed = f"no checked row of {len(predictions)}"
    claim = claim_true(
        "The model of the dry run predicts each costly timing within a factor of"
        f" {MODEL_FACTOR:g}.",
        f"every timing of at least {COSTLY_SECONDS * 1e3:g} ms of a statistic run per"
        f" null draw or surrogate within a factor of {MODEL_FACTOR:g} of the model",
        observed,
        all(prediction.factor <= MODEL_FACTOR for prediction in checked),
        full_only=True,
    )
    return table, claim


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
    existing: dict[str, float] = {}
    for row in rows:
        timing = _time_call(row.call)
        for prefix, statistic in EXISTING_ROWS:
            at_grid_size = (
                row.name != "energy distance" or row.samples == GRID_POINTS[-1][0]
            )
            # * A name ends there or at a comma, so the calibration curve stays out.
            named = row.name == prefix or row.name.startswith(f"{prefix},")
            if named and at_grid_size:
                existing[statistic] = timing.seconds
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
        " being about 6 s per set. No claim reads these rows."
    )
    predictions = _grid_predictions(context, existing, QUICK_STATISTICS)
    grid_table, claim = _grid_section(predictions)
    points = QUICK_GRID if context.quick else GRID_POINTS
    point_text = ", ".join(
        f"({grouped(count)}, {dimension})" for count, dimension in points
    )
    cost_role = role("func", "mixed_signals.engines.cost.estimate_compare_cost")
    setup += (
        f" The model of the dry run ({cost_role}) predicts the seconds of one"
        " evaluation of every statistic from the samples per set, the parameters"
        " and the work units of the statistic. The second table times each"
        " statistic as a jitted program, as in the loop of the null draws, at the"
        f" pairs of samples per set and parameters {point_text}, except the"
        " GMM-BIC fit, which is timed as the plain call that runs once per"
        " posterior. It divides the larger of model and timing by the smaller."
        " The statistics that the table above times at the largest point are not"
        " timed again, and the"
        " caps of the package act on the model as they do on the timings. A call"
        f" of more than {GRID_SLOW_SECONDS:g} s is timed once, otherwise the"
        " median of three calls counts. A claim, in the full run only, lets every"
        " statistic that runs per null draw or surrogate and takes at least"
        f" {COSTLY_SECONDS * 1e3:g} ms lie within a factor of {MODEL_FACTOR:g} of"
        " the model; the faster rows, dominated by the cost of dispatch, and the"
        " statistics that run once per call (the median bandwidth, the ε of the"
        " sliced Sinkhorn divergence and the GMM-BIC fit) stay in the table"
        " unchecked. A run of the quick counts times six cheap statistics at one"
        " point."
    )
    return Section(setup, (table, grid_table), (claim,))


CHECKS: tuple[Check, ...] = (
    Check(
        label=COSTS_LABEL,
        title="Cost of the metrics and diagnostics",
        session="8, 9, 10, 12 and 14",
        supports=(
            role("data", "mixed_signals._config.SAMPLE_CAP", short=False),
            role("data", "mixed_signals.diagnostics.dependence.PAIR_CAP", short=False),
            role(
                "data",
                "mixed_signals.metrics.ot.sliced.SLICED_SINKHORN_CAP",
                short=False,
            ),
            role(
                "func", "mixed_signals.engines.cost.estimate_compare_cost", short=False
            ),
            role(
                "func", "mixed_signals.engines.cost.estimate_diagnose_cost", short=False
            ),
            "the cost weights of the dry run in session 14",
        ),
        run=_costs,
    ),
)
"""Checks of the costs."""
