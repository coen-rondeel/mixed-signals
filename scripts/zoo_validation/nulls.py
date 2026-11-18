"""Checks of the comparison nulls: permutation, weighted, thinned and split-half."""

import math

import jax
import jax.numpy as jnp
from jax.scipy.special import ndtri
from jaxtyping import Array
from matplotlib.figure import Figure as MatplotlibFigure

from mixed_signals import zoo
from mixed_signals.core.samples import kish_effective_sample_size
from mixed_signals.mcmc.reduce import ReductionRecord, reduce_chains
from mixed_signals.metrics.kernel import energy_distance
from mixed_signals.metrics.ot.sliced import (
    SLICED_DIRECTIONS,
    sliced_directions,
)
from mixed_signals.nulls import (
    NullPlan,
    permutation_null,
    random_sides,
    split_half_null,
)
from mixed_signals.nulls.plan import Statistic
from zoo_validation import figures
from zoo_validation.common import (
    DISPLAY_NAMES,
    FIRST_PARAMETER_METRICS,
    LEVELS,
    Comparison,
    collect,
    compare_on_plan,
    compare_with_constants,
    empirical_p_values,
    first_parameter_statistics,
    rate_counts,
    rate_measurements,
    sliced_w2_builder,
)
from zoo_validation.report import (
    Cell,
    Check,
    Claim,
    Context,
    Figure,
    Measurement,
    Section,
    Table,
    claim_rate_above,
    claim_rates_at_most,
    claim_rates_within,
    grouped,
    role,
)

LEVEL_COLUMNS = tuple(f"Rate at {level:.0%}" for level in LEVELS)

PERMUTATION_NULL = role("func", "mixed_signals.nulls.permutation.permutation_null")
SHIFTED_GAUSSIANS = role("func", "mixed_signals.zoo.cases.shifted_gaussians")


# * Check permutation-uniformity.

UNIFORMITY_LABEL = "permutation-uniformity"


def _permutation_uniformity(context: Context) -> Section:
    label = UNIFORMITY_LABEL
    sample_count, param_count = 1_000, 2
    draws = context.count(199, 19)
    replicates = context.count(120, 10)
    slow_draws = context.count(99, 19)
    slow_replicates = context.count(10, 1)
    case = zoo.shifted_gaussians(0.0, param_count=param_count)
    space = case.space
    sets = case.sample(
        context.key(label, "sets"), sample_count, batch_size=replicates
    ).samples
    statistics = first_parameter_statistics(space)

    def energy(
        first: Array, first_weights: Array, second: Array, second_weights: Array
    ) -> Array:
        return energy_distance(
            first,
            first_weights,
            second,
            second_weights,
            space,
            key=context.key(label, "subsample"),
        )

    statistics["energy"] = energy
    names = [*statistics, "sliced_w2"]
    comparisons: dict[str, list[tuple[Array, Array]]] = {name: [] for name in names}
    for replicate in range(replicates):
        first, second = sets[replicate]
        plan = permutation_null(
            first,
            None,
            second,
            None,
            key=context.key(label, "null", replicate),
            draws=draws,
        )
        for name, statistic in statistics.items():
            result = compare_on_plan(statistic, plan, first, None, second, None)
            comparisons[name].append((result.value, result.draws))
        if replicate < slow_replicates:
            directions = sliced_directions(
                space,
                SLICED_DIRECTIONS,
                key=context.key(label, "directions", replicate),
            )

            sliced_plan = permutation_null(
                first,
                None,
                second,
                None,
                key=context.key(label, "null-sliced", replicate),
                draws=slow_draws,
            )
            result = compare_with_constants(
                sliced_w2_builder(space),
                (directions,),
                sliced_plan,
                first,
                None,
                second,
                None,
            )
            comparisons["sliced_w2"].append((result.value, result.draws))
    p_values = {
        name: empirical_p_values(
            jnp.stack([value for value, _ in pairs]),
            jnp.stack([null for _, null in pairs]),
            name,
        )
        for name, pairs in comparisons.items()
    }
    rows: list[list[Cell]] = []
    claims: list[Claim] = []
    for name in names:
        trials = int(p_values[name].shape[0])
        rows.append([DISPLAY_NAMES[name], trials, *rate_measurements(p_values[name])])
        claims.append(
            claim_rates_within(
                f"The {DISPLAY_NAMES[name]} rejects at its levels.",
                rate_counts(p_values[name]),
                LEVELS,
                full_only=True,
            )
        )
    table = Table.of(
        "Rejection rates of identical sets: the share of replicates whose p-value"
        " is at or below the level, with the binomial error at the level.",
        ("Statistic", "Replicates", *LEVEL_COLUMNS),
        rows,
    )
    setup = (
        f"Two sets of {grouped(sample_count)} samples are drawn from one zoo Gaussian"
        f" ({SHIFTED_GAUSSIANS} without a shift,"
        f" {param_count} parameters), {replicates} times. For each replicate,"
        f" {PERMUTATION_NULL} splits the pool"
        f" of both sets at random {draws} times,"
        f" {role('func', 'mixed_signals.nulls.plan.null_draws')}"
        " evaluates a statistic on every draw, and"
        f" {role('func', 'mixed_signals.sigma.significance.null_significance')}"
        " reads the empirical p-value of the observed value, without the tail"
        " fit, on the tail of"
        f" {role('func', 'mixed_signals.sigma.thresholds.comparison_tail')}."
        " The first parameter gives the mean shift, the width ratio and the"
        " KS statistic. The sliced W2 uses the"
        f" {SLICED_DIRECTIONS} directions of its default and costs most per"
        f" draw, so it runs on the first {slow_replicates} replicates with"
        f" {slow_draws} draws only. If the null is exact, the p-values are"
        " uniform and the rejection rate at a level equals the level; a number"
        " of draws one below a multiple of 100 makes the levels of 1%, 5% and"
        " 10% exact."
    )
    groups = [names[start : start + 3] for start in range(0, len(names), 3)]
    data = {
        "statistics": [DISPLAY_NAMES[name] for name in names],
        "p_values": {DISPLAY_NAMES[name]: p_values[name] for name in names},
    }

    def draw(style: figures.FigureStyle) -> MatplotlibFigure:
        figure, panels = figures.new_figure(style, columns=len(groups), sharey=True)
        for index, (axes, group) in enumerate(zip(panels, groups, strict=True)):
            for series, name in enumerate(group):
                ordered = jnp.sort(p_values[name])
                count = ordered.shape[0]
                x_values = jnp.concatenate([jnp.zeros(1), ordered, jnp.ones(1)])
                y_values = jnp.concatenate([jnp.arange(count + 1) / count, jnp.ones(1)])
                figures.plot_steps(
                    axes, style, series, x_values, y_values, label=DISPLAY_NAMES[name]
                )
            figures.plot_reference(axes, style, identity=True)
            figures.finish_panel(
                axes,
                style,
                x_label="p-value",
                y_label="empirical distribution function" if index == 0 else None,
            )
            axes.set_xlim(0, 1)
            axes.set_ylim(0, 1)
            figures.add_legend(axes, style, location="lower right")
        return figure

    figure = Figure(
        label=f"{label}-ecdf",
        caption="Empirical distribution function of the p-values of identical sets;"
        " the line is the uniform distribution.",
        alt="Step curves of the p-value distribution functions of five statistics,"
        " each close to the diagonal.",
        draw=draw,
        data=data,
    )
    return Section(setup, (table, figure), tuple(claims))


# * Check permutation-thinned-chains.

THINNED_LABEL = "permutation-thinned-chains"
SLOW_MEAN_ALLOWANCE = 1.5
"""Multiple of the level that the developers allow for the mean of the slowest chain."""


def _null_scores(values: list[Array], draws: list[Array]) -> Array:
    """Return (value minus null mean) over the null spread, per replicate."""
    drawn = jnp.stack(draws)
    return (jnp.stack(values) - jnp.mean(drawn, axis=-1)) / jnp.std(
        drawn, axis=-1, ddof=1
    )


def _variance_measurement(scores: Array) -> Measurement:
    """Return the variance of the scores with its standard error for normal scores."""
    count = int(scores.shape[0])
    variance = float(jnp.var(scores, ddof=1))
    return Measurement(variance, variance * math.sqrt(2 / (count - 1)), count)


def _narrowing_of_rate(level: float, rate: float) -> float:
    """Return how many times too narrow a null is that rejects at ``rate``.

    The test is two-sided and the statistic normal, so the narrowing is the
    ratio of the normal quantiles at the level and at the rate.
    """
    return float(ndtri(1.0 - level / 2.0) / ndtri(1.0 - rate / 2.0))


def _thinned_chains(context: Context) -> Section:
    label = THINNED_LABEL
    coefficients = (0.9, 0.5)
    walker_count, steps_per_tau = 32, 100
    replicate_counts = (context.count(300, 3), context.count(120, 3))
    draws = context.count(99, 19)
    property_rows: list[list[Cell]] = []
    rate_rows: list[list[Cell]] = []
    variance_rows: list[list[Cell]] = []
    variances: dict[tuple[float, str], Measurement] = {}
    thinned: dict[float, tuple[Measurement, Measurement]] = {}
    claims: list[Claim] = []
    for index, (coefficient, replicates) in enumerate(
        zip(coefficients, replicate_counts, strict=True)
    ):
        case = zoo.ar1_chains([coefficient])
        statistics = first_parameter_statistics(case.space)
        true_tau = case.reference_value("tau")
        step_count = steps_per_tau * round(true_tau)
        records: list[ReductionRecord] = []
        values: dict[str, dict[str, list[Array]]] = {
            kind: {metric: [] for metric in FIRST_PARAMETER_METRICS}
            for kind in ("default", "all")
        }
        draw_values: dict[str, dict[str, list[Array]]] = {
            kind: {metric: [] for metric in FIRST_PARAMETER_METRICS}
            for kind in ("default", "all")
        }
        for replicate in range(replicates):
            reductions = [
                reduce_chains(
                    case.sample(
                        context.key(label, "chains", index, replicate, run),
                        step_count,
                        walker_count=walker_count,
                    )
                )
                for run in (0, 1)
            ]
            records.extend(reduction.record for reduction in reductions)
            first, second = (
                reduction.sample_set.samples[0, 0] for reduction in reductions
            )
            plans = {
                "default": permutation_null(
                    first,
                    None,
                    second,
                    None,
                    key=context.key(label, "null", index, replicate, "default"),
                    first_n_eff=reductions[0].record.n_eff,
                    second_n_eff=reductions[1].record.n_eff,
                    draws=draws,
                ),
                "all": permutation_null(
                    first,
                    None,
                    second,
                    None,
                    key=context.key(label, "null", index, replicate, "all"),
                    draws=draws,
                ),
            }
            for kind, plan in plans.items():
                for metric, statistic in statistics.items():
                    result = compare_on_plan(statistic, plan, first, None, second, None)
                    values[kind][metric].append(result.value)
                    draw_values[kind][metric].append(result.draws)
        thinned[coefficient] = (
            Measurement.from_replicates([record.thinned_tau[0] for record in records]),
            Measurement.from_replicates(
                [
                    (1 + coefficient**record.thin) / (1 - coefficient**record.thin)
                    for record in records
                ]
            ),
        )
        property_rows.append(
            [
                coefficient,
                true_tau,
                Measurement.from_replicates([record.tau[0] for record in records]),
                Measurement.from_replicates([record.thin for record in records]),
                *thinned[coefficient],
                Measurement.from_replicates(
                    [record.sample_count for record in records]
                ),
                Measurement.from_replicates([record.n_eff for record in records]),
            ]
        )
        for kind, kind_name in (
            ("default", "sides at the effective sample size"),
            ("all", "all samples, sides of N"),
        ):
            variances[(coefficient, kind)] = _variance_measurement(
                _null_scores(
                    values[kind]["mean_shift"], draw_values[kind]["mean_shift"]
                )
            )
            variance_rows.append(
                [coefficient, replicates, kind_name, variances[(coefficient, kind)]]
            )
            for metric in FIRST_PARAMETER_METRICS:
                p_values = empirical_p_values(
                    jnp.stack(values[kind][metric]),
                    jnp.stack(draw_values[kind][metric]),
                    metric,
                )
                rate_rows.append(
                    [
                        coefficient,
                        kind_name,
                        DISPLAY_NAMES[metric],
                        *rate_measurements(p_values),
                    ]
                )
                if kind == "default":
                    is_slow_mean = (
                        coefficient == coefficients[0] and metric == "mean_shift"
                    )
                    statement = (
                        f"For coefficient {coefficient}, the default null"
                        " over-rejects the mean shift by at most"
                        f" {SLOW_MEAN_ALLOWANCE - 1:.0%} of the level."
                        if is_slow_mean
                        else f"For coefficient {coefficient}, the default null does"
                        f" not over-reject the {DISPLAY_NAMES[metric]}."
                    )
                    claims.append(
                        claim_rates_at_most(
                            statement,
                            rate_counts(p_values),
                            LEVELS,
                            allowance=SLOW_MEAN_ALLOWANCE if is_slow_mean else 1.0,
                            full_only=True,
                        )
                    )
    properties = Table.of(
        "AR(1) runs after reduction: the true autocorrelation time of the case and"
        " the means over runs of the estimated one, the thinning factor k, the"
        " autocorrelation time left after thinning as the reduction estimates it,"
        " the same as the thinning factor implies for an AR(1) chain with"
        " coefficient φ, (1 + φᵏ)/(1 − φᵏ), the sample count and the effective"
        " sample size.",
        (
            "Coefficient",
            "True τ",
            "Estimated τ",
            "Thinning",
            "Thinned τ",
            "Implied thinned τ",
            "N",
            "N_eff",
        ),
        property_rows,
    )
    rates = Table.of(
        "Rejection rates of two independent runs of the same chain.",
        ("Coefficient", "Null", "Statistic", *LEVEL_COLUMNS),
        rate_rows,
    )
    variance_table = Table.of(
        "Variance over the replicates of the mean shift in units of its null: the"
        " value minus the null mean, over the null standard deviation. A null of"
        " the right width gives one; the error is that of a variance of normal"
        " scores.",
        ("Coefficient", "Replicates", "Null", "Variance"),
        variance_rows,
    )
    slowest = coefficients[0]
    thinned_estimate, thinned_implied = thinned[slowest]
    default_variance = variances[(slowest, "default")]
    allowed_rate = SLOW_MEAN_ALLOWANCE * LEVELS[1]
    narrowing = _narrowing_of_rate(LEVELS[1], allowed_rate)
    setup = (
        f"Zoo chains {role('func', 'mixed_signals.zoo.cases.ar1_chains')} with the"
        f" coefficients {coefficients[0]} and {coefficients[1]}, {walker_count}"
        f" walkers and {steps_per_tau} times the true autocorrelation time τ in steps,"
        f" reduced by {role('func', 'mixed_signals.mcmc.reduce.reduce_chains')}"
        " (thinning by the autocorrelation time). Each of the"
        f" {replicate_counts[0]} replicates at coefficient {coefficients[0]} and"
        f" the {replicate_counts[1]} at {coefficients[1]} compares two"
        " independent runs, with the statistics of the first parameter and"
        f" {draws} permutation draws. The default null gives each side the"
        " effective sample size of its set, through the arguments `first_n_eff`"
        " and `second_n_eff` of"
        f" {PERMUTATION_NULL}; the contrast"
        " permutes all samples with sides of the sample count. The thinned"
        " samples keep some autocorrelation, so a null of full sides is too"
        " narrow. The reduction estimates the autocorrelation time left after"
        f" thinning at {thinned_estimate.render()}, against"
        f" {thinned_implied.render()} that the thinning factor implies for an"
        " AR(1) chain, (1 + φᵏ)/(1 − φᵏ). Sides of N/τₖ samples therefore make the"
        " default null narrower than the scatter of the mean shift by their"
        f" ratio, {thinned_implied.value / thinned_estimate.value:.3f} in variance;"
        f" the replicates at coefficient {slowest} give"
        f" {default_variance.render()} for the default null and"
        f" {variances[(slowest, 'all')].render()} for the sides of N, against one"
        " for a null of the right width. The developers allow the rejection rate"
        f" of the mean shift at coefficient {slowest}, the slowest chain, up to"
        f" {SLOW_MEAN_ALLOWANCE:g} times the level, plus three binomial errors at"
        f" that bound. At the {LEVELS[1]:.0%} level that lets σ_sig read up to"
        f" {narrowing - 1:.0%} high: a rate of {allowed_rate:.1%} is the rate of a"
        f" null {narrowing:.2f} times too narrow in standard deviation. The reason"
        " is that the τₖ of the reduction reads low. The other claims of the check"
        " stay strict."
    )
    return Section(setup, (properties, rates, variance_table), tuple(claims))


# * Check permutation-weights.

WEIGHTS_LABEL = "permutation-weights"
PROPOSAL_WIDTH = 1.5


def _weighted_case() -> zoo.ZooCase:
    identity = jnp.eye(2)
    return zoo.gaussians(
        jnp.zeros((2, 2)),
        jnp.stack([identity, PROPOSAL_WIDTH**2 * identity]),
        names=("target", "proposal"),
    )


def _importance_log_weights(case: zoo.ZooCase, points: Array) -> Array:
    log_densities = case.log_prob(points)
    return log_densities[..., 0] - log_densities[..., 1]


def _weights(context: Context) -> Section:
    label = WEIGHTS_LABEL
    weighted_count, equal_count = 4_000, 2_000
    replicates = context.count(70, 3)
    larger_replicates = context.count(100, 3)
    draws = context.count(99, 19)
    case = _weighted_case()
    statistics = first_parameter_statistics(case.space)
    sets = case.sample(
        context.key(label, "sets"), weighted_count, batch_size=replicates
    ).samples
    kinds = ("default", "resampled")
    values: dict[str, dict[str, list[Array]]] = {
        kind: {metric: [] for metric in statistics} for kind in kinds
    }
    draw_values: dict[str, dict[str, list[Array]]] = {
        kind: {metric: [] for metric in statistics} for kind in kinds
    }
    kish_sizes = []
    for replicate in range(replicates):
        weighted = sets[replicate, 1]
        equal = sets[replicate, 0, :equal_count]
        log_weights = _importance_log_weights(case, weighted)
        kish_sizes.append(float(kish_effective_sample_size(log_weights)))
        indices = jax.random.choice(
            context.key(label, "resample", replicate),
            weighted_count,
            shape=(weighted_count,),
            p=jax.nn.softmax(log_weights),
        )
        resampled = weighted[indices]
        equal_indices = jax.random.choice(
            context.key(label, "resample-equal", replicate),
            equal_count,
            shape=(equal_count,),
        )
        for kind, first, first_weights, second in (
            ("default", weighted, log_weights, equal),
            ("resampled", resampled, None, equal[equal_indices]),
        ):
            plan = permutation_null(
                first,
                first_weights,
                second,
                None,
                key=context.key(label, "null", replicate, kind),
                draws=draws,
            )
            for metric, statistic in statistics.items():
                result = compare_on_plan(
                    statistic, plan, first, first_weights, second, None
                )
                values[kind][metric].append(result.value)
                draw_values[kind][metric].append(result.draws)
    rate_rows: list[list[Cell]] = []
    claims: list[Claim] = []
    for kind, kind_name in (
        ("default", "weights kept in the pool"),
        ("resampled", "resampled to equal weights first"),
    ):
        for metric in FIRST_PARAMETER_METRICS:
            p_values = empirical_p_values(
                jnp.stack(values[kind][metric]),
                jnp.stack(draw_values[kind][metric]),
                metric,
            )
            rate_rows.append(
                [kind_name, DISPLAY_NAMES[metric], *rate_measurements(p_values)]
            )
            counts = rate_counts(p_values)
            if kind == "default":
                claims.append(
                    claim_rates_within(
                        f"With the weights kept, the {DISPLAY_NAMES[metric]} rejects"
                        " at its levels.",
                        counts,
                        LEVELS,
                        full_only=True,
                    )
                )
            else:
                claims.append(
                    claim_rate_above(
                        f"After resampling, the {DISPLAY_NAMES[metric]} rejects"
                        " too often at 5%.",
                        counts[1][0],
                        counts[1][1],
                        LEVELS[1],
                        full_only=True,
                    )
                )
    rates = Table.of(
        "Rejection rates of an importance-weighted set against equal-weight"
        " draws of its target.",
        ("Null", "Statistic", *LEVEL_COLUMNS),
        rate_rows,
    )
    larger_rows: list[list[Cell]] = []
    larger_sizes = (200, 800, 3_200)
    width_statistic = statistics["width_ratio"]
    for size in larger_sizes:
        larger_sets = case.sample(
            context.key(label, "larger", size), size, batch_size=larger_replicates
        ).samples

        def comparison_of(
            replicate: int, size: int = size, larger_sets: Array = larger_sets
        ) -> Comparison:
            return _weighted_comparison(
                context,
                case,
                larger_sets[replicate],
                width_statistic,
                label,
                size,
                replicate,
                draws,
            )

        width_values, width_draws = collect(comparison_of, larger_replicates)
        p_values = empirical_p_values(width_values, width_draws, "width_ratio")
        larger_rows.append([size, *rate_measurements(p_values)])
        claims.append(
            claim_rates_within(
                f"At {grouped(size)} samples per set, the log width ratio rejects at"
                " its levels.",
                rate_counts(p_values),
                LEVELS,
                full_only=True,
            )
        )
    larger = Table.of(
        "Rejection rates of the log width ratio of one weighted set against an"
        " equal-weight set, for growing sample counts per set.",
        ("Samples per set", *LEVEL_COLUMNS),
        larger_rows,
    )
    kish = Measurement.from_replicates(kish_sizes)
    setup = (
        "Zoo Gaussians in two parameters: the target"
        f" N(0, I) and a proposal {PROPOSAL_WIDTH} times wider. The weighted set"
        f" holds {grouped(weighted_count)} proposal samples with the importance weights"
        f" target over proposal (Kish size {kish.render()}),"
        f" the other set {grouped(equal_count)} equal-weight draws of the target."
        f" {replicates} replicates use {draws} permutation draws of"
        f" {PERMUTATION_NULL}, whose sides"
        " follow the Kish sizes. For contrast, the script resamples both sets to"
        " equal weights, with replacement, with `jax.random.choice` before the"
        " permutation, an alternative that the package does not offer. Then the"
        " width ratio"
        f" of one weighted set against an equal-weight set is checked at"
        f" {larger_sizes[0]}, {larger_sizes[1]} and {grouped(larger_sizes[2])} samples"
        f" per set with {larger_replicates} replicates each."
    )
    return Section(setup, (rates, larger), tuple(claims))


def _weighted_comparison(
    context: Context,
    case: zoo.ZooCase,
    pair: Array,
    statistic: Statistic,
    label: str,
    size: int,
    replicate: int,
    draws: int,
) -> Comparison:
    weighted, equal = pair[1], pair[0]
    log_weights = _importance_log_weights(case, weighted)
    plan = permutation_null(
        weighted,
        log_weights,
        equal,
        None,
        key=context.key(label, "larger-null", size, replicate),
        draws=draws,
    )
    return compare_on_plan(statistic, plan, weighted, log_weights, equal, None)


# * Check split-half.

SPLIT_HALF_LABEL = "split-half"


def _split_half(context: Context) -> Section:
    label = SPLIT_HALF_LABEL
    replicates = context.count(120, 3)
    draws = context.count(99, 19)
    cases = (
        ("large reference", 10_000, 1_000),
        ("small reference", 4_000, 2_000),
    )
    case = zoo.shifted_gaussians(0.0, param_count=1)
    statistics = first_parameter_statistics(case.space)
    rate_rows: list[list[Cell]] = []
    side_rows: list[list[Cell]] = []
    claims: list[Claim] = []
    for case_index, (case_name, reference_count, approximation_count) in enumerate(
        cases
    ):
        sets = case.sample(
            context.key(label, "sets", case_index),
            reference_count,
            batch_size=replicates,
        ).samples
        kinds = ["default", "approximation size"] if case_index == 0 else ["default"]
        values: dict[str, dict[str, list[Array]]] = {
            kind: {metric: [] for metric in statistics} for kind in kinds
        }
        draw_values: dict[str, dict[str, list[Array]]] = {
            kind: {metric: [] for metric in statistics} for kind in kinds
        }
        sizes = []
        for replicate in range(replicates):
            reference = sets[replicate, 0]
            approximation = sets[replicate, 1, :approximation_count]
            plans: dict[str, NullPlan] = {
                "default": split_half_null(
                    reference,
                    None,
                    approximation_n_eff=float(approximation_count),
                    key=context.key(label, "null", case_index, replicate),
                    draws=draws,
                )
            }
            sizes.append(int(plans["default"].first.shape[1]))
            if case_index == 0:
                first_sides, second_sides = random_sides(
                    context.key(label, "null", case_index, replicate, "contrast"),
                    reference_count,
                    approximation_count,
                    draws=draws,
                )
                plans["approximation size"] = NullPlan(
                    construction="split_half",
                    samples=reference,
                    log_weights=jax.nn.log_softmax(jnp.zeros(reference_count)),
                    first=first_sides,
                    second=second_sides,
                )
            for kind, plan in plans.items():
                for metric, statistic in statistics.items():
                    result = compare_on_plan(
                        statistic, plan, reference, None, approximation, None
                    )
                    values[kind][metric].append(result.value)
                    draw_values[kind][metric].append(result.draws)
        side_rows.append(
            [
                case_name,
                reference_count,
                approximation_count,
                sizes[0]
                if len(set(sizes)) == 1
                else Measurement.from_replicates(sizes),
            ]
        )
        for kind in kinds:
            for metric in FIRST_PARAMETER_METRICS:
                p_values = empirical_p_values(
                    jnp.stack(values[kind][metric]),
                    jnp.stack(draw_values[kind][metric]),
                    metric,
                )
                rate_rows.append(
                    [
                        case_name,
                        kind,
                        DISPLAY_NAMES[metric],
                        *rate_measurements(p_values),
                    ]
                )
                if kind != "default":
                    continue
                counts = rate_counts(p_values)
                if case_index == 0:
                    claims.append(
                        claim_rates_within(
                            f"With the default sides, the {DISPLAY_NAMES[metric]}"
                            " rejects at its levels.",
                            counts,
                            LEVELS,
                            full_only=True,
                        )
                    )
                else:
                    claims.append(
                        claim_rates_at_most(
                            f"With clipped sides, the {DISPLAY_NAMES[metric]} is"
                            " conservative.",
                            counts,
                            LEVELS,
                            full_only=True,
                        )
                    )
    sides = Table.of(
        "Cases of the split-half null: sample counts and the side size of the default.",
        ("Case", "Reference N", "Approximation N", "Side size"),
        side_rows,
    )
    rates = Table.of(
        "Rejection rates of an approximation drawn from the reference itself.",
        ("Case", "Sides", "Statistic", *LEVEL_COLUMNS),
        rate_rows,
    )
    setup = (
        "A reference and an approximation of equal weight are drawn from one"
        f" zoo Gaussian ({SHIFTED_GAUSSIANS}"
        " without a shift, one parameter), so that the null hypothesis holds."
        f" {role('func', 'mixed_signals.nulls.split_half.split_half_null')} cuts the"
        f" reference alone into two sides, {draws} times per replicate, over"
        f" {replicates} replicates. The default sides carry the variance of the"
        " comparison. For contrast, the script builds sides of the size of the"
        f" approximation with {role('func', 'mixed_signals.nulls.plan.random_sides')}."
        " A reference of 4 000 against 2 000 cannot supply such sides: they are"
        " clipped at half the reference, and the null warns that it is wider"
        " than the comparison."
    )
    return Section(setup, (sides, rates), tuple(claims))


CHECKS: tuple[Check, ...] = (
    Check(
        label=UNIFORMITY_LABEL,
        title="Uniform p-values under the permutation null",
        session="12a",
        supports=(
            role(
                "func", "mixed_signals.nulls.permutation.permutation_null", short=False
            ),
            role(
                "func",
                "mixed_signals.sigma.significance.null_significance",
                short=False,
            ),
            "the done criterion of session 12: uniform p-values for identical sets",
        ),
        run=_permutation_uniformity,
    ),
    Check(
        label=THINNED_LABEL,
        title="Permutation null on thinned chains",
        session="12a",
        supports=(
            role(
                "func",
                "mixed_signals.nulls.permutation.permutation_side_size",
                short=False,
            ),
            "the null sides at the effective sample size",
        ),
        run=_thinned_chains,
    ),
    Check(
        label=WEIGHTS_LABEL,
        title="Permutation null on weighted sets",
        session="12a",
        supports=(
            role(
                "func", "mixed_signals.nulls.permutation.permutation_null", short=False
            ),
            "the pool that keeps every sample with its weight",
        ),
        run=_weights,
    ),
    Check(
        label=SPLIT_HALF_LABEL,
        title="Split-half null",
        session="12a",
        supports=(
            role("func", "mixed_signals.nulls.split_half.split_half_null", short=False),
            role(
                "func",
                "mixed_signals.nulls.split_half.split_half_side_size",
                short=False,
            ),
        ),
        run=_split_half,
    ),
)
