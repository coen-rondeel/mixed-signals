"""The validation build: data types, rendering and a quick run of every check."""

import importlib
import importlib.util
import io
import json
import math
import sys
import warnings
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType

import jax
import jax.numpy as jnp
import pytest
import scipy.integrate
from zoo_validation import costs, diagnostics, effects, engines, figures, nulls
from zoo_validation.common import (
    ResolutionRuns,
    pooled_resolution,
    resolution_ratio,
    sample_shift,
)
from zoo_validation.report import (
    ROLE_PATTERN,
    Case,
    Check,
    Claim,
    Context,
    Figure,
    Measurement,
    Section,
    Table,
    _rerun_sentence,
    claim_at_least,
    claim_at_most,
    claim_cases,
    claim_cases_above,
    claim_rates_at_most,
    claim_rates_within,
    claim_true,
    claim_within,
    critical_value,
    error_text,
    grouped,
    run_check,
)

from mixed_signals import zoo
from mixed_signals.core.space import Parameter
from mixed_signals.diagnostics import MODE_FLOOR, marginal_modes
from mixed_signals.errors import ValidityWarning

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "build_validation.py"


@pytest.fixture(scope="module")
def build_module() -> ModuleType:
    """The build script, imported by its path."""
    spec = importlib.util.spec_from_file_location("build_validation", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["build_validation"] = module
    spec.loader.exec_module(module)
    return module


def make_check(label: str, claims: tuple[Claim, ...], *, warn: bool = False) -> Check:
    """Return a check that gives the claims, and raises a package warning if asked."""

    def run(context: Context) -> Section:
        del context
        if warn:
            warnings.warn(ValidityWarning("a package warning"), stacklevel=1)
        return Section("Setup of the check.", (), claims)

    return Check(label, f"Check {label}", "12b", ("a decision",), run)


def walk_measurements(entry: object) -> Iterator[dict[str, float | None]]:
    """Yield every measurement dictionary of a JSON value."""
    if isinstance(entry, dict):
        if "value" in entry and "error" in entry:
            yield entry
        else:
            for item in entry.values():
                yield from walk_measurements(item)
    elif isinstance(entry, list):
        for item in entry:
            yield from walk_measurements(item)


def resolve_target(target: str) -> object:
    """Import the object that a dotted name points at."""
    parts = target.lstrip("~").split(".")
    for split in range(len(parts), 0, -1):
        try:
            current: object = importlib.import_module(".".join(parts[:split]))
        except ModuleNotFoundError:
            continue
        for name in parts[split:]:
            current = getattr(current, name)
        return current
    raise ModuleNotFoundError(target)


def defining_path(target: str) -> str | None:
    """Return where a function or class is defined, or ``None`` for plain data."""
    resolved = resolve_target(target)
    module = getattr(resolved, "__module__", None)
    qualified = getattr(resolved, "__qualname__", None)
    if callable(resolved) and module and qualified:
        return f"{module}.{qualified}"
    return None


def test_measurement_precision_follows_the_error() -> None:
    """The error has two significant digits and the value the same decimals."""
    assert Measurement(0.051_234_5, 0.0049).render() == "0.0512 ± 0.0049"
    assert Measurement(123.456, 12.0).render() == "123 ± 12"
    assert Measurement(2.0, 0.31).render() == "2.00 ± 0.31"


def test_measurement_without_error_and_with_nan() -> None:
    """An exact value has four significant digits, and NaN reads NaN."""
    assert Measurement(0.123_456).render() == "0.1235"
    assert Measurement(math.nan, 0.1).render() == "NaN"


def test_digits_are_grouped_by_spaces() -> None:
    """Numbers write as 1 000 and 10 000, in prose and in tables."""
    assert grouped(1_000) == "1 000"
    assert grouped(10_000) == "10 000"
    assert Measurement(3_717.0, 45.0).render() == "3 717 ± 45"
    table = Table.of("Caption.", ("N",), [[1_818], [12]])
    assert "| 1 818 |" in table.render()
    assert "| 12 |" in table.render()


def test_a_constant_value_renders_without_error() -> None:
    """An error below 1e-12 of the value counts as none."""
    assert Measurement(1_818.000_000_000_000_227, 1.9e-14).render() == "1 818"
    assert Measurement(0.5, 0.0).render() == "0.5"
    assert Measurement(1_818.0, 1.9e-14, 6).margin() == 0.0


def test_error_text_names_the_factor_with_its_source() -> None:
    """Few replicates widen the factor, and the text says so with numbers."""
    assert error_text(None) == "three standard errors"
    assert error_text(6) == "5.5 standard errors (Student t, 6 replicates)"
    assert error_text(100) == "3.1 standard errors (Student t, 100 replicates)"
    assert error_text(1_000) == "three standard errors"


def test_cases_name_the_worst_with_its_deviation_and_allowance() -> None:
    """A claim over cases reports the case with the least room."""
    cases = [Case("first", 0.01, 0.02, 0.05), Case("second", -0.061, 0.03, 0.053)]
    claim = claim_cases("Cases.", cases, tolerance_text="5% of the truth", replicates=6)
    assert claim.consistent
    assert claim.expected == (
        "within 5% of the truth plus 5.5 standard errors (Student t, 6 replicates)"
    )
    assert claim.observed == "worst of 2: second: -0.061 against ±0.083"
    failing = claim_cases(
        "Cases.", [Case("only", 0.2, 0.01, 0.05)], tolerance_text=None, replicates=None
    )
    assert not failing.consistent
    assert failing.error_part == 0.01
    assert failing.tolerance_part == 0.05


def test_measurement_rate_uses_the_binomial_error_at_the_level() -> None:
    """The error of a rate at a level is sqrt(level (1 - level) / trials)."""
    rate = Measurement.from_rate(10, 200, level=0.05)
    assert rate.value == pytest.approx(0.05)
    assert rate.error == pytest.approx(math.sqrt(0.05 * 0.95 / 200))
    assert Measurement.from_rate(0, 100).error is not None


def test_measurement_from_replicates() -> None:
    """The mean and the standard error of the mean of replicates."""
    measurement = Measurement.from_replicates(jnp.array([1.0, 2.0, 3.0, 4.0]))
    assert measurement.value == pytest.approx(2.5)
    assert measurement.error == pytest.approx(math.sqrt(5 / 3) / 2)
    assert Measurement.from_replicates([1.0]).error is None


def test_table_renders_a_pipe_table_with_a_caption() -> None:
    """A table is a MyST directive with a header, a rule and escaped cells."""
    table = Table.of(
        "A caption.",
        ("Name", "Value"),
        [["a|b", Measurement(1.0, 0.1)], ["c", None], ["d", True], ["e", math.nan]],
    )
    lines = table.render().splitlines()
    assert lines[0] == ":::{table} A caption."
    assert lines[2] == "| Name | Value |"
    assert lines[3] == "| --- | --- |"
    assert lines[4] == r"| a\|b | 1.00 ± 0.10 |"
    assert lines[5:8] == ["| c | - |", "| d | yes |", "| e | NaN |"]
    assert lines[-1] == ":::"


def test_table_rejects_a_row_of_the_wrong_length() -> None:
    """Every row needs one cell per column."""
    with pytest.raises(ValueError, match="does not fit"):
        Table.of("Caption.", ("a", "b"), [["only one"]])


def test_json_round_trip_keeps_full_precision() -> None:
    """A table stored as JSON and read back holds the same numbers."""
    table = Table.of(
        "Caption.",
        ("Name", "Number", "Measured"),
        [
            ["row", 0.1 + 0.2, Measurement(1 / 3, 2 / 7)],
            ["other", 12, Measurement(math.nan, None)],
        ],
    )
    loaded = Table.from_json(json.loads(json.dumps(table.to_json(), allow_nan=False)))
    assert loaded.caption == table.caption
    assert loaded.columns == table.columns
    assert loaded.rows[0] == table.rows[0]
    other_measurement = loaded.rows[1][2]
    assert isinstance(other_measurement, Measurement)
    assert math.isnan(other_measurement.value)
    assert loaded.rows[1][:2] == table.rows[1][:2]


def test_claims_state_their_verdicts() -> None:
    """The claim helpers compute the verdict from the numbers."""
    measurement = Measurement(0.052, 0.005)
    assert claim_within("w", measurement, 0.05, 0.015, tolerance_text="t").consistent
    assert not claim_within("w", measurement, 0.1, 0.015, tolerance_text="t").consistent
    assert claim_at_most("m", measurement, 0.06, limit_text="l").consistent
    assert not claim_at_least("m", measurement, 0.06, limit_text="l").consistent
    assert claim_true("t", "e", "o", True).verdict(quick=False) == "consistent"
    assert claim_true("t", "e", "o", False).verdict(quick=False) == "not consistent"


def test_critical_value_is_the_student_quantile_of_three_sigma() -> None:
    """Few replicates widen the tolerance; many give three standard errors."""
    assert critical_value(None) == 3.0
    assert critical_value(2) == pytest.approx(235.8, abs=0.1)
    assert critical_value(3) == pytest.approx(19.2, abs=0.1)
    assert critical_value(10_000) == pytest.approx(3.0, abs=0.01)
    measurement = Measurement.from_replicates([1.0, 2.0, 3.0])
    assert measurement.error is not None
    assert measurement.margin() == pytest.approx(19.2 * measurement.error, rel=1e-3)
    assert Measurement(1.0).margin() == 0.0


def test_rate_claims_use_three_binomial_errors() -> None:
    """A rate at 5% of 200 replicates may lie 3 x 1.5% from its level."""
    assert claim_rates_within("r", [(10, 200)], [0.05]).consistent
    assert claim_rates_within("r", [(18, 200)], [0.05]).consistent
    assert not claim_rates_within("r", [(30, 200)], [0.05]).consistent
    assert claim_rates_at_most("r", [(2, 200)], [0.05]).consistent
    assert not claim_rates_at_most("r", [(30, 200)], [0.05]).consistent


def test_a_full_only_claim_is_not_evaluated_in_quick_mode() -> None:
    """The page says so, and the claim cannot fail a quick build."""
    claim = claim_true("t", "e", "o", False, full_only=True)
    assert claim.verdict(quick=True) == "not evaluated (full run only)"
    assert not claim.failed(quick=True)
    assert claim.failed(quick=False)


def test_a_check_label_must_be_kebab_case() -> None:
    """A label is a lower-case word list joined by hyphens."""
    with pytest.raises(ValueError, match="kebab case"):
        make_check("Not Kebab", ())


def test_keys_depend_on_the_label_and_not_on_the_other_checks() -> None:
    """A check derives the same key whatever else runs."""
    context = Context.create(3, quick=True)
    first = context.key("one", "a", 1)
    again = Context.create(3, quick=False).key("one", "a", 1)
    assert bool(jnp.array_equal(jax.random.key_data(first), jax.random.key_data(again)))
    assert context.count(10, 2) == 2
    assert Context.create(3, quick=False).count(10, 2) == 10


def test_run_check_records_package_warnings_with_counts() -> None:
    """A package warning raised inside a check is listed with its category."""
    check = make_check("warns", (), warn=True)
    result = run_check(check, "Topic", Context.create(1, quick=True))
    assert [(entry.category, entry.count) for entry in result.warnings] == [
        ("ValidityWarning", 1)
    ]
    assert result.seconds >= 0.0


def test_figures_are_saved_in_both_themes_and_repeat_exactly(tmp_path: Path) -> None:
    """A figure gives a light and a dark file, and equal data give equal bytes."""

    def draw(style: figures.FigureStyle):  # type: ignore[no-untyped-def]
        figure, panels = figures.new_figure(style)
        figures.plot_series(panels[0], style, 0, [1, 2, 3], [1, 4, 9], label="a")
        figures.plot_series(panels[0], style, 1, [1, 2, 3], [2, 3, 5], label="b")
        figures.plot_reference(panels[0], style, identity=True)
        figures.finish_panel(panels[0], style, x_label="x", y_label="y")
        figures.add_legend(panels[0], style)
        return figure

    figure = Figure("demo", "Caption.", "Alt text.", draw, {"x": [1, 2, 3]})
    for directory in (tmp_path / "first", tmp_path / "second"):
        figure.save(directory)
    for theme in ("light", "dark"):
        name = f"demo-{theme}.svg"
        assert (tmp_path / "first" / name).read_bytes() == (
            tmp_path / "second" / name
        ).read_bytes()
    for line in (tmp_path / "first" / "demo-light.svg").read_text().splitlines():
        assert line == line.rstrip()
    rendered = figure.render()
    assert ":class: only-light" in rendered
    assert ":class: only-dark" in rendered
    assert ":alt: Alt text." in rendered


def test_an_allowance_widens_the_bound_of_a_rate_claim() -> None:
    """A rate allowance sets the bound at a multiple of the level and says so."""
    counts = [(6, 300), (29, 300), (49, 300)]
    levels = (0.01, 0.05, 0.10)
    strict = claim_rates_at_most("Strict.", counts, levels)
    relaxed = claim_rates_at_most("Relaxed.", counts, levels, allowance=1.5)
    assert not strict.consistent
    assert relaxed.consistent
    assert relaxed.expected == (
        "rates at most 1.5 times the levels 1%, 5%, 10%, that is 1.5%, 7.5%, 15.0%,"
        " plus three binomial errors at those bounds"
    )


def test_a_rate_of_one_and_a_half_times_the_level_is_a_null_ten_percent_narrow() -> (
    None
):
    """At the 5% level, a two-sided rate of 7.5% belongs to a null 1.10 times narrow."""
    assert nulls._narrowing_of_rate(0.05, 0.075) == pytest.approx(1.10, abs=0.005)
    assert nulls._narrowing_of_rate(0.05, 0.05) == pytest.approx(1.0)


def test_an_outside_legend_hangs_below_the_axes() -> None:
    """The legend of ``outside=True`` lies fully under the axes, clear of the data."""
    style = figures.LIGHT
    figure, panels = figures.new_figure(style, extra_height=1.0)
    figures.plot_series(panels[0], style, 0, [1, 2, 3], [1, 4, 9], label="first")
    figures.plot_series(panels[0], style, 1, [1, 2, 3], [2, 3, 5], label="second")
    figures.finish_panel(panels[0], style, x_label="x", y_label="y")
    figures.add_legend(panels[0], style, outside=True)
    figure.savefig(io.BytesIO(), format="png")  # * Draws the layout.
    legend = panels[0].get_legend()
    assert legend is not None
    assert legend.get_window_extent().y1 < panels[0].get_window_extent().y0


def test_the_paired_table_has_a_dash_at_the_zero_shift() -> None:
    """The zero shift carries no claim, so its cell is empty."""
    by_shift = {0.0: Measurement(0.01, 0.02, 12), 0.5: Measurement(0.02, 0.01, 12)}
    assert effects._without_zero_shift(by_shift) == [None, by_shift[0.5]]


def test_the_small_shift_text_names_the_metrics_beyond_the_tolerance() -> None:
    """The text lists the metrics beyond their tolerance, largest deviation first."""
    paired = {
        "ks": {0.2: Measurement(-0.002, 0.001, 12), 0.5: Measurement(0.0, 0.01, 12)},
        "js kde": {0.2: Measurement(-0.016, 0.001, 12)},
        "w1": {0.2: Measurement(-0.001, 0.001, 12)},
    }
    text = effects._small_shift_text(paired, [0.5, 1.0, 2.0])
    assert "1 of 3 metrics, largest first: JS, KDE (-8.0 ± 0.5%)." in text
    assert "smallest claimed shift of 0.5" in text
    assert (
        effects._small_shift_text({"ks": {0.5: Measurement(0.0, 0.01, 2)}}, [0.5]) == ""
    )


def test_cases_above_name_the_worst_and_must_exceed_the_allowance() -> None:
    """A claim above a reference needs every deviation beyond its allowance."""
    cases = [Case("first", 0.5, 0.1, 0.2), Case("second", 0.31, 0.1, 0.2)]
    claim = claim_cases_above(
        "Above.",
        cases,
        tolerance_text="20% of the truth",
        replicates=None,
    )
    assert claim.consistent
    assert claim.expected == (
        "above the truth by more than 20% of the truth plus three standard errors"
    )
    assert claim.observed == "worst of 2: second: +0.31 against more than 0.3"
    failing = claim_cases_above(
        "Above.",
        [Case("only", 0.25, 0.1, 0.2)],
        tolerance_text=None,
        replicates=None,
        reference_text="the resolution at k = 3",
    )
    assert not failing.consistent
    assert failing.expected == (
        "above the resolution at k = 3 by more than three standard errors"
    )


def test_a_check_of_two_sessions_says_sessions() -> None:
    """The sentence under a heading names one session or several."""
    assert _rerun_sentence("12a") == "Reruns the check of session 12a."
    assert _rerun_sentence("6 and 11") == "Reruns the checks of sessions 6 and 11."


def test_the_pooled_resolution_and_its_leave_one_out_error() -> None:
    """The resolution is the mean signal over the pooled null width."""
    runs = ResolutionRuns([1.0, 2.0, 3.0], [4.0, 4.0, 4.0])
    resolution = pooled_resolution(runs)
    assert resolution.value == pytest.approx(1.0)
    # * Leave-one-out means of the signals are 2.5, 2 and 1.5 over a width of 2.
    expected_error = math.sqrt(2 / 3 * (0.25**2 * 2))
    assert resolution.error == pytest.approx(expected_error)
    assert resolution.replicates == 3


def test_the_resolution_ratio_is_paired_and_ignores_non_finite_replicates() -> None:
    """Equal runs give exactly one; a doubled signal gives two with its own error."""
    reference = ResolutionRuns([1.0, 2.0, 3.0, math.nan], [1.0, 1.0, 1.0, 1.0])
    same = resolution_ratio(reference, reference)
    assert same.value == pytest.approx(1.0)
    assert same.error == pytest.approx(0.0, abs=1e-12)
    assert same.replicates == 3
    doubled = ResolutionRuns([2.0, 4.0, 6.0, 8.0], [1.0, 1.0, 1.0, 1.0])
    ratio = resolution_ratio(doubled, reference)
    assert ratio.value == pytest.approx(2.0)
    assert ratio.error == pytest.approx(0.0, abs=1e-12)


def test_the_sample_shift_is_the_difference_of_the_means_of_the_first_axis() -> None:
    """The truth of a paired comparison reads the first parameter only."""
    first = jnp.array([[0.0, 5.0], [2.0, 5.0]])
    second = jnp.array([[3.0, 9.0], [5.0, 9.0]])
    assert sample_shift(first, second) == pytest.approx(3.0)


def banana_mutual_information_by_scipy(product: float) -> float:
    """Return the mutual information of the banana by nested adaptive quadrature.

    With y = x₂ + t(z² − 1), the mutual information is the entropy of y minus
    the entropy of a unit normal. The inner integral stops at z = 8, where the
    weight is 1e-14, because a wider range lets the adaptive rule step over the
    narrow peaks that a large twist makes at large z.
    """

    def density(value: float) -> float:
        def integrand(point: float) -> float:
            shifted = value - product * (point**2 - 1.0)
            return math.exp(-0.5 * shifted**2 - 0.5 * point**2) / (2 * math.pi)

        return float(
            scipy.integrate.quad(integrand, -8.0, 8.0, limit=400, epsabs=1e-13)[0]
        )

    def entropy_term(value: float) -> float:
        mass = density(value)
        return -mass * math.log(mass) if mass > 1e-300 else 0.0

    entropy = float(
        scipy.integrate.quad(
            entropy_term,
            -product - 10.0,
            product * 60.0 + 10.0,
            limit=400,
            epsabs=1e-11,
        )[0]
    )
    return entropy - 0.5 * math.log(2 * math.pi * math.e)


@pytest.mark.parametrize("product", [0.3, 1.5])
def test_the_banana_quadrature_matches_nested_adaptive_quadrature(
    product: float,
) -> None:
    """Gauss-Legendre in z with a trapezoid in y equals SciPy's nested quad."""
    expected = banana_mutual_information_by_scipy(product)
    assert diagnostics.banana_mutual_information(product) == pytest.approx(
        expected, abs=1e-7
    )


def test_the_banana_mutual_information_follows_the_small_signal_limit() -> None:
    """For a weak twist, I(y₁; y₂) tends to t²: half the signal variance 2t²."""
    product = 0.01
    value = diagnostics.banana_mutual_information(product)
    assert value == pytest.approx(product**2, rel=5e-3)
    assert diagnostics.banana_mutual_information(
        product, refinement=2.0
    ) == pytest.approx(value, abs=1e-10)


def test_the_fixed_point_iteration_finds_the_modes_of_a_mixture() -> None:
    """Two far components give two modes, two near ones one, nested ones one."""
    identity = jnp.eye(2)[None].repeat(2, axis=0)
    weights = jnp.log(jnp.array([0.5, 0.5]))
    far = jnp.array([[-3.0, 0.0], [3.0, 0.0]])
    near = jnp.array([[-0.5, 0.0], [0.5, 0.0]])
    assert diagnostics.mixture_modes(weights, far, identity) == 2
    assert diagnostics.mixture_modes(weights, near, identity) == 1
    nested = jnp.stack([jnp.eye(2), 9.0 * jnp.eye(2)])
    centred = jnp.zeros((2, 2))
    assert diagnostics.mixture_modes(weights, centred, nested) == 1


def test_the_fixed_point_iteration_counts_the_modes_of_three_components() -> None:
    """Three components on a line, two close and one far, give two modes."""
    identity = jnp.eye(2)[None].repeat(3, axis=0)
    weights = jnp.log(jnp.array([0.3, 0.3, 0.4]))
    means = jnp.array([[-6.0, 0.0], [-5.0, 0.0], [6.0, 0.0]])
    assert diagnostics.mixture_modes(weights, means, identity) == 2


def test_the_floor_contrast_equals_the_package_at_the_package_floor() -> None:
    """Counting at the floor of the package reproduces its mode counts."""
    case = zoo.gaussian_mixture(4.0)
    samples = case.sample(jax.random.key(3), 2_000).samples[0, 0]
    package = marginal_modes(samples, None, case.space)
    for position in range(2):
        parameter = Parameter(name="value")
        counted = diagnostics.modes_at_floor(
            samples[:, position], parameter, MODE_FLOOR
        )
        assert float(counted) == float(package.modes[position])


def test_a_lower_floor_never_counts_fewer_modes() -> None:
    """A heavy tail can only add peaks when the floor falls."""
    samples = jax.random.t(jax.random.key(5), 2.0, (3_000, 1))
    parameter = Parameter(name="value")
    strict = float(diagnostics.modes_at_floor(samples[:, 0], parameter, MODE_FLOOR))
    lenient = float(diagnostics.modes_at_floor(samples[:, 0], parameter, 0.001))
    assert lenient >= strict >= 1.0


def test_a_band_claim_names_the_entry_with_the_least_room() -> None:
    """The verdict needs every spread in its band; the text names the closest."""
    entries = [
        ("first", Measurement(1.0, 0.1)),
        ("second", Measurement(1.35, 0.1)),
    ]
    inside = diagnostics._limits_claim(
        "Band.", entries, low=0.7, high=1.4, expected="between 0.7 and 1.4"
    )
    assert inside.consistent
    assert inside.observed == "worst of 2: second: 1.35 ± 0.10"
    outside = diagnostics._limits_claim(
        "Band.",
        [*entries, ("third", Measurement(0.6, 0.1))],
        low=0.7,
        high=1.4,
        expected="between 0.7 and 1.4",
    )
    assert not outside.consistent
    assert outside.observed == "worst of 3: third: 0.60 ± 0.10"
    limit = diagnostics._limits_claim(
        "Limit.", entries, low=None, high=0.6, expected="at most 0.6"
    )
    assert not limit.consistent
    assert limit.observed == "worst of 2: second: 1.35 ± 0.10"
    undefined = diagnostics._limits_claim(
        "Band.",
        [("nan", Measurement(math.nan, 0.1))],
        low=0.7,
        high=1.4,
        expected="between 0.7 and 1.4",
    )
    assert not undefined.consistent


def test_the_cost_claim_checks_each_costly_timing_against_the_model() -> None:
    """A per-draw row of 10 ms or more is checked, and one miss fails the claim."""
    close = costs._Prediction("energy", 3_000, 16, 1.0, 0.02, 0.025)
    low = costs._Prediction("energy", 10_000, 8, 1.0, 0.02, 0.01)
    cheap = costs._Prediction("moments", 3_000, 16, 16.0, 0.005, 0.05)
    once = costs._Prediction("gmm_bic", 3_000, 16, 1.0, 0.7, 0.4)
    assert close.factor == pytest.approx(1.25)
    assert low.factor == pytest.approx(2.0)
    assert close.checked
    assert not cheap.checked
    assert not once.checked
    _, passing = costs._grid_section([close, cheap, once])
    assert passing.consistent
    assert passing.full_only
    assert "worst of 1 checked rows of 3" in passing.observed
    table, failing = costs._grid_section([close, low, cheap, once])
    assert not failing.consistent
    assert "energy at 10 000 samples and 8 parameters, factor 2.00" in failing.observed
    assert len(table.rows) == 4
    _, empty = costs._grid_section([cheap, once])
    assert empty.consistent
    assert empty.observed == "no checked row of 2"


def test_a_ratio_of_half_sample_errors_combines_both_errors() -> None:
    """The error joins the scatter of the errors with that of the true scatter."""
    errors = jnp.asarray([1.1, 0.9, 1.0, 1.2])
    ratio = engines._ratio(errors, jnp.asarray(1.0), 200)
    mean = 1.05
    error_of_mean = math.sqrt(0.05 / 3) / math.sqrt(4)
    relative = math.hypot(error_of_mean / mean, 1 / math.sqrt(2 * 199))
    assert ratio.value == pytest.approx(mean)
    assert ratio.error == pytest.approx(mean * relative)
    assert ratio.replicates == 4
    scaled = engines._scaled(ratio, 0.5)
    assert scaled.value == pytest.approx(0.5 * mean)
    assert scaled.error == pytest.approx(0.5 * mean * relative)


def test_the_flags_of_a_zoo_case_name_the_required_and_the_excluded_rows() -> None:
    """A required row must flag, an excluded row must not, a quiet case flags none."""
    cases = {case.name: case for case in engines._flag_cases()}
    banana = cases["Banana"]
    required = set(banana.required)
    assert engines._flags_met(banana, required)
    assert engines._flags_met(banana, required | {("variance", "x0")})
    assert not engines._flags_met(banana, required - {("gmm_bic", "x0,x1")})
    assert not engines._flags_met(banana, required | {("mode_dip", "x0")})
    assert not engines._flags_met(banana, required | {("spearman", "x0,x1")})
    gaussian = cases["Gaussian with correlations"]
    assert engines._flags_met(gaussian, set())
    assert not engines._flags_met(gaussian, {("skewness", "x0")})


def test_counts_and_lists_are_written_in_plain_text() -> None:
    """One problem is singular, and lists end with and, or or."""
    assert engines._plural(1, "problem") == "1 problem"
    assert engines._plural(4, "problem") == "4 problems"
    assert engines._dimension_text((2, 4, 8)) == "2, 4 and 8"
    assert engines._dimension_text((2,)) == "2"
    assert engines._value_text((0.5, 1.0)) == "0.5 or 1"
    assert engines._value_text((1.0,)) == "1"


def test_a_failing_claim_gives_a_non_zero_status(
    build_module: ModuleType, tmp_path: Path
) -> None:
    """The page is written, and the exit status is 1 if an evaluated claim fails."""
    failing = make_check("fails", (claim_true("Breaks.", "yes", "no", False),))
    passing = make_check("passes", (claim_true("Holds.", "yes", "yes", True),))
    topics = [("Topic", (passing, failing))]
    status = build_module.main(
        ["--quick", "--output-dir", str(tmp_path)], topics=topics
    )
    assert status == 1
    page = (tmp_path / "validation.md").read_text(encoding="utf-8")
    assert "(validation-fails)=" in page
    assert "not consistent" in page
    payload = json.loads((tmp_path / "validation" / "results.json").read_text())
    assert [entry["label"] for entry in payload["checks"]] == ["passes", "fails"]
    assert (
        build_module.main(
            ["--quick", "--only", "passes", "--output-dir", str(tmp_path / "ok")],
            topics=topics,
        )
        == 0
    )


def test_a_full_only_failure_passes_in_quick_mode_and_fails_in_full(
    build_module: ModuleType, tmp_path: Path
) -> None:
    """A claim that declares itself full-only does not fail a quick build."""
    check = make_check(
        "slow-claim", (claim_true("Breaks.", "yes", "no", False, full_only=True),)
    )
    topics = [("Topic", (check,))]
    assert (
        build_module.main(
            ["--quick", "--output-dir", str(tmp_path / "quick")], topics=topics
        )
        == 0
    )
    assert (
        build_module.main(["--output-dir", str(tmp_path / "full")], topics=topics) == 1
    )


def test_a_partial_run_writes_nothing_without_an_output_directory(
    build_module: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only an explicit output directory lets a partial run write the page."""
    monkeypatch.setattr(build_module, "REPOSITORY", tmp_path)
    passing = make_check("passes", (claim_true("Holds.", "yes", "yes", True),))
    topics = [("Topic", (passing,))]
    assert build_module.main(["--quick", "--only", "passes"], topics=topics) == 0
    assert not (tmp_path / "docs").exists()
    with pytest.raises(SystemExit, match="unknown check labels"):
        build_module.main(["--quick", "--only", "absent"], topics=topics)


def test_quick_build_writes_the_page_the_figures_and_the_numbers(
    build_module: ModuleType, tmp_path: Path
) -> None:
    """Every check appears with its target, every figure in both themes."""
    status = build_module.main(["--quick", "--output-dir", str(tmp_path)])
    assert status == 0
    page = (tmp_path / "validation.md").read_text(encoding="utf-8")
    payload = json.loads(
        (tmp_path / "validation" / "results.json").read_text(encoding="utf-8")
    )
    topics = build_module.load_topics()
    labels = [check.label for _, checks in topics for check in checks]
    assert len(labels) >= 16
    assert "NaN null draws" in page
    assert "Along arcs the MMD squared of the bimodal circle" in page
    for label in labels:
        assert f"(validation-{label})=" in page
    assert [entry["label"] for entry in payload["checks"]] == labels
    assert payload["stamp"]["Mode"] == "quick"
    figure_labels = [
        block["label"]
        for entry in payload["checks"]
        for block in entry["blocks"]
        if isinstance(block, dict) and block.get("type") == "figure"
    ]
    assert figure_labels
    for figure_label in figure_labels:
        for theme in ("light", "dark"):
            path = tmp_path / "validation" / "figures" / f"{figure_label}-{theme}.svg"
            assert path.stat().st_size > 0
    roles = [match["target"] for match in ROLE_PATTERN.finditer(page)]
    roles.extend(
        match["target"]
        for _, checks in topics
        for check in checks
        for entry in check.supports
        for match in ROLE_PATTERN.finditer(entry)
    )
    assert len(roles) > 20
    for target in roles:
        defined_at = defining_path(target.lstrip("~"))
        if defined_at is not None:
            assert defined_at == target.lstrip("~")
    measurements = list(walk_measurements(payload))
    assert measurements
    for measurement in measurements:
        assert measurement["value"] is not None
        assert math.isfinite(measurement["value"])
        assert measurement["error"] is None or math.isfinite(measurement["error"])
