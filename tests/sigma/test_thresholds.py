"""Tests for the default effect-size thresholds, the tails and the verdict of a row."""

import math

import jax.numpy as jnp
import pytest

from mixed_signals.diagnostics import flags
from mixed_signals.sigma import thresholds
from mixed_signals.sigma.maps import (
    KERNEL_METRICS,
    QUADRATURE_METRICS,
    SIGMA_EFF_MAPS,
    SLICED_METRICS,
    SUBTRACTION_SPACES,
)
from mixed_signals.sigma.significance import Tail
from mixed_signals.sigma.thresholds import (
    LOG_R_EFF_THRESHOLD,
    R_EFF_METRICS,
    SIGMA_EFF_THRESHOLD,
    SIGNED_METRICS,
    SIGNIFICANCE_LEVEL,
    THRESHOLDS,
    Verdict,
    comparison_tail,
    verdict,
)
from mixed_signals.zoo import closed_forms

SHAPE_TERMS = ("kl_shape", "w2_shape", "hellinger_shape")


def test_the_defaults_are_the_documented_ones() -> None:
    """0.2 for σ_eff, 0.14 for the log of r_eff and a σ_sig of 2."""
    assert SIGMA_EFF_THRESHOLD == 0.2
    assert LOG_R_EFF_THRESHOLD == 0.14
    assert SIGNIFICANCE_LEVEL == 2.0
    assert R_EFF_METRICS == ("width_ratio", "kl_shape", "w2_shape", "hellinger_shape")
    assert SIGNED_METRICS == ("mean_shift", "width_ratio")


def test_the_flags_module_keeps_exporting_the_level() -> None:
    """The diagnostics read the level from here and still export the name."""
    assert flags.SIGNIFICANCE_LEVEL is thresholds.SIGNIFICANCE_LEVEL
    assert flags.SIGNIFICANCE_LEVEL == 2.0


def test_the_thresholds_follow_from_the_js_divergence_of_a_small_shift() -> None:
    """A 0.2σ shift has JS = 0.0072 bit; the width ratios of that JS give 0.14."""
    js_nats = float(closed_forms.normal_js(0.0, 1.0, SIGMA_EFF_THRESHOLD, 1.0))
    assert js_nats / math.log(2) == pytest.approx(0.0072, abs=5e-5)
    for ratio in (1.153, 0.867):
        wider = float(closed_forms.normal_js(0.0, 1.0, 0.0, ratio))
        assert wider == pytest.approx(js_nats, rel=1e-2)
        assert round(abs(math.log(ratio)), 2) == LOG_R_EFF_THRESHOLD


def test_every_metric_with_a_map_has_a_threshold() -> None:
    """The table covers every σ_eff and r_eff metric, and no other metric."""
    sigma_metrics = {
        "mean_shift",
        *SIGMA_EFF_MAPS,
        *KERNEL_METRICS,
        *SLICED_METRICS,
        *QUADRATURE_METRICS,
    }
    assert set(THRESHOLDS) == sigma_metrics | set(R_EFF_METRICS)
    assert set(THRESHOLDS) == set(SUBTRACTION_SPACES)
    for metric in sigma_metrics:
        assert THRESHOLDS[metric] == SIGMA_EFF_THRESHOLD, metric
    for metric in R_EFF_METRICS:
        assert THRESHOLDS[metric] == LOG_R_EFF_THRESHOLD, metric
    assert {"js", "ad", "energy", "mmd", "sliced_w2", "sliced_sinkhorn"} <= set(
        THRESHOLDS
    )


def test_mean_shift_keeps_its_threshold_without_a_map_entry() -> None:
    """The key set and order are the old ones, with mean_shift named explicitly."""
    assert "mean_shift" not in SIGMA_EFF_MAPS
    assert THRESHOLDS["mean_shift"] == SIGMA_EFF_THRESHOLD
    assert list(THRESHOLDS) == [
        "mean_shift",
        "w1",
        "w2",
        "w2_location",
        "kl",
        "kl_location",
        "ks",
        "kuiper_ks",
        "hellinger",
        "hellinger_location",
        "sinkhorn",
        "energy",
        "mmd",
        "sliced_sinkhorn",
        "sliced_w2",
        "ad",
        "js",
        "width_ratio",
        "kl_shape",
        "w2_shape",
        "hellinger_shape",
    ]


def test_the_tail_is_two_sided_for_the_signed_metrics_only() -> None:
    """Mean shift and width ratio count departures in both directions."""
    for metric in THRESHOLDS:
        expected = Tail.TWO_SIDED if metric in SIGNED_METRICS else Tail.UPPER
        assert comparison_tail(metric) is expected, metric
    assert comparison_tail("mean_shift") is Tail.TWO_SIDED
    assert comparison_tail("ks") is Tail.UPPER


def test_mean_shift_fails_on_the_absolute_shift() -> None:
    """The mean shift is signed, so a shift of −0.3σ fails as one of +0.3σ does."""
    for shift in (0.3, -0.3):
        result = verdict("mean_shift", sigma_eff=shift, sigma_sig=3.0)
        assert result == Verdict(effect=0.3, resolved=True, failed=True)
    small = verdict("mean_shift", sigma_eff=-0.1, sigma_sig=5.0)
    assert small == Verdict(effect=0.1, resolved=True, failed=False)


def test_other_sigma_rows_fail_on_sigma_eff_itself() -> None:
    """A negative σ_eff, a value below the null, never fails."""
    for metric in ("ks", "kl", "energy", "js", "ad", "mmd", "sliced_w2"):
        assert verdict(metric, sigma_eff=-0.9, sigma_sig=9.0) == Verdict(
            effect=-0.9, resolved=True, failed=False
        )
        assert verdict(metric, sigma_eff=0.19, sigma_sig=9.0).failed is False
        assert verdict(metric, sigma_eff=0.2, sigma_sig=9.0).failed is True
        assert verdict(metric, sigma_eff=0.7, sigma_sig=9.0).failed is True


def test_width_ratio_fails_on_the_absolute_log() -> None:
    """A ratio r and 1/r are equally far from one."""
    for ratio in (1.2, 1 / 1.2):
        result = verdict("width_ratio", r_eff=ratio, sigma_sig=4.0)
        assert result.effect == pytest.approx(math.log(1.2), rel=1e-12)
        assert result.failed
    assert not verdict("width_ratio", r_eff=1.1, sigma_sig=4.0).failed
    assert not verdict("width_ratio", r_eff=1 / 1.1, sigma_sig=4.0).failed
    assert verdict("width_ratio", r_eff=1.0, sigma_sig=4.0).effect == 0.0


@pytest.mark.parametrize("metric", SHAPE_TERMS)
def test_shape_terms_fail_only_above_the_null(metric: str) -> None:
    """The sign of the subtracted value gives the sign of the effect."""
    above = verdict(metric, r_eff=1.2, subtracted=0.05, sigma_sig=4.0)
    assert above.effect == pytest.approx(math.log(1.2), rel=1e-12)
    assert above.failed
    narrower = verdict(metric, r_eff=1 / 1.2, subtracted=0.05, sigma_sig=4.0)
    assert narrower.failed
    below = verdict(metric, r_eff=1 / 1.2, subtracted=-0.05, sigma_sig=4.0)
    assert below.effect == pytest.approx(-math.log(1.2), rel=1e-12)
    assert not below.failed
    zero = verdict(metric, r_eff=1.0, subtracted=0.0, sigma_sig=4.0)
    assert zero == Verdict(effect=0.0, resolved=True, failed=False)


def test_a_row_needs_the_significance_level_to_fail() -> None:
    """The level itself resolves; a lower σ_sig does not, whatever the effect."""
    at_level = verdict("ks", sigma_eff=1.0, sigma_sig=SIGNIFICANCE_LEVEL)
    assert at_level == Verdict(effect=1.0, resolved=True, failed=True)
    below = verdict("ks", sigma_eff=1.0, sigma_sig=1.99)
    assert below == Verdict(effect=1.0, resolved=False, failed=False)
    assert verdict("ks", sigma_eff=1.0, sigma_sig=-math.inf).resolved is False
    assert verdict("ks", sigma_eff=1.0, sigma_sig=3.0, level=3.5).failed is False
    assert verdict("ks", sigma_eff=1.0, sigma_sig=3.0, level=2.5).failed is True


def test_nan_never_fails() -> None:
    """A NaN effect size or a NaN σ_sig leaves the row unresolved or without effect."""
    nan = math.nan
    undefined = verdict("ks", sigma_eff=nan, sigma_sig=9.0)
    assert math.isnan(undefined.effect)
    assert undefined.resolved
    assert not undefined.failed
    assert not verdict("mean_shift", sigma_eff=nan, sigma_sig=9.0).failed
    assert not verdict("ks", sigma_eff=1.0, sigma_sig=nan).failed
    assert verdict("ks", sigma_eff=1.0, sigma_sig=nan).resolved is False
    for ratio in (nan, 0.0, -1.0, math.inf):
        result = verdict("width_ratio", r_eff=ratio, sigma_sig=9.0)
        assert math.isnan(result.effect), ratio
        assert not result.failed, ratio
    nan_shape = verdict("kl_shape", r_eff=1.5, subtracted=nan, sigma_sig=9.0)
    assert math.isnan(nan_shape.effect)
    assert not nan_shape.failed


def test_thresholds_and_level_can_be_overridden() -> None:
    """A mapping of the caller replaces the defaults."""
    custom = {"ks": 0.5, "width_ratio": 0.05}
    assert (
        verdict("ks", sigma_eff=0.4, sigma_sig=3.0, thresholds=custom).failed is False
    )
    assert verdict("ks", sigma_eff=0.5, sigma_sig=3.0, thresholds=custom).failed is True
    assert verdict("width_ratio", r_eff=1.06, sigma_sig=3.0, thresholds=custom).failed
    with pytest.raises(KeyError, match="no threshold for 'mean_shift'"):
        verdict("mean_shift", sigma_eff=1.0, sigma_sig=3.0, thresholds=custom)


def test_arrays_of_one_value_are_accepted() -> None:
    """Scalar JAX arrays from the maps enter without conversion by the caller."""
    result = verdict(
        "kl_shape",
        r_eff=jnp.asarray(1.3),
        subtracted=jnp.asarray(0.04),
        sigma_sig=jnp.asarray(2.5),
    )
    assert result.failed
    assert isinstance(result.effect, float)
    assert isinstance(result.resolved, bool)
    assert isinstance(result.failed, bool)


def test_a_metric_without_a_threshold_is_a_key_error() -> None:
    """Diagnostics and unknown names have no comparison threshold."""
    for metric in ("skewness", "c2st", ""):
        with pytest.raises(KeyError, match=f"no threshold for {metric!r}"):
            verdict(metric, sigma_eff=1.0, sigma_sig=3.0)


def test_an_effect_size_that_does_not_fit_the_metric_is_a_value_error() -> None:
    """σ_eff belongs to the σ_eff rows, r_eff to the width rows, and one is enough."""
    with pytest.raises(ValueError, match="'ks' is sigma_eff"):
        verdict("ks", r_eff=1.2, sigma_sig=3.0)
    with pytest.raises(ValueError, match="'ks' is sigma_eff"):
        verdict("ks", sigma_sig=3.0)
    with pytest.raises(ValueError, match="'ks' is sigma_eff"):
        verdict("ks", sigma_eff=0.5, r_eff=1.2, sigma_sig=3.0)
    with pytest.raises(ValueError, match="'width_ratio' is r_eff"):
        verdict("width_ratio", sigma_eff=0.5, sigma_sig=3.0)
    with pytest.raises(ValueError, match="'width_ratio' is r_eff"):
        verdict("width_ratio", sigma_sig=3.0)
    with pytest.raises(ValueError, match="'kl_shape' is r_eff"):
        verdict("kl_shape", sigma_eff=0.5, r_eff=1.2, subtracted=0.1, sigma_sig=3.0)
    for metric in SHAPE_TERMS:
        with pytest.raises(ValueError, match=f"{metric!r} needs its subtracted value"):
            verdict(metric, r_eff=1.2, sigma_sig=3.0)
    # * The subtracted value of a metric without a shape term is ignored.
    assert verdict("ks", sigma_eff=0.5, subtracted=-1.0, sigma_sig=3.0).failed
