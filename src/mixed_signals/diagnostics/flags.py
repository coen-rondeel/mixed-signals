"""Default flag thresholds of the diagnostic rows, and the evaluation of one row.

A diagnostic row flags when its effect size passes the threshold of its
metric and, where the row has a null or a p-value of its own, its σ_sig
reaches :data:`SIGNIFICANCE_LEVEL`. The effect size is the value minus the
mean of the null for rows with a surrogate null, and the value itself
otherwise, so a finite-sample bias that the null carries does not count as a
departure. The YAML config of a pipeline overrides these defaults.

Basis of each default
    ``skewness``: 0.5, the skewness of the zoo's normal railed 1σ inside its
    bound (0.57), which also anchors ``anderson_darling`` at 0.005 (its
    A²/n is 0.006). ``kurtosis``: 1, the size of g₂ for an equal mixture of unit
    normals at a separation of 3.1, −2a⁴/(1 + a²)² with a = 1.55.
    ``mardia_skewness`` and ``mardia_kurtosis`` carry the marginal values
    over, since b₁ = Σ g₁² and b₂ − D(D + 2) = Σ g₂ for independent axes:
    0.25 and 1. On a narrow circle the circular skewness and kurtosis tend
    to −2√2·g₁ and 2·g₂ (the Taylor expansion of the trigonometric moments,
    checked on the zoo), so ``circular_skewness`` takes 2√2 · 0.5 and
    ``circular_kurtosis`` 2. ``spearman``, ``distance_correlation`` and
    ``mutual_information``: 0.1, 0.1 and 0.05 nats, about the excesses of the
    zoo's banana with b = 0.003, whose Gaussian approximation is off by
    0.08 nats (KL). ``mode_dip``:
    :data:`~mixed_signals.diagnostics.modes.MODE_SEPARATION`. ``gmm_bic``:
    ΔBIC of 10, very strong evidence on the scale of Kass and Raftery (1995),
    which Fraley and Raftery (2002) use for mixtures. ``railing_lower`` and
    ``railing_upper``: 0.02, the edge-to-peak ratio at which pe-configurator
    reports railing. ``ess``: below 400, the minimum of Vehtari et al.
    (2021). ``split_chain_ks`` and ``split_chain_kuiper_ks``: 0.1, a shift of
    about 0.25σ between the halves by the σ_eff map of KS.
Significance
    :data:`SIGNIFICANCE_LEVEL` is 2: the default 200 null draws resolve σ_sig
    up to 2.58 before a fit to the tail of the null.
Informational rows
    ``pearson``, ``mode_count``, ``tau``, ``rhat``, ``acceptance_rate`` and
    ``swap_rate`` have no threshold and never flag.

References:
    :cite:t:`kass1995bayes`; :cite:t:`fraley2002model`;
    :cite:t:`vehtari2021rank`; :cite:t:`estelles2023peconfigurator`;
    :cite:t:`phipson2010permutation`; :cite:t:`cowan2011asymptotic`.
"""

import math
from collections.abc import Mapping
from enum import StrEnum
from types import MappingProxyType
from typing import Final, NamedTuple

import jax.numpy as jnp
from jax.typing import ArrayLike

from mixed_signals.diagnostics.modes import MODE_SEPARATION
from mixed_signals.sigma.significance import Tail, empirical_p_value, sigma_sig

SIGNIFICANCE_LEVEL: Final = 2.0
"""σ_sig that a row with a null or a p-value of its own must reach to flag."""


class Side(StrEnum):
    """Which effect sizes pass a threshold.

    ``above``: at least the threshold. ``both``: at least the threshold in
    absolute value. ``below``: less than the threshold.
    """

    ABOVE = "above"
    BOTH = "both"
    BELOW = "below"


class Threshold(NamedTuple):
    """The flag threshold of one diagnostic metric."""

    value: float
    """The threshold of the effect size."""
    side: Side
    """Which effect sizes pass it."""
    tail: Tail | None
    """Tail of the p-value of a row with a null; ``None`` for rows without one."""


THRESHOLDS: Final[Mapping[str, Threshold]] = MappingProxyType(
    {
        "skewness": Threshold(0.5, Side.BOTH, Tail.TWO_SIDED),
        "circular_skewness": Threshold(math.sqrt(2), Side.BOTH, Tail.TWO_SIDED),
        "kurtosis": Threshold(1.0, Side.BOTH, Tail.TWO_SIDED),
        "circular_kurtosis": Threshold(2.0, Side.BOTH, Tail.TWO_SIDED),
        "anderson_darling": Threshold(0.005, Side.ABOVE, Tail.UPPER),
        "mardia_skewness": Threshold(0.25, Side.ABOVE, Tail.UPPER),
        "mardia_kurtosis": Threshold(1.0, Side.BOTH, Tail.TWO_SIDED),
        "spearman": Threshold(0.1, Side.BOTH, Tail.TWO_SIDED),
        "distance_correlation": Threshold(0.1, Side.ABOVE, Tail.UPPER),
        "mutual_information": Threshold(0.05, Side.ABOVE, Tail.UPPER),
        "mode_dip": Threshold(MODE_SEPARATION, Side.ABOVE, Tail.UPPER),
        "gmm_bic": Threshold(10.0, Side.ABOVE, None),
        "railing_lower": Threshold(0.02, Side.ABOVE, None),
        "railing_upper": Threshold(0.02, Side.ABOVE, None),
        "ess": Threshold(400.0, Side.BELOW, None),
        "split_chain_ks": Threshold(0.1, Side.ABOVE, Tail.UPPER),
        "split_chain_kuiper_ks": Threshold(0.1, Side.ABOVE, Tail.UPPER),
    }
)
"""Default threshold of every flagging diagnostic metric.

:meta hide-value:
"""


class Evaluation(NamedTuple):
    """Effect size, significance and verdict of one diagnostic row."""

    effect: float
    """Value minus the null mean, or the value for rows without a null."""
    p_value: float
    """Empirical p-value against the null, or the row's own; NaN without one."""
    sigma_sig: float
    """The p-value as a Gaussian significance; NaN without a p-value."""
    flagged: bool
    """Whether the effect passes the threshold and σ_sig the level."""


def evaluate(
    metric: str,
    value: float,
    null_draws: ArrayLike | None = None,
    *,
    p_value: float | None = None,
    thresholds: Mapping[str, Threshold] = THRESHOLDS,
    level: float = SIGNIFICANCE_LEVEL,
) -> Evaluation:
    """Return the effect size, significance and flag of one diagnostic row.

    Args:
        metric: Name of the row's metric, a key of ``thresholds``.
        value: The row's value.
        null_draws: The row's surrogate null draws, if it has a null.
        p_value: The row's own p-value, as the split-chain tests bring;
            used when there are no null draws.
        thresholds: Threshold of every metric; the defaults when not given.
        level: σ_sig that a row with a null or a p-value must reach.

    Returns:
        The effect size, the p-value, σ_sig and whether the row flags. A NaN
        effect never flags.

    Raises:
        KeyError: If the metric has no threshold.

    References:
        :cite:t:`phipson2010permutation`; :cite:t:`cowan2011asymptotic`.
    """
    threshold = thresholds[metric]
    tail = threshold.tail if threshold.tail is not None else Tail.UPPER
    effect = float(value)
    probability = math.nan
    if null_draws is not None:
        draws = jnp.asarray(null_draws, dtype=jnp.float64)
        finite = draws[jnp.isfinite(draws)]
        effect = float(value) - float(jnp.mean(finite)) if finite.size else math.nan
        probability = float(empirical_p_value(value, draws, tail))
    elif p_value is not None:
        probability = float(p_value)
    tested = null_draws is not None or p_value is not None
    significance = float(sigma_sig(probability, tail)) if tested else math.nan
    if threshold.side is Side.ABOVE:
        passes = effect >= threshold.value
    elif threshold.side is Side.BOTH:
        passes = abs(effect) >= threshold.value
    else:
        passes = effect < threshold.value
    # * A NaN σ_sig of a tested row fails the comparison, so it never flags.
    resolved = significance >= level if tested else True
    return Evaluation(
        effect=effect,
        p_value=probability,
        sigma_sig=significance,
        flagged=bool(passes and resolved and not math.isnan(effect)),
    )
