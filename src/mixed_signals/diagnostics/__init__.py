"""Single-posterior diagnostics: shape, dependence, modes, railing and chain health.

Each module gives the statistics of one posterior, and
:mod:`mixed_signals.diagnostics.flags` decides whether a row flags::

    import jax
    from mixed_signals import zoo
    from mixed_signals.diagnostics import dependence_statistics, evaluate
    from mixed_signals.nulls.surrogates import gaussian_surrogates

    case = zoo.banana()
    samples = case.sample(jax.random.key(0), 2_000).samples[0, 0]
    key = jax.random.key(1)
    observed = dependence_statistics(samples, None, case.space, key=key)
    surrogates = gaussian_surrogates(samples, None, case.space, size=2_000, key=key)
    null = jax.lax.map(
        lambda values: dependence_statistics(values, None, case.space, key=key),
        surrogates,
    )
    evaluate(
        "mutual_information",
        float(observed.mutual_information[0]),
        null.mutual_information[:, 0],
    )  # flagged: the banana's curved dependence

:mod:`~mixed_signals.diagnostics.shape` holds the marginal and circular
moments, the one-sample Anderson-Darling distance and Mardia's moments;
:mod:`~mixed_signals.diagnostics.dependence` Pearson's r and the excess of
Spearman's ρ_s, the distance correlation and the mutual information over their
Gaussian values; :mod:`~mixed_signals.diagnostics.modes` GMM-BIC and the modes
of one-dimensional kernel estimates; :mod:`~mixed_signals.diagnostics.railing`
the railing scores; :mod:`~mixed_signals.diagnostics.chains` the health of MCMC
chains; and :mod:`~mixed_signals.diagnostics.flags` the default thresholds.
Shape, dependence and mode statistics are compared with Gaussian surrogates
(:mod:`mixed_signals.nulls.surrogates`).
"""

from mixed_signals.diagnostics._common import CIRCLE_WIDTH_LIMIT, parameter_pairs
from mixed_signals.diagnostics.chains import (
    ChainHealth,
    chain_health,
    split_chain_metric_names,
)
from mixed_signals.diagnostics.dependence import (
    MUTUAL_INFORMATION_NEIGHBOURS,
    PAIR_CAP,
    Dependence,
    dependence_statistics,
    gaussian_distance_correlation,
    gaussian_mutual_information,
    gaussian_spearman,
)
from mixed_signals.diagnostics.flags import (
    SIGNIFICANCE_LEVEL,
    THRESHOLDS,
    Evaluation,
    Side,
    Threshold,
    evaluate,
)
from mixed_signals.diagnostics.modes import (
    MAX_COMPONENTS,
    MODE_FLOOR,
    MODE_SEPARATION,
    GmmBic,
    ModeCounts,
    circle_centres,
    discriminant_directions,
    gmm_bic,
    marginal_modes,
    projection_modes,
)
from mixed_signals.diagnostics.railing import (
    RailingScores,
    railed_parameters,
    railing_scores,
)
from mixed_signals.diagnostics.shape import (
    MardiaMoments,
    MarginalMoments,
    anderson_darling_normality,
    mardia_moments,
    marginal_moments,
    moment_metric_names,
)

__all__ = [
    "CIRCLE_WIDTH_LIMIT",
    "MAX_COMPONENTS",
    "MODE_FLOOR",
    "MODE_SEPARATION",
    "MUTUAL_INFORMATION_NEIGHBOURS",
    "PAIR_CAP",
    "SIGNIFICANCE_LEVEL",
    "THRESHOLDS",
    "ChainHealth",
    "Dependence",
    "Evaluation",
    "GmmBic",
    "MardiaMoments",
    "MarginalMoments",
    "ModeCounts",
    "RailingScores",
    "Side",
    "Threshold",
    "anderson_darling_normality",
    "chain_health",
    "circle_centres",
    "dependence_statistics",
    "discriminant_directions",
    "evaluate",
    "gaussian_distance_correlation",
    "gaussian_mutual_information",
    "gaussian_spearman",
    "gmm_bic",
    "mardia_moments",
    "marginal_modes",
    "marginal_moments",
    "moment_metric_names",
    "parameter_pairs",
    "projection_modes",
    "railed_parameters",
    "railing_scores",
    "split_chain_metric_names",
]
