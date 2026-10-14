"""Density estimators: Gaussian, bounded KDE, kNN and GMM, with two placeholders.

Every estimator fits the samples of one posterior of one problem, respects the
topology of its :class:`~mixed_signals.core.space.ParamSpace`, and gives log
densities with respect to the coordinates; ``jax.vmap`` fits one estimator per
problem and posterior::

    import jax
    import jax.numpy as jnp
    from mixed_signals import zoo
    from mixed_signals.estimators import KernelDensity

    case = zoo.railed_gaussians([1.0], param_count=1)
    sample_set = case.sample(jax.random.key(0), 10_000)
    kde = KernelDensity.fit(sample_set.samples[0, 0], None, case.space)
    kde.log_density(jnp.array([[0.0], [1.0]]))  # bound and peak: [-1.14, -0.74]
    kde.resolved_settings().to_json()  # '{"bandwidth":0.1678...}'

:mod:`~mixed_signals.estimators.gaussian` holds the moment-matched Gaussian,
:mod:`~mixed_signals.estimators.kde` the bounded kernel density estimate,
:mod:`~mixed_signals.estimators.knn` the nearest-neighbour distances and log
densities, and :mod:`~mixed_signals.estimators.gmm` the Gaussian mixture.
:mod:`~mixed_signals.estimators.flow` and
:mod:`~mixed_signals.estimators.classifier` are placeholders that raise
``NotImplementedError``. :mod:`~mixed_signals.estimators.base` holds the shared
settings and checks.
"""

from mixed_signals.estimators.base import (
    IMAGE_REACH,
    RAYLEIGH_THRESHOLD,
    CovarianceFactor,
    Estimator,
    EstimatorSettings,
    factor_covariance,
)
from mixed_signals.estimators.classifier import ClassifierRatio, ClassifierSettings
from mixed_signals.estimators.flow import FlowDensity, FlowSettings
from mixed_signals.estimators.gaussian import GaussianDensity, GaussianSettings
from mixed_signals.estimators.gmm import DEFAULT_STARTS, MixtureDensity, MixtureSettings
from mixed_signals.estimators.kde import (
    MINIMUM_SAMPLE_SIZES,
    SILVERMAN,
    KernelDensity,
    KernelDensitySettings,
)
from mixed_signals.estimators.knn import (
    WEIGHT_TOLERANCE,
    NearestNeighbours,
    NearestNeighbourSettings,
)

__all__ = [
    "DEFAULT_STARTS",
    "IMAGE_REACH",
    "MINIMUM_SAMPLE_SIZES",
    "RAYLEIGH_THRESHOLD",
    "SILVERMAN",
    "WEIGHT_TOLERANCE",
    "ClassifierRatio",
    "ClassifierSettings",
    "CovarianceFactor",
    "Estimator",
    "EstimatorSettings",
    "FlowDensity",
    "FlowSettings",
    "GaussianDensity",
    "GaussianSettings",
    "KernelDensity",
    "KernelDensitySettings",
    "MixtureDensity",
    "MixtureSettings",
    "NearestNeighbourSettings",
    "NearestNeighbours",
    "factor_covariance",
]
