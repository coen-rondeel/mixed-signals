"""Validation zoo: distributions with known answers for tests and calibration.

Each builder returns a case: posteriors whose divergences, shifts, width
ratios and other properties are known in closed form or by one-dimensional
quadrature. The cases serve as the package's test data, as the source of its
σ_eff and r_eff calibration tables and as the content of its validation page,
and users can test their own metrics against them::

    import jax
    from mixed_signals import zoo

    case = zoo.shifted_gaussians(1.5, param_count=3)
    sample_set = case.sample(jax.random.key(0), 10_000)  # shape (1, 2, 10000, 3)
    case.reference_value("kl", ("reference", "shifted"))  # 1.125

The families: shifted, scaled and rotated Gaussians, any K Gaussians, the
twisted Gaussian (banana) with its moment-matched Gaussian, a two-component
mixture with a moment-matched Gaussian, a dropped mode and reweighted modes,
Gaussians railed at a prior bound, wrapped Gaussians across a circle boundary,
von Mises-Fisher distributions near a pole of a sphere, and AR(1) chains with a
known autocorrelation time. :mod:`mixed_signals.zoo.cases` describes the cases
and their reference tables, :mod:`mixed_signals.zoo.distributions` the
distributions, and :mod:`mixed_signals.zoo.closed_forms` the formulas.
"""

from mixed_signals.zoo.cases import (
    DEFAULT_WALKER_COUNT,
    REFERENCE_SCHEMA,
    ChainCase,
    ReferenceBasis,
    ZooCase,
    ar1_chains,
    banana,
    gaussian_mixture,
    gaussians,
    railed_gaussians,
    rotated_gaussians,
    scaled_gaussians,
    shifted_gaussians,
    von_mises_fisher,
    wrapped_gaussians,
)

__all__ = [
    "DEFAULT_WALKER_COUNT",
    "REFERENCE_SCHEMA",
    "ChainCase",
    "ReferenceBasis",
    "ZooCase",
    "ar1_chains",
    "banana",
    "gaussian_mixture",
    "gaussians",
    "railed_gaussians",
    "rotated_gaussians",
    "scaled_gaussians",
    "shifted_gaussians",
    "von_mises_fisher",
    "wrapped_gaussians",
]
