"""Finite-sample nulls: permutation, split-half and Gaussian surrogates.

Every null matches the effective sample sizes of the inputs it compares.
:mod:`mixed_signals.nulls.surrogates` holds the Gaussian surrogates of the
posterior-shape diagnostics.
"""

from mixed_signals.nulls.surrogates import (
    CONSTRUCTION,
    gaussian_surrogates,
    surrogate_size,
)

__all__ = ["CONSTRUCTION", "gaussian_surrogates", "surrogate_size"]
