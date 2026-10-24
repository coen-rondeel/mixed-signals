"""Comparison metrics as pure JAX kernels on arrays.

:mod:`mixed_signals.metrics.moments` holds the mean shift, the width ratio and
the Gaussian divergences, and :mod:`mixed_signals.metrics.marginal` the
one-dimensional marginal metrics. Metrics whose cost grows as N² draw a capped
subsample first.
"""
