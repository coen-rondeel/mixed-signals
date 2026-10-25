"""Comparison metrics as pure JAX kernels on arrays.

:mod:`mixed_signals.metrics.moments` holds the mean shift, the width ratio and
the Gaussian divergences, :mod:`mixed_signals.metrics.marginal` the
one-dimensional marginal metrics, :mod:`mixed_signals.metrics.divergence` the
nearest-neighbour KL, JS and Hellinger estimates,
:mod:`mixed_signals.metrics.kernel` the maximum mean discrepancy and the
energy distance, and :mod:`mixed_signals.metrics.classifier` the placeholder
of the classifier two-sample test. Metrics whose cost grows as N² draw a
capped subsample first.
"""
