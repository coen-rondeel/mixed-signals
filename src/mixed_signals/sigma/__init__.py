"""Significance and effect sizes: sigma_sig, sigma_eff and r_eff.

sigma_sig comes from the null; sigma_eff and r_eff come from their maps.
:mod:`mixed_signals.sigma.significance` holds the empirical p-value, the tail
fit beyond the resolution of the null draws and sigma_sig,
:mod:`mixed_signals.sigma.pareto` the generalized Pareto fit that the tail fit
uses, and :mod:`mixed_signals.sigma.maps` the null subtraction and the maps of
the moment, marginal, kernel and optimal-transport metrics, whose kNN KL map
needs the curve that :mod:`mixed_signals.sigma.calibration` measures at run
time. :mod:`mixed_signals.sigma.thresholds` holds the default thresholds of the
effect sizes and the verdict of a comparison row.
"""
