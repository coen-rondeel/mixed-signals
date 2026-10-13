"""Convert nested-sampling weights into ``log_weights``: benched for now.

Until this module is designed, pass the natural logarithm of the weights that a
nested sampler reports as the ``log_weights`` of a
:class:`~mixed_signals.core.samples.SampleSet`. The weights need no
normalisation, and a weight of zero becomes a log weight of −∞.
"""
