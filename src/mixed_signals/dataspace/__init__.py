"""Data space: posteriors compared through the signals that they imply.

A posterior over parameters implies a distribution of signals in the data
domain, and two posteriors that look different in parameters can imply the same
signals. Data space compares the signals. It whitens each signal by the noise
model, so that the dot product of two whitened signals is their noise-weighted
inner product (Finn 1992; Cutler and Flanagan 1994), and measures distances in
units of the noise standard deviation: two signals with ‖δh‖ ≲ 1 are
indistinguishable in the data (Lindblom, Owen and Brown 2008). The energy
distance of the whitened signals vanishes only for equal distributions, because
a separable Hilbert space has strong negative type (Lyons 2013), and its root
is a metric. At linear order the squared distance between the signals of two
nearby parameter vectors is ΔθᵀΓΔθ for the Fisher matrix Γ (Vallisneri 2008),
so a distance in the Fisher chart of the parameters has the same meaning.

:func:`~mixed_signals.dataspace.templates.whitened_templates`
    Calls the generator and the noise model of the caller on chunks of samples
    and returns the whitened templates as JAX rows.
:func:`~mixed_signals.dataspace.distances.distance_matrix`
    The Euclidean distances between all pairs of rows, from the Gram matrix of
    the rows centred on their mean.
:class:`~mixed_signals.dataspace.chart.FisherChart`
    The joint local chart of a Fisher matrix, with circle axes unwrapped about
    the centre, whose distances are ΔθᵀΓΔθ.

The engine that joins them is
:func:`~mixed_signals.engines.dataspace.compare_signals`.

References:
    :cite:t:`lindblom2008model`; :cite:t:`lyons2013distance`;
    :cite:t:`vallisneri2008use`; :cite:t:`finn1992detection`;
    :cite:t:`cutler1994gravitational`.
"""

from mixed_signals.dataspace.chart import CHART_CUT_WIDTHS, FisherChart
from mixed_signals.dataspace.distances import distance_matrix
from mixed_signals.dataspace.templates import whitened_templates

__all__ = ["CHART_CUT_WIDTHS", "FisherChart", "distance_matrix", "whitened_templates"]
