"""Optimal transport: the Sinkhorn divergence, sliced W2 and its Sinkhorn variant.

:mod:`mixed_signals.metrics.ot.sinkhorn` holds the debiased Sinkhorn
divergence, and :mod:`mixed_signals.metrics.ot.sliced` sliced W2 and the
sliced Sinkhorn divergence with their directions. All three transport the
squared Euclidean distance of the chordal embedding, in which circles and
spheres are measured along chords, as for the maximum mean discrepancy and
the energy distance; ott-jax solves the transport problems.
"""
