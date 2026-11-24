"""LISA sources: holders of their samples, conventions, transforms and comparisons.

One holder class per source type of the catpylisa conventions keeps B sources
with N weighted samples each, and gives the
:class:`~mixed_signals.core.samples.SampleSet` and
:class:`~mixed_signals.core.space.ParamSpace` that the engines compare:
:class:`~mixed_signals.gw.lisa.galactic_binaries.GalacticBinaries`,
:class:`~mixed_signals.gw.lisa.massive_black_hole_binaries.MassiveBlackHoleBinaries`,
:class:`~mixed_signals.gw.lisa.extreme_mass_ratio_inspirals.ExtremeMassRatioInspirals`
and
:class:`~mixed_signals.gw.lisa.stellar_origin_binary_black_holes.StellarOriginBinaryBlackHoles`.
A catalog of injections or point estimates is a holder with one sample per
source.

The SGS conventions document (:cite:t:`baghi2026lisa`) is the ground truth for
frames; names, units and defaults follow the catpylisa conventions file. Each
holder records its conventions, :class:`LisaConventions`, and comparisons refuse
holders whose records differ, with the holder method that aligns them: the sky
frame (:meth:`~mixed_signals.gw.lisa.sources.LisaSources.to_icrs` and
``to_ecliptic``), the epoch of Galactic binaries and massive black-hole binaries
(``at_epoch``), and the opt-in forms of the parameters: the chirp mass and
distance of Galactic binaries, their phase combinations, and the total or chirp
mass of binary black holes with the mass ratio q = m₂/m₁ or the inverted mass
ratio Q = m₁/m₂ (``to_total_mass_and_mass_ratio`` and
``to_chirp_mass_and_mass_ratio``, with ``inverted=True`` for Q). A prior bound
on a transformed parameter is set after the transform with
:meth:`~mixed_signals.gw.lisa.sources.LisaSources.with_bounds`, because
transforms refuse bounded parameters. The differences between the codes of the
global fits are out of scope.

The package needs the ``gw`` extra (``pip install "mixed-signals[gw]"``) for the
constants of lisaconstants and the Planck15 cosmology of astropy. Importing this
package works without it; only a call that needs a constant or the redshift
raises :class:`~mixed_signals.errors.MissingDependencyError`.

:func:`~mixed_signals.gw.lisa.compare.compare_sources` compares two holders
through a matched index array, one problem per pair of sources. catpylisa itself
is never imported.

References:
    :cite:t:`baghi2026lisa`; :cite:t:`bayle2026lisa`.
"""

from mixed_signals.gw.lisa.compare import (
    SourceComparison,
    compare_sources,
    match_indices,
)
from mixed_signals.gw.lisa.conventions import (
    LisaConventions,
    MassFrame,
    ParameterSpec,
    SkyFrame,
    SourceFrame,
)
from mixed_signals.gw.lisa.extreme_mass_ratio_inspirals import ExtremeMassRatioInspirals
from mixed_signals.gw.lisa.galactic_binaries import GalacticBinaries
from mixed_signals.gw.lisa.massive_black_hole_binaries import MassiveBlackHoleBinaries
from mixed_signals.gw.lisa.sources import BlackHoleBinarySources, LisaSources
from mixed_signals.gw.lisa.stellar_origin_binary_black_holes import (
    StellarOriginBinaryBlackHoles,
)

__all__ = [
    "BlackHoleBinarySources",
    "ExtremeMassRatioInspirals",
    "GalacticBinaries",
    "LisaConventions",
    "LisaSources",
    "MassFrame",
    "MassiveBlackHoleBinaries",
    "ParameterSpec",
    "SkyFrame",
    "SourceComparison",
    "SourceFrame",
    "StellarOriginBinaryBlackHoles",
    "compare_sources",
    "match_indices",
]
