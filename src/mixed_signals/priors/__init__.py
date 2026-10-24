"""Prior interfaces and the adapters for eryn's priors.

:class:`~mixed_signals.priors.protocol.PriorDistribution` is the minimal
interface of a prior over the parameters of a problem. For eryn's prior
containers, :meth:`ParamSpace.from_eryn_priors
<mixed_signals.core.space.ParamSpace.from_eryn_priors>` builds the space of the
sampling coordinates, :class:`~mixed_signals.priors.eryn_container.ErynPhysical`
maps them to physical parameters, and
:class:`~mixed_signals.priors.eryn_container.ErynPrior` makes a container meet
the protocol::

    import numpy as np
    from eryn.priors import LogUniform, ProbDistContainer, UniformDistribution

    from mixed_signals.core.space import ParamSpace
    from mixed_signals.priors import ErynPhysical

    priors = ProbDistContainer(
        {
            "ln_amplitude": LogUniform(
                1e-23, 1e-21, name="ln_amplitude", name_phys="amplitude"
            ),
            "phase": UniformDistribution(0.0, 2 * np.pi),
        }
    )
    space = ParamSpace.from_eryn_priors(priors, periodic={"phase": 2 * np.pi})
    physical = ErynPhysical(priors)
    physical.map_space(space).names  # ('amplitude', 'phase')
"""

from mixed_signals.priors.eryn_container import ErynPhysical, ErynPrior
from mixed_signals.priors.protocol import PriorDistribution

__all__ = ["ErynPhysical", "ErynPrior", "PriorDistribution"]
