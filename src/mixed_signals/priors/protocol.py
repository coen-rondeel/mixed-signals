"""``PriorDistribution``: the minimal interface of a prior over parameters.

A prior distribution evaluates its log density at parameter vectors and draws
new ones. Any object with these two methods meets the protocol, whatever array
library it computes with, so engines call it at the boundary and convert its
results with :func:`~mixed_signals.io.arrays.to_jax`.

logpdf
    Maps samples shaped (N, D) to log densities shaped (N,), with respect to
    the Lebesgue measure of the coordinates that the samples use.
rvs
    Draws samples shaped ``size + (D,)`` from an explicit JAX PRNG ``key``, as
    every stochastic function of the package does, so that draws follow the
    config seed.

eryn's ``ProbDistContainer`` draws with ``rng_key`` on its JAX backend and from
the global random state of NumPy or CuPy otherwise, so it meets the protocol
through :class:`~mixed_signals.priors.eryn_container.ErynPrior`.
"""

from typing import Any, Protocol, runtime_checkable

from jax.typing import ArrayLike
from jaxtyping import PRNGKeyArray


@runtime_checkable
class PriorDistribution(Protocol):
    """A prior over the D parameters of a problem.

    ``isinstance`` checks that both methods exist, not their signatures.
    """

    def logpdf(self, samples: ArrayLike) -> Any:
        """Return the log prior density, shaped (N,), at samples shaped (N, D)."""
        ...

    def rvs(self, size: int | tuple[int, ...], *, key: PRNGKeyArray) -> Any:
        """Return draws shaped ``size + (D,)`` from the JAX PRNG ``key``."""
        ...
