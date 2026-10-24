"""Tests for ``PriorDistribution``, the minimal prior interface."""

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike
from jaxtyping import Array, PRNGKeyArray

from mixed_signals.priors import PriorDistribution


class UnitNormal:
    """A prior from outside the package: independent unit normals in JAX."""

    def __init__(self, param_count: int) -> None:
        """Store the number of parameters."""
        self.param_count = param_count

    def logpdf(self, samples: ArrayLike) -> Array:
        """Return the log density of independent unit normals."""
        values = jnp.asarray(samples)
        return -0.5 * jnp.sum(values**2 + jnp.log(2 * jnp.pi), axis=-1)

    def rvs(self, size: int | tuple[int, ...], *, key: PRNGKeyArray) -> Array:
        """Draw independent unit normals."""
        shape = (size,) if isinstance(size, int) else size
        return jax.random.normal(key, (*shape, self.param_count))


class DensityOnly:
    """An object with a density but no draws."""

    def logpdf(self, samples: Any) -> Any:
        """Return zeros."""
        return np.zeros(np.shape(samples)[:-1])


def test_objects_with_both_methods_meet_the_protocol() -> None:
    """Any object with logpdf and rvs is a PriorDistribution."""
    prior: PriorDistribution = UnitNormal(3)
    assert isinstance(prior, PriorDistribution)
    draws = prior.rvs((4, 5), key=jax.random.key(0))
    assert draws.shape == (4, 5, 3)
    assert prior.logpdf(draws).shape == (4, 5)


def test_objects_without_draws_do_not_meet_the_protocol() -> None:
    """A density alone is not a PriorDistribution."""
    assert not isinstance(DensityOnly(), PriorDistribution)
