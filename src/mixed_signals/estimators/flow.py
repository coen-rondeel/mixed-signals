"""Normalizing-flow estimator: a placeholder until its training setup is chosen.

A normalizing flow models a density as a learned bijection of a simple base
density, whose log-determinant gives exact log densities (Papamakarios et
al. 2021). The flow, its training setup and the optimiser it needs are
deferred; until then :class:`FlowDensity` has its final signatures and
raises ``NotImplementedError``.

References:
    :cite:t:`papamakarios2021normalizing`.
"""

from typing import ClassVar, Final

import equinox as eqx
from jax.typing import ArrayLike
from jaxtyping import Array, Float, PRNGKeyArray

from mixed_signals.core.space import ParamSpace
from mixed_signals.estimators.base import Estimator, EstimatorSettings

_DEFERRED: Final = (
    "the normalizing-flow estimator is deferred until its flow and training setup"
    " are chosen"
)


class FlowSettings(EstimatorSettings):
    """Settings of the normalizing-flow estimator, none until it is implemented."""

    name: ClassVar[str] = "flow"


class FlowDensity(Estimator):
    """A normalizing flow fitted to one posterior; a placeholder for now.

    References:
        :cite:t:`papamakarios2021normalizing`.
    """

    settings: FlowSettings = eqx.field(static=True)

    @classmethod
    def fit(
        cls,
        samples: ArrayLike,
        log_weights: ArrayLike | None,
        space: ParamSpace,
        settings: FlowSettings | None = None,
        *,
        key: PRNGKeyArray,
    ) -> "FlowDensity":
        """Train a normalizing flow on the samples of one posterior.

        Args:
            samples: Samples in the parameter order of ``space``, shaped
                (N, D).
            log_weights: Log weights shaped (N,); equal weights when
                ``None``.
            space: Parameter space of the samples.
            settings: The settings; the defaults when ``None``.
            key: JAX PRNG key for the initialisation and the training.

        Returns:
            The trained flow.

        Raises:
            NotImplementedError: Always, until the flow is implemented.
        """
        raise NotImplementedError(_DEFERRED)

    def log_density(
        self, points: Float[ArrayLike, "*leading param"]
    ) -> Float[Array, " *leading"]:
        """Return the log density of the flow at points.

        Raises:
            NotImplementedError: Always, until the flow is implemented.
        """
        raise NotImplementedError(_DEFERRED)
