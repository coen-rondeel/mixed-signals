"""Classifier estimator: density ratios from a probabilistic classifier; a placeholder.

A probabilistic classifier trained to tell samples of p from samples of q
estimates the log density ratio ln p/q through its log odds, corrected by the
ratio of the two sample sizes (Sugiyama, Suzuki and Kanamori 2012). The
classifier two-sample test of the metrics rests on such a classifier
(Lopez-Paz and Oquab 2017). The classifier and its training setup are
deferred; until then :class:`ClassifierRatio` has its final signatures and
raises ``NotImplementedError``.

References:
    :cite:t:`sugiyama2012density`; :cite:t:`lopezpaz2017classifier`.
"""

from typing import ClassVar, Final

import equinox as eqx
from jax.typing import ArrayLike
from jaxtyping import Array, Float, PRNGKeyArray

from mixed_signals.core.space import ParamSpace
from mixed_signals.estimators.base import EstimatorSettings

_DEFERRED: Final = (
    "the classifier estimator is deferred until its classifier and training setup"
    " are chosen"
)


class ClassifierSettings(EstimatorSettings):
    """Settings of the classifier estimator, none until it is implemented."""

    name: ClassVar[str] = "classifier"


class ClassifierRatio(eqx.Module):
    """A classifier that estimates the log density ratio of two posteriors.

    It describes two posteriors at once, so it gives a ratio rather than a
    density and is not an :class:`~mixed_signals.estimators.base.Estimator`.

    Attributes:
        settings: The settings.

    References:
        :cite:t:`sugiyama2012density`; :cite:t:`lopezpaz2017classifier`.
    """

    settings: ClassifierSettings = eqx.field(static=True)

    @classmethod
    def fit(
        cls,
        first: ArrayLike,
        first_log_weights: ArrayLike | None,
        second: ArrayLike,
        second_log_weights: ArrayLike | None,
        space: ParamSpace,
        settings: ClassifierSettings | None = None,
        *,
        key: PRNGKeyArray,
    ) -> "ClassifierRatio":
        """Train a classifier to tell the samples of two posteriors apart.

        Args:
            first: Samples of the first posterior p, shaped (N, D).
            first_log_weights: Their log weights, shaped (N,); equal weights
                when ``None``.
            second: Samples of the second posterior q, shaped (M, D).
            second_log_weights: Their log weights, shaped (M,); equal weights
                when ``None``.
            space: Parameter space of both sets.
            settings: The settings; the defaults when ``None``.
            key: JAX PRNG key for the initialisation and the training.

        Returns:
            The trained classifier.

        Raises:
            NotImplementedError: Always, until the classifier is implemented.
        """
        raise NotImplementedError(_DEFERRED)

    def log_ratio(
        self, points: Float[ArrayLike, "*leading param"]
    ) -> Float[Array, " *leading"]:
        """Return the estimated log density ratio ln p/q at points.

        Raises:
            NotImplementedError: Always, until the classifier is implemented.
        """
        raise NotImplementedError(_DEFERRED)
