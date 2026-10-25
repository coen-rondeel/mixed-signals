"""Classifier two-sample test (C2ST); a placeholder.

A classifier trained to tell the samples of two posteriors apart reaches a
held-out accuracy of one half when they agree and more when they differ
(Lopez-Paz and Oquab 2017); the simulation-based inference benchmark of
Lueckmann et al. (2021) compares approximate posteriors with references this
way. Its odds also estimate the density ratio of the two posteriors
(Sugiyama, Suzuki and Kanamori 2012). The rows read the metric ``c2st``, and
for the Bayes classifier of two unit Gaussians shifted by Δ the accuracy is
Φ(Δ/2), so σ_eff = 2Φ⁻¹(accuracy), a lower bound for real classifiers. The
classifier, its training setup and its null are deferred; until then
:func:`classifier_two_sample_test` has its final signature and raises
``NotImplementedError``.

References:
    :cite:t:`lopezpaz2017classifier`; :cite:t:`lueckmann2021simulation`;
    :cite:t:`sugiyama2012density`.
"""

from typing import Final

from jax.typing import ArrayLike
from jaxtyping import Array, Float, PRNGKeyArray

from mixed_signals.core.space import ParamSpace
from mixed_signals.estimators.classifier import ClassifierSettings

_DEFERRED: Final = (
    "the classifier two-sample test is deferred until its classifier, training"
    " setup and null are chosen"
)


def classifier_two_sample_test(
    first: ArrayLike,
    first_log_weights: ArrayLike | None,
    second: ArrayLike,
    second_log_weights: ArrayLike | None,
    space: ParamSpace,
    settings: ClassifierSettings | None = None,
    *,
    key: PRNGKeyArray,
) -> Float[Array, ""]:
    """Return the held-out accuracy of a classifier that tells two posteriors apart.

    Args:
        first: Samples of the first posterior, shaped (N, D), in the
            parameter order of ``space``.
        first_log_weights: Its log weights shaped (N,); equal weights when
            ``None``.
        second: Samples of the second posterior, shaped (M, D).
        second_log_weights: Its log weights shaped (M,); equal weights when
            ``None``.
        space: Space of the samples.
        settings: Settings of the classifier; the defaults when ``None``.
        key: JAX PRNG key for the split, the initialisation and the training.

    Returns:
        The accuracy, one half when the posteriors agree.

    Raises:
        NotImplementedError: Always, until the test is implemented.

    References:
        :cite:t:`lopezpaz2017classifier`; :cite:t:`lueckmann2021simulation`.
    """
    raise NotImplementedError(_DEFERRED)
