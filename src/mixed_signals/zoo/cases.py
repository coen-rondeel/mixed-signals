"""Validation zoo cases: posteriors with known answers, their space and their truth.

A :class:`ZooCase` holds K named posteriors of one problem, the
:class:`~mixed_signals.core.space.ParamSpace` they share and a table of their
reference values. :meth:`ZooCase.sample` draws a
:class:`~mixed_signals.core.samples.SampleSet` shaped (B, K, N, D) from an
explicit key, and :meth:`ZooCase.log_prob` evaluates the density of every
posterior. A :class:`ChainCase` holds AR(1) chains with a known
autocorrelation time, drawn as :class:`~mixed_signals.core.chains.Chains`.

Builders
    :func:`gaussians` takes any K Gaussians; :func:`shifted_gaussians`,
    :func:`scaled_gaussians` and :func:`rotated_gaussians` build the
    canonical pairs. :func:`banana` pairs a twisted Gaussian with its
    moment-matched Gaussian, :func:`gaussian_mixture` a two-component mixture
    with three partners, and :func:`railed_gaussians` gives Gaussians
    truncated at one bound. :func:`wrapped_gaussians` straddles the boundary
    of a circle, :func:`von_mises_fisher` a pole of a sphere, and
    :func:`ar1_chains` gives correlated chains.
Reference table
    One row per pair, metric and parameter subset, with the columns ``pair``,
    ``metric``, ``parameters``, ``value`` and ``basis``, typed as in the
    result table, so that results join their reference values on ``pair``,
    ``metric`` and ``parameters``. ``pair`` holds one name for a property of
    one posterior, and two names in argument order for a comparison: the
    ``kl`` row of the pair (a, b) is KL(a ‖ b). Symmetric metrics have one
    row per pair, in case order. ``basis`` says whether a value is a closed
    form or comes from quadrature; see :mod:`mixed_signals.zoo.closed_forms`.
Keys
    Posterior k draws from ``derive_key(key, name)``, and problem b folds its
    index into that key, so the draws of a posterior depend neither on the
    other posteriors nor on the number of problems. Posteriors of the same
    name in two cases reuse the same random numbers for the same key; derive
    separate keys for cases that must be independent.

The metrics of the reference table:

``mean``, ``std``
    Mean and standard deviation of one marginal of one posterior.
``circular_mean``, ``mean_resultant_length``
    Mean direction and mean resultant length of a circle marginal, or the
    mean resultant length of the unit vectors of a sphere, which is
    coth κ − 1/κ for a von Mises-Fisher distribution.
``railing_score``
    Density at the finite bound over the peak density of a railed marginal.
``tau``
    Integrated autocorrelation time of one parameter of AR(1) chains.
``mean_shift``, ``width_ratio``
    (μ_b − μ_a)/σ_a and σ_b/σ_a for one parameter of the pair (a, b), the
    true per-parameter σ_eff and r_eff. On a circle the shift is the shortest
    signed arc between the circular means.
``mahalanobis_shift``
    √(ΔμᵀΣ_a⁻¹Δμ) over all parameters, the true σ_eff of a shift. For
    wrapped Gaussians it is the shift of the unwrapped normals.
``angular_shift``
    Angle between the mean directions of two von Mises-Fisher distributions,
    in units of the width 1/√κ_a.
``kl``, ``kl_location``, ``kl_shape``
    KL(a ‖ b) in nats; Gaussian pairs also carry the location and shape
    terms.
``w2``, ``w2_location``, ``w2_shape``
    W2 distance between Gaussians, in the units of the parameters.
``hellinger``, ``hellinger_location``, ``hellinger_shape``
    Hellinger distance, over all parameters or, for Gaussian pairs, over one
    marginal.
``ks``, ``w1``, ``js``, ``ad``
    Kolmogorov-Smirnov, W1, Jensen-Shannon (nats) and Anderson-Darling
    distances between one marginal of a Gaussian pair.

References:
    :cite:t:`haario1999adaptive`; :cite:t:`sejdinovic2014kernel`;
    :cite:t:`mardia2000directional`; :cite:t:`fisher1953dispersion`;
    :cite:t:`barr1999mean`; :cite:t:`thompson2010comparison`.
"""

import dataclasses
import itertools
import math
from collections.abc import Sequence
from enum import StrEnum
from typing import Final

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import polars as pl
from jax.scipy.linalg import solve_triangular
from jax.typing import ArrayLike
from jaxtyping import Array, Float, PRNGKeyArray

from mixed_signals.core.chains import Chains
from mixed_signals.core.samples import SampleSet
from mixed_signals.core.space import Parameter, ParamSpace
from mixed_signals.core.topology import (
    AZIMUTH_PERIOD,
    PolarForm,
    Sphere,
    Topology,
    wrap,
)
from mixed_signals.prng import derive_key
from mixed_signals.zoo import closed_forms
from mixed_signals.zoo._checks import (
    check_count,
    check_finite,
    check_positive,
    cholesky,
    matrix,
    vector,
)
from mixed_signals.zoo.distributions import (
    Gaussian,
    GaussianMixture,
    TruncatedGaussian,
    TwistedGaussian,
    VonMisesFisher,
    WrappedGaussian,
    ZooDistribution,
)

REFERENCE_SCHEMA: Final = pl.Schema(
    {
        "pair": pl.List(pl.String()),
        "metric": pl.String(),
        "parameters": pl.List(pl.String()),
        "value": pl.Float64(),
        "basis": pl.String(),
    }
)
"""Column names and types of every reference table, in order.

:meta hide-value:
"""

DEFAULT_WALKER_COUNT: Final = 32
"""Walkers per AR(1) draw unless the caller sets another number."""

_MARGINAL_METRICS: Final = (
    ("hellinger", closed_forms.normal_hellinger, "closed form"),
    ("ks", closed_forms.normal_ks, "closed form"),
    ("w1", closed_forms.normal_w1, "closed form"),
    ("js", closed_forms.normal_js, "quadrature"),
    ("ad", closed_forms.normal_anderson_darling, "quadrature"),
)


class ReferenceBasis(StrEnum):
    """How a reference value was computed: ``closed form`` or ``quadrature``."""

    CLOSED_FORM = "closed form"
    QUADRATURE = "quadrature"


@dataclasses.dataclass(frozen=True, eq=False)
class ZooCase:
    """K posteriors with known answers, their space and their reference values.

    Build cases with the builders of this module, such as
    :func:`shifted_gaussians`, rather than directly.

    Attributes:
        name: Name of the case, such as ``"shifted_gaussians"``.
        space: The parameter space that every posterior shares.
        names: Names of the posteriors, in the order of the posterior axis.
        posteriors: The posteriors, one per name.
        reference: Reference values, with the columns of
            :data:`REFERENCE_SCHEMA`.

    Raises:
        ValueError: If the names repeat or do not match the posteriors, a
            posterior does not fit the space, or the table has another schema.
    """

    name: str
    space: ParamSpace
    names: tuple[str, ...]
    posteriors: tuple[ZooDistribution, ...]
    reference: pl.DataFrame

    def __post_init__(self) -> None:
        """Check that names, posteriors, space and table fit together."""
        _check_names(self.names)
        if len(self.names) != len(self.posteriors):
            raise ValueError(
                f"{len(self.names)} names for {len(self.posteriors)} posteriors"
            )
        for name, posterior in zip(self.names, self.posteriors, strict=True):
            if posterior.param_count != len(self.space):
                raise ValueError(
                    f"posterior {name!r} has {posterior.param_count} parameters,"
                    f" the space {len(self.space)}"
                )
        _check_reference(self.reference)

    def sample(
        self, key: PRNGKeyArray, sample_count: int, *, batch_size: int = 1
    ) -> SampleSet:
        """Draw independent samples of every posterior.

        Args:
            key: JAX PRNG key.
            sample_count: Number of samples N of each posterior.
            batch_size: Number of independent problems B, each a fresh draw
                of the same posteriors.

        Returns:
            Equally weighted samples shaped (B, K, N, D).

        Raises:
            TypeError: If a count is not an integer.
            ValueError: If a count is below one.
        """
        check_count("sample_count", sample_count)
        check_count("batch_size", batch_size)
        draws = [
            _draw(posterior, derive_key(key, name), sample_count, batch_size)
            for name, posterior in zip(self.names, self.posteriors, strict=True)
        ]
        return SampleSet(jnp.stack(draws, axis=1))

    def log_prob(
        self, points: Float[ArrayLike, "*leading param"]
    ) -> Float[Array, "*leading posterior"]:
        """Return the log density of every posterior at each point.

        Args:
            points: Points shaped (..., D), in the coordinates of the space.

        Returns:
            The log densities shaped (..., K), with the posteriors on the
            last axis; −inf outside a support.
        """
        values = jnp.asarray(points)
        return jnp.stack(
            [posterior.log_prob(values) for posterior in self.posteriors], axis=-1
        )

    def reference_value(
        self,
        metric: str,
        pair: str | Sequence[str],
        parameters: str | Sequence[str] | None = None,
    ) -> float:
        """Return one reference value.

        Args:
            metric: Name of the metric, such as ``"kl"``.
            pair: One posterior name, or two names in argument order.
            parameters: One parameter name, several, or ``None`` for every
                parameter of the space.

        Returns:
            The value.

        Raises:
            KeyError: If the table has no such row.
        """
        return _lookup(
            self.reference,
            metric,
            [pair] if isinstance(pair, str) else list(pair),
            _parameter_list(parameters, self.space),
        )


@dataclasses.dataclass(frozen=True, eq=False)
class ChainCase:
    """AR(1) chains with a known autocorrelation time.

    Every walker and parameter follows x_t = φ x_{t−1} + √(1 − φ²) ε_t with
    standard normal ε_t, independently, so the stationary distribution of
    each parameter is N(0, 1) and its autocorrelation at lag t is φ^|t|.

    Attributes:
        name: Name of the case; the ``pair`` entry of its reference rows.
        space: The parameter space, one real parameter per coefficient.
        coefficients: The coefficient φ of each parameter, in (−1, 1).
        reference: Reference values: ``tau``, ``mean`` and ``std`` of each
            parameter.

    Raises:
        ValueError: If the coefficients do not fit the space or lie outside
            (−1, 1), or the table has another schema.

    References:
        :cite:t:`thompson2010comparison`; :cite:t:`foremanmackey2017autocorrelation`.
    """

    name: str
    space: ParamSpace
    coefficients: tuple[float, ...]
    reference: pl.DataFrame

    def __post_init__(self) -> None:
        """Check the coefficients and the table."""
        if len(self.coefficients) != len(self.space):
            raise ValueError(
                f"{len(self.coefficients)} coefficients for {len(self.space)}"
                " parameters"
            )
        for coefficient in self.coefficients:
            closed_forms.ar1_autocorrelation_time(coefficient)
        _check_reference(self.reference)

    def sample(
        self,
        key: PRNGKeyArray,
        step_count: int,
        *,
        walker_count: int = DEFAULT_WALKER_COUNT,
        start: float | None = None,
    ) -> Chains:
        """Draw chains of independent walkers.

        Args:
            key: JAX PRNG key.
            step_count: Number of steps of each walker.
            walker_count: Number of walkers.
            start: Value of every parameter of every walker at the first
                step, in stationary standard deviations; ``None`` draws the
                first step from the stationary distribution, so the chains
                need no burn-in. A start far from zero decays as φ^t.

        Returns:
            Chains shaped (step, walker, param), with one temperature.

        Raises:
            TypeError: If a count is not an integer.
            ValueError: If a count is below one or the start is not finite.
        """
        check_count("step_count", step_count)
        check_count("walker_count", walker_count)
        if start is not None:
            check_finite("start", start)
        samples = _draw_ar1(
            jnp.asarray(self.coefficients),
            derive_key(key, self.name),
            step_count,
            walker_count,
            None if start is None else float(start),
        )
        return Chains(samples)

    def reference_value(
        self, metric: str, parameters: str | Sequence[str] | None = None
    ) -> float:
        """Return one reference value of the chains.

        Args:
            metric: ``"tau"``, ``"mean"`` or ``"std"``.
            parameters: One parameter name, or ``None`` for the only
                parameter of a one-parameter space.

        Returns:
            The value.

        Raises:
            KeyError: If the table has no such row.
        """
        return _lookup(
            self.reference, metric, [self.name], _parameter_list(parameters, self.space)
        )


def gaussians(
    means: object, covariances: object, *, names: Sequence[str] | None = None
) -> ZooCase:
    """Build a case from K Gaussians.

    Every pair carries the closed forms of KL, W2 and Hellinger with their
    location and shape terms, the moment metrics, and the marginal KS, W1,
    Hellinger, JS and AD values of every parameter.

    Args:
        means: Means shaped (K, D).
        covariances: Covariances shaped (K, D, D), each symmetric positive
            definite.
        names: Names of the posteriors; ``posterior_0``, ``posterior_1``, …
            when not given.

    Returns:
        The case ``"gaussians"``, with parameters ``x0``, ``x1``, … on the
        real line.

    Raises:
        ValueError: If the shapes do not fit or a covariance is not symmetric
            positive definite.

    References:
        :cite:t:`kullback1951information`; :cite:t:`dowson1982frechet`;
        :cite:t:`bhattacharyya1943measure`.
    """
    means_array = np.asarray(means, dtype=np.float64)
    if means_array.ndim != 2 or 0 in means_array.shape:
        raise ValueError(f"means must be shaped (K, D), got shape {means_array.shape}")
    posterior_count, param_count = means_array.shape
    covariances_array = np.asarray(covariances, dtype=np.float64)
    if covariances_array.shape != (posterior_count, param_count, param_count):
        raise ValueError(
            f"covariances must be shaped ({posterior_count}, {param_count},"
            f" {param_count}), got shape {covariances_array.shape}"
        )
    if names is None:
        names = [f"posterior_{index}" for index in range(posterior_count)]
    return _gaussian_case(
        "gaussians",
        tuple(names),
        [jnp.asarray(mean) for mean in means_array],
        [jnp.asarray(covariance) for covariance in covariances_array],
    )


def shifted_gaussians(
    shift: float,
    *,
    param_count: int = 2,
    direction: object = None,
    mean: object = None,
    covariance: object = None,
) -> ZooCase:
    """Build a Gaussian and a copy shifted by a known Mahalanobis distance.

    The reference is N(μ, Σ) and the shifted posterior N(μ + Δμ, Σ), with Δμ
    along ``direction`` and ΔμᵀΣ⁻¹Δμ = shift², so ``shift`` is the true
    σ_eff of every joint metric after whitening by Σ.

    Args:
        shift: Mahalanobis length of the shift.
        param_count: Number of parameters D.
        direction: Direction of the shift in parameter space, shaped (D,);
            the first axis when not given.
        mean: Mean μ of the reference, shaped (D,); zero when not given.
        covariance: Covariance Σ of both, shaped (D, D); the identity when not
            given.

    Returns:
        The case ``"shifted_gaussians"`` with posteriors ``reference`` and
        ``shifted``.

    Raises:
        ValueError: If a parameter is out of range or the shapes do not fit.
    """
    check_finite("shift", shift)
    base_mean, base_covariance = _base_gaussian(param_count, mean, covariance)
    if direction is None:
        direction_array = jnp.eye(param_count)[0]
    else:
        direction_array = jnp.asarray(vector("direction", direction, param_count))
    whitened = solve_triangular(
        cholesky(base_covariance, "covariance"), direction_array, lower=True
    )
    length = float(jnp.linalg.norm(whitened))
    if length == 0.0:
        raise ValueError("direction must not be the zero vector")
    offset = shift * direction_array / length
    return _gaussian_case(
        "shifted_gaussians",
        ("reference", "shifted"),
        [base_mean, base_mean + offset],
        [base_covariance, base_covariance],
    )


def scaled_gaussians(
    ratio: float,
    *,
    param_count: int = 2,
    axes: Sequence[int] = (0,),
    mean: object = None,
    covariance: object = None,
) -> ZooCase:
    """Build a Gaussian and a copy whose width along some axes is scaled.

    The scaled posterior has the covariance SΣS, with S diagonal, ``ratio``
    on the given axes and one elsewhere, so its width ratio is ``ratio`` on
    those axes and one on the others, and its correlations are unchanged.
    With one scaled axis and a diagonal Σ, ``ratio`` is the true r_eff of
    every shape term.

    Args:
        ratio: Width ratio σ_scaled/σ_reference on the given axes, positive.
        param_count: Number of parameters D.
        axes: Indices of the scaled parameters.
        mean: Mean of both, shaped (D,); zero when not given.
        covariance: Covariance Σ of the reference, shaped (D, D); the identity
            when not given.

    Returns:
        The case ``"scaled_gaussians"`` with posteriors ``reference`` and
        ``scaled``.

    Raises:
        ValueError: If a parameter is out of range or the shapes do not fit.
    """
    check_positive("ratio", ratio)
    base_mean, base_covariance = _base_gaussian(param_count, mean, covariance)
    scaled_axes = jnp.asarray(_axis_indices("axes", axes, param_count))
    scaling = jnp.ones(param_count).at[scaled_axes].set(ratio)
    scaled_covariance = scaling[:, None] * base_covariance * scaling[None, :]
    return _gaussian_case(
        "scaled_gaussians",
        ("reference", "scaled"),
        [base_mean, base_mean],
        [base_covariance, scaled_covariance],
    )


def rotated_gaussians(
    angle: float,
    *,
    param_count: int = 2,
    widths: Sequence[float] = (2.0, 0.5),
    plane: Sequence[int] = (0, 1),
    mean: object = None,
    scales: object = None,
) -> ZooCase:
    """Build two Gaussians rotated by ±angle/2 in one plane.

    In the plane, the reference has the principal widths ``widths`` along
    axes rotated by −angle/2 and the other posterior along axes rotated by
    +angle/2; every other axis has unit width. The two have the same
    marginals for every angle and differ only in the sign of the correlation
    in the plane, so every marginal and per-parameter metric is exactly zero,
    and so is their log-determinant ratio. ``scales`` then multiplies each
    parameter, which keeps the marginals equal.

    Args:
        angle: Rotation angle between the two, in radians.
        param_count: Number of parameters D, at least 2.
        widths: The two principal widths in the plane, positive.
        plane: Indices of the two parameters that span the plane.
        mean: Mean of both, shaped (D,); zero when not given.
        scales: Factor that multiplies each parameter, positive, shaped (D,);
            ones when not given.

    Returns:
        The case ``"rotated_gaussians"`` with posteriors ``reference`` and
        ``rotated``.

    Raises:
        ValueError: If a parameter is out of range or the shapes do not fit.
    """
    check_finite("angle", angle)
    check_count("param_count", param_count, minimum=2)
    if len(widths) != 2 or len(plane) != 2:
        raise ValueError(
            f"widths and plane need two entries each, got {tuple(widths)} and"
            f" {tuple(plane)}"
        )
    for width in widths:
        check_positive("each width", width)
    first, second = _axis_indices("plane", plane, param_count)
    base_mean, _ = _base_gaussian(param_count, mean, None)
    if scales is None:
        scales_array = jnp.ones(param_count)
    else:
        scales_read = vector("scales", scales, param_count)
        if np.any(scales_read <= 0):
            raise ValueError("scales must be positive")
        scales_array = jnp.asarray(scales_read)
    principal = (
        jnp.eye(param_count)
        .at[first, first]
        .set(widths[0] ** 2)
        .at[second, second]
        .set(widths[1] ** 2)
    )
    covariances = []
    for half_angle in (-0.5 * angle, 0.5 * angle):
        cosine, sine = jnp.cos(half_angle), jnp.sin(half_angle)
        rotation = (
            jnp.eye(param_count)
            .at[first, first]
            .set(cosine)
            .at[first, second]
            .set(-sine)
            .at[second, first]
            .set(sine)
            .at[second, second]
            .set(cosine)
        )
        rotated = rotation @ principal @ rotation.T
        covariances.append(scales_array[:, None] * rotated * scales_array[None, :])
    return _gaussian_case(
        "rotated_gaussians",
        ("reference", "rotated"),
        [base_mean, base_mean],
        covariances,
    )


def banana(
    twist: float = 0.03, *, param_count: int = 2, variance: float = 100.0
) -> ZooCase:
    """Build the twisted Gaussian B(b, v) and its moment-matched Gaussian.

    The two share their mean and covariance, so every moment metric reads
    zero, while KL(B ‖ N) = ½ ln(1 + 2b²v²) and
    KL(N ‖ B) = 2b²v² − ½ ln(1 + 2b²v²). The defaults are the moderately
    twisted target of Sejdinovic et al. (2014), who use it in 8 dimensions
    and also b = 0.1 as the strongly twisted one.

    Args:
        twist: The twist b.
        param_count: Number of parameters D, at least 2.
        variance: The variance v of the first coordinate, positive.

    Returns:
        The case ``"banana"`` with posteriors ``banana`` and ``gaussian``.

    Raises:
        ValueError: If a parameter is out of range.

    References:
        :cite:t:`haario1999adaptive`; :cite:t:`sejdinovic2014kernel`;
        :cite:t:`cover2006elements`.
    """
    twisted = TwistedGaussian(twist, variance, param_count)
    mean, covariance = closed_forms.twisted_gaussian_moments(
        twist, variance, param_count
    )
    names = ("banana", "gaussian")
    param_names = _parameter_names(param_count)
    rows = _Rows()
    _moment_rows(rows, names, param_names, [mean, mean], [covariance, covariance])
    banana_to_gaussian, gaussian_to_banana = closed_forms.twisted_gaussian_kl(
        twist, variance
    )
    rows.add(names, "kl", param_names, banana_to_gaussian)
    rows.add(names[::-1], "kl", param_names, gaussian_to_banana)
    return ZooCase(
        name="banana",
        space=_real_space(param_names),
        names=names,
        posteriors=(twisted, Gaussian(mean, covariance)),
        reference=rows.frame(),
    )


def gaussian_mixture(
    separation: float,
    *,
    param_count: int = 2,
    weights: Sequence[float] = (0.5, 0.5),
    reweighted: Sequence[float] = (0.8, 0.2),
) -> ZooCase:
    """Build a two-component Gaussian mixture and three partners.

    Both components have unit covariance and lie at ∓separation/2 on the
    first axis. The posteriors are the ``mixture``; the ``gaussian`` with its
    mean and covariance, which every moment metric confuses with it; the
    first ``component`` alone, an approximation that drops the second mode;
    and the ``reweighted`` mixture, whose modes carry other weights, as when
    walkers stick in one mode. With equal weights the mixture is bimodal
    exactly when the separation exceeds 2 (Schilling, Watkins and Watkins
    2002). All four agree on every axis but the first, so each KL divergence
    is a one-dimensional quadrature.

    Args:
        separation: Distance between the component means, in component
            standard deviations; non-negative.
        param_count: Number of parameters D.
        weights: Weights of the two components, positive; normalised.
        reweighted: Weights of the reweighted mixture, positive; normalised.

    Returns:
        The case ``"gaussian_mixture"`` with posteriors ``mixture``,
        ``gaussian``, ``component`` and ``reweighted``.

    Raises:
        ValueError: If a parameter is out of range.

    References:
        :cite:t:`schilling2002human`; :cite:t:`kullback1951information`.
    """
    if not (math.isfinite(separation) and separation >= 0):
        raise ValueError(
            f"the separation must be finite and non-negative, got {separation!r}"
        )
    check_count("param_count", param_count)
    mixture_weights = _normalised_weights("weights", weights)
    reweighted_weights = _normalised_weights("reweighted", reweighted)
    centres = jnp.array([-0.5 * separation, 0.5 * separation])
    component_means = jnp.zeros((2, param_count)).at[:, 0].set(centres)
    identity = jnp.eye(param_count)
    mixture_moments = _mixture_moments(mixture_weights, component_means)
    component_moments = (component_means[0], identity)
    moments = [
        mixture_moments,
        mixture_moments,
        component_moments,
        _mixture_moments(reweighted_weights, component_means),
    ]
    names = ("mixture", "gaussian", "component", "reweighted")
    posteriors: tuple[ZooDistribution, ...] = (
        GaussianMixture(mixture_weights, component_means, [identity, identity]),
        Gaussian(*mixture_moments),
        Gaussian(*component_moments),
        GaussianMixture(reweighted_weights, component_means, [identity, identity]),
    )
    # * Each posterior as a one-dimensional mixture along the separation axis.
    matched_std = jnp.sqrt(mixture_moments[1][:1, 0])
    profiles = [
        (mixture_weights, centres, jnp.ones(2)),
        (jnp.ones(1), mixture_moments[0][:1], matched_std),
        (jnp.ones(1), centres[:1], jnp.ones(1)),
        (reweighted_weights, centres, jnp.ones(2)),
    ]
    param_names = _parameter_names(param_count)
    rows = _Rows()
    _moment_rows(
        rows,
        names,
        param_names,
        [mean for mean, _ in moments],
        [covariance for _, covariance in moments],
    )
    gaussian_pair = (names.index("gaussian"), names.index("component"))
    _gaussian_pair_rows(
        rows, ("gaussian", "component"), param_names, mixture_moments, component_moments
    )
    for first, second in itertools.permutations(range(len(names)), 2):
        if {first, second} == set(gaussian_pair):
            continue
        rows.add(
            (names[first], names[second]),
            "kl",
            param_names,
            closed_forms.normal_mixture_kl(*profiles[first], *profiles[second]),
            ReferenceBasis.QUADRATURE,
        )
    return ZooCase(
        name="gaussian_mixture",
        space=_real_space(param_names),
        names=names,
        posteriors=posteriors,
        reference=rows.frame(),
    )


def railed_gaussians(
    offsets: Sequence[float],
    *,
    param_count: int = 2,
    side: str = "lower",
) -> ZooCase:
    """Build Gaussians truncated at a prior bound, one per offset.

    The first parameter of posterior k is a unit normal whose peak lies
    ``offsets[k]`` standard deviations inside the bound at zero, truncated
    there; the other parameters are independent unit normals. Its railing
    score is exp(−offset²/2), or 1 when the peak lies at or beyond the
    bound. The space makes the first parameter an interval with the finite
    bound at zero.

    Args:
        offsets: Distance from the bound to each peak before truncation, in
            standard deviations; positive inside the support.
        param_count: Number of parameters D.
        side: ``"lower"`` for the support [0, ∞), ``"upper"`` for (−∞, 0].

    Returns:
        The case ``"railed_gaussians"`` with posteriors ``railed_0``,
        ``railed_1``, …

    Raises:
        ValueError: If a parameter is out of range.

    References:
        :cite:t:`barr1999mean`.
    """
    if side not in ("lower", "upper"):
        raise ValueError(f"side must be 'lower' or 'upper', got {side!r}")
    check_count("param_count", param_count)
    offset_values = [float(offset) for offset in offsets]
    if not offset_values:
        raise ValueError("give at least one offset")
    for offset in offset_values:
        check_finite("each offset", offset)
    sign = 1.0 if side == "lower" else -1.0
    rail_lower, rail_upper = (0.0, math.inf) if side == "lower" else (-math.inf, 0.0)
    lower = jnp.full(param_count, -jnp.inf).at[0].set(rail_lower)
    upper = jnp.full(param_count, jnp.inf).at[0].set(rail_upper)
    names = tuple(f"railed_{index}" for index in range(len(offset_values)))
    posteriors = []
    means, covariances = [], []
    for offset in offset_values:
        centre = jnp.zeros(param_count).at[0].set(sign * offset)
        posteriors.append(
            TruncatedGaussian(centre, jnp.ones(param_count), lower, upper)
        )
        rail_mean, rail_std = closed_forms.truncated_normal_moments(
            sign * offset, 1.0, rail_lower, rail_upper
        )
        means.append(jnp.zeros(param_count).at[0].set(rail_mean))
        covariances.append(jnp.diag(jnp.ones(param_count).at[0].set(rail_std**2)))
    param_names = _parameter_names(param_count)
    rows = _Rows()
    _moment_rows(rows, names, param_names, means, covariances)
    for name, offset in zip(names, offset_values, strict=True):
        rows.add(
            (name,),
            "railing_score",
            param_names[:1],
            closed_forms.railing_score(offset),
        )
    for first, second in itertools.permutations(range(len(names)), 2):
        rows.add(
            (names[first], names[second]),
            "kl",
            param_names,
            closed_forms.truncated_normal_kl(
                sign * offset_values[first],
                sign * offset_values[second],
                1.0,
                rail_lower,
                rail_upper,
            ),
        )
    parameters = [
        Parameter(
            name=param_names[0],
            topology=Topology.INTERVAL,
            bounds=(rail_lower, rail_upper),
        )
    ] + [Parameter(name=name) for name in param_names[1:]]
    return ZooCase(
        name="railed_gaussians",
        space=ParamSpace(parameters=tuple(parameters)),
        names=names,
        posteriors=tuple(posteriors),
        reference=rows.frame(),
    )


def wrapped_gaussians(
    shift: float,
    *,
    width: float = 0.5,
    param_count: int = 1,
    correlation: float = 0.0,
    period: float = AZIMUTH_PERIOD,
    lower: float = 0.0,
) -> ZooCase:
    """Build a wrapped Gaussian that straddles the circle boundary, and a shifted copy.

    The first parameter is a circle [lower, lower + period); the reference
    centres its wrapped normal of width σ on the boundary itself, so its
    samples straddle the cut, and the shifted posterior centres it
    shift · σ further along the circle. The other parameters are unit
    normals, and ``correlation`` couples the circle to the second parameter
    before wrapping, which tests that whitening never mixes a periodic axis.

    Args:
        shift: Shift of the circular mean, in units of σ; the true σ_eff of
            the circle marginal.
        width: Standard deviation σ of the unwrapped circle coordinate.
        param_count: Number of parameters D.
        correlation: Correlation between the circle and the second parameter
            before wrapping, in (−1, 1); needs at least 2 parameters unless
            zero.
        period: Period of the circle, positive.
        lower: Lower end of the circle.

    Returns:
        The case ``"wrapped_gaussians"`` with posteriors ``reference`` and
        ``shifted``.

    Raises:
        ValueError: If a parameter is out of range.

    References:
        :cite:t:`mardia2000directional`.
    """
    check_finite("shift", shift)
    check_positive("width", width)
    check_positive("period", period)
    check_finite("lower", lower)
    check_count("param_count", param_count)
    if not -1 < correlation < 1:
        raise ValueError(f"the correlation must lie in (-1, 1), got {correlation!r}")
    if correlation != 0 and param_count < 2:
        raise ValueError("a correlation needs at least 2 parameters")
    covariance = jnp.eye(param_count).at[0, 0].set(width**2)
    if param_count > 1:
        coupling = correlation * width
        covariance = covariance.at[0, 1].set(coupling).at[1, 0].set(coupling)
    shifted_centre = wrap(jnp.asarray(lower + shift * width), lower, period)
    centres = [jnp.asarray(lower), shifted_centre]
    names = ("reference", "shifted")
    means = [jnp.zeros(param_count).at[0].set(centre) for centre in centres]
    posteriors = tuple(
        WrappedGaussian(mean, covariance, circle=0, lower=lower, period=period)
        for mean in means
    )
    param_names = _parameter_names(param_count)
    rows = _Rows()
    resultant = closed_forms.wrapped_normal_resultant_length(width, period)
    for name, centre in zip(names, centres, strict=True):
        rows.add((name,), "circular_mean", param_names[:1], centre)
        rows.add((name,), "mean_resultant_length", param_names[:1], resultant)
        for parameter in param_names[1:]:
            rows.add((name,), "mean", (parameter,), 0.0)
            rows.add((name,), "std", (parameter,), 1.0)
    for first, second in itertools.permutations(range(2), 2):
        pair = (names[first], names[second])
        arc = _signed_arc(centres[first], centres[second], period)
        unwrapped_shift = jnp.zeros(param_count).at[0].set(arc)
        rows.add(pair, "mean_shift", param_names[:1], arc / width)
        for parameter in param_names[1:]:
            rows.add(pair, "mean_shift", (parameter,), 0.0)
        for parameter in param_names:
            rows.add(pair, "width_ratio", (parameter,), 1.0)
        rows.add(
            pair,
            "mahalanobis_shift",
            param_names,
            closed_forms.mahalanobis_shift(
                jnp.zeros(param_count), covariance, unwrapped_shift
            ),
        )
    parameters = [
        Parameter(
            name=param_names[0],
            topology=Topology.CIRCLE,
            bounds=(lower, lower + period),
        )
    ] + [Parameter(name=name) for name in param_names[1:]]
    return ZooCase(
        name="wrapped_gaussians",
        space=ParamSpace(parameters=tuple(parameters)),
        names=names,
        posteriors=posteriors,
        reference=rows.frame(),
    )


def von_mises_fisher(
    shift: float,
    *,
    width: float = 0.1,
    colatitude: float | None = None,
    polar_form: PolarForm | str = PolarForm.LATITUDE,
    euclidean_count: int = 0,
) -> ZooCase:
    """Build a von Mises-Fisher distribution near the north pole and a rotated copy.

    Both have the concentration κ = 1/width², so the width is the angular
    standard deviation along each tangent direction when κ is large
    (Mardia and Jupp 2000). The reference is centred at the given colatitude
    on the azimuth 0, so its samples straddle the azimuth boundary. The
    other is rotated by shift · width along the same meridian towards the
    pole and, for a large enough shift, over it onto the azimuth π: the two
    centres then differ by π in azimuth but only by the shift on the sky.
    ``euclidean_count`` standard normal parameters follow the sphere,
    identical in both.

    Args:
        shift: Angle of the rotation, in units of the width.
        width: Angular width 1/√κ, in radians, positive.
        colatitude: Colatitude of the reference centre in [0, π]; one width
            from the pole when not given.
        polar_form: Coordinate of the polar axis.
        euclidean_count: Number of standard normal parameters after the
            sphere.

    Returns:
        The case ``"von_mises_fisher"`` with posteriors ``reference`` and
        ``shifted`` and parameters ``azimuth``, ``polar``, ``x2``, …

    Raises:
        ValueError: If a parameter is out of range.

    References:
        :cite:t:`fisher1953dispersion`; :cite:t:`mardia2000directional`.
    """
    check_finite("shift", shift)
    check_positive("width", width)
    check_count("euclidean_count", euclidean_count, minimum=0)
    form = PolarForm(polar_form)
    centre_colatitude = width if colatitude is None else float(colatitude)
    if not 0 <= centre_colatitude <= math.pi:
        raise ValueError(
            f"the colatitude must lie in [0, π], got {centre_colatitude!r}"
        )
    concentration = 1 / width**2
    directions = [
        jnp.stack([jnp.sin(polar), jnp.zeros(()), jnp.cos(polar)])
        for polar in jnp.array([centre_colatitude, centre_colatitude - shift * width])
    ]
    names = ("reference", "shifted")
    posteriors = tuple(
        VonMisesFisher(
            direction, concentration, polar_form=form, euclidean_count=euclidean_count
        )
        for direction in directions
    )
    sphere_names = ("azimuth", "polar")
    extra_names = _parameter_names(euclidean_count, start=2)
    param_names = sphere_names + extra_names
    rows = _Rows()
    resultant = closed_forms.von_mises_fisher_resultant_length(concentration)
    for name in names:
        rows.add((name,), "mean_resultant_length", sphere_names, resultant)
        for parameter in extra_names:
            rows.add((name,), "mean", (parameter,), 0.0)
            rows.add((name,), "std", (parameter,), 1.0)
    for first, second in itertools.permutations(range(2), 2):
        pair = (names[first], names[second])
        rows.add(
            pair,
            "kl",
            param_names,
            closed_forms.von_mises_fisher_kl(
                directions[first], concentration, directions[second], concentration
            ),
        )
        cosine = jnp.clip(directions[first] @ directions[second], -1.0, 1.0)
        rows.add(pair, "angular_shift", sphere_names, jnp.arccos(cosine) / width)
    rows.add(
        names,
        "hellinger",
        param_names,
        closed_forms.von_mises_fisher_hellinger(
            directions[0], concentration, directions[1], concentration
        ),
    )
    parameters = [
        Parameter(
            name="azimuth", topology=Topology.CIRCLE, bounds=(0.0, AZIMUTH_PERIOD)
        ),
        Parameter(name="polar", topology=Topology.INTERVAL, bounds=form.bounds),
    ] + [Parameter(name=name) for name in extra_names]
    space = ParamSpace(
        parameters=tuple(parameters),
        spheres=(Sphere(azimuth="azimuth", polar="polar", polar_form=form),),
    )
    return ZooCase(
        name="von_mises_fisher",
        space=space,
        names=names,
        posteriors=posteriors,
        reference=rows.frame(),
    )


def ar1_chains(coefficients: Sequence[float]) -> ChainCase:
    """Build AR(1) chains, one coefficient per parameter.

    Parameter i of every walker follows x_t = φ_i x_{t−1} + √(1 − φ_i²) ε_t,
    whose integrated autocorrelation time is τ = (1 + φ)/(1 − φ), summed over
    all lags as in emcee. Thompson (2010) uses φ = 0.98, with τ = 99, to
    compare estimators of τ. A negative φ gives τ < 1, which tests the cap
    on the effective sample size.

    Args:
        coefficients: Coefficient φ of each parameter, in (−1, 1).

    Returns:
        The case ``"ar1_chains"``, with parameters ``x0``, ``x1``, …

    Raises:
        ValueError: If there is no coefficient or one lies outside (−1, 1).

    References:
        :cite:t:`thompson2010comparison`; :cite:t:`foremanmackey2017autocorrelation`.
    """
    values = tuple(float(coefficient) for coefficient in coefficients)
    if not values:
        raise ValueError("give at least one coefficient")
    param_names = _parameter_names(len(values))
    name = "ar1_chains"
    rows = _Rows()
    for parameter, coefficient in zip(param_names, values, strict=True):
        rows.add(
            (name,),
            "tau",
            (parameter,),
            closed_forms.ar1_autocorrelation_time(coefficient),
        )
        rows.add((name,), "mean", (parameter,), 0.0)
        rows.add((name,), "std", (parameter,), 1.0)
    return ChainCase(
        name=name,
        space=_real_space(param_names),
        coefficients=values,
        reference=rows.frame(),
    )


class _Rows:
    """Rows of a reference table, collected in order."""

    def __init__(self) -> None:
        self.rows: list[dict[str, object]] = []

    def add(
        self,
        pair: Sequence[str],
        metric: str,
        parameters: Sequence[str],
        value: ArrayLike,
        basis: ReferenceBasis = ReferenceBasis.CLOSED_FORM,
    ) -> None:
        self.rows.append(
            {
                "pair": list(pair),
                "metric": metric,
                "parameters": list(parameters),
                "value": float(jnp.asarray(value)),
                "basis": str(basis),
            }
        )

    def frame(self) -> pl.DataFrame:
        return pl.DataFrame(self.rows, schema=REFERENCE_SCHEMA)


@eqx.filter_jit
def _draw(
    distribution: ZooDistribution,
    key: PRNGKeyArray,
    sample_count: int,
    batch_size: int,
) -> Float[Array, "batch sample param"]:
    problem_keys = jax.vmap(jax.random.fold_in, in_axes=(None, 0))(
        key, jnp.arange(batch_size)
    )
    return jax.vmap(
        lambda problem_key: distribution.sample(problem_key, (sample_count,))
    )(problem_keys)


@eqx.filter_jit
def _draw_ar1(
    coefficients: Float[Array, " param"],
    key: PRNGKeyArray,
    step_count: int,
    walker_count: int,
    start: float | None,
) -> Float[Array, "step walker param"]:
    shape = (walker_count, coefficients.shape[0])
    if start is None:
        initial = jax.random.normal(derive_key(key, "start"), shape)
    else:
        initial = jnp.full(shape, start)
    innovations = jax.random.normal(
        derive_key(key, "innovations"), (step_count - 1, *shape)
    )
    innovation_scale = jnp.sqrt(1 - coefficients**2)

    def step(previous: Array, innovation: Array) -> tuple[Array, Array]:
        current = coefficients * previous + innovation_scale * innovation
        return current, current

    _, later = jax.lax.scan(step, initial, innovations)
    return jnp.concatenate([initial[None], later], axis=0)


def _gaussian_case(
    case_name: str,
    names: tuple[str, ...],
    means: list[Array],
    covariances: list[Array],
) -> ZooCase:
    """Build a case of Gaussians with every closed-form reference value."""
    posteriors = tuple(
        Gaussian(mean, covariance)
        for mean, covariance in zip(means, covariances, strict=True)
    )
    param_names = _parameter_names(posteriors[0].param_count)
    _check_names(names)
    if len(names) != len(posteriors):
        raise ValueError(f"{len(names)} names for {len(posteriors)} posteriors")
    rows = _Rows()
    _moment_rows(rows, names, param_names, means, covariances)
    for first, second in itertools.combinations(range(len(names)), 2):
        _gaussian_pair_rows(
            rows,
            (names[first], names[second]),
            param_names,
            (means[first], covariances[first]),
            (means[second], covariances[second]),
        )
    return ZooCase(
        name=case_name,
        space=_real_space(param_names),
        names=names,
        posteriors=posteriors,
        reference=rows.frame(),
    )


def _moment_rows(
    rows: _Rows,
    names: Sequence[str],
    param_names: Sequence[str],
    means: Sequence[Array],
    covariances: Sequence[Array],
) -> None:
    """Add the moment rows: means, widths, shifts, width ratios, Mahalanobis shifts."""
    stds = [jnp.sqrt(jnp.diag(covariance)) for covariance in covariances]
    for name, mean, std in zip(names, means, stds, strict=True):
        for parameter, value, spread in zip(param_names, mean, std, strict=True):
            rows.add((name,), "mean", (parameter,), value)
            rows.add((name,), "std", (parameter,), spread)
    for first, second in itertools.permutations(range(len(names)), 2):
        pair = (names[first], names[second])
        for index, parameter in enumerate(param_names):
            rows.add(
                pair,
                "mean_shift",
                (parameter,),
                (means[second][index] - means[first][index]) / stds[first][index],
            )
            rows.add(
                pair,
                "width_ratio",
                (parameter,),
                stds[second][index] / stds[first][index],
            )
        rows.add(
            pair,
            "mahalanobis_shift",
            param_names,
            closed_forms.mahalanobis_shift(
                means[first], covariances[first], means[second]
            ),
        )


def _gaussian_pair_rows(
    rows: _Rows,
    pair: tuple[str, str],
    param_names: Sequence[str],
    first: tuple[Array, Array],
    second: tuple[Array, Array],
) -> None:
    """Add the joint and marginal closed forms of one pair of Gaussians."""
    for ordered, (source, target) in (
        (pair, (first, second)),
        (pair[::-1], (second, first)),
    ):
        kl = closed_forms.gaussian_kl(*source, *target)
        rows.add(ordered, "kl", param_names, kl.total)
        rows.add(ordered, "kl_location", param_names, kl.location)
        rows.add(ordered, "kl_shape", param_names, kl.shape)
    for metric, terms in (
        ("w2", closed_forms.gaussian_w2(*first, *second)),
        ("hellinger", closed_forms.gaussian_hellinger(*first, *second)),
    ):
        rows.add(pair, metric, param_names, terms.total)
        rows.add(pair, f"{metric}_location", param_names, terms.location)
        rows.add(pair, f"{metric}_shape", param_names, terms.shape)
    # * With one parameter the marginal Hellinger row would repeat the joint one.
    marginal_metrics = (
        _MARGINAL_METRICS[1:] if len(param_names) == 1 else _MARGINAL_METRICS
    )
    for index, parameter in enumerate(param_names):
        marginals = (
            float(first[0][index]),
            float(jnp.sqrt(first[1][index, index])),
            float(second[0][index]),
            float(jnp.sqrt(second[1][index, index])),
        )
        identical = marginals[:2] == marginals[2:]
        for metric, function, basis in marginal_metrics:
            # * Identical marginals give exact zeros, without quadrature.
            value = 0.0 if identical else function(*marginals)
            rows.add(
                pair,
                metric,
                (parameter,),
                value,
                ReferenceBasis.CLOSED_FORM if identical else ReferenceBasis(basis),
            )


def _mixture_moments(
    weights: Float[Array, " component"],
    component_means: Float[Array, "component param"],
) -> tuple[Array, Array]:
    """Return the mean and covariance of a mixture of unit-covariance Gaussians."""
    mean = weights @ component_means
    offsets = component_means - mean
    covariance = (
        jnp.eye(component_means.shape[1]) + (weights[:, None] * offsets).T @ offsets
    )
    return mean, covariance


def _normalised_weights(name: str, weights: Sequence[float]) -> Array:
    weights_read = vector(name, weights, 2)
    if np.any(weights_read <= 0):
        raise ValueError(f"{name} must be positive, got {tuple(weights)}")
    weights_array = jnp.asarray(weights_read)
    return weights_array / jnp.sum(weights_array)


def _base_gaussian(
    param_count: int, mean: object, covariance: object
) -> tuple[Array, Array]:
    """Return the checked mean and covariance of a reference Gaussian."""
    check_count("param_count", param_count)
    if mean is None:
        mean_array = jnp.zeros(param_count)
    else:
        mean_array = jnp.asarray(vector("mean", mean, param_count))
    if covariance is None:
        return mean_array, jnp.eye(param_count)
    covariance_array = jnp.asarray(matrix("covariance", covariance, param_count))
    cholesky(covariance_array, "covariance")
    return mean_array, covariance_array


def _axis_indices(name: str, axes: Sequence[int], param_count: int) -> list[int]:
    indices = [int(axis) for axis in axes]
    if not indices or len(set(indices)) != len(indices):
        raise ValueError(f"{name} must be distinct axis indices, got {tuple(axes)}")
    for index in indices:
        if not 0 <= index < param_count:
            raise ValueError(
                f"{name} index {index} is outside the {param_count} parameters"
            )
    return indices


def _parameter_names(count: int, start: int = 0) -> tuple[str, ...]:
    return tuple(f"x{index}" for index in range(start, start + count))


def _real_space(param_names: Sequence[str]) -> ParamSpace:
    return ParamSpace(parameters=tuple(Parameter(name=name) for name in param_names))


def _signed_arc(origin: ArrayLike, target: ArrayLike, period: float) -> Array:
    """Return the shortest signed arc from origin to target, in [−P/2, P/2)."""
    return jnp.mod(jnp.asarray(target) - origin + 0.5 * period, period) - 0.5 * period


def _check_names(names: Sequence[str]) -> None:
    if not names:
        raise ValueError("a case needs at least one posterior")
    for name in names:
        if not name or name != name.strip():
            raise ValueError(
                "posterior names must be non-empty, without surrounding"
                f" whitespace, got {name!r}"
            )
    repeated = sorted({name for name in names if names.count(name) > 1})
    if repeated:
        raise ValueError(f"posterior names must be unique, repeated: {repeated}")


def _check_reference(reference: pl.DataFrame) -> None:
    if reference.schema != REFERENCE_SCHEMA:
        raise ValueError(
            f"the reference table needs the schema {dict(REFERENCE_SCHEMA)}, got"
            f" {dict(reference.schema)}"
        )


def _parameter_list(
    parameters: str | Sequence[str] | None, space: ParamSpace
) -> list[str]:
    if parameters is None:
        return list(space.names)
    if isinstance(parameters, str):
        return [parameters]
    return list(parameters)


def _lookup(
    reference: pl.DataFrame, metric: str, pair: list[str], parameters: list[str]
) -> float:
    """Return the value of the row with this metric, pair and parameter subset."""
    for row in reference.iter_rows(named=True):
        if (
            row["metric"] == metric
            and row["pair"] == pair
            and row["parameters"] == parameters
        ):
            return float(row["value"])
    available = sorted(
        {
            f"{row['metric']} over {row['parameters']}"
            for row in reference.iter_rows(named=True)
            if row["pair"] == pair
        }
    )
    raise KeyError(
        f"no reference value {metric!r} for the pair {pair} over {parameters};"
        f" that pair has: {'; '.join(available) or 'no rows'}"
    )
