"""Whitening transforms: reference, pooled and Fisher.

Whitening expresses a parameter space in units of a posterior width, so that a
unit step means the same in every direction (Kessy, Lewin and Strimmer 2018).
Both inputs of a comparison share one whitening, recorded in the ``space``
column of every result row.

Blocks
    A whitening treats the parameters by their geometry, as
    :attr:`~mixed_signals.core.space.ParamSpace.axis_blocks` groups them. The
    Euclidean block, the real and interval parameters outside spheres, is
    centred and whitened as a whole. A circle is only divided by its circular
    standard deviation s (Mardia and Jupp 2000), so it is never mixed with
    another axis and stays a circle, of period P/s. Sphere pairs pass
    unchanged, because great-circle distances need radians and an azimuth of
    period 2π.
Sources
    ``reference``: the covariance of one posterior, as in mode A.
    ``pooled``: the mean of the covariances of the posteriors given, each
    weighted equally, which is the pooled within-group covariance of the
    Mahalanobis distance between groups (Mahalanobis 1936; Johnson and
    Wichern 2007, Section 6.3); circles take the root mean square of their
    circular standard deviations. The covariance of all samples together would
    add the spread between the means, which shrinks a Mahalanobis shift d
    between two equal Gaussians to d/√(1 + d²/4), below 2 for every d. Engines
    pool the two posteriors of each pair, so a row does not depend on the
    other posteriors of a run.
    ``fisher``: the inverse Γ⁻¹ of a Fisher matrix, the posterior covariance at
    high signal-to-noise ratio (Vallisneri 2008). The Euclidean block takes its
    marginal covariance, the Euclidean block of Γ⁻¹, and each circle the square
    root of its diagonal entry, so a Fisher whitening matches the covariance
    whitening of such a posterior.
Rotation
    Whitening matrices are unique up to a rotation. The Euclidean block uses
    ZCA-cor whitening, W = P^(−1/2) V^(−1/2), with P the correlation matrix and
    V the diagonal matrix of variances: of all whitenings, its axes correlate
    most with their own parameters, and it does not depend on the units of the
    parameters (Kessy, Lewin and Strimmer 2018). Each whitened axis therefore
    keeps the name of its parameter. With ``decorrelate=False`` the block is
    only scaled, W = V^(−1/2), a z-score that keeps interval bounds as bounds.
Conditioning
    The checks act on matrices scaled to unit diagonal, whose condition number
    κ lies within a factor D of the smallest over all diagonal scalings
    (van der Sluis 1969), so the units of the parameters do not enter. A matrix
    whose smallest eigenvalue lies below D·ε times its largest is singular to
    working precision. Inversion loses up to log₁₀ κ digits (Vallisneri 2008),
    so κ above :data:`CONDITION_NUMBER_LIMIT` raises a
    :class:`~mixed_signals.errors.ValidityWarning`.
Invalid inputs
    A singular covariance from samples, from a parameter without spread, too
    few samples with positive weight or collinear parameters, gives a
    whitening of NaN and a :class:`~mixed_signals.errors.ValidityWarning`, so
    the other problems of a batch go on. A Fisher matrix that is not finite,
    symmetric and positive definite raises ValueError. Inside ``jax.jit`` and
    ``jax.vmap`` the values are unknown, so nothing warns or raises, and
    invalid inputs give NaN.

References:
    :cite:t:`kessy2018optimal`; :cite:t:`mahalanobis1936generalised`;
    :cite:t:`johnson2007applied`; :cite:t:`mardia2000directional`;
    :cite:t:`vallisneri2008use`; :cite:t:`vandersluis1969condition`.
"""

import math
import os
import warnings
from collections.abc import Callable, Sequence
from enum import StrEnum
from typing import Final, NamedTuple, cast

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike
from jaxtyping import Array, Float

from mixed_signals.core.samples import circular_standard_deviation, weighted_covariance
from mixed_signals.core.space import AxisBlocks, Parameter, ParamSpace
from mixed_signals.errors import PrecisionWarning, ValidityWarning
from mixed_signals.transforms.base import Transform

FLOAT64_EPSILON: Final = float(np.finfo(np.float64).eps)
"""Spacing of float64 numbers at one, ε = 2⁻⁵² ≈ 2.2e-16 (IEEE 2019).

References:
    :cite:t:`ieee2019ieee`.
"""

CONDITION_NUMBER_LIMIT: Final = 1 / math.sqrt(FLOAT64_EPSILON)
"""Condition number above which a whitening warns, 1/√ε = 2²⁶ ≈ 6.7e7.

Inversion loses up to log₁₀ κ digits (Vallisneri 2008), so beyond this limit
the whitening may have lost more than half of its 16 digits.

References:
    :cite:t:`vallisneri2008use`.
"""

SYMMETRY_TOLERANCE: Final = 1e-10
"""Largest asymmetry of a Fisher matrix scaled to unit diagonal, as in the zoo."""

_PACKAGE_DIRECTORY: Final = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# * Loadings below this share of the largest one are left out of a direction's text.
_DIRECTION_SHARE: Final = 0.1


class WhiteningSource(StrEnum):
    """Where a whitening takes its widths from."""

    REFERENCE = "reference"
    """The covariance of one posterior."""
    POOLED = "pooled"
    """The mean of the covariances of several posteriors, each weighted equally."""
    FISHER = "fisher"
    """The inverse of a Fisher matrix."""


class _Parts(NamedTuple):
    """Arrays of a whitening, with the unmasked widths and spectrum that checks read."""

    centre: Array
    matrix: Array
    inverse_matrix: Array
    circle_scales: Array
    log_abs_det: Array
    variances: Array
    circle_widths: Array
    eigenvalues: Array
    eigenvectors: Array


class _Spectrum(NamedTuple):
    """Diagonal, asymmetry and spectrum of a Fisher matrix scaled to unit diagonal."""

    diagonal: Array
    asymmetry: Array
    eigenvalues: Array
    eigenvectors: Array


class Whitening(Transform):
    """Whitening of one problem: the Euclidean block whitened, circles scaled.

    The Euclidean block is centred and whitened, y_E = W (x_E − c); each circle
    is divided by its scale, y = x/s; sphere pairs stay as they are. Build a
    whitening with :meth:`from_samples` or :meth:`from_fisher`.

    Attributes:
        centre: Centre c of the Euclidean block, shaped (E,).
        matrix: Whitening matrix W of the Euclidean block, shaped (E, E).
        inverse_matrix: Its inverse W⁻¹, shaped (E, E).
        circle_scales: Scale s of each circle, its circular standard deviation
            in the units of the circle, shaped (C,).
        log_abs_det: ln|det ∂f/∂x|, the same at every point.
        blocks: Positions of the Euclidean block, the circles and the spheres.
        source: Where the widths come from.
        decorrelate: Whether the Euclidean block is decorrelated (ZCA-cor) or
            only scaled (z-score).

    References:
        :cite:t:`kessy2018optimal`; :cite:t:`mardia2000directional`.
    """

    centre: Float[Array, " euclidean"]
    matrix: Float[Array, "euclidean euclidean"]
    inverse_matrix: Float[Array, "euclidean euclidean"]
    circle_scales: Float[Array, " circle"]
    log_abs_det: Float[Array, ""]
    blocks: AxisBlocks = eqx.field(static=True)
    source: WhiteningSource = eqx.field(static=True)
    decorrelate: bool = eqx.field(static=True)

    @classmethod
    def from_samples(
        cls,
        samples: ArrayLike,
        log_weights: ArrayLike | None,
        space: ParamSpace,
        *,
        decorrelate: bool = True,
    ) -> "Whitening":
        """Build the whitening of one problem from the samples of its posteriors.

        One posterior gives a ``reference`` whitening, and several give a
        ``pooled`` one from the mean of their covariances, each posterior
        weighted equally. Covariances are the unbiased weighted estimates of
        :func:`~mixed_signals.core.samples.weighted_covariance`, the circle
        widths those of
        :func:`~mixed_signals.core.samples.circular_standard_deviation`, and
        the centre is the mean of the posterior means.

        Args:
            samples: Samples of K posteriors in the parameter order of
                ``space``, shaped (K, N, D); K = 1 for a reference whitening.
            log_weights: Log weights shaped (K, N); equal weights when
                ``None``.
            space: Parameter space of the samples.
            decorrelate: Whether to decorrelate the Euclidean block (ZCA-cor)
                or only scale it (z-score).

        Returns:
            The whitening, NaN if a covariance is singular.

        Raises:
            ValueError: If the shapes do not fit the space, concrete samples
                are not finite, or a posterior has no sample with positive
                weight.
            TypeError: If an input is not real-valued.

        Warns:
            ValidityWarning: If concrete samples give a singular or
                ill-conditioned covariance, or a circle without spread.
            PrecisionWarning: If an input is narrower than float64.

        References:
            :cite:t:`mahalanobis1936generalised`; :cite:t:`johnson2007applied`;
            :cite:t:`kessy2018optimal`; :cite:t:`vandersluis1969condition`.
        """
        size = len(space)
        samples_array = _float64("samples", samples)
        if (
            samples_array.ndim != 3
            or samples_array.shape[-1] != size
            or 0 in samples_array.shape
        ):
            raise ValueError(
                f"samples must be shaped (K, N, {size}) for a space of {size}"
                f" parameters, got shape {samples_array.shape}"
            )
        weights_shape = samples_array.shape[:2]
        if log_weights is None:
            weights_array = jnp.zeros(weights_shape)
        else:
            weights_array = _float64("log_weights", log_weights)
            if weights_array.shape != weights_shape:
                raise ValueError(
                    f"log_weights must be shaped {weights_shape} for samples of"
                    f" shape {samples_array.shape}, got shape {weights_array.shape}"
                )
        if _is_concrete(samples_array, weights_array):
            _check_samples(samples_array, weights_array)
        blocks = space.axis_blocks
        periods = tuple(
            space.parameters[position].bounds[1] - space.parameters[position].bounds[0]
            for position in blocks.circles
        )
        parts = _sample_parts(
            samples_array,
            weights_array,
            blocks=blocks,
            periods=periods,
            decorrelate=decorrelate,
        )
        if _is_concrete(parts.log_abs_det):
            _warn_about_parts(parts, space, decorrelate)
        source = (
            WhiteningSource.REFERENCE
            if samples_array.shape[0] == 1
            else WhiteningSource.POOLED
        )
        return cls._from_parts(parts, blocks, source, decorrelate)

    @classmethod
    def from_fisher(
        cls,
        fisher: ArrayLike,
        centre: ArrayLike,
        space: ParamSpace,
        *,
        decorrelate: bool = True,
    ) -> "Whitening":
        """Build the whitening of one problem from a Fisher matrix.

        The covariance is Γ⁻¹, the posterior covariance at high
        signal-to-noise ratio (Vallisneri 2008). The Euclidean block is
        whitened with its marginal covariance, the Euclidean block of Γ⁻¹, and
        each circle scaled by the square root of its diagonal entry of Γ⁻¹.

        Args:
            fisher: Fisher matrix Γ over all parameters of ``space``, shaped
                (D, D).
            centre: Point at which Γ holds, such as the injected or most
                probable parameters, shaped (D,); it centres the Euclidean
                block.
            space: Parameter space of Γ.
            decorrelate: Whether to decorrelate the Euclidean block (ZCA-cor)
                or only scale it (z-score).

        Returns:
            The whitening.

        Raises:
            ValueError: If the shapes do not fit the space, or concrete inputs
                are not finite, or a concrete Γ is not symmetric and positive
                definite to working precision.
            TypeError: If an input is not real-valued.

        Warns:
            ValidityWarning: If a concrete Γ, scaled to unit diagonal, has a
                condition number above :data:`CONDITION_NUMBER_LIMIT`.
            PrecisionWarning: If an input is narrower than float64.

        References:
            :cite:t:`vallisneri2008use`; :cite:t:`vandersluis1969condition`;
            :cite:t:`kessy2018optimal`.
        """
        size = len(space)
        fisher_array = _float64("fisher", fisher)
        centre_array = _float64("centre", centre)
        if fisher_array.shape != (size, size):
            raise ValueError(
                f"fisher must be shaped ({size}, {size}) for a space of {size}"
                f" parameters, got shape {fisher_array.shape}"
            )
        if centre_array.shape != (size,):
            raise ValueError(
                f"centre must be shaped ({size},) for a space of {size} parameters,"
                f" got shape {centre_array.shape}"
            )
        if _is_concrete(fisher_array, centre_array):
            _check_fisher(fisher_array, centre_array, space)
        blocks = space.axis_blocks
        parts = _fisher_parts(
            fisher_array, centre_array, blocks=blocks, decorrelate=decorrelate
        )
        return cls._from_parts(parts, blocks, WhiteningSource.FISHER, decorrelate)

    @classmethod
    def _from_parts(
        cls,
        parts: _Parts,
        blocks: AxisBlocks,
        source: WhiteningSource,
        decorrelate: bool,
    ) -> "Whitening":
        return cls(
            centre=parts.centre,
            matrix=parts.matrix,
            inverse_matrix=parts.inverse_matrix,
            circle_scales=parts.circle_scales,
            log_abs_det=parts.log_abs_det,
            blocks=blocks,
            source=source,
            decorrelate=decorrelate,
        )

    def forward(
        self, points: Float[Array, "*leading param"]
    ) -> Float[Array, "*leading param"]:
        """Return whitened points."""
        euclidean = _positions(self.blocks.euclidean)
        circles = _positions(self.blocks.circles)
        whitened = (points[..., euclidean] - self.centre) @ self.matrix.T
        scaled = points[..., circles] / self.circle_scales
        return points.at[..., euclidean].set(whitened).at[..., circles].set(scaled)

    def inverse(
        self, points: Float[Array, "*leading param"]
    ) -> Float[Array, "*leading param"]:
        """Return the original coordinates of whitened points."""
        euclidean = _positions(self.blocks.euclidean)
        circles = _positions(self.blocks.circles)
        restored = points[..., euclidean] @ self.inverse_matrix.T + self.centre
        unscaled = points[..., circles] * self.circle_scales
        return points.at[..., euclidean].set(restored).at[..., circles].set(unscaled)

    def log_abs_det_jacobian(
        self, points: Float[Array, "*leading param"]
    ) -> Float[Array, " *leading"]:
        """Return ln|det ∂f/∂x|, which is the same at every point."""
        return jnp.broadcast_to(self.log_abs_det, points.shape[:-1])

    @property
    def description(self) -> str:
        """``whiten(<source>)``, or ``zscore(<source>)`` without decorrelation."""
        operation = "whiten" if self.decorrelate else "zscore"
        return f"{operation}({self.source})"

    def map_space(self, space: ParamSpace) -> ParamSpace:
        """Return the space of the whitened coordinates.

        Whitened coordinates have no unit. A decorrelated Euclidean axis
        becomes a real line, because its interval bounds turn into oblique
        planes; methods that need bounds run in the original space. A z-scored
        axis keeps its topology, with its bounds mapped. A circle keeps its
        topology, with its bounds, and so its period, divided by its scale.
        Sphere axes stay as they are. The method reads concrete values, so it
        runs outside ``jax.jit``.

        Args:
            space: Space of the original coordinates, with the axis blocks of
                the whitening.

        Returns:
            The space of the whitened coordinates.

        Raises:
            ValueError: If the space has other axis blocks than the whitening,
                or the whitening is NaN.
        """
        if space.axis_blocks != self.blocks:
            raise ValueError(
                f"the space has the axis blocks {space.axis_blocks}, but the"
                f" whitening was built for {self.blocks}"
            )
        if math.isnan(float(self.log_abs_det)):
            raise ValueError("the whitening is NaN, so it maps to no space")
        centre = np.asarray(self.centre)
        scales = np.asarray(jnp.diagonal(self.matrix))
        circle_scales = np.asarray(self.circle_scales)
        parameters = list(space.parameters)
        for index, position in enumerate(self.blocks.euclidean):
            original = parameters[position]
            entry: dict[str, object] = {"name": original.name, "label": original.label}
            if not self.decorrelate:
                entry["topology"] = original.topology
                entry["bounds"] = tuple(
                    (bound - centre[index]) * scales[index] for bound in original.bounds
                )
            parameters[position] = Parameter.model_validate(entry)
        for index, position in enumerate(self.blocks.circles):
            original = parameters[position]
            parameters[position] = Parameter.model_validate(
                {
                    "name": original.name,
                    "label": original.label,
                    "topology": original.topology,
                    "bounds": tuple(
                        bound / circle_scales[index] for bound in original.bounds
                    ),
                }
            )
        return ParamSpace(parameters=tuple(parameters), spheres=space.spheres)


def _jit[**Parameters, Result](
    function: Callable[Parameters, Result], *static_names: str
) -> Callable[Parameters, Result]:
    """Compile a kernel with :func:`jax.jit` and keep its signature for type checks."""
    return cast(
        Callable[Parameters, Result],
        jax.jit(function, static_argnames=static_names),
    )


def _positions(positions: tuple[int, ...]) -> np.ndarray:
    """Return parameter positions as an index array, empty when there are none."""
    return np.asarray(positions, dtype=np.intp)


def _float64(name: str, value: ArrayLike) -> Array:
    """Return an input as a float64 array, warning when it was narrower."""
    array = jnp.asarray(value)
    if not (
        jnp.issubdtype(array.dtype, jnp.floating)
        or jnp.issubdtype(array.dtype, jnp.integer)
    ):
        raise TypeError(f"{name} must hold real numbers, got dtype {array.dtype}")
    if jnp.issubdtype(array.dtype, jnp.floating) and array.dtype.itemsize < 8:
        warnings.warn(
            PrecisionWarning(
                f"cast {name} ({array.dtype}) to float64; the cast cannot restore"
                " precision lost before it"
            ),
            skip_file_prefixes=(_PACKAGE_DIRECTORY,),
        )
    return array.astype(jnp.float64)


def _is_concrete(*arrays: Array) -> bool:
    """Return whether every array holds known values, outside a JAX trace."""
    return not any(isinstance(array, jax.core.Tracer) for array in arrays)


def _whitening_parts(
    centre: Float[Array, " euclidean"],
    covariance: Float[Array, "euclidean euclidean"],
    circle_scales: Float[Array, " circle"],
    decorrelate: bool,
) -> _Parts:
    """Return the arrays of a whitening, NaN when its widths are singular."""
    variances = jnp.diagonal(covariance)
    widths = jnp.sqrt(variances)
    correlation = covariance / jnp.outer(widths, widths)
    # * eigh reads one triangle only, so average away the rounding between them.
    eigenvalues, eigenvectors = jnp.linalg.eigh(0.5 * (correlation + correlation.T))
    if decorrelate:
        # * ZCA-cor: W = P^(-1/2) V^(-1/2) and its inverse V^(1/2) P^(1/2).
        matrix = (eigenvectors / jnp.sqrt(eigenvalues)) @ eigenvectors.T / widths
        inverse_matrix = widths[:, None] * (
            (eigenvectors * jnp.sqrt(eigenvalues)) @ eigenvectors.T
        )
        block_log_det = -0.5 * jnp.sum(jnp.log(eigenvalues)) - jnp.sum(jnp.log(widths))
    else:
        matrix = jnp.diag(1 / widths)
        inverse_matrix = jnp.diag(widths)
        block_log_det = -jnp.sum(jnp.log(widths))
    valid = (
        jnp.all(jnp.isfinite(centre))
        & jnp.all(jnp.isfinite(covariance))
        & jnp.all(variances > 0)
        & jnp.all(jnp.isfinite(circle_scales))
        & jnp.all(circle_scales > 0)
    )
    if decorrelate:
        size = variances.shape[0]
        smallest = jnp.min(eigenvalues, initial=jnp.inf)
        largest = jnp.max(eigenvalues, initial=0.0)
        valid = valid & (smallest > size * FLOAT64_EPSILON * largest)
    log_abs_det = block_log_det - jnp.sum(jnp.log(circle_scales))
    return _Parts(
        centre=jnp.where(valid, centre, jnp.nan),
        matrix=jnp.where(valid, matrix, jnp.nan),
        inverse_matrix=jnp.where(valid, inverse_matrix, jnp.nan),
        circle_scales=jnp.where(valid, circle_scales, jnp.nan),
        log_abs_det=jnp.where(valid, log_abs_det, jnp.nan),
        variances=variances,
        circle_widths=circle_scales,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
    )


def _sample_parts_kernel(
    samples: Float[Array, "posterior sample param"],
    log_weights: Float[Array, "posterior sample"],
    blocks: AxisBlocks,
    periods: tuple[float, ...],
    decorrelate: bool,
) -> _Parts:
    """Return the arrays of a whitening from the samples of K posteriors."""
    euclidean = _positions(blocks.euclidean)
    means, covariances = weighted_covariance(samples[..., euclidean], log_weights)
    # * Pooled circles take the root mean square, as variances average for the block.
    circle_scales = jnp.asarray(
        [
            jnp.sqrt(
                jnp.mean(
                    circular_standard_deviation(
                        samples[..., position], log_weights, period
                    )
                    ** 2
                )
            )
            for position, period in zip(blocks.circles, periods, strict=True)
        ],
        dtype=jnp.float64,
    )
    return _whitening_parts(
        jnp.mean(means, axis=0),
        jnp.mean(covariances, axis=0),
        circle_scales,
        decorrelate,
    )


def _fisher_spectrum_kernel(fisher: Float[Array, "param param"]) -> _Spectrum:
    """Return the diagonal, asymmetry and spectrum of Γ scaled to unit diagonal."""
    diagonal = jnp.diagonal(fisher)
    scale = 1 / jnp.sqrt(diagonal)
    unit = fisher * jnp.outer(scale, scale)
    eigenvalues, eigenvectors = jnp.linalg.eigh(0.5 * (unit + unit.T))
    return _Spectrum(
        diagonal, jnp.max(jnp.abs(unit - unit.T)), eigenvalues, eigenvectors
    )


def _fisher_parts_kernel(
    fisher: Float[Array, "param param"],
    centre: Float[Array, " param"],
    blocks: AxisBlocks,
    decorrelate: bool,
) -> _Parts:
    """Return the arrays of a whitening from a Fisher matrix and its centre."""
    spectrum = _fisher_spectrum_kernel(fisher)
    scale = 1 / jnp.sqrt(spectrum.diagonal)
    size = fisher.shape[0]
    regular = (
        jnp.all(spectrum.diagonal > 0)
        & jnp.all(jnp.isfinite(spectrum.eigenvalues))
        & (spectrum.eigenvalues[0] > size * FLOAT64_EPSILON * spectrum.eigenvalues[-1])
    )
    # * The inverse in the scaled basis, Γ⁻¹ = S U Λ⁻¹ Uᵀ S, keeps GW scales exact.
    covariance = jnp.outer(scale, scale) * (
        (spectrum.eigenvectors / spectrum.eigenvalues) @ spectrum.eigenvectors.T
    )
    covariance = jnp.where(regular, covariance, jnp.nan)
    euclidean = _positions(blocks.euclidean)
    circles = _positions(blocks.circles)
    return _whitening_parts(
        centre[euclidean],
        covariance[np.ix_(euclidean, euclidean)],
        jnp.sqrt(jnp.diagonal(covariance)[circles]),
        decorrelate,
    )


_sample_parts = _jit(_sample_parts_kernel, "blocks", "periods", "decorrelate")
_fisher_parts = _jit(_fisher_parts_kernel, "blocks", "decorrelate")
_fisher_spectrum = _jit(_fisher_spectrum_kernel)


def _names_text(names: Sequence[str]) -> str:
    """Return parameter names as quoted text, joined by commas."""
    return ", ".join(map(repr, names))


def _direction(vector: Array, names: Sequence[str]) -> str:
    """Return a unit vector as a combination of named parameters, largest first."""
    # ! Python floats only: NumPy flags invalid operations differently per platform.
    loadings = [float(loading) for loading in np.asarray(vector)]
    printed = [float(f"{abs(loading):.3g}") for loading in loadings]
    # * Loadings that print alike keep parameter order, whatever their last bits.
    order = sorted(range(len(loadings)), key=printed.__getitem__, reverse=True)
    sign = -1.0 if loadings[order[0]] < 0 else 1.0
    pieces = [
        f"{'-' if sign * loadings[position] < 0 else '+'}"
        f" {printed[position]:.3g} {names[position]}"
        for position in order
        if printed[position] >= _DIRECTION_SHARE * printed[order[0]]
    ]
    return " ".join(pieces).removeprefix("+ ")


def _warn(message: str) -> None:
    """Raise a validity warning that points at the caller's line."""
    warnings.warn(ValidityWarning(message), skip_file_prefixes=(_PACKAGE_DIRECTORY,))


def _check_samples(samples: Array, log_weights: Array) -> None:
    """Raise unless concrete samples and log weights can give widths."""
    if not bool(jnp.all(jnp.isfinite(samples))):
        raise ValueError("samples must be finite")
    if bool(jnp.any(jnp.isnan(log_weights) | (log_weights == jnp.inf))):
        raise ValueError("log_weights must not contain NaN or +inf")
    if not bool(jnp.all(jnp.any(log_weights > -jnp.inf, axis=-1))):
        raise ValueError(
            "every posterior needs a sample with positive weight; at least one"
            " log weight per posterior must exceed -inf"
        )


def _warn_about_parts(parts: _Parts, space: ParamSpace, decorrelate: bool) -> None:
    """Warn when concrete sample widths are singular or ill-conditioned."""
    if math.isnan(float(parts.log_abs_det)):
        _warn(f"the whitening is NaN: {_degeneracy(parts, space)}")
        return
    if decorrelate and parts.eigenvalues.size:
        condition = float(parts.eigenvalues[-1]) / float(parts.eigenvalues[0])
        if condition > CONDITION_NUMBER_LIMIT:
            _warn(
                "the correlation matrix of the Euclidean block has condition number"
                f" {condition:.3g}, above 1/sqrt(eps) = {CONDITION_NUMBER_LIMIT:.3g}:"
                " the whitening may have lost more than half of its digits along"
                f" {_weakest_direction(parts, space)}"
            )


def _weakest_direction(parts: _Parts, space: ParamSpace) -> str:
    """Return the weakest direction of the Euclidean block as named parameters."""
    names = [space.names[position] for position in space.axis_blocks.euclidean]
    return _direction(parts.eigenvectors[:, 0], names)


def _degeneracy(parts: _Parts, space: ParamSpace) -> str:
    """Return why the widths of a whitening are singular."""
    blocks = space.axis_blocks
    variances = [float(variance) for variance in np.asarray(parts.variances)]
    if any(math.isnan(variance) for variance in variances):
        return "a posterior has fewer than two samples with positive weight"
    flat = [
        space.names[position]
        for position, variance in zip(blocks.euclidean, variances, strict=True)
        if not variance > 0
    ]
    if flat:
        verb = "has" if len(flat) == 1 else "have"
        return f"{_names_text(flat)} {verb} no spread"
    circles = [
        space.names[position]
        for position, width in zip(
            blocks.circles, np.asarray(parts.circle_widths).tolist(), strict=True
        )
        if not 0 < width < math.inf
    ]
    if circles:
        return (
            f"the circular standard deviation of {_names_text(circles)} is zero or"
            " infinite: the values coincide, or spread evenly around the circle"
        )
    return (
        "the Euclidean block is singular, with the correlation eigenvalue"
        f" {float(parts.eigenvalues[0]):.3g} along {_weakest_direction(parts, space)}"
    )


def _check_fisher(fisher: Array, centre: Array, space: ParamSpace) -> None:
    """Raise unless a concrete Fisher matrix is usable; warn if ill-conditioned."""
    if not bool(jnp.all(jnp.isfinite(fisher))):
        raise ValueError("fisher must be finite")
    if not bool(jnp.all(jnp.isfinite(centre))):
        raise ValueError("centre must be finite")
    spectrum = _fisher_spectrum(fisher)
    diagonal = np.asarray(spectrum.diagonal).tolist()
    nonpositive = [
        name
        for name, information in zip(space.names, diagonal, strict=True)
        if not information > 0
    ]
    if nonpositive:
        raise ValueError(
            "fisher needs a positive diagonal, but the information of"
            f" {_names_text(nonpositive)} is not positive"
        )
    asymmetry = float(spectrum.asymmetry)
    if asymmetry > SYMMETRY_TOLERANCE:
        raise ValueError(
            "fisher must be symmetric: scaled to unit diagonal, it differs from its"
            f" transpose by up to {asymmetry:.3g}, above {SYMMETRY_TOLERANCE:.0e};"
            " symmetrise it with (fisher + fisher.T) / 2 if numerical derivatives"
            " cause the difference"
        )
    smallest = float(spectrum.eigenvalues[0])
    largest = float(spectrum.eigenvalues[-1])
    direction = _direction(spectrum.eigenvectors[:, 0], space.names)
    if not smallest > len(space) * FLOAT64_EPSILON * largest:
        raise ValueError(
            "fisher is singular or not positive definite: scaled to unit diagonal,"
            f" its smallest eigenvalue {smallest:.3g} lies below D*eps times"
            f" its largest, {largest:.3g}, along {direction}"
        )
    condition = largest / smallest
    if condition > CONDITION_NUMBER_LIMIT:
        _warn(
            f"fisher has condition number {condition:.3g} after scaling to unit"
            f" diagonal, above 1/sqrt(eps) = {CONDITION_NUMBER_LIMIT:.3g}: its"
            f" inverse may have lost more than half of its digits along {direction}"
        )
