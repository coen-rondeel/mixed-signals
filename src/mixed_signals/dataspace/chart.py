"""``FisherChart``: the joint local chart of a Fisher matrix, with circles unwrapped.

At linear order, the squared data-space distance between the signals of two
nearby parameter vectors is ΔθᵀΓΔθ for the Fisher matrix Γ (Vallisneri 2008).
:class:`FisherChart` maps parameters to y = W·arc(θ − c), where c is the centre
at which Γ holds, arc takes the shortest signed arc on every circle axis and
the plain difference on the other axes, and W satisfies WᵀW = Γ. Then
‖y₁ − y₂‖² = Δθᵀ Γ Δθ for two samples on the same side of the cut, so the
energy distance of the charted samples is the linear-order data-space distance
in the same units, the noise standard deviation.

Joint chart
    W is the ZCA-cor whitening of Σ = Γ⁻¹ over all axes, W = R^(−1/2) V^(−1/2)
    with V = diag(Σ) and the correlation matrix R = V^(−1/2) Σ V^(−1/2) (Kessy,
    Lewin and Strimmer 2018), so each chart axis keeps its parameter's name. The
    block form of
    :meth:`~mixed_signals.transforms.whiten.Whitening.from_fisher` scales each
    circle by its own entry of Γ⁻¹ and the Euclidean block by its marginal
    block, which drops the cross terms between correlated circles and other
    axes, such as the phase, the polarisation and the inclination of a
    Galactic binary. A distance in the chart would then differ from ΔθᵀΓΔθ.
Cut
    The arc wraps a difference into [−P/2, P/2) on a circle of period P, so the
    chart cuts every circle half a period from the centre. Two samples on
    opposite sides of the cut sit a period apart in the chart, where the
    circle has them close together. The chart is exact for a posterior that
    stays within half a period of the centre.
Wide posteriors
    :meth:`FisherChart.warn_wide_axes` raises a
    :class:`~mixed_signals.errors.ValidityWarning` when the weighted root mean
    square arc of a circle axis from the centre exceeds
    P/(2·:data:`CHART_CUT_WIDTHS`), the width within which a Gaussian
    posterior stays on its own side of the cut (circular statistics follow
    Mardia and Jupp 2000).

References:
    :cite:t:`vallisneri2008use`; :cite:t:`kessy2018optimal`;
    :cite:t:`mardia2000directional`.
"""

import os
import warnings
from typing import Final

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from jaxtyping import Array, Float

from mixed_signals.core.samples import _is_concrete
from mixed_signals.core.space import AxisBlocks, Parameter, ParamSpace
from mixed_signals.core.topology import wrap
from mixed_signals.errors import ValidityWarning
from mixed_signals.transforms.base import Transform
from mixed_signals.transforms.whiten import _check_fisher, _fisher_parts, _float64

CHART_CUT_WIDTHS: Final = 3.0
"""Widths of a Gaussian posterior that fit between the centre and the cut.

The chart cuts each circle half a period from the centre, and a Gaussian of
width P/(2·3) puts 0.27% of its mass beyond three widths, which then lands on
the far side of the cut. A circle axis whose posterior has a larger root mean
square arc from the centre raises a
:class:`~mixed_signals.errors.ValidityWarning`.
"""

CHART_DESCRIPTION: Final = "chart(fisher)"
"""Description of the chart in the ``space`` column of result rows."""

_PACKAGE_DIRECTORY: Final = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class FisherChart(Transform):
    """The joint local chart y = W·arc(θ − c) of the Fisher matrix of one problem.

    Build a chart with :meth:`from_fisher`. Its distances are ΔθᵀΓΔθ for
    samples on the same side of the cut of every circle.

    Attributes:
        centre: The centre c at which the Fisher matrix holds, shaped (D,).
        matrix: The matrix W with WᵀW = Γ, shaped (D, D).
        inverse_matrix: Its inverse, shaped (D, D).
        log_abs_det: ln det W, the same at every point; the arcs have a unit
            Jacobian away from the cut.
        names: The parameter names, in the order of the axes.
        periods: The period of every circle axis, ``None`` for the others.
        lower_bounds: The lower bound of every circle axis, ``None`` for the
            others, which the inverse wraps back to.

    References:
        :cite:t:`vallisneri2008use`; :cite:t:`kessy2018optimal`.
    """

    centre: Float[Array, " param"]
    matrix: Float[Array, "param param"]
    inverse_matrix: Float[Array, "param param"]
    log_abs_det: Float[Array, ""]
    names: tuple[str, ...] = eqx.field(static=True)
    periods: tuple[float | None, ...] = eqx.field(static=True)
    lower_bounds: tuple[float | None, ...] = eqx.field(static=True)

    @classmethod
    def from_fisher(
        cls, fisher: ArrayLike, centre: ArrayLike, space: ParamSpace
    ) -> "FisherChart":
        """Build the chart of one problem from a Fisher matrix and its centre.

        Args:
            fisher: The Fisher matrix Γ over all parameters of ``space``,
                shaped (D, D).
            centre: The point at which Γ holds, shaped (D,).
            space: The parameter space of Γ; its circle axes, sphere azimuths
                included, are unwrapped about the centre.

        Returns:
            The chart.

        Raises:
            ValueError: If the shapes do not fit the space, or a concrete Γ or
                centre is not finite, or Γ is not symmetric and positive
                definite to working precision.
            TypeError: If an input is not real-valued.

        Warns:
            ValidityWarning: If a concrete Γ, scaled to unit diagonal, has a
                condition number above
                :data:`~mixed_signals.transforms.whiten.CONDITION_NUMBER_LIMIT`.
            PrecisionWarning: If an input is narrower than float64.

        References:
            :cite:t:`vallisneri2008use`; :cite:t:`kessy2018optimal`.
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
            # * Eager: a closed-over matrix would stage these checks inside a trace.
            with jax.ensure_compile_time_eval():
                _check_fisher(fisher_array, centre_array, space)
        # * Every axis in the Euclidean block gives one ZCA-cor whitening of Γ⁻¹.
        parts = _fisher_parts(
            fisher_array,
            centre_array,
            blocks=AxisBlocks(tuple(range(size)), (), ()),
            polar_forms=(),
            decorrelate=True,
        )
        circles = [parameter.period is not None for parameter in space.parameters]
        return cls(
            centre=parts.centre,
            matrix=parts.matrix,
            inverse_matrix=parts.inverse_matrix,
            log_abs_det=parts.log_abs_det,
            names=space.names,
            periods=tuple(
                parameter.period if circle else None
                for parameter, circle in zip(space.parameters, circles, strict=True)
            ),
            lower_bounds=tuple(
                parameter.bounds[0] if circle else None
                for parameter, circle in zip(space.parameters, circles, strict=True)
            ),
        )

    def forward(
        self, points: Float[Array, "*leading param"]
    ) -> Float[Array, "*leading param"]:
        """Return y = W·arc(θ − c): arcs on the circles, differences elsewhere."""
        return self._arcs(points) @ self.matrix.T

    def inverse(
        self, points: Float[Array, "*leading param"]
    ) -> Float[Array, "*leading param"]:
        """Return θ = W⁻¹ y + c, with each circle axis wrapped into its range."""
        restored = points @ self.inverse_matrix.T + self.centre
        lower, period, is_circle = self._circle_arrays()
        return jnp.where(is_circle, wrap(restored, lower, period), restored)

    def log_abs_det_jacobian(
        self, points: Float[Array, "*leading param"]
    ) -> Float[Array, " *leading"]:
        """Return ln det W, which is the same at every point."""
        return jnp.broadcast_to(self.log_abs_det, points.shape[:-1])

    @property
    def description(self) -> str:
        """``chart(fisher)``."""
        return CHART_DESCRIPTION

    def map_space(self, space: ParamSpace) -> ParamSpace:
        """Return the space of the chart: every axis a real line, no spheres.

        Each axis keeps the name and the label of its parameter and has no
        unit. The axes are not the original parameters, so no topology or
        bound carries over.

        Args:
            space: The space of the parameters, with the circle axes of the
                chart.

        Returns:
            The space of the chart coordinates.

        Raises:
            ValueError: If the space has another number of parameters, or
                other circle axes, than the chart.
        """
        if len(space) != len(self.periods):
            raise ValueError(
                f"the space has {len(space)} parameters, but the chart was built"
                f" for {len(self.periods)}"
            )
        periods = tuple(parameter.period for parameter in space.parameters)
        lower_bounds = tuple(
            parameter.bounds[0] if parameter.period is not None else None
            for parameter in space.parameters
        )
        if (periods, lower_bounds) != (self.periods, self.lower_bounds):
            raise ValueError(
                "the space has other circle axes than the chart: periods"
                f" {periods}, but the chart was built for {self.periods}"
            )
        return ParamSpace(
            parameters=tuple(
                Parameter(name=parameter.name, label=parameter.label)
                for parameter in space.parameters
            )
        )

    def warn_wide_axes(
        self,
        samples: Float[ArrayLike, "sample param"],
        log_weights: Float[ArrayLike, " sample"],
    ) -> None:
        """Warn if a circle axis of the samples is too wide for the cut of the chart.

        For every circle axis, the weighted root mean square of the signed arcs
        of the samples from the centre is compared with P/(2·CHART_CUT_WIDTHS).
        One warning names every axis above the limit. The check reads concrete
        values, so call it outside ``jax.jit``; traced inputs are skipped.

        Args:
            samples: Samples shaped (N, D), in the coordinates of the chart's
                parameters.
            log_weights: Their log weights, shaped (N,).

        Warns:
            ValidityWarning: If a circle axis has a root mean square arc above
                P/(2·:data:`CHART_CUT_WIDTHS`).
        """
        samples_array = jnp.asarray(samples, dtype=jnp.float64)
        weights_array = jnp.asarray(log_weights, dtype=jnp.float64)
        if not _is_concrete(samples_array, weights_array):
            return
        weights = jax.nn.softmax(weights_array)
        arcs = self._arcs(samples_array)
        root_mean_squares = jnp.sqrt(jnp.sum(weights[:, None] * arcs**2, axis=0))
        wide = []
        for name, period, value in zip(
            self.names, self.periods, root_mean_squares.tolist(), strict=True
        ):
            if period is not None and value > period / (2 * CHART_CUT_WIDTHS):
                wide.append(
                    f"{name!r} (root mean square arc {value:.3g}, limit"
                    f" {period / (2 * CHART_CUT_WIDTHS):.3g} for the period"
                    f" {period:.3g})"
                )
        if wide:
            warnings.warn(
                ValidityWarning(
                    "the Fisher chart cuts each circle half a period from its"
                    f" centre, but the samples of {', '.join(wide)} are too wide for"
                    " it; samples beyond the cut land on the far side, where chart"
                    " distances are too large"
                ),
                skip_file_prefixes=(_PACKAGE_DIRECTORY,),
            )

    def _circle_arrays(self) -> tuple[Array, Array, Array]:
        """Return the lower bounds, periods and circle mask, with 0 and 1 elsewhere."""
        lower = jnp.asarray(
            [0.0 if bound is None else bound for bound in self.lower_bounds]
        )
        period = jnp.asarray(
            [1.0 if value is None else value for value in self.periods]
        )
        is_circle = jnp.asarray([value is not None for value in self.periods])
        return lower, period, is_circle

    def _arcs(
        self, points: Float[Array, "*leading param"]
    ) -> Float[Array, "*leading param"]:
        """Return θ − c, with the circle axes wrapped into [−P/2, P/2)."""
        offsets = points - self.centre
        _, period, is_circle = self._circle_arrays()
        arcs = jnp.mod(offsets + 0.5 * period, period) - 0.5 * period
        return jnp.where(is_circle, arcs, offsets)
