"""Toy signals of the validation zoo: generators with known Fisher matrices.

A data-space comparison needs a signal generator, a noise model and, for the
Fisher row, a Fisher matrix. The toy signals here supply all three in JAX and
float64, so that tests, the validation page and users can check
:func:`~mixed_signals.engines.dataspace.compare_signals` against a truth. Each
object meets the protocols of :mod:`mixed_signals.generators.protocol`.

:class:`WhiteNoise`
    Gaussian noise of standard deviation σ in every entry of the data vector. It
    whitens real signals to vectors divided by σ, so the noise-weighted inner
    product of two signals is their dot product over σ².
:class:`LinearSignal`
    h(θ) = Aθ + b, whose Fisher matrix is AᵀA/σ² at every θ.
    :func:`linear_signal` builds one whose Fisher matrix is exactly a given
    matrix, from the orthonormal basis of the discrete cosine transform.
:class:`SinusoidSignal`
    h(t; θ) = A sin(2π f t + φ) on a grid of times, with the parameters
    amplitude A, frequency f and phase φ. The single tone in white Gaussian
    noise is the classic estimation problem, whose Cramér-Rao bounds Rife and
    Boorstyn (1974) derive for a complex tone. Its Fisher matrix comes from
    automatic differentiation of the template, not from their formulas.
    :func:`sinusoid_signal` sets the signal-to-noise ratio exactly.
:class:`SignalCase`
    A generator, its noise model, its parameter space, its centre and its
    Fisher matrix at the centre, with :meth:`SignalCase.fisher_gaussians`:
    two Gaussians of covariance Γ⁻¹ a known Mahalanobis length apart. At linear
    order that length is the true σ_eff of the data-space comparison, in units
    of the noise standard deviation (Vallisneri 2008).

References:
    :cite:t:`rife1974single`; :cite:t:`vallisneri2008use`;
    :cite:t:`finn1992detection`.
"""

import dataclasses
import math
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.linalg import solve_triangular
from jax.typing import ArrayLike
from jaxtyping import Array, Float

from mixed_signals.core.space import Parameter, ParamSpace
from mixed_signals.generators.protocol import NoiseModel
from mixed_signals.io.arrays import to_jax
from mixed_signals.zoo._checks import (
    check_count,
    check_finite,
    check_positive,
    cholesky,
    vector,
)
from mixed_signals.zoo.cases import (
    ZooCase,
    _gaussian_case,
    _parameter_names,
    _real_space,
)

__all__ = [
    "LinearSignal",
    "SignalCase",
    "SinusoidSignal",
    "WhiteNoise",
    "linear_signal",
    "sinusoid_signal",
]

_SINUSOID_PARAMETERS = (
    Parameter(name="amplitude", label="A"),
    Parameter(name="frequency", label="f"),
    Parameter(name="phase", label=r"\phi"),
)


@dataclasses.dataclass(frozen=True, eq=False)
class WhiteNoise:
    """White Gaussian noise of standard deviation σ per entry of the data vector.

    ``whiten(signals)`` returns the real signals reshaped to (n, M) and divided
    by σ, so whiten(a)·whiten(b) = a·b/σ².

    Attributes:
        sigma: The standard deviation σ of the noise, positive.
        description: The name of the noise model in the ``space`` column of
            result rows.

    Raises:
        ValueError: If σ is not finite and positive.
    """

    sigma: float = 1.0
    description: str = "white"

    def __post_init__(self) -> None:
        """Check the standard deviation."""
        check_positive("sigma", self.sigma)

    def whiten(self, signals: object) -> Float[Array, "sample data"]:
        """Return the real signals shaped (n, ...) as vectors shaped (n, M) over σ.

        Args:
            signals: Real signals shaped (n, ...), as an array of any library
                that JAX reads.

        Returns:
            The whitened vectors as float64, shaped (n, M).

        Raises:
            ValueError: If the signals are complex or have no leading axis.
        """
        values = to_jax(signals)
        if jnp.issubdtype(values.dtype, jnp.complexfloating):
            raise ValueError("WhiteNoise whitens real signals, got complex signals")
        if values.ndim == 0:
            raise ValueError("signals must be shaped (n, ...), got a scalar")
        rows = jnp.asarray(values, dtype=jnp.float64).reshape(values.shape[0], -1)
        return rows / self.sigma


@dataclasses.dataclass(frozen=True, eq=False)
class LinearSignal:
    """The linear signal h(θ) = Aθ + b.

    Its Fisher matrix for a noise model is whiten(A)ᵀ whiten(A), the same at
    every θ, where the columns of A are whitened as signals.

    Attributes:
        matrix: The matrix A, shaped (M, D): the data entries by the
            parameters. It is stored as a float64 JAX array.
        offset: The offset b, shaped (M,), stored as a float64 JAX array.

    Raises:
        ValueError: If the shapes do not fit or a value is not finite.
    """

    # * Broad types: the checks below read shape and values, and give the messages.
    matrix: ArrayLike
    offset: ArrayLike

    def __post_init__(self) -> None:
        """Check the matrix and the offset and store them as float64 arrays."""
        matrix_array = np.asarray(self.matrix, dtype=np.float64)
        if matrix_array.ndim != 2 or 0 in matrix_array.shape:
            raise ValueError(
                f"matrix must be shaped (M, D), got shape {matrix_array.shape}"
            )
        if not np.all(np.isfinite(matrix_array)):
            raise ValueError("matrix must be finite")
        offset_array = vector("offset", self.offset, matrix_array.shape[0])
        object.__setattr__(self, "matrix", jnp.asarray(matrix_array))
        object.__setattr__(self, "offset", jnp.asarray(offset_array))

    def generate_template(self, params: object) -> Float[Array, "sample data"]:
        """Return the signals Aθ + b of parameters shaped (n, D), shaped (n, M)."""
        values = jnp.asarray(to_jax(params), dtype=jnp.float64)
        return values @ jnp.asarray(self.matrix).T + jnp.asarray(self.offset)

    def information_matrix(
        self, params: object, *, noise: NoiseModel
    ) -> Float[Array, "sample param param"]:
        """Return the Fisher matrix at each row of parameters, shaped (n, D, D).

        The D columns of A are whitened as D signals, and their Gram matrix is
        the same for every row.
        """
        count = to_jax(params).shape[0]
        whitened = jnp.asarray(
            noise.whiten(jnp.asarray(self.matrix).T), dtype=jnp.float64
        )
        gram = whitened @ whitened.T
        return jnp.broadcast_to(gram, (count, *gram.shape))


@dataclasses.dataclass(frozen=True, eq=False)
class SinusoidSignal:
    """The sinusoid h(t; θ) = A sin(2π f t + φ) on a grid of times.

    The parameters are the columns amplitude A, frequency f and phase φ.

    Attributes:
        times: The times t of the data entries, shaped (M,), stored as a
            float64 JAX array.

    Raises:
        ValueError: If the times are not a finite vector.
    """

    # * A broad type: the check below reads the shape and the values.
    times: ArrayLike

    def __post_init__(self) -> None:
        """Check the times and store them as a float64 array."""
        object.__setattr__(self, "times", jnp.asarray(vector("times", self.times)))

    def generate_template(self, params: object) -> Float[Array, "sample data"]:
        """Return the signals of parameters shaped (n, 3), shaped (n, M)."""
        values = jnp.asarray(to_jax(params), dtype=jnp.float64)
        return jax.vmap(partial(_sinusoid, jnp.asarray(self.times)))(values)

    def information_matrix(
        self, params: object, *, noise: NoiseModel
    ) -> Float[Array, "sample param param"]:
        """Return the Fisher matrix at each row of parameters, shaped (n, 3, 3).

        The template Jacobian of each row, shaped (M, 3), comes from
        ``jax.jacfwd``; its three columns are whitened as three signals and the
        Fisher matrix is their Gram matrix.
        """
        values = jnp.asarray(to_jax(params), dtype=jnp.float64)
        template = partial(_sinusoid, jnp.asarray(self.times))
        jacobians = jax.vmap(jax.jacfwd(template))(values)
        count, data_size, param_count = jacobians.shape
        columns = jnp.swapaxes(jacobians, 1, 2).reshape(count * param_count, data_size)
        whitened = jnp.asarray(noise.whiten(columns), dtype=jnp.float64)
        grouped = whitened.reshape(count, param_count, -1)
        return jnp.einsum("nik,njk->nij", grouped, grouped)


def _sinusoid(times: Float[Array, " data"], params: Float[Array, " param"]) -> Array:
    """Return A sin(2π f t + φ) at the times, for the parameters (A, f, φ)."""
    amplitude, frequency, phase = params
    template: Array = amplitude * jnp.sin(2 * jnp.pi * frequency * times + phase)
    return template


@dataclasses.dataclass(frozen=True, eq=False)
class SignalCase:
    """A toy signal with its noise, space, centre and Fisher matrix.

    Build cases with :func:`linear_signal` and :func:`sinusoid_signal`.

    Attributes:
        name: Name of the case, such as ``"linear_signal"``.
        generator: The signal generator, with ``generate_template`` and
            ``information_matrix``.
        noise: The noise model.
        space: The parameter space, with D real parameters.
        centre: The parameters at which the Fisher matrix holds, shaped (D,).
        fisher: The Fisher matrix Γ at the centre for the noise model, shaped
            (D, D).

    Raises:
        ValueError: If the centre or the Fisher matrix does not fit the space.
    """

    name: str
    generator: LinearSignal | SinusoidSignal
    noise: WhiteNoise
    space: ParamSpace
    # * The sizes are checked below against the space, which gives the messages.
    centre: Float[Array, " _"]
    fisher: Float[Array, "_ _"]

    def __post_init__(self) -> None:
        """Check that the centre and the Fisher matrix fit the space."""
        size = len(self.space)
        if self.centre.shape != (size,):
            raise ValueError(
                f"centre must be shaped ({size},), got shape {self.centre.shape}"
            )
        if self.fisher.shape != (size, size):
            raise ValueError(
                f"fisher must be shaped ({size}, {size}), got shape {self.fisher.shape}"
            )

    def fisher_gaussians(self, shift: float, *, direction: object = None) -> ZooCase:
        """Build two Gaussians of covariance Γ⁻¹ a known Mahalanobis length apart.

        The posterior ``reference`` is N(c, Γ⁻¹) at the centre c and
        ``shifted`` is N(c + Δθ, Γ⁻¹), with Δθ along ``direction`` and
        ΔθᵀΓΔθ = shift², built as
        :func:`~mixed_signals.zoo.cases.shifted_gaussians` builds its offset.
        At linear order ``shift`` is the true σ_eff of
        :func:`~mixed_signals.engines.dataspace.compare_signals`, in noise
        standard deviations, and the ``mahalanobis_shift`` row of the reference
        table holds it.

        Args:
            shift: Mahalanobis length of the shift.
            direction: Direction of the shift in parameter space, shaped (D,);
                the first axis when not given.

        Returns:
            The case ``"fisher_gaussians"`` with posteriors ``reference`` and
            ``shifted``, in the parameter names of this case.

        Raises:
            ValueError: If the shift is not finite, the direction does not fit
                or is zero, or the Fisher matrix is not symmetric positive
                definite.

        References:
            :cite:t:`vallisneri2008use`.
        """
        check_finite("shift", shift)
        size = len(self.space)
        covariance = jnp.linalg.inv(self.fisher)
        covariance = 0.5 * (covariance + covariance.T)
        if direction is None:
            direction_array = jnp.eye(size)[0]
        else:
            direction_array = jnp.asarray(vector("direction", direction, size))
        whitened = solve_triangular(
            cholesky(covariance, "the inverse of the Fisher matrix"),
            direction_array,
            lower=True,
        )
        length = float(jnp.linalg.norm(whitened))
        if length == 0.0:
            raise ValueError("direction must not be the zero vector")
        offset = shift * direction_array / length
        return _gaussian_case(
            "fisher_gaussians",
            ("reference", "shifted"),
            [self.centre, self.centre + offset],
            [covariance, covariance],
            param_names=self.space.names,
        )


def linear_signal(
    fisher: object,
    *,
    centre: object = None,
    data_size: int = 64,
    noise_sigma: float = 1.0,
) -> SignalCase:
    """Build a linear signal whose Fisher matrix in white noise is a given matrix.

    The matrix is σ·B·U, with B the first D columns of the orthonormal basis
    of the discrete cosine transform of size M, B[k, d] = √(2/M) cos(π (k + ½)
    d/M) and √(1/M) in the first column, and U the upper Cholesky factor of Γ,
    Γ = UᵀU. The whitened matrix is then B U, and its Gram matrix is exactly Γ.
    The offset is zero.

    Args:
        fisher: The Fisher matrix Γ, symmetric positive definite, shaped
            (D, D).
        centre: The parameters at which Γ holds, shaped (D,); zero when not
            given.
        data_size: The number M of data entries, at least D.
        noise_sigma: The standard deviation σ of the white noise.

    Returns:
        The case ``"linear_signal"``, with parameters ``x0``, ``x1``, … on the
        real line.

    Raises:
        ValueError: If Γ is not a symmetric positive definite matrix, M is
            below D, or a shape or value does not fit.
        TypeError: If ``data_size`` is not an integer.

    References:
        :cite:t:`vallisneri2008use`; :cite:t:`finn1992detection`.
    """
    fisher_array = np.asarray(fisher, dtype=np.float64)
    if fisher_array.ndim != 2 or fisher_array.shape[0] != fisher_array.shape[1]:
        raise ValueError(
            f"fisher must be a square matrix, got shape {fisher_array.shape}"
        )
    param_count = fisher_array.shape[0]
    check_count("data_size", data_size, 1)
    check_positive("noise_sigma", noise_sigma)
    if data_size < param_count:
        raise ValueError(
            f"data_size must be at least the {param_count} parameters, got {data_size}"
        )
    factor = cholesky(jnp.asarray(fisher_array), "fisher")
    centre_array = (
        jnp.zeros(param_count)
        if centre is None
        else jnp.asarray(vector("centre", centre, param_count))
    )
    basis = _cosine_basis(data_size, param_count)
    matrix = noise_sigma * basis @ factor.T
    return SignalCase(
        name="linear_signal",
        generator=LinearSignal(matrix, jnp.zeros(data_size)),
        noise=WhiteNoise(noise_sigma),
        space=_real_space(_parameter_names(param_count)),
        centre=centre_array,
        fisher=jnp.asarray(fisher_array),
    )


def sinusoid_signal(
    signal_to_noise: float,
    *,
    frequency: float = 40.0,
    phase: float = 1.0,
    data_size: int = 1_000,
    noise_sigma: float = 1.0,
) -> SignalCase:
    """Build the sinusoid A sin(2π f t + φ) in white noise at a given SNR.

    The times are ``jnp.linspace(-0.5, 0.5, M)``, one unit of observing time.
    The amplitude is A = SNR·σ/√(Σₖ sin²(2π f tₖ + φ)), so the signal-to-noise
    ratio of the template at the centre is exact. The parameters are
    ``amplitude``, ``frequency`` and ``phase``, all on real lines: the
    posteriors of the validation checks stay far narrower than the period of
    the phase, so no circle is needed. The Fisher matrix at the centre comes
    from :meth:`SinusoidSignal.information_matrix`. The single tone in white
    Gaussian noise is the estimation problem whose Cramér-Rao bounds Rife and
    Boorstyn (1974) derive for a complex tone.

    Args:
        signal_to_noise: The signal-to-noise ratio of the template at the
            centre, positive.
        frequency: The frequency f of the tone, in cycles per unit time.
        phase: The phase φ of the tone, in radians.
        data_size: The number M of samples of the time series, at least 3.
        noise_sigma: The standard deviation σ of the white noise.

    Returns:
        The case ``"sinusoid_signal"``.

    Raises:
        ValueError: If a value is not finite or positive where it must be, or
            the tone vanishes at every sample time.
        TypeError: If ``data_size`` is not an integer.

    References:
        :cite:t:`rife1974single`; :cite:t:`vallisneri2008use`.
    """
    check_positive("signal_to_noise", signal_to_noise)
    check_finite("frequency", frequency)
    check_finite("phase", phase)
    check_count("data_size", data_size, 3)
    check_positive("noise_sigma", noise_sigma)
    times = jnp.linspace(-0.5, 0.5, data_size)
    power = float(jnp.sum(jnp.sin(2 * jnp.pi * frequency * times + phase) ** 2))
    if not power > 0.0:
        raise ValueError("the tone vanishes at every sample time, so it has no SNR")
    amplitude = signal_to_noise * noise_sigma / math.sqrt(power)
    generator = SinusoidSignal(times)
    noise = WhiteNoise(noise_sigma)
    centre = jnp.asarray([amplitude, frequency, phase])
    fisher = generator.information_matrix(centre[None], noise=noise)[0]
    return SignalCase(
        name="sinusoid_signal",
        generator=generator,
        noise=noise,
        space=ParamSpace(parameters=_SINUSOID_PARAMETERS),
        centre=centre,
        fisher=fisher,
    )


def _cosine_basis(data_size: int, param_count: int) -> Float[Array, "data param"]:
    """Return the first D columns of the orthonormal DCT-II basis of size M."""
    rows = jnp.arange(data_size)[:, None]
    columns = jnp.arange(param_count)[None, :]
    basis = math.sqrt(2.0 / data_size) * jnp.cos(
        jnp.pi * (rows + 0.5) * columns / data_size
    )
    # * The first column is the constant vector, with norm one at 1/√M.
    return basis.at[:, 0].set(math.sqrt(1.0 / data_size))
