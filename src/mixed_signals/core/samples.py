"""``SampleSet``: flat samples shaped (B, K, N, D) with ``log_weights`` (B, K, N).

B counts independent problems, K posteriors of the same data, N samples per
posterior and D parameters, in the parameter order of the
:class:`~mixed_signals.core.space.ParamSpace` of the comparison. A set holds
arrays only. Posterior names and the parameter space go to the engines instead,
because metadata stored in a JAX pytree recompiles jitted kernels whenever it
changes.

Shapes
    One posterior (N, D), K posteriors (K, N, D) and batches (B, K, N, D)
    build the same container: missing leading axes get size one. Per-sample
    fields have the shape of the samples without the parameter axis, and
    ``n_eff`` also drops the sample axis, so for one posterior it is a scalar.
Weights
    ``log_weights`` are natural logarithms of unnormalised weights, all zero
    (equal weights) by default. A weight of zero, a log weight of −∞, removes
    a sample from every estimate. Methods normalise the weights over the
    samples of each problem and posterior where they need it.
Effective sample size
    ``n_eff`` sizes nulls and subsamples. Without it, the Kish effective
    sample size of the weights applies (Kish 1965), which assumes independent
    samples; set ``n_eff`` for correlated samples. A given ``n_eff`` must lie
    between 1 and the number of samples with positive weight, the range of
    the Kish value.
Weighted statistics
    :func:`weighted_covariance` and :func:`circular_standard_deviation` are
    the package's estimators of widths from weighted samples, shared by the
    transforms, metrics and diagnostics. :func:`circular_mean` and
    :func:`mean_resultant_length` give the direction and concentration of
    values on a circle, and :func:`unwrap_about` their representatives
    within half a period of a centre, such as the circular mean.
Precision
    Every field is stored in float64. Narrower float inputs are cast and raise
    a :class:`~mixed_signals.errors.PrecisionWarning`, because the cast cannot
    restore precision lost before it: neighbouring float32 values differ by
    6e-8 to 1.2e-7 of their magnitude (IEEE 2019).
Memory
    Inputs convert with :func:`~mixed_signals.io.arrays.to_jax`, so a float64
    torch or CuPy input shares memory with its field. Do not modify it in place
    afterwards.
Checks
    The constructor checks shapes and types, and for concrete inputs also
    their values: finite samples, log weights and log densities without NaN or
    +∞, at least one positive weight per posterior, and the range of
    ``n_eff``. Inside ``jax.jit`` the values are unknown, so only shapes and
    types are checked.

References:
    :cite:t:`kish1965survey`; :cite:t:`ieee2019ieee`;
    :cite:t:`harris2020array`; :cite:t:`chan1983algorithms`;
    :cite:t:`mardia2000directional`.
"""

import dataclasses
import os
import warnings
from collections.abc import Mapping, Sequence
from typing import Any, Final, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import logsumexp
from jax.typing import ArrayLike
from jaxtyping import Array, Bool, Float, Integer

from mixed_signals.core.topology import wrap
from mixed_signals.errors import PrecisionWarning
from mixed_signals.io.arrays import ArrayBackend, from_jax, to_jax

N_EFF_RELATIVE_TOLERANCE: Final = 1e-9
"""Relative tolerance of the ``n_eff`` bounds, letting a rounded Kish value pass."""

# * Index types stay broad: the methods check dtypes and shapes with their own messages.
type Selection = int | np.integer[Any] | Sequence[int] | np.ndarray | Array
"""Integer indices on one axis: one index, which keeps its axis with size one, or a
one-dimensional sequence or array."""

type SampleIndices = Sequence[int] | np.ndarray | Array
"""Integer positions on the sample axis, shaped (M,) for every posterior or (B, K, M)
for one row per problem and posterior."""

_SAMPLE_LAYOUTS: Final = "(N, D), (K, N, D) or (B, K, N, D)"
_PER_SAMPLE_FIELDS: Final = ("log_weights", "log_like", "log_prior", "log_q")
# * Frames in these directories are skipped, so a warning points at the caller's line.
_INTERNAL_DIRECTORIES: Final = (
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    os.path.dirname(os.path.abspath(eqx.__file__)),
)


class SampleShape(NamedTuple):
    """Axis sizes of a :class:`SampleSet`; compares equal to a plain tuple."""

    batch: int
    """Number of independent problems, B."""
    posterior: int
    """Number of posteriors of each problem, K."""
    sample: int
    """Number of samples of each posterior, N."""
    param: int
    """Number of parameters, D."""


def kish_effective_sample_size(
    log_weights: Float[Array, "*leading sample"],
) -> Float[Array, " *leading"]:
    """Return the Kish effective sample size of weighted samples.

    The size is (Σ wᵢ)² / Σ wᵢ² over the last axis, evaluated as
    exp(2 logsumexp(log w) − logsumexp(2 log w)), so that weights of any scale
    neither overflow nor underflow. It lies between 1, when one weight
    dominates, and the number of positive weights, when they are equal, and it
    assumes independent samples.

    Args:
        log_weights: Natural logarithms of unnormalised weights; −∞ for zero.

    Returns:
        The effective sample size of each row.

    References:
        :cite:t:`kish1965survey`.
    """
    return jnp.exp(
        2 * logsumexp(log_weights, axis=-1) - logsumexp(2 * log_weights, axis=-1)
    )


def weighted_covariance(
    samples: Float[Array, "*leading sample param"],
    log_weights: Float[Array, "*leading sample"],
) -> tuple[Float[Array, "*leading param"], Float[Array, "*leading param param"]]:
    """Return the weighted mean and the unbiased weighted covariance of samples.

    With the weights wᵢ normalised to sum to one, the mean is x̄ = Σ wᵢ xᵢ and
    the covariance Σ wᵢ (xᵢ − x̄)(xᵢ − x̄)ᵀ / (1 − Σ wᵢ²), the reliability-weight
    estimator of ``numpy.cov`` with ``aweights`` (Harris et al. 2020). It is
    unbiased for independent samples and reduces to the usual 1/(N − 1) form
    for equal weights. A single sample with positive weight gives NaN.

    The samples are first shifted by the sample of largest weight, which
    avoids cancellation (Chan, Golub and LeVeque 1983): a parameter that
    never changes gets a variance of exactly zero, and an offset far above
    the width, such as a frequency of 3 mHz known to 1e-11 Hz, cancels before
    any sum.

    Args:
        samples: Parameter values, shaped (..., N, D).
        log_weights: Natural logarithms of unnormalised weights, shaped
            (..., N); −∞ for zero.

    Returns:
        The mean, shaped (..., D), and the covariance, shaped (..., D, D).

    References:
        :cite:t:`harris2020array`; :cite:t:`chan1983algorithms`.
    """
    weights = jax.nn.softmax(log_weights, axis=-1)
    anchor = jnp.take_along_axis(
        samples, jnp.argmax(log_weights, axis=-1)[..., None, None], axis=-2
    )
    shifted = samples - anchor
    offset = jnp.einsum("...s,...sp->...p", weights, shifted)
    centred = shifted - offset[..., None, :]
    scatter = jnp.einsum("...s,...sp,...sq->...pq", weights, centred, centred)
    correction = 1 - jnp.sum(weights**2, axis=-1)
    return anchor[..., 0, :] + offset, scatter / correction[..., None, None]


def circular_standard_deviation(
    samples: Float[Array, "*leading sample"],
    log_weights: Float[Array, "*leading sample"],
    period: float,
) -> Float[Array, " *leading"]:
    """Return the weighted circular standard deviation of values on a circle.

    The values x map to angles 2πx/P, and R̄ is the length of the weighted
    mean of the unit vectors exp(2πi xᵢ/P), for weights summing to one. The
    circular standard deviation √(−2 ln R̄), converted back to the units of x by
    the factor P/(2π), equals σ for a normal of width σ wrapped onto the circle
    (Mardia and Jupp 2000). It is zero when all values coincide and grows
    without bound as R̄ falls to zero.

    The angles are measured from the value of largest weight, which leaves R̄
    unchanged, and 1 − R̄² follows from half-angle sines, so narrow circles keep
    their width and coinciding values give exactly zero, as in the shifted
    computation of :func:`weighted_covariance`.

    Args:
        samples: Values on the circle, shaped (..., N).
        log_weights: Natural logarithms of unnormalised weights, shaped
            (..., N); −∞ for zero.
        period: Period P of the circle.

    Returns:
        The circular standard deviation in the units of the values, shaped
        (...).

    References:
        :cite:t:`mardia2000directional`.
    """
    deficit = _resultant_deficit(samples, log_weights, period)
    return jnp.sqrt(-jnp.log1p(-deficit)) * period / (2 * jnp.pi)


def mean_resultant_length(
    samples: Float[Array, "*leading sample"],
    log_weights: Float[Array, "*leading sample"],
    period: float,
) -> Float[Array, " *leading"]:
    """Return the weighted mean resultant length R̄ of values on a circle.

    R̄ is the length of the weighted mean of the unit vectors exp(2πi xᵢ/P),
    for weights summing to one (Mardia and Jupp 2000). It is 1 when all
    values coincide, and near 0 when they spread evenly around the circle or
    form two opposite clusters of equal weight, when the mean direction is
    undefined. It shares the half-angle sums of
    :func:`circular_standard_deviation`.

    Args:
        samples: Values on the circle, shaped (..., N).
        log_weights: Natural logarithms of unnormalised weights, shaped
            (..., N); −∞ for zero.
        period: Period P of the circle.

    Returns:
        R̄ in [0, 1], shaped (...).

    References:
        :cite:t:`mardia2000directional`.
    """
    return jnp.sqrt(1 - _resultant_deficit(samples, log_weights, period))


def circular_mean(
    samples: Float[Array, "*leading sample"],
    log_weights: Float[Array, "*leading sample"],
    period: float,
    lower: float,
) -> Float[Array, " *leading"]:
    """Return the weighted circular mean of values on a circle.

    The mean direction is the argument of the weighted mean of the unit
    vectors exp(2πi xᵢ/P) (Mardia and Jupp 2000), converted back to the units
    of the values. The angles are measured from the value of largest weight,
    as in :func:`circular_standard_deviation`, so a narrow circle keeps its
    precision. Where R̄ is near zero, the direction is undefined and the
    result arbitrary; the Rayleigh test tells whether it is resolved (Mardia
    and Jupp 2000).

    Args:
        samples: Values on the circle, shaped (..., N).
        log_weights: Natural logarithms of unnormalised weights, shaped
            (..., N); −∞ for zero.
        period: Period P of the circle.
        lower: Lower end of the circle.

    Returns:
        The circular mean in [lower, lower + P), shaped (...).

    References:
        :cite:t:`mardia2000directional`.
    """
    weights = jax.nn.softmax(log_weights, axis=-1)
    anchor = jnp.take_along_axis(
        samples, jnp.argmax(log_weights, axis=-1)[..., None], axis=-1
    )
    angles = 2 * jnp.pi * (samples - anchor) / period
    offset = jnp.arctan2(
        jnp.sum(weights * jnp.sin(angles), axis=-1),
        jnp.sum(weights * jnp.cos(angles), axis=-1),
    )
    return wrap(anchor[..., 0] + offset * period / (2 * jnp.pi), lower, period)


def unwrap_about(
    values: Float[Array, " *leading"], centre: ArrayLike, period: float
) -> Float[Array, " *leading"]:
    """Return the representative of each circle value within half a period of a centre.

    Values that differ by a whole number of periods are the same point, so
    each value maps to its representative in [centre − P/2, centre + P/2).
    About the circular mean, the representatives of a narrow distribution
    form one unbroken interval, on which linear statistics apply.

    Args:
        values: Values on the circle.
        centre: Centre of the representatives, broadcast against ``values``.
        period: Period P of the circle.

    Returns:
        The representatives, shaped like ``values``.

    References:
        :cite:t:`mardia2000directional`.
    """
    return wrap(values, jnp.asarray(centre) - period / 2, period)


def _resultant_deficit(
    samples: Float[Array, "*leading sample"],
    log_weights: Float[Array, "*leading sample"],
    period: float,
) -> Float[Array, " *leading"]:
    """Return 1 − R̄², computed from half-angle sines so that narrow circles keep it."""
    weights = jax.nn.softmax(log_weights, axis=-1)
    anchor = jnp.take_along_axis(
        samples, jnp.argmax(log_weights, axis=-1)[..., None], axis=-1
    )
    half_angles = jnp.pi * (samples - anchor) / period
    total = jnp.sum(weights, axis=-1)
    # * With versine v = Σ w (1 − cos Δφ) and sine s = Σ w sin Δφ, R̄² = (1 − v)² + s².
    versine = jnp.sum(weights * 2 * jnp.sin(half_angles) ** 2, axis=-1) / total
    sine = jnp.sum(weights * jnp.sin(2 * half_angles), axis=-1) / total
    return jnp.clip(2 * versine - versine**2 - sine**2, 0.0, 1.0)


class SampleSet(eqx.Module):
    """Weighted samples of B problems with K posteriors each, shaped (B, K, N, D).

    Every input may be a JAX, NumPy, CuPy or PyTorch array, or a nested
    sequence.

    Args:
        samples: Parameter values shaped (N, D), (K, N, D) or (B, K, N, D).
        log_weights: Natural logarithm of the unnormalised weight of each
            sample, shaped like ``samples`` without the parameter axis; equal
            weights when not given.
        n_eff: Effective sample size of each posterior, shaped like ``samples``
            without the sample and parameter axes; the Kish value of the
            weights when not given.
        log_like: Log-likelihood of each sample, shaped like ``log_weights``.
        log_prior: Log-prior density of each sample, shaped like
            ``log_weights``.
        log_q: Log-density of each sample under an approximation q of the
            posterior, such as a normalizing flow, shaped like ``log_weights``.

    Attributes:
        samples: Parameter values, shaped (B, K, N, D).
        log_weights: Log weights, shaped (B, K, N).
        n_eff: Effective sample sizes, shaped (B, K), or ``None``.
        log_like: Log-likelihoods, shaped (B, K, N), or ``None``.
        log_prior: Log-prior densities, shaped (B, K, N), or ``None``.
        log_q: Log-densities under the approximation, shaped (B, K, N), or
            ``None``.

    Raises:
        ValueError: If the shapes do not fit together, or concrete inputs have
            invalid values.
        TypeError: If an input is not real-valued.

    Example:
        >>> import numpy as np
        >>> draws = np.random.default_rng(1).normal(size=(2, 500, 3))
        >>> SampleSet(draws).shape
        SampleShape(batch=1, posterior=2, sample=500, param=3)

    References:
        :cite:t:`kish1965survey`.
    """

    samples: Float[Array, "batch posterior sample param"]
    log_weights: Float[Array, "batch posterior sample"]
    n_eff: Float[Array, "batch posterior"] | None
    log_like: Float[Array, "batch posterior sample"] | None
    log_prior: Float[Array, "batch posterior sample"] | None
    log_q: Float[Array, "batch posterior sample"] | None

    def __init__(
        self,
        samples: object,
        *,
        log_weights: object = None,
        n_eff: object = None,
        log_like: object = None,
        log_prior: object = None,
        log_q: object = None,
    ) -> None:
        """Check the inputs, add missing leading axes and cast them to float64."""
        inputs = {
            "samples": samples,
            "log_weights": log_weights,
            "n_eff": n_eff,
            "log_like": log_like,
            "log_prior": log_prior,
            "log_q": log_q,
        }
        given = {
            name: to_jax(value) for name, value in inputs.items() if value is not None
        }
        input_shape = given["samples"].shape
        if len(input_shape) not in (2, 3, 4):
            raise ValueError(
                f"samples must be shaped {_SAMPLE_LAYOUTS}, got shape {input_shape}"
            )
        if 0 in input_shape:
            raise ValueError(
                f"samples need entries on every axis, got shape {input_shape}"
            )
        expected_shapes = {
            "samples": input_shape,
            "n_eff": input_shape[:-2],
        } | dict.fromkeys(_PER_SAMPLE_FIELDS, input_shape[:-1])
        for name, array in given.items():
            _check_real(name, array)
            if array.shape != expected_shapes[name]:
                raise ValueError(
                    f"{name} has shape {array.shape}; samples of shape {input_shape}"
                    f" need {expected_shapes[name]}"
                )
        _warn_narrow_floats(given)
        leading_axes = (1,) * (4 - len(input_shape))
        fields = {
            name: jnp.asarray(array, dtype=jnp.float64).reshape(
                leading_axes + array.shape
            )
            for name, array in given.items()
        }
        if "log_weights" not in fields:
            fields["log_weights"] = jnp.zeros(fields["samples"].shape[:-1])
        if _is_concrete(*fields.values()):
            _check_values(fields)
        self.samples = fields["samples"]
        self.log_weights = fields["log_weights"]
        self.n_eff = fields.get("n_eff")
        self.log_like = fields.get("log_like")
        self.log_prior = fields.get("log_prior")
        self.log_q = fields.get("log_q")

    @property
    def shape(self) -> SampleShape:
        """Axis sizes (B, K, N, D)."""
        return SampleShape(*self.samples.shape)

    def effective_sample_size(self) -> Float[Array, "batch posterior"]:
        """Return ``n_eff`` if set, otherwise the Kish value of the weights.

        References:
            :cite:t:`kish1965survey`.
        """
        if self.n_eff is not None:
            return self.n_eff
        return kish_effective_sample_size(self.log_weights)

    def normalized_log_weights(self) -> Float[Array, "batch posterior sample"]:
        """Return log weights whose weights sum to one over each posterior."""
        return jax.nn.log_softmax(self.log_weights, axis=-1)

    def normalized_weights(self) -> Float[Array, "batch posterior sample"]:
        """Return weights that sum to one over each posterior."""
        return jax.nn.softmax(self.log_weights, axis=-1)

    def select(
        self,
        *,
        problems: Selection | None = None,
        posteriors: Selection | None = None,
        params: Selection | None = None,
    ) -> "SampleSet":
        """Return the set restricted to some problems, posteriors or parameters.

        Every field follows the samples. A single index keeps its axis with
        size one, and indices may reorder or repeat entries. The log densities
        stay those of the full parameter vector: as values per sample, they
        remain valid for a subset of the parameters, whose distribution is a
        marginal of the full posterior.

        Args:
            problems: Indices on the batch axis; ``None`` keeps every problem.
            posteriors: Indices on the posterior axis; ``None`` keeps all.
            params: Indices on the parameter axis; ``None`` keeps all.

        Returns:
            The selected set.

        Raises:
            IndexError: If an index lies outside its axis.
            ValueError: If a selection is empty or not one-dimensional.
            TypeError: If the indices are not integers.
        """
        fields = self._present_fields()
        batch_size, posterior_count, _, param_count = self.shape
        for axis, selection, size, axis_name in (
            (0, problems, batch_size, "problem"),
            (1, posteriors, posterior_count, "posterior"),
        ):
            if selection is not None:
                index = _axis_index(selection, size, axis_name)
                fields = {
                    name: jnp.take(array, index, axis=axis)
                    for name, array in fields.items()
                }
        if params is not None:
            index = _axis_index(params, param_count, "parameter")
            fields["samples"] = jnp.take(fields["samples"], index, axis=3)
        return SampleSet(**fields)

    def take_samples(self, indices: SampleIndices) -> "SampleSet":
        """Return the samples at the given positions on the sample axis.

        Every per-sample field follows the samples. ``n_eff`` is dropped,
        because it described the full set, so the Kish value of the remaining
        weights applies; set it again for correlated samples. Indices may
        repeat, but the Kish value then counts repeated samples as independent.

        Args:
            indices: Positions shaped (M,), shared by every problem and
                posterior, or (B, K, M), one row per problem and posterior.

        Returns:
            The set with M samples per posterior.

        Raises:
            IndexError: If a position lies outside the sample axis.
            ValueError: If the indices are empty or have the wrong shape, or a
                posterior keeps no sample with positive weight.
            TypeError: If the indices are not integers.
        """
        batch_size, posterior_count, sample_count, _ = self.shape
        index = _sample_index(indices, batch_size, posterior_count, sample_count)
        fields = self._present_fields()
        fields.pop("n_eff", None)
        taken = {
            name: jnp.take_along_axis(
                array, index[..., None] if name == "samples" else index, axis=2
            )
            for name, array in fields.items()
        }
        return SampleSet(**taken)

    def to_arrays(
        self, backend: ArrayBackend = "numpy", *, copy: bool = False
    ) -> dict[str, Any]:
        """Return every field that is set, as arrays of the given library.

        The keys are the field names, so ``SampleSet(**sample_set.to_arrays())``
        rebuilds the set.

        Args:
            backend: ``"numpy"``, ``"jax"``, ``"torch"`` or ``"cupy"``.
            copy: If true, return independent copies; otherwise the arrays may
                share memory with the fields and must not be modified in place.

        Returns:
            The arrays, keyed by field name, in field order.
        """
        return {
            name: from_jax(array, backend, copy=copy)
            for name, array in self._present_fields().items()
        }

    def _present_fields(self) -> dict[str, Array]:
        values = {
            field.name: getattr(self, field.name) for field in dataclasses.fields(self)
        }
        return {name: value for name, value in values.items() if value is not None}


def _check_real(name: str, array: Array) -> None:
    """Raise TypeError unless the array holds booleans, integers or real floats."""
    dtype = array.dtype
    if not (
        jnp.issubdtype(dtype, jnp.floating)
        or jnp.issubdtype(dtype, jnp.integer)
        or jnp.issubdtype(dtype, jnp.bool_)
    ):
        raise TypeError(f"{name} must hold real numbers, got dtype {dtype}")


def _warn_narrow_floats(arrays: Mapping[str, Array]) -> None:
    """Warn once about every float input narrower than float64."""
    narrow = [
        f"{name} ({array.dtype})"
        for name, array in arrays.items()
        if jnp.issubdtype(array.dtype, jnp.floating) and array.dtype.itemsize < 8
    ]
    if narrow:
        warnings.warn(
            PrecisionWarning(
                f"cast {', '.join(narrow)} to float64; the cast cannot restore"
                " precision lost before it: neighbouring float32 values differ by"
                " up to 1.2e-7 of their magnitude, and narrower types by more"
            ),
            skip_file_prefixes=_INTERNAL_DIRECTORIES,
        )


def _is_concrete(*arrays: Array) -> bool:
    """Return whether every array holds known values, outside a JAX trace."""
    return not any(isinstance(array, jax.core.Tracer) for array in arrays)


def _first_true(mask: Bool[Array, "..."]) -> tuple[int, ...] | None:
    """Return the index of the first true entry in C order, or ``None``."""
    if not bool(jnp.any(mask)):
        return None
    # * One axis at a time, so only scalars leave the device, whatever the mask size.
    position = []
    remaining = mask
    for _ in range(mask.ndim):
        rows_with_true = jnp.any(remaining.reshape(remaining.shape[0], -1), axis=1)
        row = int(jnp.argmax(rows_with_true))
        position.append(row)
        remaining = remaining[row]
    return tuple(position)


def _check_values(fields: Mapping[str, Array]) -> None:
    """Check the values of concrete fields in the (B, K, N, D) layout."""
    samples = fields["samples"]
    if (failure := _first_true(~jnp.isfinite(samples))) is not None:
        raise ValueError(
            f"samples of problem {failure[0]}, posterior {failure[1]} contain NaN"
            " or infinite values"
        )
    for name in _PER_SAMPLE_FIELDS:
        if name not in fields:
            continue
        values = fields[name]
        if (
            failure := _first_true(jnp.isnan(values) | (values == jnp.inf))
        ) is not None:
            raise ValueError(
                f"{name} of problem {failure[0]}, posterior {failure[1]} contain NaN"
                " or +inf"
            )
    positive_counts = jnp.sum(fields["log_weights"] > -jnp.inf, axis=-1)
    if (failure := _first_true(positive_counts == 0)) is not None:
        raise ValueError(
            f"posterior {failure[1]} of problem {failure[0]} has no sample with"
            " positive weight; at least one log weight must exceed -inf"
        )
    if "n_eff" not in fields:
        return
    n_eff = fields["n_eff"]
    outside = (
        ~jnp.isfinite(n_eff)
        | (n_eff < 1 - N_EFF_RELATIVE_TOLERANCE)
        | (n_eff > positive_counts * (1 + N_EFF_RELATIVE_TOLERANCE))
    )
    if (failure := _first_true(outside)) is not None:
        raise ValueError(
            f"n_eff of problem {failure[0]}, posterior {failure[1]} is"
            f" {float(n_eff[failure])!r}; it must lie between 1 and"
            f" {int(positive_counts[failure])}, the number of samples with positive"
            " weight"
        )


def _host_indices(indices: object, axis_name: str) -> np.ndarray:
    """Return concrete indices as a NumPy integer array, rejecting other types."""
    if isinstance(indices, bool | np.bool_):
        raise TypeError(f"{axis_name} indices must be integers, got a bool")
    host_indices = np.asarray(indices)
    if host_indices.size == 0:
        raise ValueError(f"select at least one {axis_name}")
    if not np.issubdtype(host_indices.dtype, np.integer):
        raise TypeError(
            f"{axis_name} indices must be integers, got dtype {host_indices.dtype}"
        )
    return host_indices


def _in_range(host_indices: np.ndarray, size: int, axis_name: str) -> np.ndarray:
    """Check that indices lie in [−size, size) and turn negative ones positive."""
    outside = (host_indices < -size) | (host_indices >= size)
    if outside.any():
        raise IndexError(
            f"{axis_name} index {int(host_indices[outside][0])} is out of range for"
            f" an axis of size {size}"
        )
    return np.where(host_indices < 0, host_indices + size, host_indices)


def _axis_index(selection: Selection, size: int, axis_name: str) -> Array:
    """Turn a selection on one axis into a checked one-dimensional index array."""
    if isinstance(selection, jax.core.Tracer):
        return jnp.where(selection < 0, selection + size, selection)
    if isinstance(selection, int | np.integer) and not isinstance(selection, bool):
        selection = [int(selection)]
    host_indices = _host_indices(selection, axis_name)
    if host_indices.ndim != 1:
        raise ValueError(
            f"{axis_name} indices must be one-dimensional, got shape"
            f" {host_indices.shape}"
        )
    return jnp.asarray(_in_range(host_indices, size, axis_name))


def _sample_index(
    indices: SampleIndices, batch_size: int, posterior_count: int, sample_count: int
) -> Integer[Array, "batch posterior selected"]:
    """Turn sample positions into a checked index array shaped (B, K, M)."""
    if isinstance(indices, jax.core.Tracer):
        index = jnp.where(indices < 0, indices + sample_count, indices)
    else:
        host_indices = _host_indices(indices, "sample")
        if host_indices.ndim == 3 and host_indices.shape[:2] != (
            batch_size,
            posterior_count,
        ):
            raise ValueError(
                f"sample indices of shape {host_indices.shape} need the leading"
                f" axes ({batch_size}, {posterior_count}) of the set"
            )
        if host_indices.ndim not in (1, 3):
            raise ValueError(
                "sample indices must be shaped (M,) or (B, K, M), got shape"
                f" {host_indices.shape}"
            )
        index = jnp.asarray(_in_range(host_indices, sample_count, "sample"))
    return jnp.broadcast_to(index, (batch_size, posterior_count, index.shape[-1]))
