"""``Chains``: MCMC chains in eryn's axis layout.

A chain array is shaped (step, temp, walker, leaf, param), the layout of one
eryn branch (Karnesis et al. 2023): stored steps, the temperatures of a
parallel-tempering ladder, the walkers of an ensemble, leaves and parameters.
The inverse temperatures β of each step are shaped (step, temp). The cold
chain, at β = 1, samples the posterior; a chain at β < 1 samples the prior
times the likelihood to the power β (Vousden, Farr and Mandel 2016).

Layouts
    The constructor takes the eryn layout, or (step, walker, param) for a
    sampler with one temperature, such as the chains of emcee
    (Foreman-Mackey et al. 2013). It adds the temperature and leaf axes to
    the second layout, with size one and β = 1. Per-walker fields have the
    shape of the samples without the leaf and parameter axes; counts per
    walker also drop the step axis.
Scope of v1
    One branch, one leaf and one independent sampler. Runs with several
    leaves are rejected, because leaves are not batch entries; multi-leaf and
    trans-dimensional runs, and eryn runs with several independent samplers,
    are deferred. eryn stores inactive leaves as NaN, so they fail the check
    of finite samples.
Log densities
    ``log_like`` and ``log_prior`` may be missing, as for samplers that store
    only the log posterior. The autocorrelation time, the effective sample
    size and the split-chain tests need the samples only.
Acceptance and swaps
    ``accepted`` counts the accepted in-model proposals of each walker at
    each temperature over the whole run, and ``proposed`` the proposals
    behind them, so accepted / proposed is the acceptance rate also when the
    stored steps were cut or thinned. ``swaps_accepted`` and
    ``swaps_proposed`` count the swaps between temperatures t and t + 1 in
    the same way.
Checks
    As for :class:`~mixed_signals.core.samples.SampleSet`: shapes and types
    always, values when the inputs are concrete arrays.

References:
    :cite:t:`karnesis2023eryn`; :cite:t:`vousden2016dynamic`;
    :cite:t:`foremanmackey2013emcee`.
"""

from collections.abc import Mapping
from typing import Final, NamedTuple

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, Int

from mixed_signals.core.samples import (
    _check_real,
    _first_true,
    _is_concrete,
    _warn_narrow_floats,
)
from mixed_signals.io.arrays import to_jax

_CHAIN_LAYOUTS: Final = (
    "(step, temp, walker, leaf, param), the eryn layout, or (step, walker, param)"
)
_FLOAT_FIELDS: Final = ("samples", "betas", "log_like", "log_prior")
_COUNT_PAIRS: Final = (("accepted", "proposed"), ("swaps_accepted", "swaps_proposed"))


class ChainShape(NamedTuple):
    """Axis sizes of :class:`Chains`; compares equal to a plain tuple."""

    step: int
    """Number of stored steps."""
    temp: int
    """Number of temperatures."""
    walker: int
    """Number of walkers per temperature."""
    leaf: int
    """Number of leaves, one in v1."""
    param: int
    """Number of parameters."""


class Chains(eqx.Module):
    """MCMC chains of one branch in eryn's axis layout.

    Every input may be a JAX, NumPy, CuPy or PyTorch array, or a nested
    sequence.

    Args:
        samples: Parameter values shaped (step, temp, walker, leaf, param), or
            (step, walker, param) for one temperature.
        betas: Inverse temperatures shaped (step, temp); all ones when not
            given, which needs a single temperature.
        log_like: Log-likelihood of each walker at each step, shaped like
            ``samples`` without the leaf and parameter axes.
        log_prior: Log-prior density of each walker at each step, shaped like
            ``log_like``.
        accepted: Accepted in-model proposals of each walker, shaped (temp,
            walker), or (walker,) for one temperature.
        proposed: Proposals behind ``accepted``, shaped like it.
        swaps_accepted: Accepted swaps between neighbouring temperatures,
            shaped (temp − 1,).
        swaps_proposed: Proposed swaps behind ``swaps_accepted``, shaped like
            it.

    Attributes:
        samples: Parameter values, shaped (step, temp, walker, 1, param).
        betas: Inverse temperatures, shaped (step, temp).
        log_like: Log-likelihoods, shaped (step, temp, walker), or ``None``.
        log_prior: Log-prior densities, shaped (step, temp, walker), or
            ``None``.
        accepted: Accepted in-model proposals, shaped (temp, walker), or
            ``None``.
        proposed: In-model proposals, shaped (temp, walker), or ``None``.
        swaps_accepted: Accepted swaps, shaped (temp − 1,), or ``None``.
        swaps_proposed: Proposed swaps, shaped (temp − 1,), or ``None``.

    Raises:
        ValueError: If the shapes do not fit together, or concrete inputs have
            invalid values.
        TypeError: If an input is not real-valued, or a count is a bool.
        NotImplementedError: If the chains hold several leaves.

    References:
        :cite:t:`karnesis2023eryn`; :cite:t:`vousden2016dynamic`;
        :cite:t:`foremanmackey2013emcee`.
    """

    samples: Float[Array, "step temp walker leaf param"]
    betas: Float[Array, "step temp"]
    log_like: Float[Array, "step temp walker"] | None
    log_prior: Float[Array, "step temp walker"] | None
    accepted: Int[Array, "temp walker"] | None
    proposed: Int[Array, "temp walker"] | None
    swaps_accepted: Int[Array, " temp-1"] | None
    swaps_proposed: Int[Array, " temp-1"] | None

    def __init__(
        self,
        samples: object,
        *,
        betas: object = None,
        log_like: object = None,
        log_prior: object = None,
        accepted: object = None,
        proposed: object = None,
        swaps_accepted: object = None,
        swaps_proposed: object = None,
    ) -> None:
        """Check the inputs, bring them to the eryn layout and cast them."""
        inputs = {
            "samples": samples,
            "betas": betas,
            "log_like": log_like,
            "log_prior": log_prior,
            "accepted": accepted,
            "proposed": proposed,
            "swaps_accepted": swaps_accepted,
            "swaps_proposed": swaps_proposed,
        }
        given = {
            name: to_jax(value) for name, value in inputs.items() if value is not None
        }
        input_shape = given["samples"].shape
        if len(input_shape) == 3:
            step_count, walker_count, param_count = input_shape
            temp_count = leaf_count = 1
            for name in ("betas", "swaps_accepted", "swaps_proposed"):
                if name in given:
                    raise ValueError(
                        f"{name} needs samples in the eryn layout (step, temp, walker,"
                        " leaf, param); samples shaped (step, walker, param) have one"
                        " temperature"
                    )
        elif len(input_shape) == 5:
            step_count, temp_count, walker_count, leaf_count, param_count = input_shape
        else:
            raise ValueError(
                f"samples must be shaped {_CHAIN_LAYOUTS}, got shape {input_shape}"
            )
        if 0 in input_shape:
            raise ValueError(
                f"samples need entries on every axis, got shape {input_shape}"
            )
        if leaf_count > 1:
            raise NotImplementedError(
                f"chains with {leaf_count} leaves are not supported yet: v1 reads one"
                " leaf per branch, because leaves are not batch entries"
            )
        per_walker = input_shape[:-1] if len(input_shape) == 3 else input_shape[:-2]
        per_count = per_walker[1:]
        expected_shapes = {
            "samples": input_shape,
            "betas": (step_count, temp_count),
            "log_like": per_walker,
            "log_prior": per_walker,
            "accepted": per_count,
            "proposed": per_count,
            "swaps_accepted": (temp_count - 1,),
            "swaps_proposed": (temp_count - 1,),
        }
        for name, array in given.items():
            if name in _FLOAT_FIELDS:
                _check_real(name, array)
            else:
                _check_count_type(name, array)
            if array.shape != expected_shapes[name]:
                raise ValueError(
                    f"{name} has shape {array.shape}; samples of shape {input_shape}"
                    f" need {expected_shapes[name]}"
                )
        for count_name, total_name in _COUNT_PAIRS:
            if (count_name in given) != (total_name in given):
                raise ValueError(
                    f"{count_name} and {total_name} come together: give both or neither"
                )
        if "betas" not in given and temp_count > 1:
            raise ValueError(
                f"betas are needed for chains with {temp_count} temperatures"
            )
        _warn_narrow_floats(
            {name: array for name, array in given.items() if name in _FLOAT_FIELDS}
        )
        target_shapes = {
            "samples": (step_count, temp_count, walker_count, 1, param_count),
            "betas": (step_count, temp_count),
            "log_like": (step_count, temp_count, walker_count),
            "log_prior": (step_count, temp_count, walker_count),
            "accepted": (temp_count, walker_count),
            "proposed": (temp_count, walker_count),
            "swaps_accepted": (temp_count - 1,),
            "swaps_proposed": (temp_count - 1,),
        }
        concrete = _is_concrete(*given.values())
        fields: dict[str, Array] = {}
        for name, array in given.items():
            if name in _FLOAT_FIELDS:
                fields[name] = jnp.asarray(array, dtype=jnp.float64)
            else:
                if concrete:
                    _check_whole_numbers(name, array)
                fields[name] = jnp.asarray(array, dtype=jnp.int64)
            fields[name] = fields[name].reshape(target_shapes[name])
        if "betas" not in fields:
            fields["betas"] = jnp.ones((step_count, temp_count))
        if concrete:
            _check_values(fields)
        self.samples = fields["samples"]
        self.betas = fields["betas"]
        self.log_like = fields.get("log_like")
        self.log_prior = fields.get("log_prior")
        self.accepted = fields.get("accepted")
        self.proposed = fields.get("proposed")
        self.swaps_accepted = fields.get("swaps_accepted")
        self.swaps_proposed = fields.get("swaps_proposed")

    @property
    def shape(self) -> ChainShape:
        """Axis sizes (step, temp, walker, leaf, param)."""
        return ChainShape(*self.samples.shape)

    def cold_index(self) -> int:
        """Return the index of the cold chain, the first temperature at β = 1.

        The inverse temperature must equal 1 exactly at every step, as for
        eryn's cold chain. Call it outside ``jax.jit``, because it reads the
        values of ``betas``.

        Returns:
            The index on the temperature axis.

        Raises:
            ValueError: If no temperature stays at β = 1.

        References:
            :cite:t:`vousden2016dynamic`.
        """
        at_one = np.asarray(jnp.all(self.betas == 1.0, axis=0))
        if not at_one.any():
            raise ValueError(
                "no temperature stays at beta = 1 in every step, so the chains"
                " have no cold chain"
            )
        return int(np.argmax(at_one))

    def cold(self) -> "Chains":
        """Return the cold chain alone, with a temperature axis of size one.

        Acceptance counts keep the row of the cold chain. Swap counts are
        dropped, because one temperature has no neighbour; read swap rates
        from the full chains.

        Returns:
            The chains at the temperature of :meth:`cold_index`.

        Raises:
            ValueError: If no temperature stays at β = 1.
        """
        cold_index = self.cold_index()
        cold_slice = slice(cold_index, cold_index + 1)
        per_step = ("samples", "betas", "log_like", "log_prior")
        per_temperature = ("accepted", "proposed")
        fields: dict[str, Array] = {}
        for name in per_step + per_temperature:
            value = getattr(self, name)
            if value is not None:
                fields[name] = (
                    value[:, cold_slice] if name in per_step else value[cold_slice]
                )
        return Chains(**fields)


def _check_count_type(name: str, array: Array) -> None:
    """Raise TypeError unless a count array holds integers or floats."""
    dtype = array.dtype
    if not (jnp.issubdtype(dtype, jnp.integer) or jnp.issubdtype(dtype, jnp.floating)):
        raise TypeError(f"{name} must hold whole numbers, got dtype {dtype}")


def _check_whole_numbers(name: str, array: Array) -> None:
    """Check that a concrete count array holds non-negative whole numbers."""
    values = jnp.asarray(array, dtype=jnp.float64)
    invalid = ~jnp.isfinite(values) | (values != jnp.round(values)) | (values < 0)
    if (failure := _first_true(invalid)) is not None:
        raise ValueError(
            f"{name} must hold non-negative whole numbers, got"
            f" {float(values[failure])!r} at index {failure}"
        )


def _check_values(fields: Mapping[str, Array]) -> None:
    """Check the values of concrete fields in the eryn layout."""
    if (failure := _first_true(~jnp.isfinite(fields["samples"]))) is not None:
        raise ValueError(
            f"samples contain NaN or infinite values at step {failure[0]},"
            f" temperature {failure[1]}, walker {failure[2]}; eryn stores inactive"
            " leaves as NaN, and v1 reads fixed-dimensional runs only"
        )
    betas = fields["betas"]
    if (failure := _first_true(~((betas >= 0) & (betas <= 1)))) is not None:
        raise ValueError(
            f"betas must lie in [0, 1], got {float(betas[failure])!r} at step"
            f" {failure[0]}, temperature {failure[1]}"
        )
    for name in ("log_like", "log_prior"):
        if name not in fields:
            continue
        values = fields[name]
        if (
            failure := _first_true(jnp.isnan(values) | (values == jnp.inf))
        ) is not None:
            raise ValueError(
                f"{name} contains NaN or +inf at step {failure[0]}, temperature"
                f" {failure[1]}, walker {failure[2]}"
            )
    for count_name, total_name in _COUNT_PAIRS:
        if count_name not in fields:
            continue
        counts, totals = fields[count_name], fields[total_name]
        if (failure := _first_true(counts > totals)) is not None:
            raise ValueError(
                f"{count_name} exceeds {total_name} at index {failure}:"
                f" {int(counts[failure])} > {int(totals[failure])}"
            )
