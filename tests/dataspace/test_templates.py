"""Tests for the whitened templates of a generator and a noise model."""

import inspect
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mixed_signals.dataspace.templates import whitened_templates
from mixed_signals.errors import CapabilityError, PrecisionWarning

unchecked_templates = inspect.unwrap(whitened_templates)


class RampGenerator:
    """A generator with three entries per signal that records its calls."""

    def __init__(self) -> None:
        """Start without calls."""
        self.sizes: list[int] = []
        self.types: list[type] = []

    def generate_template(self, params: Any) -> Any:
        """Return (a, a + b, a * b) for each row (a, b)."""
        self.sizes.append(len(params))
        self.types.append(type(params))
        first, second = params[:, 0], params[:, 1]
        if isinstance(params, np.ndarray):
            return np.stack([first, first + second, first * second], axis=1)
        return jnp.stack([first, first + second, first * second], axis=1)


class ScalingNoise:
    """A noise model that divides the signals by a standard deviation."""

    def __init__(self, sigma: float) -> None:
        """Store the standard deviation."""
        self.sigma = sigma

    def whiten(self, signals: Any) -> Any:
        """Return the signals divided by sigma."""
        return signals / self.sigma


class ComplexSpectrumNoise:
    """A noise model for complex signals: real and imaginary parts, stacked."""

    def whiten(self, signals: Any) -> Any:
        """Return the real and the imaginary parts of each bin, side by side."""
        return np.concatenate([signals.real, signals.imag], axis=1)


class ComplexGenerator:
    """A generator of complex signals, one bin per parameter."""

    def generate_template(self, params: Any) -> Any:
        """Return the bin a + i b of each row (a, b)."""
        return (params[:, 0] + 1j * params[:, 1])[:, None]


class BadNoise:
    """A noise model that returns what it is told to."""

    def __init__(self, result: Any) -> None:
        """Store the result for a call."""
        self.result = result

    def whiten(self, signals: Any) -> Any:
        """Return the stored result, whatever the signals."""
        return self.result(signals) if callable(self.result) else self.result


def points(count: int = 7) -> np.ndarray:
    """Return parameter vectors shaped (count, 2)."""
    return np.stack([np.arange(count, dtype=float), np.linspace(1.0, 2.0, count)], 1)


def test_the_templates_are_whitened_rows_in_the_order_of_the_points() -> None:
    """Each row is the whitened signal of its point, as float64 shaped (N, M)."""
    rows = points()
    whitened = whitened_templates(RampGenerator(), ScalingNoise(2.0), rows)
    expected = np.stack(
        [rows[:, 0], rows[:, 0] + rows[:, 1], rows[:, 0] * rows[:, 1]], axis=1
    )
    assert whitened.shape == (7, 3)
    assert whitened.dtype == jnp.float64
    np.testing.assert_allclose(whitened, expected / 2.0, rtol=1e-14)


def test_chunks_bound_the_calls_of_the_generator() -> None:
    """Seven points in chunks of three give calls of 3, 3 and 1, and the same rows."""
    generator = RampGenerator()
    chunked = whitened_templates(generator, ScalingNoise(1.0), points(), chunk_size=3)
    assert generator.sizes == [3, 3, 1]
    whole = whitened_templates(RampGenerator(), ScalingNoise(1.0), points())
    np.testing.assert_array_equal(chunked, whole)


def test_the_generator_receives_arrays_of_the_backend() -> None:
    """NumPy and JAX backends hand over arrays of their own type."""
    numpy_generator, jax_generator = RampGenerator(), RampGenerator()
    from_numpy = whitened_templates(
        numpy_generator, ScalingNoise(1.0), points(), backend="numpy"
    )
    from_jax = whitened_templates(
        jax_generator, ScalingNoise(1.0), points(), backend="jax"
    )
    assert all(issubclass(kind, np.ndarray) for kind in numpy_generator.types)
    assert all(issubclass(kind, jax.Array) for kind in jax_generator.types)
    np.testing.assert_allclose(from_numpy, from_jax, rtol=1e-14)


def test_the_torch_backend_hands_tensors_to_the_generator() -> None:
    """Under torch, the generator and the noise model run on tensors."""
    torch = pytest.importorskip("torch")

    class TorchGenerator:
        def generate_template(self, params: Any) -> Any:
            return torch.stack([params[:, 0], 2 * params[:, 1]], dim=1)

    class TorchNoise:
        def whiten(self, signals: Any) -> Any:
            assert isinstance(signals, torch.Tensor)
            return signals

    whitened = whitened_templates(
        TorchGenerator(), TorchNoise(), points(), backend="torch"
    )
    np.testing.assert_allclose(whitened[:, 1], 2 * points()[:, 1], rtol=1e-14)


def test_complex_signals_cross_as_real_whitened_vectors() -> None:
    """A complex template is whitened by the caller, so only real rows leave."""
    whitened = whitened_templates(ComplexGenerator(), ComplexSpectrumNoise(), points())
    assert whitened.shape == (7, 2)
    np.testing.assert_allclose(whitened, points(), rtol=1e-14)


def test_a_generator_or_noise_model_without_its_method_raises() -> None:
    """The capability checks name the function, the role and the type.

    The test session checks the protocol annotations first, so the test calls
    the unwrapped function, as runs without runtime type checks do.
    """
    with pytest.raises(
        CapabilityError,
        match=r"whitened_templates needs a generator with generate_template\(params\)"
        "; ScalingNoise lacks it",
    ):
        unchecked_templates(ScalingNoise(1.0), ScalingNoise(1.0), points())
    with pytest.raises(
        CapabilityError,
        match=r"whitened_templates needs a noise model with whiten\(signals\)"
        "; RampGenerator lacks it",
    ):
        unchecked_templates(RampGenerator(), RampGenerator(), points())


@pytest.mark.parametrize(
    "result",
    [
        np.zeros(7),
        np.zeros((7, 2, 2)),
        np.zeros((7, 0)),
    ],
)
def test_whiten_must_return_vectors_shaped_n_by_m(result: np.ndarray) -> None:
    """A wrong rank or an empty vector raises with the shape that came back."""
    with pytest.raises(
        ValueError, match=r"whiten must return real vectors shaped \(n, M\), got"
    ):
        whitened_templates(RampGenerator(), BadNoise(result), points())


def test_a_wrong_number_of_rows_is_rejected_by_the_chunked_call() -> None:
    """Whitened vectors for another number of points raise in call_in_chunks."""
    with pytest.raises(ValueError, match=r"returned shape \(5, 3\) for a chunk of 7"):
        whitened_templates(RampGenerator(), BadNoise(np.zeros((5, 3))), points())


def test_complex_whitened_vectors_are_rejected() -> None:
    """The whitened vectors must be real."""
    with pytest.raises(TypeError, match="real numbers, got dtype complex128"):
        whitened_templates(
            RampGenerator(), BadNoise(lambda signals: signals + 1j), points()
        )


def test_non_finite_rows_are_counted() -> None:
    """The message names the number of rows with a NaN or an infinity."""

    def poisoned(signals: Any) -> Any:
        values = np.array(signals, dtype=float)
        values[1, 0] = np.nan
        values[4, 2] = np.inf
        values[5, 1] = -np.inf
        return values

    with pytest.raises(ValueError, match="not finite in 3 of 7 rows"):
        whitened_templates(RampGenerator(), BadNoise(poisoned), points())


def test_float32_whitened_vectors_warn_once() -> None:
    """A narrow float result is cast with the PrecisionWarning of call_in_chunks."""
    with pytest.warns(PrecisionWarning, match=r"results \(float32\)") as records:
        whitened = whitened_templates(
            RampGenerator(),
            BadNoise(lambda signals: np.asarray(signals, dtype=np.float32)),
            points(),
            chunk_size=3,
        )
    assert len(records) == 1
    assert whitened.dtype == jnp.float64


def test_the_arguments_of_the_chunked_call_are_checked() -> None:
    """A bad chunk size or an empty set of points raises before any template."""
    with pytest.raises(ValueError, match="chunk_size must be at least 1"):
        whitened_templates(RampGenerator(), ScalingNoise(1.0), points(), chunk_size=0)
    with pytest.raises(ValueError, match="at least one point"):
        unchecked_templates(RampGenerator(), ScalingNoise(1.0), np.zeros((0, 2)))
