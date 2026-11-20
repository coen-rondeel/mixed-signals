"""Tests for the toy signals of the zoo: shapes, Fisher matrices, SNR and Gaussians."""

import dataclasses
import math
from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mixed_signals.generators import FisherGenerator, NoiseModel, SignalGenerator
from mixed_signals.zoo import (
    LinearSignal,
    SignalCase,
    SinusoidSignal,
    WhiteNoise,
    ZooCase,
    linear_signal,
    sinusoid_signal,
)
from mixed_signals.zoo.cases import _real_space
from mixed_signals.zoo.distributions import Gaussian

FISHER = np.array(
    [
        [1.0, 0.8, 0.2, 0.0],
        [0.8, 4.0, 0.9, 0.1],
        [0.2, 0.9, 9.0, 1.2],
        [0.0, 0.1, 1.2, 16.0],
    ]
)


def linear_generator(case: SignalCase) -> LinearSignal:
    """Return the generator of a case that the test builds as a linear signal."""
    assert isinstance(case.generator, LinearSignal)
    return case.generator


def sinusoid_generator(case: SignalCase) -> SinusoidSignal:
    """Return the generator of a case that the test builds as a sinusoid."""
    assert isinstance(case.generator, SinusoidSignal)
    return case.generator


def gaussian_pair(case: ZooCase) -> tuple[Gaussian, Gaussian]:
    """Return the two Gaussians of a case of ``fisher_gaussians``."""
    reference, shifted = case.posteriors
    assert isinstance(reference, Gaussian)
    assert isinstance(shifted, Gaussian)
    return reference, shifted


def sinusoid_jacobian(centre: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Return the template Jacobian by central differences in NumPy, shaped (M, 3)."""

    def template(parameters: np.ndarray) -> np.ndarray:
        amplitude, frequency, phase = parameters
        signal: np.ndarray = amplitude * np.sin(2 * np.pi * frequency * times + phase)
        return signal

    steps = 1e-6 * np.maximum(np.abs(centre), 1.0)
    columns = []
    for index, step in enumerate(steps):
        shift = np.zeros(3)
        shift[index] = step
        columns.append(
            (template(centre + shift) - template(centre - shift)) / (2 * step)
        )
    jacobian: np.ndarray = np.stack(columns, axis=1)
    return jacobian


# * White noise


def test_white_noise_divides_by_sigma_and_flattens_the_data_axes() -> None:
    """Signals become real vectors (n, M) over σ: whiten(a)·whiten(b) = a·b/σ²."""
    noise = WhiteNoise(2.0)
    signals = np.arange(24.0).reshape(2, 3, 4)
    whitened = noise.whiten(signals)
    assert whitened.shape == (2, 12)
    assert whitened.dtype == jnp.float64
    np.testing.assert_allclose(whitened, signals.reshape(2, 12) / 2.0, rtol=1e-15)
    first, second = np.asarray(whitened[0]), np.asarray(whitened[1])
    np.testing.assert_allclose(
        first @ second, signals[0].ravel() @ signals[1].ravel() / 4.0, rtol=1e-14
    )
    assert noise.whiten(np.arange(3.0)).shape == (3, 1)


def test_white_noise_describes_itself_and_rejects_bad_input() -> None:
    """The description is the default 'white' or the caller's; complex input raises."""
    assert WhiteNoise().description == "white"
    assert WhiteNoise(description="detector").description == "detector"
    with pytest.raises(ValueError, match="whitens real signals"):
        WhiteNoise().whiten(np.ones((2, 3), dtype=complex))
    with pytest.raises(ValueError, match=r"shaped \(n, \.\.\.\), got a scalar"):
        WhiteNoise().whiten(np.float64(1.0))
    for bad_sigma in (0.0, -1.0, math.nan, math.inf):
        with pytest.raises(ValueError, match="sigma must be finite and positive"):
            WhiteNoise(bad_sigma)


# * The generators


def test_the_linear_signal_maps_parameters_to_data_entries() -> None:
    """The template θ Aᵀ + b is shaped (n, M) and linear in the parameters."""
    matrix = np.arange(12.0).reshape(4, 3) / 5
    offset = np.array([1.0, -1.0, 0.5, 0.0])
    generator = LinearSignal(matrix, offset)
    params = np.random.default_rng(0).normal(size=(5, 3))
    templates = generator.generate_template(params)
    assert templates.shape == (5, 4)
    np.testing.assert_allclose(templates, params @ matrix.T + offset, rtol=1e-14)
    assert isinstance(generator.matrix, jax.Array)


def test_the_linear_fisher_matrix_is_the_same_gram_matrix_for_every_row() -> None:
    """Γ = whiten(A)ᵀ whiten(A) for the noise it receives, broadcast to (n, D, D)."""
    matrix = np.random.default_rng(1).normal(size=(6, 3))
    generator = LinearSignal(matrix, np.zeros(6))
    gram = generator.information_matrix(np.zeros((4, 3)), noise=WhiteNoise(2.0))
    assert gram.shape == (4, 3, 3)
    np.testing.assert_allclose(gram[2], matrix.T @ matrix / 4.0, rtol=1e-13)


def test_the_linear_signal_checks_its_inputs() -> None:
    """Shape, value and offset-length errors name the argument."""
    with pytest.raises(ValueError, match=r"matrix must be shaped \(M, D\)"):
        LinearSignal(np.zeros(3), np.zeros(3))
    with pytest.raises(ValueError, match="matrix must be finite"):
        LinearSignal(np.array([[1.0, np.nan]]), np.zeros(1))
    with pytest.raises(ValueError, match=r"offset must be shaped \(2,\)"):
        LinearSignal(np.ones((2, 2)), np.zeros(3))
    with pytest.raises(ValueError, match="offset must be finite"):
        LinearSignal(np.ones((2, 2)), np.array([0.0, np.inf]))


def test_the_sinusoid_template_follows_its_formula() -> None:
    """The template A sin(2π f t + φ) is shaped (n, M) for (A, f, φ)."""
    times = np.linspace(-0.5, 0.5, 50)
    generator = SinusoidSignal(times)
    params = np.array([[2.0, 3.0, 0.4], [0.5, 7.0, -1.0]])
    templates = generator.generate_template(params)
    assert templates.shape == (2, 50)
    expected = params[:, :1] * np.sin(
        2 * np.pi * params[:, 1:2] * times + params[:, 2:3]
    )
    np.testing.assert_allclose(templates, expected, rtol=1e-13, atol=1e-14)
    with pytest.raises(
        ValueError, match=r"times must be shaped \(D,\), got shape \(2, 2\)"
    ):
        SinusoidSignal(np.zeros((2, 2)))
    with pytest.raises(ValueError, match="times must be finite"):
        SinusoidSignal(np.array([0.0, np.nan]))


def test_the_toy_generators_and_the_noise_meet_the_protocols() -> None:
    """The three objects of the zoo are what the data-space engine asks for."""
    for generator in (
        LinearSignal(np.ones((3, 2)), np.zeros(3)),
        SinusoidSignal(np.zeros(3)),
    ):
        assert isinstance(generator, SignalGenerator)
        assert isinstance(generator, FisherGenerator)
    assert isinstance(WhiteNoise(), NoiseModel)


# * The builders


def test_the_linear_fisher_matrix_equals_the_given_matrix() -> None:
    """The whitened matrix BU has the Gram matrix Γ, to 1e-12, in noise of any σ."""
    for noise_sigma in (1.0, 3.0):
        case = linear_signal(FISHER, data_size=32, noise_sigma=noise_sigma)
        assert case.name == "linear_signal"
        assert case.space.names == ("x0", "x1", "x2", "x3")
        assert case.generator.generate_template(case.centre[None]).shape == (1, 32)
        gram = case.generator.information_matrix(case.centre[None], noise=case.noise)[0]
        np.testing.assert_allclose(gram, FISHER, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(case.fisher, FISHER, rtol=1e-15)
        np.testing.assert_array_equal(case.centre, np.zeros(4))
        np.testing.assert_array_equal(linear_generator(case).offset, np.zeros(32))


def test_the_cosine_basis_is_orthonormal_and_a_given_centre_is_kept() -> None:
    """The matrix is σ B U with B orthonormal, and a centre stays as given."""
    case = linear_signal(FISHER, centre=[1.0, 2.0, 3.0, 4.0], data_size=7)
    np.testing.assert_allclose(case.centre, [1.0, 2.0, 3.0, 4.0])
    upper = np.linalg.cholesky(FISHER).T
    basis = np.asarray(linear_generator(case).matrix) @ np.linalg.inv(upper)
    np.testing.assert_allclose(basis.T @ basis, np.eye(4), atol=1e-13)
    # * Data size equal to D: B is a square orthogonal matrix.
    square = linear_signal(FISHER, data_size=4)
    gram = square.generator.information_matrix(np.zeros((1, 4)), noise=square.noise)[0]
    np.testing.assert_allclose(gram, FISHER, rtol=1e-12, atol=1e-12)


def test_the_sinusoid_signal_to_noise_ratio_is_exact() -> None:
    """The whitened template at the centre has the norm SNR, to 1e-12."""
    for signal_to_noise, noise_sigma in ((5.0, 1.0), (30.0, 2.5), (100.0, 1.0)):
        case = sinusoid_signal(signal_to_noise, noise_sigma=noise_sigma)
        template = case.generator.generate_template(case.centre[None])
        norm = float(jnp.linalg.norm(case.noise.whiten(template)))
        assert norm == pytest.approx(signal_to_noise, rel=1e-12)
        assert case.name == "sinusoid_signal"
        assert jnp.shape(sinusoid_generator(case).times) == (1_000,)
        np.testing.assert_allclose(case.centre[1:], [40.0, 1.0])
    custom = sinusoid_signal(10.0, frequency=12.5, phase=-0.3, data_size=200)
    np.testing.assert_allclose(custom.centre[1:], [12.5, -0.3])
    assert jnp.shape(sinusoid_generator(custom).times) == (200,)
    assert float(jnp.asarray(sinusoid_generator(custom).times)[0]) == -0.5
    assert float(jnp.asarray(sinusoid_generator(custom).times)[-1]) == 0.5


def test_the_sinusoid_space_has_real_parameters_with_labels() -> None:
    """Amplitude, frequency and phase, on real lines, with plot labels."""
    space = sinusoid_signal(10.0).space
    assert space.names == ("amplitude", "frequency", "phase")
    assert [parameter.label for parameter in space.parameters] == ["A", "f", r"\phi"]
    assert space.axis_blocks.circles == ()
    assert all(parameter.topology == "real" for parameter in space.parameters)


@pytest.mark.parametrize("noise_sigma", [1.0, 2.0])
def test_the_sinusoid_fisher_matrix_matches_finite_differences(
    noise_sigma: float,
) -> None:
    """Γ from autodiff equals JᵀJ/σ² from central differences, to 1e-6 relative."""
    case = sinusoid_signal(12.0, noise_sigma=noise_sigma)
    jacobian = sinusoid_jacobian(
        np.asarray(case.centre), np.asarray(sinusoid_generator(case).times)
    )
    expected = jacobian.T @ jacobian / noise_sigma**2
    np.testing.assert_allclose(case.fisher, expected, rtol=1e-6)
    batched = case.generator.information_matrix(
        np.broadcast_to(np.asarray(case.centre), (3, 3)), noise=case.noise
    )
    assert batched.shape == (3, 3, 3)
    np.testing.assert_allclose(batched[1], case.fisher, rtol=1e-12)


# * The Gaussians of the Fisher matrix


def test_fisher_gaussians_have_the_inverse_fisher_covariance_and_the_shift() -> None:
    """Both posteriors have Γ⁻¹, the second is shifted by the Mahalanobis length."""
    case = linear_signal(FISHER, centre=[0.5, 0.0, -1.0, 2.0], data_size=16)
    gaussians = case.fisher_gaussians(1.5)
    assert isinstance(gaussians, ZooCase)
    assert gaussians.name == "fisher_gaussians"
    assert gaussians.names == ("reference", "shifted")
    assert gaussians.space.names == case.space.names
    reference, shifted = gaussian_pair(gaussians)
    np.testing.assert_allclose(reference.covariance, np.linalg.inv(FISHER), rtol=1e-10)
    np.testing.assert_array_equal(reference.covariance, shifted.covariance)
    np.testing.assert_allclose(reference.mean, case.centre)
    offset = np.asarray(shifted.mean - reference.mean)
    assert offset @ FISHER @ offset == pytest.approx(1.5**2, rel=1e-10)
    # * The default direction is the first axis.
    np.testing.assert_allclose(offset[1:] / offset[0], 0.0, atol=1e-15)
    assert gaussians.reference_value(
        "mahalanobis_shift", ("reference", "shifted")
    ) == pytest.approx(1.5, rel=1e-10)


def test_fisher_gaussians_along_a_direction_and_in_the_names_of_the_case() -> None:
    """An oblique direction keeps its Mahalanobis length; names come from the case."""
    case = sinusoid_signal(20.0)
    direction = [0.0, 1.0, 0.3]
    gaussians = case.fisher_gaussians(0.5, direction=direction)
    assert gaussians.space.names == ("amplitude", "frequency", "phase")
    reference, shifted = gaussian_pair(gaussians)
    offset = np.asarray(shifted.mean - reference.mean)
    assert offset @ np.asarray(case.fisher) @ offset == pytest.approx(0.25, rel=1e-9)
    np.testing.assert_allclose(offset / offset[1], np.array(direction) / direction[1])
    zero = case.fisher_gaussians(0.0)
    assert zero.reference_value("mahalanobis_shift", ("reference", "shifted")) == 0.0
    negative = case.fisher_gaussians(-0.5)
    assert negative.reference_value(
        "mahalanobis_shift", ("reference", "shifted")
    ) == pytest.approx(0.5, rel=1e-9)


def test_fisher_gaussians_check_their_inputs() -> None:
    """A shift that is not finite, a bad direction and a bad matrix raise."""
    case = linear_signal(FISHER, data_size=8)
    with pytest.raises(ValueError, match="shift must be finite"):
        case.fisher_gaussians(math.nan)
    with pytest.raises(ValueError, match=r"direction must be shaped \(4,\)"):
        case.fisher_gaussians(1.0, direction=[1.0, 0.0])
    with pytest.raises(ValueError, match="direction must not be the zero vector"):
        case.fisher_gaussians(1.0, direction=np.zeros(4))
    singular = dataclasses.replace(case, fisher=jnp.zeros((4, 4)).at[0, 0].set(1.0))
    with pytest.raises(
        ValueError, match="the inverse of the Fisher matrix must be finite"
    ):
        singular.fisher_gaussians(1.0)
    indefinite = dataclasses.replace(case, fisher=jnp.diag(jnp.array([1.0, 1, 1, -1])))
    with pytest.raises(
        ValueError, match="the inverse of the Fisher matrix must be positive definite"
    ):
        indefinite.fisher_gaussians(1.0)


# * The checks of the builders and the case


@pytest.mark.parametrize(
    ("build", "message"),
    [
        (lambda: linear_signal(np.ones(3)), "fisher must be a square matrix"),
        (lambda: linear_signal(np.ones((2, 3))), "fisher must be a square matrix"),
        (
            lambda: linear_signal(FISHER, data_size=3),
            "data_size must be at least the 4",
        ),
        (lambda: linear_signal(FISHER, data_size=0), "data_size must be at least 1"),
        (lambda: linear_signal(FISHER, noise_sigma=0.0), "noise_sigma must be finite"),
        (
            lambda: linear_signal(np.diag([1.0, -1.0])),
            "fisher must be positive definite",
        ),
        (
            lambda: linear_signal(np.array([[1.0, 0.5], [0.0, 1.0]])),
            "fisher must be symmetric",
        ),
        (lambda: linear_signal(np.diag([1.0, np.nan])), "fisher must be finite"),
        (
            lambda: linear_signal(FISHER, centre=[0.0, 1.0]),
            r"centre must be shaped \(4,\)",
        ),
        (lambda: sinusoid_signal(0.0), "signal_to_noise must be finite and positive"),
        (lambda: sinusoid_signal(math.inf), "signal_to_noise must be finite"),
        (lambda: sinusoid_signal(5.0, frequency=math.nan), "frequency must be finite"),
        (lambda: sinusoid_signal(5.0, phase=math.inf), "phase must be finite"),
        (lambda: sinusoid_signal(5.0, data_size=2), "data_size must be at least 3"),
        (lambda: sinusoid_signal(5.0, noise_sigma=-1.0), "noise_sigma must be finite"),
        (
            lambda: sinusoid_signal(5.0, frequency=0.0, phase=0.0),
            "the tone vanishes at every sample time",
        ),
    ],
)
def test_the_builders_check_their_inputs(
    build: Callable[[], object], message: str
) -> None:
    """Every invalid argument raises with a message that names it."""
    with pytest.raises((ValueError, TypeError), match=message):
        build()


def test_a_signal_case_checks_that_its_centre_and_fisher_fit_the_space() -> None:
    """The container rejects a centre or a Fisher matrix of another size."""
    generator = LinearSignal(np.ones((3, 2)), np.zeros(3))
    space = _real_space(("x0", "x1"))
    with pytest.raises(ValueError, match=r"centre must be shaped \(2,\)"):
        SignalCase("case", generator, WhiteNoise(), space, jnp.zeros(3), jnp.eye(2))
    with pytest.raises(ValueError, match=r"fisher must be shaped \(2, 2\)"):
        SignalCase("case", generator, WhiteNoise(), space, jnp.zeros(2), jnp.eye(3))
