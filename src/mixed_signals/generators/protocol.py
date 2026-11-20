"""Protocols of the user code that data space calls: signals, Fisher matrices, noise.

A data-space comparison compares the signals that posteriors imply, so it calls
user code for templates, for the noise-weighted inner product and, optionally,
for Fisher matrices. Any object with the right methods meets a protocol,
whatever array library it computes with, so engines call it at the boundary, in
chunks, through :func:`~mixed_signals.io.arrays.call_in_chunks`. ``isinstance``
checks that the methods exist, not their signatures. The full generator
contract, with the likelihood methods, stays deferred; the protocols hold what
data space needs.

Data grid
    All signals of one problem lie on one data grid: entry j means the same
    frequency bin, time sample and channel in every signal. A Galactic-binary
    generator whose frequency band starts at a bin that depends on the
    frequency f₀ must therefore place its templates on one band per problem.
Noise-weighted inner product
    For stationary Gaussian noise of one-sided power spectral density S(f) on
    bins of width Δf, ⟨a|b⟩ = 4 Re Σ a(f) b*(f) Δf/S(f) (Finn 1992; Cutler and
    Flanagan 1994). A :class:`NoiseModel` makes it a dot product of real
    vectors, which is what the engines evaluate.
Fisher matrix
    Γᵢⱼ = ⟨∂ᵢh|∂ⱼh⟩ for the noise model in use, and the squared distance of
    two nearby signals is ΔθᵀΓΔθ at linear order (Vallisneri 2008).

References:
    :cite:t:`finn1992detection`; :cite:t:`cutler1994gravitational`;
    :cite:t:`vallisneri2008use`.
"""

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class SignalGenerator(Protocol):
    """A generator of signals in the data domain, one per parameter vector.

    ``isinstance`` checks that the method exists, not its signature.
    """

    def generate_template(self, params: Any) -> Any:
        """Return the signals of parameters shaped (n, D), shaped (n, ...).

        Args:
            params: Parameters shaped (n, D), as arrays of the library of the
                caller, in the parameter order of the space.

        Returns:
            One signal per row, shaped (n, ...), real or complex, in the data
            domain. All signals of one problem lie on one data grid.
        """
        ...


@runtime_checkable
class NoiseModel(Protocol):
    """A noise model, as the whitening of signals to real vectors.

    The whitened vectors have the noise-weighted inner product as their dot
    product. A model may carry an optional string attribute ``description``,
    which names it in the ``space`` column of result rows; it is not a member
    of the protocol, so ``isinstance`` does not require it. ``isinstance``
    checks that the method exists, not its signature.
    """

    def whiten(self, signals: Any) -> Any:
        """Return the whitened signals of signals shaped (n, ...), shaped (n, M).

        The result is real, and its dot products are the noise-weighted inner
        products, whiten(a)·whiten(b) = ⟨a|b⟩. Whitening is linear. For
        stationary Gaussian noise of one-sided power spectral density S(f) on
        bins of width Δf, it is 2√(Δf/S(f)) times the real and imaginary parts
        of each bin, stacked (Finn 1992; Cutler and Flanagan 1994).

        Args:
            signals: Signals shaped (n, ...), as arrays of the library of the
                caller, real or complex.

        Returns:
            Real vectors shaped (n, M).
        """
        ...


@runtime_checkable
class FisherGenerator(Protocol):
    """A generator that also gives Fisher matrices for a noise model.

    ``isinstance`` checks that the method exists, not its signature.
    """

    def information_matrix(self, params: Any, *, noise: NoiseModel) -> Any:
        """Return the Fisher matrices of parameters shaped (n, D), shaped (n, D, D).

        Γᵢⱼ = ⟨∂ᵢh|∂ⱼh⟩ for the noise model that the method receives
        (Vallisneri 2008). Whitening is linear, so a generator can return
        whiten(∂ᵢh)·whiten(∂ⱼh) from its template derivatives. The engine passes
        its noise model by keyword.

        Args:
            params: Parameters shaped (n, D), as arrays of the library of the
                caller.
            noise: The noise model of the comparison.

        Returns:
            One Fisher matrix per row, shaped (n, D, D).
        """
        ...
