"""Read eryn runs into ``Chains``.

:func:`from_eryn`, which :meth:`~mixed_signals.core.chains.Chains.from_eryn`
calls, reads one branch of an eryn run with all its temperatures: the chain in
eryn's sampling coordinates, the log-likelihood, the log-prior and the inverse
temperatures of every stored step (Karnesis et al. 2023).

Sources
    An ``EnsembleSampler``, an eryn backend, or the path of an HDF5 file that
    an ``HDFBackend`` wrote, read without write access from eryn's default
    group.
Acceptance counts
    eryn counts the accepted in-model proposals of each walker at the stored
    steps only, ``num_repeats_in_model`` per step, whatever ``thin_by`` is, so
    the proposals behind the counts are the stored steps times
    ``num_repeats_in_model`` (Foreman-Mackey et al. 2013). The backend does
    not store that number: a sampler supplies it, and for a backend or file
    the caller may; without it, ``Chains`` hold no acceptance counts, so a
    rate is never wrong.
Swap counts
    Each stored step proposes one swap per walker between every pair of
    neighbouring temperatures (Vousden, Farr and Mandel 2016), so the
    proposals behind the swap counts are the stored steps times the walkers.
    A run with one temperature has none.
Fixed parameters
    With the priors, from a sampler or the ``priors`` argument, the reader
    drops the columns of ``DeltaFunction`` parameters, which the space of
    :meth:`~mixed_signals.core.space.ParamSpace.from_eryn_priors` leaves out
    too, after checking the parameter order that the backend stored against
    the priors. Without priors every column stays.
Scope of v1
    Runs with several independent samplers (``nsamplers`` > 1) and
    reversible-jump runs raise ``NotImplementedError``; ``Chains`` rejects
    several leaves.

eryn is optional (the ``eryn`` extra), so this module imports it inside
functions.

References:
    :cite:t:`karnesis2023eryn`; :cite:t:`foremanmackey2013emcee`;
    :cite:t:`vousden2016dynamic`.
"""

import os
from collections.abc import Mapping
from typing import Any

import numpy as np

from mixed_signals.core.chains import Chains
from mixed_signals.priors.eryn_container import _free_positions, _import_eryn


def from_eryn(
    source: object,
    *,
    branch: str | None = None,
    num_repeats_in_model: int | None = None,
    priors: object = None,
) -> Chains:
    """Read one branch of an eryn run into ``Chains``.

    :meth:`Chains.from_eryn <mixed_signals.core.chains.Chains.from_eryn>`
    calls this function; the module docstring describes what it reads.

    Args:
        source: An ``EnsembleSampler``, an eryn ``Backend`` or
            ``HDFBackend``, or the path of an HDF5 file of an ``HDFBackend``.
        branch: Name of the branch; needed only when the run has several.
        num_repeats_in_model: In-model proposals per step of the sampler,
            for the acceptance counts of a backend or file. A sampler
            supplies it.
        priors: eryn ``ProbDistContainer`` of the branch, for a backend or
            file, so that the ``DeltaFunction`` columns are dropped. A sampler
            supplies it.

    Returns:
        The chains of the branch, in eryn's sampling coordinates.

    Raises:
        TypeError: If ``source`` or ``priors`` has the wrong type, or
            ``num_repeats_in_model`` is not an integer.
        ValueError: If a sampler comes with ``num_repeats_in_model`` or
            ``priors``, the backend holds no stored step, the branch is
            missing or ambiguous, or the priors do not fit the stored
            parameters.
        NotImplementedError: For runs with several independent samplers or
            reversible jumps.
        MissingDependencyError: If eryn is not installed.

    References:
        :cite:t:`karnesis2023eryn`; :cite:t:`foremanmackey2013emcee`;
        :cite:t:`vousden2016dynamic`.
    """
    backend, num_repeats_in_model, sampler_priors = _resolve_source(
        source, num_repeats_in_model, priors
    )
    branch = _check_run(backend, branch)
    if sampler_priors is not None:
        priors = _branch_priors(sampler_priors, branch)
    samples = np.asarray(backend.get_chain(branch_names=[branch])[branch])
    if priors is not None:
        _check_parameter_order(backend, branch, priors, samples.shape[-1])
        samples = samples[..., list(_free_positions(priors))]
    stored_steps = int(backend.iteration)
    temp_count, walker_count = samples.shape[1], samples.shape[2]
    counts: dict[str, np.ndarray] = {}
    if num_repeats_in_model is not None:
        accepted = np.asarray(backend.accepted)
        counts["accepted"] = accepted
        counts["proposed"] = np.full(
            accepted.shape, stored_steps * num_repeats_in_model
        )
    if temp_count > 1:
        swaps = np.asarray(backend.swaps_accepted)
        counts["swaps_accepted"] = swaps
        counts["swaps_proposed"] = np.full(swaps.shape, stored_steps * walker_count)
    return Chains(
        samples,
        betas=np.asarray(backend.get_betas()),
        log_like=np.asarray(backend.get_log_like()),
        log_prior=np.asarray(backend.get_log_prior()),
        **counts,
    )


def _resolve_source(
    source: object, num_repeats_in_model: int | None, priors: object
) -> tuple[Any, int | None, Mapping[str, object] | None]:
    """Return the backend, the repeats and, for a sampler, its priors by branch."""
    ensemble = _import_eryn("eryn.ensemble")
    backends = _import_eryn("eryn.backends")
    if isinstance(source, ensemble.EnsembleSampler):
        if num_repeats_in_model is not None or priors is not None:
            raise ValueError(
                "an EnsembleSampler supplies num_repeats_in_model and its priors;"
                " leave both out"
            )
        return source.backend, int(source.num_repeats_in_model), source.priors
    if num_repeats_in_model is not None:
        if isinstance(num_repeats_in_model, bool) or not isinstance(
            num_repeats_in_model, int | np.integer
        ):
            raise TypeError(
                "num_repeats_in_model must be an integer, got"
                f" {type(num_repeats_in_model).__name__}"
            )
        if num_repeats_in_model < 1:
            raise ValueError(
                f"num_repeats_in_model must be at least 1, got {num_repeats_in_model}"
            )
        num_repeats_in_model = int(num_repeats_in_model)
    if isinstance(source, backends.Backend):
        return source, num_repeats_in_model, None
    if isinstance(source, str | os.PathLike):
        path = os.fspath(source)
        return backends.HDFBackend(path, read_only=True), num_repeats_in_model, None
    raise TypeError(
        "source must be an eryn EnsembleSampler, an eryn backend or the path of"
        f" an HDF5 file, got {type(source).__name__}"
    )


def _branch_priors(sampler_priors: Mapping[str, object], branch: str) -> object:
    """Return a sampler's container for the branch, or ``None`` for other priors.

    eryn also samples with any object that has ``logpdf``; such priors name no
    fixed parameters, so every column stays.
    """
    priors = sampler_priors.get(branch)
    container_type = _import_eryn("eryn.priors").ProbDistContainer
    return priors if isinstance(priors, container_type) else None


def _check_run(backend: Any, branch: str | None) -> str:
    """Check that v1 can read the run, and return the name of its branch."""
    if not backend.initialized:
        location = getattr(backend, "filename", None)
        where = f" in {location!r}" if location is not None else ""
        raise ValueError(f"the eryn backend holds no run{where}")
    sampler_count = int(getattr(backend, "nsamplers", 1))
    if sampler_count > 1:
        raise NotImplementedError(
            f"the run has {sampler_count} independent samplers; v1 reads runs with"
            " one sampler"
        )
    if bool(backend.rj):
        raise NotImplementedError(
            "the run uses reversible jumps, which are trans-dimensional; v1 reads"
            " fixed-dimensional runs only"
        )
    if int(backend.iteration) == 0:
        raise ValueError("the eryn backend holds no stored step")
    branch_names = [str(name) for name in backend.branch_names]
    if branch is None:
        if len(branch_names) > 1:
            raise ValueError(
                f"the run has the branches {', '.join(map(repr, branch_names))};"
                " choose one with branch="
            )
        return branch_names[0]
    if branch not in branch_names:
        raise ValueError(
            f"the run has no branch {branch!r}; it has"
            f" {', '.join(map(repr, branch_names))}"
        )
    return branch


def _check_parameter_order(
    backend: Any, branch: str, priors: object, param_count: int
) -> None:
    """Check that the priors describe the stored parameters, in the stored order."""
    priors_module = _import_eryn("eryn.priors")
    if not isinstance(priors, priors_module.ProbDistContainer):
        raise TypeError(
            f"priors must be an eryn ProbDistContainer, got {type(priors).__name__}"
        )
    if priors.ndim != param_count:
        raise ValueError(
            f"the priors have {priors.ndim} parameters, but branch {branch!r}"
            f" stores {param_count}"
        )
    stored = getattr(backend, "key_order", None)
    if isinstance(stored, Mapping) and branch in stored:
        stored_keys = [str(key) for key in stored[branch]]
        prior_keys = [str(key) for key in priors.key_order]
        if stored_keys != prior_keys:
            raise ValueError(
                f"branch {branch!r} stores the parameters {stored_keys}, but the"
                f" priors have {prior_keys}"
            )
