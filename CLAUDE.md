# mixed-signals

A JAX package that checks the quality of single posteriors and compares posteriors from different analyses of the same data, reporting every number against a finite-sample null. Built for gravitational-wave data analysis (LISA, LVK), but the core is domain-agnostic.

## Read first

`DESIGN.md` holds the agreed architecture: 95 decisions, the package layout, what is open or deferred, and the sources of every method. Follow it.

`ROADMAP.md` splits the implementation into sessions 0 to 19 and tracks their status. Take the next open session, follow its brief and the routine at the top of the file, and update the status table when done.

- Items marked **Proposed** or **Open** are not agreed yet. Say so when work depends on one, and ask before treating it as settled.
- When a decision changes, update `DESIGN.md` in the same change.
- When a choice is settled neither here nor in `DESIGN.md`, ask the developers before picking one.

## Status

The design is closed: 8 design rounds, then the close-out in session 0 (2026-10-13). Every item in `DESIGN.md` is Decided or Deferred, except the form of eryn transforms, which session 7 settles. Session 1 (Foundations, 2026-10-13) implemented `errors.py`, `prng.py`, `_config.py`, `core/topology.py`, `core/space.py`, `core/result.py` and the citation setup in `docs/`. Session 2 (Sample containers, 2026-10-13) implemented `core/samples.py`, `core/chains.py` and `io/arrays.py`, and benched `io/weights.py`. The other modules in the Package layout are docstring-only stubs, except `cli.py` (`--version` only). Next are the wave B sessions 3 to 6 of `ROADMAP.md`, in any order; session 3, Validation zoo, is marked next.

## Commands

- `uv sync`: development environment with every extra except the CUDA ones, plus the test and lint tools.
- `uv run pytest`: CPU tests. `uv run pytest -m gpu`: GPU suite, run by hand on a CUDA machine after `uv sync --extra cuda12` or `--extra cuda13`.
- `uv run ruff check`, `uv run ruff format`, `uv run mypy` and `uv run python scripts/check_style.py` must all pass; pre-commit runs everything except mypy.
- `uv run --group docs sphinx-build -W -b html docs docs/_build/html`: docs build, where warnings fail.
- CI (`.github/workflows/ci.yml`) runs the same checks plus the pre-commit hygiene hooks, CPU only, with the tests on Python 3.12 and 3.13. Its actions are pinned by commit SHA with a version comment; update both together.

## Commits

End every commit message with these two trailers and no Co-Authored-By line. Use the model name only, without the context window:

```text
Implemented-and-tested-by: <model name> (Anthropic AI)
Designed-verified-and-reviewed-by: coen-rondeel <coen.rondeel@gmail.com>
```

## Conventions

- uv project (`uv_build` backend), uv ≥ 0.13, Python ≥ 3.12, Apache-2.0. Distribution `mixed-signals`, import `mixed_signals`, CLI `mixed-signals`.
- Core dependencies: jax, ott-jax, equinox, jaxtyping, numpy, polars, pydantic, pyyaml. Everything else is an extra; add each new extra to the `mixed-signals[...]` entry of the `dev` group as well, except the CUDA extras `cuda12` and `cuda13`, which need a CUDA machine. uv takes CPU-only torch from the PyTorch index on Linux and Windows, so the dev environment and CI stay small.
- eryn is the `eryn` extra, from the `l2d-dev` branch of https://github.com/mikekatz04/Eryn as a uv git source until it is released. eryn-specific code lives only in `io/eryn_backend.py` and `priors/eryn_container.py`, which import eryn inside functions so `import mixed_signals` works without it. Use `eryn.priors`; `eryn.prior` is deprecated.
- JAX is a hard dependency. Importing the package enables `jax_enable_x64`; work in float64. Re-exports in `__init__.py` go below the float64 switch.
- equinox modules for anything that crosses `jit`; pydantic for configs, `ParamSpace` and provenance; polars for result tables, written to parquet.
- jaxtyping shape annotations on every kernel. The pytest configuration runs beartype on every annotated function in `mixed_signals`; library code never imports beartype.
- jaxtyping dimension names are words: `batch posterior sample param` for the (B, K, N, D) axes of `DESIGN.md`, and `step temp walker leaf param` for eryn chains, for example `Float[Array, "sample param"]`. So far two more exist: `*leading` for any leading axes of a kernel, and `selected` for positions picked from an axis. Add new words sparingly and reuse them across modules. Ruff ignores F722, because shape strings are not Python; write a single-word shape with a leading space, such as `" sample"`, so that F821 does not read it as a name.
- Google-style docstrings: ruff `D` rules with the google convention, rendered by Sphinx napoleon.
- Tests mirror `src/` under `tests/`. GPU tests live in `tests/gpu/` and carry `pytestmark = pytest.mark.gpu`. Warnings fail a test unless it expects them.
- Every stochastic function takes an explicit JAX PRNG `key`; pipelines derive all keys from one config seed. Derive each key with `prng.derive_key` from a path of names, never by counting calls, so that keys do not depend on call order.
- Raise package warnings from the classes in `errors.py`, outside jit, where a cached call would not warn again; engines capture them per row with `errors.capture_warnings`.
- Do not annotate with `typing.Self`: beartype rejects it in classes that it does not decorate, and the test hook decorates functions only. Write the class name instead.
- The test hook checks the arguments of an equinox module's own `__init__`, not its field annotations, so containers check their shapes themselves. They check values only for concrete inputs, since traced values are unknown inside `jax.jit`. Annotate inputs whose dtype and shape the code checks with broad types, so that tests reach the package's own messages.
- Engines (`diagnose`, `compare`, `reweight`, data space) are plain functions that return results.
- Every comparison returns a finite-sample null plus σ_sig and an effect size: σ_eff, or r_eff for rows that measure only a difference in width. Pass/fail thresholds act on the effect size.
- Every diagnostic, metric, estimator and null cites the paper, official document or reference package that defines it, in a Google-style `References` docstring section with `:cite:t:` roles and keys from `docs/references.bib`. Keys are first author, year and first title word in lowercase ASCII, such as `mardia2000directional`. Every assumption (a default, a threshold, a σ_eff or r_eff map, a null construction) cites a paper or a section of the validation page. `DESIGN.md` lists the sources of every v1 method; a new method needs a verifiable source before it lands, and a citation is never guessed.
- Both inputs to a comparison share one whitening, recorded in the result.
- Never assume user code is JAX: generators and likelihoods may be CuPy or PyTorch and are called at the boundary.
- YAML configs, argparse CLI, Sphinx docs.
- GPU tests are a marked suite, run by hand; v1 targets a single GPU.
- The developers own optimal-transport method choices: propose options rather than picking one.

## Style rules

These apply to code, docstrings, comments and documentation; `scripts/check_style.py` enforces the mechanical ones.

- Plain language (ISO 24495) in a scientific register, as a gravitational-wave data analyst would write it.
- No em-dashes and no emojis. Docstrings may use Unicode mathematics such as σ, π and −; ruff allows σ and − as confusable characters.
- No one-letter names for variables, parameters, attributes, functions, classes or type parameters (`_` is fine). Abbreviations are allowed with a short comment that expands them.
- Number literals with five or more digits in a row use `_` separators: `10_000`, `0.000_01`.
- Comments are one or two lines and explain why, not what. Use Better Comments tags where they help: `# !` warning, `# ?` question or uncertainty, `# *` highlight, `# TODO` gap. Longer explanations belong in the docs.
- State reasons directly; never justify a change by pointing at a label in another document, such as a question number.
- Refer to "the developers", not to a person by name; the README author line and the commit trailers are the exceptions.
- A line containing `# style: allow` is exempt from the name and number rules; use it only when an external API forces the name.
