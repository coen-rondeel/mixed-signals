# mixed-signals

mixed signals checks the quality of single posteriors and compares posteriors
from different analyses of the same data. Every number it reports comes with a
finite-sample null, an effect size (σ_eff, or r_eff for a difference in width
alone) and a significance (σ_sig). The main field is gravitational-wave data
analysis (LISA, LVK), but the core does not depend on it.

**Status:** early development. The architecture is agreed in
[DESIGN.md](DESIGN.md) and the package layout is in place; the modules are being
implemented.

## Development setup

The project needs [uv](https://docs.astral.sh/uv/) 0.13 or newer and Python 3.12
or newer.

```bash
git clone https://github.com/coen-rondeel/mixed-signals.git
cd mixed-signals
uv sync                      # .venv with every extra and the development tools
uv run pre-commit install    # run the checks on every commit
```

Importing `mixed_signals` switches JAX to float64 for the whole process.

## Checks

```bash
uv run pytest                          # CPU test suite
uv run pytest -m gpu                   # GPU suite, run by hand on a CUDA machine
uv run ruff check
uv run ruff format --check
uv run mypy                            # strict type checking
uv run python scripts/check_style.py   # style rules that ruff cannot enforce
uv run --group docs sphinx-build -W -b html docs docs/_build/html
```

## Optional extras

| Extra | Adds |
| --- | --- |
| `eryn` | Reading eryn backends and priors; during development eryn comes from its `l2d-dev` branch |
| `plot` | Corner plots with cantuccio |
| `report` | Report notebooks with marimo |

## Author

Coen Rondeel

## License

Apache License 2.0; see [LICENSE](LICENSE).
