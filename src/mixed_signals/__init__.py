"""Posterior quality checks and posterior-to-posterior comparisons.

Importing the package switches JAX to float64 for the whole process, including
any JAX code the host pipeline runs afterwards.
"""

from importlib.metadata import version as _distribution_version

from mixed_signals._config import enable_float64

# ! Switch to float64 before any subpackage import can create a JAX array.
enable_float64()

__version__: str = _distribution_version("mixed-signals")

__all__ = ["__version__"]
