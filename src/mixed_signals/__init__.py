"""Posterior quality checks and posterior-to-posterior comparisons.

Importing the package switches JAX to float64 for the whole process, including
any JAX code the host pipeline runs afterwards.
"""

from importlib.metadata import version as _distribution_version

from mixed_signals._config import enable_float64

# ! Switch to float64 before any subpackage import can create a JAX array.
enable_float64()

__version__: str = _distribution_version("mixed-signals")

# * Re-exports of the main objects, below the float64 switch.
from mixed_signals.core.chains import Chains
from mixed_signals.core.result import Result
from mixed_signals.core.samples import SampleSet
from mixed_signals.core.space import ParamSpace
from mixed_signals.engines import Reweighting, compare, diagnose, reweight
from mixed_signals.mcmc.reduce import reduce_chains
from mixed_signals.prng import root_key

__all__ = [
    "Chains",
    "ParamSpace",
    "Result",
    "Reweighting",
    "SampleSet",
    "__version__",
    "compare",
    "diagnose",
    "reduce_chains",
    "reweight",
    "root_key",
]
