"""Process-wide JAX settings and package defaults.

The package works in float64 throughout: galactic-binary frequency posteriors
have a relative width near 1e-8, below the resolution of float32.

Package defaults are plain constants. There are no process-wide settings that
change them at run time, so a result never depends on hidden state; callers
pass other values as keyword arguments or through the YAML config.
"""

from typing import Final

import jax

NULL_DRAWS: Final = 200
"""Default number of null draws per result row.

With the +1 correction of the empirical p-value, n draws resolve p-values down
to 1/(n + 1) (Phipson and Smyth 2010). For 200 draws that floor is 0.005, a
one-sided Gaussian significance of 2.58σ (Cowan et al. 2011), so claims of
higher significance need more draws.

References:
    :cite:t:`phipson2010permutation`; :cite:t:`cowan2011asymptotic`.
"""

SAMPLE_CAP: Final = 10_000
"""Default number of samples per posterior above which an O(N²) metric subsamples.

Metrics whose cost grows as N², such as the energy distance, run on a uniform
random subsample of this many samples per posterior, drawn without
replacement. One energy distance in 16 parameters, three sums over all N²
pairwise distances, took 0.06 s at N = 5 000, 0.48 s at N = 10 000 and 2.0 s
at N = 20 000 on one Apple M4 Pro CPU (JAX 0.11.2, float64): about 1.6 min
per row with 200 null draws at this cap. A keyword of each metric overrides
it.
"""

PRNG_IMPLEMENTATION: Final = "threefry2x32"
"""JAX random generator of every root key, Threefry-2x32, which is the JAX default.

Root keys name it explicitly, so a different process default cannot change the
keys that a config seed gives.

References:
    :cite:t:`salmon2011parallel`; :cite:t:`jax2019prng`.
"""


def enable_float64() -> None:
    """Switch JAX to float64 for the whole process.

    The JAX setting is global, so it also applies to JAX code outside this
    package that runs in the same process.
    """
    jax.config.update("jax_enable_x64", True)  # type: ignore[no-untyped-call]


def float64_enabled() -> bool:
    """Return whether JAX currently uses float64 as its default float type."""
    return bool(jax.config.jax_enable_x64)
