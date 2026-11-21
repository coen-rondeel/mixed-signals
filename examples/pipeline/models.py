"""The objects that ``config.yaml`` names by import path.

The case is the linear signal of the validation zoo with three correlated
parameters: a data vector that depends linearly on the parameters, in white
noise, whose Fisher matrix is ``FISHER`` at every point. ``generator`` and
``noise`` give the data-space comparison, and ``priors`` are the eryn priors of
the sampler in ``make_data.py``: uniform, ten posterior widths either side of
``CENTRE``, with the integer keys 0, 1 and 2, so that the parameters are named
``x0``, ``x1`` and ``x2`` as the zoo names them.
"""

import numpy as np
from eryn.priors import ProbDistContainer, UniformDistribution

from mixed_signals import zoo

# * Fisher matrix and centre of the toy posterior, symmetric positive definite.
FISHER = np.array(
    [
        [4.0, 1.2, 0.8],
        [1.2, 2.5, -0.6],
        [0.8, -0.6, 3.0],
    ]
)
CENTRE = np.array([1.0, -0.5, 0.25])
PRIOR_WIDTHS = 10.0  # Prior half-width in posterior standard deviations.

case = zoo.linear_signal(FISHER, centre=CENTRE)
generator = case.generator
noise = case.noise

_standard_deviations = np.sqrt(np.diag(np.linalg.inv(FISHER)))
priors = ProbDistContainer(
    {
        position: UniformDistribution(
            CENTRE[position] - PRIOR_WIDTHS * _standard_deviations[position],
            CENTRE[position] + PRIOR_WIDTHS * _standard_deviations[position],
        )
        for position in range(len(CENTRE))
    }
)
