import numba
import numpy as np
from numpy.random import randint

from constants import (DATA_SEQUENCE_SIZE,
                       PARAMETERS_SIZE,
                       MUTATION_RATE_PARAMS,
                       MUTATION_RATE_DATA_SEQ,
                       POPULATION_SIZE)
from utils import calculate_output_from_ndm

# NumPy's randint: https://numpy.org/doc/stable/reference/random/generated/numpy.random.randint.html

# NDM from the article "Neural collision ..." by Tomek
"""hardcoded_ndm = np.array([
    [0, 0.2, 0.3, 0, -0.7, 0.1],
    [-0.9, 0, 1, -0.5, -1, 0.9],
    [0.5, 0, 0, -0.5, 0.3, 0.2],
    [0, 0.3, 0, 0.6, 0.1, 0.5]
])"""

hardcoded_ndm = np.ones((5, 7)) * 0.5
x = np.linspace(-0.8, 0.8, 1601)
y = np.linspace(-0.8, 0.8, 1601)
samples = np.column_stack([x, y])

# checking NDM calculations for single output value
ndm_out = calculate_output_from_ndm(
    hardcoded_ndm,
    in_neurons=np.array([[0, 1]]),
    out_neurons=np.array([[4]]),
    in_neurons_value=samples[0]
)

# single output from the NDM for the samples[0]
print(f"Single {ndm_out=}")
