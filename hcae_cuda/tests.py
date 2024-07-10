import random

from numba import cuda, jit
import numpy as np
from numpy.random import randint
from tqdm import tqdm

from constants import (DATA_SEQUENCE_SIZE,
                       PARAMETERS_SIZE,
                       MUTATION_RATE_PARAMS,
                       MUTATION_RATE_DATA_SEQ,
                       POPULATION_SIZE, NDM_ROWS, NDM_COLUMNS)
from hcae_operations import oper2
from utils import calculate_output_from_ndm, calculate_error

# NumPy's randint: https://numpy.org/doc/stable/reference/random/generated/numpy.random.randint.html

# NDM from the article "Neural collision ..." by Tomek
"""hardcoded_ndm = np.array([
    [0, 0.2, 0.3, 0, -0.7, 0.1],
    [-0.9, 0, 1, -0.5, -1, 0.9],
    [0.5, 0, 0, -0.5, 0.3, 0.2],
    [0, 0.3, 0, 0.6, 0.1, 0.5]
])"""


@jit
def initialize_ndm() -> np.ndarray:
    # initialization of the NDM - random values from range (-1.0; 1.0)
    # init_ndm = 2 * np.random.rand(NDM_ROWS, NDM_COLUMNS) - 1

    # initialization of the NDM - zero values matrix
    init_ndm = np.zeros((NDM_ROWS, NDM_COLUMNS))

    # zero the values under the 1st diagonal
    # https: // numpy.org / doc / stable / reference / generated / numpy.triu.html
    # init_ndm = np.triu(init_ndm, k=1)
    return init_ndm


hardcoded_ndm = np.ones((5, 7)) * 0.5
dev_hardcoded_ndm = cuda.to_device(hardcoded_ndm)

x = np.linspace(-0.8, 0.8, 1601)
y = np.linspace(-0.8, 0.8, 1601)
samples = np.column_stack([x, y])
dev_samples = cuda.to_device(samples)

# checking NDM calculations for single output value
ndm_out = calculate_output_from_ndm(
    hardcoded_ndm,
    in_neurons=np.array([[0, 1]]),
    out_neurons=np.array([[4]]),
    in_neurons_value=samples[0]
)

# single output from the NDM for the samples[0]
print(f"Single {ndm_out=}")

# two first neurons
input_neurons = np.array([[0, 1]])
dev_input_neurons = cuda.to_device(input_neurons)

# last neuron
output_neurons = np.array([[NDM_ROWS - 1]])
dev_output_neurons = cuda.to_device(output_neurons)

population_data_seq = [
    (2 * np.random.rand(1, DATA_SEQUENCE_SIZE) - 1)[0]
    for _ in range(POPULATION_SIZE)
]
dev_population_data_seq = cuda.to_device(population_data_seq)

# initialize "best" params_1, data_seq and params_2
# initial candidates - two for operations and one for data sequence
initial_ndm = initialize_ndm()
best_ndm_for_params_1 = initial_ndm.copy()

best_data_seq = random.choice(population_data_seq)
dev_best_data_seq = cuda.to_device(best_data_seq)

population_params_1 = [
    np.random.randint(
        0, [2, 3, NDM_ROWS, NDM_ROWS, DATA_SEQUENCE_SIZE, DATA_SEQUENCE_SIZE], dtype=np.int64
    )
    for _ in range(POPULATION_SIZE)
]
dev_population_params_1 = cuda.to_device(population_params_1)


@cuda.jit
def my_kernel(scores_for_params_1):
    """
    Return the absolute position of the current thread in the entire grid of blocks.
    *ndim* should correspond to the number of dimensions declared when instantiating the kernel.
    If *ndim* is 1, a single integer is returned.
    If *ndim* is 2 or 3, a tuple of the given number of integers is returned.
    Computation of the first integer is as follows::  cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    and is similar for the other two indices, but using the `y` and `z` attributes.
    """
    pos = cuda.grid(1)
    if pos < scores_for_params_1.size:
        """scores_for_params_1[pos] = calculate_error(
            oper2(population_params_1[pos], best_data_seq, best_ndm_for_params_1),
            # updated NDM after changing operation parameters_1
            samples,
            in_neurons=input_neurons,
            out_neurons=output_neurons,
        )"""
        scores_for_params_1[pos] = pos ** 2

threads_per_block = 256
blocks_per_grid = (POPULATION_SIZE + (threads_per_block - 1)) // threads_per_block
some_scores = np.zeros(POPULATION_SIZE, dtype=np.float32)
dev_scores_for_params_1 = cuda.to_device(some_scores)

my_kernel[blocks_per_grid, threads_per_block](dev_scores_for_params_1)
host_scores = dev_scores_for_params_1.copy_to_host()
print(f"{host_scores=}")

