import time

import numpy as np
from numba import cuda, float32

from constants import NDM_ROWS, NDM_COLUMNS, DATA_SEQUENCE_SIZE

threadsperblock = (16, 16)  # Should be a multiple of 32 if possible.
blockspergrid = (256, 256)  # Blocks per grid


@cuda.jit
def main():
    # 'cuda.shared.array' cannot be called from host code!
    init_ndm = cuda.shared.array(shape=(NDM_ROWS, NDM_COLUMNS), dtype=float32)
    x, y = cuda.grid(2)

if __name__ == "__main__":
    # start CPU measurement
    start_cpu = time.perf_counter()

    # NDM, parameters and data sequence initialization
    # initial_ndm = cuda.shared.array(shape=(NDM_ROWS, NDM_COLUMNS), dtype=float32)

    # create samples of input variables
    x = np.linspace(-0.8, 0.8, 1601)
    y = np.linspace(-0.8, 0.8, 1601)
    samples = np.column_stack([x, y])

    # indexes of input and output neurons (depends on the task, that author had in mind)
    # two first neurons
    input_neurons = np.array([[0, 1]])
    # last neuron
    output_neurons = np.array([[NDM_ROWS - 1]])

    current_error = 2000.0

    init_oper_params_1 = np.random.randint(
            0, [2, 3, NDM_ROWS, NDM_ROWS, 2 * DATA_SEQUENCE_SIZE, DATA_SEQUENCE_SIZE]
        )
    device_init_oper_params_1 = cuda.to_device(init_oper_params_1)

    init_oper_params_2 = np.random.randint(
        0, [2, 3, NDM_ROWS, NDM_ROWS, 2 * DATA_SEQUENCE_SIZE, DATA_SEQUENCE_SIZE]
    )
    device_init_oper_params_2 = cuda.to_device(init_oper_params_2)

    # initialization of the data_sequence - random values from range (-1.0; 1.0)
    init_data_seq = (2 * np.random.rand(1, DATA_SEQUENCE_SIZE) - 1)[0]
    device_init_data_seq = cuda.to_device(init_data_seq)

    main[blockspergrid, threadsperblock]()

    # 'device_array' is to be called in host code, not within CUDA-jitted functions
    device_array = cuda.device_array((10,), dtype=np.float32)

    end_cpu = time.perf_counter()
    print(f"\nElapsed time: {end_cpu - start_cpu:.3f} seconds.")
