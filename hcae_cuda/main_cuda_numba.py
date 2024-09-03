import time

import numpy as np
from numba import cuda, float32

from constants import NDM_ROWS, NDM_COLUMNS, DATA_SEQUENCE_SIZE

threadsperblock = (16, 16)  # Should be a multiple of 32 if possible.
blockspergrid = (256, 256)  # Blocks per grid


@cuda.jit
def main():
    init_ndm = cuda.shared.array(shape=(NDM_ROWS, NDM_COLUMNS), dtype=float32)


    x, y = cuda.grid(2)


# start CPU measurement
start_cpu = time.perf_counter()

x = np.linspace(-0.8, 0.8, 1601)
print(f"{x.shape=}")
print(f"{np.array([1, 2]).shape}")
y = np.linspace(-0.8, 0.8, 1601)
samples = np.column_stack([x, y])

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

end_cpu = time.perf_counter()
print(f"\nElapsed time: {end_cpu - start_cpu:.3f} seconds.")
