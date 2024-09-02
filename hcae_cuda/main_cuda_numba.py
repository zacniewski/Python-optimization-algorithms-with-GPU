import time

import numpy as np
from numba import cuda, float32

from constants import NDM_ROWS, NDM_COLUMNS

threadsperblock = (16, 16)  # Should be a multiple of 32 if possible.
blockspergrid = (256, 256)  # Blocks per grid


@cuda.jit
def main():
    # start CPU measurement
    init_ndm = cuda.shared.array(shape=(NDM_ROWS, NDM_COLUMNS), dtype=float32)
    x, y = cuda.grid(2)


start_cpu = time.perf_counter()

x = np.linspace(-0.8, 0.8, 1601)
y = np.linspace(-0.8, 0.8, 1601)
samples = np.column_stack([x, y])

main[blockspergrid, threadsperblock]()

end_cpu = time.perf_counter()
print(f"\nElapsed time: {end_cpu - start_cpu:.3f} seconds.")
