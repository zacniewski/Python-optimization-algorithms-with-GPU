import time

import numpy as np
from numba import cuda, float32

from constants import NDM_ROWS, NDM_COLUMNS

@cuda.jit
def main():
    # start CPU measurement
    init_ndm = cuda.shared.array(shape=(NDM_ROWS, NDM_COLUMNS), dtype=float32)
    x, y = cuda.grid(2)

start_cpu = time.perf_counter()
