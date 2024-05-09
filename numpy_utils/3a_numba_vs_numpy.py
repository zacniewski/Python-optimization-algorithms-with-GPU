import math
import time

from numba import jit
import numpy as np

x = np.arange(1_000_000).reshape(1000, 1000)


def go_python(a):
    trace = 0.0
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            trace += math.tanh(a[i, j])
    return a + trace


def go_numpy(a):
    trace = 0.0
    iter1 = (np.tanh(a[i, j]) for i in range(a.shape[0]) for j in range(a.shape[1]))
    trace += np.fromiter(iter1, dtype=a.dtype).sum()
    return a + trace


@jit(nopython=True)
def go_numba(a):  # Function is compiled and runs in machine code
    trace = 0.0
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            trace += np.tanh(a[i, j])
    return a + trace


if __name__ == '__main__':
    # ONLY PYTHON
    start = time.perf_counter()
    go_python(x)
    end = time.perf_counter()
    print(f"Elapsed time (only Python): {end - start:.6f} seconds.")

    # ONLY NUMPY
    start = time.perf_counter()
    go_numpy(x)
    end = time.perf_counter()
    print(f"Elapsed time (only Numpy): {end - start:.6f} seconds.")

    # DO NOT REPORT THIS FOR NUMBA... COMPILATION TIME IS INCLUDED IN THE EXECUTION TIME!
    start = time.perf_counter()
    go_numba(x)
    end = time.perf_counter()
    print(f"Elapsed time (with Numba compilation): {end - start:.6f} seconds.")

    # NOW THE FUNCTION IS COMPILED, RE-TIME IT EXECUTING FROM CACHE
    start = time.time()
    go_numba(x)
    end = time.time()
    print(f"Elapsed time (after Numba compilation): {end - start:.6f} seconds.")
