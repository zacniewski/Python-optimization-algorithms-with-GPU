import numpy as np
from numba import vectorize, cuda


@vectorize(['float32(float32, float32)'], target='cuda')
def rel_diff(x, y):
    return 2 * (x - y) / (x + y)


a = np.arange(1000, dtype=np.float32)
print(f"{a.shape=}")

b = a * 2 + 1
print(f"{b.shape=}")

c = rel_diff(a, b)
print(f"{c=}")
