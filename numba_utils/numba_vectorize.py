import numpy as np
from numba import vectorize, int64


@vectorize([int64(int64, int64)])
def vec_add(x, y):
    return x + y


a = np.arange(6, dtype=np.int64)
b = np.linspace(0, 10, 6, dtype=np.int64)
print(vec_add(a, a))
print(vec_add(b, b))
