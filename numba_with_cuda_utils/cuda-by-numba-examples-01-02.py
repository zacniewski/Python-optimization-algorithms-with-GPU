from numba import cuda
import numpy as np


@cuda.jit
def add_scalars(a, b, c):
    c[0] = a + b


dev_c = cuda.device_array((1,), np.float32)

add_scalars[1, 1](2.0, 7.0, dev_c)

c = dev_c.copy_to_host()
print(f"2.0 + 7.0 = {c[0]}")
#  2.0 + 7.0 = 9.0
