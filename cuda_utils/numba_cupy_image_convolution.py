"""
In this example, all three libraries are used to perform image convolution using a 3x3 kernel.
NumPy utilizes the convolve2d function from scipy.signal,
cuPy provides a GPU-accelerated version of convolve2d,
and Numba compiles the convolution function using JIT compilation.
"""

import numpy as np
import cupy as cp
import numba as nb
from scipy.signal import convolve2d

# NumPy code
image_np = np.random.rand(512, 512)
kernel_np = np.ones((3, 3))
result_np = convolve2d(image_np, kernel_np, mode='same')

# cuPy code
image_cp = cp.asarray(image_np)
kernel_cp = cp.ones((3, 3))
result_cp = cp.convolve2d(image_cp, kernel_cp, mode='same').get()

# Numba code
@nb.njit
def image_convolution(image, kernel):
    return convolve2d(image, kernel, mode='same')

result_nb = image_convolution(image_np, kernel_np)

print("NumPy result:", result_np)
print("cuPy result:", result_cp)
print("Numba result:", result_nb)
