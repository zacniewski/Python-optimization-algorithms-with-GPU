# from __future__ import division
from numba import cuda
import numpy
import math


# CUDA kernel
@cuda.jit
def my_kernel(io_array):
    """
    Return the absolute position of the current thread in the entire grid of blocks.
    *ndim* should correspond to the number of dimensions declared when instantiating the kernel.
    If *ndim* is 1, a single integer is returned.
    If *ndim* is 2 or 3, a tuple of the given number of integers is returned.
    Computation of the first integer is as follows::  cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    and is similar for the other two indices, but using the `y` and `z` attributes.
    """
    pos = cuda.grid(1)
    if pos < io_array.size:
        io_array[pos] *= 2  # do the computation


# Host code
data = numpy.ones(256)
threadsperblock = 256
blockspergrid = math.ceil(data.shape[0] / threadsperblock)
my_kernel[blockspergrid, threadsperblock](data)
print(data)
