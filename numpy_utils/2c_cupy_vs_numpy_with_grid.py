import math

import cupy as cp
import numpy as np


def vector_add(A, B, C, size):
    for item in range(0, size):
        C[item] = A[item] + B[item]

# size of the vectors
size = 1024

# allocating and populating the vectors
a_gpu = cp.random.rand(size, dtype=cp.float32)
b_gpu = cp.random.rand(size, dtype=cp.float32)
c_gpu = cp.zeros(size, dtype=cp.float32)

# CUDA version of vector_add
kernel_add_cuda = r'''
extern "C"
__global__ void vector_add(const float * A, const float * B, float * C, const int size)
{
   int item = (blockIdx.x * blockDim.x) + threadIdx.x;
   if ( item < size )
   {
      C[item] = A[item] + B[item];
   }
}
'''
vector_add_gpu = cp.RawKernel(kernel_add_cuda, "vector_add")
threads_per_block = 1024
grid_size = (int(math.ceil(size / threads_per_block)), 1, 1)
block_size = (threads_per_block, 1, 1)

vector_add_gpu(grid_size, block_size, (a_gpu, b_gpu, c_gpu, size))

a_cpu = cp.asnumpy(a_gpu)
b_cpu = cp.asnumpy(b_gpu)
c_cpu = np.zeros(size, dtype=np.float32)

vector_add(a_cpu, b_cpu, c_cpu, size)

# test
print(f"{c_cpu=}")
print(f"{cp.asnumpy(c_gpu)=}")
print(f"{c_gpu.device=}")
print((c_cpu-cp.asnumpy(c_gpu)).sum())
