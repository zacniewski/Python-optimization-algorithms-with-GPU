from numba import cuda

# prints general information about GPU (if found)
print(cuda.gpus)

# prints more detailed info
cuda.detect()