# Need to: pip install --upgrade cuda-python
# Docs: https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__DEVICE.html

from cuda.cuda import CUdevice_attribute, cuDeviceGetAttribute, cuDeviceGetName, cuInit

# Initialize CUDA Driver API
(err,) = cuInit(0)

# Get attributes
err, DEVICE_NAME = cuDeviceGetName(128, 0)
DEVICE_NAME = DEVICE_NAME.decode("ascii").replace("\x00", "")

err, MAX_THREADS_PER_BLOCK = cuDeviceGetAttribute(
    CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, 0
)
err, MAX_BLOCK_DIM_X = cuDeviceGetAttribute(
    CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X, 0
)
err, MAX_GRID_DIM_X = cuDeviceGetAttribute(
    CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X, 0
)
err, SMs = cuDeviceGetAttribute(
    CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, 0
)
err, MAX_SHARED_MEMORY_PER_BLOCK = cuDeviceGetAttribute(
    CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK, 0
)

print(f"Device Name: {DEVICE_NAME}")
print(f"Maximum number of multiprocessors: {SMs}")
print(f"Maximum number of threads per block: {MAX_THREADS_PER_BLOCK:10}")
print(f"Maximum x-dimension of a block:   {MAX_BLOCK_DIM_X:10}")
print(f"Maximum x-dimension of a grid:  {MAX_GRID_DIM_X:10}")
print(f"Maximum amount of shared memory available to a thread block in bytes:  {MAX_SHARED_MEMORY_PER_BLOCK:10}")

#  Device Name: NVIDIA TITAN Xp
#  Maximum number of multiprocessors: 30
#  Maximum number of threads per block:       1024
#  Maximum number of blocks per grid:         1024
#  Maximum number of threads per grid:  2147483647
#  Maximum amount of shared memory available to a thread block in bytes:  49152
