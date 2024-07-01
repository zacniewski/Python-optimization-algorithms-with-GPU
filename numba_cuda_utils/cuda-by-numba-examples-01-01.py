import numpy as np
import numba
from numba import cuda

print(f"Numpy's version: {np.__version__}")
print(f"Numba's version: {numba.__version__}")

print("CUDA detection:")
cuda.detect()

#
# Found 1 CUDA devices
# id 0             b'Tesla T4'                              [SUPPORTED]
#                       Compute Capability: 7.5
#                            PCI Device ID: 4
#                               PCI Bus ID: 0
#                                     UUID: GPU-e0b8547a-62e9-2ea2-44f6-9cd43bf7472d
#                                 Watchdog: Disabled
#              FP32/FP64 Performance Ratio: 32
# Summary:
# 	1/1 devices are supported
