import numpy as np
import cupy as cp

# NumPy code
a_np = np.array([1, 2, 3, 4, 5])
result_np = np.square(a_np)

# cuPy code
a_cp = cp.array([1, 2, 3, 4, 5])
result_cp = cp.square(a_cp)

# Transfer the result back to NumPy for comparison
result_cp_np = cp.asnumpy(result_cp)

print("NumPy result:", result_np)
print("cuPy result:", result_cp_np)
