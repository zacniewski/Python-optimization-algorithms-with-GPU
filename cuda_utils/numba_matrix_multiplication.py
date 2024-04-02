import numpy as np
import numba as nb

# NumPy code
a_np = np.random.rand(1000, 1000)
b_np = np.random.rand(1000, 1000)
result_np = np.matmul(a_np, b_np)

# Numba code
@nb.njit(parallel=True)
def matrix_multiply(a, b):
    return np.matmul(a, b)

result_nb = matrix_multiply(a_np, b_np)

print("NumPy result:", result_np)
print("Numba result:", result_nb)
