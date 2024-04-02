import numpy as np
import numba as nb

@nb.njit(fastmath=True)
def nb_dot(mat, q):
  return np.dot(mat, q)

@nb.njit(fastmath=True)
def nb_op(mat, q):
  return mat @ q

mat = np.array([[  2.44 ,  -0.01 , -74.526],
                [  0.578,   0.873, -86.261],
                [  0.003,  -0.   ,   1.   ]])
q = np.array([100., 200, 1])

print("NumPy result:", np.matmul(mat, q))
print("Numba result (ver. 1):", nb_dot(mat, q))
print("Numba result (ver. 2):", nb_op(mat, q))
