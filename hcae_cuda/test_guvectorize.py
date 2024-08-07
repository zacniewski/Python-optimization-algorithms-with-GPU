# Very simple example:
import numpy as np
from numba import guvectorize, int64, float64


@guvectorize([(int64[:], int64[:], int64[:])], '(n),()->(n)')
def g(x, y, res):
    z_for_first_neuron = x[0] * 1 + y[0]
    for i in range(x.shape[0]):
        res[i] = x[i] + y[0]


# if using a scalar 'y' it must be declared with 'int64'
# if using an array it must be declared with 'int64[:]'
@guvectorize([(int64[:], int64, int64[:])], '(n),()->(n)', target='cuda')
def g2(x, y, res):
    for i in range(x.shape[0]):
        res[i] = x[i] + y


a = np.arange(6)
print(g(a, 10))

b = np.arange(6).reshape(2, 3)
print(g(b, 10))

print(g2(b, np.array([10, 20])))
