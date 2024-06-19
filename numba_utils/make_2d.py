from numba import njit
import numpy as np

@njit
def make_2d(arraylist):
    """
    :param arraylist: list of Numpy's arrays
    :return: 2D Numpy's array with particular input arrays
    Link: https://numba.discourse.group/t/passing-a-list-of-numpy-arrays-into-np-array-with-numba/278
    """
    n = len(arraylist)
    k = arraylist[0].shape[0]
    a2d = np.zeros((n, k))
    for i in range(n):
        a2d[i] = arraylist[i]
    return(a2d)

a = np.array((0, 1, 2, 3))
b = np.array((4, 5, 6, 7))
c = np.array((9, 10, 11, 12))

d = make_2d([a, b, c])

print(f"{d=}")
print(f"{d.shape=}")
print(f"{type(d)=}")
