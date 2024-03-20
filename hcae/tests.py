import numpy as np
from numpy.random import randint

from constants import PARAMETERS_SIZE
from hcae_operations import fill, oper2

# NumPy's randint: https://numpy.org/doc/stable/reference/random/generated/numpy.random.randint.html

# NDM from the article "Neural collision ..." by Tomek
hardcoded_ndm = np.array([
    [0, 0.2, 0.3, 0, -0.7, 0.1],
    [-0.9, 0, 1, -0.5, -1, 0.9],
    [0.5, 0, 0, -0.5, 0.3, 0.2],
    [0, 0.3, 0, 0.6, 0.1, 0.5]
])

hardcoded_input_neuron_values = np.array([[1, 1]])

hardcoded_operation_parameters = np.array([1, 2, 3, 1, 2, 3])

print(hardcoded_operation_parameters.shape)
random_operation_parameters = randint(hardcoded_ndm.shape[0], size=PARAMETERS_SIZE)
print(random_operation_parameters)