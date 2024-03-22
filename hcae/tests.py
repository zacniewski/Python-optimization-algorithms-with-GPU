import numpy as np
from numpy import asarray, ogrid
from numpy.random import randint, rand

from constants import DATA_SEQUENCE_SIZE, PARAMETERS_SIZE, MUTATION_RATE
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
print(f"{hardcoded_operation_parameters.shape=}")

random_operation_parameters = randint(hardcoded_ndm.shape[0], size=PARAMETERS_SIZE)
print(f"{random_operation_parameters=}")

# checking bounds calculations
bounds = asarray([[-3.0, 3.0], [-5.0, 5.0]])
print(f"{bounds[0]=}")
print(f"{bounds[1]=}")
print(f"{bounds[:, 0]=}")
print(f"{bounds[:, 1]=}")
solution = bounds[:, 0] + rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0])
print(f"{solution=}")


def mutation_of_parameters(params, mutation_rate=MUTATION_RATE):
    random_index = randint(len(params))
    if np.random.rand() < mutation_rate:
        # change the value at random index
        print("Mutation of parameters!")
        params[random_index] = randint(PARAMETERS_SIZE)
    return params


def mutation_of_data_sequence(data_seq, mutation_rate=MUTATION_RATE):
    random_index = randint(len(data_seq[0]))
    print(f"{len(data_seq[0])=}")
    if np.random.rand() < mutation_rate:
        # change the value at random index
        print("Mutation of data sequence!")
        data_seq[0][random_index] = randint(DATA_SEQUENCE_SIZE)
    return data_seq


print(f"{hardcoded_operation_parameters=}")
print(f"Mutated (or not) parameters: {mutation_of_parameters(hardcoded_operation_parameters, mutation_rate=MUTATION_RATE)}")

testing_data_sequence = 2 * np.random.rand(1, DATA_SEQUENCE_SIZE) - 1

print(f"Testing data sequence: {testing_data_sequence}")

print(f"Mutated (or not) data sequence: {mutation_of_data_sequence(testing_data_sequence)}")

X, Y = np.mgrid[-2:2:41j, -2:2:41j]
samples = np.column_stack([X.ravel(), Y.ravel()])
print(samples)
