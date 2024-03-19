import numpy as np

from constants import *
from activation_functions import activation_function, linear, sigmoid


def objective(x, y):
    return np.sin(x*y)


def tournament_selection(population, scores, k=TOURNAMENT_CANDIDATES):
    """
    :param population: population of individuals (chromosomes)
    :param scores:  values of fitness function for every individual from population
    :param k: number of tournament candidates
    :return: chromosome who won the selection
    """

    # first random selected index
    selection_index = np.random.randint(len(population))

    # checking another (k-1) candidates
    for ix in np.random.randint(0, len(population), k - 1):
        # check if better (perform a tournament)
        if scores[ix] < scores[selection_index]:
            selection_index = ix
    return population[selection_index]


def crossover(parent1, parent2, r_cross=CROSSOVER_RATE):
    """
    :param parent1: first parent
    :param parent2: second parent
    :param r_cross: crossover rate
    :return: a pair of children after crossover
    """

    # children are created from parents
    child1, child2 = parent1.copy(), parent2.copy()

    # check for recombination
    if np.random.rand() < r_cross:
        # select crossover point that is not on the end of the chromosome
        pt = np.random.randint(1, len(parent1) - 2)

        # perform crossover
        child1 = np.concatenate((parent1[:pt], parent2[pt:]), axis=0)
        child2 = np.concatenate((parent2[:pt], parent1[pt:]), axis=0)
    return [child1, child2]


def mutation(bitstring, mutation_rate=MUTATION_RATE):
    for i in range(len(bitstring)):
        # check for a mutation
        if np.random.rand() < mutation_rate:
            # flip the bit
            bitstring[i] = 1 - bitstring[i]


def calculate_output_from_ndm(in_ndm: np.array, in_neurons, out_neurons, in_neurons_value) -> float:
    # the column with index '-2' of 'ndm' is a bias column
    # the column with index '-1' of 'ndm' stores the type of activation function for the given neuron,
    # e.g. 'linear' or 'sigmoid'
    z_for_first_neuron = in_neurons_value[0][0] * 1 + in_ndm[0][-2]

    # storage for sigma values, first pair is for the '0' neuron
    # (first neuron, with '1' index on the figures and with '0' index in the NDM)
    sigma = {0: activation_function(input_value=z_for_first_neuron, type_of_neuron_value=in_ndm[0][-1])}

    # calculating output values of neurons and storing them in the 'sigma' dictionary
    z = 0
    for j in range(1, NDM_COLUMNS - 2):
        for i in range(j):
            z = z + in_ndm[i][j] * sigma[i]

        # adding bias
        z = z + in_ndm[j][-2]

        # adding input value (if exists) multiplied by its weight (so far it's 1 by default)
        if j in in_neurons:
            z = z + in_neurons_value[0][j] * 1

        # determine the activation function of 'z' basing on the values of input and 'type of neuron' cell
        sigma[j] = activation_function(input_value=z, type_of_neuron_value=abs(in_ndm[j][-1]))
        z = 0

    print("\nOutputs of activation functions:")
    for key, value in sigma.items():
        print(f"{key}: {value}")

    # Output value from FFN
    out_value = sigma[out_neurons[0]]
    print(f"\nOutput value: {out_value}")
    return out_value


if __name__ == '__main__':
    # initialization - random values from range (-1; 1)
    ndm = 2 * np.random.rand(NDM_ROWS, NDM_COLUMNS) - 1

    # NDM from the article "Neural collision ..." by Tomek
    """hardcoded_ndm = np.array([
        [0, 0.2, 0.3, 0, -0.7, 0.1],
        [-0.9, 0, 1, -0.5, -1, 0.9],
        [0.5, 0, 0, -0.5, 0.3, 0.2],
        [0, 0.3, 0, 0.6, 0.1, 0.5]
    ])

    ndm = hardcoded_ndm
    input_neuron_values = np.array([[1, 1]])"""

    # indexes of input and output neurons
    input_neurons = np.array([[0, 1]])
    output_neurons = np.array([3])

    # random input values in the range (-1; 1)
    input_neuron_values = 2 * np.random.rand(input_neurons.shape[0], input_neurons.shape[1]) - 1
    print(f"Values of input neurons: {input_neuron_values}")

    output_value = calculate_output_from_ndm(
        ndm,
        in_neurons=input_neurons,
        out_neurons=output_neurons,
        in_neurons_value=input_neuron_values
    )

    # Value of the objective function
    objective_value = objective(input_neuron_values[0][0], input_neuron_values[0][1])
    print(f"\nObjective value: {objective_value}")

    # Error value
    print(f"\nError: {np.abs(output_value - objective_value)}")
