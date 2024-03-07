import numpy as np

from constants import *
from activation_functions import activation_type, linear, sigmoid


def objective(x):
    return sum(x)


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


def mutation(bitstring, prob_of_mutation):
    for i in range(len(bitstring)):
        # check for a mutation
        if np.random.rand() < prob_of_mutation:
            # flip the bit
            bitstring[i] = 1 - bitstring[i]


if __name__ == '__main__':
    # random values from range (-1; 1)
    ndm = 2 * np.random.rand(NDM_ROWS, NDM_COLUMNS) - 1

    # indexes of input and output neurons
    input_neurons = np.array([0, 1])
    output_neurons = np.array([3])

    # random input values
    input_neuron_values = 2 * np.random.rand(input_neurons.shape[0], input_neurons.shape[1]) - 1

    # storage for sigma values, first pair is for the '0' neuron
    z_for_neuron_zero = input_neuron_values[0] + ndm[0][-2]
    sigma = {0: activation_type(z_for_neuron_zero, ndm[0][-1])}

    # calculating output values of neurons and storing them in 'sigma' dictionary
    z = 0
    for j in range(NDM_COLUMNS - 2):
        for i in range(j):
            z = z + ndm[j][i]
        z = z + ndm[j][-2]

        # determine the activation function of 'z' basing on the value of 'type of neuron' cell
        sigma[j] = activation_type(z, abs(ndm[j][-1]))
        if abs(ndm[j][-1]) <= 0.5:
            sigma[j] = sigmoid(z)
        else:
            sigma[j] = linear(z)

    print(f"{ndm=}")
