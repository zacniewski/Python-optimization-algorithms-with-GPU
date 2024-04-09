import numpy as np

from activation_functions import activation_function
from constants import (
    CROSSOVER_RATE,
    DATA_SEQUENCE_SIZE,
    MUTATION_RATE,
    NDM_COLUMNS,
    NDM_ROWS,
    NUMBER_OF_ITERATIONS,
    PARAMETERS_SIZE,
    TOURNAMENT_CANDIDATES, POPULATION_SIZE,
)
from hcae.hcae_operations import oper2


# first objective is a simple trigonometric function
# the second will be the Ackley's function
def objective(v):
    x, y = v
    return np.sin(x * y)


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


def calculate_output_from_ndm(
        in_ndm: np.array,
        in_neurons: np.array,
        out_neurons: np.array,
        in_neurons_value: np.array,
) -> float:
    """
    :param in_ndm: input NDM matrix
    :param in_neurons: numbers of input neurons (starting from zero, first neuron has '0' index, etc., like in the 0-indexed array)
    :param out_neurons: numbers of output neurons
    :param in_neurons_value: values of the input neurons, e.g. 0.7
    """
    # the column with index '-2' of 'ndm' is a bias column
    # the column with index '-1' of 'ndm' stores the type of activation function for the given neuron,
    # e.g. 'linear' or 'sigmoid'
    z_for_first_neuron = in_neurons_value[0] * 1 + in_ndm[0][-2]

    # storage for sigma values, first pair is for the '0' neuron
    # (first neuron, with '1' index on the figures and with '0' index in the NDM)
    sigma = {
        0: activation_function(
            input_value=z_for_first_neuron, type_of_neuron_value=in_ndm[0][-1]
        )
    }

    # calculating output values of neurons and storing them in the 'sigma' dictionary
    z = 0
    for j in range(1, in_ndm.shape[1] - 2):
        for i in range(j):
            z = z + in_ndm[i][j] * sigma[i]

        # adding bias
        z = z + in_ndm[j][-2]

        # adding input value (if exists) multiplied by its weight (so far it's 1 by default)
        if j in in_neurons:
            z = z + in_neurons_value[j] * 1

        # determine the activation function of 'z' basing on the values of input and 'type of neuron' cell
        sigma[j] = activation_function(
            input_value=z, type_of_neuron_value=abs(in_ndm[j][-1])
        )
        z = 0

    # uncomment to check the outputs value of all neurons (after activation function)
    # for key, value in sigma.items():
    # print(f"{key}: {value}")

    # output value from FFN
    out_value = sigma[out_neurons[0][0]]
    return out_value


def initialize_all() -> tuple:
    # initialization of the NDM - random values from range (-1.0; 1.0)
    init_ndm = 2 * np.random.rand(NDM_ROWS, NDM_COLUMNS) - 1

    # zero the values under the 1st diagonale
    # https: // numpy.org / doc / stable / reference / generated / numpy.triu.html
    init_ndm = np.triu(init_ndm, k=1)

    # initialization of the operations_parameters - random int values from range <0; NDM_ROWS)
    init_oper_params_1 = np.random.randint(init_ndm.shape[0], size=PARAMETERS_SIZE)
    init_oper_params_2 = np.random.randint(init_ndm.shape[0], size=PARAMETERS_SIZE)

    # initialization of the data_sequence - random values from range (-1.0; 1.0)
    init_data_seq = (2 * np.random.rand(1, DATA_SEQUENCE_SIZE) - 1)[0]
    print(f"{init_data_seq=}")

    return init_ndm, init_oper_params_1, init_oper_params_2, init_data_seq


def calculate_error(current_ndm, samples_values, in_neurons, out_neurons):
    # output values from NDM for all input samples
    iterable1 = (
        calculate_output_from_ndm(
            current_ndm,
            in_neurons,
            out_neurons,
            in_neurons_value=s,
        )
        for s in samples_values
    )
    output_values_for_samples = np.fromiter(iterable1, dtype=np.dtype(list))
    # print(f"\n{output_values_for_samples[0:10]=}")

    # values of the objective function for all samples
    iterable2 = (objective(s) for s in samples_values)
    objective_values_for_samples = np.fromiter(iterable2, dtype=np.dtype(list))
    # print(f"{objective_values_for_samples[0: 10]=}")

    # error value
    return np.sum(np.abs(output_values_for_samples[0:10] - objective_values_for_samples[0:10]))


if __name__ == "__main__":
    (
        initial_ndm,
        initial_operation_parameters_1,
        initial_operation_parameters_2,
        initial_data_sequence,
    ) = initialize_all()

    # initial candidates - two for operations and one for data sequence
    # print(f"{initial_operation_parameters_1=}")
    # print(f"{initial_operation_parameters_2=}")
    # print(f"{initial_data_sequence[-9:]=}")

    # create samples of input variables
    # X in <-2; 2> and Y in <-2; 2>
    X, Y = np.mgrid[-2:2:41j, -2:2:41j]
    samples = np.column_stack([X.ravel(), Y.ravel()])
    # print(f"{samples.shape=}")

    # indexes of input and output neurons (depends on the task, that author had in mind)
    input_neurons = np.array([[0, 1]])
    output_neurons = np.array([[3]])

    # calculate initial error
    # the algorithm's task is to minimalize it
    best_ndm, minimal_error = initial_ndm, calculate_error(
        initial_ndm,
        samples,
        in_neurons=input_neurons,
        out_neurons=output_neurons)
    best_op_params_1, best_op_params_2, best_data_seq = (initial_operation_parameters_1,
                                                         initial_operation_parameters_2,
                                                         initial_data_sequence)
    print(f"Initial error = {minimal_error}")

    # to modify NDM one need to invoke the 'oper2' function
    # its arguments are: parameters, data sequence and "current" NDM
    # when storing temporarily best NDM, we need to store also data sequence and parameters related to this NDM
    # cause for example when we'd like to select the best parameters candidate(s),
    # we need to have the data sequence unchanged in this process

    iterable_params_1 = (np.random.randint(best_ndm.shape[0], size=PARAMETERS_SIZE) for _ in
                         range(POPULATION_SIZE))
    population_params_1 = np.fromiter(iterable_params_1, dtype=np.dtype(list))
    print(f"{population_params_1.shape=}")

    iterable_params_2 = (np.random.randint(best_ndm.shape[0], size=PARAMETERS_SIZE) for _ in
                         range(POPULATION_SIZE))
    population_params_2 = np.fromiter(iterable_params_2, dtype=np.dtype(list))
    print(f"{population_params_2.shape=}")

    iterable_data_seq = ((2 * np.random.rand(1, DATA_SEQUENCE_SIZE) - 1)[0] for _ in range(3))
    population_data_seq = np.fromiter(iterable_data_seq, dtype=np.dtype(list))
    print(f"{population_data_seq[0].shape=}")

    for gen in range(NUMBER_OF_ITERATIONS):
        # the first step in the algorithm iteration is to evaluate all candidate solutions
        print(f"--- Iteration {gen} ---")
        ndm_after_oper2 = oper2(
            random_operation_parameters,
            testing_data_sequence,
            hardcoded_ndm
        )


