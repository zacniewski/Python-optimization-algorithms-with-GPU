import numba
import numpy as np
import time

from tqdm import tqdm

from activation_functions import activation_function
from constants import (
    ACCEPTED_ERROR,
    CROSSOVER_RATE,
    DATA_SEQUENCE_SIZE,
    MUTATION_RATE_PARAMS,
    MUTATION_RATE_PARAMS_ZERO,
    MUTATION_RATE_DATA_SEQ,
    MAX_ITER_NO_PROG,
    NDM_COLUMNS,
    NDM_ROWS,
    TOTAL_NUMBER_OF_ITERATIONS,
    PARAMETERS_SIZE,
    TOURNAMENT_CANDIDATES, POPULATION_SIZE,
)
from hcae_operations import oper2


# first objective is a simple trigonometric function
# the second will be the Ackley's function
@numba.njit
def objective(v):
    xx, yy = v
    return np.sin(xx) * np.cos(yy)


@numba.jit
def tournament_selection(population, scores, k=TOURNAMENT_CANDIDATES):
    """
    :param population: population of individuals (chromosomes)
    :param scores:  values of fitness function for every individual from population
    :param k: number of tournament candidates
    :return: chromosome who won the selection
    """
    print("population=", population)
    print("scores=", scores)

    # first random selected index
    selection_index = np.random.randint(len(population))
    print("selection_index=", selection_index)

    # checking another (k-1) candidates
    for ix in np.random.randint(0, len(population), k - 1):
        # check if better (perform a tournament)
        if scores[ix] < scores[selection_index]:
            selection_index = ix
    return population[selection_index]


@numba.jit
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
    return np.array([child1, child2])


@numba.jit
def mutation_of_parameters(params, mutation_rate=MUTATION_RATE_PARAMS):
    random_index = np.random.randint(params.size)
    if np.random.rand() < mutation_rate:
        # change the value at random index
        # print("Mutation of parameters!")
        b = 5
        params[random_index] = np.abs(params[random_index] + np.random.randint(-b, b))
    # return params


@numba.jit
def mutation_of_data_sequence(d_s, mutation_rate=MUTATION_RATE_DATA_SEQ):
    random_index = np.random.randint(d_s.size)
    if np.random.rand() < mutation_rate:
        # change the value at random index
        # print("Mutation of data sequence!")
        a = 2  # this value could be changed if necessary
        d_s[random_index] = d_s[random_index] + a * (np.random.rand() - 0.5)
        if d_s[random_index] < -1:
            d_s[random_index] = -1
        if d_s[random_index] > 1:
            d_s[random_index] = 1
    # return params


@numba.jit
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
    # it's good for testing purposes!
    # for key, value in sigma.items():
    #     print(f"{key}: {value}")

    # output value from FFN
    out_value = sigma[out_neurons[0][0]]
    return out_value


@numba.jit
def initialize_ndm() -> np.ndarray:
    # initialization of the NDM - random values from range (-1.0; 1.0)
    # init_ndm = 2 * np.random.rand(NDM_ROWS, NDM_COLUMNS) - 1

    # initialization of the NDM - zero values matrix
    init_ndm = np.zeros((NDM_ROWS, NDM_COLUMNS))

    # zero the values under the 1st diagonal
    # https: // numpy.org / doc / stable / reference / generated / numpy.triu.html
    # init_ndm = np.triu(init_ndm, k=1)
    return init_ndm


@numba.jit
def initialize_test_ndm() -> np.ndarray:
    # initialization of the NDM - random values from range (-1.0; 1.0)
    # init_ndm = 2 * np.random.rand(NDM_ROWS, NDM_COLUMNS) - 1

    # initialization of the NDM - zero values matrix
    test_ndm = 2 * np.random.rand(NDM_ROWS, NDM_COLUMNS) - 1

    # zero the values under the 1st diagonal
    # https: // numpy.org / doc / stable / reference / generated / numpy.triu.html
    # init_ndm = np.triu(init_ndm, k=1)
    return test_ndm


@numba.jit
def initialize_params_and_data_seq() -> tuple:
    # initialization of the operations_parameters - random int values from range <0; NDM_ROWS)
    init_oper_params_1 = np.random.randint(0, [2, 3, NDM_ROWS, NDM_ROWS, 2 * DATA_SEQUENCE_SIZE, DATA_SEQUENCE_SIZE])
    init_oper_params_2 = np.random.randint(0, [2, 3, NDM_ROWS, NDM_ROWS, 2 * DATA_SEQUENCE_SIZE, DATA_SEQUENCE_SIZE])

    # initialization of the data_sequence - random values from range (-1.0; 1.0)
    init_data_seq = (2 * np.random.rand(1, DATA_SEQUENCE_SIZE) - 1)[0]
    return init_oper_params_1, init_oper_params_2, init_data_seq


@numba.jit
def calculate_error(current_ndm, samples_values, in_neurons, out_neurons) -> float:
    # values calculated from the NDM
    output_values_for_samples = [calculate_output_from_ndm(
        current_ndm,
        in_neurons,
        out_neurons,
        in_neurons_value=s,
    )
        for s in samples_values]

    # values of the objective function for all samples
    objective_values_for_samples = [objective(s) for s in samples_values]

    # error value
    return sum([np.abs(out - obj) for out, obj in zip(output_values_for_samples, objective_values_for_samples)])
    # return np.sum(np.abs(output_values_for_samples - objective_values_for_samples))


if __name__ == "__main__":
    # draw input function
    # draw_sinus()

    # start CPU measurement
    start_cpu = time.perf_counter()

    # NDM, parameters and data sequence initialization
    initial_ndm = initialize_ndm()

    # create samples of input variables
    # X in <-2; 2> and Y in <-2; 2>
    # X, Y = np.mgrid[-2:2:41j, -2:2:41j]
    # samples = np.column_stack([X.ravel(), Y.ravel()])
    x = np.linspace(-0.8, 0.8, 1601)
    y = np.linspace(-0.8, 0.8, 1601)
    samples = np.column_stack([x, y])

    # indexes of input and output neurons (depends on the task, that author had in mind)
    # two first neurons
    input_neurons = np.array([[0, 1]])
    # last neuron
    output_neurons = np.array([[NDM_ROWS - 1]])

    # calculate initial error
    # the algorithm's task is to minimalize it
    # best_ndm, current_error = initial_ndm, calculate_error(
    #    initial_ndm,
    #    samples,
    #    in_neurons=input_neurons,
    #    out_neurons=output_neurons)
    best_ndm = initial_ndm
    current_error = 2000.0

    # Three copies of the "original" NDM
    best_ndm_for_params_1 = best_ndm.copy()
    best_ndm_for_params_2 = best_ndm.copy()
    best_ndm_for_data_seq = best_ndm.copy()

    # Three "backups" of NDM, used for the replacement of the current NDM, if there's no progress
    # after MAX_ITER_NO_PROG iterations
    # one with the lowest error will be chosen
    backup_ndm_for_params_1 = best_ndm_for_params_1.copy()
    backup_ndm_for_params_2 = best_ndm_for_params_2.copy()
    backup_ndm_for_data_seq = best_ndm_for_data_seq.copy()

    # Three copies of the current error
    # The lowest will be chosen to the next iteration as an indicator
    error_of_best_ndm_for_params_1 = current_error
    error_of_best_ndm_for_data_seq = current_error
    error_of_best_ndm_for_params_2 = current_error

    # to modify NDM one need to invoke the 'oper2' function
    # its arguments are: parameters, data sequence and "current" NDM
    # when storing temporarily best NDM, we need to store also data sequence and parameters related to this NDM
    # cause for example when we'd like to select the best parameters_1 candidates,
    # we need to have the data sequence and best parameters_2 unchanged in this process

    # initial params_1 population
    iterable_params_1 = (np.random.randint(0, [2, 3, NDM_ROWS, NDM_ROWS, DATA_SEQUENCE_SIZE, DATA_SEQUENCE_SIZE]) for _
                         in
                         range(POPULATION_SIZE))
    population_params_1 = np.fromiter(iterable_params_1, dtype='O')

    # initial params_2 population
    iterable_params_2 = (np.random.randint(0, [2, 3, NDM_ROWS, NDM_ROWS, DATA_SEQUENCE_SIZE, DATA_SEQUENCE_SIZE]) for _
                         in
                         range(POPULATION_SIZE))
    population_params_2 = np.fromiter(iterable_params_2, dtype='O')

    # initial data_seq population
    iterable_data_seq = ((2 * np.random.rand(1, DATA_SEQUENCE_SIZE) - 1)[0] for _ in range(POPULATION_SIZE))
    population_data_seq = np.fromiter(iterable_data_seq, dtype='O')

    # initialize "best" params_1, data_seq and params_2
    # initial candidates - two for operations and one for data sequence

    best_op_params_1 = np.random.choice(population_params_1, 1)[0]
    best_op_params_2 = np.random.choice(population_params_2, 1)[0]
    best_data_seq = np.random.choice(population_data_seq, 1)[0]

    # Three backups of the best individuals from every population
    backup_op_params_1 = best_op_params_1.copy()
    backup_op_params_2 = best_op_params_2.copy()
    backup_data_seq = best_data_seq.copy()

    print(f"{best_op_params_1=}")
    print(f"{best_op_params_2=}")
    print(f"{best_data_seq=}")

    print("START!")
    print(f"Initial error = {current_error}")
    number_of_iteration = 0
    iterations_without_progress = 0
    change_in_current_iteration = False

    # The adventure starts here!
    while number_of_iteration < TOTAL_NUMBER_OF_ITERATIONS and ACCEPTED_ERROR < current_error:

        print(f"\n ***** ITERATION #{number_of_iteration + 1} *****")

        # the first step in the algorithm iteration is to evaluate all candidates in the population
        # we need to invoke calculate_error() for every NDM
        # every NDM is changed by given argument only, i.e. params_1 or params_2 or data_sequence
        # every population has size POPULATION_SIZE :)

        # ---- FIRST COMPONENT -----
        # evaluate all candidates in the population (params_1)
        # best_data_seq and best_ndm are constant during evaluating candidates for params_1 population!

        print(f"\n Evaluating parameters_1 in iteration #{number_of_iteration + 1} ...")
        print(f"{best_ndm_for_params_1.sum()=}")
        print(f"{best_ndm_for_params_2.sum()=}")
        print(f"{best_ndm_for_data_seq.sum()=}")

        # ndm_for_params_1 should be the same for every params_1
        # during calculations of error in the given iteration!
        scores_for_params_1 = np.empty(100, dtype='float')
        for i in range(POPULATION_SIZE):
            scores_for_params_1[i] = calculate_error(
                oper2(population_params_1[i], best_data_seq, best_ndm_for_params_1),
                samples,
                in_neurons=input_neurons,
                out_neurons=output_neurons)

        # selecting the best params_1 candidates
        for i in range(POPULATION_SIZE):
            if scores_for_params_1[i] < error_of_best_ndm_for_params_1:
                error_of_best_ndm_for_params_1 = scores_for_params_1[i]
                backup_op_params_1 = population_params_1[i]  # new best params_1
                backup_ndm_for_params_1 = oper2(best_op_params_1, best_data_seq, best_ndm_for_params_1.copy())
                iterations_without_progress = 0
                change_in_current_iteration = True
                print(f"New {error_of_best_ndm_for_params_1= } (for params_1)")

        # ---- SECOND COMPONENT -----
        # evaluate all candidates in the population (data_sequence)
        # best_op_params_1 and best_ndm are constant during evaluating candidates for data_seq population!

        print(f"\n Evaluating data sequence in iteration #{number_of_iteration + 1} ...")
        scores_for_data_seq = [
            calculate_error(
                oper2(best_op_params_1, pop_data_seq, best_ndm_for_data_seq),
                samples,
                in_neurons=input_neurons,
                out_neurons=output_neurons)
            for pop_data_seq in tqdm(population_data_seq)
        ]

        # selecting the best data_seq candidates
        for i in range(POPULATION_SIZE):
            if scores_for_data_seq[i] < error_of_best_ndm_for_data_seq:
                # current_error = scores_for_data_seq[i]
                error_of_best_ndm_for_data_seq = scores_for_data_seq[i]
                backup_data_seq = population_data_seq[i]  # new best data_seq
                backup_ndm_for_data_seq = oper2(best_op_params_1, best_data_seq, best_ndm_for_data_seq.copy())
                iterations_without_progress = 0
                change_in_current_iteration = True
                print(f"New {error_of_best_ndm_for_data_seq= } (for data_seq)")

        # ---- THIRD COMPONENT -----
        # evaluate all candidates in the population (params_2)
        # best_data_seq and best_ndm are constant during evaluating candidates for params_2 population!

        print(f"\n Evaluating parameters_2 in iteration #{number_of_iteration + 1} ...")
        scores_for_params_2 = [
            calculate_error(
                oper2(pop_par_2, best_data_seq, best_ndm_for_params_2),
                # updated NDM after changing operation parameters_1
                samples,
                in_neurons=input_neurons,
                out_neurons=output_neurons)
            for pop_par_2 in tqdm(population_params_2)
        ]

        # selecting the best params_2 candidates
        for i in range(POPULATION_SIZE):
            if scores_for_params_2[i] < error_of_best_ndm_for_params_2:
                # current_error = scores_for_params_2[i]
                error_of_best_ndm_for_params_2 = scores_for_params_2[i]
                backup_op_params_2 = population_params_2[i]  # new best params_2
                backup_ndm_for_params_2 = oper2(best_op_params_1, best_data_seq, best_ndm_for_params_2.copy())
                iterations_without_progress = 0
                change_in_current_iteration = True
                print(f"New {error_of_best_ndm_for_params_2= } (for params_2)")

        # All 3 populations are now evaluated
        current_errors = np.array([
            error_of_best_ndm_for_params_1,
            error_of_best_ndm_for_params_2,
            error_of_best_ndm_for_data_seq
        ])
        current_error = current_errors.min()
        print(f"\n{current_error=} (after evaluations)")

        # select parents
        print("\n Selecting parents from parameters_1 ...")
        print(f"{population_params_1[:3]=}")
        print(f"{tournament_selection(population_params_1[0], scores_for_params_1[0])=}")

        selected_params_1 = [tournament_selection(population_params_1, scores_for_params_1) for _ in
                             tqdm(range(POPULATION_SIZE))]

        print("\n Selecting parents from data sequence ...")
        selected_data_seq = [tournament_selection(population_data_seq, scores_for_data_seq) for _ in
                             tqdm(range(POPULATION_SIZE))]

        print("\n Selecting parents from parameters_2 ...")
        selected_params_2 = [tournament_selection(population_params_2, scores_for_params_2) for _ in
                             tqdm(range(POPULATION_SIZE))]

        # create the next generation
        children_of_params_1 = np.zeros((POPULATION_SIZE, PARAMETERS_SIZE), dtype=int)
        children_of_data_seq = np.zeros((POPULATION_SIZE, DATA_SEQUENCE_SIZE))
        children_of_params_2 = np.zeros((POPULATION_SIZE, PARAMETERS_SIZE), dtype=int)

        for i in range(0, POPULATION_SIZE, 2):
            # get selected parents in pairs
            parent_1, parent_2 = selected_params_1[i], selected_params_1[i + 1]
            parent_3, parent_4 = selected_data_seq[i], selected_data_seq[i + 1]
            parent_5, parent_6 = selected_params_2[i], selected_params_2[i + 1]

            # crossover and mutation for params_1
            for index, c in enumerate(crossover(parent_1, parent_2, CROSSOVER_RATE)):
                # mutation
                mutation_of_parameters(c, MUTATION_RATE_PARAMS)

                # store for next generation
                children_of_params_1[i + index] = c

            # crossover and mutation for data_seq
            for index, d in enumerate(crossover(parent_3, parent_4, CROSSOVER_RATE)):
                # mutation
                mutation_of_data_sequence(d, MUTATION_RATE_DATA_SEQ)

                # store for next generation
                children_of_data_seq[i + index] = d

            # crossover and mutation for params_2
            for index, e in enumerate(crossover(parent_5, parent_6, CROSSOVER_RATE)):
                # mutation
                mutation_of_parameters(e, MUTATION_RATE_PARAMS)

                # store for next generation
                children_of_params_2[i + index] = e

        # replace populations
        population_params_1 = children_of_params_1.copy()
        population_data_seq = children_of_data_seq.copy()
        population_params_2 = children_of_params_2.copy()

        # replacing best individuals
        best_op_params_1 = backup_op_params_1.copy()
        best_op_params_2 = backup_op_params_2.copy()
        best_data_seq = backup_data_seq.copy()

        # checking changes during iteration
        if not change_in_current_iteration:
            iterations_without_progress += 1
            print(f"\nIterations without progress: {iterations_without_progress}.")
        if iterations_without_progress == MAX_ITER_NO_PROG:
            print("\nNO PROGRESS!")
            print("Current best NDM will be used as a starting NDM in the next iteration!")

            # Which NDM gives the smallest error
            where_min_error = np.argmin(current_errors)
            if where_min_error == 0:
                best_ndm_for_params_1 = backup_ndm_for_params_1.copy()
                best_ndm_for_params_2 = backup_ndm_for_params_1.copy()
                best_ndm_for_data_seq = backup_ndm_for_params_1.copy()

            if where_min_error == 1:
                best_ndm_for_params_1 = backup_ndm_for_params_2.copy()
                best_ndm_for_params_2 = backup_ndm_for_params_2.copy()
                best_ndm_for_data_seq = backup_ndm_for_params_2.copy()
            if where_min_error == 2:
                best_ndm_for_params_1 = backup_ndm_for_data_seq.copy()
                best_ndm_for_params_2 = backup_ndm_for_data_seq.copy()
                best_ndm_for_data_seq = backup_ndm_for_data_seq.copy()

            error_of_best_ndm_for_params_1 = current_errors[where_min_error]
            error_of_best_ndm_for_params_2 = current_errors[where_min_error]
            error_of_best_ndm_for_params_1 = current_errors[where_min_error]

            iterations_without_progress = 0

        change_in_current_iteration = False
        number_of_iteration += 1
        # END OF THE WHILE LOOP!

    print(f"\nFinished, the smallest error is: {current_error}")
    end_cpu = time.perf_counter()
    print(f"\nElapsed time: {end_cpu - start_cpu:.3f} seconds.")
