import numpy as np


# objective function
# minimize the sum
def zero_max(x):
    return sum(x)


# tournament selection
def selection(population, scores, k=3):
    # first random selected index
    selection_ix = np.random.randint(len(population))

    for ix in np.random.randint(0, len(population), k - 1):
        # check if better (perform a tournament)
        if scores[ix] < scores[selection_ix]:
            selection_ix = ix
    return population[selection_ix]


# crossover two parents to create two children
def crossover(p1, p2, r_cross):
    # children are copies of parents by default
    c1, c2 = p1.copy(), p2.copy()

    # check for recombination
    if np.random.rand() < r_cross:
        # select crossover point that is not on the end of the string
        pt = np.random.randint(1, len(p1) - 2)

        # perform crossover
        c1 = np.concatenate((p1[:pt], p2[pt:]), axis=0)
        c2 = np.concatenate((p2[:pt], p1[pt:]), axis=0)
    return [c1, c2]


# mutation operator
def mutation(bitstring, prob_of_mutation):
    for i in range(len(bitstring)):
        # check for a mutation
        if np.random.rand() < prob_of_mutation:
            # flip the bit
            bitstring[i] = 1 - bitstring[i]


# genetic algorithm
def genetic_algorithm(objective, number_of_bits, number_of_iterations, population_size, crossover_rate, mutation_rate):
    # initial population of random bitstring - only zeros (0) and ones (1)
    # pop = [randint(0, 2, number_of_bits).tolist() for _ in range(population_size)]
    pop = np.random.randint(0, 2, size=(population_size, number_of_bits))

    # keep track of best solution
    best_individual, best_eval = 0, objective(pop[0])

    # enumerate generations
    for gen in range(number_of_iterations):
        # evaluate all candidates in the population
        scores = [objective(c) for c in pop]

        # check for new best solution
        for i in range(population_size):
            if scores[i] < best_eval:
                best_individual, best_eval = pop[i], scores[i]
                print(f"#{gen}, new best f({pop[i]}) = {scores[i]:.3f}")

        # select parents
        selected = [selection(pop, scores) for _ in range(population_size)]

        # create the next generation
        children = list()
        for i in range(0, population_size, 2):
            # get selected parents in pairs
            p1, p2 = selected[i], selected[i + 1]

            # crossover and mutation
            for c in crossover(p1, p2, crossover_rate):
                # mutation
                mutation(c, mutation_rate)

                # store for next generation
                children.append(c)

        # replace population
        pop = children
    return [best_individual, best_eval]


# define the total iterations
n_iter = 100

# bits
n_bits = 25

# define the population size
n_pop = 100

# crossover rate
crossover_rate = 0.9

# mutation probability
mutation_rate = 1.0 / float(n_bits)

# perform the genetic algorithm search
best_individual, score = genetic_algorithm(zero_max, n_bits, n_iter, n_pop, crossover_rate, mutation_rate)
print('Done!')
print(f'f{best_individual}) = {score}')
