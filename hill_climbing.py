from numpy.random import randn, rand


# hill climbing local search algorithm
def hill_climbing(objective, bounds, n_iterations, step_size):
    """
    :param objective: objective function is specific to the problem domain
    :param bounds: function is constraint to a specific range
    :param n_iterations: number of iterations
    :param step_size:  size of the algorithm iteration's step, relative to the bounds of the search space

    :return: list with:
        solution - the best point in the search space
        solution_evaluation - the value for the 'solution' point
        scores - list with the consecutive improvements of solution
    """

    # generate an initial point
    solution = bounds[:, 0] + rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0])

    # evaluate the initial point
    solution_evaluation = objective(solution)

    # keep points in the 'solutions' list
    solutions = list()
    solutions.append(solution)

    # keep evaluations in the 'scores' list
    scores = list()
    scores.append(solution_evaluation)

    # run the hill climbing algorithm
    for i in range(n_iterations):
        # take a step
        candidate = solution + randn(len(bounds)) * step_size

        # evaluate candidate point
        candidate_evaluation = objective(candidate)

        # check if we should keep the new point
        if candidate_evaluation <= solution_evaluation:
            # store the new point
            solution, solution_evaluation = candidate, candidate_evaluation

            # keep track of points and scores
            solutions.append(solution)
            scores.append(solution_evaluation)

            # report progress
            print(f"#{i} f({solution}) = {solution_evaluation:.5f}")
    return [solution, solution_evaluation, solutions, scores]
