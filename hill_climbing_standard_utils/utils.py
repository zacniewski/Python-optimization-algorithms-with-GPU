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


# hill climbing local search algorithm (version with bounds)
def hill_climbing_with_bounds(objective, bounds, n_iterations, step_size):
    # generate an initial point
    solution = None
    while solution is None or not in_bounds(solution, bounds):
        solution = bounds[:, 0] + rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0])

    # evaluate the initial point
    solution_eval = objective(solution)

    # run the hill climb
    for i in range(n_iterations):
        # take a step
        candidate = None
        while candidate is None or not in_bounds(candidate, bounds):
            candidate = solution + randn(len(bounds)) * step_size

        # evaluate candidate point
        candidate_eval = objective(candidate)

        # check if we should keep the new point
        if candidate_eval <= solution_eval:
            # store the new point
            solution, solution_eval = candidate, candidate_eval

            # report progress
            print(f"#{i} f({solution}) = {solution_eval:.5f}")
    return [solution, solution_eval]


# hill climbing local search algorithm (with random restarts)
def hill_climbing_with_starting_point(
    objective, bounds, n_iterations, step_size, start_pt
):
    # store the initial point
    solution = start_pt

    # evaluate the initial point
    solution_eval = objective(solution)

    # run the hill climb
    for i in range(n_iterations):
        # take a step
        candidate = None
        while candidate is None or not in_bounds(candidate, bounds):
            candidate = solution + randn(len(bounds)) * step_size

        # evaluate candidate point
        candidate_eval = objective(candidate)

        # check if we should keep the new point
        if candidate_eval <= solution_eval:
            # store the new point
            solution, solution_eval = candidate, candidate_eval
    return [solution, solution_eval]


# check if a point is within the bounds of the search
def in_bounds(point, bounds):
    # enumerate all dimensions of the point
    for d in range(len(bounds)):
        # check if out of bounds for this dimension
        if point[d] < bounds[d, 0] or point[d] > bounds[d, 1]:
            return False
    return True
