# iterated local search of the ackley objective function
from numpy import asarray, cos, e, exp, pi, sqrt
from numpy.random import rand, randn, seed

from hill_climbing_standard_utils.utils import hill_climbing_with_starting_point, in_bounds


# objective function
def objective(v):
    x, y = v
    return (
        -20.0 * exp(-0.2 * sqrt(0.5 * (x**2 + y**2)))
        - exp(0.5 * (cos(2 * pi * x) + cos(2 * pi * y)))
        + e
        + 20
    )


# iterated local search algorithm
def iterated_local_search(objective, bounds, n_iter, step_size, n_restarts, p_size):
    # define starting point
    best = None
    while best is None or not in_bounds(best, bounds):
        best = bounds[:, 0] + rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0])

    # evaluate current best point
    best_eval = objective(best)

    # enumerate restarts
    for n in range(n_restarts):
        # generate an initial point as a perturbed version of the last best
        start_pt = None
        while start_pt is None or not in_bounds(start_pt, bounds):
            start_pt = best + randn(len(bounds)) * p_size

        # perform a stochastic hill climbing search
        solution, solution_eval = hill_climbing_with_starting_point(
            objective, bounds, n_iter, step_size, start_pt
        )

        # check for new best
        if solution_eval < best_eval:
            best, best_eval = solution, solution_eval
            print(f"Restart #{n}, best: f({best}) = {best_eval:.5f}")
    return [best, best_eval]


# seed the pseudorandom number generator
seed(1)

# define range for input
bounds = asarray([[-5.0, 5.0], [-5.0, 5.0]])

# define the total iterations
n_iter = 1000

# define the maximum step size
s_size = 0.05

# total number of random restarts
n_restarts = 30

# perturbation step size
p_size = 1.0

# perform the hill climbing search
best, score = iterated_local_search(
    objective, bounds, n_iter, s_size, n_restarts, p_size
)
print("Done!")
print(f"f({best}) = {score}")
