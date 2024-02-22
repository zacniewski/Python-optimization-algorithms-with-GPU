# hill climbing search with random restarts of the ackley objective function
from numpy import asarray, cos, e, exp, sqrt, pi
from numpy.random import rand
from numpy.random import seed

from utils import hill_climbing_with_starting_point, in_bounds


# objective function
def objective(v):
    x, y = v
    return (
        -20.0 * exp(-0.2 * sqrt(0.5 * (x**2 + y**2)))
        - exp(0.5 * (cos(2 * pi * x) + cos(2 * pi * y)))
        + e
        + 20
    )


# hill climbing with random restarts algorithm
def random_restarts(objective, bounds, n_iter, step_size, n_restarts):
    best, best_eval = None, 1e10
    # enumerate restarts
    for n in range(n_restarts):
        # generate a random initial point for the search
        start_pt = None
        while start_pt is None or not in_bounds(start_pt, bounds):
            start_pt = bounds[:, 0] + rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0])
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
step_size = 0.05

# total number of random restarts
n_restarts = 30

# perform the hill climbing search
best, score = random_restarts(objective, bounds, n_iter, step_size, n_restarts)
print("Done!")
print(f"f({best}) = {score}")
