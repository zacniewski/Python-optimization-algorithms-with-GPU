# hill climbing search of the Ackley objective function
from numpy import asarray, cos, e, exp, pi, sqrt
from numpy.random import seed

from hill_climbing_standard_utils.utils import hill_climbing_with_bounds


# objective function
def objective(v):
    x, y = v
    return (
        -20.0 * exp(-0.2 * sqrt(0.5 * (x**2 + y**2)))
        - exp(0.5 * (cos(2 * pi * x) + cos(2 * pi * y)))
        + e
        + 20
    )


# seed the pseudorandom number generator
seed(1)

# define range for input
bounds = asarray([[-5.0, 5.0], [-5.0, 5.0]])

# define the total iterations
n_iterations = 1000

# define the maximum step size
step_size = 0.05

# perform the hill climbing search (with bounds checking)
best, score = hill_climbing_with_bounds(objective, bounds, n_iterations, step_size)
print("Done!")
print(f"f({best}) = {score}")
