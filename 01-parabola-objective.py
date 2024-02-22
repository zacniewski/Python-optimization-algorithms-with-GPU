import matplotlib.pyplot as plt
from numpy import asarray, arange
from numpy.random import seed

from utils import hill_climbing


# objective function
def objective(x):
    return x[0] ** 2.0 + x[0] - 4


# seed the pseudorandom number generator to get the same results at every run
seed(5)

# define range for input
bounds = asarray([[-5.0, 5.0]])

# define the total iterations number
n_iterations = 1000

# define the maximum step size
step_size = 0.1

# start the hill climbing search
best_point, value_for_best_point, solutions, scores = hill_climbing(
    objective, bounds, n_iterations, step_size
)
print("Finished!")
print(f"f({best_point}) = {value_for_best_point}")

# PLOTS
# sample input range uniformly at 0.1 increments
inputs = arange(bounds[0, 0], bounds[0, 1], 0.1)

# create a line plot of input vs result
plt.figure(1)
plt.plot(inputs, [objective([x]) for x in inputs], "--")
plt.xlabel("Input points")
plt.ylabel("Solution evaluation f(x)")

# draw a vertical line at the optimal input
plt.axvline(x=best_point, ls="--", color="red")

# save the figure
plt.savefig("figures/01-parabola-objective.png")

# plot the sample as black circles
plt.plot(solutions, [objective(x) for x in solutions], "o", color="black")

# line plot of best scores
plt.figure(2)
plt.plot(scores, ".-")
plt.xlabel("Improvement number")
plt.ylabel("Solution evaluation f(x)")
plt.savefig("figures/01-parabola-improvement-number.png")

# show the plot
plt.show()
