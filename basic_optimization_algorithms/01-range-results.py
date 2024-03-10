from matplotlib import pyplot
from numpy import arange


# objective function
def objective(x):
    return x ** 2


# range for input
r_min, r_max = -5.0, 5.0

# sample input range uniformly at 0.1 increments
inputs = arange(r_min, r_max, 0.1)

# summarize some of the input domain
print("Some inputs: ", inputs[:5])

# compute targets
results = objective(inputs)

# summarize some of the results
print("Some results: ", results[:5])

# create a mapping of some inputs to some results
for i in range(5):
    print(f"f({inputs[i]:.3f}) = {results[i]:.3f}")

# create a line plot of input vs result
pyplot.plot(inputs, results)

# define the known function optima
optima_x = 0.0
# optima_y = objective(optima_x)

# draw the function optima as a red square
# pyplot.plot([optima_x], [optima_y], 's', color='r')

# draw a vertical line at the optimal input
pyplot.axvline(x=optima_x, ls='--', color='red')

# show the plot
pyplot.show()
