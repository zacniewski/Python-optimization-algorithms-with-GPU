import numba
import numpy as np


@numba.jit
def linear(z):
    return z


@numba.jit
def tanh(z):
    return np.tanh(z)


@numba.jit
def activation_function(input_value, type_of_neuron_value):
    if abs(type_of_neuron_value) < 0.5:
        return tanh(input_value)
    else:
        return linear(input_value)
