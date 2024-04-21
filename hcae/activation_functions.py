import numpy as np


def linear(z):
    return z


def tanh(z):
    return np.tanh(z)


def activation_function(input_value, type_of_neuron_value):
    if abs(type_of_neuron_value) <= 0.5:
        return tanh(input_value)
    else:
        return linear(input_value)
