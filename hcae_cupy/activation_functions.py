import numpy as np


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def relu(z):
    return np.maximum(0, z)


def linear(z):
    return z


def tanh(z):
    return np.tanh(z)


def activation_function(input_value, type_of_neuron_value):
    if abs(type_of_neuron_value) <= 0.5:
        return sigmoid(input_value)
    else:
        return linear(input_value)
