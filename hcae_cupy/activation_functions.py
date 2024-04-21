import cupy as cp


def sigmoid(z):
    return 1.0 / (1.0 + cp.exp(-z))


def relu(z):
    return cp.maximum(0, z)


def linear(z):
    return z


def tanh(z):
    return cp.tanh(z)


def activation_function(input_value, type_of_neuron_value):
    if abs(type_of_neuron_value) <= 0.5:
        return sigmoid(input_value)
    else:
        return linear(input_value)
