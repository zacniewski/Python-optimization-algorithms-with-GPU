import numba
import numpy as np


@numba.jit
def activation_function(input_value, type_of_neuron_value):
    if abs(type_of_neuron_value) < 0.5:
        return np.tanh(input_value)
    else:
        return input_value


@numba.jit
def calculate_output_from_ndm(
        in_ndm: np.array,
        in_neurons: np.array,
        out_neurons: np.array,
        in_neurons_value: np.array,
) -> float:
    """
    :param in_ndm: input NDM matrix
    :param in_neurons: numbers of input neurons (starting from zero, first neuron has '0' index, etc., like in the 0-indexed array)
    :param out_neurons: numbers of output neurons
    :param in_neurons_value: values of the input neurons, e.g. 0.7
    """
    # the column with index '-2' of 'ndm' is a bias column
    # the column with index '-1' of 'ndm' stores the type of activation function for the given neuron,
    # e.g. 'linear' or 'sigmoid'
    z_for_first_neuron = in_neurons_value[0] * 1 + in_ndm[0][-2]

    # storage for sigma values, first pair is for the '0' neuron
    # (first neuron, with '1' index on the figures and with '0' index in the NDM)
    sigma = {
        0: activation_function(
            input_value=z_for_first_neuron, type_of_neuron_value=in_ndm[0][-1]
        )
    }

    # calculating output values of neurons and storing them in the 'sigma' dictionary
    z = 0
    for j in range(1, in_ndm.shape[1] - 2):
        for i in range(j):
            z = z + in_ndm[i][j] * sigma[i]

        # adding bias
        z = z + in_ndm[j][-2]

        # adding input value (if exists) multiplied by its weight (so far it's 1 by default)
        if j in in_neurons:
            z = z + in_neurons_value[j] * 1

        # determine the activation function of 'z' basing on the values of input and 'type of neuron' cell
        sigma[j] = activation_function(
            input_value=z, type_of_neuron_value=abs(in_ndm[j][-1])
        )
        z = 0

    # uncomment to check the outputs value of all neurons (after activation function)
    # it's good for testing purposes!
    # for key, value in sigma.items():
    #     print(f"{key}: {value}")

    # output value from FFN
    out_value = sigma[out_neurons[0][0]]
    return out_value


if __name__ == '__main__':
    # checking NDM calculations for single output value
    hardcoded_ndm = np.ones((5, 7)) * 0.5
    x = np.linspace(-0.8, 0.8, 1601)
    y = np.linspace(-0.8, 0.8, 1601)
    samples = np.column_stack([x, y])

    ndm_out = calculate_output_from_ndm(
        hardcoded_ndm,
        in_neurons=np.array([[0, 1]]),
        out_neurons=np.array([[4]]),
        in_neurons_value=samples[0]
    )

    # single output from the NDM for the samples[0]
    print(f"Single {ndm_out=}")
