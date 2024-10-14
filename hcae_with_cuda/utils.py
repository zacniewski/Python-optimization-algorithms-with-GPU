from numba import jit, vectorize, guvectorize, int64, float64
import numpy as np

from activation_functions import activation_function
from constants import NDM_ROWS


@jit
def objective(v):
    xx, yy = v
    return np.sin(xx) * np.cos(yy)


@jit
def calculate_output_from_ndm(
        in_ndm: np.array,
        in_neurons: np.array,
        out_neurons: np.array,
        in_neurons_value: np.array,
):
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


sigma_2 = np.arange(NDM_ROWS, dtype=np.float64)
print(sigma_2[0])


@guvectorize([(float64[:, :], float64[:, :], float64[:, :], float64[:],  float64)], '(m, n), (o, p), (o, o), (p)->()', target='cuda')
def cuda_calculate_output_from_ndm(
        in_ndm: np.array,
        in_neurons: np.array,
        out_neurons: np.array,
        in_neurons_value: np.array,
        out_value
):
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
    #sigma_2[0] = 88
    """sigma_2[0] = activation_function(
        input_value=z_for_first_neuron, type_of_neuron_value=in_ndm[0][-1]
    )

    # calculating output values of neurons and storing them in the 'sigma' dictionary
    z = 0
    for j in range(1, in_ndm.shape[1] - 2):
        for i in range(j):
            z = z + in_ndm[i][j] * sigma_2[i]

        # adding bias
        z = z + in_ndm[j][-2]

        # adding input value (if exists) multiplied by its weight (so far it's 1 by default)
        if j in in_neurons:
            z = z + in_neurons_value[j] * 1

        # determine the activation function of 'z' basing on the values of input and 'type of neuron' cell
        sigma_2[j] = activation_function(
            input_value=z, type_of_neuron_value=abs(in_ndm[j][-1])
        )
        z = 0

    # uncomment to check the outputs value of all neurons (after activation function)
    # it's good for testing purposes!
    # print(sigma_2)
    """
    # output value from FFN
    out_value = 66# sigma_2[out_neurons[0][0]]


@jit
def calculate_error(current_ndm, samples_values, in_neurons, out_neurons) -> float:
    # values calculated from the NDM
    output_values_for_samples = [
        calculate_output_from_ndm(
            current_ndm,
            in_neurons,
            out_neurons,
            in_neurons_value=s,
        )
        for s in samples_values
    ]

    # values of the objective function for all samples
    objective_values_for_samples = [objective(s) for s in samples_values]

    # error value
    return sum(
        [
            np.abs(out - obj)
            for out, obj in zip(output_values_for_samples, objective_values_for_samples)
        ]
    )

