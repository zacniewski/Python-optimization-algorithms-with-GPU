import numpy as np

# In all documentation comments, after the colon a name of the variable from original paper is given.


def oper1(operation_parameters, data_sequence: np.ndarray, ndm) -> np.ndarray:
    """
     This function is an adaptation of a solution applied in AEEO.
     It is of a global range, which means that it can modify any element of NDM,
     and it uses a small feed-forward neural network, say, ANN operation, in the decision-making process.

     The task of ANN operation is to decide which NDM items are to be updated and how they are to be updated.
     The architecture of each ANN operation is determined by parameters this function,
     whereas inputs to the ANN operation are taken from the data sequence of AEP.

     Each ANN operation has two inputs and five outputs.
     The inputs indicate individual items of NDM.
     ANN operations are fed with data items which correspond to i and j, that is, with row [i] and column [j].
     Vectors 'row' and 'column' are filled with appropriate data items.

    :param operation_parameters: 'p'
    :param data_sequence: 'd'
    :param ndm: 'NDM'
    :return: 'NDM'
    """

    number_of_rows, number_of_columns = ndm.shape

    # 'row' and 'column' vectors as a containers for data from 'd'
    row = np.zeros(number_of_rows)
    column = np.zeros(number_of_columns)

    for i in range(number_of_rows):
        row[i] = data_sequence[i % len(data_sequence)]
    for i in range(number_of_columns):
        column[i] = data_sequence[(i + number_of_rows) % len(data_sequence)]

    # Q: how is the architecture of ANN operation created from the operation parameters?
    ann_oper = getANN(operation_parameters)

    # ann_inputs and ann_outputs are created only for the discussion purposes!
    # These inputs and outputs will be replaced by the ANN created from the operation parameters 'p'!
    ann_inputs = np.zeros(2)
    ann_outputs = np.zeros(5)

    for i in range(number_of_columns):
        for j in range(number_of_rows):
            ann_inputs[0] = row[j]
            ann_inputs[1] = column[i]

            # ann_oper.run() - running the ANN

            if ann_outputs[1] < ann_outputs[2]:
                if ann_outputs[3] < ann_outputs[4]:
                    ndm[j, i] = M * ann_outputs[5]
                else:
                    ndm[j, i] = 0

    return ndm


def getANN(operation_parameters: np.ndarray) -> np.ndarray:
    """
    This function generates the NDM operation and consequently, an ANN operation
    Like resultant ANNs, ANN operations are also represented in the form of NDMs,
    say, NDM operations.
    It fills all matrix items with subsequent parameters of 'oper1' divided by a scaling coefficient, N.
    If the number of parameters is too small to fill the entire matrix, they are used many times.

    :param operation_parameters: 'p'
    :return: ANN-operation (encoded in the NDM)

    Probably indexing of operation_parameters should be zero-based, but it can be easily corrected.

    """

    # it's a bad shape. It is used only temporarily!
    # Q: where to take the proper shape from?
    bad_hardcoded_shape = (5, 6)

    ndm_operation = np.zeros(bad_hardcoded_shape)
    number_of_item = 0

    # i - number of columns, j - number of rows
    number_of_rows, number_of_columns = ndm_operation.shape
    for i in range(number_of_columns):
        for j in range(number_of_rows):
            ndm_operation[j, i] = (
                operation_parameters[number_of_item % len(operation_parameters)] / N
            )
            number_of_item += 1

    return ndm_operation


def oper2(operation_parameters, data_sequence: np.ndarray, ndm) -> np.ndarray:
    """
    This function  directly fills NDM with values from the data sequence of AEP.
    The operation parameters determine:
        - where NDM (Network Definition Matrix) is updated,
        - and which and how many data items are used.

    The first parameter (index 1) indicates the direction according to which NDM is modified,
    that is, whether it is changed along columns or rows.

    The second parameter (index 2) determines the size of holes between NDM updates, that is,
    the number of zeros that separate consecutive updates.

    The next two parameters (indexes 3 and 4) point out the location in NDM where the operation starts to work, i.e.,
    they indicate the starting row and column.

    The fifth parameter (index 5) determines the size of the altered NDM area, in other words,
    it indicates how many NDM items are updated.

    The last, sixth parameter (index 6) points out location in the sequence of data
    from where the operation starts to take data items and put them into the NDM.

    :param operation_parameters: 'p'
    :param data_sequence: 'd'
    :param ndm: 'NDM'
    :return: 'NDM'
    """

    filled = 0
    where = operation_parameters[6]
    holes = 0
    num_of_ndm_rows, num_of_ndm_columns = ndm.shape

    # check direction of filling
    if operation_parameters[1] % 2 == 0:
        for k in range(num_of_ndm_columns):
            for j in range(num_of_ndm_rows):
                ndm[j][k] = fill(
                    k, j, operation_parameters, data_sequence, filled, where, holes
                )
    else:
        for k in range(num_of_ndm_rows):
            for j in range(num_of_ndm_columns):
                ndm[k][j] = fill(
                    j, k, operation_parameters, data_sequence, filled, where, holes
                )

    return ndm


def fill(
    number_of_column: int,
    number_of_row: int,
    operation_parameters: np.ndarray,
    data_sequence: np.ndarray,
    number_of_updated_items: int,
    starting_position_in_data: int,
    number_of_holes: int,
) -> int:
    """
    This function is used in the 'oper2' function to update the given cell of the NDM matrix.
    :param number_of_column: 'c'
    :param number_of_row: 'r'
    :param operation_parameters: 'p'
    :param data_sequence: 'd'
    :param number_of_updated_items: 'f'
    :param starting_position_in_data: 'w'
    :param number_of_holes: 'h'

    :return: new value for NDM item
    """

    if (
        number_of_updated_items < operation_parameters[5]
        and number_of_column > operation_parameters[4]
        and number_of_row > operation_parameters[3]
    ):
        number_of_updated_items += 1
        if number_of_holes == operation_parameters[2]:
            number_of_holes = 0
            starting_position_in_data += 1
            return data_sequence[starting_position_in_data % len(data_sequence)]
        else:
            number_of_holes += 1
            return 0


def hcae_evolution():
    pass


if __name__ == "__main__":
    # Parameters
    M = 0.5
    N = 0.5
    population_size = 100
    operations_parameters_size = 6
    data_sequence_size = 50
