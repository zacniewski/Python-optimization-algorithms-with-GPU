import time

import numpy as np


def create_range():
    """
    Function to create 'range' object with 100 million numbers
    :return: time of execution in seconds
    """
    start = time.perf_counter()
    # *: argument-unpacking operator
    x = [*range(100_000_000)]
    end = time.perf_counter()
    times = end - start
    return f"Standard range - executed in {np.round(times, 3)} seconds."


def create_array():
    """
    Function to create NumPy array with 100 million numbers
    :return: time of execution in seconds
    """
    start = time.perf_counter()
    y = np.arange(100_000_000)
    end = time.perf_counter()
    times = end - start
    return f"Numpy's array - executed in {np.round(times, 3)} seconds."


if __name__ == "__main__":
    print(create_range())
    print(create_array())
