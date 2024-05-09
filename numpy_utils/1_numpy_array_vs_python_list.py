import random
import time

import numpy as np

NUMBER_OF_ELEMENTS = 100_000_000
NUMBER_OF_INTEGERS = 10_000_000


def python_create_range():
    """
    Function to create 'range' object with 100 million numbers
    :return: time of execution in seconds
    """
    start = time.perf_counter()
    # *: argument-unpacking operator
    x = [*range(NUMBER_OF_ELEMENTS)]
    end = time.perf_counter()
    times = end - start
    return f"Standard list with range() - created in {np.round(times, 3)} seconds."


def numpy_create_array():
    """
    Function to create NumPy array with 100 million numbers
    :return: time of execution in seconds
    """
    start = time.perf_counter()
    y = np.arange(NUMBER_OF_ELEMENTS)
    end = time.perf_counter()
    times = end - start
    return f"Numpy's array - created in {np.round(times, 3)} seconds."


def python_sum():
    """
    Function to sum 100 million integers (from 1 to 10) of Python's list
    """
    start = time.perf_counter()
    sum_of_list = sum([random.randint(1, 10) for _ in range(NUMBER_OF_ELEMENTS)])
    end = time.perf_counter()
    times = end - start
    return f"Sum of list (Python) - calculated in {np.round(times, 3)} seconds."


def numpy_sum():
    """
    Function to sum 100 million integers (from 1 to 10) of Numpy's array
    """
    start = time.perf_counter()
    sum_of_array = np.random.randint(1, 10, size=NUMBER_OF_ELEMENTS).sum()
    end = time.perf_counter()
    times = end - start
    return f"Sum of array (Numpy) - calculated in {np.round(times, 3)} seconds."


if __name__ == "__main__":
    print(python_create_range())
    print(numpy_create_array())
    print(python_sum())
    print(numpy_sum())
