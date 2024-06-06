from time import perf_counter

import numba
import numpy as np
from skimage import io


def tg_numpy(color_image):
    result = np.round(
        0.299 * color_image[:, :, 0] +
        0.587 * color_image[:, :, 1] +
        0.114 * color_image[:, :, 2]
    )
    return result.astype(np.uint8)


@numba.jit
def tg_numba(color_image):
    result = np.round(
        0.299 * color_image[:, :, 0] +
        0.587 * color_image[:, :, 1] +
        0.114 * color_image[:, :, 2]
    )
    return result.astype(np.uint8)


@numba.jit
def tg_numba_for_loop(color_image):
    result = np.empty(color_image.shape[:2], dtype=np.uint8)
    for y in range(color_image.shape[0]):
        for x in range(color_image.shape[1]):
            r, g, b = color_image[y, x, :]
            result[y, x] = np.round(
                0.299 * r + 0.587 * g + 0.114 * b
            )
    return result


if __name__ == "__main__":
    # Please, download your image. My is around 50 MB and I don't want to upload it.
    RGB_IMAGE = io.imread("images/large-image.jpg")
    print("Shape:", RGB_IMAGE.shape)
    print("dtype:", RGB_IMAGE.dtype)
    print("Memory usage (bytes):", RGB_IMAGE.size)

    # Only NumPy
    start1 = perf_counter()
    GRAYSCALE = tg_numpy(RGB_IMAGE)
    end1 = perf_counter()
    print(f"Elapsed time for NumPy: {end1 - start1:.4f} sec.")

    start2 = perf_counter()
    GRAYSCALE2 = tg_numba(RGB_IMAGE)
    end2 = perf_counter()
    print(f"Elapsed time for Numba (wrong way): {end2 - start2:.4f} sec.")
    assert np.array_equal(GRAYSCALE, GRAYSCALE2)

    # Numba used in the good way
    start3 = perf_counter()
    GRAYSCALE3 = tg_numba_for_loop(RGB_IMAGE)
    end3 = perf_counter()
    print(f"Elapsed time for Numba (good way): {end3 - start3:.4f} sec.")
    assert np.array_equal(GRAYSCALE, GRAYSCALE3)
