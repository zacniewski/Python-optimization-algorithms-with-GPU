import math
import numpy as np
from numba import cuda
from skimage import data
import matplotlib.pyplot as plt


moon = data.moon().astype(np.float32) / 255.


@cuda.jit
def adjust_log(inp, gain, out):
    ix, iy = cuda.grid(2)  # The first index is the fastest dimension
    # ix and iy change from 0 to 1023

    threads_per_grid_x, threads_per_grid_y = cuda.gridsize(2)  # threads per grid dimension
    # in this case threads_per_grid_x = threads_per_block_2d[0] * blocks_per_grid_2d[0]
    # in this case threads_per_grid_y = threads_per_block_2d[1] * blocks_per_grid_2d[1]
    # it gives (16 * 64 = 1024, 16 * 64 = 1024)

    n0, n1 = inp.shape  # The last index is the fastest dimension, shape should be (512, 512) in this case (moon image shape)

    # Stride each dimension independently
    for i0 in range(iy, n0, threads_per_grid_y):
        for i1 in range(ix, n1, threads_per_grid_x):
            out[i0, i1] = gain * math.log2(1 + inp[i0, i1])


threads_per_block_2d = (16, 16)  # 256 threads total
blocks_per_grid_2d = (64, 64)

moon_gpu = cuda.to_device(moon)
moon_corr_gpu = cuda.device_array_like(moon_gpu)

adjust_log[blocks_per_grid_2d, threads_per_block_2d](moon_gpu, 1.0, moon_corr_gpu)
moon_corr = moon_corr_gpu.copy_to_host()

fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.imshow(moon, cmap="gist_earth")
ax2.imshow(moon_corr, cmap="gist_earth")
ax1.set(title="Original image")
ax2.set(title="Log-corrected image")
for ax in (ax1, ax2):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
fig.tight_layout()
plt.show()
