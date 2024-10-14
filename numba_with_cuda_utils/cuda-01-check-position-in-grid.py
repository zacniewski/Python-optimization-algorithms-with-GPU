from numba import cuda


@cuda.jit
def another_kernel():
    """Commands to get thread positions"""
    # Get the thread position in a thread block
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    tz = cuda.threadIdx.z

    # Get the id of the thread block
    block_x = cuda.blockIdx.x
    block_y = cuda.blockIdx.y
    block_z = cuda.blockIdx.z

    # Number of threads per block
    dim_x = cuda.blockDim.x
    dim_y = cuda.blockDim.y
    dim_z = cuda.blockDim.z

    # Global thread position
    pos_x = tx + block_x * dim_x
    pos_y = ty + block_y * dim_y
    pos_z = tz + block_z * dim_z

    # We can also use the grid function to get
    # the global position

    (pos_x, pos_y, pos_z) = cuda.grid(3)
    # For a 1-or 2-d grid use grid(1) or grid(2)
    # to return a scalar or a two tuple.


threadsperblock = (16, 16, 4)  # Should be a multiple of 32 if possible.
blockspergrid = (256, 256, 256)  # Blocks per grid

another_kernel[blockspergrid, threadsperblock]()
