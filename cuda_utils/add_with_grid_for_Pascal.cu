#include <iostream>
#include <math.h>

//  Kernel to initialize the data.
// We can just replace the host code that initializes x and y with a launch of this kernel.
__global__ void init(int n, float *x, float *y) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }
}

// Kernel function to add the elements of two arrays
__global__
void add(int n, float *x, float *y)
{
  // gridDim.x contains the number of blocks in the grid,
  // blockIdx.x contains the index of the current thread block in the grid,
  // blockDim.x contains the number of threads in the block.
  // each thread gets its index by computing the offset to the beginning of its block:
  // (the block index times the block size: blockIdx.x * blockDim.x) and adding the thread’s index within the block (threadIdx.x)
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride)
    y[i] = x[i] + y[i];
}

int main(void)
{
  int N = 1<<20;
  float *x, *y;

  // Allocate Unified Memory – accessible from CPU or GPU
  cudaMallocManaged(&x, N*sizeof(float));
  cudaMallocManaged(&y, N*sizeof(float));

  // initialize x and y arrays on the host
  //for (int i = 0; i < N; i++) {
  //  x[i] = 1.0f;
  //  y[i] = 2.0f;
  // }

  // Run kernel on 1M elements on the GPU
  int blockSize = 256;
  int numBlocks = (N + blockSize - 1) / blockSize;
  printf("%d \n", numBlocks);
  init<<<numBlocks, blockSize>>>(N, x, y);
  add<<<numBlocks, blockSize>>>(N, x, y);

  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();

  // Check for errors (all values should be 3.0f)
  float maxError = 0.0f;
  for (int i = 0; i < N; i++)
    maxError = fmax(maxError, fabs(y[i]-3.0f));
  std::cout << "Max error: " << maxError << std::endl;

  // Free memory
  cudaFree(x);
  cudaFree(y);

  return 0;
}