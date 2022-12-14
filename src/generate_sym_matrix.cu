#include <cuComplex.h>
#include <cuda_runtime.h>
#include <curand.h>

__global__ void
generate_hermitian_matrix(const double2* __restrict__ Rng,
                          double2* A_dev,
                          const int n,
                          const int lda)
{
  const int row = blockDim.x * blockIdx.x + threadIdx.x;
  const int col = blockDim.y * blockIdx.y + threadIdx.y;

  if ((row >= n) || (col >= n)) return;

  if (row <= col) {
    const double angle1      = 2.0 * M_PI * Rng[row * lda + col].x;
    const double norm1       = Rng[row * lda + col].y;
    const double2 cs         = make_double2(norm1 * cos(angle1), norm1 * sin(angle1));
    A_dev[row * lda + col].x = cs.x;
    A_dev[row * lda + col].y = cs.y * (row != col);
  }

  if (row > col) {
    const double angle1      = 2.0 * M_PI * Rng[col * lda + row].x;
    const double norm1       = Rng[col * lda + row].y;
    const double2 cs         = make_double2(norm1 * cos(angle1), norm1 * sin(angle1));
    A_dev[row * lda + col].x = cs.x;
    A_dev[row * lda + col].y = cs.y;
  }
}

__host__ void
generate_hermitian_matrix(double2* A_dev, int n, int lda)
{
  curandGenerator_t gen;
  dim3 block, thread;

  curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_XORWOW);

  block.x  = n / 128 + ((n % 128) != 0);
  block.y  = n / 128 + ((n % 128) != 0);
  thread.x = 128;
  thread.y = 128;

  double2* Rng;
  cudaMalloc(&Rng, sizeof(double2) * n * lda);
  curandGenerateUniformDouble(gen, reinterpret_cast<double*>(Rng), 2 * n * lda);
  generate_hermitian_matrix<<<block, thread>>>(Rng, A_dev, n, lda);
}
