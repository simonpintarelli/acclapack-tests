#include "common/common.hpp"
#include "generate_sym_matrix.hpp"
#include "util.hpp"
#include <chrono>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <cusolver_common.h>
#include <cxxopts.hpp>
#include <exception>
#include <iostream>

#define CALL_CUDA(func__, args__)                                                             \
  {                                                                                           \
    cudaError_t error = func__ args__;                                                        \
    if (error != cudaSuccess) {                                                               \
      char nm[1024];                                                                          \
      gethostname(nm, 1024);                                                                  \
      printf("hostname: %s\n", nm);                                                           \
      printf("Error in %s at line %i of file %s: %s\n",                                       \
             #func__,                                                                         \
             __LINE__,                                                                        \
             __FILE__,                                                                        \
             cudaGetErrorString(error));                                                      \
      stack_backtrace();                                                                      \
    }                                                                                         \
  }

#define CALL_CUSOLVER(func__, args__)                                                         \
  {                                                                                           \
    cusolverStatus_t status;                                                                  \
    if ((status = func__ args__) != CUSOLVER_STATUS_SUCCESS) {                                \
      cusolver::error_message(status);                                                        \
      char nm[1024];                                                                          \
      gethostname(nm, 1024);                                                                  \
      std::printf("hostname: %s\n", nm);                                                      \
      std::printf("Error in %s at line %i of file %s\n", #func__, __LINE__, __FILE__);        \
      stack_backtrace();                                                                      \
    }                                                                                         \
  }

void zheevd(cusolverDnHandle_t handle,
            cusolverEigMode_t jobz,
            cublasFillMode_t uplo,
            int n,
            cuDoubleComplex *A,
            int lda,
            double *w,
            int &Info) {
  int lwork           = 0;
  CALL_CUSOLVER(cusolverDnZheevd_bufferSize, (handle, jobz, uplo, n, A, lda, w, &lwork));

  cuDoubleComplex *work_ptr;
  CALL_CUDA(cudaMalloc, (&work_ptr, lwork * sizeof(cuDoubleComplex)));
  int *dev_Info;
  CALL_CUDA(cudaMalloc, ((void **)&dev_Info, sizeof(int)));
  CALL_CUSOLVER(cusolverDnZheevd,
                (handle, jobz, uplo, n, A, lda, w, work_ptr, lwork, dev_Info));
  CALL_CUDA(cudaMemcpy, (&Info, dev_Info, sizeof(int), cudaMemcpyDeviceToHost));
  CALL_CUDA(cudaFree, (work_ptr));
}

double run(int n, int lda) {
  cusolverDnHandle_t handle;

  CALL_CUSOLVER(cusolverDnCreate, (&handle));

  // allocate device memory and create Hermitian (random) matrix A
  double2 *A_dev;
  CALL_CUDA(cudaMalloc, (&A_dev, sizeof(double2) * n * lda));
  double *w_dev;
  CALL_CUDA(cudaMalloc, (&w_dev, sizeof(double) * n));
  generate_hermitian_matrix(A_dev, n, lda);

  auto fill_mode = cublasFillMode_t::CUBLAS_FILL_MODE_LOWER;
  int info;

  CALL_CUDA(cudaDeviceSynchronize, ());
  auto start = std::chrono::high_resolution_clock::now();
  zheevd(handle,
         cusolverEigMode_t::CUSOLVER_EIG_MODE_VECTOR,
         fill_mode,
         n,
         reinterpret_cast<cuDoubleComplex *>(A_dev),
         lda,
         w_dev,
         info);
  CALL_CUDA(cudaDeviceSynchronize, ());
  double t =
      std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start).count();

  CALL_CUSOLVER(cusolverDnDestroy, (handle));
  if (info != 0) {
    std::fprintf(stderr, "Error: info: %d", info);
    std::terminate();
  }

  return t;
}

int main(int argc, char *argv[]) {
  cxxopts::Options options("zheev", "run rocsolver zheev");

  /* clang-format off */
  options.add_options()
      ("n,size", "matrix size", cxxopts::value<std::vector<int>>())
      ("r,repeat", "number of repetitions", cxxopts::value<int>()->default_value("1"));

  auto results = options.parse(argc, argv);

  auto ns  = results["n"].as<std::vector<int>>();
  int nrep = results["r"].as<int>();

  for (int n : ns) {
    for (int i = 0; i < nrep; ++i) {
      double t = run(n, n);
      std::printf("n: %40d\t%f\n", n, t);
    }
  }
  return 0;
}
