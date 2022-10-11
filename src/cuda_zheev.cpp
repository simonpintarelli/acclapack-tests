#include <cuda.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <cusolver_common.h>

#include <iostream>

#include "common/common.hpp"
#include "generate_sym_matrix.hpp"
#include "util.hpp"

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

void
zheevd(cusolverDnHandle_t handle,
       cusolverEigMode_t jobz,
       cublasFillMode_t uplo,
       int n,
       cuDoubleComplex* A,
       int lda,
       double* w,
       int& Info)
{
  cuDoubleComplex* cA = reinterpret_cast<cuDoubleComplex*>(A);
  int lwork           = 0;
  CALL_CUSOLVER(cusolverDnZheevd_bufferSize, (handle, jobz, uplo, n, cA, lda, w, &lwork));

  cuDoubleComplex* work_ptr;
  CALL_CUDA(cudaMalloc, (&work_ptr, lwork * sizeof(cuDoubleComplex)));
  int* dev_Info;
  CALL_CUDA(cudaMalloc, ((void**)&dev_Info, sizeof(int)));
  CALL_CUSOLVER(cusolverDnZheevd,
                (handle, jobz, uplo, n, cA, lda, w, work_ptr, lwork, dev_Info));
  CALL_CUDA(cudaMemcpy, (&Info, dev_Info, sizeof(int), cudaMemcpyDeviceToHost));
  CALL_CUDA(cudaFree, (work_ptr));
}

int
main(int argc, char* argv[])
{
  cusolverDnHandle_t handle;

  CALL_CUSOLVER(cusolverDnCreate, (&handle));

  int n   = 1000;
  int lda = n;

  // allocate device memory and create Hermitian (random) matrix A
  double2* A_dev;
  CALL_CUDA(cudaMalloc, (&A_dev, sizeof(double2) * n * lda));
  double* w_dev;
  CALL_CUDA(cudaMalloc, (&w_dev, sizeof(double) * n));
  generate_hermitian_matrix(A_dev, n, lda);

  auto fill_mode = cublasFillMode_t::CUBLAS_FILL_MODE_FULL;
  int info;
  zheevd(handle,
         cusolverEigMode_t::CUSOLVER_EIG_MODE_VECTOR,
         fill_mode,
         n,
         reinterpret_cast<cuDoubleComplex*>(A_dev),
         lda,
         w_dev,
         info);

  return 0;
}
