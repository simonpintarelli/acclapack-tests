#include <cuda.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <cusolver_common.h>
#include <iostream>
#include "common/common.hpp"

#define CALL_CUDA(func__, args__)                                                                                      \
    {                                                                                                                  \
        cudaError_t error = func__ args__;                                                                             \
        if (error != cudaSuccess) {                                                                                    \
            char nm[1024];                                                                                             \
            gethostname(nm, 1024);                                                                                     \
            printf("hostname: %s\n", nm);                                                                              \
            printf("Error in %s at line %i of file %s: %s\n", #func__, __LINE__, __FILE__, cudaGetErrorString(error)); \
            stack_backtrace();                                                                                         \
        }                                                                                                              \
    }

#define CALL_CUSOLVER(func__, args__)                                                                                  \
    {                                                                                                                  \
        cusolverStatus_t status;                                                                                       \
        if ((status = func__ args__) != CUSOLVER_STATUS_SUCCESS) {                                                     \
            cusolver::error_message(status);                                                                           \
            char nm[1024];                                                                                             \
            gethostname(nm, 1024);                                                                                     \
            std::printf("hostname: %s\n", nm);                                                                         \
            std::printf("Error in %s at line %i of file %s\n", #func__, __LINE__, __FILE__);                           \
            stack_backtrace();                                                                                         \
        }                                                                                                              \
    }

namespace cusolver {

void error_message(cusolverStatus_t status);

struct cusolverDnHandle
{
    static cusolverDnHandle_t& _get()
    {
        static cusolverDnHandle_t handle{nullptr};
        return handle;
    }

    static cusolverDnHandle_t& get()
    {
        auto& handle = _get();
        if (!handle) {
            CALL_CUSOLVER(cusolverDnCreate, (&handle));
        }
        return handle;
    }

    static void destroy()
    {
        if (!_get())
            CALL_CUSOLVER(cusolverDnDestroy, (_get()));
        _get() = nullptr;
    }

    static cusolverDnHandle_t handle;
};
} // namespace cusolver

cusolverStatus_t
zheevd(cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, cuDoubleComplex* A, int lda,
       double* w, int& Info)
{
    cuDoubleComplex* cA              = reinterpret_cast<cuDoubleComplex*>(A);
    int lwork                        = 0;
    cusolverStatus_t ret_buffer_size = cusolverDnZheevd_bufferSize(handle, jobz, uplo, n, cA, lda, w, &lwork);
    if (ret_buffer_size != CUSOLVER_STATUS_SUCCESS) {
        std::cerr << "Something went wrong\n"
                  << "return value: " << ret_buffer_size << "\n";
        exit(1);
    }

    cuDoubleComplex* work_ptr;
    CALL_CUDA(cudaMalloc, (&work_ptr, lwork * sizeof(cuDoubleComplex)));
    int* dev_Info;
    CALL_CUDA(cudaMalloc, ((void**)&dev_Info, sizeof(int)));
    cusolverStatus_t ret_cusolver = cusolverDnZheevd(handle, jobz, uplo, n, cA, lda, w, work_ptr, lwork, dev_Info);
    CALL_CUDA(cudaDeviceSynchronize, ());
    CALL_CUDA(cudaMemcpy, (&Info, dev_Info, sizeof(int), cudaMemcpyDeviceToHost));
    CALL_CUDA(cudaFree, (work_ptr));
    if (ret_cusolver != CUSOLVER_STATUS_SUCCESS) {
        std::cerr << "Error in DnZheevd\n"
                  << "return value: " << ret_cusolver << "\n"
                  << "info: " << Info << "\n";

        exit(1);
    }
    return ret_cusolver;
}

int
main(int argc, char* argv[])
{
    cusolverDnHandle_t handle;
    CALL_CUSOLVER(cusolverDnCreate, (&handle));
    return 0;
}
