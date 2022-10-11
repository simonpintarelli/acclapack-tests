#include <cusolverDn.h>
#include <cusolver_common.h>
#include <iostream>
#include "common/common.hpp"

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

cusolverStatus_t
zheevd(cusolverDnHandle handle, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, cuDoubleComplex* A, int lda, double* w, int& Info)
{
    cuDoubleComplex* cA                = reinterpret_cast<cuDoubleComplex*>(A);
    int lwork                          = 0;
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
    cusolverStatus_t ret_cusolver =
        cusolverDnZheevd(cusolver_handle, jobz, uplo, n, cA, lda, w, work_ptr, lwork, dev_Info);
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

    return 0;
}
