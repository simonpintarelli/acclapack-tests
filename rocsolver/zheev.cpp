#include <rocsolver.h>
#include <rocblas.h>
#include <complex>
#include <stdexcept>
#include "common/common.hpp"
#include "generate_sym_matrix.hpp"

#define CALL_ROCBLAS(func__, args__)                                                                                   \
    {                                                                                                                  \
        rocblas_status status = func__ args__;                                                                         \
        if (status != rocblas_status::rocblas_status_success) {                                                        \
            char nm[1024];                                                                                             \
            gethostname(nm, 1024);                                                                                     \
            printf("hostname: %s\n", nm);                                                                              \
            printf("Error in %s at line %i of file %s: %s\n", #func__, __LINE__, __FILE__,                             \
                   rocblas_status_to_string(status));                                                                  \
            stack_backtrace();                                                                                         \
        }                                                                                                              \
    }

#define CALL_HIP(func__, args__)                                                                                       \
    {                                                                                                                  \
        hipError_t status = func__ args__;                                                                             \
        if (status != hipSuccess) {                                                                                    \
            char nm[1024];                                                                                             \
            gethostname(nm, 1024);                                                                                     \
            printf("hostname: %s\n", nm);                                                                              \
            printf("Error in %s at line %i of file %s: %s\n", #func__, __LINE__, __FILE__, hipGetErrorString(status)); \
            stack_backtrace();                                                                                         \
        }                                                                                                              \
    }

void
zheevd(rocblas_handle& handle, rocblas_evect mode, rocblas_fill uplo, int n, rocblas_double_complex* A, int lda,
       double* w)
{

    // rocsolver_zh
    if (mode != rocblas_evect::rocblas_evect_original) {
        throw std::runtime_error("unsupported mode in rocm::heevd");
    }

    rocblas_double_complex* A_ptr = reinterpret_cast<rocblas_double_complex*>(A);

    int* dev_info{nullptr};
    CALL_HIP(hipMalloc, (&dev_info, sizeof(rocblas_int)));

    double* E;
    CALL_HIP(hipMalloc, (&E, n * sizeof(double)));

    CALL_ROCBLAS(rocsolver_zheevd, (handle, mode, uplo, n, A_ptr, lda, w, E, dev_info));

    int info;
    CALL_HIP(hipMemcpyDtoH, (&info, dev_info, sizeof(int)));
    CALL_HIP(hipFree, (E));
}

int
main(int argc, char* argv[])
{

    rocblas_handle handle;
    rocblas_create_handle(&handle);

    int n   = 1000;
    int lda = n;

    double2* A_dev;
    CALL_HIP(hipMalloc, (&A_dev, sizeof(double2) * n * lda));
    generate_hermitian_matrix(A_dev, n, lda);

    double* w_dev;
    CALL_HIP(hipMalloc, (&w_dev, sizeof(double) * n));

    auto fill_mode = rocblas_fill::rocblas_fill_full;
    zheevd(handle, rocblas_evect::rocblas_evect_original, fill_mode, n,
           reinterpret_cast<rocblas_double_complex*>(A_dev), lda, w_dev);

    return 0;
}
