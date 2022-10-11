#pragma once

#if defined(ALT_USE_CUDA)
#include <cuda_runtime.h>
#elif defined(ALT_USE_ROCM)
#include <hip/hip_runtime.h>
#endif

extern "C" void generate_hermitian_matrix(double2* A_dev, int n, int lda);
