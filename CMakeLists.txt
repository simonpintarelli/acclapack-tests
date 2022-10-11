cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(acclapack-tests LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
# user variables
set(USE_OPENMP On CACHE BOOL "use OpenMP")
set(USE_CUDA Off CACHE BOOL "use cuda")
set(USE_ROCM Off CACHE BOOL "use amd gpus")

set(LAPACK_VENDOR "OpenBLAS" CACHE STRING "lapack vendor")
set_property(CACHE LAPACK_VENDOR PROPERTY STRINGS "OpenBLAS" "MKL")

if(USE_OPENMP)
  find_package(OpenMP REQUIRED)
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS "YES")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")


if(USE_CUDA)
  find_package(CUDAToolkit REQUIRED)
endif()

if(USE_ROCM)
  # find_package(hip REQUIRED)
  find_package(rocblas REQUIRED)
  find_package(rocsolver REQUIRED)
endif()

# find_package(MPI REQUIRED)

if(LAPACK_VENDOR MATCHES OpenBLAS)
  find_package(OpenBLAS REQUIRED)
  add_library(my_lapack INTERFACE IMPORTED)
  set_target_properties(my_lapack PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${OpenBLAS_INCLUDE_DIRS}"
    INTERFACE_LINK_LIBRARIES "${OpenBLAS_LIBRARIES}")
elseif(LAPACK_VENDOR MATCHES MKL)
  message("LAPACK VENDOR MKL")
  find_package(MKL REQUIRED)
elseif(LAPACK_VENDOR MATCHES CRAY_LIBSCI)
  message("LAPACK VENDOR Cray Libsci")
  find_package(SCI REQUIRED)
  add_library(lapack INTERFACE IMPORTED)
  set_target_properties(lapack PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${SCI_INCLUDE_DIRS}"
    INTERFACE_LINK_LIBRARIES "${SCI_LIBRARIES}")
else()
  message(FATAL_ERROR "must specify a LAPACK_VENDOR")
endif()


if(USE_CUDA)
  add_executable(cuda_zheev cusolver/zheev.cpp)
  target_include_directories(cuda_zheev PRIVATE ${CMAKE_SOURCE_DIR})
  target_link_libraries(cuda_zheev PRIVATE CUDA::cudart CUDA::cublas CUDA::cusolver)
  install(TARGETS cuda_zheev)
endif()

if(USE_ROCM)
  add_executable(rocm_zheev rocsolver/zheev.cpp)
  target_include_directories(rocm_zheev PRIVATE ${CMAKE_SOURCE_DIR})
  target_link_libraries(rocm_zheev PRIVATE roc::rocblas hip::device hip::host roc::rocsolver)
  install(TARGETS rocm_zheev)
endif()
