# Copyright 2013-2022 Lawrence Livermore National Security, LLC and other
# Spack Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

from spack.package import *


class AcclapackTests(CMakePackage, CudaPackage,  ROCmPackage):
    """Nonlinear CG methods for wave-function optimization in DFT."""

    homepage = "https://github.com/simonpintarelli/acclapack-tests"
    git = "https://github.com/simonpintarelli/acclapack-tests.git"
    url = "https://github.com/simonpintarelli/acclapack-tests/archive/"


    version('master', branch="master")

    variant('openmp', default=True)
    variant('cuda', default=False)
    variant('rocm', default=False)
    variant('build_type',
            default="Release",
            description="CMake build type",
            values=("Debug", "Release", "RelWithDebInfo"),
            )

    depends_on('lapack')
    depends_on('rocblas', when='+rocm')
    depends_on('rocsolver', when='+rocm')

    def cmake_args(self):

        options = [
            self.define_from_variant('USE_OPENMP', 'openmp'),
            self.define_from_variant('USE_CUDA', 'cuda'),
            self.define_from_variant('USE_ROCM', 'rocm')
                        ]
        if self.spec['blas'].name in ['intel-mkl', 'intel-parallel-studio']:
            options.append('-DLAPACK_VENDOR=MKL')
        elif self.spec['blas'].name in ['openblas']:
            options.append('-DLAPACK_VENDOR=OpenBLAS')
        else:
            raise Exception('blas/lapack must be either openblas or mkl.')

        if '+rocm' in self.spec:
            options.append('-DUSE_ROCM=On')
            options.append(self.define(
                'CMAKE_CXX_COMPILER', self.spec['hip'].hipcc))

        return options
