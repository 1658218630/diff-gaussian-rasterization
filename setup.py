#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os

here = os.path.dirname(os.path.abspath(__file__))

setup(
    name="diff_gaussian_rasterization",
    packages=['diff_gaussian_rasterization'],
    ext_modules=[
        CUDAExtension(
            name="diff_gaussian_rasterization._C",
            sources=[
                "cuda_rasterizer/statistical_constants.cu",
                "cuda_rasterizer/rasterizer_impl.cu",
                "cuda_rasterizer/forward.cu",
                "cuda_rasterizer/backward.cu",
                "rasterize_points.cu",
                "ext.cpp",
            ],
            include_dirs=[
                os.path.join(here, "cuda_rasterizer"),
                os.path.join(here, "third_party", "glm"),
            ],
            # NVCC: 开启设备代码重定位，加个优化等级
            extra_compile_args={
                "nvcc": [
                    "-allow-unsupported-compiler",
                    "-rdc=true",
                    "-O3",
                ],
                # CXX: 给 C++ 源也加优化
                "cxx": ["-O3"],
            },
            # 链接时显式拉入 cudart 库，否则找不到 __cudaRegisterLinkedBinary
            extra_link_args=["-lcudart", "-lcudadevrt"],
        )
    ],
    cmdclass={'build_ext': BuildExtension},
)
