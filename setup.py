# setup.py  (放在项目根目录)

from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os

# 1️⃣ CUDA 源文件及头文件所在目录
cuda_src_dir = "cuda_rasterizer"          # 视你的项目结构调整
third_party_glm = os.path.join("third_party", "glm")

# 2️⃣ 列出所有 .cu / .cpp / .cxx 源文件
sources = [
    os.path.join(cuda_src_dir, "rasterizer_impl.cu"),
    os.path.join(cuda_src_dir, "forward.cu"),
    os.path.join(cuda_src_dir, "backward.cu"),
    os.path.join(cuda_src_dir, "statistical_constants.cu"),
    "rasterize_points.cu",
    "ext.cpp",
]

setup(
    name="diff_gaussian_rasterization",
    packages=["diff_gaussian_rasterization"],
    ext_modules=[
        CUDAExtension(
            name="diff_gaussian_rasterization._C",
            sources=sources,
            # 3️⃣ 让 g++ / nvcc 都能找到 header
            include_dirs=[cuda_src_dir, third_party_glm],
            # 4️⃣ 关键编译/链接参数
            extra_compile_args={
                "cxx": [
                    "-std=c++17",
                    "-O3",
                    f"-I{cuda_src_dir}",      # 冗余一份，保险起见
                    f"-I{third_party_glm}",
                ],
                "nvcc": [
                    "-std=c++17",
                    "-O3",
                    "-rdc=true",              # 允许可重定位设备代码
                    "--expt-relaxed-constexpr",
                    f"-I{third_party_glm}",
                ],
            },
            # 让 Torch 帮你做 device-link 阶段；编译器会自动加 -lcudadevrt
            dlink=True,
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
