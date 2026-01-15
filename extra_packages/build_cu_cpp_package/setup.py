from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="build_cu_cpp",
    version="0.0.1",
    packages=["build_cu_cpp"],
    ext_modules=[
        CUDAExtension(
            name="build_cu_cpp.my_add_ext",  # <-- important: put extension inside package
            sources=[
                "build_cu_cpp/add_cuda.cpp",
                "build_cu_cpp/add_kernel.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3"],
            },
        ),
        CUDAExtension(
            name="build_cu_cpp.my_add_forward_backward_ext",  # <-- important: put extension inside package
            sources=[
                "build_cu_cpp/add_forward_backward_cuda.cpp",
                "build_cu_cpp/add_forward_backward_kernel.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3"],
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
