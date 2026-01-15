# python demo/build_cu_cpp_NoPackage/setup.py build_ext --build-lib demo/build_cu_cpp_NoPackage --build-temp demo/build_cu_cpp_NoPackage/build
# 

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os
R = os.path.dirname(__file__)

setup(
    name="my_add_NoPackage_ext",
    ext_modules=[
        CUDAExtension(
            name="my_add_NoPackage_ext",
            sources=[
                f"{R}/add_cuda.cpp",
                f"{R}/add_kernel.cu"
            ],
            # extra_compile_args={
            #     "cxx": ["-O3"],
            #     "nvcc": ["-O3"],
            # },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
