from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='remap_cuda',
    ext_modules=[
        CUDAExtension('remap_cuda', [
            './map_cuda/remap_cuda.cpp',
            './map_cuda/remap_cuda_kernel.cu'
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
