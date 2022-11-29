from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='prediction collection',
    ext_modules=[
        CUDAExtension('pred_collect_ext', [
            'src/pred_collect_ext.cpp',
            'src/pred_collect_cuda_kernel.cu'
        ]),
    ],
    cmdclass={'build_ext': BuildExtension})