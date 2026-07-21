from setuptools import setup, find_packages
import glob

def read_requirements():
    with open('requirements.txt', 'r') as f:
        return [line.strip() for line in f.readlines() if line.strip()]

def get_extensions():
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension 

    cuda_source_files = glob.glob("csrc/*/*.cu") + glob.glob("csrc/*/*.cpp") + glob.glob("csrc/*.cpp")
    print(cuda_source_files)
    return  CUDAExtension(
            'ccdl_cuda_ops', 
            cuda_source_files,
            extra_compile_args={'cxx': [], 'nvcc': ['-O3', '-U__CUDA_NO_HALF_OPERATORS__']}
        )

def get_build_extensions():
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension 
    return BuildExtension.with_options(verbose_build=True)



setup(
    name='ccdl', 
    version='0.0.1', 
    author='ccdl', 
    author_email='ccdl@ccdl.com',  
    packages=find_packages(),
    ext_modules=[
        get_extensions()
    ],
    cmdclass={
        'build_ext': get_build_extensions()
    },
    install_requires=[
        "torch",
        "ninja",
        "pybind11"
    ]
)
