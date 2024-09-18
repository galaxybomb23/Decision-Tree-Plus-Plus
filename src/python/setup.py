# setup.py
import os
from setuptools import setup, Extension
from pybind11.setup_helpers import build_ext
import pybind11
# python setup.py build_ext --inplace

#change the directory to the file location
os.chdir(os.path.dirname(os.path.abspath(__file__)))
ext_modules = [
    Extension(
        'demoClass',  # Name of the resulting Python module
        ['DemoClass.cpp', 'bindings.cpp'],  # Source file
        include_dirs=[pybind11.get_include()],  # Include `pybind11` headers
        language='c++',
    ),
]

setup(
    name='demoClass',
    ext_modules=ext_modules,
    cmdclass={'build_ext': build_ext},  # Use `build_ext` imported from `pybind11.setup_helpers`
)