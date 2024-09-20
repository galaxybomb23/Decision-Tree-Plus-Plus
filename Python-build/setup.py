# setup.py
import os
from setuptools import setup, Extension
from pybind11.setup_helpers import build_ext
import pybind11
import shutil
# python setup.py build_ext --inplace

#change the directory to the file location
os.chdir(os.path.dirname(os.path.abspath(__file__)))

#copy the src and includes dir locally
if os.path.exists('src'):
    shutil.rmtree('src')
shutil.copytree('../src', 'src')
if os.path.exists('include'):
    shutil.rmtree('include')
shutil.copytree('../include', 'include')


#get name in src dir
src_files = os.listdir('src')
src_files = ['src/' + file for file in src_files]
src_files.append("bindings.cpp")
includes_files = os.listdir('include')
includes_files = ['include/' + file for file in includes_files]



ext_modules = [
    Extension(
        'DecisionTreePP',  # Name of the resulting Python module
        src_files,  # Source file
        include_dirs=[pybind11.get_include()],  # Include `pybind11` headers
        language='c++',
    ),
]

setup(
    name='DecisionTreePP',
    ext_modules=ext_modules,
    cmdclass={'build_ext': build_ext},  # Use `build_ext` imported from `pybind11.setup_helpers`
)