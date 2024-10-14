# setup.py
import os
from setuptools import setup, Extension
from pybind11.setup_helpers import build_ext
import pybind11
import shutil
import subprocess
# python setup.py build_ext --inplace

def find_eigen_include_dir():
    try:
        # Run pkg-config to get the include directory
        result = subprocess.run(
            ["pkg-config", "--cflags", "eigen3"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            text=True
        )
        return result.stdout.strip().split()[0][2:]  # Remove the leading '-I'
    except subprocess.CalledProcessError:
        print("Eigen not found. Make sure it's installed and pkg-config is configured.")
        return None

#change the directory to the file location
os.chdir(os.path.dirname(os.path.abspath(__file__)))

eigen_include_dir = find_eigen_include_dir()

#copy the src and includes dir locally
if os.path.exists('src'):
    shutil.rmtree('src')
shutil.copytree('../src', 'src')



#get name in src dir
src_files = os.listdir('../src')
src_files = ['src/' + file for file in src_files]
src_files.append("bindings.cpp")

ext_modules = [
    Extension(
        name = 'DecisionTreePP',  # Name of the resulting Python module
        sources=src_files,  # Source file
        include_dirs=[pybind11.get_include(), "../include", eigen_include_dir],  # Include `pybind11` headers
        language='c++',  # This is a C++ extension
        extra_compile_args=['-O3', '-Wall', '-shared', '-fPIC'],  # Optimization flags

    ),
]

setup(
    name='DecisionTreePP',
    version='0.5.0',
    author='N/A',
    author_email='N/A',
    description='Decision Tree C++ implementation',
    long_description='',
    ext_modules=ext_modules,
    cmdclass={'build_ext': build_ext},  # Use `build_ext` imported from `pybind11.setup_helpers`
)

#remove the src dir
shutil.rmtree('src')