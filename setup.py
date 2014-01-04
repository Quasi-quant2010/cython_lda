# -*- coding: utf-8 -*-

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy as np
import sys, os, shutil
import itertools


path = os.getenv('PATH').split(':')
ld_library_path = os.getenv('LD_LIBRARY_PATH').split(':')
man_path = os.getenv('MANPATH').split(':')
cplus_include_path = os.getenv('CPLUS_INCLUDE_PATH').split(':')

# clean previous build
for root, dirs, files in os.walk('.',topdown=False):
    for name in files:
        if name.endswith('.so'):
            os.remove(os.path.join(root,name))
    for name in dirs:
        if (name=='build'):
            shutil.rmtree(name)

# build

ext0 = Extension(
    'vocabulary', ['vocabulary.py'],
    include_dirs = [element for element in itertools.chain([np.get_include()], cplus_include_path)],
    library_dirs = [element for element in itertools.chain(ld_library_path, man_path)],
    libraries=['gsl','openblas'],
    extra_compile_args=['-fopenmp','-O3']
    )

ext1 = Extension(
    'memoryview1', ['memoryview1.pyx'],
    include_dirs = [element for element in itertools.chain([np.get_include()], cplus_include_path)],
    library_dirs = [element for element in itertools.chain(ld_library_path, man_path)],
    libraries=['gsl','openblas'],
    extra_compile_args=['-fopenmp','-O3']
    )

ext2 = Extension(
    'pointer1', ['pointer1.pyx'],
    include_dirs = [element for element in itertools.chain([np.get_include()], cplus_include_path)],
    library_dirs = [element for element in itertools.chain(ld_library_path, man_path)],
    libraries=['gsl','openblas'],
    extra_compile_args=['-fopenmp','-O3']
    )

ext3 = Extension(
    'pointer2', ['pointer2.pyx'],
    include_dirs = [element for element in itertools.chain([np.get_include()], cplus_include_path)],
    library_dirs = [element for element in itertools.chain(ld_library_path, man_path)],
    libraries=['gsl','openblas'],
    extra_compile_args=['-fopenmp','-O3']
    )


setup(ext_modules=[ext0,ext1,ext2,ext3],
      cmdclass = {'build_ext': build_ext})
