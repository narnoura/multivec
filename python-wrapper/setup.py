from distutils.core import setup, Extension
import numpy
module = Extension("word2vec", ["wrapper.cpp", "../bivec/word2vec.cpp", "../bivec/bivec.cpp"],
    undef_macros=['NDEBUG'])
module.extra_compile_args = ['--std=c++0x', '-w', '-I../bivec', '-O3']
module.libraries = ['m', 'boost_serialization']
setup(name="word2vec", version="1.0", ext_modules=[module], include_dirs=[numpy.get_include()])
