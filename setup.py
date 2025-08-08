from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import os, numpy, pufferlib


class BuildExt(build_ext):
    def build_extensions(self):
        puffer_include = os.path.join(os.path.dirname(pufferlib.__file__), 'ocean')
        for ext in self.extensions:
            ext.include_dirs.extend([numpy.get_include(), puffer_include])
        super().build_extensions()


binding = Extension(
    name='four_rooms.binding',
    sources=['four_rooms/binding.c'],
    include_dirs=['.', 'four_rooms'],
    extra_compile_args=['-O3', '-DNO_RAYLIB=1'],
    libraries=[],
)

setup(
    name='four_rooms',
    version='0.0.1',
    packages=['four_rooms'],
    ext_modules=[binding],
    cmdclass={'build_ext': BuildExt},
)


