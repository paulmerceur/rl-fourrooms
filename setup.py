from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import os, numpy, pufferlib


class BuildExt(build_ext):
    def build_extensions(self):
        puffer_include = os.path.join(os.path.dirname(pufferlib.__file__), 'ocean')
        for ext in self.extensions:
            ext.include_dirs.extend([numpy.get_include(), puffer_include])
        super().build_extensions()


def get_raylib_paths():
    inc = os.environ.get('RAYLIB_INCLUDE_DIR')
    lib = os.environ.get('RAYLIB_LIB_DIR')
    include_dirs = ['.', 'four_rooms']
    library_dirs = []

    if inc:
        include_dirs.append(inc)
    if lib:
        library_dirs.append(lib)

    # Heuristic for Homebrew on Apple Silicon
    if os.name == 'posix' and not inc and os.path.isdir('/opt/homebrew/include'):
        include_dirs.append('/opt/homebrew/include')
    if os.name == 'posix' and not lib and os.path.isdir('/opt/homebrew/lib'):
        library_dirs.append('/opt/homebrew/lib')

    return include_dirs, library_dirs


inc_dirs, lib_dirs = get_raylib_paths()

binding = Extension(
    name='four_rooms.binding',
    sources=['four_rooms/binding.c'],
    include_dirs=inc_dirs,
    library_dirs=lib_dirs,
    extra_compile_args=['-O3'],
    libraries=['raylib'],
    extra_link_args=[
        '-framework', 'Cocoa',
        '-framework', 'IOKit',
        '-framework', 'CoreVideo',
        '-framework', 'OpenGL',
    ],
)

setup(
    name='four_rooms',
    version='0.0.1',
    packages=['four_rooms'],
    ext_modules=[binding],
    cmdclass={'build_ext': BuildExt},
)


