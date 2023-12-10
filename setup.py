""" This module is based on and modified from:
https://github.com/sdpa-python/sdpa-python/blob/main/setup.py
"""
import os
import platform
import sys
from distutils.command.build_ext import build_ext
from pathlib import Path

from pybind11.setup_helpers import Pybind11Extension as Extension  # type: ignore
from setuptools import setup

# from setuptools.command.build_ext import build_ext

ON_WINDOWS = platform.system() == "Windows"
ON_MACOS = platform.system() == "Darwin"

if ON_WINDOWS:
    if "mingw64\\bin" not in os.environ["PATH"]:
        raise RuntimeError(
            "On Windows, the binaries of MinGW are required to compile the extensions. "
            "Please, add the corresponding directory to your PATH environment variable."
            " For example, if you installed MinGW using MSYS2 with standard settings, "
            "C:\\msys64\\mingw64\\bin should be added to your PATH env. variable."
        )


USEGMP = False

# Paths that depend on how dependencies like SDPA and MUMPS have been installed.
SDPA_DIR = os.environ.get("SDPA_DIR", str(Path("path/to/folder/sdpa-7.3.17")))
# FIXME: If we are building a source dist. (python -m build --sidst), SDPA_DIR is not needed.
# assert Path(SDPA_DIR).is_dir(), f"SDPA_DIR is not a folder: {SDPA_DIR}"

MUMPS_DIR = os.path.join(SDPA_DIR, "mumps", "build")

if ON_WINDOWS:
    MSYS2_ROOT = os.environ.get("MSYS2_ROOT", os.path.join("C:\\", "msys64"))
    if sys.maxsize > 2 * 32:
        # 64-bit Python
        MINGW_LIBS = os.path.join(MSYS2_ROOT, "mingw64", "lib")
    else:
        # 32-bit Python
        MINGW_LIBS = os.path.join(MSYS2_ROOT, "mingw32", "lib")
    # MINGW_LIBS = os.environ.get(
    #     "MINGW_LIBS",
    #     os.path.join("C:\\", "msys64", "mingw64", "lib"),
    # )
    assert Path(SDPA_DIR).is_dir(), f"SDPA_DIR is not a folder: {SDPA_DIR}"
    assert Path(MINGW_LIBS).is_dir(), f"MINGW_LIBS is not a folder: {MINGW_LIBS}"

# We use BLAS/LAPACK from Accelerate on MacOS, while on Linux/Windows we use OpenBLAS.
BLAS_LAPACK_LIBS = [] if ON_MACOS else ["openblas", "gomp"]
# gfortran on MacOS is installed and symlinked in a non-standard location.
GFORTRAN_LIBS = "/usr/local/gfortran/lib"


def pjoin(parent, children):
    return [os.path.join(parent, child) for child in children]


libraries = ["sdpa", "dmumps", "mumps_common", "pord", "mpiseq"] + BLAS_LAPACK_LIBS
library_dirs = [SDPA_DIR] + pjoin(MUMPS_DIR, ["lib", "libseq"])
include_dirs = [SDPA_DIR] + pjoin(MUMPS_DIR, ["include"])

if ON_MACOS:
    extra_objects = pjoin(GFORTRAN_LIBS, ["libgfortran.a", "libquadmath.a"])
else:
    libraries += ["gfortran", "quadmath"]
    extra_objects = []

if ON_WINDOWS:
    import distutils.cygwinccompiler

    distutils.cygwinccompiler.get_msvcr = lambda: []
    library_dirs += [MINGW_LIBS]  # type: ignore

extra_link_args = (
    ["-static"] if ON_WINDOWS else ["-framework", "Accelerate"] if ON_MACOS else []
)

ext_modules = [
    Extension(
        "nonmin_pose.sdp_zhao",
        ["src/sdp_zhao.cpp"],
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=libraries,
        extra_objects=extra_objects,
        extra_compile_args=["-DUSEGMP=1"] if USEGMP else ["-DUSEGMP=0"],
        extra_link_args=extra_link_args,
        cxx_std="11" if ON_MACOS else None,
    ),
    Extension(
        "nonmin_pose.sdpa",
        ["src/sdpa.cpp"],
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=libraries,
        extra_objects=extra_objects,
        extra_compile_args=["-DUSEGMP=1"] if USEGMP else ["-DUSEGMP=0"],
        extra_link_args=extra_link_args,
        cxx_std="11" if ON_MACOS else None,
    ),
]


class CustomBuildExtCommand(build_ext):
    def initialize_options(self):
        super().initialize_options()
        if ON_WINDOWS:
            # set compiler type to mingw32.
            self.compiler = "mingw32"


if ON_WINDOWS:
    # Remove the MSVC specific flags that Pybind11 adds (we use MINGW compiler type).
    for ext in ext_modules:
        ext.extra_compile_args = [
            x for x in ext.extra_compile_args if x not in ("/EHsc", "/bigobj")
        ]


setup_args = {
    "ext_modules": ext_modules,
    "cmdclass": {"build_ext": CustomBuildExtCommand},
}
setup(**setup_args)
