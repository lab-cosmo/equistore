# -* coding: utf-8 -*
import os
import sys
from ctypes import cdll

from ._c_api import setup_functions

_HERE = os.path.realpath(os.path.dirname(__file__))

EQUISTORE_LIBRARY_PATH = None


def _set_equistore_library_path(path):
    """
    Set the path of the shared library exporting the equistore functions.

    This is an advanced functionality most users should not need. There can only
    be one call to this function, before trying to create any equistore object.
    """
    global EQUISTORE_LIBRARY_PATH
    if EQUISTORE_LIBRARY_PATH is not None:
        raise ValueError("Trying to set the EQUISTORE library path twice")
    EQUISTORE_LIBRARY_PATH = str(path)


class LibraryFinder(object):
    def __init__(self):
        self._cached_dll = None

    def __call__(self):
        if self._cached_dll is None:
            path = _lib_path()
            self._cached_dll = cdll.LoadLibrary(path)
            setup_functions(self._cached_dll)

            # initial setup, disable printing of the error in case of panic
            # the error will be transformed to a Python exception anyway
            self._cached_dll.eqs_disable_panic_printing()

        return self._cached_dll


def _lib_path():
    global EQUISTORE_LIBRARY_PATH
    if EQUISTORE_LIBRARY_PATH is not None:
        return EQUISTORE_LIBRARY_PATH
    elif sys.platform.startswith("darwin"):
        windows = False
        name = "libequistore.dylib"
    elif sys.platform.startswith("linux"):
        windows = False
        name = "libequistore.so"
    elif sys.platform.startswith("win"):
        windows = True
        name = "equistore.dll"
    else:
        raise ImportError("Unknown platform. Please edit this file")

    path = os.path.join(os.path.join(_HERE, "lib"), name)
    EQUISTORE_LIBRARY_PATH = path

    if os.path.isfile(path):
        if windows:
            _check_dll(path)
        return path

    raise ImportError("Could not find equistore shared library at " + path)


def _check_dll(path):
    """
    Check if the DLL pointer size matches Python (32-bit or 64-bit)
    """
    import platform
    import struct

    IMAGE_FILE_MACHINE_I386 = 332
    IMAGE_FILE_MACHINE_AMD64 = 34404

    machine = None
    with open(path, "rb") as fd:
        header = fd.read(2).decode(encoding="utf-8", errors="strict")
        if header != "MZ":
            raise ImportError(path + " is not a DLL")
        else:
            fd.seek(60)
            header = fd.read(4)
            header_offset = struct.unpack("<L", header)[0]
            fd.seek(header_offset + 4)
            header = fd.read(2)
            machine = struct.unpack("<H", header)[0]

    arch = platform.architecture()[0]
    if arch == "32bit":
        if machine != IMAGE_FILE_MACHINE_I386:
            raise ImportError("Python is 32-bit, but this DLL is not")
    elif arch == "64bit":
        if machine != IMAGE_FILE_MACHINE_AMD64:
            raise ImportError("Python is 64-bit, but this DLL is not")
    else:
        raise ImportError("Could not determine pointer size of Python")


_get_library = LibraryFinder()
