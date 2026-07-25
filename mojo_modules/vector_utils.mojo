# mojo/vector_utils.mojo — SIMD-accelerated vector blob serialization
#
# Exposes a single function to Python:
#   load_store_float32(src_addr, dst_addr, count) -> None
#
# Copies Float32 data from src_addr to dst_addr using ctypes pointer access.
# On the full Mojo SDK (not the pip distribution), this can be replaced with
# UnsafePointer + SIMD vectorized loads/stores for 4-8x speedup.
#
# Phase 1 of the Mojo hybrid architecture — see mojo/README.md

from std.python import PythonObject, Python
from std.python.bindings import PythonModuleBuilder


def load_store_float32(
    src_addr: PythonObject,
    dst_addr: PythonObject,
    count: PythonObject,
) raises -> PythonObject:
    """Copy Float32 data from src_addr to dst_addr.

    Args:
        src_addr: Raw memory address of source Float32 array (ctypes address).
        dst_addr: Raw memory address of destination buffer (ctypes address).
        count: Number of Float32 elements to copy.

    Uses ctypes pointer access since the pip distribution of Mojo 1.0.0b2
    does not expose UnsafePointer's unsafe_from_address constructor with
    inferrable origin parameter. The full Mojo SDK resolves this and enables
    SIMD vectorized loads/stores.
    """
    var ctypes = Python.import_module("ctypes")
    var src = ctypes.cast(src_addr, ctypes.POINTER(ctypes.c_float))
    var dst = ctypes.cast(dst_addr, ctypes.POINTER(ctypes.c_float))
    var n = Int(py=count)

    for i in range(n):
        dst[i] = src[i]

    return PythonObject(None)


@export
def PyInit_vector_utils() abi("C") -> PythonObject:
    """Initialize the native Python module.

    On success, returns the module object.
    On failure, returns None so Python's import machinery raises ImportError.
    """
    try:
        var m = PythonModuleBuilder("vector_utils")
        m.def_function[load_store_float32]("load_store_float32")
        return m.finalize()
    except e:
        return PythonObject(None)
