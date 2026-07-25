"""
Python bridge for Mojo-accelerated vector serialization (Phase 1).

Provides a drop-in replacement for _vec_to_blob with SIMD-accelerated copy.
Falls back to pure NumPy when the Mojo shared library is unavailable.

Usage:
    from dspytools.graph.cache_mojo_bridge import vec_to_blob
    blob = vec_to_blob(numpy_array_of_float32)
"""

from __future__ import annotations

import ctypes
from types import ModuleType
from typing import Optional

import numpy as np

from dspytools.core.logging_config import get_logger
from dspytools.core.mojo_bridge import try_load_mojo

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Mojo module loading — shared utility
# ---------------------------------------------------------------------------

HAS_MOJO: bool = False
"""True when the Mojo module was loaded successfully."""

_mojo_module: Optional[ModuleType] = None
"""Cached reference to the loaded Mojo module."""

HAS_MOJO, _mojo_module = try_load_mojo("vector_utils", "load_store_float32", logger)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def vec_to_blob(arr: np.ndarray, buffer: Optional[bytearray] = None) -> bytes:
    """Serialize a Float32 numpy array to a compact bytes blob.

    Args:
        arr: 1-D numpy array of dtype float32 (contiguous C-order).
        buffer: Optional pre-allocated bytearray for zero-alloc path.
                If provided, must have length >= arr.nbytes. The buffer
                is NOT truncated — only the first arr.nbytes bytes
                represent meaningful data.

    Returns:
        bytes object containing the raw Float32 data. If a buffer was
        provided, returns bytes(memoryview of buffer)[:arr.nbytes].
    """
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32)
    if not arr.flags["C_CONTIGUOUS"]:
        arr = np.ascontiguousarray(arr)

    count = arr.size
    nbytes = arr.nbytes  # count * 4

    if HAS_MOJO and _mojo_module is not None:
        if buffer is None:
            buffer = bytearray(nbytes)
        elif len(buffer) < nbytes:
            raise ValueError(f"buffer too small: {len(buffer)} < {nbytes}")

        src_addr = arr.ctypes.data
        dst_addr = ctypes.addressof(ctypes.c_char.from_buffer(buffer))
        _mojo_module.load_store_float32(src_addr, dst_addr, count)

        return bytes(memoryview(buffer)[:nbytes])
    else:
        # Pure NumPy fallback
        return arr.tobytes()


def has_mojo() -> bool:
    """Check whether the Mojo accelerator is active."""
    return HAS_MOJO
