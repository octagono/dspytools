"""Shared Mojo module loader — eliminates 3× duplicated bridge loading boilerplate."""

from __future__ import annotations

import importlib
import os
import sys
from types import ModuleType
from typing import Any

from dspytools.core.logging_config import get_logger

_log = get_logger(__name__)


def try_load_mojo(
    module_name: str, attr_name: str, logger: Any = None
) -> tuple[bool, ModuleType | None]:
    """Load a Mojo-compiled module and check for an expected attribute.

    Args:
        module_name: The Python import name of the Mojo module (e.g. "sprt").
        attr_name: The expected attribute/function name (e.g. "sprt_evaluate").
        logger: A logger instance (defaults to this module's logger).

    Returns:
        A tuple of (HAS_MOJO, module).  Callers typically:
            HAS_MOJO, _mojo_module = try_load_mojo(...)
    """
    if logger is None:
        logger = _log

    mojo_dir = os.path.normpath(
        os.path.join(
            os.path.dirname(__file__), os.pardir, os.pardir, os.pardir, "mojo_modules"
        )
    )
    if mojo_dir not in sys.path:
        sys.path.insert(0, mojo_dir)

    try:
        # Register the .mojo import hook — only succeeds if mojo SDK is installed
        import mojo.importer  # noqa: F401
    except (ImportError, ModuleNotFoundError):
        pass

    try:
        mod = importlib.import_module(module_name)
    except (ImportError, ModuleNotFoundError, OSError) as e:
        logger.debug("mojo_load_skipped", module=module_name, error=str(e))
        return False, None

    if mod is not None and hasattr(mod, attr_name):
        logger.info("mojo_loaded", module=module_name, attr=attr_name)
        return True, mod

    logger.info("mojo_no_attr", module=module_name, attr=attr_name)
    return False, mod
