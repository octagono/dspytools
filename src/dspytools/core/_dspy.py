"""Lazy DSPy import helper — avoids heavy import chain on every CLI invocation.

Use: from dspytools.core._dspy import dspy
Instead of: import dspy  (inline in every function)

The first access triggers the actual import. Subsequent accesses are instant.
"""

from __future__ import annotations

import importlib
from typing import Any


class _LazyDSPy:
    """Lazy import wrapper for dspy — defers heavy LiteLLM import chain."""

    _module: Any = None

    def __getattr__(self, name: str) -> Any:
        if self._module is None:
            self._module = importlib.import_module("dspy")
        return getattr(self._module, name)

    def __setattr__(self, name: str, value: Any) -> None:
        # Internal attributes go on the instance
        if name == "_module":
            object.__setattr__(self, name, value)
            return
        # All other attributes delegate to the real dspy module
        if self._module is None:
            self._module = importlib.import_module("dspy")
        setattr(self._module, name, value)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if self._module is None:
            self._module = importlib.import_module("dspy")
        return self._module(*args, **kwargs)

    def __dir__(self) -> list[str]:
        if self._module is None:
            self._module = importlib.import_module("dspy")
        return dir(self._module)


dspy: Any = _LazyDSPy()
