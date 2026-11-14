"""Compiled-program provenance — was this artifact built by this DSPy?

DSPy records the saving interpreter's version in a saved program's
``metadata.dspy_version``. DSPy 3.5 changed prompts, made ``XMLAdapter`` the
default and moved LM calls onto the canonical ``Request``/``Response`` path,
so a program compiled under 3.3.x is not a drop-in: loading one logs a
``dspy.primitives.base_module`` mismatch warning and silently degrades quality
(upstream: "re-evaluate optimized programs").

These helpers let a caller detect the mismatch *before* loading and decide
what to do — recompile, warn, or fall back.
"""

from __future__ import annotations

import json
from pathlib import Path


def saved_dspy_version(path: str | Path) -> str | None:
    """Return ``metadata.dspy_version`` from a saved DSPy program.

    Returns ``None`` when the file is unreadable, is not a saved program, or
    was saved by a DSPy that did not record a version.
    """
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    meta = raw.get("metadata")
    if not isinstance(meta, dict):
        return None
    version = meta.get("dspy_version")
    if isinstance(version, str):
        return version
    # Programs saved by some DSPy builds record it in the dependency map.
    deps = meta.get("dependency_versions")
    if isinstance(deps, dict):
        version = deps.get("dspy")
        if isinstance(version, str):
            return version
    return None


def is_current_dspy_version(path: str | Path, current: str | None = None) -> bool:
    """True when the program matches the running DSPy.

    A program with no recorded version is treated as current — refusing to
    load an unknown artifact would be worse than the warning DSPy emits.
    """
    saved = saved_dspy_version(path)
    if not saved:
        return True
    if current is None:
        from dspytools.core._dspy import dspy

        current = dspy.__version__
    return saved == current
