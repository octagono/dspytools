"""SSOT file I/O utilities for JSON reading and writing — fail-fast.

CONTRACT: All dspytools modules that read/write JSON files MUST use these utilities.
No module may call json.loads(Path(...).read_text()) directly.

Atomic writes via temp + rename. No error-swallowing defaults — fail fast
on corrupt files so bugs are caught immediately.
"""

from __future__ import annotations

import json as _json
import os
import tempfile
from pathlib import Path
from typing import Any

from dspytools.core.logging_config import get_logger

_log = get_logger(__name__)


def read_json(path: str | Path) -> Any:
    """Read and parse a JSON file. Fail-fast on corrupt or missing files.

    Raises:
        FileNotFoundError: File does not exist.
        json.JSONDecodeError: File is corrupt.
        OSError: IO errors.
    """
    p = Path(path)
    return _json.loads(p.read_text(encoding="utf-8"))


def try_read_json(path: str | Path, default: Any = None) -> Any:
    """Read JSON safely, returning default on any failure.

    Use ONLY for optional/cache files where missing data is acceptable.
    Use read_json() for critical program state.
    """
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return default
    try:
        return _json.loads(p.read_text(encoding="utf-8"))
    except (_json.JSONDecodeError, OSError, UnicodeDecodeError):
        return default


def write_json(path: str | Path, data: Any, indent: int = 2) -> None:
    """Write data as JSON atomically (temp file + rename). Fail-fast."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    content = _json.dumps(data, indent=indent, default=str)
    fd, tmp = tempfile.mkstemp(dir=str(p.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
        os.replace(tmp, str(p))
    except (OSError, ValueError, TypeError) as e:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        _log.warning("write_json_failed", error=str(e))
        raise


def read_text(path: str | Path) -> str:
    """Read a text file. Fail-fast on missing files or encoding errors.

    Raises:
        FileNotFoundError: File does not exist.
        OSError: IO errors.
        UnicodeDecodeError: Encoding issues.
    """
    return Path(path).read_text(encoding="utf-8")


def try_read_text(path: str | Path, default: str = "") -> str:
    """Read a text file safely, returning default on any failure.

    Use ONLY for optional files.
    """
    p = Path(path)
    if not p.exists():
        return default
    try:
        return p.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return default
