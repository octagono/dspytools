"""Shared embedder singleton for SemanticCache + MemoryManager.

Avoids creating duplicate dspy.Embedder instances across modules.
"""

from __future__ import annotations

from typing import Any

_embedder: Any = None


def get_embedder() -> Any:
    """Get or create shared dspy.Embedder instance.

    Singleton per process — prevents redundant HTTP connection pools
    to the embedding server.
    """
    global _embedder
    if _embedder is None:
        from dspytools.config.settings import embedder_kwargs
        from dspytools.core._dspy import dspy

        _embedder = dspy.Embedder(**embedder_kwargs())
    return _embedder


def clear_embedder() -> None:
    """Reset the embedder singleton (for testing)."""
    global _embedder
    _embedder = None
