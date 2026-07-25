"""Memory module — FalkorDB-native persistent agent memory for dspytools.

Provides entity extraction, deduplication, semantic search, and graph relationships.
"""

from __future__ import annotations

from dspytools.memory.manager import MemoryManager, get_memory_manager

__all__ = [
    "MemoryManager",
    "get_memory_manager",
]
