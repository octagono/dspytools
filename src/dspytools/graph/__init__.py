"""Graph module — FalkorDB + Redis integration for dspytools.

Provides graph-backed skill dependencies, semantic caching, and vector search.
"""

from __future__ import annotations

from dspytools.graph.client import GraphClient, get_graph_client
from dspytools.graph.skill_graph import FalkorDBSkillGraph

__all__ = [
    "GraphClient",
    "get_graph_client",
    "FalkorDBSkillGraph",
]
