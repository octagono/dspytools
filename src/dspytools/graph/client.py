"""FalkorDB + Redis client singleton for dspytools.

Provides unified access to FalkorDB (graph queries) and Redis (vector search, caching).
"""

from __future__ import annotations

from typing import Any

import redis
from falkordb import FalkorDB
from redis.exceptions import ResponseError

from dspytools.config.settings import embedder_dimension, load_config
from dspytools.core.errors import ServiceUnavailableError


class GraphClient:
    """Singleton FalkorDB + Redis client with connection pooling.

    FalkorDB handles graph queries (Cypher).
    Redis handles vector search, semantic caching, and JSON storage.
    """

    _instance: GraphClient | None = None

    def __new__(cls) -> GraphClient:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self._initialized = True
        self._cfg = self._load_config()
        self._falkordb = None
        self._redis_obj_obj = None
        self._database = self._cfg.get("database", "dspytools")

    def _ensure_clients(self) -> None:
        """Lazy-init FalkorDB + Redis clients on first use."""
        if self._falkordb is not None:
            return

        cfg = self._cfg
        self._falkordb = FalkorDB(
            host=cfg.get("host", "localhost"),
            port=cfg.get("port", 6379),
            password=cfg.get("password") or None,
        )
        self._redis_obj_obj = redis.Redis(
            host=cfg.get("host", "localhost"),
            port=cfg.get("port", 6379),
            db=cfg.get("db", 0),
            password=cfg.get("password") or None,
            decode_responses=True,
        )

    def _load_config(self) -> dict:
        """Load graph config from dspytools ConfigCache (hot-reload)."""

        return load_config().get("graph", {})

    def graph(self, name: str | None = None) -> Any:
        """Select a named FalkorDB graph."""
        self._ensure_clients()
        if self._falkordb is None:
            raise ServiceUnavailableError("falkordb", "FalkorDB client not initialized")
        graph_name = name or self._database
        return self._falkordb.select_graph(graph_name)

    def ensure_vector_index(
        self, graph_name: str, label: str, property: str, dim: int
    ) -> None:
        """Create a FalkorDB vector index idempotently.

        FalkorDB lacks CREATE VECTOR INDEX IF NOT EXISTS, so we catch
        the "already indexed" error and ignore it. All other errors propagate.
        """
        g = self.graph(graph_name)
        try:
            g.query(
                f"CREATE VECTOR INDEX FOR (n:{label}) ON (n.{property}) "
                f"OPTIONS {{dimension:{dim}, similarityFunction:'cosine'}}"
            )
        except (RuntimeError, OSError, ValueError, ResponseError) as e:
            # FalkorDB raises "already indexed" when index exists — idempotent
            if "already indexed" not in str(e).lower():
                raise

    def redis(self) -> Any:
        """Get Redis client for vector search and caching."""
        self._ensure_clients()
        if self._redis_obj_obj is None:
            raise ServiceUnavailableError("falkordb", "Redis client not initialized")
        return self._redis_obj_obj

    def ping(self) -> bool:
        """Check if Redis/FalkorDB is reachable.

        Raises ConnectionError or Redis exception if unreachable.
        """
        self._ensure_clients()
        if self._redis_obj_obj is None:
            raise ServiceUnavailableError("falkordb", "Redis client not initialized")
        return bool(self._redis_obj_obj.ping())

    def ensure_indexes(self) -> None:
        """Create FalkorDB vector indexes for semantic search.

        All indexes use FalkorDB's native vector index (db.idx.vector.queryNodes).
        No RediSearch/FT.SEARCH dependency — FalkorDB container lacks that module.

        Idempotent: ignores "already indexed" errors.
        """

        dim = embedder_dimension()
        self.ensure_vector_index(self._database, "Skill", "embedding", dim)
        self.ensure_vector_index("llm_cache", "CacheVec", "embedding", dim)
        self.ensure_vector_index("memories", "Memory", "embedding", dim)

    def flush_all(self) -> None:
        """Flush all data (use with caution).

        Raises on failure — does not silently skip any graph deletion.
        """
        self._ensure_clients()
        if self._redis_obj_obj is None:
            raise ServiceUnavailableError("falkordb", "Redis client not initialized")
        if self._falkordb is None:
            raise ServiceUnavailableError("falkordb", "FalkorDB client not initialized")
        self._redis_obj_obj.flushall()
        # Flush all FalkorDB graphs
        graphs = self._falkordb.list_graphs()
        for g in graphs:
            graph = self._falkordb.select_graph(g)
            graph.delete()


def get_graph_client() -> GraphClient:
    """Get GraphClient singleton (already guarded by __new__)."""
    return GraphClient()
