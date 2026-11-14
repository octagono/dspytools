"""FalkorDB + Redis client singleton for dspytools.

Provides unified access to FalkorDB (graph queries) and Redis (vector search, caching).
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import Any, Self, TypeVar

import redis
from falkordb import FalkorDB
from redis.exceptions import ResponseError

from dspytools.config.settings import embedder_dimension, load_config
from dspytools.core.errors import ServiceUnavailableError

T = TypeVar("T")


class CircuitBreaker:
    """Circuit breaker for external-service calls (FalkorDB/Redis).

    State machine:
      closed    → calls pass through; failures increment a counter
      open      → after ``failure_threshold`` consecutive failures, calls are
                  rejected fast (raise the stored ``error_type``) for
                  ``cooldown_seconds``, giving the service time to recover
      half-open → after the cooldown, one probe call is allowed; success
                  resets to closed, failure re-opens the circuit

    Thread-safe via a lock. This is the resilience layer documented in
    graph/AGENTS.md ("service down doesn't crash the process") — callers that
    can degrade (semantic cache, drift hooks) catch the typed error and skip;
    callers that must fail (explicit ``graph query``) surface it cleanly.
    """

    def __init__(
        self,
        service: str,
        error_type: type[ServiceUnavailableError] = ServiceUnavailableError,
        failure_threshold: int = 3,
        cooldown_seconds: float = 10.0,
    ) -> None:
        self._service = service
        self._error_type = error_type
        self._threshold = failure_threshold
        self._cooldown = cooldown_seconds
        self._failures = 0
        self._state = "closed"  # closed | open | half-open
        self._opened_at = 0.0
        self._lock = threading.Lock()

    @property
    def state(self) -> str:
        return self._state

    def call(self, fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """Run ``fn`` under the breaker. Raises the typed error when open."""
        with self._lock:
            if self._state == "open":
                if time.time() - self._opened_at >= self._cooldown:
                    self._state = "half-open"
                else:
                    raise self._error_type(
                        self._service,
                        f"circuit open after {self._failures} consecutive failures; "
                        f"retry in {self._cooldown - (time.time() - self._opened_at):.0f}s",
                    )
        try:
            result = fn(*args, **kwargs)
        except (ConnectionError, OSError, ResponseError, redis.RedisError) as e:
            self._record_failure()
            raise self._error_type(self._service, f"{type(e).__name__}: {e}") from e
        self._record_success()
        return result

    def _record_failure(self) -> None:
        with self._lock:
            self._failures += 1
            if self._failures >= self._threshold:
                self._state = "open"
                self._opened_at = time.time()

    def _record_success(self) -> None:
        with self._lock:
            self._failures = 0
            self._state = "closed"

    def reset(self) -> None:
        with self._lock:
            self._failures = 0
            self._state = "closed"
            self._opened_at = 0.0


class GraphClient:
    """Singleton FalkorDB + Redis client with connection pooling.

    FalkorDB handles graph queries (Cypher).
    Redis handles vector search, semantic caching, and JSON storage.
    """

    _instance: GraphClient | None = None

    def __new__(cls) -> Self:
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
        self._redis_obj = None
        self._database = self._cfg.get("database", "dspytools")
        # Circuit breakers: one per service. Graph ops trip the GraphError
        # breaker; Redis ops (cache) trip the CacheError breaker.
        from dspytools.core.errors import CacheError, GraphError

        self._graph_breaker = CircuitBreaker("falkordb", GraphError)
        self._redis_breaker = CircuitBreaker("redis", CacheError)

    def _ensure_clients(self) -> None:
        """Lazy-init FalkorDB + Redis clients on first use (breaker-guarded)."""
        if self._falkordb is not None:
            return

        def _init() -> None:
            cfg = self._cfg
            # Reset both to avoid a partially-initialized retry after a
            # failed attempt (FalkorDB connects eagerly; Redis is lazy).
            self._falkordb = None
            self._redis_obj = None
            self._falkordb = FalkorDB(
                host=cfg.get("host", "localhost"),
                port=cfg.get("port", 6379),
                password=cfg.get("password") or None,
            )
            self._redis_obj = redis.Redis(
                host=cfg.get("host", "localhost"),
                port=cfg.get("port", 6379),
                db=cfg.get("db", 0),
                password=cfg.get("password") or None,
                decode_responses=True,
            )

        self._graph_breaker.call(_init)

    def _load_config(self) -> dict:
        """Load FalkorDB/Redis config from dspytools ConfigCache (hot-reload).

        Reads the ``falkordb`` section (host/port/password/database/db), matching
        the graph/AGENTS.md contract. Falls back to localhost:6379 defaults.
        """
        return load_config().get("falkordb", {})

    def graph(self, name: str | None = None) -> Any:
        """Select a named FalkorDB graph (breaker-guarded)."""
        self._ensure_clients()
        if self._falkordb is None:
            raise ServiceUnavailableError("falkordb", "FalkorDB client not initialized")
        graph_name = name or self._database
        return self._graph_breaker.call(self._falkordb.select_graph, graph_name)

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
        """Get the Redis client for vector search and caching.

        Returns the raw (lazy-connecting) client. Redis operations themselves
        should be wrapped via :meth:`cache_breaker` so a down Redis degrades
        to a typed ``CacheError`` instead of a raw connection exception.
        """
        self._ensure_clients()
        if self._redis_obj is None:
            raise ServiceUnavailableError("redis", "Redis client not initialized")
        return self._redis_obj

    def cache_breaker(self) -> CircuitBreaker:
        """Return the CacheError circuit breaker for Redis-backed cache ops.

        Cache consumers (SemanticCache, RedisCache, MCP cache) that can degrade
        call ``breaker.call(fn)`` and catch ``CacheError`` to skip caching when
        Redis is down instead of crashing the request.
        """
        return self._redis_breaker

    def ping(self) -> bool:
        """Check if Redis/FalkorDB is reachable.

        Returns True when reachable. Raises ``GraphError``/``CacheError``
        (breaker-routed) when the service is down or the circuit is open.
        """
        self._ensure_clients()
        if self._redis_obj is None:
            raise ServiceUnavailableError("redis", "Redis client not initialized")
        return self._graph_breaker.call(self._redis_obj.ping)

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
        if self._redis_obj is None:
            raise ServiceUnavailableError("falkordb", "Redis client not initialized")
        if self._falkordb is None:
            raise ServiceUnavailableError("falkordb", "FalkorDB client not initialized")
        self._redis_obj.flushall()
        # Flush all FalkorDB graphs
        graphs = self._falkordb.list_graphs()
        for g in graphs:
            graph = self._falkordb.select_graph(g)
            graph.delete()


def get_graph_client() -> GraphClient:
    """Get GraphClient singleton (already guarded by __new__)."""
    return GraphClient()
