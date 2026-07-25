"""General-purpose Redis cache for dspytools.

Provides namespaced, TTL-aware caching backed by the Redis Stack instance.
Used for MCP tool responses, compile results, and cross-process shared state.
"""

from __future__ import annotations

import json
import random
from typing import Any


class RedisCache:
    """Redis-backed cache with namespace isolation and TTL support.

    Keys are prefixed with `{namespace}:` to avoid collisions with
    FalkorDB graph data and semantic cache entries.
    """

    def __init__(
        self,
        namespace: str = "dspytools",
        default_ttl: int = 300,
        max_entries: int = 1024,
    ) -> None:
        self._ns = namespace
        self._default_ttl = default_ttl
        self._max = max_entries
        self._redis = None  # Lazy — connected on first use
        self._write_count = 0  # Probabilistic eviction counter

    def _get_redis(self):
        """Lazy Redis connection — connects on first use."""
        if self._redis is None:
            from dspytools.graph.client import get_graph_client

            self._redis = get_graph_client().redis()
        return self._redis

    def _key(self, key: str) -> str:
        """Namespace a key."""
        return f"{self._ns}:{key}"

    def get(self, key: str) -> Any | None:
        """Get a cached value. Returns None on miss."""
        raw = self._get_redis().get(self._key(key))
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return raw

    def get_str(self, key: str) -> str | None:
        """Get a cached string value."""
        raw = self._get_redis().get(self._key(key))
        return raw if raw else None

    def set(
        self,
        key: str,
        value: Any,
        ttl: int | None = None,
    ) -> None:
        """Set a cached value with optional TTL override."""
        ttl = ttl if ttl is not None else self._default_ttl
        ttl = int(ttl)  # Redis setex requires integer TTL
        if isinstance(value, str):
            serialized = value
        else:
            serialized = json.dumps(value, default=str)
        self._get_redis().setex(self._key(key), ttl, serialized)
        # Probabilistic eviction — only check every 50th write (avoids full SCAN per write)
        self._write_count += 1
        if self._write_count % 50 == 0:
            self._evict_if_needed()

    def delete(self, key: str) -> bool:
        """Delete a cached key. Returns True if key existed."""
        return bool(self._get_redis().delete(self._key(key)))

    def exists(self, key: str) -> bool:
        """Check if a key exists."""
        return bool(self._get_redis().exists(self._key(key)))

    def ttl(self, key: str) -> int:
        """Get remaining TTL in seconds. Returns -1 (no expiry) or -2 (missing)."""
        return self._get_redis().ttl(self._key(key))

    def expire(self, key: str, ttl: int) -> bool:
        """Update TTL on an existing key."""
        return bool(self._get_redis().expire(self._key(key), ttl))

    def _scan_keys(self, pattern: str = "*", count: int = 200) -> list[str]:
        """Iterate keys using SCAN cursor instead of blocking KEYS."""
        full = self._key(pattern)
        cursor = 0
        keys: list[str] = []
        while True:
            cursor, batch = self._get_redis().scan(cursor, match=full, count=count)
            keys.extend(batch)
            if cursor == 0:
                break
        prefix = f"{self._ns}:"
        return [k[len(prefix) :] if k.startswith(prefix) else k for k in keys]

    def keys(self, pattern: str = "*") -> list[str]:
        """List keys matching a pattern (within this namespace)."""
        return self._scan_keys(pattern)

    def flush(self) -> int:
        """Delete all keys in this namespace. Returns count deleted."""
        keys = self._scan_keys("*")
        if keys:
            full_keys = [self._key(k) for k in keys]
            return self._get_redis().delete(*full_keys)
        return 0

    def count(self) -> int:
        """Count keys in this namespace."""
        return len(self._scan_keys("*"))

    def memory_usage(self) -> int:
        """Get total memory usage in bytes for this namespace."""
        total = 0
        for k in self._scan_keys("*"):
            full_key = self._key(k)
            usage = self._get_redis().memory_usage(full_key)
            if usage:
                total += usage
        return total

    def _evict_if_needed(self) -> None:
        """Evict oldest entries when over capacity.

        Uses SCAN cursor to avoid blocking Redis.
        When over capacity, evicts a random sample of excess keys
        (instead of sorting all keys by TTL).
        """
        keys = self._scan_keys("*")
        if len(keys) <= self._max:
            return
        excess = len(keys) - self._max
        sample_size = min(excess * 2, len(keys))
        sample = random.sample(keys, sample_size)
        scored = []
        for k in sample:
            t = self._get_redis().ttl(self._key(k))
            scored.append((t, k))
        scored.sort()
        to_delete = [self._key(k) for _, k in scored[:excess]]
        if to_delete:
            self._get_redis().delete(*to_delete)

    def stats(self) -> dict:
        """Return cache statistics."""
        # Use SCAN instead of blocking KEYS (Item 1 fix)
        keys = [self._key(k) for k in self._scan_keys("*")]
        total_bytes = 0
        ttl_sum = 0
        ttl_count = 0
        for k in keys:
            usage = self._get_redis().memory_usage(k)
            if usage:
                total_bytes += usage
            t = self._get_redis().ttl(k)
            if t >= 0:
                ttl_sum += t
                ttl_count += 1
        avg_ttl = ttl_sum / ttl_count if ttl_count else 0
        return {
            "namespace": self._ns,
            "entries": len(keys),
            "max_entries": self._max,
            "memory_bytes": total_bytes,
            "memory_human": f"{total_bytes / 1024:.1f}KB",
            "default_ttl": self._default_ttl,
            "avg_ttl_remaining": round(avg_ttl, 1),
        }


# Module-level singleton cache for MCP tool responses
_mcp_cache: RedisCache | None = None


def get_mcp_cache() -> RedisCache:
    """Get the MCP tool response cache singleton."""
    global _mcp_cache
    if _mcp_cache is None:
        _mcp_cache = RedisCache(namespace="mcp", default_ttl=5, max_entries=256)
    return _mcp_cache


# Module-level singleton for compile results
_compile_cache: RedisCache | None = None


def get_compile_cache() -> RedisCache:
    """Get the compile result cache singleton."""
    global _compile_cache
    if _compile_cache is None:
        _compile_cache = RedisCache(
            namespace="compile", default_ttl=3600, max_entries=64
        )
    return _compile_cache
