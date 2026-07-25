"""Semantic cache for LLM responses backed by Redis + FalkorDB vector index.

Two-tier cache:
  1. Exact-match: SHA-256 hash of prompt → stored response (O(1) lookup via Redis)
  2. Semantic: FalkorDB native vector index (db.idx.vector.queryNodes, O(log N))

Reduces LLM API costs by caching semantically similar responses.
Uses shared embedder singleton and FalkorDB graph for scalable vector search.
"""

from __future__ import annotations

import hashlib
import json
import os
import time

import numpy as np

from dspytools.config.settings import (
    cache_threshold_path,
    embedder_dimension,
    load_config,
)
from dspytools.core._embedder import get_embedder
from dspytools.core._io import read_json
from dspytools.graph.client import get_graph_client


class SemanticCache:
    """Two-tier LLM response cache backed by Redis + FalkorDB vector index.

    Tier 1: Exact-match via SHA-256 hashed prompt keys (O(1), Redis GET/SETEX).
    Tier 2: Semantic similarity via FalkorDB db.idx.vector.queryNodes (O(log N)).

    Vectors stored as vecf32() on :CacheVec graph nodes. Score = distance
    (lower = more similar). Threshold controls recall precision.

    Tracks hit/miss counters for observability. Threshold is configurable
    via constructor (default 0.15) and can be overridden from config.toml.
    """

    def __init__(
        self,
        name: str = "llm_cache",
        ttl_seconds: int = 3600,
        distance_threshold: float = 0.15,
        model_name: str | None = None,
    ) -> None:

        self._name = name
        self._ttl = ttl_seconds
        self._model_name = model_name
        env_threshold = os.environ.get("DSPYTOOLS_CACHE_THRESHOLD")
        base_threshold = (
            float(env_threshold) if env_threshold is not None else distance_threshold
        )
        # Per-model threshold from config file overrides base threshold
        self._threshold = self._model_threshold(base_threshold)
        client = get_graph_client()
        self._redis = client.redis()
        self._graph = client.graph("llm_cache")
        self._embedder = get_embedder()
        # Create vector index (idempotent — ignores "already indexed")
        dim = embedder_dimension()
        client.ensure_vector_index("llm_cache", "CacheVec", "embedding", dim)
        # Hit/miss counters for observability
        self._hit_count: int = 0
        self._miss_count: int = 0

    def _model_threshold(self, fallback: float) -> float:
        """Load per-model threshold override from config file."""
        path = cache_threshold_path()
        if path.exists():
            data = read_json(path)
            thresholds = data.get("thresholds", {})
            if self._model_name and self._model_name in thresholds:
                return float(thresholds[self._model_name])
            if "default" in thresholds:
                return float(thresholds["default"])
        return fallback

    def adjust_threshold(self, drift_delta: float) -> float:
        """Auto-tune cache threshold based on drift delta.

        Closed-loop feedback: drift monitoring → cache strictness.

        Positive drift_delta (quality dropping) → tighten threshold (more strict)
        Negative/zero drift_delta (stable/improving) → loosen (more cache hits)

        Adjustments are gradual (±10%) to avoid oscillation.
        Clamped to [0.01, 0.5] to stay within useful range.

        Args:
            drift_delta: Degradation percentage (0.05 = 5% drop)

        Returns:
            New threshold value
        """
        if drift_delta > 0.05:
            self._threshold = max(0.01, self._threshold * 0.9)
        elif drift_delta < -0.02:
            self._threshold = min(0.5, self._threshold * 1.1)
        return self._threshold

    @staticmethod
    def _prompt_key(prompt: str) -> str:
        return hashlib.sha256(prompt.encode()).hexdigest()[:32]

    def _embed(self, prompt: str) -> list[float] | None:
        """Embed a prompt using the shared embedder. Fail-fast if embedder is down."""
        vec = self._embedder(prompt)
        return vec.tolist() if isinstance(vec, np.ndarray) else vec

    def check(self, prompt: str) -> dict | None:
        """Check cache. Tier 1 exact → Tier 2 FalkorDB vector similarity.

        Returns cached response dict on hit, None on miss.
        Updates internal hit/miss counters for observability.
        """
        prompt_hash = self._prompt_key(prompt)

        # Tier 1: exact match
        exact_key = f"{self._name}:exact:{prompt_hash}"
        cached = self._redis.get(exact_key)
        if cached:
            self._hit_count += 1
            entry = json.loads(cached)
            return {
                "response": entry["response"],
                "metadata": entry.get("metadata"),
                "distance": 0.0,
                "prompt": prompt,
                "tier": "exact",
                "_vector": entry.get("_vector"),
            }

        # Tier 2: vector similarity via FalkorDB native vector index
        prompt_vec = self._embed(prompt)
        if prompt_vec is None:
            self._miss_count += 1
            return None

        result = self._graph.query(
            """CALL db.idx.vector.queryNodes('CacheVec', 'embedding', 5, vecf32($vec))
               YIELD node, score
               RETURN node.response, node.prompt, node.metadata, score
               ORDER BY score ASC""",
            {"vec": prompt_vec},
        )

        if result.result_set:
            for row in result.result_set:
                distance = row[3]
                if distance <= self._threshold:
                    self._hit_count += 1
                    response_raw = row[0]
                    response_entry = (
                        json.loads(response_raw)
                        if isinstance(response_raw, str)
                        else response_raw
                    )
                    metadata_raw = row[2] or "{}"
                    metadata = (
                        json.loads(metadata_raw)
                        if isinstance(metadata_raw, str)
                        else metadata_raw
                    )
                    return {
                        "response": response_entry.get("response", response_entry)
                        if isinstance(response_entry, dict)
                        else response_entry,
                        "metadata": metadata,
                        "distance": distance,
                        "prompt": row[1] or prompt,
                        "tier": "semantic",
                        "_vector": prompt_vec,
                    }

        self._miss_count += 1
        return None

    def store(
        self,
        prompt: str,
        response: str,
        metadata: dict | None = None,
        _vector: list[float] | None = None,
    ) -> None:
        """Store prompt-response pair in both tiers.

        Args:
            prompt: Input prompt
            response: LLM response text
            metadata: Optional metadata dict
            _vector: Pre-computed embedding (from check() miss). Avoids recomputation.
        """
        prompt_hash = self._prompt_key(prompt)
        entry = {
            "response": response,
            "metadata": metadata or {},
            "stored_at": time.time(),
        }

        # Tier 1: exact match (Redis)
        exact_key = f"{self._name}:exact:{prompt_hash}"
        vec = _vector or self._embed(prompt)
        if vec is not None:
            entry["_vector"] = vec
        entry_json = json.dumps(entry)
        self._redis.setex(exact_key, self._ttl, entry_json)

        # Tier 2: FalkorDB graph node with vecf32() embedding
        if vec is not None:
            self._graph.query(
                """CREATE (c:CacheVec {
                    key: $key,
                    embedding: vecf32($vec),
                    response: $response,
                    prompt: $prompt,
                    metadata: $metadata,
                    stored_at: $stored_at
                })""",
                {
                    "key": f"{self._name}:{prompt_hash}",
                    "vec": vec,
                    "response": entry_json,
                    "prompt": prompt,
                    "metadata": json.dumps(metadata or {}),
                    "stored_at": str(time.time()),
                },
            )

    def _scan_exact_keys(self) -> list[str]:
        """SCAN all exact-match keys for this cache (non-blocking, cursor-based)."""
        keys: list[str] = []
        cursor = 0
        while True:
            cursor, batch = self._redis.scan(
                cursor, match=f"{self._name}:exact:*", count=100
            )
            keys.extend(batch)
            if cursor == 0:
                break
        return keys

    def clear(self) -> None:
        """Clear all cached entries from both tiers."""
        keys = self._scan_exact_keys()
        if keys:
            self._redis.delete(*keys)
        self._graph.query("MATCH (c:CacheVec) DETACH DELETE c")

    def count(self) -> int:
        """Get number of cached entries across both tiers."""
        exact_count = len(self._scan_exact_keys())
        result = self._graph.query("MATCH (c:CacheVec) RETURN count(c)")
        vec_count = result.result_set[0][0] if result.result_set else 0
        return exact_count + vec_count

    def stats(self) -> dict:
        """Get cache statistics."""
        exact_count = 0
        cursor = 0
        while True:
            cursor, keys = self._redis.scan(
                cursor, match=f"{self._name}:exact:*", count=100
            )
            exact_count += len(keys)
            if cursor == 0:
                break
        vec_result = self._graph.query("MATCH (c:CacheVec) RETURN count(c)")
        vec_count = vec_result.result_set[0][0] if vec_result.result_set else 0
        total_requests = self._hit_count + self._miss_count
        hit_rate = (
            (self._hit_count / total_requests * 100) if total_requests > 0 else 0.0
        )
        return {
            "name": self._name,
            "exact_entries": exact_count,
            "semantic_entries": vec_count,
            "total_entries": exact_count + vec_count,
            "size_bytes": 0,  # FalkorDB doesn't expose per-node memory
            "hit_count": self._hit_count,
            "miss_count": self._miss_count,
            "total_requests": total_requests,
            "hit_rate_pct": round(hit_rate, 1),
        }


_cache: SemanticCache | None = None


def get_semantic_cache() -> SemanticCache:
    """Get or create semantic cache singleton.

    Reads the configured student model name so per-model cache
    thresholds from config are applied automatically.
    """
    global _cache
    if _cache is None:
        cfg = load_config()
        model_name = None
        student = cfg.get("lm", {}).get("student")
        if student and isinstance(student, dict):
            model_name = student.get("model")
        _cache = SemanticCache(model_name=model_name)
    return _cache
