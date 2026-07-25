"""Tests for SemanticCache — Redis + FalkorDB two-tier LLM response cache.

Uses mocked Redis (exact-match tier) and FalkorDB graph (semantic tier).
No actual Redis/FalkorDB connection needed.
"""

from __future__ import annotations

import json
import time
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def mock_cache():
    """Create a SemanticCache with mocked Redis, graph, and embedder.

    Returns (cache, mock_redis, mock_graph, mock_embedder) for assertions.

    Patches at the LOCAL binding site (``dspytools.graph.cache.*``), not at the
    definition site, because ``from X import Y`` creates a local binding that
    ``mock.patch("X.Y")`` does not reach.
    """
    with (
        patch("dspytools.graph.cache.get_graph_client") as mock_client,
        patch("dspytools.graph.cache.get_embedder") as mock_embedder,
        patch("dspytools.graph.cache.embedder_dimension", return_value=768),
    ):
        mock_redis = MagicMock()
        mock_graph = MagicMock()
        mock_client.return_value.redis.return_value = mock_redis
        mock_client.return_value.graph.return_value = mock_graph

        mock_emb = MagicMock()
        mock_emb.return_value = [0.1] * 768
        mock_embedder.return_value = mock_emb

        from dspytools.graph.cache import SemanticCache

        cache = SemanticCache(
            name="test_cache", ttl_seconds=60, distance_threshold=0.15
        )
        yield cache, mock_redis, mock_graph, mock_emb


def _entry(response: str, metadata: dict | None = None) -> str:
    return json.dumps(
        {
            "response": response,
            "metadata": metadata or {},
            "stored_at": time.time(),
        }
    )


def _mock_graph_result(rows: list) -> MagicMock:
    """Create a mock graph query result with result_set."""
    result = MagicMock()
    result.result_set = rows
    return result


# ── check: exact match ─────────────────────────────────────────────────────


class TestCheckExact:
    def test_exact_hit(self, mock_cache):
        cache, redis, graph, _ = mock_cache
        redis.get.return_value = _entry("cached_value")

        result = cache.check("hello world")
        assert result is not None
        assert result["response"] == "cached_value"
        assert result["distance"] == 0.0
        assert result["tier"] == "exact"

    def test_exact_miss(self, mock_cache):
        cache, redis, graph, _ = mock_cache
        redis.get.return_value = None
        graph.query.return_value = _mock_graph_result([])

        result = cache.check("unknown prompt")
        assert result is None

    def test_exact_with_metadata(self, mock_cache):
        cache, redis, graph, _ = mock_cache
        redis.get.return_value = _entry("val", {"source": "test"})

        result = cache.check("prompt")
        assert result is not None
        assert result["metadata"] == {"source": "test"}

    def test_exact_key_format(self, mock_cache):
        cache, redis, graph, _ = mock_cache
        redis.get.return_value = _entry("r")

        cache.check("test_prompt")
        called_key = redis.get.call_args[0][0]
        assert called_key.startswith("test_cache:exact:")
        assert len(called_key.split(":")[-1]) == 32


# ── check: semantic match ──────────────────────────────────────────────────


class TestCheckSemantic:
    def test_semantic_hit(self, mock_cache):
        cache, redis, graph, _ = mock_cache
        redis.get.return_value = None  # exact miss
        # FalkorDB returns (response, prompt, metadata, score)
        # score = distance (lower = more similar); 0.05 <= 0.15 threshold
        graph.query.return_value = _mock_graph_result(
            [
                [_entry("semantic_val"), "similar prompt", "{}", 0.05],
            ]
        )

        result = cache.check("close prompt")
        assert result is not None
        assert result["tier"] == "semantic"
        assert result["distance"] <= 0.15

    def test_semantic_outside_threshold(self, mock_cache):
        cache, redis, graph, _ = mock_cache
        redis.get.return_value = None
        # distance 0.50 > 0.15 threshold → miss
        graph.query.return_value = _mock_graph_result(
            [
                [_entry("far"), "far prompt", "{}", 0.50],
            ]
        )

        result = cache.check("different prompt")
        assert result is None

    def test_semantic_no_results(self, mock_cache):
        cache, redis, graph, _ = mock_cache
        redis.get.return_value = None
        graph.query.return_value = _mock_graph_result([])

        result = cache.check("any prompt")
        assert result is None

    def test_embedder_failure(self, mock_cache):
        cache, redis, graph, embedder = mock_cache
        redis.get.return_value = None
        embedder.return_value = None

        result = cache.check("prompt")
        assert result is None

    def test_semantic_returns_vector(self, mock_cache):
        cache, redis, graph, _ = mock_cache
        redis.get.return_value = None
        graph.query.return_value = _mock_graph_result(
            [
                [_entry("v"), "similar", "{}", 0.05],
            ]
        )

        result = cache.check("prompt")
        assert result is not None
        assert "_vector" in result
        assert isinstance(result["_vector"], list)


# ── store ──────────────────────────────────────────────────────────────────


class TestStore:
    def test_stores_both_tiers(self, mock_cache):
        cache, redis, graph, _ = mock_cache
        cache.store("my prompt", "my response")

        # Tier 1: Redis setex
        assert redis.setex.call_count >= 1
        # Tier 2: FalkorDB graph query (CREATE node)
        assert graph.query.call_count >= 1

    def test_store_with_vector_skip(self, mock_cache):
        cache, redis, graph, embedder = mock_cache
        cache.store("prompt", "response", _vector=[0.5] * 768)

        assert embedder.call_count == 0

    def test_store_with_metadata(self, mock_cache):
        cache, redis, graph, _ = mock_cache
        cache.store("prompt", "response", metadata={"source": "test"})

        setex_call = redis.setex.call_args
        stored = json.loads(setex_call[0][2])
        assert stored["metadata"] == {"source": "test"}

    def test_store_embedder_failure(self, mock_cache):
        cache, redis, graph, embedder = mock_cache
        embedder.return_value = None
        cache.store("prompt", "response")


# ── clear ──────────────────────────────────────────────────────────────────


class TestClear:
    def test_clear_deletes_keys(self, mock_cache):
        cache, redis, graph, _ = mock_cache
        redis.scan.side_effect = [(0, ["test_cache:exact:k1", "test_cache:exact:k2"])]

        cache.clear()
        # Redis exact keys deleted
        redis.delete.assert_called_once_with(
            "test_cache:exact:k1", "test_cache:exact:k2"
        )
        # FalkorDB nodes deleted
        assert graph.query.call_count >= 1

    def test_clear_empty(self, mock_cache):
        cache, redis, graph, _ = mock_cache
        redis.scan.side_effect = [(0, [])]

        cache.clear()


# ── stats ──────────────────────────────────────────────────────────────────


class TestStats:
    def test_stats_structure(self, mock_cache):
        cache, redis, graph, _ = mock_cache
        redis.scan.side_effect = [(0, ["k1"])]
        # graph.query is called for count — mock returns 1 vec entry
        graph.query.return_value = _mock_graph_result([[1]])

        s = cache.stats()
        assert s["name"] == "test_cache"
        assert s["exact_entries"] == 1
        assert s["semantic_entries"] == 1
        assert s["total_entries"] == 2
        assert "hit_count" in s
        assert "miss_count" in s

    def test_stats_empty_cache(self, mock_cache):
        cache, redis, graph, _ = mock_cache
        redis.scan.side_effect = [(0, [])]
        graph.query.return_value = _mock_graph_result([[0]])

        s = cache.stats()
        assert s["total_entries"] == 0


class TestCount:
    def test_count(self, mock_cache):
        cache, redis, graph, _ = mock_cache
        redis.scan.side_effect = [(0, ["e1", "e2"])]
        graph.query.return_value = _mock_graph_result([[1]])

        assert cache.count() == 3

    def test_count_empty(self, mock_cache):
        cache, redis, graph, _ = mock_cache
        redis.scan.side_effect = [(0, [])]
        graph.query.return_value = _mock_graph_result([[0]])

        assert cache.count() == 0
