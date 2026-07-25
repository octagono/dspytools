"""Tests for AnalysisCache — AST-based dependency caching for llms.txt generation.

Pure logic tests — no LM calls, no DSPy import needed.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from dspytools.generate.cache import AnalysisCache


def _cache(tmp_path: Path, ttl: float = 86400.0) -> AnalysisCache:
    """Create an AnalysisCache isolated to a temp path."""
    cache_dir = tmp_path / "analysis_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return AnalysisCache(_cache_path=str(cache_dir), ttl=ttl)


# ── hash_file ──────────────────────────────────────────────────────────────


class TestHashFile:
    def test_existing_file_returns_hex(self, tmp_path: Path):
        f = tmp_path / "test.py"
        f.write_text("x = 1")
        c = _cache(tmp_path)
        h = c.hash_file(f)
        assert isinstance(h, str)
        assert len(h) == 16  # truncated SHA-256 hex
        assert all(c in "0123456789abcdef" for c in h)

    def test_nonexistent_file_returns_empty(self, tmp_path: Path):
        c = _cache(tmp_path)
        h = c.hash_file(tmp_path / "nope.py")
        assert h == ""

    def test_same_content_same_hash(self, tmp_path: Path):
        f1 = tmp_path / "a.py"
        f2 = tmp_path / "b.py"
        f1.write_text("content")
        f2.write_text("content")
        c = _cache(tmp_path)
        assert c.hash_file(f1) == c.hash_file(f2)

    def test_different_content_different_hash(self, tmp_path: Path):
        f1 = tmp_path / "a.py"
        f2 = tmp_path / "b.py"
        f1.write_text("content_a")
        f2.write_text("content_b")
        c = _cache(tmp_path)
        assert c.hash_file(f1) != c.hash_file(f2)


# ── get / set round-trip ───────────────────────────────────────────────────


class TestGetSet:
    def test_set_and_get(self, tmp_path: Path):
        c = _cache(tmp_path)
        c.set("key1", {"result": "hello"})
        got = c.get("key1")
        assert got == {"result": "hello"}

    def test_miss_returns_none(self, tmp_path: Path):
        c = _cache(tmp_path)
        assert c.get("nonexistent") is None

    def test_overwrite(self, tmp_path: Path):
        c = _cache(tmp_path)
        c.set("k", {"v": 1})
        c.set("k", {"v": 2})
        assert c.get("k") == {"v": 2}

    def test_multiple_keys(self, tmp_path: Path):
        c = _cache(tmp_path)
        c.set("a", {"x": 1})
        c.set("b", {"y": 2})
        assert c.get("a") == {"x": 1}
        assert c.get("b") == {"y": 2}


# ── composite_key ──────────────────────────────────────────────────────────


class TestCompositeKey:
    def test_same_inputs_same_key(self, tmp_path: Path):
        c = _cache(tmp_path)
        k1 = c.composite_key({"a.py": "hash1", "b.py": "hash2"})
        k2 = c.composite_key({"a.py": "hash1", "b.py": "hash2"})
        assert k1 == k2

    def test_different_inputs_different_key(self, tmp_path: Path):
        c = _cache(tmp_path)
        k1 = c.composite_key({"a.py": "hash1"})
        k2 = c.composite_key({"a.py": "hash2"})
        assert k1 != k2

    def test_sort_order(self, tmp_path: Path):
        """Keys are sorted, so dict insertion order doesn't matter."""
        c = _cache(tmp_path)
        k1 = c.composite_key({"b.py": "h2", "a.py": "h1"})
        k2 = c.composite_key({"a.py": "h1", "b.py": "h2"})
        assert k1 == k2

    def test_length(self, tmp_path: Path):
        c = _cache(tmp_path)
        k = c.composite_key({"x": "y"})
        assert len(k) == 24


# ── invalidate ─────────────────────────────────────────────────────────────


class TestInvalidate:
    def test_single_key(self, tmp_path: Path):
        c = _cache(tmp_path)
        c.set("k", {"v": 1})
        assert c.get("k") is not None
        n = c.invalidate("k")
        assert n == 1
        assert c.get("k") is None

    def test_invalidate_all(self, tmp_path: Path):
        c = _cache(tmp_path)
        c.set("a", {"v": 1})
        c.set("b", {"v": 2})
        assert len(c) == 2
        n = c.invalidate(None)
        assert n == 2
        assert c.get("a") is None
        assert c.get("b") is None

    def test_invalidate_missing_key(self, tmp_path: Path):
        c = _cache(tmp_path)
        n = c.invalidate("nonexistent")
        assert n == 0

    def test_disk_cleanup(self, tmp_path: Path):
        c = _cache(tmp_path)
        c.set("k", {"v": 1})
        disk_files = list(Path(c.cache_dir).glob("*.json"))
        assert len(disk_files) >= 1
        c.invalidate("k")
        assert c.get("k") is None


# ── stats ──────────────────────────────────────────────────────────────────


class TestStats:
    def test_empty_stats(self, tmp_path: Path):
        c = _cache(tmp_path)
        s = c.stats
        assert s["memory_entries"] == 0
        assert s["ttl_seconds"] == 86400.0

    def test_after_set(self, tmp_path: Path):
        c = _cache(tmp_path)
        c.set("k", {"v": 1})
        s = c.stats
        assert s["memory_entries"] >= 1
        assert s["disk_entries"] >= 1

    def test_cache_dir_path(self, tmp_path: Path):
        c = _cache(tmp_path)
        assert "analysis_cache" in c.stats["cache_dir"]


# ── hash_directory ─────────────────────────────────────────────────────────


class TestHashDirectory:
    def test_scans_py_files(self, tmp_path: Path):
        src = tmp_path / "src"
        src.mkdir()
        (src / "mod.py").write_text("x = 1")
        (src / "util.py").write_text("y = 2")
        c = _cache(tmp_path)
        hashes = c.hash_directory(str(src))
        assert "mod.py" in hashes
        assert "util.py" in hashes
        assert len(hashes) == 2

    def test_skips_pycache(self, tmp_path: Path):
        src = tmp_path / "src"
        src.mkdir()
        (src / "good.py").write_text("x = 1")
        pycache = src / "__pycache__"
        pycache.mkdir()
        (pycache / "cache.py").write_text("y = 2")
        c = _cache(tmp_path)
        hashes = c.hash_directory(str(src))
        assert "good.py" in hashes
        assert "__pycache__/cache.py" not in hashes

    def test_skips_dotfiles(self, tmp_path: Path):
        src = tmp_path / "src"
        src.mkdir()
        (src / ".hidden.py").write_text("x = 1")
        (src / "visible.py").write_text("y = 2")
        c = _cache(tmp_path)
        hashes = c.hash_directory(str(src))
        assert "visible.py" in hashes
        assert ".hidden.py" not in hashes

    def test_nonexistent_directory(self, tmp_path: Path):
        c = _cache(tmp_path)
        hashes = c.hash_directory(str(tmp_path / "nope"))
        assert hashes == {}


# ── TTL expiry ──────────────────────────────────────────────────────────────


class TestTtl:
    def test_disk_ttl_expiry(self, tmp_path: Path):
        """Disk cache respects TTL (memory cache uses entry-level TTL)."""
        c = _cache(tmp_path, ttl=0.01)  # 10ms TTL
        c.set("k", {"v": 1})
        assert c.get("k") is not None  # memory cache hit
        # Clear memory cache so next read goes to disk
        c._memory.clear()
        time.sleep(0.02)
        assert c.get("k") is None  # disk TTL expired

    def test_within_ttl(self, tmp_path: Path):
        c = _cache(tmp_path, ttl=60.0)
        c.set("k", {"v": 1})
        assert c.get("k") is not None


# ── Disk persistence ───────────────────────────────────────────────────────


class TestDiskPersistence:
    def test_set_persists_to_disk(self, tmp_path: Path):
        cache_dir = tmp_path / "persist"
        cache_dir.mkdir(parents=True, exist_ok=True)
        c1 = AnalysisCache(_cache_path=str(cache_dir))
        c1.set("persist_key", {"data": "value"})

        # New instance reads from disk
        c2 = AnalysisCache(_cache_path=str(cache_dir))
        got = c2.get("persist_key")
        assert got == {"data": "value"}

    def test_disk_file_format(self, tmp_path: Path):
        c = _cache(tmp_path)
        c.set("fmt", {"x": 1})
        # Find the disk file for key 'fmt'
        disk_files = list(Path(c.cache_dir).glob("*.json"))
        assert len(disk_files) >= 1
        # Each file should have hash, analysis, cached_at
        for f in disk_files:
            data = json.loads(f.read_text())
            assert "hash" in data
            assert "analysis" in data
            assert "cached_at" in data
