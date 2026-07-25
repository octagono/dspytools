"""AST-based dependency caching for llms.txt generation.

Caches analysis results keyed by file content hashes.
Subsequent runs only re-analyze changed modules.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from dspytools.config.settings import cache_dir
from dspytools.core._io import read_json, write_json
from dspytools.generate.explorer import gather_repository_info


@dataclass
class CacheEntry:
    """Cached analysis result for a single module/file."""

    path: str
    hash: str
    analysis: dict = field(default_factory=dict)
    cached_at: float = field(default_factory=time.time)
    ttl: float = 86400.0  # 24 hours


class AnalysisCache:
    """File-hash-based cache for RepositoryAnalyzer results.

    Keys are SHA-256 hashes of file contents.
    Values are cached analysis outputs.

    Usage:
        cache = AnalysisCache()
        file_hash = cache.hash_file("/path/to/file.py")
        cached = cache.get(file_hash)
        if cached is None:
            result = analyzer.forward(...)
            cache.set(file_hash, result)
    """

    def __init__(self, _cache_path: str | None = None, ttl: float = 86400.0):
        self.cache_dir = Path(_cache_path) if _cache_path else cache_dir()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl = ttl
        self._memory: dict[str, CacheEntry] = {}
        self._load_index()

    def hash_file(self, filepath: str | Path) -> str:
        """Compute SHA-256 hash of a file's contents."""
        path = Path(filepath)
        if not path.exists():
            return ""

        if not path.is_file():
            return ""
        hasher = hashlib.sha256()
        hasher.update(path.read_bytes())
        return hasher.hexdigest()[:16]

    def hash_directory(
        self, directory: str | Path, pattern: str = "*.py"
    ) -> dict[str, str]:
        """Hash all files matching pattern in a directory.

        Returns:
            {relative_path: file_hash} dict
        """
        root = Path(directory)
        if not root.exists():
            return {}

        hashes = {}
        for f in sorted(root.rglob(pattern)):
            if any(p.startswith((".", "__pycache__")) for p in f.parts):
                continue
            rel = f.relative_to(root).as_posix()
            h = self.hash_file(f)
            if h:
                hashes[rel] = h

        return hashes

    def get(self, key: str) -> dict | None:
        """Get cached analysis for a file hash key."""
        # Check memory cache first
        if key in self._memory:
            entry = self._memory[key]
            if time.time() - entry.cached_at < entry.ttl:
                return entry.analysis
            del self._memory[key]

        # Check disk cache
        cache_file = self.cache_dir / f"{key}.json"
        if cache_file.exists():
            try:
                data = read_json(cache_file)
                cached_at = data.get("cached_at", 0)
                if time.time() - cached_at < self.ttl:
                    return data.get("analysis", {})
            except (json.JSONDecodeError, KeyError):
                pass

        return None

    def set(self, key: str, analysis: dict) -> None:
        """Cache analysis result keyed by file hash."""
        entry = CacheEntry(path="", hash=key, analysis=analysis)
        self._memory[key] = entry

        # Persist to disk
        cache_file = self.cache_dir / f"{key}.json"
        write_json(
            cache_file,
            {
                "hash": key,
                "analysis": analysis,
                "cached_at": entry.cached_at,
            },
        )

    def invalidate(self, key: str | None = None) -> int:
        """Invalidate cache entries. If key is None, invalidate all.

        Returns number of entries invalidated.
        """
        if key is None:
            count = len(self._memory)
            self._memory.clear()
            for f in self.cache_dir.glob("*.json"):
                f.unlink()
            return count

        count = 1 if key in self._memory else 0
        self._memory.pop(key, None)
        cache_file = self.cache_dir / f"{key}.json"
        if cache_file.exists():
            cache_file.unlink()
            count = 1
        return count

    def composite_key(self, file_hashes: dict[str, str]) -> str:
        """Create a composite hash key from multiple file hashes.

        This represents the entire repository state as a single key.
        """
        hasher = hashlib.sha256()
        for path, h in sorted(file_hashes.items()):
            hasher.update(path.encode())
            hasher.update(h.encode())
        return hasher.hexdigest()[:24]

    @property
    def stats(self) -> dict:
        return {
            "memory_entries": len(self._memory),
            "disk_entries": len(list(self.cache_dir.glob("*.json"))),
            "cache_dir": str(self.cache_dir),
            "ttl_seconds": self.ttl,
        }

    def warmup(self, paths: list[str]) -> dict:
        """Pre-compute real composite cache keys for local repositories.

        Scans each local path to gather file_tree, readme, and package_files,
        then computes the exact same composite key that forward() will compute.
        This ensures the warmup entry is actually hit on the next analysis run.

        Args:
            paths: List of LOCAL repository paths to pre-register.
                   Remote URLs are not supported (would require cloning).

        Returns:
            {path: composite_key} mapping for successfully registered paths.
        """
        registered: dict[str, str] = {}
        for path_str in paths:
            path = Path(path_str)
            # Fail-fast: path validation is the caller's responsibility
            file_tree, readme, packages, _ = gather_repository_info(str(path))
            # Compute the EXACT same key as RepositoryAnalyzer.forward()
            key_material = f"{path_str}|{file_tree}|{readme}|{packages}"
            composite = hashlib.sha256(key_material.encode()).hexdigest()[:24]
            self.set(
                composite,
                {
                    "_warmup": True,
                    "path": path_str,
                    "cached_at": time.time(),
                },
            )
            registered[path_str] = composite

        return registered

    def _load_index(self) -> None:
        """Load cached hashes from disk on startup."""
        pass

    def __len__(self) -> int:
        return len(self._memory)


# Module-level singleton
_cache: AnalysisCache | None = None


def get_analysis_cache() -> AnalysisCache:
    global _cache
    if _cache is None:
        _cache = AnalysisCache()
    return _cache
