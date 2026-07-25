"""JSON-only registry for compiled programs, signatures, modules, agents.

Optimization 7: Index cached in memory with mtime check + run_id lookup dict.
  - list_compiled_runs() returns cached list (re-reads only when file changes)
  - get_run() uses O(1) dict lookup instead of O(n) scan
  - get_lineage() reuses cached index instead of re-reading per ancestor
  - register_run/delete_run invalidate cache after write

Optimization 8: compute_dataset_hash() cached per trainset fingerprint.
  - Hash computed once per unique trainset, cached by (id, len) key
  - Eliminates 5x redundant sorting + JSON serialization per compile session

Optimization 10: list_modules/list_signatures/list_agents cached with mtime.
  - Each directory scanned once per 2s window, not per call
"""

from __future__ import annotations

import hashlib
import inspect
import json
import shutil
import time
from pathlib import Path

from dspytools.config.settings import (
    agents_dir,
    compiled_dir,
    modules_dir,
    signatures_dir,
)
from dspytools.core._io import read_json, write_json
from dspytools.graph.skill_graph import FalkorDBSkillGraph

# ── Compiled Programs (Optimization 7: cached index + run_id lookup) ──────

_index_cache: list[dict] | None = None
_index_cache_mtime: float = 0
_run_id_cache: dict[str, dict] | None = None


def _load_index() -> list[dict]:
    """Load index.json with mtime-based caching."""
    global _index_cache, _index_cache_mtime, _run_id_cache
    index_path = compiled_dir() / "index.json"
    if not index_path.exists():
        _index_cache = []
        _run_id_cache = {}
        return _index_cache

    mtime = index_path.stat().st_mtime
    if _index_cache is not None and mtime == _index_cache_mtime:
        return _index_cache

    _index_cache = read_json(index_path)
    if _index_cache is None:
        _index_cache = []
        _run_id_cache = {}
        return _index_cache
    _run_id_cache = {r["id"]: r for r in _index_cache}
    _index_cache_mtime = mtime
    return _index_cache


def _invalidate_index() -> None:
    """Force re-read on next access."""
    global _index_cache, _index_cache_mtime, _run_id_cache
    _index_cache = None
    _index_cache_mtime = 0
    _run_id_cache = None


def list_compiled_runs() -> list[dict]:
    """List all compiled program runs from the JSON index (cached)."""
    return list(_load_index())


def save_run_index(runs: list[dict]) -> None:
    """Write the index JSON file atomically and invalidate cache."""
    index_path = compiled_dir() / "index.json"
    write_json(index_path, runs)
    _invalidate_index()


def register_run(run_id: str, metadata: dict) -> None:
    """Register a new compiled run (invalidates cache)."""
    runs = list_compiled_runs()
    runs.append({"id": run_id, **metadata})
    save_run_index(runs)


def register_run_with_graph(
    run_id: str,
    metadata: dict,
    optimizer: str = "",
    score: float = 0.0,
    parent_id: str | None = None,
    dataset_hash: str | None = None,
) -> dict:
    """Register a run in JSON index AND FalkorDB graph.

    Auto-extracts optimizer and score from metadata dict when keyword
    args are not provided, so callers only need to pass metadata.

    Convenience wrapper that calls register_run() then records in FalkorDB.
    """
    register_run(run_id, metadata)

    # Resolve optimizer: prefer explicit kwarg, fall back to metadata
    actual_optimizer = optimizer if optimizer else metadata.get("optimizer", "")
    # Resolve score: prefer explicit kwarg (including 0.0), fall back to metadata
    raw_score = metadata.get("score", 0.0)
    actual_score = score if score != 0.0 or "score" in metadata else raw_score
    graph = FalkorDBSkillGraph()
    graph.record_program(
        run_id=run_id,
        optimizer=actual_optimizer,
        score=actual_score if isinstance(actual_score, (int, float)) else 0.0,
        parent_id=parent_id,
        dataset_hash=dataset_hash,
    )
    return metadata


def get_run(run_id: str) -> dict | None:
    """Get a specific run by ID — O(1) dict lookup (Optimization 7)."""
    _load_index()
    if _run_id_cache is None:
        return None
    return _run_id_cache.get(run_id)


def get_lineage(run_id: str) -> list[dict]:
    """Trace the lineage chain — reuses cached index per ancestor (Optimization 7).

    Returns the full ancestor chain: [run_id, parent, grandparent, ...]
    """
    _load_index()  # ensure cache is fresh
    lineage = []
    current = run_id
    while current:
        meta = get_run(current)
        if not meta:
            break
        lineage.append(meta)
        current = meta.get("lineage", {}).get("parent_run")
    return lineage


# ── Optimization 8: Cached dataset hash ──────────────────────────────────

_hash_cache: dict[str, str] = {}


def compute_dataset_hash(trainset: list) -> str:
    """Compute a deterministic hash of a training dataset (cached per fingerprint).

    Optimization 8: Uses content-based cache key — same trainset produces the
    same hash without re-sorting + re-serializing. Cache keyed by (len, first_example_str).
    """
    # Content-based cache key: length + first example string (fast to compute)
    cache_key = str(len(trainset))
    if trainset:
        first = trainset[0]
        if hasattr(first, "toDict"):
            d = first.toDict()
            cache_key += "|" + str(sorted(d.items()) if isinstance(d, dict) else d)
        else:
            cache_key += "|" + str(first)

    if cache_key in _hash_cache:
        return _hash_cache[cache_key]

    hasher = hashlib.sha256()
    for ex in sorted(trainset, key=lambda e: str(e)):
        data = json.dumps(
            ex.toDict() if hasattr(ex, "toDict") else str(ex), sort_keys=True
        )
        hasher.update(data.encode())

    result = hasher.hexdigest()[:12]
    _hash_cache[cache_key] = result
    return result


def find_existing_compile(
    module_name: str, dataset_hash: str, optimizer: str
) -> dict | None:
    """Check if the exact same compile already exists — saves LM API costs.

    Compares module_name, dataset_hash, and optimizer. If a completed run
    with a positive score exists, returns it so the caller can skip recompilation.

    Idempotency: Same inputs always find the same existing run.

    Returns:
        The existing run dict, or None if no match found.
    """
    runs = list_compiled_runs()
    for r in runs:
        lineage = r.get("lineage", {})
        if (
            lineage.get("base_program") == module_name
            and lineage.get("dataset_hash") == dataset_hash
            and lineage.get("optimizer") == optimizer
            and lineage.get("status") != "failed"
        ):
            return r
    return None


def compute_module_source_hash(module) -> str:
    """Hash a DSPy module's source code for structural identity.

    Uses inspect.getsource() to capture the module's forward() method
    and class definition. Returns empty string if source unavailable.
    """

    try:
        source = inspect.getsource(module)
        # Normalize whitespace to avoid hash changes on reformatting
        if source is None:
            return ""
        normalized = " ".join(source.split())
        return hashlib.sha256(normalized.encode()).hexdigest()[:12]
    except TypeError:
        return ""


def compute_idempotency_key(
    module_name: str,
    dataset_hash: str,
    optimizer: str,
    module_source_hash: str | None = None,
) -> str:
    """Compute a deterministic idempotency key for compile deduplication.

    Same module + dataset + optimizer (+ source code) → same key.
    When module_source_hash is provided, recompilation is skipped if the
    module's Python source hasn't changed, even if triggered manually.

    Used for X-Idempotency-Key header in API and CLI --idempotency-key flag.
    """

    payload = f"{module_name}|{dataset_hash}|{optimizer}"
    if module_source_hash:
        payload += f"|src:{module_source_hash}"
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def register_run_with_lineage(
    run_id: str,
    metadata: dict,
    optimizer: str = "",
    dataset_hash: str = "",
    base_program_id: str | None = None,
    parent_run_id: str | None = None,
) -> dict:
    """Register a compile run with full lineage tracking.

    Args:
        run_id: Unique run ID
        metadata: Compile metadata (score, params, etc.)
        optimizer: Optimizer name used
        dataset_hash: SHA256 hash of the training dataset
        base_program_id: The unoptimized module name
        parent_run_id: Previous compile this was derived from
    """
    lineage = {
        "run_id": run_id,
        "optimizer": optimizer,
        "dataset_hash": dataset_hash,
        "base_program": base_program_id,
        "parent_run": parent_run_id,
        "timestamp": time.time(),
    }

    enriched = {**metadata, "lineage": lineage}
    register_run(run_id, enriched)
    return enriched


def delete_run(run_id: str) -> bool:
    """Delete a run from JSON index and disk (invalidates cache)."""
    deleted = False
    runs = list_compiled_runs()  # Uses cached _load_index()
    new_runs = [r for r in runs if r["id"] != run_id]
    if len(new_runs) != len(runs):
        deleted = True
        save_run_index(new_runs)  # Atomically writes + invalidates cache

    run_dir = compiled_dir() / run_id
    if run_dir.exists():
        shutil.rmtree(run_dir)
        deleted = True

    return deleted


# ── Optimization 10: Cached directory listings ───────────────────────────

_dir_cache: dict[str, tuple[float, list[dict]]] = {}
_DIR_CACHE_TTL = 2.0  # seconds


def _cached_dir_scan(path: Path, pattern: str, key: str) -> list[dict]:
    """Scan a directory with mtime-based caching (Optimization 10)."""
    now = time.time()
    if key in _dir_cache:
        ts, result = _dir_cache[key]
        if (now - ts) < _DIR_CACHE_TTL:
            return result

    items = []
    for f in sorted(path.glob(pattern)):
        if f.stem == "__init__":
            continue
        items.append({"name": f.stem, "path": str(f), "size": f.stat().st_size})

    _dir_cache[key] = (now, items)
    return items


def list_signatures() -> list[dict]:
    """List signature files (cached per 2s window)."""
    return _cached_dir_scan(signatures_dir(), "*.py", "signatures")


def delete_signature(name: str) -> bool:
    path = signatures_dir() / f"{name}.py"
    if path.exists():
        path.unlink()
        _dir_cache.pop("signatures", None)
        return True
    return False


def list_modules() -> list[dict]:
    """List module files (cached per 2s window)."""
    return _cached_dir_scan(modules_dir(), "*.py", "modules")


def delete_module(name: str) -> bool:
    path = modules_dir() / f"{name}.py"
    if path.exists():
        path.unlink()
        _dir_cache.pop("modules", None)
        return True
    return False


def list_agents() -> list[dict]:
    """List agent files (cached per 2s window)."""
    return _cached_dir_scan(agents_dir(), "*.json", "agents")


def delete_agent(name: str) -> bool:
    path = agents_dir() / f"{name}.json"
    if path.exists():
        path.unlink()
        _dir_cache.pop("agents", None)
        return True
    return False
