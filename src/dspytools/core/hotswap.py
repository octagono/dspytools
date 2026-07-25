"""Hot-swap engine for compiled DSPy programs.

Optimization 1: LRU bounded cache — max N loaded programs, evict least recently used.
Optimization 2: Lazy loading — index loaded on startup, programs loaded on first access.
Optimization 3: Single-score infer — auto_metric called once, reused for drift + evolve.
Optimization 6: Deterministic loader — use metadata.json module_type, no fallback chain.
Optimization 17: Metadata cache — metadata.json reads cached per run_id.
Optimization 24: auto_metric imported at module level (was imported inside infer()).
"""

from __future__ import annotations

import json
import re as _re
import threading
import time as _time
from collections import OrderedDict, deque
from typing import TYPE_CHECKING, Any

from dspytools.core.logging_config import get_logger

if TYPE_CHECKING:
    import dspy
else:
    from dspytools.core._dspy import dspy

import numpy as np

from dspytools.config.settings import compiled_dir
from dspytools.core._io import read_json
from dspytools.core.drift_monitor import get_drift_monitor
from dspytools.core.loaders import prediction_to_dict
from dspytools.core.metrics import auto_metric
from dspytools.graph.cache import get_semantic_cache

_log = get_logger(__name__)

# Maximum programs to hold in memory simultaneously
MAX_LOADED = 16

# Bounded quality samples for self-evolve engine (max 200, FIFO eviction)
_quality_samples: deque[dict] = deque(maxlen=200)

# Stochastic scoring counter — only evaluate 1-in-10 inferences
_infer_count: int = 0

# Drift monitor singleton cache (avoids re-import + re-instantiation per infer call)
_drift_monitor = None

# Optimization 17: Metadata cache — avoids re-reading metadata.json per load
# Bounded LRU cache via OrderedDict (max 256 entries)
_metadata_cache: OrderedDict[str, dict] = OrderedDict()

# Optimization 28: Signature cache — avoids re-reading signature.json per load
# Bounded LRU cache via OrderedDict (max 256 entries)
_signature_cache: OrderedDict[str, dict] = OrderedDict()
_MAX_CACHE = 256


def _load_program_from_run(run_id: str) -> dspy.Module | None:
    """Load a compiled program deterministically from its metadata.

    Optimization 9: Reads program.json ONCE and reuses the parsed data
    for both the initial load and signature-mismatch recovery.
    Optimization 17: Caches metadata.json reads to avoid repeated disk I/O.

    Uses metadata.json's module_type field to pick the right DSPy class
    on first try — no Predict→ChainOfThought fallback chain.
    """
    run_path = compiled_dir() / run_id
    meta_path = run_path / "metadata.json"

    # Optimization 17: Check metadata cache first
    if run_id in _metadata_cache:
        meta = _metadata_cache[run_id]
    else:
        if not meta_path.exists():
            return None
        meta = read_json(meta_path)
        _metadata_cache[run_id] = meta
        _metadata_cache.move_to_end(run_id)
        if len(_metadata_cache) > _MAX_CACHE:
            _metadata_cache.popitem(last=False)

    sig_path = run_path / "signature.json"

    # Optimization 28: Check signature cache first
    sig_cache_key = f"{run_id}:sig"
    if sig_cache_key in _signature_cache:
        sig_data = _signature_cache[sig_cache_key]
    else:
        if not sig_path.exists():
            return None
        sig_data = read_json(sig_path)
        _signature_cache[sig_cache_key] = sig_data
        _signature_cache.move_to_end(sig_cache_key)
        if len(_signature_cache) > _MAX_CACHE:
            _signature_cache.popitem(last=False)

    module_type = meta.get("module_type", sig_data.get("module_type", "predict"))

    prog_path = run_path / "program.json"
    if not prog_path.exists():
        return None

    # Optimization 9: Read program.json ONCE, reuse for both load paths
    prog_text = prog_path.read_text()  # Fail fast on read errors

    saved = json.loads(prog_text)

    # Determine the correct signature: use real field names from
    # program.json instructions ("fields `X`, produce fields `Y`")
    real_sig = None
    if "predictor" in saved:
        sig = saved["predictor"].get("signature", {})
        instr = sig.get("instructions", "")
        input_m = _re.search(r"fields `([^`]+)`", instr)
        output_m = _re.search(r"produce the fields `([^`]+)`", instr)
        if input_m and output_m:
            real_inputs = [f.strip() for f in input_m.group(1).split(",")]
            real_outputs = [f.strip() for f in output_m.group(1).split(",")]
            real_sig = ", ".join(real_inputs) + " -> " + ", ".join(real_outputs)
    if not real_sig:
        real_sig = f"{', '.join(sig_data.get('inputs', ['input']))} -> {', '.join(sig_data.get('outputs', ['output']))}"

    if module_type == "cot" or module_type == "chain_of_thought":
        program = dspy.ChainOfThought(real_sig)
    elif module_type == "react":
        program = dspy.ReAct(real_sig, tools=[])
    else:
        program = dspy.Predict(real_sig)

    # Use program.load() first — works for pipeline format (stages[0], ...)
    # and any format DSPy natively supports.
    try:
        program.load(str(prog_path))
    except (KeyError, ValueError):
        # Predict format: program.json wraps state in {predictor: {...}}
        # but load_state() needs flat {signature, demos, ...}
        if "predictor" in saved:
            program.load_state(saved["predictor"])

    return program


class HotSwapManager:
    """Optimization 1: LRU bounded cache. Optimization 2: Lazy loading.
    Optimization 3: Single-score infer. Optimization 6: Deterministic loader.

    Loads compiled programs into memory with LRU eviction.
    Switching active program is O(1). Inference uses active program.

    Reference counting: Each infer() call increments refcount for the active
    program. swap() awaits zero refcount before unloading the previous program.
    """

    def __init__(self, max_loaded: int = MAX_LOADED) -> None:
        self._max_loaded = max_loaded
        self._programs: OrderedDict[str, dspy.Module] = OrderedDict()
        self._metadata: dict[str, dict] = {}
        self._active: str | None = None
        self._index_loaded = False  # Optimization 2: lazy index flag
        self._refcounts: dict[str, int] = {}  # program_id → active inference count
        self._reflock = (
            threading.Lock()
        )  # guards _refcounts under concurrent infer/swap
        self._cache = (
            None  # Cached SemanticCache singleton (avoids function call per infer)
        )

    @property
    def active_id(self) -> str | None:
        return self._active

    @property
    def active_program(self) -> dspy.Module:
        if self._active is None or self._active not in self._programs:
            raise RuntimeError("No active program. Call swap() first.")
        return self._programs[self._active]

    def _ensure_index(self) -> None:
        """Optimization 2: Load index metadata lazily on first access."""
        if self._index_loaded:
            return
        self._index_loaded = True
        index_path = compiled_dir() / "index.json"
        if not index_path.exists():
            return
        runs = read_json(index_path)
        for run in runs:
            rid = run["id"]
            if rid not in self._metadata:
                self._metadata[rid] = {"id": rid}
        # Auto-activate first program if none active
        if self._active is None and self._metadata:
            self._active = next(iter(self._metadata))

    def load_all(self) -> list[str]:
        """Load all compiled programs from disk into memory."""
        self._ensure_index()
        for rid in list(self._metadata.keys()):
            self._ensure_loaded(rid)
        return list(self._programs.keys())

    def _ensure_loaded(self, run_id: str) -> None:
        """Load a run into cache if not present, evicting LRU if full."""
        if run_id in self._programs:
            self._programs.move_to_end(run_id)
            return

        program = _load_program_from_run(run_id)
        if program is None:
            return

        # Evict LRU if at capacity
        while len(self._programs) >= self._max_loaded:
            self._programs.popitem(last=False)

        self._programs[run_id] = program
        if run_id not in self._metadata:
            self._metadata[run_id] = {"id": run_id}

    def load_single(self, run_id: str) -> bool:
        """Load a single run by ID."""
        self._ensure_loaded(run_id)
        if run_id in self._programs:
            if self._active is None:
                self._active = run_id
            return True
        return False

    def _inc_ref(self, run_id: str) -> None:
        """Increment reference count for a program (thread-safe)."""
        with self._reflock:
            self._refcounts[run_id] = self._refcounts.get(run_id, 0) + 1

    def _dec_ref(self, run_id: str) -> None:
        """Decrement reference count for a program (thread-safe)."""
        with self._reflock:
            current = self._refcounts.get(run_id, 1)
            if current <= 1:
                self._refcounts.pop(run_id, None)
            else:
                self._refcounts[run_id] = current - 1

    def _refcount(self, run_id: str) -> int:
        """Get current reference count for a program (thread-safe)."""
        with self._reflock:
            return self._refcounts.get(run_id, 0)

    def swap(self, run_id: str, wait_for_drain: bool = False) -> str | None:
        """Switch active program. Returns previous active ID or None.

        If wait_for_drain is True, blocks until the previous program's
        refcount reaches zero (up to 30s timeout). Use for production
        zero-downtime swaps.
        """
        if run_id not in self._programs:
            if not self.load_single(run_id):
                raise KeyError(f"Program '{run_id}' not found")
        self._programs.move_to_end(run_id)
        prev = self._active
        self._active = run_id

        # Wait for previous program to drain if requested
        if prev and wait_for_drain and self._refcount(prev) > 0:
            deadline = _time.time() + 30.0
            while _time.time() < deadline and self._refcount(prev) > 0:
                _time.sleep(0.05)

        # Predictive prefetching: pre-load dependent programs (best-effort)
        self._prefetch_dependents(run_id)

        return prev

    def _prefetch_dependents(self, run_id: str) -> None:
        """Pre-load programs that depend on the swapped one into the LRU cache.

        Uses SkillGraph lineage to identify likely-next programs.
        Best-effort: silently skips if SkillGraph unavailable or cache full.
        Non-activating: loads into cache but doesn't change active program.
        """
        from dspytools.evolve.self_evolve import (
            SkillGraph,  # lazy: breaks core↔evolve cycle
        )

        sg = SkillGraph()
        dependents = sg.transitive_dependents(run_id)
        for dep in dependents:
            if dep not in self._programs and len(self._programs) < self._max_loaded - 1:
                self.load_single(dep)

    def warm_swap(self, run_id: str) -> str | None:
        """Pre-load a program and verify it works before switching.

        Loads the program, runs a verification inference using the program's
        own signature fields, then swaps if successful.
        Returns previous active ID or raises on failure.
        """
        # Ensure loaded
        if run_id not in self._programs:
            if not self.load_single(run_id):
                raise KeyError(f"Program '{run_id}' not found")

        # Verify: construct test input from the program's signature fields
        program = self._programs[run_id]
        try:
            test_input = self._build_test_input(run_id, program)
            result = program(**test_input)
            if result is None:
                raise RuntimeError("Test inference returned None")
        except (RuntimeError, OSError, ValueError, TypeError) as e:
            # Unload failed program
            self._programs.pop(run_id, None)
            self._metadata.pop(run_id, None)
            raise RuntimeError(
                f"Warm swap verification failed for '{run_id}': {e}"
            ) from e

        # Swap in
        self._programs.move_to_end(run_id)
        prev = self._active
        self._active = run_id
        return prev

    @staticmethod
    def _build_test_input(run_id: str, program: Any) -> dict[str, str]:
        """Construct a minimal test input from the program's signature fields.

        Reads signature.json from the run directory to determine input field
        names. Falls back to {"input": "test"} if signature cannot be read.
        """
        sig_path = compiled_dir() / run_id / "signature.json"
        if sig_path.exists():
            sig = read_json(sig_path)
            input_fields = sig.get("inputs", [])
            if isinstance(input_fields, str):
                input_fields = [f.strip() for f in input_fields.split(",") if f.strip()]
            if input_fields:
                return {f: "test" for f in input_fields}
        return {"input": "test"}

    def infer(self, **inputs: Any) -> dict[str, Any]:
        """Run inference with the active program.

        Optimization 3: Scores once, reuses for self-evolve + drift monitoring.
        SSOT: Checks semantic cache (RedisVL) before hitting the LM.

        Reference counting: increments refcount for the active program during
        inference and decrements it after completion. swap() can wait for
        refcount to drain before unloading.
        """
        global _drift_monitor
        prog = self.active_program
        active_id = self._active

        # Increment refcount for this inference — must decrement in finally
        if active_id:
            self._inc_ref(active_id)

        try:
            # Semantic cache check (SSOT: RedisVL)
            # Note: get_semantic_cache() can raise ConnectionError when Redis is
            # down — must be inside try/except, not before it.
            cache_key = json.dumps(inputs, sort_keys=True, default=str)
            cache = self._cache or get_semantic_cache()
            self._cache = cache
            cached = cache.check(cache_key)
            if cached:
                response = cached.get("response", {})
                if isinstance(response, str):
                    response = json.loads(response)
                _log.debug(
                    "semantic cache hit — tier=%s distance=%.4f",
                    cached.get("tier", "?"),
                    cached.get("distance", 0.0),
                )
                return response

            # Reuse the embedding vector from cache.check() miss to avoid
            # a redundant embedder call in cache.store()
            cache_vector = cached.get("_vector") if cached else None

            result = prog(**inputs)
            output = prediction_to_dict(result)

            # Semantic cache store (SSOT)
            if cache:
                # Convert ndarray values to lists for JSON serialization
                serializable = {
                    k: v.tolist() if isinstance(v, np.ndarray) else v
                    for k, v in output.items()
                }
                cache.store(cache_key, json.dumps(serializable), _vector=cache_vector)

            # Stochastic scoring: only evaluate 1-in-10 inferences
            # (drift monitoring only needs statistical signal, not every sample)
            global _infer_count
            content = str(next(iter(output.values()), ""))
            score = 0.5
            if content and _infer_count % 10 == 0:
                score = auto_metric(content)
            _infer_count += 1

            # Self-evolve: collect quality sample (bounded deque)
            if self._active:
                _quality_samples.append({"program": self._active, "score": score})

            # Drift monitoring: cached monitor instance, reuse score
            if self._active and score < 0.3:
                if _drift_monitor is None:
                    _drift_monitor = get_drift_monitor()
                alert = _drift_monitor.check(self._active, score)
                if alert and alert.severity == "critical":
                    _log.warning("drift_detected", message=alert.message)
                    # Auto-queue for recompile
                    _drift_monitor.request_recompile(self._active)
                    _log.info(
                        "drift recompile queued for %s — run `dspytools self auto-fix` to process",
                        self._active,
                    )
                    # Dynamic cache tuning: tighten threshold on drift
                    if cache:
                        cache.adjust_threshold(alert.degradation_pct / 100)

            return output
        finally:
            # Always decrement refcount — covers success, cache hit, and exception paths
            if active_id:
                self._dec_ref(active_id)

    def list(self) -> list[dict]:
        """List all loaded programs with active flag."""
        return [
            {
                "id": rid,
                "active": rid == self._active,
                **self._metadata.get(rid, {}),
            }
            for rid in self._programs
        ]

    def get_metadata(self, run_id: str) -> dict | None:
        return self._metadata.get(run_id)

    def get(self, run_id: str) -> dspy.Module | None:
        """Load and return a program by run_id, or None if not found."""
        self._ensure_loaded(run_id)
        return self._programs.get(run_id)

    def unload(self, run_id: str) -> bool:
        if run_id in self._programs:
            del self._programs[run_id]
            self._metadata.pop(run_id, None)
            if self._active == run_id:
                self._active = next(iter(self._programs)) if self._programs else None
            return True
        return False

    def is_loaded(self, run_id: str) -> bool:
        return run_id in self._programs
