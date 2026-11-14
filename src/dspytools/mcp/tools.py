"""Built-in MCP tools with Redis-backed response caching.

Optimization 6: MCP tool responses are cached in Redis with TTL (5s default)
and LRU eviction (max 256 entries), eliminating redundant file reads on every
agent call. Cache persists across MCP server restarts.

Schema note: tool specs declare `description`, `inputSchema` and `annotations`
only. `outputSchema` is deliberately **not** declared — the server does not
forward it (see `mcp/server.py::handle_list_tools`), so the SDK's output
validation never ran and the old entries were inaccurate dead weight. Do not
reintroduce the field without also emitting it from `handle_list_tools`.

Imports stay lazy on the stdio startup path: `mlflow` is imported inside the
`doctor` handler (it is used for one tracking-URI lookup) so
`import dspytools.mcp.server` never pulls it in.
"""

from __future__ import annotations

import fnmatch
import functools
import io
import json
import json as _j
import pathlib
import subprocess as _sp
import sys
import threading
import time
import urllib.request
from collections import OrderedDict
from collections.abc import Callable
from contextlib import nullcontext
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import dspy
else:
    from dspytools.core._dspy import dspy

from dspytools.commands.evaluate import _get_metric, _load_program
from dspytools.commands.lora import _adapter_model_name, _get_base_model
from dspytools.commands.self import _get_status
from dspytools.config.settings import adapters_dir as _adapters_dir
from dspytools.config.settings import llama_cpp_url
from dspytools.core._io import read_json
from dspytools.core.drift_monitor import get_drift_monitor
from dspytools.core.holdout import get_holdout_gate
from dspytools.core.hotswap import HotSwapManager
from dspytools.core.loaders import load_module_by_name, load_trainset
from dspytools.core.logging_config import get_logger
from dspytools.core.metrics import exact_match_metric
from dspytools.core.registry import (
    get_lineage,
    get_run,
    list_compiled_runs,
    list_modules,
    list_signatures,
)
from dspytools.core.setup import LMRegistry, setup_dspy
from dspytools.generate import RepositoryAnalyzer, gather_repository_info
from dspytools.generate.cache import get_analysis_cache
from dspytools.generate.module import get_sandbox_pool
from dspytools.graph.cache import get_semantic_cache
from dspytools.graph.client import get_graph_client
from dspytools.graph.redis_cache import RedisCache, get_compile_cache, get_mcp_cache
from dspytools.graph.skill_graph import FalkorDBSkillGraph
from dspytools.memory.manager import get_memory_manager
from dspytools.skills import SkillManager
from dspytools.skills.discovery import (
    list_categories,
    search_external,
    try_skills_sh_api,
)

_hotswap: HotSwapManager | None = None

# Optimization 6: Redis-backed cache with TTL
_CACHE_TTL = 5.0  # seconds
# Optimization: cache the resolved cache instance — avoids _get_redis_cache()
# try/except overhead on every _cached/_set_cache/_invalidate call
_resolved_cache: _FallbackCache | Any | None = None


def _get_redis_cache():
    """Lazy Redis cache — falls back to in-memory if Redis unavailable."""
    try:
        return get_mcp_cache()
    except (ConnectionError, OSError) as e:
        _log.warning("redis_unavailable", error=str(e))
        return None


class _FallbackCache:
    """In-memory fallback when Redis is unavailable."""

    def __init__(self):
        self._data: OrderedDict[str, tuple[float, str]] = OrderedDict()
        self._max = 128

    def get(self, key: str) -> str | None:
        if key in self._data:
            ts, val = self._data[key]
            if time.time() - ts < _CACHE_TTL:
                self._data.move_to_end(key)
                return val
            del self._data[key]
        return None

    def set(self, key: str, value: str, ttl: float = _CACHE_TTL) -> None:
        if key in self._data:
            self._data.move_to_end(key)
        self._data[key] = (time.time(), value)
        while len(self._data) > self._max:
            self._data.popitem(last=False)

    def delete(self, key: str) -> None:
        self._data.pop(key, None)

    def flush(self) -> int:
        n = len(self._data)
        self._data.clear()
        return n

    def keys(self, pattern: str = "*") -> list[str]:
        return list(self._data.keys())


_fallback_instance: _FallbackCache | None = None


def _cache() -> _FallbackCache | Any:
    """Get cache instance — Redis or in-memory fallback.

    Optimization: caches the resolved instance so subsequent calls skip
    the _get_redis_cache() try/except overhead.
    """
    global _resolved_cache, _fallback_instance
    if _resolved_cache is not None:
        return _resolved_cache
    rc = _get_redis_cache()
    if rc is not None:
        _resolved_cache = rc
        return rc
    if _fallback_instance is None:
        _fallback_instance = _FallbackCache()
    _resolved_cache = _fallback_instance
    return _fallback_instance


def _cached(key: str, ttl: float = _CACHE_TTL) -> str | None:
    c = _cache()
    val = c.get(key)
    return val


def _set_cache(key: str, val: str) -> None:
    c = _cache()
    c.set(key, val, ttl=_CACHE_TTL)


def _invalidate(pattern: str) -> None:
    """Invalidate cache keys matching a glob pattern."""
    c = _cache()
    all_keys: list[str] = []
    if hasattr(c, "keys"):
        all_keys = list(c.keys(pattern))
    elif hasattr(c, "_data"):
        all_keys = list(c._data.keys())
    for k in all_keys:
        if fnmatch.fnmatch(k, pattern):
            c.delete(k)


def _cached_call(key: str, fn: Callable[[], str], ttl: float = _CACHE_TTL) -> str:
    """Cache helper: check cache, call fn on miss, store result."""
    cached = _cached(key)
    if cached:
        return cached
    result = fn()
    _set_cache(key, result)
    return result


def _get_hotswap() -> HotSwapManager:
    global _hotswap
    if _hotswap is None:
        # Optimization 2: Don't call load_all() eagerly — lazy index + on-demand loading
        _hotswap = HotSwapManager()
    return _hotswap


_log = get_logger(__name__)


def _error(exception: Exception | str, detail: str | None = None) -> str:
    """Standardized JSON error response for MCP tool handlers.

    Logs the exception at ERROR level before returning structured JSON.

    Args:
        exception: Exception or string error message
        detail: Optional additional context

    Returns:
        JSON string: {"error": str(exception)} with optional "detail" key
    """
    msg = str(exception)
    if isinstance(exception, Exception):
        _log.exception("mcp_tool_error", error=str(msg))
    else:
        _log.error("mcp_tool_error", error=str(msg))
    payload: dict[str, str] = {"error": msg}
    if detail:
        payload["detail"] = detail
    return json.dumps(payload)


_setup_lock = threading.Lock()


def _ensure_lm_configured() -> None:
    """Configure DSPy at most once, and only if the LM is not already configured.

    MCP tool handlers run on worker threads. `dspy.configure()` may only be
    called by the thread that first configured settings — any other thread
    raises `RuntimeError`, and `core.setup.setup_dspy`'s documented fallback
    then writes `dspy.dsp.utils.settings.main_thread_config` directly. Calling
    `setup_dspy()` from every handler would therefore re-enter that fallback on
    every tool call, so the MCP side guards it:

    - if `dspy.settings.lm` is already set, do nothing (reads are global);
    - otherwise configure once under a lock.

    Per-call isolation belongs to the caller via `with dspy.context(...)`.
    """
    if getattr(dspy.settings, "lm", None) is not None:
        return
    with _setup_lock:
        if getattr(dspy.settings, "lm", None) is not None:
            return
        setup_dspy()


def _lm_tool(fn: Callable) -> Callable:
    """Guard one LM-touching tool handler.

    Every handler that reaches an LM is wrapped in this: it configures DSPy
    once (`_ensure_lm_configured`) and turns a configuration failure into the
    documented in-band error shape (`_error`, i.e. ``{"error", "detail"}``)
    instead of letting a raw exception reach the MCP transport. Handlers that
    only read files, caches or the registry are not wrapped.
    """

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> str:
        try:
            _ensure_lm_configured()
        except Exception as e:  # noqa: BLE001 - reported in-band, never raised
            return _error(
                e,
                detail=(
                    "DSPy has no usable LM. Configure one with "
                    "`dspytools configure lm set student <model>` (and a teacher "
                    "for GEPA) or set the API key, then restart the MCP server."
                ),
            )
        return fn(*args, **kwargs)

    return wrapper


# Optimizer families that exist only as `dspytools compile <command>` commands.
# They are not constructible from `_OPTIMIZER_SPECS`, so `compile_optimizer`
# cannot run them; they are reported separately instead of being advertised as
# callable (that mismatch used to make every list-then-compile fail).
_CLI_ONLY_OPTIMIZERS: dict[str, str] = {
    "avatar": "dspy.AvatarOptimizer - upstream still calls the removed TypedPredictor",
    "better-together": "dspy.BetterTogether program merger",
    "bootstrap-finetune": "alias of `finetune`",
    "distill": "LoRA distillation pipeline (teacher-driven)",
    "ensemble": "dspy.Ensemble over already-compiled modules",
    "finetune": "dspy.BootstrapFinetune - needs a fine-tuning API, not llama.cpp",
    "flex": "dspy.Flex + GEPA wrapper (needs an interpreter pool)",
    "grpo": "dspy.teleprompt.grpo.GRPO - finetune-only, refuses llama.cpp",
}

# Underscore spellings the older tool surface used; accepted as aliases only.
_OPTIMIZER_ALIASES: dict[str, str] = {
    "bootstrap_few_shot": "bootstrap-few-shot",
    "bootstrap_few_shot_optuna": "bootstrap-few-shot-optuna",
    "bootstrap_few_shot_random": "bootstrap-few-shot-random",
    "bootstrap_finetune": "bootstrap-finetune",
    "infer_rules": "infer-rules",
    "labeled_few_shot": "labeled-few-shot",
}

# Legacy spellings of names that only exist as CLI commands.
_CLI_ONLY_ALIASES: dict[str, str] = {
    "better_together": "better-together",
}

# Lazily populated copy of `commands.compile._OPTIMIZER_SPECS` (see
# `_optimizer_specs()` — `commands.compile` imports the MLflow SDK).
_OPTIMIZER_SPECS_CACHE: dict[str, Any] | None = None


def _optimizer_specs() -> dict[str, Any]:
    """The `_OPTIMIZER_SPECS` table from `commands/compile.py`, imported lazily.

    `commands.compile` imports `core.mlflow_tracker`, which imports the MLflow
    SDK at module scope. Importing the table lazily keeps MLflow off the stdio
    startup path while still giving `list_optimizers`, the
    `dspytools://optimizers` resource and `compile_optimizer` one shared source.
    """
    global _OPTIMIZER_SPECS_CACHE
    if _OPTIMIZER_SPECS_CACHE is None:
        from dspytools.commands.compile import _OPTIMIZER_SPECS

        _OPTIMIZER_SPECS_CACHE = _OPTIMIZER_SPECS
    return _OPTIMIZER_SPECS_CACHE


def optimizer_catalog() -> dict[str, Any]:
    """Return the single source of truth for optimizer names.

    `tool_list_optimizers`, the `dspytools://optimizers` resource and
    `tool_compile_optimizer` all derive from `_OPTIMIZER_SPECS`, so a name that
    is listed under ``optimizers`` is exactly a name `compile_optimizer`
    resolves. Families that only exist as CLI commands live under ``cli_only``
    and are explicitly not callable through this tool.
    """
    return {
        "optimizers": sorted(_optimizer_specs()),
        "cli_only": dict(sorted(_CLI_ONLY_OPTIMIZERS.items())),
    }


def _canonical_optimizer_name(name: str) -> str:
    """Map a listed/legacy optimizer spelling to its `_OPTIMIZER_SPECS` key."""
    key = (name or "").strip().lower()
    key = _OPTIMIZER_ALIASES.get(key, _CLI_ONLY_ALIASES.get(key, key))
    specs = _optimizer_specs()
    if key in specs or key in _CLI_ONLY_OPTIMIZERS:
        return key
    hyphenated = key.replace("_", "-")
    if hyphenated in specs or hyphenated in _CLI_ONLY_OPTIMIZERS:
        return hyphenated
    return key


def tool_list_programs() -> str:
    def _compute() -> str:
        mgr = _get_hotswap()
        mgr._ensure_index()
        return json.dumps(mgr.list(), indent=2)

    return _cached_call("list_programs", _compute)


def tool_swap_program(program_id: str) -> str:
    # Invalidate cache on mutation
    _invalidate("list_programs")
    mgr = _get_hotswap()
    try:
        prev = mgr.swap(program_id)
        return json.dumps({"status": "ok", "active": program_id, "previous": prev})
    except KeyError as e:
        return _error(e)


@_lm_tool
def tool_infer(**inputs: Any) -> str:
    mgr = _get_hotswap()
    result = mgr.infer(**inputs)
    return json.dumps({"status": "ok", "result": result}, default=str)


def tool_get_program_metadata(program_id: str) -> str:
    mgr = _get_hotswap()
    meta = mgr.get_metadata(program_id)
    if meta:
        return json.dumps(meta, indent=2)
    return _error(f"Program '{program_id}' not found")


def tool_list_signatures() -> str:
    return _cached_call(
        "list_signatures", lambda: json.dumps(list_signatures(), indent=2)
    )


def tool_list_modules() -> str:
    return _cached_call("list_modules", lambda: json.dumps(list_modules(), indent=2))


def tool_list_compiled_runs() -> str:
    return _cached_call(
        "list_compiled_runs", lambda: json.dumps(list_compiled_runs(), indent=2)
    )


BUILTIN_TOOLS: dict[str, dict] = {
    "list_programs": {
        "description": "List all loaded compiled programs with active flag",
        "inputSchema": {"type": "object", "properties": {}},
        "annotations": {
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        },
        "handler": tool_list_programs,
    },
    "swap_program": {
        "description": "Switch the active compiled program",
        "inputSchema": {
            "type": "object",
            "properties": {
                "program_id": {
                    "type": "string",
                    "description": "ID of the compiled program to activate",
                }
            },
            "required": ["program_id"],
        },
        "annotations": {
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        },
        "handler": tool_swap_program,
    },
    "infer": {
        "description": "Run inference with the active compiled program",
        "inputSchema": {
            "type": "object",
            "properties": {},
            "additionalProperties": True,
        },
        "annotations": {
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        },
        "handler": tool_infer,
    },
    "get_program_metadata": {
        "description": "Get metadata for a compiled program",
        "inputSchema": {
            "type": "object",
            "properties": {
                "program_id": {
                    "type": "string",
                    "description": "ID of the compiled program",
                }
            },
            "required": ["program_id"],
        },
        "annotations": {
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        },
        "handler": tool_get_program_metadata,
    },
    "list_signatures": {
        "description": "List available generated signatures",
        "inputSchema": {"type": "object", "properties": {}},
        "annotations": {
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        },
        "handler": tool_list_signatures,
    },
    "list_modules": {
        "description": "List available generated modules",
        "inputSchema": {"type": "object", "properties": {}},
        "annotations": {
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        },
        "handler": tool_list_modules,
    },
    "list_compiled_runs": {
        "description": "List all compiled runs in the registry",
        "inputSchema": {"type": "object", "properties": {}},
        "annotations": {
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        },
        "handler": tool_list_compiled_runs,
    },
}


def tool_list_optimizers() -> str:
    """List optimizers this server can compile with, and CLI-only families.

    ``optimizers`` are the names `compile_optimizer` accepts (derived from
    `_OPTIMIZER_SPECS`). ``cli_only`` families are available only through
    ``dspytools compile <name>`` and are reported separately rather than being
    advertised as callable here.
    """
    return json.dumps(optimizer_catalog(), indent=2)


def tool_skills_list() -> str:
    """List all available skills."""

    mgr = SkillManager()
    skills = mgr.list_skills()
    return json.dumps(
        [
            {"name": s.name, "description": s.description, "compiled": s.has_program}
            for s in skills
        ],
        indent=2,
    )


def tool_skills_search(query: str, k: int = 5) -> str:
    """BM25 + embedding search for skills matching query."""

    mgr = SkillManager()
    results = mgr.search(query, k)
    return json.dumps(
        [{"name": s.name, "description": s.description[:100]} for s in results],
        indent=2,
    )


def tool_self_status() -> str:
    """Show self-optimization status."""
    return json.dumps(_get_status(), indent=2)


def tool_inspect_history(n: int = 5) -> str:
    """Show last N LM call history entries.

    DSPy 3.5's `inspect_history` accepts a `file` sink, so the process-wide
    `sys.stdout` is never swapped (an exception used to leave the server's
    stdout pointing at a dead StringIO for the rest of its life).
    """

    buf = io.StringIO()
    dspy.inspect_history(n=n, file=buf)
    return buf.getvalue()[:2000]


def tool_mlflow_status() -> str:
    """Tool: Report MLflow tracking status including async queue stats."""
    # Imported here so the stdio startup path never imports the MLflow SDK.
    from dspytools.core.mlflow_tracker import get_tracker

    tracker = get_tracker()
    tracker._ensure_initialized()

    result = {
        "enabled": True,
        "tracking_uri": tracker.tracking_uri,
        "experiment": tracker.experiment_name,
    }

    if hasattr(tracker, "stats"):
        result["async"] = tracker.stats  # type: ignore[union-attr]

    return json.dumps(result, indent=2)


@_lm_tool
def tool_generate_llms_txt(target: str, local: bool = False, output: str = "") -> str:
    """Generate llms.txt for a repository."""
    repo_url = target
    file_tree, readme, packages, _history = gather_repository_info(
        str(target) if local else ""
    )
    analyzer = RepositoryAnalyzer()
    result = analyzer(
        repo_url=repo_url,
        file_tree=file_tree,
        readme_content=readme,
        package_files=packages,
    )
    if output:
        pathlib.Path(output).write_text(result.llms_txt_content)
    return json.dumps(
        {
            "llms_txt_content": result.llms_txt_content[:2000],
            "chars": len(result.llms_txt_content),
        }
    )


@_lm_tool
def tool_compile_optimizer(optimizer: str, module_name: str, trainset_path: str) -> str:
    """Trigger a DSPy compile with a specific optimizer.

    Accepts any name from `list_optimizers`' ``optimizers`` list (the
    `_OPTIMIZER_SPECS` keys) plus the legacy underscore spellings. Two
    contracts matter here:

    - the spec's ``params`` dict is owned by the process-wide optimizer table,
      so it is copied before the teacher LM is attached. Writing the teacher
      (and its ``api_key``) into the shared dict used to poison every later
      compile — including the CLI's.
    - Golden Rule 3: `HoldoutGate.split()` runs **before** the optimizer is
      constructed and only the train split reaches `compile`.
    """
    canonical = _canonical_optimizer_name(optimizer)
    if canonical in _CLI_ONLY_OPTIMIZERS:
        return _error(
            f"Optimizer '{canonical}' is not callable through this tool",
            detail=(
                f"{_CLI_ONLY_OPTIMIZERS[canonical]}. Run it from the CLI: "
                f"`dspytools compile {canonical} --help`."
            ),
        )
    spec = _optimizer_specs().get(canonical)
    if spec is None:
        return _error(
            f"Unknown optimizer: {optimizer}",
            detail=(
                "Call list_optimizers for the callable names: "
                + ", ".join(sorted(_optimizer_specs()))
            ),
        )

    student = load_module_by_name(module_name)
    trainset = load_trainset(trainset_path)
    if len(trainset) < 2:
        return _error(
            f"Trainset '{trainset_path}' has {len(trainset)} example(s)",
            detail=(
                "At least 2 examples are required so the holdout gate can carve "
                "out a validation split before compiling."
            ),
        )

    # Copy — `_OPTIMIZER_SPECS[...]["params"]` is process-wide state.
    extra = dict(spec.get("params") or {})
    metric: Callable = exact_match_metric()
    if canonical == "gepa":
        # GEPA validates a five-argument metric and needs a reflection LM.
        metric = _gepa_feedback_metric
    if spec.get("needs_teacher"):
        teacher = LMRegistry.get_teacher()
        if teacher is None:
            return _error(
                "gepa requires a teacher (reflection) LM",
                detail=(
                    "Configure one with `dspytools configure lm set teacher <model>` "
                    "or set DEEPSEEK_API_KEY, then restart the MCP server."
                ),
            )
        extra["reflection_lm"] = teacher

    compile_id = f"mcp:{module_name}:{canonical}"
    train_set, holdout = get_holdout_gate().split(trainset, compile_id=compile_id)

    opt = spec["cls_lambda"](metric=metric, **extra)
    opt.compile(student=student, trainset=train_set)
    return json.dumps(
        {
            "status": "compiled",
            "optimizer": canonical,
            "module": module_name,
            "trainset_size": len(train_set),
            "holdout_size": len(holdout),
            "compile_id": compile_id,
        }
    )


def tool_archive_search(query: str, top_k: int = 3) -> str:
    """Search compiled program archive for similar past compilations (Meta Agent Search)."""
    # Imported here: `dspytools.evolve` pulls the MLflow tracing SDK, which must
    # stay off the stdio startup path.
    from dspytools.evolve.self_evolve import get_engine

    engine = get_engine()
    results = engine.archive_search(query, top_k=top_k)
    return json.dumps(results, indent=2, default=str)


def tool_validate_deploy(
    program_id: str,
    holdout_path: str = "",
    alpha: float = 0.05,
    beta: float = 0.2,
) -> str:
    """SPRT-powered validation: test if a compiled program is ready to deploy."""
    # Load holdout from path or use empty set
    holdout = []
    if holdout_path:
        data = read_json(holdout_path)
        holdout = [dspy.Example(**item) for item in data]
    from dspytools.evolve.self_evolve import get_engine

    engine = get_engine()
    result = engine.validate_and_deploy(
        None, program_id, holdout, alpha=alpha, beta=beta
    )
    return json.dumps(result, indent=2, default=str)


def tool_sandbox_execute(code: str, timeout: float = 30.0) -> str:
    """Execute code in a warm sandbox worker (Python)."""
    pool = get_sandbox_pool()
    result = pool.execute(code)
    return json.dumps(
        {
            "success": result["success"],
            "output": result.get("output", "")[:2000],
            "error": result.get("error", ""),
            "worker_reused": result.get("worker_reused", False),
        }
    )


def tool_sandbox_stats() -> str:
    """Get sandbox pool statistics."""
    pool = get_sandbox_pool()
    return json.dumps(pool.stats, indent=2)


@_lm_tool
def tool_gfl_run_halving(module_name: str, trainset_path: str) -> str:
    """Run GFL pipeline with Successive Halving early pruning."""
    # `dspytools.gfl` imports the MLflow tracing SDK via `evolve`; keep it off
    # the stdio startup path.
    from dspytools.gfl.pipeline import GFLPipeline

    student = load_module_by_name(module_name)
    trainset = load_trainset(trainset_path)
    pipeline = GFLPipeline()
    result = pipeline.run_halving(student=student, trainset=trainset)
    return json.dumps(
        {
            "best_optimizer": result["best_optimizer"],
            "best_score": result["best_score"],
            "survivors": result["survivors"],
            "pruned": result["pruned"],
        },
        indent=2,
    )


@_lm_tool
def tool_challenger_solver(rounds: int = 5) -> str:
    """Run R-Zero Challenger-Solver co-evolution."""
    from dspytools.gfl.synthetic import ChallengerSolver

    challenger = dspy.Predict("difficulty, previous_tasks -> task")
    solver = dspy.Predict("task -> output")
    cs = ChallengerSolver(challenger_program=challenger, solver_program=solver)
    result = cs.co_evolve(num_rounds=rounds)
    return json.dumps(result, indent=2, default=str)


@_lm_tool
def tool_meta_prompt_learn(tasks_json: str = "", iterations: int = 5) -> str:
    """Meta-learn system prompts across tasks (MetaSPO).

    Scores are only reported when the optimizer actually measured them: with no
    tasks/devsets there is nothing to evaluate, so the payload says
    ``status: not_evaluated`` instead of inventing a 0.5 score.
    """
    # If tasks_json provided, parse it; otherwise use empty dicts (self-test)
    tasks = {}
    devsets = {}
    if tasks_json:
        data = _j.loads(tasks_json)
        tasks = {k: dspy.Predict("input -> output") for k in data}
    from dspytools.gfl.paper_optimizers import MetaPromptOptimizer

    opt = MetaPromptOptimizer()
    result = opt.meta_learn(
        task_programs=tasks, dev_sets=devsets, num_iterations=iterations
    )
    payload: dict[str, Any] = {
        "meta_prompt": (result.get("final_meta_prompt") or "")[:500],
    }
    if tasks and result.get("final_score") is not None:
        payload["final_score"] = result["final_score"]
        payload["iterations"] = result.get("iterations", [])
    else:
        payload["final_score"] = None
        payload["status"] = "not_evaluated"
        payload["reason"] = (
            "no task programs/devsets were supplied, so no score could be measured"
        )
    return json.dumps(payload, indent=2)


def tool_compile_cost(run_id: str) -> str:
    """Get cost and lineage details for a compiled run."""

    meta = get_run(run_id)
    lineage = get_lineage(run_id) if meta else []

    return json.dumps(
        {
            "run_id": run_id,
            "metadata": meta,
            "lineage_chain": [
                entry.get("lineage", {}).get("optimizer", "?") for entry in lineage
            ],
            "lineage_depth": len(lineage),
        },
        indent=2,
        default=str,
    )


def tool_skills_external_search(query: str, k: int = 10, category: str = "") -> str:
    """Search the open agent skills ecosystem (skills.sh) for matching skills."""

    live = try_skills_sh_api(query)
    if live:
        results_data = live[:k]
        source_label = "skills.sh"
    else:
        skills = search_external(query, k=k)
        if category:
            skills = [s for s in skills if s.category == category]
            skills = skills[:k]
        results_data = [
            {
                "name": s.name,
                "source": s.source,
                "description": s.description,
                "installs": s.installs,
                "category": s.category,
                "install_command": s.install_command,
                "url": s.browse_url,
            }
            for s in skills
        ]
        source_label = "curated"

    return json.dumps(
        {
            "query": query,
            "source": source_label,
            "results": results_data,
            "categories": list_categories() if not category else [],
        },
        indent=2,
    )


def tool_drift_status() -> str:
    """Check all programs for quality drift."""

    monitor = get_drift_monitor()
    return json.dumps(monitor.status, indent=2)


def tool_drift_history(run_id: str, n: int = 10) -> str:
    """Get recent quality snapshots for a program."""

    monitor = get_drift_monitor()
    return json.dumps(monitor.get_history(run_id, n), indent=2)


def tool_drift_auto_fix(dry_run: bool = True) -> str:
    """Check and process programs degraded by drift that need recompilation.

    Args:
        dry_run: If True, just report pending recompiles. If False, trigger them.
    """

    monitor = get_drift_monitor()
    pending = monitor.pending_recompiles()

    if not pending:
        return json.dumps({"status": "ok", "message": "No programs need recompilation"})

    if dry_run:
        return json.dumps(
            {
                "status": "pending",
                "programs": pending,
                "message": f"{len(pending)} program(s) queued — call with dry_run=false to fix",
            }
        )

    results = monitor.process_recompile_requests(auto_fix=True)
    return json.dumps(
        {
            "status": "processed",
            "results": results,
            "remaining": monitor.pending_recompiles(),
        }
    )


@_lm_tool
def tool_stream_infer(input_text: str, program_id: str = "") -> str:
    """Run inference with the active compiled program (simulated streaming).

    Returns full output with token count metadata.

    The module-level `_get_hotswap()` manager is used (not a fresh
    `HotSwapManager()`): swaps and the LRU cache are process state, and
    `mgr.swap()` on a throwaway instance mutated an object nobody kept.
    """

    mgr = _get_hotswap()

    if program_id:
        mgr.swap(program_id)  # auto-loads single program on demand

    result = mgr.infer(input=input_text)
    output = result.get("output", str(result))

    return json.dumps(
        {
            "streaming": True,
            "output": output,
            "tokens": len(output.split()),
            "program_id": program_id or mgr.active_id or "none",
        },
        indent=2,
        default=str,
    )


def tool_analysis_cache_stats() -> str:
    """Show analysis cache statistics."""

    cache = get_analysis_cache()
    return json.dumps(cache.stats, indent=2)


def tool_analysis_cache_invalidate(key: str = "") -> str:
    """Invalidate analysis cache entries."""

    cache = get_analysis_cache()
    count = cache.invalidate(key if key else None)
    return json.dumps(
        {"invalidated": count, "remaining": cache.stats["memory_entries"]}
    )


@_lm_tool
def tool_doctor() -> str:
    """Run system diagnostics — LLM health, GPU status, config, dependencies.

    Every check is a measurement: ``dspy`` reports whether an LM is actually
    configured (not a hardcoded True), and the MLflow probe imports the SDK
    on demand so server import stays light.
    """
    checks = {
        "llm": False,
        "gpu": False,
        "dspy": getattr(dspy.settings, "lm", None) is not None,
        "mlflow": False,
    }
    details = {"dspy_version": getattr(dspy, "__version__", "unknown")}

    # LLM check
    try:
        base = llama_cpp_url()
        checks["llm"] = False
        for path in ("/v1/models", "/api/tags", "/health"):
            try:
                resp = urllib.request.urlopen(f"{base}{path}", timeout=3)
                if resp.status == 200:
                    checks["llm"] = True
                    break
            except (ConnectionError, OSError, TimeoutError) as e:
                _log.debug("llm_health_check_failed", error=str(e))
                continue
        details["llm_url"] = base
    except (ConnectionError, OSError, TimeoutError, ValueError) as e:
        _log.error("doctor_llm_check_failed", error=str(e))
        details["llm_error"] = str(e)

    # GPU check
    try:
        result = _sp.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            checks["gpu"] = True
            details["gpu"] = result.stdout.strip()
    except (OSError, FileNotFoundError, TimeoutError) as e:
        _log.error("doctor_gpu_check_failed", error=str(e))
        details["gpu_error"] = str(e)

    # MLflow check — imported here (not at module import) so the stdio startup
    # path never pulls the MLflow SDK in.
    try:
        import mlflow

        details["mlflow_tracking_uri"] = mlflow.get_tracking_uri()
        checks["mlflow"] = True
    except (ConnectionError, OSError, ImportError) as e:
        _log.error("doctor_mlflow_check_failed", error=str(e))
        details["mlflow_error"] = str(e)

    return json.dumps(
        {
            "healthy": all(checks.values()),
            "checks": checks,
            "details": details,
        },
        indent=2,
    )


@_lm_tool
def tool_trace2skill_evolve(
    program_id: str, tasks_json: str = "", skill_name: str = "trace2skill"
) -> str:
    """Run Trace2Skill evolution: consolidate execution trajectories into a reusable skill.

    Uses the 3-stage pipeline from arXiv 2603.25158:
      1. Rollout — parallel task execution
      2. Analyze — LLM-driven Success + Error Analysts
      3. Consolidate — hierarchical merge with inductive reasoning
    """

    mgr = _get_hotswap()
    mgr._ensure_index()

    try:
        mgr.swap(program_id)
    except (RuntimeError, OSError, ValueError, TypeError) as e:
        _log.warning("program_swap_failed", error=str(e))
        return _error(f"Program '{program_id}' not found")

    program = mgr.get(program_id)
    if program is None:
        return _error(f"Program '{program_id}' not loaded")

    # Parse tasks
    if tasks_json:
        tasks = json.loads(tasks_json)
    else:
        tasks = [{"input": {"input": "hello"}, "expected": "Hello, World!"}]

    from dspytools.gfl.consolidation import SkillConsolidator

    metric = exact_match_metric()
    consolidator = SkillConsolidator()
    result = consolidator.evolve(
        program=program,
        tasks=tasks,
        metric=metric,
        skill_name=skill_name,
        mode="creation",
    )

    return json.dumps(
        {
            "skill_name": result.skill_name,
            "patches_generated": result.patches_generated,
            "patches_accepted": result.patches_accepted,
            "patches_discarded": result.patches_discarded,
            "trajectories_analyzed": result.trajectories_analyzed,
            "success_trajectories": result.success_trajectories,
            "error_trajectories": result.error_trajectories,
            "guardrail_failures": result.guardrail_failures,
            "quality_dropped": result.quality_dropped,
            "elapsed_seconds": result.elapsed_seconds,
            "mode": result.mode,
            "evolved_skill_preview": result.evolved_skill[:200],
        },
        indent=2,
    )


@_lm_tool
def tool_spin_optimize(
    module_name: str, trainset_json: str = "", num_iterations: int = 3
) -> str:
    """Run SPIN self-play optimization (arXiv 2401.01335).

    Self-Play fIne-tuNing: generates candidate outputs, then uses teacher LM
    as discriminator to score how well model distinguishes own generations from gold.
    """

    student = load_module_by_name(module_name)
    trainset = _parse_trainset(trainset_json)

    from dspytools.gfl.paper_optimizers import SPINOptimizer

    opt = SPINOptimizer(student=student)
    result = opt.iterate(trainset, num_iterations=num_iterations)

    return json.dumps(
        {
            "optimizer": "SPIN",
            "module": module_name,
            "iterations": result["iterations"],
            "final_score": result["final_score"],
            "improvement": result["improvement"],
        },
        indent=2,
    )


def _parse_trainset(
    trainset_json: str = "",
    default_input: str = "hello",
    default_output: str = "world",
) -> list:
    """Parse a JSON trainset, falling back to one documented placeholder example.

    The placeholder is a *self-test* example (it is never used as compile data
    where a real score is reported): `tool_lse_explore` skips instead of scoring
    against it, while SPIN/OPSD keep their pre-existing self-test behaviour.
    """
    if trainset_json:
        raw = json.loads(trainset_json)
        return [
            dspy.Example(**item).with_inputs(next(iter(item.keys()))) for item in raw
        ]
    return [
        dspy.Example(input=default_input, output=default_output).with_inputs("input")
    ]


@_lm_tool
def tool_opsd_purify(
    module_name: str,
    iterations: int = 3,
    trainset_json: str = "",
    base_optimizer: str = "",
    beta: float = 1.0,
    clip_c: float = 10.0,
) -> str:
    """Purified OPSD (arXiv 2607.02234) — PMI-refined teacher distillation.

    Replaces teacher log-probs with PMI-based target that preserves
    question-relevant knowledge while rejecting reference-induced shortcuts.
    Can wrap any existing optimizer for combined purify+optimize.
    """
    student = load_module_by_name(module_name)
    trainset = _parse_trainset(trainset_json)

    base_opt = None
    if base_optimizer:
        base_opt = _make_optimizer_instance(base_optimizer)

    from dspytools.gfl.paper_optimizers import PurifiedOPSDOptimizer

    opt = PurifiedOPSDOptimizer(
        student=student,
        base_optimizer=base_opt,
        beta=beta,
        clip_c=clip_c,
    )
    result = opt.iterate(trainset, num_iterations=iterations)
    stats = result["purification_stats"]

    return json.dumps(
        {
            "optimizer": "PurifiedOPSD",
            "module": module_name,
            "iterations": result["iterations"],
            "final_score": result["final_score"],
            "improvement": result["improvement"],
            "purification_stats": {
                "avg_pmi_weight": stats["avg_pmi_weight"],
                "positive_pmi_count": stats["positive_pmi_count"],
                "negative_pmi_count": stats["negative_pmi_count"],
                "total_pmi_signals": stats["total_pmi_signals"],
            },
        },
        indent=2,
    )


def _measure(program: Any, devset: list) -> float:
    """Score one compiled program on a devset — the only source of LSE scores."""
    evaluator = dspy.Evaluate(
        devset=devset,
        metric=exact_match_metric(),
        num_threads=1,
        display_progress=False,
    )
    return float(evaluator(program).score)


@_lm_tool
def tool_lse_explore(
    module_name: str, trainset_json: str = "", max_depth: int = 3
) -> str:
    """Run LSE tree-guided evolution (arXiv 2603.18620).

    Tree-guided evolution with UCB selection. Each node is an optimization
    attempt; its score is a **real measurement** — the compiled program is
    evaluated on the holdout half of ``trainset_json``. Without a trainset there
    is nothing to measure, so the tool reports ``status: skipped`` instead of
    inventing scores from the tree depth.
    """
    if not (trainset_json or "").strip():
        return json.dumps(
            {
                "status": "skipped",
                "reason": (
                    "lse_explore scores every node by evaluating the compiled "
                    "program on the holdout split of `trainset_json`; without a "
                    "trainset there is no measurement to report."
                ),
            },
            indent=2,
        )

    student = load_module_by_name(module_name)
    trainset = _parse_trainset(trainset_json)
    if len(trainset) < 2:
        return json.dumps(
            {
                "status": "skipped",
                "reason": (
                    f"needs at least 2 examples to split a holdout; got {len(trainset)}"
                ),
            },
            indent=2,
        )

    from dspytools.gfl.paper_optimizers import LSETreeExplorer

    lse = LSETreeExplorer(max_depth=max_depth)
    root = lse.new_root()
    results = []
    gate = get_holdout_gate()

    for depth in range(max_depth):
        for optimizer in ["bootstrap-few-shot", "mipro", "gepa"]:
            try:
                opt = _make_optimizer_instance(optimizer)
                compile_id = f"mcp:lse:{optimizer}:{depth}"
                train_set, holdout = gate.split(trainset, compile_id=compile_id)
                compiled = opt.compile(student=student, trainset=train_set)
                score = _measure(compiled, holdout)
                node = lse.expand(root, optimizer, score, f"depth={depth}")
                selected = lse.select(node)
                results.append(
                    {
                        "depth": depth,
                        "optimizer": optimizer,
                        "score": score,
                        "scored_on": f"holdout[{len(holdout)}]",
                        "selected": selected,
                    }
                )
            except (RuntimeError, OSError, ValueError, TypeError) as e:
                _log.warning("lse_tree_node_failed", error=str(e))
                results.append(
                    {
                        "depth": depth,
                        "optimizer": optimizer,
                        "status": "failed",
                        "error": str(e),
                    }
                )

    return json.dumps(
        {
            "tree_size": len(lse.tree),
            "max_depth": max_depth,
            "nodes": results,
        },
        indent=2,
    )


def tool_gepa_frontier(module_name: str, scores_json: str = "") -> str:
    """Run GEPA Pareto frontier optimization (arXiv 2507.19457).

    Pareto frontier optimization with coverage-weighted selection.
    Candidates dominate if score not exceeded by any frontier member.
    """

    from dspytools.gfl.paper_optimizers import GEPAParetoFrontier

    frontier = GEPAParetoFrontier()

    if scores_json:
        candidates = json.loads(scores_json)
        for c in candidates:
            frontier.add(
                optimizer=c.get("optimizer", "unknown"),
                score=c.get("score", 0.0),
                feedback=c.get("feedback", ""),
            )
    else:
        frontier.add("bootstrap", 0.60, "Baseline")
        frontier.add("mipro", 0.75, "Bayesian improvement")
        frontier.add("gepa", 0.82, "Evolutionary refinement")

    next_opt = frontier.select_next()
    return json.dumps(
        {
            "frontier_size": len(frontier.frontier),
            "candidates": len(frontier.candidates),
            "next_optimizer": next_opt,
        },
        indent=2,
    )


def _gepa_feedback_metric(
    gold: Any,
    pred: Any,
    trace: Any = None,
    pred_name: Any = None,
    pred_trace: Any = None,
) -> float:
    """Five-argument GEPA feedback metric `(gold, pred, trace, pred_name, pred_trace)`.

    GEPA validates its metric by binding five positional arguments and refuses
    the three-argument `exact_match_metric()` factories — use this for any
    GEPA optimizer created from MCP tools.
    """
    return exact_match_metric()(gold, pred, trace)


def _make_optimizer_instance(name: str):
    """Construct an optimizer from the shared `_OPTIMIZER_SPECS` table.

    Accepts any listed optimizer name (hyphenated canonical spelling) plus the
    legacy underscore aliases, so a name from `list_optimizers` always resolves
    here too. GEPA needs two things the shared `exact_match_metric()` factory
    cannot supply: a five-argument metric and a reflection LM. The reflection LM
    is always the teacher role (`LMRegistry.get_teacher()`) — it is never
    satisfied by the student, and a missing teacher raises instead of silently
    degrading the tool run.
    """
    canonical = _canonical_optimizer_name(name)
    if canonical in _CLI_ONLY_OPTIMIZERS:
        raise ValueError(
            f"Optimizer '{canonical}' exists only as a CLI command "
            f"(`dspytools compile {canonical}`): {_CLI_ONLY_OPTIMIZERS[canonical]}"
        )
    spec = _optimizer_specs().get(canonical)
    if spec is None:
        raise ValueError(
            f"Unknown optimizer: {name}. Available: "
            + ", ".join(sorted(_optimizer_specs()))
        )

    # Copy: the spec's params dict is process-wide state.
    params = dict(spec.get("params") or {})
    metric: Any = exact_match_metric()
    if canonical == "gepa":
        metric = _gepa_feedback_metric
    if spec.get("needs_teacher"):
        teacher = LMRegistry.get_teacher()
        if teacher is None:
            raise ValueError(
                "gepa requires a teacher (reflection) LM — configure one with "
                "`dspytools configure lm set teacher <model>` (or set "
                "DEEPSEEK_API_KEY) and retry. LMRegistry.get_teacher() returned None."
            )
        params["reflection_lm"] = teacher
    return spec["cls_lambda"](metric=metric, **params)


# Register additional tools


def tool_compile_stats() -> str:
    """Show compile statistics including retry history."""

    runs = list_compiled_runs()
    recent = runs[:10] if len(runs) > 10 else runs

    stats = []
    for run in recent:
        lineage = run.get("lineage", {})
        cost_info = run.get("metadata", {})
        stats.append(
            {
                "run_id": run.get("id", "?"),
                "optimizer": lineage.get("optimizer", "?"),
                "score": cost_info.get("score", "?"),
                "timestamp": lineage.get("timestamp", "?"),
            }
        )

    return json.dumps({"recent_runs": len(stats), "runs": stats}, indent=2, default=str)


def tool_holdout_status() -> str:
    """Show holdout gate status — which programs are gated."""

    gate = get_holdout_gate()
    return json.dumps(gate.stats, indent=2)


_EXTRA_TOOLS = {
    "list_optimizers": {
        "description": (
            "List optimizers compile_optimizer can run, plus CLI-only optimizer "
            "families that it cannot"
        ),
        "inputSchema": {"type": "object", "properties": {}},
        "annotations": {
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        },
        "handler": tool_list_optimizers,
    },
    "skills_list": {
        "description": "List all available skills (BM25-indexed library)",
        "inputSchema": {"type": "object", "properties": {}},
        "annotations": {
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        },
        "handler": tool_skills_list,
    },
    "skills_search": {
        "description": "BM25 + embedding search for skills matching a query",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "k": {"type": "integer", "default": 5},
            },
            "required": ["query"],
        },
        "annotations": {
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        },
        "handler": tool_skills_search,
    },
    "self_status": {
        "description": "Show self-optimization status (compiled help cache, self-evolve state)",
        "inputSchema": {"type": "object", "properties": {}},
        "annotations": {
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        },
        "handler": tool_self_status,
    },
    "inspect_history": {
        "description": "Show last N DSPy LM call history entries",
        "inputSchema": {
            "type": "object",
            "properties": {"n": {"type": "integer", "default": 5}},
        },
        "annotations": {
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        },
        "handler": tool_inspect_history,
    },
    "mlflow_status": {
        "description": "Report MLflow tracking status (enabled, tracking URI, experiment)",
        "inputSchema": {"type": "object", "properties": {}},
        "annotations": {
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        },
        "handler": tool_mlflow_status,
    },
    "generate_llms_txt": {
        "description": "Generate llms.txt for a repository (URL or local path)",
        "inputSchema": {
            "type": "object",
            "properties": {
                "target": {"type": "string"},
                "local": {"type": "boolean", "default": False},
                "output": {"type": "string", "default": ""},
            },
            "required": ["target"],
        },
        "annotations": {
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": False,
            "openWorldHint": True,
        },
        "handler": tool_generate_llms_txt,
    },
    "compile_optimizer": {
        "description": "Trigger a DSPy compile with a specific optimizer",
        "inputSchema": {
            "type": "object",
            "properties": {
                "optimizer": {"type": "string"},
                "module_name": {"type": "string"},
                "trainset_path": {"type": "string"},
            },
            "required": ["optimizer", "module_name", "trainset_path"],
        },
        "annotations": {
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": False,
            "openWorldHint": True,
        },
        "handler": tool_compile_optimizer,
    },
    "archive_search": {
        "description": "Search compiled program archive for similar past compilations (Meta Agent Search)",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "top_k": {"type": "integer", "default": 3},
            },
            "required": ["query"],
        },
        "annotations": {
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        },
        "handler": tool_archive_search,
    },
    "validate_deploy": {
        "description": "SPRT-powered validation: test if a compiled program is ready to deploy",
        "inputSchema": {
            "type": "object",
            "properties": {
                "program_id": {"type": "string"},
                "holdout_path": {"type": "string", "default": ""},
                "alpha": {"type": "number", "default": 0.05},
                "beta": {"type": "number", "default": 0.2},
            },
            "required": ["program_id"],
        },
        "annotations": {
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        },
        "handler": tool_validate_deploy,
    },
    "sandbox_execute": {
        "description": "Execute Python code in a warm sandbox worker (pooled subprocess)",
        "inputSchema": {
            "type": "object",
            "properties": {
                "code": {"type": "string"},
                "timeout": {"type": "number", "default": 30.0},
            },
            "required": ["code"],
        },
        "annotations": {
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        },
        "handler": tool_sandbox_execute,
    },
    "sandbox_stats": {
        "description": "Get sandbox worker pool statistics (active/in-use/available)",
        "inputSchema": {"type": "object", "properties": {}},
        "annotations": {
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        },
        "handler": tool_sandbox_stats,
    },
    "gfl_run_halving": {
        "description": "Run GFL 4-way comparison with Successive Halving early pruning",
        "inputSchema": {
            "type": "object",
            "properties": {
                "module_name": {"type": "string"},
                "trainset_path": {"type": "string"},
            },
            "required": ["module_name", "trainset_path"],
        },
        "annotations": {
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": False,
            "openWorldHint": True,
        },
        "handler": tool_gfl_run_halving,
    },
    "challenger_solver": {
        "description": "Run R-Zero Challenger-Solver co-evolution (task generation + solving)",
        "inputSchema": {
            "type": "object",
            "properties": {"rounds": {"type": "integer", "default": 5}},
        },
        "annotations": {
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": False,
            "openWorldHint": True,
        },
        "handler": tool_challenger_solver,
    },
    "trace2skill_evolve": {
        "description": "Trace2Skill: evolve agent skills from execution trajectories (arXiv 2603.25158). 3-stage pipeline — rollout, analyze, consolidate.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "program_id": {
                    "type": "string",
                    "description": "ID of the compiled program to evolve skills from",
                },
                "tasks_json": {
                    "type": "string",
                    "default": "",
                    "description": "JSON array of {input: dict, expected: str} tasks",
                },
                "skill_name": {
                    "type": "string",
                    "default": "trace2skill",
                    "description": "Name for the evolved skill",
                },
            },
            "required": ["program_id"],
        },
        "annotations": {
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": False,
            "openWorldHint": True,
        },
        "handler": tool_trace2skill_evolve,
    },
    "spin_optimize": {
        "description": "SPIN self-play optimization (arXiv 2401.01335) — teacher LM discriminates model outputs vs gold",
        "inputSchema": {
            "type": "object",
            "properties": {
                "module_name": {"type": "string", "description": "DSPy module name"},
                "trainset_json": {
                    "type": "string",
                    "default": "",
                    "description": "JSON array of training examples",
                },
                "num_iterations": {
                    "type": "integer",
                    "default": 3,
                    "description": "Number of SPIN iterations",
                },
            },
            "required": ["module_name"],
        },
        "annotations": {
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": False,
            "openWorldHint": True,
        },
        "handler": tool_spin_optimize,
    },
    "opsd_purify": {
        "description": "Purified OPSD (arXiv 2607.02234) — PMI-refined teacher distillation without losing how to think",
        "inputSchema": {
            "type": "object",
            "properties": {
                "module_name": {"type": "string", "description": "DSPy module name"},
                "iterations": {
                    "type": "integer",
                    "default": 3,
                    "description": "Number of purification iterations",
                },
                "trainset_json": {
                    "type": "string",
                    "default": "",
                    "description": "JSON array of training examples",
                },
                "base_optimizer": {
                    "type": "string",
                    "default": "",
                    "description": "Optional base optimizer to wrap (e.g. 'spin', 'gepa')",
                },
                "beta": {
                    "type": "number",
                    "default": 1.0,
                    "description": "PMI correction strength",
                },
                "clip_c": {
                    "type": "number",
                    "default": 10.0,
                    "description": "Tanh soft clipping threshold",
                },
            },
            "required": ["module_name"],
        },
        "annotations": {
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": False,
            "openWorldHint": True,
        },
        "handler": tool_opsd_purify,
    },
    "lse_explore": {
        "description": "LSE tree-guided evolution (arXiv 2603.18620) — UCB-balanced exploration tree",
        "inputSchema": {
            "type": "object",
            "properties": {
                "module_name": {"type": "string", "description": "DSPy module name"},
                "trainset_json": {
                    "type": "string",
                    "default": "",
                    "description": (
                        "JSON array of examples; required for a real score — each "
                        "node is evaluated on the holdout split of this data"
                    ),
                },
                "max_depth": {
                    "type": "integer",
                    "default": 3,
                    "description": "Maximum tree depth",
                },
            },
            "required": ["module_name"],
        },
        "annotations": {
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": False,
            "openWorldHint": True,
        },
        "handler": tool_lse_explore,
    },
    "gepa_frontier": {
        "description": "GEPA Pareto frontier (arXiv 2507.19457) — coverage-weighted candidate selection",
        "inputSchema": {
            "type": "object",
            "properties": {
                "module_name": {"type": "string", "description": "DSPy module name"},
                "scores_json": {
                    "type": "string",
                    "default": "",
                    "description": "JSON array of {optimizer, score, feedback}",
                },
            },
            "required": ["module_name"],
        },
        "annotations": {
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": False,
            "openWorldHint": True,
        },
        "handler": tool_gepa_frontier,
    },
    "meta_prompt_learn": {
        "description": "Meta-learn system prompts across tasks (MetaSPO bilevel optimization)",
        "inputSchema": {
            "type": "object",
            "properties": {
                "tasks_json": {"type": "string", "default": ""},
                "iterations": {"type": "integer", "default": 5},
            },
        },
        "annotations": {
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": False,
            "openWorldHint": True,
        },
        "handler": tool_meta_prompt_learn,
    },
    "compile_cost": {
        "description": "Get cost breakdown and lineage chain for a compiled program",
        "inputSchema": {
            "type": "object",
            "properties": {
                "run_id": {
                    "type": "string",
                    "description": "Run ID from list_compiled_runs",
                }
            },
            "required": ["run_id"],
        },
        "annotations": {
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        },
        "handler": tool_compile_cost,
    },
    "skills_external_search": {
        "description": "Search the open agent skills ecosystem (skills.sh) for matching skills by keyword",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query (e.g., 'react', 'testing', 'deploy')",
                },
                "k": {
                    "type": "integer",
                    "default": 10,
                    "description": "Number of results",
                },
                "category": {
                    "type": "string",
                    "default": "",
                    "description": "Filter by category",
                },
            },
            "required": ["query"],
        },
        "annotations": {
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        },
        "handler": tool_skills_external_search,
    },
    "drift_status": {
        "description": "Check all compiled programs for quality drift from baseline",
        "inputSchema": {"type": "object", "properties": {}},
        "annotations": {
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        },
        "handler": tool_drift_status,
    },
    "drift_history": {
        "description": "Get recent quality snapshots for a specific program",
        "inputSchema": {
            "type": "object",
            "properties": {
                "run_id": {"type": "string"},
                "n": {"type": "integer", "default": 10},
            },
            "required": ["run_id"],
        },
        "annotations": {
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        },
        "handler": tool_drift_history,
    },
    "drift_auto_fix": {
        "description": "Check and auto-fix programs degraded by drift that need recompilation",
        "inputSchema": {
            "type": "object",
            "properties": {
                "dry_run": {"type": "boolean", "default": True},
            },
        },
        "annotations": {
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        },
        "handler": tool_drift_auto_fix,
    },
    "stream_infer": {
        "description": "Run streaming inference with the active compiled program (simulated streaming)",
        "inputSchema": {
            "type": "object",
            "properties": {
                "input_text": {
                    "type": "string",
                    "description": "Input text for inference",
                },
                "program_id": {
                    "type": "string",
                    "default": "",
                    "description": "Specific program to use",
                },
            },
            "required": ["input_text"],
        },
        "annotations": {
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": True,
        },
        "handler": tool_stream_infer,
    },
    "compile_stats": {
        "description": "Show recent compile statistics and retry history",
        "inputSchema": {"type": "object", "properties": {}},
        "annotations": {
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        },
        "handler": tool_compile_stats,
    },
    "doctor": {
        "description": "Run system diagnostics — LLM, GPU, MLflow, config health check",
        "inputSchema": {"type": "object", "properties": {}},
        "annotations": {
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        },
        "handler": tool_doctor,
    },
    "holdout_status": {
        "description": "Show holdout isolation gate status and stored splits",
        "inputSchema": {"type": "object", "properties": {}},
        "annotations": {
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        },
        "handler": tool_holdout_status,
    },
    "analysis_cache_stats": {
        "description": "Show analysis cache statistics (entries, disk usage, TTL)",
        "inputSchema": {"type": "object", "properties": {}},
        "annotations": {
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        },
        "handler": tool_analysis_cache_stats,
    },
    "analysis_cache_invalidate": {
        "description": "Invalidate analysis cache (all or by key)",
        "inputSchema": {
            "type": "object",
            "properties": {
                "key": {
                    "type": "string",
                    "default": "",
                    "description": "Cache key to invalidate (empty = all)",
                }
            },
        },
        "annotations": {
            "readOnlyHint": False,
            "destructiveHint": True,
            "idempotentHint": True,
            "openWorldHint": False,
        },
        "handler": tool_analysis_cache_invalidate,
    },
}

BUILTIN_TOOLS.update(_EXTRA_TOOLS)


@_lm_tool
def tool_evaluate(module: str, devset_path: str) -> str:
    """Evaluate a module on a devset."""

    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        program = _load_program(module)
        examples = load_trainset(devset_path)
        metric = _get_metric("exact_match")

        evaluator = dspy.Evaluate(
            devset=examples, metric=metric, num_threads=1, display_progress=False
        )
        result = evaluator(program)
        return json.dumps({"score": result.score, "examples": len(examples)})
    finally:
        sys.stdout = old_stdout


def tool_gfl_synthesize(seed_path: str, target: int = 10) -> str:
    """Generate synthetic training data from seed examples."""

    from dspytools.gfl.synthetic import DataSynthesizer

    synth = DataSynthesizer()
    result = synth.generate(seed_path, target_count=target)
    return json.dumps(result)


def _last_lm_output() -> str:
    """Return the most recent raw LM response text recorded by the active LM.

    DSPy LMs append every call to `lm.history`; lm15 entries carry
    ``messages`` / ``prompt`` / ``outputs`` / ``response`` / ``timestamp``.
    ``outputs`` is the documented, adapter-independent field (a list of typed
    outputs — plain strings, ``{"text": ...}`` dicts or objects exposing
    ``.text``); older or alternate shapes are covered by the ``response``
    fallback. GEPA/ReAct failures are far easier to diagnose with the model
    output that failed to parse, so `tool_agent_run`'s error payload includes
    it.
    """
    lm = getattr(dspy.settings, "lm", None)
    history = getattr(lm, "history", None) or []
    if not history:
        return ""
    last = history[-1]
    if not isinstance(last, dict):
        return str(last)
    outputs = last.get("outputs")
    if isinstance(outputs, (list, tuple)) and outputs:
        first = outputs[0]
        if isinstance(first, dict):
            return str(first.get("text", first))
        text = getattr(first, "text", None)
        return str(text) if text is not None else str(first)
    # Older/alternate entry shapes.
    for key in ("response", "output"):
        value = last.get(key)
        if value is not None:
            text = getattr(value, "text", None)
            return str(text) if text is not None else str(value)
    return ""


def _configured_adapter() -> Any | None:
    """The adapter `setup_dspy()` configured (honours config `dspy.adapter`).

    `run.py` scopes work with the *active* adapter instead of rebuilding a
    hardcoded one; `tool_agent_run` follows the same rule. No clone is needed
    because this turn changes no adapter option — it is only scoped with
    `dspy.context(adapter=...)`, so global settings stay untouched.
    """
    return getattr(dspy.settings, "adapter", None)


def _agent_tools(tool_names: list[str]) -> list:
    """Wrap built-in handlers as `dspy.Tool`, keeping their MCP input schemas.

    `dspy.Tool(handler)` would re-derive the schema from the Python signature,
    which loses every description/default declared in the tool spec. The spec's
    `inputSchema` goes through the loader's `_schema_to_tool_args` so optional
    MCP arguments are not advertised as required.
    """
    from dspytools.mcp.loader import _schema_to_tool_args

    tools: list = []
    for name in tool_names:
        spec = BUILTIN_TOOLS.get(name)
        if spec is None:
            continue
        args, arg_types, arg_desc = _schema_to_tool_args(spec.get("inputSchema") or {})
        tools.append(
            dspy.Tool(
                spec["handler"],
                name=name,
                desc=spec["description"],
                args=args,
                arg_types=arg_types,
                arg_desc=arg_desc,
            )
        )
    return tools


@_lm_tool
def tool_agent_run(question: str, tools: str = "[]", max_iters: int = 8) -> str:
    """Run a ReAct agent with specified tools.

    TOOLS: JSON array of tool names registered in this MCP server (see
    `list_tools`), or ``[]`` for a no-tool reasoning agent.
    MAX_ITERS: ReAct turn budget (default 8).

    The adapter is the one `setup_dspy()` configured (config `dspy.adapter`),
    scoped with `dspy.context` — never a hardcoded `BAMLAdapter`.

    A failed forced submit raises a plain `ValueError` (`react.py` raises
    `ValueError` for both forced-submit paths; `AdapterParseError` and
    `ContextWindowExceededError` are surfaced the same way). Those are returned
    as a structured error payload naming the exception and the last model
    output, never propagated to the MCP transport.
    """

    tool_names = json.loads(tools) if tools else []
    dspy_tools = _agent_tools(tool_names)
    adapter = _configured_adapter()
    try:
        # DSPy 3.5: dspy.ReAct is the structured agent loop (typed `submit`
        # tool, History trace, termination_reason); ReActV2 is a deprecated
        # alias of it.
        agent = dspy.ReAct("question -> answer", tools=dspy_tools, max_iters=max_iters)
        context = (
            dspy.context(adapter=adapter) if adapter is not None else nullcontext()
        )
        with context:
            result = agent(question=question)
    except Exception as e:  # noqa: BLE001 — AdapterParseError is not a ValueError
        # MCP handlers report in-band — see the `_error` layering contract.
        return json.dumps(
            {
                "error": f"{type(e).__name__}: {e}",
                "exception": type(e).__name__,
                "last_model_output": _last_lm_output()[:1000],
                "detail": (
                    "agent run failed (a failed forced submit raises ValueError "
                    "from dspy's react loop)"
                ),
            }
        )
    return json.dumps(
        {
            "answer": getattr(result, "answer", str(result))[:500],
            "termination_reason": getattr(result, "termination_reason", "unknown"),
            "turns": len(
                getattr(getattr(result, "history", None), "messages", None) or []
            ),
        }
    )


def tool_lora_list_adapters() -> str:
    """List all LoRA-derived models in llama-cpp-server."""
    resp = urllib.request.urlopen(f"{llama_cpp_url()}/api/tags", timeout=5)
    data = json.loads(resp.read())
    models = data.get("models", [])
    lora_models = [
        {"name": m["name"], "status": "loaded"}
        for m in models
        if "-lora-" in m.get("name", "")
    ]
    return json.dumps(lora_models if lora_models else models, indent=2)


def tool_lora_load_adapter(name: str, path: str = "") -> str:
    """Create a LoRA-derived model in llama-cpp-server via /api/generate."""

    adapter_path = Path(path) if path else _adapters_dir() / name
    if not adapter_path.exists():
        return _error(f"Adapter not found: {adapter_path}")
    if not (adapter_path / "adapter_model.safetensors").exists():
        return _error(f"No adapter_model.safetensors in {adapter_path}")

    base_model = _get_base_model()
    model_name = _adapter_model_name(name)

    # llama-cpp uses /api/generate with adapters parameter
    payload = json.dumps(
        {
            "model": base_model,
            "adapter": str(adapter_path.resolve()),
            "prompt": "{{ .Prompt }}",
            "stream": False,
        }
    ).encode()
    req = urllib.request.Request(
        f"{llama_cpp_url()}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        resp.read()
    return json.dumps({"status": "loaded", "name": model_name}, indent=2)


def tool_lora_unload_adapter(name: str) -> str:
    """Report LoRA adapter unload status.

    llama.cpp's server exposes no unload endpoint: an adapter is bound when the
    model is created (`/api/generate` with `adapter`) and only disappears when
    that derived model is removed. This tool is therefore an explicit no-op
    rather than a fake ``{"status": "unloaded"}`` — the caller is told nothing
    happened and what to do instead.
    """

    try:
        model_name = name if "-lora-" in name else _adapter_model_name(name)
    except (AttributeError, KeyError, TypeError):
        # `_adapter_model_name` reads `lm.student` from config; an unset student
        # entry must not turn a status report into a crash.
        model_name = name
    return json.dumps(
        {
            "status": "unsupported",
            "name": model_name,
            "detail": (
                "llama.cpp has no LoRA unload endpoint; the adapter stays bound "
                "to the derived model until the server drops that model. Restart "
                "llama-cpp-server (or create a model without `adapter`) to unload it."
            ),
        },
        indent=2,
    )


_FINAL_TOOLS = {
    "evaluate": {
        "description": "Evaluate a module on a devset dataset",
        "inputSchema": {
            "type": "object",
            "properties": {
                "module": {"type": "string"},
                "devset_path": {"type": "string"},
            },
            "required": ["module", "devset_path"],
        },
        "annotations": {
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": False,
            "openWorldHint": True,
        },
        "handler": tool_evaluate,
    },
    "gfl_synthesize": {
        "description": "Generate synthetic training data from seed examples",
        "inputSchema": {
            "type": "object",
            "properties": {
                "seed_path": {"type": "string"},
                "target": {"type": "integer", "default": 10},
            },
            "required": ["seed_path"],
        },
        "annotations": {
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": False,
            "openWorldHint": True,
        },
        "handler": tool_gfl_synthesize,
    },
    "agent_run": {
        "description": "Run a ReAct agent to answer a question",
        "inputSchema": {
            "type": "object",
            "properties": {
                "question": {"type": "string", "description": "Question to answer"},
                "tools": {
                    "type": "string",
                    "default": "[]",
                    "description": (
                        "JSON array of tool names from list_tools, e.g. "
                        '["skills_search", "graph_stats"]; "[]" runs a no-tool agent'
                    ),
                },
                "max_iters": {
                    "type": "integer",
                    "default": 8,
                    "description": "ReAct turn budget",
                },
            },
            "required": ["question"],
        },
        "annotations": {
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": False,
            "openWorldHint": True,
        },
        "handler": tool_agent_run,
    },
    "lora_list_adapters": {
        "description": "List all LoRA-derived models in llama-cpp-server",
        "inputSchema": {"type": "object", "properties": {}},
        "annotations": {
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        },
        "handler": tool_lora_list_adapters,
    },
    "lora_load_adapter": {
        "description": "Create a LoRA-derived model in llama-cpp-server via /api/generate",
        "inputSchema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Adapter name"},
                "path": {"type": "string", "description": "Path to adapter directory"},
            },
            "required": ["name"],
        },
        "annotations": {
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": False,
            "openWorldHint": False,
        },
        "handler": tool_lora_load_adapter,
    },
    "lora_unload_adapter": {
        "description": "Unload a LoRA adapter from llama-cpp-server",
        "inputSchema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Adapter name"},
            },
            "required": ["name"],
        },
        "annotations": {
            "readOnlyHint": False,
            "destructiveHint": True,
            "idempotentHint": True,
            "openWorldHint": False,
        },
        "handler": tool_lora_unload_adapter,
    },
}

# ═══════════════════════════════════════════════════════════════════════════
# Graph Tools — FalkorDB


def tool_graph_query(cypher: str, params: str = "{}") -> str:
    """Execute a Cypher query on FalkorDB."""
    client = get_graph_client()
    graph = client.graph()
    parsed_params = json.loads(params)
    result = graph.query(cypher, parsed_params)
    # FalkorDB returns list of lists: [[val1, val2], ...]
    rows = [list(row) for row in result.result_set] if result.result_set else []
    return json.dumps({"status": "ok", "rows": rows}, default=str)


def tool_graph_skill_tree() -> str:
    """List all skills in the FalkorDB skill graph."""
    cached = _cached("graph_skill_tree")
    if cached:
        return cached
    graph = FalkorDBSkillGraph()
    skills = graph.list_skills()
    result = json.dumps({"status": "ok", "skills": skills})
    _set_cache("graph_skill_tree", result)
    return result


def tool_graph_program_lineage(run_id: str) -> str:
    """Show program ancestry chain."""
    graph = FalkorDBSkillGraph()
    lineage = graph.program_lineage(run_id)
    return json.dumps({"status": "ok", "lineage": lineage})


def tool_graph_stats() -> str:
    """Get graph statistics."""
    cached = _cached("graph_stats")
    if cached:
        return cached
    client = get_graph_client()
    g = client.graph()
    node_result = g.query("MATCH (n) RETURN labels(n) AS labels, count(n) AS cnt")
    edge_result = g.query("MATCH ()-[r]->() RETURN type(r) AS type, count(r) AS cnt")
    nodes = (
        {str(row[0]): row[1] for row in node_result.result_set}
        if node_result.result_set
        else {}
    )
    edges = (
        {str(row[0]): row[1] for row in edge_result.result_set}
        if edge_result.result_set
        else {}
    )
    total_nodes = sum(nodes.values())
    total_edges = sum(edges.values())
    result = json.dumps(
        {
            "status": "ok",
            "total_nodes": total_nodes,
            "total_edges": total_edges,
            "node_labels": nodes,
            "edge_types": edges,
        }
    )
    _set_cache("graph_stats", result)
    return result


def tool_graph_add_dependency(skill: str, depends_on: str) -> str:
    """Add a skill dependency edge in FalkorDB."""
    graph = FalkorDBSkillGraph()
    graph.add_dependency(skill, depends_on)
    _invalidate("graph_skill_tree")
    return json.dumps({"status": "ok", "from": skill, "to": depends_on})


def tool_graph_dependents(skill: str, transitive: bool = False) -> str:
    """Get skills that depend on a given skill."""
    graph = FalkorDBSkillGraph()
    if transitive:
        deps = graph.transitive_dependents(skill)
    else:
        deps = graph.get_dependents(skill)
    return json.dumps({"status": "ok", "skill": skill, "dependents": deps})


def tool_graph_record_program(
    run_id: str,
    optimizer: str,
    score: float = 0.0,
    dataset_hash: str = "",
    parent_id: str = "",
) -> str:
    """Record a compiled program in the graph."""
    graph = FalkorDBSkillGraph()
    graph.record_program(
        run_id=run_id,
        optimizer=optimizer,
        score=score,
        dataset_hash=dataset_hash or None,
        parent_id=parent_id or None,
    )
    return json.dumps(
        {"status": "ok", "run_id": run_id, "optimizer": optimizer, "score": score}
    )


def tool_memory_add(content: str, user_id: str = "dspytools") -> str:
    """Add a memory."""
    manager = get_memory_manager()
    result = manager.add(content, user_id=user_id)
    # Invalidate related caches
    _invalidate(f"memory_all:{user_id}")
    _invalidate(f"memory_stats:{user_id}")
    return json.dumps({"status": "ok", "result": result}, default=str)


def tool_memory_search(query: str, user_id: str = "dspytools", limit: int = 10) -> str:
    """Search memories."""
    manager = get_memory_manager()
    results = manager.search(query, user_id=user_id, limit=limit)
    return json.dumps({"status": "ok", "memories": results}, default=str)


def tool_memory_get_all(user_id: str = "dspytools") -> str:
    """Get all memories."""
    cache_key = f"memory_all:{user_id}"
    cached = _cached(cache_key)
    if cached:
        return cached
    manager = get_memory_manager()
    results = manager.get_all(user_id=user_id)
    result = json.dumps({"status": "ok", "memories": results}, default=str)
    _set_cache(cache_key, result)
    return result


def tool_memory_delete(memory_id: str) -> str:
    """Delete a memory."""
    manager = get_memory_manager()
    result = manager.delete(memory_id)
    # Invalidate every memory cache (user_id unknown) — a glob, not a literal
    # prefix: `_invalidate("memory_")` matched no key at all.
    _invalidate("memory_*")
    return json.dumps({"status": "ok", "result": result})


def tool_memory_update(memory_id: str, content: str) -> str:
    """Update a memory's content."""
    manager = get_memory_manager()
    result = manager.update(memory_id, content)
    _invalidate("memory_*")
    return json.dumps({"status": "ok", "result": result}, default=str)


def tool_memory_stats(user_id: str = "dspytools") -> str:
    """Get memory statistics."""
    cache_key = f"memory_stats:{user_id}"
    cached = _cached(cache_key)
    if cached:
        return cached
    manager = get_memory_manager()
    stats = manager.stats(user_id=user_id)
    result = json.dumps({"status": "ok", "stats": stats})
    _set_cache(cache_key, result)
    return result


def tool_cache_check(prompt: str) -> str:
    """Check if prompt has cached response."""
    cache = get_semantic_cache()
    result = cache.check(prompt)
    if result:
        return json.dumps({"status": "hit", "response": result})
    return json.dumps({"status": "miss"})


def tool_cache_store(prompt: str, response: str) -> str:
    """Store prompt-response pair in cache."""
    cache = get_semantic_cache()
    cache.store(prompt, response)
    return json.dumps({"status": "ok"})


def tool_cache_stats() -> str:
    """Get cache statistics."""
    cache = get_semantic_cache()
    stats = cache.stats()
    return json.dumps({"status": "ok", "stats": stats})


def tool_cache_clear() -> str:
    """Clear all cached entries."""
    cache = get_semantic_cache()
    cache.clear()
    return json.dumps({"status": "ok"})


def tool_redis_stats() -> str:
    """Get Redis cache statistics for all namespaces."""
    mcp = get_mcp_cache()
    compile_c = get_compile_cache()
    return json.dumps(
        {
            "status": "ok",
            "mcp_cache": mcp.stats(),
            "compile_cache": compile_c.stats(),
        },
        indent=2,
    )


def tool_redis_flush(namespace: str = "mcp") -> str:
    """Flush Redis cache for a namespace (mcp, compile, or all)."""
    flushed = 0
    if namespace in ("mcp", "all"):
        flushed += get_mcp_cache().flush()
    if namespace in ("compile", "all"):
        flushed += get_compile_cache().flush()
    return json.dumps({"status": "ok", "flushed": flushed, "namespace": namespace})


def tool_redis_get(key: str, namespace: str = "mcp") -> str:
    """Get a value from Redis cache by key."""
    cache = RedisCache(namespace=namespace)
    val = cache.get(key)
    if val is None:
        return json.dumps({"status": "miss", "key": key})
    return json.dumps({"status": "hit", "key": key, "value": val}, default=str)


def tool_redis_set(key: str, value: str, namespace: str = "mcp", ttl: int = 300) -> str:
    """Set a value in Redis cache."""
    cache = RedisCache(namespace=namespace)
    cache.set(key, value, ttl=ttl)
    return json.dumps({"status": "ok", "key": key, "namespace": namespace, "ttl": ttl})


_GRAPH_TOOLS = {
    "graph_query": {
        "description": "Execute a Cypher query on FalkorDB graph database",
        "inputSchema": {
            "type": "object",
            "properties": {
                "cypher": {"type": "string", "description": "Cypher query"},
                "params": {"type": "string", "description": "JSON parameters"},
            },
            "required": ["cypher"],
        },
        "annotations": {
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        },
        "handler": tool_graph_query,
    },
    "graph_skill_tree": {
        "description": "List all skills in the FalkorDB skill graph",
        "inputSchema": {"type": "object", "properties": {}},
        "annotations": {
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        },
        "handler": tool_graph_skill_tree,
    },
    "graph_program_lineage": {
        "description": "Show program ancestry chain from FalkorDB",
        "inputSchema": {
            "type": "object",
            "properties": {
                "run_id": {"type": "string", "description": "Run ID to trace"}
            },
            "required": ["run_id"],
        },
        "annotations": {
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        },
        "handler": tool_graph_program_lineage,
    },
    "graph_stats": {
        "description": "Get FalkorDB graph statistics",
        "inputSchema": {"type": "object", "properties": {}},
        "annotations": {
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        },
        "handler": tool_graph_stats,
    },
    "graph_add_dependency": {
        "description": "Add a skill dependency edge in FalkorDB",
        "inputSchema": {
            "type": "object",
            "properties": {
                "skill": {"type": "string", "description": "Skill name"},
                "depends_on": {"type": "string", "description": "Dependency name"},
            },
            "required": ["skill", "depends_on"],
        },
        "annotations": {
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        },
        "handler": tool_graph_add_dependency,
    },
    "graph_dependents": {
        "description": "Get skills that depend on a given skill",
        "inputSchema": {
            "type": "object",
            "properties": {
                "skill": {"type": "string", "description": "Skill name"},
                "transitive": {
                    "type": "boolean",
                    "description": "Include transitive dependents",
                    "default": False,
                },
            },
            "required": ["skill"],
        },
        "annotations": {
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        },
        "handler": tool_graph_dependents,
    },
    "graph_record_program": {
        "description": "Record a compiled program in the FalkorDB graph",
        "inputSchema": {
            "type": "object",
            "properties": {
                "run_id": {"type": "string", "description": "Run ID"},
                "optimizer": {"type": "string", "description": "Optimizer used"},
                "score": {"type": "number", "description": "Quality score"},
                "dataset_hash": {"type": "string", "description": "Dataset hash"},
                "parent_id": {"type": "string", "description": "Parent run ID"},
            },
            "required": ["run_id", "optimizer"],
        },
        "annotations": {
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        },
        "handler": tool_graph_record_program,
    },
    "memory_add": {
        "description": "Add a memory to FalkorDB persistent memory store",
        "inputSchema": {
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "Memory content"},
                "user_id": {
                    "type": "string",
                    "description": "User ID (default: dspytools)",
                },
            },
            "required": ["content"],
        },
        "annotations": {
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": False,
            "openWorldHint": True,
        },
        "handler": tool_memory_add,
    },
    "memory_search": {
        "description": "Search memories using semantic similarity",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "user_id": {"type": "string", "description": "User ID"},
                "limit": {
                    "type": "integer",
                    "description": "Max results",
                    "default": 10,
                },
            },
            "required": ["query"],
        },
        "annotations": {
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": True,
        },
        "handler": tool_memory_search,
    },
    "memory_get_all": {
        "description": "Get all memories for a user",
        "inputSchema": {
            "type": "object",
            "properties": {"user_id": {"type": "string", "description": "User ID"}},
        },
        "annotations": {
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": True,
        },
        "handler": tool_memory_get_all,
    },
    "memory_update": {
        "description": "Update a memory's content",
        "inputSchema": {
            "type": "object",
            "properties": {
                "memory_id": {"type": "string", "description": "Memory ID"},
                "content": {"type": "string", "description": "New content"},
            },
            "required": ["memory_id", "content"],
        },
        "annotations": {
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": True,
        },
        "handler": tool_memory_update,
    },
    "memory_delete": {
        "description": "Delete a memory by ID",
        "inputSchema": {
            "type": "object",
            "properties": {"memory_id": {"type": "string", "description": "Memory ID"}},
            "required": ["memory_id"],
        },
        "annotations": {
            "readOnlyHint": False,
            "destructiveHint": True,
            "idempotentHint": True,
            "openWorldHint": True,
        },
        "handler": tool_memory_delete,
    },
    "memory_stats": {
        "description": "Get memory statistics for a user",
        "inputSchema": {
            "type": "object",
            "properties": {"user_id": {"type": "string", "description": "User ID"}},
        },
        "annotations": {
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": True,
        },
        "handler": tool_memory_stats,
    },
    "cache_check": {
        "description": "Check if a prompt has a cached LLM response (semantic cache)",
        "inputSchema": {
            "type": "object",
            "properties": {
                "prompt": {"type": "string", "description": "Prompt to check"}
            },
            "required": ["prompt"],
        },
        "annotations": {
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        },
        "handler": tool_cache_check,
    },
    "cache_store": {
        "description": "Store a prompt-response pair in semantic cache",
        "inputSchema": {
            "type": "object",
            "properties": {
                "prompt": {"type": "string", "description": "Prompt"},
                "response": {"type": "string", "description": "Response to cache"},
            },
            "required": ["prompt", "response"],
        },
        "annotations": {
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        },
        "handler": tool_cache_store,
    },
    "cache_stats": {
        "description": "Get semantic cache statistics",
        "inputSchema": {"type": "object", "properties": {}},
        "annotations": {
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        },
        "handler": tool_cache_stats,
    },
    "cache_clear": {
        "description": "Clear all cached entries from semantic cache",
        "inputSchema": {"type": "object", "properties": {}},
        "annotations": {
            "readOnlyHint": False,
            "destructiveHint": True,
            "idempotentHint": True,
            "openWorldHint": False,
        },
        "handler": tool_cache_clear,
    },
    "redis_stats": {
        "description": "Get Redis cache statistics (MCP response cache + compile cache)",
        "inputSchema": {"type": "object", "properties": {}},
        "annotations": {
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        },
        "handler": tool_redis_stats,
    },
    "redis_flush": {
        "description": "Flush Redis cache for a namespace (mcp, compile, or all)",
        "inputSchema": {
            "type": "object",
            "properties": {
                "namespace": {
                    "type": "string",
                    "description": "Namespace to flush (mcp/compile/all)",
                    "default": "mcp",
                },
            },
        },
        "annotations": {
            "readOnlyHint": False,
            "destructiveHint": True,
            "idempotentHint": True,
            "openWorldHint": False,
        },
        "handler": tool_redis_flush,
    },
    "redis_get": {
        "description": "Get a value from Redis cache by key and namespace",
        "inputSchema": {
            "type": "object",
            "properties": {
                "key": {"type": "string", "description": "Cache key"},
                "namespace": {
                    "type": "string",
                    "description": "Namespace (mcp/compile)",
                    "default": "mcp",
                },
            },
            "required": ["key"],
        },
        "annotations": {
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        },
        "handler": tool_redis_get,
    },
    "redis_set": {
        "description": "Set a value in Redis cache with optional TTL",
        "inputSchema": {
            "type": "object",
            "properties": {
                "key": {"type": "string", "description": "Cache key"},
                "value": {"type": "string", "description": "Value to cache"},
                "namespace": {
                    "type": "string",
                    "description": "Namespace (mcp/compile)",
                    "default": "mcp",
                },
                "ttl": {
                    "type": "integer",
                    "description": "TTL in seconds",
                    "default": 300,
                },
            },
            "required": ["key", "value"],
        },
        "annotations": {
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        },
        "handler": tool_redis_set,
    },
}

BUILTIN_TOOLS.update(_GRAPH_TOOLS)

BUILTIN_TOOLS.update(_FINAL_TOOLS)
