# DOX — core engine

- DOX is a highly performant AGENTS.md hierarchy installed here
- This child doc is owned by the root AGENTS.md at `/home/octagono/dev/dspytools/AGENTS.md`
- This doc does not weaken DOX

## Purpose

The `core/` directory is the engine of the dspytools CLI package. It provides lazy DSPy importing, LM instance management (singleton registry with student/teacher roles), LRU-cached program hot-swapping, JSON-based run registry, async compile scheduling, shared dataset/module loading, and output directory management. Every other dspytools subpackage depends on core.

## Ownership

This doc owns all files directly in `src/dspytools/core/` and their contracts. There are no subdirectories, so no child DOX docs are needed.

## Local Contracts

### File index (20 source files)

| File | Role |
|------|------|
| `_dspy.py` | Lazy DSPy import — sole import path for `dspy` |
| `mojo_bridge.py` | Shared Mojo module loader — `try_load_mojo()` eliminates boilerplate in 3 bridge files |
| `_embedder.py` | Shared embedder singleton for SemanticCache + MemoryManager |
| `_io.py` | Atomic JSON read/write utilities |
| `setup.py` | LMRegistry singleton + `setup_dspy()` configuration |
| `hotswap.py` | LRU-cached program hot-swap with semantic cache integration |
| `registry.py` | JSON registry with lineage, idempotency, and module source hashing |
| `scheduler.py` | Async compile scheduler with ThreadPoolExecutor |
| `output.py` | Run directory management and program persistence |
| `mlflow_tracker.py` | MLflow experiment tracking (sync + async) |
| `drift_monitor.py` | Quality drift detection and auto-recompile cascade |
| `holdout.py` | Programmatic holdout enforcement (Invariant 5) |
| `loaders.py` | Shared dataset/module loading + SSOT utilities (`prediction_to_dict`, `get_example_inputs`) |
| `metrics.py` | Shared DSPy metric factories (SSOT) |
| `cost_tracker.py` | Token counting and cost estimation |
| `sprt_mojo_bridge.py` | Mojo-accelerated SPRT bridge |
| `retry.py` | Auto-retry with exponential backoff |
| `dspy_modules.py` | Quality scoring and diagnostic DSPy module definitions |
| `errors.py` | Typed exception hierarchy |
| `logging_config.py` | Centralized structured logging via structlog — `get_logger(__name__)` is the standard logger accessor for all dspytools modules |

- `_LazyDSPy` defers the `import dspy` (which triggers the LiteLLM import chain costing 300ms+) until the first attribute access.
- **This is the SOLE import path for dspy across all dspytools modules.** All other modules must use:
  ```python
  from dspytools.core._dspy import dspy
  ```
  instead of `import dspy` directly.
- `_LazyDSPy.__getattr__`, `__call__`, and `__dir__` all trigger the lazy import on first use.
- The module-level `dspy: Any = _LazyDSPy()` singleton is the exposed name.

### `mojo_bridge.py` — Shared Mojo module loader

- `try_load_mojo(module_name, attr_name, logger) → tuple[bool, ModuleType | None]` — inserts `mojo_modules/` into `sys.path`, attempts `importlib.import_module(module_name)`, and checks for `attr_name` on the loaded module.
- Returns `(True, module)` on success, `(False, None)` on `ImportError`/`ModuleNotFoundError`/`OSError`, `(False, module)` if module loaded but lacked the expected attr.
- Consumed by `src/dspytools/skills/bm25_mojo_bridge.py`, `src/dspytools/graph/cache_mojo_bridge.py`, and `src/dspytools/core/sprt_mojo_bridge.py` — replaces 17-line boilerplate blocks in each file.

### `_embedder.py` — Shared embedder singleton

- `get_embedder()` — process-level singleton for `dspy.Embedder`. Prevents duplicate HTTP connection pools to the embedding server across `SemanticCache` and `MemoryManager`.
- Called via `from dspytools.core._embedder import get_embedder`.
- `clear_embedder()` — resets singleton for testing.
- Lazy init: avoids importing `dspy.Embedder` until first use.

### `setup.py` — LMRegistry singleton + DSPy configuration

- `LMRegistry` is a **class-level singleton** via `_instances: dict[str, dspy.LM]` and `_default: dspy.LM | None`.
- LM instances are keyed by `"{model}|{api_base}"` — sharing instances ensures DSPy's built-in LM cache (`cache=True`) hits across modules.
- `LMRegistry.get_or_default()` — prefers the student model from `config["lm"]["student"]`. Falls back to `config["lm"]["default"]`, then to `openai/gpt-4o` with `cache=False`.
- `LMRegistry.get_teacher()` — returns the teacher LM from `config["lm"]["teacher"]`. **MUST NEVER be used for inference.** Only for GEPA/distill/finetune optimization paths.
- `LMRegistry.clear()` — resets all cached instances (used in tests).
- `setup_dspy()` — calls `dspy.configure(lm=...)` with an LM from the registry. Auto-resolves `api_base` from the configured student model or registry entries when not explicitly provided. Reads `.env` for API keys. Call once at startup.
  - **Adapter**: reads `cfg["dspy"]["adapter"]` from config (set via `dspytools configure adapter set`). Defaults to `BAMLAdapter(use_native_function_calling=False)`. Supports `"baml"`, `"chat"`, `"json"`, `"xml"` types.
  - **Reasoning patch**: always patched into `baml_adapter._render_type_str` — renders `dspy.Reasoning` as plain `"string"` instead of expanding the Pydantic schema (prevents Qwen 7B outputting `"reasoning"` key instead of `"next_thought"`).

### `hotswap.py` — LRU-cached program hot-swap

- `HotSwapManager` holds up to `MAX_LOADED = 16` compiled programs in an `OrderedDict` for LRU eviction.
- `_load_program_from_run(run_id)` — deterministic loader using `metadata.json`'s `module_type` field (no Predict→ChainOfThought fallback chain). Reads `signature.json` and `program.json` from the run directory.
- `swap(run_id)` — switches active program, O(1). Returns previous active ID.
- `infer(**inputs)` — runs inference on the active program. Returns a dict (not a DSPy `Prediction`). **SSOT: checks semantic cache (`get_semantic_cache()`) before hitting the LM. Stores result after LM call.** Scores once via `auto_metric`, reuses score for drift monitoring and self-evolve quality tracking. Increments refcount on entry (try), decrements in `finally` block — guarantees cleanup on cache hit, exception, or normal return. On critical drift alert, calls `SemanticCache.adjust_threshold()` to dynamically tighten the cache threshold.
- `swap(run_id, wait_for_drain=False)` — if `wait_for_drain=True` and the active program has outstanding `_refcount > 0`, blocks up to 30s polling until refcount reaches 0. Raises `TimeoutError` if drain times out. After swap, calls `_prefetch_dependents()` to pre-load downstream programs into LRU cache (predictive prefetching via SkillGraph lineage).
- `warm_swap(run_id)` — loads, reads signature.json to construct field-appropriate test input, runs a verification inference, then swaps. Returns previous active ID (str) or raises `RuntimeError` on test failure / `KeyError` if not found. Uses `_build_test_input()` to inspect signature fields instead of hardcoding `{"input": "test"}`.
- `load_all()` — loads all programs from `compiled/index.json`, auto-activates first.
- `_refcounts: dict[str, int]` — tracks concurrent `infer()` usage per program. Guarded by `_reflock` (`threading.Lock`) for thread-safe concurrent access. Used by `swap(wait_for_drain)` to avoid swapping out an active program mid-request.
- `list()` — returns loaded programs with `active` flag.
- `unload(run_id)` — removes from cache; auto-selects next program if the active one was unloaded.

### `registry.py` — JSON registry with lineage tracking

- JSON index at `compiled/index.json` — single source of truth for compiled runs.
- `save_run_index(runs)` — writes the full JSON index file.
- `register_run(run_id, metadata)` — appends via `save_run_index`.
- `register_run_with_lineage(run_id, metadata, optimizer, dataset_hash, base_program_id, parent_run_id)` — registers a run with a `lineage` dict embedded in metadata, tracking optimizer, dataset hash, base program, and parent run for ancestry chains.
- `compute_dataset_hash(trainset)` — returns a deterministic 12-char SHA256 hex prefix for a training dataset (sorted by `str(ex)`, JSON-serialized via `toDict` if available).
- `get_lineage(run_id)` — returns the full ancestor chain `[run_id, parent, grandparent, ...]` by following `lineage.parent_run` links in the index.
- `list_compiled_runs()` — reads all runs from the JSON index.
- `get_run(run_id)` — looks up a run by ID from the JSON index.
- `delete_run(run_id)` — removes from JSON index and disk (shutil.rmtree).
- `list_signatures()`, `delete_signature(name)` — file-based (`.py` files in signatures dir).
- `list_modules()`, `delete_module(name)` — file-based (`.py` files in modules dir).
- `list_agents()`, `delete_agent(name)` — file-based (`.json` files in agents dir).

### `scheduler.py` — Async compile scheduler

- `CompileScheduler` uses a `ThreadPoolExecutor(max_workers=2)` for background compilation.
- `submit(optimizer, module_name, compile_fn, label)` — returns `job_id` immediately. `compile_fn` is a `Callable[[], str]` that returns the `run_id`.
- `CompileJob` status lifecycle: `queued → running → completed | failed | cancelled`.
- `get_status(job_id)`, `list_jobs()`, `cancel(job_id)` for job management.
- Job IDs are prefixed `job_` with 8 hex chars (`uuid.uuid4().hex[:8]`).

### `output.py` — Run directory management

- `create_run_dir(optimizer, label, metadata)` — creates `compiled/{timestamp}_{optimizer}[_{label}]/` with `metadata.json`. Returns `(run_id, run_path)`.
- `save_program(run_path, program, signature_data, metrics, module_type)` — saves `program.json`, `signature.json`, `metrics.json`, and updates `metadata.json` with `module_type` (for the deterministic loader in hotswap).
- `clean_old_runs(max_age_days=30)` — removes run directories older than threshold by reading `metadata.json` timestamps.

### `mlflow_tracker.py` — MLflow experiment tracking

- `MLflowTracker` wraps MLflow (v3.5.1+) for dspytools experiment tracking, tracing, and feedback logging.
- **Lazy initialization**: `_ensure_initialized()` called on first use. Falls back to local file store (`~/.config/dspytools/mlruns/`) if MLflow server is unavailable.
- **`trace(name)`** — context manager for MLflow autologging a compile operation. Wraps code in `start_run`/`end_run`.
- **`log_compile(optimizer, module, score, params, metrics)`** — logs optimizer name, module name, quality score, and extra params/metrics to a dedicated MLflow run. Returns `run_id`. Raises on MLflow failure (fail-fast).
- **`log_gfl_comparison(pipeline_result)`** — logs GFL 4-way comparison results including best optimizer, baseline, improvement, and per-optimizer scores. Raises on MLflow failure (fail-fast).
- **`log_feedback(trace_id, name, value, rationale)`** — logs evaluation feedback for a trace via `mlflow.log_feedback()`. Raises on MLflow failure (fail-fast).
- **Singleton access**: `get_tracker(async_mode=True)` returns a global `MLflowTracker` or `MLflowAsyncTracker` instance. Defaults to async mode.
- **Fail-fast**: All public methods propagate MLflow exceptions rather than silently swallowing them. Only retains `try/except` for: the recovery pattern in `_ensure_initialized` (get → create experiment), and control-flow exceptions (`queue.Empty`, `queue.Full`) in the async worker.

### `mlflow_tracker.py` — `MLflowAsyncTracker`

- `MLflowAsyncTracker` extends `MLflowTracker` with a background worker queue.
- Uses `queue.Queue(maxsize=500)` + single daemon `threading.Thread` to drain log operations.
- `_enqueue()` puts log tuples `(method, args, kwargs)` on the queue. Falls back to synchronous execution if the queue is full.
- `log_compile()`, `log_gfl_comparison()`, `log_feedback()` — async overrides that enqueue instead of blocking on HTTP.
- `shutdown(wait=True, timeout=5.0)` — sends `None` sentinel, joins worker, shuts down `ThreadPoolExecutor`.
- `flush(timeout=3.0)` — drains the queue synchronously and waits for in-flight tasks to complete. Returns `{drained, remaining, timed_out, dropped_total}`. Should be called before process exit to prevent telemetry loss.
- `stats` property exposes `queue_size`, `dropped`, `running`, `worker_alive`.
- `get_tracker(async_mode=True)` creates an `MLflowAsyncTracker` by default. Switching modes calls `shutdown()` on the old tracker.

### `drift_monitor.py` — Drift detection for compiled programs

- `DriftMonitor` — monitors compiled program quality over time. Re-evaluates on a fixed holdout set at configurable intervals. Alerts when quality drops below acceptable thresholds.
- `DriftSnapshot` — single quality check result with score, timestamp, delta from baseline.
- `DriftAlert` — dataclass with severity (`warning`/`critical`), degradation percentage, consecutive drops, and actionable message.
- `DriftMonitor.check(run_id, current_score, holdout_size)` — returns `DriftAlert | None`. Warning at 5% degradation, critical at 15% (configurable via constructor).
- `DriftMonitor.status()` — returns full status dict with programs_tracked, thresholds, and per-program baseline/current/delta.
- `DriftMonitor.request_recompile(run_id)` — queues a program for automatic recompilation when critical drift is detected. Called by `HotSwapManager.infer()` on critical alerts.
- `DriftMonitor.pending_recompiles()` — returns list of programs queued for recompilation.
- `DriftMonitor.process_recompile_requests(auto_fix=False)` — processes pending recompile queue. Dry-run mode by default; `auto_fix=True` triggers recompilation via GFLPipeline. **Delta-optimization**: uses `compile_draft()` for minor drift (<10%) and `run_single()` for major drift, selecting strategy based on drift severity.
- `get_drift_monitor()` — module-level singleton accessor.
- **Drift → recompile loop**: `hotswap.py` calls `request_recompile()` on critical alerts. `dspytools self auto-fix` inspects and processes the queue.
- State persisted to `~/.config/dspytools/drift_state.json`.

### `retry.py` — Auto-retry with exponential backoff

- `retry(max_retries=3, base_delay=2.0, max_delay=60.0, backoff_factor=2.0, retryable_exceptions=...)` — Decorator that wraps a function with exponential backoff retry logic. Retries on `TimeoutError`, `ConnectionError`, `OSError` by default. Non-retryable exceptions are re-raised immediately.
- `compile_with_retry(compile_fn, *args, max_retries=3, base_delay=2.0, **kwargs)` — Function wrapper that runs a compile call with retry and returns `(result, stats_dict)`. The stats dict includes `attempts`, `retries`, `errors`, and `total_delay`. Also retries on `RuntimeError` (covers OOM conditions).
- Used by `commands/compile.py` (3 retries, 2s base delay) and `gfl/pipeline.py` (2 retries, 2s base delay via `@retry` decorator on `_dispatch_optimizer`).

### `holdout.py` — Programmatic holdout enforcement (Invariant 5)

- `HoldoutGate` singleton enforces holdout isolation: splits trainset into train + holdout before any compile call, stores holdout in-memory, and NEVER passes it to the optimizer.
- `split(trainset, compile_id)` — deterministic shuffle (seed=42), splits by `holdout_fraction` (default 0.2). Returns `(train, holdout)` tuple. Holdout keyed by `compile_id` in `_splits` dict.
- `validate_gate(compile_id, compiled_program)` — SPRT-based evaluation on the stored holdout. Uses p0=0.50, p1=0.65, alpha=0.05, beta=0.20 for early termination. Returns `{accepted, score, n_evaluated, reason, holdout_size}`.
- `get_holdout_gate()` — module-level singleton accessor. Used by `commands/compile.py` for post-compile validation and by `mcp/tools.py` for the `holdout_status` MCP tool.
- `gated_compile` decorator — wraps any compile function to auto-split and validate. Not yet wired into the compile pipeline (reserved for future `@gated_compile` usage).
- `stats` property — returns number of stored splits and their IDs.

### `loaders.py` — Shared dataset and module loading (SSOT)

- **Single Source of Truth** for loading trainsets and DSPy modules across all commands.
- `load_trainset(path: str) → list` — loads a DSPy trainset from a JSON or JSONL file (auto-detected by first character: `[` → JSON array, `{` → JSONL). Auto-detects input keys: `"input"`, `"inputs"`, or falls back to the first key in each item.
- `load_jsonl(path: str) → list` — loads examples from a JSONL file (one JSON per line). Parses `input` field (JSON string) into individual fields. Used by distill auto-eval.
- `load_module_by_name(name: str) → dspy.Module` — dynamically loads and instantiates a DSPy module from `modules_dir/{name}.py`. Raises `FileNotFoundError` if the file is missing, `ImportError` if the module lacks a class matching the name.
- Used by `commands/compile.py`, `commands/evaluate.py`, `commands/compare.py`.

### `metrics.py` — Shared DSPy metric functions (SSOT)

- **Single Source of Truth** for all DSPy-compatible scoring functions across compile commands and GFL pipeline.
- `exact_match_metric(val_field="output")` — factory returning a `(example, prediction, trace) -> float` metric.
  - Compares `getattr(prediction, val_field, "")` to `getattr(example, val_field, "")`.
  - Returns `1.0` on exact match, `0.0` otherwise.
  - `val_field` parameter allows flexible field selection (e.g. `"answer"`, `"output"`).
  - Used by `commands/compile.py`, `gfl/pipeline.py`, `gfl/optimizer.py`.
- All metric factories accept configurable field names — no hardcoded `"output"` in call sites.

### `cost_tracker.py` — Token counting and cost estimation

- `PRICING` dict maps model names to `{"input": price, "output": price}` per 1M tokens. Includes DeepSeek V4 Flash ($0.14/$0.28), DeepSeek V3 ($0.27/$1.10), Qwen 3B local ($0/$0), and a `"default"` fallback ($0.15/$0.60).
- `TokenUsage` dataclass: model, prompt/completion/total tokens, cost_estimate. `TokenUsage.estimate(model, prompt_tokens, completion_tokens)` computes cost using `PRICING`.
- `CompileCost` dataclass: aggregates token usage across LM calls in a compile operation. Properties: `teacher_tokens` (deepseek models), `student_tokens` (non-deepseek), `elapsed_seconds`.
- `CompileCost.add_call(model, prompt_tokens, completion_tokens)` — records a single LM call and updates totals.
- `CompileCost.finish()` — records `finished_at` timestamp.
- `CompileCost.summary()` — returns a dict with compile_id, optimizer, elapsed_seconds, total/teacher/student tokens, total_cost_usd, and call count.

### `sprt_mojo_bridge.py` — Mojo-accelerated SPRT bridge (Phase 2)

- Mojo loading delegated to `mojo_bridge.try_load_mojo()`.
- `sprt_evaluate(outcomes, p0, p1, alpha, beta) → dict` — runs SPRT on binary outcome array.
- `HAS_MOJO` flag + pure Python fallback with identical semantics.
- Consumed by `SelfEvolveEngine.validate_and_deploy()` in `evolve/self_evolve.py`.

### `errors.py` — Typed exception hierarchy

- `DspyToolsError` — base for all dspytools exceptions.
- `ServiceUnavailableError` with subclasses `CacheError` (Redis), `GraphError` (FalkorDB), `LlamaCppError` (llama-cpp-server). Each carries `service_name` and optional `retry_after`.
- `CompileError` — carries `optimizer`, `module_name`, original `cause`.
- `ValidationError` — carries `field`, `value`, `constraint`.
- `ConfigError` — carries missing/invalid `key`.
- `RateLimitError` — carries `endpoint`, `retry_after`.
- All subclasses of `DspyToolsError` for clean `except DspyToolsError` catch-all.

### `logging_config.py` — Centralized structured logging via structlog

- `configure_logging(*, force=False)` — one-time global structlog configuration. Idempotent; `force=True` reconfigures. Sets up stdlib integration so existing `logging.getLogger()` calls are rendered through structlog. Console output in dev, JSON in production (`DSPYTOOLS_ENV=production`). Respects `DSPYTOOLS_LOG_LEVEL` env var (default INFO). Quiets noisy third-party loggers (urllib3, httpx, litellm, falkordb, redis).
- `get_logger(name)` — returns a `structlog.stdlib.BoundLogger` bound with `module=name`. **This is the standard logger accessor for all dspytools modules.** Use `log = get_logger(__name__)` at module level. Calls `configure_logging()` on first use.
- **Backward compatible**: stdlib `logging.getLogger()` calls still work and are rendered through the structlog pipeline. New code should prefer `get_logger()` for structured key-value logging (`log.info("event", key=val)`).

### `registry.py` — Idempotency tokens (new)

- `compute_idempotency_key(module_name, dataset_hash, optimizer, module_source_hash=None)` — deterministic SHA256 key (16-char hex). Optional `module_source_hash` includes AST hash of the module's Python source for structural identity — skips recompilation when code hasn't changed.
- `compute_module_source_hash(module) → str` — hashes a DSPy module's source code via `inspect.getsource()` for structural identity. Returns empty string if source unavailable.
- `find_existing_compile()` — enhanced to skip failed runs for safe retry without duplicate compiles.

## Work Guidance

- Never `import dspy` directly in any dspytools module. Always use `from dspytools.core._dspy import dspy`.
- LM instances must go through `LMRegistry` — never create `dspy.LM(...)` directly in other modules.
- `LMRegistry.get_teacher()` is ONLY for GEPA/distill/finetune paths. Do not use for inference or dspy.configure().
- `setup_dspy()` should be called once at CLI startup, not per-command.
- When adding new LM configuration fields, update `setup.py`'s `LMRegistry` methods.
- HotSwapManager max_loaded cap (16) is a soft limit via `while` eviction — be mindful of memory when adding programs.
- Registry is JSON-only — `compiled/index.json` is the single source of truth for compiled runs.
- All trainset and module loading must go through `loaders.py` functions — do not duplicate loading logic in command files.

## Verification

- `ruff check --fix --unsafe-fixes` must pass with zero errors.
- Expected false positives from LSP stubs (`click`, `dspy`, `mcp`, `fastapi`, `dspy.LM`) are acceptable and ignored.
- No additional CI or test framework is currently enforced beyond ruff. Update this section when one is added.

## Child DOX Index

Core owns no subdirectories and has no child AGENTS.md files. All contracts are documented here.
