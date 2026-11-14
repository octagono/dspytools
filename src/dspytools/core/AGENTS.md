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
| `programs.py` | Compiled-program provenance — reads `metadata.dspy_version` (or `metadata.dependency_versions.dspy`) so callers can refuse a program compiled by another DSPy |
| `setup.py` | LMRegistry singleton + `setup_dspy()` configuration |
| `hotswap.py` | LRU-cached program hot-swap with semantic cache integration |
| `registry.py` | JSON registry with lineage, idempotency, and module source hashing |
| `scheduler.py` | Async compile scheduler with ThreadPoolExecutor |
| `output.py` | Run directory management and program persistence |
| `mlflow_tracker.py` | MLflow experiment tracking (sync + async) |
| `drift_monitor.py` | Quality drift detection and auto-recompile cascade |
| `holdout.py` | Programmatic holdout enforcement (Invariant 5) |
| `loaders.py` | Shared dataset/module loading + SSOT utilities (`prediction_to_dict`, `prediction_output`, `get_example_inputs`, `join_inputs`) |
| `metrics.py` | Shared DSPy metric factories (SSOT) |
| `cost_tracker.py` | Real token accounting (`Prediction.get_lm_usage()` / `lm.history`) plus `PRICING`-based cost estimates |
| `sprt_mojo_bridge.py` | Mojo-accelerated SPRT bridge |
| `retry.py` | DSPy-aligned auto-retry with exponential backoff and `retry_after` hints |
| `dspy_modules.py` | Quality scoring and diagnostic DSPy module definitions |
| `errors.py` | Typed exception hierarchy |
| `logging_config.py` | Centralized structured logging via structlog — `get_logger(__name__)` is the standard logger accessor for all dspytools modules |

- `_LazyDSPy` defers `import dspy` (adapter + engine stack, hundreds of ms on a cold process) until the first attribute access.
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
- `sampling_lm(lm, temperature)` is the only supported way to vary sampling: `temperature` is absent from DSPy's `DEFAULT_CONFIG` and `Predict` reads it from `config`/`lm.kwargs`, so `dspy.context(temperature=...)` is a no-op. The helper returns `lm.copy(temperature=...)` (shared HTTP pool, isolated kwargs/history) and passes non-LM stand-ins through unchanged.
- LM instances are keyed by `_lm_cache_key(model, lm_kw)` — the model plus **every output-affecting kwarg** (`temperature`, `max_tokens`, `reasoning_effort`, `model_type`, …); `api_key` is deliberately excluded. Sharing instances ensures DSPy's built-in LM cache (`cache=True`) hits across modules; differing thinking/temperature configs must never collide on one cached LM.
- Every LM is built with `model_type="chat"` by default (config `dspy.model_type`, per-section `model_type` override) — llama.cpp serves only `/v1/chat/completions` while 3.5's `"auto"` routes to `/v1/responses`. Embedding models are exempt: `LMRegistry` is for chat/completion LMs, embeddings go through `dspy.Embedder`.
- Thinking is controlled by `reasoning_effort`, never `chat_template_kwargs` (3.5 rejects the latter on the canonical Request path). `_reasoning_kwargs()` maps `enable_thinking: false` → `"none"`, `true`/absent → omit the dial; an explicit `reasoning_effort` in config wins.
- `LMRegistry.get_or_default()` — prefers the student model from `config["lm"]["student"]`. Falls back to `config["lm"]["default"]`, then to `openai/gpt-4o` with `cache=False`.
- `LMRegistry.get_teacher()` — returns the teacher LM from `config["lm"]["teacher"]`. **MUST NEVER be used for inference.** Only for GEPA/distill/finetune optimization paths.
- `LMRegistry.clear()` — resets all cached instances (used in tests).
- `setup_dspy()` — calls `dspy.configure(lm=...)` with an LM from the registry. Auto-resolves `api_base` from the configured student model or registry entries when not explicitly provided. Reads `.env` for API keys. Call once at startup.
  - **Usage tracking defaults on**: `setup_dspy()` forwards `track_usage=True` as a real `dspy.configure()` kwarg whenever config does not set it (public API, no writes into DSPy's settings dict), so `Prediction.get_lm_usage()` is populated for `core.cost_tracker`. A persisted `dspy.track_usage: false` still wins via `_runtime_settings`; the configure-owner-thread fallback applies the same settings.
  - **Adapter**: reads `cfg["dspy"]["adapter"]` from config (set via `dspytools configure adapter set`). Defaults to `BAMLAdapter(use_native_function_calling=False)`. Supports `"baml"`, `"chat"`, `"json"`, `"xml"` types. Only the selected adapter is constructed (not all four); unknown/stale config values fall back to `BAMLAdapter`.
  - **Reasoning patch**: always patched into `baml_adapter._render_type_str` — renders `dspy.Reasoning` as plain `"string"` instead of expanding the Pydantic schema (prevents Qwen 7B outputting `"reasoning"` key instead of `"next_thought"`).
  - **Parse patch**: `JSONAdapter.parse`/`BAMLAdapter.parse` are replaced with a reasoning-tolerant parse. It first tries strict JSON (json_repair + `{...}` regex), then falls back to ChatAdapter-style `[[ ## field ## ]]` section parsing — small local models (Qwen) echo the BAML template back verbatim instead of emitting JSON. After the `parse_value` cast it applies DSPy 3.5's `apply_output_field_defaults` (so omitted defaulted/nullable output fields are filled, matching stock JSONAdapter behavior) and only then enforces presence of required (non-Reasoning) output fields; missing `dspy.Reasoning` fields are tolerated (filled from `reasoning_content` in `_call_postprocess`). **Version-compat**: `apply_output_field_defaults` is imported with feature-detection (try/except ImportError) so a future DSPy that moves/renames the helper degrades to skip-default-filling rather than hard-crashing setup — this is a symbol-level compat guard on a hard dependency, the sole exception to Golden Rule 4's no-guarded-imports.

### `hotswap.py` — LRU-cached program hot-swap

- `HotSwapManager` holds up to `MAX_LOADED = 16` compiled programs in an `OrderedDict` for LRU eviction.
- `_load_program_from_run(run_id, run_path=None)` — deterministic loader using `metadata.json`'s `module_type` field (no Predict→ChainOfThought fallback chain). Supports `module_type` values: `"predict"`/default → Predict, `"cot"` → ChainOfThought, `"react"` → ReAct, `"flex"` → `dspy.Flex` (reconstructed with the default `PythonInterpreter` sandbox). Reads `signature.json` and `program.json` from the run directory. `run_path` overrides the `compiled_dir()/run_id` lookup so a run directory elsewhere (`dspytools evaluate run <path>`) loads in its saved shape. **Provenance**: before loading, `core.programs.is_current_dspy_version()` is checked; a program saved by another DSPy logs `dspy_version_mismatch` (saved vs. current version + recompile hint) and is still loaded — 3.5 changed prompts/adapters, so the warning is the signal to recompile.
- `swap(run_id)` — switches active program, O(1). Returns previous active ID.
- `infer(**inputs)` — runs inference on the active program. Returns a dict (not a DSPy `Prediction`). **SSOT: checks semantic cache (`get_semantic_cache()`) before hitting the LM. Stores result after LM call.** Scores once via `auto_metric`, reuses score for drift monitoring and self-evolve quality tracking. Increments refcount on entry (try), decrements in `finally` block — guarantees cleanup on cache hit, exception, or normal return. On critical drift alert, calls `SemanticCache.adjust_threshold()` to dynamically tighten the cache threshold. **Cache serialization rule**: the stored payload is built by `_json_safe_output()` (`dspy.Reasoning` and other non-JSON values become `str`, ndarrays become lists); a payload that still cannot be JSON-encoded is logged (`semantic_cache_serialize_failed`) and treated as a **cache miss** — serialization never fails a request. **Scoring rule**: the drift/quality score is computed from `loaders.prediction_output(result)` (the `answer`/`output` field), never from the first output field — for CoT programs the first field is `reasoning`, and grading it produced meaningless drift alerts and recompile queues. **Graceful degradation**: a Redis/FalkorDB outage during cache init/check/store/adjust degrades to "cache miss → hit the LM directly" (catches `ServiceUnavailableError`/`ConnectionError`, logs `semantic_cache_unavailable`) — inference never crashes because the semantic cache is down.
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
- **Lazy initialization**: `_ensure_initialized()` called on first use. Uses `MLFLOW_TRACKING_URI` when set, otherwise the local sqlite store (`~/.config/dspytools/mlruns/mlflow.db`) — never a dead remote default.
- **`trace(name)`** — context manager for MLflow autologging a compile operation. Wraps code in `start_run`/`end_run`. Yields `None` when the tracker is disabled.
- **`log_compile(optimizer, module, score, params, metrics)`** — logs optimizer name, module name, quality score, and extra params/metrics to a dedicated MLflow run. Returns `run_id`, or `None` when tracking is disabled.
- **`log_gfl_comparison(pipeline_result)`** — logs GFL 4-way comparison results including best optimizer, baseline, improvement, and per-optimizer scores. Returns `None` when tracking is disabled.
- **`log_feedback(trace_id, name, value, rationale)`** — logs evaluation feedback for a trace via `mlflow.log_feedback()`. Returns `None` when tracking is disabled.
- **Singleton access**: `get_tracker(async_mode=True)` returns a global `MLflowTracker` or `MLflowAsyncTracker` instance. Defaults to async mode.
- **Degrade, never block**: tracking is telemetry. The HTTP client is pinned (`MLFLOW_HTTP_REQUEST_TIMEOUT`, `MLFLOW_HTTP_REQUEST_MAX_RETRIES`, `MLFLOW_HTTP_REQUEST_BACKOFF_FACTOR`) before the first call; if the store is unreachable, `_ensure_initialized` logs one `mlflow_unavailable` warning, sets `_disabled`, and every public method becomes a no-op for the rest of the process. `DSPYTOOLS_MLFLOW=0` disables it outright. The only surviving `try/except` besides this is the get → create-experiment recovery and the async worker's `queue.Empty`/`queue.Full` control flow.

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
- `DriftMonitor.process_recompile_requests(auto_fix=False)` — processes pending recompile queue. Dry-run mode by default; `auto_fix=True` triggers recompilation via GFLPipeline. **Trainset requirement**: auto-fix only recompiles against a real trainset — registered via `DriftMonitor.register_trainset(run_id, trainset)` (in-memory) or discovered from the run's registry metadata (`trainset`/`trainset_path`/`dataset`/`dataset_path` file path). When neither exists the request is **refused** (`status: "skipped"`, actionable message, request stays queued) instead of compiling against a fabricated one-example placeholder and hot-swapping the result into service. **Delta-optimization**: uses `compile_draft()` for minor drift (<10%) and `run_single()` for major drift, selecting strategy based on drift severity; the saved `signature.json` is derived from the trainset (`_signature_from_trainset`) rather than hardcoded `input`/`output`. **Closes the loop**: on successful recompile the freshly compiled run is registered, baseline-updated, and hot-swapped into active service via `HotSwapManager` (best-effort — a swap failure logs `drift_fix_swap_failed` and leaves the run persisted but not active, never crashes).
- `DriftMonitor.register_trainset(run_id, trainset)` / `get_trainset(run_id)` — supply the real data for drift auto-fix (empty trainsets raise `ValueError`). In-memory only: trainsets are never serialized into `drift_state.json`.
- `get_drift_monitor()` — module-level singleton accessor.
- **Drift → recompile loop**: `hotswap.py` calls `request_recompile()` on critical alerts. `dspytools self auto-fix` inspects and processes the queue; it needs a registered trainset for each queued run.
- State persisted to `~/.config/dspytools/drift_state.json`.

### `retry.py` — Auto-retry with exponential backoff

- **DSPy-aligned taxonomy**: the retryable set is `retryable_exceptions()` — stdlib transients (`TimeoutError`, `ConnectionError`, `OSError`) plus DSPy's own `_RETRYABLE_LM_ERRORS` (`LMRateLimitError`, `LMTimeoutError`, `LMServerError`, `LMTransportError`, `LMLockTimeoutError`), read from the installed dspy on first use, never hardcoded. `non_retryable_exceptions()` is the never-retry set (`ContextWindowExceededError`, `LMInvalidRequestError`, `LMAuthError`, `LMBillingError`, `LMConfigurationError`, `LMNotConfiguredError`, `LMUnsupportedFeatureError`, `LMUnsupportedModelError`, `LMCollectionLimitError`).
- **Never-retry wins**: `_is_retryable()` checks the never-retry set first, so even a caller who widens `retryable_exceptions=(Exception,)` cannot retry a context-window/auth/billing failure — those can never succeed and only burn money. `LMUnexpectedError` is *not* retried: upstream deliberately excludes it (`is_retryable_lm_error` returns False), since unclassified engine failures carry no evidence they are transient.
- `retry_after_seconds(exc)` — `DSPyError.__init__` always sets `retry_after` (float seconds or `None`); invalid values (negative, NaN/inf, bools, non-numeric) are ignored. When a hint exists the delay is `min(retry_after, max_delay)`, otherwise `min(base_delay * backoff_factor**attempt, max_delay)`.
- `retry(max_retries=3, base_delay=2.0, max_delay=60.0, backoff_factor=2.0, retryable_exceptions=None)` — decorator. `None` resolves the default tuple lazily; retries are logged as `retry_attempt`. Non-retryable exceptions are re-raised immediately.
- `compile_with_retry(compile_fn, *args, max_retries=3, base_delay=2.0, **kwargs)` — runs a compile call and returns `(result, stats_dict)` with `attempts`, `retries`, `errors`, `total_delay`. Also retries `RuntimeError` (OOM) and honours `retry_after`.
- Importing this module never imports dspy (the exception types load on first retry call).
- Used by `commands/compile.py` (3 retries, 2s base delay) and `gfl/pipeline.py` (2 retries, 2s base delay via `@retry` decorator on `_dispatch_optimizer`).

### `holdout.py` — Programmatic holdout enforcement (Invariant 5)

- `HoldoutGate` singleton enforces holdout isolation: splits trainset into train + holdout before any compile call, stores holdout in-memory, and NEVER passes it to the optimizer.
- `split(trainset, compile_id)` — deterministic shuffle (seed=42), splits by `holdout_fraction` (default 0.2). Returns `(train, holdout)` tuple. Holdout keyed by `compile_id` in `_splits` dict.
- `validate_gate(compile_id, compiled_program, baseline_program=None)` — SPRT-based evaluation on the stored holdout. Uses p0=0.50, p1=0.65, alpha=0.05, beta=0.20 for early termination. Returns `{accepted, score, n_evaluated, reason, holdout_size}`.
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
- `normalize_text(text)` — comparison normalization: strip, casefold, collapse whitespace, strip surrounding punctuation/quotes (so `"Paris."` == `"Paris"`).
- `answer_exact_match(gold, pred)` — normalized equality of two answers (symmetric: two absent answers compare equal).
- `answer_overlap_score(gold, pred)` — `1.0` on normalized equality, else fraction of gold tokens covered by the prediction, capped at `0.99` so a non-exact answer is never "perfect" (GEPA's `skip_perfect_score` would drop it).
- `exact_match_metric(val_field="output")` — factory returning a `(example, prediction, trace) -> float` metric.
  - Compares `getattr(prediction, val_field, "")` to `getattr(example, val_field, "")` **after normalization** — byte-exactness scored `"Paris."` vs `"Paris"` as 0 and left bootstrap optimizers with zero demos.
  - Returns `1.0` on match, `0.0` otherwise.
  - `val_field` parameter allows flexible field selection (e.g. `"answer"`, `"output"`).
  - Used by `commands/compile.py`, `gfl/pipeline.py`, `skills/manager.py`.
- `gepa_metric(gold, pred, trace, pred_name, pred_trace, *, val_field="output")` — GEPA metric returning `dspy.Prediction(score=<float in [0,1]>, feedback=<short specific text>)`: missing field, empty output, gold-vs-prediction delta and the score. A bare float left the reflection teacher optimizing markdown shape instead of correctness. Used by `commands/compile.py` and `gfl/pipeline.py`.
- `gepa_metric_float(...)` — plain-float fallback for callers that consume the metric numerically (`dspy.Evaluate` sums metric returns, so `dspy.BetterTogether` must use this one).
- `gepa_metric_cost_aware(val_field="output", lm_call_penalty=0.01)` — factory returning a 6-arg GEPA metric `(gold, pred, trace, pred_name, pred_trace, program_trace)`. Scores via `auto_metric` then subtracts `lm_call_penalty * len(program_trace)` when `program_trace` is populated (Flex optimization scoring), clamped to `[0.0, 1.0]`. Enables cost-aware Flex optimization via GEPA's `program_trace` parameter. Used by `commands/compile.py` (`compile flex`).
- All metric factories accept configurable field names — no hardcoded `"output"` in call sites.

### `cost_tracker.py` — Token counting and cost estimation

- `PRICING` dict maps model names to `{"input": price, "output": price}` per 1M tokens. Includes DeepSeek V4 Flash ($0.14/$0.28), DeepSeek V3 ($0.27/$1.10), Qwen 3B local ($0/$0), and a `"default"` fallback ($0.15/$0.60).
- **Two sources, one sink.** Measured usage comes from DSPy; estimates come from `PRICING`. Both land in the same `CompileCost.calls` list, distinguished by `TokenUsage.measured`.
- `TokenUsage` dataclass: model, prompt/completion/total tokens, cost_estimate, `measured`. `TokenUsage.estimate(model, prompt_tokens, completion_tokens)` prices counts from `PRICING` (`measured=False`); `TokenUsage.from_usage(model, usage, cost=None)` builds a measured record from one DSPy usage entry, preferring a provider-reported `cost` over the estimate.
- **Measured shape (verified against dspy 3.5.0)**: `Prediction.get_lm_usage()` returns `{model: {prompt_tokens, completion_tokens, total_tokens, prompt_tokens_details, completion_tokens_details, ...}}` and is `None` unless `settings.track_usage` is on; a `DummyLM`-backed call reports zeros. `lm.history[-1]` carries `usage` (same inner dict) and `cost` (float USD or `None`) — **cached calls report `None` for both**, since no request was made.
- `CompileCost` dataclass: aggregates token usage across LM calls in a compile operation. Properties: `teacher_tokens` (deepseek models), `student_tokens` (non-deepseek), `elapsed_seconds`.
- `CompileCost.add_call(model, prompt_tokens, completion_tokens)` — estimate-based path, unchanged: records one call and prices it from `PRICING`.
- `CompileCost.add_usage(usage_by_model) → int` — records measured usage in `get_lm_usage()` shape; entries with zero tokens and zero cost (cache hits) are skipped and do not inflate the call count.
- `CompileCost.add_result(result, lm=None) → int` — records **real** usage from a DSPy result: `result.get_lm_usage()` first, falling back to `lm.history[-1]` when tracking was off. Returns how many records were added (0 for a cached call), so the estimate path remains the fallback for callers with no result object.
- `usage_from_result(result)` / `usage_from_lm_history(lm)` — the pure extractors behind `add_result`; both tolerate non-`Prediction` results and return empty/`None` instead of raising.
- `CompileCost.finish()` — records `finished_at` timestamp.
- `CompileCost.summary()` — returns compile_id, optimizer, elapsed_seconds, total/teacher/student tokens, total_cost_usd, call count, and the `measured_calls` / `estimated_calls` split (0 measured means pure estimates).

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
- Retry decisions come from `retry.retryable_exceptions()` / `non_retryable_exceptions()` — never from a hand-written exception list at a call site. Context-window, auth and billing failures must never be retried.
- Cost reporting must prefer measured usage (`CompileCost.add_result`) over `PRICING` estimates (`add_call`) whenever a result object exists; `setup_dspy()` keeps `track_usage` on so measured usage is available.

## Verification

- `ruff check --fix --unsafe-fixes` must pass with zero errors.
- `uv run --no-sync pytest tests/test_optimizer_integrity.py -q` covers the `metrics.py` contracts (normalization, `gepa_metric` feedback, float fallback).
- `uv run --no-sync pytest tests/test_dspy_optimizations.py -q` covers the retry taxonomy (`retry.py`), measured token accounting (`cost_tracker.py` + `setup_dspy`'s `track_usage` default), and the CLI `--parallel-tools` / `--stream` adapter paths.
- Expected false positives from LSP stubs (`click`, `dspy`, `mcp`, `fastapi`, `dspy.LM`) are acceptable and ignored.
- No other CI or test framework is enforced beyond ruff + those test modules. Update this section when more are added.

## Child DOX Index

Core owns no subdirectories and has no child AGENTS.md files. All contracts are documented here.
