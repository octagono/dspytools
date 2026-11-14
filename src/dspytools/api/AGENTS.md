# DOX — REST API server

## Purpose

FastAPI-based REST API that exposes compiled DSPy programs for inference over HTTP. Enables external services, scripts, and integrations to run hot-swapped programs without a CLI dependency.

## Ownership

Owns the `server.py` FastAPI application — the single source file in this directory. All API endpoints, request/response models, and the `run_api()` entry point are defined here.

## Local Contracts

### `server.py` — FastAPI hot-swap inference server

- **LM lifecycle (startup contract)**: `lifespan` calls `_ensure_lm_configured()` → `setup_dspy()` before any request is served; `get_hotswap()` repeats the same idempotent guard so no entry point can serve with `dspy.settings.lm is None`. If DSPy still has no LM, `LMConfigurationError` is raised (startup fails loudly) and `/infer` answers **503** via the registered exception handler — never an opaque 500. `dspytools server start` also calls `setup_dspy()` in the forked API child before uvicorn binds.
- **Serving LM + per-request isolation (config-change contract)**: the process holds `_serving_lm`. Blocking work runs inside `_with_serving_lm()`, which pins that LM with `dspy.context()` — `dspy.configure()` is single-owner (one thread **and** one async task), so a running server cannot re-configure from a request task. `_current_lm()` falls back to `dspy.settings.lm` when `_serving_lm` is unset (health probe, tests).
- **Model config change contract**: `/config/models/*` persists through `save_user_config()`, then calls `_reload_lm_configuration()` → `LMRegistry.clear()` + `LMRegistry.get_or_default()` (fresh LM from the newly persisted config) and best-effort `dspy.configure(lm=...)`. A configure refusal (single-owner rule) is logged as `api_lm_configure_deferred`; serving still switches because requests pin `_serving_lm` through `dspy.context()`. Responses include `active_lm`. No restart needed.
- **Async contract**: `/infer` and `POST /swap/{id}?warm=true` run the synchronous, LM-bound manager calls on a worker thread via `dspy.asyncify` (anyio thread pool bounded by the `async_max_workers` setting, default 8) — the event loop never blocks. The `/infer` response schema is unchanged: `{"status": "ok", "result": {...}}`.
- **Hot-Swap lifecycle**: `_hotswap` is a module-level global initialized lazily by `get_hotswap()`. It creates a `HotSwapManager` without `load_all()` (lazy index, on-demand program loading).
- **Endpoints**:

  | Method | Path | Description |
  |--------|------|-------------|
  | GET | `/programs` | List all loaded programs with metadata |
  | GET | `/programs/{id}` | Get metadata for a specific program |
  | POST | `/swap/{id}` | Activate a loaded program by ID. Query param `?warm=true` triggers `warm_swap()` (load + test inference + swap) |
  | POST | `/infer` | Run inference on the active program |
  | GET | `/active` | Get the currently active program ID |
  | GET | `/health` | Probe: `status` (`ok`/`degraded`), `active_program`, `lm_configured`, `lm` (model name) |
  | GET | `/config/models` | Get current student/teacher model config + `active_lm` |
  | POST | `/config/models/student` | Set student (inference) model, re-applied in-process |
  | POST | `/config/models/teacher` | Set teacher (optimization) model, re-applied in-process |
  | PUT | `/config/models` | **Optimization 10**: Atomic batch update of student + teacher models |

- **Request/Response models** defined inline with Pydantic:
  - `InferRequest` — `{"inputs": {...}}` dict passed as kwargs to `HotSwapManager.infer()`
  - `SwapResponse` — status, active, previous IDs
  - `ProgramInfo` — id, active, optimizer, created, score
  - `ModelConfig` — model, api_base, api_key, temperature, max_tokens
  - `ModelsConfigRequest` — atomic student+teacher update payload

- **Inference contract**: `POST /infer` calls `HotSwapManager.infer(**inputs)` (off-loop, LM pinned by `dspy.context`) which runs on the currently swapped program. Returns `{"status": "ok", "result": {...}}` on success, `500` on failure, `503` when no LM can be configured.

- **Hot-swap contract**: `POST /swap/{id}` calls `HotSwapManager.swap(program_id)`. Returns `404` if the ID is not in the loaded cache, not if the program JSON is absent from disk (cache miss is a separate concern).

- **Model config persistence**: Config writes go through `config.settings.save_user_config()`, which writes to `~/.config/dspytools/config.toml` and invalidates the `ConfigCache`. Reads go through `config.settings.load_config()` for hot-reload support. A persisted change is re-applied in-process (see the config-change contract above).

- **Entry point**: `run_api(host="0.0.0.0", port=8080)` starts uvicorn. Called from `commands/server.py` via `dspytools server start`.

### Dependencies

| Module | Usage |
|--------|-------|
| `dspytools.core.hotswap.HotSwapManager` | Program lifecycle (load, swap, infer, list, metadata) |
| `dspytools.core.setup.setup_dspy` / `LMRegistry` | Startup LM configuration; registry clear + rebuild after a model config change |
| `dspytools.core._dspy.dspy` | `dspy.context` (per-request LM pinning) and `dspy.asyncify` (thread offload) |
| `dspytools.config.settings.load_config` | Read model configuration with hot-reload |
| `dspytools.config.settings.save_user_config` | Persist model configuration changes |

## Work Guidance

- `get_hotswap()` is the single accessor for `HotSwapManager`. Do not create additional instances — the module-level `_hotswap` ensures one manager per server process.
- Never call `dspy.configure()` from an endpoint. Pin the serving LM with `_with_serving_lm()` / `dspy.context()` instead; configure only at startup (`lifespan`) or through `_ensure_lm_configured(force=True)`.
- When adding new endpoints, add new Pydantic models at the top of `server.py` following the existing patterns.
- New model config fields added to `ModelConfig` must also be supported in `config/settings.py`'s TOML schema and in the corresponding CLI commands.
- Error responses use FastAPI `HTTPException` with appropriate status codes — do not return raw Python exceptions.
- The `/health` endpoint is meant for load balancers and container orchestrators. Keep it lightweight (no DB queries, no LM calls, no program loading) — it reports `lm_configured` from existing state only.
- Any new LM-bound endpoint must run its DSPy work through `dspy.asyncify` + `_with_serving_lm()`.

## Verification

- Start the server: `dspytools server start --port 8080`
- Health check: `curl http://localhost:8080/health` (`lm_configured` must be `true`)
- List programs: `curl http://localhost:8080/programs`
- Swap: `curl -X POST http://localhost:8080/swap/{run_id}`
- Infer: `curl -X POST http://localhost:8080/infer -H 'Content-Type: application/json' -d '{"inputs": {...}}'`
- Model config: `curl http://localhost:8080/config/models`
- Offline tests: `uv run --no-sync pytest tests/test_serving_integrity.py -q` (TestClient + `DummyLM`, no network)

## Child DOX Index

No child directories. The `api/` directory contains only `server.py` and this `AGENTS.md`.
