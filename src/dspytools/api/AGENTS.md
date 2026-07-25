# DOX — REST API server

## Purpose

FastAPI-based REST API that exposes compiled DSPy programs for inference over HTTP. Enables external services, scripts, and integrations to run hot-swapped programs without a CLI dependency.

## Ownership

Owns the `server.py` FastAPI application — the single source file in this directory. All API endpoints, request/response models, and the `run_api()` entry point are defined here.

## Local Contracts

### `server.py` — FastAPI hot-swap inference server

- **Hot-Swap lifecycle**: `_hotswap` is a module-level global initialized lazily by `get_hotswap()`. On first call, it creates a `HotSwapManager` and calls `load_all()` to populate the LRU cache.
- **Endpoints**:

  | Method | Path | Description |
  |--------|------|-------------|
  | GET | `/programs` | List all loaded programs with metadata |
  | GET | `/programs/{id}` | Get metadata for a specific program |
  | POST | `/swap/{id}` | Activate a loaded program by ID. Query param `?warm=true` triggers `warm_swap()` (load + test inference + swap) |
  | POST | `/infer` | Run inference on the active program |
  | GET | `/active` | Get the currently active program ID |
  | GET | `/health` | Health check with active program info |
  | GET | `/config/models` | Get current student/teacher model config |
  | POST | `/config/models/student` | Set student (inference) model |
  | POST | `/config/models/teacher` | Set teacher (optimization) model |
  | PUT | `/config/models` | **Optimization 10**: Atomic batch update of student + teacher models |

- **Request/Response models** defined inline with Pydantic:
  - `InferRequest` — `{"inputs": {...}}` dict passed as kwargs to `HotSwapManager.infer()`
  - `SwapResponse` — status, active, previous IDs
  - `ProgramInfo` — id, active, optimizer, created, score
  - `ModelConfig` — model, api_base, api_key, temperature, max_tokens
  - `ModelsConfigRequest` — atomic student+teacher update payload

- **Inference contract**: `POST /infer` calls `HotSwapManager.infer(**inputs)` which runs on the currently swapped program. Returns `{"status": "ok", "result": {...}}` on success, `500` on failure.

- **Hot-swap contract**: `POST /swap/{id}` calls `HotSwapManager.swap(program_id)`. Returns `404` if the ID is not in the loaded cache, not if the program JSON is absent from disk (cache miss is a separate concern).

- **Model config persistence**: Config writes go through `config.settings.save_user_config()`, which writes to `~/.config/dspytools/config.toml` and invalidates the `ConfigCache`. Reads go through `config.settings.load_config()` for hot-reload support.

- **Entry point**: `run_api(host="0.0.0.0", port=8080)` starts uvicorn. Called from `commands/server.py` via `dspytools server start`.

### Dependencies

| Module | Usage |
|--------|-------|
| `dspytools.core.hotswap.HotSwapManager` | Program lifecycle (load, swap, infer, list, metadata) |
| `dspytools.config.settings.load_config` | Read model configuration with hot-reload |
| `dspytools.config.settings.save_user_config` | Persist model configuration changes |

## Work Guidance

- `get_hotswap()` is the single accessor for `HotSwapManager`. Do not create additional instances — the module-level `_hotswap` ensures one manager per server process.
- When adding new endpoints, add new Pydantic models at the top of `server.py` following the existing patterns.
- New model config fields added to `ModelConfig` must also be supported in `config/settings.py`'s TOML schema and in the corresponding CLI commands.
- Error responses use FastAPI `HTTPException` with appropriate status codes — do not return raw Python exceptions.
- The `/health` endpoint is meant for load balancers and container orchestrators. Keep it lightweight (no DB queries, no LM calls).

## Verification

- Start the server: `dspytools server start --port 8080`
- Health check: `curl http://localhost:8080/health`
- List programs: `curl http://localhost:8080/programs`
- Swap: `curl -X POST http://localhost:8080/swap/{run_id}`
- Infer: `curl -X POST http://localhost:8080/infer -H 'Content-Type: application/json' -d '{"inputs": {...}}'`
- Model config: `curl http://localhost:8080/config/models`

## Child DOX Index

No child directories. The `api/` directory contains only `server.py` and this `AGENTS.md`.
