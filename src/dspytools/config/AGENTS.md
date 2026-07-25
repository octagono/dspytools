# DOX — configuration management

## Purpose

Central configuration and environment management for dspytools. Provides hot-reload config caching (TOML-based, single `~/.config/dspytools/config.toml`), `.env` file read/write, and path resolution for all dspytools data directories (SSOT under `~/.config/dspytools/`).

## Ownership

Owns two source files:

| File | Role |
|------|------|
| `settings.py` | `ConfigCache` singleton, `load_config`/`save_config`/`save_user_config`, path helpers (project_root, config_dir, data_dir, compiled_dir, etc.) |
| `env.py` | `.env` file loading and API key management (set_key, get_key, list_keys) |

## Local Contracts

### `settings.py` — ConfigCache with hot-reload

- **ConfigCache is a class-level singleton** with a 1-second stat throttle. On `get()`, it stats the config files' mtime and only re-reads when changed. Saves ~1ms per read in hot paths (FastAPI serving 100+ req/s).
- **Config merge order** (later values win):
  1. Hardcoded defaults in `ConfigCache.get()`:
     - `server.mcp.enabled: false, transport: stdio, port: 8002`
     - `server.api.enabled: false, port: 8080`
     - `lm.student: null, lm.teacher: null`
     - `keys: {}`
  2. User config: `~/.config/dspytools/config.toml` — **SSOT**, single config file
  3. ~~Project config~~ — removed. All config lives under user config only.
- **`_deep_merge(base, update)`** — recursive dict merge. Lists are replaced, not extended. Nested dicts are merged recursively.
- **`load_config()`** — alias for `ConfigCache.get()`. Use this for reads to get hot-reload behavior.
- **`save_config(cfg)`** — writes to project config dir, invalidates cache.
- **`save_user_config(cfg)`** — writes to user config dir, invalidates cache.
- **`ConfigCache.invalidate()`** — forces a full re-read on next `get()`.
- **Path helpers** — functions returning `Path` objects, all calling `.mkdir(parents=True, exist_ok=True)`:
  - `project_root()` — walks up from cwd looking for `.dspytools/` dir or a parent dir named `dspytools`. Returns `cwd` as fallback.
  - `config_dir()` — `~/.config/dspytools/` — **SSOT base** for all persistent data
  - `data_dir()` — `~/.config/dspytools/data/`
  - `cache_dir()` — `~/.config/dspytools/cache/`
  - `skills_dir()` — `~/.config/dspytools/skills/`
  - `project_config_dir()` — `<project_root>/.dspytools/` (minimal, project-override only)
  - `compiled_dir()` — `~/.config/dspytools/compiled/`
  - `signatures_dir()` — `~/.config/dspytools/signatures/`
  - `modules_dir()` — `~/.config/dspytools/modules/`
  - `agents_dir()` — `~/.config/dspytools/agents/`
  - `distill_dir()` — `~/.config/dspytools/distill/`
  - `dspy_lora_dir()` — `~/.config/dspytools/lora/` (or `DSPY_LORA_DIR` env var)
  - `adapters_dir()` — `~/.config/dspytools/adapters/`
  - `mlflow_tracking_uri()` — `os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")`
  - `mlflow_experiment_name()` — `os.environ.get("MLFLOW_EXPERIMENT_NAME", "dspytools")`
  - `embedder_kwargs()` — returns `{model, api_base, api_key}` dict for `dspy.Embedder` (SSOT)
  - `embedder_dimension()` — reads `DSPYTOOLS_EMBEDDING_DIM` env var, returns int (default 768 for embeddinggemma)

- **Settings schema** (TOML keys):

  | Key | Type | Default | Description |
  |-----|------|---------|-------------|
  | `server.mcp.enabled` | bool | false | Enable MCP server |
  | `server.mcp.transport` | string | "stdio" | MCP transport: stdio or sse |
  | `server.mcp.port` | int | 8002 | MCP SSE port |
  | `server.api.enabled` | bool | false | Enable FastAPI server |
  | `server.api.port` | int | 8080 | FastAPI port |
  | `lm.student` | dict | null | Student model config (model, api_base, api_key, temperature, max_tokens) |
  | `lm.teacher` | dict | null | Teacher model config (same fields) |
  | `keys` | dict | {} | API key overrides |

### `env.py` — .env file management

- **`load_env(path=None)`** — reads `.env` from `~/.config/dspytools/.env` (or given path), returns `dict[str, str]`. Does NOT modify `os.environ`. Override via `DSPYTOOLS_ENV_FILE` env var.
- **`merge_environ(env_dict)`** — calls `os.environ.setdefault(k, v)` for each pair. Only sets keys not already in environment.
- **`set_key(provider, key, env_path=None)`** — sets/updates an API key in `.env` and `os.environ` (default: `~/.config/dspytools/.env`). Variable name format: `{PROVIDER}_API_KEY` (uppercased).
- **`get_key(provider)`** — checks `os.environ` first, then `.env`. Returns the key value or `None`.
- **`list_keys()`** — returns `{provider: masked_value}` dict from `.env`. Values truncated to first 8 chars + `...`.

### Shared contracts

- **Config loading order**: environment variables → `ConfigCache` (user) → defaults. Environment variables take highest priority. Project config (`<project>/.dspytools/config.toml`) is no longer merged — SSOT is the user config.
- **API key resolution**: `LMRegistry` in `core/setup.py` calls `env.get_key(provider)` when constructing `dspy.LM` instances. Keys can live in `.env` or `os.environ`.
- **Path determinism**: All path helpers use `mkdir(parents=True, exist_ok=True)` — they always succeed and return a valid path. No file existence checks needed at call sites.

## Work Guidance

- `ConfigCache` is the sole config access path for all dspytools modules. Do not parse TOML files directly elsewhere.
- When adding new config keys, update the defaults dict in `ConfigCache.get()`, the merge logic, and the schema table above.
- Path helpers should remain stateless — they derive from `Path.cwd()` at call time, not at import time.
- `env.py` functions are thin wrappers over file I/O and `os.environ`. They have no dspytools package dependencies and can be imported by setup scripts.
- `merge_environ` uses `setdefault` intentionally — it never overwrites an existing env var. This preserves secrets injected by Docker/CI/Orchestrator.
- When adding a new data directory (e.g., for a new artifact type), add a path helper in `settings.py` following the existing pattern.

## Verification

- Load config: `python -c "from dspytools.config.settings import load_config; print(load_config())"`
- Save config: `python -c "from dspytools.config.settings import save_config; save_config({'lm': {'student': {'model': 'test'}}})"`
- Cache invalidation: call `save_config` then `load_config` — should reflect the change without restart.
- Env round-trip: `python -c "from dspytools.config.env import set_key, get_key; set_key('test', 'sk-abc123'); print(get_key('test'))"`
- Path helpers: `python -c "from dspytools.config.settings import config_dir; print(config_dir())"` — should print `~/.config/dspytools`.

## Child DOX Index

No child directories. All config source files are flat in this directory.
