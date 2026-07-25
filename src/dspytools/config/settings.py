"""Configuration management with hot-reload.

Optimization 4: project_root() cached per process — eliminates 10+ fs walks per CLI invocation.
Optimization 5: load_env() cached — avoids re-reading .env on every setup_dspy() call.
Optimization 13: _env_path() cached — eliminates redundant mkdir syscalls (10+ per CLI).

All paths support DSPYTOOLS_<NAME>_DIR env var overrides.
Every other module imports from here instead of hardcoding paths.
"""

import os
import time
from pathlib import Path

import toml  # type: ignore[import-untyped]

# ═══════════════════════════════════════════════════════════════════════════
# SSOT project-wide constants — every module imports from here
# instead of hardcoding these values.
# ═══════════════════════════════════════════════════════════════════════════

DEFAULT_SEED: int = 42
"""Default random seed for all deterministic operations (holdout split, probe split,
valset shuffle, LoRA/distill training). Change once to affect all subsystems."""

DEFAULT_STUDENT_MODEL: str = "unsloth/Qwen3.5-9B-GGUF"
"""Default student model name without provider prefix. Used in lora.py as fallback
when no student model is configured. Override via config lm.student.model."""


# Optimization 13: Cache _env_path() results — avoid redundant mkdir syscalls
_env_path_cache: dict[str, Path] = {}


def _env_path(name: str, default: Path) -> Path:
    """Return env override or default path, creating parents once.

    Optimization 13: Caches the result per (name, default) key.
    The first call creates the directory; subsequent calls return the cached path.
    """
    cache_key = f"{name}|{default}"
    if cache_key in _env_path_cache:
        return _env_path_cache[cache_key]

    val = os.environ.get(f"DSPYTOOLS_{name}_DIR")
    p = Path(val) if val else default
    p.mkdir(parents=True, exist_ok=True)
    _env_path_cache[cache_key] = p
    return p


# ── Project root (Optimization 4: cached per process) ─────────────────────

_project_root_cache: Path | None = None


def project_root() -> Path:
    """Project root (project/.dspytools/ marker). Override: DSPYTOOLS_PROJECT_DIR.

    Optimization 4: Cached per process — the project root never changes within
    a single CLI invocation. Eliminates 10+ filesystem walks per command.
    """
    global _project_root_cache
    if _project_root_cache is not None:
        return _project_root_cache

    env = os.environ.get("DSPYTOOLS_PROJECT_DIR")
    if env:
        _project_root_cache = Path(env)
        return _project_root_cache

    cwd = Path.cwd()
    for parent in [cwd] + list(cwd.parents):
        if (parent / ".dspytools").is_dir():
            _project_root_cache = parent
            return _project_root_cache
        if (parent / "pyproject.toml").is_file() and "dspytools" in parent.name:
            _project_root_cache = parent
            return _project_root_cache

    _project_root_cache = cwd
    return _project_root_cache


# ── User-wide directories ──────────────────────────────────────────────────


def config_dir() -> Path:
    """User config dir. Default: ~/.config/dspytools. Env: DSPYTOOLS_CONFIG_DIR."""
    return _env_path("CONFIG", Path.home() / ".config" / "dspytools")


def data_dir() -> Path:
    """Dataset storage. Default: ~/.config/dspytools/data. Env: DSPYTOOLS_DATA_DIR."""
    return _env_path("DATA", config_dir() / "data")


def cache_dir() -> Path:
    """Analysis cache dir. Default: ~/.config/dspytools/cache. Env: DSPYTOOLS_CACHE_DIR."""
    return _env_path("CACHE", config_dir() / "cache")


def skills_dir() -> Path:
    """User-wide skills dir. Default: ~/.config/dspytools/skills. Env: DSPYTOOLS_SKILLS_DIR."""
    return _env_path("SKILLS", config_dir() / "skills")


def adapters_dir() -> Path:
    """LoRA adapter storage. Default: ~/.config/dspytools/adapters. Env: DSPYTOOLS_ADAPTERS_DIR."""
    return _env_path("ADAPTERS", config_dir() / "adapters")


# ── SSOT data directories (all under ~/.config/dspytools/) ────────────────


def project_config_dir() -> Path:
    """Project-local config. Default: <project>/.dspytools. Env: DSPYTOOLS_PROJECT_CONFIG_DIR.
    Only for project-specific overrides — not for persistent data."""
    return _env_path("PROJECT_CONFIG", project_root() / ".dspytools")


def compiled_dir() -> Path:
    """Compiled program output. Default: ~/.config/dspytools/compiled. Env: DSPYTOOLS_COMPILED_DIR."""
    return _env_path("COMPILED", config_dir() / "compiled")


def signatures_dir() -> Path:
    """Signature file storage. Default: ~/.config/dspytools/signatures. Env: DSPYTOOLS_SIGNATURES_DIR."""
    return _env_path("SIGNATURES", config_dir() / "signatures")


def modules_dir() -> Path:
    """Module file storage. Default: ~/.config/dspytools/modules. Env: DSPYTOOLS_MODULES_DIR."""
    return _env_path("MODULES", config_dir() / "modules")


def agents_dir() -> Path:
    """Agent file storage. Default: ~/.config/dspytools/agents. Env: DSPYTOOLS_AGENTS_DIR."""
    return _env_path("AGENTS", config_dir() / "agents")


def distill_dir() -> Path:
    """Distillation output. Default: ~/.config/dspytools/distill. Env: DSPYTOOLS_DISTILL_DIR."""
    return _env_path("DISTILL", config_dir() / "distill")


def dspy_lora_dir() -> Path | None:
    """Path to the dspy-lora project (external dependency for distillation).
    Env: DSPY_LORA_DIR. Default: ~/.config/dspytools/lora.

    Self-sufficient: loads .env first so the env var is visible without
    callers needing to call load_env()/merge_environ() manually.
    """
    from dspytools.config.env import load_env, merge_environ

    merge_environ(load_env())
    env = os.environ.get("DSPY_LORA_DIR")
    return Path(env) if env else config_dir() / "lora"


# ── LLM / Embedding server URLs ───────────────────────────────────────────


def llm_url() -> str:
    """LLM inference base URL.

    Resolution order:
    1. DSPYTOOLS_LLM_URL env var
    2. Student LM's api_base from config (strips /v1 or /v1/ suffix)
    3. Fallback: http://127.0.0.1:8000
    """
    env = os.environ.get("DSPYTOOLS_LLM_URL")
    if env:
        return env
    cfg = load_config()
    student = cfg.get("lm", {}).get("student", {})
    api_base = student.get("api_base", "")
    if api_base:
        # Strip /v1 or /v1/ suffix to get base URL
        for suffix in ("/v1/", "/v1"):
            if api_base.endswith(suffix):
                return api_base[: -len(suffix)]
        return api_base.rstrip("/")
    return "http://127.0.0.1:8000"


def embedding_url() -> str:
    """Embedding server URL. Default: http://127.0.0.1:8001/v1. Env: DSPYTOOLS_EMBEDDING_URL."""
    cfg = load_config().get("embedding", {})
    return cfg.get("url") or os.environ.get(
        "DSPYTOOLS_EMBEDDING_URL", "http://127.0.0.1:8001/v1"
    )


def embedding_model() -> str:
    """Embedding model name. Env: DSPYTOOLS_EMBEDDING_MODEL."""
    cfg = load_config().get("embedding", {})
    return cfg.get("model") or os.environ.get(
        "DSPYTOOLS_EMBEDDING_MODEL", "embeddinggemma"
    )


def embedder_kwargs() -> dict:
    """Embedding config for dspy.Embedder — model + url + api_key. Single source of truth."""
    model = embedding_model()
    if not model.startswith(("openai/", "llama_cpp/", "text-")):
        model = f"openai/{model}"
    return {
        "model": model,
        "api_base": embedding_url(),
        "api_key": "sk-local",
    }


def embedder_dimension() -> int:
    """Embedding vector dimension. Env: DSPYTOOLS_EMBEDDING_DIM. Default: 768 (embeddinggemma)."""
    cfg = load_config().get("embedding", {})
    dim = cfg.get("dimension")
    if dim is not None:
        return int(dim)
    return int(os.environ.get("DSPYTOOLS_EMBEDDING_DIM", "768"))


def llama_cpp_url() -> str:
    """llama-cpp-server API URL. Default: http://127.0.0.1:8080. Env: DSPYTOOLS_LLAMA_CPP_URL."""
    return os.environ.get("DSPYTOOLS_LLAMA_CPP_URL", "http://127.0.0.1:8080")


# ── Persistent state files (all under config_dir) ─────────────────────────


def help_cache_path() -> Path:
    """Compiled help cache. Env: DSPYTOOLS_HELP_CACHE."""
    env = os.environ.get("DSPYTOOLS_HELP_CACHE")
    return Path(env) if env else config_dir() / "help" / "compiled.json"


def help_meta_path() -> Path:
    """Help cache metadata. Env: DSPYTOOLS_HELP_META."""
    env = os.environ.get("DSPYTOOLS_HELP_META")
    return Path(env) if env else config_dir() / "help" / "meta.json"


def cache_threshold_path() -> Path:
    """Cache threshold config. Env: DSPYTOOLS_CACHE_THRESHOLD_PATH."""
    env = os.environ.get("DSPYTOOLS_CACHE_THRESHOLD_PATH")
    return Path(env) if env else config_dir() / "cache" / "thresholds.json"


def drift_state_path() -> Path:
    """Drift monitor state. Env: DSPYTOOLS_DRIFT_STATE."""
    env = os.environ.get("DSPYTOOLS_DRIFT_STATE")
    return Path(env) if env else config_dir() / "monitor" / "drift_state.json"


def lse_log_path() -> Path:
    """LSE tracker log. Env: DSPYTOOLS_LSE_LOG."""
    env = os.environ.get("DSPYTOOLS_LSE_LOG")
    return Path(env) if env else config_dir() / "gfl" / "lse_log.json"


def grao_log_path() -> Path:
    """GRAO meta-optimizer log. Env: DSPYTOOLS_GRAO_LOG."""
    env = os.environ.get("DSPYTOOLS_GRAO_LOG")
    return Path(env) if env else config_dir() / "gfl" / "grao_log.json"


def meta_optimizer_path() -> Path:
    """Meta optimizer state. Env: DSPYTOOLS_META_OPTIMIZER."""
    env = os.environ.get("DSPYTOOLS_META_OPTIMIZER")
    return Path(env) if env else config_dir() / "gfl" / "meta_optimizer.json"


def quality_log_path() -> Path:
    """Quality monitor log. Env: DSPYTOOLS_QUALITY_LOG."""
    env = os.environ.get("DSPYTOOLS_QUALITY_LOG")
    return Path(env) if env else config_dir() / "monitor" / "quality_log.json"


def trajectories_db_path() -> Path:
    """Trajectory DB. Env: DSPYTOOLS_TRAJECTORIES_DB."""
    env = os.environ.get("DSPYTOOLS_TRAJECTORIES_DB")
    return Path(env) if env else config_dir() / "evolve" / "trajectories.db"


def morphology_path() -> Path:
    """Morphology tracker state. Env: DSPYTOOLS_MORPHOLOGY."""
    env = os.environ.get("DSPYTOOLS_MORPHOLOGY")
    return Path(env) if env else config_dir() / "evolve" / "morphology.json"


def ucb_explorer_path() -> Path:
    """UCB explorer state. Env: DSPYTOOLS_UCB_EXPLORER."""
    env = os.environ.get("DSPYTOOLS_UCB_EXPLORER")
    return Path(env) if env else config_dir() / "evolve" / "ucb_explorer.json"


def evolve_scores_path() -> Path:
    """Evolve scores history. Env: DSPYTOOLS_EVOLVE_SCORES."""
    env = os.environ.get("DSPYTOOLS_EVOLVE_SCORES")
    return Path(env) if env else config_dir() / "evolve" / "scores.json"


def skill_graph_path() -> Path:
    """Skill graph state. Env: DSPYTOOLS_SKILL_GRAPH."""
    env = os.environ.get("DSPYTOOLS_SKILL_GRAPH")
    return Path(env) if env else config_dir() / "evolve" / "skill_graph.json"


# ── MLflow Settings ───────────────────────────────────────────────────────


def mlflow_tracking_uri() -> str:
    return os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")


def mlflow_experiment_name() -> str:
    return os.environ.get("MLFLOW_EXPERIMENT_NAME", "dspytools")


class ConfigCache:
    """Optimization 5: Cached config with mtime check — stat-only on read.

    Only re-reads files when they change, saving ~1ms per read.
    In the FastAPI server serving 100+ req/s, this eliminates redundant I/O.
    """

    _cfg: dict | None = None
    _user_mtime: float = 0
    _proj_mtime: float = 0
    _last_check: float = 0
    _check_interval: float = 1.0  # Don't stat more than once per second

    @classmethod
    def get(cls) -> dict:
        now = time.time()

        # Fast path: return cached config if within check interval
        if cls._cfg is not None and (now - cls._last_check) < cls._check_interval:
            return cls._cfg

        cls._last_check = now

        # Stat files to check if anything changed since last full read
        # Optimization: single stat() call instead of exists()+stat() (N+1 syscall)
        user_path = config_dir() / "config.toml"
        proj_path = project_config_dir() / "config.toml"
        try:
            user_mtime = user_path.stat().st_mtime
        except OSError:
            user_mtime = 0
        try:
            proj_mtime = proj_path.stat().st_mtime
        except OSError:
            proj_mtime = 0

        # If cache exists and no files changed, return cached config
        if (
            cls._cfg is not None
            and user_mtime == cls._user_mtime
            and proj_mtime == cls._proj_mtime
        ):
            return cls._cfg

        # Rebuild from defaults + config files
        cfg: dict = {
            "server": {
                "mcp": {"enabled": False, "transport": "stdio", "port": 8002},
                "api": {"enabled": False, "port": 8080},
            },
            "lm": {"default": None, "registry": {}, "student": None, "teacher": None},
            "keys": {},
        }

        # User config (~/.config/dspytools/config.toml)
        # mtime != 0 means file exists — skip redundant exists() call
        if user_mtime:
            user_cfg = toml.loads(user_path.read_text())
            _deep_merge(cfg, user_cfg)
        cls._user_mtime = user_mtime

        # Project config (<project>/.dspytools/config.toml)
        if proj_mtime:
            proj_cfg = toml.loads(proj_path.read_text())
            _deep_merge(cfg, proj_cfg)
        cls._proj_mtime = proj_mtime

        cls._cfg = cfg
        return cfg

    @classmethod
    def invalidate(cls) -> None:
        """Force re-read on next access."""
        cls._cfg = None
        cls._last_check = 0


def load_config() -> dict:
    """Load config via ConfigCache for hot-reload support."""
    return ConfigCache.get()


def save_config(cfg: dict) -> None:
    """Write project-level config and invalidate cache."""
    path = project_config_dir() / "config.toml"
    path.write_text(toml.dumps(cfg))
    ConfigCache.invalidate()


def save_user_config(cfg: dict) -> None:
    """Write user-wide config and invalidate cache."""
    path = config_dir() / "config.toml"
    path.write_text(toml.dumps(cfg))
    ConfigCache.invalidate()


def _deep_merge(base: dict, update: dict) -> None:
    for k, v in update.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v
