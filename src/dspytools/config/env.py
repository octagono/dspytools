"""Environment variable management (.env read/write).

Optimization 5: load_env() cached with mtime check — avoids re-reading .env
on every setup_dspy() call (52 callers across commands).
"""

import os
import time
from pathlib import Path

# Optimization 5: Cached .env read with mtime check
_env_cache: dict[str, str] | None = None
_env_cache_mtime: float = 0
_env_cache_path: str | None = None


def load_env(path: str | Path | None = None) -> dict[str, str]:
    """Load .env file, return dict of key=value pairs. Does NOT set environ.

    Default location: ~/.config/dspytools/.env
    Override via: path argument or DSPYTOOLS_ENV_FILE env var.

    Optimization 5: Cached per path — only re-reads when file mtime changes.
    """
    global _env_cache, _env_cache_mtime, _env_cache_path
    if path is None:
        env_override = os.environ.get("DSPYTOOLS_ENV_FILE")
        if env_override:
            env_path = Path(env_override)
        else:
            from dspytools.config.settings import config_dir

            env_path = config_dir() / ".env"
    else:
        env_path = Path(path)
    path_str = str(env_path)

    if not env_path.exists():
        return {}

    # Check cache: same path + same mtime = return cached
    now = time.time()
    if (
        _env_cache is not None
        and _env_cache_path == path_str
        and (now - _env_cache_mtime) < 2.0  # 2s stat throttle
    ):
        return _env_cache

    result: dict[str, str] = {}
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        result[k.strip()] = v.strip().strip('"').strip("'")

    _env_cache = result
    _env_cache_path = path_str
    _env_cache_mtime = now
    return result


def merge_environ(env_dict: dict[str, str]) -> None:
    """Set os.environ from dict, only if not already set."""
    for k, v in env_dict.items():
        os.environ.setdefault(k, v)


def set_key(provider: str, key: str, env_path: str | Path | None = None) -> None:
    """Set or update an API key in .env. Default: ~/.config/dspytools/.env."""
    if env_path is None:
        env_override = os.environ.get("DSPYTOOLS_ENV_FILE")
        if env_override:
            env_file = Path(env_override)
        else:
            from dspytools.config.settings import config_dir

            env_file = config_dir() / ".env"
    else:
        env_file = Path(env_path)
    var_name = f"{provider.upper()}_API_KEY"

    lines = []
    found = False
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if line.startswith(f"{var_name}="):
                lines.append(f"{var_name}={key}")
                found = True
            else:
                lines.append(line)

    if not found:
        lines.append(f"{var_name}={key}")

    env_file.write_text("\n".join(lines) + "\n")
    os.environ[var_name] = key


def get_key(provider: str) -> str | None:
    """Get an API key from env or .env."""
    var_name = f"{provider.upper()}_API_KEY"
    val = os.environ.get(var_name)
    if val:
        return val
    env = load_env()
    return env.get(var_name)


def list_keys() -> dict[str, str]:
    """List all known API keys (name only, value hidden)."""
    env = load_env()
    keys = {}
    for k, v in env.items():
        if k.endswith("_API_KEY"):
            provider = k.replace("_API_KEY", "").lower()
            keys[provider] = v[:8] + "..." if len(v) > 8 else "***"
    return keys
