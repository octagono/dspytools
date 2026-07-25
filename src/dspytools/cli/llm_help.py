"""LLM-powered help for all CLI commands.

Default: instant static rich-click help (no LM call, no network).
Use `--llm-help` flag on any command to get LLM-generated help.

Disk cache at ~/.cache/dspytools/help_cache.json persists across restarts.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import rich_click as click
from rich.markdown import Markdown
from rich.panel import Panel

from dspytools.cli.output import console
from dspytools.core._io import read_json, write_json
from dspytools.core.logging_config import get_logger

_log = get_logger(__name__)

# ── Disk cache (permanent — help text doesn't change) ────────────────────
_DISK_CACHE_PATH = Path.home() / ".cache" / "dspytools" / "help_cache.json"
_disk_cache_lock = threading.Lock()
_mem_cache: dict[str, str] = {}
_mem_loaded = False


def _load_disk_cache() -> None:
    """Load disk cache into memory (once per process)."""
    global _mem_loaded
    if _mem_loaded:
        return
    _mem_loaded = True
    if _DISK_CACHE_PATH.exists():
        _mem_cache.update(read_json(_DISK_CACHE_PATH))


def _save_disk_cache() -> None:
    """Persist memory cache to disk."""
    _DISK_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _disk_cache_lock:
        write_json(_DISK_CACHE_PATH, _mem_cache)


def _get_cached(key: str) -> str | None:
    _load_disk_cache()
    return _mem_cache.get(key)


def _set_cache(key: str, text: str) -> None:
    _mem_cache[key] = text
    _save_disk_cache()


# ── LLM help generation (only for --llm-help flag) ───────────────────────
def _llm_help(ctx: click.Context) -> str | None:
    """Generate LLM help via direct LM call. Returns None on failure."""
    key = ctx.command_path or "dspytools"
    cached = _get_cached(key)
    if cached is not None:
        return cached

    from dspytools.core.setup import LMRegistry

    lm = LMRegistry.get_or_default()
    if lm is None:
        return None

    cmd = ctx.command

    # Build subcommands string
    sub_list = ""
    if hasattr(cmd, "commands"):
        for name in sorted(cmd.commands):
            sub = cmd.commands[name]
            desc = sub.help or sub.short_help or ""
            sub_list += f"- {name}: {desc}\n"

    # Build options string
    opt_list = ""
    for p in cmd.params or []:
        if hasattr(p, "opts") and p.opts:
            h = p.help or ""
            req = " [required]" if getattr(p, "required", False) else ""
            opt_list += f"- {p.opts[0]}: {h}{req}\n"

    # Dynamic max_tokens: more subcommands = more output needed
    n_subs = len(cmd.commands) if hasattr(cmd, "commands") else 0
    max_tok = 256 if n_subs <= 3 else 512 if n_subs <= 8 else 768

    prompt = (
        f"Command: {key}\n"
        f"Description: {cmd.help or cmd.short_help or ''}\n"
        f"Subcommands:\n{sub_list or '(none)'}\n"
        f"Options:\n{opt_list or '(none)'}\n\n"
        f"Write a short, clear help text for this command. "
        f"Include usage examples. Be concise (3-8 lines max)."
    )

    response = lm(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tok,
        temperature=0.3,
    )

    answer = response[0] if isinstance(response, list) else str(response)
    answer = answer.strip()
    if answer and len(answer) > 20:
        _set_cache(key, answer)
        return answer
    return None


def _print_llm_help(ctx: click.Context) -> None:
    """Generate and print LLM help for the current command."""
    answer = _llm_help(ctx)
    if answer:
        console.print(
            Panel(
                Markdown(answer),
                title=f"[bold cyan]{ctx.command_path}[/]",
                border_style="cyan",
            )
        )
    else:
        console.print("[yellow]LLM help unavailable. Is the LLM server running?[/]")


# ── Custom Click classes — instant static help by default ───────────────
class LLMCommand(click.RichCommand):
    """Click Command with instant static help + optional LLM help via --llm-help."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    def get_help(self, ctx: click.Context) -> str:
        return super().get_help(ctx)


class LLMGroup(click.RichGroup):
    """Click Group with instant static help + optional LLM help via --llm-help."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    def get_help(self, ctx: click.Context) -> str:
        return super().get_help(ctx)


# ── Decorator shortcuts ─────────────────────────────────────────────────
def llm_command(*args: Any, **kwargs: Any) -> Any:
    kwargs.setdefault("cls", LLMCommand)
    return click.command(*args, **kwargs)


def llm_group(*args: Any, **kwargs: Any) -> Any:
    kwargs.setdefault("cls", LLMGroup)
    return click.group(*args, **kwargs)
