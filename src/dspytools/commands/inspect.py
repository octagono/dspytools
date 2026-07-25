"""dspytools inspect — Inspect DSPy internals (history, config, caches)."""

from __future__ import annotations

import io
import sys

from dspytools import __version__
from dspytools.cli.output import console, info, panel, table
from dspytools.cli.rich_config import LLMCommand, LLMGroup, click


@click.group(name="inspect", cls=LLMGroup)
def inspect_cmd():
    """Inspect DSPy internals (history, config, LM calls)."""


@inspect_cmd.command(name="history", cls=LLMCommand)
@click.option("--n", "-n", default=5, type=int, help="Number of last LM calls to show")
def inspect_history(n: int):
    """Show the last N LM call history entries."""
    from dspytools.core._dspy import dspy

    dspy.inspect_history(n=n)


@inspect_cmd.command(name="config", cls=LLMCommand)
def inspect_config():
    """Show current DSPy configuration."""
    from dspytools.core._dspy import dspy

    settings = dspy.settings
    config_items = {
        "LM": str(settings.lm)
        if hasattr(settings, "lm") and settings.lm
        else "not set",
        "Adapter": type(settings.adapter).__name__
        if hasattr(settings, "adapter") and settings.adapter
        else "default",
        "Num Threads": str(getattr(settings, "num_threads", "N/A")),
        "Track Usage": str(getattr(settings, "track_usage", "N/A")),
    }
    text = "\n".join(f"[bold]{k}:[/] {v}" for k, v in config_items.items())
    panel("DSPy Configuration", text, border_style="cyan")


@inspect_cmd.command(name="cache", cls=LLMCommand)
def inspect_cache():
    """Show cache configuration."""
    info("Cache config: check ~/.cache/dspy/")


@inspect_cmd.command(name="tools", cls=LLMCommand)
def inspect_tools():
    """Show registered DSPy tools and their signatures."""
    from dspytools.mcp.tools import BUILTIN_TOOLS

    rows = []
    for name, spec in BUILTIN_TOOLS.items():
        rows.append([name, spec.get("description", "")[:80]])

    table("Registered DSPy Tools", ["Name", "Description"], rows)


@inspect_cmd.command(name="history-detail", cls=LLMCommand)
@click.option(
    "--n", default=1, type=int, help="Which history entry to inspect (1=latest)"
)
def inspect_history_detail(n: int):
    """Show detailed information about a specific LM call history entry."""

    from dspytools.core._dspy import dspy

    old = sys.stdout
    buf = io.StringIO()
    sys.stdout = buf
    dspy.inspect_history(n=n)
    sys.stdout = old
    console.print(buf.getvalue()[:3000])


@inspect_cmd.command(name="experimental", cls=LLMCommand)
def inspect_experimental():
    """Show DSPy experimental features (Citations, Document)."""
    panel(
        "DSPy Experimental",
        "[bold]Available:[/]\n"
        "  • dspy.experimental.Citations — Citation tracking\n"
        "  • dspy.experimental.Document — Document handling\n",
        border_style="cyan",
    )


@inspect_cmd.command(name="version", cls=LLMCommand)
def inspect_version():
    """Show DSPy and dspytools version info."""
    from dspytools.core._dspy import dspy

    panel(
        "Version Info",
        f"[bold]dspytools:[/] {__version__}\n"
        f"[bold]dspy:[/] {dspy.__version__}\n"
        f"[bold]installed modules:[/] {len([m for m in dir(dspy) if not m.startswith('_')])}",
        border_style="cyan",
    )
