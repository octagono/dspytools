"""Self-optimizing CLI help system.

HelpManager loads a cached compiled DSPy program (or None when no cache
exists — callers fall through to static rich-click help). SelfOptimizingCLI
overrides click's get_help to use the DSPy-powered help, with rich-click as
the fallback for beautiful help output.
"""

from __future__ import annotations

from typing import Any

import rich_click as click
from rich.panel import Panel

from dspytools.cli.llm_help import LLMGroup
from dspytools.cli.output import console
from dspytools.core._dspy import dspy
from dspytools.core._io import read_json
from dspytools.core.logging_config import get_logger
from dspytools.core.setup import LMRegistry
from dspytools.help.context import (
    _build_examples,
    _format_subcommands,
    get_all_commands,
)

_log = get_logger(__name__)

__all__ = [
    "HelpManager",
    "SelfOptimizingCLI",
    "AutoCompiler",
]


class HelpManager:
    """Singleton manager for the self-help DSPy program."""

    # Bounded budget for LLM help generation. --help falls back to static
    # rich-click help if the LM is slow or unreachable within this window.
    _HELP_TIMEOUT: float = 15.0

    _module: Any = None
    _compiled: bool = False
    _cli: Any = None

    @classmethod
    def init(cls, cli: Any) -> None:
        cls._cli = cli

    @classmethod
    def _get_compiler(cls):
        """Lazily import AutoCompiler — defers the heavy DSPy import chain
        (help.optimize → help.module → dspy.predict, ~1s) until the help
        system is actually used. Keeps plain ``dspytools --help`` and every
        non-help command import-cheap."""
        from dspytools.help.optimize import AutoCompiler

        return AutoCompiler

    @classmethod
    def get_answer(cls, ctx: Any, command_path: str) -> str | None:
        """Return DSPy-generated help for the command, or None to use static help.

        Runs the compiled help module under a bounded timeout so ``--help``
        never blocks indefinitely on a slow or unreachable LM — on timeout the
        call returns None and Click falls back to rich-click static help.
        """
        import queue
        import threading

        if cls._cli is None:
            return None

        if cls._module is None:
            cls._module = cls._get_compiler().compile_if_needed(cls._cli)
            cls._compiled = cls._module is not None

        if cls._module is None:
            return None

        q: queue.Queue = queue.Queue(maxsize=1)

        def _generate() -> None:
            try:
                subcommands = ""
                if hasattr(ctx.command, "commands"):
                    cmds = get_all_commands(cls._cli)
                    cmd_name = (
                        command_path.replace("dspytools ", "").split()[-1]
                        if " " in command_path
                        else command_path
                    )
                    if cmd_name in cmds:
                        subcommands = _format_subcommands(
                            cmds[cmd_name].get("subcommands", [])
                        )
                examples = _build_examples(command_path, [])
                lm = LMRegistry.get_or_default()
                with dspy.context(lm=lm):
                    result = cls._module(
                        command=command_path,
                        subcommands=subcommands,
                        examples=examples,
                    )
                q.put(getattr(result, "answer", str(result)))
            except Exception as e:  # noqa: BLE001 — help must never crash the CLI
                _log.warning("help_generation_failed", error=str(e))
                q.put(None)

        # Bounded worker thread: a dead/slow LM falls back to static help.
        worker = threading.Thread(target=_generate, daemon=True)
        worker.start()
        try:
            return q.get(timeout=cls._HELP_TIMEOUT)
        except queue.Empty:
            _log.warning("help_generation_timeout", command_path=command_path)
            return None

    @classmethod
    def status(cls) -> dict:
        compiler = cls._get_compiler()
        compiled = compiler.is_compiled()
        meta = {}
        if compiled and compiler.META_PATH.exists():
            meta = read_json(compiler.META_PATH)
        return {"compiled": compiled, "cache": str(compiler.CACHE_PATH), **meta}


class SelfOptimizingCLI(LLMGroup):
    """Click Group that uses DSPy-powered help with rich-click fallback.

    Inherits from LLMGroup (dspytools.cli.llm_help) so every dspytools
    group — including the root CLI — stays rooted in the same rich-click
    group base used by the 24 subcommand groups.

    When the DSPy help module is compiled, it renders AI-generated help.
    Otherwise, falls back to rich-click's beautiful grouped help output.
    """

    def get_help(self, ctx: click.Context) -> str:
        if ctx.invoked_subcommand:
            return click.RichGroup.get_help(self, ctx)
        HelpManager.init(ctx.find_root().command)
        answer = HelpManager.get_answer(ctx, self.name or "dspytools")
        if answer and answer.strip():
            console.print(Panel(answer, title="Help", border_style="cyan"))
            return ""
        return click.RichGroup.get_help(self, ctx)


def __getattr__(name: str):
    """PEP-562 lazy module attribute.

    Supports ``from dspytools.help import AutoCompiler`` without paying the
    ~1s DSPy import chain at module load — AutoCompiler is imported on first
    actual access (compile_if_needed / is_compiled / force_compile).
    """
    if name == "AutoCompiler":
        from dspytools.help.optimize import AutoCompiler

        return AutoCompiler
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
