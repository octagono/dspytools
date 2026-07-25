"""Self-optimizing CLI help system.

HelpManager loads a cached compiled DSPy program or auto-compiles on first use.
SelfOptimizingCLI overrides click's get_help to use the DSPy-powered help,
with rich-click as the fallback for beautiful help output.
"""

from __future__ import annotations

from typing import Any

import rich_click as click
from rich.panel import Panel

from dspytools.cli.output import console
from dspytools.core._dspy import dspy
from dspytools.core._io import read_json
from dspytools.core.setup import LMRegistry
from dspytools.help.context import (
    _build_examples,
    _format_subcommands,
    get_all_commands,
)
from dspytools.help.optimize import AutoCompiler

__all__ = [
    "HelpManager",
    "SelfOptimizingCLI",
    "AutoCompiler",
]


class HelpManager:
    """Singleton manager for the self-help DSPy program."""

    _module: Any = None
    _compiled: bool = False
    _cli: Any = None

    @classmethod
    def init(cls, cli: Any) -> None:
        cls._cli = cli

    @classmethod
    def get_answer(cls, ctx: Any, command_path: str) -> str | None:
        if cls._cli is None:
            return None

        if cls._module is None:
            cls._module = AutoCompiler.compile_if_needed(cls._cli)
            cls._compiled = cls._module is not None

        if cls._module is None:
            return None

        subcommands = ""
        if hasattr(ctx.command, "commands"):
            cmds = get_all_commands(cls._cli)
            cmd_name = (
                command_path.replace("dspytools ", "").split()[-1]
                if " " in command_path
                else command_path
            )
            if cmd_name in cmds:
                subcommands = _format_subcommands(cmds[cmd_name].get("subcommands", []))

        examples = _build_examples(command_path, [])

        lm = LMRegistry.get_or_default()
        with dspy.context(lm=lm):
            result = cls._module(
                command=command_path, subcommands=subcommands, examples=examples
            )
        return getattr(result, "answer", str(result))

    @classmethod
    def _init_module(cls) -> None:
        cls._module = AutoCompiler.compile_if_needed(cls._cli)
        cls._compiled = AutoCompiler.is_compiled()

    @classmethod
    def status(cls) -> dict:
        compiled = AutoCompiler.is_compiled()
        meta = {}
        if compiled and AutoCompiler.META_PATH.exists():
            meta = read_json(AutoCompiler.META_PATH)
        return {"compiled": compiled, "cache": str(AutoCompiler.CACHE_PATH), **meta}


class SelfOptimizingCLI(click.RichGroup):
    """Click Group that uses DSPy-powered help with rich-click fallback.

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
