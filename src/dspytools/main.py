"""DSPyTools CLI — Entry point.

Usage:
    dspytools [command] [options]
"""

import atexit
import importlib
import signal
import sys
import time
from typing import Any

from dspytools import __version__
from dspytools.cli.llm_help import LLMGroup
from dspytools.cli.rich_config import click
from dspytools.core.logging_config import get_logger

_log = get_logger(__name__)

# Cached MLflow tracker reference — set by get_tracker() when first created.
# Avoids importing mlflow during atexit shutdown (causes RuntimeError).
_mlflow_tracker_ref = None


class LazyGroup(LLMGroup):
    """Click group that lazily imports subcommand modules.

    Uses LLM-powered help for all --help invocations.
    Only imports command modules when the subcommand is actually invoked.
    """

    def __init__(self, name=None, commands=None, **attrs):
        super().__init__(name, commands, **attrs)
        self._lazy_commands: dict[str, tuple[str, str]] = {}

    def add_lazy_command(self, name: str, module_path: str, attr_name: str) -> None:
        """Register a command to be lazily loaded.

        Args:
            name: CLI subcommand name (e.g., 'compile')
            module_path: Python import path (e.g., 'dspytools.commands.compile')
            attr_name: Attribute name in module (e.g., 'compile_cmd')
        """
        self._lazy_commands[name] = (module_path, attr_name)

    def get_command(self, ctx, cmd_name):
        """Override: lazy-load command if not yet imported."""
        cmd = super().get_command(ctx, cmd_name)
        if cmd is not None:
            return cmd

        if cmd_name in self._lazy_commands:
            module_path, attr_name = self._lazy_commands[cmd_name]
            mod = importlib.import_module(module_path)
            cmd = getattr(mod, attr_name, None)
            if cmd is not None:
                self.add_command(cmd, cmd_name)
                return cmd

        return None

    def list_commands(self, ctx):
        """List both registered and lazy commands."""
        base = list(super().list_commands(ctx))
        lazy = list(self._lazy_commands.keys())
        return sorted(set(base + lazy))

    def invoke(self, ctx: click.Context) -> Any:
        """Override: log CLI command lifecycle events.

        Captures every CLI invocation — command name, subcommand,
        duration, and exit status — as a structured structlog event.
        """
        start = time.monotonic()
        try:
            return super().invoke(ctx)
        except SystemExit as e:
            _log.info(
                "cli_command_exit",
                command=ctx.command.name,
                subcommand=ctx.invoked_subcommand,
                duration_s=round(time.monotonic() - start, 3),
                exit_code=e.code if e.code is not None else 0,
            )
            raise
        except click.ClickException as e:
            _log.warning(
                "cli_command_error",
                command=ctx.command.name,
                subcommand=ctx.invoked_subcommand,
                duration_s=round(time.monotonic() - start, 3),
                error=str(e),
                exit_code=e.exit_code,
            )
            raise
        except RuntimeError as e:
            # click.exceptions.Exit is a RuntimeError subclass (raised by --help)
            if type(e).__name__ == "Exit":
                _log.info(
                    "cli_command",
                    command=ctx.command.name,
                    subcommand=ctx.invoked_subcommand,
                    duration_s=round(time.monotonic() - start, 3),
                    exit_code=getattr(e, "code", 0),
                )
                raise
            _log.exception(
                "cli_command_crash",
                command=ctx.command.name,
                subcommand=ctx.invoked_subcommand,
                duration_s=round(time.monotonic() - start, 3),
                error=str(e),
            )
            raise
        except Exception as e:
            _log.exception(
                "cli_command_crash",
                command=ctx.command.name,
                subcommand=ctx.invoked_subcommand,
                duration_s=round(time.monotonic() - start, 3),
                error=str(e),
            )
            raise


@click.version_option(version=__version__, prog_name="dspytools")
def _cli():
    """DSPyTools — CLI for DSPy program management, MCP agents, and hot-swap inference."""


cli = LazyGroup(
    name="dspytools",
    help="DSPyTools — CLI for DSPy program management, MCP agents, and hot-swap inference.",
    context_settings={"help_option_names": ["-h", "--help"]},
    callback=_cli,
)

cli.add_lazy_command("configure", "dspytools.commands.configure", "configure_cmd")
cli.add_lazy_command("signature", "dspytools.commands.signature", "signature_cmd")
cli.add_lazy_command("module", "dspytools.commands.module", "module_cmd")
cli.add_lazy_command("run", "dspytools.commands.run", "run_cmd")
cli.add_lazy_command("compile", "dspytools.commands.compile", "compile_cmd")
cli.add_lazy_command("agent", "dspytools.commands.agent", "agent_cmd")
cli.add_lazy_command("tool", "dspytools.commands.tool", "tool_cmd")
cli.add_lazy_command("evaluate", "dspytools.commands.evaluate", "evaluate_cmd")
cli.add_lazy_command("data", "dspytools.commands.data", "data_cmd")
cli.add_lazy_command("inspect", "dspytools.commands.inspect", "inspect_cmd")
cli.add_lazy_command("mcp", "dspytools.commands.mcp", "mcp_cmd")
cli.add_lazy_command("server", "dspytools.commands.server", "server_cmd")
cli.add_lazy_command("self", "dspytools.commands.self", "self_cmd")
cli.add_lazy_command("gfl", "dspytools.commands.gfl", "gfl_cmd")
cli.add_lazy_command("skills", "dspytools.commands.skills", "skills_cmd")
cli.add_lazy_command("generate", "dspytools.commands.generate", "generate_cmd")
cli.add_lazy_command("pipeline", "dspytools.commands.pipeline", "pipeline_cmd")
cli.add_lazy_command("export", "dspytools.commands.export", "export_cmd")
cli.add_lazy_command("compare", "dspytools.commands.compare", "compare_cmd")
cli.add_lazy_command("doctor", "dspytools.commands.doctor", "doctor_cmd")
cli.add_lazy_command("lora", "dspytools.commands.lora", "lora_cmd")
cli.add_lazy_command("distill", "dspytools.commands.distill", "distill_cmd")
cli.add_lazy_command("graph", "dspytools.commands.graph", "graph_cmd")
cli.add_lazy_command("memory", "dspytools.commands.memory", "memory_cmd")


# ── MLflow flush on exit ────────────────────────────────────────────────────


def _flush_mlflow():
    """Flush MLflow async queue on process exit if tracker was ever used.

    Uses cached tracker reference to avoid importing mlflow during shutdown
    (mlflow's import chain triggers concurrent.futures which fails with
    'can't register atexit after shutdown' during interpreter exit).
    """
    if _mlflow_tracker_ref is None:
        return  # Never used — nothing to flush
    if hasattr(_mlflow_tracker_ref, "flush"):
        result = _mlflow_tracker_ref.flush(timeout=2.0)
        if result.get("drained", 0) > 0:
            _log.debug("mlflow_flushed", result=str(result))


atexit.register(_flush_mlflow)


# Also flush on SIGTERM/SIGINT
def _signal_handler(signum, frame):
    _flush_mlflow()
    sys.exit(0)


signal.signal(signal.SIGTERM, _signal_handler)
signal.signal(signal.SIGINT, _signal_handler)


if __name__ == "__main__":
    cli()
