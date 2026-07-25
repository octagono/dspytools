"""dspytools tool — Manage DSPy tools (MCP, built-in, langchain)."""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

from dspytools.cli.output import console, error, info, ok, panel, table, warn
from dspytools.cli.rich_config import LLMCommand, LLMGroup, click
from dspytools.core._io import read_json


@click.group(name="tool", cls=LLMGroup)
def tool_cmd():
    """Manage DSPy tools (MCP servers, built-in tools)."""


@tool_cmd.command(name="list", cls=LLMCommand)
@click.option("--mcp-config", "-c", default=".mcp.json", help="MCP config file")
def tool_list(mcp_config: str):
    """List all available MCP tools from config."""
    cfg_path = Path(mcp_config)
    if not cfg_path.exists():
        warn("No .mcp.json found")
        info("Create one with MCP server configuration")
        return

    config = read_json(cfg_path)
    servers = config.get("mcpServers", {})

    if not servers:
        warn("No MCP servers configured in .mcp.json")
        return

    # Load tools from each server
    from dspytools.mcp.loader import MCPSessionPool

    pool = MCPSessionPool()
    all_tools = pool.get_tools(mcp_config)

    if not all_tools:
        warn("Could not connect to any MCP servers")
        return

    # Group tools by server
    console.print(f"\n[bold cyan]Available MCP Tools[/] ({len(all_tools)} total)")

    # Since we can't map tools back to servers easily, show flat list
    rows = []
    for t in all_tools:
        desc = (t.desc or "")[:60]
        rows.append([t.name, desc])

    table("MCP Tools", ["Name", "Description"], rows)

    info("Tip: use --tools 'server:tool_name' when running agents")


@tool_cmd.command(name="show", cls=LLMCommand)
@click.argument("tool_name")
@click.option("--mcp-config", "-c", default=".mcp.json", help="MCP config file")
def tool_show(tool_name: str, mcp_config: str):
    """Show details and signature of a specific tool."""
    from dspytools.mcp.tools import BUILTIN_TOOLS

    # Check built-in tools first
    if tool_name in BUILTIN_TOOLS:
        spec = BUILTIN_TOOLS[tool_name]
        panel(
            f"Tool: {tool_name}",
            f"[bold]Name:[/] {tool_name}\n"
            f"[bold]Description:[/] {spec.get('description', 'N/A')}\n"
            f"[bold]Args:[/] {json.dumps(spec.get('args', {}), indent=2)}\n"
            f"[bold]Type:[/] builtin",
            border_style="cyan",
        )
        return

    # Check MCP tools
    from dspytools.mcp.loader import MCPSessionPool

    pool = MCPSessionPool()
    all_tools = pool.get_tools(mcp_config)

    for t in all_tools:
        if t.name == tool_name:
            panel(
                f"Tool: {t.name}",
                f"[bold]Name:[/] {t.name}\n"
                f"[bold]Description:[/] {t.desc or 'N/A'}\n"
                f"[bold]Args:[/] {t.args if hasattr(t, 'args') else 'auto-inferred'}",
                border_style="cyan",
            )
            return

    error(f"Tool '{tool_name}' not found (checked built-in + MCP servers)")


@tool_cmd.command(name="from-mcp", cls=LLMCommand)
@click.argument("server_name")
@click.argument("tool_name")
@click.option("--mcp-config", "-c", default=".mcp.json", help="MCP config file")
@click.option("--output", "-o", help="Save tool to file")
def tool_from_mcp(
    server_name: str, tool_name: str, mcp_config: str, output: str | None
):
    """Create a DSPy Tool from an MCP tool and optionally save it.

    SERVER_NAME: MCP server name from .mcp.json
    TOOL_NAME: Tool name to convert
    """

    from dspytools.mcp.loader import MCPSessionPool

    pool = MCPSessionPool()
    all_tools = pool.get_tools(mcp_config)

    for t in all_tools:
        if t.name == tool_name:
            ok(f"Loaded DSPy Tool '{tool_name}' from MCP server '{server_name}'")
            panel(
                "Tool Info",
                f"[bold]Name:[/] {t.name}\n"
                f"[bold]Description:[/] {t.desc or 'N/A'}\n"
                f"[bold]Type:[/] {type(t).__name__}",
                border_style="green",
            )
            return

    error(f"Tool '{tool_name}' not found in MCP server '{server_name}'")
    info("Use `dspytools tool list` to see available tools")


@tool_cmd.command(name="inspect", cls=LLMCommand)
@click.argument("name")
@click.option("--mcp-config", "-c", default=".mcp.json", help="MCP config file")
def tool_inspect(name: str, mcp_config: str):
    """Inspect a tool's input/output schema."""
    from dspytools.mcp.tools import BUILTIN_TOOLS

    # Check built-in tools first
    if name in BUILTIN_TOOLS:
        spec = BUILTIN_TOOLS[name]
        args_str = json.dumps(spec.get("args", {}), indent=2)
        panel(
            f"Tool: {name}",
            f"[bold]Type:[/] builtin\n"
            f"[bold]Description:[/] {spec.get('description', 'N/A')}\n"
            f"[bold]Args:[/] {args_str}",
            border_style="cyan",
        )
        return

    # Check MCP tools
    from dspytools.mcp.loader import MCPSessionPool

    pool = MCPSessionPool()
    all_tools = pool.get_tools(mcp_config)

    for t in all_tools:
        if t.name == name:
            args_str = (
                json.dumps(t.args, indent=2)
                if hasattr(t, "args") and isinstance(t.args, dict)
                else str(getattr(t, "args", "auto-inferred"))
            )
            panel(
                f"Tool: {name}",
                f"[bold]Type:[/] {type(t).__name__}\n[bold]Args:[/] {args_str}",
                border_style="cyan",
            )
            return

    error(f"Tool '{name}' not found (checked built-in + MCP servers)")


@tool_cmd.command(name="python-interpreter", cls=LLMCommand)
@click.argument("code")
def tool_python_interpreter(code: str):
    """Execute Python code in DSPy sandbox (Deno/Pyodide)."""
    from dspytools.core._dspy import dspy

    interpreter = dspy.PythonInterpreter()
    result = interpreter.execute(code)
    console.print(f"[bold]Output:[/]\n{result}")


@tool_cmd.command(name="history", cls=LLMCommand)
@click.argument("n", type=int, default=5)
def tool_history(n: int):
    """Show DSPy History as structured messages."""
    from dspytools.core._dspy import dspy

    old = sys.stdout
    buf = io.StringIO()
    sys.stdout = buf
    dspy.inspect_history(n=n)
    sys.stdout = old
    output = buf.getvalue()
    console.print(f"[bold]Last {n} LM calls:[/]\n{output[:2000]}")
