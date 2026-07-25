"""dspytools mcp — Manage MCP servers and tools."""

from dspytools.cli.rich_config import LLMCommand, LLMGroup, click
from dspytools.config.settings import load_config


@click.group(name="mcp", cls=LLMGroup)
def mcp_cmd():
    """Manage MCP servers and tools."""


@mcp_cmd.command(name="tools", cls=LLMCommand)
@click.option("--config", "-c", "cfg_path", default=".mcp.json", help="MCP config file")
def mcp_tools(cfg_path: str):
    """List available MCP tools from config."""
    from dspytools.mcp.loader import load_mcp_tools_sync

    click.echo(f"  Loading MCP tools from {cfg_path}...")
    sessions, tools = load_mcp_tools_sync(cfg_path)

    if tools:
        click.echo(f"\n  Loaded {len(tools)} tools:")
        for t in tools:
            click.echo(f"    • {t.name}")
            if t.desc:
                click.echo(f"      {t.desc[:120]}")
    else:
        click.echo("  No MCP tools loaded")

        # Show config guide
        load_config()
        click.echo("\n  To enable MCP, create .mcp.json:")
        click.echo(
            '    {"mcpServers": {"git": {"command": "uvx", "args": ["git-mcp"]}}}'
        )


@mcp_cmd.command(name="serve", cls=LLMCommand)
@click.option(
    "--transport",
    "-t",
    default="stdio",
    type=click.Choice(["stdio", "sse"]),
    help="Transport protocol",
)
@click.option("--port", "-p", default=8002, type=int, help="Port for SSE transport")
def mcp_serve(transport: str, port: int):
    """Start the dspytools MCP server.

    Other agents (OpenCode, Codex, Claude) can connect to this server
    to list programs, swap programs, and run inference.

    For local agents: --transport stdio
    For remote agents: --transport sse --port 8002
    """
    if transport == "stdio":
        click.echo("  Starting MCP server (stdio)...", err=True)
        from dspytools.mcp.server import run_stdio

        run_stdio()
    else:
        click.echo(f"  Starting MCP server (SSE) on port {port}...", err=True)
        from dspytools.mcp.server import run_sse

        run_sse(port=port)


@mcp_cmd.command(name="config", cls=LLMCommand)
@click.option("--show", is_flag=True, help="Show current MCP config")
def mcp_config(show: bool):
    """Show MCP configuration."""
    from pathlib import Path

    cfg_path = Path(".mcp.json")
    if cfg_path.exists():
        content = cfg_path.read_text()
        click.echo(f"  MCP config ({cfg_path}):")
        click.echo(f"  {content[:500]}")
        if len(content) > 500:
            click.echo(f"  ... ({len(content) - 500} more chars)")
    else:
        click.echo("  No .mcp.json found")
        click.echo("\n  Create one with:")
        click.echo("  {")
        click.echo('    "mcpServers": {')
        click.echo('      "git": {')
        click.echo('        "command": "uvx",')
        click.echo('        "args": ["git-mcp"]')
        click.echo("      }")
        click.echo("    }")
        click.echo("  }")
