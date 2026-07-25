"""dspytools server — Start, stop, manage the hot-swap server.

Enables/disables the MCP and FastAPI servers, manages PID files,
and provides status information.
"""

from __future__ import annotations

import json as _json
import os
import signal
import sys
import urllib.request
from pathlib import Path

from dspytools.cli.rich_config import LLMCommand, LLMGroup, click
from dspytools.config.settings import data_dir, load_config, save_config


def _pidfile(name: str) -> Path:
    return data_dir() / f"{name}.pid"


def _write_pid(name: str) -> None:
    _pidfile(name).write_text(str(os.getpid()))


def _read_pid(name: str) -> int | None:
    pidfile = _pidfile(name)
    if pidfile.exists():
        try:
            return int(pidfile.read_text().strip())
        except (ValueError, OSError):
            return None
    return None


def _remove_pid(name: str) -> None:
    _pidfile(name).unlink(missing_ok=True)


def _is_running(name: str) -> bool:
    pid = _read_pid(name)
    if pid is None:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        _remove_pid(name)
        return False


@click.group(name="server", cls=LLMGroup)
def server_cmd():
    """Manage the hot-swap servers (MCP + FastAPI)."""


@server_cmd.command(name="enable", cls=LLMCommand)
@click.option("--mcp/--no-mcp", default=True, help="Enable MCP server")
@click.option("--api/--no-api", default=True, help="Enable API server")
def server_enable(mcp: bool, api: bool):
    """Enable servers for auto-start."""
    cfg = load_config()
    cfg.setdefault("server", {})
    if mcp:
        cfg["server"]["mcp"] = {**cfg["server"].get("mcp", {}), "enabled": True}
    if api:
        cfg["server"]["api"] = {**cfg["server"].get("api", {}), "enabled": True}
    save_config(cfg)

    parts = []
    if mcp:
        parts.append("MCP")
    if api:
        parts.append("API")
    click.echo(f"  Enabled: {' + '.join(parts)}")


@server_cmd.command(name="disable", cls=LLMCommand)
@click.option("--mcp/--no-mcp", default=True, help="Disable MCP server")
@click.option("--api/--no-api", default=True, help="Disable API server")
def server_disable(mcp: bool, api: bool):
    """Disable servers from auto-start."""
    cfg = load_config()
    cfg.setdefault("server", {})
    if mcp:
        cfg["server"]["mcp"] = {**cfg["server"].get("mcp", {}), "enabled": False}
    if api:
        cfg["server"]["api"] = {**cfg["server"].get("api", {}), "enabled": False}
    save_config(cfg)

    parts = []
    if mcp:
        parts.append("MCP")
    if api:
        parts.append("API")
    click.echo(f"  Disabled: {' + '.join(parts)}")


@server_cmd.command(name="start", cls=LLMCommand)
@click.option("--mcp/--no-mcp", default=None, help="Start MCP server")
@click.option("--api/--no-api", default=None, help="Start API server")
@click.option("--mcp-port", default=8002, type=int, help="MCP SSE port")
@click.option("--api-port", default=8080, type=int, help="API port")
def server_start(mcp: bool | None, api: bool | None, mcp_port: int, api_port: int):
    """Start the server(s)."""
    cfg = load_config()

    # Determine what to start based on flags or config
    start_mcp = (
        mcp
        if mcp is not None
        else cfg.get("server", {}).get("mcp", {}).get("enabled", False)
    )
    start_api = (
        api
        if api is not None
        else cfg.get("server", {}).get("api", {}).get("enabled", False)
    )

    if not start_mcp and not start_api:
        click.echo("  No servers to start. Enable with `dspytools server enable`")
        click.echo("  Or pass --mcp / --api flags")
        return

    if start_mcp:
        click.echo("  Starting MCP server (stdio)...")
        pid = os.fork()
        if pid == 0:
            _write_pid("mcp")
            # Redirect stdin to /dev/null in the forked child — stdin is owned
            # by the parent terminal and the MCP stdio server must not read it.
            devnull = os.open(os.devnull, os.O_RDWR)
            os.dup2(devnull, 0)
            os.close(devnull)
            from dspytools.mcp.server import run_stdio

            run_stdio()
            sys.exit(0)
        click.echo(f"    PID: {pid}")

    if start_api:
        click.echo(f"  Starting API server on port {api_port}...")
        pid = os.fork()
        if pid == 0:
            _write_pid("api")
            from dspytools.api.server import run_api

            run_api(port=api_port)
            sys.exit(0)
        click.echo(f"    PID: {pid}")

    click.echo("  Servers started in background")


@server_cmd.command(name="stop", cls=LLMCommand)
@click.option("--mcp/--no-mcp", default=True, help="Stop MCP server")
@click.option("--api/--no-api", default=True, help="Stop API server")
def server_stop(mcp: bool, api: bool):
    """Stop running server(s)."""
    for name in ["mcp", "api"]:
        if (name == "mcp" and not mcp) or (name == "api" and not api):
            continue
        pid = _read_pid(name)
        if pid and _is_running(name):
            try:
                os.kill(pid, signal.SIGTERM)
                _remove_pid(name)
                click.echo(f"  Stopped {name.upper()} server (PID {pid})")
            except OSError as e:
                click.echo(f"  Error stopping {name.upper()}: {e}")
        else:
            click.echo(f"  {name.upper()} server not running")


@server_cmd.command(name="restart", cls=LLMCommand)
@click.option("--mcp/--no-mcp", default=True, help="Restart MCP server")
@click.option("--api/--no-api", default=True, help="Restart API server")
def server_restart(mcp: bool, api: bool):
    """Restart server(s)."""
    server_stop.callback(mcp, api)  # type: ignore[attr-defined]
    server_start.callback(mcp, api, 8002, 8080)  # type: ignore[attr-defined]


@server_cmd.command(name="status", cls=LLMCommand)
def server_status():
    """Show server status."""
    cfg = load_config()

    click.echo("  Server Configuration:")
    click.echo(
        f"    MCP enabled: {cfg.get('server', {}).get('mcp', {}).get('enabled', False)}"
    )
    click.echo(
        f"    API enabled: {cfg.get('server', {}).get('api', {}).get('enabled', False)}"
    )

    click.echo("\n  Runtime Status:")
    for name, label in [("mcp", "MCP"), ("api", "API")]:
        running = _is_running(name)
        pid = _read_pid(name)
        status = "running" if running else "stopped"
        pid_str = f" (PID {pid})" if running and pid else ""
        click.echo(f"    {label}: {status}{pid_str}")

    # Show active program if API server is running
    if _is_running("api"):
        resp = urllib.request.urlopen("http://localhost:8080/active", timeout=2)
        data = _json.loads(resp.read())
        click.echo(f"    Active program: {data.get('active', 'none')}")

    # Show loaded programs
    from dspytools.core.hotswap import HotSwapManager

    mgr = HotSwapManager()
    programs = mgr.load_all()
    if programs:
        click.echo(f"\n  Loaded Programs ({len(programs)}):")
        for p in mgr.list():
            marker = "→" if p.get("active") else " "
            click.echo(f"    {marker} {p['id']}")
    else:
        click.echo("\n  No compiled programs loaded")


@server_cmd.command(name="swap", cls=LLMCommand)
@click.argument("program_id")
def server_swap(program_id: str):
    """Hot-swap the active compiled program via local API."""

    req = urllib.request.Request(
        f"http://localhost:8080/swap/{program_id}",
        method="POST",
    )
    resp = urllib.request.urlopen(req, timeout=5)
    data = _json.loads(resp.read())
    if data.get("status") == "ok":
        click.echo(f"  Swapped to: {data['active']}")
        if data.get("previous"):
            click.echo(f"  Previous: {data['previous']}")
    else:
        click.echo(f"  Error: {data.get('message', 'unknown')}")


@server_cmd.command(name="list", cls=LLMCommand)
def server_list():
    """List loaded compiled programs."""
    from dspytools.core.hotswap import HotSwapManager

    mgr = HotSwapManager()
    programs = mgr.load_all()
    if programs:
        click.echo(f"  Loaded {len(programs)} programs:")
        for p in mgr.list():
            marker = "→" if p.get("active") else " "
            click.echo(f"    {marker} {p['id']}")
    else:
        click.echo("  No compiled programs found")
