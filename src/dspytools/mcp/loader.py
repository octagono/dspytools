"""MCP tool loader — reads .mcp.json and converts tools to DSPy tools.

Architecture
────────────
MCPSessionPool keeps MCP sessions alive on a dedicated event loop thread.
This is necessary because:

1. MCP sessions are event-loop-bound (stdio pipes register on creation loop).
2. DSPy ReAct calls tools synchronously.
3. DSPy's `_run_async_in_sync` uses `asyncio.run()` which creates a *new*
   event loop each time — but the MCP pipes only respond on the original loop.

Solution: a background `_MCPEventLoop` thread keeps the original loop alive.
All MCP connections and tool calls dispatch on that same loop via
`asyncio.run_coroutine_threadsafe()`. Sync wrapper `dspy.Tool` objects
hide the async machinery from DSPy.
"""

from __future__ import annotations

import asyncio
import os
import sys
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from dspytools.core._io import read_json
from dspytools.core.logging_config import get_logger

if TYPE_CHECKING:
    import dspy
else:
    from dspytools.core._dspy import dspy

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

_log = get_logger(__name__)


# ── Dedicated event loop thread ───────────────────────────────────────────


class _MCPEventLoop(threading.Thread):
    """Dedicated event loop thread for all MCP operations.

    Keeps MCP sessions alive for the lifetime of the process so synchronous
    callers (DSPy ReAct) can dispatch async tool calls via run_coro().
    """

    def __init__(self) -> None:
        super().__init__(daemon=True, name="mcp-event-loop")
        self._loop: asyncio.AbstractEventLoop | None = None
        self._ready = threading.Event()

    def run(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._ready.set()
        self._loop.run_forever()

    def run_coro(self, coro) -> Any:
        """Run a coroutine on the dedicated loop (thread-safe, blocking)."""
        self._ready.wait()
        assert self._loop is not None
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result()

    @property
    def loop(self) -> asyncio.AbstractEventLoop:
        self._ready.wait()
        assert self._loop is not None
        return self._loop

    def stop(self) -> None:
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)


# ── Session pool ──────────────────────────────────────────────────────────


class MCPSessionPool:
    """Singleton MCP session pool.

    All MCP connections live on a dedicated background event loop thread so
    that sync callers (DSPy ReAct) can dispatch tool calls without
    ClosedResourceError from mismatched event loops.
    """

    _instance: MCPSessionPool | None = None
    _loop: _MCPEventLoop | None = None
    _tools: list[dspy.Tool] = []
    _sessions: list[tuple[str, Any, Any]] = []  # (name, session, list_tools_result)
    _cms: list[Any] = []  # kept-alive context managers (stdio_client)
    _last_connect: float = 0

    def __new__(cls) -> MCPSessionPool:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    # ── Lifecycle ──────────────────────────────────────────────────────

    @classmethod
    def _get_loop(cls) -> _MCPEventLoop:
        if cls._loop is None or not cls._loop.is_alive():
            cls._loop = _MCPEventLoop()
            cls._loop.start()
        return cls._loop

    @classmethod
    def get_tools(
        cls, config_path: str = ".mcp.json", force_reconnect: bool = False
    ) -> list[dspy.Tool]:
        """Return cached tools, reconnecting only if forced or config changed."""
        now = time.time()
        if force_reconnect or not cls._tools or (now - cls._last_connect > 300):
            cls._disconnect_all()
            cls._tools = cls._connect_all(config_path)
            cls._last_connect = now
        return cls._tools

    @classmethod
    def _connect_all(cls, config_path: str) -> list[dspy.Tool]:
        """Connect to all MCP servers and create sync-wrapped DSPy tools."""
        cfg_path = Path(config_path)
        if not cfg_path.exists():
            return []
        config = read_json(cfg_path)
        servers = config.get("mcpServers", {})
        if not servers:
            return []

        loop = cls._get_loop()

        async def _connect() -> list[dspy.Tool]:

            all_tools: list[dspy.Tool] = []

            for name, server_cfg in servers.items():
                server_params = StdioServerParameters(
                    command=server_cfg["command"],
                    args=server_cfg.get("args", []),
                    env={**os.environ, **(server_cfg.get("env", {}))},
                )
                cm = stdio_client(server_params)
                try:
                    read, write = await cm.__aenter__()
                    session = ClientSession(read, write)
                    await session.__aenter__()
                    await session.initialize()
                    result = await session.list_tools()

                    # Create sync-wrapper tools that dispatch on OUR loop
                    for mcp_tool in result.tools:
                        tool = _make_sync_tool(
                            name=mcp_tool.name,
                            description=mcp_tool.description or "",
                            input_schema=mcp_tool.inputSchema or {},
                            session=session,
                        )
                        all_tools.append(tool)

                    cls._sessions.append((name, session, result))
                    cls._cms.append(cm)

                except (ConnectionError, OSError, TimeoutError, RuntimeError) as e:
                    _log.warning(
                        "MCP [%s]: connection failed (%s)", name, type(e).__name__
                    )

            return all_tools

        with _suppress_stderr():
            return loop.run_coro(_connect())

    @classmethod
    def _disconnect_all(cls) -> None:
        async def _cleanup():
            for cm in cls._cms:
                await cm.__aexit__(None, None, None)
            cls._sessions.clear()
            cls._cms.clear()
            cls._tools.clear()

        if cls._loop and cls._loop.is_alive():
            cls._loop.run_coro(_cleanup())

    @classmethod
    def refresh(cls) -> None:
        """Force reconnection on next get_tools() call."""
        cls._last_connect = 0


# ── Sync wrapper tool factory ─────────────────────────────────────────────


def _make_sync_tool(
    name: str,
    description: str,
    input_schema: dict[str, Any],
    session: Any,
) -> dspy.Tool:
    """Create a dspy.Tool that calls an MCP session tool synchronously.

    The returned tool's __call__ dispatches session.call_tool() on the
    dedicated MCP event loop thread, avoiding ClosedResourceError from
    DSPy's asyncio.run() creating a new loop.
    """
    pool = MCPSessionPool()
    loop = pool._get_loop()

    def _call(**kwargs: Any) -> Any:
        """Sync wrapper — runs session.call_tool on the MCP event loop."""
        result = loop.run_coro(
            session.call_tool(name, arguments=kwargs if kwargs else None)
        )
        # Extract text content from MCP result
        if hasattr(result, "content"):
            texts = [c.text for c in result.content if hasattr(c, "text") and c.text]
            return "\n".join(texts) if texts else str(result)
        return str(result)

    # Build DSPy Tool
    args = _json_schema_to_args(input_schema)
    return dspy.Tool(
        _call,
        name=name,
        desc=description,
        args=args,
    )


def _json_schema_to_args(schema: dict[str, Any]) -> dict[str, dict[str, str]]:
    """Convert JSON Schema properties to DSPy Tool's simple args format.

    DSPy's Tool.args expects {name: {type: str}}.
    """
    if not schema:
        return {}
    props = schema.get("properties", schema) if isinstance(schema, dict) else {}
    args: dict[str, dict[str, str]] = {}
    for pname, pinfo in props.items():
        if isinstance(pinfo, dict):
            js_type = pinfo.get("type", "string")
            args[pname] = {"type": js_type}
        else:
            args[pname] = {"type": "string"}
    return args


# ── Utilities ─────────────────────────────────────────────────────────────


class _suppress_stderr:
    """Redirect stderr to /dev/null at the OS file descriptor level.

    Unlike contextlib.redirect_stderr, this works across threads —
    the upstream mcp library prints JSON parse warnings from a background
    stdout_reader thread, which redirect_stderr cannot suppress.
    """

    def __enter__(self):
        self._original = os.dup(sys.stderr.fileno())
        self._devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(self._devnull, sys.stderr.fileno())
        return self

    def __exit__(self, *args):
        os.dup2(self._original, sys.stderr.fileno())
        os.close(self._original)
        os.close(self._devnull)


def load_mcp_tools_sync(config_path: str = ".mcp.json") -> tuple[list, list[dspy.Tool]]:
    """Compatibility wrapper — returns (sessions, tools) like before.

    Uses MCPSessionPool internally for connection reuse.
    """
    pool = MCPSessionPool()
    tools = pool.get_tools(config_path)
    return pool._sessions, tools
