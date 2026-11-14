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
import keyword
import os
import re
import sys
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self

from dspytools.core._io import read_json
from dspytools.core.logging_config import get_logger

if TYPE_CHECKING:
    import dspy
else:
    from dspytools.core._dspy import dspy

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

_log = get_logger(__name__)

# Bound on any single cross-thread MCP operation. Without it, one hung server
# blocks the sync caller forever (`future.result()` had no timeout).
MCP_CALL_TIMEOUT = 120.0


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

    def run_coro(self, coro, timeout: float = MCP_CALL_TIMEOUT) -> Any:
        """Run a coroutine on the dedicated loop (thread-safe, blocking).

        The wait is bounded: a hung MCP server raises `TimeoutError` to the
        caller instead of blocking it forever.
        """
        self._ready.wait()
        assert self._loop is not None
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        try:
            return future.result(timeout=timeout)
        except TimeoutError:
            future.cancel()
            raise TimeoutError(
                f"MCP operation timed out after {timeout:.0f}s on the MCP event loop"
            ) from None

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

    Sessions are kept for the lifetime of the process: `get_tools()` connects
    once (or after `refresh()` / `force_reconnect=True`) and never tears a
    working session down on a timer. Servers that failed to connect are
    recorded in `failed_servers` so callers can report *why* the tool list is
    empty or partial instead of claiming the tool does not exist.
    """

    _instance: MCPSessionPool | None = None
    _loop: _MCPEventLoop | None = None
    _tools: list[dspy.Tool] = []
    _sessions: list[tuple[str, Any, Any]] = []  # (name, session, list_tools_result)
    _cms: list[Any] = []  # kept-alive context managers (stdio_client)
    _last_connect: float = 0
    _failed: list[dict[str, str]] = []  # {"server": name, "error": reason}
    _connected_servers: list[str] = []

    def __new__(cls) -> Self:
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
        """Return cached tools; (re)connect only when forced or not yet connected.

        There is no keepalive timer: a healthy session is never dropped just
        because time passed (the old 300 s sweep reconmected — and briefly
        emptied — a working pool mid-agent-turn).
        """
        if force_reconnect or not cls._tools:
            cls._disconnect_all()
            cls._tools = cls._connect_all(config_path)
            cls._last_connect = time.time()
        return cls._tools

    @classmethod
    def failed_servers(cls) -> list[dict[str, str]]:
        """Servers whose last connection attempt failed, with the reason."""
        return list(cls._failed)

    @classmethod
    def describe_failures(cls) -> str:
        """One-line, human-readable summary of failed MCP server connections."""
        if not cls._failed:
            return ""
        return "; ".join(f"{f['server']}: {f['error']}" for f in cls._failed)

    @classmethod
    def _connect_all(cls, config_path: str) -> list[dspy.Tool]:
        """Connect to all MCP servers and create sync-wrapped DSPy tools."""
        cfg_path = Path(config_path)
        cls._failed = []
        cls._connected_servers = []
        if not cfg_path.exists():
            return []
        config = read_json(cfg_path)
        servers = config.get("mcpServers", {})
        if not servers:
            return []

        loop = cls._get_loop()

        async def _connect() -> list[dspy.Tool]:
            all_tools: list[dspy.Tool] = []
            taken_names: set[str] = set()

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
                            taken_names=taken_names,
                        )
                        all_tools.append(tool)

                    cls._sessions.append((name, session, result))
                    cls._cms.append(cm)
                    cls._connected_servers.append(name)

                except (ConnectionError, OSError, TimeoutError, RuntimeError) as e:
                    reason = f"{type(e).__name__}: {e}"
                    cls._failed.append({"server": name, "error": reason})
                    _log.error("MCP server [%s] failed to connect (%s)", name, reason)

            if cls._failed:
                _log.error(
                    "MCP servers unavailable: %s (connected: %s)",
                    cls.describe_failures(),
                    ", ".join(cls._connected_servers) or "none",
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
            cls._connected_servers.clear()

        if cls._loop and cls._loop.is_alive():
            cls._loop.run_coro(_cleanup())
        else:
            cls._tools.clear()

    @classmethod
    def refresh(cls) -> None:
        """Drop the cached tools so the next `get_tools()` reconnects.

        This is what a forced reconnect means: the cached tool list is cleared
        here, and `get_tools()` re-runs `_connect_all()`. (It used to only reset
        a timestamp, which the timer has since stopped reading.)
        """
        cls._last_connect = 0
        cls._tools = []


# ── Sync wrapper tool factory ─────────────────────────────────────────────


class MCPToolError(RuntimeError):
    """An external MCP tool call failed (`result.is_error`).

    Raised instead of returning the error text as a normal result, so the
    agent cannot proceed as if the call succeeded.
    """

    def __init__(self, tool_name: str, message: str) -> None:
        super().__init__(f"MCP tool '{tool_name}' failed: {message}")
        self.tool_name = tool_name
        self.message = message


def _result_text(result: Any) -> str:
    """Join the text parts of an MCP call result (media parts are dropped)."""
    if hasattr(result, "content"):
        texts = [c.text for c in result.content if hasattr(c, "text") and c.text]
        return "\n".join(texts)
    return ""


def _make_sync_tool(
    name: str,
    description: str,
    input_schema: dict[str, Any],
    session: Any,
    taken_names: set[str] | None = None,
) -> dspy.Tool:
    """Create a dspy.Tool that calls an MCP session tool synchronously.

    The returned tool's __call__ dispatches session.call_tool() on the
    dedicated MCP event loop thread, avoiding ClosedResourceError from
    DSPy's asyncio.run() creating a new loop.

    The advertised name is sanitized to a valid Python identifier (see
    `_sanitize_tool_name`); the original MCP name is preserved in the tool
    description and is the name actually sent over the wire.
    """
    pool = MCPSessionPool()
    loop = pool._get_loop()

    def _call(**kwargs: Any) -> Any:
        """Sync wrapper — runs session.call_tool on the MCP event loop.

        Text-only by design (image/file parts are dropped; see AGENTS.md).
        A failed external tool is **not** silently turned into plausible text:
        `result.is_error` raises, mirroring DSPy's own `convert_mcp_tool`.
        """
        result = loop.run_coro(
            session.call_tool(name, arguments=kwargs if kwargs else None)
        )
        text = _result_text(result)
        if getattr(result, "is_error", False) or getattr(result, "isError", False):
            raise MCPToolError(name, text or "MCP tool reported an error")
        return text if text else str(result)

    # Build DSPy Tool — DSPy's own converter preserves descriptions and the
    # JSON-schema fragments (nested objects, anyOf, ...) that a hand-rolled
    # {name: {type}} mapping would drop.
    args, arg_types, arg_desc = _schema_to_tool_args(input_schema)
    tool_name = _sanitize_tool_name(name, taken_names)
    if tool_name != name:
        description = f"{description}\n\nMCP tool name: {name}".strip()
    return dspy.Tool(
        _call,
        name=tool_name,
        desc=description,
        args=args,
        arg_types=arg_types,
        arg_desc=arg_desc,
    )


def _schema_to_tool_args(
    schema: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
    """Convert an MCP input JSON Schema into DSPy Tool ``args``/``arg_types``/``arg_desc``.

    DSPy's `convert_input_schema_to_tool_args` keeps each property's schema
    (description, default, nested pieces) and types, but it does not encode
    optionality: `dspy.Tool.as_function_tool()` derives JSON-schema
    ``required`` from the *absence* of a ``"default"`` key on the arg. So every
    property the server did not list in ``required`` gets a ``default`` key
    (the schema's own default when present, else ``None``) — otherwise an
    optional MCP argument is advertised to the model as mandatory.
    """
    from dspy.adapters.types.tool import convert_input_schema_to_tool_args

    schema = schema or {}
    args, arg_types, arg_desc = convert_input_schema_to_tool_args(schema)
    required = schema.get("required") or []
    for pname, prop in list(args.items()):
        if pname in required or not isinstance(prop, dict):
            continue
        if "default" not in prop:
            args[pname] = {**prop, "default": None}
    return args, arg_types, arg_desc


def _sanitize_tool_name(raw_name: str, taken_names: set[str] | None = None) -> str:
    """Return a collision-free, valid Python identifier for an MCP tool name.

    MCP servers expose names that are not valid identifiers (``git-mcp_list_files``).
    ``dspy.ReAct`` tolerates those, but ``dspy.RLM``/``dspy.Flex`` raise
    `ValueError: Invalid tool name ...` for anything that fails
    ``str.isidentifier()`` or is a keyword. Deterministic rules:

    1. non-alphanumeric/underscore characters → ``_``
    2. empty result → ``"tool"``
    3. leading digit → prefix ``_``
    4. Python keyword → suffix ``_``
    5. collision with an already-taken name → suffix ``_2``, ``_3``, ...

    ``taken_names`` is updated in place. The original name stays visible to the
    model through the tool description and is what `_call` sends to the server.
    """
    taken = taken_names if taken_names is not None else set()
    ident = re.sub(r"[^0-9a-zA-Z_]", "_", raw_name)
    if not ident:
        ident = "tool"
    if ident[0].isdigit():
        ident = "_" + ident
    if keyword.iskeyword(ident):
        ident = ident + "_"

    base = ident
    suffix = 2
    while ident in taken:
        ident = f"{base}_{suffix}"
        suffix += 1
    taken.add(ident)
    return ident


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

    Uses MCPSessionPool internally for connection reuse. When the pool comes
    back empty or partial, `MCPSessionPool.failed_servers()` carries the
    per-server reasons (also logged at ERROR) — "connected to nothing" is never
    silent.
    """
    pool = MCPSessionPool()
    tools = pool.get_tools(config_path)
    failures = pool.failed_servers()
    if failures:
        _log.error(
            "mcp_partial_connection: %d/%d server(s) failed (%s)",
            len(failures),
            len(failures) + len(pool._connected_servers),
            pool.describe_failures(),
        )
    return pool._sessions, tools
