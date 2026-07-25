# DOX — MCP Directory

## Purpose

Model Context Protocol (MCP) server and tools layer that exposes all dspytools features for agent interoperability. Agents (OpenCode, Codex, Claude Desktop, web-based) connect via MCP to manage compiled programs, llama-cpp-server LoRA models, skills, GFL pipelines, evaluation, and DSPy inference.

## Ownership

Owns the MCP transport layer — both server (accepting inbound agent connections) and client (connecting to external MCP tool servers like git-mcp). Four source files:

| File | Role |
|------|------|
| `server.py` | Unified MCP server — exposes all dspytools features (built-in tools) over stdio or SSE transport. Uses `list_tools`, `call_tool`, `list_resources`, `read_resource` handlers. |
| `tools.py` | 65 MCP tool handlers with response caching (5s TTL), merged into BUILTIN_TOOLS from 4 sub-dicts: initial (7), _EXTRA_TOOLS (29), _GRAPH_TOOLS (21), _FINAL_TOOLS (8). Covers: programs, signatures, modules, compiled runs, optimizers, skills, self-status, inspect history, evaluate, GFL synthesize, agent, mlflow_status, drift, sandbox, compile, archive search, validate_deploy, challenger_solver, meta_prompt_learn, opsd_purify, cache, LoRA, graph, memory. |
| `loader.py` | `MCPSessionPool` singleton — manages MCP client sessions to external servers (e.g. git-mcp). Connection reuse with 5-min keepalive, config-driven from `.mcp.json`. Converts external MCP tools to `dspy.Tool` via `dspy.Tool.from_mcp_tool()`. |

## Local Contracts

- **Unified exposure**: `server.py` exposes `BUILTIN_TOOLS` from `tools.py` as the single entry point for agent interoperability.
- **Transport duality**: stdio transport for local agents (Claude Desktop, OpenCode); SSE transport for web-based agents. Both activated via `dspytools mcp serve --transport`.
- **MCPSessionPool singleton**: Only one pool instance exists. Sessions are reused for up to 5 minutes before forced reconnection. `refresh()` clears the pool for next `get_tools()` call. Cache invalidation on mutation (e.g. `tool_swap_program` clears the list programs cache).
- **Response caching**: `tools.py` caches tool responses in-memory with 5-second TTL (`_CACHE_TTL`). Mutating tools (swap, etc.) invalidate relevant cache keys.
- **Tool conversion**: External MCP tools are converted to `dspy.Tool` via `dspy.Tool.from_mcp_tool(session, mcp_tool)`. The `tools.py` built-in handlers are plain synchronous functions wrapped in the `BUILTIN_TOOLS` dict with descriptions and input schemas.
- **Shared `_error()` helper**: `tools.py` defines `_error(exception, detail=None)` producing standardized `{"error": ..., "detail": ...}` JSON responses. All error returns in `tools.py` and `server.py` use this helper — never raw `json.dumps({"error": ...})`. `server.py` import it from `tools.py`.
- **Resource URIs**: The unified server registers `dspytools://` resource URIs for programs, config, llama-cpp-server profiles/status, MLflow, skills, evolve, GFL status, sandbox, and optimizers.
- **Resource handler return type**: `@server.read_resource()` handler must return `Iterable[ReadResourceContents]` (list of `ReadResourceContents(content=..., mime_type=...)` dataclass instances from `mcp.server.lowlevel.helper_types`). This is the low-level SDK's expected format — returning `ReadResourceResult` or plain dicts causes client-side `TypeError`.

## Work Guidance

- When adding a new CLI command group that should be agent-accessible, register a corresponding tool handler in `tools.py` and add it to `BUILTIN_TOOLS` or `_EXTRA_TOOLS` / `_FINAL_TOOLS`.
- When adding a new transport or modifying the MCP server initialization in `server.py`, keep `run_stdio()` and `run_sse()` signatures compatible with the CLI entry point in `commands/mcp.py`.
- Response caching TTL in `tools.py` is a simple time-based dict. For long-running servers, consider migrating to a proper cache with LRU eviction. Cache keys must be invalidated on any mutating tool call.
- `MCPSessionPool.get_tools()` is synchronous but internally runs an async event loop. Avoid calling from within an already-running async context without careful handling.
## Verification

- **Comprehensive MCP test suite**: tests covering all built-in tools, resources, and prompts. Runs with real Qwen student LM via stdio transport. Passes at 189/189.
- Manual smoke test: `dspytools mcp serve --transport stdio` then connect with an MCP client (Claude Desktop, MCP inspector).
- - Key assertions to re-verify after edits:
  - `list_tools` returns exactly 65 tools with correct names and descriptions
  - `list_resources` returns all 8 static resource patterns + per-program entries
  - `read_resource` for any URI returns `ReadResourceContents` (not `ReadResourceResult` or dict) — verified by client-side deserialization
  - `get_prompt` returns `content` as `{"type": "text", "text": "..."}` dict, not a raw string
  - All 3 prompts return valid `PromptMessage` objects with `TextContent` content

## Child DOX Index

No child directories. All MCP source files are flat in this directory.
