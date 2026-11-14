# DOX — MCP Directory

## Purpose

Model Context Protocol (MCP) server and tools layer that exposes all dspytools features for agent interoperability. Agents (OpenCode, Codex, Claude Desktop, web-based) connect via MCP to manage compiled programs, llama-cpp-server LoRA models, skills, GFL pipelines, evaluation, and DSPy inference.

## Ownership

Owns the MCP transport layer — both server (accepting inbound agent connections) and client (connecting to external MCP tool servers like git-mcp). Three source files:

| File | Role |
|------|------|
| `server.py` | Unified MCP server — exposes all dspytools features (built-in tools) over stdio or SSE transport. Uses `list_tools`, `call_tool`, `list_resources`, `read_resource` handlers. |
| `tools.py` | 65 MCP tool handlers with response caching (5s TTL), merged into BUILTIN_TOOLS from a base dict (7) + 3 sub-dicts: _EXTRA_TOOLS (31), _GRAPH_TOOLS (21), _FINAL_TOOLS (6). Covers: programs, signatures, modules, compiled runs, optimizers, skills, self-status, inspect history, evaluate, GFL synthesize, agent, mlflow_status, drift, sandbox, compile, archive search, validate_deploy, challenger_solver, meta_prompt_learn, opsd_purify, cache, LoRA, graph, memory. |
| `loader.py` | `MCPSessionPool` singleton — manages MCP client sessions to external servers (e.g. git-mcp). Connection reuse with 5-min keepalive, config-driven from `.mcp.json`. Converts external MCP tools to `dspy.Tool` via the event-loop-thread-aware `_make_sync_tool()` helper, which uses DSPy's `convert_input_schema_to_tool_args` and sanitizes MCP tool names to valid Python identifiers. |

## Local Contracts

- **Unified exposure**: `server.py` exposes `BUILTIN_TOOLS` from `tools.py` as the single entry point for agent interoperability.
- **Transport duality**: stdio transport for local agents (Claude Desktop, OpenCode); SSE transport for web-based agents. Both activated via `dspytools mcp serve --transport`.
- **MCPSessionPool singleton**: Only one pool instance exists. Sessions are reused for up to 5 minutes before forced reconnection. `refresh()` clears the pool for next `get_tools()` call. Cache invalidation on mutation (e.g. `tool_swap_program` clears the list programs cache).
- **Response caching**: `tools.py` caches tool responses in-memory with 5-second TTL (`_CACHE_TTL`). Mutating tools (swap, etc.) invalidate relevant cache keys.
- **Tool conversion**: External MCP tools are converted to `dspy.Tool` via the event-loop-thread-aware `_make_sync_tool()` helper in `loader.py` — it hand-builds `dspy.Tool(_call, name, desc, args, arg_types, arg_desc)` whose `__call__` dispatches `session.call_tool()` on the pool's dedicated MCP event-loop thread (avoids `ClosedResourceError` from DSPy's `asyncio.run()` creating a new loop). `dspy.Tool.from_mcp_tool()` is deliberately NOT used because it would own its own event loop and reintroduce that error. The `tools.py` built-in handlers are plain synchronous functions wrapped in the `BUILTIN_TOOLS` dict with descriptions and input schemas. **Tool results are text-only**: `_make_sync_tool._call` joins `result.content` text parts and returns a string, dropping image/file parts. Do not upgrade this to `dspy.Image`/`dspy.File` content parts without also moving off the `chat` API family — lm15's chat-completions presets declare `tool_result_media="reject"`, which aborts an agent mid-turn (see root AGENTS.md "Not adopted").
- **Tool schema fidelity**: `_schema_to_tool_args()` delegates to DSPy's `convert_input_schema_to_tool_args` (it returns `(args, arg_types, arg_desc)`) so property descriptions, types, and nested schema fragments survive. Hand-rolled `{name: {type: str}}` mappings are forbidden — they drop descriptions and mark every argument required. `dspy.Tool.as_function_tool()` derives JSON-schema `required` from the *absence* of a `"default"` key on an arg, so `_schema_to_tool_args` adds `default` (the schema's own default, else `None`) to every property the server did not list in `required`. Contract: optional MCP arguments must never be advertised as required.
- **Tool name sanitization**: `_sanitize_tool_name(raw, taken)` maps MCP names to valid, unique Python identifiers — non-alphanumerics → `_`, leading digit → `_` prefix, keyword → `_` suffix, collisions → `_2`/`_3`. `dspy.ReAct` tolerates names like `git-mcp_list_files`; `dspy.RLM`/`dspy.Flex` raise `ValueError: Invalid tool name ...` unless the name is a valid identifier. Sanitize in the loader, never at the `commands/run.py` call site. The original MCP name stays in the tool description (`\n\nMCP tool name: <raw>`) and is what `_call` sends to the server.
- **LM configuration on worker threads**: MCP handlers run off the configuring thread. `dspy.configure` (via `setup_dspy`) raises `RuntimeError` for any thread other than the one that first configured settings, and `core/setup.py`'s documented fallback then writes `dspy.dsp.utils.settings.main_thread_config` directly. Handlers therefore call `_ensure_lm_configured()`: if `dspy.settings.lm` is already set it does nothing; otherwise it calls `setup_dspy()` once under a lock. Per-call isolation belongs to `dspy.context(...)` (e.g. `tool_agent_run` pins its adapter that way). Never call `setup_dspy()` directly from an MCP tool handler.
- **GEPA needs a teacher**: `_make_optimizer_instance("gepa")` uses a five-argument metric (`_gepa_feedback_metric(gold, pred, trace, pred_name, pred_trace)`) and `reflection_lm=LMRegistry.get_teacher()`. GEPA rejects three-argument metrics (`TypeError`) and hard-asserts that a reflection LM exists (`AssertionError`). When no teacher is configured the helper raises an actionable `ValueError` — `tool_lse_explore` records the failed node, `tool_opsd_purify` surfaces the message. Never degrade silently to the student LM.
- **Agent failure payload**: `tool_agent_run` wraps `dspy.ReAct` construction and invocation. A failed forced submit raises `AdapterParseError` or `ContextWindowExceededError` and `issubclass(dspy.AdapterParseError, ValueError)` is **False**, so the handler catches broadly and returns `{"error": "<Type>: <msg>", "exception", "last_model_output", "detail"}` instead of propagating to the MCP transport.
- **Error handling layering**: `tools.py` defines `_error(exception, detail=None)` → JSON string `{"error": ..., "detail": ...}`, used by all tool handlers for in-band (returned-as-text) errors. `server.py` does NOT import `_error` — its `handle_call_tool` dispatcher returns its own structured dispatch errors (`{"error", "message", "hint", "available_tools"}` for unknown tools, `{"error", "message"}` for execution/internal failures). These are two distinct layers: tools report domain errors inside their result; the server reports dispatch/transport failures. Do not conflate them.
- **Resource URIs**: The unified server registers `dspytools://` resource URIs for programs (+ per-program `programs/{rid}`), config, MLflow, skills, evolve, GFL status, sandbox, and optimizers — 8 static + per-program entries. (llama-cpp/LoRA state is exposed via tools, not resources.)
- **Resource handler return type**: `@server.read_resource()` handler must return `Iterable[ReadResourceContents]` (list of `ReadResourceContents(content=..., mime_type=...)` dataclass instances from `mcp.server.lowlevel.helper_types`). This is the low-level SDK's expected format — returning `ReadResourceResult` or plain dicts causes client-side `TypeError`.

## Work Guidance

- When adding a new CLI command group that should be agent-accessible, register a corresponding tool handler in `tools.py` and add it to `BUILTIN_TOOLS` or `_EXTRA_TOOLS` / `_FINAL_TOOLS`.
- When adding a new transport or modifying the MCP server initialization in `server.py`, keep `run_stdio()` and `run_sse()` signatures compatible with the CLI entry point in `commands/mcp.py`.
- Response caching TTL in `tools.py` is a simple time-based dict. For long-running servers, consider migrating to a proper cache with LRU eviction. Cache keys must be invalidated on any mutating tool call.
- `MCPSessionPool.get_tools()` is synchronous but internally runs an async event loop. Avoid calling from within an already-running async context without careful handling.
## Verification

- Functional smoke test: `dspytools mcp serve --transport stdio` then connect with an MCP client (Claude Desktop, MCP inspector, or an in-process `mcp.ClientSession` via `stdio_client`).
- - Key assertions to re-verify after edits:
  - `list_tools` returns exactly 65 tools with correct names and descriptions
  - `list_resources` returns all 8 static resource patterns + per-program entries
  - `read_resource` for any URI returns `ReadResourceContents` (not `ReadResourceResult` or dict) — verified by client-side deserialization
  - `get_prompt` returns `content` as `{"type": "text", "text": "..."}` dict, not a raw string
  - All 3 prompts return valid `PromptMessage` objects with `TextContent` content
  - `call_tool` with an unknown tool returns a graceful `tool_not_found` response (not an unhandled exception)
- Functional check (verified against a live stdio server): 65 tools, 8 static + N per-program resources, 3 prompts, resource reads and unknown-tool handling all work over `mcp.ClientSession`.
- Regression tests: `uv run --no-sync pytest tests/test_mcp_tools.py -q` — schema fidelity (descriptions/types/`required`/defaults), identifier sanitization + collision handling, `dspy.RLM` accepting a sanitized tool, `tool_agent_run` structured error on `AdapterParseError`, GEPA teacher requirement, and the `tool python-interpreter` sandbox shutdown.

## Child DOX Index

No child directories. All MCP source files are flat in this directory.
