"""Unified MCP server for dspytools — all tools in one place.

Exposes:
  - Program management: list_programs, swap_program, infer, ...
  - Compilation: compile_optimizer, compile_cost, holdout_status, ...
  - Evaluation: evaluate, drift_status, validate_deploy, ...

Run:
    dspytools mcp serve --transport stdio
    dspytools mcp serve --transport sse --port 8002
"""

from __future__ import annotations

import json
from typing import Any

import anyio
import uvicorn
from mcp.server import Server
from mcp.server.lowlevel.helper_types import ReadResourceContents
from mcp.server.sse import SseServerTransport
from mcp.server.stdio import stdio_server
from mcp.types import TextContent
from mcp.types import Tool as MCPTool
from starlette.applications import Starlette
from starlette.routing import Mount, Route

from dspytools.commands.compile import _OPTIMIZER_SPECS
from dspytools.config.settings import load_config
from dspytools.core.logging_config import get_logger
from dspytools.core.mlflow_tracker import MLflowAsyncTracker, get_tracker
from dspytools.core.registry import get_run, list_compiled_runs
from dspytools.evolve import SelfEvolve
from dspytools.generate.module import get_sandbox_pool
from dspytools.gfl.pipeline import GFLPipeline
from dspytools.mcp.tools import BUILTIN_TOOLS
from dspytools.skills import SkillManager

_ALL_TOOLS = BUILTIN_TOOLS

_log = get_logger(__name__)


async def _route_tool(name: str, arguments: dict) -> str:
    """Route a tool call to the right handler."""
    if name in BUILTIN_TOOLS:
        handler = BUILTIN_TOOLS[name]["handler"]
        return handler(**arguments)
    raise ValueError(f"Unknown tool: {name}")


def create_mcp_server() -> Any:
    """Unified MCP server: dspytools programs + compilation management."""

    server = Server("dspytools")

    # Add prompts capability
    @server.list_prompts()
    async def handle_list_prompts():
        return [
            {
                "name": "compile",
                "description": "Compile a DSPy program",
                "arguments": [],
            },
            {
                "name": "gfl",
                "description": "Run GFL 4-way optimizer comparison",
                "arguments": [],
            },
            {
                "name": "validate",
                "description": "Validate a program before deployment",
                "arguments": [],
            },
        ]

    @server.get_prompt()
    async def handle_get_prompt(name: str, arguments: dict | None):
        if name == "compile":
            return {
                "messages": [
                    {
                        "role": "user",
                        "content": {
                            "type": "text",
                            "text": "I need to compile a DSPy program. First, use list_optimizers to see available optimizers. Then use compile_optimizer with the chosen optimizer, module name, and a trainset path. After compiling, use list_compiled_runs to verify the result.",
                        },
                    }
                ]
            }
        elif name == "gfl":
            return {
                "messages": [
                    {
                        "role": "user",
                        "content": {
                            "type": "text",
                            "text": "I need to run the GFL pipeline. Use gfl_run_halving to run a 4-way optimizer comparison with Successive Halving early pruning. This evaluates all optimizers on 10% of data, prunes the worst 50%, and runs survivors on the full dataset.",
                        },
                    }
                ]
            }
        elif name == "validate":
            return {
                "messages": [
                    {
                        "role": "user",
                        "content": {
                            "type": "text",
                            "text": "I need to validate a compiled program before deployment. Use validate_deploy with the program ID. This uses SPRT (Sequential Probability Ratio Test) for early stopping on clear wins/losses, saving API tokens.",
                        },
                    }
                ]
            }
        return {
            "messages": [
                {"role": "user", "content": {"type": "text", "text": "Unknown prompt"}}
            ]
        }

    @server.list_tools()
    async def handle_list_tools() -> list[MCPTool]:
        tools = []
        for name, spec in _ALL_TOOLS.items():
            tool_kwargs = {
                "name": name,
                "description": spec["description"],
                "inputSchema": spec.get(
                    "inputSchema", {"type": "object", "properties": {}}
                ),
            }
            # Include annotations if MCP SDK supports them (silently skip if not)
            ann = spec.get("annotations")
            if ann is not None:
                try:
                    tool_kwargs["annotations"] = ann
                except TypeError:
                    pass  # MCP SDK version doesn't support annotations kwarg
            tools.append(MCPTool(**tool_kwargs))
        return tools

    @server.call_tool()
    async def handle_call_tool(
        name: str, arguments: dict[str, Any] | None
    ) -> list[TextContent]:
        try:
            result = await _route_tool(name, arguments or {})
            return [TextContent(type="text", text=result)]
        except ValueError as e:
            msg = str(e)
            hint = ""
            if "Unknown tool" in msg:
                hint = "Use list_* tools to see available tools."
            return [
                TextContent(
                    type="text",
                    text=json.dumps(
                        {
                            "error": "tool_not_found"
                            if "Unknown tool" in msg
                            else "invalid_request",
                            "message": msg,
                            "hint": hint,
                            "available_tools": list(_ALL_TOOLS.keys())[:10],
                        }
                    ),
                )
            ]
        except (RuntimeError, OSError, ValueError, TypeError) as e:
            _log.error("mcp_tool_dispatch_failed", tool=name, error=str(e))
            return [
                TextContent(
                    type="text",
                    text=json.dumps(
                        {
                            "error": "execution_failed",
                            "message": str(e),
                            "hint": "Check input parameters and try again. Use inspect_history to debug LM calls.",
                        }
                    ),
                )
            ]
        except Exception as e:
            _log.error(
                "MCP tool unexpected error: tool=%s error=%s", name, e, exc_info=True
            )
            return [
                TextContent(
                    type="text",
                    text=json.dumps(
                        {
                            "error": "internal_error",
                            "message": str(e),
                        }
                    ),
                )
            ]

    @server.list_resources()
    async def handle_list_resources():

        runs = list_compiled_runs()
        resources = []
        for run in runs:
            rid = run["id"]
            resources.append(
                {
                    "uri": f"dspytools://programs/{rid}",
                    "name": f"Program: {rid}",
                    "mimeType": "application/json",
                }
            )
        resources.extend(
            [
                {
                    "uri": "dspytools://programs",
                    "name": "All compiled programs",
                    "mimeType": "application/json",
                },
                {
                    "uri": "dspytools://config",
                    "name": "DSPyTools configuration",
                    "mimeType": "application/json",
                },
                {
                    "uri": "dspytools://mlflow",
                    "name": "MLflow tracking status",
                    "mimeType": "application/json",
                },
                {
                    "uri": "dspytools://skills",
                    "name": "All skills in library",
                    "mimeType": "application/json",
                },
                {
                    "uri": "dspytools://evolve",
                    "name": "Self-evolve engine state",
                    "mimeType": "application/json",
                },
                {
                    "uri": "dspytools://gfl/status",
                    "name": "GFL pipeline status",
                    "mimeType": "application/json",
                },
                {
                    "uri": "dspytools://sandbox",
                    "name": "Sandbox worker pool status",
                    "mimeType": "application/json",
                },
                {
                    "uri": "dspytools://optimizers",
                    "name": "Available DSPy optimizers",
                    "mimeType": "application/json",
                },
            ]
        )
        return resources

    @server.read_resource()
    async def handle_read_resource(uri: str):

        uri_str = str(uri)  # MCP SDK may pass AnyUrl object, ensure string

        if uri_str == "dspytools://programs":
            data = json.dumps(list_compiled_runs(), indent=2)
        elif uri_str.startswith("dspytools://programs/"):
            rid = uri_str.split("/")[-1]
            meta = get_run(rid)
            data = json.dumps(meta or {"error": "not found"}, indent=2)
        elif uri_str == "dspytools://config":
            data = json.dumps(load_config(), indent=2, default=str)
        elif uri_str == "dspytools://mlflow":
            tracker = get_tracker()
            tracker._ensure_initialized()
            info = {
                "tracking_uri": tracker.tracking_uri,
                "experiment": tracker.experiment_name,
            }
            if isinstance(tracker, MLflowAsyncTracker):
                info["async"] = tracker.stats
            data = json.dumps(info, indent=2)
        elif uri_str == "dspytools://skills":
            mgr = SkillManager()
            skills = [
                {"name": s.name, "description": s.description}
                for s in mgr.list_skills()
            ]
            data = json.dumps(skills, indent=2)
        elif uri_str == "dspytools://evolve":
            try:
                evolve = SelfEvolve()
                data = json.dumps(evolve.status, indent=2)
            except (OSError, RuntimeError, ValueError) as e:
                _log.error("evolve_resource_failed", error=str(e))
                data = json.dumps(
                    {"error": "evolve status unavailable", "detail": str(e)}, indent=2
                )
        elif uri_str == "dspytools://gfl/status":
            pipeline = GFLPipeline()
            data = json.dumps(
                {
                    "tracker": {
                        "baseline": pipeline.tracker.baseline,
                        "trend": pipeline.tracker.improvement_trend,
                        "total_improvement": pipeline.tracker.total_improvement,
                    },
                    "budget": str(pipeline.budget.summary),
                },
                indent=2,
            )
        elif uri_str == "dspytools://sandbox":
            pool = get_sandbox_pool()
            data = json.dumps(pool.stats, indent=2)
        elif uri_str == "dspytools://optimizers":
            data = json.dumps(list(_OPTIMIZER_SPECS.keys()), indent=2)
        else:
            raise ValueError(f"Unknown resource: {uri_str}")
        # Return Iterable[ReadResourceContents] — what the low-level SDK expects
        return [
            ReadResourceContents(content=data, mime_type="application/json"),
        ]

    return server


def run_stdio() -> None:

    server = create_mcp_server()

    async def _run():
        async with stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                server.create_initialization_options(),
            )

    anyio.run(_run)


def run_sse(host: str = "0.0.0.0", port: int = 8002) -> None:

    server = create_mcp_server()
    sse = SseServerTransport("/mcp/messages")

    async def handle_sse(request):
        async with sse.connect_sse(request.scope, request.receive, request._send) as (
            read,
            write,
        ):
            await server.run(
                read,
                write,
                server.create_initialization_options(),
            )

    app = Starlette(
        routes=[
            Route("/mcp/sse", endpoint=handle_sse),
            Mount("/mcp/messages", app=sse.handle_post_message),
        ]
    )
    uvicorn.run(app, host=host, port=port)
