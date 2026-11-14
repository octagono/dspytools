"""dspytools run — Run DSPy modules (predict, cot, react, etc.)."""

from __future__ import annotations

import importlib
import inspect
import json
import json as _json
import re as _re
import sys as _sys
import time
from collections.abc import Callable
from contextlib import contextmanager, nullcontext
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import dspy
else:
    from dspytools.core._dspy import dspy

from dspytools.cli.output import (
    fail,
    info,
    ok,
    warn,
)
from dspytools.cli.output import header as section
from dspytools.cli.output import panel as rich_panel
from dspytools.cli.output import spinner as rich_spinner
from dspytools.cli.output import table as rich_table
from dspytools.cli.rich_config import LLMCommand, LLMGroup, click
from dspytools.config.settings import data_dir, embedder_kwargs
from dspytools.core.setup import LMRegistry, setup_dspy

# ── Colour palette constants ────────────────────────────────────────────

_CYAN = "bold cyan"
_GREEN = "bold green"
_YELLOW = "bold yellow"
_DIM = "dim"
_RESULT = "cyan"
_AGENT = "magenta"
_CODE = "green"
_ERROR = "red"


# ── Display helpers ─────────────────────────────────────────────────────


def _show_command_header(cmd: str) -> None:
    name, _ = _MODULE_TYPES.get(cmd, (cmd, ""))
    section(f" {name} ")


def _show_inputs(kwargs: dict[str, str]) -> None:
    if not kwargs:
        return
    parts = "  ".join(f"[{_CYAN}]{k}[/]=[bold]{v}[/]" for k, v in kwargs.items())
    info(f"Inputs: {parts}")


def _show_model_info(lm: str | None = None) -> None:
    model = lm or "default (student Qwen)"
    info(f"Model: [{_DIM}]{model}[/]")


def _show_timing(start: float) -> None:
    elapsed = time.time() - start
    info(f"Time:  [{_DIM}]{elapsed:.2f}s[/]")


# Fields DSPy 3.5 attaches to a Prediction alongside the user's outputs
# (ReAct: history/termination_reason, RLM: history/final_reasoning/trajectory,
# ChainOfThought: reasoning — the whole rationale, rendered separately).
_RESULT_META_FIELDS = frozenset(
    {
        "history",
        "termination_reason",
        "final_reasoning",
        "trajectory",
        "reasoning",
        "rationale",
    }
)

# The reasoning/rationale field is never an answer: it is a str-like
# ``dspy.Reasoning`` carrying the full chain of thought.
_REASONING_FIELDS = ("reasoning", "rationale")
_REASONING_PREVIEW_CHARS = 200


def _result_fields(result: Any) -> list[str]:
    """Return a Prediction's declared output fields (DSPy 3.5 has no
    ``_output_field_names``; fall back to the underlying store).

    Reasoning/rationale and DSPy's bookkeeping fields are excluded: callers
    treat every returned name as an answer to display.
    """
    names = getattr(result, "_output_field_names", None)
    if names:
        return [name for name in names if name not in _RESULT_META_FIELDS]
    store = getattr(result, "_store", None)
    if isinstance(store, dict):
        return [name for name in store if name not in _RESULT_META_FIELDS]
    return []


def _reasoning_preview(
    result: Any, *, limit: int = _REASONING_PREVIEW_CHARS
) -> str | None:
    """Return a one-line, truncated preview of the model's reasoning."""
    for fname in _REASONING_FIELDS:
        raw = getattr(result, fname, None)
        if not raw:
            continue
        text = " ".join(str(raw).split())
        if not text:
            continue
        return text if len(text) <= limit else text[:limit].rstrip() + "…"
    return None


def _show_reasoning(result: Any) -> None:
    """Show the chain of thought as a deliberate, labelled preview."""
    preview = _reasoning_preview(result)
    if preview:
        info(f"[{_YELLOW}]Reasoning:[/] {preview}")


def _show_result(
    result: Any,
    *,
    title: str = "Result",
    border: str = _RESULT,
) -> None:
    """Display a DSPy prediction in a panel with labelled output fields."""
    _show_reasoning(result)
    names = _result_fields(result)
    lines = [
        f"[{_CYAN}]{fname}:[/]  {getattr(result, fname, '')}"
        for fname in names
        if getattr(result, fname, "")
    ]
    content = "\n".join(lines) if lines else str(result)
    rich_panel(title, content, border_style=border)


def _show_answer(result_or_str: Any) -> None:
    """Show a simple answer in a panel."""
    if _result_fields(result_or_str):
        _show_result(result_or_str)
        return
    rich_panel("Answer", str(result_or_str), border_style=_RESULT)


def _show_react_trace(result: Any) -> None:
    """Show a ReAct Prediction's termination reason and turn count.

    DSPy 3.5 returns ``termination_reason`` (``submit`` or ``forced_submit``)
    and the structured ``history`` instead of the old trajectory dict.
    """
    term = getattr(result, "termination_reason", None)
    turns = len(getattr(getattr(result, "history", None), "messages", None) or [])
    parts = []
    if term:
        parts.append(f"Terminated: [{_CYAN}]{term}[/]")
    if turns:
        parts.append(f"Turns: [{_DIM}]{turns}[/]")
    if parts:
        info("  ".join(parts))


# ── Tool loading ────────────────────────────────────────────────────────


def _load_tools(tool_specs: tuple[str, ...]) -> list:
    """Load tools from MCP servers or imported Python functions.

    Accepts:
      - ``server:tool_name`` — load tool from an MCP server
      - ``module.attr`` or ``module:attr`` — import a Python function
        (e.g. ``math.sqrt``, ``json.dumps``) and wrap it as a ``dspy.Tool``
    """

    if not tool_specs:
        return []

    tools: list = []
    for spec in tool_specs:
        if _load_mcp_tool(spec, tools):
            continue
        try:
            if ":" in spec:
                module_path, attr_name = spec.split(":", 1)
            else:
                module_path, attr_name = spec.rsplit(".", 1)
            mod = importlib.import_module(module_path)
            func = getattr(mod, attr_name)
            t = _build_dspy_tool(func, name=attr_name)
            tools.append(t)
            ok(f"Loaded tool '{attr_name}' from {module_path}")
        except (ImportError, AttributeError, ValueError, TypeError):
            if ":" in spec:
                detail = _mcp_failure_detail()
                suffix = f" ({detail})" if detail else ""
                hint = f" in any MCP server{suffix}"
            else:
                hint = ""
            warn(
                f"Tool '{spec}' not found{hint}. Use qualified names "
                f"(e.g. ``math.sqrt``) or register a function first."
            )
    return tools


def _mcp_failure_detail() -> str:
    """Per-server MCP connection failures, when the loader was already used.

    Reads the module out of ``sys.modules`` instead of importing it so a plain
    ``run --tools math.sqrt`` never pays the MCP client import.
    """

    loader = _sys.modules.get("dspytools.mcp.loader")
    if loader is None:
        return ""
    return loader.MCPSessionPool.describe_failures()


def _load_mcp_tool(spec: str, tools: list) -> bool:
    """Append the MCP tool named by ``server:tool``; ``False`` when not found.

    A colon is ambiguous (``server:tool`` vs ``module:attr``), so a miss is not
    an error: the caller falls back to importing the spec as a Python attribute.
    """
    if ":" not in spec or spec.endswith(":") or "/" in spec:
        return False
    server_name, tool_name = spec.split(":", 1)
    for t in _load_mcp_tools_for_server(server_name):
        if t.name == tool_name:
            tools.append(t)
            ok(f"Loaded tool '{tool_name}' from MCP server '{server_name}'")
            return True
    return False


def _tool_arg_schema(param: inspect.Parameter) -> dict[str, Any]:
    """JSON schema for one parameter, derived from its annotation."""
    from pydantic import TypeAdapter

    annotation = (
        Any if param.annotation is inspect.Parameter.empty else param.annotation
    )
    schema = TypeAdapter(annotation).json_schema()
    schema.pop("title", None)
    if param.default is not inspect.Parameter.empty and isinstance(
        param.default, (str, int, float, bool, type(None))
    ):
        schema["default"] = param.default
    return schema


def _build_dspy_tool(func: Callable, name: str | None = None) -> dspy.Tool:
    """Build a ``dspy.Tool`` that also works for C extension / built-in functions.

    Python functions are wrapped directly: ``dspy.Tool`` infers their args and
    validates/coerces values against the declared type hints. Builtins expose a
    text signature (``round(number, ndigits=None)``) but no annotations, so the
    arg schema is built explicitly and the callable is invoked with keyword
    arguments — never by positional remapping, and never with ad-hoc string to
    number coercion (``dspy.Tool`` already coerces where a type is known).
    """
    if inspect.isfunction(func) or inspect.ismethod(func):
        return dspy.Tool(func)

    tool_name = name or getattr(func, "__name__", "tool")
    desc = getattr(func, "__doc__", "") or ""

    params = [
        param
        for param in inspect.signature(func).parameters.values()
        if param.kind
        not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    ]
    if not params:
        return dspy.Tool(func, name=tool_name)

    arg_names = [param.name for param in params]
    defaults = {param.name: param.default for param in params}
    args = {param.name: _tool_arg_schema(param) for param in params}
    arg_types = {
        param.name: (
            Any if param.annotation is inspect.Parameter.empty else param.annotation
        )
        for param in params
    }
    # Positional-only builtins (math.sqrt, len, ...) refuse keyword arguments.
    positional_only = any(
        param.kind is inspect.Parameter.POSITIONAL_ONLY for param in params
    )

    def _call(**kwargs: Any) -> Any:
        if not positional_only:
            return func(**kwargs)
        positional: list[Any] = []
        for arg_name in arg_names:
            if arg_name in kwargs:
                positional.append(kwargs.pop(arg_name))
            elif defaults[arg_name] is not inspect.Parameter.empty:
                positional.append(defaults[arg_name])
            else:
                raise TypeError(
                    f"{tool_name}() missing required argument: {arg_name!r}"
                )
        return func(*positional, **kwargs)

    return dspy.Tool(
        _call,
        name=tool_name,
        desc=desc,
        args=args,
        arg_types=arg_types,
    )


def _sanitize_tool_names(tools: list) -> list:
    """Rename tools to valid Python identifiers for RLM/Flex sandboxes.

    MCP tool names are hyphenated (``git-mcp_list_files``) and ``dspy.RLM`` /
    ``dspy.Flex`` reject them (``must be a valid Python identifier``). Renamed
    copies keep the original function; renames are warned about.
    """
    used: set[str] = set()
    sanitized: list = []
    for tool in tools:
        original = getattr(tool, "name", "") or type(tool).__name__
        safe = _re.sub(r"\W", "_", original)
        if not safe or safe[0].isdigit():
            safe = f"tool_{safe}"
        candidate, suffix = safe, 2
        while candidate in used:
            candidate = f"{safe}_{suffix}"
            suffix += 1
        if candidate != original:
            warn(
                f"Renamed tool '{original}' to '{candidate}' (must be a Python identifier)"
            )
            tool = tool.model_copy(update={"name": candidate})
        used.add(candidate)
        sanitized.append(tool)
    return sanitized


def _load_mcp_tools_for_server(server_name: str) -> list:
    from dspytools.mcp.loader import MCPSessionPool

    pool = MCPSessionPool()
    return pool.get_tools(".mcp.json")


# ── DSPy 3.5 adapter / streaming helpers ────────────────────────────────


def _parallel_tools_adapter(adapter: Any) -> Any:
    """Clone the active adapter with provider-side parallel tool calls on.

    ``parallel_tool_calls`` only reaches the provider through
    ``Adapter._call_preprocess`` when the adapter is built with
    ``use_native_function_calling=True``; otherwise DSPy pops
    ``tools``/``tool_choice``/``parallel_tool_calls`` from the LM kwargs and the
    flag becomes a silent no-op.  So the clone always turns native function
    calling on.

    The *active* adapter is cloned (same class, same options) instead of being
    rebuilt from config: that keeps the user's configured adapter and any
    class-level patches ``setup_dspy()`` applied.  The global default is never
    mutated — the clone is only used inside ``dspy.context(adapter=...)``.
    """
    if adapter is None:
        raise click.ClickException(
            "  no DSPy adapter is configured; call setup_dspy() before --parallel-tools"
        )
    params = inspect.signature(type(adapter).__init__).parameters
    if "parallel_tool_calls" not in params:
        raise click.ClickException(
            f"  adapter {type(adapter).__name__} does not accept parallel_tool_calls "
            "in DSPy 3.5; use --adapter chat, json, xml or baml"
        )
    kwargs: dict[str, Any] = {
        "parallel_tool_calls": True,
        "use_native_function_calling": True,
    }
    for name in ("callbacks", "native_response_types", "use_json_adapter_fallback"):
        if name in params and hasattr(adapter, name):
            kwargs[name] = getattr(adapter, name)
    return type(adapter)(**kwargs)


@contextmanager
def _parallel_tools_context(enabled: bool):
    """Scope the active adapter to one that requests parallel tool calls.

    ``--parallel-tools`` is opt-in and off by default: with ``enabled=False``
    this yields the unchanged global settings, so behaviour is identical to a
    run without the flag.  When the configured LM cannot do native function
    calling the flag cannot take effect, so it fails loudly instead of being
    silently dropped.
    """
    if not enabled:
        yield
        return

    lm = getattr(dspy.settings, "lm", None)
    if not getattr(lm, "supports_function_calling", False):
        model = getattr(lm, "model", None) or "the configured LM"
        raise click.ClickException(
            f"  --parallel-tools needs native function calling, which {model} does "
            "not advertise; drop the flag or serve a function-calling endpoint"
        )
    with dspy.context(
        adapter=_parallel_tools_adapter(getattr(dspy.settings, "adapter", None))
    ):
        yield


def _stream_output_field(module: Any) -> str | None:
    """Last declared output field of a Predict/ChainOfThought module.

    ``dspy.ChainOfThought`` prepends a ``reasoning`` output field (rendered as
    ``plan`` for LMs without native reasoning), so the *last* field is the
    user's answer for both module types.
    """
    signature = getattr(module, "signature", None)
    if signature is None:
        signature = getattr(getattr(module, "predict", None), "signature", None)
    fields = list(getattr(signature, "output_fields", ()) or ())
    return fields[-1] if fields else None


def _run_streaming(
    module: Any, kwargs: dict[str, Any], *, adapter: Any
) -> tuple[Any, int]:
    """Run a program through ``dspy.streamify`` and echo chunks as they arrive.

    Returns ``(final_prediction, chunks_printed)``.  ``async_streaming=False``
    makes DSPy hand back a plain generator, so ``run predict``/``run cot`` stay
    synchronous.  A cached result produces no ``StreamResponse`` chunks but DSPy
    still emits the final ``Prediction`` (``include_final_prediction_in_output_stream``
    defaults to true), so a cache hit renders normally instead of hanging.
    """
    field = _stream_output_field(module)
    if field is None:
        raise click.ClickException(
            "  --stream needs a signature with at least one output field"
        )

    stream = dspy.streamify(
        module,
        stream_listeners=[dspy.streaming.StreamListener(signature_field_name=field)],
        async_streaming=False,
    )
    info(f"Streaming: [{_DIM}]field '{field}'[/]")

    result = None
    chunks = 0
    with dspy.context(adapter=adapter):
        for value in stream(**kwargs):
            if isinstance(value, dspy.Prediction):
                result = value
                continue
            chunk = getattr(value, "chunk", None)
            if chunk:
                click.echo(chunk, nl=False)
                chunks += 1
    if chunks:
        click.echo()
    if result is None:
        raise click.ClickException("  stream ended without a final prediction")
    return result, chunks


def _run_module(
    module_type: str,
    signature: str,
    inputs: tuple[str, ...],
    lm: str | None,
    temperature: float,
    max_tokens: int,
    adapter: str,
    stream: bool = False,
) -> None:
    """Generic runner for Predict and ChainOfThought."""
    _show_command_header(module_type)
    _show_inputs(_parse_inputs(inputs))
    _show_model_info(lm)
    info(f"Adapter: [{_DIM}]{adapter}[/]")
    info(f"Signature: [bold]{signature}[/]")

    t0 = time.time()
    setup_dspy(model=lm, temperature=temperature, max_tokens=max_tokens)
    kwargs = _parse_inputs(inputs)

    adapters = {
        "chat": dspy.ChatAdapter(),
        "json": dspy.JSONAdapter(),
        "xml": dspy.XMLAdapter(),
    }
    adapter_obj = adapters.get(adapter)
    if adapter_obj is None:
        # BAMLAdapter is not exported at the dspy top level — import it only
        # when it is actually selected (or as the unknown-adapter fallback).
        from dspy.adapters.baml_adapter import BAMLAdapter

        adapter_obj = BAMLAdapter(use_native_function_calling=False)

    module = {
        "predict": dspy.Predict,
        "cot": dspy.ChainOfThought,
    }[module_type](signature)

    if stream:
        result, chunks = _run_streaming(module, kwargs, adapter=adapter_obj)
        if not chunks:
            info(
                f"[{_DIM}]no streamed chunks "
                "(cached result or a non-streaming adapter)[/]"
            )
    else:
        sp = rich_spinner()
        with sp:
            sp.add_task(f"Running {module_type}...")
            with dspy.context(adapter=adapter_obj):
                result = module(**kwargs)

    _show_result(result)
    _show_timing(t0)


def _parse_inputs(inputs: tuple[str, ...]) -> dict[str, str]:
    kwargs = {}
    for inp in inputs:
        if "=" in inp:
            k, v = inp.split("=", 1)
            kwargs[k.strip()] = v.strip()
        else:
            # Try JSON format: --inputs '{"key": "val"}'
            try:
                parsed = json.loads(inp)
                if isinstance(parsed, dict):
                    for k, v in parsed.items():
                        kwargs[str(k)] = str(v)
            except json.JSONDecodeError:
                pass
    return kwargs


_MODULE_TYPES = {
    "predict": ("Predict", "dspy.Predict(signature)"),
    "cot": ("ChainOfThought", "dspy.ChainOfThought(signature)"),
    "react": ("ReAct", "dspy.ReAct(signature, tools=[...])"),
    "react-v2": ("ReAct", "dspy.ReAct(signature, tools=[...])"),
    "pot": ("RLM", "dspy.RLM(signature, max_iters=10)"),
    "code-act": ("RLM", "dspy.RLM(signature, tools=[...], max_iters=10)"),
    "rlm": ("RLM", "dspy.RLM(signature, max_iters=20)"),
    "flex": (
        "Flex",
        "dspy.Flex(signature, tools=[...], max_predictor_calls=100)",
    ),
    "best-of-n": ("BestOfN", "dspy.BestOfN(module, N=5, ...)"),
    "refine": ("Refine", "dspy.Refine(module, N=5, ...)"),
    "multi-chain": (
        "MultiChainComparison",
        "dspy.MultiChainComparison(signature, M=3)",
    ),
}


# ═══════════════════════════════════════════════════════════════════════════
#  Commands
# ═══════════════════════════════════════════════════════════════════════════


@click.group(name="run", cls=LLMGroup)
def run_cmd():
    """Run DSPy modules (Predict, ChainOfThought, ReAct, RLM, ...)."""


# ── predict ──────────────────────────────────────────────────────────────


@run_cmd.command(name="predict", cls=LLMCommand)
@click.argument("signature")
@click.option("--inputs", "-i", multiple=True, help="Inputs as KEY=VALUE")
@click.option("--lm", help="LM model to use")
@click.option("--temperature", type=float, default=0.2)
@click.option("--max-tokens", type=int, default=4096)
@click.option(
    "--adapter", type=click.Choice(["chat", "json", "xml", "baml"]), default="chat"
)
@click.option(
    "--stream",
    is_flag=True,
    default=False,
    help="Stream output-field chunks as they arrive (falls back to the result panel on cache hits)",
)
def run_predict(
    signature: str,
    inputs: tuple[str, ...],
    lm: str | None,
    temperature: float,
    max_tokens: int,
    adapter: str,
    stream: bool,
):
    """Run a Predict module.  SIGNATURE: e.g. ``"question -> answer"``"""
    _run_module(
        "predict", signature, inputs, lm, temperature, max_tokens, adapter, stream
    )


# ── cot ──────────────────────────────────────────────────────────────────


@run_cmd.command(name="cot", cls=LLMCommand)
@click.argument("signature")
@click.option("--inputs", "-i", multiple=True, help="Inputs as KEY=VALUE")
@click.option("--lm", help="LM model to use")
@click.option("--temperature", type=float, default=0.2)
@click.option("--max-tokens", type=int, default=4096)
@click.option(
    "--adapter", type=click.Choice(["chat", "json", "xml", "baml"]), default="chat"
)
@click.option(
    "--stream",
    is_flag=True,
    default=False,
    help="Stream output-field chunks as they arrive (falls back to the result panel on cache hits)",
)
def run_cot(
    signature: str,
    inputs: tuple[str, ...],
    lm: str | None,
    temperature: float,
    max_tokens: int,
    adapter: str,
    stream: bool,
):
    """Run a ChainOfThought module."""
    _run_module("cot", signature, inputs, lm, temperature, max_tokens, adapter, stream)


# ── agent execution ──────────────────────────────────────────────────────

# A failed forced submit is not a plain ValueError: ReAct re-raises the parse
# error and the context-window error from ``_forced_submit``. Resolved lazily so
# importing this module never triggers the DSPy import.
_AGENT_ERRORS: tuple[type[BaseException], ...] | None = None


def _agent_errors() -> tuple[type[BaseException], ...]:
    """Exception family raised by a failed ReAct forced submit."""
    global _AGENT_ERRORS
    if _AGENT_ERRORS is None:
        _AGENT_ERRORS = (
            ValueError,
            dspy.AdapterParseError,
            dspy.ContextWindowExceededError,
        )
    return _AGENT_ERRORS


def _agent_failure(err: Exception, start: float | None = None) -> click.ClickException:
    """Turn a failed agent run into a CLI error naming the last model output."""
    rich_panel("Agent failed", str(err), border_style=_ERROR)
    last_reply = _extract_reply_from_lm_history()
    if last_reply:
        rich_panel("Last model output", last_reply, border_style=_YELLOW)
    if start is not None:
        _show_timing(start)
    return click.ClickException(str(err))


def _run_agent(
    module: Any,
    kwargs: dict[str, Any],
    *,
    label: str = "Thinking...",
    start: float | None = None,
) -> Any:
    """Run an agent module, reporting a failed forced submit as a CLI error."""
    sp = rich_spinner()
    with sp:
        sp.add_task(label)
        try:
            return module(**kwargs)
        except _agent_errors() as err:
            raise _agent_failure(err, start) from err


# ── react ────────────────────────────────────────────────────────────────


@run_cmd.command(name="react", cls=LLMCommand)
@click.argument("signature")
@click.option("--inputs", "-i", multiple=True, help="Inputs as KEY=VALUE")
@click.option("--tools", "-t", multiple=True, help="Tool names or MCP server:tool")
@click.option("--max-iters", default=10, type=int)
@click.option("--lm", help="LM model to use")
@click.option("--temperature", type=float, default=0.2)
@click.option("--max-tokens", type=int, default=4096)
@click.option(
    "--resume-history",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help=(
        "Resume from a saved ReAct history JSON (replays conversation evidence "
        "only; live tool state is not restored)"
    ),
)
@click.option(
    "--no-save-history",
    is_flag=True,
    help="Do not auto-save this run's history JSON",
)
@click.option(
    "--parallel-tools",
    is_flag=True,
    default=False,
    help=(
        "Let one agent turn issue several tool calls (native function calling "
        "required); off by default"
    ),
)
def run_react(
    signature: str,
    inputs: tuple[str, ...],
    tools: tuple[str, ...],
    max_iters: int,
    lm: str | None,
    temperature: float,
    max_tokens: int,
    resume_history: Path | None,
    no_save_history: bool,
    parallel_tools: bool,
):
    """Run a ReAct agent with tools.  Tools format: ``"server:tool_name"``
    from ``.mcp.json``, or ``module.attr`` (e.g. ``math.sqrt``)."""

    _show_command_header("react")
    _show_inputs(_parse_inputs(inputs))
    _show_model_info(lm)
    if tools:
        info(f"Tools: [bold]{', '.join(tools)}[/]")
    info(f"Max iters: [{_DIM}]{max_iters}[/]")
    if parallel_tools:
        info("Parallel tool calls: [bold]on[/]")

    # Load before configuring the LM so a bad file fails fast and cleanly.
    history = _load_agent_history(resume_history)

    t0 = time.time()
    setup_dspy(model=lm, temperature=temperature, max_tokens=max_tokens)
    dspy_tools = _load_tools(tools)
    kwargs = _parse_inputs(inputs)
    if history is not None:
        kwargs["history"] = history

    with _parallel_tools_context(parallel_tools):
        result = _run_agent(
            dspy.ReAct(signature, tools=dspy_tools, max_iters=max_iters),
            kwargs,
            start=t0,
        )

    _show_answer(result)
    _show_react_trace(result)
    if not no_save_history:
        _save_agent_history(getattr(result, "history", None), kind="react")
    _show_timing(t0)


# ── react-v2 ─────────────────────────────────────────────────────────────


@run_cmd.command(name="react-v2", cls=LLMCommand)
@click.argument("signature")
@click.option("--inputs", "-i", multiple=True, help="Inputs as KEY=VALUE")
@click.option("--tools", "-t", multiple=True, help="Tool names or MCP server:tool")
@click.option("--max-iters", default=10, type=int)
@click.option("--lm", help="LM model to use")
@click.option("--temperature", type=float, default=0.2)
@click.option("--max-tokens", type=int, default=4096)
@click.option(
    "--parallel-tools",
    is_flag=True,
    default=False,
    help=(
        "Let one agent turn issue several tool calls (native function calling "
        "required); off by default"
    ),
)
def run_react_v2(
    signature: str,
    inputs: tuple[str, ...],
    tools: tuple[str, ...],
    max_iters: int,
    lm: str | None,
    temperature: float,
    max_tokens: int,
    parallel_tools: bool,
):
    """Run the structured ReAct agent with tools (legacy command name).

    In DSPy 3.5 ``dspy.ReAct`` *is* the structured loop: the model calls the
    typed ``submit`` tool and the result carries the declared outputs plus
    ``termination_reason`` (``submit`` / ``forced_submit``) and a structured
    ``history``. A forced final submit that produces no usable outputs raises.
    """

    _show_command_header("react-v2")
    _show_inputs(_parse_inputs(inputs))
    _show_model_info(lm)
    if tools:
        info(f"Tools: [bold]{', '.join(tools)}[/]")
    info(f"Max iters: [{_DIM}]{max_iters}[/]")
    if parallel_tools:
        info("Parallel tool calls: [bold]on[/]")

    t0 = time.time()
    setup_dspy(model=lm, temperature=temperature, max_tokens=max_tokens)
    dspy_tools = _load_tools(tools)
    kwargs = _parse_inputs(inputs)

    agent = dspy.ReAct(signature, tools=dspy_tools, max_iters=max_iters)
    with _parallel_tools_context(parallel_tools):
        result = _run_agent(agent, kwargs, start=t0)

    _show_answer(result)
    _show_react_trace(result)

    used = [t.name for t in dspy_tools] if dspy_tools else []
    if used:
        info(f"Tools: [{_DIM}]{', '.join(f'[bold]{n}[/]' for n in used)}[/]")

    _show_timing(t0)


def _next_thought_from_output(text: str) -> str | None:
    """Pull ``next_thought`` out of one LM output in any adapter format.

    DSPy 3.5 renders the ReAct ``next_thought`` field as ChatAdapter sections
    (``[[ ## next_thought ## ]]``), XML (``<next_thought>``) or a JSON object,
    depending on the active adapter.
    """
    stripped = text.strip()
    if stripped.startswith("{"):
        try:
            parsed = _json.loads(stripped)
        except _json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict) and parsed.get("next_thought"):
            return str(parsed["next_thought"]).strip()
    for pattern in (
        r"\[\[ ## next_thought ## \]\]\s*(.+?)\s*(?:\[\[ ## |\Z)",
        r"<next_thought>\s*(.+?)\s*</next_thought>",
    ):
        m = _re.search(pattern, text, _re.DOTALL)
        if m:
            return m.group(1).strip()
    return None


def _extract_reply_from_lm_history() -> str | None:
    """Extract the model's ``next_thought`` from the raw LM call history.

    DSPy 3.5 records canonical Request/Response entries: ``outputs`` holds
    plain strings for text completions and dicts carrying ``text`` /
    ``tool_calls`` for native tool-calling responses.
    """

    lm_obj = getattr(dspy.settings, "lm", None)
    raw_hist = getattr(lm_obj, "history", None)
    if not raw_hist:
        return None
    for call in reversed(raw_hist):
        for output in reversed(call.get("outputs") or []):
            text = output.get("text") if isinstance(output, dict) else output
            if not isinstance(text, str) or not text.strip():
                continue
            thought = _next_thought_from_output(text)
            if thought:
                return thought
    return None


# ── flex ──────────────────────────────────────────────────────────────────


@contextmanager
def _interpreter_factory(pool_size: int = 0):
    """Yield an interpreter factory for the Flex/RLM sandbox paths.

    ``pool_size=0`` (the default) yields ``None`` so the module keeps DSPy's
    own one-interpreter-per-invocation behaviour. A positive size yields a
    ``dspy.InterpreterPool`` that prewarms that many interpreters and is always
    closed (idle sandboxes shut down) when the command finishes.
    """
    if pool_size <= 0:
        yield None
        return
    pool = dspy.InterpreterPool(size=pool_size)
    try:
        yield pool
    finally:
        pool.close()


@run_cmd.command(name="flex", cls=LLMCommand)
@click.argument("signature")
@click.option("--inputs", "-i", multiple=True, help="Inputs as KEY=VALUE")
@click.option("--tools", "-t", multiple=True, help="Tool names or MCP server:tool")
@click.option("--lm", help="LM model to use")
@click.option("--temperature", type=float, default=0.2)
@click.option("--max-tokens", type=int, default=4096)
@click.option(
    "--max-predictor-calls",
    type=int,
    default=100,
    help=(
        "Cap on LM calls per forward, including nested/candidate calls "
        "(DSPy 3.5 counts every call)"
    ),
)
@click.option(
    "--interpreter-pool-size",
    type=int,
    default=0,
    help="Prewarm N sandbox interpreters via dspy.InterpreterPool (0 = off)",
)
def run_flex(
    signature: str,
    inputs: tuple[str, ...],
    tools: tuple[str, ...],
    lm: str | None,
    temperature: float,
    max_tokens: int,
    max_predictor_calls: int,
    interpreter_pool_size: int,
):
    """Run a dspy.Flex program (optimizable module code).

    Optionally with tools; the baseline becomes dspy.RLM when tools are given.
    """

    _show_command_header("flex")
    _show_inputs(_parse_inputs(inputs))
    _show_model_info(lm)
    if tools:
        info(f"Tools: [bold]{', '.join(tools)}[/]")
    info(f"Max predictor calls: [{_DIM}]{max_predictor_calls}[/]")
    if interpreter_pool_size:
        info(f"Interpreter pool: [{_DIM}]{interpreter_pool_size}[/]")

    t0 = time.time()
    setup_dspy(model=lm, temperature=temperature, max_tokens=max_tokens)
    dspy_tools = _sanitize_tool_names(_load_tools(tools))
    kwargs = _parse_inputs(inputs)

    sp = rich_spinner()
    with sp, _interpreter_factory(interpreter_pool_size) as factory:
        sp.add_task("Running Flex...")
        extra = {"interpreter_factory": factory} if factory is not None else {}
        prog = dspy.Flex(
            signature,
            tools=dspy_tools or None,
            max_predictor_calls=max_predictor_calls,
            **extra,
        )
        try:
            result = prog(**kwargs)
        except Exception as e:
            rich_panel("Error", str(e), border_style=_ERROR)
            _show_timing(t0)
            raise click.ClickException(str(e)) from e

    answer = getattr(result, "answer", getattr(result, "output", str(result)))
    _show_answer(answer)
    _show_timing(t0)


# ── pot ──────────────────────────────────────────────────────────────────


@run_cmd.command(name="pot", cls=LLMCommand)
@click.argument("signature")
@click.option("--inputs", "-i", multiple=True, help="Inputs as KEY=VALUE")
@click.option("--max-iters", default=10, type=int)
@click.option("--max-llm-calls", default=30, type=int)
@click.option("--lm", help="LM model to use")
def run_pot(
    signature: str,
    inputs: tuple[str, ...],
    max_iters: int,
    max_llm_calls: int,
    lm: str | None,
):
    """Run a ProgramOfThought-style module on the RLM engine.

    ``dspy.ProgramOfThought`` was removed in DSPy 3.5 and RLM is its
    replacement. This command keeps the familiar ``pot`` surface but
    executes on the sandboxed RLM REPL (reasoning via code execution).
    """
    _show_command_header("pot")
    _show_inputs(_parse_inputs(inputs))
    _show_model_info(lm)
    info(f"Max iters: [{_DIM}]{max_iters}[/]")

    t0 = time.time()
    setup_dspy(model=lm)
    kwargs = _parse_inputs(inputs)

    sp = rich_spinner()
    with sp:
        sp.add_task("Reasoning & executing code...")
        result = dspy.RLM(
            signature,
            max_iters=max_iters,
            max_llm_calls=max_llm_calls,
        )(**kwargs)

    _show_result(result, border=_CODE)
    _show_timing(t0)


# ── code-act ─────────────────────────────────────────────────────────────


@run_cmd.command(name="code-act", cls=LLMCommand)
@click.argument("signature")
@click.option("--inputs", "-i", multiple=True, help="Inputs as KEY=VALUE")
@click.option("--tools", "-t", multiple=True, help="Tool names (plain functions only)")
@click.option("--max-iters", default=10, type=int)
@click.option("--max-llm-calls", default=30, type=int)
@click.option("--lm", help="LM model to use")
def run_code_act(
    signature: str,
    inputs: tuple[str, ...],
    tools: tuple[str, ...],
    max_iters: int,
    max_llm_calls: int,
    lm: str | None,
):
    """Run a CodeAct-style module on the RLM engine.

    ``dspy.CodeAct`` was removed in DSPy 3.5 and RLM is its replacement. This
    command keeps the familiar ``code-act`` surface but executes on the
    sandboxed RLM REPL (code generation + tool execution).
    """
    _show_command_header("code-act")
    _show_inputs(_parse_inputs(inputs))
    _show_model_info(lm)
    if tools:
        info(f"Tools: [bold]{', '.join(tools)}[/]")
    info(f"Max iters: [{_DIM}]{max_iters}[/]")

    t0 = time.time()
    setup_dspy(model=lm)
    kwargs = _parse_inputs(inputs)
    dspy_tools = _sanitize_tool_names(_load_tools(tools))

    sp = rich_spinner()
    with sp:
        sp.add_task("Reasoning, executing code & calling tools...")
        result = dspy.RLM(
            signature,
            tools=dspy_tools,
            max_iters=max_iters,
            max_llm_calls=max_llm_calls,
        )(**kwargs)

    _show_result(result, border=_CODE)
    _show_timing(t0)


# ── rlm ──────────────────────────────────────────────────────────────────


def _load_agent_history(path: Path | None) -> dspy.History | None:
    """Load a saved agent history (RLM or ReAct) for replay.

    History replays conversation evidence only — live Python variables and tool
    state are not restored — and the loaded object is passed as the ``history``
    input without being modified. A bad file raises a clean CLI error.
    """
    if path is None:
        return None
    try:
        history = dspy.History.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as err:
        raise click.ClickException(f"cannot resume from {path}: {err}") from err
    info(f"Resumed: [{_DIM}]{path}[/] ({len(history.messages)} turns)")
    return history


def _save_agent_history(history: Any, *, kind: str = "rlm") -> None:
    """Persist an agent ``History`` as JSON for later ``--resume-history``.

    These commands keep no run directory, so the file lands flat in the
    toolkit's data dir — where their other one-off runtime state lives. Empty
    histories are skipped, and a write failure warns instead of failing an
    otherwise successful run.
    """
    if not getattr(history, "messages", None):
        return
    path = data_dir() / f"{kind}_history_{datetime.now():%Y%m%d_%H%M%S}.json"
    try:
        path.write_text(history.model_dump_json(), encoding="utf-8")
    except (OSError, ValueError) as err:
        warn(f"could not save {kind.upper()} history: {err}")
        return
    info(f"History: [{_DIM}]{path}[/]  resume with --resume-history {path}")


@run_cmd.command(name="rlm", cls=LLMCommand)
@click.argument("signature")
@click.option("--inputs", "-i", multiple=True, help="Inputs as KEY=VALUE")
@click.option(
    "--tools", "-t", multiple=True, help="Tool names (plain functions or server:tool)"
)
@click.option("--max-iters", default=10, type=int)
@click.option("--max-llm-calls", default=30, type=int)
@click.option("--lm", help="Main LM")
@click.option("--sub-lm", help="Sub-LLM for llm_query() calls")
@click.option(
    "--interpreter-pool-size",
    type=int,
    default=0,
    help="Prewarm N sandbox interpreters via dspy.InterpreterPool (0 = off)",
)
@click.option(
    "--resume-history",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help=(
        "Resume from a saved RLM history JSON (replays conversation evidence "
        "only; Python variables are not restored)"
    ),
)
def run_rlm(
    signature: str,
    inputs: tuple[str, ...],
    tools: tuple[str, ...],
    max_iters: int,
    max_llm_calls: int,
    lm: str | None,
    sub_lm: str | None,
    interpreter_pool_size: int,
    resume_history: Path | None,
):
    """Run a Recursive Language Model (RLM) module."""
    _show_command_header("rlm")
    _show_inputs(_parse_inputs(inputs))
    _show_model_info(lm)
    if tools:
        info(f"Tools: [bold]{', '.join(tools)}[/]")
    if sub_lm:
        info(f"Sub-LM: [{_DIM}]{sub_lm}[/]")
    info(
        f"Iterations: [{_DIM}]{max_iters}[/]   "
        f"Max LLM calls: [{_DIM}]{max_llm_calls}[/]"
    )
    if interpreter_pool_size:
        info(f"Interpreter pool: [{_DIM}]{interpreter_pool_size}[/]")

    # Load before configuring the LM so a bad file fails fast and cleanly.
    history = _load_agent_history(resume_history)

    t0 = time.time()
    setup_dspy(model=lm, temperature=0.2, max_tokens=4096)
    kwargs = _parse_inputs(inputs)
    dspy_tools = _sanitize_tool_names(_load_tools(tools))
    sub_lm_obj = LMRegistry.get(model=sub_lm, temperature=0.0) if sub_lm else None
    if history is not None:
        kwargs["history"] = history

    sp = rich_spinner()
    with sp, _interpreter_factory(interpreter_pool_size) as factory:
        sp.add_task("Running RLM...")
        extra = {"interpreter_factory": factory} if factory is not None else {}
        result = dspy.RLM(
            signature,
            tools=dspy_tools,
            max_iters=max_iters,
            max_llm_calls=max_llm_calls,
            sub_lm=sub_lm_obj,
            **extra,
        )(**kwargs)

    _show_answer(result)
    _save_agent_history(getattr(result, "history", None), kind="rlm")
    _show_timing(t0)


# ── best-of-n / refine ───────────────────────────────────────────────────


def _reward_fn(
    keyword: str | None, scored: list[float]
) -> Callable[[dict, Any], float]:
    """Structural reward: every declared output field is non-empty.

    ``BestOfN``/``Refine`` break as soon as ``reward >= threshold``, so a
    constant reward ends sampling after one attempt. This reward is honest by
    construction: ``0.0`` for a missing/empty answer, ``1.0`` for a complete
    one, plus the optional ``--require-keyword`` quality gate. Every call is
    recorded in ``scored`` so the real attempt count can be reported.
    """

    def reward(_args: dict, pred: Any) -> float:
        values: list[str] = []
        if pred is not None:
            values = [
                str(getattr(pred, name, "") or "").strip()
                for name in _result_fields(pred)
            ]
        passed = bool(values) and all(values)
        if passed and keyword:
            passed = any(keyword.lower() in value.lower() for value in values)
        value = 1.0 if passed else 0.0
        scored.append(value)
        return value

    return reward


def _effective_threshold(threshold: float, keyword: str | None) -> tuple[float, bool]:
    """Return the threshold to use and whether early exit is meaningful.

    Without ``--require-keyword`` the only signal is structure, which any
    well-formed answer satisfies — so early exit would stop after the first
    sample and make ``-n`` a lie. Disable it by making the threshold
    unreachable, and say so in the output.
    """
    if keyword:
        return threshold, True
    return max(threshold, 1.01), False


def _run_sampler(
    kind: str,
    signature: str,
    kwargs: dict[str, Any],
    *,
    n: int,
    threshold: float,
    keyword: str | None,
    fail_count: int | None,
) -> Any:
    """Run ``BestOfN``/``Refine`` with the structural reward and report attempts."""
    effective, early_exit = _effective_threshold(threshold, keyword)
    if early_exit:
        info(
            f"Quality signal: [{_DIM}]--require-keyword {keyword!r}[/]   "
            f"Threshold: [{_DIM}]{effective}[/]"
        )
    else:
        info(
            f"Quality signal: [{_DIM}]none (--require-keyword not set)[/]   "
            f"Threshold: [{_DIM}]{effective}[/]"
        )
        info(f"Early exit disabled: running all [bold]{n}[/] attempts")

    scored: list[float] = []
    reward = _reward_fn(keyword, scored)
    sampler_cls = dspy.BestOfN if kind == "best-of-n" else dspy.Refine
    extra = {"fail_count": fail_count} if fail_count is not None else {}
    sampler = sampler_cls(
        module=dspy.ChainOfThought(signature),
        N=n,
        threshold=effective,
        reward_fn=reward,
        **extra,
    )

    label = f"Sampling {n}x..." if kind == "best-of-n" else f"Refining ({n} rounds)..."
    sp = rich_spinner()
    with sp:
        sp.add_task(label)
        result = sampler(**kwargs)

    best = max(scored, default=0.0)
    info(f"Attempts: [bold]{len(scored)}/{n}[/]   Best reward: [{_DIM}]{best:.2f}[/]")
    if result is None:
        raise click.ClickException(f"{kind}: all {n} attempts failed")
    return result


@run_cmd.command(name="best-of-n", cls=LLMCommand)
@click.argument("signature")
@click.option("--inputs", "-i", multiple=True, help="Inputs as KEY=VALUE")
@click.option("--samples", "-n", default=5, type=int, help="Number of samples")
@click.option(
    "--threshold",
    default=0.8,
    type=float,
    help=(
        "Reward needed for early exit. With --require-keyword this is a real "
        "quality gate; without it early exit is disabled and all N run"
    ),
)
@click.option(
    "--require-keyword",
    default=None,
    help=(
        "Only count an attempt as a success when this keyword appears in an "
        "output field (enables early exit at --threshold)"
    ),
)
@click.option(
    "--fail-count",
    type=int,
    default=None,
    help="Raise after this many failed attempts (default: --samples)",
)
@click.option("--lm", help="LM model to use")
def run_best_of_n(
    signature: str,
    inputs: tuple[str, ...],
    samples: int,
    threshold: float,
    require_keyword: str | None,
    fail_count: int | None,
    lm: str | None,
):
    """Run a BestOfN module (sample N times, pick best)."""
    _show_command_header("best-of-n")
    _show_inputs(_parse_inputs(inputs))
    _show_model_info(lm)
    info(f"Samples: [bold]{samples}[/]")

    t0 = time.time()
    setup_dspy(model=lm)
    kwargs = _parse_inputs(inputs)

    result = _run_sampler(
        "best-of-n",
        signature,
        kwargs,
        n=samples,
        threshold=threshold,
        keyword=require_keyword,
        fail_count=fail_count,
    )

    _show_answer(result)
    _show_timing(t0)


# ── refine ───────────────────────────────────────────────────────────────


@run_cmd.command(name="refine", cls=LLMCommand)
@click.argument("signature")
@click.option("--inputs", "-i", multiple=True, help="Inputs as KEY=VALUE")
@click.option("--rounds", "-n", default=5, type=int, help="Refinement rounds")
@click.option(
    "--threshold",
    default=0.8,
    type=float,
    help=(
        "Reward needed for early exit. With --require-keyword this is a real "
        "quality gate; without it early exit is disabled and all N run"
    ),
)
@click.option(
    "--require-keyword",
    default=None,
    help=(
        "Only count an attempt as a success when this keyword appears in an "
        "output field (enables early exit at --threshold)"
    ),
)
@click.option(
    "--fail-count",
    type=int,
    default=None,
    help="Raise after this many failed attempts (default: --rounds)",
)
@click.option("--lm", help="LM model to use")
def run_refine(
    signature: str,
    inputs: tuple[str, ...],
    rounds: int,
    threshold: float,
    require_keyword: str | None,
    fail_count: int | None,
    lm: str | None,
):
    """Run a Refine module (iterative improvement with feedback)."""
    _show_command_header("refine")
    _show_inputs(_parse_inputs(inputs))
    _show_model_info(lm)
    info(f"Rounds: [bold]{rounds}[/]")

    t0 = time.time()
    setup_dspy(model=lm)
    kwargs = _parse_inputs(inputs)

    result = _run_sampler(
        "refine",
        signature,
        kwargs,
        n=rounds,
        threshold=threshold,
        keyword=require_keyword,
        fail_count=fail_count,
    )

    _show_answer(result)
    _show_timing(t0)


# ── multi-chain ──────────────────────────────────────────────────────────


@run_cmd.command(name="multi-chain", cls=LLMCommand)
@click.argument("signature")
@click.option("--inputs", "-i", multiple=True, help="Inputs as KEY=VALUE")
@click.option("--chains", "-m", default=3, type=int, help="Number of reasoning chains")
@click.option("--temperature", default=0.7, type=float)
@click.option("--lm", help="LM model to use")
def run_multi_chain(
    signature: str,
    inputs: tuple[str, ...],
    chains: int,
    temperature: float,
    lm: str | None,
):
    """Run a MultiChainComparison module (compare M reasoning chains)."""
    _show_command_header("multi-chain")
    _show_inputs(_parse_inputs(inputs))
    _show_model_info(lm)
    info(f"Chains: [bold]{chains}[/]   Temperature: [{_DIM}]{temperature}[/]")

    t0 = time.time()
    setup_dspy(model=lm, temperature=temperature)
    kwargs = _parse_inputs(inputs)

    sp = rich_spinner()
    with sp:
        sp.add_task(f"Generating {chains} reasoning chains...")
        judge = dspy.MultiChainComparison(signature, M=chains, temperature=temperature)
        predictor = dspy.ChainOfThought(signature, temperature=temperature)
        # Identical inputs at identical config are a cache hit: without a
        # distinct rollout id the judge would compare M copies of one draft.
        lm_obj = dspy.settings.lm
        completions: list = []
        for i in range(chains):
            rollout = lm_obj.copy(rollout_id=i) if lm_obj is not None else None
            context = dspy.context(lm=rollout) if rollout is not None else nullcontext()
            with context:
                completions.append(predictor(**kwargs))
        if not completions:
            fail("All chain attempts failed")
            raise click.ClickException("all M completion attempts failed")
        drafts = {repr(completion) for completion in completions}
        if len(drafts) < len(completions):
            warn(
                f"Only {len(drafts)}/{len(completions)} drafts differ after "
                f"distinct rollout ids — the judge is comparing near-duplicates"
            )
        try:
            result = judge(completions, **kwargs)
        except (RuntimeError, ValueError, KeyError, TypeError) as e:
            fail(f"MultiChainComparison judge failed: {e}")
            raise click.ClickException(f"MultiChainComparison judge failed: {e}")

    if _result_fields(result):
        parts = [
            getattr(result, fname, "")
            for fname in _result_fields(result)
            if getattr(result, fname, "")
        ]
        rich_panel("Best Answer", "\n".join(parts), border_style=_RESULT)
    else:
        rich_panel("Result", str(result), border_style=_RESULT)
    _show_timing(t0)


# ── list ─────────────────────────────────────────────────────────────────


@run_cmd.command(name="list", cls=LLMCommand)
def run_list():
    """List available module types."""
    section(" Available Modules ")
    rich_table(
        "DSPy Modules",
        ["Command", "Type", "Description"],
        [
            [cmd, mod, f"dspytools run {cmd} --help"]
            for cmd, (mod, _) in sorted(_MODULE_TYPES.items())
        ],
    )


# ── parallel ─────────────────────────────────────────────────────────────


@run_cmd.command(name="parallel", cls=LLMCommand)
@click.option("--modules", "-m", multiple=True, required=True, help="Module signatures")
@click.option("--inputs", "-i", multiple=True, help="Inputs as KEY=VALUE")
@click.option("--lm", help="LM model to use")
@click.option(
    "--timeout",
    type=int,
    default=120,
    show_default=True,
    help="Per-module timeout in seconds",
)
@click.option(
    "--straggler-limit",
    type=int,
    default=3,
    show_default=True,
    help="Abort a module after this many consecutive timeouts",
)
def run_parallel(
    modules: tuple[str, ...],
    inputs: tuple[str, ...],
    lm: str | None,
    timeout: int,
    straggler_limit: int,
):
    """Run multiple modules in parallel."""
    section(" Parallel ")
    _show_inputs(_parse_inputs(inputs))
    _show_model_info(lm)
    info(f"Modules: [bold]{len(modules)}[/]")

    t0 = time.time()
    setup_dspy(model=lm)
    kwargs = _parse_inputs(inputs)

    sp = rich_spinner()
    with sp:
        sp.add_task("Running modules in parallel...")
        # build the (module, example) pairs dspy.Parallel consumes
        exec_pairs: list[tuple[Any, Any]] = []
        for sig in modules:
            mod = dspy.ChainOfThought(sig)
            ex = dspy.Example(**kwargs).with_inputs(*kwargs.keys())
            exec_pairs.append((mod, ex))
        parallel = dspy.Parallel(
            num_threads=len(modules),
            timeout=timeout,
            straggler_limit=straggler_limit,
            return_failed_examples=True,
        )
        results, failed_examples, exceptions = parallel(exec_pairs)

    successes = [item for item in results if item is not None]
    if failed_examples:
        fail(f"{len(failed_examples)}/{len(exec_pairs)} modules failed")
        for example, error in zip(failed_examples, exceptions, strict=False):
            warn(f"{type(error).__name__}: {error} — inputs {dict(example.inputs())}")
        if not successes:
            raise click.ClickException(
                f"all {len(exec_pairs)} modules failed; see the errors above"
            )
        warn(f"Showing the {len(successes)} module(s) that completed")

    _show_result(successes, title="Parallel Results")
    _show_timing(t0)


# ── retrieve ─────────────────────────────────────────────────────────────


@run_cmd.command(name="retrieve", cls=LLMCommand)
@click.argument("query")
@click.option("--k", default=5, type=int, help="Number of passages to retrieve")
@click.option(
    "--url",
    default="http://20.102.90.50:2017/wiki17_abstracts",
    help="ColBERTv2 server URL",
    show_default=True,
)
def run_retrieve(query: str, k: int, url: str):
    """Retrieve k passages using ColBERTv2 or configured retriever."""
    section(" Retrieve ")
    info(f"Query: [bold]{query}[/]")
    info(f"K:     [bold]{k}[/]")
    info(f"URL:   [{_DIM}]{url}[/]")

    t0 = time.time()
    sp = rich_spinner()
    with sp:
        sp.add_task("Retrieving...")
        retriever = dspy.ColBERTv2(url=url)
        results = retriever(query, k=k)

    if results:
        lines = [
            f"[{_CYAN}]{i + 1}.[/] {passage[:200]}" for i, passage in enumerate(results)
        ]
        rich_panel(f"Top {k} Passages", "\n".join(lines), border_style=_RESULT)
    else:
        info("No passages retrieved.")
    _show_timing(t0)


# ── cot-hint ─────────────────────────────────────────────────────────────


def _cot_with_hint(signature: Any, hint: str | None) -> Any:
    """Build a ChainOfThought whose instructions include ``hint``.

    There is no ``ChainOfThoughtWithHint`` in DSPy 3.5, so the hint is layered
    on with ``append_instructions`` — a *new* signature class. The task
    docstring survives, the caller's signature class is never mutated, and a
    reused module cannot keep the hint.
    """
    module = dspy.ChainOfThought(signature)
    if not hint or module.predict is None or module.predict.signature is None:
        return module
    hinted = module.predict.signature.append_instructions(f"Hint: {hint}")
    module.predict = dspy.Predict(hinted)
    return module


@run_cmd.command(name="cot-hint", cls=LLMCommand)
@click.argument("signature")
@click.option("--inputs", "-i", multiple=True, help="Inputs as KEY=VALUE")
@click.option("--hint", help="Hint string for the chain-of-thought")
@click.option("--lm", help="LM model to use")
def run_cot_hint(
    signature: str, inputs: tuple[str, ...], hint: str | None, lm: str | None
):
    """Run a ChainOfThoughtWithHint module."""
    _show_command_header("cot-hint")
    _show_inputs(_parse_inputs(inputs))
    _show_model_info(lm)
    if hint:
        info(f"Hint: [bold]{hint}[/]")

    t0 = time.time()
    setup_dspy(model=lm)
    kwargs = _parse_inputs(inputs)

    sp = rich_spinner()
    with sp:
        sp.add_task("Thinking with hint...")
        module = _cot_with_hint(signature, hint)
        result = module(**kwargs)

    _show_answer(result)
    _show_timing(t0)


# ── knn ──────────────────────────────────────────────────────────────────


@run_cmd.command(name="knn", cls=LLMCommand)
@click.argument("query")
@click.option("--k", default=5, type=int, help="Number of neighbors")
@click.option("--trainset", help="Path to trainset JSON file for KNN training")
def run_knn(query: str, k: int, trainset: str | None):
    """Run KNN classification with few-shot retrieval.

    Requires --trainset for training data (JSON array of examples with 'text' field).
    """
    section(" KNN ")
    info(f"Query: [bold]{query}[/]")
    info(f"K:     [bold]{k}[/]")
    if not trainset:
        warn(
            "KNN requires a trainset. Use: dspytools run knn QUERY --trainset data.json --k 5"
        )
        warn("Trainset must be a JSON array of objects with a 'text' field.")
        return

    t0 = time.time()
    from dspytools.core.loaders import load_trainset

    examples = load_trainset(trainset)
    vectorizer = dspy.Embedder(**embedder_kwargs())
    knn = dspy.KNN(k=k, trainset=examples, vectorizer=vectorizer)
    result = knn(query)

    rich_panel(f"{k} Nearest Neighbors", str(result), border_style=_RESULT)
    _show_timing(t0)
