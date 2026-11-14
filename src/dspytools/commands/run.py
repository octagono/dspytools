"""dspytools run — Run DSPy modules (predict, cot, react, etc.)."""

from __future__ import annotations

import importlib
import inspect
import json
import json as _json
import re as _re
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

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
# (ReAct: history/termination_reason, RLM: history/final_reasoning/trajectory).
_RESULT_META_FIELDS = frozenset(
    {"history", "termination_reason", "final_reasoning", "trajectory"}
)


def _result_fields(result: Any) -> list[str]:
    """Return a Prediction's declared output fields (DSPy 3.5 has no
    ``_output_field_names``; fall back to the underlying store)."""
    names = getattr(result, "_output_field_names", None)
    if names:
        return list(names)
    store = getattr(result, "_store", None)
    if isinstance(store, dict):
        return [name for name in store if name not in _RESULT_META_FIELDS]
    return []


def _show_result(
    result: Any,
    *,
    title: str = "Result",
    border: str = _RESULT,
) -> None:
    """Display a DSPy prediction in a panel with labelled output fields."""
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
        if ":" in spec and not spec.endswith(":") and "/" not in spec:
            server_name, tool_name = spec.split(":", 1)
            mcp_tools = _load_mcp_tools_for_server(server_name)
            found = False
            for t in mcp_tools:
                if t.name == tool_name:
                    tools.append(t)
                    ok(f"Loaded tool '{tool_name}' from MCP server '{server_name}'")
                    found = True
                    break
            if not found:
                warn(f"Tool '{tool_name}' not found in MCP server '{server_name}'")
        else:
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
                warn(
                    f"Tool '{spec}' not found. Use qualified names "
                    f"(e.g. ``math.sqrt``) or register a function first."
                )
    return tools


def _build_dspy_tool(func: Callable, name: str | None = None) -> dspy.Tool:
    """Build a ``dspy.Tool`` that handles C extension / built-in functions."""

    if inspect.isfunction(func) or inspect.ismethod(func):
        return dspy.Tool(func)

    desc = getattr(func, "__doc__", "") or ""
    tool_name = name or getattr(func, "__name__", "tool")

    sig = inspect.signature(func)
    params: dict[str, Any] = {}
    for pname, param in sig.parameters.items():
        if param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        params[pname] = {"type": ["string", "number", "integer"]}
        if param.default is not inspect.Parameter.empty:
            params[pname]["default"] = param.default

    if not params:
        return dspy.Tool(func)

    param_names = list(params.keys())

    def _wrapped(**kwargs: object) -> Any:
        mapped: list[object] = []
        for p in param_names:
            v = kwargs.pop(p, ...)
            if v is ...:
                continue
            if isinstance(v, str):
                for caster in (int, float):
                    try:
                        v = caster(v)
                        break
                    except (ValueError, TypeError):
                        continue
            mapped.append(v)
        return func(*mapped, **kwargs)

    return dspy.Tool(_wrapped, name=tool_name, desc=desc, args=params)


def _load_mcp_tools_for_server(server_name: str) -> list:
    from dspytools.mcp.loader import MCPSessionPool

    pool = MCPSessionPool()
    return pool.get_tools(".mcp.json")


def _run_module(
    module_type: str,
    signature: str,
    inputs: tuple[str, ...],
    lm: str | None,
    temperature: float,
    max_tokens: int,
    adapter: str,
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

    sp = rich_spinner()
    with sp:
        sp.add_task(f"Running {module_type}...")
        with dspy.context(adapter=adapter_obj):
            modules = {
                "predict": dspy.Predict(signature),
                "cot": dspy.ChainOfThought(signature),
            }
            result = modules[module_type](**kwargs)

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
def run_predict(
    signature: str,
    inputs: tuple[str, ...],
    lm: str | None,
    temperature: float,
    max_tokens: int,
    adapter: str,
):
    """Run a Predict module.  SIGNATURE: e.g. ``"question -> answer"``"""
    _run_module("predict", signature, inputs, lm, temperature, max_tokens, adapter)


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
def run_cot(
    signature: str,
    inputs: tuple[str, ...],
    lm: str | None,
    temperature: float,
    max_tokens: int,
    adapter: str,
):
    """Run a ChainOfThought module."""
    _run_module("cot", signature, inputs, lm, temperature, max_tokens, adapter)


# ── react ────────────────────────────────────────────────────────────────


@run_cmd.command(name="react", cls=LLMCommand)
@click.argument("signature")
@click.option("--inputs", "-i", multiple=True, help="Inputs as KEY=VALUE")
@click.option("--tools", "-t", multiple=True, help="Tool names or MCP server:tool")
@click.option("--max-iters", default=10, type=int)
@click.option("--lm", help="LM model to use")
@click.option("--temperature", type=float, default=0.2)
@click.option("--max-tokens", type=int, default=4096)
def run_react(
    signature: str,
    inputs: tuple[str, ...],
    tools: tuple[str, ...],
    max_iters: int,
    lm: str | None,
    temperature: float,
    max_tokens: int,
):
    """Run a ReAct agent with tools.  Tools format: ``"server:tool_name"``
    from ``.mcp.json``, or ``module.attr`` (e.g. ``math.sqrt``)."""

    _show_command_header("react")
    _show_inputs(_parse_inputs(inputs))
    _show_model_info(lm)
    if tools:
        info(f"Tools: [bold]{', '.join(tools)}[/]")
    info(f"Max iters: [{_DIM}]{max_iters}[/]")

    t0 = time.time()
    setup_dspy(model=lm, temperature=temperature, max_tokens=max_tokens)
    dspy_tools = _load_tools(tools)
    kwargs = _parse_inputs(inputs)

    sp = rich_spinner()
    with sp:
        sp.add_task("Thinking...")
        result = dspy.ReAct(signature, tools=dspy_tools, max_iters=max_iters)(**kwargs)

    _show_answer(result)
    _show_react_trace(result)
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
def run_react_v2(
    signature: str,
    inputs: tuple[str, ...],
    tools: tuple[str, ...],
    max_iters: int,
    lm: str | None,
    temperature: float,
    max_tokens: int,
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

    t0 = time.time()
    setup_dspy(model=lm, temperature=temperature, max_tokens=max_tokens)
    dspy_tools = _load_tools(tools)
    kwargs = _parse_inputs(inputs)

    agent = dspy.ReAct(signature, tools=dspy_tools, max_iters=max_iters)

    sp = rich_spinner()
    with sp:
        sp.add_task("Thinking...")
        try:
            result = agent(**kwargs)
        except ValueError as err:
            # 3.5 raises when the forced submit yields no usable outputs.
            # Surface it instead of guessing an answer from the raw LM text.
            rich_panel("ReAct failed", str(err), border_style=_ERROR)
            last_reply = _extract_reply_from_lm_history()
            if last_reply:
                rich_panel("Last model output", last_reply, border_style=_YELLOW)
            _show_timing(t0)
            raise click.ClickException(str(err)) from err

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
    dspy_tools = _load_tools(tools)
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
    dspy_tools = _load_tools(tools)

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


def _load_rlm_history(path: Path | None) -> dspy.History | None:
    """Load a saved RLM history for replay.

    History replays conversation evidence only — live Python variables are
    not restored — and the loaded object is passed as the ``history`` input
    without being modified. A bad file raises a clean CLI error.
    """
    if path is None:
        return None
    try:
        history = dspy.History.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as err:
        raise click.ClickException(f"cannot resume from {path}: {err}") from err
    info(f"Resumed: [{_DIM}]{path}[/] ({len(history.messages)} turns)")
    return history


def _save_rlm_history(history: Any) -> None:
    """Persist an RLM ``History`` as JSON for later ``--resume-history``.

    ``run rlm`` keeps no run directory, so the file lands flat in the toolkit's
    data dir — where the command's other one-off runtime state lives. Empty
    histories are skipped, and a write failure warns instead of failing an
    otherwise successful run.
    """
    if not getattr(history, "messages", None):
        return
    path = data_dir() / f"rlm_history_{datetime.now():%Y%m%d_%H%M%S}.json"
    try:
        path.write_text(history.model_dump_json(), encoding="utf-8")
    except (OSError, ValueError) as err:
        warn(f"could not save RLM history: {err}")
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
    history = _load_rlm_history(resume_history)

    t0 = time.time()
    setup_dspy(model=lm, temperature=0.2, max_tokens=4096)
    kwargs = _parse_inputs(inputs)
    dspy_tools = _load_tools(tools)
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
    _save_rlm_history(getattr(result, "history", None))
    _show_timing(t0)


# ── best-of-n ────────────────────────────────────────────────────────────


@run_cmd.command(name="best-of-n", cls=LLMCommand)
@click.argument("signature")
@click.option("--inputs", "-i", multiple=True, help="Inputs as KEY=VALUE")
@click.option("--samples", "-n", default=5, type=int, help="Number of samples")
@click.option("--threshold", default=0.8, type=float)
@click.option("--lm", help="LM model to use")
def run_best_of_n(
    signature: str,
    inputs: tuple[str, ...],
    samples: int,
    threshold: float,
    lm: str | None,
):
    """Run a BestOfN module (sample N times, pick best)."""
    _show_command_header("best-of-n")
    _show_inputs(_parse_inputs(inputs))
    _show_model_info(lm)
    info(f"Samples: [bold]{samples}[/]   Threshold: [{_DIM}]{threshold}[/]")

    t0 = time.time()
    setup_dspy(model=lm)
    kwargs = _parse_inputs(inputs)

    sp = rich_spinner()
    with sp:
        sp.add_task(f"Sampling {samples}x...")
        module = dspy.ChainOfThought(signature)
        result = dspy.BestOfN(
            module=module,
            N=samples,
            threshold=threshold,
            reward_fn=lambda args, pred: 1.0 if pred else 0.0,
        )(**kwargs)

    _show_answer(result)
    _show_timing(t0)


# ── refine ───────────────────────────────────────────────────────────────


@run_cmd.command(name="refine", cls=LLMCommand)
@click.argument("signature")
@click.option("--inputs", "-i", multiple=True, help="Inputs as KEY=VALUE")
@click.option("--rounds", "-n", default=5, type=int, help="Refinement rounds")
@click.option("--threshold", default=0.8, type=float)
@click.option("--lm", help="LM model to use")
def run_refine(
    signature: str,
    inputs: tuple[str, ...],
    rounds: int,
    threshold: float,
    lm: str | None,
):
    """Run a Refine module (iterative improvement with feedback)."""
    _show_command_header("refine")
    _show_inputs(_parse_inputs(inputs))
    _show_model_info(lm)
    info(f"Rounds: [bold]{rounds}[/]   Threshold: [{_DIM}]{threshold}[/]")

    t0 = time.time()
    setup_dspy(model=lm)
    kwargs = _parse_inputs(inputs)

    sp = rich_spinner()
    with sp:
        sp.add_task(f"Refining ({rounds} rounds)...")
        module = dspy.ChainOfThought(signature)
        result = dspy.Refine(
            module=module,
            N=rounds,
            threshold=threshold,
            reward_fn=lambda args, pred: 1.0 if pred else 0.0,
        )(**kwargs)

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
        completions: list = []
        for i in range(chains):
            completions.append(predictor(**kwargs))
        if not completions:
            fail("All chain attempts failed")
            raise click.ClickException("all M completion attempts failed")
        try:
            result = judge(completions, **kwargs)
        except (RuntimeError, ValueError, KeyError, TypeError) as e:
            fail(f"MultiChainComparison judge failed: {e}")
            raise click.ClickException(f"MultiChainComparison judge failed: {e}")

    if _result_fields(result):
        parts = [
            getattr(result, fname, "")
            for fname in _result_fields(result)
            if getattr(result, fname, "") and fname != "rationale"
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
        # build (module, example) pairs for new Parallel API in DSPy 3.3.0b1
        exec_pairs: list[tuple[Any, Any]] = []
        for sig in modules:
            mod = dspy.ChainOfThought(sig)
            ex = dspy.Example(**kwargs).with_inputs(*kwargs.keys())
            exec_pairs.append((mod, ex))
        parallel = dspy.Parallel(
            num_threads=len(modules),
            timeout=timeout,
            straggler_limit=straggler_limit,
        )
        result = parallel(exec_pairs)

    _show_result(result, title="Parallel Results")
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
        # ChainOfThoughtWithHint removed in DSPy 3.3.0b1 — inject hint via instructions
        module = dspy.ChainOfThought(signature)
        if hint and module.predict is not None and module.predict.signature is not None:
            module.predict.signature.instructions = f"Hint: {hint}"
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
