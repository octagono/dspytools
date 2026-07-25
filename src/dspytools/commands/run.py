"""dspytools run — Run DSPy modules (predict, cot, react, etc.)."""

from __future__ import annotations

import importlib
import inspect
import json
import json as _json
import re as _re
import time
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
from dspytools.cli.output import syntax as rich_syntax
from dspytools.cli.output import table as rich_table
from dspytools.cli.rich_config import LLMCommand, LLMGroup, click
from dspytools.config.settings import embedder_kwargs
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


def _show_result(
    result: Any,
    *,
    title: str = "Result",
    border: str = _RESULT,
) -> None:
    """Display a DSPy prediction in a panel with labelled output fields."""
    if hasattr(result, "_output_field_names"):
        names = result._output_field_names
        lines = [
            f"[{_CYAN}]{fname}:[/]  {getattr(result, fname, '')}"
            for fname in names
            if getattr(result, fname, "")
        ]
        content = "\n".join(lines) if lines else str(result)
    else:
        content = str(result)
    rich_panel(title, content, border_style=border)


def _show_answer(result_or_str: Any) -> None:
    """Show a simple answer in a panel."""
    if hasattr(result_or_str, "_output_field_names"):
        _show_result(result_or_str)
        return
    rich_panel("Answer", str(result_or_str), border_style=_RESULT)


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
    from dspy.adapters.baml_adapter import BAMLAdapter

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
        "baml": BAMLAdapter(use_native_function_calling=False),
    }
    adapter_obj = adapters.get(adapter)

    sp = rich_spinner()
    with sp:
        sp.add_task(f"Running {module_type}...")
        with dspy.context(adapter=adapter_obj or adapters["baml"]):
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
    "react-v2": ("ReActV2", "dspy.ReActV2(signature, tools=[...])"),
    "pot": ("ProgramOfThought", "dspy.ProgramOfThought(signature)"),
    "code-act": ("CodeAct", "dspy.CodeAct(signature, tools=[...])"),
    "rlm": ("RLM", "dspy.RLM(signature, max_iters=20)"),
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
    """Run a ReActV2 agent with tools.  When the model does not produce
    ``submit`` tool calls, the CLI extracts the answer from its last
    output (``next_thought``) automatically."""

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

    sp = rich_spinner()
    with sp:
        sp.add_task("Thinking...")
        result = dspy.ReActV2(signature, tools=dspy_tools, max_iters=max_iters)(
            **kwargs
        )

    term = getattr(result, "termination_reason", "unknown")
    answer = getattr(result, "answer", None)

    if answer is not None:
        _show_answer(answer)
    elif term in ("empty_tool_calls", "parse_error", "failed", "max_iters"):
        fallback = None

        # 1) Try the ReActV2 history
        history = getattr(result, "history", None)
        last_tool_result = None
        last_thought = None
        if history and history.messages:
            for msg in reversed(history.messages):
                tc = msg.get("tool_calls")
                if tc is not None:
                    tcr = getattr(tc, "tool_call_results", None)
                    if tcr and tcr.tool_call_results:
                        last_tool_result = str(tcr.tool_call_results[-1].value)
                thought = msg.get("next_thought")
                if thought and last_thought is None:
                    last_thought = thought
        fallback = last_tool_result or last_thought

        # 2) Fall back to raw LM output
        fallback = fallback or _extract_reply_from_lm_history()

        if fallback:
            rich_panel("Answer", fallback, border_style=_RESULT)
        else:
            info(f"[{_YELLOW}]ReActV2 terminated with[/] [bold]{term}[/]")

        used = [t.name for t in dspy_tools] if dspy_tools else []
        if used:
            info(f"Tools: [{_DIM}]{', '.join(f'[bold]{n}[/]' for n in used)}[/]")
    else:
        info(f"[{_YELLOW}]Unexpected termination:[/] {term}")

    _show_timing(t0)


def _extract_reply_from_lm_history() -> str | None:
    """Extract the model's ``next_thought`` from the raw LM output history."""

    raw_hist = (
        getattr(dspy.settings.lm, "history", None)
        if hasattr(dspy, "settings")
        else None
    )
    if not raw_hist:
        return None
    for call in reversed(raw_hist):
        for output in reversed(call.get("outputs") or []):
            if not isinstance(output, str):
                continue
            m = _re.search(
                r"\[\[ ## next_thought ## \]\]\s*(.+?)\s*\[\[ ## tool_calls ## \]\]",
                output,
                _re.DOTALL,
            )
            if m:
                return m.group(1).strip()
            parsed = _json.loads(output)
            nt = parsed.get("next_thought")
            if nt:
                return nt.strip()
    return None


# ── pot ──────────────────────────────────────────────────────────────────


@run_cmd.command(name="pot", cls=LLMCommand)
@click.argument("signature")
@click.option("--inputs", "-i", multiple=True, help="Inputs as KEY=VALUE")
@click.option("--max-iters", default=3, type=int)
@click.option("--lm", help="LM model to use")
def run_pot(signature: str, inputs: tuple[str, ...], max_iters: int, lm: str | None):
    """Run a ProgramOfThought module (generates and executes Python code)."""
    _show_command_header("pot")
    _show_inputs(_parse_inputs(inputs))
    _show_model_info(lm)
    info(f"Max iters: [{_DIM}]{max_iters}[/]")

    t0 = time.time()
    setup_dspy(model=lm)
    kwargs = _parse_inputs(inputs)

    sp = rich_spinner()
    with sp:
        sp.add_task("Generating & executing code...")
        result = dspy.ProgramOfThought(signature, max_iters=max_iters)(**kwargs)

    answer = getattr(result, "answer", None)
    reasoning = getattr(result, "reasoning", None)

    if reasoning:
        if reasoning.strip().startswith("def ") or "import " in reasoning[:200]:
            rich_syntax(reasoning, lang="python")
        else:
            info(
                f"Reasoning: [{_DIM}]{reasoning[:300]}...[/]"
                if len(reasoning) > 300
                else f"Reasoning: {reasoning}"
            )
    if answer is not None:
        rich_panel("Answer", str(answer), border_style=_CODE)
    else:
        _show_result(result, border=_CODE)
    _show_timing(t0)


# ── code-act ─────────────────────────────────────────────────────────────


@run_cmd.command(name="code-act", cls=LLMCommand)
@click.argument("signature")
@click.option("--inputs", "-i", multiple=True, help="Inputs as KEY=VALUE")
@click.option("--tools", "-t", multiple=True, help="Tool names (plain functions only)")
@click.option("--max-iters", default=5, type=int)
@click.option("--lm", help="LM model to use")
def run_code_act(
    signature: str,
    inputs: tuple[str, ...],
    tools: tuple[str, ...],
    max_iters: int,
    lm: str | None,
):
    """Run a CodeAct module (code generation + tool execution)."""
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
        sp.add_task("Running code-act...")
        result = dspy.CodeAct(signature, tools=dspy_tools, max_iters=max_iters)(
            **kwargs
        )

    answer = getattr(result, "answer", None)
    if answer is not None:
        rich_panel("Answer", str(answer), border_style=_CODE)
    else:
        _show_result(result, border=_CODE)
    _show_timing(t0)


# ── rlm ──────────────────────────────────────────────────────────────────


@run_cmd.command(name="rlm", cls=LLMCommand)
@click.argument("signature")
@click.option("--inputs", "-i", multiple=True, help="Inputs as KEY=VALUE")
@click.option("--max-iters", default=10, type=int)
@click.option("--max-llm-calls", default=30, type=int)
@click.option("--lm", help="Main LM")
@click.option("--sub-lm", help="Sub-LLM for llm_query() calls")
def run_rlm(
    signature: str,
    inputs: tuple[str, ...],
    max_iters: int,
    max_llm_calls: int,
    lm: str | None,
    sub_lm: str | None,
):
    """Run a Recursive Language Model (RLM) module."""
    _show_command_header("rlm")
    _show_inputs(_parse_inputs(inputs))
    _show_model_info(lm)
    if sub_lm:
        info(f"Sub-LM: [{_DIM}]{sub_lm}[/]")
    info(
        f"Iterations: [{_DIM}]{max_iters}[/]   "
        f"Max LLM calls: [{_DIM}]{max_llm_calls}[/]"
    )

    t0 = time.time()
    setup_dspy(model=lm, temperature=0.2, max_tokens=4096)
    kwargs = _parse_inputs(inputs)
    sub_lm_obj = LMRegistry.get(model=sub_lm, temperature=0.0) if sub_lm else None

    sp = rich_spinner()
    with sp:
        sp.add_task("Running RLM...")
        result = dspy.RLM(
            signature,
            max_iters=max_iters,
            max_llm_calls=max_llm_calls,
            sub_lm=sub_lm_obj,
        )(**kwargs)

    _show_answer(result)
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

    if hasattr(result, "_output_field_names"):
        parts = [
            getattr(result, fname, "")
            for fname in result._output_field_names
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
def run_parallel(modules: tuple[str, ...], inputs: tuple[str, ...], lm: str | None):
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
        parallel = dspy.Parallel(num_threads=len(modules))
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
