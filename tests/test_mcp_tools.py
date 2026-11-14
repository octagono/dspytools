"""Regression tests for the MCP tool layer.

Covers four defects fixed together:

1. lossy MCP tool schemas (`mcp/loader.py`) — descriptions, types, and
   `required` must survive the conversion to `dspy.Tool`;
2. hyphens/digits in MCP tool names (`mcp/loader.py`) — sanitized to a valid
   Python identifier for `dspy.RLM`/`dspy.Flex`, with collisions deduped;
3. `tool_agent_run` (`mcp/tools.py`) — a failed forced submit returns a
   structured error instead of propagating;
4. `_make_optimizer_instance("gepa")` (`mcp/tools.py`) — five-arg metric plus a
   teacher reflection LM, with an actionable error when no teacher is set.

No LM calls and no network: the agent and optimizer paths are stubbed.
"""

from __future__ import annotations

import json

import pytest

from dspytools.core._dspy import dspy
from dspytools.core.setup import LMRegistry
from dspytools.mcp import tools as mcp_tools
from dspytools.mcp.loader import _make_sync_tool, _sanitize_tool_name


class _StubSession:
    """Stand-in for an `mcp.ClientSession` — never called in these tests."""

    async def call_tool(self, name, arguments=None):  # pragma: no cover - stub
        raise AssertionError(f"unexpected MCP call: {name}")


# ── 1. Schema fidelity ────────────────────────────────────────────────────

_SCHEMA = {
    "type": "object",
    "properties": {
        "path": {"type": "string", "description": "Repository path"},
        "limit": {"type": "integer"},
    },
    "required": ["path"],
}


def test_make_sync_tool_preserves_schema_semantics():
    tool = _make_sync_tool(
        name="read_path",
        description="Read a path",
        input_schema=_SCHEMA,
        session=_StubSession(),
    )

    assert tool.name == "read_path"
    # Descriptions reach the model
    assert tool.args["path"]["description"] == "Repository path"
    assert tool.arg_desc["path"] == "Repository path (Required)"
    assert tool.arg_desc["limit"] == "No description provided."
    # Types survive as real Python types
    assert tool.arg_types == {"path": str, "limit": int}

    parameters = tool.as_function_tool().parameters
    assert parameters["properties"]["path"]["type"] == "string"
    assert parameters["properties"]["limit"]["type"] == "integer"
    # The optional argument is NOT advertised as required
    assert parameters["required"] == ["path"]


def test_make_sync_tool_keeps_declared_default():
    tool = _make_sync_tool(
        name="read_path",
        description="Read a path",
        input_schema={
            "type": "object",
            "properties": {"limit": {"type": "integer", "default": 10}},
        },
        session=_StubSession(),
    )
    assert tool.as_function_tool().parameters["properties"]["limit"]["default"] == 10
    assert tool.as_function_tool().parameters["required"] == []


# ── 2. Identifier sanitization ────────────────────────────────────────────


def test_sanitize_hyphenated_name():
    taken: set[str] = set()
    assert _sanitize_tool_name("git-mcp_list_files", taken) == "git_mcp_list_files"
    # deterministic + collision-free
    assert _sanitize_tool_name("git-mcp_list_files", taken) == "git_mcp_list_files_2"


def test_sanitize_digit_leading_and_keyword_names():
    taken: set[str] = set()
    assert _sanitize_tool_name("2fast_tool", taken) == "_2fast_tool"
    assert _sanitize_tool_name("class", taken) == "class_"
    assert _sanitize_tool_name("!!!", taken) == "___"


def test_sanitized_tool_is_accepted_by_rlm():
    tool = _make_sync_tool(
        name="git-mcp_list_files",
        description="List files",
        input_schema={"type": "object", "properties": {}},
        session=_StubSession(),
    )
    assert tool.name == "git_mcp_list_files"
    # The original MCP name stays visible to the model
    assert "git-mcp_list_files" in tool.desc
    # dspy.RLM/Flex reject invalid identifiers — this must not raise
    dspy.RLM("question -> answer", tools=[tool])


# ── 3. tool_agent_run error payload ───────────────────────────────────────


def test_tool_agent_run_returns_structured_error(monkeypatch):
    signature = dspy.Signature("question -> answer")

    def _boom(*_args, **_kwargs):
        raise dspy.AdapterParseError(
            adapter_name="BAMLAdapter",
            signature=signature,
            lm_response="<answer>",
            message="forced submit failed",
        )

    monkeypatch.setattr(mcp_tools, "_ensure_lm_configured", lambda: None)
    monkeypatch.setattr(dspy, "ReAct", _boom)

    payload = json.loads(mcp_tools.tool_agent_run("hello"))

    assert payload["exception"] == "AdapterParseError"
    assert "AdapterParseError" in payload["error"]
    assert payload["last_model_output"] == ""
    assert "forced submit" in payload["detail"]


# ── 4. GEPA optimizer construction ────────────────────────────────────────


def test_make_optimizer_gepa_requires_teacher(monkeypatch):
    monkeypatch.setattr(LMRegistry, "get_teacher", classmethod(lambda cls: None))

    with pytest.raises(ValueError) as excinfo:
        mcp_tools._make_optimizer_instance("gepa")

    message = str(excinfo.value)
    assert "teacher" in message
    assert "configure lm set teacher" in message


def test_make_optimizer_gepa_constructs_with_teacher(monkeypatch):
    teacher = dspy.utils.DummyLM([{"answer": "reflection"}])
    monkeypatch.setattr(LMRegistry, "get_teacher", classmethod(lambda cls: teacher))

    opt = mcp_tools._make_optimizer_instance("gepa")

    assert isinstance(opt, dspy.GEPA)
    assert opt.reflection_lm is teacher
    # GEPA's required five-argument metric signature
    assert opt.metric_fn(1, 2, None, None, None) is not None


def test_gepa_feedback_metric_accepts_five_args():
    metric = mcp_tools._gepa_feedback_metric
    gold = dspy.Example(output="a")
    assert metric(gold, dspy.Example(output="a"), None, None, None) == 1.0
    assert metric(gold, dspy.Example(output="b"), None, None, None) == 0.0


# ── 5. PythonInterpreter lifecycle (commands/tool.py) ─────────────────────


def test_python_interpreter_command_shuts_sandbox_down(monkeypatch):
    from click.testing import CliRunner

    from dspytools.commands.tool import tool_python_interpreter

    state: dict[str, int] = {"shutdown": 0}

    class _FakeInterpreter:
        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            state["shutdown"] += 1

        def execute(self, code):
            return "42"

    monkeypatch.setattr(dspy, "PythonInterpreter", _FakeInterpreter)

    result = CliRunner().invoke(tool_python_interpreter, ["print(42)"])

    assert result.exit_code == 0
    assert state["shutdown"] == 1
    assert "42" in result.output
