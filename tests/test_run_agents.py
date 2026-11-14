"""Hermetic tests for the `dspytools run` agent and sampling commands.

Everything here is offline: a stub ``complete(Request) -> Response`` engine
(``dspy.LM(..., engine=...)``) or ``dspy.utils.DummyLM``. ``setup_dspy`` is
patched out and every command runs inside an explicit ``dspy.context``, so no
config file, env var, socket or MCP server is touched.

Coverage:

1. ``multi-chain`` samples distinct rollouts instead of one cached draft.
2. ``parallel`` surfaces failed examples and their exceptions.
3. ``_build_dspy_tool`` maps keyword arguments without positional sliding.
4. A failed agent run becomes a ``ClickException`` naming the last output.
5. ``react`` history save/replay round-trip.
6. Chain-of-thought reasoning never counts as a result field.
7. ``cot-hint`` appends instructions without mutating the signature.
8. ``best-of-n``/``refine`` stop lying about sample counts.
9. Tool loading/sanitizing for RLM and Flex.
"""

from __future__ import annotations

import json
import math
import re

import pytest
from click import ClickException
from click.testing import CliRunner

from dspytools.commands import run as run_mod
from dspytools.config.settings import data_dir
from dspytools.core._dspy import dspy

# ── Hermetic scaffolding ───────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _no_setup_dspy(monkeypatch):
    """Never read config or configure a real LM from the commands."""
    monkeypatch.setattr(run_mod, "setup_dspy", lambda *args, **kwargs: None)


@pytest.fixture(autouse=True)
def _memory_only_cache(monkeypatch):
    """Keep DSPy's cache in memory so a test can prove cache bypass."""
    monkeypatch.setattr(dspy.cache, "enable_disk_cache", False)
    monkeypatch.setattr(dspy.cache, "disk_cache", {})
    dspy.cache.reset_memory_cache()
    yield
    dspy.cache.reset_memory_cache()


def _prompt_text(request) -> str:
    """Flatten a canonical Request into one searchable string."""
    return "\n".join(
        getattr(part, "text", "") or ""
        for message in request.messages
        for part in message.parts
    )


def _output_fields(text: str) -> list[str]:
    """Output fields the ChatAdapter asked for, in declared order."""
    marker = "Respond with the corresponding output fields"
    block = text[text.index(marker) :] if marker in text else text
    return list(dict.fromkeys(re.findall(r"\[\[ ## (\w+) ## \]\]", block)))


class _ScriptedEngine:
    """Stub sync engine: answers the fields ChatAdapter asks for."""

    def __init__(self, fail_if=None) -> None:
        self.requests: list = []
        self.fail_if = fail_if

    def complete(self, request):
        text = _prompt_text(request)
        if self.fail_if is not None and self.fail_if(text):
            raise RuntimeError("stub engine failure")
        self.requests.append(request)
        attempt = len(self.requests)
        fields = _output_fields(text) or ["answer"]
        body = "".join(
            f"[[ ## {name} ## ]]\n"
            f"{f'step {attempt}' if name in ('plan', 'rationale', 'reasoning') else f'answer {attempt}'}"
            "\n\n"
            for name in fields
        )
        return dspy.lm15.Response(
            id=None,
            model=request.model,
            message=dspy.lm15.Message.assistant([dspy.lm15.TextPart(body)]),
            finish_reason="stop",
            usage=dspy.lm15.Usage(input_tokens=0, output_tokens=0, total_tokens=0),
        )


def _stub_lm(engine: _ScriptedEngine, *, cache: bool = True):
    return dspy.LM("openai/stub-model", engine=engine, cache=cache)


def _invoke(args: list[str], lm) -> object:
    with dspy.context(lm=lm, adapter=dspy.ChatAdapter()):
        return CliRunner().invoke(run_mod.run_cmd, args)


# ── 1. multi-chain ─────────────────────────────────────────────────────────


def test_multi_chain_samples_distinct_rollouts():
    """M chains must be M LM calls, not M cache hits of one draft."""
    engine = _ScriptedEngine()
    result = _invoke(
        ["multi-chain", "question -> answer", "-i", "question=q", "-m", "3"],
        _stub_lm(engine),
    )

    assert result.exit_code == 0, result.output
    # 3 drafts + 1 judge call. A cached loop would have made 2 calls.
    assert len(engine.requests) == 4


# ── 2. parallel ────────────────────────────────────────────────────────────


def test_parallel_surfaces_failed_examples():
    engine = _ScriptedEngine(fail_if=lambda text: "[[ ## summary ## ]]" in text)
    result = _invoke(
        [
            "parallel",
            "-m",
            "question -> answer",
            "-m",
            "question -> summary",
            "-i",
            "question=q",
        ],
        _stub_lm(engine),
    )

    assert result.exit_code == 0, result.output
    assert "1/2 modules failed" in result.output
    assert "stub engine failure" in result.output
    assert "question" in result.output


# ── 3. _build_dspy_tool ────────────────────────────────────────────────────


def test_build_dspy_tool_maps_kwargs_with_omitted_optional_arg():
    """An omitted earlier optional arg must not shift later values left."""
    tool = run_mod._build_dspy_tool(round)

    assert tool(number=3.14159, ndigits=2) == 3.14
    assert tool(number=3.14159) == 3


def test_build_dspy_tool_supports_positional_only_builtins():
    tool = run_mod._build_dspy_tool(math.sqrt)

    assert tool(x=9.0) == 3.0


def test_build_dspy_tool_does_not_coerce_strings():
    """dspy.Tool does the coercion; no ad-hoc ``int()``/``float()`` guessing."""
    tool = run_mod._build_dspy_tool(json.dumps)

    assert tool(obj={"a": 1}) == '{"a": 1}'


# ── 4. agent failures ──────────────────────────────────────────────────────


def _parse_error() -> dspy.AdapterParseError:
    return dspy.AdapterParseError(
        "ChatAdapter", dspy.Signature("question -> answer"), "nothing usable"
    )


def test_forced_submit_failure_becomes_click_exception():
    exc = run_mod._agent_failure(_parse_error())

    assert isinstance(exc, ClickException)
    assert "failed to parse the LM response" in str(exc)


def test_run_agent_converts_parse_errors():
    """AdapterParseError is not a ValueError in 3.5 — it must be handled."""

    class _Failing:
        def __call__(self, **kwargs):
            raise _parse_error()

    assert not issubclass(dspy.AdapterParseError, ValueError)
    with pytest.raises(ClickException, match="failed to parse the LM response"):
        run_mod._run_agent(_Failing(), {})


def test_run_agent_converts_context_window_errors():
    class _Failing:
        def __call__(self, **kwargs):
            raise dspy.ContextWindowExceededError(message="stub overflow")

    with pytest.raises(ClickException, match="stub overflow"):
        run_mod._run_agent(_Failing(), {})


# ── 5. react history round-trip ────────────────────────────────────────────


def _react_lm(answer: str):
    return dspy.utils.DummyLM(
        [
            {
                "next_thought": "I know it",
                "tool_calls": {"name": "submit", "args": {"answer": answer}},
            }
        ]
    )


def test_react_history_save_and_replay():
    signature = "question -> answer: str"
    first = _invoke(["react", signature, "-i", "question=2+2?"], _react_lm("4"))

    assert first.exit_code == 0, first.output
    saved = sorted(data_dir().glob("react_history_*.json"))
    assert saved, first.output

    history = dspy.History.model_validate_json(saved[-1].read_text(encoding="utf-8"))
    assert len(history.messages) == 1
    assert history.messages[0]["answer"] == "4"

    second = _invoke(
        ["react", signature, "-i", "question=2+3?", "--resume-history", str(saved[-1])],
        _react_lm("5"),
    )

    assert second.exit_code == 0, second.output
    assert "Resumed" in second.output


def test_react_result_fields_exclude_meta_fields():
    with dspy.context(lm=_react_lm("4"), adapter=dspy.ChatAdapter()):
        prediction = dspy.ReAct("question -> answer: str", tools=[])(question="2+2?")

    assert run_mod._result_fields(prediction) == ["answer"]


# ── 6. reasoning is never a result field ───────────────────────────────────


def test_result_fields_exclude_reasoning():
    engine = _ScriptedEngine()
    lm = _stub_lm(engine, cache=False)
    with dspy.context(lm=lm, adapter=dspy.ChatAdapter()):
        prediction = dspy.ChainOfThought("question -> answer")(question="q")

    assert set(prediction._store) >= {"reasoning", "answer"}
    assert run_mod._result_fields(prediction) == ["answer"]


def test_show_result_renders_reasoning_preview_separately():
    prediction = dspy.Prediction(reasoning="a" * 500, answer="42")

    preview = run_mod._reasoning_preview(prediction)

    assert preview.endswith("…")
    assert len(preview) == run_mod._REASONING_PREVIEW_CHARS + 1


# ── 7. cot-hint ────────────────────────────────────────────────────────────


def test_cot_hint_appends_without_mutating_the_signature():
    signature = dspy.Signature("question -> answer")
    original_instructions = signature.instructions

    module = run_mod._cot_with_hint(signature, "be brief")
    hinted = module.predict.signature

    assert hinted is not signature
    assert original_instructions in hinted.instructions
    assert "Hint: be brief" in hinted.instructions
    # The caller's signature class is untouched.
    assert signature.instructions == original_instructions
    assert (
        "Hint: be brief"
        not in dspy.ChainOfThought(signature).predict.signature.instructions
    )


def test_cot_hint_command_keeps_task_and_hint_in_the_prompt():
    engine = _ScriptedEngine()
    result = _invoke(
        ["cot-hint", "question -> answer", "-i", "question=q", "--hint", "be brief"],
        _stub_lm(engine, cache=False),
    )

    assert result.exit_code == 0, result.output
    system = engine.requests[0].system or ""
    assert "produce the fields `answer`" in system
    assert "Hint: be brief" in system


# ── 8. best-of-n / refine ──────────────────────────────────────────────────


def test_effective_threshold_disables_early_exit_without_a_quality_signal():
    effective, early_exit = run_mod._effective_threshold(0.8, None)

    assert early_exit is False
    assert effective > 1.0


def test_effective_threshold_keeps_quality_gate():
    effective, early_exit = run_mod._effective_threshold(0.8, "Paris")

    assert early_exit is True
    assert effective == 0.8


def test_reward_fn_is_structural_and_tracks_attempts():
    scored: list[float] = []
    reward = run_mod._reward_fn(None, scored)

    assert reward({}, dspy.Prediction(answer="Paris")) == 1.0
    assert reward({}, dspy.Prediction(answer="")) == 0.0
    assert reward({}, None) == 0.0
    assert scored == [1.0, 0.0, 0.0]

    keyword_scored: list[float] = []
    keyword_reward = run_mod._reward_fn("paris", keyword_scored)
    assert keyword_reward({}, dspy.Prediction(answer="Lyon")) == 0.0
    assert keyword_reward({}, dspy.Prediction(answer="Paris, France")) == 1.0


def test_best_of_n_runs_all_samples_without_a_keyword():
    engine = _ScriptedEngine()
    result = _invoke(
        ["best-of-n", "question -> answer", "-i", "question=q", "-n", "3"],
        _stub_lm(engine),
    )

    assert result.exit_code == 0, result.output
    assert len(engine.requests) == 3
    assert "Attempts: 3/3" in result.output
    assert "Early exit disabled" in result.output


# ── 9. tool loading / sanitizing ───────────────────────────────────────────


def test_load_tools_falls_back_to_python_attribute(monkeypatch):
    """``module:attr`` misses MCP and must still import the Python attribute."""
    monkeypatch.setattr(run_mod, "_load_mcp_tools_for_server", lambda server: [])

    tools = run_mod._load_tools(("json:dumps",))

    assert [tool.name for tool in tools] == ["dumps"]
    assert tools[0](obj={"a": 1}) == '{"a": 1}'


def test_sanitize_tool_names_for_rlm_and_flex():
    def echo(value: str) -> str:
        "Echo a value."
        return value

    tools = [
        dspy.Tool(echo, name="git-mcp_list_files"),
        dspy.Tool(echo, name="git-mcp_list_files"),
        dspy.Tool(echo, name="2weird.name"),
    ]

    sanitized = run_mod._sanitize_tool_names(tools)

    assert [tool.name for tool in sanitized] == [
        "git_mcp_list_files",
        "git_mcp_list_files_2",
        "tool_2weird_name",
    ]
