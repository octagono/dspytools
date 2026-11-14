"""DSPy 3.3.1 -> 3.5.0 migration regression tests.

Everything here is offline: scripted ``dspy.utils.DummyLM`` answers or a stub
engine (``complete(Request) -> Response``) passed as ``dspy.LM(engine=...)``.
No socket is opened, no config file is read (``load_config``/``load_env`` are
patched per test) and the LM registry is cleared around every test.

Coverage:

1. ``model_type`` injection + config / per-section overrides.
2. ``reasoning_effort`` translation from legacy ``enable_thinking``.
3. LM instance cache keying (never share two differently-configured LMs).
4. ``cli/llm_help.py`` builds a canonical ``dspy.lm15.Request``.
5. Adapter selection from ``dspy.adapter``.
6. ReAct's 3.5 Prediction contract (``history`` + ``termination_reason``).
7. The reasoning-tolerant ``JSONAdapter.parse``/``BAMLAdapter.parse`` patch.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import rich_click as click

from dspytools.core._dspy import dspy


# ── Hermetic scaffolding ───────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _hermetic_setup(monkeypatch):
    """No config file, no .env, no cached LM instances across tests."""
    import dspytools.core.setup as setup_mod

    state: dict = {"cfg": {}}
    monkeypatch.setattr(setup_mod, "load_config", lambda: state["cfg"])
    monkeypatch.setattr(setup_mod, "load_env", lambda: {})
    monkeypatch.setattr(setup_mod, "merge_environ", lambda env=None: None)
    setup_mod.LMRegistry.clear()
    yield state
    setup_mod.LMRegistry.clear()


class _RecordingEngine:
    """Stub sync engine: records the canonical request, returns fixed text."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.requests: list = []

    def complete(self, request):
        assert isinstance(request, dspy.lm15.Request)
        self.requests.append(request)
        return dspy.lm15.Response(
            id=None,
            model=request.model,
            message=dspy.lm15.Message.assistant([dspy.lm15.TextPart(self.text)]),
            finish_reason="stop",
            usage=dspy.lm15.Usage(input_tokens=0, output_tokens=0, total_tokens=0),
        )


def _stub_lm(text: str = "stub answer") -> tuple:
    engine = _RecordingEngine(text)
    return dspy.LM("openai/gpt-4o", engine=engine, cache=False), engine


def _run_setup_dspy(monkeypatch, state, *, adapter: str | None = None, cfg=None):
    """Call ``setup_dspy()`` hermetically and return the ``dspy.configure`` kwargs."""
    import dspytools.core.setup as setup_mod

    state["cfg"] = (
        cfg
        if cfg is not None
        else ({"dspy": {"adapter": adapter}} if adapter is not None else {})
    )
    lm, _ = _stub_lm()
    monkeypatch.setattr(setup_mod, "_setup_configured", False)
    monkeypatch.setattr(
        setup_mod.LMRegistry, "get_or_default", classmethod(lambda cls: lm)
    )
    captured: dict = {}
    monkeypatch.setattr(dspy, "configure", lambda **kwargs: captured.update(kwargs))
    setup_mod.setup_dspy()
    return captured


# ── 1. model_type ─────────────────────────────────────────────────────────


def test_model_type_defaults_to_chat():
    from dspytools.core.setup import LMRegistry

    assert LMRegistry.get(model="openai/gpt-4o").model_type == "chat"


@pytest.mark.parametrize("value", ["responses", "auto", "text"])
def test_model_type_config_override_survives(_hermetic_setup, value):
    from dspytools.core.setup import LMRegistry

    _hermetic_setup["cfg"] = {"dspy": {"model_type": value}}
    assert LMRegistry.get(model="openai/gpt-4o").model_type == value


def test_model_type_per_section_overrides_config(_hermetic_setup):
    from dspytools.core.setup import LMRegistry

    _hermetic_setup["cfg"] = {
        "dspy": {"model_type": "chat"},
        "lm": {"student": {"model": "openai/gpt-4o", "model_type": "responses"}},
    }
    assert LMRegistry.get_or_default().model_type == "responses"


# ── 2. reasoning_effort mapping ───────────────────────────────────────────


def test_reasoning_kwargs_mapping():
    from dspytools.core.setup import _reasoning_kwargs

    assert _reasoning_kwargs({}) == {}
    assert _reasoning_kwargs({"enable_thinking": False}) == {"reasoning_effort": "none"}
    assert _reasoning_kwargs({"enable_thinking": True}) == {}
    assert _reasoning_kwargs({"reasoning_effort": "high"}) == {
        "reasoning_effort": "high"
    }
    # Explicit reasoning_effort always wins over the legacy flag.
    assert _reasoning_kwargs({"reasoning_effort": "low", "enable_thinking": False}) == {
        "reasoning_effort": "low"
    }


@pytest.mark.parametrize(
    ("student", "expected"),
    [
        ({}, None),
        ({"enable_thinking": False}, "none"),
        ({"enable_thinking": True}, None),
        ({"reasoning_effort": "high", "enable_thinking": False}, "high"),
    ],
)
def test_reasoning_effort_reaches_the_lm(_hermetic_setup, student, expected):
    from dspytools.core.setup import LMRegistry

    _hermetic_setup["cfg"] = {"lm": {"student": {"model": "openai/gpt-4o", **student}}}
    lm = LMRegistry.get_or_default()

    assert lm.kwargs.get("reasoning_effort") == expected
    # DSPy 3.5 rejects chat_template_kwargs outright; it must never be forwarded.
    assert "chat_template_kwargs" not in lm.kwargs


# ── 3. LM instance cache ──────────────────────────────────────────────────


def test_identical_configs_reuse_one_lm():
    from dspytools.core.setup import LMRegistry

    first = LMRegistry.get(model="openai/gpt-4o", temperature=0.2)
    second = LMRegistry.get(model="openai/gpt-4o", temperature=0.2)

    assert first is second
    assert len(LMRegistry._instances) == 1


def test_different_thinking_configs_never_share_an_lm():
    from dspytools.core.setup import LMRegistry

    without = LMRegistry.get(model="openai/gpt-4o")
    with_none = LMRegistry.get(model="openai/gpt-4o", reasoning_effort="none")

    assert without is not with_none
    assert with_none.kwargs["reasoning_effort"] == "none"
    assert "reasoning_effort" not in without.kwargs
    assert len(LMRegistry._instances) == 2


def test_cache_key_ignores_api_key_but_sees_behavioral_kwargs():
    from dspytools.core.setup import _lm_cache_key

    base = {"model": "openai/gpt-4o", "temperature": 0.2, "cache": True}
    assert _lm_cache_key("m", {**base, "api_key": "a"}) == _lm_cache_key(
        "m", {**base, "api_key": "b"}
    )
    assert _lm_cache_key("m", base) != _lm_cache_key(
        "m", {**base, "reasoning_effort": "none"}
    )


def test_enable_thinking_configs_produce_distinct_lm_instances(
    monkeypatch, _hermetic_setup
):
    """A config reload flipping enable_thinking must not reuse the cached LM."""
    from dspytools.core.setup import LMRegistry

    def _default_for(cfg):
        _hermetic_setup["cfg"] = cfg
        # Simulate a config reload without dropping the instance cache.
        monkeypatch.setattr(LMRegistry, "_default", None)
        return LMRegistry.get_or_default()

    off = _default_for(
        {"lm": {"student": {"model": "openai/gpt-4o", "enable_thinking": False}}}
    )
    on = _default_for(
        {"lm": {"student": {"model": "openai/gpt-4o", "enable_thinking": True}}}
    )

    assert off is not on
    assert off.kwargs["reasoning_effort"] == "none"
    assert "reasoning_effort" not in on.kwargs
    assert len(LMRegistry._instances) == 2


# ── 4. cli/llm_help.py canonical Request ──────────────────────────────────


def test_llm_help_sends_canonical_request(monkeypatch, tmp_path: Path):
    from dspytools.cli import llm_help
    import dspytools.core.setup as setup_mod

    @click.group(help="Test group.")
    def cli() -> None: ...

    @cli.command(help="Do things.")
    def thing() -> None: ...

    answer = "Run `dspytools thing` to do the thing. It is quite simple."
    lm, engine = _stub_lm(answer)
    monkeypatch.setattr(
        setup_mod.LMRegistry, "get_or_default", classmethod(lambda cls: lm)
    )

    # Keep the disk cache out of the real home directory.
    monkeypatch.setattr(llm_help, "_DISK_CACHE_PATH", tmp_path / "help_cache.json")
    monkeypatch.setattr(llm_help, "_mem_cache", {})
    monkeypatch.setattr(llm_help, "_mem_loaded", True)

    ctx = click.Context(cli, info_name="dspytools")
    text = llm_help._llm_help(ctx)

    assert text == answer
    assert len(engine.requests) == 1
    request = engine.requests[0]
    assert isinstance(request, dspy.lm15.Request)
    assert request.system is None
    assert [m.role for m in request.messages] == ["user"]
    prompt_text = "".join(
        part.text for part in request.messages[-1].parts if hasattr(part, "text")
    )
    assert "Command: dspytools" in prompt_text
    assert request.config.max_tokens == 256
    assert request.config.temperature == 0.3
    # The cached copy short-circuits the second call — no extra LM turn.
    assert llm_help._llm_help(ctx) == answer
    assert len(engine.requests) == 1


def test_llm_help_returns_none_for_short_answer(monkeypatch, tmp_path: Path):
    from dspytools.cli import llm_help
    import dspytools.core.setup as setup_mod

    @click.group(help="Test group.")
    def cli() -> None: ...

    lm, _ = _stub_lm("too short")
    monkeypatch.setattr(
        setup_mod.LMRegistry, "get_or_default", classmethod(lambda cls: lm)
    )
    monkeypatch.setattr(llm_help, "_DISK_CACHE_PATH", tmp_path / "help_cache.json")
    monkeypatch.setattr(llm_help, "_mem_cache", {})
    monkeypatch.setattr(llm_help, "_mem_loaded", True)

    assert llm_help._llm_help(click.Context(cli, info_name="dspytools")) is None


# ── 5. Adapter selection ──────────────────────────────────────────────────


def test_adapter_selection_from_config(monkeypatch, _hermetic_setup):
    from dspy.adapters.baml_adapter import BAMLAdapter

    expected = {
        "baml": BAMLAdapter,
        "chat": dspy.ChatAdapter,
        "json": dspy.JSONAdapter,
        "xml": dspy.XMLAdapter,
    }
    for name, cls in expected.items():
        captured = _run_setup_dspy(monkeypatch, _hermetic_setup, adapter=name)
        assert isinstance(captured["adapter"], cls), name
        assert captured["lm"] is not None


def test_default_adapter_is_baml_without_native_function_calling(
    monkeypatch, _hermetic_setup
):
    from dspy.adapters.baml_adapter import BAMLAdapter

    captured = _run_setup_dspy(monkeypatch, _hermetic_setup, cfg={})
    adapter = captured["adapter"]
    assert isinstance(adapter, BAMLAdapter)
    assert adapter.use_native_function_calling is False
    assert "warn_on_type_mismatch" not in captured


# ── 6. ReAct contract on DSPy 3.5 ─────────────────────────────────────────


def test_react_returns_history_and_termination_reason():
    lm = dspy.utils.DummyLM(
        [
            {
                "next_thought": "I already know the answer",
                "tool_calls": {"name": "submit", "args": {"answer": "4"}},
            }
        ]
    )
    program = dspy.ReAct("question -> answer: str", tools=[], max_iters=2)

    with dspy.context(lm=lm, adapter=dspy.ChatAdapter()):
        result = program(question="What is 2+2?")

    assert result.answer == "4"
    assert isinstance(result.history, dspy.History)
    assert result.termination_reason == "submit"
    assert len(result.history.messages) == 1
    event = result.history.messages[0]
    assert set(event) >= {"question", "next_thought", "tool_calls", "answer"}
    assert event["answer"] == "4"


def test_react_rejects_reserved_prediction_field_names():
    with pytest.raises(ValueError, match="reserved"):
        dspy.ReAct("question -> history: str", tools=[])


# ── 7. reasoning-tolerant parse patch ─────────────────────────────────────


@pytest.fixture
def patched_parsers(monkeypatch, _hermetic_setup):
    """Install the setup.py adapter patches (idempotent within a process)."""
    _run_setup_dspy(monkeypatch, _hermetic_setup, cfg={})
    return True


def _reasoning_signature():
    return dspy.Signature(
        {
            "question": (str, dspy.InputField()),
            "reasoning": (dspy.Reasoning, dspy.OutputField()),
            "answer": (str, dspy.OutputField()),
        }
    )


def test_parse_tolerates_missing_reasoning_field(patched_parsers):
    from dspy.adapters.baml_adapter import BAMLAdapter

    signature = _reasoning_signature()
    completion = '{"question": "2+2?", "answer": "4"}'

    for adapter in (dspy.JSONAdapter(), BAMLAdapter(use_native_function_calling=False)):
        parsed = adapter.parse(signature, completion)
        assert parsed["answer"] == "4"


def test_parse_still_requires_non_reasoning_fields(patched_parsers):
    from dspy.adapters.baml_adapter import BAMLAdapter
    from dspy.utils.exceptions import AdapterParseError

    signature = _reasoning_signature()
    completion = '{"question": "2+2?", "reasoning": "because"}'

    for adapter in (dspy.JSONAdapter(), BAMLAdapter(use_native_function_calling=False)):
        with pytest.raises(AdapterParseError):
            adapter.parse(signature, completion)


def test_parse_accepts_baml_section_completion(patched_parsers):
    signature = _reasoning_signature()
    completion = "[[ ## answer ## ]]\n4"

    parsed = dspy.JSONAdapter().parse(signature, completion)
    assert parsed["answer"] == "4"


# ── 8. compiled-program provenance ──────────────────────────────────────


def _write_program(
    path: Path, version: str | None, *, dependency_map: bool = False
) -> Path:
    payload: dict = {"answer.predict": {"demos": []}}
    if version is not None:
        payload["metadata"] = (
            {"dependency_versions": {"python": "3.12", "dspy": version}}
            if dependency_map
            else {"dspy_version": version}
        )
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_saved_dspy_version_reads_metadata(tmp_path):
    from dspytools.core.programs import saved_dspy_version

    assert (
        saved_dspy_version(_write_program(tmp_path / "p.json", "3.3.0b1")) == "3.3.0b1"
    )
    # dspy records it under metadata.dependency_versions on some save paths.
    assert (
        saved_dspy_version(
            _write_program(tmp_path / "d.json", "3.3.0b1", dependency_map=True)
        )
        == "3.3.0b1"
    )
    assert saved_dspy_version(_write_program(tmp_path / "n.json", None)) is None
    assert saved_dspy_version(tmp_path / "missing.json") is None


def test_is_current_dspy_version(tmp_path):
    from dspytools.core.programs import is_current_dspy_version

    assert is_current_dspy_version(
        _write_program(tmp_path / "a.json", "3.5.0"), "3.5.0"
    )
    assert not is_current_dspy_version(
        _write_program(tmp_path / "b.json", "3.3.0b1"), "3.5.0"
    )
    # No recorded version: load it rather than refuse — DSPy warns on its own.
    assert is_current_dspy_version(_write_program(tmp_path / "c.json", None), "3.5.0")


def test_stale_help_cache_is_ignored(tmp_path, monkeypatch):
    """A cache compiled by another DSPy must not be served as help."""
    from dspytools.help.optimize import AutoCompiler

    cache = _write_program(tmp_path / "compiled.json", "3.3.0b1")
    monkeypatch.setattr(AutoCompiler, "CACHE_PATH", cache)
    monkeypatch.setattr(AutoCompiler, "META_PATH", tmp_path / "meta.json")

    assert AutoCompiler.compile_if_needed(cli=None) is None
    assert not AutoCompiler.is_compiled()
