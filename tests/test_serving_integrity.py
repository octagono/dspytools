"""Serving / evaluation / persistence integrity tests.

Covers the defects fixed in this lane:

- the API process configures a DSPy LM at startup (and refuses to serve blind),
- ``/config/models/*`` re-applies the LM inside the running process,
- ``/infer`` runs the blocking, LM-bound call off the event loop,
- the semantic cache serializes ``dspy.Reasoning`` instead of raising TypeError,
- drift/quality scoring grades the answer, not the chain of thought,
- hot-swap loading reconstructs the saved module shape and checks provenance,
- ``module new --signature`` uses ``Signature.signature`` as the prompt string,
- ``evaluate`` reconstructs compiled runs and scores the program's real field.

All LMs are ``dspy.utils.DummyLM`` — no network, no live LM.
"""

from __future__ import annotations

import copy
import json
from collections import deque
from pathlib import Path

import pytest
from click.testing import CliRunner
from fastapi.testclient import TestClient

from dspytools.core._dspy import dspy


@pytest.fixture(autouse=True)
def _fresh_dspy_settings():
    """Give every test a clean DSPy settings slate.

    DSPy allows ``dspy.configure`` from exactly one thread; the fixture resets
    the recorded owner so a test can configure from the main thread and the next
    one from a ``TestClient`` portal thread without cross-test failure.
    """
    import importlib

    from dspytools.api import server as api_server

    dspy_settings = importlib.import_module("dspy.dsp.utils.settings")
    saved_config = copy.deepcopy(dspy_settings.main_thread_config)
    saved_owner = dspy_settings.config_owner_thread_id
    saved_task = dspy_settings.config_owner_async_task
    saved_serving = api_server._serving_lm

    dspy_settings.main_thread_config["lm"] = None
    dspy_settings.config_owner_thread_id = None
    dspy_settings.config_owner_async_task = None
    api_server._serving_lm = None
    api_server._lm_configured = False
    api_server._hotswap = None
    yield
    dspy_settings.main_thread_config.clear()
    dspy_settings.main_thread_config.update(saved_config)
    dspy_settings.config_owner_thread_id = saved_owner
    dspy_settings.config_owner_async_task = saved_task
    api_server._serving_lm = saved_serving
    api_server._lm_configured = False
    api_server._hotswap = None


# ── helpers ──────────────────────────────────────────────────────────


class FakeCache:
    """Minimal SemanticCache stand-in: JSON-decoding store, always-miss check."""

    def __init__(self):
        self.stored: dict[str, str] = {}

    def check(self, key):
        return None

    def store(self, key, payload, _vector=None):
        # Must be JSON — the previous implementation stored a raw Python repr.
        json.loads(payload)
        self.stored[key] = payload

    def adjust_threshold(self, delta):
        return None


class StubProgram:
    """Returns a fixed prediction — no LM involved."""

    def __init__(self, prediction):
        self.prediction = prediction

    def __call__(self, **_inputs):
        return self.prediction


def _manager_for(program):
    from dspytools.core.hotswap import HotSwapManager

    mgr = HotSwapManager()
    mgr._programs["run_1"] = program
    mgr._active = "run_1"
    mgr._cache = FakeCache()
    return mgr


def _configure_dummy(dummy):
    def _setup_dspy(**_kwargs):
        dspy.configure(lm=dummy)
        return dummy

    return _setup_dspy


def _reasoning(text: str):
    from dspy.adapters.types.reasoning import Reasoning

    return Reasoning(content=text)


# ── 1. API startup configures the LM ─────────────────────────────────


def test_startup_configures_lm(monkeypatch):
    from dspytools.api import server as api_server

    dummy = dspy.utils.DummyLM([{"answer": "4"}])
    monkeypatch.setattr(api_server, "_lm_configured", False)
    monkeypatch.setattr(api_server, "setup_dspy", _configure_dummy(dummy))

    api_server._ensure_lm_configured()

    assert dspy.settings.lm is dummy
    assert api_server._lm_configured is True


def test_startup_fails_loudly_without_lm(monkeypatch):
    from dspytools.api import server as api_server

    monkeypatch.setattr(api_server, "_lm_configured", False)
    monkeypatch.setattr(api_server, "setup_dspy", lambda **_kwargs: None)

    with pytest.raises(api_server.LMConfigurationError, match="No LM"):
        api_server._ensure_lm_configured()


def test_infer_returns_200_with_configured_lm(monkeypatch):
    """The old server never configured the LM: every /infer was a 500."""
    from dspytools.api import server as api_server

    dummy = dspy.utils.DummyLM([{"answer": "4"}])
    monkeypatch.setattr(api_server, "_lm_configured", False)
    monkeypatch.setattr(api_server, "setup_dspy", _configure_dummy(dummy))
    monkeypatch.setattr(
        api_server, "_hotswap", _manager_for(dspy.Predict("question -> answer"))
    )

    client = TestClient(api_server.app)
    resp = client.post("/infer", json={"inputs": {"question": "2+2?"}})

    assert resp.status_code == 200, resp.text
    assert resp.json() == {"status": "ok", "result": {"answer": "4"}}


def test_infer_returns_503_when_lm_cannot_be_configured(monkeypatch):
    from dspytools.api import server as api_server

    monkeypatch.setattr(api_server, "_lm_configured", False)
    monkeypatch.setattr(api_server, "setup_dspy", lambda **_kwargs: None)
    monkeypatch.setattr(api_server, "_hotswap", None)

    client = TestClient(api_server.app)
    resp = client.post("/infer", json={"inputs": {"question": "2+2?"}})

    assert resp.status_code == 503
    assert "No LM" in resp.json()["detail"]


def test_health_reports_lm_state(monkeypatch):
    import importlib

    dspy_settings = importlib.import_module("dspy.dsp.utils.settings")
    from dspytools.api import server as api_server

    monkeypatch.setattr(api_server, "_hotswap", None)
    client = TestClient(api_server.app)

    body = client.get("/health").json()
    assert body["lm_configured"] is False
    assert body["status"] == "degraded"

    dummy = dspy.utils.DummyLM([{"answer": "4"}])
    dspy_settings.main_thread_config["lm"] = dummy

    body = client.get("/health").json()
    assert body["lm_configured"] is True
    assert body["status"] == "ok"


# ── 2. Model config changes take effect in the running process ────────


def test_model_config_endpoint_reapplies_lm(monkeypatch):
    from dspytools.api import server as api_server

    old_lm = dspy.utils.DummyLM([{"answer": "old"}])
    new_lm = dspy.utils.DummyLM([{"answer": "new"}])
    # "before": the process is serving with the old LM.
    api_server._serving_lm = old_lm

    cleared: list[bool] = []
    monkeypatch.setattr(
        api_server.LMRegistry, "clear", classmethod(lambda cls: cleared.append(True))
    )
    monkeypatch.setattr(
        api_server.LMRegistry,
        "get_or_default",
        classmethod(lambda cls: new_lm),
    )
    monkeypatch.setattr(api_server, "_lm_configured", True)
    monkeypatch.setattr(api_server, "load_config", lambda: {"lm": {}})
    monkeypatch.setattr(api_server, "save_user_config", lambda cfg: None)

    client = TestClient(api_server.app)
    resp = client.post("/config/models/student", json={"model": "openai/new-model"})

    assert resp.status_code == 200, resp.text
    assert cleared == [True], "registry cache must be cleared before re-applying"
    # "after": requests are served by the new LM, no restart needed.
    assert api_server._serving_lm is new_lm
    assert api_server._current_lm() is not old_lm
    assert resp.json()["active_lm"] == getattr(new_lm, "model", None)


# ── 4. Semantic cache serialization ────────────────────────────────


def test_cot_reasoning_prediction_is_cached_as_json():
    prediction = dspy.Prediction(reasoning=_reasoning("think"), answer="4")
    mgr = _manager_for(StubProgram(prediction))

    result = mgr.infer(question="2+2?")

    assert result["answer"] == "4"
    assert len(mgr._cache.stored) == 1
    payload = json.loads(next(iter(mgr._cache.stored.values())))
    assert payload["answer"] == "4"
    # dspy.Reasoning is str-like; it is coerced to a plain string for JSON.
    assert payload["reasoning"] == "think"


def test_cache_serialization_failure_is_a_miss(monkeypatch):
    from dspytools.core import hotswap

    def _boom(_output):
        raise TypeError("Object of type Reasoning is not JSON serializable")

    monkeypatch.setattr(hotswap, "_json_safe_output", _boom)
    mgr = _manager_for(
        StubProgram(dspy.Prediction(reasoning=_reasoning("think"), answer="4"))
    )

    result = mgr.infer(question="2+2?")

    assert result["answer"] == "4"
    assert mgr._cache.stored == {}, "an unserializable result is a cache miss"


# ── 5. Drift/quality score tracks the answer, not the reasoning ──────


def test_score_uses_answer_not_reasoning(monkeypatch):
    from dspytools.core import hotswap

    rich_answer = "# Answer\n\n## Detail\n\n- first\n- second\n\n" + "word " * 80
    prediction = dspy.Prediction(
        reasoning=_reasoning("x"),  # old code scored this: 0.0
        answer=rich_answer,
    )
    mgr = _manager_for(StubProgram(prediction))

    monkeypatch.setattr(hotswap, "_infer_count", 0)
    monkeypatch.setattr(hotswap, "_quality_samples", deque(maxlen=200))

    mgr.infer(question="q")

    score = hotswap._quality_samples[-1]["score"]
    assert score > 0.3, "score must reflect the answer field, not `reasoning`"


# ── 8. evaluate reconstructs the saved program shape ─────────────────


def _write_cot_run(root: Path, run_id: str = "cot_run") -> Path:
    run_dir = root / run_id
    run_dir.mkdir(parents=True)
    dspy.ChainOfThought("question -> answer").save(str(run_dir / "program.json"))
    (run_dir / "metadata.json").write_text(
        json.dumps({"id": run_id, "module_type": "cot"})
    )
    (run_dir / "signature.json").write_text(
        json.dumps({"inputs": ["question"], "outputs": ["answer"]})
    )
    return run_dir


def test_evaluate_loads_saved_module_type(tmp_path):
    from dspytools.commands.evaluate import _resolve_answer_field

    run_dir = _write_cot_run(tmp_path)
    program = _load_evaluate_program(run_dir)

    assert isinstance(program, dspy.ChainOfThought)
    assert _resolve_answer_field(program) == "answer"


def test_evaluate_rejects_non_run_paths(tmp_path):
    from dspytools.cli.rich_config import LLMCommand, click
    from dspytools.commands.evaluate import _load_program

    @click.command(name="_load_program", cls=LLMCommand)
    @click.argument("module")
    def _cmd(module: str):
        _load_program(module)

    stray = tmp_path / "weights.json"
    stray.write_text(json.dumps({}))
    assert CliRunner().invoke(_cmd, [str(stray)]).exit_code != 0

    # A bare program.json without metadata.json/signature.json is refused too.
    bare = tmp_path / "bare_run"
    bare.mkdir()
    (bare / "program.json").write_text(json.dumps({}))
    assert CliRunner().invoke(_cmd, [str(bare / "program.json")]).exit_code != 0


def _load_evaluate_program(path: Path):
    from dspytools.commands.evaluate import _load_program

    return _load_program(str(path))


# ── 9. evaluate metrics use the resolved field ───────────────────────


def test_evaluate_metrics_resolve_output_field():
    from dspytools.commands.evaluate import _get_metric, _resolve_answer_field

    example = dspy.Example(input="q", output="the answer").with_inputs("input")
    prediction = dspy.Prediction(output="the answer")

    assert _get_metric("exact_match", "output")(example, prediction) == 1.0
    assert _get_metric("passage_match", "output")(example, prediction) == 1.0
    assert _get_metric("semantic_f1", "output")(example, prediction) == 1.0
    # The old hardcoded `answer` field never existed on this schema, so every
    # example scored 0.0.
    assert (
        _get_metric("exact_match", "answer")(example, dspy.Prediction(answer="x"))
        == 0.0
    )

    # callers that pass no field (the MCP `evaluate` tool) resolve per pair
    assert _get_metric("exact_match")(example, prediction) == 1.0
    answer_example = dspy.Example(input="q", answer="the answer").with_inputs("input")
    answer_prediction = dspy.Prediction(answer="the answer")
    assert _get_metric("exact_match")(answer_example, answer_prediction) == 1.0
    # exact_match shares core.metrics normalization ("The answer." == "the answer")
    assert (
        _get_metric("exact_match", "output")(
            example, dspy.Prediction(output="The answer.")
        )
        == 1.0
    )

    # Field resolution comes from the program's own signature.
    assert _resolve_answer_field(dspy.Predict("input -> output")) == "output"
    assert _resolve_answer_field(dspy.ChainOfThought("question -> answer")) == "answer"


# ── 7. module new --signature emits the prompt string ────────────────


def test_module_new_signature_uses_signature_string(monkeypatch):
    from dspytools.commands import module as module_cmd
    from dspytools.config.settings import signatures_dir

    (signatures_dir() / "QA.py").write_text(
        "from dspytools.core._dspy import dspy\n\n"
        "class QA(dspy.Signature):\n"
        '    """Answer questions."""\n\n'
        "    question: str = dspy.InputField()\n"
        "    answer: str = dspy.OutputField()\n"
    )

    captured: dict[str, str] = {}

    class FakeGenerator:
        def generate(self, name, sig_str, mod_type, instructions):
            captured["sig"] = sig_str
            return ("class Generated(dspy.Module):\n    pass\n" + "#" * 120, False, [])

    monkeypatch.setattr(module_cmd, "setup_dspy", lambda **_kwargs: None)
    # module.py imports the generator lazily (it is a real dspy.Module
    # subclass), so the patch target is the generators package.
    monkeypatch.setattr("dspytools.generators.ModuleGeneratorDSPy", FakeGenerator)

    result = CliRunner().invoke(module_cmd.module_new, ["QAThing", "--signature", "QA"])

    assert result.exit_code == 0, result.output
    assert captured["sig"] == "question -> answer"


# ── 10. drift auto-fix requires a real trainset ─────────────────────


def test_drift_auto_fix_refuses_without_trainset(mock_drift_monitor):
    from dspytools.core.registry import register_run

    register_run("drifted_run", {"optimizer": "test", "module": "ghost_module"})
    monitor = mock_drift_monitor
    monitor.request_recompile("drifted_run")

    results = monitor.process_recompile_requests(auto_fix=True)

    assert results[0]["status"] == "skipped"
    assert "no trainset registered" in results[0]["action"]
    # The request stays queued so a later run with data can fix it.
    assert "drifted_run" in monitor.pending_recompiles()


def test_drift_trainset_registration_and_resolution(tmp_path, mock_drift_monitor):
    monitor = mock_drift_monitor
    trainset = [dspy.Example(input="q", output="a").with_inputs("input")]

    monitor.register_trainset("run_a", trainset)
    assert monitor.get_trainset("run_a") == trainset

    with pytest.raises(ValueError):
        monitor.register_trainset("run_b", [])

    dataset = tmp_path / "data.json"
    dataset.write_text(json.dumps([{"input": "q", "output": "a"}]))
    assert monitor._resolve_trainset("run_c", {"trainset": str(dataset)})
    assert monitor.get_trainset("run_c")
    assert monitor._resolve_trainset("run_d", {}) is None
