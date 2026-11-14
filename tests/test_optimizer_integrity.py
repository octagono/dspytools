"""Optimizer/metric integrity tests (Golden Rule 3 + metric contracts).

Covers the defects fixed in the optimizer-integrity pass:

* the holdout split happens before the compile, so no optimizer input (trainset
  or GEPA valset) contains a holdout example;
* metrics normalize answers, so ``"Paris."`` matches ``"Paris"``;
* ``gepa_metric`` returns evidence (``dspy.Prediction(score, feedback)``);
* ``_build_gepa`` rejects conflicting budgets with a Click error;
* optimizer specs are never mutated and only JSON-safe params are logged;
* the GRPO adapter matches the installed ``dspy.GRPO`` signature.

Hermetic: GEPA and the CLI collaborators are faked, no LM is constructed and no
network is touched.
"""

from __future__ import annotations

import inspect

import pytest
from click.testing import CliRunner
from dspy.teleprompt.grpo import GRPO

import dspytools.commands.compile as compile_mod
from dspytools.core._dspy import dspy
from dspytools.core.holdout import HoldoutGate
from dspytools.core.metrics import (
    answer_overlap_score,
    exact_match_metric,
    gepa_metric,
    gepa_metric_float,
)
from dspytools.gfl import grpo as grpo_mod

# ── Fakes ──────────────────────────────────────────────────────────────


class _FakePrediction:
    def __init__(self, output: str):
        self.output = output


class _FakeCompiled:
    """Stand-in compiled program: scores as one exact match."""

    detailed_results = None

    def __call__(self, **kwargs: object) -> _FakePrediction:
        return _FakePrediction("Paris")


class _FakeGEPA:
    """Signature-compatible GEPA stand-in that records what it was given."""

    constructed: list[_FakeGEPA] = []

    def __init__(
        self,
        metric=None,
        *,
        auto=None,
        max_full_evals=None,
        max_metric_calls=None,
        reflection_lm=None,
        instruction_proposer=None,
        use_merge=True,
        max_merge_invocations=5,
        component_selector="round_robin",
        perfect_score=1.0,
        track_stats=False,
        track_best_outputs=False,
        seed=0,
        gepa_kwargs=None,
        use_mlflow=False,
    ):
        self.metric = metric
        self.kwargs = {
            "auto": auto,
            "max_full_evals": max_full_evals,
            "max_metric_calls": max_metric_calls,
            "seed": seed,
        }
        self.seen: dict[str, object] = {}
        _FakeGEPA.constructed.append(self)

    def compile(self, student=None, trainset=None, valset=None, **kwargs):
        self.seen = {
            "trainset": list(trainset or []),
            "valset": None if valset is None else list(valset),
        }
        return _FakeCompiled()


class _FakeMorphology:
    def profile_task(self, **kwargs: object) -> str:
        return "profile"


class _FakeEngine:
    def __init__(self, *args: object, **kwargs: object):
        self.morphology = _FakeMorphology()

    def on_compile(self, **kwargs: object) -> None:
        return None


class _FakeDriftMonitor:
    def __init__(self) -> None:
        self.baselines: list[tuple[str, float]] = []

    def update_baseline(self, run_id: str, score: float) -> None:
        self.baselines.append((run_id, score))


class _RecordingTracker:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def log_compile(self, **kwargs: object) -> str:
        self.calls.append(kwargs)
        return "mlflow-run"


def _examples(n: int = 10) -> list:
    return [
        dspy.Example(input=f"q{i}", output=f"a{i}").with_inputs("input")
        for i in range(n)
    ]


# ── (1) golden rule 3: optimizer never sees holdout ────────────────────


def test_optimizer_never_receives_holdout_examples(monkeypatch, tmp_path):
    gate = HoldoutGate()
    tracker = _RecordingTracker()
    registered: list[tuple] = []
    trainset = _examples(10)

    monkeypatch.setattr(compile_mod, "setup_dspy", lambda: None)
    monkeypatch.setattr(compile_mod, "load_module_by_name", lambda name: object())
    monkeypatch.setattr(compile_mod, "load_trainset", lambda path: list(trainset))
    monkeypatch.setattr(compile_mod, "compute_dataset_hash", lambda data: "deadbeef")
    monkeypatch.setattr(compile_mod, "find_existing_compile", lambda *a, **k: None)
    monkeypatch.setattr(
        compile_mod, "create_run_dir", lambda *a, **k: ("run_test", tmp_path)
    )
    monkeypatch.setattr(compile_mod, "save_program", lambda *a, **k: None)
    monkeypatch.setattr(
        compile_mod,
        "register_run_with_lineage",
        lambda run_id, metadata, **kw: registered.append((run_id, metadata)),
    )
    monkeypatch.setattr(compile_mod, "get_tracker", lambda: tracker)
    monkeypatch.setattr(compile_mod, "get_holdout_gate", lambda: gate)
    monkeypatch.setattr(compile_mod, "SelfEvolveEngine", _FakeEngine)
    monkeypatch.setattr(compile_mod, "get_drift_monitor", lambda: _FakeDriftMonitor())
    monkeypatch.setattr(compile_mod.dspy, "GEPA", _FakeGEPA)

    class _Teacher:
        def __repr__(self) -> str:
            return "<teacher-lm>"

    monkeypatch.setattr(
        compile_mod.LMRegistry, "get_teacher", classmethod(lambda cls: _Teacher())
    )

    result = CliRunner().invoke(
        compile_mod.compile_cmd, ["gepa", "demo", "data.json", "--force"]
    )
    assert result.exit_code == 0, result.output

    holdout = gate.get_holdout("run_test")
    assert len(holdout) == 2
    holdout_ids = {id(ex) for ex in holdout}

    gepa = _FakeGEPA.constructed[-1]
    optimizer_inputs = gepa.seen["trainset"] + (gepa.seen["valset"] or [])
    assert optimizer_inputs, "optimizer received no data"
    assert not holdout_ids & {id(ex) for ex in optimizer_inputs}
    # GEPA's valset comes out of the train split, not out of the holdout.
    assert len(gepa.seen["trainset"]) == 5
    assert len(gepa.seen["valset"]) == 3

    # Metadata still reports the real dataset total.
    _, metadata = registered[-1]
    assert metadata["trainset_size"] == 10
    assert metadata["holdout_size"] == 2
    # ... and never leaks the live LM object into the index/MLflow params.
    assert "reflection_lm" not in metadata
    assert "reflection_lm" not in tracker.calls[-1]["params"]


def test_no_gate_path_compiles_on_full_dataset(monkeypatch, tmp_path):
    """A 1-example dataset cannot be split — the old path is unchanged."""
    gate = HoldoutGate()
    trainset = _examples(1)

    monkeypatch.setattr(compile_mod, "setup_dspy", lambda: None)
    monkeypatch.setattr(compile_mod, "load_module_by_name", lambda name: object())
    monkeypatch.setattr(compile_mod, "load_trainset", lambda path: list(trainset))
    monkeypatch.setattr(compile_mod, "compute_dataset_hash", lambda data: "deadbeef")
    monkeypatch.setattr(compile_mod, "find_existing_compile", lambda *a, **k: None)
    monkeypatch.setattr(
        compile_mod, "create_run_dir", lambda *a, **k: ("run_tiny", tmp_path)
    )
    monkeypatch.setattr(compile_mod, "save_program", lambda *a, **k: None)
    monkeypatch.setattr(compile_mod, "register_run_with_lineage", lambda *a, **k: None)
    monkeypatch.setattr(compile_mod, "get_tracker", lambda: _RecordingTracker())
    monkeypatch.setattr(compile_mod, "get_holdout_gate", lambda: gate)
    monkeypatch.setattr(compile_mod, "SelfEvolveEngine", _FakeEngine)
    monkeypatch.setattr(compile_mod, "get_drift_monitor", lambda: _FakeDriftMonitor())
    monkeypatch.setattr(compile_mod.dspy, "GEPA", _FakeGEPA)
    monkeypatch.setattr(
        compile_mod.LMRegistry, "get_teacher", classmethod(lambda cls: object())
    )

    result = CliRunner().invoke(
        compile_mod.compile_cmd, ["gepa", "demo", "data.json", "--force"]
    )
    assert result.exit_code == 0, result.output
    assert len(_FakeGEPA.constructed[-1].seen["trainset"]) == 1


# ── (2) normalized exact match ─────────────────────────────────────────


def test_exact_match_metric_normalizes_case_and_punctuation():
    metric = exact_match_metric()
    gold = dspy.Example(output="Paris")
    assert metric(gold, _FakePrediction("Paris.")) == 1.0
    assert metric(gold, _FakePrediction('  "paris"  ')) == 1.0
    assert metric(gold, _FakePrediction("Paris, France")) == 0.0
    assert metric(gold, _FakePrediction("")) == 0.0


def test_answer_overlap_score_is_bounded_and_sub_perfect():
    assert answer_overlap_score("Paris", "paris") == 1.0
    assert 0.0 < answer_overlap_score("Paris France", "Paris") < 1.0
    assert answer_overlap_score("Paris", "London") == 0.0
    # A non-exact answer must never read as perfect (GEPA skip_perfect_score).
    assert answer_overlap_score("the capital of France", "France") < 1.0


# ── (3) gepa_metric carries evidence ───────────────────────────────────


def test_gepa_metric_returns_prediction_with_feedback():
    gold = dspy.Example(output="Paris")
    result = gepa_metric(gold, _FakePrediction("Lyon"))
    assert isinstance(result, dspy.Prediction)
    assert 0.0 <= result.score <= 1.0
    assert "Paris" in result.feedback and "Lyon" in result.feedback

    perfect = gepa_metric(gold, _FakePrediction("paris."))
    assert perfect.score == 1.0


def test_gepa_metric_feedback_names_missing_field():
    gold = dspy.Example(output="Paris")
    result = gepa_metric(gold, dspy.Example(answer="Paris"))
    assert result.score == 0.0
    assert "missing the 'output' field" in result.feedback


def test_gepa_metric_float_is_plain_float():
    gold = dspy.Example(output="Paris")
    score = gepa_metric_float(gold, _FakePrediction("Paris."))
    assert isinstance(score, float)
    assert score == 1.0


# ── (4) budget conflict is a CLI error ─────────────────────────────────


def test_build_gepa_rejects_two_budgets():
    with pytest.raises(compile_mod.click.ClickException) as excinfo:
        compile_mod._build_gepa(None, auto="light", max_metric_calls=100)
    assert "max_metric_calls" in str(excinfo.value)
    assert "mutually exclusive" in str(excinfo.value)


def test_build_gepa_defaults_to_light_auto(monkeypatch):
    monkeypatch.setattr(compile_mod.dspy, "GEPA", _FakeGEPA)
    compile_mod._build_gepa(None)
    assert _FakeGEPA.constructed[-1].kwargs["auto"] == "light"


# ── (5) optimizer spec params are copied and JSON-safe ─────────────────


def test_optimizer_params_are_copied_not_mutated():
    spec = compile_mod._OPTIMIZER_SPECS["gepa"]["params"]
    assert spec == {}

    first = compile_mod._optimizer_run_params(spec)
    first["reflection_lm"] = object()
    first["gepa_kwargs"] = {"x": 1}

    second = compile_mod._optimizer_run_params(spec)
    assert spec == {}
    assert second == {}


def test_json_safe_params_drops_live_objects():
    safe = compile_mod._json_safe_params(
        {
            "reflection_lm": object(),
            "seed": 3,
            "use_merge": True,
            "gepa_kwargs": {"a": 1},
            "nested": {"bad": object()},
            "callable": lambda: None,
        }
    )
    assert safe == {"seed": 3, "use_merge": True, "gepa_kwargs": {"a": 1}}


# ── (6) GRPO adapter matches the installed signature ───────────────────


def test_grpo_construction_matches_installed_signature(monkeypatch):
    captured: dict[str, dict] = {}

    class _CapturingGRPO:
        def __init__(self, **kwargs: object):
            captured["init"] = kwargs

        def compile(self, **kwargs: object):
            captured["compile"] = kwargs
            return "compiled"

    monkeypatch.setattr(grpo_mod, "_grpo_class", lambda: _CapturingGRPO)

    result = grpo_mod.compile_grpo(
        object(), _examples(3), max_steps=7, num_rollouts=2, num_threads=1
    )
    assert result == "compiled"

    accepted = set(inspect.signature(GRPO.__init__).parameters)
    assert set(captured["init"]) <= accepted
    assert captured["init"]["num_train_steps"] == 7
    assert captured["init"]["num_rollouts_per_grpo_step"] == 2
    assert "lora" not in captured["init"]
    assert "beta" not in captured["init"]
