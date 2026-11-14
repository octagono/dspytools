"""Hermetic optimizer behavior tests — optimizers really run, no LM server.

These tests used to replay vcrpy cassettes of real llama.cpp traffic.  DSPy
3.5's lm15 engine calls HTTP over bundled stdlib raw-socket transports
(``dspy/_vendor/lm15/transports/_http11.py``), which vcrpy cannot intercept:
every replay missed, fell through to a real ``openai/gpt-4o`` call and died on
``SSLCertVerificationError``.  The cassettes and ``tests/vcr_config.py`` were
therefore deleted; each test below is driven by ``dspy.utils.DummyLM`` and
asserts wiring/contracts (fields, demos, call counts), never model prose.

``tests/conftest.py::probe_lm_endpoint`` is the only liveness check in the
tree; it is exercised by ``test_probe_lm_endpoint_*`` with zero network I/O.
"""

from __future__ import annotations

import time

import pytest


def test_predict_produces_expected_answer(dummy_lm):
    """A plain dspy.Predict maps the LM answer into the signature output field.

    Intent preserved from the cassette test: one Predict call, one LM turn,
    the ``answer`` field carries the LM's text.
    """
    from dspytools.core._dspy import dspy

    lm = dummy_lm([{"answer": "4"}])

    with dspy.context(lm=lm, adapter=dspy.ChatAdapter()):
        result = dspy.Predict("question -> answer")(question="What is 2+2?")

    assert result.answer == "4"
    assert isinstance(result.answer, str)
    # Exactly one LM turn was consumed by exactly one predictor call.
    assert len(lm.history) == 1
    assert lm.history[0]["messages"][-1]["content"].strip()


def test_chain_of_thought_returns_answer_field(dummy_lm):
    """ChainOfThought exposes both the reasoning and the declared output field.

    DSPy 3.5 types the CoT rationale as ``dspy.Reasoning`` and renders it under
    the ``plan`` fallback name for text adapters, so DummyLM scripts ``plan``.
    """
    from dspytools.core._dspy import dspy

    lm = dummy_lm([{"plan": "recall the sentiment lexicon", "sentiment": "positive"}])

    with dspy.context(lm=lm, adapter=dspy.ChatAdapter()):
        result = dspy.ChainOfThought("sentence -> sentiment: str")(
            sentence="dspytools is a great tool"
        )

    assert result.sentiment == "positive"
    assert hasattr(result, "reasoning")
    assert result.reasoning is not None
    assert str(result.reasoning) == "recall the sentiment lexicon"
    assert isinstance(result.reasoning, dspy.Reasoning)
    assert len(lm.history) == 1


def test_labeled_few_shot_compile_improves_demos(dummy_lm):
    """LabeledFewShot attaches every train example as a demo, and the compiled
    program still answers through the LM.

    The optimizer itself makes no LM call; the assertion is that the demos land
    on the compiled predictor and the compiled program is callable.
    """
    from dspytools.core._dspy import dspy

    trainset = [
        dspy.Example(question="What is 1+1?", answer="2").with_inputs("question"),
        dspy.Example(question="What is 2+2?", answer="4").with_inputs("question"),
    ]
    lm = dummy_lm({"What is 3+3?": {"answer": "6"}})

    with dspy.context(lm=lm, adapter=dspy.ChatAdapter()):
        compiled = dspy.LabeledFewShot(k=2).compile(
            student=dspy.Predict("question -> answer"),
            trainset=trainset,
        )
        demos = [demo for pred in compiled.predictors() for demo in pred.demos]
        result = compiled(question="What is 3+3?")

    assert len(demos) == 2, "labeled demos were not attached to the compiled program"
    assert {d.question for d in demos} == {ex.question for ex in trainset}
    assert all(getattr(d, "answer", "") for d in demos), "demos lost their label"
    assert result.answer == "6"
    assert len(lm.history) == 1, "compiled program did not make exactly one LM call"


def test_bootstrap_few_shot_compiles_and_scores(dummy_lm):
    """BootstrapFewShot runs the full bootstrap loop and scores >= baseline.

    Intent preserved: prove the optimizer loop actually executes (teacher tries
    each train example, a passing trace becomes a demo) rather than only
    constructing.  DummyLM answers by prompt content, so bootstrapping succeeds
    deterministically and the metric can be replayed on the trainset.
    """
    from dspytools.core._dspy import dspy

    trainset = [
        dspy.Example(question="Capital of France?", answer="Paris").with_inputs(
            "question"
        ),
        dspy.Example(question="Capital of Japan?", answer="Tokyo").with_inputs(
            "question"
        ),
    ]

    def metric(gold, pred, trace=None):
        return float(getattr(pred, "answer", "") == getattr(gold, "answer", ""))

    lm = dummy_lm(
        {
            "Capital of France": {"answer": "Paris"},
            "Capital of Japan": {"answer": "Tokyo"},
        }
    )

    with dspy.context(lm=lm, adapter=dspy.ChatAdapter()):
        compiled = dspy.BootstrapFewShot(
            metric=metric, max_bootstrapped_demos=2
        ).compile(
            student=dspy.Predict("question -> answer"),
            trainset=trainset,
        )
        demos = [demo for pred in compiled.predictors() for demo in pred.demos]
        correct = sum(metric(ex, compiled(question=ex.question)) for ex in trainset)

    assert len(demos) == 2, "bootstrap produced no demos"
    assert all(getattr(d, "answer", "") for d in demos), (
        "bootstrapped demo has no answer"
    )
    assert correct == len(trainset), (
        f"compiled program scored {correct}/{len(trainset)} on the trainset"
    )
    # 2 bootstrap attempts + 2 evaluation calls: the loop really ran.
    assert len(lm.history) >= 4


def test_probe_lm_endpoint_reports_closed_port_fast():
    """Liveness probe fails fast on a closed local port and never raises."""
    from tests import conftest as conftest_mod

    start = time.monotonic()
    assert conftest_mod.probe_lm_endpoint("http://127.0.0.1:1/v1") is False
    assert time.monotonic() - start < 2.0, "probe blocked instead of timing out"


def test_probe_lm_endpoint_targets_configured_host_and_port(monkeypatch):
    """Probe resolves scheme defaults and passes the 0.25s timeout (no I/O)."""
    import contextlib

    from tests import conftest as conftest_mod

    seen: dict = {}

    def fake_connect(address, timeout=None):
        seen["address"] = address
        seen["timeout"] = timeout
        return contextlib.nullcontext()

    monkeypatch.setattr(conftest_mod.socket, "create_connection", fake_connect)

    assert conftest_mod.probe_lm_endpoint("http://127.0.0.1:9999/v1") is True
    assert seen == {"address": ("127.0.0.1", 9999), "timeout": 0.25}

    assert conftest_mod.probe_lm_endpoint("https://example.invalid/v1") is True
    assert seen["address"] == ("example.invalid", 443)


def test_live_lane_is_opt_in_and_never_probes(live_lm):
    """The old live llama.cpp lane stays off: no LM built, no endpoint contacted."""
    import os

    if os.environ.get("DSPYTOOLS_LIVE_LM") == "1":
        pytest.skip("live lane explicitly enabled via DSPYTOOLS_LIVE_LM=1")
    assert live_lm is None
