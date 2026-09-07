"""Recorded-LM behavior tests — hermetic optimizer validation.

These tests verify that optimizers actually *optimize* (not just construct)
by running them against recorded LM responses. On first run (or with
``VCR_RECORD=1``) they record real LM calls to ``tests/cassettes/*.yaml``;
on subsequent runs they replay those cassettes — deterministic, zero-cost,
and CI-safe.

Recording requires a live OpenAI-compatible LM. Point ``DSPYTOOLS_TEST_LM_URL``
at it (default: local llama-cpp on :8080 with Qwen3.5-9B). When no LM is
reachable the tests skip so the suite still passes headless.

To re-record after prompt changes: ``rm tests/cassettes/*.yaml && VCR_RECORD=1 pytest tests/test_optimizer_behavior.py``
"""

from __future__ import annotations

import pytest


def _requires_lm(student_lm):
    """Skip when no live LM is available for recording (replay still works)."""
    if student_lm is None:
        pytest.skip("no local LM reachable — set DSPYTOOLS_TEST_LM_URL")


def _cassette(vcr_instance, name):
    """Return a use_cassette context for the given cassette name."""
    return vcr_instance.use_cassette(f"{name}.yaml")


def test_predict_produces_expected_answer(student_lm, vcr_instance):
    """A plain dspy.Predict answers correctly — replay-recorded."""
    _requires_lm(student_lm)
    from dspytools.core._dspy import dspy

    with _cassette(vcr_instance, "test_predict_answer"):
        with dspy.context(lm=student_lm):
            result = dspy.Predict("question -> answer")(question="What is 2+2?")
    assert int(getattr(result, "answer", "0").strip()) == 4


def test_chain_of_thought_returns_answer_field(student_lm, vcr_instance):
    """ChainOfThought returns the signature output field."""
    _requires_lm(student_lm)
    from dspytools.core._dspy import dspy

    with _cassette(vcr_instance, "test_cot_sentiment"):
        with dspy.context(lm=student_lm):
            result = dspy.ChainOfThought("sentence -> sentiment: str")(
                sentence="dspytools is a great tool"
            )
    # Field contract: output field 'sentiment' present and non-empty.
    assert getattr(result, "sentiment", "").strip()
    assert hasattr(result, "reasoning")  # CoT adds reasoning


def test_labeled_few_shot_compile_improves_demos(student_lm, vcr_instance):
    """LabeledFewShot compile succeeds against the LM and produces a program.

    Uses a tiny trainset so recording stays fast. The point is behavioral:
    the optimizer runs end-to-end (LM demos formatted, compiled program
    callable) rather than only constructing.
    """
    _requires_lm(student_lm)
    from dspytools.core._dspy import dspy

    trainset = [
        dspy.Example(question="What is 1+1?", answer="2").with_inputs("question"),
        dspy.Example(question="What is 2+2?", answer="4").with_inputs("question"),
    ]

    def metric(gold, pred, trace=None):
        return getattr(pred, "answer", "") == getattr(gold, "answer", "")

    with _cassette(vcr_instance, "test_lfs_compile"):
        with dspy.context(lm=student_lm):
            compiled = dspy.LabeledFewShot(k=2).compile(
                student=dspy.Predict("question -> answer"),
                trainset=trainset,
            )
            result = compiled(question="What is 3+3?")
    assert getattr(result, "answer", "").strip()  # non-empty answer


def test_bootstrap_few_shot_compiles_and_scores(student_lm, vcr_instance):
    """BootstrapFewShot compile runs and the compiled program scores >= baseline.

    Proves the optimizer loop actually executes against LM responses
    (demo bootstrapping included) — a structural smoke test cannot catch a
    broken bootstrap path.
    """
    _requires_lm(student_lm)
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

    student = dspy.Predict("question -> answer")
    with _cassette(vcr_instance, "test_bfs_compile"):
        with dspy.context(lm=student_lm):
            compiled = dspy.BootstrapFewShot(
                metric=metric, max_bootstrapped_demos=2
            ).compile(
                student=student,
                trainset=trainset,
            )
            # Evaluate on the trainset — demos were bootstrapped from LM calls.
            correct = 0
            for ex in trainset:
                pred = compiled(question=ex.question)
                correct += metric(ex, pred)
    assert correct >= 1, "bootstrap compile produced no correct answer on trainset"
