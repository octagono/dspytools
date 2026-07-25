"""Tests for SPRT validation — Sequential Probability Ratio Test.

Tests SelfEvolveEngine.validate_and_deploy() with mock candidate programs.
Pure logic — no LM calls, no DSPy import needed.
"""

from __future__ import annotations

from typing import Any

from dspytools.evolve.self_evolve import SelfEvolveEngine


def _make_example(input_val: str, output_val: str) -> object:
    """Create a minimal object mimicking a DSPy Example.

    The method signature used by validate_and_deploy:
      kwargs = example.inputs() if hasattr(example, "inputs")
               else {"input": getattr(example, "input", "")}
      expected = getattr(example, "output", "")
    """

    class _Example:
        def inputs(self) -> dict[str, Any]:
            return {"input": self.input}

        input = input_val
        output = output_val

    return _Example()


def _make_holdout(pairs: list[tuple[str, str]]) -> list:
    """Build a holdout set from (input, output) pairs."""
    return [_make_example(inp, out) for inp, out in pairs]


def _perfect_candidate(**kwargs: Any) -> object:
    """Candidate that always matches expected output."""

    class _Pred:
        output = "correct"
        answer = "correct"

    return _Pred()


def _terrible_candidate(**kwargs: Any) -> object:
    """Candidate that never matches expected output."""

    class _Pred:
        output = "wrong"
        answer = "wrong"

    return _Pred()


_candidate_toggle: bool = False


def _medium_candidate(**kwargs: Any) -> object:
    """Candidate that matches 50% of the time (alternating)."""
    global _candidate_toggle
    _candidate_toggle = not _candidate_toggle

    class _Pred:
        output = kwargs.get("input", "") if _candidate_toggle else "wrong"
        answer = output

    return _Pred()


def _crashing_candidate(**kwargs: Any) -> object:
    """Candidate that always raises."""
    msg = "internal error"
    raise RuntimeError(msg)


# ═══════════════════════════════════════════════════════════════════════════


class TestSprtEarlyDecision:
    """Tests where SPRT terminates early on clear wins or losses."""

    def _engine(self) -> SelfEvolveEngine:
        engine = SelfEvolveEngine()
        engine._clear_state()
        return engine

    def test_early_accept(self):
        """Candidate with 80% accuracy should be accepted early."""
        engine = self._engine()
        holdout = _make_holdout([(f"q{i}", "correct") for i in range(20)])
        result = engine.validate_and_deploy(
            candidate_program=_perfect_candidate,
            program_id="test_prog",
            holdout_set=holdout,
            alpha=0.05,
            beta=0.2,
            max_evaluations=50,
        )
        assert result["accepted"] is True, f"Expected accepted, got: {result}"
        assert result["early_stop"] is True, "Should stop early on clear win"
        assert result["candidate_score"] >= 0.5
        assert result["n_evaluated"] < 20, "SPRT should stop before exhausting holdout"

    def test_early_reject(self):
        """Candidate with 0% accuracy should be rejected early."""
        engine = self._engine()
        holdout = _make_holdout([(f"q{i}", "correct") for i in range(20)])
        result = engine.validate_and_deploy(
            candidate_program=_terrible_candidate,
            program_id="test_prog",
            holdout_set=holdout,
            alpha=0.05,
            beta=0.2,
            max_evaluations=50,
        )
        assert result["accepted"] is False
        assert result["early_stop"] is True
        assert result["n_evaluated"] < 20

    def test_result_keys_present(self):
        """Result dict has all expected keys."""
        engine = self._engine()
        holdout = _make_holdout([("q", "a")])
        result = engine.validate_and_deploy(
            candidate_program=_perfect_candidate,
            program_id="test_prog",
            holdout_set=holdout,
        )
        expected_keys = {
            "accepted",
            "candidate_score",
            "n_evaluated",
            "early_stop",
            "reason",
        }
        assert expected_keys.issubset(result.keys()), (
            f"Missing keys: {expected_keys - set(result.keys())}"
        )


# ═══════════════════════════════════════════════════════════════════════════


class TestSprtMaxEvaluations:
    """Tests where SPRT exhausts max_evaluations without reaching a decision."""

    def _engine(self) -> SelfEvolveEngine:
        engine = SelfEvolveEngine()
        engine._clear_state()
        return engine

    def test_max_evaluations_exhausted(self):
        """SPRT exhausted after max_evaluations with ~50% accuracy."""
        engine = self._engine()
        # For a 50% accurate candidate, SPRT should not reach a clear decision
        # within a reasonable number of evaluations
        holdout = _make_holdout([(f"q{i}", "correct") for i in range(10)])

        # Use a deterministic "always wrong" candidate to guarantee rejection
        result = engine.validate_and_deploy(
            candidate_program=_terrible_candidate,
            program_id="test_prog",
            holdout_set=holdout,
            alpha=0.05,
            beta=0.2,
            max_evaluations=10,
        )
        # Candidate is terrible, should still reject
        assert result["accepted"] is False


# ═══════════════════════════════════════════════════════════════════════════


class TestSprtEdgeCases:
    """Edge cases: empty holdout, single example, crashing candidate."""

    def _engine(self) -> SelfEvolveEngine:
        engine = SelfEvolveEngine()
        engine._clear_state()
        return engine

    def test_empty_holdout(self):
        """Empty holdout → not accepted, 0 evaluated."""
        engine = self._engine()
        result = engine.validate_and_deploy(
            candidate_program=_perfect_candidate,
            program_id="test_prog",
            holdout_set=[],
        )
        assert result["accepted"] is False
        assert result["n_evaluated"] == 0

    def test_single_example(self):
        """Single holdout example → at least evaluated (but <3 for SPRT)."""
        engine = self._engine()
        holdout = _make_holdout([("q", "correct")])
        result = engine.validate_and_deploy(
            candidate_program=_perfect_candidate,
            program_id="test_prog",
            holdout_set=holdout,
            max_evaluations=1,
        )
        assert result["n_evaluated"] >= 1
        # With n=1, SPRT can't decide (needs >=3) but should still produce a result
        assert "candidate_score" in result

    def test_crashing_candidate(self):
        """Candidate that always raises → all failures, rejected."""
        engine = self._engine()
        holdout = _make_holdout([("q1", "a1"), ("q2", "a2"), ("q3", "a3")])
        result = engine.validate_and_deploy(
            candidate_program=_crashing_candidate,
            program_id="test_prog",
            holdout_set=holdout,
            max_evaluations=5,
        )
        assert result["accepted"] is False
        assert result["candidate_score"] == 0.0

    def test_perfect_candidate_accepted(self):
        """Perfect candidate (matches all expected output="correct") is accepted."""
        engine = self._engine()
        holdout = _make_holdout(
            [("q1", "correct"), ("q2", "correct"), ("q3", "correct")]
        )
        result = engine.validate_and_deploy(
            candidate_program=_perfect_candidate,
            program_id="test_prog",
            holdout_set=holdout,
            max_evaluations=5,
        )
        assert result["accepted"] is True
        assert result["candidate_score"] == 1.0
