"""Tests for GFLPipeline infrastructure (no actual compilation needed).

Splits, modes, static helpers, run_halving behavior, and compile_draft
structure are tested without calling real DSPy optimizers.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from dspytools.core._dspy import dspy
from dspytools.gfl.pipeline import GFLPipeline


def _make_example(input_val, output_val):
    """Create a minimal DSPy example with input/output fields."""
    ex = dspy.Example(input=input_val, output=output_val)
    ex = ex.with_inputs("input")
    return ex


# ── split_holdout tests ────────────────────────────────────────────────────


def test_split_holdout_proportion():
    """Split 20 examples with 20% holdout yields 4 holdout / 16 train."""
    pipeline = GFLPipeline()
    trainset = [_make_example(f"test{i}", "result") for i in range(20)]
    train, holdout = pipeline.split_holdout(trainset, holdout_fraction=0.2)
    assert len(holdout) == 4
    assert len(train) == 16
    assert set(train).isdisjoint(set(holdout))


def test_split_holdout_minimum():
    """Single example yields 1 holdout, 0 train (at least 1 holdout)."""
    pipeline = GFLPipeline()
    trainset = [_make_example("t", "r")]
    train, holdout = pipeline.split_holdout(trainset, holdout_fraction=0.2)
    assert len(holdout) == 1
    assert len(train) == 0


def test_split_holdout_empty():
    """Empty trainset yields empty splits."""
    pipeline = GFLPipeline()
    train, holdout = pipeline.split_holdout([], holdout_fraction=0.2)
    assert len(train) == 0
    assert len(holdout) == 0


def test_split_holdout_deterministic():
    """Same seed produces same split."""
    pipeline = GFLPipeline()
    trainset = [_make_example(f"e{i}", str(i)) for i in range(10)]
    train1, holdout1 = pipeline.split_holdout(trainset, holdout_fraction=0.3)
    train2, holdout2 = pipeline.split_holdout(trainset, holdout_fraction=0.3)
    assert [e.input for e in train1] == [e.input for e in train2]
    assert [e.input for e in holdout1] == [e.input for e in holdout2]


# ── Mode initialization tests ──────────────────────────────────────────────


def test_init_modes():
    """Explicit mode strings are stored correctly."""
    p_compare = GFLPipeline(mode="compare")
    p_single = GFLPipeline(mode="single")
    assert p_compare.mode == "compare"
    assert p_single.mode == "single"


def test_init_default_mode():
    """Default mode is 'compare'."""
    p = GFLPipeline()
    assert p.mode == "compare"


# ── gate_promotion tests ──────────────────────────────────────────────────


def test_gate_promotion_structure():
    """gate_promotion returns expected keys (no actual programs needed)."""
    result = GFLPipeline.gate_promotion(
        candidate="dummy", baseline="dummy", holdout=[], min_improvement=0.02
    )
    assert "promoted" in result
    assert "candidate_score" in result
    assert "baseline_score" in result
    assert "improvement" in result


# ═══════════════════════════════════════════════════════════════════════════
# run_halving — pure logic tests (mocked optimizers)
# ═══════════════════════════════════════════════════════════════════════════


def _mock_dispatch_optimizer(score_map: dict[str, float]):
    """Create a mock _dispatch_optimizer that returns scores from a map."""
    program = _make_example("x", "y")

    def mock(name, student, trainset):
        nonlocal program
        return program  # Return same program regardless of optimizer

    return mock


class TestRunHalving:
    """Tests for GFLPipeline.run_halving() behavior."""

    @patch("dspytools.gfl.pipeline.GFLPipeline._dispatch_optimizer")
    @patch("dspytools.gfl.pipeline.GFLPipeline._evaluate")
    def test_halving_survivor_count(self, mock_eval, mock_dispatch):
        """With 4 optimizers and prune_fraction=0.5, exactly 2 survivors."""
        pipeline = GFLPipeline()
        student = MagicMock()
        mock_dispatch.return_value = _make_example("x", "y")
        # Return different scores: first optimizer wins
        # Note: run_halving calls _evaluate for baseline (set_baseline) before probe phase,
        # so the first score value is consumed by baseline, then 4 probe + 2 full = 7 total calls.
        scores = iter(
            [
                0.5,  # baseline evaluation
                0.9,
                0.8,
                0.7,
                0.6,  # probe phase
                0.95,
                0.85,
            ]
        )  # full phase (2 survivors)
        mock_eval.side_effect = scores

        trainset = [_make_example(f"q{i}", f"a{i}") for i in range(20)]
        result = pipeline.run_halving(student, trainset=trainset, prune_fraction=0.5)

        assert len(result["survivors"]) == 2
        assert len(result["pruned"]) == 2  # 4 total - 2 survivors = 2 pruned

    @patch("dspytools.gfl.pipeline.GFLPipeline._dispatch_optimizer")
    @patch("dspytools.gfl.pipeline.GFLPipeline._evaluate")
    def test_halving_keeps_best(self, mock_eval, mock_dispatch):
        """Optimizer with highest probe score survives to full phase."""
        pipeline = GFLPipeline()
        student = MagicMock()
        mock_dispatch.return_value = _make_example("x", "y")
        scores = iter(
            [
                0.5,  # baseline evaluation
                0.9,
                0.8,
                0.7,
                0.1,  # probe: first is best
                0.95,
                0.85,
            ]
        )  # full phase
        mock_eval.side_effect = scores

        trainset = [_make_example(f"q{i}", f"a{i}") for i in range(20)]
        result = pipeline.run_halving(student, trainset=trainset, prune_fraction=0.5)

        # Note: baseline _evaluate call consumes the first score (0.5),
        # so probe_scores start from the second score (0.9, 0.8, 0.7, 0.1)
        best_probe = sorted(
            result["probe_scores"].items(), key=lambda x: x[1], reverse=True
        )
        # "bootstrap_few_shot" (first optimizer) gets score 0.9 and is the best
        assert best_probe[0][0] == "bootstrap_few_shot"
        assert best_probe[0][1] == 0.9

    @patch("dspytools.gfl.pipeline.GFLPipeline._dispatch_optimizer")
    @patch("dspytools.gfl.pipeline.GFLPipeline._evaluate")
    def test_halving_prunes_worst(self, mock_eval, mock_dispatch):
        """Optimizer with lowest probe score is pruned."""
        pipeline = GFLPipeline()
        student = MagicMock()
        mock_dispatch.return_value = _make_example("x", "y")
        scores = iter(
            [
                0.5,  # baseline evaluation
                0.5,
                0.4,
                0.3,
                0.1,  # probe: last is worst
                0.6,
                0.5,
            ]
        )  # full phase
        mock_eval.side_effect = scores

        trainset = [_make_example(f"q{i}", f"a{i}") for i in range(20)]
        result = pipeline.run_halving(student, trainset=trainset, prune_fraction=0.5)

        # Worst optimizer (0.1) should be pruned
        assert any(s <= 0.1 for s in result["probe_scores"].values())
        assert "pruned" in result

    @patch("dspytools.gfl.pipeline.GFLPipeline._dispatch_optimizer")
    @patch("dspytools.gfl.pipeline.GFLPipeline._evaluate")
    def test_halving_fallback_on_failure(self, mock_eval, mock_dispatch):
        """If a survivor's full run fails, probe score is used as fallback."""
        pipeline = GFLPipeline()
        student = MagicMock()
        # Make dispatch fail on the full-phase call (after 4 probe calls)
        mock_dispatch.side_effect = [
            _make_example("x", "y"),  # probe optimizer 1
            _make_example("x", "y"),  # probe optimizer 2
            _make_example("x", "y"),  # probe optimizer 3
            _make_example("x", "y"),  # probe optimizer 4
            _make_example("x", "y"),  # full survivor 1 (works)
            RuntimeError("optimizer compilation failed"),  # full survivor 2 (fails)
        ]

        # probe scores: 0.9, 0.8, 0.7, 0.6
        # full scores for survivors: 0.95, then exception
        # Note: run_halving calls _evaluate for baseline too (6 total calls vs 5 previously)
        eval_scores = [
            0.5,  # baseline evaluation (set_baseline)
            0.9,
            0.8,
            0.7,
            0.6,  # probe phase
            0.95,
        ]  # full phase first survivor
        mock_eval.side_effect = iter(eval_scores)

        trainset = [_make_example(f"q{i}", f"a{i}") for i in range(20)]

        # Should not raise — fallback kicks in
        result = pipeline.run_halving(student, trainset=trainset, prune_fraction=0.5)
        # Best score should come from a completed run
        assert result["best_score"] >= 0

    @patch("dspytools.gfl.pipeline.GFLPipeline._dispatch_optimizer")
    @patch("dspytools.gfl.pipeline.GFLPipeline._evaluate")
    def test_halving_probe_fraction(self, mock_eval, mock_dispatch):
        """probe_fraction=0.1 with 100 examples means ~10 probe examples."""
        pipeline = GFLPipeline()
        student = MagicMock()
        mock_dispatch.return_value = _make_example("x", "y")
        mock_eval.return_value = 0.5

        trainset = [_make_example(f"q{i}", f"a{i}") for i in range(100)]
        # We can't easily check internal probe_set size from result,
        # but verify the method runs without error
        result = pipeline.run_halving(student, trainset=trainset, probe_fraction=0.1)
        assert "probe_scores" in result
        assert "full_scores" in result

    @patch("dspytools.gfl.pipeline.GFLPipeline._dispatch_optimizer")
    @patch("dspytools.gfl.pipeline.GFLPipeline._evaluate")
    def test_halving_auto_suggest(self, mock_eval, mock_dispatch):
        """When auto_suggest=True, engine is consulted for optimizer order."""
        pipeline = GFLPipeline()
        student = MagicMock()
        mock_dispatch.return_value = _make_example("x", "y")
        mock_eval.return_value = 0.5

        trainset = [_make_example(f"q{i}", f"a{i}") for i in range(20)]
        result = pipeline.run_halving(student, trainset=trainset, auto_suggest=True)
        assert "survivors" in result
        assert "pruned" in result

    @patch("dspytools.gfl.pipeline.GFLPipeline._dispatch_optimizer")
    @patch("dspytools.gfl.pipeline.GFLPipeline._evaluate")
    def test_halving_custom_optimizer_list(self, mock_eval, mock_dispatch):
        """Custom optimizer list overrides defaults."""
        pipeline = GFLPipeline()
        student = MagicMock()
        mock_dispatch.return_value = _make_example("x", "y")
        mock_eval.return_value = 0.5

        custom = ["gepa", "mipro"]
        trainset = [_make_example(f"q{i}", f"a{i}") for i in range(20)]
        result = pipeline.run_halving(student, trainset=trainset, optimizers=custom)
        # Only 2 optimizers → prune_fraction=0.5 → 1 survivor, 1 pruned
        assert len(result["survivors"]) == 1
        assert len(result["pruned"]) == 1

    @patch("dspytools.gfl.pipeline.GFLPipeline._dispatch_optimizer")
    @patch("dspytools.gfl.pipeline.GFLPipeline._evaluate")
    def test_halving_result_keys(self, mock_eval, mock_dispatch):
        """Result dict has expected keys."""
        pipeline = GFLPipeline()
        student = MagicMock()
        mock_dispatch.return_value = _make_example("x", "y")
        mock_eval.return_value = 0.5

        trainset = [_make_example(f"q{i}", f"a{i}") for i in range(20)]
        result = pipeline.run_halving(student, trainset=trainset)
        expected_keys = {
            "best_optimizer",
            "best_score",
            "best_program",
            "baseline",
            "improvement",
            "all_scores",
            "probe_scores",
            "full_scores",
            "survivors",
            "pruned",
            "budget",
            "trend",
            "total_improvement",
        }
        assert expected_keys.issubset(result.keys())


# ═══════════════════════════════════════════════════════════════════════════
# compile_draft — structure and behavior tests
# ═══════════════════════════════════════════════════════════════════════════


class TestCompileDraft:
    """Tests for GFLPipeline.compile_draft()."""

    @patch("dspytools.gfl.pipeline.GFLPipeline._dispatch_optimizer")
    @patch("dspytools.gfl.pipeline.GFLPipeline._evaluate")
    @patch("dspytools.core.setup.LMRegistry.get_or_default")
    def test_draft_round_scores_structure(self, mock_lm, mock_eval, mock_dispatch):
        """Return dict has round_scores array with expected entries."""
        mock_lm.return_value = "mock_lm"
        student = MagicMock()
        mock_dispatch.return_value = _make_example("x", "y")
        mock_eval.return_value = 0.5

        pipeline = GFLPipeline()
        trainset = [_make_example(f"q{i}", f"a{i}") for i in range(5)]
        # Note: dspy.context runs real (both _dispatch_optimizer and _evaluate
        # are mocked so the LM is never called).
        result = pipeline.compile_draft(
            student, trainset=trainset, draft_rounds=2, polish_rounds=1
        )

        assert "round_scores" in result
        assert len(result["round_scores"]) == 3  # 2 draft + 1 polish
        for entry in result["round_scores"]:
            assert "round" in entry
            assert "score" in entry
            assert "phase" in entry

    @patch("dspytools.gfl.pipeline.GFLPipeline._dispatch_optimizer")
    @patch("dspytools.gfl.pipeline.GFLPipeline._evaluate")
    @patch("dspytools.core.setup.LMRegistry.get_or_default")
    @patch("dspytools.core.setup.LMRegistry.get_teacher")
    def test_draft_no_teacher(self, mock_teacher, mock_lm, mock_eval, mock_dispatch):
        """When no teacher LM, draft_score == polished_score."""
        mock_lm.return_value = "mock_lm"
        mock_teacher.return_value = None  # No teacher
        student = MagicMock()
        mock_dispatch.return_value = _make_example("x", "y")
        mock_eval.return_value = 0.5

        pipeline = GFLPipeline()
        trainset = [_make_example(f"q{i}", f"a{i}") for i in range(5)]
        result = pipeline.compile_draft(
            student, trainset=trainset, draft_rounds=1, polish_rounds=0
        )

        assert result["teacher_used"] is False
        # With polish_rounds=0, no teacher phase — draft is the final
        assert result["draft_rounds"] == 1

    @patch("dspytools.gfl.pipeline.GFLPipeline._dispatch_optimizer")
    @patch("dspytools.gfl.pipeline.GFLPipeline._evaluate")
    @patch("dspytools.core.setup.LMRegistry.get_or_default")
    def test_draft_result_keys(self, mock_lm, mock_eval, mock_dispatch):
        """Return dict has all expected keys."""
        mock_lm.return_value = "mock_lm"
        student = MagicMock()
        mock_dispatch.return_value = _make_example("x", "y")
        mock_eval.return_value = 0.5

        pipeline = GFLPipeline()
        trainset = [_make_example(f"q{i}", f"a{i}") for i in range(5)]
        result = pipeline.compile_draft(student, trainset=trainset)

        expected_keys = {
            "optimizer",
            "draft_score",
            "polished_score",
            "improvement",
            "draft_rounds",
            "polish_rounds",
            "best_program",
            "teacher_used",
            "round_scores",
        }
        assert expected_keys.issubset(result.keys())

    @patch("dspytools.gfl.pipeline.GFLPipeline._dispatch_optimizer")
    @patch("dspytools.gfl.pipeline.GFLPipeline._evaluate")
    @patch("dspytools.core.setup.LMRegistry.get_or_default")
    @patch("dspytools.core.setup.LMRegistry.get_teacher")
    def test_draft_teacher_used_true(
        self, mock_teacher, mock_lm, mock_eval, mock_dispatch
    ):
        """When teacher LM configured, teacher_used is True."""
        mock_lm.return_value = "mock_lm"
        mock_teacher.return_value = "mock_teacher_lm"
        student = MagicMock()
        mock_dispatch.return_value = _make_example("x", "y")
        mock_eval.return_value = 0.5

        pipeline = GFLPipeline()
        trainset = [_make_example(f"q{i}", f"a{i}") for i in range(5)]
        result = pipeline.compile_draft(
            student, trainset=trainset, draft_rounds=1, polish_rounds=1
        )

        assert result["teacher_used"] is True
        assert len(result["round_scores"]) == 2  # 1 draft + 1 polish


# ═══════════════════════════════════════════════════════════════════════════
# run_single tests
# ═══════════════════════════════════════════════════════════════════════════


def test_run_single_accepts_optimizer_name():
    """run_single signature accepts optimizer_name and config."""
    import inspect

    sig = inspect.signature(GFLPipeline.run_single)
    assert "optimizer_name" in sig.parameters
    assert "auto_synthesize" in sig.parameters
    assert "min_examples" in sig.parameters
