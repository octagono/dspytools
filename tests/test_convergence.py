"""Tests for convergence guardrails — Goodhart detection, output degradation.

Tests the SelfEvolveEngine's detect_metric_cheating, detect_output_degradation,
and check_convergence methods. Pure logic — no LM calls, no DSPy import needed.
"""

from __future__ import annotations

# ═══════════════════════════════════════════════════════════════════════════
# detect_metric_cheating — repetition detection
# ═══════════════════════════════════════════════════════════════════════════


class TestDetectMetricCheating:
    """SelfEvolveEngine.detect_metric_cheating detects repetitive outputs.

    Uses max repetition frequency (mode ratio): if the most common output
    appears more than `repetition_threshold` (default 85%) of the time,
    the model may be collapsing to a single answer (Goodhart).
    """

    def _detect(self, outputs, threshold=0.85):
        from dspytools.evolve.self_evolve import SelfEvolveEngine

        return SelfEvolveEngine.detect_metric_cheating(outputs, threshold)

    def test_all_unique(self):
        """All unique outputs → no cheating (max_repetition_ratio = 0.2)."""
        result = self._detect(["a", "b", "c", "d", "e"])
        assert result["cheating"] is False
        assert result["max_repetition_ratio"] == 0.2

    def test_all_identical(self):
        """All identical → cheating (max_repetition_ratio = 1.0)."""
        result = self._detect(["hello"] * 10)
        assert result["cheating"] is True
        assert result["max_repetition_ratio"] == 1.0

    def test_mostly_identical(self):
        """90% identical → cheating (max_repetition_ratio = 0.9 > 0.85)."""
        outputs = ["hello"] * 9 + ["world"]
        result = self._detect(outputs)
        assert result["cheating"] is True
        assert result["max_repetition_ratio"] == 0.9

    def test_mostly_unique(self):
        """80% unique output per value → no cheating."""
        outputs = ["a"] * 2 + ["b", "c", "d", "e", "f", "g", "h", "i"]
        result = self._detect(outputs)
        assert result["cheating"] is False
        assert result["max_repetition_ratio"] == 0.2  # 2/10

    def test_empty_input(self):
        """Empty list → safe (no cheating)."""
        result = self._detect([])
        assert result["cheating"] is False
        assert result["n_samples"] == 0

    def test_single_output(self):
        """Single output → not cheating (below min_samples)."""
        result = self._detect(["only"])
        assert result["cheating"] is False
        assert result["max_repetition_ratio"] == 0.0  # below min_samples

    def test_threshold_boundary(self):
        """50/50 split with threshold 0.5 is not cheating (0.5 not > 0.5)."""
        outputs = ["hello"] * 5 + ["world"] * 5
        result = self._detect(outputs, threshold=0.5)
        assert result["cheating"] is False  # 0.5 NOT > 0.5

    def test_just_above_threshold(self):
        """51% > 50% threshold → cheating."""
        outputs = ["hello"] * 6 + ["world"] * 4  # 6/10 = 0.6
        result = self._detect(outputs, threshold=0.5)
        assert result["cheating"] is True  # 0.6 > 0.5

    def test_trigger_message_present(self):
        """Cheating result includes a trigger message mentioning repetition."""
        result = self._detect(["same"] * 5)
        assert result["trigger"] is not None
        assert "repetition" in result["trigger"].lower()

    def test_trigger_message_absent_when_safe(self):
        """Safe result has None trigger."""
        result = self._detect(["a", "b", "c"])
        assert result["trigger"] is None


# ═══════════════════════════════════════════════════════════════════════════
# detect_output_degradation — stagnation/oscillation detection
# ═══════════════════════════════════════════════════════════════════════════


class TestDetectOutputDegradation:
    """SelfEvolveEngine.detect_output_degradation detects suspicious score patterns."""

    def _detect(self, scores, window=5, threshold=0.01):
        from dspytools.evolve.self_evolve import SelfEvolveEngine

        return SelfEvolveEngine.detect_output_degradation(scores, window, threshold)

    def test_improving_scores(self):
        """Consistently improving → not degraded."""
        scores = [0.5, 0.6, 0.7, 0.8, 0.9]
        result = self._detect(scores)
        assert result["degraded"] is False
        assert len(result["triggers"]) == 0

    def test_stagnant_scores(self):
        """All identical scores → degraded (stagnation)."""
        scores = [0.7, 0.7, 0.7, 0.7, 0.7]
        result = self._detect(scores)
        assert result["degraded"] is True
        assert any("stagnation" in t.lower() for t in result["triggers"])

    def test_oscillating_scores(self):
        """Strong alternating high/low → degraded (oscillation)."""
        scores = [0.9, 0.3, 0.9, 0.3, 0.9]
        result = self._detect(scores)
        assert result["degraded"] is True
        assert result["oscillating"] is True

    def test_small_oscillation_not_flagged(self):
        """Small wobbles in improving trend → not oscillation (amplitude too low)."""
        scores = [0.5, 0.7, 0.6, 0.8, 0.7]
        result = self._detect(scores)
        # Amplitude = 0.3 > 0.2, but check whether it's flagged
        # The oscillation needs AMPLITUDE > 0.2 and all signs alternating
        # diffs = [0.2, -0.1, 0.2, -0.1] → 3/3 sign changes → oscillating=True
        # But amplitude_check: not all within 0.2
        assert result["oscillating"] is False

    def test_insufficient_data(self):
        """Less than window samples → not degraded."""
        result = self._detect([0.5, 0.6], window=5)
        assert result["degraded"] is False
        assert result["n_samples"] == 2

    def test_empty_scores(self):
        """Empty list → not degraded."""
        result = self._detect([])
        assert result["degraded"] is False

    def test_wide_variance_not_degraded(self):
        """Scores with healthy variance but no pattern → not degraded."""
        scores = [0.5, 0.7, 0.6, 0.8, 0.7]
        result = self._detect(scores)
        assert result["degraded"] is False

    def test_larger_window_stagnant(self):
        """Stagnation detected with larger window."""
        scores = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
        result = self._detect(scores, window=7)
        assert result["degraded"] is True

    def test_variance_threshold_custom(self):
        """Higher variance threshold catches smaller fluctuations as stagnation."""
        scores = [0.7, 0.71, 0.7, 0.71, 0.7]
        result = self._detect(scores, threshold=0.02)
        # variance = 0.01 < 0.02 → flagged as stagnation
        assert result["degraded"] is True

    def test_lower_variance_threshold_no_stagnation(self):
        """Lower variance threshold doesn't flag small fluctuations."""
        scores = [0.7, 0.71, 0.7, 0.71, 0.7]
        result = self._detect(scores, threshold=0.005)
        # variance = 0.01 >= 0.005 → not stagnant
        assert result["degraded"] is False

    def test_mean_calculated(self):
        """Mean of recent scores is computed."""
        scores = [0.5, 0.6, 0.7, 0.8, 0.9]
        result = self._detect(scores)
        assert result["mean"] == 0.7

    def test_no_oscillation_with_improvement(self):
        """Monotonic improvement → not oscillating."""
        scores = [0.1, 0.2, 0.3, 0.4, 0.5]
        result = self._detect(scores)
        assert result["oscillating"] is False


# ═══════════════════════════════════════════════════════════════════════════
# check_convergence — unified guardrail
# ═══════════════════════════════════════════════════════════════════════════


class TestCheckConvergence:
    """SelfEvolveEngine.check_convergence unifies repetition + degradation checks."""

    def _engine(self):
        from dspytools.evolve.self_evolve import SelfEvolveEngine

        engine = SelfEvolveEngine()
        engine._clear_state()
        return engine

    def test_safe_outputs(self):
        """Diverse outputs, improving scores → safe."""
        engine = self._engine()
        engine.record_score(0.5)
        engine.record_score(0.7)
        engine.record_score(0.9)
        result = engine.check_convergence(["a", "b", "c"])
        assert result["safe"] is True

    def test_repetition_warning(self):
        """Identical outputs → warning."""
        engine = self._engine()
        engine.record_score(0.5)
        result = engine.check_convergence(["same"] * 10)
        assert result["safe"] is False
        assert result["repetition_warning"] is not None

    def test_degradation_warning(self):
        """Stagnant scores → warning."""
        engine = self._engine()
        for _ in range(10):
            engine.record_score(0.5)
        result = engine.check_convergence(["a", "b", "c"])
        assert result["safe"] is False
        assert len(result["degradation_warning"]) > 0

    def test_both_warnings(self):
        """Both repetition and degradation → multiple warnings."""
        engine = self._engine()
        for _ in range(10):
            engine.record_score(0.5)
        result = engine.check_convergence(["same"] * 10)
        assert result["safe"] is False
        assert result["repetition_warning"] is not None
        assert len(result["degradation_warning"]) > 0

    def test_prediction_cache_bounded(self):
        """Prediction cache stays at max 50 entries."""
        engine = self._engine()
        engine.record_score(0.5)
        engine.check_convergence(["a"] * 100)
        assert len(engine._prediction_cache) <= 50

    def test_score_history_bounded(self):
        """Score history stays at max 100 entries."""
        engine = self._engine()
        for i in range(200):
            engine.record_score(i % 10 / 10)
        assert len(engine._score_history) <= 100

    def test_consecutive_check_convergence(self):
        """Multiple check_convergence calls accumulate history."""
        engine = self._engine()
        engine.record_score(0.5)
        engine.record_score(0.7)

        r1 = engine.check_convergence(["a"])
        assert r1["safe"] is True

        r2 = engine.check_convergence(["a"] * 20)
        assert r2["safe"] is False
        assert r2["repetition_warning"] is not None

    def test_clear_state(self):
        """New engine starts with empty caches."""
        engine = self._engine()
        assert len(engine._prediction_cache) == 0
        assert len(engine._score_history) == 0

    def test_max_repetition_ratio_in_result(self):
        """Result includes max_repetition_ratio from repetition check."""
        engine = self._engine()
        result = engine.check_convergence(["a", "b", "c", "a"])
        assert "max_repetition_ratio" in result
        # "a" appears 2/4 times, "b" and "c" once each
        assert result["max_repetition_ratio"] == 0.5

    def test_score_variance_in_result(self):
        """Result includes score_variance from degradation check (window=5)."""
        engine = self._engine()
        for s in [0.5, 0.6, 0.7, 0.8, 0.9]:
            engine.record_score(s)
        result = engine.check_convergence(["a"])
        assert "score_variance" in result
        assert result["score_variance"] == 0.4  # max - min = 0.9 - 0.5


# ═══════════════════════════════════════════════════════════════════════════
# Integration: on_compile triggers record_score
# ═══════════════════════════════════════════════════════════════════════════


class TestOnCompileScoreRecording:
    """on_compile automatically records the score for convergence tracking."""

    def test_on_compile_records_score(self):
        """Calling on_compile appends to score_history."""
        from dspytools.evolve.self_evolve import SelfEvolveEngine

        engine = SelfEvolveEngine()
        engine._clear_state()
        engine.on_compile("test_profile", "gepa", 0.85, success=True)
        assert len(engine._score_history) == 1
        assert engine._score_history[0] == 0.85

    def test_multiple_on_compile(self):
        """Multiple on_compile calls accumulate scores."""
        from dspytools.evolve.self_evolve import SelfEvolveEngine

        engine = SelfEvolveEngine()
        engine._clear_state()
        engine.on_compile("p1", "mipro", 0.7)
        engine.on_compile("p2", "gepa", 0.85)
        engine.on_compile("p3", "copro", 0.9)
        assert engine._score_history == [0.7, 0.85, 0.9]
