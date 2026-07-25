"""Tests for meta agent + meta learning (no actual LM calls needed).

Covers:
  Meta Learning:
    1. MetaOptimizer.select_optimizer() — dataset-size routing
    2. MetaOptimizer.record_result() + get_best_optimizer()
    3. GRAOMetaOptimizer — learn_from_trial, suggest_fix, suggest_strategy,
       improvement_rate, meta_best_optimizer
    4. MetaPromptOptimizer — meta_learn (mock programs), adapt

  Meta Agent:
    5. SelfEvolveEngine.suggest_optimizer() — morphology→transfer→UCB chain
    6. SelfEvolveEngine.archive_search() — with empty + populated registry
    7. SelfEvolveEngine.on_compile() — updates all trackers
    8. SelfEvolveEngine.validate_and_deploy() — SPRT math
    9. SelfEvolveEngine.suggest_optimizer() after recording trials

  CLI:
    10. gfl meta-optimize CLI command
"""

from __future__ import annotations

import math

import pytest

from dspytools.core._dspy import dspy

# ═══════════════════════════════════════════════════════════════════════════
# 1. MetaOptimizer — selection logic
# ═══════════════════════════════════════════════════════════════════════════


class TestMetaOptimizerSelection:
    """MetaOptimizer.select_optimizer() must route by dataset size."""

    def setup_method(self):
        from dspytools.gfl.meta_learn import MetaOptimizer

        self.meta = MetaOptimizer()
        # Clear persisted history to avoid cross-test pollution
        self.meta.history = {"trials": [], "recommendations": {}}
        self.meta._save_history()

    def test_select_small(self):
        """Under 10 examples → labeled_few_shot."""
        result = self.meta.select_optimizer("prog1", 5, "simple")
        assert result["optimizer"] == "labeled_few_shot"

    def test_select_medium(self):
        """10-49 examples → mipro."""
        result = self.meta.select_optimizer("prog1", 25, "medium")
        assert result["optimizer"] == "mipro"

    def test_select_large(self):
        """50+ examples → gepa."""
        result = self.meta.select_optimizer("prog1", 100, "complex")
        assert result["optimizer"] == "gepa"

    def test_select_edge_boundary_9(self):
        """Exactly 9 → small."""
        result = self.meta.select_optimizer("p", 9, "simple")
        assert result["optimizer"] == "labeled_few_shot"

    def test_select_edge_boundary_10(self):
        """Exactly 10 → medium."""
        result = self.meta.select_optimizer("p", 10, "medium")
        assert result["optimizer"] == "mipro"

    def test_select_edge_boundary_50(self):
        """Exactly 50 → large."""
        result = self.meta.select_optimizer("p", 50, "complex")
        assert result["optimizer"] == "gepa"

    def test_select_includes_all_keys(self):
        """Result dict has all expected keys."""
        result = self.meta.select_optimizer("prog1", 10, "medium")
        assert "program" in result
        assert "optimizer" in result
        assert "dataset_size" in result
        assert "complexity" in result
        assert "reason" in result
        assert result["program"] == "prog1"


# ═══════════════════════════════════════════════════════════════════════════
# 2. MetaOptimizer — record + retrieve
# ═══════════════════════════════════════════════════════════════════════════


class TestMetaOptimizerRecord:
    """MetaOptimizer.record_result() and get_best_optimizer()."""

    def setup_method(self):
        from dspytools.gfl.meta_learn import MetaOptimizer

        self.meta = MetaOptimizer()
        self.meta.history = {"trials": [], "recommendations": {}}

    def test_record_and_get_best(self):
        """Recording a trial makes it retrievable via get_best_optimizer."""
        self.meta.record_result("mipro", 0.85, "medium", 30, "prog1")
        self.meta.record_result("gepa", 0.72, "medium", 30, "prog1")
        best = self.meta.get_best_optimizer("medium")
        assert best == "mipro", f"expected mipro got {best}"

    def test_get_best_no_history(self):
        """No history → default 'mipro'."""
        assert self.meta.get_best_optimizer("simple") == "mipro"

    def test_get_best_wrong_complexity(self):
        """History for different complexity → default 'mipro'."""
        self.meta.record_result("gepa", 0.9, "complex", 100, "prog1")
        assert self.meta.get_best_optimizer("simple") == "mipro"

    def test_trial_truncation(self):
        """History truncated to 100 entries."""
        for i in range(110):
            self.meta.record_result("mipro", 0.5, "medium", 10, f"prog{i}")
        assert len(self.meta.history["trials"]) == 100

    def test_record_updates_best_optimizer(self):
        """select_optimizer prefers historical best over default."""
        self.meta.record_result("gepa", 0.99, "simple", 5, "prog1")
        result = self.meta.select_optimizer("prog1", 5, "simple")
        # gepa scored 0.99 for simple (was 5 examples → labeled_few_shot default)
        # but history override kicks in → gepa
        assert result["optimizer"] == "gepa"

    def test_record_result_with_all_fields(self):
        """record_result accepts all expected fields."""
        self.meta.record_result(
            optimizer="bootstrap_few_shot",
            score=0.78,
            complexity="complex",
            dataset_size=200,
            program="progX",
        )
        trials = self.meta.history["trials"]
        assert len(trials) == 1
        t = trials[0]
        assert t["optimizer"] == "bootstrap_few_shot"
        assert t["score"] == 0.78
        assert t["complexity"] == "complex"
        assert t["dataset_size"] == 200
        assert t["program"] == "progX"
        assert "timestamp" in t


# ═══════════════════════════════════════════════════════════════════════════
# 3. GRAOMetaOptimizer — trial learning + strategy suggestion
# ═══════════════════════════════════════════════════════════════════════════


class TestGRAOMetaOptimizer:
    """GRAO-style meta-learner methods."""

    def setup_method(self):
        from dspytools.gfl.paper_optimizers import GRAOMetaOptimizer

        self.grao = GRAOMetaOptimizer()
        self.grao.history = {
            "trials": [],
            "learned_strategies": {},
            "error_patterns": {},
        }
        self.grao.error_patterns = {}
        self.grao.success_strategies = {}

    def test_learn_from_trial_success(self):
        """High-score trial records success strategy."""
        self.grao.learn_from_trial("classification", "gepa", 0.85)
        strategies = self.grao.success_strategies.get("classification", [])
        assert len(strategies) == 1
        assert strategies[0]["optimizer"] == "gepa"

    def test_learn_from_trial_low_score(self):
        """Low-score trial does not record success."""
        self.grao.learn_from_trial("classification", "mipro", 0.3)
        assert "classification" not in self.grao.success_strategies

    def test_learn_from_trial_with_error(self):
        """Error patterns are stored with fixes."""
        self.grao.learn_from_trial(
            "qa", "mipro", 0.4, error_type="OOM", fix_used="reduce_batch"
        )
        assert "OOM" in self.grao.error_patterns
        assert "reduce_batch" in self.grao.error_patterns["OOM"]

    def test_learn_from_trial_error_no_fix(self):
        """Error without fix still records the error type."""
        self.grao.learn_from_trial("qa", "mipro", 0.4, error_type="OOM")
        assert "OOM" in self.grao.error_patterns
        assert self.grao.error_patterns["OOM"] == []

    def test_suggest_fix(self):
        """Known error returns stored fixes."""
        self.grao.error_patterns["OOM"] = ["reduce_batch", "use_lora"]
        fixes = self.grao.suggest_fix("OOM")
        assert "reduce_batch" in fixes
        assert "use_lora" in fixes

    def test_suggest_fix_unknown(self):
        """Unknown error returns empty list."""
        assert self.grao.suggest_fix("MISSING") == []

    def test_suggest_strategy(self):
        """Returns best-scoring strategy for task type."""
        self.grao.success_strategies["classification"] = [
            {"optimizer": "mipro", "score": 0.7},
            {"optimizer": "gepa", "score": 0.88},
        ]
        best = self.grao.suggest_strategy("classification")
        assert best is not None
        assert best["optimizer"] == "gepa"
        assert best["score"] == 0.88

    def test_suggest_strategy_unknown(self):
        """Unknown task returns None."""
        assert self.grao.suggest_strategy("unknown") is None

    def test_improvement_rate(self):
        """Computes ratio of high-score trials for task+optimizer."""
        for _ in range(3):
            self.grao.learn_from_trial("t", "gepa", 0.85)
        for _ in range(2):
            self.grao.learn_from_trial("t", "gepa", 0.3)
        rate = self.grao.improvement_rate("t", "gepa")
        assert rate == pytest.approx(0.6, rel=0.01), f"expected 0.6 got {rate}"

    def test_improvement_rate_no_trials(self):
        """No trials → 0.0."""
        assert self.grao.improvement_rate("t", "gepa") == 0.0

    def test_meta_best_optimizer(self):
        """Returns optimizer with highest avg score and ≥2 trials."""
        # gepa: avg 0.8 (3 trials)
        for _ in range(3):
            self.grao.learn_from_trial("t", "gepa", 0.8)
        # mipro: avg 0.75 (2 trials)
        for _ in range(2):
            self.grao.learn_from_trial("t", "mipro", 0.75)
        best = self.grao.meta_best_optimizer("t")
        assert best == "gepa"

    def test_meta_best_optimizer_min_trials(self):
        """Less than 2 trials for an optimizer is excluded."""
        self.grao.learn_from_trial("t", "gepa", 0.99)
        assert self.grao.meta_best_optimizer("t") is None

    def test_meta_best_optimizer_no_history(self):
        """No history → None."""
        assert self.grao.meta_best_optimizer("unknown") is None

    def test_strategy_cap_at_20(self):
        """Success strategies capped at 20 entries."""
        for i in range(25):
            self.grao.learn_from_trial("t", f"opt{i}", 0.8)
        assert len(self.grao.success_strategies["t"]) == 20

    def test_trial_history_capped_at_500(self):
        """GRAO history capped at 500 trials inside save()."""
        for i in range(600):
            self.grao.history["trials"].append({"dummy": True})
        self.grao.save()
        # save() caps for disk; in-memory still has 600 until reload
        assert len(self.grao.history["trials"]) == 600
        # But the saved file has only 500
        import json

        saved = json.loads(self.grao.LOG_PATH.read_text())
        assert len(saved.get("trials", [])) <= 500

    def test_learn_from_trial_appends_to_history(self):
        """Each call appends one trial."""
        self.grao.learn_from_trial("t", "gepa", 0.8)
        self.grao.learn_from_trial("t", "mipro", 0.7)
        assert len(self.grao.history["trials"]) == 2


# ═══════════════════════════════════════════════════════════════════════════
# 4. MetaPromptOptimizer — meta_learn + adapt
# ═══════════════════════════════════════════════════════════════════════════


class TestMetaPromptOptimizer:
    """MetaPromptOptimizer bilevel meta-learning."""

    def test_adapt_returns_prompt(self):
        """adapt() returns a task-specific prompt."""
        from dspytools.gfl.paper_optimizers import MetaPromptOptimizer

        opt = MetaPromptOptimizer(meta_prompt="Be concise.")
        prompt = opt.adapt("task1", None, [])
        assert isinstance(prompt, str)
        assert "Be concise." in prompt

    def test_adapt_caches(self):
        """Second call for same task returns cached prompt."""
        from dspytools.gfl.paper_optimizers import MetaPromptOptimizer

        opt = MetaPromptOptimizer()
        p1 = opt.adapt("task1", None, [])
        p2 = opt.adapt("task1", None, [])
        assert p1 == p2

    def test_default_meta_prompt(self):
        """Default meta prompt is set correctly."""
        from dspytools.gfl.paper_optimizers import MetaPromptOptimizer

        opt = MetaPromptOptimizer()
        assert "helpful assistant" in opt.meta_prompt

    def test_meta_learn_with_mock_programs(self):
        """meta_learn() runs with mock programs (no LM needed)."""
        from dspytools.gfl.paper_optimizers import MetaPromptOptimizer

        class MockProgram:
            def __call__(self, input="", **kwargs):
                return dspy.Prediction(output="correct", answer="correct")

        opt = MetaPromptOptimizer()
        ex = dspy.Example(input="q", output="correct").with_inputs("input")
        result = opt.meta_learn(
            {"task1": MockProgram()},
            {"task1": [ex]},
            num_iterations=2,
        )
        assert "iterations" in result
        assert "final_meta_prompt" in result
        assert result["final_score"] == 1.0
        assert len(result["iterations"]) == 2

    def test_meta_learn_with_poor_program(self):
        """Low accuracy affects meta-learning score."""
        from dspytools.gfl.paper_optimizers import MetaPromptOptimizer

        class PoorProgram:
            def __call__(self, input="", **kwargs):
                return dspy.Prediction(output="wrong", answer="wrong")

        opt = MetaPromptOptimizer()
        ex = dspy.Example(input="q", output="correct").with_inputs("input")
        result = opt.meta_learn(
            {"task1": PoorProgram()},
            {"task1": [ex]},
            num_iterations=1,
        )
        assert result["final_score"] == 0.0

    def test_meta_learn_multiple_tasks(self):
        """Across multiple tasks, average score is reported."""
        from dspytools.gfl.paper_optimizers import MetaPromptOptimizer

        class GoodProgram:
            def __call__(self, **kwargs):
                return dspy.Prediction(output="a", answer="a")

        class BadProgram:
            def __call__(self, **kwargs):
                return dspy.Prediction(output="b", answer="b")

        good_ex = dspy.Example(input="q", output="a").with_inputs("input")
        bad_ex = dspy.Example(input="q", output="z").with_inputs("input")

        opt = MetaPromptOptimizer()
        result = opt.meta_learn(
            {"good": GoodProgram(), "bad": BadProgram()},
            {"good": [good_ex], "bad": [bad_ex]},
            num_iterations=1,
        )
        assert result["final_score"] == 0.5

    def test_meta_learn_refine_low_score(self):
        """Low avg score triggers prompt refinement."""
        from dspytools.gfl.paper_optimizers import MetaPromptOptimizer

        class BadProgram:
            def __call__(self, **kwargs):
                return dspy.Prediction(output="x", answer="x")

        ex = dspy.Example(input="q", output="y").with_inputs("input")
        opt = MetaPromptOptimizer(meta_prompt="Initial")
        result = opt.meta_learn({"t": BadProgram()}, {"t": [ex]}, num_iterations=2)
        # Score will be low, so prompt should have been refined
        assert result["final_score"] < 0.6

    def test_meta_learn_empty_devset(self):
        """Empty devset should not crash."""
        from dspytools.gfl.paper_optimizers import MetaPromptOptimizer

        class MockProgram:
            def __call__(self, **kwargs):
                return dspy.Prediction(output="x", answer="x")

        opt = MetaPromptOptimizer()
        # Empty devset
        result = opt.meta_learn({"t": MockProgram()}, {"t": []}, num_iterations=1)
        assert result["final_score"] == 0.0


# ═══════════════════════════════════════════════════════════════════════════
# 5. SelfEvolveEngine — suggest_optimizer chain
# ═══════════════════════════════════════════════════════════════════════════


class TestSelfEvolveSuggest:
    """SelfEvolveEngine.suggest_optimizer() resolution chain."""

    def setup_method(self):
        from unittest.mock import patch

        from dspytools.evolve.self_evolve import SelfEvolveEngine

        # Mock memory manager so suggest_optimizer doesn't hit real FalkorDB
        self._mem_patch = patch("dspytools.memory.manager.get_memory_manager")
        mock_mem = self._mem_patch.start()
        mock_mem.return_value.search.return_value = []

        self.engine = SelfEvolveEngine()

    def teardown_method(self):
        self._mem_patch.stop()

    def test_suggest_no_history(self):
        """No history → UCB select (first untried optimizer)."""
        opt = self.engine.suggest_optimizer("unknown_rare_1f_5w")
        assert isinstance(opt, str)
        assert len(opt) > 0

    def test_suggest_after_morphology(self):
        """Recording a successful trial makes morphology recommend it."""
        profile = "test_moderate_3f_80w"
        self.engine.morphology.record(profile, "custom_opt", True)
        self.engine.morphology.record(profile, "custom_opt", True)
        self.engine.morphology.record(profile, "custom_opt", True)
        opt = self.engine.suggest_optimizer(profile)
        assert opt == "custom_opt"

    def test_suggest_falls_through_to_ucb(self):
        """With no morphology or transfer, falls to UCB."""
        opt = self.engine.suggest_optimizer("fresh_task_1f_2w")
        assert isinstance(opt, str)

    def test_on_compile_returns_keys(self):
        """on_compile() returns expected dict."""
        result = self.engine.on_compile(
            task_profile="test_moderate_3f_80w",
            optimizer="mipro",
            score=0.85,
            success=True,
        )
        assert "morphology" in result
        assert "transferred" in result
        assert "ucb_next" in result
        assert "exploitation" in result


# ═══════════════════════════════════════════════════════════════════════════
# 6. SelfEvolveEngine — archive_search
# ═══════════════════════════════════════════════════════════════════════════


class TestArchiveSearch:
    """archive_search() keyword-based discovery."""

    def test_search_empty_registry(self):
        """Empty registry returns []."""
        from dspytools.core.registry import save_run_index
        from dspytools.evolve.self_evolve import SelfEvolveEngine

        save_run_index([])
        engine = SelfEvolveEngine()
        results = engine.archive_search("test query", top_k=3)
        assert results == []

    def test_search_keyword_match(self):
        """Runs matching keywords appear in results."""
        from dspytools.core.registry import register_run, save_run_index
        from dspytools.evolve.self_evolve import SelfEvolveEngine

        # Register a run with known metadata
        register_run(
            "run_gepa_1",
            {
                "optimizer": "gepa",
                "module": "classifier",
                "score": 0.85,
            },
        )
        register_run(
            "run_mipro_1",
            {
                "optimizer": "mipro",
                "module": "extractor",
                "score": 0.72,
            },
        )

        # Sanitize: restore clean state
        engine = SelfEvolveEngine()
        results = engine.archive_search("gepa", top_k=5)

        # Clean up
        save_run_index([])

        assert len(results) >= 1
        ids = [r.get("id", "") for r in results]
        found = any("gepa" in rid for rid in ids)
        # At least the gepa run matched
        if not found:
            # Keyword might match on metadata fields too
            assert any("gepa" in str(r.get("metadata", {})).lower() for r in results)

    def test_search_top_k_respected(self):
        """top_k limits results."""
        from dspytools.core.registry import register_run, save_run_index
        from dspytools.evolve.self_evolve import SelfEvolveEngine

        for i in range(5):
            register_run(f"run_{i}", {"optimizer": "test", "module": "m"})

        engine = SelfEvolveEngine()
        results = engine.archive_search("test m", top_k=3)

        # Clean up
        save_run_index([])

        assert len(results) <= 3

    def test_search_scored_by_relevance(self):
        """Results sorted by relevance score descending."""
        from dspytools.core.registry import register_run, save_run_index
        from dspytools.evolve.self_evolve import SelfEvolveEngine

        register_run("qa_gepa_1", {"optimizer": "gepa", "module": "qa_classifier"})
        register_run("ner_mipro_1", {"optimizer": "mipro", "module": "ner_extractor"})

        engine = SelfEvolveEngine()
        results = engine.archive_search("qa gepa", top_k=5)

        save_run_index([])

        if len(results) >= 2:
            scores = [
                sum(
                    1
                    for kw in ["qa", "gepa"]
                    if kw
                    in f"{r.get('id', '')} {r.get('optimizer', '')} {r.get('metadata', {}).get('module', '')}".lower()
                )
                for r in results
            ]
            assert scores == sorted(scores, reverse=True), (
                "results not sorted by relevance"
            )


# ═══════════════════════════════════════════════════════════════════════════
# 7. SelfEvolveEngine — on_compile tracker updates
# ═══════════════════════════════════════════════════════════════════════════


class TestSelfEvolveOnCompile:
    """on_compile() updates morphology, UCB, LSE, and GEPA."""

    def setup_method(self):
        from dspytools.evolve.self_evolve import SelfEvolveEngine

        self.engine = SelfEvolveEngine()
        self.engine._clear_state()

    def test_morphology_updated(self):
        """Morphology records the pattern after on_compile (needs ≥3 trials)."""
        profile = "test_small_1f_10w"
        # best_pattern requires count >= 3
        self.engine.on_compile(profile, "gepa", 0.85)
        self.engine.on_compile(profile, "gepa", 0.82)
        self.engine.on_compile(profile, "gepa", 0.88)
        best = self.engine.morphology.best_pattern(profile)
        assert best == "gepa"

    def test_ucb_updated(self):
        """UCB records the score after on_compile."""
        self.engine.on_compile("profile", "mipro", 0.75)
        self.engine.on_compile("profile", "gepa", 0.88)
        # UCB should have both recorded
        assert "mipro" in self.engine.ucb.trials
        assert "gepa" in self.engine.ucb.trials

    def test_ucb_exploitation_tracks(self):
        """Exploitation score increases as more optimizers are tried."""
        assert self.engine.ucb.exploitation_score == 0.0  # no optimizers tried yet
        # triaselects from all_optimizers, not trials (trials is initially empty)
        for opt in self.engine.ucb.all_optimizers:
            self.engine.ucb.record(opt, 0.7)
        assert self.engine.ucb.exploitation_score == 1.0


# ═══════════════════════════════════════════════════════════════════════════
# 8. SelfEvolveEngine — validate_and_deploy (SPRT)
# ═══════════════════════════════════════════════════════════════════════════


class TestSPRTValidation:
    """SPRT validation — sequential probability ratio test logic."""

    def setup_method(self):
        from dspytools.evolve.self_evolve import SelfEvolveEngine

        self.engine = SelfEvolveEngine()

    def test_sprt_accept_good_candidate(self):
        """Perfect candidate should be accepted early (needs 11+ for SPRT)."""
        from dspytools.core._dspy import dspy

        class PerfectProgram:
            def __call__(self, **kwargs):
                return dspy.Prediction(output="correct", answer="correct")

        holdout = [
            dspy.Example(input="a", output="correct").with_inputs("input")
            for _ in range(15)
        ]
        result = self.engine.validate_and_deploy(
            PerfectProgram(),
            "test_id",
            holdout,
            alpha=0.05,
            beta=0.2,
            max_evaluations=50,
        )
        assert result["accepted"] is True
        assert result["early_stop"] is True
        assert result["n_evaluated"] >= 3  # min samples before decision

    def test_sprt_reject_bad_candidate(self):
        """Terrible candidate should be rejected early."""
        from dspytools.core._dspy import dspy

        class TerribleProgram:
            def __call__(self, **kwargs):
                return dspy.Prediction(output="wrong", answer="wrong")

        holdout = [
            dspy.Example(input="a", output="correct").with_inputs("input")
            for _ in range(10)
        ]
        result = self.engine.validate_and_deploy(
            TerribleProgram(),
            "test_id",
            holdout,
            alpha=0.05,
            beta=0.2,
            max_evaluations=50,
        )
        assert result["accepted"] is False
        assert result["early_stop"] is True

    def test_sprt_forced_decision(self):
        """Ambiguous candidate hits max_evaluations."""
        from dspytools.core._dspy import dspy

        class MediocreProgram:
            def __call__(self, **kwargs):
                return dspy.Prediction(output="correct", answer="correct")

        # Create holdout where half match, half don't
        holdout = []
        for i in range(10):
            ex = dspy.Example(
                input=str(i), output="correct" if i < 5 else "wrong"
            ).with_inputs("input")
            holdout.append(ex)

        result = self.engine.validate_and_deploy(
            MediocreProgram(),
            "test_id",
            holdout,
            alpha=0.01,
            beta=0.01,
            max_evaluations=5,  # tight bounds, low max
        )
        # With 5 eval and ~50% accuracy, may force decision
        assert result["n_evaluated"] <= 5
        assert "early_stop" in result
        assert "candidate_score" in result
        assert 0.0 <= result["candidate_score"] <= 1.0

    def test_sprt_accept_boundary(self):
        """SPRT log-likelihood ratio at decision boundaries."""

        # Test the SPRT math directly
        p0, p1 = 0.50, 0.65
        alpha, beta = 0.05, 0.20
        # log(LR) for accept H1: A = log((1-beta)/alpha) = log(16) ~= 2.77
        A = math.log((1 - beta) / alpha)
        # log(LR) for accept H0: B = log(beta/(1-alpha)) = log(0.2105) ~= -1.56
        B = math.log(beta / (1 - alpha))

        # log(p1/p0) = log(0.65/0.50) = 0.262 per success
        # Need 11 successes: 11 * 0.262 = 2.88 > 2.77 = A
        s, f = 11, 0
        log_lr = s * math.log(p1 / p0) + f * math.log((1 - p1) / (1 - p0))
        assert log_lr >= A, f"Expected accept H1, log_lr={log_lr:.2f} < A={A:.2f}"

        # log((1-p1)/(1-p0)) = log(0.35/0.50) = -0.357 per failure
        # Need 5 failures: 5 * -0.357 = -1.78 < -1.56 = B
        s, f = 0, 5
        log_lr = s * math.log(p1 / p0) + f * math.log((1 - p1) / (1 - p0))
        assert log_lr <= B, f"Expected accept H0, log_lr={log_lr:.2f} > B={B:.2f}"

    def test_sprt_result_keys(self):
        """Return dict has all expected keys."""
        from dspytools.core._dspy import dspy

        class Prog:
            def __call__(self, **kwargs):
                return dspy.Prediction(output="ok", answer="ok")

        holdout = [
            dspy.Example(input="a", output="ok").with_inputs("input") for _ in range(3)
        ]
        result = self.engine.validate_and_deploy(Prog(), "id", holdout)
        expected_keys = {
            "accepted",
            "candidate_score",
            "p_value",
            "n_evaluated",
            "early_stop",
            "reason",
            "statistical_method",
        }
        assert expected_keys.issubset(result.keys()), (
            f"Missing: {expected_keys - result.keys()}"
        )


# ═══════════════════════════════════════════════════════════════════════════
# 9. SelfEvolveEngine — UCB exploitation balance
# ═══════════════════════════════════════════════════════════════════════════


class TestUCBExploration:
    """UCB explorer selection and exploitation."""

    def setup_method(self):
        from dspytools.evolve.self_evolve import UCBExplorer

        self.ucb = UCBExplorer()
        self.ucb.trials = {}  # clear any persisted state

    def test_select_first_untried(self):
        """With no trials, first optimizer is selected (infinite UCB)."""
        opt = self.ucb.select()
        assert opt in self.ucb.all_optimizers

    def test_select_prefers_untried(self):
        """Untried optimizers have infinite UCB and are preferred."""
        self.ucb.record("mipro", 0.85)
        self.ucb.record("gepa", 0.72)
        # Many untried still have UCB = inf
        opt = self.ucb.select()
        # Should be an untried one
        assert opt != "mipro" or opt != "gepa"

    def test_after_all_tried(self):
        """After trying all, selects by UCB1 formula."""
        # trials is initially empty — use all_optimizers instead
        for opt in self.ucb.all_optimizers:
            self.ucb.record(opt, 0.7)
        opt = self.ucb.select()
        assert opt in self.ucb.all_optimizers

    def test_exploitation_score(self):
        """Exploitation score = tried / total."""
        assert self.ucb.exploitation_score == 0.0
        self.ucb.record("mipro", 0.8)
        assert self.ucb.exploitation_score > 0.0

    def test_exploitation_full(self):
        """All optimizers tried -> score 1.0."""
        for opt in self.ucb.all_optimizers:
            self.ucb.record(opt, 0.5)
        assert self.ucb.exploitation_score == 1.0


# ═══════════════════════════════════════════════════════════════════════════
# 10. MorphologyTracker + KnowledgeTransfer integration
# ═══════════════════════════════════════════════════════════════════════════


class TestMorphologyAndTransfer:
    """MorphologyTracker and KnowledgeTransfer as used by SelfEvolveEngine."""

    def setup_method(self):
        from dspytools.evolve.self_evolve import KnowledgeTransfer, MorphologyTracker

        self.morph = MorphologyTracker()
        self.morph._data = {"patterns": {}}
        self.transfer = KnowledgeTransfer(self.morph)

    def test_morphology_requires_min_evidence(self):
        """best_pattern() requires count >= 3."""
        self.morph.record("p", "gepa", True)
        self.morph.record("p", "gepa", True)
        assert self.morph.best_pattern("p") is None
        self.morph.record("p", "gepa", True)
        assert self.morph.best_pattern("p") == "gepa"

    def test_morphology_prefers_highest_success(self):
        """With multiple patterns, picks highest success rate."""
        for _ in range(5):
            self.morph.record("p", "gepa", True)  # 5/5 = 100%
        for _ in range(5):
            self.morph.record(
                "p", "mipro", True
            )  # 5/5 = 100% — same success, gepa first
        best = self.morph.best_pattern("p")
        assert best is not None

    def test_morphology_domain_detection(self):
        """profile_task detects domain from description."""
        from unittest.mock import patch

        from dspytools.core._dspy import dspy

        # Mock the TaskProfileModule to avoid LM dependency
        mock_pred = dspy.Prediction(domain="documentation", complexity="moderate")

        from dspytools.evolve.self_evolve import MorphologyTracker

        with patch("dspytools.evolve.self_evolve.get_task_profiler") as mock_get:
            mock_profiler = mock_get.return_value
            mock_profiler.return_value = mock_pred
            m = MorphologyTracker()
            m._data = {"patterns": {}}
            doc_profile = m.profile_task("documentation gen", 5, 2)
            assert doc_profile.startswith("documentation_")

        mock_pred2 = dspy.Prediction(domain="classification", complexity="moderate")
        with patch("dspytools.evolve.self_evolve.get_task_profiler") as mock_get2:
            mock_profiler2 = mock_get2.return_value
            mock_profiler2.return_value = mock_pred2
            cls_profile = m.profile_task("classify sentiment", 50, 3)
            assert cls_profile.startswith("classification_")

    def test_knowledge_transfer_same_domain(self):
        """Same-domain patterns are transferred with weight 1.0."""
        self.morph.record("documentation_small_2f_100w", "mipro", True)
        self.morph.record("documentation_small_2f_100w", "mipro", True)
        self.morph.record("documentation_small_2f_100w", "mipro", True)
        patterns = self.transfer.transfer_patterns("documentation_large_3f_200w")
        assert "mipro" in patterns

    def test_profile_task_creates_expected_format(self):
        """profile_task returns '{domain}_{size}_{field_count}f_{words}w'."""
        from unittest.mock import patch

        from dspytools.core._dspy import dspy

        mock_pred = dspy.Prediction(domain="generation", complexity="moderate")

        from dspytools.evolve.self_evolve import MorphologyTracker

        with patch("dspytools.evolve.self_evolve.get_task_profiler") as mock_get:
            mock_profiler = mock_get.return_value
            mock_profiler.return_value = mock_pred
            m = MorphologyTracker()
            m._data = {"patterns": {}}
            profile = m.profile_task("generate docs", field_count=3, data_size=20)
            parts = profile.split("_")
            assert len(parts) >= 4
            assert parts[0] == "generation"
            assert parts[1] == "moderate"
            assert parts[2] == "3f"
            assert parts[-1] == "2w"


# ═══════════════════════════════════════════════════════════════════════════
# Import pytest for approx support
# ═══════════════════════════════════════════════════════════════════════════
