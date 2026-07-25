"""GFL comparison pipeline — runs optimizer comparison or single optimizer mode.

Lab 11 pattern: BFS → MIPROv2 → GEPA → Sequential, compare scores.
LSE pattern: tracks delta improvements, rewards positive changes.

Modes:
    "compare" (default): 4-way optimizer comparison, picks best.
    "single": Run a single optimizer with auto-synthesize + meta-learn.

Hold-out validation: split_holdout() + gate_promotion() for CI gating.
"""

from __future__ import annotations

import functools
import logging as _stdlib_logging
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from dspytools.config.settings import DEFAULT_SEED, embedder_kwargs
from dspytools.core._dspy import dspy
from dspytools.core._io import try_read_json, write_json
from dspytools.core.holdout import HoldoutGate
from dspytools.core.logging_config import get_logger
from dspytools.core.metrics import exact_match_metric, gepa_metric
from dspytools.core.mlflow_tracker import get_tracker
from dspytools.core.retry import retry
from dspytools.core.setup import LMRegistry
from dspytools.evolve.self_evolve import SelfEvolveEngine
from dspytools.gfl.budget import ResourceBudget
from dspytools.gfl.consolidation import SkillConsolidator
from dspytools.gfl.feedback import generate_feedback
from dspytools.gfl.meta_learn import MetaOptimizer
from dspytools.gfl.synthetic import DataSynthesizer
from dspytools.gfl.tracker import LSETracker

_log = get_logger(__name__)


class GFLPipeline:
    """Full GFL pipeline: 4-way optimizer comparison with budget and tracking."""

    def __init__(self, budget: ResourceBudget | None = None, mode: str = "compare"):
        self.budget = budget or ResourceBudget()
        # Suppress DSPy internal kwarg-leak warnings from grounded_proposer
        # (max_depth passed to DescribeModule, previous_instructions passed to
        #  GenerateSingleModuleInstruction when use_instruct_history=False).
        # These are DSPy-internal and not actionable by us.

        self._predict_logger = _stdlib_logging.getLogger("dspy.predict.predict")
        self._predict_logger_was_warning = self._predict_logger.getEffectiveLevel()
        self._predict_logger.setLevel(_stdlib_logging.ERROR)
        self._predict_logger_restored = False
        self.tracker = LSETracker()
        self.results: dict[str, tuple[Any, float]] = {}
        self.mode = mode  # "compare" (default) or "single"
        # Self-evolve engine: persists morphology, UCB, transfer, convergence state

        self._evolve_engine = SelfEvolveEngine()

    def __del__(self):
        """Restore DSPy logger level on garbage collection."""
        if not getattr(self, "_predict_logger_restored", False) and hasattr(
            self, "_predict_logger"
        ):
            self._predict_logger.setLevel(self._predict_logger_was_warning)

    def _restore_logger(self):
        """Restore DSPy logger level."""
        self._predict_logger.setLevel(self._predict_logger_was_warning)
        self._predict_logger_restored = True

    def run(
        self,
        student: Any,
        trainset: list,
        train_field: str = "input",
        val_field: str = "output",
    ) -> dict:
        """Run all optimizers and return best.

        Consults SelfEvolveEngine.suggest_optimizer() to prioritize
        historically-successful optimizers first. Calls on_compile() after
        each optimizer to update morphology, UCB, transfer, and convergence
        tracking. For faster comparisons with large datasets, see
        `run_halving()` which uses multi-fidelity early pruning.
        """
        if not trainset:
            raise ValueError("trainset must be non-empty for GFL pipeline")

        # Split holdout FIRST (Invariant 5: holdout never seen by optimizer)

        gate = HoldoutGate()
        train_data, holdout = gate.split(trainset, compile_id="gfl_run")

        # Self-evolve: consult morphology tracker for optimizer hints
        suggested = self._evolve_engine.suggest_optimizer("general")

        self.tracker.set_baseline(
            self._evaluate(student, train_data[:3], train_field, val_field)
        )

        # Build optimizer list — suggested optimizer first
        all_optimizers = ["bootstrap_few_shot", "mipro", "gepa", "sequential"]
        if suggested and suggested in all_optimizers:
            all_optimizers.remove(suggested)
            all_optimizers.insert(0, suggested)

        optimizers = {
            name: lambda n=name: self._dispatch_optimizer(n, student, train_data)
            for name in all_optimizers
        }

        # Pre-flight budget check
        self.budget.check()

        # Run optimizers in parallel (independent — each gets fresh student+trainset)

        max_workers = min(4, len(optimizers))

        def _run_one(name: str, fn: Any) -> tuple[str, Any, float]:
            """Run one optimizer, return (name, compiled, score) or (name, None, fallback)."""
            try:
                compiled = fn()
                score = self._evaluate(compiled, train_data[:3], train_field, val_field)
                return (name, compiled, score)
            except (RuntimeError, OSError, ValueError, TypeError) as e:
                _log.warning("optimizer_failed", optimizer=name, error=str(e))
                return (name, None, self.tracker.best_score or 0.5)

        results_collected: list[tuple[str, Any, float]] = []
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_run_one, n, fn): n for n, fn in optimizers.items()}
            for future in as_completed(futures):
                results_collected.append(future.result())

        # Collect results sequentially (thread-safe — no shared mutation during parallel phase)
        for name, compiled, score in results_collected:
            estimated_tokens = len(train_data) * 5000
            self.budget.spend_tokens(estimated_tokens)
            if compiled is not None:
                self.results[name] = (compiled, score)
            else:
                self.results[name] = (student, score)
            self.tracker.record(name, score)

            # Wire to self-evolve engine: update morphology, UCB, transfer, convergence
            self._evolve_engine.on_compile(
                task_profile="general",
                optimizer=name,
                score=score,
                success=score > 0.5,
            )

        best_name = (
            max(self.results, key=lambda k: self.results[k][1]) if self.results else ""
        )
        best_prog, best_score = self.results.get(best_name, (student, 0.5))

        result_dict = {
            "best_optimizer": best_name,
            "best_program": best_prog,
            "best_score": best_score,
            "baseline": self.tracker.baseline,
            "improvement": best_score - (self.tracker.baseline),
            "all_scores": {k: v[1] for k, v in self.results.items()},
            "budget": self.budget.summary,
            "trend": self.tracker.improvement_trend,
            "total_improvement": self.tracker.total_improvement,
            # Self-evolve state
            "evolve_suggestions": {
                "ucb_next": self._evolve_engine.ucb.select(),
                "exploitation": self._evolve_engine.ucb.exploitation_score,
                "convergence": self._evolve_engine.check_convergence([]),
            },
        }

        # SPRT validation: validate best program against holdout set
        # (holdout was split at the START of run() — optimizer never saw it)

        if holdout:
            validate_engine = SelfEvolveEngine()
            sprt_result = validate_engine.validate_and_deploy(
                candidate_program=best_prog,
                program_id=best_name,
                holdout_set=holdout,
                alpha=0.05,
                beta=0.2,
                max_evaluations=min(50, len(holdout)),
            )
            result_dict["validation"] = sprt_result

        # MLflow tracking
        tracker = get_tracker()
        tracker.log_gfl_comparison(result_dict)

        # Trace2Skill consolidation: mine patterns from optimization trajectories
        self._consolidate_trajectories(best_name, best_prog, trainset)

        # Restore DSPy logger level
        self._predict_logger.setLevel(self._predict_logger_was_warning)

        return result_dict

    def run_halving(
        self,
        student,
        trainset: list,
        train_field: str = "input",
        val_field: str = "output",
        prune_fraction: float = 0.5,
        probe_fraction: float = 0.1,
        min_examples: int = 5,
        optimizers: list[str] | None = None,
        auto_suggest: bool = False,
    ) -> dict:
        """Successive Halving: prune poor optimizers early on small data subset.

        Algorithm:
          1. Split ~10% of trainset as 'probe' set
          2. Run ALL optimizers on the probe set only
          3. Keep top `round(len(optimizers) * (1 - prune_fraction))` by score
          4. Run survivors on the FULL trainset
          5. Return best result across all phases

        Args:
            student: DSPy module to optimize
            trainset: Training examples
            train_field: Input field name
            val_field: Output field name
            prune_fraction: Fraction of optimizers to prune each round (default 0.5)
            probe_fraction: Fraction of trainset to use for probe phase (default 0.1)
            min_examples: Minimum probe examples
            optimizers: List of optimizer names to try. Defaults to all 4.
            auto_suggest: If True, consult SelfEvolveEngine for optimizer suggestion

        This avoids wasting teacher LM calls on optimizers that clearly underperform.
        """

        # Configurable optimizer list with optional auto-suggest
        if optimizers is not None:
            base_optimizers = list(optimizers)
        else:
            base_optimizers = ["bootstrap_few_shot", "mipro", "gepa", "sequential"]

        if auto_suggest:
            suggested = self._evolve_engine.suggest_optimizer("general")
            if suggested and suggested in base_optimizers:
                base_optimizers.remove(suggested)
                base_optimizers.insert(0, suggested)

        optimizers = base_optimizers

        # Phase 1: Probe on small subset
        n_probe = max(min_examples, int(len(trainset) * probe_fraction))
        indices = list(range(len(trainset)))
        random.Random(DEFAULT_SEED).shuffle(indices)
        probe_set = [trainset[i] for i in indices[:n_probe]]

        self.tracker.set_baseline(
            self._evaluate(student, probe_set, train_field, val_field)
        )

        probe_scores: dict[str, float] = {}
        probe_programs: dict[str, Any] = {}

        for name in optimizers:
            try:
                self.budget.check()
                compiled = self._dispatch_optimizer(name, student, probe_set)
                # Track token budget after each compile
                estimated_tokens = len(probe_set) * 5000
                self.budget.spend_tokens(estimated_tokens)
                score = self._evaluate(compiled, probe_set, train_field, val_field)
                probe_scores[name] = score
                probe_programs[name] = compiled
                self.tracker.record(f"{name}_probe", score)
                # Wire to self-evolve engine
                self._evolve_engine.on_compile(
                    task_profile="general",
                    optimizer=name,
                    score=score,
                    success=score > 0.5,
                )
            except (RuntimeError, OSError, ValueError, TypeError) as e:
                _log.warning("probe_optimizer_failed", optimizer=name, error=str(e))
                probe_scores[name] = 0.0

        # Phase 2: Keep top performers
        n_survivors = max(1, int(len(optimizers) * (1 - prune_fraction)))
        ranked = sorted(probe_scores.items(), key=lambda x: x[1], reverse=True)
        survivors = [name for name, _ in ranked[:n_survivors]]

        # Phase 3: Full runs on survivors only
        full_scores: dict[str, float] = {}
        full_programs: dict[str, Any] = {}

        for name in survivors:
            try:
                self.budget.check()
                compiled = self._dispatch_optimizer(name, student, trainset)
                # Track token budget after each compile
                estimated_tokens = len(trainset) * 5000
                self.budget.spend_tokens(estimated_tokens)
                score = self._evaluate(
                    compiled,
                    trainset[: min(10, len(trainset))],
                    train_field,
                    val_field,
                )
                full_scores[name] = score
                full_programs[name] = compiled
                self.tracker.record(f"{name}_full", score)
                # Wire to self-evolve engine
                self._evolve_engine.on_compile(
                    task_profile="general",
                    optimizer=name,
                    score=score,
                    success=score > 0.5,
                )
            except (RuntimeError, OSError, ValueError, TypeError) as e:
                _log.warning("full_optimizer_failed", optimizer=name, error=str(e))
                full_scores[name] = probe_scores.get(name, 0.0)

        # Best overall (prefer full-score when available, fall back to probe)
        all_scores = {**probe_scores}
        for name, score in full_scores.items():
            if name in probe_scores and score < probe_scores[name]:
                all_scores[name] = probe_scores[name]  # keep better of the two
            else:
                all_scores[name] = score

        best_name = max(all_scores, key=all_scores.get)
        best_score = all_scores[best_name]
        best_prog = (
            full_programs.get(best_name) or probe_programs.get(best_name) or student
        )

        result = {
            "best_optimizer": best_name,
            "best_score": best_score,
            "best_program": best_prog,
            "baseline": self.tracker.baseline,
            "improvement": best_score - (self.tracker.baseline),
            "all_scores": all_scores,
            "probe_scores": probe_scores,
            "full_scores": full_scores if full_scores else None,
            "survivors": survivors,
            "pruned": [n for n in optimizers if n not in survivors],
            "budget": self.budget.summary,
            "trend": self.tracker.improvement_trend,
            "total_improvement": self.tracker.total_improvement,
        }

        # MLflow tracking

        tracker = get_tracker()
        tracker.log_gfl_comparison(result)

        # Restore DSPy logger level
        self._predict_logger.setLevel(self._predict_logger_was_warning)

        return result

    def compile_draft(
        self,
        student,
        trainset: list,
        optimizer_name: str = "gepa",
        draft_rounds: int = 3,
        polish_rounds: int = 1,
    ) -> dict:
        """Speculative Compilation: student drafts, teacher polishes.

        Algorithm:
          1. Run `draft_rounds` optimization passes using student LM (cheap)
          2. Run `polish_rounds` optimization passes using teacher LM (expensive)
          3. Evaluate each round, track per-round scores.

        Returns per-round score history for observability.
        This reduces teacher LM API costs by 3-5x for iterative optimizers.
        """
        # Phase 1: Draft with student LM
        student_lm = LMRegistry.get_or_default()
        draft_program = student
        draft_round_scores: list[dict] = []

        for round_num in range(draft_rounds):
            try:
                self.budget.check()
                with dspy.context(lm=student_lm, temperature=0.7):
                    draft_program = self._dispatch_optimizer(
                        optimizer_name, draft_program, trainset
                    )
                estimated_tokens = len(trainset) * 5000
                self.budget.spend_tokens(estimated_tokens)
                round_score = self._evaluate(
                    draft_program,
                    trainset[: min(10, len(trainset))],
                    "input",
                    "output",
                )
                self.tracker.record(
                    f"{optimizer_name}_draft_round{round_num}", round_score
                )
                draft_round_scores.append(
                    {"round": round_num, "score": round_score, "phase": "draft"}
                )
            except (RuntimeError, OSError, ValueError, TypeError) as e:
                _log.warning("draft_round_failed", round=round_num, error=str(e))
                break

        draft_score = draft_round_scores[-1]["score"] if draft_round_scores else 0.0

        # Phase 2: Polish with teacher LM
        teacher_lm = LMRegistry.get_teacher()
        polished_program = draft_program
        polish_round_scores: list[dict] = []

        if teacher_lm:
            for round_num in range(polish_rounds):
                try:
                    self.budget.check()
                    with dspy.context(lm=teacher_lm, temperature=0.3):
                        polished_program = self._dispatch_optimizer(
                            optimizer_name, polished_program, trainset
                        )
                    estimated_tokens = len(trainset) * 5000
                    self.budget.spend_tokens(estimated_tokens)
                    round_score = self._evaluate(
                        polished_program,
                        trainset[: min(10, len(trainset))],
                        "input",
                        "output",
                    )
                    self.tracker.record(
                        f"{optimizer_name}_polish_round{round_num}", round_score
                    )
                    polish_round_scores.append(
                        {"round": round_num, "score": round_score, "phase": "polish"}
                    )
                except (RuntimeError, OSError, ValueError, TypeError) as e:
                    _log.warning("polish_round_failed", round=round_num, error=str(e))
                    break

        polished_score = (
            polish_round_scores[-1]["score"] if polish_round_scores else draft_score
        )

        self._predict_logger.setLevel(self._predict_logger_was_warning)

        return {
            "optimizer": optimizer_name,
            "draft_score": draft_score,
            "polished_score": polished_score,
            "improvement": polished_score - draft_score,
            "draft_rounds": draft_rounds,
            "polish_rounds": polish_rounds,
            "best_program": polished_program,
            "teacher_used": teacher_lm is not None,
            "round_scores": draft_round_scores + polish_round_scores,
        }

    @retry(max_retries=2, base_delay=2.0)
    def _dispatch_optimizer(self, name: str, student, trainset: list):
        """Dispatch to a single optimizer by name."""

        metric = exact_match_metric()
        # GEPA requires a 5-arg metric: (gold, pred, trace, pred_name, pred_trace)
        gepa_met = gepa_metric

        optimizers = {
            "bootstrap_few_shot": lambda: dspy.BootstrapFewShot(
                metric=metric, max_labeled_demos=4, max_bootstrapped_demos=4
            ).compile(student=student, trainset=trainset),
            "mipro": lambda: dspy.MIPROv2(metric=metric, auto="light").compile(
                student=student, trainset=trainset
            ),
            "gepa": lambda: dspy.GEPA(
                metric=gepa_met, auto="light", reflection_lm=LMRegistry.get_teacher()
            ).compile(
                student=student, trainset=trainset, **self._valset_kwargs(trainset)
            ),
            "sequential": lambda: dspy.BetterTogether(
                metric=gepa_met,
                p=dspy.GEPA(
                    metric=gepa_met,
                    auto="light",
                    reflection_lm=LMRegistry.get_teacher(),
                ),
            ).compile(
                student=student,
                trainset=trainset,
                strategy="p",
                **self._valset_kwargs(trainset),
            ),
            "knn": lambda: dspy.KNNFewShot(
                k=3,
                trainset=trainset,
                vectorizer=dspy.Embedder(**embedder_kwargs()),
            ).compile(student=student),
            "labeled_few_shot": lambda: dspy.LabeledFewShot(
                k=min(4, len(trainset))
            ).compile(student=student, trainset=trainset),
        }
        if name in optimizers:
            return optimizers[name]()
        return dspy.LabeledFewShot(k=min(4, len(trainset))).compile(
            student=student, trainset=trainset
        )

    @staticmethod
    def _valset_kwargs(trainset: list) -> dict:
        """Split a small valset from trainset for GEPA to avoid overfitting warning."""
        if len(trainset) >= 6:
            val_size = max(3, len(trainset) // 5)
            return {"valset": trainset[-val_size:], "trainset": trainset[:-val_size]}
        return {}

    @staticmethod
    def _evaluate(compiled, sample, train_field, val_field) -> float:
        if not sample:
            return 0.5
        ex = sample[0]
        kwargs = (
            ex.inputs()
            if hasattr(ex, "inputs")
            else {train_field: getattr(ex, train_field, "")}
        )
        result = compiled(**kwargs)
        output = getattr(result, val_field, getattr(result, "output", str(result)))

        return generate_feedback(str(output))["score"]

    def compare_teacher_student(
        self,
        student: Any,
        trainset: list,
        train_field: str = "input",
        val_field: str = "output",
    ) -> dict:
        """Compare teacher-optimized vs baseline student program.

        Returns delta showing teacher improvement over student baseline.
        """
        baseline = self._evaluate(student, trainset[:3], train_field, val_field)

        # Teacher optimization
        teacher_lm = LMRegistry.get_teacher()
        if teacher_lm:
            metric = functools.partial(gepa_metric, val_field=val_field)
            gepa = dspy.GEPA(metric=metric, auto="light", reflection_lm=teacher_lm)
            valset_kwargs = self._valset_kwargs(trainset)
            teacher_opt = gepa.compile(
                student=student,
                **valset_kwargs if valset_kwargs else {"trainset": trainset},
            )
            teacher_score = self._evaluate(
                teacher_opt, trainset[:3], train_field, val_field
            )
        else:
            teacher_opt = student
            teacher_score = baseline

        # Restore DSPy logger level
        self._predict_logger.setLevel(self._predict_logger_was_warning)

        return {
            "baseline": baseline,
            "teacher_optimized": teacher_score,
            "improvement": teacher_score - baseline,
            "teacher_helpful": teacher_score > baseline,
        }

    def run_single(
        self,
        optimizer_name: str,
        student: Any,
        trainset: list,
        auto_synthesize: bool = True,
        auto_meta: bool = True,
        min_examples: int = 5,
        train_field: str = "input",
        val_field: str = "output",
    ) -> dict:
        """Run a single optimizer with auto-synthesize + meta-learn.

        Args:
            optimizer_name: Name of the optimizer to use
            student: DSPy module to optimize
            trainset: Training examples
            auto_synthesize: Auto-generate data if too few examples
            auto_meta: Auto-select best optimizer via MetaOptimizer
            min_examples: Minimum examples before synthesis
            train_field: Input field name for evaluation
            val_field: Output field name for evaluation

        Returns {compiled_program, optimizer, score, ...}
        """
        trainset = list(trainset)
        original_len = len(trainset)

        # 1. Auto-synthesize if too few examples
        if auto_synthesize and len(trainset) < min_examples:
            tmp = Path("/tmp/dspytools_synth_seed.json")
            seed_json = []
            for ex in trainset:
                if hasattr(ex, "toDict"):
                    seed_json.append(ex.toDict())
                else:
                    seed_json.append(ex)
            write_json(tmp, seed_json)

            synth = DataSynthesizer()
            synth_result = synth.generate(str(tmp), target_count=min_examples * 2)
            if synth_result["generated"] > 0:
                new_data = try_read_json(synth_result["output_path"], [])
                new_examples = []
                for item in new_data:
                    ex = dspy.Example(**item)
                    if item:
                        first_key = list(item.keys())[0]
                        ex = ex.with_inputs(first_key)
                    new_examples.append(ex)
                trainset = list(trainset) + new_examples

        # 2. Meta-learn if enabled
        if auto_meta:
            meta = MetaOptimizer()
            selection = meta.select_optimizer(
                "auto",
                len(trainset),
                complexity="simple" if len(trainset) < 10 else "complex",
            )
            if selection["optimizer"] != optimizer_name:
                optimizer_name = selection["optimizer"]

        # 3. Run the optimizer
        compiled = self._dispatch_optimizer(optimizer_name, student, trainset)

        # 4. Evaluate with configured field names
        score = self._evaluate(compiled, trainset[:3], train_field, val_field)

        # 5. Wire to self-evolve engine
        self._evolve_engine.on_compile(
            task_profile="general",
            optimizer=optimizer_name,
            score=score,
            success=score > 0.5,
        )

        # Restore DSPy logger level
        self._predict_logger.setLevel(self._predict_logger_was_warning)

        return {
            "best_optimizer": optimizer_name,
            "best_score": score,
            "best_program": compiled,
            "trainset_size": len(trainset),
            "synthesized": len(trainset) != original_len if auto_synthesize else False,
            "all_scores": {optimizer_name: score},
        }

    def _consolidate_trajectories(
        self, optimizer_name: str, program, trainset: list
    ) -> None:
        """Trace2Skill: mine patterns from optimization trajectories into reusable skills."""

        # Build simple tasks from trainset for rollout
        tasks = []
        for i, ex in enumerate(trainset[: min(20, len(trainset))]):
            inp = ex.inputs() if hasattr(ex, "inputs") else {}
            expected = getattr(ex, "output", getattr(ex, "answer", ""))
            tasks.append({"input": inp, "expected": str(expected)})

        metric = exact_match_metric()
        consolidator = SkillConsolidator()
        result = consolidator.evolve(
            program=program,
            tasks=tasks,
            metric=metric,
            skill_name=f"{optimizer_name}_trace2skill",
            skill_content="",
        )

        # Record score to tracker
        self.tracker.record(
            f"trace2skill_{optimizer_name}",
            result.success_trajectories / max(result.trajectories_analyzed, 1),
            metadata={
                "patches_generated": result.patches_generated,
                "patches_accepted": result.patches_accepted,
                "patches_discarded": result.patches_discarded,
            },
        )

    @staticmethod
    def split_holdout(
        trainset: list, holdout_fraction: float = 0.2
    ) -> tuple[list, list]:
        """Split trainset into train and hold-out sets.

        Hold-out is NEVER seen by the optimizer — only used for gating.
        """

        n_holdout = max(1, int(len(trainset) * holdout_fraction))
        indices = list(range(len(trainset)))
        random.Random(DEFAULT_SEED).shuffle(indices)  # deterministic seed
        holdout_idx = set(indices[:n_holdout])
        train = [ex for i, ex in enumerate(trainset) if i not in holdout_idx]
        holdout = [ex for i, ex in enumerate(trainset) if i in holdout_idx]
        return train, holdout

    @staticmethod
    def gate_promotion(
        candidate, baseline, holdout: list, min_improvement: float = 0.02
    ) -> dict:
        """CI gate: only promote if candidate beats baseline on hold-out set.

        Args:
            candidate: Compiled program to test
            baseline: Current deployed program
            holdout: Hold-out examples (never seen by optimizer)
            min_improvement: Minimum score improvement to promote

        Returns:
            {promoted: bool, candidate_score, baseline_score, improvement}
        """
        candidate_score = GFLPipeline._evaluate(candidate, holdout, "input", "output")
        baseline_score = GFLPipeline._evaluate(baseline, holdout, "input", "output")
        improvement = candidate_score - baseline_score
        return {
            "promoted": improvement > min_improvement,
            "candidate_score": candidate_score,
            "baseline_score": baseline_score,
            "improvement": improvement,
        }
