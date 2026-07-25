"""Continuous self-evolution engine — morphology, transfer, UCB, skill graph.

Optimization 20: SelfEvolveEngine is a process-level singleton — avoids loading
4 JSON files per MCP tool call.
Optimization 21: on_compile() uses dirty flags — batches 4 JSON writes into 1 flush.
Optimization 22: check_convergence() defers _save_scores with dirty flag.
Optimization 24: Counter imported at module level (was imported inside method).

These four modules transform the system from batch recompilation to continuous learning:
  - Morphology Tracker: learns which instruction patterns work for which tasks
  - Knowledge Transfer: shares patterns across similar tasks
  - UCB Explorer: proactively searches untried optimizer combinations
  - Skill Graph: builds dependency chains for transitive improvement

State is persisted to ~/.config/dspytools/ and survives process restarts.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
import re as _out_re
import subprocess
import sys
import time
import time as _time
import urllib.request
from collections import Counter, defaultdict, deque
from pathlib import Path as _Path

import numpy as np

from dspytools.commands.lora import _adapter_model_name, _get_base_model
from dspytools.config.settings import (
    DEFAULT_SEED,
    adapters_dir,
    distill_dir,
    evolve_scores_path,
    morphology_path,
    llama_cpp_url,
    skill_graph_path,
    ucb_explorer_path,
)
from dspytools.config.settings import (
    compiled_dir as _compiled_dir,
)
from dspytools.core._dspy import dspy
from dspytools.core._io import read_json, write_json
from dspytools.core.dspy_modules import get_task_profiler
from dspytools.core.logging_config import get_logger
from dspytools.core.metrics import exact_match_metric
from dspytools.core.sprt_mojo_bridge import sprt_evaluate

_log = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# 1. Morphology Tracker — learns task→instruction patterns
# ═══════════════════════════════════════════════════════════════════════════


class MorphologyTracker:
    """Tracks which instruction patterns work for which task types.

    Builds a morphology graph: task_profile → {pattern_type: success_rate}
    Over time, learns that "repo documentation tasks" benefit from
    "structured markdown with ## sections" patterns.
    """

    _STATE_PATH = morphology_path()

    def __init__(self):
        self.patterns: dict[str, dict[str, tuple[int, int]]] = defaultdict(dict)
        self._load()

    def _load(self) -> None:
        """Load persisted state from disk."""
        if self._STATE_PATH.exists():
            data = read_json(self._STATE_PATH)
            for profile, patterns in data.items():
                for pattern, vals in patterns.items():
                    if isinstance(vals, list) and len(vals) == 2:
                        self.patterns[profile][pattern] = (vals[0], vals[1])

    def save(self) -> None:
        """Persist state to disk."""
        data = {}
        for profile, patterns in self.patterns.items():
            data[profile] = {}
            for pattern, val in patterns.items():
                if isinstance(val, (tuple, list)):
                    data[profile][pattern] = [val[0], val[1]]
        write_json(self._STATE_PATH, data)

    def record(self, task_profile: str, pattern_type: str, success: bool) -> None:
        """Record whether a pattern type succeeded for a task profile.

        Optimization 21: Marks dirty instead of saving immediately.
        Caller batches saves via _flush_dirty().
        """
        prev = self.patterns.get(task_profile, {}).get(pattern_type)
        if isinstance(prev, (tuple, list)):
            count, successes = int(prev[0]), int(prev[1])
        else:
            count, successes = 0, 0
        count += 1
        if success:
            successes += 1
        self.patterns[task_profile][pattern_type] = (count, successes)

    def best_pattern(self, task_profile: str) -> str | None:
        """Return the best-known pattern type for a task profile."""
        if task_profile not in self.patterns:
            return None
        best = None
        best_rate = 0.0
        for pattern, val in self.patterns[task_profile].items():
            if isinstance(val, (tuple, list)):
                count, successes = int(val[0]), int(val[1])
            else:
                continue
            if count >= 3:  # Minimum evidence
                rate = successes / count
                if rate > best_rate:
                    best_rate = rate
                    best = pattern
        return best

    def profile_task(self, description: str, field_count: int, data_size: int) -> str:
        """Create a task profile string from description + metadata using DSPy module."""
        words = len(description.split())
        if data_size < 10:
            size = "sparse"
        elif data_size < 50:
            size = "moderate"
        else:
            size = "dense"

        profiler = get_task_profiler()
        result = profiler(
            description=description, field_count=field_count, data_size=data_size
        )
        domain = getattr(result, "domain", "").strip().lower()
        complexity = getattr(result, "complexity", "").strip().lower()
        if complexity in ("simple", "moderate", "complex"):
            size = complexity

        return f"{domain}_{size}_{field_count}f_{words}w"


# ═══════════════════════════════════════════════════════════════════════════
# 2. Knowledge Transfer — cross-task pattern sharing
# ═══════════════════════════════════════════════════════════════════════════


class KnowledgeTransfer:
    """Transfers successful patterns between similar task profiles.

    When a pattern works for "numpy documentation", it's likely to work
    for "pandas documentation" too. This module finds similar tasks and
    transfers successful patterns.
    """

    def __init__(self, morphology: MorphologyTracker):
        self.morphology = morphology

    def find_similar_tasks(self, task_profile: str, max_results: int = 3) -> list[str]:
        """Find task profiles similar to the given one."""
        domain = task_profile.split("_")[0] if "_" in task_profile else task_profile
        similar = []
        for profile in self.morphology.patterns:
            if profile != task_profile and domain in profile:
                similar.append(profile)
        return similar[:max_results]

    def transfer_patterns(self, target_profile: str) -> dict[str, float]:
        """Transfer best patterns from similar tasks to the target."""
        similar = self.find_similar_tasks(target_profile)
        transferred = {}
        for source in similar:
            best = self.morphology.best_pattern(source)
            if best:
                # Weight by similarity (same domain = 1.0, cross-domain = 0.5)
                weight = (
                    1.0 if source.split("_")[0] == target_profile.split("_")[0] else 0.5
                )
                transferred[best] = transferred.get(best, 0) + weight
        return transferred


# ═══════════════════════════════════════════════════════════════════════════
# 3. UCB Explorer — proactive optimizer search
# ═══════════════════════════════════════════════════════════════════════════


class UCBExplorer:
    """UCB (Upper Confidence Bound) exploration for optimizer selection.

    Instead of always using the best-known optimizer, UCB balances
    exploitation (use best) with exploration (try untried optimizers).
    """

    _STATE_PATH = ucb_explorer_path()

    def __init__(self):
        self.trials: dict[str, tuple[int, float]] = {}
        self.costs: dict[str, float] = {}
        self.all_optimizers = [
            "bootstrap_few_shot",
            "mipro",
            "gepa",
            "copro",
            "simba",
            "labeled_few_shot",
            "knn",
            "better_together",
            "grpo",
        ]
        self._load()

    def _load(self) -> None:
        if self._STATE_PATH.exists():
            data = read_json(self._STATE_PATH)
            if "trials" in data:
                trials_data = data["trials"]
                self.costs = data.get("costs", {})
            else:
                trials_data = data
            for opt, vals in trials_data.items():
                if isinstance(vals, list) and len(vals) == 2:
                    self.trials[opt] = (vals[0], vals[1])

    def save(self) -> None:
        data = {
            "trials": {opt: [count, avg] for opt, (count, avg) in self.trials.items()},
            "costs": self.costs,
        }
        write_json(self._STATE_PATH, data)

    def record(self, optimizer: str, score: float, cost: float = 0.0) -> None:
        """Record a trial. Optimization 21: Marks dirty instead of saving immediately.

        Args:
            optimizer: Optimizer name
            score: Quality score (0.0-1.0)
            cost: Estimated token cost for this trial (cost-aware UCB)
        """
        prev = self.trials.get(optimizer, (0, 0.0))
        count, avg = prev
        new_count = count + 1
        new_avg = (avg * count + score) / new_count
        self.trials[optimizer] = (new_count, new_avg)
        if cost > 0:
            prev_cost = self.costs.get(optimizer, 0.0)
            self.costs[optimizer] = (prev_cost * count + cost) / new_count

    def select(self, c: float = 2.0, cost_weight: float = 0.0) -> str:
        """UCB selection: pick optimizer with highest upper confidence bound.

        c=2.0 encourages exploration. Decrease for more exploitation.
        cost_weight > 0 penalizes expensive optimizers (cost-aware UCB).
        """
        total = sum(cnt for cnt, _ in self.trials.values()) + 1
        best_opt = self.all_optimizers[0]
        best_score = -float("inf")

        for opt in self.all_optimizers:
            count, avg = self.trials.get(opt, (0, 0.0))
            if count == 0:
                # Never tried — prioritize exploration
                ucb = float("inf")
            else:
                ucb = avg + c * math.sqrt(math.log(total) / count)
                # Cost-aware: penalize expensive optimizers
                if cost_weight > 0:
                    avg_cost = self.costs.get(opt, 0.0)
                    ucb -= cost_weight * avg_cost

            if ucb > best_score:
                best_score = ucb
                best_opt = opt

        return best_opt

    @property
    def exploitation_score(self) -> float:
        """Ratio of exploitation vs exploration. 1.0 = pure exploitation."""
        if not self.trials:
            return 0.0
        tried = len([o for o in self.all_optimizers if o in self.trials])
        return tried / len(self.all_optimizers)


# ═══════════════════════════════════════════════════════════════════════════
# 4. Skill Graph — transitive improvement chains
# ═══════════════════════════════════════════════════════════════════════════


class SkillGraph:
    """Directed graph of skill dependencies for transitive improvement.

    SSOT: delegates to FalkorDBSkillGraph when FalkorDB is available.
    Falls back to JSON file only when FalkorDB is unreachable.
    """

    _STATE_PATH = skill_graph_path()

    def __init__(self):
        self.edges: dict[str, set[str]] = defaultdict(set)
        self._falkordb = None
        self._falkordb_unreachable = False
        self._load()

    def _get_falkordb(self):
        """Lazy-init FalkorDB backend (SSOT). Returns None if FalkorDB unavailable."""
        if self._falkordb is None and not self._falkordb_unreachable:
            from dspytools.graph.skill_graph import FalkorDBSkillGraph

            self._falkordb = FalkorDBSkillGraph()
        return self._falkordb

    def _load(self) -> None:
        """Load JSON fallback state from disk."""
        if self._STATE_PATH.exists():
            data = read_json(self._STATE_PATH)
            for skill, deps in data.items():
                self.edges[skill] = set(deps)

    def save(self) -> None:
        """Persist JSON fallback state to disk."""
        data = {skill: sorted(deps) for skill, deps in self.edges.items()}
        write_json(self._STATE_PATH, data)

    def add_dependency(self, skill: str, depends_on: str) -> None:
        """Record that 'skill' depends on 'depends_on'.

        Dual-write: FalkorDB (SSOT) + JSON fallback. If FalkorDB fails,
        JSON fallback is still updated for graceful degradation.
        """
        # SSOT: FalkorDB first (graceful degradation on failure)
        graph = self._get_falkordb()
        if graph:
            try:
                graph.add_dependency(skill, depends_on)
            except (ConnectionError, OSError, RuntimeError) as e:
                _log.warning(
                    "FalkorDB add_dependency failed, using JSON fallback: %s", e
                )

        # Fallback: JSON mirror (always updated)
        self.edges[depends_on].add(skill)

    def get_dependents(self, skill: str) -> list[str]:
        """Get all skills that depend on this skill."""
        # SSOT: FalkorDB first
        graph = self._get_falkordb()
        if graph:
            result = graph.get_dependents(skill)
            if result is not None:
                return result

        # Fallback: JSON
        return list(self.edges.get(skill, set()))

    def transitive_dependents(self, skill: str) -> list[str]:
        """Get ALL transitive dependents (direct + indirect)."""
        # SSOT: FalkorDB first
        graph = self._get_falkordb()
        if graph:
            result = graph.transitive_dependents(skill)
            if result is not None:
                return result

        # Fallback: JSON BFS
        visited = set()
        queue = deque([skill])
        while queue:
            current = queue.popleft()
            for dep in self.edges.get(current, set()):
                if dep not in visited:
                    visited.add(dep)
                    queue.append(dep)
        return list(visited)

    def on_improvement(self, skill: str) -> list[str]:
        """Called when a skill improves. Returns skills that need re-evaluation."""
        return self.transitive_dependents(skill)


# ═══════════════════════════════════════════════════════════════════════════
# 5. Continuous Self-Evolve Engine — wires all four together
# ═══════════════════════════════════════════════════════════════════════════


class SelfEvolveEngine:
    """Continuous self-evolution engine combining all four components.

    Optimization 20: Process-level singleton — avoids 4 JSON loads per MCP call.
    Optimization 21: on_compile() uses dirty flags — batches saves into 1 flush.
    Optimization 22: check_convergence() defers _save_scores with dirty flag.

    Usage:
        engine = get_engine()  # use singleton accessor
        engine.on_compile("documentation_moderate_3f_80w", "gepa", 0.85, success=True)
        best = engine.suggest_optimizer("documentation_moderate_3f_80w")
    """

    _SCORES_PATH = evolve_scores_path()

    def __init__(self):
        self.morphology = MorphologyTracker()
        self.transfer = KnowledgeTransfer(self.morphology)
        self.ucb = UCBExplorer()
        self.graph = SkillGraph()
        self._score_history: list[float] = []
        self._prediction_cache: list[str] = []
        self._scores_dirty = False
        self._morphology_dirty = False
        self._ucb_dirty = False
        self._graph_dirty = False
        self._load_scores()

    def _load_scores(self) -> None:
        """Load persisted score history from disk."""
        if self._SCORES_PATH.exists():
            data = read_json(self._SCORES_PATH)
            self._score_history = data.get("scores", [])[-100:]
            self._prediction_cache = data.get("predictions", [])[-50:]

    def _save_scores(self) -> None:
        """Persist score history to disk."""
        data = {
            "scores": self._score_history[-100:],
            "predictions": self._prediction_cache[-50:],
        }
        write_json(self._SCORES_PATH, data)
        self._scores_dirty = False

    def _flush_dirty(self) -> None:
        """Optimization 21: Batch-save only dirty components in one pass.

        Called after on_compile() finishes updating all trackers.
        Saves morphology, UCB, skill graph, and scores only if dirty.
        """
        if self._morphology_dirty:
            self.morphology.save()
            self._morphology_dirty = False
        if self._ucb_dirty:
            self.ucb.save()
            self._ucb_dirty = False
        if self._graph_dirty:
            self.graph.save()
            self._graph_dirty = False
        if self._scores_dirty:
            self._save_scores()
            self._scores_dirty = False

    def _clear_state(self) -> None:
        """Clear all in-memory and persisted state. For testing only."""
        # Remove persisted state files FIRST so fresh constructors load empty state
        for f in [
            self._SCORES_PATH,
            MorphologyTracker._STATE_PATH,
            UCBExplorer._STATE_PATH,
            SkillGraph._STATE_PATH,
        ]:
            f.unlink(missing_ok=True)
        self.morphology = MorphologyTracker()
        self.transfer = KnowledgeTransfer(self.morphology)
        self.ucb = UCBExplorer()
        self.graph = SkillGraph()
        self._score_history = []
        self._prediction_cache = []

    # ═══════════════════════════════════════════════════════════════════════
    # Convergence Guardrails — Goodhart/Repetition/Degradation Detection
    # ═══════════════════════════════════════════════════════════════════════

    @staticmethod
    def detect_metric_cheating(
        outputs: list[str],
        repetition_threshold: float = 0.85,
        min_samples: int = 3,
    ) -> dict:
        """Detect if candidate programs are producing repetitive outputs (Goodhart).

        When >repetition_threshold of outputs are identical or near-identical
        (i.e. the most common output makes up >85% of all outputs), the
        optimizer may be gaming the metric by collapsing to a single answer.

        Uses max repetition frequency (mode ratio) instead of unique ratio
        to correctly catch the "all but one identical" case.
        Considers at least ``min_samples`` needed before flagging.
        """
        if len(outputs) < min_samples:
            return {
                "cheating": False,
                "max_repetition_ratio": 0.0,
                "n_samples": len(outputs),
            }

        counts = Counter(outputs)
        max_repetition_ratio = max(counts.values()) / len(outputs)
        return {
            "cheating": max_repetition_ratio > repetition_threshold,
            "max_repetition_ratio": max_repetition_ratio,
            "n_samples": len(outputs),
            "trigger": (
                f"Most common output ({max(counts.values())}/{len(outputs)} = "
                f"{max_repetition_ratio:.0%}) exceeds {repetition_threshold:.0%} "
                "repetition threshold"
                if max_repetition_ratio > repetition_threshold
                else None
            ),
        }

    @staticmethod
    def detect_output_degradation(
        scores: list[float], window: int = 5, variance_threshold: float = 0.01
    ) -> dict:
        """Detect suspicious optimization behavior in recent score history.

        Checks:
        - Stagnation: scores are all identical (no variance)
        - Oscillation: alternating high/low (common in reward-hacking)
        - Plateau: no improvement over window despite many iterations
        """
        if len(scores) < window:
            return {"degraded": False, "n_samples": len(scores)}

        recent = scores[-window:]
        variance = max(recent) - min(recent)
        is_flat = variance < variance_threshold

        # Oscillation: check for alternating pattern with significant amplitude
        # AND no overall improvement (true oscillation = cycle without progress)
        diffs = [recent[i + 1] - recent[i] for i in range(len(recent) - 1)]
        sign_changes = sum(
            1 for i in range(1, len(diffs)) if diffs[i] * diffs[i - 1] < 0
        )
        amplitude = max(recent) - min(recent)
        net_change = recent[-1] - recent[0]
        oscillating = (
            sign_changes >= len(diffs) - 1
            and amplitude > 0.2
            and net_change <= 0  # no net improvement = true oscillation
            and len(diffs) >= 3
        )

        triggers = []
        if is_flat:
            triggers.append(f"Score stagnation: variance={variance:.4f}")
        if oscillating:
            triggers.append(
                f"Score oscillation: {sign_changes} sign changes in {len(diffs)} steps"
            )

        return {
            "degraded": bool(triggers),
            "variance": variance,
            "oscillating": oscillating,
            "mean": sum(recent) / len(recent),
            "triggers": triggers,
        }

    def check_convergence(self, predictions: list[str]) -> dict:
        """Unified convergence check: repetition + degradation.

        Optimization 22: Defers _save_scores — only persists when called
        from on_compile() via _flush_dirty().
        """
        outputs = self._prediction_cache + predictions
        self._prediction_cache = outputs[-50:]  # keep last 50

        cheating = self.detect_metric_cheating(outputs)
        degradation = self.detect_output_degradation(self._score_history)

        # degradation may not have 'variance' when n_samples < window
        score_variance = degradation.get("variance", 0.0)

        self._scores_dirty = True  # Defer save to _flush_dirty()

        return {
            "repetition_warning": cheating.get("trigger"),
            "degradation_warning": degradation.get("triggers", []),
            "safe": not cheating["cheating"] and not degradation["degraded"],
            "max_repetition_ratio": cheating["max_repetition_ratio"],
            "score_variance": score_variance,
        }

    def record_score(self, score: float):
        """Record a quality score for convergence tracking.

        Optimization 22: Marks dirty instead of saving immediately.
        """
        self._score_history.append(score)
        if len(self._score_history) > 100:
            self._score_history = self._score_history[-100:]
        self._scores_dirty = True  # Defer save to _flush_dirty()

    def on_compile(
        self, task_profile: str, optimizer: str, score: float, success: bool = True
    ) -> dict:
        """Called after every compilation. Updates all trackers.

        Optimization 21: All tracker updates are in-memory only.
        _flush_dirty() persists only dirty components once at the end.

        SSOT: Stores optimization lessons in MemoryManager (FalkorDB-native)
        so future compiles can retrieve past insights.

        Cascade: If the task_profile matches a known skill, trigger
        on_skill_improvement to queue downstream skills for re-optimization.
        """
        # Record score for convergence tracking
        self.record_score(score)

        # Record morphology
        self.morphology.record(task_profile, optimizer, success)
        self._morphology_dirty = True

        # Record UCB
        self.ucb.record(optimizer, score)
        self._ucb_dirty = True

        # Transfer knowledge
        transferred = self.transfer.transfer_patterns(task_profile)

        # Cascade: if task_profile matches a known skill, propagate improvement
        # This ensures downstream skills are ACTUALLY queued for re-optimization
        # via the DriftMonitor's pending recompile queue. `dspytools self auto-fix`
        # processes these queue entries.
        downstream = self.graph.transitive_dependents(task_profile)
        if downstream:
            from dspytools.core.drift_monitor import get_drift_monitor

            dm = get_drift_monitor()
            for dep in downstream:
                dm.request_recompile(f"cascade:{dep}")
            _log.info(
                "Skill '%s' improved — %d downstream skills queued via drift monitor: %s",
                task_profile,
                len(downstream),
                ", ".join(downstream),
            )

        # Optimization 21: Batch-save only dirty components
        self._flush_dirty()

        # Record in FalkorDB graph (SSOT: via SkillGraph's FalkorDB connection)
        falkordb = self.graph._get_falkordb()
        if falkordb:
            falkordb.record_program(
                run_id=f"evolve_{task_profile}",
                optimizer=optimizer,
                score=score,
            )

        # Store optimization lesson in MemoryManager (SSOT: FalkorDB-native memory)
        from dspytools.memory.manager import get_memory_manager

        memory = get_memory_manager()
        memory.add(
            content=f"Optimizer '{optimizer}' scored {score:.2f} on task profile '{task_profile}' (success={success})",
            user_id="self_evolve_engine",
            metadata={
                "task_profile": task_profile,
                "optimizer": optimizer,
                "score": score,
                "success": success,
            },
        )

        return {
            "morphology": self.morphology.best_pattern(task_profile),
            "transferred": transferred,
            "ucb_next": self.ucb.select(),
            "exploitation": self.ucb.exploitation_score,
        }

    def suggest_optimizer(self, task_profile: str) -> str:
        """Suggest the best optimizer for a task profile.

        SSOT: Checks memory (MemoryManager) for past lessons first,
        then morphology, transfer, and finally UCB exploration.
        """
        # Check memory (SSOT: FalkorDB-native) for past lessons
        from dspytools.memory.manager import get_memory_manager

        memory = get_memory_manager()
        results = memory.search(
            query=f"best optimizer for {task_profile}",
            user_id="self_evolve_engine",
            limit=3,
        )
        for result in results:
            content = result.get("content", "")
            if "Optimizer '" in content and "scored" in content:
                # Extract optimizer name from stored memory
                m = re.search(r"Optimizer '(\w+)' scored ([\d.]+)", content)
                if m and float(m.group(2)) > 0.7:
                    return m.group(1)

        # Check morphology
        best = self.morphology.best_pattern(task_profile)
        if best:
            return best

        # Check transfer
        transferred = self.transfer.transfer_patterns(task_profile)
        if transferred:
            return max(transferred, key=transferred.get)  # type: ignore[arg-type]

        # Fall back to UCB
        return self.ucb.select()

    def on_skill_improvement(self, skill: str) -> list[str]:
        """Called when a skill improves. Returns skills needing re-evaluation."""
        return self.graph.on_improvement(skill)

    def add_skill_dependency(self, skill: str, depends_on: str) -> None:
        self.graph.add_dependency(skill, depends_on)
        self._graph_dirty = True

    def trigger_trace2skill(
        self, program, tasks: list[dict], metric, skill_name: str = "evolved"
    ) -> dict | None:
        """Trigger Trace2Skill consolidation after a successful compile.

        Called when a compiled program achieves a strong score — evolves
        the agent skill from execution trajectories.
        """

        from dspytools.gfl.consolidation import (
            SkillConsolidator,  # lazy: breaks evolve→gfl cycle
        )

        consolidator = SkillConsolidator()
        result = consolidator.evolve(
            program=program,
            tasks=tasks,
            metric=metric,
            skill_name=skill_name,
            mode="deepening",
        )
        # Record on skill graph
        self.graph.add_dependency(skill_name, "GFLPipeline")
        self._graph_dirty = True
        self._flush_dirty()
        return {
            "skill_name": skill_name,
            "patches_accepted": result.patches_accepted,
            "patches_discarded": result.patches_discarded,
            "trajectories_analyzed": result.trajectories_analyzed,
            "elapsed_seconds": result.elapsed_seconds,
        }

    # ═══════════════════════════════════════════════════════════════════════
    # 6. Gödel Agent — Validate-Before-Deploy (arXiv 2410.04444)
    # ═══════════════════════════════════════════════════════════════════════

    def validate_and_deploy(
        self,
        candidate_program,
        program_id: str,
        holdout_set: list,
        alpha: float = 0.05,
        beta: float = 0.2,
        max_evaluations: int = 50,
    ) -> dict:
        """SPRT-based validation: accept/reject early on clear wins/losses.

        Sequential Probability Ratio Test (Wald, 1945):
        - H₀: candidate is NOT better than baseline (p ≤ p₀)
        - H₁: candidate IS better than baseline (p ≥ p₁)
        - Accept H₀ (reject candidate) when LR ≤ β/(1-α)
        - Accept H₁ (deploy candidate) when LR ≥ (1-β)/α
        - Continue testing otherwise

        This terminates early on clear wins/losses, saving API tokens.

        Args:
            candidate_program: Compiled program to test
            program_id: Registry program ID
            holdout_set: Holdout examples
            alpha: Type I error (false positive — deploy bad candidate)
            beta: Type II error (false negative — miss good candidate)
            max_evaluations: Maximum evaluations before forced decision

        Returns:
            {accepted, candidate_score, p_value, n_evaluated, early_stop, reason}
        """

        # Shuffle holdout for sequential evaluation

        indices = list(range(len(holdout_set)))
        random.Random(DEFAULT_SEED).shuffle(indices)

        # Collect outcomes as float32 array (1.0 = success, 0.0 = failure)
        # for Mojo-accelerated SPRT evaluation

        outcomes = np.empty(max_evaluations, dtype=np.float32)
        n_evaluated = 0

        for idx in indices[:max_evaluations]:
            example = holdout_set[idx]

            # Evaluate one example — crashes count as failures (SPRT protocol)
            kwargs = (
                example.inputs()
                if hasattr(example, "inputs")
                else {"input": getattr(example, "input", "")}
            )
            try:
                pred = candidate_program(**kwargs)
                expected = getattr(example, "output", "")
                got = getattr(pred, "output", getattr(pred, "answer", str(pred)))
                match = got == expected
            except (RuntimeError, OSError, ValueError, TypeError) as e:
                _log.warning("sprt_eval_failed", example=n_evaluated, error=str(e))
                match = False

            outcomes[n_evaluated] = 1.0 if match else 0.0
            n_evaluated += 1

        # Delegate the SPRT math to Mojo bridge (or pure Python fallback)

        result = sprt_evaluate(
            outcomes[:n_evaluated],
            p0=0.50,
            p1=0.65,
            alpha=alpha,
            beta=beta,
        )

        # Translate bridge result to the existing return schema
        return {
            "accepted": result["accepted"],
            "candidate_score": result["candidate_score"],
            "p_value": None,  # SPRT doesn't produce p-values
            "n_evaluated": result["n_evaluated"],
            "early_stop": result["early_stop"],
            "reason": result["reason"],
            "statistical_method": result["statistical_method"],
        }

    def self_validate(
        self,
        program_id: str,
        holdout: list,
        alpha: float = 0.05,
        beta: float = 0.2,
    ) -> dict:
        """Validate self-evolved programs using SPRT before accepting them."""
        from dspytools.core.registry import get_run

        get_run(program_id)

        candidate = self.suggest_optimizer(program_id)
        if candidate is None:
            return {
                "accepted": False,
                "reason": "no candidate available",
                "statistical_method": "SPRT",
            }

        return self.validate_and_deploy(
            candidate, program_id, holdout, alpha=alpha, beta=beta
        )

    # ═══════════════════════════════════════════════════════════════════════
    # 7. Meta Agent Search — Archive-Based Discovery (arXiv 2408.08435)
    # ═══════════════════════════════════════════════════════════════════════

    def archive_search(self, task_description: str, top_k: int = 3) -> list[dict]:
        """Meta Agent Search pattern: find similar past compilations in the archive.

        Uses the compiled program registry as the 'archive of discovered agents'.
        Returns similar past programs that can be used as initialization.
        """
        from dspytools.core.registry import list_compiled_runs

        all_runs = list_compiled_runs()
        if not all_runs:
            return []

        # Simple keyword-based relevance scoring
        keywords = set(task_description.lower().split())
        scored = []
        for run in all_runs:
            metadata = run.get("metadata", {})
            run_text = f"{run.get('id', '')} {run.get('optimizer', '')} {metadata.get('module', '')}".lower()
            score = sum(1 for kw in keywords if kw in run_text)
            if score > 0:
                scored.append((score, run))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [run for _, run in scored[:top_k]]

    # ═══════════════════════════════════════════════════════════════════════
    # 8. LSE-GEPA Integration — Tree-Guided Context Evolution
    # ═══════════════════════════════════════════════════════════════════════

    def evolve_context_lse(
        self,
        current_context: str,
        performance_history: list[dict],
        holdout: list[dict] | None = None,
        max_depth: int = 3,
    ) -> dict:
        """Use LSE tree-guided evolution to improve the instruction context.

        Wires LSESelfEvolveModule (compilable f_ψ policy) into the self-evolve
        engine. Builds an LSE exploration tree, evolves contexts using the DSPy
        module, and selects the best via UCB.

        Paper: arXiv 2603.18620 — Learning to Self-Evolve
        """

        from dspytools.gfl.paper_optimizers import (
            LSETreeExplorer,  # lazy: breaks evolve→gfl cycle
        )

        lse = LSETreeExplorer(max_depth=max_depth)
        if holdout:
            lse.set_holdout(holdout)

        root = lse.new_root()
        lse.tree[root]["context"] = current_context
        best = current_context
        best_score = 0.0

        for depth in range(max_depth):
            if performance_history:
                problems = [h.get("input", {}) for h in performance_history]
                outputs = [str(h.get("output", "")) for h in performance_history]
                ground_truth = [str(h.get("expected", "")) for h in performance_history]
                scores = [float(h.get("score", 0.0)) for h in performance_history]
                summary = lse.build_performance_summary(
                    problems, outputs, ground_truth, scores
                )
            else:
                summary = "No performance history available."

            new_context, improvement = lse.evolve_context(best, summary)
            if holdout:
                score = lse.evaluate_holdout(new_context, None)
            else:
                score = improvement

            node = lse.expand(
                root, f"lse_depth_{depth}", score or 0.5, context=new_context
            )
            lse.update(node, score or 0.5)

            if (score or 0) >= best_score:
                best_score = score or 0.0
                best = new_context

        return {
            "best_context": best,
            "tree_depth": max_depth,
            "nodes_explored": len(lse.tree),
            "improvement_estimate": best_score,
            "tree": lse.to_dict(),
        }

    # ═══════════════════════════════════════════════════════════════════════
    # 9. Trace2Skill Integration — Pattern Mining from Trajectories
    # ═══════════════════════════════════════════════════════════════════════

    def consolidate_skills(
        self,
        program,
        tasks: list[dict],
        skill_name: str = "self_evolve_skill",
        skill_content: str = "",
        mode: str = "creation",
        merge_width: int = 2,
    ) -> dict:
        """Use Trace2Skill to mine reusable patterns from execution trajectories.

        Wires the full 3-stage pipeline (Rollout → Analyze → Consolidate)
        with compilable DSPy modules.

        Paper: arXiv 2603.25158 — Trace2Skill
        """
        from dspytools.gfl.consolidation import (
            SkillConsolidator,  # lazy: breaks evolve→gfl cycle
        )

        consolidator = SkillConsolidator(merge_width=merge_width)
        metric = exact_match_metric()
        result = consolidator.evolve(
            program=program,
            tasks=tasks,
            metric=metric,
            skill_name=skill_name,
            skill_content=skill_content,
            mode=mode,
        )

        self.graph.add_dependency(skill_name, "compile")
        self._graph_dirty = True

        return {
            "skill_name": result.skill_name,
            "evolved_skill": result.evolved_skill,
            "patches_generated": result.patches_generated,
            "patches_accepted": result.patches_accepted,
            "patches_discarded": result.patches_discarded,
            "trajectories_analyzed": result.trajectories_analyzed,
            "success_trajectories": result.success_trajectories,
            "error_trajectories": result.error_trajectories,
            "guardrail_failures": result.guardrail_failures,
            "quality_dropped": result.quality_dropped,
            "elapsed_seconds": result.elapsed_seconds,
            "mode": result.mode,
            "audit_trail": result.audit_trail,
        }

    # ═══════════════════════════════════════════════════════════════════════
    # 10. Closed Self-Evolve Cycle — Compile → Evolve → Distill → LoRA
    # ═══════════════════════════════════════════════════════════════════════

    def auto_evolve_cycle(
        self,
        program,
        trainset: list,
        holdout: list | None = None,
        task_profile: str = "general_moderate_3f_50w",
        skill_name: str = "self_evolve_skill",
        max_lse_depth: int = 3,
        distill_to_lora: bool = False,
    ) -> dict:
        """Run the full closed self-evolve cycle.

        Pipeline: Suggest → LSE-evolve → Compile → Trace2Skill → LoRA distill.
        """

        results: dict = {"cycle_timestamp": _time.time(), "task_profile": task_profile}

        optimizer = self.suggest_optimizer(task_profile)
        results["suggested_optimizer"] = optimizer

        if holdout:
            context_result = self.evolve_context_lse(
                current_context="",
                performance_history=holdout,
                holdout=holdout,
                max_depth=max_lse_depth,
            )
            results["lse"] = {
                "best_context": context_result["best_context"][:500],
                "tree_depth": context_result["tree_depth"],
                "nodes_explored": context_result["nodes_explored"],
            }
        else:
            results["lse"] = {"status": "skipped", "reason": "No holdout set"}

        results["compilation"] = {
            "optimizer": optimizer,
            "needs_compile": True,
        }

        tasks = []
        for ex in trainset[: min(20, len(trainset))]:
            inp = ex.inputs() if hasattr(ex, "inputs") else {}
            expected = getattr(ex, "output", getattr(ex, "answer", ""))
            tasks.append({"input": inp, "expected": str(expected)})

        consolidation = self.consolidate_skills(
            program=program,
            tasks=tasks,
            skill_name=skill_name,
            mode="creation",
        )
        results["trace2skill"] = {
            "skill_name": consolidation["skill_name"],
            "patches_accepted": consolidation["patches_accepted"],
            "trajectories_analyzed": consolidation["trajectories_analyzed"],
        }

        if distill_to_lora:
            results["lora"] = self._check_lora_readiness(skill_name)
        else:
            results["lora"] = {"status": "skipped"}

        results["ucb_next_suggestion"] = self.ucb.select()
        results["exploitation_score"] = self.ucb.exploitation_score
        self._flush_dirty()
        return results

    def _check_lora_readiness(self, skill_name: str) -> dict:
        """Check if enough skills accumulated for LoRA distillation via llama-cpp-server.

        When enough skills accumulate, the auto_evolve_cycle extracts training
        data from the best compiled program and trains a LoRA adapter.
        The adapter is then loaded into llama-cpp-server.
        """
        from dspytools.skills.manager import SkillManager

        mgr = SkillManager()
        all_skills = mgr.list_skills()
        skill_count = len(all_skills)

        if skill_count < 3:
            return {
                "status": "deferred",
                "reason": f"Only {skill_count}/3 skills accumulated",
                "skills_available": [s.name for s in all_skills],
            }

        return {
            "status": "ready",
            "skills_count": skill_count,
            "target_skill": skill_name,
            "next_steps": [
                f"dspytools lora extract {skill_name} --min-score 0.5",
                f"dspytools lora train {skill_name} --data <jsonl-path> --rank 64",
                f"dspytools lora load {skill_name}",
            ],
        }

    def distill_to_lora(
        self,
        run_id: str,
        adapter_name: str = "distilled",
        rank: int = 64,
        min_score: float = 0.5,
        local: bool = False,
        colab: bool = False,
        devset: str | None = None,
    ) -> dict:
        """Full teacher→LoRA distillation pipeline: extract → train → load.

        Chains:
          1. Extract best outputs from compiled program using exact_match_metric
          2. Train LoRA adapter via Unsloth (local or Colab)
          3. Load into llama-cpp-server with adapter

        Returns:
            dict with extraction_stats, training_status, llama_cpp_model_name, llama_cpp_status
        """
        results: dict = {
            "started_at": time.time(),
            "run_id": run_id,
            "adapter_name": adapter_name,
        }

        # ── Step 1: Setup LM ──
        from dspytools.core.setup import LMRegistry, setup_dspy

        setup_dspy()
        lm = LMRegistry.get_or_default()
        dspy.configure(lm=lm)

        # ── Step 2: Load compiled program ──
        from dspytools.core.hotswap import (
            HotSwapManager,  # lazy: breaks circular import
        )

        mgr = HotSwapManager()
        mgr.load_all()
        programs = mgr.list()
        matching = [p for p in programs if run_id in p["id"]]
        if not matching:
            return {"error": f"Run '{run_id}' not found in compiled programs"}
        run_id = matching[0]["id"]
        mgr.swap(run_id)
        results["compiled_program"] = run_id

        # ── Step 3: Extract training data ──
        from dspytools.core.loaders import load_trainset

        testset = (
            load_trainset(devset)
            if devset
            else load_trainset("data/commitmessagegen_trainset.json")
        )

        # Detect the program's actual output field from program.json
        out_field = "output"
        prog_json_path = _Path(_compiled_dir()) / run_id / "program.json"
        if prog_json_path.exists():
            _pj = read_json(prog_json_path)
            _sig = (_pj.get("predictor") or _pj).get("signature") or {}
            _instr = _sig.get("instructions", "")
            _om = _out_re.search(r"produce the fields `([^`]+)`", _instr)
            if _om:
                out_field = _om.group(1).strip()

        metric_fn = exact_match_metric(val_field=out_field)
        training_data = []

        for ex in testset:
            inputs = ex.inputs() if hasattr(ex, "inputs") else {}
            if not inputs:
                inputs = {
                    k: v
                    for k, v in vars(ex).items()
                    if not k.startswith("_") and not callable(v)
                }
            result = mgr.infer(**inputs)
            output_str = result.get(out_field, str(result))
            score = metric_fn(ex, type("Pred", (), {out_field: output_str})())
            if score >= min_score:
                instruction = f"Generate output for: {next(iter(inputs.values()), 'task') if inputs else 'task'}"
                training_data.append(
                    {
                        "instruction": instruction,
                        "input": json.dumps(inputs, default=str),
                        "output": output_str,
                        "framework": "dspy",
                        "score": round(score, 2),
                        "format": "extracted",
                        "source_run": run_id,
                    }
                )

        results["extraction"] = {
            "total_processed": len(testset),
            "extracted": len(training_data),
            "min_score": min_score,
        }
        if not training_data:
            return {**results, "error": f"No examples passed min_score={min_score}"}

        # Deduplicate
        seen = set()
        deduped = []
        for item in training_data:
            h = hashlib.sha256(item["output"].encode()).hexdigest()
            if h not in seen:
                seen.add(h)
                deduped.append(item)
        training_data = deduped
        results["extraction"]["deduplicated"] = len(training_data)

        # Save JSONL
        output_path = distill_dir() / f"distilled_{adapter_name}.jsonl"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            for item in training_data:
                f.write(json.dumps(item) + "\n")
        results["extraction"]["jsonl_path"] = str(output_path)
        results["extraction"]["avg_score"] = sum(
            item["score"] for item in training_data
        ) / len(training_data)

        # ── Step 4: Train LoRA adapter using Unsloth ──
        gpu_info = self._get_gpu_info()
        can_train_local = (
            (not colab) and (gpu_info.get("free_mb", 0) >= 6000) if gpu_info else False
        )

        if local or can_train_local:
            base_model_name = "Qwen/Qwen3.5-9B"
            adapters_out = adapters_dir() / adapter_name

            script = f'''"""LoRA training via Unsloth for {adapter_name} (score≥{min_score})."""
import json, torch
from datasets import Dataset
from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel, is_bfloat16_supported

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="{base_model_name}", max_seq_length=4096,
    dtype=torch.bfloat16 if is_bfloat16_supported() else torch.float16,
    load_in_4bit=False,
)
model = FastLanguageModel.get_peft_model(
    model, r={rank},
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    lora_alpha={rank}, lora_dropout=0.0, bias="none",
    use_gradient_checkpointing="unsloth", random_state=42,
)
with open("{output_path}") as f:
    raw = [json.loads(l) for l in f if l.strip()]
def fmt(e):
    i, o, inp = e.get("instruction",""), e.get("output",""), e.get("input","")
    t = f"### Instruction:\\n{{i}}\\n" + (f"### Input:\\n{{inp}}\\n" if inp else "") + f"### Response:\\n{{o}}"
    return {{"text": t}}
dataset = Dataset.from_list([fmt(e) for e in raw])
trainer = SFTTrainer(model=model, tokenizer=tokenizer, train_dataset=dataset,
    dataset_text_field="text", max_seq_length=4096,
    args=TrainingArguments(per_device_train_batch_size=2, gradient_accumulation_steps=4,
        warmup_ratio=0.1, num_train_epochs=3, learning_rate=2e-4,
        fp16=not is_bfloat16_supported(), bf16=is_bfloat16_supported(),
        logging_steps=1, optim="adamw_8bit", weight_decay=0.01,
        lr_scheduler_type="cosine", seed=42,
        output_dir="adapters/{adapter_name}", report_to="none"),
)
trainer.train()
model.save_pretrained("adapters/{adapter_name}")
tokenizer.save_pretrained("adapters/{adapter_name}")
print(f"Adapter saved to adapters/{adapter_name}/")
'''
            script_path = output_path.parent / f"train_{adapter_name}.py"
            script_path.write_text(script)
            proc = subprocess.Popen(
                [sys.executable, str(script_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            if proc.stdout:
                for line in proc.stdout:
                    pass  # consume silently
            proc.wait()
            success = proc.returncode == 0
            results["training"] = {
                "mode": "local",
                "success": success,
                "adapter_path": str(adapters_out),
            }
        else:
            # Colab staging
            colab_dir = distill_dir() / f"colab_{adapter_name}"
            colab_dir.mkdir(parents=True, exist_ok=True)
            dest = colab_dir / "training_data.jsonl"
            dest.write_bytes(output_path.read_bytes())
            results["training"] = {
                "mode": "colab",
                "staged": True,
                "colab_dir": str(colab_dir),
                "colab_command": f"# Upload {colab_dir}/ to Colab and run training script",
            }
            results["training"]["success"] = True

        # ── Step 5: Load into llama-cpp-server ──

        base = _get_base_model()
        llama_cpp_model_name = _adapter_model_name(adapter_name)
        adapter_path = adapters_dir() / adapter_name
        
        # llama-cpp uses /api/generate with adapters parameter
        payload = json.dumps({
            "model": base,
            "adapter": str(adapter_path.resolve()),
            "prompt": "{{ .Prompt }}",
            "stream": False,
        }).encode()
        req = urllib.request.Request(
            f"{llama_cpp_url()}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            resp.read()
        results["llama_cpp"] = {
            "model_name": llama_cpp_model_name,
            "status": "loaded",
            "adapter_path": str(adapter_path),
        }

        results["elapsed_seconds"] = time.time() - results["started_at"]

        # Record in FalkorDB for lineage
        self.graph.add_dependency(f"distilled_{adapter_name}", run_id)
        self._graph_dirty = True
        self._flush_dirty()

        return results

    @staticmethod
    def _get_gpu_info() -> dict:
        """Get GPU memory info from nvidia-smi."""
        r = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        parts = [p.strip() for p in r.stdout.strip().split(", ")]
        if len(parts) >= 3:
            return {
                "used_mb": int(parts[0]),
                "total_mb": int(parts[1]),
                "free_mb": int(parts[2]),
            }
        return {}


# ═══════════════════════════════════════════════════════════════════════════
# 8. Process-level singleton — avoids 4 JSON loads per MCP call
# ═══════════════════════════════════════════════════════════════════════════

_engine: SelfEvolveEngine | None = None


def get_engine() -> SelfEvolveEngine:
    """Optimization 20: Process-level singleton for SelfEvolveEngine.

    Avoids loading morphology.json, ucb_explorer.json, skill_graph.json,
    and evolve_scores.json on every MCP tool call.
    """
    global _engine
    if _engine is None:
        _engine = SelfEvolveEngine()
    return _engine
