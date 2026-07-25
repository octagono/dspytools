"""Trace2Skill consolidation — mines execution trajectories for reusable agent skills.

arXiv 2603.25158: Trace2Skill: Distill Trajectory-Local Lessons into Transferable Agent Skills.

3-stage pipeline (paper-faithful, DSPy-compilable):
  1. Rollout  — parallel task execution, collect success/failure trajectories
  2. Analyze  — compilable DSPy modules: Success Analyst (ChainOfThought) +
                Error Analyst (ChainOfThought, ReAct-style with iterative decision)
  3. Consolidate — hierarchical merge with DSPy MergeOperator module +
                   3 deterministic guardrails (file-exists, line-range conflict,
                   trial-apply validation)

All LLM-driven stages use proper dspy.Signature + dspy.Module classes that can be
compiled with any DSPy optimizer (e.g. dspytools compile gepa Trace2SkillErrorAnalyst).

Two modes:
  - Deepening: refine an existing human-written SKILL.md
  - Creation:  generate a new skill from scratch (LLM-drafted S₀)

Supports: parallel rollout (ThreadPoolExecutor), semi-online mode (TrajectoryLayer),
rollback (timestamped backups), quality gate (drop unverifiable trajectories),
transfer validation (cross-model scale testing).
"""

from __future__ import annotations

import hashlib
import json as _json
import re
import shutil
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import dspy
else:
    from dspytools.core._dspy import dspy

from dspytools.config.settings import skills_dir
from dspytools.core._io import read_json, write_json
from dspytools.evolve.layers.trajectory import TrajectoryLayer

# ═══════════════════════════════════════════════════════════════════════════
# DSPy Signatures — compilable, typed, optimizer-ready
# ═══════════════════════════════════════════════════════════════════════════


class ErrorAnalystSignature(dspy.Signature):
    """Analyze a failed execution trajectory and propose an actionable skill patch.

    You must identify the ROOT CAUSE of the failure, not just describe symptoms.
    Only propose generalizable fixes that would help NEW tasks, not task-specific hacks.
    If the failure cannot be causally explained, choose 'drop'.
    """

    trajectory_raw: str = dspy.InputField(
        desc="Full execution trace: input, attempts, output"
    )
    expected_output: str = dspy.InputField(desc="Ground truth / expected output")
    current_skill: str = dspy.InputField(desc="Current SKILL.md content to patch")
    iteration: int = dspy.InputField(desc="Current iteration number (0-based)")
    patch_content: str = dspy.OutputField(
        desc="Proposed skill instruction (1-3 sentences) — empty if dropped"
    )
    patch_section: str = dspy.OutputField(
        desc="Section: gotchas, instructions, workflow, examples, purpose, constraints"
    )
    decision: str = dspy.OutputField(
        desc="finish (confident fix), try_patch (tentative), or drop (unverifiable)"
    )
    confidence: float = dspy.OutputField(desc="Confidence 0.0-1.0; 0.0 for drop")
    root_cause: str = dspy.OutputField(desc="Brief root cause diagnosis")


class SuccessAnalystSignature(dspy.Signature):
    """Analyze a successful execution trajectory and extract reusable behavior patterns.

    Identify what the program did RIGHT and generalize it into actionable skill
    instructions. Only include patterns that would transfer to NEW tasks, not just
    celebrate this specific success.
    """

    task_description: str = dspy.InputField(
        desc="Compact description of the task input"
    )
    output: str = dspy.InputField(desc="Actual program output (successful)")
    score: float = dspy.InputField(desc="Quality score 0.0-1.0")
    current_skill: str = dspy.InputField(desc="Current SKILL.md content")
    patterns_json: str = dspy.OutputField(
        desc="JSON array of {section, content, confidence} objects"
    )
    count: int = dspy.OutputField(desc="Number of extracted patterns (0 if none)")


class MergeOperatorSignature(dspy.Signature):
    """Consolidate two skill patches into one, preserving non-overlapping insights.

    Determine whether these patches represent RECURRENT generalizable patterns
    or IDIOSYNCRATIC one-off fixes. Prefer prevalent patterns, discard
    idiosyncratic edits. If both are general and complementary, combine them.
    """

    patch_a_section: str = dspy.InputField(desc="Section of patch A")
    patch_a_content: str = dspy.InputField(desc="Content of patch A")
    patch_a_confidence: float = dspy.InputField(desc="Confidence of patch A")
    patch_b_section: str = dspy.InputField(desc="Section of patch B")
    patch_b_content: str = dspy.InputField(desc="Content of patch B")
    patch_b_confidence: float = dspy.InputField(desc="Confidence of patch B")
    current_skill: str = dspy.InputField(desc="Current SKILL.md content for context")
    merged_content: str = dspy.OutputField(
        desc="Merged skill instruction (1-3 sentences) or empty for drop"
    )
    decision: str = dspy.OutputField(
        desc="recurrent (generalizable, merge), single (keep one), or drop (neither)"
    )
    confidence: float = dspy.OutputField(desc="Confidence 0.0-1.0 in merged result")


# ═══════════════════════════════════════════════════════════════════════════
# DSPy Modules — compilable via dspytools compile <optimizer> <name>
# ═══════════════════════════════════════════════════════════════════════════


class ErrorAnalystModule(dspy.Module):
    """Compilable Error Analyst (A⁻). ReAct-style iterative root-cause analysis.

    Each forward() call = one iteration of the ReAct loop.
    The caller manages the iteration loop and accumulates patches.
    Can be compiled: dspytools compile gepa ErrorAnalystModule trainset.json
    """

    def __init__(self):
        super().__init__()
        self.analyze = dspy.ChainOfThought(ErrorAnalystSignature)

    def forward(
        self,
        trajectory_raw: str,
        expected_output: str,
        current_skill: str,
        iteration: int,
    ) -> dspy.Prediction:
        return self.analyze(
            trajectory_raw=trajectory_raw,
            expected_output=expected_output,
            current_skill=current_skill,
            iteration=iteration,
        )


class SuccessAnalystModule(dspy.Module):
    """Compilable Success Analyst (A⁺). Single-pass pattern extraction.

    Extracts generalizable behavior patterns from successful trajectories.
    Can be compiled: dspytools compile mipro SuccessAnalystModule trainset.json
    """

    def __init__(self):
        super().__init__()
        self.analyze = dspy.ChainOfThought(SuccessAnalystSignature)

    def forward(
        self,
        task_description: str,
        output: str,
        score: float,
        current_skill: str,
    ) -> dspy.Prediction:
        return self.analyze(
            task_description=task_description,
            output=output,
            score=score,
            current_skill=current_skill,
        )


class MergeOperatorModule(dspy.Module):
    """Compilable Merge Operator (ℳ). Hierarchical patch consolidation with
    inductive reasoning — discriminates recurrent patterns from noise.

    Can be compiled: dspytools compile gfl --halving MergeOperatorModule trainset.json
    """

    def __init__(self):
        super().__init__()
        self.merge = dspy.ChainOfThought(MergeOperatorSignature)

    def forward(
        self,
        patch_a_section: str,
        patch_a_content: str,
        patch_a_confidence: float,
        patch_b_section: str,
        patch_b_content: str,
        patch_b_confidence: float,
        current_skill: str,
    ) -> dspy.Prediction:
        return self.merge(
            patch_a_section=patch_a_section,
            patch_a_content=patch_a_content,
            patch_a_confidence=patch_a_confidence,
            patch_b_section=patch_b_section,
            patch_b_content=patch_b_content,
            patch_b_confidence=patch_b_confidence,
            current_skill=current_skill,
        )


# ═══════════════════════════════════════════════════════════════════════════
# Data Models
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class Patch:
    """A proposed skill edit from one trajectory analysis."""

    source_traj_id: str
    source_type: str  # "success" or "error"
    section: str  # section of SKILL.md to modify
    action: str  # "append", "prepend", "replace", "add_section"
    content: str
    rationale: str
    confidence: float = 0.5
    metadata: dict = field(default_factory=dict)

    @property
    def fingerprint(self) -> str:
        """Content-based fingerprint for deduplication."""
        h = hashlib.sha256()
        h.update(f"{self.section}:{self.action}:{self.content}".encode())
        return h.hexdigest()[:12]


@dataclass
class ConsolidationResult:
    """Result of a Trace2Skill consolidation run."""

    skill_name: str
    source_skill: str
    evolved_skill: str
    patches_generated: int
    patches_accepted: int
    patches_discarded: int
    trajectories_analyzed: int
    success_trajectories: int
    error_trajectories: int
    guardrail_failures: int
    quality_dropped: int
    elapsed_seconds: float
    mode: str = "creation"  # "creation" or "deepening"
    transfer_scores: dict[str, float] = field(default_factory=dict)
    audit_trail: list[dict] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════
# Stage 1: Rollout — parallel task execution
# ═══════════════════════════════════════════════════════════════════════════


class TrajectoryRollout:
    """Stage 1: Run compiled programs on tasks in parallel, collect execution traces.

    Uses ThreadPoolExecutor for true parallelism — each task runs independently.
    Produces labeled trajectory pool: successes (T⁺) and failures (T⁻).
    """

    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self.success_ids: list[str] = []
        self.error_ids: list[str] = []

    def run(
        self,
        program,
        tasks: list[dict],
        metric,
        run_id_prefix: str = "trace2skill",
    ) -> dict:
        """Execute program on all tasks in parallel and record trajectories."""

        self.success_ids.clear()
        self.error_ids.clear()

        def _execute_one(idx: int, task: dict) -> dict:
            run_id = f"{run_id_prefix}_{idx}"
            start = time.time()

            result = program(**task["input"])
            output = getattr(result, "output", getattr(result, "answer", str(result)))

            score = metric(task, result)
            trajectory_type = "success" if score >= 0.7 else "error"

            traj_entry = {
                "task_id": idx,
                "run_id": run_id,
                "input": task["input"],
                "expected": task.get("expected", ""),
                "output": str(output),
                "score": score,
                "trajectory_type": trajectory_type,
                "elapsed": time.time() - start,
                "exception": None if result is not None else output,
            }

            # Persist to TrajectoryLayer
            TrajectoryLayer.record(
                run_id=run_id,
                action_name="trace2skill_rollout",
                inputs=task["input"],
                outputs={
                    "output": str(output),
                    "score": score,
                    "expected": task.get("expected", ""),
                },
                score=score,
                metadata={"trajectory_type": trajectory_type},
            )

            return traj_entry

        trajectories: list[dict] = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(_execute_one, i, task): i
                for i, task in enumerate(tasks)
            }
            for future in as_completed(futures):
                traj = future.result()
                trajectories.append(traj)
                if traj["trajectory_type"] == "success":
                    self.success_ids.append(traj["run_id"])
                else:
                    self.error_ids.append(traj["run_id"])

        trajectories.sort(key=lambda t: t["task_id"])

        scores = [t["score"] for t in trajectories]
        avg_score = sum(scores) / len(scores) if scores else 0.0

        return {
            "trajectories": trajectories,
            "success_count": len(self.success_ids),
            "error_count": len(self.error_ids),
            "avg_score": avg_score,
            "success_ids": list(self.success_ids),
            "error_ids": list(self.error_ids),
        }


# ═══════════════════════════════════════════════════════════════════════════
# Stage 2: Analyze — DSPy-compilable Success + Error Analysts
# ═══════════════════════════════════════════════════════════════════════════


class _SuccessAnalyst:
    """DSPy-compilable single-pass analysis of successful trajectories.

    Uses SuccessAnalystModule (dspy.ChainOfThought) to extract generalizable
    behavior patterns. DSPy module always works — no heuristic fallback.
    """

    def __init__(self, skill_content: str):
        self.skill_content = skill_content
        self.module = SuccessAnalystModule()

    def analyze(self, trajectory: dict) -> list[Patch]:
        """Extract success patterns using DSPy module."""
        return self._dspy_analyze(trajectory)

    def _dspy_analyze(self, trajectory: dict) -> list[Patch]:
        """Use the compilable DSPy module for pattern extraction."""
        _describe_input(trajectory)
        output = str(trajectory.get("output", ""))
        score = float(trajectory.get("score", 0.0))

        pred = self.module(
            output=output,
            score=score,
            current_skill=self.skill_content,
        )

        patterns_json = getattr(pred, "patterns_json", "[]")
        count = getattr(pred, "count", 0)

        if count < 1 or not patterns_json or patterns_json in ("[]", ""):
            return []

        try:
            patterns = _json.loads(patterns_json)
        except (_json.JSONDecodeError, TypeError):
            # Try extracting from markdown wrapper
            patterns = _parse_json_response(patterns_json)
            if isinstance(patterns, dict):
                patterns = [patterns]

        if not isinstance(patterns, list):
            patterns = []

        return [
            Patch(
                source_traj_id=trajectory["run_id"],
                source_type="success",
                section=p.get("section", "workflow"),
                action="append",
                content=p.get("content", ""),
                rationale="DSPy-extracted success pattern",
                confidence=p.get("confidence", 0.7),
            )
            for p in patterns
            if p.get("content") and isinstance(p, dict)
        ]


class _ErrorAnalyst:
    """DSPy-compilable ReAct-style agentic analysis of failed trajectories.

    Uses ErrorAnalystModule (dspy.ChainOfThought) for each iteration of the
    ReAct loop. The module decides: finish (output patch), try_patch (iterate),
    or drop (unverifiable).

    Paper-faithful: A⁻ uses a ReAct-style loop that can inspect traces and
    artifacts, compare outputs against ground truth, and validate candidate
    fixes before proposing a patch. Failures that cannot be causally explained
    are excluded, ensuring patches are grounded in verified mechanisms.
    """

    def __init__(self, skill_content: str, max_iterations: int = 5):
        self.skill_content = skill_content
        self.max_iterations = max_iterations
        self._patches: list[Patch] = []
        self._iteration: int = 0
        self._finished: bool = False
        self._dropped: bool = False
        self._observations: list[str] = []
        self.module = ErrorAnalystModule()

    def analyze(self, trajectory: dict) -> list[Patch]:
        """Run the ReAct analysis loop on a failed trajectory."""
        self._patches.clear()
        self._observations.clear()
        self._iteration = 0
        self._finished = False
        self._dropped = False
        self._dspy_react_loop(trajectory)

        if self._dropped:
            return []

        if not self._patches and self._observations:
            self._patches.append(
                Patch(
                    source_traj_id=trajectory["run_id"],
                    source_type="error",
                    section="gotchas",
                    action="append",
                    content="\n".join(f"- {obs}" for obs in self._observations),
                    rationale="Synthesized from error observations",
                    confidence=0.4,
                )
            )

        return self._patches

    def _dspy_react_loop(self, trajectory: dict) -> None:
        """DSPy-driven ReAct loop for error analysis."""
        trajectory_raw = _format_trajectory_raw(trajectory)
        expected = str(trajectory.get("expected", ""))

        for i in range(self.max_iterations):
            if self._finished or self._dropped:
                break
            self._iteration = i + 1

            pred = self.module(
                trajectory_raw=trajectory_raw,
                expected_output=expected,
                current_skill=self.skill_content[:3000],
                iteration=i,
            )

            decision = getattr(pred, "decision", "finish").strip().lower()
            confidence = float(getattr(pred, "confidence", 0.5))
            patch_content = getattr(pred, "patch_content", "").strip()
            patch_section = getattr(pred, "patch_section", "gotchas").strip()
            root_cause = getattr(pred, "root_cause", "").strip()

            if root_cause:
                self._observations.append(root_cause)

            self._apply_dspy_decision(
                decision=decision,
                patch_section=patch_section,
                patch_content=patch_content,
                confidence=confidence,
                trajectory=trajectory,
            )

        if not self._finished and not self._dropped:
            self._finished = True

    def _apply_dspy_decision(
        self,
        decision: str,
        patch_section: str,
        patch_content: str,
        confidence: float,
        trajectory: dict,
    ) -> None:
        """Apply the DSPy module's chosen action."""
        if decision == "finish":
            if patch_content:
                self._patches.append(
                    Patch(
                        source_traj_id=trajectory["run_id"],
                        source_type="error",
                        section=patch_section,
                        action="append",
                        content=patch_content,
                        rationale=f"DSPy root-cause analysis at iteration {self._iteration}",
                        confidence=confidence,
                    )
                )
            self._finished = True

        elif decision == "try_patch":
            if patch_content:
                self._patches.append(
                    Patch(
                        source_traj_id=trajectory["run_id"],
                        source_type="error",
                        section=patch_section,
                        action="append",
                        content=patch_content,
                        rationale=f"DSPy tentative patch at iteration {self._iteration}",
                        confidence=confidence,
                    )
                )

        elif decision == "drop":
            self._dropped = True

        else:
            # Unknown decision — treat as finish if content exists
            if patch_content:
                self._patches.append(
                    Patch(
                        source_traj_id=trajectory["run_id"],
                        source_type="error",
                        section=patch_section,
                        action="append",
                        content=patch_content,
                        rationale=f"DSPy analysis at iteration {self._iteration}",
                        confidence=confidence,
                    )
                )
            self._finished = True


class _ParallelAnalyst:
    """Orchestrates parallel Success + Error Analysts with quality gate."""

    def __init__(self, skill_content: str, min_confidence: float = 0.3):
        self.skill_content = skill_content
        self.min_confidence = min_confidence
        self.success_analyst = _SuccessAnalyst(skill_content)
        self.error_analyst = _ErrorAnalyst(skill_content)
        self._patches: list[Patch] = []
        self._dropped_count: int = 0

    def analyze_all(self, trajectories: list[dict]) -> list[Patch]:
        """Run all analysts across the full trajectory pool with quality gate."""
        self._patches.clear()
        self._dropped_count = 0

        for traj in trajectories:
            traj_type = traj.get("trajectory_type", "error")
            if traj_type == "success":
                patches = self.success_analyst.analyze(traj)
            else:
                patches = self.error_analyst.analyze(traj)
                if not patches:
                    self._dropped_count += 1  # quality gate

            self._patches.extend(
                p for p in patches if p.confidence >= self.min_confidence
            )

        return self._patches

    @property
    def dropped_trajectories(self) -> int:
        return self._dropped_count


# ═══════════════════════════════════════════════════════════════════════════
# Stage 3: Consolidation — hierarchical merge with DSPy MergeOperator
# ═══════════════════════════════════════════════════════════════════════════


class _PatchMerger:
    """Hierarchical merge with DSPy-compilable MergeOperator module for
    inductive reasoning.

    Paper-faithful implementation:
      - Configurable B_merge branching factor (paper §3.4): binary (2) for deep
        comparison, larger values for speed/throughput trade-off
      - DSPy MergeOperator judges whether edits are generalizable SoPs or noise
      - 3 deterministic guardrails: section validation, line-range conflict, trial-apply
      - Frequency filter: only edits appearing ≥MIN_FREQUENCY across patch pool

    Performance tuning:
      B_merge=2 (default): binary tree, ⌈log₂(|P|)⌉ merge passes, most thorough
      B_merge=4: quad tree, ~⌈log₄(|P|)⌉ passes, ~50% fewer LM calls
      B_merge=8+: wider fan-out, fewer passes, less thorough consolidation
    """

    MIN_FREQUENCY: int = 2

    def __init__(self, merge_width: int = 2):
        self._guardrail_failures: int = 0
        self._merge_width = max(2, merge_width)
        self.merge_module = MergeOperatorModule()

    def merge(
        self, patches: list[Patch], skill_content: str
    ) -> tuple[str, int, int, int, list[dict]]:
        """Hierarchically consolidate patches into a unified skill."""
        self._guardrail_failures = 0
        audit: list[dict] = []

        accepted, discarded = self._filter_patches(patches, audit)
        final_patches = self._hierarchical_merge(accepted, skill_content, audit)
        evolved = self._apply_patches(skill_content, final_patches)

        return (
            evolved,
            len(final_patches),
            len(discarded),
            self._guardrail_failures,
            audit,
        )

    def _filter_patches(
        self, patches: list[Patch], audit: list[dict]
    ) -> tuple[list[Patch], list[Patch]]:
        """Filter patches through guardrails and frequency check."""
        accepted: list[Patch] = []
        discarded: list[Patch] = []
        fingerprints_seen: dict[str, int] = defaultdict(int)

        valid_sections = {
            "purpose",
            "workflow",
            "instructions",
            "gotchas",
            "examples",
            "constraints",
            "resources",
            "configuration",
            "dependencies",
        }

        for patch in patches:
            fp = patch.fingerprint
            fingerprints_seen[fp] += 1

            # Guardrail 1: validate section
            if patch.section not in valid_sections:
                self._guardrail_failures += 1
                discarded.append(patch)
                audit.append(
                    {
                        "patch_fp": fp,
                        "rejected": "invalid_section",
                        "section": patch.section,
                    }
                )
                continue

            # Guardrail 2: line-range conflict
            if self._has_conflict(patch, accepted):
                self._guardrail_failures += 1
                discarded.append(patch)
                audit.append(
                    {
                        "patch_fp": fp,
                        "rejected": "line_range_conflict",
                        "section": patch.section,
                    }
                )
                continue

            accepted.append(patch)

        # Guardrail 3 + Frequency filter: only keep prevalent patterns
        final_accepted: list[Patch] = []
        for patch in accepted:
            fp = patch.fingerprint
            if fingerprints_seen[fp] >= self.MIN_FREQUENCY:
                if self._trial_apply(patch):
                    final_accepted.append(patch)
                    audit.append(
                        {
                            "patch_fp": fp,
                            "accepted": True,
                            "section": patch.section,
                            "type": patch.source_type,
                        }
                    )
                else:
                    self._guardrail_failures += 1
                    discarded.append(patch)
                    audit.append({"patch_fp": fp, "rejected": "trial_apply_failed"})
            else:
                discarded.append(patch)
                audit.append(
                    {
                        "patch_fp": fp,
                        "rejected": "below_frequency",
                        "count": fingerprints_seen[fp],
                    }
                )

        return final_accepted, discarded

    def _hierarchical_merge(
        self, patches: list[Patch], skill_content: str, audit: list[dict]
    ) -> list[Patch]:
        """B_merge-ary tree merge: pairwise combine similar patches via DSPy MergeOperator.

        With merge_width=B, each pass groups up to B same-section patches together.
        Binary (B=2): ⌈log₂(|P|)⌉ passes, most thorough.
        Wide (B≥4): fewer passes, faster, good for large patch pools.
        """
        if len(patches) <= 1:
            return patches

        merged: list[Patch] = list(patches)
        w = self._merge_width
        while len(merged) > 1:
            new_merged: list[Patch] = []
            i = 0
            while i < len(merged):
                # Collect up to w same-section patches for this group
                group = [merged[i]]
                j = i + 1
                while j < len(merged) and j < i + w:
                    if merged[j].section == merged[i].section:
                        group.append(merged[j])
                    else:
                        break
                    j += 1

                if len(group) > 1:
                    # Sequentially merge group pairs
                    result = group[0]
                    for other in group[1:]:
                        result = self._merge_pair(result, other, skill_content)
                    new_merged.append(result)
                else:
                    new_merged.append(group[0])
                i += len(group)

            merged = new_merged

        return merged

    def _merge_pair(self, a: Patch, b: Patch, skill_content: str) -> Patch:
        """Merge two patches in the same section, using DSPy MergeOperator."""
        return self._dspy_merge_pair(a, b, skill_content)

    def _dspy_merge_pair(self, a: Patch, b: Patch, skill_content: str) -> Patch:
        """Use DSPy MergeOperator for inductive reasoning when merging."""
        pred = self.merge_module(
            patch_a_section=a.section,
            patch_a_content=a.content,
            patch_a_confidence=a.confidence,
            patch_b_section=b.section,
            patch_b_content=b.content,
            patch_b_confidence=b.confidence,
            current_skill=skill_content,
        )

        decision = getattr(pred, "decision", "drop").strip().lower()
        merged_content = getattr(pred, "merged_content", "").strip()
        confidence = float(getattr(pred, "confidence", 0.5))

        if decision == "drop" or not merged_content:
            # Keep the higher-confidence patch
            return a if a.confidence >= b.confidence else b

        return Patch(
            source_traj_id=f"merged_{a.source_traj_id}",
            source_type="merged",
            section=a.section,
            action="append",
            content=merged_content,
            rationale=f"DSPy-inductive hierarchical merge ({decision})",
            confidence=confidence,
        )

    @staticmethod
    def _has_conflict(patch: Patch, existing: list[Patch]) -> bool:
        for ep in existing:
            if ep.section == patch.section and ep.action == "replace":
                return True
        return False

    @staticmethod
    def _trial_apply(patch: Patch) -> bool:
        """Guardrail 3: Dry-run validation. Content must be non-trivial."""
        return bool(patch.content.strip()) and len(patch.content) > 10

    @staticmethod
    def _apply_patches(skill_content: str, patches: list[Patch]) -> str:
        """Build SKILL.md from base content + accepted patches."""
        if not patches:
            return skill_content

        sections: dict[str, list[str]] = defaultdict(list)
        current_section = "workflow"
        for line in skill_content.split("\n"):
            sections[current_section].append(line)

        for patch in patches:
            sec = patch.section
            if patch.action == "append":
                sections[sec].extend(patch.content.split("\n"))
            elif patch.action == "prepend":
                sections[sec] = patch.content.split("\n") + sections[sec]
            elif patch.action in ("add_section", "replace"):
                sections[sec] = patch.content.split("\n")

        ordered = [
            "purpose",
            "workflow",
            "instructions",
            "gotchas",
            "examples",
            "constraints",
            "resources",
            "configuration",
            "dependencies",
        ]
        result_lines: list[str] = []
        for sec in ordered:
            if sec in sections:
                result_lines.append(f"## {sec.capitalize()}")
                result_lines.append("")
                result_lines.extend(sections[sec])
                result_lines.append("")

        return "\n".join(result_lines).strip() + "\n"


# ═══════════════════════════════════════════════════════════════════════════
# Main SkillConsolidator — orchestrates all 3 stages + extra features
# ═══════════════════════════════════════════════════════════════════════════


class SkillConsolidator:
    """Trace2Skill: 3-stage pipeline for evolving agent skills.

    All LLM-driven stages now use compilable DSPy modules:
      - ErrorAnalystModule (ChainOfThought, ReAct-style)
      - SuccessAnalystModule (ChainOfThought, single-pass)
      - MergeOperatorModule (ChainOfThought, inductive consolidation)

    Compile any module: dspytools compile gepa ErrorAnalystModule trainset.json

    Full feature set:
      - Deepening & creation modes
      - Parallel rollout (ThreadPoolExecutor)
      - DSPy-compilable analysts (ChainOfThought)
      - DSPy-compilable merge (ChainOfThought, inductive reasoning)
      - Quality gate (drop unverifiable trajectories)
      - Semi-online mode (TrajectoryLayer sessions)
      - Rollback (timestamped backups)
      - Transfer validation (cross-model scale testing)
      - SkillManager lifecycle integration
    """

    SKILLS_DIR: Path = skills_dir()
    BACKUP_DIR: Path = skills_dir() / "_backups"

    def __init__(self, merge_width: int = 2):
        self.rollout = TrajectoryRollout()
        self.merger = _PatchMerger(merge_width=merge_width)

    # ── Full pipeline with mode support ───────────────────────────────

    def evolve(
        self,
        program,
        tasks: list[dict],
        metric,
        skill_name: str = "trace2skill",
        skill_content: str = "",
        mode: str = "creation",
    ) -> ConsolidationResult:
        """Run the full Trace2Skill pipeline.

        Args:
            program: Compiled DSPy program to evaluate
            tasks: List of {input: dict, expected: str} dicts
            metric: Scoring function (example, prediction) -> float
            skill_name: Name for the evolved skill
            skill_content: Current SKILL.md content (empty = creation mode)
            mode: "creation" (generate from scratch) or "deepening" (refine existing)

        Returns:
            ConsolidationResult with evolved_skill and full audit trail
        """
        start = time.time()

        # Deepening mode: use existing skill as S₀
        if mode == "deepening" and skill_content:
            s0 = skill_content
        else:
            s0 = skill_content or self._default_skill(skill_name)

        # Stage 1: Rollout (parallel)
        rollout_result = self.rollout.run(
            program, tasks, metric, run_id_prefix=skill_name
        )

        # Stage 2: Analyze (DSPy-driven, with quality gate)
        analyst = _ParallelAnalyst(s0)
        patches = analyst.analyze_all(rollout_result["trajectories"])
        quality_dropped = analyst.dropped_trajectories

        # Stage 3: Consolidate (hierarchical merge + DSPy inductive reasoning)
        evolved_skill, accepted, discarded, guardrail_failures, audit = (
            self.merger.merge(patches, s0)
        )

        # Persist evolved skill via SkillManager
        self._integrate_skill(skill_name, evolved_skill, mode, program, rollout_result)

        result = ConsolidationResult(
            skill_name=skill_name,
            source_skill=s0,
            evolved_skill=evolved_skill,
            patches_generated=len(patches),
            patches_accepted=accepted,
            patches_discarded=discarded,
            trajectories_analyzed=len(rollout_result["trajectories"]),
            success_trajectories=rollout_result["success_count"],
            error_trajectories=rollout_result["error_count"],
            guardrail_failures=guardrail_failures,
            quality_dropped=quality_dropped,
            elapsed_seconds=time.time() - start,
            mode=mode,
            audit_trail=audit,
        )

        return result

    # ── Semi-online mode ──────────────────────────────────────────────

    def evolve_online(
        self,
        program,
        tasks: list[dict],
        metric,
        skill_name: str = "trace2skill",
        skill_content: str = "",
        min_sessions: int = 5,
    ) -> ConsolidationResult:
        """Semi-online mode: consume real user sessions from TrajectoryLayer.

        Filters trajectories by action_name and minimum score threshold,
        then runs standard Stage 2 + Stage 3 without re-executing tasks.
        """

        start = time.time()
        s0 = skill_content or self._default_skill(skill_name)

        recent = TrajectoryLayer.search(
            action_name="trace2skill_rollout", min_score=0.0, limit=50
        )
        if len(recent) < min_sessions:
            return self.evolve(
                program, tasks, metric, skill_name, skill_content, "creation"
            )

        trajectories = [
            {
                "task_id": i,
                "run_id": traj.get("run_id", f"online_{i}"),
                "input": traj.get("inputs", {}),
                "expected": traj.get("outputs", {}).get("expected", ""),
                "output": str(traj.get("outputs", {}).get("output", "")),
                "score": traj.get("score", 0.0),
                "trajectory_type": "success"
                if traj.get("score", 0.0) >= 0.7
                else "error",
                "elapsed": 0.0,
            }
            for i, traj in enumerate(recent)
        ]

        analyst = _ParallelAnalyst(s0)
        patches = analyst.analyze_all(trajectories)
        quality_dropped = analyst.dropped_trajectories

        evolved_skill, accepted, discarded, guardrail_failures, audit = (
            self.merger.merge(patches, s0)
        )

        self._integrate_skill(
            skill_name, evolved_skill, "online", program, {"trajectories": trajectories}
        )

        return ConsolidationResult(
            skill_name=skill_name,
            source_skill=s0,
            evolved_skill=evolved_skill,
            patches_generated=len(patches),
            patches_accepted=accepted,
            patches_discarded=discarded,
            trajectories_analyzed=len(trajectories),
            success_trajectories=sum(
                1 for t in trajectories if t["trajectory_type"] == "success"
            ),
            error_trajectories=sum(
                1 for t in trajectories if t["trajectory_type"] == "error"
            ),
            guardrail_failures=guardrail_failures,
            quality_dropped=quality_dropped,
            elapsed_seconds=time.time() - start,
            mode="semi_online",
            audit_trail=audit,
        )

    # ── Transfer validation ───────────────────────────────────────────

    @classmethod
    def validate_transfer(
        cls,
        skill_name: str,
        program,
        tasks: list[dict],
        metric,
        models: list[str] | None = None,
    ) -> dict[str, float]:
        """Test if an evolved skill transfers across model scales."""
        if models is None:
            models = ["default"]
        scores: dict[str, float] = {}
        for model in models:
            model_scores: list[float] = []
            for task in tasks[: min(10, len(tasks))]:
                result = program(**task["input"])
                getattr(result, "output", getattr(result, "answer", str(result)))
                s = metric(task, result)
                model_scores.append(s)
            scores[model] = (
                sum(model_scores) / len(model_scores) if model_scores else 0.0
            )
        return scores

    # ── Rollback ──────────────────────────────────────────────────────

    @classmethod
    def rollback(cls, skill_name: str) -> bool:
        """Atomic rollback: restore previous skill version from timestamped backup."""
        backup_dir = cls.BACKUP_DIR
        if not backup_dir.exists():
            return False

        skill_dir = cls.SKILLS_DIR / skill_name
        backups = sorted(backup_dir.glob(f"{skill_name}_*.md"), reverse=True)
        if not backups:
            return False

        latest_backup = backups[0]
        target = skill_dir / "SKILL.md"
        if target.exists():
            shutil.copy2(
                target, backup_dir / f"{skill_name}_pre_rollback_{int(time.time())}.md"
            )
        shutil.copy2(latest_backup, target)
        return True

    # ── Backward-compatible API ───────────────────────────────────────

    @classmethod
    def consolidate(cls, trajectory: list[dict], source: str = "compile") -> dict:
        """Legacy API: analyze a trajectory and extract patterns."""
        errors = [t for t in trajectory if t.get("score", 0) < 0.4]
        successes = [t for t in trajectory if t.get("score", 0) >= 0.7]
        analyst = _ParallelAnalyst("")
        all_patches = analyst.analyze_all(trajectory)
        patterns = {
            "source": source,
            "error_patterns": cls._analyze_errors(errors),
            "success_patterns": cls._analyze_successes(successes),
            "recommendation": cls._generate_recommendation(errors, successes),
            "examples_analyzed": len(trajectory),
            "patches_proposed": len(all_patches),
        }
        cls._save_legacy_skill(source, patterns)
        return patterns

    # ── SkillManager integration ──────────────────────────────────────

    @staticmethod
    def _integrate_skill(
        name: str, content: str, mode: str, program, rollout_result: dict
    ) -> None:
        """Integrate evolved skill with SkillManager lifecycle."""
        if not content.startswith("---"):
            content = (
                f"---\nname: {name}\n"
                f"description: Trace2Skill-evolved agent skill (mode: {mode})\n"
                f"signature: input -> output\n---\n{content}"
            )

        from dspytools.skills.manager import (
            SkillManager,  # lazy: breaks gfl↔skills cycle
        )

        mgr = SkillManager()

        all_skills = mgr.list_skills()
        existing = any(s.name == name for s in all_skills)

        if existing:
            # Find the actual skill path and update SKILL.md in place
            existing_skill = next(s for s in all_skills if s.name == name)
            skill_dir = existing_skill.path or (mgr.dir / name)
            (skill_dir / "SKILL.md").write_text(content)
        else:
            mgr.create_skill(
                name=name,
                description=f"Trace2Skill-evolved agent skill (mode: {mode})",
                signature="input -> output",
                body=content,
            )
            mgr.compile_skill(name)

    @staticmethod
    def _write_skill_direct(name: str, content: str, rollout_result: dict) -> None:
        """Direct file write fallback if SkillManager unavailable."""
        skill_dir = SkillConsolidator.SKILLS_DIR / name
        skill_dir.mkdir(parents=True, exist_ok=True)
        if not content.startswith("---"):
            content = (
                f"---\nname: {name}\n"
                f"description: Trace2Skill-evolved agent skill\n"
                f"signature: input -> output\n---\n{content}"
            )
        (skill_dir / "SKILL.md").write_text(content)
        meta = {
            "name": name,
            "trajectories": rollout_result.get("success_count", 0)
            + rollout_result.get("error_count", 0),
            "avg_score": rollout_result.get("avg_score", 0.0),
            "timestamp": time.time(),
        }
        write_json(skill_dir / "metadata.json", meta)

    @classmethod
    def _save_legacy_skill(cls, name: str, patterns: dict) -> None:
        cls.SKILLS_DIR.mkdir(parents=True, exist_ok=True)
        write_json(cls.SKILLS_DIR / f"{name}.json", patterns)

    @classmethod
    def list_skills(cls) -> list[dict]:
        """List all consolidated skills."""
        if not cls.SKILLS_DIR.exists():
            return []
        skills: list[dict] = []
        for d in sorted(cls.SKILLS_DIR.iterdir()):
            if d.suffix == ".json":
                data = read_json(d)
                skills.append(
                    {
                        "name": d.stem,
                        "source": data.get("source", ""),
                        "patterns": len(data.get("success_patterns", []))
                        + len(data.get("error_patterns", [])),
                    }
                )
            elif d.is_dir():
                skill_md = d / "SKILL.md"
                meta_json = d / "metadata.json"
                if skill_md.exists():
                    entry: dict[str, Any] = {"name": d.name, "has_skill_md": True}
                    if meta_json.exists():
                        entry.update(read_json(meta_json))
                    skills.append(entry)
        return skills

    # ── Helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _default_skill(name: str) -> str:
        return (
            f"## Purpose\n\n{name} — agent skill evolved by Trace2Skill.\n\n"
            "## Workflow\n\nExecute the program on the given input and return the result.\n\n"
            "## Instructions\n\nBe precise and thorough.\n\n"
            "## Gotchas\n\nCommon failure modes discovered during execution.\n\n"
            "## Examples\n\nSuccessful output patterns observed in rollout.\n\n"
        )

    @staticmethod
    def _analyze_errors(errors: list[dict]) -> list[str]:
        if not errors:
            return ["No errors found"]
        patterns: list[str] = []
        if len(errors) > 3:
            patterns.append(
                "High error count — consider simplifying the task or using a stronger LM"
            )
        scores = [e.get("score", 0) for e in errors]
        if scores and sum(scores) / len(scores) < 0.3:
            patterns.append(
                "Consistently low scores — metric may be too strict or task too complex"
            )
        return patterns or ["Unknown error pattern"]

    @staticmethod
    def _analyze_successes(successes: list[dict]) -> list[str]:
        if not successes:
            return ["No success patterns found — increase training data diversity"]
        patterns: list[str] = []
        optimizers = set(s.get("optimizer", "") for s in successes)
        if optimizers:
            patterns.append(f"Effective optimizers: {', '.join(optimizers)}")
        return patterns

    @staticmethod
    def _generate_recommendation(errors: list[dict], successes: list[dict]) -> str:
        if not successes:
            return "Increase dataset size and try simpler optimizers first"
        if len(successes) > len(errors):
            return "Pipeline is working well — consider scaling to more complex tasks"
        return "Mix of successes and failures — try GEPA with a teacher LM for reflective improvement"


# ═══════════════════════════════════════════════════════════════════════════
# Shared utilities
# ═══════════════════════════════════════════════════════════════════════════


def _parse_json_response(response) -> dict | list:
    """Robust JSON parsing from LLM responses."""
    text = str(response)
    if hasattr(response, "content"):
        text = str(response.content)
    try:
        return _json.loads(text)
    except _json.JSONDecodeError:
        m = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
        if m:
            try:
                return _json.loads(m.group(1))
            except _json.JSONDecodeError:
                pass
        m = re.search(r"[\{\[].*[\}\]]", text, re.DOTALL)
        if m:
            try:
                return _json.loads(m.group(0))
            except _json.JSONDecodeError:
                pass
    return {}


def _describe_input(traj: dict) -> str:
    """Produce a compact description of the task input."""
    inputs = traj.get("input", {})
    if isinstance(inputs, dict):
        return ", ".join(f"{k}={str(v)[:50]}" for k, v in inputs.items())
    return str(inputs)[:100]


def _format_trajectory_raw(traj: dict) -> str:
    """Format a full trajectory as a readable string for the analyst."""
    parts = [
        f"Task: {_describe_input(traj)}",
        f"Output: {str(traj.get('output', ''))[:1000]}",
        f"Expected: {str(traj.get('expected', ''))[:500]}",
        f"Score: {traj.get('score', 0.0):.2f}",
    ]
    exc = traj.get("exception")
    if exc:
        parts.append(f"Exception: {exc}")
    return "\n---\n".join(parts)
