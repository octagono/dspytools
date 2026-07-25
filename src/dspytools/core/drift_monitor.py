"""Drift detection — monitors compiled program quality over time.

Optimization 14: Deferred state saves — only persists to disk when drift
is detected (warning/critical), not on every check() call. Reduces I/O
from O(N) writes per inference to O(drift_events) writes.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

from dspytools.config.settings import drift_state_path
from dspytools.core._io import read_json, write_json


@dataclass
class DriftSnapshot:
    """A single quality check for a compiled program."""

    run_id: str
    score: float
    checked_at: float = field(default_factory=time.time)
    holdout_size: int = 0
    delta_from_baseline: float = 0.0


@dataclass
class DriftAlert:
    """Alert when a program has degraded."""

    run_id: str
    severity: str  # "warning" or "critical"
    current_score: float
    baseline_score: float
    degradation_pct: float
    consecutive_drops: int
    message: str


class DriftMonitor:
    """Monitors compiled program quality for drift.

    Usage:
        monitor = DriftMonitor()
        monitor.update_baseline("run_123", 0.85)
        alert = monitor.check("run_123", current_score=0.72)
        if alert:
            print(f"DRIFT ALERT: {alert.message}")
    """

    def __init__(
        self,
        drift_threshold: float = 0.05,
        critical_threshold: float = 0.15,
        state_file: str | None = None,
    ):
        self.drift_threshold = drift_threshold  # 5% degradation = warning
        self.critical_threshold = critical_threshold  # 15% degradation = critical
        self.state_file = state_file or str(drift_state_path())

        self._baselines: dict[str, float] = {}
        self._history: dict[str, list[DriftSnapshot]] = {}
        self._recompile_requests: set[str] = set()
        self._load_state()

    def update_baseline(self, run_id: str, score: float) -> None:
        """Set or update the baseline quality for a program."""
        self._baselines[run_id] = score
        if run_id not in self._history:
            self._history[run_id] = []
        self._save_state()

    def check(
        self, run_id: str, current_score: float, holdout_size: int = 0
    ) -> DriftAlert | None:
        """Check if a program has drifted from its baseline.

        Returns DriftAlert if degradation detected, None otherwise.
        """
        baseline = self._baselines.get(run_id)
        if baseline is None:
            return None

        delta = baseline - current_score
        snapshot = DriftSnapshot(
            run_id=run_id,
            score=current_score,
            holdout_size=holdout_size,
            delta_from_baseline=delta,
        )
        self._history.setdefault(run_id, []).append(snapshot)

        # Keep last 50 snapshots
        if len(self._history[run_id]) > 50:
            self._history[run_id] = self._history[run_id][-50:]

        degradation_pct = (delta / baseline) * 100 if baseline > 0 else 0

        # Count consecutive drops
        consecutive = 0
        for snap in reversed(self._history[run_id]):
            if snap.delta_from_baseline > 0:
                consecutive += 1
            else:
                break

        if degradation_pct >= self.critical_threshold * 100:
            self._save_state()  # Only persist on drift events
            return DriftAlert(
                run_id=run_id,
                severity="critical",
                current_score=current_score,
                baseline_score=baseline,
                degradation_pct=round(degradation_pct, 1),
                consecutive_drops=consecutive,
                message=f"CRITICAL: {run_id} degraded {degradation_pct:.1f}% from baseline {baseline:.2f}. Re-compile recommended.",
            )
        elif degradation_pct >= self.drift_threshold * 100:
            self._save_state()  # Only persist on drift events
            return DriftAlert(
                run_id=run_id,
                severity="warning",
                current_score=current_score,
                baseline_score=baseline,
                degradation_pct=round(degradation_pct, 1),
                consecutive_drops=consecutive,
                message=f"WARNING: {run_id} drifted {degradation_pct:.1f}% from baseline {baseline:.2f}. Monitor closely.",
            )

        return None

    def get_history(self, run_id: str, last_n: int = 10) -> list[dict]:
        """Get recent quality snapshots for a program."""
        snaps = self._history.get(run_id, [])[-last_n:]
        return [
            {
                "score": s.score,
                "delta": round(s.delta_from_baseline, 4),
                "checked_at": s.checked_at,
                "holdout_size": s.holdout_size,
            }
            for s in snaps
        ]

    @property
    def status(self) -> dict:
        """Full drift monitor status."""
        results = {}
        for run_id, baseline in self._baselines.items():
            history = self._history.get(run_id, [])
            current = history[-1].score if history else baseline
            delta = baseline - current
            results[run_id] = {
                "baseline": round(baseline, 4),
                "current": round(current, 4),
                "delta": round(delta, 4),
                "checks": len(history),
                "last_checked": history[-1].checked_at if history else None,
            }
        return {
            "programs_tracked": len(self._baselines),
            "thresholds": {
                "warning": self.drift_threshold,
                "critical": self.critical_threshold,
            },
            "programs": results,
        }

    def request_recompile(self, run_id: str) -> None:
        """Mark a program for automatic recompilation due to drift."""
        self._recompile_requests.add(run_id)
        self._save_state()

    def pending_recompiles(self) -> list[str]:
        """Get list of programs queued for automatic recompilation."""
        return list(self._recompile_requests)

    def clear_recompile_request(self, run_id: str) -> None:
        """Clear a recompile request after processing."""
        self._recompile_requests.discard(run_id)
        self._save_state()

    def process_recompile_requests(self, auto_fix: bool = False) -> list[dict]:
        """Process pending recompile requests.

        Args:
            auto_fix: If True, actually trigger recompilation via GFLPipeline.
                      If False, just report what would be done.

        Returns:
            List of dicts with {run_id, status, action} for each request.
        """
        results = []
        if not self._recompile_requests:
            return results

        for run_id in list(self._recompile_requests):
            if not auto_fix:
                results.append(
                    {
                        "run_id": run_id,
                        "status": "pending",
                        "action": "dry-run — run with --auto-fix to recompile",
                    }
                )
                continue

            # Attempt auto-recompile via GFLPipeline
            from dspytools.core.loaders import load_module_by_name
            from dspytools.core.registry import get_run
            from dspytools.gfl.pipeline import GFLPipeline

            meta = get_run(run_id)
            if not meta:
                results.append(
                    {
                        "run_id": run_id,
                        "status": "failed",
                        "action": f"Run '{run_id}' not found in registry",
                    }
                )
                continue

            module_name = meta.get("module", "unknown")
            student = load_module_by_name(module_name)

            from dspytools.core._dspy import dspy

            default_trainset = [
                dspy.Example(
                    input="generate code output", output="code example"
                ).with_inputs("input"),
            ]

            history = self._history.get(run_id, [])
            baseline = self._baselines.get(run_id, 0.5)
            latest_delta = (baseline - history[-1].score) if history else 0.0
            use_draft = latest_delta < 0.10

            from dspytools.evolve.self_evolve import get_engine

            engine = get_engine()
            task_profile = engine.morphology.profile_task(
                description=run_id,
                field_count=2,
                data_size=10,
            )
            suggested = engine.suggest_optimizer(task_profile)

            pipeline = GFLPipeline(mode="single")
            if use_draft:
                compile_result = pipeline.compile_draft(
                    student=student,
                    trainset=default_trainset,
                    optimizer_name=suggested,
                    draft_rounds=2,
                    polish_rounds=1,
                )
                compile_result["strategy"] = "draft"
            else:
                compile_result = pipeline.run_single(
                    optimizer_name=suggested,
                    student=student,
                    trainset=default_trainset,
                    auto_synthesize=True,
                    auto_meta=True,
                    min_examples=5,
                )
                compile_result["strategy"] = "full"

            new_score = compile_result.get("best_score", 0.0)

            from dspytools.core.output import create_run_dir, save_program

            new_run_id, run_path = create_run_dir(f"drift_fix_{suggested}", run_id)
            save_program(
                run_path,
                compile_result.get("best_program", student),
                {"inputs": ["input"], "outputs": ["output"]},
                module_type="predict",
            )

            self.update_baseline(new_run_id, new_score)
            self.clear_recompile_request(run_id)

            results.append(
                {
                    "run_id": run_id,
                    "new_run_id": new_run_id,
                    "status": "recompiled",
                    "action": f"Recompiled with {suggested} (score: {new_score:.4f})",
                }
            )

        return results

    def _load_state(self) -> None:
        path = Path(self.state_file)
        if not path.exists():
            return
        data = read_json(path)
        self._baselines = data.get("baselines", {})
        self._recompile_requests = set(data.get("recompile_requests", []))
        # Restore history snapshots (backward compat: old state files lack "history")
        for run_id, snaps in data.get("history", {}).items():
            self._history[run_id] = [
                DriftSnapshot(
                    run_id=run_id,
                    score=s.get("score", 0.0),
                    holdout_size=s.get("holdout_size", 0),
                    delta_from_baseline=s.get("delta_from_baseline", 0.0),
                    checked_at=s.get("checked_at"),
                )
                for s in snaps
            ]

    def _save_state(self) -> None:
        write_json(
            self.state_file,
            {
                "baselines": self._baselines,
                "recompile_requests": list(self._recompile_requests),
                "history": {
                    run_id: [
                        {
                            "score": s.score,
                            "delta_from_baseline": s.delta_from_baseline,
                            "checked_at": s.checked_at,
                            "holdout_size": s.holdout_size,
                        }
                        for s in snaps[-50:]
                    ]
                    for run_id, snaps in self._history.items()
                    if snaps
                },
            },
        )


# Module-level singleton
_monitor: DriftMonitor | None = None


def get_drift_monitor() -> DriftMonitor:
    global _monitor
    if _monitor is None:
        _monitor = DriftMonitor()
    return _monitor
