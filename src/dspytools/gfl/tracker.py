"""LSE delta tracker — tracks improvement as r = R̄(c₁) − R̄(c₀).

Optimization 15: Deferred state saves — only persists to disk when the best
score improves or every 10 records. Reduces I/O from O(N) writes per compile
to O(improvement_events + 10) writes.

From "Learning to Self-Evolve" (Chen et al., 2026):
  A 4B model trained with LSE outperforms GPT-5 as a self-evolving policy.
  Key insight: reward improvement, not absolute performance.
"""

from __future__ import annotations

import time
from pathlib import Path

from dspytools.config.settings import lse_log_path
from dspytools.core._io import read_json, write_json


class LSETracker:
    """Track improvement deltas like LSE: reward the improvement, not the score.

    r_LSE = R̄(c₁) − R̄(c₀) — only positive deltas count.
    """

    LOG_PATH: Path = lse_log_path()

    def __init__(self):
        self.history = self._load()
        self._baseline: float | None = None
        self._dirty = False
        self._records_since_save = 0

    def _load(self) -> dict:
        if self.LOG_PATH.exists():
            return read_json(self.LOG_PATH)
        return {"iterations": [], "best_score": 0.0, "total_improvement": 0.0}

    def _save(self) -> None:
        write_json(self.LOG_PATH, self.history)
        self._dirty = False
        self._records_since_save = 0

    def set_baseline(self, score: float) -> None:
        """Set the baseline context performance c₀."""
        self._baseline = score

    def record(
        self, optimizer: str, score: float, metadata: dict | None = None
    ) -> dict:
        """Record a score and compute delta from baseline."""
        if self._baseline is None:
            self._baseline = score

        delta = score - self._baseline
        improved = delta > 0

        entry = {
            "timestamp": time.time(),
            "optimizer": optimizer,
            "score": score,
            "baseline": self._baseline,
            "delta": delta,
            "improved": improved,
            "metadata": metadata or {},
        }

        self.history["iterations"].append(entry)
        if score > self.history["best_score"]:
            self.history["best_score"] = score
        if delta > 0:
            self.history["total_improvement"] += delta

        self.history["iterations"] = self.history["iterations"][-500:]

        # Optimization 15: Deferred state saves — save on improvement or every 10 records
        self._records_since_save += 1
        if improved or self._records_since_save >= 10:
            self._save()

        return entry

    @property
    def best_score(self) -> float:
        return self.history["best_score"]

    @property
    def baseline(self) -> float:
        """Current baseline score (0.5 if not set)."""
        return self._baseline or 0.5

    @property
    def total_improvement(self) -> float:
        return self.history["total_improvement"]

    @property
    def average_delta(self) -> float:
        deltas = [e["delta"] for e in self.history["iterations"] if "delta" in e]
        if not deltas:
            return 0.0
        return sum(deltas) / len(deltas)

    @property
    def improvement_trend(self) -> str:
        """Is the system improving? 'up', 'down', 'stable'."""
        recent = self.history["iterations"][-5:]
        if len(recent) < 3:
            return "stable"
        deltas = [e.get("delta", 0) for e in recent]
        if all(d > 0 for d in deltas):
            return "up"
        if all(d <= 0 for d in deltas):
            return "down"
        return "stable"
