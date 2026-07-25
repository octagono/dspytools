"""Meta-optimizer — learns which optimizer works best for each task type.

GFL Stage: Learn

Maintains a meta-model mapping task profiles → best optimizer.
"""

from __future__ import annotations

import time
from pathlib import Path

from dspytools.config.settings import meta_optimizer_path
from dspytools.core._io import read_json, write_json


class MetaOptimizer:
    """Meta-learner that selects the best optimizer for a given task.

    The meta-model tracks:
      - Task signature complexity (input/output count, field types)
      - Dataset size
      - Historical optimizer performance per task
    """

    META_PATH: Path = meta_optimizer_path()

    def __init__(self):
        self.history = self._load_history()

    def _load_history(self) -> dict:
        if self.META_PATH.exists():
            return read_json(self.META_PATH)
        return {"trials": [], "recommendations": {}}

    def _save_history(self) -> None:
        self.META_PATH.parent.mkdir(parents=True, exist_ok=True)
        write_json(self.META_PATH, self.history)

    def select_optimizer(
        self, program_id: str, dataset_size: int, complexity: str = "medium"
    ) -> dict:
        """Select the best optimizer based on task profile and history."""
        recommendations = {
            "small": "labeled_few_shot",  # < 10 examples
            "medium": "mipro",  # 10-50 examples
            "large": "gepa",  # 50+ examples
            "quick": "knn",  # fast, uses embeddings
            "thorough": "better_together",  # chains GEPA + finetune
        }

        if dataset_size < 10:
            key = "small"
        elif dataset_size < 50:
            key = "medium"
        else:
            key = "large"

        # Check history for overrides
        past_trials = [
            t for t in self.history["trials"] if t.get("complexity") == complexity
        ]
        if past_trials:
            best = max(past_trials, key=lambda t: t.get("score", 0))
            optimizer = best.get("optimizer", recommendations.get(key, "mipro"))
        else:
            optimizer = recommendations.get(key, "mipro")

        return {
            "program": program_id,
            "optimizer": optimizer,
            "dataset_size": dataset_size,
            "complexity": complexity,
            "reason": f"Selected {optimizer} for {complexity} complexity with {dataset_size} examples",
        }

    def record_result(
        self,
        optimizer: str,
        score: float,
        complexity: str,
        dataset_size: int,
        program: str,
    ) -> None:
        """Record an optimization result for future meta-learning."""
        self.history["trials"].append(
            {
                "timestamp": time.time(),
                "optimizer": optimizer,
                "score": score,
                "complexity": complexity,
                "dataset_size": dataset_size,
                "program": program,
            }
        )
        # Keep last 100 trials
        self.history["trials"] = self.history["trials"][-100:]
        self._save_history()

    def get_best_optimizer(self, complexity: str = "medium") -> str:
        """Return the best optimizer for a given complexity based on history."""
        trials = [
            t for t in self.history["trials"] if t.get("complexity") == complexity
        ]
        if not trials:
            return "mipro"
        best = max(trials, key=lambda t: t.get("score", 0))
        return best.get("optimizer", "mipro")
