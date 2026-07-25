"""GFL (Generative Feedback Loop) orchestrator.

The GFL module implements the full self-meta-learning loop:
  Generate → Evaluate → Keep → Learn → Deploy

CLI: dspytools gfl synthesize|meta-optimize|decompose|ab-test|consolidate|spin|lse|gepa
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from dspytools.core.metrics import auto_metric
from dspytools.core.registry import get_run
from dspytools.evolve.self_evolve import SelfEvolveEngine
from dspytools.gfl.consolidation import (
    ErrorAnalystModule,
    MergeOperatorModule,
    SkillConsolidator,
    SuccessAnalystModule,
)
from dspytools.gfl.meta_learn import MetaOptimizer
from dspytools.gfl.paper_optimizers import (
    GEPAParetoFrontier,
    GRAOMetaOptimizer,
    LSESelfEvolveModule,
    LSETreeExplorer,
    MetaPromptOptimizer,
    PurifiedOPSDModule,
    PurifiedOPSDOptimizer,
    SpinDiscriminateModule,
    SPINOptimizer,
)
from dspytools.gfl.synthetic import ChallengerSolver, DataSynthesizer

__all__ = [
    "GFLLoop",
    "SPINOptimizer",
    "SpinDiscriminateModule",
    "MetaPromptOptimizer",
    "LSETreeExplorer",
    "LSESelfEvolveModule",
    "GEPAParetoFrontier",
    "GRAOMetaOptimizer",
    "PurifiedOPSDModule",
    "PurifiedOPSDOptimizer",
    "ChallengerSolver",
    "SkillConsolidator",
    "ErrorAnalystModule",
    "SuccessAnalystModule",
    "MergeOperatorModule",
]


class GFLLoop:
    """Full GFL cycle: monitor → synthesize → optimize → evaluate → deploy."""

    def __init__(
        self, quality_threshold: float = 0.6, improvement_threshold: float = 0.05
    ):
        self.quality_threshold = quality_threshold
        self.improvement_threshold = improvement_threshold
        self.history: list[dict] = []
        self._loaded: bool = False
        # Optimization 12: Reuse SelfEvolveEngine instead of creating per call
        self._engine = None
        # Optimization 16: Reuse MetaOptimizer and DataSynthesizer per GFLLoop instance
        self._meta_optimizer = None
        self._data_synth = None
        # Optimization 29: Reuse HotSwapManager per GFLLoop instance
        self._hotswap = None

    def _get_engine(self):
        """Lazy-init SelfEvolveEngine singleton per GFLLoop instance (Optimization 12)."""
        if self._engine is None:
            self._engine = SelfEvolveEngine()
        return self._engine

    def run(self, program_id: str, trainset_path: str, max_iterations: int = 3) -> dict:
        """Run the full GFL loop."""
        results = {"iterations": [], "final": {}}

        for i in range(max_iterations):
            iteration = {"iteration": i + 1, "timestamp": datetime.now().isoformat()}

            # 1. Monitor quality
            iteration["monitor"] = self._monitor(program_id)

            # 2. If regression, synthesize more data
            if iteration["monitor"]["needs_improvement"]:
                iteration["synthesize"] = self._synthesize(trainset_path, target=10)

            # 3. Meta-optimize: pick best optimizer
            iteration["meta"] = self._meta_learn(program_id)

            # 4. Evaluate result
            iteration["evaluate"] = self._evaluate(program_id)

            results["iterations"].append(iteration)

            # 5. If good enough, stop
            if not iteration["monitor"]["needs_improvement"]:
                break

        results["final"] = {
            "iterations_completed": len(results["iterations"]),
            "final_program": program_id,
            "history": self.history,
        }
        return results

    def _monitor(self, program_id: str) -> dict:
        """Monitor quality by querying the self-evolve engine's convergence state.

        Optimization 12: Reuses cached SelfEvolveEngine instead of creating new one per call.
        """
        engine = self._get_engine()
        convergence = engine.check_convergence([])
        return {
            "program": program_id,
            "needs_improvement": not convergence.get("safe", True),
            "reason": (
                convergence.get("repetition_warning")
                or convergence.get("degradation_warning")
                or "Quality stable"
            ),
            "score_variance": convergence.get("score_variance", 0.0),
            "max_repetition_ratio": convergence.get("max_repetition_ratio", 0.0),
        }

    def _synthesize(self, trainset_path: str, target: int = 10) -> dict:
        """Synthesize training data. Optimization 16: Reuses cached DataSynthesizer."""

        if self._data_synth is None:
            self._data_synth = DataSynthesizer()
        result = self._data_synth.generate(Path(trainset_path), target_count=target)
        return {"generated": result["generated"], "path": result["output_path"]}

    def _meta_learn(self, program_id: str) -> dict:
        """Meta-learn best optimizer. Optimization 16: Reuses cached MetaOptimizer."""

        if self._meta_optimizer is None:
            self._meta_optimizer = MetaOptimizer()
        return self._meta_optimizer.select_optimizer(program_id, len(self.history))

    def _evaluate(self, program_id: str) -> dict:
        """Evaluate a program's quality using registry metadata + auto_metric."""

        # Optimization 29: Reuse HotSwapManager
        if self._hotswap is None:
            from dspytools.core.hotswap import HotSwapManager  # lazy: breaks cycle

            self._hotswap = HotSwapManager()

        run = get_run(program_id)
        if not run:
            return {
                "program": program_id,
                "score": 0.0,
                "improved": False,
                "error": "Program not found",
            }

        # Use the shared HotSwapManager
        if not self._hotswap.is_loaded(program_id):
            self._hotswap.load_single(program_id)

        if self._hotswap.is_loaded(program_id):
            prog = self._hotswap._programs[program_id]
            result = prog(question="test")
            content = getattr(result, "answer", getattr(result, "output", str(result)))
            score = auto_metric(str(content))
        else:
            score = 0.5  # Can't load, neutral score

        improved = len(self.history) > 0 and score > self.history[-1].get(
            "evaluate", {}
        ).get("score", 0.5)

        return {
            "program": program_id,
            "score": score,
            "improved": improved,
        }
