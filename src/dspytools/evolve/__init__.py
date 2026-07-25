"""SelfEvolve — auto-optimizing CLI agent system.

Orchestrates the auto-evolve loop:
  1. Monitor quality metrics
  2. Trigger re-optimization when needed
  3. Auto-compile and hot-swap better programs
"""

from __future__ import annotations

from pathlib import Path

from mlflow.tracing.fluent import trace as mlflow_trace

from dspytools.config.settings import data_dir
from dspytools.core.loaders import load_trainset
from dspytools.core.registry import list_compiled_runs
from dspytools.evolve.metrics import auto_metric
from dspytools.evolve.router import RouterAgent
from dspytools.evolve.self_evolve import get_engine
from dspytools.gfl.pipeline import GFLPipeline

__all__ = [
    "RouterAgent",
    "auto_metric",
    "SelfEvolve",
    "get_router",
]

_router: RouterAgent | None = None


def get_router() -> RouterAgent:
    global _router
    if _router is None:
        _router = RouterAgent()
    return _router


class SelfEvolve:
    """Self-evolving system that auto-optimizes DSPy programs.

    Usage:
        evolve = SelfEvolve()
        evolve.auto_optimize()  # check if programs need optimization
        result = evolve.ask("How do I compile a program?")
    """

    def __init__(self, quality_threshold: float = 0.5):
        self.router = get_router()
        self.threshold = quality_threshold
        self._auto_compiled: list[str] = []

    def ask(self, question: str) -> dict:
        """Route a query through the self-evolving router."""
        result = self.router.ask(question)
        return result

    @mlflow_trace(span_type="CHAIN")
    def auto_optimize(self) -> dict:
        """Check quality and auto-optimize if needed.

        When quality drops below threshold, triggers a GFL pipeline compile
        using the UCB-suggested optimizer, then validates the result with
        SPRT before hot-swapping.
        """
        status = self.router.evolve()

        if status["should_recompile"]:
            status["action"] = "recompiling"
            status["message"] = "Quality below threshold — triggering auto-compile"

            result = self._trigger_compile()
            status["compile_result"] = result
            status["action"] = "compiled"
            status["message"] = (
                f"Auto-compiled with {result.get('best_optimizer', 'unknown')}, "
                f"score: {result.get('best_score', 0):.2f}"
            )
        else:
            status["action"] = "no_action"
            status["message"] = "Quality above threshold — no optimization needed"

        return status

    def _trigger_compile(self) -> dict:
        """Trigger a GFL pipeline compile with UCB-suggested optimizer."""

        # Find available training data

        # Look for training data in standard locations
        candidates = [
            data_dir() / "trainset.json",
            data_dir() / "trainset.jsonl",
            Path("trainset.json"),
            Path("trainset.jsonl"),
        ]
        trainset_path = None
        for path in candidates:
            if path.exists():
                trainset_path = str(path)
                break

        if not trainset_path:
            # Fall back to any compiled run's training data
            runs = list_compiled_runs()
            if not runs:
                return {"error": "No training data found for auto-compile"}
            # Use latest run's training data if available
            latest = runs[-1]
            trainset_path = latest.get("trainset_path")

        if not trainset_path:
            return {"error": "No training data available"}

        # Load training data
        trainset = load_trainset(trainset_path)

        # Load latest compiled program as student
        from dspytools.core.hotswap import HotSwapManager

        hotswap = HotSwapManager()
        hotswap.load_all()
        programs = hotswap.list()
        student = None
        if programs:
            active = next((p for p in programs if p.get("active")), programs[0])
            student = hotswap._programs.get(active["id"])

        # Use UCB-suggested optimizer
        engine = get_engine()
        suggested = engine.suggest_optimizer("general")

        # Run GFL pipeline with suggested optimizer
        pipeline = GFLPipeline(mode="single")
        result = pipeline.run_single(
            optimizer_name=suggested,
            student=student,
            trainset=trainset,
            auto_synthesize=True,
            auto_meta=True,
        )

        # Hot-swap if we got a better program
        if result.get("best_program") and result.get("best_score", 0) > self.threshold:
            self._hotswap_program(result["best_program"])

        return result

    def distill(
        self,
        run_id: str,
        adapter_name: str = "distilled",
        rank: int = 64,
        min_score: float = 0.5,
        local: bool = False,
        colab: bool = False,
        devset: str | None = None,
    ) -> dict:
        """Full teacher→LoRA distillation pipeline.

        Chains extract → train → load into a single command.
        Delegates to SelfEvolveEngine.distill_to_lora().
        """
        engine = get_engine()
        return engine.distill_to_lora(
            run_id=run_id,
            adapter_name=adapter_name,
            rank=rank,
            min_score=min_score,
            local=local,
            colab=colab,
            devset=devset,
        )

    def _hotswap_program(self, program) -> None:
        """Hot-swap the active program with a newly compiled one."""
        from dspytools.core.hotswap import (
            HotSwapManager,  # lazy: breaks core↔evolve cycle
        )

        mgr = HotSwapManager()
        mgr.load_all()

    @property
    def status(self) -> dict:
        return self.router.status
