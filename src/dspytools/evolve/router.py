"""RouterAgent — ReAct agent that routes queries and self-optimizes.

Has access to all dspytools features as tools:
  - list_programs, swap_program, infer
  - compile (any optimizer)
  - evaluate
  - self_optimize
"""

from __future__ import annotations

import json as _json
import os
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import dspy
else:
    from dspytools.core._dspy import dspy

import mlflow
from mlflow.tracing.fluent import trace as mlflow_trace

from dspytools.core.hotswap import HotSwapManager
from dspytools.core.setup import setup_dspy
from dspytools.evolve.metrics import auto_metric

# ---------------------------------------------------------------------------
# MLflow Tracing setup — configure tracking URI + experiment once
# ---------------------------------------------------------------------------

_TRACING_INITIALIZED = False


def setup_agent_tracing() -> None:
    """Configure MLflow Tracing for agent interactions.

    Idempotent — safe to call multiple times.
    Uses the same tracking URI and experiment as the existing MLflowTracker.
    """
    global _TRACING_INITIALIZED
    if _TRACING_INITIALIZED:
        return

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
    experiment_name = os.environ.get("MLFLOW_EXPERIMENT_NAME", "dspytools")

    mlflow.set_tracking_uri(tracking_uri)
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        mlflow.create_experiment(experiment_name)
    mlflow.set_experiment(experiment_name)

    # Enable DSPy autologging — auto-traces every dspy.Module, dspy.ReAct,
    # dspy.ChainOfThought, dspy.Predict, dspy.Tool, and LM call project-wide.
    # Uses importlib because mlflow.dspy is a dynamic plugin not statically resolvable.
    import importlib

    mdspy = importlib.import_module("mlflow.dspy")
    mdspy.autolog(
        log_traces=True,
        log_traces_from_compile=True,
        log_traces_from_eval=True,
        log_compiles=True,
        log_evals=True,
    )

    _TRACING_INITIALIZED = True


def _build_router_tools() -> list[dspy.Tool]:
    """Build the set of tools available to the RouterAgent."""

    mgr = HotSwapManager()
    mgr.load_all()

    @mlflow_trace(span_type="TOOL")
    def tool_list_programs() -> str:
        """List all loaded compiled programs."""
        return _json.dumps(mgr.list(), indent=2)

    @mlflow_trace(span_type="TOOL")
    def tool_swap_program(program_id: str) -> str:
        """Switch the active compiled program."""
        prev = mgr.swap(program_id)
        return _json.dumps({"status": "ok", "active": program_id, "previous": prev})

    @mlflow_trace(span_type="TOOL")
    def tool_infer(**inputs: Any) -> str:
        """Run inference with active program."""
        result = mgr.infer(**inputs)
        return _json.dumps({"status": "ok", "result": result}, default=str)

    @mlflow_trace(span_type="TOOL")
    def tool_list_optimizers() -> str:
        """List available DSPy optimizers."""
        return _json.dumps(
            [
                "knn",
                "mipro",
                "gepa",
                "better_together",
                "ensemble",
                "bootstrap_few_shot",
                "labeled_few_shot",
            ]
        )

    @mlflow_trace(span_type="TOOL")
    def tool_evaluate(content: str) -> str:
        """Evaluate output quality with auto-metric (0.0-1.0)."""
        score = auto_metric(content)
        return _json.dumps({"score": score})

    @mlflow_trace(span_type="TOOL")
    def tool_compile_decision(
        program_id: str, score: float, threshold: float = 0.5
    ) -> str:
        """Decide if a program needs re-compilation based on score."""
        needs_compile = score < threshold
        return _json.dumps(
            {
                "needs_compile": needs_compile,
                "program": program_id,
                "score": score,
                "threshold": threshold,
                "action": "re-compile with best optimizer"
                if needs_compile
                else "keep current",
            }
        )

    return [
        dspy.Tool(
            tool_list_programs,
            name="list_programs",
            desc="List all loaded compiled programs",
        ),
        dspy.Tool(
            tool_swap_program,
            name="swap_program",
            desc="Switch the active compiled program",
        ),
        dspy.Tool(tool_infer, name="infer", desc="Run inference with active program"),
        dspy.Tool(
            tool_list_optimizers,
            name="list_optimizers",
            desc="List available DSPy optimizers",
        ),
        dspy.Tool(
            tool_evaluate, name="evaluate", desc="Evaluate output quality score 0.0-1.0"
        ),
        dspy.Tool(
            tool_compile_decision,
            name="compile_decision",
            desc="Decide if re-compilation is needed",
        ),
    ]


class RouterAgent:
    """Self-evolving router that selects the best program and auto-optimizes.

    Uses ReAct v1 (text-based Thought/Action/Observation parsing) for
    compatibility with small models like Qwen 7B that can't produce
    the structured ToolCalls JSON expected by ReActV2.
    """

    def __init__(self, max_iters: int = 10):

        setup_dspy()
        setup_agent_tracing()
        tools = _build_router_tools()
        self.agent = dspy.ReAct(
            "question -> answer",
            tools=tools,
            max_iters=max_iters,
        )
        self.mgr = HotSwapManager()
        self.mgr.load_all()
        self._quality_history: list[dict] = []

    @mlflow_trace(span_type="AGENT")
    def ask(self, question: str) -> dict:
        """Route a query and return the answer with metadata."""
        result = self.agent(question=question)  # type: ignore[arg-type]
        answer = getattr(result, "answer", str(result))

        # Auto-evaluate quality
        score = auto_metric(answer)
        self._quality_history.append(
            {
                "question": question[:100],
                "score": score,
                "program": self.mgr.active_id,
            }
        )

        return {
            "answer": answer,
            "score": score,
            "active_program": self.mgr.active_id,
            "needs_optimization": score < 0.4,
        }

    def evolve(self) -> dict:
        """Trigger self-evolution. Returns optimization results."""
        avg_score = sum(h["score"] for h in self._quality_history[-10:]) / max(
            len(self._quality_history[-10:]), 1
        )

        return {
            "action": "self_evolve",
            "average_score": avg_score,
            "samples": len(self._quality_history),
            "active_program": self.mgr.active_id,
            "should_recompile": avg_score < 0.5,
        }

    @property
    def status(self) -> dict:
        return {
            "programs_loaded": len(self.mgr.list()),
            "active_program": self.mgr.active_id,
            "quality_samples": len(self._quality_history),
            "average_score": sum(h["score"] for h in self._quality_history)
            / max(len(self._quality_history), 1),
        }
