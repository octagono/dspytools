"""GRPO optimizer adapter — RL-based optimization for DSPy pipelines.

GRPO (Group Relative Policy Optimization) applies reinforcement learning
to composed DSPy programs, treating the entire pipeline as a single policy.
Combined with GEPA via BetterTogether: 5-11% improvement (arXiv:2508.04660).
"""

from __future__ import annotations

from typing import Any

from dspy.teleprompt import GRPO


def compile_grpo(
    student: Any,
    trainset: list,
    lora: bool = False,
    beta: float = 0.01,
    max_steps: int = 10,
) -> Any:
    """Compile a program with GRPO (experimental, fails fast if unavailable).

    GRPO treats the pipeline as a policy:
      - Samples multiple outputs per input
      - Computes relative advantage between groups
      - Updates via policy gradient
    """

    # GRPO-specific metric: reward any valid output (truthy check)
    def _grpo_metric(example, prediction, trace=None):
        return (
            1.0
            if getattr(prediction, "output", getattr(prediction, "answer", ""))
            else 0.0
        )

    opt = GRPO(metric=_grpo_metric, lora=lora, beta=beta, max_steps=max_steps)
    return opt.compile(student=student, trainset=trainset)
