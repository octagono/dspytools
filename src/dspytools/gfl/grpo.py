"""GRPO optimizer adapter — RL-based optimization for DSPy programs.

GRPO (Group Relative Policy Optimization) applies reinforcement learning to
composed DSPy programs, treating the program as a single policy: it samples
multiple rollouts per input, computes relative advantages between groups and
updates the predictor LMs through ``LM.finetune()``.

Because the update *is* a weight update, GRPO only works with an LM that
implements ``.finetune()`` (a provider fine-tuning API). Local llama.cpp
endpoints do not, so callers must guard the same way ``compile finetune`` does.
"""

from __future__ import annotations

from typing import Any, Callable

from dspytools.core.metrics import exact_match_metric


def _grpo_class() -> Any:
    """Resolve ``GRPO`` lazily.

    DSPy 3.5 exports GRPO from neither ``dspy`` nor ``dspy.teleprompt``, so the
    submodule is the only import path — resolved on use to keep the CLI's
    import chain lazy (Golden Rule 1).
    """
    from dspy.teleprompt.grpo import GRPO

    return GRPO


def _default_metric() -> Callable[..., float]:
    """Normalized exact-match reward (correctness, not mere truthiness)."""
    return exact_match_metric()


def compile_grpo(
    student: Any,
    trainset: list,
    *,
    metric: Callable[..., float] | None = None,
    valset: list | None = None,
    max_steps: int = 10,
    num_rollouts: int = 1,
    num_threads: int = 1,
    seed: int = 0,
    train_kwargs: dict[str, Any] | None = None,
    multitask: bool = True,
) -> Any:
    """Compile a program with GRPO (3.5 signature).

    Args:
        student: Program whose predictor LMs will be fine-tuned.
        trainset: Training examples.
        metric: Reward metric; defaults to normalized exact match.
        valset: Optional validation examples (never part of the training set).
        max_steps: Mapped to GRPO's ``num_train_steps``.
        num_rollouts: Mapped to ``num_rollouts_per_grpo_step``.
        num_threads: Rollout parallelism (keep at 1 for a single local model).
        seed: GRPO sampling seed.
        train_kwargs: Forwarded to the provider's ``LM.finetune()``.
        multitask: GRPO's multi-predictor grouping mode.

    Returns:
        The compiled program.

    Raises:
        TypeError: The installed ``GRPO`` signature no longer accepts these
            kwargs (the adapter is signature-checked, never version-pinned).
    """
    # `lora` and `beta` from older DSPy GRPO releases no longer exist: LoRA is a
    # provider-side decision expressed through train_kwargs, and KL weighting is
    # not configurable in 3.5's GRPO.
    optimizer = _grpo_class()(
        metric=metric or _default_metric(),
        multitask=multitask,
        train_kwargs=train_kwargs,
        num_threads=num_threads,
        num_train_steps=max_steps,
        seed=seed,
        num_rollouts_per_grpo_step=num_rollouts,
    )
    return optimizer.compile(student=student, trainset=trainset, valset=valset)
