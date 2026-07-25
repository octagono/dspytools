"""Cost tracking — token counting and cost estimation for compile operations.

Tracks token usage for teacher LM (DeepSeek V4 Flash) and student LM (Qwen 9B local LLM).
Estimates costs based on published pricing.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

# Pricing per 1M tokens (as of 2025-07)
PRICING = {
    "deepseek/deepseek-v4-flash": {"input": 0.14, "output": 0.28},
    "deepseek/deepseek-v3": {"input": 0.27, "output": 1.10},
    "unsloth/Qwen3.5-9B-GGUF": {"input": 0.0, "output": 0.0},  # local
    "Qwen/Qwen3.5-9B": {"input": 0.0, "output": 0.0},  # local
    "default": {"input": 0.15, "output": 0.60},
}


@dataclass
class TokenUsage:
    """Token usage for a single LM call."""

    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_estimate: float = 0.0

    @classmethod
    def estimate(
        cls, model: str, prompt_tokens: int = 0, completion_tokens: int = 0
    ) -> TokenUsage:
        pricing = PRICING.get(model, PRICING["default"])
        input_cost = (prompt_tokens / 1_000_000) * pricing["input"]
        output_cost = (completion_tokens / 1_000_000) * pricing["output"]
        return cls(
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            cost_estimate=round(input_cost + output_cost, 6),
        )


@dataclass
class CompileCost:
    """Aggregated cost for a compile operation."""

    compile_id: str
    optimizer: str
    started_at: float = field(default_factory=time.time)
    finished_at: float = 0.0
    calls: list[TokenUsage] = field(default_factory=list)
    total_tokens: int = 0
    total_cost: float = 0.0

    @property
    def elapsed_seconds(self) -> float:
        return self.finished_at - self.started_at if self.finished_at > 0 else 0

    @property
    def teacher_tokens(self) -> int:
        return sum(c.total_tokens for c in self.calls if "deepseek" in c.model)

    @property
    def student_tokens(self) -> int:
        return sum(c.total_tokens for c in self.calls if "deepseek" not in c.model)

    def add_call(
        self, model: str, prompt_tokens: int = 0, completion_tokens: int = 0
    ) -> None:
        usage = TokenUsage.estimate(model, prompt_tokens, completion_tokens)
        self.calls.append(usage)
        self.total_tokens += usage.total_tokens
        self.total_cost += usage.cost_estimate

    def finish(self) -> None:
        self.finished_at = time.time()

    def summary(self) -> dict:
        return {
            "compile_id": self.compile_id,
            "optimizer": self.optimizer,
            "elapsed_seconds": round(self.elapsed_seconds, 1),
            "total_tokens": self.total_tokens,
            "teacher_tokens": self.teacher_tokens,
            "student_tokens": self.student_tokens,
            "total_cost_usd": round(self.total_cost, 6),
            "calls": len(self.calls),
        }
