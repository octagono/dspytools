"""ResourceBudget — hard limits on LLM calls, wall time, agents.

Lab 11 / LSE pattern: prevents runaway costs in autonomous operation.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class ResourceBudget:
    """Enforce hard limits on autonomous optimization resources.

    Default: 100 LLM calls, 5 minutes wall time, 10 agents generated.
    Production: 200/600/20. Light: 50/180/5.
    """

    max_llm_calls: int = 100
    max_wall_seconds: int = 300
    max_agents_generated: int = 10
    max_iterations: int = 20

    _llm_calls_used: int = 0
    _agents_generated: int = 0
    _tokens_spent: int = 0
    _start_time: float = field(default_factory=time.time)

    def check_llm(self) -> None:
        self._llm_calls_used += 1
        if self._llm_calls_used > self.max_llm_calls:
            raise RuntimeError(f"LLM call budget exceeded ({self.max_llm_calls})")

    def check_agent(self) -> None:
        self._agents_generated += 1
        if self._agents_generated > self.max_agents_generated:
            raise RuntimeError(
                f"Agent generation budget exceeded ({self.max_agents_generated})"
            )

    def check_wall(self) -> None:
        elapsed = time.time() - self._start_time
        if elapsed > self.max_wall_seconds:
            raise RuntimeError(f"Wall time budget exceeded ({self.max_wall_seconds}s)")

    def check(self) -> None:
        """Check all budgets at once."""
        self.check_wall()
        self.check_llm()

    def spend_tokens(self, tokens: int) -> None:
        """Track token spending."""
        self._tokens_spent += tokens

    @property
    def tokens_spent(self) -> int:
        return self._tokens_spent

    @property
    def remaining_llm(self) -> int:
        return max(0, self.max_llm_calls - self._llm_calls_used)

    @property
    def elapsed_seconds(self) -> float:
        return time.time() - self._start_time

    @property
    def summary(self) -> dict:
        return {
            "llm_calls_used": self._llm_calls_used,
            "llm_calls_remaining": self.remaining_llm,
            "tokens_spent": self._tokens_spent,
            "elapsed_seconds": round(self.elapsed_seconds, 1),
            "agents_generated": self._agents_generated,
        }

    PRODUCTION = {
        "max_llm_calls": 200,
        "max_wall_seconds": 600,
        "max_agents_generated": 20,
    }
    LIGHT = {"max_llm_calls": 50, "max_wall_seconds": 180, "max_agents_generated": 5}
