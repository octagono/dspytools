"""Cost tracking — real token usage plus cost estimation for compile/LM runs.

Tracks token usage for teacher LM (DeepSeek V4 Flash) and student LM (Qwen 9B
local LLM) and prices it from the published table below.

Two sources, one sink
─────────────────────
1. **Measured** — a DSPy ``Prediction`` carries the real usage DSPy observed
   when ``track_usage`` is enabled (``setup_dspy()`` turns it on by default):

       result.get_lm_usage()

   returns ``{model_name: {usage_fields...}}`` where ``usage_fields`` are what
   the engine reported, merged across every call the module made::

       {"openai/gpt-4o": {"prompt_tokens": 120,
                          "completion_tokens": 42,
                          "total_tokens": 162,
                          "prompt_tokens_details": None,
                          "completion_tokens_details": None}}

   Provider-specific extra fields (``is_byok`` flags, cache-token counters, …)
   flow through verbatim; only numeric non-bool fields are summed.  Measured
   with dspy 3.5.0 against a stub engine: ``get_lm_usage()`` is ``None`` unless
   ``settings.track_usage`` is on, and a DummyLM-backed call reports zeros.

2. **Per-call history** — when a single LM object is in hand,
   ``lm.history[-1]`` holds ``{"usage": {...}, "cost": float | None,
   "model": str, ...}``.  ``cost`` is the provider-reported USD cost and is
   ``None`` for cached calls (no request was made).

3. **Estimated** — ``CompileCost.add_call()`` keeps the original API for callers
   with no result object (optimizer cost tables).  It prices token counts from
   ``PRICING``.

``CompileCost.add_usage()`` / ``add_result()`` are additive: they record
measured usage and never change the meaning of the estimate-based path.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

# Pricing per 1M tokens (as of 2025-07)
PRICING = {
    "deepseek/deepseek-v4-flash": {"input": 0.14, "output": 0.28},
    "deepseek/deepseek-v3": {"input": 0.27, "output": 1.10},
    "unsloth/Qwen3.5-9B-GGUF": {"input": 0.0, "output": 0.0},  # local
    "Qwen/Qwen3.5-9B": {"input": 0.0, "output": 0.0},  # local
    "default": {"input": 0.15, "output": 0.60},
}

#: Keys used when a usage entry does not report ``total_tokens``.
_PROMPT_KEYS = ("prompt_tokens", "input_tokens")
_COMPLETION_KEYS = ("completion_tokens", "output_tokens")


@dataclass
class TokenUsage:
    """Token usage for a single LM call (measured or estimated)."""

    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_estimate: float = 0.0
    #: ``True`` when the numbers came from DSPy/the provider, ``False`` when
    #: they were inferred from ``PRICING`` by ``estimate()``.
    measured: bool = False

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

    @classmethod
    def from_usage(
        cls,
        model: str,
        usage: dict[str, Any] | None,
        *,
        cost: float | None = None,
    ) -> TokenUsage:
        """Build from one DSPy usage entry.

        ``usage`` is one value of ``Prediction.get_lm_usage()`` (or
        ``lm.history[-1]["usage"]``).  A provider-reported ``cost`` wins over
        the ``PRICING`` estimate.  An empty/``None`` entry (cached call, no
        request made) yields a zero-usage record marked measured.
        """
        entry = usage or {}
        prompt = _first_int(entry, _PROMPT_KEYS)
        completion = _first_int(entry, _COMPLETION_KEYS)
        total = _first_int(entry, ("total_tokens",)) or (prompt + completion)
        if cost is None:
            pricing = PRICING.get(model, PRICING["default"])
            cost = (prompt / 1_000_000) * pricing["input"] + (
                completion / 1_000_000
            ) * pricing["output"]
        return cls(
            model=model,
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=total,
            cost_estimate=round(float(cost), 6),
            measured=True,
        )


def _first_int(entry: dict[str, Any], keys: tuple[str, ...]) -> int:
    """First non-bool numeric value among ``keys``, as an int (0 if absent)."""
    for key in keys:
        value = entry.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        return int(value)
    return 0


def usage_from_result(result: Any) -> dict[str, dict[str, Any]]:
    """``Prediction.get_lm_usage()`` as ``{model: usage}``; ``{}`` when absent.

    Tolerates non-``Prediction`` results (plain dicts, ``None``) so callers can
    pass whatever a program returned.  ``get_lm_usage()`` returns ``None`` when
    usage tracking is off or the program returned no ``Prediction``.
    """
    getter = getattr(result, "get_lm_usage", None)
    if getter is None:
        return {}
    usage = getter()
    return usage if isinstance(usage, dict) else {}


def usage_from_lm_history(lm: Any) -> TokenUsage | None:
    """Last measured call from ``lm.history``, or ``None`` when unavailable.

    Cached calls report ``usage=None``/``cost=None`` — a cache hit made no
    request, so there is nothing to bill and nothing to record.
    """
    history = getattr(lm, "history", None)
    if not history:
        return None
    entry = history[-1]
    if not isinstance(entry, dict):
        return None
    usage = entry.get("usage")
    if not usage:
        return None
    model = entry.get("model") or getattr(lm, "model", "") or "default"
    return TokenUsage.from_usage(str(model), usage, cost=entry.get("cost"))


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
        """Record one call from token counts and price it from ``PRICING``."""
        usage = TokenUsage.estimate(model, prompt_tokens, completion_tokens)
        self.calls.append(usage)
        self.total_tokens += usage.total_tokens
        self.total_cost += usage.cost_estimate

    def add_usage(self, usage_by_model: dict[str, dict[str, Any]]) -> int:
        """Record measured usage in ``get_lm_usage()`` shape; return call count.

        Entries whose token counts are all zero (a cache hit, or an engine that
        reports nothing) are skipped — they cost nothing and would only inflate
        the call count.
        """
        added = 0
        for model, entry in (usage_by_model or {}).items():
            usage = TokenUsage.from_usage(model, entry)
            if usage.total_tokens == 0 and usage.cost_estimate == 0:
                continue
            self.calls.append(usage)
            self.total_tokens += usage.total_tokens
            self.total_cost += usage.cost_estimate
            added += 1
        return added

    def add_result(self, result: Any, lm: Any = None) -> int:
        """Record real usage from a DSPy result (and its LM's last history call).

        ``result.get_lm_usage()`` is preferred; when usage tracking was off it
        is ``None`` and this falls back to ``lm.history[-1]``.  A cached call
        reports ``None`` for both, so nothing is recorded and ``0`` is
        returned — the estimate-based ``add_call`` path still works.
        """
        usage = usage_from_result(result)
        if usage:
            return self.add_usage(usage)
        if lm is None:
            return 0
        single = usage_from_lm_history(lm)
        if single is None:
            return 0
        self.calls.append(single)
        self.total_tokens += single.total_tokens
        self.total_cost += single.cost_estimate
        return 1

    def finish(self) -> None:
        self.finished_at = time.time()

    def summary(self) -> dict:
        measured = sum(1 for c in self.calls if c.measured)
        return {
            "compile_id": self.compile_id,
            "optimizer": self.optimizer,
            "elapsed_seconds": round(self.elapsed_seconds, 1),
            "total_tokens": self.total_tokens,
            "teacher_tokens": self.teacher_tokens,
            "student_tokens": self.student_tokens,
            "total_cost_usd": round(self.total_cost, 6),
            "calls": len(self.calls),
            # Measured-vs-estimated split: 0 means pure PRICING estimates.
            "measured_calls": measured,
            "estimated_calls": len(self.calls) - measured,
        }
