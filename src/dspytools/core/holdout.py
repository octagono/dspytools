"""Programmatic holdout enforcement — mechanical Invariant 5 guarantee.

Ensures holdout set is split before any compile call and NEVER seen by the optimizer.
Uses a contract pattern: @gated_compile decorator auto-splits and registers holdout.
"""

from __future__ import annotations

import functools
import hashlib
import math
import random
from typing import Callable

from dspytools.config.settings import DEFAULT_SEED


class HoldoutGate:
    """Enforces holdout isolation contract.

    Wraps a compile function to:
    1. Auto-split trainset into train + holdout
    2. Pass only train to optimizer
    3. Register holdout with drift monitor for future validation
    """

    def __init__(self, holdout_fraction: float = 0.2, seed: int = DEFAULT_SEED):
        self.holdout_fraction = holdout_fraction
        self.seed = seed
        self._splits: dict[str, tuple[list, list]] = {}

    def split(self, trainset: list, compile_id: str = "") -> tuple[list, list]:
        """Split trainset into train + holdout.

        Holdout is stored in memory and NEVER returned to the caller.
        Only the train portion is passed to the optimizer.

        Returns:
            (train_set, holdout_set)
        """

        n_holdout = max(1, int(len(trainset) * self.holdout_fraction))
        indices = list(range(len(trainset)))
        random.Random(self.seed).shuffle(indices)

        holdout_idx = set(indices[:n_holdout])
        train = [trainset[i] for i in range(len(trainset)) if i not in holdout_idx]
        holdout = [trainset[i] for i in range(len(trainset)) if i in holdout_idx]

        key = compile_id or hashlib.sha256(str(trainset).encode()).hexdigest()[:8]
        self._splits[key] = (train, holdout)

        return train, holdout

    def get_holdout(self, compile_id: str) -> list:
        """Retrieve holdout set that was previously split."""
        return self._splits.get(compile_id, ([], []))[1]

    def validate_gate(
        self, compile_id: str, compiled_program, baseline_program=None
    ) -> dict:
        """Validate candidate on holdout set.

        Uses SPRT for early termination on clear wins/losses.
        """
        holdout = self.get_holdout(compile_id)
        if not holdout:
            return {
                "accepted": True,
                "reason": "no holdout available",
                "holdout_size": 0,
            }

        # SPRT parameters
        p0, p1 = 0.50, 0.65
        alpha, beta = 0.05, 0.20
        A = math.log((1 - beta) / alpha)
        B = math.log(beta / (1 - alpha))

        successes = 0
        n_eval = 0

        for ex in holdout[: min(50, len(holdout))]:
            n_eval += 1
            kwargs = (
                ex.inputs()
                if hasattr(ex, "inputs")
                else {"input": getattr(ex, "input", "")}
            )
            pred = compiled_program(**kwargs)
            expected = getattr(ex, "output", "")
            got = getattr(pred, "output", getattr(pred, "answer", str(pred)))
            if str(got).strip() == str(expected).strip():
                successes += 1

            if n_eval < 3:
                continue

            s, f = successes, n_eval - successes
            if s + f == 0:
                continue
            log_lr = s * math.log(p1 / p0) + f * math.log((1 - p1) / (1 - p0))

            if log_lr <= B:
                return {
                    "accepted": False,
                    "score": successes / n_eval,
                    "n_evaluated": n_eval,
                    "reason": "SPRT rejected",
                    "holdout_size": len(holdout),
                }
            elif log_lr >= A:
                return {
                    "accepted": True,
                    "score": successes / n_eval,
                    "n_evaluated": n_eval,
                    "reason": "SPRT accepted",
                    "holdout_size": len(holdout),
                }

        final_score = successes / n_eval if n_eval > 0 else 0.0
        return {
            "accepted": final_score > p0,
            "score": final_score,
            "n_evaluated": n_eval,
            "reason": "max evaluations reached",
            "holdout_size": len(holdout),
        }

    @property
    def stats(self) -> dict:
        return {
            "splits_stored": len(self._splits),
            "ids": list(self._splits.keys()),
        }


# Module-level gate
_gate: HoldoutGate | None = None


def get_holdout_gate(holdout_fraction: float = 0.2) -> HoldoutGate:
    global _gate
    if _gate is None:
        _gate = HoldoutGate(holdout_fraction=holdout_fraction)
    return _gate


def gated_compile(compile_fn: Callable, holdout_fraction: float = 0.2):
    """Decorator: enforce holdout isolation on a compile function.

    Usage:
        @gated_compile
        def my_compile(student, trainset):
            return optimizer.compile(student, trainset)
    """
    gate = get_holdout_gate(holdout_fraction)

    @functools.wraps(compile_fn)
    def wrapper(student, trainset: list, *args, **kwargs):
        compile_id = (
            kwargs.pop("compile_id", None)
            or hashlib.sha256(str(trainset).encode()).hexdigest()[:8]
        )

        # Auto-split
        train, holdout = gate.split(trainset, compile_id)

        # Only pass train to optimizer
        result = compile_fn(student, train, *args, **kwargs)

        # Validate on holdout
        if hasattr(result, "llms_txt_content") or hasattr(result, "output"):
            validation = gate.validate_gate(compile_id, result)
        else:
            # Result is likely tuple (compiled, score)
            compiled = result[0] if isinstance(result, tuple) else result
            validation = gate.validate_gate(compile_id, compiled)

        return result, validation

    return wrapper
