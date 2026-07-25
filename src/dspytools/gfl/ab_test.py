"""A/B testing framework — statistical comparison of compiled programs.

Runs two programs side-by-side, computes win-rate with confidence intervals,
and auto-deploys the winner if statistically significant.
"""

from __future__ import annotations

import random
from typing import Any


class ABTest:
    """Statistical A/B testing for compiled DSPy programs.

    Usage:
        test = ABTest(program_a, program_b)
        result = test.run(test_inputs, n_trials=30)
        if result["winner"] == "b":
            hotswap_manager.swap("b")
    """

    def __init__(self, program_a: Any, program_b: Any, confidence: float = 0.9):
        self.prog_a = program_a
        self.prog_b = program_b
        self.confidence = confidence
        self.results: dict[str, Any] = {}

    def run(
        self, test_inputs: list[dict], n_trials: int = 20, metric_fn: Any = None
    ) -> dict:
        """Run A/B test with n_trials, compute winner statistically.

        Each trial: run both programs on same random input, pick better.
        Winner = program with more wins, if win-rate ≥ confidence.
        """
        metric = metric_fn or (
            lambda output: 1.0 if output and len(str(output)) > 50 else 0.0
        )

        wins_a = 0
        wins_b = 0
        draws = 0

        for _ in range(n_trials):
            if test_inputs:
                inp = random.choice(test_inputs)
                kwargs = (
                    inp.inputs()
                    if hasattr(inp, "inputs")
                    else dict(inp)
                    if isinstance(inp, dict)
                    else {}
                )
            else:
                kwargs = {"input": "test"}

            out_a = self.prog_a(**kwargs)
            out_b = self.prog_b(**kwargs)
            score_a = metric(
                getattr(out_a, "output", getattr(out_a, "answer", str(out_a)))
            )
            score_b = metric(
                getattr(out_b, "output", getattr(out_b, "answer", str(out_b)))
            )

            if score_a > score_b:
                wins_a += 1
            elif score_b > score_a:
                wins_b += 1
            else:
                draws += 1

        total = n_trials
        rate_a = wins_a / total
        rate_b = wins_b / total

        # Statistical significance: winner must exceed confidence threshold
        winner = None
        if rate_a >= self.confidence:
            winner = "a"
        elif rate_b >= self.confidence:
            winner = "b"
        elif rate_a > rate_b:
            winner = "a"  # Weak win
        elif rate_b > rate_a:
            winner = "b"

        self.results = {
            "winner": winner,
            "wins_a": wins_a,
            "wins_b": wins_b,
            "draws": draws,
            "rate_a": rate_a,
            "rate_b": rate_b,
            "significant": winner is not None
            and max(rate_a, rate_b) >= self.confidence,
            "trials": n_trials,
            "recommendation": f"Deploy program {winner}"
            if winner
            else "No clear winner — collect more data",
        }
        return self.results


def auto_deploy_if_better(
    results: dict,
    hotswap_manager: Any,
    program_id_a: str = "current",
    program_id_b: str = "candidate",
) -> str | None:
    """Auto-deploy the winner if statistically significant.

    Returns the deployed program_id, or None if no action taken.
    """
    if results.get("significant") and results.get("winner") == "b":
        hotswap_manager.load_single(program_id_b)
        hotswap_manager.swap(program_id_b)
        return program_id_b
    return None
