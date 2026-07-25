"""
Python bridge for Mojo-accelerated SPRT (Phase 2).

Provides a drop-in replacement for the SPRT evaluation loop in
SelfEvolveEngine.validate_and_deploy().  Falls back to pure Python
when the Mojo shared library is unavailable.

Usage:
    from dspytools.core.sprt_mojo_bridge import sprt_evaluate
    result = sprt_evaluate(outcomes, p0=0.50, p1=0.65, alpha=0.05, beta=0.20)
"""

from __future__ import annotations

import math
from types import ModuleType
from typing import Any, Optional

import numpy as np

from dspytools.core.logging_config import get_logger
from dspytools.core.mojo_bridge import try_load_mojo

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Mojo module loading — shared utility
# ---------------------------------------------------------------------------

HAS_MOJO: bool = False
_mojo_module: Optional[ModuleType] = None

HAS_MOJO, _mojo_module = try_load_mojo("sprt", "sprt_evaluate", logger)

# ---------------------------------------------------------------------------
# SPRT return schema (shared by both paths)
# ---------------------------------------------------------------------------

SPRTResult = dict[str, Any]
"""
Accepted result dict keys:
    accepted: bool          – True if H₁ accepted (deploy candidate)
    candidate_score: float  – success rate on evaluated samples
    n_evaluated: int        – number of evaluations performed
    early_stop: bool        – True if SPRT terminated before all samples
    log_likelihood_ratio: float – cumulative log-likelihood ratio
    reason: str             – human-readable termination reason
    statistical_method: str – "SPRT (Mojo)" or "SPRT (Python)"
"""

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def sprt_evaluate(
    outcomes: list[float] | np.ndarray,
    p0: float = 0.50,
    p1: float = 0.65,
    alpha: float = 0.05,
    beta: float = 0.20,
) -> SPRTResult:
    """Run SPRT on an iterable of binary outcomes.

    Args:
        outcomes: Sequence where > 0.5 means "success", ≤ 0.5 means "failure".
        p0: H₀ baseline accuracy (default 0.50).
        p1: H₁ target accuracy (default 0.65).
        alpha: Type I error probability (default 0.05).
        beta: Type II error probability (default 0.20).

    Returns:
        Dict matching SelfEvolveEngine.validate_and_deploy() schema.
    """
    if isinstance(outcomes, np.ndarray):
        arr = outcomes.astype(np.float32, order="C", copy=False)
    else:
        arr = np.asarray(outcomes, dtype=np.float32)

    if arr.ndim != 1:
        raise ValueError(f"outcomes must be 1-D, got {arr.ndim}-D")
    count = arr.size

    if HAS_MOJO and _mojo_module is not None:
        return _mojo_evaluate(arr, count, p0, p1, alpha, beta)
    else:
        return _py_evaluate(arr, count, p0, p1, alpha, beta)


# ---------------------------------------------------------------------------
# Mojo path
# ---------------------------------------------------------------------------


def _mojo_evaluate(
    arr: np.ndarray,
    count: int,
    p0: float,
    p1: float,
    alpha: float,
    beta: float,
) -> SPRTResult:
    """Delegate to the Mojo shared library via pointer passing."""
    if _mojo_module is None or not HAS_MOJO:
        return _py_evaluate(arr, count, p0, p1, alpha, beta)
    src_addr = arr.ctypes.data
    result_pydict = _mojo_module.sprt_evaluate(src_addr, count, p0, p1, alpha, beta)
    return dict(result_pydict)


# ---------------------------------------------------------------------------
# Pure Python fallback (mirrors Mojo logic exactly)
# ---------------------------------------------------------------------------


def _py_evaluate(
    arr: np.ndarray,
    count: int,
    p0: float,
    p1: float,
    alpha: float,
    beta: float,
) -> SPRTResult:
    """Pure Python SPRT — reference implementation."""
    A = math.log((1.0 - beta) / alpha)
    B = math.log(beta / (1.0 - alpha))
    log_ratio_win = math.log(p1 / p0)
    log_ratio_loss = math.log((1.0 - p1) / (1.0 - p0))

    successes = 0
    failures = 0
    log_lr = 0.0

    for i in range(count):
        if arr[i] > 0.5:
            successes += 1
            log_lr += log_ratio_win
        else:
            failures += 1
            log_lr += log_ratio_loss

        observed = successes + failures
        if observed < 3:
            continue
        if log_lr <= B:
            return _build_result(
                accepted=False,
                score=successes / observed,
                n=observed,
                early_stop=True,
                log_lr=log_lr,
                reason=f"SPRT rejected H₁ at n={observed} "
                f"(log_lr={log_lr:.2f} ≤ B={B:.2f})",
            )
        if log_lr >= A:
            return _build_result(
                accepted=True,
                score=successes / observed,
                n=observed,
                early_stop=True,
                log_lr=log_lr,
                reason=f"SPRT accepted H₁ at n={observed} "
                f"(log_lr={log_lr:.2f} ≥ A={A:.2f})",
            )

    total = successes + failures
    final_score = successes / total if total > 0 else 0.0
    return _build_result(
        accepted=final_score > p0,
        score=final_score,
        n=total,
        early_stop=False,
        log_lr=log_lr,
        reason=f"Forced decision after {total} evaluations (no SPRT threshold met)",
    )


def _build_result(
    accepted: bool,
    score: float,
    n: int,
    early_stop: bool,
    log_lr: float,
    reason: str,
) -> SPRTResult:
    """Build the standard SPRT result dict."""
    return {
        "accepted": accepted,
        "candidate_score": score,
        "n_evaluated": n,
        "early_stop": early_stop,
        "log_likelihood_ratio": log_lr,
        "reason": reason,
        "statistical_method": "SPRT (Mojo)" if HAS_MOJO else "SPRT (Python)",
    }


def has_mojo() -> bool:
    """Check whether the Mojo SPRT accelerator is active."""
    return HAS_MOJO
