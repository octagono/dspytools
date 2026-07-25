"""Shared DSPy metric functions — Single Source of Truth for scoring.

All compile commands, GFL pipeline, and generate module use these metrics.
Consolidates auto_metric (evolve/metrics.py), llms_txt_quality (generate/quality.py),
and _calc_score (gfl/feedback.py) into one canonical implementation.
"""

from __future__ import annotations

from typing import Any


def exact_match_metric(val_field: str = "output"):
    """Factory: returns a DSPy-compatible metric for exact string match.

    Args:
        val_field: Field name on example/prediction to compare (default "output")

    Returns:
        A function with signature (example, prediction, trace) -> float
    """

    def _metric(example, prediction, trace=None):
        pred = getattr(prediction, val_field, "")
        gold = getattr(example, val_field, "")
        return 1.0 if pred == gold else 0.0

    return _metric


def content_quality_score(
    content: str,
    *,
    target_format: str = "markdown",
    length_sweet_spot: tuple[int, int] = (300, 5000),
    long_penalty_threshold: int = 10000,
) -> float:
    """Score content quality 0.0–1.0 based on structure and format.

    SSOT for all heuristic quality scoring across the project:
    - evolve/metrics.py auto_metric()
    - generate/quality.py llms_txt_quality()
    - gfl/feedback.py _calc_score()

    Scoring:
        Empty/short (<50 chars): 0.0
        JSON echo (model parroting): 0.0
        Markdown heading (#):     +0.15
        Markdown sections (##):   +0.15
        Bullet lists:             +0.10
        Bold emphasis:            +0.05
        Code blocks:              +0.10
        Length in sweet spot:     +0.20  (configurable)
        Length > threshold:       -0.10
        Full output code-wrapped: -0.15
    """
    if not content or len(content.strip()) < 50:
        return 0.0

    stripped = content.strip()
    score = 0.0

    # Penalize JSON echo (model parroting input fields)
    if stripped.startswith("{") and '"repo_url"' in stripped:
        return 0.0

    if target_format == "markdown":
        if stripped.startswith("#"):
            score += 0.15
        if "\n## " in stripped:
            score += 0.15
        if "\n- " in stripped or "\n* " in stripped:
            score += 0.10
        if "**" in stripped:
            score += 0.05

    # Code blocks
    if "```" in stripped:
        score += 0.10

    # Length adequacy
    length = len(stripped)
    lo, hi = length_sweet_spot
    if lo <= length <= hi:
        score += 0.20
    elif length > long_penalty_threshold:
        score -= 0.10

    # Penalize entire output wrapped in code block
    if stripped.startswith("```") and stripped.rstrip().endswith("```"):
        score -= 0.15

    return max(0.0, min(1.0, score))


def content_quality_score_dspy(
    content: str,
    *,
    target_format: str = "markdown",
) -> float:
    """DSPy-based content quality scoring.

    Uses a learned DSPy ChainOfThought module for scoring.
    Falls back to heuristic scoring if LM is unavailable.
    """
    from dspytools.core.dspy_modules import get_content_scorer

    scorer = get_content_scorer()
    result = scorer(content=content, target_format=target_format)
    return float(result.score)


def auto_metric(content: str, target_format: str = "markdown") -> float:
    """Heuristic quality metric — SSOT wrapper around content_quality_score.

    Backward-compatible signature for evolve/metrics.py callers.
    """
    return content_quality_score(content, target_format=target_format)


def gepa_metric(
    gold: Any,
    pred: Any,
    trace: Any = None,
    pred_name: Any = None,
    pred_trace: Any = None,
    *,
    val_field: str = "output",
) -> float:
    """GEPA-compatible metric — extracts prediction answer, delegates to auto_metric."""
    output = getattr(
        pred, val_field, getattr(pred, "answer", getattr(pred, "output", str(pred)))
    )
    return auto_metric(str(output))


def simple_metric(example: Any, prediction: Any, trace: Any = None) -> float:
    """Simple metric for BootstrapFewShot/MIPROv2 — extracts prediction, delegates to auto_metric."""
    output = getattr(
        prediction, "answer", getattr(prediction, "output", str(prediction))
    )
    return auto_metric(str(output))


def llms_txt_quality(content: str) -> float:
    """llms.txt quality scoring — SSOT wrapper with generate-specific length tiers.

    Uses content_quality_score with tighter sweet spot for documentation.
    """
    # Documentation-specific: 300–5000 is sweet, 10000+ penalized more
    return content_quality_score(
        content,
        target_format="markdown",
        length_sweet_spot=(300, 5000),
        long_penalty_threshold=10000,
    )


def llms_txt_metric(example: Any, prediction: Any, trace: Any = None) -> float:
    """DSPy metric for llms.txt quality — wraps llms_txt_quality()."""
    _ = trace
    return llms_txt_quality(prediction.llms_txt_content)
