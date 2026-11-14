"""Shared DSPy metric functions — Single Source of Truth for scoring.

All compile commands, GFL pipeline, and generate module use these metrics.
Consolidates auto_metric (evolve/metrics.py), llms_txt_quality (generate/quality.py),
and _calc_score (gfl/feedback.py) into one canonical implementation.
"""

from __future__ import annotations

import re
import string
from typing import Any, Callable

from dspytools.core._dspy import dspy
from dspytools.core.loaders import prediction_output

# Characters stripped from the ends of a normalized answer: ASCII punctuation plus
# the typographic quotes/dashes small models emit around short answers.
_EDGE_CHARS = (
    string.punctuation + "\u2018\u2019\u201c\u201d\u00ab\u00bb\u2013\u2014\u2026"
)
_WS_RE = re.compile(r"\s+")
# Feedback quoting budget: GEPA reflection prompts must stay small.
_FEEDBACK_VALUE_CHARS = 200


def normalize_text(text: Any) -> str:
    """Normalize an answer for comparison (diving-deeper/metrics-and-evaluation §6).

    Strip surrounding whitespace, casefold, collapse internal whitespace, then
    strip surrounding punctuation/quotes so ``"Paris."`` and ``' "paris" '``
    compare equal.
    """
    if text is None:
        return ""
    normalized = _WS_RE.sub(" ", str(text).strip().casefold())
    normalized = normalized.strip(_EDGE_CHARS + " ")
    return _WS_RE.sub(" ", normalized).strip()


def answer_exact_match(gold: Any, pred: Any) -> bool:
    """True when both answers normalize to the same text.

    Symmetric by design: two absent answers (``""`` vs ``None``) compare equal,
    matching DSPy's ``pred == gold`` semantics for degenerate examples.
    """
    return normalize_text(gold) == normalize_text(pred)


def answer_overlap_score(gold: Any, pred: Any) -> float:
    """Fractional correctness score in ``[0.0, 1.0]`` for a free-text answer.

    Normalized equality scores ``1.0``; otherwise the score is the fraction of
    gold tokens covered by the prediction, capped at ``0.99`` so a non-exact
    answer is never mistaken for a perfect one (GEPA's ``skip_perfect_score``
    would otherwise drop it from reflection).
    """
    gold_n = normalize_text(gold)
    pred_n = normalize_text(pred)
    if not gold_n:
        return 0.0
    if gold_n == pred_n:
        return 1.0
    if not pred_n:
        return 0.0
    gold_tokens = gold_n.split()
    pred_tokens = set(pred_n.split())
    if not gold_tokens or not pred_tokens:
        return 0.0
    covered = sum(1 for token in gold_tokens if token in pred_tokens)
    return round(min(0.99, covered / len(gold_tokens)), 4)


def exact_match_metric(val_field: str = "output"):
    """Factory: returns a DSPy-compatible metric for normalized exact match.

    Both sides are normalized (whitespace, case, surrounding punctuation) before
    comparison, so ``"Paris."`` vs ``"Paris"`` scores 1.0 instead of 0.0.

    Args:
        val_field: Field name on example/prediction to compare (default "output")

    Returns:
        A function with signature (example, prediction, trace) -> float
    """

    def _metric(example, prediction, trace=None):
        pred = getattr(prediction, val_field, "")
        gold = getattr(example, val_field, "")
        return 1.0 if answer_exact_match(gold, pred) else 0.0

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


def _truncate(text: str) -> str:
    """Shorten a value for GEPA feedback so reflections stay within budget."""
    if len(text) <= _FEEDBACK_VALUE_CHARS:
        return text
    return text[:_FEEDBACK_VALUE_CHARS] + "…"


def _gepa_feedback(
    *,
    field: str,
    gold_text: str,
    pred_text: str,
    score: float,
    missing_field: bool,
    pred: Any,
    component: Any,
) -> str:
    """Build short, specific reflection feedback for one GEPA example."""
    target = f"[{component}] " if component else ""
    if missing_field:
        return (
            f"{target}prediction is missing the '{field}' field "
            f"(got fields: {', '.join(sorted(getattr(pred, '_output_field_names', []) or [])) or 'none'}). "
            f"Expected {_truncate(gold_text)!r}. score={score:.2f}"
        )
    if not pred_text.strip():
        return (
            f"{target}'{field}' was empty. Expected {_truncate(gold_text)!r}. "
            f"score={score:.2f}"
        )
    if score >= 1.0:
        return f"{target}correct: {_truncate(pred_text)!r}. score=1.00"
    return (
        f"{target}expected {_truncate(gold_text)!r} but got {_truncate(pred_text)!r} "
        f"(gold has {len(normalize_text(gold_text).split())} token(s), "
        f"prediction covers {answer_overlap_score(gold_text, pred_text):.2f}). "
        f"score={score:.2f}"
    )


def gepa_metric(
    gold: Any,
    pred: Any,
    trace: Any = None,
    pred_name: Any = None,
    pred_trace: Any = None,
    *,
    val_field: str = "output",
) -> Any:
    """GEPA-compatible metric returning ``dspy.Prediction(score, feedback)``.

    Scores correctness against ``gold`` (normalized exact match, fractional
    token-coverage credit otherwise) and returns the evidence GEPA's reflection
    LM needs: the missing field, the gold-vs-prediction delta, and the resulting
    score. A bare float would leave the teacher optimizing markdown shape
    because it only sees "score of N" (diving-deeper/gepa-in-depth §1).

    The score is always in ``[0.0, 1.0]``. Callers that feed a generic evaluator
    which sums metric outputs (``dspy.Evaluate``, and therefore
    ``dspy.BetterTogether``) must use :func:`gepa_metric_float` instead —
    ``Prediction`` objects are not summable.
    """
    gold_raw = getattr(gold, val_field, None)
    pred_raw = getattr(pred, val_field, None)
    gold_text = "" if gold_raw is None else str(gold_raw)
    pred_text = "" if pred_raw is None else str(pred_raw)

    score = answer_overlap_score(gold_text, pred_text)
    feedback = _gepa_feedback(
        field=val_field,
        gold_text=gold_text,
        pred_text=pred_text,
        score=score,
        missing_field=pred_raw is None and gold_raw is not None,
        pred=pred,
        component=pred_name,
    )
    return dspy.Prediction(score=score, feedback=feedback)


def gepa_metric_float(
    gold: Any,
    pred: Any,
    trace: Any = None,
    pred_name: Any = None,
    pred_trace: Any = None,
    *,
    val_field: str = "output",
) -> float:
    """Plain-float fallback for :func:`gepa_metric`.

    Use wherever the metric output is consumed numerically instead of by GEPA's
    reflection loop (``dspy.Evaluate`` / ``dspy.BetterTogether`` sum metric
    returns, so they cannot accept a ``dspy.Prediction``).
    """
    gold_raw = getattr(gold, val_field, None)
    pred_raw = getattr(pred, val_field, None)
    return answer_overlap_score(
        "" if gold_raw is None else str(gold_raw),
        "" if pred_raw is None else str(pred_raw),
    )


def gepa_metric_cost_aware(
    val_field: str = "output", lm_call_penalty: float = 0.01
) -> Callable[..., float]:
    """Factory: cost-aware GEPA metric for Flex optimization.

    Enables cost-aware Flex program optimization via GEPA's ``program_trace``
    parameter (populated at scoring time for Flex submodules). The returned
    metric scores prediction quality via :func:`auto_metric` and subtracts a
    penalty for each LM call recorded in ``program_trace``, so GEPA favours
    programs that achieve the same quality with fewer LM calls.

    The returned callable follows the 6-arg GEPA metric protocol:
    ``(gold, pred, trace, pred_name, pred_trace, program_trace)``.

    Args:
        val_field: Field name on the prediction to extract the output from.
        lm_call_penalty: Score subtracted per LM call in ``program_trace``.

    Returns:
        A metric function returning a float clamped to ``[0.0, 1.0]``.
    """

    def _metric(
        gold: Any,
        pred: Any,
        trace: Any = None,
        pred_name: Any = None,
        pred_trace: Any = None,
        program_trace: Any = None,
    ) -> float:
        output = prediction_output(pred, field=val_field)
        score = auto_metric(str(output))
        if program_trace is not None:
            score -= lm_call_penalty * len(program_trace)
        return max(0.0, min(1.0, score))

    return _metric


def simple_metric(example: Any, prediction: Any, trace: Any = None) -> float:
    """Simple metric for BootstrapFewShot/MIPROv2 — extracts prediction, delegates to auto_metric."""
    output = prediction_output(prediction)
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
