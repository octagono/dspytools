"""Rich textual feedback generation — GEPA-compatible diagnostics.

Optimization 18: Cached generate_feedback — identical predictions return
cached results without re-scoring. Uses content hash as cache key.

GFL Stage: Evaluate → Feedback

Converts scalar scores into structured diagnostic feedback for GEPA's reflection_lm.
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict

from dspytools.core.dspy_modules import get_rich_feedback_generator

_feedback_cache: OrderedDict[str, dict] = OrderedDict()
_CACHE_MAX = 256


def generate_feedback(prediction: str, gold: str | None = None) -> dict:
    """Generate rich textual feedback from prediction and optional gold answer.

    Optimization 18: Caches results for identical prediction strings.
    Gold answers are not cached (rarely used, and would complicate the key).

    Returns: {"score": float, "feedback": str}
    """
    # Cache only when no gold answer (gold path is rare and would complicate the key)
    cache_key = ""
    use_cache = gold is None
    if use_cache:
        cache_key = hashlib.md5(prediction.encode(), usedforsecurity=False).hexdigest()
        if cache_key in _feedback_cache:
            _feedback_cache.move_to_end(cache_key)
            return _feedback_cache[cache_key]

    # Use DSPy module for rich feedback
    generator = get_rich_feedback_generator()
    result = generator(prediction=prediction, gold=gold or "")
    feedback_result = {"score": float(result.score), "feedback": result.feedback}

    if use_cache:
        _feedback_cache[cache_key] = feedback_result
        if len(_feedback_cache) > _CACHE_MAX:
            _feedback_cache.popitem(last=False)

    return feedback_result
