"""Self-evolving metrics — auto-generate evaluation functions.

The SelfEvolve system creates DSPy metrics on the fly based on task
requirements, then optimizes them using DSPy optimizers.

All scoring is delegated to core/metrics.py (SSOT).
"""

from __future__ import annotations

from dspytools.core.metrics import (
    auto_metric,
    content_quality_score,
    gepa_metric,
    simple_metric,
)

__all__ = ["auto_metric", "content_quality_score", "gepa_metric", "simple_metric"]
