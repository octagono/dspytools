"""
Python bridge for Mojo-accelerated BM25 scoring (Phase 3).

Provides the inner BM25 scoring loop in Mojo for vectorized throughput.
Falls back to pure Python when the Mojo shared library is unavailable.

Usage:
    from dspytools.skills.bm25_mojo_bridge import score_documents
    scores = score_documents(query_tf_matrix, idf_values, doc_lengths,
                             avg_doc_len, k1=1.2, b=0.75)
"""

from __future__ import annotations

from types import ModuleType
from typing import Optional

import numpy as np

from dspytools.core.logging_config import get_logger
from dspytools.core.mojo_bridge import try_load_mojo

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Mojo module loading — shared utility
# ---------------------------------------------------------------------------

HAS_MOJO: bool
_mojo_module: Optional[ModuleType] = None

HAS_MOJO, _mojo_module = try_load_mojo("bm25", "bm25_score_docs", logger)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def score_documents(
    query_tf_matrix: list[list[float]] | np.ndarray,
    idf_values: list[float] | np.ndarray,
    doc_lengths: list[float] | np.ndarray,
    avg_doc_len: float,
    k1: float = 1.2,
    b: float = 0.75,
) -> np.ndarray:
    """Compute BM25 scores for all documents against a query.

    Args:
        query_tf_matrix: Shape (n_docs, n_terms) — TF of each query term
                         in each document.
        idf_values: Shape (n_terms,) — IDF of each query term.
        doc_lengths: Shape (n_docs,) — length of each document.
        avg_doc_len: Average document length across the corpus.
        k1: BM25 k1 parameter (default 1.2).
        b: BM25 b parameter (default 0.75).

    Returns:
        Float32 numpy array of shape (n_docs,) with BM25 scores.
    """
    # Convert inputs to contiguous Float32 arrays
    tf_arr = np.asarray(query_tf_matrix, dtype=np.float32, order="C")
    idf_arr = np.asarray(idf_values, dtype=np.float32, order="C")
    len_arr = np.asarray(doc_lengths, dtype=np.float32, order="C")

    if tf_arr.ndim != 2:
        raise ValueError(f"query_tf_matrix must be 2-D, got {tf_arr.ndim}-D")

    n_docs, n_terms = tf_arr.shape

    if idf_arr.ndim != 1 or idf_arr.size != n_terms:
        raise ValueError(
            f"idf_values must be 1-D with {n_terms} elements, got {idf_arr.shape}"
        )
    if len_arr.ndim != 1 or len_arr.size != n_docs:
        raise ValueError(
            f"doc_lengths must be 1-D with {n_docs} elements, got {len_arr.shape}"
        )

    if n_docs == 0 or n_terms == 0:
        return np.zeros(n_docs, dtype=np.float32)

    # Pre-allocate output buffer
    scores_out = np.empty(n_docs, dtype=np.float32, order="C")

    if HAS_MOJO and _mojo_module is not None:
        _mojo_module.bm25_score_docs(
            tf_arr.ctypes.data,
            idf_arr.ctypes.data,
            len_arr.ctypes.data,
            float(avg_doc_len),
            (n_docs, n_terms),  # dims tuple (Mojo supports ≤8 params)
            float(k1),
            float(b),
            scores_out.ctypes.data,
        )
    else:
        # Pure Python fallback (vectorized inner loop)
        k1_f = float(k1)
        b_f = float(b)
        k1_plus_1 = k1_f + 1.0
        inv_avg_len = 1.0 / avg_doc_len

        for doc_idx in range(n_docs):
            doc_len = float(len_arr[doc_idx])
            norm_factor = 1.0 - b_f + b_f * doc_len * inv_avg_len
            score = 0.0
            tf_row_start = doc_idx * n_terms
            for term_idx in range(n_terms):
                tf_val = float(tf_arr.flat[tf_row_start + term_idx])
                if tf_val == 0.0:
                    continue
                score += float(idf_arr[term_idx]) * (
                    (tf_val * k1_plus_1) / (tf_val + k1_f * norm_factor)
                )
            scores_out[doc_idx] = score

    return scores_out


def has_mojo() -> bool:
    """Check whether the Mojo BM25 accelerator is active."""
    return HAS_MOJO
