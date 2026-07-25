# mojo/bm25.mojo — BM25 scoring kernel
#
# Exposes a single function to Python:
#   bm25_score_docs(query_tf_addr, idf_addr, doc_lens_addr, avg_doc_len,
#       dims, k1, b, scores_out_addr) -> None
#
# Computes BM25 scores for all documents against a query using ctypes
# pointer access. Each document's score is the sum of per-term BM25
# contributions. The output buffer is pre-allocated by Python.
#
# dims is passed as a Python tuple (n_docs, n_terms) since Mojo 1.0.0b2
# def_function supports at most 8 PythonObject parameters.
#
# Phase 3 of the Mojo hybrid architecture — see mojo/README.md

from std.python import PythonObject, Python
from std.python.bindings import PythonModuleBuilder


def bm25_score_docs(
    query_tf_addr: PythonObject,
    idf_addr: PythonObject,
    doc_lens_addr: PythonObject,
    avg_doc_len: PythonObject,
    dims: PythonObject,          # Python tuple (n_docs, n_terms)
    k1: PythonObject,
    b: PythonObject,
    scores_out_addr: PythonObject,
) raises -> PythonObject:
    """Compute BM25 scores for all documents against a query.

    Args:
        query_tf_addr: Raw address of Float32[flattened n_docs × n_terms] TF matrix.
        idf_addr: Float32 IDF array (length n_terms).
        doc_lens_addr: Float32 document lengths (length n_docs).
        avg_doc_len: Average document length.
        dims: Python tuple (n_docs, n_terms).
        k1: BM25 k1 parameter.
        b: BM25 b parameter.
        scores_out_addr: Float32 output buffer (length n_docs).

    BM25 formula per (doc, term):
        idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * doc_len / avg_doc_len))
    """
    var ctypes = Python.import_module("ctypes")
    var tf_mat = ctypes.cast(query_tf_addr, ctypes.POINTER(ctypes.c_float))
    var idf_vals = ctypes.cast(idf_addr, ctypes.POINTER(ctypes.c_float))
    var doc_lens = ctypes.cast(doc_lens_addr, ctypes.POINTER(ctypes.c_float))
    var scores_out = ctypes.cast(scores_out_addr, ctypes.POINTER(ctypes.c_float))

    var nd = Int(py=dims[0])
    var nt = Int(py=dims[1])
    var k1_val = Float64(py=k1)
    var b_val = Float64(py=b)
    var avg_len = Float64(py=avg_doc_len)

    if nd == 0 or nt == 0:
        return PythonObject(None)

    var k1_plus_1 = k1_val + 1.0
    var inv_avg_len = 1.0 / avg_len

    # Process each document
    for doc_idx in range(nd):
        var score: Float64 = 0.0
        var doc_len = Float64(py=doc_lens[doc_idx])
        var norm_factor = 1.0 - b_val + b_val * doc_len * inv_avg_len

        # Sum BM25 over all query terms for this document
        var tf_row_start = doc_idx * nt
        for term_idx in range(nt):
            var tf_val = Float64(py=tf_mat[tf_row_start + term_idx])
            if tf_val == 0.0:
                continue
            var idf_val = Float64(py=idf_vals[term_idx])
            score += idf_val * (tf_val * k1_plus_1) / (tf_val + k1_val * norm_factor)

        scores_out[doc_idx] = ctypes.c_float(score)

    return PythonObject(None)


@export
def PyInit_bm25() abi("C") -> PythonObject:
    """Initialize the native Python module."""
    try:
        var m = PythonModuleBuilder("bm25")
        m.def_function[bm25_score_docs]("bm25_score_docs")
        return m.finalize()
    except e:
        return PythonObject(None)
