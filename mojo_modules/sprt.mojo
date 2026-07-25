# mojo/sprt.mojo — Sequential Probability Ratio Test
#
# Exposes a single function to Python:
#   sprt_evaluate(outcomes_addr, count, p0, p1, alpha, beta) -> dict
#
# Runs SPRT on an array of binary outcomes with early stopping.
# Accepts Float32 outcome array via ctypes pointer + integer count.
# Returns Python dict with accepted, candidate_score, n_evaluated, etc.
#
# Phase 2 of the Mojo hybrid architecture — see mojo/README.md

from std.python import PythonObject, Python
from std.python.bindings import PythonModuleBuilder
from std import math


def sprt_evaluate(
    outcomes_addr: PythonObject,
    count: PythonObject,
    p0: PythonObject,
    p1: PythonObject,
    alpha: PythonObject,
    beta: PythonObject,
) raises -> PythonObject:
    """Run SPRT on an array of binary outcomes with early stopping.

    Accepts H₁ (program is better) when cumulative evidence crosses the
    upper threshold A. Accepts H₀ (program is NOT better) when it crosses
    the lower threshold B. Early termination saves API tokens.

    Returns a Python dict compatible with SelfEvolveEngine.validate_and_deploy().
    """
    var ctypes = Python.import_module("ctypes")
    var outcomes = ctypes.cast(outcomes_addr, ctypes.POINTER(ctypes.c_float))
    var n_total = Int(py=count)
    var p0_val = Float64(py=p0)
    var p1_val = Float64(py=p1)
    var alpha_val = Float64(py=alpha)
    var beta_val = Float64(py=beta)

    # SPRT thresholds
    var A = math.log((1.0 - beta_val) / alpha_val)   # upper bound → accept H₁
    var B = math.log(beta_val / (1.0 - alpha_val))    # lower bound → accept H₀

    # Precompute log ratios
    var log_ratio_win = math.log(p1_val / p0_val)
    var log_ratio_loss = math.log((1.0 - p1_val) / (1.0 - p0_val))

    var successes: Int = 0
    var log_lr: Float64 = 0.0

    for i in range(n_total):
        var outcome = outcomes[i]
        var is_win = outcome > 0.5

        if is_win:
            successes += 1
            log_lr += log_ratio_win
        else:
            log_lr += log_ratio_loss

        # SPRT early stopping — check after at least 3 observations
        var observed = i + 1
        if observed < 3:
            continue

        if log_lr <= B:
            # Accept H₀: candidate is NOT better than baseline
            return _make_result(
                accepted=False,
                score=Float64(successes) / Float64(observed),
                n=observed,
                early_stop=True,
                log_lr=log_lr,
                reason=(
                    "SPRT rejected H₁ at n=" + String(observed) +
                    " (log_lr=" + String(log_lr) + " ≤ B=" + String(B) + ")"
                ),
            )
        elif log_lr >= A:
            # Accept H₁: candidate IS better than baseline
            return _make_result(
                accepted=True,
                score=Float64(successes) / Float64(observed),
                n=observed,
                early_stop=True,
                log_lr=log_lr,
                reason=(
                    "SPRT accepted H₁ at n=" + String(observed) +
                    " (log_lr=" + String(log_lr) + " ≥ A=" + String(A) + ")"
                ),
            )

    # Forced decision: no SPRT boundary crossed within count evaluations
    var final_score = Float64(successes) / Float64(n_total) if n_total > 0 else 0.0
    return _make_result(
        accepted=final_score > p0_val,
        score=final_score,
        n=n_total,
        early_stop=False,
        log_lr=log_lr,
        reason="Forced decision after " + String(n_total) + " evaluations (no SPRT threshold met)",
    )


def _make_result(
    accepted: Bool,
    score: Float64,
    n: Int,
    early_stop: Bool,
    log_lr: Float64,
    reason: String,
) raises -> PythonObject:
    """Build a Python dict return value matching SelfEvolveEngine schema."""
    var dict_type = Python.evaluate("dict")
    var result = dict_type()
    var py_bool = Python.evaluate("bool")

    result["accepted"] = py_bool(accepted)
    result["candidate_score"] = PythonObject(score)
    result["n_evaluated"] = PythonObject(n)
    result["early_stop"] = py_bool(early_stop)
    result["log_likelihood_ratio"] = PythonObject(log_lr)
    result["reason"] = PythonObject(reason)
    result["statistical_method"] = PythonObject("SPRT (Mojo)")

    return result


@export
def PyInit_sprt() abi("C") -> PythonObject:
    """Initialize the native Python module."""
    try:
        var m = PythonModuleBuilder("sprt")
        m.def_function[sprt_evaluate]("sprt_evaluate")
        return m.finalize()
    except e:
        return PythonObject(None)
