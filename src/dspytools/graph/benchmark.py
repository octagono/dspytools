#!/usr/bin/env python3
"""
Benchmark suite for Mojo-accelerated vector serialization (Phase 1).

Usage:
    python -m dspytools.graph.benchmark             # quick smoke
    python -m dspytools.graph.benchmark --fuzz       # correctness fuzz
    python -m dspytools.graph.benchmark --bench      # throughput benchmark
    python -m dspytools.graph.benchmark --all        # fuzz + bench
    python -m dspytools.graph.benchmark --ci         # CI mode (fuzz + bench, non-zero exit on fail)

Exit code:
    0 — all checks pass
    1 — correctness failure or regression > 20%
"""

from __future__ import annotations

import argparse
import random
import sys
import time

import numpy as np

from dspytools.graph.cache_mojo_bridge import HAS_MOJO, vec_to_blob

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_SIZES = [1, 3, 7, 64, 128, 384, 512, 1024, 2048, 4096]
"""Vector sizes to test — covers scalar tails, full SIMD, and large payloads."""

_FUZZ_TRIALS = 500
"""Number of random trials in fuzz mode."""

_BENCH_WARMUP = 3
_BENCH_TRIALS = 10
"""Benchmark warmup and measurement iterations."""

_REGRESSION_THRESHOLD = 0.20
"""Fail CI if Mojo is slower than NumPy by more than 20%."""

# ---------------------------------------------------------------------------
# Correctness fuzz
# ---------------------------------------------------------------------------


def fuzz() -> int:
    """Test vec_to_blob against NumPy tobytes for correctness.

    Returns: number of failures (0 = all good).
    """
    failures = 0
    for trial in range(_FUZZ_TRIALS):
        n = random.choice(_SIZES)
        arr = np.random.randn(n).astype(np.float32)

        expected = arr.tobytes()
        actual = vec_to_blob(arr)

        if expected != actual:
            failures += 1
            if failures <= 5:
                print(f"  ✗ MISMATCH n={n} size={arr.nbytes}B (trial {trial})")

    status = "✓" if failures == 0 else "✗"
    print(f"  {status} fuzz: {_FUZZ_TRIALS} trials, {failures} failures")
    return failures


# ---------------------------------------------------------------------------
# Throughput benchmark
# ---------------------------------------------------------------------------


def bench() -> dict[int, float]:
    """Benchmark vec_to_blob throughput across sizes.

    Returns: {n_bytes: mb_per_sec} dict.
    """
    results: dict[int, float] = {}

    for n in _SIZES:
        arr = np.random.randn(n).astype(np.float32)
        nbytes = arr.nbytes

        # Warmup (Mojo may JIT on first call)
        for _ in range(_BENCH_WARMUP):
            _ = vec_to_blob(arr)

        # Measurement
        start = time.perf_counter()
        for _ in range(_BENCH_TRIALS):
            _ = vec_to_blob(arr)
        elapsed = time.perf_counter() - start
        avg_s = elapsed / _BENCH_TRIALS
        mb_per_s = (nbytes / 1_000_000) / avg_s if avg_s > 0 else 0.0
        results[nbytes] = mb_per_s

    return results


# ---------------------------------------------------------------------------
# NumPy reference baseline
# ---------------------------------------------------------------------------


def bench_numpy() -> dict[int, float]:
    """Benchmark numpy.tobytes as a baseline reference."""
    results: dict[int, float] = {}
    for n in _SIZES:
        arr = np.random.randn(n).astype(np.float32)
        nbytes = arr.nbytes

        for _ in range(_BENCH_WARMUP):
            _ = arr.tobytes()

        start = time.perf_counter()
        for _ in range(_BENCH_TRIALS):
            _ = arr.tobytes()
        elapsed = time.perf_counter() - start
        avg_s = elapsed / _BENCH_TRIALS
        mb_per_s = (nbytes / 1_000_000) / avg_s if avg_s > 0 else 0.0
        results[nbytes] = mb_per_s

    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def print_header():
    impl = "Mojo SIMD" if HAS_MOJO else "Python fallback"
    col_label = "Mojo SIMD" if HAS_MOJO else "Bridge (MB/s)"
    print(f"  Implementation: {impl}")
    print()
    print(
        "  {:>8}  {:>10}  {:>12}  {:>12}  {:>10}".format(
            "Size", "Bytes", col_label, "NumPy (MB/s)", "Speedup"
        )
    )
    print("  " + "-" * 64)


def print_row(n: int, nbytes: int, mojo_speed: float, numpy_speed: float):
    if mojo_speed > 0 and numpy_speed > 0:
        speedup = mojo_speed / numpy_speed
        tag = "[FAST]" if speedup > 1.1 else ("[OK]" if speedup > 0.8 else "[SLOW]")
        speedup_str = f"{speedup:.2f}x {tag}"
    else:
        speedup_str = "—"

    print(
        "  {:>8}  {:>10}  {:>12.2f}  {:>12.2f}  {:>10}".format(
            f"n={n}",
            f"{nbytes}B",
            mojo_speed,
            numpy_speed,
            speedup_str,
        )
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Mojo-accelerated vector serialization"
    )
    parser.add_argument("--fuzz", action="store_true", help="Run correctness fuzz")
    parser.add_argument("--bench", action="store_true", help="Run throughput benchmark")
    parser.add_argument(
        "--all",
        action="store_true",
        dest="all_",
        help="Run fuzz + benchmark",
    )
    parser.add_argument(
        "--ci",
        action="store_true",
        help="CI mode — strict checks, non-zero exit on failure",
    )
    args = parser.parse_args()

    # Default: smoke-fuzz and quick bench
    do_fuzz = args.fuzz or args.all_ or args.ci or (not args.bench and not args.all_)
    do_bench = args.bench or args.all_ or args.ci or (not args.fuzz and not args.all_)

    failures = 0
    print("=" * 70)
    print("  Phase 1 — Vector Serialization Benchmark")
    print("=" * 70)
    print()

    # --- Correctness ---
    if do_fuzz:
        print("[Fuzz]")
        failures += fuzz()
        print()

    # --- Throughput ---
    if do_bench:
        print("[Throughput]")
        print_header()
        mojo_results = bench()
        numpy_results = bench_numpy()

        for n in _SIZES:
            nbytes = n * 4
            print_row(n, nbytes, mojo_results[nbytes], numpy_results[nbytes])

        print()

        # Regression check for CI
        if args.ci and HAS_MOJO:
            regressed = False
            for n in _SIZES:
                nbytes = n * 4
                mojo = mojo_results.get(nbytes, 0)
                numpy_ref = numpy_results.get(nbytes, 1)
                if numpy_ref > 0 and mojo / numpy_ref < (1.0 - _REGRESSION_THRESHOLD):
                    print(
                        f"  ⚠  REGRESSION at n={n}: "
                        f"Mojo={mojo:.2f} MB/s vs NumPy={numpy_ref:.2f} MB/s "
                        f"(speedup={mojo / numpy_ref:.2f}x below "
                        f"{1 - _REGRESSION_THRESHOLD:.0%} threshold)"
                    )
                    regressed = True
                    failures += 1
            if not regressed:
                print("  ✓ No regressions detected")

    # Final summary
    print()
    if failures == 0:
        print("✅ All checks passed")
    else:
        print(f"❌ {failures} failure(s) detected")
    print()

    sys.exit(1 if failures > 0 else 0)


if __name__ == "__main__":
    main()
