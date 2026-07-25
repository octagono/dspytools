"""Tests for SandboxPool — resource limits, pool lifecycle, and output guardrails.

Runs real subprocess workers but with short timeouts and small limits.
"""

from __future__ import annotations

import threading

import pytest


class TestSandboxPoolConstruction:
    """SandboxPool creates warm workers on init."""

    def test_empty_pool_size_zero(self):
        """pool_size=0 creates no workers."""
        from dspytools.generate.module import SandboxPool

        pool = SandboxPool(pool_size=0)
        assert len(pool._workers) == 0
        pool.shutdown()

    def test_pool_creates_warm_workers(self):
        """Default pool_size=2 creates 2 warm workers."""
        from dspytools.generate.module import SandboxPool

        pool = SandboxPool(pool_size=2)
        assert len(pool._workers) == 2
        alive = sum(1 for p in pool._workers if p.poll() is None)
        assert alive == 2
        pool.shutdown()

    def test_pool_size_five(self):
        """pool_size=5 creates 5 warm workers."""
        from dspytools.generate.module import SandboxPool

        pool = SandboxPool(pool_size=5)
        assert len(pool._workers) == 5
        pool.shutdown()


class TestSandboxPoolExecute:
    """Executing code on warm workers."""

    @pytest.fixture(autouse=True)
    def _pool(self):
        from dspytools.generate.module import SandboxPool

        pool = SandboxPool(pool_size=2, timeout=5.0)
        yield pool
        pool.shutdown()

    def test_simple_expression(self, _pool):
        """Simple Python expression returns correct output."""
        result = _pool.execute("print('hello world')")
        assert result["success"] is True
        assert "hello world" in result["output"]

    def test_multi_line_code(self, _pool):
        """Multi-line code executes correctly."""
        code = """x = 1 + 2
y = x * 3
print(y)"""
        result = _pool.execute(code)
        assert result["success"] is True
        assert "9" in result["output"]

    def test_error_handling(self, _pool):
        """Python runtime error is caught and returned."""
        result = _pool.execute("raise ValueError('boom')")
        assert result["success"] is True  # worker catches exceptions
        assert "boom" in result["output"]

    def test_worker_reuse(self, _pool):
        """Same worker is reused across calls."""
        r1 = _pool.execute("print('first')")
        r2 = _pool.execute("print('second')")
        assert r1["success"] is True
        assert r2["success"] is True
        assert r1["worker_reused"] is True
        assert r2["worker_reused"] is True

    def test_concurrent_execution(self, _pool):
        """Two parallel executions on different workers."""
        results: list[dict] = []
        errors: list[Exception] = []

        def run():
            try:
                results.append(_pool.execute("print('concurrent')"))
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=run)
        t2 = threading.Thread(target=run)
        t1.start()
        t2.start()
        t1.join(timeout=5)
        t2.join(timeout=5)

        assert len(errors) == 0
        assert sum(1 for r in results if r["success"]) == 2

    def test_state_across_calls(self, _pool):
        """Worker maintains state between calls."""
        _pool.execute("x = 42")
        result = _pool.execute("print(x)")
        assert result["success"] is True
        assert "42" in result["output"]


class TestSandboxPoolFallback:
    """Pool exhaustion triggers one-shot fallback."""

    def test_fallback_on_exhaustion(self):
        """When all workers are in use, fallback is used."""
        from dspytools.generate.module import SandboxPool

        pool = SandboxPool(pool_size=1, timeout=5.0)
        # Acquire the only worker
        idx, proc = pool._acquire()
        assert idx is not None
        # Second call should fall back
        result = pool.execute("print('fallback')")
        assert result["success"] is True
        assert result["worker_reused"] is False
        # Release
        pool._release(idx)
        pool.shutdown()

    def test_fallback_timeout(self):
        """Fallback handles timeout gracefully."""
        from dspytools.generate.module import SandboxPool

        pool = SandboxPool(pool_size=0, timeout=0.1)
        result = pool.execute("import time; time.sleep(10)")
        assert result["success"] is False
        assert "Timeout" in result.get("error", "")
        pool.shutdown()


class TestSandboxPoolOutputGuardrail:
    """max_output_size prevents runaway output."""

    def test_output_limit_enforced(self):
        """Output exceeding max_output_size is truncated with error."""
        from dspytools.generate.module import SandboxPool

        pool = SandboxPool(pool_size=1, timeout=5.0, max_output_size=100)
        code = "for i in range(1000): print('x' * 80)"
        result = pool.execute(code)
        assert result["success"] is False
        assert "exceeded" in result["error"].lower()
        pool.shutdown()

    def test_output_limit_not_reached(self):
        """Small output passes without error."""
        from dspytools.generate.module import SandboxPool

        pool = SandboxPool(pool_size=1, timeout=5.0, max_output_size=10000)
        result = pool.execute("print('small')")
        assert result["success"] is True
        pool.shutdown()

    def test_fallback_output_limit(self):
        """Fallback also enforces output limit."""
        from dspytools.generate.module import SandboxPool

        pool = SandboxPool(pool_size=0, timeout=5.0, max_output_size=10)
        result = pool.execute("print('x' * 100)")
        assert result["success"] is False
        assert "exceeded" in result["error"].lower()
        pool.shutdown()


class TestSandboxPoolRecycle:
    """Worker recycling when max_reuse exceeded."""

    def test_recycle_on_max_reuse(self):
        """Worker exceeding max_reuse is recycled."""
        from dspytools.generate.module import SandboxPool

        pool = SandboxPool(pool_size=1, timeout=5.0, max_reuse=2)
        # Use the worker 3 times (triggers recycle on 3rd)
        for i in range(3):
            result = pool.execute(f"print({i})")
            assert result["success"] is True
        # Worker should have been recycled at least once
        assert pool.stats["total_recycled"] >= 1
        pool.shutdown()

    def test_reuse_count_tracking(self):
        """Reuse count increments per worker."""
        from dspytools.generate.module import SandboxPool

        pool = SandboxPool(pool_size=1, timeout=5.0, max_reuse=10)
        for i in range(3):
            pool.execute(f"print({i})")
        assert pool.stats["reuse_counts"] and any(pool.stats["reuse_counts"].values())
        pool.shutdown()


class TestSandboxPoolStats:
    """stats property reports pool state."""

    def test_stats_empty_pool(self):
        from dspytools.generate.module import SandboxPool

        pool = SandboxPool(pool_size=0, timeout=5.0)
        s = pool.stats
        assert s["pool_size"] == 0
        assert s["workers_alive"] == 0
        assert s["workers_in_use"] == 0
        pool.shutdown()

    def test_stats_shows_in_use(self):
        from dspytools.generate.module import SandboxPool

        pool = SandboxPool(pool_size=2, timeout=5.0)
        idx, proc = pool._acquire()
        assert idx is not None
        s = pool.stats
        assert s["workers_in_use"] == 1
        assert s["workers_available"] == 1
        pool._release(idx)
        pool.shutdown()

    def test_stats_after_shutdown(self):
        from dspytools.generate.module import SandboxPool

        pool = SandboxPool(pool_size=1, timeout=5.0)
        pool.shutdown()
        s = pool.stats
        assert s["workers_alive"] == 0

    def test_stats_singleton_consistency(self):
        """get_sandbox_pool returns same pool within test."""
        from dspytools.generate.module import get_sandbox_pool

        pool1 = get_sandbox_pool()
        pool2 = get_sandbox_pool()
        assert pool1 is pool2


class TestSandboxPoolShutdown:
    """shutdown kills all workers."""

    def test_shutdown_clears_workers(self):
        from dspytools.generate.module import SandboxPool

        pool = SandboxPool(pool_size=2, timeout=5.0)
        assert len(pool._workers) == 2
        pool.shutdown()
        assert len(pool._workers) == 0
        assert len(pool._in_use) == 0

    def test_shutdown_idempotent(self):
        """Calling shutdown twice is safe."""
        from dspytools.generate.module import SandboxPool

        pool = SandboxPool(pool_size=1, timeout=5.0)
        pool.shutdown()
        pool.shutdown()  # should not raise
