"""MLflow tracing and experiment tracking for dspytools.

Integrates MLflow (v3.5.1+) with the dspytools compile pipeline.
Provides experiment tracking, trace logging, and feedback logging.
Uses sqlite:/// backend for local tracking (file:// is deprecated in MLflow v3).
"""

from __future__ import annotations

import os
import queue
import threading
import time
from contextlib import contextmanager
from typing import Any

import mlflow

from dspytools.config.settings import config_dir as _config_dir


class MLflowTracker:
    """Wrapper around MLflow for dspytools experiment tracking.

    Usage:
        tracker = MLflowTracker()

        with tracker.trace("compile_mipro"):
            compiled = optimizer.compile(student=student, trainset=trainset)

        tracker.log_compile(
            optimizer="mipro",
            module="TestMod",
            score=0.85,
            params={"auto": "light"},
        )
    """

    def __init__(
        self, tracking_uri: str | None = None, experiment_name: str = "dspytools"
    ):
        # Allow env override, else None so _ensure_initialized chooses sqlite
        self.tracking_uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI")
        self.experiment_name = experiment_name
        self._mlflow = None
        self._initialized = False

    @property
    def mlflow(self):
        """Lazy import mlflow."""
        if self._mlflow is None:
            self._mlflow = mlflow
        return self._mlflow

    def _ensure_initialized(self):
        """Initialize MLflow tracking with sqlite:/// backend.

        Uses sqlite:/// for local tracking (file:// is deprecated in MLflow v3).
        Remote tracking (e.g. MLflow server) is handled via MLFLOW_TRACKING_URI env var.
        """
        if self._initialized:
            return

        if self.tracking_uri:
            self.mlflow.set_tracking_uri(self.tracking_uri)
        else:
            db_path = str(_config_dir() / "mlruns" / "mlflow.db")
            os.makedirs(os.path.dirname(db_path), exist_ok=True)
            self.mlflow.set_tracking_uri(f"sqlite:///{db_path}")

        # Create or get experiment — only catch "not found", re-raise everything else
        from mlflow.exceptions import MlflowException

        try:
            self.mlflow.get_experiment_by_name(self.experiment_name)
        except MlflowException:
            self.mlflow.create_experiment(self.experiment_name)

        # Enable DSPy autologging — auto-traces every dspy.Module, dspy.ReAct,
        # dspy.ChainOfThought, dspy.Predict, dspy.Tool, and LM call project-wide.
        # Uses importlib because mlflow.dspy is a dynamic plugin not statically resolvable.
        self._enable_dspy_autolog()

        self._initialized = True

    @contextmanager
    def trace(self, name: str):
        """Context manager for MLflow autologging a compile operation.

        Example:
            with tracker.trace("compile_mipro"):
                compiled = optimizer.compile(...)
        """
        self._ensure_initialized()
        run = self.mlflow.start_run(run_name=name)
        try:
            yield run
        finally:
            self.mlflow.end_run()

    def _enable_dspy_autolog(self) -> None:
        """Enable MLflow DSPy autologging via importlib.

        mlflow.dspy is a dynamic plugin registered at runtime, so static
        analysis (pyright) cannot resolve it. Using importlib avoids
        false-positive reportPrivateImportUsage errors while keeping
        full runtime behavior.
        """
        import importlib

        mdspy = importlib.import_module("mlflow.dspy")
        mdspy.autolog(
            log_traces=True,
            log_traces_from_compile=True,
            log_traces_from_eval=True,
            log_compiles=True,
            log_evals=True,
        )

    def log_compile(
        self,
        optimizer: str,
        module: str,
        score: float,
        params: dict[str, Any] | None = None,
        metrics: dict[str, float] | None = None,
    ) -> str | None:
        """Log a compile run to MLflow.

        Args:
            optimizer: Optimizer name (e.g., "mipro", "gepa")
            module: Module name
            score: Quality score (0.0-1.0)
            params: Optimizer parameters
            metrics: Additional metrics

        Returns:
            run_id

        Raises:
            Exception: Propagates any MLflow errors (fail-fast).
        """
        self._ensure_initialized()
        self.mlflow.set_experiment(self.experiment_name)

        with self.mlflow.start_run(run_name=f"{optimizer}_{module}") as run:
            # Log parameters
            self.mlflow.log_param("optimizer", optimizer)
            self.mlflow.log_param("module", module)
            if params:
                for key, value in params.items():
                    self.mlflow.log_param(key, str(value))

            # Log metrics
            self.mlflow.log_metric("score", score)
            if metrics:
                for key, value in metrics.items():
                    self.mlflow.log_metric(key, value)

            return run.info.run_id

    def log_gfl_comparison(self, pipeline_result: dict) -> str | None:
        """Log GFL pipeline 4-way comparison results.

        Args:
            pipeline_result: Dict from GFLPipeline.run()

        Returns:
            run_id

        Raises:
            Exception: Propagates any MLflow errors (fail-fast).
        """
        self._ensure_initialized()

        with self.mlflow.start_run(run_name="gfl_comparison") as run:
            self.mlflow.log_param(
                "best_optimizer", pipeline_result.get("best_optimizer", "unknown")
            )
            self.mlflow.log_metric("best_score", pipeline_result.get("best_score", 0.0))
            self.mlflow.log_metric("baseline", pipeline_result.get("baseline", 0.5))
            self.mlflow.log_metric(
                "improvement", pipeline_result.get("improvement", 0.0)
            )

            # Log per-optimizer scores
            all_scores = pipeline_result.get("all_scores", {})
            for name, score in all_scores.items():
                self.mlflow.log_metric(f"score_{name}", score)

            return run.info.run_id

    def log_feedback(
        self, trace_id: str, name: str, value: float, rationale: str = ""
    ) -> None:
        """Log evaluation feedback for a trace.

        Args:
            trace_id: MLflow trace ID
            name: Feedback name (e.g., "quality", "accuracy")
            value: Numeric score
            rationale: Explanation

        Raises:
            Exception: Propagates any MLflow errors (fail-fast).
        """
        self._ensure_initialized()
        self.mlflow.log_feedback(
            trace_id=trace_id,
            name=name,
            value=value,
            rationale=rationale,
        )


class MLflowAsyncTracker(MLflowTracker):
    """Async MLflow tracker with background worker queue.

    Enqueues log operations to a background thread, so compile commands
    never block on MLflow HTTP POST requests.

    Falls back to synchronous logging if the queue is full or background
    worker is not running.
    """

    def __init__(
        self,
        tracking_uri: str | None = None,
        experiment_name: str = "dspytools",
        max_queue_size: int = 500,
        max_workers: int = 1,
    ):
        super().__init__(tracking_uri, experiment_name)
        from concurrent.futures import (
            ThreadPoolExecutor,  # lazy: avoids threading._register_atexit at module level
        )

        self._queue: queue.Queue = queue.Queue(maxsize=max_queue_size)
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="mlflow-async"
        )
        self._running = True
        self._worker_thread = threading.Thread(
            target=self._process_queue, daemon=True, name="mlflow-worker"
        )
        self._worker_thread.start()
        self._dropped: int = 0  # count of dropped log events

    def _process_queue(self):
        """Background worker: drain queue and log to MLflow."""
        while self._running:
            try:
                # Block for up to 1 second, then check _running
                item = self._queue.get(timeout=1.0)
                if item is None:  # sentinel for shutdown
                    break

                method, args, kwargs = item
                try:
                    method(*args, **kwargs)
                finally:
                    self._queue.task_done()
            except queue.Empty:
                continue

    def _enqueue(self, method, *args, **kwargs):
        """Enqueue a log operation, or execute synchronously if queue full."""
        if not self._running:
            return

        try:
            self._queue.put_nowait((method, args, kwargs))
        except queue.Full:
            self._dropped += 1
            # Execute synchronously as fallback
            method(*args, **kwargs)

    def log_compile(
        self,
        optimizer: str,
        module: str,
        score: float,
        params: dict[str, Any] | None = None,
        metrics: dict[str, float] | None = None,
    ) -> str | None:
        """Async version: enqueues and returns immediately."""
        self._enqueue(super().log_compile, optimizer, module, score, params, metrics)
        return None  # run_id not available in async mode

    def log_gfl_comparison(self, pipeline_result: dict) -> str | None:
        """Async version."""
        self._enqueue(super().log_gfl_comparison, pipeline_result)
        return None

    def log_feedback(
        self, trace_id: str, name: str, value: float, rationale: str = ""
    ) -> None:
        """Async version."""
        self._enqueue(super().log_feedback, trace_id, name, value, rationale)

    def shutdown(self, wait: bool = True, timeout: float = 5.0):
        """Graceful shutdown: drain queue and stop worker."""
        self._running = False
        if wait:
            self._queue.put_nowait(None)  # sentinel
            self._worker_thread.join(timeout=timeout)
        self._executor.shutdown(wait=False)

    def flush(self, timeout: float = 3.0) -> dict:
        """Drain the queue and wait for pending logs to complete.

        Should be called before process exit to prevent telemetry loss.
        Note: does NOT call queue.join() to avoid deadlock with the worker thread.
        Items currently being processed by the worker will complete on their own.

        Returns: {drained: int, remaining: int, timed_out: bool, dropped_total: int}
        """

        start = time.time()
        drained = 0

        # Drain queue
        while time.time() - start < timeout:
            try:
                item = self._queue.get_nowait()
                if item is None:  # sentinel
                    self._queue.task_done()
                    break
                method, args, kwargs = item
                try:
                    method(*args, **kwargs)
                finally:
                    self._queue.task_done()
                drained += 1
            except queue.Empty:
                break

        remaining = self._queue.qsize()

        return {
            "drained": drained,
            "remaining": remaining,
            "timed_out": time.time() - start >= timeout,
            "dropped_total": self._dropped,
        }

    @property
    def stats(self) -> dict:
        """Return queue statistics."""
        return {
            "queue_size": self._queue.qsize(),
            "dropped": self._dropped,
            "running": self._running,
            "worker_alive": self._worker_thread.is_alive(),
        }


# Singleton instance
_tracker: MLflowTracker | None = None


def get_tracker(async_mode: bool = True) -> MLflowTracker:
    """Get or create the global MLflow tracker.

    Args:
        async_mode: Use async background queue (default True).
    """
    global _tracker

    if async_mode:
        if _tracker is None or not isinstance(_tracker, MLflowAsyncTracker):
            if _tracker is not None:
                _tracker.shutdown()  # type: ignore[union-attr]
            _tracker = MLflowAsyncTracker()
    else:
        if _tracker is None or isinstance(_tracker, MLflowAsyncTracker):
            if _tracker is not None:
                _tracker.shutdown()  # type: ignore[union-attr]
            _tracker = MLflowTracker()

    # Ensure DSPy autolog is set up from the main thread before any worker
    # thread touches MLflow. DSPy's settings.configure() has thread-ownership
    # tracking and must be called from the initiator thread.
    _tracker._ensure_initialized()

    # Cache tracker ref in main module so atexit callback doesn't need to import mlflow
    from dspytools import main as _main

    _main._mlflow_tracker_ref = _tracker

    return _tracker
