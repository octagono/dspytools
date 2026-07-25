"""Background compile job scheduler.

Optimization 4: Async compile queue — runs optimizers in background threads,
returns job_id immediately for status polling.
"""

from __future__ import annotations

import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Callable

from dspytools.core.logging_config import get_logger

log = get_logger(__name__)


class CompileJob:
    """Represents a single compilation job in the queue."""

    def __init__(
        self,
        job_id: str,
        optimizer: str,
        module_name: str,
        label: str | None = None,
    ):
        self.job_id = job_id
        self.optimizer = optimizer
        self.module_name = module_name
        self.label = label
        self.status = "queued"  # queued → running → completed/failed
        self.progress: float = 0.0
        self.message: str = ""
        self.created = datetime.now().isoformat()
        self.completed: str | None = None
        self.run_id: str | None = None
        self.error: str | None = None


class CompileScheduler:
    """Background compile job scheduler.

    Runs optimizers asynchronously in a thread pool so the CLI/API
    remains responsive during long compilations.
    """

    _executor = ThreadPoolExecutor(max_workers=2)
    _jobs: dict[str, CompileJob] = {}
    _lock = threading.Lock()

    @classmethod
    def submit(
        cls,
        optimizer: str,
        module_name: str,
        compile_fn: Callable[[], str],
        label: str | None = None,
    ) -> str:
        """Submit a compile job. Returns job_id immediately.

        The compile_fn should run the optimizer and return the run_id.
        """
        job_id = f"job_{uuid.uuid4().hex[:8]}"
        job = CompileJob(job_id, optimizer, module_name, label)

        with cls._lock:
            cls._jobs[job_id] = job
            # Prune old terminal jobs to prevent unbounded growth (max 500)
            if len(cls._jobs) > 500:
                terminal = [
                    jid
                    for jid, j in cls._jobs.items()
                    if j.status in ("completed", "failed", "cancelled")
                ]
                for jid in terminal[: len(terminal) - 450]:
                    del cls._jobs[jid]

        def _run():
            job.status = "running"
            job.message = "Starting compilation..."
            try:
                run_id = compile_fn()
                job.status = "completed"
                job.progress = 1.0
                job.run_id = run_id
                job.message = f"Compiled → {run_id}"
            except (RuntimeError, OSError, ValueError, TypeError) as e:
                log.warning("async_compile_failed", job_id=job.job_id, error=str(e))
                job.status = "failed"
                job.error = str(e)
                job.message = f"Failed: {e}"
            finally:
                job.completed = datetime.now().isoformat()

        cls._executor.submit(_run)
        return job_id

    @classmethod
    def get_status(cls, job_id: str) -> CompileJob | None:
        with cls._lock:
            return cls._jobs.get(job_id)

    @classmethod
    def list_jobs(cls) -> list[dict]:
        with cls._lock:
            return [
                {
                    "job_id": j.job_id,
                    "optimizer": j.optimizer,
                    "module_name": j.module_name,
                    "status": j.status,
                    "progress": j.progress,
                    "message": j.message,
                    "created": j.created,
                    "completed": j.completed,
                    "run_id": j.run_id,
                    "error": j.error,
                }
                for j in cls._jobs.values()
            ]

    @classmethod
    def cancel(cls, job_id: str) -> bool:
        with cls._lock:
            if job_id in cls._jobs:
                job = cls._jobs[job_id]
                if job.status in ("queued", "running"):
                    job.status = "cancelled"
                    job.message = "Cancelled by user"
                    return True
        return False
