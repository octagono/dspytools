"""Auto-retry with exponential backoff for compile operations.

Handles transient failures: OOM, API timeouts, rate limits.
"""

from __future__ import annotations

import functools
import time
from typing import Any, Callable


def retry(
    max_retries: int = 3,
    base_delay: float = 2.0,
    max_delay: float = 60.0,
    backoff_factor: float = 2.0,
    retryable_exceptions: tuple = (TimeoutError, ConnectionError, OSError),
):
    """Decorator: retry a function with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay in seconds
        max_delay: Maximum delay cap
        backoff_factor: Multiplier for each subsequent retry
        retryable_exceptions: Exception types that trigger a retry

    Example:
        @retry(max_retries=3, base_delay=2.0)
        def compile_program(student, trainset):
            return optimizer.compile(student=student, trainset=trainset)
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except retryable_exceptions as e:
                    last_exception = e
                    if attempt == max_retries:
                        raise

                    delay = min(base_delay * (backoff_factor**attempt), max_delay)
                    time.sleep(delay)

            raise last_exception  # type: ignore[misc]

        return wrapper

    return decorator


def compile_with_retry(
    compile_fn: Callable,
    *args: Any,
    max_retries: int = 3,
    base_delay: float = 2.0,
    **kwargs: Any,
) -> tuple[Any, dict]:
    """Run a compile operation with auto-retry.

    Handles OOM errors, API timeouts, and rate limits.
    Tracks retry statistics for observability.

    Returns:
        (compiled_program, retry_stats_dict)
    """
    stats = {"attempts": 0, "retries": 0, "errors": [], "total_delay": 0.0}
    last_error = None

    for attempt in range(max_retries + 1):
        stats["attempts"] += 1

        try:
            result = compile_fn(*args, **kwargs)
            return result, stats
        except (TimeoutError, ConnectionError, OSError, RuntimeError) as e:
            stats["retries"] += 1
            stats["errors"].append(str(e)[:200])

            if attempt == max_retries:
                raise

            delay = min(base_delay * (2**attempt), 60.0)
            stats["total_delay"] += delay
            time.sleep(delay)

    if last_error:
        raise last_error
    return None, stats
