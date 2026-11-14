"""Auto-retry with exponential backoff for compile and LM operations.

DSPy 3.5 owns LM error classification. Its taxonomy is three levels deep
(verified against ``dspy/utils/exceptions.py`` in 3.5.0)::

    DSPyError
    └── LMError
        ├── LMTransportError          retryable (network/DNS/TLS/reset)
        ├── LMLockTimeoutError        retryable (local credential-lock contention)
        ├── LMUnexpectedError         NOT retryable by DSPy (unclassified)
        ├── LMConfigurationError
        │   └── LMNotConfiguredError
        ├── LMCollectionLimitError    NOT retryable (local collector budget)
        ├── LMUnsupportedFeatureError
        └── LMProviderError
            ├── LMRateLimitError      retryable, carries ``retry_after``
            ├── LMTimeoutError        retryable
            ├── LMServerError         retryable
            ├── LMAuthError           NOT retryable (bad credentials)
            ├── LMBillingError        NOT retryable (billing/quota)
            ├── LMInvalidRequestError NOT retryable (bad request shape)
            │   ├── ContextWindowExceededError   NOT retryable
            │   └── LMUnsupportedModelError      NOT retryable

Rather than hardcoding that list, this module delegates the decision to
``dspy.utils.exceptions.is_retryable_lm_error`` — upstream's own classifier,
whose retryable set is ``(LMRateLimitError, LMTimeoutError, LMServerError,
LMTransportError, LMLockTimeoutError)``. Retrying a context-window, auth or
billing error can never succeed and burns tokens/money, so those are excluded
even when a caller widens ``retryable_exceptions``.

Every ``DSPyError`` carries ``retry_after`` (a float in seconds or ``None``,
set by the base ``DSPyError.__init__``), which is honoured here when present
and capped at ``max_delay``.
"""

from __future__ import annotations

import functools
import time
from collections.abc import Callable
from typing import Any

from dspytools.core.logging_config import get_logger

log = get_logger(__name__)

#: Exceptions that must never be retried, whatever the caller asks for.
#: Retrying these can never succeed and spends money to prove it.
#: Resolved lazily so importing this module never imports dspy.
_NON_RETRYABLE_NAMES = frozenset(
    {
        "ContextWindowExceededError",
        "LMInvalidRequestError",
        "LMAuthError",
        "LMBillingError",
        "LMConfigurationError",
        "LMNotConfiguredError",
        "LMUnsupportedFeatureError",
        "LMUnsupportedModelError",
        "LMCollectionLimitError",
    }
)

#: Plain-Python transient failures that predate the DSPy taxonomy.
_BASE_RETRYABLE: tuple[type[BaseException], ...] = (
    TimeoutError,
    ConnectionError,
    OSError,
)

_lm_retryable_cache: tuple[type[BaseException], ...] | None = None
_lm_non_retryable_cache: tuple[type[BaseException], ...] | None = None

#: Fallback spelling of DSPy's retryable set, used only if the installed
#: ``_RETRYABLE_LM_ERRORS`` symbol is ever renamed.  The authoritative source is
#: upstream's own tuple.
_RETRYABLE_NAMES = (
    "LMRateLimitError",
    "LMTimeoutError",
    "LMServerError",
    "LMTransportError",
    "LMLockTimeoutError",
)


def _lm_exception_sets() -> tuple[
    tuple[type[BaseException], ...], tuple[type[BaseException], ...]
]:
    """``(retryable, never_retryable)`` DSPy LM exception types.

    Built on first use from the installed DSPy so this module stays importable
    without dspy (the laziness contract from Golden Rule 1).
    """
    global _lm_retryable_cache, _lm_non_retryable_cache
    if _lm_retryable_cache is None or _lm_non_retryable_cache is None:
        from dspy.utils import exceptions as lm_exc

        upstream = getattr(lm_exc, "_RETRYABLE_LM_ERRORS", None)
        if upstream is None:
            log.warning("retry_retryable_symbol_missing")
            upstream = tuple(
                cls
                for cls in (getattr(lm_exc, name, None) for name in _RETRYABLE_NAMES)
                if isinstance(cls, type)
            )
        _lm_retryable_cache = tuple(upstream)
        # Name-driven on purpose: the class names are the contract, not the
        # private module layout.  A missing name degrades to the safe default
        # below (never retry an LMError upstream does not classify as retryable).
        non_retryable: list[type[BaseException]] = [
            cls
            for cls in (getattr(lm_exc, name, None) for name in _NON_RETRYABLE_NAMES)
            if isinstance(cls, type)
        ]
        if not non_retryable:
            log.warning("retry_non_retryable_resolution_fallback")
            non_retryable = [lm_exc.LMError]
        _lm_non_retryable_cache = tuple(non_retryable)
    return _lm_retryable_cache, _lm_non_retryable_cache


def _default_retryable_types() -> tuple[type[BaseException], ...]:
    lm_retryable, _ = _lm_exception_sets()
    return _BASE_RETRYABLE + lm_retryable


def retryable_exceptions() -> tuple[type[BaseException], ...]:
    """Default retryable exception tuple: stdlib transients + DSPy transients."""
    return _default_retryable_types()


def non_retryable_exceptions() -> tuple[type[BaseException], ...]:
    """Exceptions excluded from retry even if a caller widens the tuple."""
    _, lm_non_retryable = _lm_exception_sets()
    return lm_non_retryable


def retry_after_seconds(exc: BaseException) -> float | None:
    """Provider-suggested retry delay carried by a DSPy error, if any.

    ``DSPyError.__init__`` always sets ``retry_after`` (float seconds or
    ``None``); non-DSPy errors generally have no such attribute.  Invalid
    values (negative, NaN, bools, non-numeric) are ignored.
    """
    value = getattr(exc, "retry_after", None)
    if value is None or isinstance(value, bool):
        return None
    if not isinstance(value, (int, float)):
        return None
    seconds = float(value)
    if seconds != seconds or seconds in (float("inf"), float("-inf")):
        return None  # NaN / inf
    return seconds if seconds >= 0 else None


def _delay_for(
    attempt: int,
    base_delay: float,
    backoff_factor: float,
    max_delay: float,
    exc: BaseException | None = None,
) -> float:
    """Backoff for ``attempt``, overridden by a capped ``retry_after``."""
    hinted = retry_after_seconds(exc) if exc is not None else None
    if hinted is not None:
        return min(hinted, max_delay)
    return min(base_delay * (backoff_factor**attempt), max_delay)


def _is_retryable(
    exc: BaseException, retryable: tuple[type[BaseException], ...]
) -> bool:
    if isinstance(exc, non_retryable_exceptions()):
        return False
    return isinstance(exc, retryable)


def retry(
    max_retries: int = 3,
    base_delay: float = 2.0,
    max_delay: float = 60.0,
    backoff_factor: float = 2.0,
    retryable_exceptions: tuple | None = None,
):
    """Decorator: retry a function with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay in seconds
        max_delay: Maximum delay cap (also caps ``retry_after`` hints)
        backoff_factor: Multiplier for each subsequent retry
        retryable_exceptions: Exception types that trigger a retry.  Defaults to
            :func:`retryable_exceptions` (stdlib transients + DSPy's own
            retryable LM errors).  DSPy's never-retryable errors are excluded
            even when the tuple is widened.

    Example:
        @retry(max_retries=3, base_delay=2.0)
        def compile_program(student, trainset):
            return optimizer.compile(student=student, trainset=trainset)
    """
    types = (
        retryable_exceptions
        if retryable_exceptions is not None
        else _default_retryable_types()
    )

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if not _is_retryable(e, types):
                        raise
                    last_exception = e
                    if attempt == max_retries:
                        raise

                    delay = _delay_for(
                        attempt, base_delay, backoff_factor, max_delay, e
                    )
                    log.warning(
                        "retry_attempt",
                        attempt=attempt + 1,
                        max_retries=max_retries,
                        delay=round(delay, 3),
                        error=type(e).__name__,
                    )
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

    Retries DSPy's transient LM errors plus ``RuntimeError`` (OOM conditions),
    and honours a provider ``retry_after`` hint when one is present.
    Tracks retry statistics for observability.

    Returns:
        (compiled_program, retry_stats_dict)
    """
    types = retryable_exceptions() + (RuntimeError,)
    stats = {"attempts": 0, "retries": 0, "errors": [], "total_delay": 0.0}
    last_error = None

    for attempt in range(max_retries + 1):
        stats["attempts"] += 1

        try:
            result = compile_fn(*args, **kwargs)
            return result, stats
        except Exception as e:
            if not _is_retryable(e, types):
                raise
            stats["retries"] += 1
            stats["errors"].append(str(e)[:200])

            if attempt == max_retries:
                raise

            delay = _delay_for(attempt, base_delay, 2.0, 60.0, e)
            stats["total_delay"] += delay
            log.warning(
                "compile_retry_attempt",
                attempt=attempt + 1,
                max_retries=max_retries,
                delay=round(delay, 3),
                error=type(e).__name__,
            )
            time.sleep(delay)

    if last_error:
        raise last_error
    return None, stats
