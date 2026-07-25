"""Typed exception hierarchy for dspytools.

Replaces bare except:pass with structured error handling.
Every external service failure carries service_name and can be caught selectively.
"""

from __future__ import annotations

from typing import Any


class DspyToolsError(Exception):
    """Base for all dspytools exceptions."""

    pass


class ServiceUnavailableError(DspyToolsError):
    """External service is unreachable (FalkorDB, Redis, llama-cpp-server, MLflow).

    Attributes:
        service_name: Name of the unreachable service
        retry_after: Suggested retry delay in seconds, or None
    """

    def __init__(
        self, service_name: str, message: str = "", retry_after: float | None = None
    ) -> None:
        self.service_name = service_name
        self.retry_after = retry_after
        super().__init__(message or f"Service '{service_name}' is unavailable")


class CacheError(ServiceUnavailableError):
    """Redis or semantic cache is unavailable."""

    pass


class GraphError(ServiceUnavailableError):
    """FalkorDB graph database is unavailable."""

    pass


class LlamaCppError(ServiceUnavailableError):
    """llama-cpp-server is unreachable or model is not loaded."""

    pass


class CompileError(DspyToolsError):
    """Compilation failed (optimizer error, LM failure, timeout).

    Attributes:
        optimizer: Name of the optimizer that failed
        module_name: Name of the module being compiled
        cause: Original exception that triggered the failure
    """

    def __init__(
        self,
        optimizer: str,
        module_name: str = "",
        message: str = "",
        cause: BaseException | None = None,
    ) -> None:
        self.optimizer = optimizer
        self.module_name = module_name
        self.cause = cause
        super().__init__(message or f"Compile failed: {optimizer} on {module_name}")


class ValidationError(DspyToolsError):
    """Input validation or SPRT rejection.

    Attributes:
        field: Name of the field that failed validation
        value: The invalid value
        constraint: Description of the violated constraint
    """

    def __init__(
        self,
        field: str = "",
        value: Any = None,
        constraint: str = "",
        message: str = "",
    ) -> None:
        self.field = field
        self.value = value
        self.constraint = constraint
        super().__init__(message or f"Validation failed: {field} — {constraint}")


class ConfigError(DspyToolsError):
    """Configuration is missing or invalid.

    Attributes:
        key: The config key that is missing/invalid
    """

    def __init__(self, key: str = "", message: str = "") -> None:
        self.key = key
        super().__init__(message or f"Configuration error: {key}")


class RateLimitError(DspyToolsError):
    """API rate limit exceeded.

    Attributes:
        endpoint: The rate-limited endpoint
        retry_after: Seconds until retry is allowed
    """

    def __init__(
        self, endpoint: str = "", retry_after: float = 0.0, message: str = ""
    ) -> None:
        self.endpoint = endpoint
        self.retry_after = retry_after
        super().__init__(
            message or f"Rate limit exceeded: {endpoint} (retry in {retry_after:.0f}s)"
        )
