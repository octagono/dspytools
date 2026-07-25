"""Central structured logging — structlog with stdlib integration.

Two output modes:
  PRODUCTION (DSPYTOOLS_ENV=production): JSON — for log aggregation.
  DEV (default): Compact console output — no timestamps, no module noise.

Operational events (cli_command, cli_command_exit) are suppressed at INFO
level in dev mode to keep CLI output clean for end users.

Usage:
    from dspytools.core.logging_config import get_logger
    log = get_logger(__name__)
    log.info("event_name", key=value)
"""

from __future__ import annotations

import logging
import os
import sys

import structlog

# ── Event names that are operational noise in CLI mode ───────────────────
_CLI_NOISE_EVENTS = frozenset({"cli_command", "cli_command_exit"})
"""Event names to suppress at INFO level in dev mode (operational noise)."""


def _drop_cli_noise(
    logger: logging.Logger, method_name: str, event: structlog.types.EventDict
) -> structlog.types.EventDict:
    """Drop operational CLI events at info level in dev mode.

    Keeps WARNING+ events visible so errors are never hidden.
    In production/JSON mode all events pass through.
    """
    if method_name == "info" and event.get("event") in _CLI_NOISE_EVENTS:
        raise structlog.DropEvent
    return event


def configure_logging(*, force: bool = False) -> None:
    """Configure structlog globally. Idempotent unless force=True.

    Called automatically by `get_logger()` on first use.
    Consumers should not need to call this directly.
    """
    if structlog.is_configured() and not force:
        return

    is_production = os.environ.get("DSPYTOOLS_ENV") == "production"

    # ── Shared processors (applied to both structlog and stdlib logs) ─────
    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.dev.set_exc_info,
    ]

    # ── Production: JSON output — all events, timestamps, full metadata ───
    if is_production:
        shared_processors.insert(2, structlog.processors.TimeStamper(fmt="iso"))
        renderer = structlog.processors.JSONRenderer()
    # ── Dev: compact console — no timestamps, filter CLI noise ────────────
    else:
        # Insert CLI noise filter AFTER add_log_level (which runs at index 2)
        shared_processors.insert(2, _drop_cli_noise)
        renderer = structlog.dev.ConsoleRenderer(
            pad_level=False,
            force_colors=True,
        )

    structlog.configure(
        processors=shared_processors
        + [
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)

    # Default level: INFO; respect DSPYTOOLS_LOG_LEVEL env var
    level_name = os.environ.get("DSPYTOOLS_LOG_LEVEL", "INFO").upper()
    root_logger.setLevel(getattr(logging, level_name, logging.INFO))

    # Quiet noisy third-party loggers
    for noisy in ("urllib3", "httpx", "httpcore", "litellm", "falkordb", "redis"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Return a structlog logger bound with `module=name`.

    Prefer `get_logger(__name__)` at module level.
    """
    configure_logging()
    return structlog.get_logger(name)


# Module-level singleton for the config module itself
log = get_logger(__name__)
