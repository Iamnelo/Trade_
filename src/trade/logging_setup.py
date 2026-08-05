"""Structured logging setup using structlog.

The same processors run in local console and production JSON modes; only the
final renderer changes. That keeps developer output readable while production
emits Loki/CloudWatch-friendly JSON.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog

from trade.config import LogFormat, get_settings


def configure_logging() -> None:
    settings = get_settings()
    numeric_level = logging.getLevelName(settings.log_level.upper())
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {settings.log_level!r}")

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=numeric_level,
        force=True,
    )

    processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]
    if settings.log_format is LogFormat.JSON:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer(colors=sys.stdout.isatty()))

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(numeric_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str | None = None) -> Any:
    return structlog.get_logger(name)
