"""Smoke test for the structured logging setup."""

from __future__ import annotations

from trade.config import get_settings
from trade.logging_setup import configure_logging, get_logger


def test_configure_and_log_does_not_raise() -> None:
    get_settings.cache_clear()
    configure_logging()
    log = get_logger("test")
    log.info("smoke", key="value", n=1)


def test_get_logger_returns_bound_logger() -> None:
    configure_logging()
    log = get_logger("some.module")
    # Duck-typed check — the bound logger accepts kwargs.
    log.info("event", answer=42)
