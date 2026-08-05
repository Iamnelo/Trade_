"""Tests for trade.config: defaults, env-var overrides, and validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from trade.config import Environment, LogFormat, Settings, get_settings


@pytest.fixture(autouse=True)
def _clear_settings_cache() -> None:
    get_settings.cache_clear()


def test_defaults_are_sane() -> None:
    s = Settings()
    assert s.environment is Environment.LOCAL
    assert s.log_format in {LogFormat.CONSOLE, LogFormat.JSON}
    assert s.daily_drawdown_pct <= s.weekly_drawdown_pct <= s.monthly_drawdown_pct
    assert s.default_symbols == ("BTCUSDT", "ETHUSDT")
    assert s.default_category == "linear"
    assert s.prometheus_port == 9464


def test_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TRADE_ENVIRONMENT", "paper")
    monkeypatch.setenv("TRADE_LOG_FORMAT", "json")
    monkeypatch.setenv("TRADE_DAILY_DRAWDOWN_PCT", "0.02")
    s = Settings()
    assert s.environment is Environment.PAPER
    assert s.log_format is LogFormat.JSON
    assert s.daily_drawdown_pct == 0.02


def test_drawdown_bounds_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TRADE_DAILY_DRAWDOWN_PCT", "0.9")
    with pytest.raises(ValidationError):
        Settings()


def test_drawdown_ordering_enforced(monkeypatch: pytest.MonkeyPatch) -> None:
    # Weekly must not be tighter than daily; the layered-safeguard invariant.
    monkeypatch.setenv("TRADE_DAILY_DRAWDOWN_PCT", "0.10")
    monkeypatch.setenv("TRADE_WEEKLY_DRAWDOWN_PCT", "0.05")
    with pytest.raises(ValidationError):
        Settings()


def test_get_settings_is_cached() -> None:
    a = get_settings()
    b = get_settings()
    assert a is b
