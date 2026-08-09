"""Notifications for the paper-trading system.

`Notifier` is a tiny protocol; the engine calls `notify(text)` for every
lifecycle event, decision, trade, halt, and summary. Two implementations:

- `TelegramNotifier` — posts to the Telegram Bot API. Sends are best-effort
  and NON-FATAL: any transport error is swallowed (and counted) so a flaky
  network can never interfere with trading logic. The transport is injectable
  so message formatting can be unit-tested without a real network call.
- `NullNotifier` — records messages in memory (used when Telegram is not
  configured, and in tests).

Zero third-party dependency: the default transport uses `urllib`.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import Protocol

from trade.paper.config import TelegramConfig

# A transport takes (url, json_body) and performs the POST. Injectable for tests.
Transport = Callable[[str, dict[str, object]], None]


class Notifier(Protocol):
    def notify(self, text: str) -> bool:
        """Send a message. Returns True if delivered, False otherwise.
        Implementations MUST NOT raise on transport failure."""
        ...


class NullNotifier:
    """No-op notifier that keeps a record of messages (for tests / disabled)."""

    def __init__(self) -> None:
        self.messages: list[str] = []

    def notify(self, text: str) -> bool:
        self.messages.append(text)
        return True


def _urllib_transport(url: str, body: dict[str, object]) -> None:
    if not url.startswith("https://"):  # defensive: only the Telegram HTTPS API
        raise ValueError("telegram transport requires an https URL")
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as resp:
        resp.read()


class TelegramNotifier:
    """Best-effort Telegram sender. Never raises on transport failure."""

    def __init__(
        self,
        config: TelegramConfig,
        *,
        transport: Transport | None = None,
    ) -> None:
        self._config = config
        self._transport = transport or _urllib_transport
        self.sent = 0
        self.failed = 0

    @property
    def is_configured(self) -> bool:
        return self._config.is_configured

    def notify(self, text: str) -> bool:
        if not self._config.is_configured:
            self.failed += 1
            return False
        url = f"https://api.telegram.org/bot{self._config.bot_token}/sendMessage"
        body: dict[str, object] = {
            "chat_id": self._config.chat_id,
            "text": text,
            "disable_web_page_preview": True,
        }
        try:
            self._transport(url, body)
        except (urllib.error.URLError, OSError, ValueError):
            # Non-fatal by design: a notification failure must never disturb
            # the trading loop.
            self.failed += 1
            return False
        self.sent += 1
        return True


def build_notifier(config: TelegramConfig, *, transport: Transport | None = None) -> Notifier:
    """Return a Telegram notifier if configured, else a NullNotifier."""
    if config.is_configured:
        return TelegramNotifier(config, transport=transport)
    return NullNotifier()
