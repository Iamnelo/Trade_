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

import contextlib
import json
import queue
import threading
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import Protocol

import structlog

from trade.paper.config import TelegramConfig

_log = structlog.get_logger(__name__)

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
        except urllib.error.HTTPError as exc:
            # Telegram returns 4xx with a JSON body explaining the problem
            # (bad token, wrong chat_id, bot not started, ...). Surface it so
            # the operator can see WHY delivery failed in the logs.
            detail = ""
            with contextlib.suppress(Exception):
                detail = exc.read().decode("utf-8", "replace")[:300]
            self.failed += 1
            _log.warning(
                "telegram_notify_failed", status=exc.code, detail=detail, kind="http_error"
            )
            return False
        except (urllib.error.URLError, OSError, ValueError) as exc:
            # Non-fatal by design: a notification failure must never disturb
            # the trading loop. Log it so it is diagnosable, not silent.
            self.failed += 1
            _log.warning("telegram_notify_failed", error=str(exc), kind="transport_error")
            return False
        self.sent += 1
        return True


class BackgroundNotifier:
    """Deliver notifications on a daemon thread so sends never block the loop.

    `notify` enqueues and returns immediately (True = accepted for delivery, not
    delivered yet). A worker thread drains the queue and calls the wrapped
    notifier; any error there is logged, never raised. This decouples Telegram
    latency and failures entirely from the trading engine (goals: non-fatal +
    non-blocking).
    """

    _SENTINEL = object()

    def __init__(self, inner: Notifier, *, max_queue: int = 1000) -> None:
        self._inner = inner
        self._queue: queue.Queue[object] = queue.Queue(maxsize=max_queue)
        self.dropped = 0
        self._thread = threading.Thread(target=self._run, name="paper-notifier", daemon=True)
        self._thread.start()

    def notify(self, text: str) -> bool:
        try:
            self._queue.put_nowait(text)
        except queue.Full:
            # Never block the trading loop; drop and count instead.
            self.dropped += 1
            _log.warning("notify_queue_full_dropped", dropped=self.dropped)
            return False
        return True

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is self._SENTINEL:
                    return
                try:
                    self._inner.notify(str(item))
                except Exception as exc:
                    _log.warning("notifier_worker_error", error=str(exc))
            finally:
                self._queue.task_done()

    def flush(self, timeout: float = 5.0) -> None:
        """Best-effort wait for the queue to drain (used on shutdown)."""
        deadline = threading.Event()
        threading.Timer(timeout, deadline.set).start()
        while not self._queue.empty() and not deadline.is_set():
            deadline.wait(0.05)

    def close(self, timeout: float = 5.0) -> None:
        self.flush(timeout=timeout)
        self._queue.put(self._SENTINEL)
        self._thread.join(timeout=timeout)


def log_notifier_status(notifier: Notifier, config: TelegramConfig) -> None:
    """Emit a clear one-line startup signal about notification delivery.

    This is the missing diagnostic: without it, an unconfigured Telegram looks
    identical to a working one (silent NullNotifier).
    """
    if config.is_configured:
        _log.info("telegram_configured", chat_id_set=True)
    else:
        _log.warning(
            "telegram_not_configured",
            hint="set TRADE_TELEGRAM_BOT_TOKEN and TRADE_TELEGRAM_CHAT_ID; "
            "notifications are disabled (no-op) until both are present",
        )


def build_notifier(config: TelegramConfig, *, transport: Transport | None = None) -> Notifier:
    """Return a Telegram notifier if configured, else a NullNotifier."""
    if config.is_configured:
        return TelegramNotifier(config, transport=transport)
    return NullNotifier()
