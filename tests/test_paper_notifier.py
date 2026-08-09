"""Telegram notifier: formatting via injected transport + non-fatal failures."""

from __future__ import annotations

from trade.paper.config import TelegramConfig
from trade.paper.notifier import NullNotifier, TelegramNotifier, build_notifier


def test_null_notifier_records_messages() -> None:
    n = NullNotifier()
    assert n.notify("hello") is True
    assert n.messages == ["hello"]


def test_telegram_notifier_posts_via_transport() -> None:
    sent: list[tuple[str, dict[str, object]]] = []

    def transport(url: str, body: dict[str, object]) -> None:
        sent.append((url, body))

    cfg = TelegramConfig(bot_token="TOKEN", chat_id="CHAT")
    n = TelegramNotifier(cfg, transport=transport)
    assert n.notify("hi there") is True
    assert n.sent == 1
    url, body = sent[0]
    assert "botTOKEN/sendMessage" in url
    assert body["chat_id"] == "CHAT"
    assert body["text"] == "hi there"


def test_telegram_notifier_swallows_transport_errors() -> None:
    def boom(url: str, body: dict[str, object]) -> None:
        raise OSError("network down")

    n = TelegramNotifier(TelegramConfig(bot_token="T", chat_id="C"), transport=boom)
    assert n.notify("x") is False  # non-fatal
    assert n.failed == 1


def test_telegram_notifier_noop_when_unconfigured() -> None:
    n = TelegramNotifier(TelegramConfig(bot_token=None, chat_id=None))
    assert n.notify("x") is False
    assert n.sent == 0


def test_build_notifier_falls_back_to_null_when_unconfigured() -> None:
    assert isinstance(build_notifier(TelegramConfig()), NullNotifier)
    assert isinstance(build_notifier(TelegramConfig(bot_token="a", chat_id="b")), TelegramNotifier)
