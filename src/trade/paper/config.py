"""Configuration for the Phase 4c paper-trading system.

The single most important field here is `execution_enabled` — the HARD MASTER
SWITCH. It defaults to ``False`` and gates whether the engine may place ANY
simulated order. With it off, the system still consumes live data, generates
predictions, records every decision, and sends notifications, but it never
opens or closes a position. This is the "fully integrated, execution disabled"
posture required until the Phase 5c forward-test gates pass and the operator
explicitly approves activation.

A second, independent guarantee lives in the engine, not here: the paper
system only ever talks to the in-memory simulated venue, so it can never place
a real or testnet order regardless of this switch.

`from_settings` reads non-secret operational values; Telegram credentials come
from the environment (`TRADE_TELEGRAM_BOT_TOKEN`, `TRADE_TELEGRAM_CHAT_ID`) so
no secret is ever committed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_MANIFEST = _REPO_ROOT / "artifacts" / "frozen" / "freeze_manifest.json"
_DEFAULT_JOURNAL = _REPO_ROOT / "paper_journal"


@dataclass(frozen=True, slots=True)
class TelegramConfig:
    """Telegram notification credentials. Both fields must be set for sends to
    be attempted; otherwise the notifier degrades to a no-op."""

    bot_token: str | None = None
    chat_id: str | None = None

    @property
    def is_configured(self) -> bool:
        return bool(self.bot_token) and bool(self.chat_id)

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> TelegramConfig:
        e = env if env is not None else dict(os.environ)
        return cls(
            bot_token=e.get("TRADE_TELEGRAM_BOT_TOKEN") or None,
            chat_id=e.get("TRADE_TELEGRAM_CHAT_ID") or None,
        )


@dataclass(frozen=True, slots=True)
class PaperTradingConfig:
    # --- HARD MASTER SWITCH ---------------------------------------------
    # Default OFF. Must be flipped deliberately, only after the forward-test
    # gates pass and the operator approves. Even when True, execution is
    # simulated — never a real or testnet order.
    execution_enabled: bool = False

    # --- What to run ----------------------------------------------------
    manifest_path: Path = _DEFAULT_MANIFEST
    symbols: tuple[str, ...] = ("BTCUSDT", "ETHUSDT")

    # --- Simulated account ---------------------------------------------
    initial_equity: float = 10_000.0

    # --- Risk (layered drawdown halts, same engine as backtest) --------
    daily_drawdown_pct: float = 0.035
    weekly_drawdown_pct: float = 0.08
    monthly_drawdown_pct: float = 0.12

    # --- Kill switch ----------------------------------------------------
    # For daily bars a healthy gap between confirmed bars is ~24h; allow a
    # generous margin before declaring the data stale.
    data_staleness_kill_seconds: float = 60.0 * 60.0 * 30.0  # 30h

    # --- Journalling ----------------------------------------------------
    journal_dir: Path = _DEFAULT_JOURNAL

    # --- Notifications --------------------------------------------------
    telegram: TelegramConfig = field(default_factory=TelegramConfig)

    def __post_init__(self) -> None:
        if self.initial_equity <= 0:
            raise ValueError("initial_equity must be positive")
        if not (
            0 < self.daily_drawdown_pct <= self.weekly_drawdown_pct <= self.monthly_drawdown_pct < 1
        ):
            raise ValueError("drawdown limits must satisfy 0 < daily <= weekly <= monthly < 1")
        if not self.symbols:
            raise ValueError("at least one symbol is required")
        if self.data_staleness_kill_seconds <= 0:
            raise ValueError("data_staleness_kill_seconds must be positive")

    @classmethod
    def from_env(
        cls,
        *,
        execution_enabled: bool = False,
        manifest_path: Path | None = None,
        symbols: tuple[str, ...] | None = None,
        journal_dir: Path | None = None,
        env: dict[str, str] | None = None,
    ) -> PaperTradingConfig:
        return cls(
            execution_enabled=execution_enabled,
            manifest_path=manifest_path or _DEFAULT_MANIFEST,
            symbols=symbols or ("BTCUSDT", "ETHUSDT"),
            journal_dir=journal_dir or _DEFAULT_JOURNAL,
            telegram=TelegramConfig.from_env(env),
        )
