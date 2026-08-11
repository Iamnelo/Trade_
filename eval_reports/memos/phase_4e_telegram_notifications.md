# Telegram notifications — fix + hardening

## Root cause (why nothing was delivered)

The notification system already existed and was correctly wired into the engine
(`TelegramNotifier` / `NullNotifier` / `build_notifier`, fed by `TelegramConfig`
from env). The failure was **configuration + observability**, not missing code:

`TelegramConfig.from_env()` reads `TRADE_TELEGRAM_BOT_TOKEN` /
`TRADE_TELEGRAM_CHAT_ID` from `os.environ`. **A systemd service does not inherit
your shell environment** — it only sees what the unit declares via
`Environment=` / `EnvironmentFile=`. With those two vars absent from
`trade-paper.service`, `build_notifier()` returned a **silent `NullNotifier`** and
every `notify()` was a no-op. Nothing was logged, so it looked broken.

Two aggravating factors: sends were logged nowhere on failure, and startup gave
no signal about whether Telegram was on.

## What changed (smallest clean fix — no parallel system)

The engine still calls the same `self._notify(text)`. Changes:

- **`src/trade/paper/notifier.py`** — added structured logging on every send
  failure (`telegram_notify_failed` with the HTTP status + Telegram's error
  body, so bad token / wrong chat_id / "bot not started" is visible in
  `journalctl`); added `log_notifier_status()` (a clear `telegram_configured` /
  `telegram_not_configured` line at startup); added **`BackgroundNotifier`**, a
  daemon-thread queue so sends happen OFF the trading loop (non-blocking) and any
  delivery error is logged, never raised (non-fatal). Existing classes unchanged.
- **`src/trade/paper/engine.py`** — emit the previously-missing events reusing
  existing state (no parallel trade tracker): **data feed CONNECTED** (first
  batch, once), **WS RECONNECTED** (`on_ws_reconnect()` hook), **HALT CLEARED**,
  and **CRASH** (`run()` reports the exception to Telegram + journal, then
  re-raises so systemd restarts). Enriched OPEN/EXIT/SCALE/REVERSE messages with
  direction, entry vs exit price (from the cost basis), quantity, realized P&L,
  fee, cumulative P&L, equity, model confidence (from the triggering decision),
  and timestamp. `stop()` flushes a background notifier. **No change** to
  strategy, model, risk, fill accounting, or journal records.
- **`src/trade/cli/paper.py`** — `run` now logs/echoes Telegram-config status,
  wraps the notifier in `BackgroundNotifier`, wires `on_reconnect` to the
  engine, and flushes on exit. Added **`trade paper test-telegram`** to safely
  send one message using the env vars (no trading, no venue).
- **`.env.example`, `deploy/trade-paper.env.example`,
  `deploy/trade-paper.service.example`** — document the two env vars and give a
  reference systemd unit + EnvironmentFile (the missing `EnvironmentFile=` line
  is the actual fix on the VPS).

## Duplicates / safety

- Restart replays the journal to rebuild state via `oms.apply_fill` directly —
  **not** through the notifying path — so historical fills are never re-notified
  (test: `test_no_duplicate_notifications_on_restart`).
- Bybit resends only new confirmed bars after a reconnect, so no duplicate
  trades/notifications; the reconnect message is a single event.
- **Paper-only is unchanged**: the engine still constructs only the in-memory
  `SimulatedVenue`; no real or testnet path is introduced. `--arm-execution`
  remains simulated-only.

## Tests

`tests/test_paper_notifications.py` (12): Telegram success (injected transport),
API failure is non-fatal, `BackgroundNotifier` delivery + error isolation,
failure does not stop trading, OPEN + EXIT content, halt + halt-cleared,
reconnect, feed-connected-once, no duplicate on restart. Full suite: 562 pass.
