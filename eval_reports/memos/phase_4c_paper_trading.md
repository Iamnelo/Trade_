# Phase 4c — integrated paper-trading system (execution disabled)

Built the full paper-trading loop you asked for: it uses the frozen winner
artifacts, consumes live Bybit data, generates predictions, manages simulated
positions, handles exits automatically, records every decision and trade in a
tamper-evident journal, and sends Telegram notifications. **Execution is off
behind a hard master switch, and no real or testnet order can be placed.**

## Two independent safety guarantees

1. **Hard master switch — `execution_enabled`, default `False`.**
   With it off, the engine still consumes bars, predicts, journals every
   decision, and notifies — but it never submits a simulated order, so no
   position is ever opened or closed and equity stays flat. Turning it on is a
   deliberate act; the CLI additionally requires the exact phrase
   `--confirm "ARM PAPER EXECUTION"` alongside `--arm-execution`, so a stray
   flag can't arm it.
2. **Paper-only by construction.** The engine builds and uses only the
   in-memory `SimulatedVenue`. There is no code path — armed or not — to the
   signed Bybit venue. This is asserted at construction. Even fully armed,
   every fill is simulated.

## How it works (and why it's consistent with what was gated)

The engine reuses the exact machinery the backtest and forward-test run on:

- **Live data**: `BybitKlineStream` (WebSocket, confirmed bars only).
- **Predictions**: the frozen winners loaded from `freeze_manifest.json`,
  driving the *same* `ModelDrivenStrategy` / `BinaryModelDrivenStrategy` used in
  WFO and the forward test. Given the same bars, paper decisions match the
  validated ones bit-for-bit.
- **Execution/accounting**: the same `SimulatedVenue` + `OrderManager` +
  `RiskManager` as `run_backtest`. Orders decided at a bar's close fill at the
  next bar's open — the look-ahead-safe convention.
- **Exits are automatic and identical to the validated path**: the strategy
  re-decides every bar; when conviction drops below θ it targets flat (→ exit)
  and when it flips it reverses. I deliberately did **not** bolt on a bespoke
  TP/SL layer, because that would make paper behaviour diverge from the exact
  strategy the forward-test gates authorize. Exit fills are surfaced explicitly
  in the journal and notifications (OPEN / EXIT / REVERSE / SCALE).
- **Risk + kill switch**: layered daily/weekly/monthly drawdown halts (same as
  live risk limits) plus a data-staleness kill switch. When halted, only
  flatten-to-zero targets pass.
- **Journal**: every decision, order, fill, exit, halt, and lifecycle event is
  written to a sha256-chained `AuditLog` (tamper-evident, `verify()`-able) and a
  human-readable `decisions.jsonl`.
- **Notifications**: `TelegramNotifier` (zero-dependency `urllib`). Sends are
  best-effort and non-fatal — a network failure can never disturb the loop. If
  no Telegram credentials are configured it degrades to a silent no-op.

## Using it

```
# Observe-only (default). Live data, predictions, journal, notifications —
# but zero simulated trades:
uv run python -m trade.cli paper run

# Offline dry run against committed bars (no network):
uv run python -m trade.cli paper run --replay BTCUSDT_D_5y.csv --replay ETHUSDT_D_5y.csv

# Inspect / verify the journal:
uv run python -m trade.cli paper status
uv run python -m trade.cli paper verify

# Arm SIMULATED execution — only after the forward-test gates pass + approval:
uv run python -m trade.cli paper run --arm-execution --confirm "ARM PAPER EXECUTION"
```

Telegram: set `TRADE_TELEGRAM_BOT_TOKEN` and `TRADE_TELEGRAM_CHAT_ID` in the
environment (never committed).

## Verified locally

- **Observe mode** over the real frozen winners on 3,652 daily bars:
  3,652 decisions, **0 fills**, equity untouched — the disabled posture holds
  end-to-end.
- **Armed mode** (dry run): trades and records OPEN/EXIT events; the audit
  chain verifies. (The dry-run replays the models over their own training data,
  so its P&L is meaningless in-sample — a plumbing check, not a result.)
- Arming is refused without the exact confirm phrase.
- 17 unit tests: master-switch-off = zero orders, armed trading with exits,
  paper-only venue guard, kill-switch staleness halt, notifier formatting +
  non-fatal failure, journal chain integrity, CLI arming guard.

## Status and the line we do not cross

The system is **built, tested, and wired**, sitting in the observe-only
posture. It is **not** armed and must not be armed until:

1. The Phase 5c forward-test gates are evaluated on genuinely-unseen data
   (which does not exist yet — our bars end 2026-08-07), and
2. You explicitly approve activation.

Even after approval, "armed" means simulated paper execution to gather a live
track record — never real money, and never a real or testnet order.
