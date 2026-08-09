# Phase 4d — paper execution ARMED (strictly simulated) + continuous operation

You asked to arm Phase 4c and run it continuously with live data, full record
keeping (including reason and P&L), Telegram notifications, and daily/weekly
reports — while keeping everything strictly paper-only and never touching a
real or testnet path. This memo records exactly what was done and the one
environment constraint that governs *where* the live loop runs.

## What "armed" means here

Arming is an operational act, not a code change: the master switch stays
`False` by default in code, and the run is armed at invocation with
`--arm-execution --confirm "ARM PAPER EXECUTION"`. Armed = the engine opens and
closes **simulated** positions and books simulated P&L. It remains paper-only
by construction — the engine only ever builds the in-memory `SimulatedVenue`;
there is no code path, armed or not, to the signed Bybit venue or a testnet.
No venue credentials are read. This is unchanged and asserted at startup.

Nothing about the models was touched: frozen artifacts, features, thresholds,
risk gates, and strategy logic are all exactly as validated. No retraining, no
tuning on live results — the engine only ever *loads and replays* the frozen
winners.

## What was added for continuous operation

- **Richer records.** Every decision now carries an explicit `reason`
  (e.g. `argmax=up p=0.574 ≥ θ=0.55 → LONG`, or
  `P(up)=0.48 within [0.45,0.55] → FLAT`). Every closing fill carries
  per-trade `realized_pnl` and a running `cumulative_realized_pnl`, via a
  signed average-cost basis tracker. Alongside the existing confidence,
  probabilities, position, equity, and timestamps.
- **Daily & weekly performance reports.** Reconstructed purely from the
  journal (single source of truth): window return, realized P&L, max drawdown,
  decisions, fills (opens/exits), wins/losses, win rate, average confidence,
  per-symbol activity, halts, and cumulative figures. The engine emits a report
  automatically at each UTC day / ISO-week rollover (writing markdown under
  `<journal>/reports/` and sending a one-line Telegram summary), and you can
  generate one on demand:
  `trade paper report --period daily|weekly`.
- **Restart durability.** After every bar the engine snapshots state; on
  startup it rebuilds positions, equity, cost basis, and counters by replaying
  the tamper-evident journal, and restores the recent bar buffer + pending
  orders from `state.json`. A killed/redeployed process resumes exactly where
  it left off — verified by a test that a fresh engine over the same journal
  reproduces equity, positions, and fill count.

## The environment constraint — where the live loop runs

This managed sandbox has **no outbound network path to Bybit** (the egress
proxy allows only package registries + Anthropic; `api.bybit.com` returns
`403 CONNECT tunnel failed`). It is the same wall that forced every data
backfill in this project to be done locally. So the live WebSocket loop
**cannot connect from here**, and this container is ephemeral besides — a
"continuous" process would die when the session ends. The live loop must run on
a host that has Bybit access and stays up (your machine or a deployment host).

Launch it there with one command:

```
export TRADE_TELEGRAM_BOT_TOKEN=...   # optional, for notifications
export TRADE_TELEGRAM_CHAT_ID=...
./scripts/run_paper_live.sh
```

That runs armed (simulated) against `wss://stream.bybit.com/v5/public/linear`,
reconnects automatically, journals every decision/trade, emits daily/weekly
reports, and runs until you stop it. Note the winners are **daily-bar** models,
so the loop makes at most one decision per symbol per day (at the 00:00 UTC
daily close) — it will sit idle between closes by design.

## Verified locally (offline, no network)

Armed replay of the frozen winners over the committed 5y daily bars: 3,652
decisions, 1,638 simulated fills, decisions carry `reason`, exits carry
`realized_pnl` + `cumulative_realized_pnl`, daily/weekly reports generate, and
the sha256 audit chain verifies. (That replay runs the models over their own
training data, so its P&L is meaningless in-sample — it is a plumbing check,
not a performance result.) 22 paper unit tests pass, including realized-P&L
cost-basis math, report generation, and restart resumption.

## Boundaries that still hold

- **Strictly simulated. No real or testnet order path exists.**
- **Models are frozen** — not retrained or tuned on live paper results.
- **Phase 5c forward-test is a separate exercise** and still must be evaluated
  when enough genuinely-unseen data exists (our bars end 2026-08-07). Paper
  trading gathers a live simulated track record; it is not a substitute for the
  forward-test gates, and neither one moves real money.
