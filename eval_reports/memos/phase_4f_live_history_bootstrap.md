# Live history bootstrap — fix for permanent WARMUP

## Root cause

The live paper engine accumulated feature history **only** from newly-arrived
WebSocket candles. On daily bars the longest-lookback feature per winner needs
**87 bars (BTC, `macd_hist@12_26_9`) / 121 bars (ETH, `vol_regime@20_120`)**, so a
freshly-started service would emit only WARMUP decisions for ~87/121 calendar
days before its first trade. The 4 WARMUP / 0-fill journal entries were the
correct symptom of a **missing bootstrap step**, not a feature/model/journal bug.
(The earlier replay runs traded because `--replay` pre-loads the full CSV history
into the buffers; the live feed started cold.)

## Fix (minimal, live-path only)

Seed the engine's history buffers at live startup with the last ~250 **closed**
daily candles from Bybit's **public** REST endpoint, reusing the existing
backfill stack. Nothing about the models, features, thresholds, timeframe, risk
manager, or fill accounting changes — this supplies the genuine history the
features already require; it does **not** lower the warmup requirement.

- **`src/trade/paper/bootstrap.py`** (new) — `fetch_seed_bars()` reuses
  `BybitPublicClient` + `backfill_bybit_klines` (public market data; no API keys,
  no order/testnet endpoints). `end` is pinned to the current interval boundary
  so the in-progress candle is excluded — **closed candles only**. `client` is
  injectable for tests.
- **`src/trade/paper/engine.py`** — `seed_history(bars)`: fills buffers +
  `latest_close`, **deduplicated by event_time (live bars win ties), sorted,
  idempotent** (skips a symbol already holding ≥ `max_lookback`, so restarts
  don't refetch). Runs **no** decisions, fills, or trade notifications — seeded
  bars are warmup context, not events. Journals a `history_seeded` record.
- **`src/trade/cli/paper.py`** — live branch fetches + seeds before the WS loop;
  `--bootstrap-bars` (default 250, `0` disables). Best-effort: a REST hiccup logs
  and continues (warm up from live bars). **Replay path unchanged** — no bootstrap.

## Safety / invariants preserved

- Strictly paper/simulated: only `BybitPublicClient` public GETs; no real or
  testnet order path introduced. `--arm-execution` semantics unchanged.
- Daily timeframe unchanged (matches the daily-trained frozen winners).
- Warmup thresholds unchanged; frozen models / strategy / risk / fills untouched.

## Tests (`tests/test_paper_bootstrap.py`, 5)

- seeded engine's **first live bar decides (not WARMUP)** — the core proof;
- `seed_history` idempotent / restart-safe (second seed adds 0);
- dedup + sort, existing bar wins on event_time tie;
- interval boundary excludes the in-progress candle;
- `fetch_seed_bars` returns only closed candles (injected fake client, no network).

Full suite: **567 passed**.

## What still holds

Bootstrap gives the live bot genuine historical context so it participates from
day one. It does not change the forward-test story: the Phase 5c forward-test
remains a separate exercise, and paper trading stays strictly simulated.
