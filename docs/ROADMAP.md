# Implementation Roadmap — Version 1

Sized for a small team (1-3 engineers). Autonomous execution proceeds through
these phases without per-step approval, escalating only on:

- A major architectural change.
- A technology choice changing significantly (ML framework, database, engine).
- A legal or security concern that could affect the product.
- A materially better design surfacing after V1 lock.
- A critical blocker that requires changing the roadmap.
- A feature outside the agreed V1 scope.

## Phase 0 — Foundation

- Monorepo layout with `src/trade`, `tests`, `scripts`, `ops`, `docs`.
- Python 3.12 pinned via `.python-version`; deps managed by `uv`; packaged
  via `pyproject.toml` + hatchling.
- Ruff (lint + format), mypy strict, pytest with coverage.
- Pre-commit hooks; GitHub Actions CI: lint, type, test on every push and PR.
- Docker Compose dev stack: QuestDB, MinIO, Redis, Prometheus, Grafana, Loki.
- Structured logging (structlog), Prometheus metrics primitive, typed config
  (pydantic-settings), UTC clock utilities.
- Bybit v5 public REST client (read-only, unauthenticated) with
  respx-mocked tests.
- Manual `scripts/hello_bybit.py` spike proving end-to-end data pull.

**Exit criteria**: `make ci` green locally; GitHub Actions green on push;
`make hello` fetches real BTC/ETH klines from Bybit.

## Phase 1 — Data platform

- Historical backfill (Bybit + Binance + reference source), stored raw in
  parquet on S3/MinIO, curated into QuestDB.
- Live WebSocket ingest (public trades, klines, orderbook L2 snapshots,
  funding), with REST reconciliation and gap detection.
- Data-quality checks: gap percentage, staleness, price sanity band,
  cross-source deltas. Publish DQ metrics + Grafana dashboard.
- UTC discipline enforced end-to-end; every row carries `event_time` and
  `ingest_time`.
- Reproducible dataset snapshots (content-hashed parquet manifests).

**Exit criteria**: at least 3 years of clean 1m OHLCV + funding + trades for
BTC/ETH stored, live tick stream running for 7 continuous days without a
gap alert, DQ dashboard live.

## Phase 2 — Market Replay Engine + benchmarks

- Event-driven Market Replay Engine behind the same interfaces
  (`MarketDataSource`, `RiskManager`, `OrderManager`, `ExecutionVenue`) that
  paper/live will use. Monotonic sim clock; PIT-correct queries enforced at
  the API surface; deterministic replay of any manifest ID.
- Realistic costs: taker fee, funding accrual, slippage curve (initialised
  conservatively, updated from live fills later).
- Walk-forward validation framework with purged/embargoed k-fold.
- Benchmark suite: buy-hold BTC, buy-hold ETH, MA cross, 12-1 momentum,
  risk-parity BTC/ETH, random-signal-with-risk-overlay control.
- Metrics library: Sharpe, Sortino, Calmar, max DD, Ulcer, hit rate, turnover,
  cost-adjusted Sharpe, CVaR.
- MLflow tracking wired in; research notebook conventions documented.

**Exit criteria**: benchmark suite reproduces expected returns within a
tolerance; framework used to publish a pre-model baseline report; identical
run twice is bit-for-bit reproducible.

## Phase 2.5 — Feature Store + versioning + reproducibility (~2 weeks)

- **Offline Feature Store**: DuckDB over content-addressed parquet keyed by
  `(feature_set_id, entity_id, event_time)`. Feature module contract:
  declared name, version, inputs, lookback, formula (pure function of inputs
  at times <= t).
- **PIT-only training API (HARD REQUIREMENT)**: the ONLY supported way to
  build a training set is `feature_store.point_in_time_join(...)`. No latest-
  features shortcut. Adding one is a spec violation.
- **Feature contract tests (HARD REQUIREMENT)**: every feature module ships
  with a `hypothesis` property test proving leakage-free behaviour. CI blocks
  merge without it.
- **FeatureSetManifest**: SHA-256 per partition, `derived_from` dataset
  manifest IDs, `feature_spec_sha256`, `code_git_sha`.
- **ExperimentRecord**: MLflow autolog of dataset/feature manifest IDs,
  model config hash, code git sha, python lockfile sha.
- **Reproducibility hash (HARD REQUIREMENT)**: computed for every training
  run; a model may not be released if the hash cannot be reproduced from
  committed artifacts.

**Exit criteria**: two independent runs against the same reproducibility
hash produce byte-identical model artifacts; contract test framework in CI;
sample feature (e.g., RSI-14) implemented as reference.

**BLOCKS Phase 3.**

## Phase 3 — Feature engineering and first model

- Feature set v1, materialised into the Phase 2.5 Feature Store: returns,
  volatility (realized + Parkinson), RSI/MACD/ATR family, order-book
  imbalance from L2 snapshots, funding-rate features, cross-asset (BTC-ETH
  spread, dominance). Every feature has a passing contract test.
- Label design: triple-barrier (Lopez de Prado) with documented leakage
  safeguards.
- Model v1: LightGBM classifier per (symbol, timeframe); direction +
  confidence; isotonic calibration on held-out. Trained EXCLUSIVELY through
  `feature_store.point_in_time_join(...)`. Every run emits a reproducibility
  hash.
- SHAP explanations logged per prediction; top-N features surfaced to the
  future signal payload.
- Position sizing: fractional Kelly capped at 1/4 Kelly, or fixed-fraction
  fallback.

**Exit criteria**: walk-forward backtest report showing model + risk overlay
vs. the full benchmark suite over at least 2 years, including a loss regime.
If the model does not beat the random-signal-with-risk-overlay control on
cost-adjusted Sharpe, do not advance — iterate on features.

## Phase 4 — Execution engine and risk manager

- OMS: idempotent client order IDs, exponential-backoff retry, cancel-all on
  shutdown, sanity gates on every order.
- Risk manager: per-trade stop, daily/weekly/monthly DD, kill switches,
  drift monitor, staleness monitor. AI Health Score computed and published.
- Reconciliation loop (local position/PnL vs. exchange, every N seconds).
- Event-sourced append-only audit log with content-hashed records.
- Same code path drives backtest, paper, and live via an exchange-adapter
  interface (`MarketDataSource` + `ExecutionVenue`).

**Exit criteria**: paper trading running from a live signal; dashboards live
for system, strategy, and model health; dry-run kill-switch drills pass.

## Phase 5 — Paper trading (at least 3 months, wall-clock)

- Continuous paper trading against Bybit testnet or spot-with-zero-capital
  behaviour.
- No production code path exists that is not exercised in paper.
- Weekly review: paper-vs-model-expected divergence, DQ incidents, drift.
- Iterate on features/model only via the WFO framework — no adjustments made
  because "paper looks bad this week".

**Exit criteria** (all required):

- At least 3 months elapsed including one BTC drawdown of at least 15%
  intraweek.
- Realized Sharpe (net of modelled costs) >= 0.6x the backtest.
- No unplanned kill-switch trips in the last 30 days.
- Slippage residual < 2x the modelled slippage.

## Phase 6 — Live micro-capital (at least 3 months)

- Real capital small enough that total loss is a rounding error ($100-$500).
- Compare live fills vs. paper for slippage and latency; update the fill
  model from real data.
- Add second symbol only after 1 month clean on the first.

**Exit criteria**: live risk-adjusted return within 30% of paper; no
reconciliation breaks; no kill-switch trips in last 30 days.

## Phase 7 — Model iteration and alt-data (parallel to 5-6)

- Add funding-rate model as an input, then on-chain features. News and
  sentiment stay deferred to V2.
- Champion-challenger: challenger runs in shadow (logged decisions only) and
  never touches capital until it wins on a preregistered metric across a
  preregistered window.

## Phase 8 — Legal and business model (start in parallel with Phase 5)

- Securities counsel in each target rest-of-world jurisdiction.
- Terms of service, risk disclosures, marketing rules.
- Geo-blocking design for US/EU/UK.
- No custody, no fiat on-ramps — reconfirm this stays out of V1.

## Phase 9 — Subscription platform (4-8 weeks, after 8)

- Next.js + Clerk (auth) + Stripe (billing).
- Signal delivery: web dashboard, webhook, email, Telegram bot.
- User-facing dashboards: signal history, hypothetical performance, model
  health score, prominent disclaimers.
- Closed beta with at least 5 external users for at least 1 month.

## Phase 10 — Steady-state operations

- Weekly model-health review, monthly re-training window.
- Quarterly stress-test replay of historical shocks (2020-03, 2022 LUNA week,
  2022 FTX week, plus new material shocks as they occur).
- Postmortems mandatory for any kill-switch trip.
