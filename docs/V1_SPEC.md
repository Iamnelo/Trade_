# Version 1 Specification — Locked

Date locked: 2026-08-05
Status: FROZEN. Changes require explicit sign-off from the product owner.

Version 1's purpose is to prove — end-to-end, with real evidence — that a
disciplined, explainable, risk-managed AI signal service can be operated on
crypto perpetuals without blowing up. It is deliberately narrow. Anything not
listed here is out of scope for V1 and belongs in V2+ enhancements at the
bottom of this file.

## Product

- **Business model**: Signal service, rest-of-world subscribers only. The
  platform does NOT trade on subscribers' accounts. Users receive signals and
  place their own trades.
- **Delivery**: Web dashboard, email, and Telegram bot (Phase 9).
- **Explainability**: Every signal carries a confidence score, a top-N feature
  attribution (SHAP), and a "why now" summary derived from the model's inputs.

## Trading

- **Venue**: Bybit v5 API, USDT-margined linear perpetuals. All execution and
  paper trading goes through Bybit. No other venues in V1.
- **Instruments (initial)**: BTCUSDT, ETHUSDT. SOLUSDT is deferred to V1.1,
  unlocked only after the model survives one full drawdown regime on BTC/ETH.
- **Decision cadence**: Hourly. Target holding periods 4 hours to several days
  (swing).
- **Position policy**: At most one open position per symbol. Portfolio gross
  exposure never exceeds 1x account equity.
- **Leverage**: Governed by position sizing (fractional Kelly, cap 1/4 Kelly).
  No fixed leverage setting; the sizing module computes contract quantity from
  equity and risk budget.

## Data

- **Historical**: Maximum available from Bybit for the traded venue,
  cross-validated against Binance perps and one reference source
  (CryptoCompare or Kaiko). Raw parquet in S3-compatible object storage
  (MinIO locally); curated series in a time-series database (QuestDB).
- **Live**: Bybit v5 WebSocket for market data with REST reconciliation and
  gap detection.
- **Alt data (V1)**: Funding rate, open interest, order-book imbalance from
  Bybit L2.
- **Alt data explicitly deferred to V2**: News, social sentiment, on-chain
  metrics.
- **Time discipline**: UTC everywhere. Every record carries `event_time`
  (source-reported) and `ingest_time` (platform-received) columns.

## Risk management

- **Per-trade stop**: Structural, ATR-based, set at signal time and enforced
  by the OMS.
- **Portfolio drawdown limits (from V1 equity high)**:
  - Daily: 3.5% — trading halted for the calendar day (UTC).
  - Weekly: 8% — trading halted until the next Monday 00:00 UTC.
  - Monthly: 12% — trading halted; requires operator sign-off to resume.
- **Kill switches** (any trigger auto-flattens and halts):
  - Market-data staleness > 90 seconds on the signal timeframe.
  - Exchange API error rate > 20% over any 60-second window.
  - Local position vs. exchange position mismatch after two reconciliation
    attempts.
  - Model prediction is out of the training distribution (PSI-based).
  - AI Health Score drops below the configured floor.

## Feature Store

Two-tier, custom in V1; Feast migration path preserved for V2.

- **Offline (research + training)**: Content-addressed parquet keyed by
  `(feature_set_id, entity_id, event_time)`, queried via **DuckDB** — embedded,
  SQL, native parquet, no server. Every feature declares a name + version, its
  input columns and lookback, and a formula that is a pure function of inputs
  at times <= t.
- **Online (live signal delivery, Phase 4+)**: **Redis** hash-per-entity for
  O(1) latest-feature reads. The online writer subscribes to feature-emitter
  streams and updates Redis; training and inference share the exact same
  feature code, never a rewritten copy.
- **Point-in-time correctness (HARD REQUIREMENT)**: every training pull
  specifies `as_of_time`; the store returns only features whose
  `event_time + availability_delay <= as_of_time`. The only supported API for
  building a training set is `feature_store.point_in_time_join(labels,
  as_of_column="event_time")`. There is no "latest features" convenience for
  training. Leakage is the loud path, not the silent one.
- **Feature contract tests (HARD REQUIREMENT)**: every feature module ships
  with a `hypothesis`-based property test asserting that for any two input
  histories identical up to time t, the feature value at t is identical. CI
  fails without the test.

## Market Replay Engine

The unified engine used for backtesting, walk-forward validation, and — via
the same interfaces — paper and live trading.

- **Same interfaces as live**: `MarketDataSource`, `RiskManager`,
  `OrderManager`, `ExecutionVenue`. Only the exchange adapter differs across
  backtest / paper / live.
- **Point-in-time correct queries**: the engine holds a monotonic `sim_clock`;
  every data query returns only records with `event_time <= sim_clock`. The
  API surface enforces this, not convention.
- **Deterministic clock advance**: events emitted in `event_time` order, ties
  broken by a stable secondary key. Replay of the same manifest ID always
  produces bit-for-bit identical PnL.
- **Bar-and-tick modes**: V1 runs on bars (hourly signal cadence); the engine
  supports finer replay (1m or trades) for fill realism without a rewrite.
- **Realistic fills**: fill logic lives in the `ExecutionVenue` adapter, not
  the strategy. Slippage curve is a parameter, updated from live fills during
  Phases 5-6.

Migration to Nautilus Trader is a V2 option and NOT scoped for V1.

## Reproducibility and versioning

- **DatasetManifest**: SHA-256 per partition, coverage bounds. Every backfill
  produces one.
- **FeatureSetManifest**: SHA-256 per partition, `derived_from` list of
  dataset manifest IDs, `feature_spec_sha256`, `code_git_sha`.
- **ExperimentRecord** (in MLflow): every run auto-logs
  `dataset_manifest_ids`, `feature_manifest_ids`, `model_config_sha256`,
  `code_git_sha`, `python_lockfile_sha`.
- **Reproducibility hash (HARD REQUIREMENT)**: for any model release,
  `sha256(dataset || features || model_config || code_sha || lockfile_sha)`.
  Two runs with the same reproducibility hash MUST produce byte-identical
  models. Release is blocked if the hash cannot be reproduced from committed
  artifacts.

## Models

- **Primary**: LightGBM classifier per symbol per timeframe, trained on the
  signal timeframe with features aggregated from context timeframes. Features
  consumed EXCLUSIVELY through the Feature Store's PIT-join API.
- **Explainability**: SHAP (TreeExplainer) values logged for every prediction.
- **Calibration**: Isotonic regression fitted on a held-out set; reliability
  diagrams monitored; recalibration triggered on ECE drift.
- **Labels**: Triple-barrier (Lopez de Prado) as the default; documented
  leakage safeguards.
- **Validation**: Walk-forward with purged/embargoed k-fold for hyperparameter
  and feature selection. No random-shuffle CV anywhere in the training
  pipeline.
- **Model registry**: MLflow. Production loads only signed, registry-blessed
  artifacts.
- **Research/production separation**: Two distinct code paths and two distinct
  dependency sets. The production runtime cannot import research notebooks or
  ad-hoc scripts.

## AI Health Score

Composite gauge (0-100) surfaced on the ops dashboard. Sub-scores are what the
risk manager acts on. Components:

- Calibration: Expected Calibration Error and Brier score on rolling window.
- Feature drift: PSI/KS vs. the training distribution, per feature and
  aggregate.
- Prediction drift: KS on the model-output distribution.
- Rolling hit rate and rolling PnL vs. training-time expectation.
- Confidence coherence: does high-confidence actually outperform low-confidence?
- Operational: latency, data-quality flags, dropped ticks.

## Backtesting and benchmarking

- **Engine**: See "Market Replay Engine" section above. The same event-driven
  engine drives every backtest and walk-forward evaluation.
- **Costs modelled**: taker fee, funding accrual, slippage curve (updated
  from live fills during paper/live phases).
- **Benchmarks required for every strategy release**:
  - Buy-and-hold BTC
  - Buy-and-hold ETH
  - MA cross (documented parameters)
  - Momentum 12-1
  - Risk-parity BTC/ETH
  - Random-signal-with-identical-risk-overlay (control)
- **Metrics reported**: Sharpe, Sortino, Calmar, max drawdown, Ulcer index,
  hit rate, turnover, cost-adjusted Sharpe, tail metrics (CVaR).

## Observability and audit

- Structured logging (JSON in non-local environments) via structlog.
- Prometheus metrics; Grafana dashboards for system, strategy, and model
  health.
- Loki for logs; Sentry for exceptions.
- **Event-sourced audit log**: append-only, content-hashed record of
  `(signal, features_snapshot, decision, order, fill, PnL, kill_switch_events)`.
  Every decision — including skipped trades — is reproducible from the log.

## Security

- Bybit API keys: withdrawal-disabled, IP-whitelisted, rotated on a defined
  schedule.
- Secrets stored via Doppler or AWS Secrets Manager. No secrets in git, ever.
- Least-privilege service accounts. Human overrides recorded in the audit log.

## Legal (V1 rest-of-world only)

- V1 does NOT accept US, EU, or UK subscribers. Geo-block at the subscription
  tier.
- Risk disclosures and terms of service are required before any signal is
  delivered.
- Securities counsel review is required before Phase 9 goes live to real
  users, even in the rest-of-world scope.
- No custody, no fiat on-ramps, no discretionary trading of user funds.

## "V1 is done" — deliverable definition

1. Paper trading has run continuously for at least 3 months including one BTC
   drawdown of at least 15% intraweek.
2. Small-live-capital phase has run at least 3 months with realized
   risk-adjusted return within 30% of paper.
3. Cost-adjusted Sharpe consistently beats every benchmark listed above across
   the evaluation window. If not, V1 does not ship to subscribers.
4. Kill-switch and reconciliation systems have been exercised in production
   and pass a documented incident-response drill.
5. Legal review is complete for at least one target rest-of-world jurisdiction.
6. Subscription platform (Phase 9) has passed a closed beta with at least five
   external users for one month.
7. Every released model has a matching reproducibility hash and can be
   rebuilt bit-for-bit from committed artifacts.
8. Every feature in production ships with a passing contract test in CI.

## V2+ enhancements (NOT V1)

Recorded here so we do not lose them:

- News and social-sentiment features (with LLM-based enrichment, versioned,
  timestamped on arrival).
- On-chain features (Glassnode / Coin Metrics).
- Additional symbols beyond BTC/ETH (SOL first, after the V1.1 gate).
- Additional venues (multi-venue routing, cross-exchange basis).
- Deep learning sequence models (only if boosted-tree baseline plateaus).
- Multi-strategy portfolio construction with correlation-aware sizing.
- US / EU / UK subscriber support (requires new legal work).
- Copy-trading integration with the Bybit copy-trading product.
- Regime detector (HMM / changepoint) driving per-regime model routing.
- Migration to Nautilus Trader as the unified backtest/live engine.
