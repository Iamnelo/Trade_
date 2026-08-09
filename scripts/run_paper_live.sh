#!/usr/bin/env bash
# Launch the Phase 4c paper-trading loop ARMED (simulated execution ON) against
# the live Bybit public WebSocket feed.
#
# STRICTLY SIMULATED. This never places a real or testnet order — the engine
# only ever uses the in-memory SimulatedVenue. "Armed" means it opens/closes
# SIMULATED positions and records P&L; no venue credentials are used.
#
# Run this on a host that has outbound network access to Bybit
# (stream.bybit.com). It will NOT connect from a network-restricted sandbox.
#
# Telegram (optional): export TRADE_TELEGRAM_BOT_TOKEN and TRADE_TELEGRAM_CHAT_ID
# before running to receive notifications; otherwise notifications are a no-op.
#
# The loop reconnects automatically and runs until you stop it (Ctrl-C). State
# and the tamper-evident journal live under --journal-dir and survive restarts.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

JOURNAL_DIR="${PAPER_JOURNAL_DIR:-$REPO_ROOT/paper_journal}"

echo "Starting ARMED paper trading (SIMULATED). Journal: $JOURNAL_DIR"
exec uv run python -m trade.cli paper run \
  --arm-execution \
  --confirm "ARM PAPER EXECUTION" \
  --journal-dir "$JOURNAL_DIR"
