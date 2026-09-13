#!/usr/bin/env bash
# Dedicated ETH 12H shadow-paper launcher. Simulated execution only.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

MANIFEST="$REPO_ROOT/artifacts/frozen/eth_12h_challenger/freeze_manifest.json"
JOURNAL_DIR="${PAPER_12H_JOURNAL_DIR:-$REPO_ROOT/paper_journal_eth_12h}"

echo "Starting ETH 12H shadow paper trading (SIMULATED). Journal: $JOURNAL_DIR"
exec uv run python -m trade.cli paper run \
  --manifest "$MANIFEST" \
  --symbol ETHUSDT \
  --arm-execution \
  --confirm "ARM PAPER EXECUTION" \
  --journal-dir "$JOURNAL_DIR"
