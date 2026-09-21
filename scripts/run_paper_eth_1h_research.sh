#!/usr/bin/env bash
# Rejected ETH 1H candidate: isolated simulated-paper research only.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

MANIFEST="$REPO_ROOT/artifacts/frozen/eth_1h_research/freeze_manifest.json"
JOURNAL_DIR="${PAPER_1H_RESEARCH_JOURNAL_DIR:-$REPO_ROOT/paper_journal_eth_1h_research}"

echo "Starting RESEARCH 1H ETH paper trading (SIMULATED). Journal: $JOURNAL_DIR"
exec uv run python -m trade.cli paper run \
  --manifest "$MANIFEST" \
  --symbol ETHUSDT \
  --arm-execution \
  --confirm "ARM PAPER EXECUTION" \
  --journal-dir "$JOURNAL_DIR"
