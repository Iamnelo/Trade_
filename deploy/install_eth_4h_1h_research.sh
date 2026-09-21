#!/usr/bin/env bash
# Install isolated ETH 4H + 1H rejected-candidate simulated-paper services.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEPLOY_USER="$(id -un)"
DAILY_REPO="$(systemctl show trade-paper.service --property=WorkingDirectory --value 2>/dev/null || true)"
DAILY_ENV_FILE="$(systemctl show trade-paper.service --property=EnvironmentFiles --value 2>/dev/null \
  | awk '{print $1}' | sed 's/^-//' || true)"
DAILY_ENV_FILE="${DAILY_ENV_FILE:-/etc/trade-paper.env}"

if command -v uv >/dev/null 2>&1; then
  UV_BIN="$(command -v uv)"
elif [[ -x /home/whyfavour/.local/bin/uv ]]; then
  UV_BIN="/home/whyfavour/.local/bin/uv"
else
  echo "missing uv executable; install uv for $DEPLOY_USER before deploying" >&2
  exit 2
fi

if [[ -n "$DAILY_REPO" && "$REPO_ROOT" == "$DAILY_REPO" ]]; then
  echo "REFUSING: install from a separate clone, not Daily V1 repository: $DAILY_REPO" >&2
  exit 2
fi

daily_before="$(systemctl is-active trade-paper.service 2>/dev/null || true)"
hour12_before="$(systemctl is-active trade-paper-eth-12h.service 2>/dev/null || true)"

install_service() {
  local timeframe="$1"
  local service_name="trade-paper-eth-${timeframe}-research.service"
  local launcher="$REPO_ROOT/scripts/run_paper_eth_${timeframe}_research.sh"
  local manifest="$REPO_ROOT/artifacts/frozen/eth_${timeframe}_research/freeze_manifest.json"
  local journal="$REPO_ROOT/paper_journal_eth_${timeframe}_research"
  local prefix="RESEARCH-${timeframe^^}-REJECTED-CANDIDATE"
  local unit_file

  if [[ ! -f "$launcher" || ! -f "$manifest" ]]; then
    echo "missing launcher or frozen manifest for $timeframe research service" >&2
    exit 2
  fi

  unit_file="$(mktemp)"
  cat >"$unit_file" <<EOF
[Unit]
Description=Trade ETH ${timeframe^^} rejected-candidate research paper bot (simulated)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=$DEPLOY_USER
WorkingDirectory=$REPO_ROOT
EnvironmentFile=-$DAILY_ENV_FILE
Environment=TRADE_TELEGRAM_PREFIX=$prefix
Environment=PAPER_${timeframe^^}_RESEARCH_JOURNAL_DIR=$journal
Environment=PATH=$(dirname "$UV_BIN"):/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
ExecStart=/usr/bin/env bash $launcher
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF
  sudo install -o root -g root -m 0644 "$unit_file" "/etc/systemd/system/$service_name"
  rm -f "$unit_file"
}

"$UV_BIN" sync --frozen
install_service "4h"
install_service "1h"
sudo systemctl daemon-reload
sudo systemctl enable trade-paper-eth-4h-research.service
sudo systemctl enable trade-paper-eth-1h-research.service
sudo systemctl restart trade-paper-eth-4h-research.service
sudo systemctl restart trade-paper-eth-1h-research.service

daily_after="$(systemctl is-active trade-paper.service 2>/dev/null || true)"
hour12_after="$(systemctl is-active trade-paper-eth-12h.service 2>/dev/null || true)"
if [[ "$daily_before" != "$daily_after" || "$hour12_before" != "$hour12_after" ]]; then
  echo "REFUSING: protected service state changed" >&2
  echo "Daily V1: before=$daily_before after=$daily_after" >&2
  echo "ETH 12H: before=$hour12_before after=$hour12_after" >&2
  exit 3
fi

systemctl --no-pager --full status trade-paper-eth-4h-research.service
systemctl --no-pager --full status trade-paper-eth-1h-research.service
echo "Daily V1 unchanged: ${daily_after:-not-found}"
echo "ETH 12H unchanged: ${hour12_after:-not-found}"
