#!/usr/bin/env bash
# Install the isolated ETH 12H simulated-paper systemd service.

set -euo pipefail

SERVICE_NAME="trade-paper-eth-12h.service"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEPLOY_USER="$(id -un)"
UNIT_PATH="/etc/systemd/system/$SERVICE_NAME"
LAUNCHER="$REPO_ROOT/scripts/run_paper_eth_12h_shadow.sh"
MANIFEST="$REPO_ROOT/artifacts/frozen/eth_12h_challenger/freeze_manifest.json"
DAILY_REPO="$(systemctl show trade-paper.service --property=WorkingDirectory --value 2>/dev/null || true)"

if command -v uv >/dev/null 2>&1; then
  UV_BIN="$(command -v uv)"
elif [[ -x /home/whyfavour/.local/bin/uv ]]; then
  UV_BIN="/home/whyfavour/.local/bin/uv"
else
  echo "missing uv executable; install uv for $DEPLOY_USER before deploying" >&2
  exit 2
fi

if [[ -n "$DAILY_REPO" && "$REPO_ROOT" == "$(realpath "$DAILY_REPO")" ]]; then
  echo "REFUSING: install from a separate clone, not the Daily V1 repository: $DAILY_REPO" >&2
  exit 2
fi

if [[ ! -x "$LAUNCHER" ]]; then
  echo "missing executable launcher: $LAUNCHER" >&2
  exit 2
fi

if [[ ! -f "$MANIFEST" ]]; then
  echo "missing frozen challenger manifest: $MANIFEST" >&2
  exit 2
fi

daily_before="$(systemctl is-active trade-paper.service 2>/dev/null || true)"

"$UV_BIN" sync --frozen

unit_file="$(mktemp)"
trap 'rm -f "$unit_file"' EXIT
cat >"$unit_file" <<EOF
[Unit]
Description=Trade ETH 12H shadow paper bot (simulated)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=$DEPLOY_USER
WorkingDirectory=$REPO_ROOT
EnvironmentFile=-/etc/trade-paper.env
Environment=PAPER_12H_JOURNAL_DIR=$REPO_ROOT/paper_journal_eth_12h
Environment=PATH=$(dirname "$UV_BIN"):/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
ExecStart=/usr/bin/env bash $LAUNCHER
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

sudo install -o root -g root -m 0644 "$unit_file" "$UNIT_PATH"
sudo systemctl daemon-reload
sudo systemctl enable --now "$SERVICE_NAME"

daily_after="$(systemctl is-active trade-paper.service 2>/dev/null || true)"
if [[ "$daily_before" != "$daily_after" ]]; then
  echo "WARNING: Daily service state changed: before=$daily_before after=$daily_after" >&2
  exit 3
fi

systemctl --no-pager --full status "$SERVICE_NAME"
echo "Daily V1 state unchanged: ${daily_after:-not-found}"
