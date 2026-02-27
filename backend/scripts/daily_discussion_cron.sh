#!/usr/bin/env bash
# MarsLab Daily Discussion Generator — Cron Wrapper
#
# Cron setup (daily at 7am KST):
#   0 7 * * * /disk1/cspark/MarsLab/backend/scripts/daily_discussion_cron.sh >> /disk1/cspark/MarsLab/backend/daily_discussions/cron.log 2>&1
#
# Or via systemd timer for better logging:
#   See daily_discussion.timer and daily_discussion.service

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "=== MarsLab Daily Discussion — $(date -Iseconds) ==="

# Activate conda/venv if needed
if [ -f "$PROJECT_ROOT/backend/.venv/bin/activate" ]; then
    source "$PROJECT_ROOT/backend/.venv/bin/activate"
fi

# Load environment
if [ -f "$PROJECT_ROOT/backend/.env" ]; then
    set -a
    source "$PROJECT_ROOT/backend/.env"
    set +a
fi

cd "$PROJECT_ROOT"
/home/cspark/miniconda3/bin/python backend/scripts/daily_discussion.py "$@"

echo "=== Done — $(date -Iseconds) ==="
