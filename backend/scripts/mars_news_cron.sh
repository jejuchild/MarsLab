#!/usr/bin/env bash
# Cron setup (daily at 7:05am):
#   5 7 * * * /disk1/cspark/MarsLab/backend/scripts/mars_news_cron.sh >> /disk1/cspark/MarsLab/backend/mars_news/cron.log 2>&1

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "=== MarsLab Mars News Crawler — $(date -Iseconds) ==="

if [ -f "$PROJECT_ROOT/backend/.venv/bin/activate" ]; then
    source "$PROJECT_ROOT/backend/.venv/bin/activate"
fi

if [ -f "$PROJECT_ROOT/backend/.env" ]; then
    set -a
    source "$PROJECT_ROOT/backend/.env"
    set +a
fi

cd "$PROJECT_ROOT"
python backend/scripts/mars_news_crawler.py "$@"

echo "=== Done — $(date -Iseconds) ==="
