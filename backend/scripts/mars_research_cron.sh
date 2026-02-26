#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "=== MarsLab Mars Research Crawler - $(date -Iseconds) ==="

if [ -f "$PROJECT_ROOT/backend/.venv/bin/activate" ]; then
    source "$PROJECT_ROOT/backend/.venv/bin/activate"
fi

if [ -f "$PROJECT_ROOT/backend/.env" ]; then
    set -a
    source "$PROJECT_ROOT/backend/.env"
    set +a
fi

cd "$PROJECT_ROOT"
python backend/scripts/mars_research_crawler.py "$@"

echo "=== Done - $(date -Iseconds) ==="
