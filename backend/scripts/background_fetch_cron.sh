#!/usr/bin/env bash
# MarsLab background data fetcher — cron wrapper.
#
# Suggested crontab line (nightly at 01:30 local time):
#   30 1 * * * /disk1/cspark/MarsLab/backend/scripts/background_fetch_cron.sh
#
# Everything (wrapper lines, the fetcher's key=value log, tracebacks) goes to
#   backend/logs/background_fetch.log
# Extra arguments are passed through, e.g. "--max-run-gb 50 --site jezero".
# Environment overrides:
#   MARSLAB_DATA_ROOT      data root (default: this backend/)
#   BG_FETCH_MAX_HOURS     internal wall-clock limit in hours (default 5)

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON=/home/cspark/miniconda3/bin/python3
MAX_HOURS="${BG_FETCH_MAX_HOURS:-5}"
# Hard backstop: internal limit + 20 min (HiRISE JP2->GeoTIFF of the last product).
HARD_LIMIT_MIN=$(awk -v h="$MAX_HOURS" 'BEGIN { printf "%d", h * 60 + 20 }')

LOG_DIR="${MARSLAB_DATA_ROOT:-$BACKEND_DIR}/logs"
LOG_FILE="$LOG_DIR/background_fetch.log"
mkdir -p "$LOG_DIR"

# cron has a minimal environment: make sure aria2c (miniconda) is on PATH.
export PATH="/home/cspark/miniconda3/bin:/usr/local/bin:/usr/bin:/bin"
export PYTHONUNBUFFERED=1
umask 002

cd "$BACKEND_DIR" || exit 1
echo "$(date -Iseconds) wrapper start pid=$$ args=$*" >> "$LOG_FILE"

# SIGTERM at the hard limit (the fetcher cleans up partial files and saves
# state), SIGKILL 2 minutes later if it is still alive.
timeout --signal=TERM --kill-after=120 "${HARD_LIMIT_MIN}m" \
    "$PYTHON" "$SCRIPT_DIR/background_fetch.py" --max-hours "$MAX_HOURS" "$@" \
    >> "$LOG_FILE" 2>&1
rc=$?

echo "$(date -Iseconds) wrapper end rc=$rc" >> "$LOG_FILE"
exit $rc
