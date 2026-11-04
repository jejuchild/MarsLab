#!/usr/bin/env bash
# Start any MarsLab server that is not already listening. Safe to run
# repeatedly: used at boot and as a watchdog from cron.
#
#   backend  uvicorn     127.0.0.1:8000  (local only, proxied by the frontends)
#   dev      vite        127.0.0.1:5174  (hot reload, local only)
#   public   vite preview 0.0.0.0:5173   (production build)
#
# Suggested crontab:
#   @reboot     /disk1/cspark/MarsLab/scripts/start_servers.sh
#   */5 * * * * /disk1/cspark/MarsLab/scripts/start_servers.sh
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$ROOT/backend/logs"
LOG="$LOG_DIR/servers.log"
mkdir -p "$LOG_DIR"

# cron has a minimal environment.
NODE_BIN=/home/cspark/.nvm/versions/node/v24.13.0/bin
export PATH="$NODE_BIN:/home/cspark/miniconda3/bin:/usr/local/bin:/usr/bin:/bin"
export PYTHONUNBUFFERED=1
# API keys (Groq, Gemini) for the backend.
if [[ -f "$ROOT/backend/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT/backend/.env"
  set +a
fi

# One instance at a time (boot and watchdog can overlap).
exec 9>"$LOG_DIR/.start_servers.lock"
flock -n 9 || exit 0

log() { echo "$(date '+%F %T') $*" >> "$LOG"; }
listening() { ss -ltn | awk '{print $4}' | grep -qx "$1"; }

start() { # name addr dir cmd...
  local name=$1 addr=$2 dir=$3; shift 3
  listening "$addr" && return 0
  log "starting $name on $addr"
  (cd "$dir" && setsid nohup "$@" >> "$LOG_DIR/$name.log" 2>&1 < /dev/null &)
}

start backend 127.0.0.1:8000 "$ROOT/backend" \
  /home/cspark/miniconda3/bin/python -m uvicorn app:app --host 127.0.0.1 --port 8000 --reload
start dev 127.0.0.1:5174 "$ROOT/frontend" \
  npx vite --host 127.0.0.1 --port 5174 --strictPort

if ! listening 0.0.0.0:5173; then
  if [[ ! -f "$ROOT/frontend/dist/index.html" ]]; then
    log "no production build; building"
    (cd "$ROOT/frontend" && npm run build >> "$LOG_DIR/public.log" 2>&1) || { log "build failed"; exit 1; }
  fi
  start public 0.0.0.0:5173 "$ROOT/frontend" \
    npx vite preview --host 0.0.0.0 --port 5173 --strictPort
fi
exit 0
