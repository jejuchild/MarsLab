#!/usr/bin/env bash
# Build the frontend and (re)start the public production server on :5173.
# The dev server with hot reload stays on 127.0.0.1:5174 (local only).
set -euo pipefail
cd "$(dirname "$0")/../frontend"

npm run build

# vite preview indexes dist/ at startup, so restart it after every build.
for pid in $(ss -ltnp 2>/dev/null | awk '/0.0.0.0:5173 /' | grep -o 'pid=[0-9]*' | cut -d= -f2); do
  kill "$pid"
done
sleep 2
(setsid nohup npx vite preview --host 0.0.0.0 --port 5173 --strictPort > preview.log 2>&1 < /dev/null &)
for _ in $(seq 1 30); do
  if ss -ltn | grep -q '0.0.0.0:5173 '; then echo "production build live on :5173"; exit 0; fi
  sleep 1
done
echo "preview failed to start; see frontend/preview.log"; exit 1
