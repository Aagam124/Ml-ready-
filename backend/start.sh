#!/bin/bash
set -e

echo "=============================================="
echo "    Starting MLReady on Render"
echo "=============================================="

# Start background worker process in background
echo "[*] Starting background worker..."
python -m app.jobs.worker &

# Start FastAPI server on port provided by Render ($PORT) or fallback to 8000
PORT="${PORT:-8000}"
echo "[*] Starting FastAPI on 0.0.0.0:${PORT}..."
exec uvicorn app.main:app --host 0.0.0.0 --port "${PORT}"
