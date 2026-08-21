#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."
source .venv/bin/activate
echo "Starting Reconciliation Agent API on http://localhost:8000"
echo "Dashboard: http://localhost:8000"
exec uvicorn api.main:app --reload --port 8000 --host 127.0.0.1
