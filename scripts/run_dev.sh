#!/usr/bin/env bash
# Sobe backend (8000) e frontend Vite (5173) em paralelo.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

export DATABASE_URL="${DATABASE_URL:-postgresql://vedas:vedas@localhost:5432/vedas}"

echo "→ Backend: http://127.0.0.1:8000"
echo "→ Frontend dev: http://127.0.0.1:5173"
echo

uvicorn vedic_knowledge_pipeline:app --host 127.0.0.1 --port 8000 --reload &
BACK_PID=$!

cleanup() {
  kill "$BACK_PID" 2>/dev/null || true
}
trap cleanup EXIT

cd frontend
npm run dev -- --host 127.0.0.1 --port 5173
