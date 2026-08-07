#!/usr/bin/env bash
# Build do frontend + API servindo dist em :8000
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

export DATABASE_URL="${DATABASE_URL:-postgresql://vedas:vedas@localhost:5432/vedas}"

echo "→ Building frontend…"
(cd frontend && npm install && npm run build)

echo "→ Serving app at http://127.0.0.1:8000"
exec uvicorn vedic_knowledge_pipeline:app --host 0.0.0.0 --port 8000
