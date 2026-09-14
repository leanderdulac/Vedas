#!/usr/bin/env bash
# Build do frontend + API servindo dist em :8000
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

# DATABASE_URL é opcional: a API carrega .env e usa o índice NumPy sem banco.

echo "→ Building frontend…"
(cd frontend && npm install && npm run build)

echo "→ Serving app at http://127.0.0.1:8000"
exec uvicorn vedic_knowledge_pipeline:app --host 0.0.0.0 --port 8000 --proxy-headers --forwarded-allow-ips '*'
