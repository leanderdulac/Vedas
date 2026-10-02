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

export VEDIC_BIND_HOST="${VEDIC_BIND_HOST:-0.0.0.0}"
export VEDIC_FORWARDED_ALLOW_IPS="${VEDIC_FORWARDED_ALLOW_IPS:-10.0.0.0/8,172.16.0.0/12,192.168.0.0/16,127.0.0.1,::1}"

echo "→ Serving app at http://127.0.0.1:8000"
exec uvicorn vedic_knowledge_pipeline:app --host "${VEDIC_BIND_HOST}" --port 8000 --proxy-headers --forwarded-allow-ips "${VEDIC_FORWARDED_ALLOW_IPS}"
