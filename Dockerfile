# Veda Knowledge — API + UI estática (produção)
# Node 22 alinha com CI e testes do frontend (strip-types >= 22.6)
FROM node:22-bookworm-slim AS frontend-build
WORKDIR /frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM python:3.12-slim-bookworm AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000

WORKDIR /app

# deps de sistema mínimas (psycopg, lxml) — build-essential removido após build se possível
RUN apt-get update && apt-get install -y --no-install-recommends \
      libpq-dev \
      curl \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd -g 10001 appuser && useradd -u 10001 -g appuser -s /bin/bash -m appuser

COPY requirements-api.txt pyproject.toml ./
# Runtime slim: sem datasets/accelerate/sentencepiece (só treino local).
# Embeddings (sentence-transformers) trazem torch+transformers transitivamente.
RUN pip install --upgrade pip \
 && pip install -r requirements-api.txt

COPY vedic_pipeline ./vedic_pipeline
COPY vedic_knowledge_pipeline.py ./
COPY scripts ./scripts
COPY fixtures ./fixtures
COPY --from=frontend-build /frontend/dist ./frontend/dist

# dados montados em runtime (volumes)
RUN mkdir -p data/raw artifacts/embeddings artifacts/tokenizer \
 && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
  CMD curl -fsS "http://127.0.0.1:${PORT}/api/v1/health" || exit 1

CMD ["sh", "-c", "python -m uvicorn vedic_knowledge_pipeline:app --host 0.0.0.0 --port ${PORT:-8000} --proxy-headers --forwarded-allow-ips '*'"]
