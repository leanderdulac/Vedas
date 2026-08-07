# Veda Knowledge — API + UI estática (produção)
FROM node:20-bookworm-slim AS frontend-build
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

# deps de sistema mínimas (psycopg, lxml)
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential \
      libpq-dev \
      curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements_vedic_pipeline.txt ./
# torch CPU wheel é grande; em container de API preferimos instalar o que a API precisa.
# Para treino use a imagem host / venv local.
RUN pip install --upgrade pip \
 && pip install -r requirements_vedic_pipeline.txt

COPY vedic_pipeline ./vedic_pipeline
COPY vedic_knowledge_pipeline.py ./
COPY scripts ./scripts
COPY fixtures ./fixtures
COPY --from=frontend-build /frontend/dist ./frontend/dist

# dados montados em runtime (volumes)
RUN mkdir -p data/raw artifacts/embeddings artifacts/tokenizer

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
  CMD curl -fsS "http://127.0.0.1:${PORT}/api/v1/health" || exit 1

CMD ["python", "-m", "uvicorn", "vedic_knowledge_pipeline:app", "--host", "0.0.0.0", "--port", "8000"]
