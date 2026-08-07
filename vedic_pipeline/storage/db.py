"""Conexão e schema PostgreSQL + extensão pgvector."""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import Any, Generator, Optional

logger = logging.getLogger("vedic_pipeline.storage")

DEFAULT_DATABASE_URL = "postgresql://vedas:vedas@localhost:5432/vedas"


def get_database_url() -> Optional[str]:
    """None se DB desabilitado; string se configurado."""
    # VEDIC_DATABASE_URL tem prioridade; DATABASE_URL é o padrão de mercado
    url = os.environ.get("VEDIC_DATABASE_URL") or os.environ.get("DATABASE_URL")
    if url is None:
        return None
    if url.strip().lower() in {"", "0", "false", "off", "none"}:
        return None
    if url.strip().lower() == "default":
        return DEFAULT_DATABASE_URL
    return url.strip()


def require_database_url() -> str:
    url = get_database_url()
    if not url:
        raise RuntimeError(
            "PostgreSQL não configurado. Defina DATABASE_URL ou "
            "VEDIC_DATABASE_URL (ex.: postgresql://vedas:vedas@localhost:5432/vedas) "
            "ou use 'default' com o docker-compose."
        )
    return url


@contextmanager
def get_connection(url: Optional[str] = None) -> Generator[Any, None, None]:
    try:
        import psycopg
        from psycopg.rows import dict_row
    except ImportError as exc:
        raise ImportError(
            "psycopg não instalado. pip install 'psycopg[binary]' pgvector"
        ) from exc

    dsn = url or require_database_url()
    conn = psycopg.connect(dsn, row_factory=dict_row)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


SCHEMA_SQL = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    id            TEXT PRIMARY KEY,
    title         TEXT,
    source_url    TEXT,
    tradition     TEXT,
    language      TEXT,
    license       TEXT,
    text          TEXT NOT NULL,
    fingerprint   TEXT,
    char_count    INTEGER,
    retrieved_at  TIMESTAMPTZ,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_documents_tradition ON documents (tradition);
CREATE INDEX IF NOT EXISTS idx_documents_language ON documents (language);
CREATE INDEX IF NOT EXISTS idx_documents_fingerprint ON documents (fingerprint);
CREATE INDEX IF NOT EXISTS idx_documents_source_url ON documents (source_url);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id      TEXT PRIMARY KEY,
    doc_id        TEXT REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index   INTEGER NOT NULL DEFAULT 0,
    text          TEXT NOT NULL,
    source_url    TEXT,
    title         TEXT,
    tradition     TEXT,
    language      TEXT,
    license       TEXT,
    char_count    INTEGER,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks (doc_id);
CREATE INDEX IF NOT EXISTS idx_chunks_tradition ON chunks (tradition);
CREATE INDEX IF NOT EXISTS idx_chunks_language ON chunks (language);

-- embedding dim default 384 (MiniLM multilingual); recriado se mudar o modelo
CREATE TABLE IF NOT EXISTS chunk_embeddings (
    chunk_id      TEXT PRIMARY KEY REFERENCES chunks(chunk_id) ON DELETE CASCADE,
    model_name    TEXT NOT NULL,
    embedding     vector(384) NOT NULL,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_chunk_embeddings_model
    ON chunk_embeddings (model_name);
"""


def init_schema(url: Optional[str] = None) -> dict[str, Any]:
    with get_connection(url) as conn:
        with conn.cursor() as cur:
            cur.execute(SCHEMA_SQL)
            # índice HNSW (pode falhar se tabela vazia em algumas versões — ok)
            try:
                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_chunk_embeddings_hnsw
                    ON chunk_embeddings
                    USING hnsw (embedding vector_cosine_ops);
                    """
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Índice HNSW não criado agora: %s", exc)
    logger.info("Schema PostgreSQL/pgvector inicializado")
    return {"ok": True, "schema": "documents+chunks+chunk_embeddings"}


def check_db(url: Optional[str] = None) -> dict[str, Any]:
    configured = get_database_url()
    if not configured and url is None:
        return {"configured": False, "reachable": False}
    try:
        with get_connection(url or configured) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1 AS ok")
                cur.fetchone()
                cur.execute(
                    "SELECT extname FROM pg_extension WHERE extname = 'vector'"
                )
                has_vector = cur.fetchone() is not None
                cur.execute(
                    """
                    SELECT to_regclass('public.documents') AS documents,
                           to_regclass('public.chunks') AS chunks,
                           to_regclass('public.chunk_embeddings') AS embeddings
                    """
                )
                regs = cur.fetchone() or {}
                schema_ready = all(regs.get(k) for k in ("documents", "chunks", "embeddings"))
                counts: dict[str, Any] = {}
                if schema_ready:
                    cur.execute(
                        """
                        SELECT
                          (SELECT COUNT(*) FROM documents) AS documents,
                          (SELECT COUNT(*) FROM chunks) AS chunks,
                          (SELECT COUNT(*) FROM chunk_embeddings) AS embeddings
                        """
                    )
                    counts = dict(cur.fetchone() or {})
                else:
                    counts = {"note": "schema ausente — rode db-init"}
        return {
            "configured": True,
            "reachable": True,
            "pgvector": has_vector,
            "schema_ready": schema_ready,
            "counts": counts,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "configured": True,
            "reachable": False,
            "error": str(exc),
        }
