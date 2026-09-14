"""Conexão e schema PostgreSQL + extensão pgvector."""

from __future__ import annotations

import logging
import os
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

logger = logging.getLogger("vedic_pipeline.storage")

DEFAULT_DATABASE_URL = os.environ.get(
    "VEDIC_DEFAULT_DATABASE_URL", "postgresql://vedas:vedas@127.0.0.1:5433/vedas"
)

DEFAULT_EMBEDDING_DIM = 384


def get_embedding_dim(default: int = DEFAULT_EMBEDDING_DIM) -> int:
    """Dimensão do vetor pgvector. Configurável via VEDIC_EMBEDDING_DIM (64..3072)."""
    raw = os.environ.get("VEDIC_EMBEDDING_DIM", str(default)).strip()
    try:
        dim = int(raw)
    except ValueError:
        return default
    if 64 <= dim <= 3072:
        return dim
    return default


def get_database_url() -> str | None:
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
def get_connection(url: str | None = None) -> Generator[Any, None, None]:
    try:
        import psycopg
        from psycopg.rows import dict_row
    except ImportError as exc:
        raise ImportError(
            "psycopg não instalado. pip install 'psycopg[binary]' pgvector"
        ) from exc

    dsn = url or require_database_url()
    conn = psycopg.connect(dsn, row_factory=dict_row, connect_timeout=5)
    try:
        with conn.cursor() as _cur:
            _cur.execute("SET statement_timeout = '15s'")
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
    work          TEXT,
    verse_id      TEXT,
    locator       TEXT,
    book          INTEGER,
    hymn          INTEGER,
    verse         INTEGER,
    verse_end     INTEGER,
    heading       TEXT,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks (doc_id);
CREATE INDEX IF NOT EXISTS idx_chunks_tradition ON chunks (tradition);
CREATE INDEX IF NOT EXISTS idx_chunks_language ON chunks (language);
"""


def build_schema_sql(dim: int) -> str:
    """DDL completo com vector(dim) validado (64..3072)."""
    if not isinstance(dim, int) or isinstance(dim, bool) or not 64 <= dim <= 3072:
        raise ValueError(f"embedding dim inválida: {dim!r} (esperado 64..3072)")
    return (
        SCHEMA_SQL
        + f"""
CREATE TABLE IF NOT EXISTS chunk_embeddings (
    chunk_id      TEXT PRIMARY KEY REFERENCES chunks(chunk_id) ON DELETE CASCADE,
    model_name    TEXT NOT NULL,
    embedding     vector({dim}) NOT NULL,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_chunk_embeddings_model
    ON chunk_embeddings (model_name);
"""
    )


def get_embedding_column_dim(conn: Any) -> int | None:
    """Retorna a dimensão atual da coluna chunk_embeddings.embedding, ou None se ausente."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT atttypmod
            FROM pg_attribute
            JOIN pg_class ON pg_class.oid = pg_attribute.attrelid
            WHERE pg_class.relname = 'chunk_embeddings'
              AND pg_attribute.attname = 'embedding'
            """
        )
        row = cur.fetchone()
    if not row:
        return None
    typmod = row["atttypmod"] if isinstance(row, dict) else row[0]
    # pgvector/pg16 grava atttypmod = dimensão (ex.: vector(384) -> 384). -1 = sem typmod.
    if typmod is None or typmod == -1:
        return None
    return int(typmod)


_CHUNK_LOCATOR_COLUMNS_SQL = """
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS work TEXT;
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS verse_id TEXT;
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS locator TEXT;
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS book INTEGER;
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS hymn INTEGER;
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS verse INTEGER;
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS verse_end INTEGER;
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS heading TEXT;
CREATE INDEX IF NOT EXISTS idx_chunks_verse_id ON chunks (verse_id);
CREATE INDEX IF NOT EXISTS idx_chunks_work ON chunks (work);
"""


def init_schema(url: str | None = None, dim: int | None = None) -> dict[str, Any]:
    target_dim = dim if dim is not None else get_embedding_dim()
    if not isinstance(target_dim, int) or isinstance(target_dim, bool) or not 64 <= target_dim <= 3072:
        raise ValueError(f"embedding dim inválida: {target_dim!r} (esperado 64..3072)")
    with get_connection(url) as conn, conn.cursor() as cur:
        cur.execute(SCHEMA_SQL)
        cur.execute(_CHUNK_LOCATOR_COLUMNS_SQL)
        # Garante chunk_embeddings com a dimensão alvo; recria se divergir
        # (índice é reconstruível via build-index --backend pgvector).
        existing: int | None = None
        try:
            existing = get_embedding_column_dim(conn)
        except Exception:  # noqa: BLE001
            existing = None
        if existing is None:
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS chunk_embeddings (
                    chunk_id      TEXT PRIMARY KEY REFERENCES chunks(chunk_id) ON DELETE CASCADE,
                    model_name    TEXT NOT NULL,
                    embedding     vector({target_dim}) NOT NULL,
                    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
                """
            )
        elif existing != target_dim:
            logger.warning(
                "Dimensão pgvector %s != alvo %s: recriando chunk_embeddings (reindex necessário)",
                existing,
                target_dim,
            )
            cur.execute("DROP TABLE IF EXISTS chunk_embeddings")
            cur.execute(
                f"""
                CREATE TABLE chunk_embeddings (
                    chunk_id      TEXT PRIMARY KEY REFERENCES chunks(chunk_id) ON DELETE CASCADE,
                    model_name    TEXT NOT NULL,
                    embedding     vector({target_dim}) NOT NULL,
                    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
                """
            )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_chunk_embeddings_model
                ON chunk_embeddings (model_name);
            """
        )
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
    logger.info("Schema PostgreSQL/pgvector inicializado (dim=%s)", target_dim)
    return {"ok": True, "schema": "documents+chunks+chunk_embeddings", "embedding_dim": target_dim}


def _sanitize_db_error(exc: Exception) -> str:
    import re
    msg = str(exc)
    # Remove credenciais/DSN completos — não expõe host/user/db em /health público.
    msg = re.sub(r"postgresql(\+\w+)?://\S+", "postgresql://******", msg)
    msg = re.sub(r"://([^:]+):([^@]+)@", r"://\1:******@", msg)
    msg = re.sub(r"password=\S+", "password=******", msg)
    msg = re.sub(r"host=\S+", "host=******", msg)
    # Mensagem genérica + prefixo curto para diagnóstico sem vazar topologia.
    short = msg.strip().splitlines()[0][:160] if msg.strip() else "erro de banco"
    return f"indisponível ({type(exc).__name__}: {short})" if short else "indisponível"


def check_db(url: str | None = None) -> dict[str, Any]:
    configured = get_database_url()
    if not configured and url is None:
        return {"configured": False, "reachable": False}
    try:
        with get_connection(url or configured) as conn, conn.cursor() as cur:
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
            embedding_dim: int | None = None
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
                try:
                    embedding_dim = get_embedding_column_dim(conn)
                except Exception:  # noqa: BLE001
                    embedding_dim = None
            else:
                counts = {"note": "schema ausente — rode db-init"}
        return {
            "configured": True,
            "reachable": True,
            "pgvector": has_vector,
            "schema_ready": schema_ready,
            "counts": counts,
            "embedding_dim": embedding_dim,
            "expected_embedding_dim": get_embedding_dim(),
        }
    except Exception as exc:  # noqa: BLE001
        sanitized = _sanitize_db_error(exc)
        logger.warning("Falha na verificação do banco: %s", sanitized)
        return {
            "configured": True,
            "reachable": False,
            "error": sanitized,
        }
