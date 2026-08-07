"""Catálogo de documentos no PostgreSQL."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

from vedic_pipeline.common.constants import DEFAULT_CORPUS
from vedic_pipeline.common.corpus import load_corpus
from vedic_pipeline.storage.db import get_connection, init_schema

logger = logging.getLogger("vedic_pipeline.storage.catalog")


def count_documents(url: Optional[str] = None) -> int:
    with get_connection(url) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) AS n FROM documents")
            row = cur.fetchone()
            return int(row["n"]) if row else 0


def upsert_document(conn: Any, rec: dict[str, Any]) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO documents (
                id, title, source_url, tradition, language, license,
                text, fingerprint, char_count, retrieved_at, updated_at
            ) VALUES (
                %(id)s, %(title)s, %(source_url)s, %(tradition)s, %(language)s,
                %(license)s, %(text)s, %(fingerprint)s, %(char_count)s,
                %(retrieved_at)s, NOW()
            )
            ON CONFLICT (id) DO UPDATE SET
                title = EXCLUDED.title,
                source_url = EXCLUDED.source_url,
                tradition = EXCLUDED.tradition,
                language = EXCLUDED.language,
                license = EXCLUDED.license,
                text = EXCLUDED.text,
                fingerprint = EXCLUDED.fingerprint,
                char_count = EXCLUDED.char_count,
                retrieved_at = EXCLUDED.retrieved_at,
                updated_at = NOW()
            """,
            {
                "id": rec.get("id"),
                "title": rec.get("title"),
                "source_url": rec.get("source_url"),
                "tradition": rec.get("tradition"),
                "language": rec.get("language"),
                "license": rec.get("license"),
                "text": rec.get("text") or "",
                "fingerprint": rec.get("fingerprint"),
                "char_count": rec.get("char_count") or len(rec.get("text") or ""),
                "retrieved_at": rec.get("retrieved_at"),
            },
        )


def sync_corpus_to_db(
    corpus_path: Path = DEFAULT_CORPUS,
    url: Optional[str] = None,
    init: bool = True,
    purge: bool = True,
) -> dict[str, Any]:
    """Sincroniza data/corpus.jsonl → tabela documents.

    Com purge=True (padrão), remove documentos (e chunks/embeddings em CASCADE)
    que não estão mais no corpus JSONL — alinha contagens corpus ↔ DB.
    """
    if init:
        init_schema(url)

    records = load_corpus(Path(corpus_path))
    ids = [rec["id"] for rec in records if rec.get("id")]
    if not records:
        if purge:
            with get_connection(url) as conn:
                with conn.cursor() as cur:
                    cur.execute("DELETE FROM documents")
                    deleted = cur.rowcount
            return {
                "synced": 0,
                "purged": deleted,
                "corpus": str(corpus_path),
            }
        return {"synced": 0, "corpus": str(corpus_path)}

    purged = 0
    with get_connection(url) as conn:
        for rec in records:
            if not rec.get("id"):
                continue
            upsert_document(conn, rec)
        if purge:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM documents WHERE id <> ALL(%s)",
                    (ids,),
                )
                purged = cur.rowcount or 0

    logger.info(
        "Sincronizados %d documentos para PostgreSQL (purged=%d)",
        len(records),
        purged,
    )
    return {
        "synced": len(records),
        "purged": purged,
        "corpus": str(corpus_path),
    }