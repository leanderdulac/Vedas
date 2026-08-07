"""Upsert e busca de embeddings via pgvector."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from vedic_pipeline.common.constants import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CORPUS,
    DEFAULT_EMBEDDING_MODEL,
)
from vedic_pipeline.common.corpus import load_corpus
from vedic_pipeline.etl.chunking import chunk_records
from vedic_pipeline.storage.catalog import sync_corpus_to_db
from vedic_pipeline.storage.db import get_connection, init_schema

logger = logging.getLogger("vedic_pipeline.storage.vectors")


def _vector_literal(vec: Sequence[float] | np.ndarray) -> str:
    arr = np.asarray(vec, dtype=np.float32).tolist()
    return "[" + ",".join(f"{x:.8f}" for x in arr) + "]"


def upsert_chunk_embeddings(
    chunks: list[dict[str, Any]],
    vectors: np.ndarray,
    model_name: str,
    url: Optional[str] = None,
) -> int:
    if len(chunks) != len(vectors):
        raise ValueError("chunks e vectors com tamanhos diferentes")

    with get_connection(url) as conn:
        with conn.cursor() as cur:
            for ch, vec in zip(chunks, vectors):
                cur.execute(
                    """
                    INSERT INTO chunks (
                        chunk_id, doc_id, chunk_index, text, source_url,
                        title, tradition, language, license, char_count
                    ) VALUES (
                        %(chunk_id)s, %(doc_id)s, %(chunk_index)s, %(text)s,
                        %(source_url)s, %(title)s, %(tradition)s, %(language)s,
                        %(license)s, %(char_count)s
                    )
                    ON CONFLICT (chunk_id) DO UPDATE SET
                        text = EXCLUDED.text,
                        title = EXCLUDED.title,
                        tradition = EXCLUDED.tradition,
                        language = EXCLUDED.language,
                        license = EXCLUDED.license,
                        char_count = EXCLUDED.char_count
                    """,
                    {
                        "chunk_id": ch["chunk_id"],
                        "doc_id": ch.get("doc_id"),
                        "chunk_index": ch.get("chunk_index", 0),
                        "text": ch.get("text") or "",
                        "source_url": ch.get("source_url"),
                        "title": ch.get("title"),
                        "tradition": ch.get("tradition"),
                        "language": ch.get("language"),
                        "license": ch.get("license"),
                        "char_count": ch.get("char_count")
                        or len(ch.get("text") or ""),
                    },
                )
                lit = _vector_literal(vec)
                cur.execute(
                    """
                    INSERT INTO chunk_embeddings (chunk_id, model_name, embedding)
                    VALUES (%s, %s, %s::vector)
                    ON CONFLICT (chunk_id) DO UPDATE SET
                        model_name = EXCLUDED.model_name,
                        embedding = EXCLUDED.embedding,
                        created_at = NOW()
                    """,
                    (ch["chunk_id"], model_name, lit),
                )
    return len(chunks)


def build_pgvector_index(
    corpus_path: Path = DEFAULT_CORPUS,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
    batch_size: int = 32,
    url: Optional[str] = None,
) -> dict[str, Any]:
    """Chunka corpus, gera embeddings e grava em PostgreSQL/pgvector."""
    from vedic_pipeline.search.embeddings import _load_st_model

    init_schema(url)
    sync_corpus_to_db(corpus_path, url=url, init=False, purge=True)

    records = load_corpus(Path(corpus_path))
    if not records:
        raise ValueError("Corpus vazio.")

    chunks = list(chunk_records(records, chunk_size=chunk_size, overlap=overlap))
    if not chunks:
        raise ValueError("Nenhum chunk gerado.")

    doc_ids = [r["id"] for r in records if r.get("id")]
    # remove chunks antigos dos docs do corpus (evita órfãos se chunking mudou)
    with get_connection(url) as conn:
        with conn.cursor() as cur:
            if doc_ids:
                cur.execute("DELETE FROM chunks WHERE doc_id = ANY(%s)", (doc_ids,))
                logger.info("Chunks antigos removidos para %d docs", len(doc_ids))

    model = _load_st_model(model_name)
    texts = [c["text"] for c in chunks]
    logger.info("Gerando embeddings pgvector para %d chunks...", len(texts))
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.shape[1] != 384:
        raise ValueError(
            f"Dimensão {vectors.shape[1]} != 384. "
            "Ajuste o schema vector(N) se usar outro modelo."
        )

    n = upsert_chunk_embeddings(chunks, vectors, model_name, url=url)
    return {
        "backend": "pgvector",
        "model_name": model_name,
        "n_docs": len(records),
        "n_chunks": n,
        "dim": int(vectors.shape[1]),
        "chunk_size": chunk_size,
        "overlap": overlap,
    }

def search_pgvector(
    query: str,
    top_k: int = 5,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    tradition: Optional[str] = None,
    language: Optional[str] = None,
    url: Optional[str] = None,
) -> list[dict[str, Any]]:
    from vedic_pipeline.search.embeddings import _load_st_model

    model = _load_st_model(model_name)
    q = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )[0].astype(np.float32)
    lit = _vector_literal(q)

    # cosine distance <=> ; score = 1 - distance (vetores normalizados ≈ cosseno)
    clauses = ["e.model_name = %s"]
    params: list[Any] = [model_name]
    if tradition:
        clauses.append("LOWER(c.tradition) = %s")
        params.append(tradition.lower())
    if language:
        clauses.append("LOWER(c.language) = %s")
        params.append(language.lower())
    where = " AND ".join(clauses)

    sql = f"""
        SELECT
            c.chunk_id,
            c.doc_id,
            c.chunk_index,
            c.text,
            c.source_url,
            c.title,
            c.tradition,
            c.language,
            c.license,
            c.char_count,
            (1 - (e.embedding <=> %s::vector)) AS score
        FROM chunk_embeddings e
        JOIN chunks c ON c.chunk_id = e.chunk_id
        WHERE {where}
        ORDER BY e.embedding <=> %s::vector
        LIMIT %s
    """
    # placeholders: score_vec, where..., order_vec, limit
    exec_params: list[Any] = [lit, *params, lit, top_k]

    with get_connection(url) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, exec_params)
            rows = cur.fetchall()

    results: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        if item.get("score") is not None:
            item["score"] = round(float(item["score"]), 4)
        results.append(item)
    return results
