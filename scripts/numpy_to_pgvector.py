#!/usr/bin/env python3
"""Copia índice numpy (embeddings.npy + chunks.jsonl) para PostgreSQL/pgvector sem re-encodar."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vedic_pipeline.common.constants import DEFAULT_CORPUS, DEFAULT_EMBED_DIR, DEFAULT_EMBEDDING_MODEL
from vedic_pipeline.common.corpus import load_corpus
from vedic_pipeline.storage.catalog import sync_corpus_to_db
from vedic_pipeline.storage.db import init_schema
from vedic_pipeline.storage.vectors import upsert_chunk_embeddings


def main() -> int:
    embed_dir = Path(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_EMBED_DIR)
    emb_path = embed_dir / "embeddings.npy"
    chunks_path = embed_dir / "chunks.jsonl"
    meta_path = embed_dir / "index_meta.json"

    if not emb_path.exists() or not chunks_path.exists():
        print(f"Índice incompleto em {embed_dir}")
        return 1

    vectors = np.load(emb_path)
    chunks = []
    with chunks_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))

    if len(chunks) != len(vectors):
        print(f"Mismatch chunks={len(chunks)} vectors={len(vectors)}")
        return 1

    model = DEFAULT_EMBEDDING_MODEL
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        model = meta.get("model_name") or model

    print("Sync schema + documents (purge órfãos)…")
    init_schema()
    sync_corpus_to_db(DEFAULT_CORPUS, init=False, purge=True)

    # limpa chunks antigos dos docs do corpus (CASCADE limpa embeddings)
    from vedic_pipeline.storage.db import get_connection

    doc_ids = sorted(
        {c.get("doc_id") for c in chunks if c.get("doc_id")}
        | {r.get("id") for r in load_corpus(DEFAULT_CORPUS) if r.get("id")}
    )
    with get_connection() as conn:
        with conn.cursor() as cur:
            if doc_ids:
                cur.execute("DELETE FROM chunks WHERE doc_id = ANY(%s)", (doc_ids,))
                print(f"Chunks antigos removidos (docs={len(doc_ids)})")

    print(f"Upsert {len(chunks)} embeddings → pgvector (model={model})…")
    # batch to avoid huge transactions
    batch = 500
    total = 0
    for i in range(0, len(chunks), batch):
        n = upsert_chunk_embeddings(
            chunks[i : i + batch],
            vectors[i : i + batch],
            model,
        )
        total += n
        print(f"  … {total}/{len(chunks)}")

    print(json.dumps({"ok": True, "n_chunks": total, "model": model}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
