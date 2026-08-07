#!/usr/bin/env python3
"""
Constrói um corpus védico amplo a partir de fixtures locais de domínio público
e (opcionalmente) URLs do manifesto.

Uso:
  python scripts/bootstrap_vedic_corpus.py
  python scripts/bootstrap_vedic_corpus.py --rebuild-index --backend both
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vedic_pipeline.crawler.ingest import ingest_manifest
from vedic_pipeline.common.constants import DEFAULT_CORPUS, DEFAULT_EMBED_DIR
from vedic_pipeline.common.corpus import load_corpus, rewrite_corpus


def main() -> int:
    parser = argparse.ArgumentParser(description="Bootstrap corpus védico PD")
    parser.add_argument(
        "--manifest",
        default=str(ROOT / "fixtures" / "sources_vedic_corpus.json"),
    )
    parser.add_argument("--corpus", default=str(DEFAULT_CORPUS))
    parser.add_argument("--reset", action="store_true", help="Apaga corpus antes")
    parser.add_argument("--rebuild-index", action="store_true")
    parser.add_argument(
        "--backend",
        choices=["numpy", "pgvector", "both"],
        default="both",
    )
    parser.add_argument("--sync-db", action="store_true")
    args = parser.parse_args()

    corpus_path = Path(args.corpus)
    if args.reset and corpus_path.exists():
        corpus_path.unlink()
        print(f"Corpus resetado: {corpus_path}")

    stats = ingest_manifest(args.manifest, corpus_path=corpus_path, min_chars=40)
    print(json.dumps(stats, ensure_ascii=False, indent=2))

    records = load_corpus(corpus_path)
    # reescreve deduplicado por segurança
    from vedic_pipeline.common.corpus import deduplicate_records

    unique = deduplicate_records(records)
    rewrite_corpus(corpus_path, unique)
    print(f"Corpus final: {len(unique)} documentos em {corpus_path}")

    if args.sync_db:
        from vedic_pipeline.storage.catalog import sync_corpus_to_db

        print(json.dumps(sync_corpus_to_db(corpus_path), ensure_ascii=False, indent=2))

    if args.rebuild_index:
        if args.backend in {"numpy", "both"}:
            from vedic_pipeline.search.embeddings import build_embedding_index
            from vedic_pipeline.search.rag import get_index

            meta = build_embedding_index(
                corpus_path=corpus_path,
                out_dir=DEFAULT_EMBED_DIR,
                chunk_size=700,
                overlap=120,
            )
            get_index(DEFAULT_EMBED_DIR, reload=True)
            print("numpy index:", json.dumps(meta, ensure_ascii=False))
        if args.backend in {"pgvector", "both"}:
            try:
                from vedic_pipeline.storage.vectors import build_pgvector_index

                meta = build_pgvector_index(
                    corpus_path=corpus_path,
                    chunk_size=700,
                    overlap=120,
                )
                print("pgvector index:", json.dumps(meta, ensure_ascii=False))
            except Exception as exc:  # noqa: BLE001
                print(f"pgvector skip: {exc}")

    return 0 if stats.get("failed", 0) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
