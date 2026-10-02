#!/usr/bin/env python3
"""Reconstrói o índice (embeddings + BM25) sem derrubar a API que o está servindo.

1. chunka o corpus com o mesmo chunk_size/overlap do índice atual;
2. reaproveita o vetor de cada chunk que já existe (mesmo chunk_id e mesmo
   texto) e só codifica os novos (``--full`` recodifica tudo);
3. grava tudo em ``artifacts/embeddings.next-<ts>`` e valida a carga;
4. troca de diretório com dois ``rename`` no mesmo volume: o atual vira
   ``artifacts/embeddings.bak-<ts>`` e o novo assume o nome. A API recarrega
   sozinha ao ver o mtime novo de ``embeddings.npy`` (``rag.get_index``).

Com ``--corpus-next`` o corpus novo (já com os registros ingeridos) também é
trocado atomicamente, com cópia do antigo em ``artifacts/backups/``.

Uso:
  python scripts/rebuild_index_safe.py --corpus-next artifacts/staging/corpus.jsonl
  python scripts/rebuild_index_safe.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from vedic_pipeline.common.constants import DEFAULT_CORPUS, DEFAULT_EMBED_DIR  # noqa: E402
from vedic_pipeline.common.corpus import load_corpus, utc_now_iso  # noqa: E402
from vedic_pipeline.crawler.licensed_sources import is_restricted, records_for_index  # noqa: E402
from vedic_pipeline.etl.chunking import chunk_records  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("rebuild_index_safe")


def load_current(index_dir: Path) -> tuple[dict[str, Any], dict[str, tuple[int, str]], np.ndarray | None]:
    meta_path = index_dir / "index_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    by_id: dict[str, tuple[int, str]] = {}
    vectors = None
    chunks_path = index_dir / "chunks.jsonl"
    if chunks_path.exists() and (index_dir / "embeddings.npy").exists():
        vectors = np.load(index_dir / "embeddings.npy", mmap_mode="r")
        with chunks_path.open(encoding="utf-8") as f:
            for row, line in enumerate(f):
                ch = json.loads(line)
                by_id[ch["chunk_id"]] = (row, ch.get("text") or "")
    return meta, by_id, vectors


def build(
    corpus: Path,
    index_dir: Path,
    out_dir: Path,
    *,
    full: bool,
    batch_size: int,
) -> dict[str, Any]:
    from vedic_pipeline.search.embeddings import _load_st_model, _passage_texts
    from vedic_pipeline.search.lexical_index import save_lexical_index

    meta, by_id, old_vectors = load_current(index_dir)
    model_name = meta.get("model_name") or "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    chunk_size = int(meta.get("chunk_size") or 800)
    overlap = int(meta.get("overlap") or 120)
    # fontes licenciadas (data/raw, fora do corpus) só com VEDIC_ENABLE_LICENSED_SOURCES=1
    records = records_for_index(load_corpus(corpus), ROOT)
    chunks = list(chunk_records(records, chunk_size=chunk_size, overlap=overlap))
    dim = int(meta.get("dim") or (old_vectors.shape[1] if old_vectors is not None else 384))
    vectors = np.zeros((len(chunks), dim), dtype=np.float32)
    todo: list[int] = []
    reused = 0
    for i, ch in enumerate(chunks):
        hit = None if full else by_id.get(ch["chunk_id"])
        if hit is not None and hit[1] == (ch.get("text") or "") and old_vectors is not None:
            vectors[i] = old_vectors[hit[0]]
            reused += 1
        else:
            todo.append(i)
    logger.info("%d chunks: %d reaproveitados, %d a codificar (%s)", len(chunks), reused, len(todo), model_name)
    if todo:
        model = _load_st_model(model_name)
        texts = _passage_texts(model_name, [chunks[i]["text"] for i in todo])
        enc = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        vectors[np.asarray(todo)] = np.asarray(enc, dtype=np.float32)

    out_dir.mkdir(parents=True, exist_ok=False)
    np.save(out_dir / "embeddings.npy", vectors)
    with (out_dir / "chunks.jsonl").open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    new_meta = {
        "model_name": model_name,
        "corpus": str(DEFAULT_CORPUS),
        "n_docs": len(records),
        "n_chunks": len(chunks),
        "dim": dim,
        "chunk_size": chunk_size,
        "overlap": overlap,
        "built_at": utc_now_iso(),
        "normalized": True,
        "reused_vectors": reused,
        "encoded_vectors": len(todo),
        "previous_built_at": meta.get("built_at"),
        "licensed_docs": sum(1 for r in records if is_restricted(r)),
    }
    (out_dir / "index_meta.json").write_text(json.dumps(new_meta, ensure_ascii=False, indent=2), encoding="utf-8")
    save_lexical_index(out_dir, chunks)
    return new_meta


def validate(out_dir: Path, expected_chunks: int) -> None:
    from vedic_pipeline.search.embeddings import load_embedding_index
    from vedic_pipeline.search.lexical_index import fingerprint, load_lexical_index

    idx = load_embedding_index(out_dir)
    if len(idx["chunks"]) != expected_chunks or idx["vectors"].shape[0] != expected_chunks:
        raise RuntimeError("índice novo com contagem inesperada")
    norms = np.linalg.norm(idx["vectors"], axis=1)
    if not np.allclose(norms, 1.0, atol=1e-3):
        raise RuntimeError("vetores não normalizados no índice novo")
    if load_lexical_index(out_dir, expected_fingerprint=fingerprint(idx["chunks"])) is None:
        raise RuntimeError("índice lexical ausente ou de outro conjunto de chunks")


def swap(index_dir: Path, out_dir: Path, stamp: str) -> Path:
    backup = index_dir.with_name(f"{index_dir.name}.bak-{stamp}")
    # dois renames no mesmo volume; a janela entre eles é de microssegundos
    os.rename(index_dir, backup)
    os.rename(out_dir, index_dir)
    return backup


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--index-dir", default=str(DEFAULT_EMBED_DIR))
    ap.add_argument("--corpus", default=str(DEFAULT_CORPUS))
    ap.add_argument("--corpus-next", default="", help="corpus novo a promover junto com o índice")
    ap.add_argument("--full", action="store_true", help="recodifica todos os chunks")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--no-swap", action="store_true", help="só constrói e valida o diretório novo")
    args = ap.parse_args()

    index_dir = Path(args.index_dir).resolve()
    corpus = Path(args.corpus).resolve()
    source_corpus = Path(args.corpus_next).resolve() if args.corpus_next else corpus
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_dir = index_dir.with_name(f"{index_dir.name}.next-{stamp}")
    meta = build(source_corpus, index_dir, out_dir, full=args.full, batch_size=args.batch_size)
    validate(out_dir, meta["n_chunks"])
    logger.info("Índice novo validado em %s", out_dir)
    if args.no_swap:
        print(json.dumps({"built": str(out_dir), **meta}, ensure_ascii=False, indent=2))
        return 0
    result: dict[str, Any] = {**meta}
    if args.corpus_next:
        backups = index_dir.parent / "backups"
        backups.mkdir(parents=True, exist_ok=True)
        corpus_backup = backups / f"corpus-{stamp}.jsonl"
        shutil.copy2(corpus, corpus_backup)
        os.replace(source_corpus, corpus)
        result["corpus_backup"] = str(corpus_backup)
    result["index_backup"] = str(swap(index_dir, out_dir, stamp))
    result["index_dir"] = str(index_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
