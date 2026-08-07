"""Índice de embeddings local (numpy) para busca semântica."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np

from vedic_pipeline.common.constants import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CORPUS,
    DEFAULT_EMBED_DIR,
    DEFAULT_EMBEDDING_MODEL,
)
from vedic_pipeline.common.corpus import load_corpus, utc_now_iso
from vedic_pipeline.etl.chunking import chunk_records

logger = logging.getLogger("vedic_pipeline.search")

_MODEL_CACHE: dict[str, Any] = {}


def _load_st_model(model_name: str):
    from sentence_transformers import SentenceTransformer

    if model_name not in _MODEL_CACHE:
        logger.info("Carregando embedding model: %s", model_name)
        _MODEL_CACHE[model_name] = SentenceTransformer(model_name)
    return _MODEL_CACHE[model_name]


def build_embedding_index(
    corpus_path: Path = DEFAULT_CORPUS,
    out_dir: Path = DEFAULT_EMBED_DIR,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
    batch_size: int = 32,
) -> dict[str, Any]:
    """
    Chunka o corpus, gera embeddings e salva:
      - embeddings.npy
      - chunks.jsonl
      - index_meta.json
    """
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus não encontrado: {corpus_path}")

    records = load_corpus(corpus_path)
    if not records:
        raise ValueError("Corpus vazio.")

    chunks = list(
        chunk_records(records, chunk_size=chunk_size, overlap=overlap)
    )
    if not chunks:
        raise ValueError("Nenhum chunk gerado.")

    model = _load_st_model(model_name)
    texts = [c["text"] for c in chunks]
    logger.info("Gerando embeddings para %d chunks...", len(texts))
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    vectors = np.asarray(vectors, dtype=np.float32)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "embeddings.npy", vectors)

    chunks_path = out_dir / "chunks.jsonl"
    with chunks_path.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    meta = {
        "model_name": model_name,
        "corpus": str(corpus_path),
        "n_docs": len(records),
        "n_chunks": len(chunks),
        "dim": int(vectors.shape[1]),
        "chunk_size": chunk_size,
        "overlap": overlap,
        "built_at": utc_now_iso(),
        "normalized": True,
    }
    (out_dir / "index_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("Índice salvo em %s (%d chunks, dim=%d)", out_dir, len(chunks), vectors.shape[1])
    return meta


def load_embedding_index(index_dir: Path = DEFAULT_EMBED_DIR) -> dict[str, Any]:
    index_dir = Path(index_dir)
    emb_path = index_dir / "embeddings.npy"
    chunks_path = index_dir / "chunks.jsonl"
    meta_path = index_dir / "index_meta.json"
    if not emb_path.exists() or not chunks_path.exists():
        raise FileNotFoundError(
            f"Índice incompleto em {index_dir}. Execute build-index primeiro."
        )

    vectors = np.load(emb_path)
    chunks: list[dict[str, Any]] = []
    with chunks_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))

    meta: dict[str, Any] = {}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))

    if len(chunks) != len(vectors):
        raise ValueError(
            f"Mismatch chunks ({len(chunks)}) vs embeddings ({len(vectors)})"
        )

    return {"vectors": vectors, "chunks": chunks, "meta": meta, "index_dir": index_dir}


def search_index(
    query: str,
    index: dict[str, Any],
    top_k: int = 5,
    model_name: Optional[str] = None,
    tradition: Optional[str] = None,
    language: Optional[str] = None,
) -> list[dict[str, Any]]:
    """Busca por similaridade cosseno (vetores já normalizados → dot product)."""
    model_name = model_name or index.get("meta", {}).get(
        "model_name", DEFAULT_EMBEDDING_MODEL
    )
    model = _load_st_model(model_name)
    q = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )[0].astype(np.float32)

    vectors: np.ndarray = index["vectors"]
    scores = vectors @ q  # (n,)

    chunks = index["chunks"]
    candidates: list[tuple[float, int]] = []
    for i, score in enumerate(scores):
        ch = chunks[i]
        if tradition and (ch.get("tradition") or "").lower() != tradition.lower():
            continue
        if language and (ch.get("language") or "").lower() != language.lower():
            continue
        candidates.append((float(score), i))

    candidates.sort(key=lambda x: x[0], reverse=True)
    results: list[dict[str, Any]] = []
    for score, i in candidates[:top_k]:
        item = dict(chunks[i])
        item["score"] = round(score, 4)
        results.append(item)
    return results
