"""Índice de embeddings local (numpy) para busca semântica."""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any

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
_MODEL_LOCK = threading.Lock()


def needs_e5_prompts(model_name: str) -> bool:
    """Modelos da família E5 exigem prefixos `query: `/`passage: `."""
    lowered = (model_name or "").lower()
    return "e5" in lowered and "instructor" not in lowered


def embedding_device() -> str:
    """CPU por omissão. `VEDIC_DEVICE=mps` (ou cuda) tira o encode da CPU."""
    from vedic_pipeline.train.device import torch_device

    return str(torch_device().type)


def _load_st_model(model_name: str):
    from sentence_transformers import SentenceTransformer

    device = embedding_device()
    cache_key = f"{model_name}\0{device}"
    cached = _MODEL_CACHE.get(cache_key)
    if cached is not None:
        return cached
    with _MODEL_LOCK:
        cached = _MODEL_CACHE.get(cache_key)
        if cached is not None:
            return cached
        logger.info("Carregando embedding model: %s (%s)", model_name, device)
        # Dois loads em paralelo no MPS já derrubaram o processo (SIGSEGV).
        # O lock serializa. O device só muda com VEDIC_DEVICE / VEDIC_ENABLE_MPS.
        model = SentenceTransformer(model_name, device=device)
        _MODEL_CACHE[cache_key] = model
        return model


def _passage_texts(model_name: str, texts: list[str]) -> list[str]:
    if needs_e5_prompts(model_name):
        return [f"passage: {t}" for t in texts]
    return texts


def _query_text(model_name: str, query: str) -> str:
    if needs_e5_prompts(model_name):
        return f"query: {query}"
    return query


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
      - lexical.npz (BM25 invertido; a busca não retokeniza o corpus)
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
    texts = _passage_texts(model_name, [c["text"] for c in chunks])
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

    tmp_emb = out_dir / "embeddings_tmp.npy"
    tmp_chunks = out_dir / "chunks_tmp.jsonl"
    tmp_meta = out_dir / "index_meta_tmp.json"

    np.save(tmp_emb, vectors)

    with tmp_chunks.open("w", encoding="utf-8") as f:
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
    tmp_meta.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    import os
    os.replace(tmp_emb, out_dir / "embeddings.npy")
    os.replace(tmp_chunks, out_dir / "chunks.jsonl")
    os.replace(tmp_meta, out_dir / "index_meta.json")

    from vedic_pipeline.search.lexical_index import save_lexical_index

    save_lexical_index(out_dir, chunks)

    logger.info("Índice salvo atomicamente em %s (%d chunks, dim=%d)", out_dir, len(chunks), vectors.shape[1])
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

    vectors = np.load(emb_path, allow_pickle=False)
    chunks: list[dict[str, Any]] = []
    with chunks_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if line:
                try:
                    chunks.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.warning("Linha %d inválida em %s — ignorada", line_no, chunks_path)

    meta: dict[str, Any] = {}
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            logger.warning("index_meta.json corrompido em %s — ignorado", index_dir)
            meta = {}

    if len(chunks) != len(vectors):
        raise ValueError(
            f"Mismatch chunks ({len(chunks)}) vs embeddings ({len(vectors)})"
        )

    from vedic_pipeline.search.lexical_index import ensure_lexical_index

    try:
        ensure_lexical_index(index_dir, chunks)
    except Exception:  # noqa: BLE001
        logger.exception("Falha ao preparar o índice lexical; a busca usa o BM25 em memória")

    return {"vectors": vectors, "chunks": chunks, "meta": meta, "index_dir": index_dir}


def search_index(
    query: str,
    index: dict[str, Any],
    top_k: int = 5,
    model_name: str | None = None,
    tradition: str | None = None,
    language: str | None = None,
) -> list[dict[str, Any]]:
    """Busca por similaridade cosseno (vetores já normalizados → dot product)."""
    model_name = model_name or index.get("meta", {}).get(
        "model_name", DEFAULT_EMBEDDING_MODEL
    )
    model = _load_st_model(model_name)
    q = model.encode(
        [_query_text(model_name, query)],
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
