"""Índice lexical invertido ao lado do índice numpy.

A busca híbrida calculava o BM25 percorrendo todos os chunks e, com o
corpus atual (dezenas de milhares), ainda retokenizava o texto. Este
arquivo guarda, por termo, os chunks e a frequência. A fórmula é a mesma
de `hybrid._lexical_scores_scan`, inclusive a expansão por documento.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger("vedic_pipeline.search.lexical")

LEXICAL_VERSION = 1
_META = "lexical_meta.json"
_VOCAB = "lexical_vocab.json"
_NPZ = "lexical.npz"

# chunk_id -> índice carregado. Vários diretórios podem coexistir em testes.
_BY_CHUNK: dict[str, LexicalIndex] = {}


@dataclass
class LexicalIndex:
    chunk_ids: list[str]
    row_of: dict[str, int]
    vocab: dict[str, int]
    post_ptr: np.ndarray
    post_docs: np.ndarray
    post_tfs: np.ndarray
    df: np.ndarray
    dl: np.ndarray
    avgdl: float

    @property
    def n_docs(self) -> int:
        return len(self.chunk_ids)

    def score(self, query: str, rows: list[int]) -> list[float]:
        from vedic_pipeline.search.hybrid import QUERY_EXPANSIONS, tokenize

        n = len(rows)
        if n == 0:
            return []
        q_tokens = tokenize(query)
        if not q_tokens:
            return [0.0] * n

        full = n == self.n_docs and rows == list(range(self.n_docs))
        mask = None
        if not full:
            mask = np.zeros(self.n_docs, dtype=bool)
            mask[np.asarray(rows, dtype=np.int32)] = True
        row_to_pos = np.full(self.n_docs, -1, dtype=np.int32)
        for pos, row in enumerate(rows):
            row_to_pos[row] = pos

        if full:
            avgdl = self.avgdl or 1.0
        else:
            avgdl = float(self.dl[np.asarray(rows, dtype=np.int32)].sum()) / n or 1.0

        scores = np.zeros(n, dtype=np.float64)
        k1, b = 1.5, 0.75
        for raw in q_tokens:
            claimed = np.zeros(n, dtype=bool)
            for term in (raw, *QUERY_EXPANSIONS.get(raw, [])):
                tid = self.vocab.get(term)
                if tid is None:
                    continue
                start = int(self.post_ptr[tid])
                end = int(self.post_ptr[tid + 1])
                if start == end:
                    continue
                docs = self.post_docs[start:end]
                tfs = self.post_tfs[start:end]
                if mask is not None:
                    keep = mask[docs]
                    if not keep.any():
                        continue
                    docs = docs[keep]
                    tfs = tfs[keep]
                    df_t = int(keep.sum())
                else:
                    df_t = int(self.df[tid])
                pos = row_to_pos[docs]
                free = ~claimed[pos]
                if not free.any():
                    continue
                pos = pos[free]
                docs = docs[free]
                tfs = tfs[free].astype(np.float64)
                claimed[pos] = True
                idf = np.log(1.0 + (n - df_t + 0.5) / (df_t + 0.5))
                dl = np.maximum(self.dl[docs].astype(np.float64), 1.0)
                contrib = idf * (tfs * (k1 + 1.0)) / (tfs + k1 * (1.0 - b + b * dl / avgdl))
                scores[pos] += contrib
        return [float(s) for s in scores]


def fingerprint(chunks: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for chunk in chunks:
        digest.update(str(chunk.get("chunk_id") or "").encode())
        digest.update(b"\0")
        digest.update(str(len(chunk.get("text") or "")).encode())
        digest.update(b"\0")
        digest.update(str(chunk.get("title") or "").encode())
        digest.update(b"\n")
    return digest.hexdigest()


def register(index: LexicalIndex) -> None:
    for cid in index.chunk_ids:
        if cid:
            _BY_CHUNK[cid] = index


def scores_for_chunks(query: str, chunks: list[dict[str, Any]]) -> list[float] | None:
    """None se estes chunks não são os do índice carregado."""
    if not chunks:
        return []
    ids: list[str] = []
    for chunk in chunks:
        cid = chunk.get("chunk_id")
        if not cid:
            return None
        ids.append(str(cid))
    if len(set(ids)) != len(ids):
        return None
    index = _BY_CHUNK.get(ids[0])
    if index is None:
        return None
    rows: list[int] = []
    for cid in ids:
        if _BY_CHUNK.get(cid) is not index:
            return None
        row = index.row_of.get(cid)
        if row is None:
            return None
        rows.append(row)
    return index.score(query, rows)


def build_lexical_index(chunks: list[dict[str, Any]]) -> LexicalIndex:
    from vedic_pipeline.search.hybrid import clear_chunk_token_cache, get_chunk_tokens

    vocab: dict[str, int] = {}
    post_docs: list[list[int]] = []
    post_tfs: list[list[int]] = []
    df_counts: list[int] = []
    dl = np.zeros(len(chunks), dtype=np.int32)
    chunk_ids: list[str] = []

    def tid_of(token: str) -> int:
        tid = vocab.get(token)
        if tid is None:
            tid = len(vocab)
            vocab[token] = tid
            post_docs.append([])
            post_tfs.append([])
            df_counts.append(0)
        return tid

    for i, chunk in enumerate(chunks):
        cid = str(chunk.get("chunk_id") or f"row-{i}")
        chunk_ids.append(cid)
        tokens = get_chunk_tokens(chunk)
        dl[i] = len(tokens)
        if not tokens:
            continue
        for token, tf in Counter(tokens).items():
            tid = tid_of(token)
            post_docs[tid].append(i)
            post_tfs[tid].append(int(tf))
            df_counts[tid] += 1

    n_vocab = len(vocab)
    post_ptr = np.zeros(n_vocab + 1, dtype=np.int64)
    for tid in range(n_vocab):
        post_ptr[tid + 1] = post_ptr[tid] + len(post_docs[tid])
    total = int(post_ptr[-1]) if n_vocab else 0
    post_docs_arr = np.zeros(total, dtype=np.int32)
    post_tfs_arr = np.zeros(total, dtype=np.int32)
    for tid in range(n_vocab):
        start = int(post_ptr[tid])
        end = int(post_ptr[tid + 1])
        if start == end:
            continue
        post_docs_arr[start:end] = post_docs[tid]
        post_tfs_arr[start:end] = post_tfs[tid]

    clear_chunk_token_cache()
    n = len(chunks)
    avgdl = float(dl.sum()) / n if n else 0.0
    row_of = {cid: i for i, cid in enumerate(chunk_ids)}
    return LexicalIndex(
        chunk_ids=chunk_ids,
        row_of=row_of,
        vocab=vocab,
        post_ptr=post_ptr,
        post_docs=post_docs_arr,
        post_tfs=post_tfs_arr,
        df=np.asarray(df_counts, dtype=np.int32),
        dl=dl,
        avgdl=avgdl,
    )


def save_lexical_index(index_dir: Path, chunks: list[dict[str, Any]]) -> LexicalIndex:
    index_dir = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)
    index = build_lexical_index(chunks)
    fp = fingerprint(chunks)
    vocab_list = [""] * len(index.vocab)
    for token, tid in index.vocab.items():
        vocab_list[tid] = token

    tmp_npz = index_dir / "lexical_tmp.npz"
    tmp_vocab = index_dir / "lexical_vocab_tmp.json"
    tmp_meta = index_dir / "lexical_meta_tmp.json"
    np.savez_compressed(
        tmp_npz,
        post_ptr=index.post_ptr,
        post_docs=index.post_docs,
        post_tfs=index.post_tfs,
        df=index.df,
        dl=index.dl,
        chunk_ids=np.asarray(index.chunk_ids),
    )
    tmp_vocab.write_text(json.dumps(vocab_list, ensure_ascii=False), encoding="utf-8")
    tmp_meta.write_text(
        json.dumps(
            {
                "version": LEXICAL_VERSION,
                "fingerprint": fp,
                "n_chunks": len(chunks),
                "n_vocab": len(vocab_list),
                "avgdl": index.avgdl,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    os.replace(tmp_npz, index_dir / _NPZ)
    os.replace(tmp_vocab, index_dir / _VOCAB)
    os.replace(tmp_meta, index_dir / _META)
    register(index)
    logger.info(
        "Índice lexical em %s (%d chunks, %d termos)",
        index_dir,
        len(chunks),
        len(vocab_list),
    )
    return index


def load_lexical_index(index_dir: Path, *, expected_fingerprint: str | None = None) -> LexicalIndex | None:
    index_dir = Path(index_dir)
    meta_path = index_dir / _META
    npz_path = index_dir / _NPZ
    vocab_path = index_dir / _VOCAB
    if not meta_path.exists() or not npz_path.exists() or not vocab_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("version") != LEXICAL_VERSION:
            return None
        if expected_fingerprint and meta.get("fingerprint") != expected_fingerprint:
            return None
        vocab_list = json.loads(vocab_path.read_text(encoding="utf-8"))
        data = np.load(npz_path, allow_pickle=False)
        chunk_ids = [str(c) for c in data["chunk_ids"].tolist()]
        vocab = {token: i for i, token in enumerate(vocab_list)}
        dl = np.asarray(data["dl"], dtype=np.int32)
        n = len(chunk_ids)
        index = LexicalIndex(
            chunk_ids=chunk_ids,
            row_of={cid: i for i, cid in enumerate(chunk_ids)},
            vocab=vocab,
            post_ptr=np.asarray(data["post_ptr"], dtype=np.int64),
            post_docs=np.asarray(data["post_docs"], dtype=np.int32),
            post_tfs=np.asarray(data["post_tfs"], dtype=np.int32),
            df=np.asarray(data["df"], dtype=np.int32),
            dl=dl,
            avgdl=float(dl.sum()) / n if n else 0.0,
        )
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        logger.warning("Índice lexical ilegível em %s (%s) — será refeito", index_dir, exc)
        return None
    register(index)
    return index


def ensure_lexical_index(index_dir: Path, chunks: list[dict[str, Any]]) -> LexicalIndex:
    fp = fingerprint(chunks)
    loaded = load_lexical_index(index_dir, expected_fingerprint=fp)
    if loaded is not None and loaded.n_docs == len(chunks):
        return loaded
    return save_lexical_index(index_dir, chunks)
