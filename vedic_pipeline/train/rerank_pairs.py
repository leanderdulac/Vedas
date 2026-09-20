"""Geração de pares fracos para fine-tune do CrossEncoder.

Positivos: hits cujo título casa com ``expect_title_any``.
Negativos hard: hits de alto score híbrido que *não* casam o título.

Pode ler candidatos pré-computados (JSON) — caminho de CI / dry-run —
ou recuperar via híbrido local quando o índice existir.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from vedic_pipeline.common.constants import DEFAULT_EMBED_DIR, PROJECT_ROOT

logger = logging.getLogger("vedic_pipeline.train.rerank_pairs")

DEFAULT_GOLD = PROJECT_ROOT / "fixtures" / "smoke_queries.json"
DEFAULT_TINY_CANDIDATES = PROJECT_ROOT / "fixtures" / "rerank" / "candidates_tiny.json"
DEFAULT_PAIRS_OUT = PROJECT_ROOT / "data" / "rerank" / "pairs.jsonl"

PAIR_FIELDS = ("query", "text", "label", "query_id", "doc_title", "split")


def title_matches(title: str, patterns: Iterable[str]) -> bool:
    haystack = (title or "").lower()
    return any(pattern.lower() in haystack for pattern in patterns if pattern)


def load_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_gold_queries(path: Path) -> list[dict[str, Any]]:
    payload = load_json(path)
    if isinstance(payload, list):
        return list(payload)
    queries = payload.get("queries")
    if not isinstance(queries, list):
        raise ValueError(f"Gold set sem lista 'queries': {path}")
    return queries


def load_candidates_payload(path: Path) -> dict[str, list[dict[str, Any]]]:
    """Aceita ``{query_id: [hits]}`` ou ``{queries: [{id, hits}]}``."""
    payload = load_json(path)
    if isinstance(payload, dict) and "queries" in payload:
        by_id: dict[str, list[dict[str, Any]]] = {}
        for item in payload.get("queries") or []:
            qid = str(item.get("id") or item.get("query_id") or "")
            hits = item.get("hits") or item.get("candidates") or []
            if qid:
                by_id[qid] = list(hits)
        return by_id
    if isinstance(payload, dict):
        return {str(key): list(value) for key, value in payload.items()}
    raise ValueError(f"Candidatos JSON inválidos: {path}")


def assign_splits(query_ids: Iterable[str], holdout_ratio: float = 0.2) -> dict[str, str]:
    ids = sorted({str(qid) for qid in query_ids})
    if len(ids) < 2 or holdout_ratio <= 0:
        return {qid: "train" for qid in ids}
    n_hold = max(1, int(round(len(ids) * holdout_ratio)))
    n_hold = min(n_hold, len(ids) - 1)
    hold = set(ids[-n_hold:])
    return {qid: ("eval" if qid in hold else "train") for qid in ids}


def pair_record(
    *,
    query: str,
    text: str,
    label: int,
    query_id: str,
    doc_title: str,
    split: str,
) -> dict[str, Any]:
    return {
        "query": query,
        "text": text,
        "label": int(label),
        "query_id": query_id,
        "doc_title": doc_title,
        "split": split,
    }


def pairs_from_hits(
    *,
    query: str,
    query_id: str,
    expect_title_any: list[str],
    hits: list[dict[str, Any]],
    split: str = "train",
    max_negatives: int = 8,
) -> list[dict[str, Any]]:
    """Emite pares 0/1 a partir de hits híbridos já recuperados."""
    positives: list[dict[str, Any]] = []
    negatives: list[dict[str, Any]] = []
    for hit in hits:
        text = str(hit.get("text") or "").strip()
        if not text:
            continue
        title = str(hit.get("title") or hit.get("doc_title") or "")
        rec = pair_record(
            query=query,
            text=text,
            label=1 if title_matches(title, expect_title_any) else 0,
            query_id=query_id,
            doc_title=title,
            split=split,
        )
        if rec["label"] == 1:
            positives.append(rec)
        else:
            negatives.append(rec)
    return positives + negatives[: max(0, max_negatives)]


def build_pairs(
    queries: list[dict[str, Any]],
    candidates_by_id: dict[str, list[dict[str, Any]]],
    *,
    holdout_ratio: float = 0.2,
    max_negatives: int = 8,
) -> list[dict[str, Any]]:
    splits = assign_splits((q.get("id") or q.get("query") or "") for q in queries)
    pairs: list[dict[str, Any]] = []
    for spec in queries:
        qid = str(spec.get("id") or spec.get("query") or "")
        query = str(spec.get("query") or "")
        expect = [str(p) for p in (spec.get("expect_title_any") or [])]
        hits = candidates_by_id.get(qid) or spec.get("hits") or []
        if not query or not hits:
            logger.info("Sem candidatos para query_id=%s — pulando", qid)
            continue
        pairs.extend(
            pairs_from_hits(
                query=query,
                query_id=qid,
                expect_title_any=expect,
                hits=list(hits),
                split=splits.get(qid, "train"),
                max_negatives=max_negatives,
            )
        )
    return pairs


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with dest.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def summarize_pairs(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    n_pos = sum(1 for p in pairs if int(p.get("label") or 0) == 1)
    n_neg = len(pairs) - n_pos
    by_split: dict[str, int] = {}
    by_query: dict[str, int] = {}
    for pair in pairs:
        split = str(pair.get("split") or "train")
        qid = str(pair.get("query_id") or "")
        by_split[split] = by_split.get(split, 0) + 1
        by_query[qid] = by_query.get(qid, 0) + 1
    return {
        "pairs": len(pairs),
        "positives": n_pos,
        "negatives": n_neg,
        "by_split": by_split,
        "queries": len(by_query),
    }


@contextmanager
def reranker_forced_off() -> Iterator[None]:
    """Garante híbrido sem CE ao minerar candidatos."""
    from vedic_pipeline.search.reranker import invalidate_reranker

    previous = os.environ.get("VEDIC_ENABLE_RERANKER")
    os.environ["VEDIC_ENABLE_RERANKER"] = "false"
    invalidate_reranker()
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("VEDIC_ENABLE_RERANKER", None)
        else:
            os.environ["VEDIC_ENABLE_RERANKER"] = previous
        invalidate_reranker()


def index_is_ready(index_dir: Path) -> bool:
    root = Path(index_dir)
    return (root / "embeddings.npy").exists() and (root / "chunks.jsonl").exists()


def retrieve_candidates(
    queries: list[dict[str, Any]],
    *,
    index_dir: Path = DEFAULT_EMBED_DIR,
    backend: str = "numpy",
    top_n: int = 40,
) -> dict[str, list[dict[str, Any]]]:
    """Recupera top-N híbrido (CE off). Exige índice local."""
    if not index_is_ready(index_dir) and backend != "pgvector":
        raise FileNotFoundError(
            f"Índice numpy ausente em {index_dir}. "
            "Passe --candidates com JSON pré-computado ou rode build-index."
        )

    from vedic_pipeline.llm.ask import retrieve_hits

    by_id: dict[str, list[dict[str, Any]]] = {}
    with reranker_forced_off():
        for spec in queries:
            qid = str(spec.get("id") or spec.get("query") or "")
            query = str(spec.get("query") or "")
            if not qid or not query:
                continue
            hits, _used = retrieve_hits(
                query,
                backend=backend,
                index_dir=Path(index_dir),
                top_k=top_n,
                tradition=spec.get("tradition") or None,
                language=spec.get("language") or None,
            )
            by_id[qid] = hits
    return by_id
