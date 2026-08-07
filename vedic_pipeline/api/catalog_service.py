"""Serviço de catálogo para a aplicação web (corpus JSONL + filtros)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Optional

from vedic_pipeline.common.constants import DEFAULT_CORPUS
from vedic_pipeline.common.corpus import load_corpus


def _preview(text: str, n: int = 280) -> str:
    text = (text or "").strip().replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    if len(text) <= n:
        return text
    return text[: n - 1] + "…"


def document_summary(rec: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": rec.get("id"),
        "title": rec.get("title") or "Sem título",
        "source_url": rec.get("source_url"),
        "tradition": rec.get("tradition") or "unknown",
        "language": rec.get("language") or "und",
        "license": rec.get("license"),
        "char_count": rec.get("char_count") or len(rec.get("text") or ""),
        "retrieved_at": rec.get("retrieved_at"),
        "preview": _preview(rec.get("text") or ""),
    }


def list_documents(
    corpus_path: Path = DEFAULT_CORPUS,
    *,
    tradition: Optional[str] = None,
    language: Optional[str] = None,
    q: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
) -> dict[str, Any]:
    records = load_corpus(Path(corpus_path))
    items: list[dict[str, Any]] = []

    q_norm = (q or "").strip().lower()
    for rec in records:
        if tradition and (rec.get("tradition") or "").lower() != tradition.lower():
            continue
        if language and (rec.get("language") or "").lower() != language.lower():
            continue
        if q_norm:
            blob = " ".join(
                [
                    str(rec.get("title") or ""),
                    str(rec.get("text") or "")[:4000],
                    str(rec.get("source_url") or ""),
                ]
            ).lower()
            if q_norm not in blob:
                continue
        items.append(document_summary(rec))

    total = len(items)
    page = items[offset : offset + limit]
    traditions = sorted({(r.get("tradition") or "unknown") for r in records})
    languages = sorted({(r.get("language") or "und") for r in records})
    return {
        "total": total,
        "offset": offset,
        "limit": limit,
        "items": page,
        "facets": {"traditions": traditions, "languages": languages},
    }


def get_document(
    doc_id: str,
    corpus_path: Path = DEFAULT_CORPUS,
    *,
    include_text: bool = True,
) -> Optional[dict[str, Any]]:
    records = load_corpus(Path(corpus_path))
    for rec in records:
        if rec.get("id") == doc_id:
            out = document_summary(rec)
            if include_text:
                out["text"] = rec.get("text") or ""
            return out
    return None


def corpus_stats(corpus_path: Path = DEFAULT_CORPUS) -> dict[str, Any]:
    records = load_corpus(Path(corpus_path))
    by_tradition: dict[str, int] = {}
    by_language: dict[str, int] = {}
    total_chars = 0
    for rec in records:
        t = (rec.get("tradition") or "unknown").lower()
        lang = (rec.get("language") or "und").lower()
        by_tradition[t] = by_tradition.get(t, 0) + 1
        by_language[lang] = by_language.get(lang, 0) + 1
        total_chars += rec.get("char_count") or len(rec.get("text") or "")
    return {
        "documents": len(records),
        "total_chars": total_chars,
        "by_tradition": by_tradition,
        "by_language": by_language,
        "corpus_path": str(corpus_path),
        "corpus_exists": Path(corpus_path).exists(),
    }
