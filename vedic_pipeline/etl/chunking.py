"""Chunking de documentos para embeddings / RAG."""

from __future__ import annotations

from typing import Any, Iterator

from vedic_pipeline.common.constants import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE
from vedic_pipeline.common.corpus import stable_id


def chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[str]:
    """Divide texto em janelas com sobreposição (por caracteres)."""
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        # tenta quebrar em fronteira de parágrafo/espaço
        if end < n:
            window = text[start:end]
            br = max(window.rfind("\n\n"), window.rfind("\n"), window.rfind(" "))
            if br > chunk_size // 3:
                end = start + br
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= n:
            break
        start = max(0, end - overlap)
        if start >= end:
            start = end
    return chunks


def chunk_records(
    records: list[dict[str, Any]],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> Iterator[dict[str, Any]]:
    """Gera chunks com metadados herdados do documento-pai."""
    for rec in records:
        pieces = chunk_text(rec.get("text") or "", chunk_size=chunk_size, overlap=overlap)
        for i, piece in enumerate(pieces):
            yield {
                "chunk_id": stable_id(rec.get("id", ""), str(i), piece[:48]),
                "doc_id": rec.get("id"),
                "chunk_index": i,
                "text": piece,
                "source_url": rec.get("source_url"),
                "title": rec.get("title"),
                "tradition": rec.get("tradition"),
                "language": rec.get("language"),
                "license": rec.get("license"),
                "char_count": len(piece),
            }
