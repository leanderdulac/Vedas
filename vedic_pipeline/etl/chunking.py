"""Chunking de documentos para embeddings / RAG."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from vedic_pipeline.common.constants import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE
from vedic_pipeline.common.corpus import stable_id
from vedic_pipeline.etl.structure import (
    format_chunk_text,
    format_locator,
    group_units,
    parse_document_units,
)


def chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[str]:
    """Divide texto em janelas com sobreposição (por caracteres)."""
    if chunk_size <= 0:
        raise ValueError("chunk_size deve ser positivo")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap deve ser >= 0 e menor que chunk_size")
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
        # Uma quebra antecipada pode deixar o trecho menor que overlap.
        # Nesse caso, avance até end para não repetir a mesma janela.
        next_start = end - overlap
        start = next_start if next_start > start else end
    return chunks


def _base_chunk(rec: dict[str, Any], index: int, text: str) -> dict[str, Any]:
    return {
        "chunk_id": stable_id(rec.get("id", ""), str(index), text[:48]),
        "doc_id": rec.get("id"),
        "chunk_index": index,
        "text": text,
        "source_url": rec.get("source_url"),
        "title": rec.get("title"),
        "tradition": rec.get("tradition"),
        "language": rec.get("language"),
        "license": rec.get("license"),
        "char_count": len(text),
    }


def chunk_records(
    records: list[dict[str, Any]],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> Iterator[dict[str, Any]]:
    """Gera chunks com metadados herdados do documento-pai.

    Ṛgveda, Gītā e Yoga-sūtra usam unidades canônicas (verso/sūtra).
    Demais obras caem no recorte por caracteres.
    """
    for rec in records:
        units = parse_document_units(rec)
        index = 0
        if units:
            for group in group_units(units, max_chars=chunk_size, max_verses=2):
                first, last = group[0], group[-1]
                text = format_chunk_text(group)
                verse_end = last.verse if last.verse is not None else first.verse
                locator = format_locator(
                    first.work,
                    first.book,
                    first.hymn,
                    first.verse,
                    verse_end=verse_end,
                )
                chunk = _base_chunk(rec, index, text)
                chunk.update(first.to_chunk_fields(verse_end=verse_end, locator=locator))
                yield chunk
                index += 1
            continue
        pieces = chunk_text(rec.get("text") or "", chunk_size=chunk_size, overlap=overlap)
        for piece in pieces:
            yield _base_chunk(rec, index, piece)
            index += 1
