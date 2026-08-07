"""Orquestra manifesto → download → ETL → corpus JSONL."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from vedic_pipeline.common.constants import DEFAULT_CORPUS, DEFAULT_RAW_DIR
from vedic_pipeline.common.corpus import (
    append_corpus,
    content_fingerprint,
    load_corpus,
    stable_id,
    utc_now_iso,
)
from vedic_pipeline.crawler.download import download_source
from vedic_pipeline.crawler.licenses import validate_source
from vedic_pipeline.etl.extractors import extract_text

logger = logging.getLogger("vedic_pipeline.ingest")


def load_manifest(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Manifesto não encontrado: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    sources = data.get("sources") if isinstance(data, dict) else data
    if not isinstance(sources, list):
        raise ValueError("Manifesto inválido: esperado {'sources': [...]}")
    return sources


def build_record(src: dict[str, Any], text: str) -> dict[str, Any]:
    url = src.get("url") or ""
    title = src.get("title") or ""
    fp = content_fingerprint(text)
    return {
        "id": stable_id(url, title, fp[:16]),
        "text": text,
        "source_url": url,
        "title": title,
        "tradition": (src.get("tradition") or "unknown").lower(),
        "language": (src.get("language") or "und").lower(),
        "license": (src.get("license") or "").lower(),
        "retrieved_at": utc_now_iso(),
        "fingerprint": fp,
        "char_count": len(text),
    }


def ingest_manifest(
    manifest_path: str | Path,
    corpus_path: Path = DEFAULT_CORPUS,
    raw_dir: Path = DEFAULT_RAW_DIR,
    min_chars: int = 80,
) -> dict[str, Any]:
    sources = load_manifest(manifest_path)
    existing = load_corpus(corpus_path)
    existing_fps = {r.get("fingerprint") for r in existing if r.get("fingerprint")}
    existing_urls = {r.get("source_url") for r in existing if r.get("source_url")}

    stats: dict[str, Any] = {
        "manifest": str(manifest_path),
        "total_sources": len(sources),
        "accepted": 0,
        "skipped_license": 0,
        "skipped_download_flag": 0,
        "failed": 0,
        "empty_or_short": 0,
        "duplicates": 0,
        "added": 0,
        "corpus_path": str(corpus_path),
        "errors": [],
    }

    new_records: list[dict[str, Any]] = []

    for src in sources:
        ok, reason = validate_source(src)
        if not ok:
            if "download=false" in reason:
                stats["skipped_download_flag"] += 1
            else:
                stats["skipped_license"] += 1
            logger.warning(
                "Fonte ignorada (%s): %s",
                reason,
                src.get("title") or src.get("url"),
            )
            continue

        stats["accepted"] += 1
        try:
            local = download_source(src, raw_dir=raw_dir)
            text = extract_text(local)
        except Exception as exc:  # noqa: BLE001
            stats["failed"] += 1
            msg = f"{src.get('url')}: {exc}"
            stats["errors"].append(msg)
            logger.error("Falha no download/extração: %s", msg)
            continue

        if len(text) < min_chars:
            stats["empty_or_short"] += 1
            logger.warning(
                "Texto muito curto (%d chars) para %s",
                len(text),
                src.get("title"),
            )
            continue

        rec = build_record(src, text)
        if rec["fingerprint"] in existing_fps or rec["source_url"] in existing_urls:
            stats["duplicates"] += 1
            logger.info("Duplicata ignorada: %s", rec["title"])
            continue

        existing_fps.add(rec["fingerprint"])
        existing_urls.add(rec["source_url"])
        new_records.append(rec)

    if new_records:
        added = append_corpus(corpus_path, new_records)
        stats["added"] = added
        logger.info("Adicionados %d registros em %s", added, corpus_path)
    else:
        logger.info("Nenhum registro novo para adicionar.")

    return stats
