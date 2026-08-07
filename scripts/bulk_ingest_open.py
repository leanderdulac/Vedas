#!/usr/bin/env python3
"""
Ingestão em massa de fontes abertas (public-domain / Gutenberg / sacred-texts).

Recursos:
  - manifesto JSON com gate de licença
  - rate limit entre downloads
  - estado resumível (data/bulk_state.json)
  - dry-run e limite de itens
  - rebuild de índice (numpy / pgvector / both)
  - log JSONL de erros

Uso:
  python scripts/bulk_ingest_open.py --manifest fixtures/sources_open_web.json --dry-run
  python scripts/bulk_ingest_open.py --manifest fixtures/sources_open_web.json --limit 5
  python scripts/bulk_ingest_open.py --manifest fixtures/sources_open_web.json \\
      --rebuild-index --backend both --sync-db
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vedic_pipeline.common.constants import DEFAULT_CORPUS, DEFAULT_EMBED_DIR, DEFAULT_RAW_DIR
from vedic_pipeline.common.corpus import deduplicate_records, load_corpus, rewrite_corpus
from vedic_pipeline.crawler.ingest import build_record, load_manifest
from vedic_pipeline.crawler.licenses import validate_source
from vedic_pipeline.crawler.download import download_source
from vedic_pipeline.etl.extractors import extract_text

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("bulk_ingest")

STATE_PATH = ROOT / "data" / "bulk_state.json"
ERROR_LOG = ROOT / "data" / "bulk_errors.jsonl"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"completed_urls": [], "failed_urls": {}, "updated_at": None}
    return json.loads(path.read_text(encoding="utf-8"))


def save_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state["updated_at"] = utc_now()
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def log_error(entry: dict[str, Any]) -> None:
    ERROR_LOG.parent.mkdir(parents=True, exist_ok=True)
    with ERROR_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def strip_gutenberg_boilerplate(text: str) -> str:
    """Remove cabeçalho/rodapé típicos do Project Gutenberg."""
    start_markers = [
        "*** START OF THE PROJECT GUTENBERG EBOOK",
        "*** START OF THIS PROJECT GUTENBERG EBOOK",
        "*END*THE SMALL PRINT",
    ]
    end_markers = [
        "*** END OF THE PROJECT GUTENBERG EBOOK",
        "*** END OF THIS PROJECT GUTENBERG EBOOK",
        "End of the Project Gutenberg",
        "End of Project Gutenberg",
    ]
    upper = text
    start = 0
    for m in start_markers:
        i = upper.find(m)
        if i >= 0:
            nl = upper.find("\n", i)
            start = nl + 1 if nl >= 0 else i + len(m)
            break
    end = len(text)
    for m in end_markers:
        i = upper.find(m, start)
        if i >= 0:
            end = i
            break
    return text[start:end].strip()


def process_source(
    src: dict[str, Any],
    *,
    raw_dir: Path,
    min_chars: int,
) -> dict[str, Any]:
    ok, reason = validate_source(src)
    if not ok:
        return {"status": "skipped", "reason": reason, "url": src.get("url")}

    local = download_source(src, raw_dir=raw_dir)
    text = extract_text(local)
    if "gutenberg.org" in (src.get("url") or ""):
        text = strip_gutenberg_boilerplate(text)
    if len(text) < min_chars:
        return {
            "status": "empty",
            "reason": f"texto curto ({len(text)} chars)",
            "url": src.get("url"),
        }
    rec = build_record(src, text)
    return {"status": "ok", "record": rec, "url": src.get("url"), "chars": len(text)}


def rebuild_indexes(corpus: Path, backend: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if backend in {"numpy", "both"}:
        from vedic_pipeline.search.embeddings import build_embedding_index
        from vedic_pipeline.search.rag import get_index

        meta = build_embedding_index(
            corpus_path=corpus,
            out_dir=DEFAULT_EMBED_DIR,
            chunk_size=900,
            overlap=150,
        )
        get_index(DEFAULT_EMBED_DIR, reload=True)
        out["numpy"] = meta
    if backend in {"pgvector", "both"}:
        from vedic_pipeline.storage.vectors import build_pgvector_index

        out["pgvector"] = build_pgvector_index(
            corpus_path=corpus,
            chunk_size=900,
            overlap=150,
        )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Bulk ingest de fontes abertas")
    parser.add_argument(
        "--manifest",
        default=str(ROOT / "fixtures" / "sources_open_web.json"),
    )
    parser.add_argument("--corpus", default=str(DEFAULT_CORPUS))
    parser.add_argument("--raw-dir", default=str(DEFAULT_RAW_DIR))
    parser.add_argument("--state", default=str(STATE_PATH))
    parser.add_argument("--delay", type=float, default=1.2, help="Segundos entre downloads")
    parser.add_argument("--min-chars", type=int, default=120)
    parser.add_argument("--limit", type=int, default=0, help="0 = todos")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true", help="Reprocessa URLs já completadas")
    parser.add_argument("--rebuild-index", action="store_true")
    parser.add_argument("--backend", choices=["numpy", "pgvector", "both"], default="both")
    parser.add_argument("--sync-db", action="store_true")
    parser.add_argument(
        "--include-local-fixtures",
        action="store_true",
        help="Também roda fixtures/sources_vedic_corpus.json",
    )
    args = parser.parse_args()

    sources = load_manifest(args.manifest)
    # remove entradas download=false (catálogo)
    sources = [s for s in sources if s.get("download") is not False]

    if args.include_local_fixtures:
        local_m = ROOT / "fixtures" / "sources_vedic_corpus.json"
        if local_m.exists():
            sources = load_manifest(local_m) + sources

    state = load_state(Path(args.state))
    completed = set(state.get("completed_urls") or [])
    corpus_path = Path(args.corpus)
    raw_dir = Path(args.raw_dir)

    existing = load_corpus(corpus_path)
    existing_fps = {r.get("fingerprint") for r in existing if r.get("fingerprint")}
    existing_urls = {r.get("source_url") for r in existing if r.get("source_url")}

    stats = {
        "manifest": args.manifest,
        "total": len(sources),
        "ok": 0,
        "skipped": 0,
        "failed": 0,
        "empty": 0,
        "duplicates": 0,
        "added": 0,
        "started_at": utc_now(),
    }

    new_records: list[dict[str, Any]] = []
    processed = 0

    for src in sources:
        url = (src.get("url") or "").strip()
        if not url:
            stats["skipped"] += 1
            continue
        if url.startswith("fixtures/") and not Path(url).exists():
            # caminho relativo
            pass

        if not args.force and url in completed:
            stats["skipped"] += 1
            logger.info("skip (state): %s", url[:80])
            continue

        if args.limit and processed >= args.limit:
            break
        processed += 1

        if args.dry_run:
            ok, reason = validate_source(src)
            logger.info("dry-run %s | %s | %s", "OK" if ok else "NO", reason, src.get("title"))
            if ok:
                stats["ok"] += 1
            else:
                stats["skipped"] += 1
            continue

        logger.info("(%d) Baixando: %s", processed, src.get("title") or url)
        try:
            result = process_source(src, raw_dir=raw_dir, min_chars=args.min_chars)
        except Exception as exc:  # noqa: BLE001
            stats["failed"] += 1
            state.setdefault("failed_urls", {})[url] = str(exc)
            log_error({"url": url, "error": str(exc), "at": utc_now(), "title": src.get("title")})
            logger.error("Falha: %s — %s", url, exc)
            save_state(Path(args.state), state)
            time.sleep(args.delay)
            continue

        status = result["status"]
        if status == "skipped":
            stats["skipped"] += 1
            logger.warning("Ignorado: %s", result.get("reason"))
        elif status == "empty":
            stats["empty"] += 1
            logger.warning("Vazio: %s", result.get("reason"))
            state.setdefault("failed_urls", {})[url] = result.get("reason")
        elif status == "ok":
            rec = result["record"]
            if rec["fingerprint"] in existing_fps or rec["source_url"] in existing_urls:
                stats["duplicates"] += 1
                logger.info("Duplicata: %s", rec.get("title"))
            else:
                new_records.append(rec)
                existing_fps.add(rec["fingerprint"])
                existing_urls.add(rec["source_url"])
                stats["added"] += 1
                stats["ok"] += 1
                logger.info("OK +%d chars — %s", result.get("chars"), rec.get("title"))
            completed.add(url)
            state["completed_urls"] = sorted(completed)
            state.get("failed_urls", {}).pop(url, None)

        save_state(Path(args.state), state)
        time.sleep(args.delay)

    if new_records and not args.dry_run:
        corpus_path.parent.mkdir(parents=True, exist_ok=True)
        # append + dedupe rewrite
        from vedic_pipeline.common.corpus import append_corpus

        append_corpus(corpus_path, new_records)
        all_recs = deduplicate_records(load_corpus(corpus_path))
        rewrite_corpus(corpus_path, all_recs)
        stats["corpus_documents"] = len(all_recs)
        logger.info("Corpus agora com %d documentos", len(all_recs))

    if args.sync_db and not args.dry_run:
        from vedic_pipeline.storage.catalog import sync_corpus_to_db

        stats["db_sync"] = sync_corpus_to_db(corpus_path)

    if args.rebuild_index and not args.dry_run:
        logger.info("Reconstruindo índices (%s)…", args.backend)
        try:
            stats["index"] = rebuild_indexes(corpus_path, args.backend)
        except Exception as exc:  # noqa: BLE001
            logger.error("Falha no índice: %s", exc)
            stats["index_error"] = str(exc)

    stats["finished_at"] = utc_now()
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0 if stats["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
