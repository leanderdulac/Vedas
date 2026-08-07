#!/usr/bin/env python3
"""
Smoke + regressão de retrieval para o corpus Veda Knowledge.

Verifica:
  1. Saúde do corpus / índice numpy / PostgreSQL (opcional)
  2. Busca semântica com 10 queries gold (fixtures/smoke_queries.json)
  3. Ask extractive em 1 query de amostragem

Uso:
  python scripts/smoke_rag.py
  python scripts/smoke_rag.py --backend numpy --strict
  python scripts/smoke_rag.py --skip-ask
  python scripts/smoke_rag.py --json-out data/smoke_report.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:
    pass

from vedic_pipeline.common.constants import DEFAULT_CORPUS, DEFAULT_EMBED_DIR
from vedic_pipeline.common.corpus import load_corpus

DEFAULT_QUERIES = ROOT / "fixtures" / "smoke_queries.json"


def corpus_stats(corpus_path: Path) -> dict[str, Any]:
    from collections import Counter

    docs = load_corpus(corpus_path)
    chars = sum(len(d.get("text") or "") for d in docs)
    return {
        "documents": len(docs),
        "chars": chars,
        "by_tradition": dict(Counter(d.get("tradition") for d in docs)),
        "by_language": dict(Counter(d.get("language") for d in docs)),
        "rv_hymns": sum(
            1
            for d in docs
            if "rigveda rv" in (d.get("title") or "").lower()
            or "rigveda rv" in (d.get("source_url") or "").lower()
        ),
    }


def index_stats(index_dir: Path) -> dict[str, Any]:
    meta_path = index_dir / "index_meta.json"
    emb_path = index_dir / "embeddings.npy"
    chunks_path = index_dir / "chunks.jsonl"
    out: dict[str, Any] = {
        "index_dir": str(index_dir),
        "embeddings_npy": emb_path.exists(),
        "chunks_jsonl": chunks_path.exists(),
        "meta": {},
    }
    if meta_path.exists():
        out["meta"] = json.loads(meta_path.read_text(encoding="utf-8"))
    return out


def match_title(title: str, patterns: list[str]) -> bool:
    t = (title or "").lower()
    return any(p.lower() in t for p in patterns)


def run_retrieval_suite(
    queries: list[dict[str, Any]],
    *,
    backend: str,
    index_dir: Path,
) -> dict[str, Any]:
    from vedic_pipeline.llm.ask import retrieve_hits

    results: list[dict[str, Any]] = []
    passed = 0
    failed = 0

    for q in queries:
        qid = q.get("id") or q["query"][:40]
        top_k = int(q.get("top_k") or 8)
        min_hits = int(q.get("min_hits") or 1)
        expect = list(q.get("expect_title_any") or [])
        tradition = q.get("tradition") or None
        language = q.get("language") or None
        t0 = time.perf_counter()
        try:
            hits, used = retrieve_hits(
                q["query"],
                backend=backend,
                index_dir=index_dir,
                top_k=top_k,
                tradition=tradition,
                language=language,
            )
            elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
            titles = [str(h.get("title") or "") for h in hits]
            ok_hits = len(hits) >= min_hits
            ok_title = (not expect) or any(match_title(t, expect) for t in titles)
            ok = ok_hits and ok_title
            if ok:
                passed += 1
            else:
                failed += 1
            results.append(
                {
                    "id": qid,
                    "query": q["query"],
                    "ok": ok,
                    "backend": used,
                    "n_hits": len(hits),
                    "elapsed_ms": elapsed_ms,
                    "top_titles": titles[:5],
                    "expect_title_any": expect,
                    "reason": (
                        None
                        if ok
                        else (
                            "few_hits"
                            if not ok_hits
                            else "title_mismatch"
                        )
                    ),
                }
            )
        except Exception as exc:  # noqa: BLE001
            failed += 1
            results.append(
                {
                    "id": qid,
                    "query": q["query"],
                    "ok": False,
                    "error": str(exc),
                    "reason": "exception",
                }
            )

    return {
        "passed": passed,
        "failed": failed,
        "total": len(queries),
        "results": results,
    }


def run_ask_sample(backend: str, index_dir: Path) -> dict[str, Any]:
    from vedic_pipeline.llm.ask import ask

    q = "Explain verse 6 of the Isha Upanishad from the sources"
    t0 = time.perf_counter()
    res = ask(
        q,
        backend=backend,
        index_dir=index_dir,
        top_k=5,
        provider="extractive",
        include_hits=True,
        include_prompt=False,
    )
    return {
        "query": q,
        "provider": res.get("provider"),
        "n_hits": len(res.get("hits") or []),
        "answer_chars": len(res.get("answer") or ""),
        "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1),
        "ok": bool((res.get("answer") or "").strip()) and len(res.get("hits") or []) > 0,
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Smoke + regressão de retrieval")
    p.add_argument("--corpus", default=str(DEFAULT_CORPUS))
    p.add_argument("--index", default=str(DEFAULT_EMBED_DIR))
    p.add_argument("--queries", default=str(DEFAULT_QUERIES))
    p.add_argument("--backend", choices=["auto", "numpy", "pgvector"], default="auto")
    p.add_argument("--skip-ask", action="store_true")
    p.add_argument(
        "--strict",
        action="store_true",
        help="Exit 1 se qualquer query de ouro falhar",
    )
    p.add_argument("--json-out", default="", help="Grava relatório JSON")
    args = p.parse_args()

    corpus_path = Path(args.corpus)
    index_dir = Path(args.index)
    report: dict[str, Any] = {
        "corpus": corpus_stats(corpus_path),
        "numpy_index": index_stats(index_dir),
        "database": None,
        "retrieval": None,
        "ask_extractive": None,
    }

    try:
        from vedic_pipeline.storage.db import check_db

        report["database"] = check_db()
    except Exception as exc:  # noqa: BLE001
        report["database"] = {"error": str(exc)}

    if not (index_dir / "embeddings.npy").exists() and args.backend != "pgvector":
        print("AVISO: índice numpy ausente — use build-index ou --backend pgvector")
        if args.backend == "numpy":
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return 2

    payload = json.loads(Path(args.queries).read_text(encoding="utf-8"))
    queries = payload.get("queries") if isinstance(payload, dict) else payload
    report["retrieval"] = run_retrieval_suite(
        queries,
        backend=args.backend,
        index_dir=index_dir,
    )

    if not args.skip_ask:
        try:
            report["ask_extractive"] = run_ask_sample(args.backend, index_dir)
        except Exception as exc:  # noqa: BLE001
            report["ask_extractive"] = {"ok": False, "error": str(exc)}

    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    # resumo legível
    c = report["corpus"]
    r = report["retrieval"]
    print("=== smoke_rag ===")
    print(
        f"corpus: {c['documents']} docs · {c['chars']:,} chars · "
        f"RV hymns≈{c['rv_hymns']} · langs={c['by_language']}"
    )
    meta = report["numpy_index"].get("meta") or {}
    if meta:
        print(
            f"numpy index: {meta.get('n_chunks')} chunks · "
            f"model={meta.get('model_name')} · built={meta.get('built_at')}"
        )
    db = report.get("database") or {}
    if db.get("reachable"):
        print(f"db: ok · counts={db.get('counts')}")
    else:
        print(f"db: {db}")

    print(f"retrieval: {r['passed']}/{r['total']} passed")
    for item in r["results"]:
        mark = "✓" if item.get("ok") else "✗"
        extra = item.get("reason") or item.get("error") or ""
        titles = ", ".join((item.get("top_titles") or [])[:2])
        print(
            f"  {mark} {item['id']}: hits={item.get('n_hits', 0)} "
            f"{item.get('elapsed_ms', '?')}ms  {extra}"
        )
        if titles:
            print(f"      top: {titles}")

    ask = report.get("ask_extractive")
    if ask:
        mark = "✓" if ask.get("ok") else "✗"
        print(
            f"ask extractive: {mark} provider={ask.get('provider')} "
            f"hits={ask.get('n_hits')} answer_chars={ask.get('answer_chars')} "
            f"{ask.get('elapsed_ms', '?')}ms"
        )
        if ask.get("error"):
            print(f"  error: {ask['error']}")

    if args.json_out:
        print(f"report: {args.json_out}")

    if args.strict and r["failed"] > 0:
        return 1
    if ask and not ask.get("ok") and args.strict:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
