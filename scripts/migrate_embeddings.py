#!/usr/bin/env python3
"""Migração de modelo de embeddings: backup + reindex (numpy e/ou pgvector).

Trocar o modelo de embeddings exige reconstruir o índice por completo
(vetores não são intercambiáveis entre modelos). Este script:

  1. faz backup do índice numpy atual (artifacts/embeddings -> .bak-<ts>);
  2. reconstrói o índice com o novo modelo (--model; default
     intfloat/multilingual-e5-small, 384-dim — mesma dim do MiniLM atual);
  3. reconstrói o pgvector se --backend incluir pgvector e o DB estiver
     configurado (o schema é recriado se a dim mudar);
  4. roda o smoke do gold set para validar a migração.

Exemplos:
  VEDIC_EMBEDDING_MODEL=intfloat/multilingual-e5-small \
    python scripts/migrate_embeddings.py --backend both --smoke

  python scripts/migrate_embeddings.py --model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 \
    --backend numpy --no-smoke
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vedic_pipeline.common.constants import DEFAULT_CORPUS, DEFAULT_EMBED_DIR  # noqa: E402

DEFAULT_TARGET_MODEL = "intfloat/multilingual-e5-small"


def backup_index() -> Path | None:
    if not (DEFAULT_EMBED_DIR / "embeddings.npy").exists():
        return None
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    dest = DEFAULT_EMBED_DIR.parent / f"embeddings.bak-{stamp}"
    shutil.copytree(DEFAULT_EMBED_DIR, dest)
    print(f"backup: {DEFAULT_EMBED_DIR} -> {dest}")
    return dest


def run_smoke_checked(queries_path: Path) -> dict:
    """Roda o smoke_rag como subprocesso (mesma semântica do CI) e devolve o resumo."""
    import os
    import subprocess

    env = dict(os.environ)
    env["DATABASE_URL"] = ""  # força numpy no smoke (pgvector é validado no build)
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "smoke_rag.py"),
        "--backend",
        "numpy",
        "--queries",
        str(queries_path),
        "--strict",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3600, env=env)
    except subprocess.TimeoutExpired:
        return {"passed": None, "total": None, "error": "timeout do smoke"}
    tail = (proc.stdout or "").strip().splitlines()
    retrieval_line = next((ln for ln in tail if ln.startswith("retrieval:")), "")
    out = {"exit": proc.returncode, "retrieval": retrieval_line}
    if proc.returncode != 0:
        out["stderr_tail"] = "\n".join((proc.stderr or "").strip().splitlines()[-6:])
    return out


def migration_status(summary: dict) -> int:
    """1 se o smoke falhou ou estourou o tempo; 0 se passou ou não rodou."""
    smoke = summary.get("smoke")
    if not isinstance(smoke, dict):
        return 0
    if smoke.get("error"):
        return 1
    code = smoke.get("exit")
    if isinstance(code, int) and code != 0:
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_TARGET_MODEL)
    parser.add_argument("--corpus", default=str(DEFAULT_CORPUS))
    parser.add_argument("--backend", choices=["numpy", "pgvector", "both"], default="both")
    parser.add_argument("--chunk-size", type=int, default=800)
    parser.add_argument("--overlap", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--no-backup", action="store_true")
    parser.add_argument("--smoke", choices=["ci", "gold", "off"], default="ci")
    args = parser.parse_args()

    import os

    if os.environ.get("HF_HUB_OFFLINE", "").strip() == "1":
        print("⚠ HF_HUB_OFFLINE=1 — o download do modelo falhará se ele não estiver em cache local.")
        print("  Desative temporariamente (unset HF_HUB_OFFLINE) para baixar o novo modelo.")

    summary: dict = {"model": args.model, "backend": args.backend, "steps": []}
    started = time.time()

    if not args.no_backup:
        summary["backup"] = str(backup_index())

    if args.backend in {"numpy", "both"}:
        from vedic_pipeline.search.embeddings import build_embedding_index

        meta = build_embedding_index(
            corpus_path=Path(args.corpus),
            out_dir=DEFAULT_EMBED_DIR,
            model_name=args.model,
            chunk_size=args.chunk_size,
            overlap=args.overlap,
            batch_size=args.batch_size,
        )
        summary["numpy"] = {
            "n_chunks": meta.get("n_chunks"),
            "dim": meta.get("dim"),
            "model": meta.get("model_name"),
        }
        print(f"numpy: {meta.get('n_chunks')} chunks, dim={meta.get('dim')}, model={meta.get('model_name')}")

    if args.backend in {"pgvector", "both"}:
        from vedic_pipeline.storage.db import get_database_url

        if get_database_url():
            from vedic_pipeline.storage.vectors import build_pgvector_index

            pg = build_pgvector_index(
                corpus_path=Path(args.corpus),
                model_name=args.model,
                chunk_size=args.chunk_size,
                overlap=args.overlap,
                batch_size=args.batch_size,
            )
            summary["pgvector"] = {
                "n_chunks": pg.get("n_chunks"),
                "dim": pg.get("dim"),
            }
            print(f"pgvector: {pg.get('n_chunks')} chunks, dim={pg.get('dim')}")
        else:
            print("pgvector: pulado (DATABASE_URL não configurado)")

    if args.smoke != "off":
        gold = ROOT / "fixtures" / ("smoke_queries.json" if args.smoke == "gold" else "smoke_queries_ci.json")
        if gold.exists():
            summary["smoke"] = run_smoke_checked(gold)
            print(f"smoke ({args.smoke}):", summary["smoke"].get("retrieval") or summary["smoke"])
        else:
            print(f"smoke: fixture {gold.name} ausente — pulado")

    summary["elapsed_s"] = round(time.time() - started, 1)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return migration_status(summary)


if __name__ == "__main__":
    raise SystemExit(main())
