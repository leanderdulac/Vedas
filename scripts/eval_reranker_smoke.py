#!/usr/bin/env python3
"""Gold smoke A/B: híbrido (CE off) vs Cross-Encoder de domínio.

Por omissão lê o gold expandido (`fixtures/smoke_queries.json`, 19 queries).
O gate de promote deve usar esse arquivo — não o snapshot histórico de 10
nem o CI reduzido (`smoke_queries_ci.json`).

Uso:
  python scripts/eval_reranker_smoke.py --model artifacts/reranker
  python scripts/eval_reranker_smoke.py \\
    --model artifacts/reranker \\
    --queries fixtures/smoke_queries.json \\
    --json-out data/rerank_eval.json

Exit 0: CE de domínio >= híbrido no pass rate E o CE passa Nasadiya.
Exit 1: gate de promote falhou.
Exit 2: índice/modelo local ausente ou suíte não rodou.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from smoke_rag import DEFAULT_QUERIES, run_retrieval_suite  # noqa: E402

from vedic_pipeline.common.constants import DEFAULT_EMBED_DIR
from vedic_pipeline.search.rerank_eval import compare_smoke_suites, gate_exit_code
from vedic_pipeline.search.reranker import (
    looks_like_filesystem_ref,
    reranker_runtime,
    resolve_reranker_model_source,
)
from vedic_pipeline.train.rerank_pairs import index_is_ready, load_gold_queries


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compara gold smoke com e sem Cross-Encoder de domínio.",
    )
    parser.add_argument(
        "--model",
        default="artifacts/reranker",
        help="Diretório local (ex. artifacts/reranker) ou id HF",
    )
    parser.add_argument("--queries", default=str(DEFAULT_QUERIES))
    parser.add_argument("--index", default=str(DEFAULT_EMBED_DIR))
    parser.add_argument("--backend", choices=["auto", "numpy", "pgvector"], default="numpy")
    parser.add_argument("--json-out", default="", help="Grava o relatório JSON")
    return parser


def _load_queries(path: Path) -> list[dict[str, Any]]:
    return load_gold_queries(path)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    queries_path = Path(args.queries)
    index_dir = Path(args.index)
    model_ref = (args.model or "").strip()
    resolved = resolve_reranker_model_source(model_ref)

    if looks_like_filesystem_ref(model_ref) and not Path(resolved).exists():
        print(
            f"Modelo local não encontrado: {model_ref} (resolvido={resolved}). "
            "Treine com scripts/train_reranker.py --out artifacts/reranker",
            file=sys.stderr,
        )
        return 2

    if args.backend != "pgvector" and not index_is_ready(index_dir):
        print(
            f"Índice numpy ausente em {index_dir}. Rode build-index ou passe --backend pgvector.",
            file=sys.stderr,
        )
        return 2

    queries = _load_queries(queries_path)

    with reranker_runtime(enabled=False):
        hybrid = run_retrieval_suite(queries, backend=args.backend, index_dir=index_dir)

    with reranker_runtime(enabled=True, model=resolved):
        ce = run_retrieval_suite(queries, backend=args.backend, index_dir=index_dir)

    comparison = compare_smoke_suites(hybrid, ce, model=resolved)
    report = {
        "hybrid": hybrid,
        "cross_encoder": {**ce, "model": resolved},
        "comparison": comparison,
    }

    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=== eval_reranker_smoke ===")
    print(f"model: {resolved}")
    print(
        f"hybrid: {comparison['hybrid_passed']}/{comparison['hybrid_total']}  "
        f"ce: {comparison['ce_passed']}/{comparison['ce_total']}"
    )
    nas = comparison.get("nasadiya")
    if nas:
        print(
            f"nasadiya: hybrid={'✓' if nas.get('hybrid_ok') else '✗'}  "
            f"ce={'✓' if nas.get('ce_ok') else '✗'}"
        )
        if nas.get("hybrid_top_titles") or nas.get("ce_top_titles"):
            print(f"  hybrid top: {', '.join((nas.get('hybrid_top_titles') or [])[:3])}")
            print(f"  ce top:     {', '.join((nas.get('ce_top_titles') or [])[:3])}")
    for row in comparison.get("per_query") or []:
        mark_h = "✓" if row.get("hybrid_ok") else "✗"
        mark_c = "✓" if row.get("ce_ok") else "✗"
        flag = "  [nasadiya]" if row.get("nasadiya") else ""
        print(f"  {row.get('id')}: hybrid={mark_h} ce={mark_c}{flag}")
    if comparison.get("promote"):
        print("gate: PROMOTE (CE >= híbrido, Nasadiya ok)")
    else:
        print(f"gate: HOLD  reasons={comparison.get('fail_reasons')}")
    if args.json_out:
        print(f"report: {args.json_out}")

    return gate_exit_code(comparison)


if __name__ == "__main__":
    raise SystemExit(main())
