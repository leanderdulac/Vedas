#!/usr/bin/env python3
"""Gera JSONL de pares (query, text, label) para fine-tune do CrossEncoder.

Uso:
  # Dry-run com a fixture minúscula (sem embeddings / sem CE)
  python scripts/build_rerank_pairs.py --dry-run

  # A partir de candidatos pré-computados
  python scripts/build_rerank_pairs.py \\
    --gold fixtures/smoke_queries.json \\
    --candidates fixtures/rerank/candidates_tiny.json \\
    --out data/rerank/pairs.jsonl

  # Híbrido local (exige artifacts/embeddings; força VEDIC_ENABLE_RERANKER=false)
  python scripts/build_rerank_pairs.py --index artifacts/embeddings --top-n 40

Ver docs/reranker_decision.md.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vedic_pipeline.common.constants import DEFAULT_EMBED_DIR
from vedic_pipeline.train.rerank_pairs import (
    DEFAULT_GOLD,
    DEFAULT_PAIRS_OUT,
    DEFAULT_TINY_CANDIDATES,
    build_pairs,
    index_is_ready,
    load_candidates_payload,
    load_gold_queries,
    retrieve_candidates,
    summarize_pairs,
    write_jsonl,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Gera pares JSONL para fine-tune do reranker (fracos + hard negatives).",
    )
    parser.add_argument(
        "--gold",
        default=str(DEFAULT_GOLD),
        help="Gold expandido (fixtures/smoke_queries.json; 19 queries)",
    )
    parser.add_argument(
        "--candidates",
        default="",
        help="JSON pré-computado {query_id: [hits]} ou {queries: [{id, hits}]}",
    )
    parser.add_argument("--out", default=str(DEFAULT_PAIRS_OUT), help="Destino JSONL")
    parser.add_argument("--index", default=str(DEFAULT_EMBED_DIR))
    parser.add_argument("--backend", choices=["auto", "numpy", "pgvector"], default="numpy")
    parser.add_argument("--top-n", type=int, default=40, help="Hits híbridos por query (live)")
    parser.add_argument("--max-negatives", type=int, default=8)
    parser.add_argument("--holdout-ratio", type=float, default=0.2)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Usa a fixture minúscula (ou --candidates) e não chama o híbrido.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    gold_path = Path(args.gold)

    if args.dry_run and not args.candidates:
        gold = load_gold_queries(DEFAULT_TINY_CANDIDATES)
        candidates = load_candidates_payload(DEFAULT_TINY_CANDIDATES)
    elif args.candidates:
        gold = load_gold_queries(gold_path)
        candidates = load_candidates_payload(Path(args.candidates))
    else:
        gold = load_gold_queries(gold_path)
        try:
            candidates = retrieve_candidates(
                gold,
                index_dir=Path(args.index),
                backend=args.backend,
                top_n=args.top_n,
            )
        except FileNotFoundError as exc:
            print(str(exc), file=sys.stderr)
            print(
                "Dica: use --dry-run ou --candidates fixtures/rerank/candidates_tiny.json",
                file=sys.stderr,
            )
            return 2

    pairs = build_pairs(
        gold,
        candidates,
        holdout_ratio=args.holdout_ratio,
        max_negatives=args.max_negatives,
    )
    summary = summarize_pairs(pairs)
    written = write_jsonl(Path(args.out), pairs)
    summary["out"] = str(Path(args.out))
    summary["written"] = written
    summary["dry_run"] = bool(args.dry_run)
    summary["source"] = args.candidates or (
        str(DEFAULT_TINY_CANDIDATES) if args.dry_run else str(args.index)
    )
    if not args.dry_run and not args.candidates:
        summary["index_ready"] = index_is_ready(Path(args.index))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if written else 1


if __name__ == "__main__":
    raise SystemExit(main())
