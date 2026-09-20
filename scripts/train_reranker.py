#!/usr/bin/env python3
"""Fine-tune do CrossEncoder de domínio.

Uso:
  python scripts/train_reranker.py --help
  python scripts/train_reranker.py --dry-run --pairs data/rerank/pairs.jsonl
  python scripts/train_reranker.py --pairs data/rerank/pairs.jsonl --out artifacts/reranker

Não promove o modelo. Só ligue em prod após
`scripts/eval_reranker_smoke.py` passar o gate — ver docs/reranker_decision.md.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vedic_pipeline.search.reranker import DEFAULT_RERANKER_DIR, DEFAULT_RERANKER_MODEL
from vedic_pipeline.train.rerank_pairs import DEFAULT_PAIRS_OUT
from vedic_pipeline.train.reranker import train_reranker


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fine-tune CrossEncoder (sentence-transformers) a partir de pairs.jsonl.",
    )
    parser.add_argument("--pairs", default=str(DEFAULT_PAIRS_OUT), help="JSONL de treino")
    parser.add_argument("--out", default=str(DEFAULT_RERANKER_DIR), help="Diretório do modelo")
    parser.add_argument("--base-model", default=DEFAULT_RERANKER_MODEL)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Não baixa modelo nem treina; só valida pares e grava train_meta.json",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    code, meta = train_reranker(
        pairs_path=Path(args.pairs),
        out_dir=Path(args.out),
        base_model=args.base_model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_length=args.max_length,
        learning_rate=args.lr,
        dry_run=bool(args.dry_run),
    )
    stream = sys.stdout if code == 0 else sys.stderr
    print(json.dumps(meta, ensure_ascii=False, indent=2), file=stream)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
