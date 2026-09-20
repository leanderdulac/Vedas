#!/usr/bin/env python3
"""Fine-tune do CrossEncoder de domínio (scaffold).

Uso:
  python scripts/train_reranker.py --help
  python scripts/train_reranker.py --dry-run --pairs data/rerank/pairs.jsonl
  python scripts/train_reranker.py --pairs data/rerank/pairs.jsonl --out artifacts/reranker

Não promove o modelo. Só ligue em prod após o CE domain-adapted
bater o híbrido no gold expandido — ver docs/reranker_decision.md.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vedic_pipeline.search.reranker import DEFAULT_RERANKER_MODEL
from vedic_pipeline.train.rerank_pairs import DEFAULT_PAIRS_OUT, iter_jsonl, summarize_pairs

DEFAULT_OUT = ROOT / "artifacts" / "reranker"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fine-tune CrossEncoder (sentence-transformers) a partir de pairs.jsonl.",
    )
    parser.add_argument("--pairs", default=str(DEFAULT_PAIRS_OUT), help="JSONL de treino")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="Diretório do modelo")
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


def load_pairs(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [row for row in iter_jsonl(path) if row.get("query") and row.get("text")]


def write_train_meta(out_dir: Path, meta: dict[str, Any]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / "train_meta.json"
    dest.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return dest


def fit_cross_encoder(
    *,
    base_model: str,
    pairs: list[dict[str, Any]],
    out_dir: Path,
    epochs: int,
    batch_size: int,
    max_length: int,
    learning_rate: float,
) -> None:
    from sentence_transformers import CrossEncoder, InputExample
    from torch.utils.data import DataLoader

    train_rows = [p for p in pairs if str(p.get("split") or "train") != "eval"]
    if not train_rows:
        train_rows = pairs
    examples = [
        InputExample(
            texts=[str(p["query"]), str(p["text"])],
            label=float(p.get("label") or 0),
        )
        for p in train_rows
    ]
    model = CrossEncoder(base_model, max_length=max_length)
    loader = DataLoader(examples, shuffle=True, batch_size=batch_size)
    try:
        model.fit(
            train_dataloader=loader,
            epochs=epochs,
            optimizer_params={"lr": learning_rate},
            show_progress_bar=True,
        )
    except TypeError:
        # sentence-transformers 3.x+: fit() deixou de aceitar train_dataloader
        model.fit(
            examples,
            epochs=epochs,
            batch_size=batch_size,
            show_progress_bar=True,
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(out_dir))


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    pairs_path = Path(args.pairs)
    out_dir = Path(args.out)
    pairs = load_pairs(pairs_path)
    summary = summarize_pairs(pairs)
    meta: dict[str, Any] = {
        "base_model": args.base_model,
        "pairs_path": str(pairs_path),
        "out_dir": str(out_dir),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "learning_rate": args.lr,
        "trained_at": utc_now_iso(),
        "dry_run": bool(args.dry_run),
        "trained": False,
        **summary,
    }

    if args.dry_run:
        meta["skipped"] = "dry-run: CrossEncoder.fit não executado"
        dest = write_train_meta(out_dir, meta)
        print(json.dumps({**meta, "train_meta": str(dest)}, ensure_ascii=False, indent=2))
        return 0

    if not pairs:
        meta["error"] = f"Nenhum par em {pairs_path}. Gere com scripts/build_rerank_pairs.py"
        write_train_meta(out_dir, meta)
        print(json.dumps(meta, ensure_ascii=False, indent=2), file=sys.stderr)
        return 2

    try:
        fit_cross_encoder(
            base_model=args.base_model,
            pairs=pairs,
            out_dir=out_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            max_length=args.max_length,
            learning_rate=args.lr,
        )
    except Exception as exc:  # noqa: BLE001
        meta["error"] = str(exc)
        write_train_meta(out_dir, meta)
        print(json.dumps(meta, ensure_ascii=False, indent=2), file=sys.stderr)
        return 1

    meta["trained"] = True
    dest = write_train_meta(out_dir, meta)
    print(json.dumps({**meta, "train_meta": str(dest)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
