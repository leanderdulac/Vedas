"""Fine-tune do CrossEncoder de domínio a partir de pares JSONL."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from vedic_pipeline.common.corpus import utc_now_iso
from vedic_pipeline.search.reranker import DEFAULT_RERANKER_DIR, DEFAULT_RERANKER_MODEL
from vedic_pipeline.train.rerank_pairs import iter_jsonl, summarize_pairs

logger = logging.getLogger("vedic_pipeline.train.reranker")

EVAL_SPLITS = frozenset({"holdout", "eval", "test", "valid", "validation", "dev"})


def detect_device() -> str:
    """Prefer CUDA, then Apple MPS, then CPU. Never silently skip MPS on Mac."""
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        mps = getattr(getattr(torch, "backends", None), "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
        return "cpu"
    except ImportError:
        return "cpu"


def resolve_device(device: str | None = None) -> str:
    """`auto` / None → detect_device(); otherwise honor an explicit backend."""
    if device in (None, "", "auto"):
        return detect_device()
    return str(device).strip().lower()


def coerce_label(value: Any) -> float:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "pos", "positive"}:
        return 1.0
    if text in {"0", "false", "no", "neg", "negative"}:
        return 0.0
    return float(text)


def normalize_pair_row(row: dict[str, Any]) -> dict[str, Any] | None:
    """Aceita query/text/label (aliases: passage, document) + query_id/split opcionais."""
    query = str(row.get("query") or "").strip()
    text = str(row.get("text") or row.get("passage") or row.get("document") or "").strip()
    if not query or not text:
        return None
    label_raw = row.get("label", 0)
    try:
        label = coerce_label(label_raw)
    except (TypeError, ValueError):
        return None
    out = dict(row)
    out["query"] = query
    out["text"] = text
    out["label"] = label
    if "query_id" in row and row["query_id"] is not None:
        out["query_id"] = str(row["query_id"])
    if "split" in row and row["split"] is not None:
        out["split"] = str(row["split"]).strip().lower() or "train"
    else:
        out.setdefault("split", "train")
    return out


def load_pairs(path: Path) -> list[dict[str, Any]]:
    if not Path(path).exists():
        return []
    rows: list[dict[str, Any]] = []
    for raw in iter_jsonl(path):
        normalized = normalize_pair_row(raw)
        if normalized:
            rows.append(normalized)
    return rows


def is_train_split(split: Any) -> bool:
    value = str(split or "train").strip().lower() or "train"
    return value not in EVAL_SPLITS


def select_train_pairs(
    pairs: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], bool]:
    """Prefere split != holdout/eval; se não houver treino, usa todos (fallback)."""
    train = [p for p in pairs if is_train_split(p.get("split"))]
    skipped = [p for p in pairs if not is_train_split(p.get("split"))]
    if train:
        return train, skipped, False
    return list(pairs), [], bool(pairs)


def write_train_meta(out_dir: Path, meta: dict[str, Any]) -> Path:
    dest_dir = Path(out_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / "train_meta.json"
    dest.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return dest


def build_train_meta(
    *,
    base_model: str,
    pairs_path: Path,
    out_dir: Path,
    pairs: list[dict[str, Any]],
    train_rows: list[dict[str, Any]],
    skipped: list[dict[str, Any]],
    used_eval_fallback: bool,
    epochs: int,
    batch_size: int,
    max_length: int,
    learning_rate: float,
    dry_run: bool,
    device: str | None = None,
) -> dict[str, Any]:
    summary = summarize_pairs(pairs)
    train_summary = summarize_pairs(train_rows)
    return {
        "base_model": base_model,
        "pairs_path": str(pairs_path),
        "out_dir": str(out_dir),
        "epochs": epochs,
        "batch_size": batch_size,
        "max_length": max_length,
        "learning_rate": learning_rate,
        "trained_at": utc_now_iso(),
        "dry_run": bool(dry_run),
        "trained": False,
        "device": resolve_device(device),
        "n_pairs_total": summary["pairs"],
        "n_train": len(train_rows),
        "n_skipped_eval_split": len(skipped),
        "used_eval_fallback": bool(used_eval_fallback),
        "train_positives": train_summary["positives"],
        "train_negatives": train_summary["negatives"],
        **summary,
    }


def _persist_model(model: Any, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if hasattr(model, "save_pretrained"):
        model.save_pretrained(str(out_dir))
    elif hasattr(model, "save"):
        model.save(str(out_dir))


def _fit_with_trainer(
    *,
    model: Any,
    train_rows: list[dict[str, Any]],
    out_dir: Path,
    epochs: int,
    batch_size: int,
    learning_rate: float,
) -> None:
    from datasets import Dataset
    from sentence_transformers.cross_encoder.losses import BinaryCrossEntropyLoss
    from sentence_transformers.cross_encoder.trainer import CrossEncoderTrainer
    from sentence_transformers.cross_encoder.training_args import CrossEncoderTrainingArguments

    dataset = Dataset.from_dict(
        {
            "text_1": [str(p["query"]) for p in train_rows],
            "text_2": [str(p["text"]) for p in train_rows],
            "label": [float(p["label"]) for p in train_rows],
        }
    )
    args = CrossEncoderTrainingArguments(
        output_dir=str(Path(out_dir) / "checkpoints"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_ratio=0.1,
        fp16=False,
        report_to=[],
        save_strategy="no",
        logging_steps=10,
    )
    trainer = CrossEncoderTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        loss=BinaryCrossEntropyLoss(model),
    )
    trainer.train()
    _persist_model(model, Path(out_dir))


def _fit_with_legacy(
    *,
    model: Any,
    train_rows: list[dict[str, Any]],
    out_dir: Path,
    epochs: int,
    batch_size: int,
    learning_rate: float,
) -> None:
    from torch.utils.data import DataLoader

    try:
        from sentence_transformers import InputExample
    except ImportError:
        from sentence_transformers.readers import InputExample

    examples = [
        InputExample(texts=[str(p["query"]), str(p["text"])], label=float(p["label"]))
        for p in train_rows
    ]
    loader = DataLoader(examples, shuffle=True, batch_size=batch_size)
    try:
        model.fit(
            train_dataloader=loader,
            epochs=epochs,
            optimizer_params={"lr": learning_rate},
            show_progress_bar=True,
        )
    except TypeError:
        model.fit(
            examples,
            epochs=epochs,
            batch_size=batch_size,
            show_progress_bar=True,
        )
    _persist_model(model, Path(out_dir))


def fit_cross_encoder(
    *,
    base_model: str,
    pairs: list[dict[str, Any]],
    out_dir: Path,
    epochs: int,
    batch_size: int,
    max_length: int,
    learning_rate: float,
    device: str | None = None,
    model: Any | None = None,
) -> list[dict[str, Any]]:
    """Treina o CE. Retorna as linhas de treino efetivamente usadas."""
    train_rows, _skipped, _fallback = select_train_pairs(pairs)
    if not train_rows:
        raise ValueError("Nenhum par de treino após filtrar holdout/eval.")

    if model is None:
        from sentence_transformers import CrossEncoder

        kwargs: dict[str, Any] = {"max_length": max_length}
        resolved = resolve_device(device)
        if resolved:
            kwargs["device"] = resolved
        model = CrossEncoder(base_model, **kwargs)

    try:
        _fit_with_trainer(
            model=model,
            train_rows=train_rows,
            out_dir=out_dir,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
        )
    except ImportError as exc:
        logger.info("CrossEncoderTrainer indisponível (%s); usando fit() legado", exc)
        _fit_with_legacy(
            model=model,
            train_rows=train_rows,
            out_dir=out_dir,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
        )
    return train_rows


def train_reranker(
    *,
    pairs_path: Path,
    out_dir: Path = DEFAULT_RERANKER_DIR,
    base_model: str = DEFAULT_RERANKER_MODEL,
    epochs: int = 1,
    batch_size: int = 16,
    max_length: int = 256,
    learning_rate: float = 2e-5,
    dry_run: bool = False,
    device: str | None = None,
) -> tuple[int, dict[str, Any]]:
    """CLI-friendly: 0 ok, 1 falha de treino, 2 pares ausentes."""
    pairs_path = Path(pairs_path)
    out_dir = Path(out_dir)
    pairs = load_pairs(pairs_path)
    train_rows, skipped, used_fallback = select_train_pairs(pairs)
    device = resolve_device(device)
    meta = build_train_meta(
        base_model=base_model,
        pairs_path=pairs_path,
        out_dir=out_dir,
        pairs=pairs,
        train_rows=train_rows,
        skipped=skipped,
        used_eval_fallback=used_fallback,
        epochs=epochs,
        batch_size=batch_size,
        max_length=max_length,
        learning_rate=learning_rate,
        dry_run=dry_run,
        device=device,
    )

    if dry_run:
        meta["skipped"] = "dry-run: CrossEncoder.fit não executado"
        dest = write_train_meta(out_dir, meta)
        meta["train_meta"] = str(dest)
        return 0, meta

    if not pairs:
        meta["error"] = f"Nenhum par em {pairs_path}. Gere com scripts/build_rerank_pairs.py"
        write_train_meta(out_dir, meta)
        return 2, meta

    if not train_rows:
        meta["error"] = "Nenhum par de treino (todos holdout/eval) e fallback vazio"
        write_train_meta(out_dir, meta)
        return 2, meta

    try:
        fit_cross_encoder(
            base_model=base_model,
            pairs=pairs,
            out_dir=out_dir,
            epochs=epochs,
            batch_size=batch_size,
            max_length=max_length,
            learning_rate=learning_rate,
            device=device,
        )
    except Exception as exc:  # noqa: BLE001
        meta["error"] = str(exc)
        write_train_meta(out_dir, meta)
        return 1, meta

    meta["trained"] = True
    dest = write_train_meta(out_dir, meta)
    meta["train_meta"] = str(dest)
    return 0, meta
