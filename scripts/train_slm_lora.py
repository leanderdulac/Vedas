#!/usr/bin/env python3
"""Continued-pretraining de um SLM causal com LoRA sobre o corpus védico.

Base recomendada: Qwen/Qwen2.5-0.5B (vocab cobre Devanāgarī; 1 GB fp16).
No Apple Silicon, use VEDIC_DEVICE=mps (torch>=2.2; bf16 automático). O
script usa o tokenizador do modelo-base (BPE próprio do projeto serve para
experimentos from-scratch, não para fine-tune).

Filtro de qualidade: documentos com OCR ruidoso (ex.: scans do Mārkaṇḍeya)
são excluídos por padrão — `build_texts(..., exclude_noise=True)`.

Exemplos:
  HF_HUB_OFFLINE=0 VEDIC_DEVICE=mps python scripts/train_slm_lora.py \
    --base-model Qwen/Qwen2.5-0.5B --max-steps 500 --out artifacts/slm-qwen05

  python scripts/train_slm_lora.py --dry-run   # dataset stats sem download
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vedic_pipeline.common.corpus import iter_corpus_records, utc_now_iso  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_slm_lora")

DEFAULT_BASE = "Qwen/Qwen2.5-0.5B"
_NOISE_MARKERS = ("ocr", "scan ocr")


def filter_records(records: list[dict[str, Any]], *, exclude_noise: bool = True) -> list[dict[str, Any]]:
    """Remove docs inadequados para pré-treino (OCR ruidoso etc.)."""
    out: list[dict[str, Any]] = []
    for rec in records:
        title = (rec.get("title") or "").lower()
        url = (rec.get("source_url") or "").lower()
        if exclude_noise:
            if any(m in title for m in _NOISE_MARKERS) or "markandeya" in title and "ocr" in title:
                continue
            if "archive.org" in url and "ocr" in title:
                continue
        text = (rec.get("text") or "").strip()
        if len(text) >= 40:
            out.append({"text": text, "title": rec.get("title"), "language": rec.get("language")})
    return out


def build_texts(corpus_path: Path, *, exclude_noise: bool = True, max_chars: int | None = None) -> list[str]:
    records = [r for r in iter_corpus_records(corpus_path)]
    kept = filter_records(records, exclude_noise=exclude_noise)
    texts = [r["text"] for r in kept]
    if max_chars:
        total = 0
        capped: list[str] = []
        for t in texts:
            if total + len(t) > max_chars:
                break
            capped.append(t)
            total += len(t)
        texts = capped
    return texts


def dataset_stats(texts: list[str]) -> dict[str, Any]:
    sa = 0
    en = 0
    chars = 0
    for t in texts:
        chars += len(t)
        if any("\u0900" <= ch <= "\u097F" for ch in t):
            sa += 1
        else:
            en += 1
    return {"docs": len(texts), "chars": chars, "sanskritish_docs": sa, "latin_docs": en}


def build_dataset(texts: list[str], tokenizer: Any, block_size: int):
    """Tokeniza e agrupa em blocos causais (imports pesados ficam aqui)."""
    from datasets import Dataset

    raw = Dataset.from_dict({"text": texts})

    def tokenize_fn(examples: dict[str, list[str]]) -> dict[str, Any]:
        return tokenizer(examples["text"], truncation=False, add_special_tokens=True)

    tokenized = raw.map(tokenize_fn, batched=True, remove_columns=["text"], desc="Tokenizando")

    def group(examples: dict[str, list]) -> dict[str, list]:
        concat = {k: sum(examples[k], []) for k in examples}
        total = (len(concat["input_ids"]) // block_size) * block_size
        sliced = {k: [v[i : i + block_size] for i in range(0, total, block_size)] for k, v in concat.items()}
        return sliced | {"labels": [list(b) for b in sliced["input_ids"]]}

    lm = tokenized.map(group, batched=True, desc="Agrupando blocos")
    return lm


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default=str(ROOT / "data" / "corpus.jsonl"))
    parser.add_argument("--base-model", default=DEFAULT_BASE)
    parser.add_argument("--out", default=str(ROOT / "artifacts" / "slm"))
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--block-size", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--keep-noise", action="store_true", help="Inclui docs de OCR ruidoso")
    parser.add_argument("--max-chars", type=int, default=None, help="Teto de chars do treino (teste rápido)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    from vedic_pipeline.train.device import detect_device

    device = detect_device()

    texts = build_texts(Path(args.corpus), exclude_noise=not args.keep_noise, max_chars=args.max_chars)
    stats = dataset_stats(texts)
    logger.info("dataset: %s", stats)
    if not texts:
        logger.error("Nenhum texto utilizável no corpus.")
        return 1

    if args.dry_run:
        print(
            json.dumps(
                {
                    "base_model": args.base_model,
                    "device": device,
                    "dataset": stats,
                    "lora": {"r": args.lora_r, "alpha": args.lora_alpha},
                    "block_size": args.block_size,
                    "dry_run": True,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, trust_remote_code=False, torch_dtype=torch.bfloat16 if device == "mps" else None
    )

    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules="all-linear",
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    lm_ds = build_dataset(texts, tokenizer, args.block_size)
    if len(lm_ds) == 0:
        logger.error("Corpus insuficiente para block_size=%s", args.block_size)
        return 1

    out = Path(args.out)
    training_args = TrainingArguments(
        output_dir=str(out / "checkpoints"),
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=0.01,
        logging_steps=10,
        save_steps=250,
        save_total_limit=2,
        prediction_loss_only=True,
        bf16=(device == "mps"),
        fp16=(device == "cuda"),
        report_to=[],
        max_steps=args.max_steps if args.max_steps is not None else -1,
        dataloader_drop_last=False,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_ds,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )
    trainer.train()
    model.save_pretrained(str(out))
    tokenizer.save_pretrained(str(out))

    meta = {
        "base_model": args.base_model,
        "corpus": str(args.corpus),
        "epochs": args.epochs,
        "block_size": args.block_size,
        "max_steps": args.max_steps,
        "lora": {"r": args.lora_r, "alpha": args.lora_alpha},
        "dataset": stats,
        "device": device,
        "trained_at": utc_now_iso(),
    }
    (out / "slm_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("SLM (adapter LoRA) salvo em %s — use VEDIC_LOCAL_LM=%s", out, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
