"""Treinamento contínuo de modelo causal."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

from vedic_pipeline.common.constants import DEFAULT_BASE_MODEL, DEFAULT_MODEL_DIR
from vedic_pipeline.common.corpus import iter_corpus_texts, utc_now_iso

logger = logging.getLogger("vedic_pipeline.train.model")


def train_causal_model(
    corpus_path: Path,
    out_dir: Path = DEFAULT_MODEL_DIR,
    base_model: str = DEFAULT_BASE_MODEL,
    tokenizer_dir: Optional[Path] = None,
    epochs: int = 1,
    block_size: int = 512,
    batch_size: int = 2,
    learning_rate: float = 5e-5,
    max_steps: Optional[int] = None,
    fp16: bool = False,
) -> Path:
    """
    Fine-tune causal. Usa o tokenizador do modelo-base para evitar
    incompatibilidade embeddings↔vocabulário. O BPE separado serve para
    treino do zero ou adaptação arquitetural posterior.
    """
    import torch
    from datasets import Dataset
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
    )

    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus não encontrado: {corpus_path}")

    texts = list(iter_corpus_texts(corpus_path))
    if not texts:
        raise ValueError("Corpus vazio — execute ingest antes de treinar.")

    logger.info("Carregando base_model=%s (%d documentos)", base_model, len(texts))

    try:
        tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=False)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

    if tokenizer_dir and Path(tokenizer_dir).exists():
        logger.info(
            "Tokenizer BPE em %s NÃO é aplicado no fine-tune do base_model "
            "(evita mismatch de embeddings). Use-o para treino do zero.",
            tokenizer_dir,
        )

    try:
        model = AutoModelForCausalLM.from_pretrained(base_model)
    except Exception as exc:
        logger.warning(
            "AutoModelForCausalLM falhou para %s (%s). "
            "Prefira um decoder causal (gpt2, distilgpt2, etc.).",
            base_model,
            exc,
        )
        raise

    if len(tokenizer) != model.get_input_embeddings().weight.shape[0]:
        logger.info("Redimensionando embeddings ao vocab do tokenizer.")
        model.resize_token_embeddings(len(tokenizer))

    raw_ds = Dataset.from_dict({"text": texts})

    def tokenize_fn(examples: dict[str, list[str]]) -> dict[str, Any]:
        return tokenizer(examples["text"], truncation=False, add_special_tokens=True)

    tokenized = raw_ds.map(
        tokenize_fn,
        batched=True,
        remove_columns=["text"],
        desc="Tokenizando corpus",
    )

    def group_texts(examples: dict[str, list]) -> dict[str, list]:
        concatenated = {k: sum(examples[k], []) for k in examples.keys()}
        total_len = len(concatenated["input_ids"])
        total_len = (total_len // block_size) * block_size
        result = {
            k: [t[i : i + block_size] for i in range(0, total_len, block_size)]
            for k, t in concatenated.items()
        }
        result["labels"] = [list(x) for x in result["input_ids"]]
        return result

    lm_ds = tokenized.map(group_texts, batched=True, desc="Agrupando em blocos")
    if len(lm_ds) == 0:
        raise ValueError(
            f"Corpus insuficiente para block_size={block_size}. "
            "Adicione mais texto ou reduza --block-size."
        )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    use_cuda = torch.cuda.is_available()
    args = TrainingArguments(
        output_dir=str(out_dir / "checkpoints"),
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=0.01,
        logging_steps=10,
        save_steps=500,
        save_total_limit=2,
        prediction_loss_only=True,
        fp16=fp16 and use_cuda,
        report_to=[],
        max_steps=max_steps if max_steps is not None else -1,
        dataloader_drop_last=False,
        remove_unused_columns=False,
    )

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=lm_ds,
        data_collator=collator,
    )

    logger.info(
        "Iniciando treino: epochs=%s blocks=%d cuda=%s",
        epochs,
        len(lm_ds),
        use_cuda,
    )
    trainer.train()
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    meta = {
        "base_model": base_model,
        "corpus": str(corpus_path),
        "epochs": epochs,
        "block_size": block_size,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "documents": len(texts),
        "lm_blocks": len(lm_ds),
        "trained_at": utc_now_iso(),
        "device": "cuda" if use_cuda else "cpu",
    }
    (out_dir / "train_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("Modelo salvo em %s", out_dir)
    return out_dir
