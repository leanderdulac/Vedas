"""Treinamento de tokenizador BPE (Hugging Face tokenizers)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterator, Optional

from vedic_pipeline.common.constants import (
    DEFAULT_TOKENIZER_DIR,
    SANSKRIT_SPECIAL_TOKENS,
)
from vedic_pipeline.common.corpus import iter_corpus_texts, utc_now_iso

logger = logging.getLogger("vedic_pipeline.train.tokenizer")


def train_bpe_tokenizer(
    corpus_path: Path,
    out_dir: Path = DEFAULT_TOKENIZER_DIR,
    vocab_size: int = 32_000,
    min_frequency: int = 2,
    special_tokens: Optional[list[str]] = None,
) -> Path:
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers
    from tokenizers.normalizers import NFC, Sequence as NormSequence

    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus não encontrado: {corpus_path}")

    specials = list(special_tokens or SANSKRIT_SPECIAL_TOKENS)
    for t in ("<pad>", "<unk>", "<s>", "</s>"):
        if t not in specials:
            specials.insert(0, t)

    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.normalizer = NormSequence([NFC()])
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=specials,
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )

    def batch_iterator(batch_size: int = 1_000) -> Iterator[list[str]]:
        batch: list[str] = []
        for text in iter_corpus_texts(corpus_path):
            batch.append(text)
            if len(batch) >= batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

    logger.info("Treinando BPE vocab_size=%d a partir de %s", vocab_size, corpus_path)
    tokenizer.train_from_iterator(batch_iterator(), trainer=trainer)

    bos_id = tokenizer.token_to_id("<s>")
    eos_id = tokenizer.token_to_id("</s>")
    tokenizer.post_processor = processors.TemplateProcessing(
        single="<s> $A </s>",
        pair="<s> $A </s> $B:1 </s>:1",
        special_tokens=[
            ("<s>", bos_id if bos_id is not None else 0),
            ("</s>", eos_id if eos_id is not None else 1),
        ],
    )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(out_dir / "tokenizer.json"))

    try:
        from transformers import PreTrainedTokenizerFast

        wrap = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer,
            bos_token="<s>",
            eos_token="</s>",
            unk_token="<unk>",
            pad_token="<pad>",
            mask_token="<mask>",
            model_max_length=2048,
        )
        wrap.save_pretrained(str(out_dir))
        logger.info("Tokenizer HF salvo em %s", out_dir)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Salvo apenas tokenizer.json: %s", exc)

    meta = {
        "vocab_size": vocab_size,
        "min_frequency": min_frequency,
        "special_tokens": specials,
        "corpus": str(corpus_path),
        "trained_at": utc_now_iso(),
        "algorithm": "BPE+ByteLevel+NFC",
        "notes": (
            "ByteLevel cobre Devanāgarī, IAST e acentos védicos. "
            "Compare taxa de compressão em 32k–64k."
        ),
    }
    (out_dir / "tokenizer_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return out_dir


def evaluate_tokenizer_compression(
    tokenizer_dir: Path,
    corpus_path: Path,
    sample_limit: int = 200,
) -> dict:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(tokenizer_dir), use_fast=True)
    n = 0
    total_chars = 0
    total_tokens = 0
    for text in iter_corpus_texts(corpus_path):
        ids = tok.encode(text, add_special_tokens=False)
        total_chars += len(text)
        total_tokens += len(ids)
        n += 1
        if n >= sample_limit:
            break
    if total_tokens == 0 or n == 0:
        return {"samples": 0}
    return {
        "samples": n,
        "total_chars": total_chars,
        "total_tokens": total_tokens,
        "chars_per_token": round(total_chars / total_tokens, 3),
        "avg_tokens_per_sample": round(total_tokens / n, 2),
    }
