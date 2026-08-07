"""Constantes compartilhadas entre os serviços."""

from __future__ import annotations

from pathlib import Path

ALLOWED_LICENSES = frozenset(
    {
        "public-domain",
        "cc0",
        "cc-by",
        "cc-by-sa",
        "authorized",
        "official-api",
    }
)

DEFAULT_CORPUS = Path("data/corpus.jsonl")
DEFAULT_TOKENIZER_DIR = Path("artifacts/tokenizer")
DEFAULT_MODEL_DIR = Path("artifacts/model")
DEFAULT_RAW_DIR = Path("data/raw")
DEFAULT_EMBED_DIR = Path("artifacts/embeddings")

# Decoder causal pequeno para smoke tests / fine-tune inicial.
# IndicBARTSS é encoder-decoder e não serve em AutoModelForCausalLM.
DEFAULT_BASE_MODEL = "gpt2"

SANSKRIT_SPECIAL_TOKENS = [
    "<pad>",
    "<unk>",
    "<s>",
    "</s>",
    "<mask>",
    "<vedic>",
    "<upanishad>",
    "<itihasa>",
    "<purana>",
    "<vaishnava>",
    "<jyotisha>",
    "<tantra>",
    "<yoga>",
    "<grammar>",
    "<sa>",
    "<en>",
    "<hi>",
    "<bn>",
]

# Chunking para embeddings / RAG
DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 120

# Modelo de embeddings multilingue (bom para en/hi; sa se beneficia de Unicode)
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
