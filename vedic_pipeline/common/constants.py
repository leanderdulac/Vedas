"""Constantes compartilhadas entre os serviços."""

from __future__ import annotations

from pathlib import Path

ALLOWED_LICENSES = frozenset(
    {
        "public-domain",
        "cc0",
        "cc-by",
        "cc-by-sa",
        "odbl",
        "authorized",
        "official-api",
        # Conteúdo do portal governamental indiano (Ministério da Cultura): as
        # escrituras estão em domínio público como obras antigas; a transcrição
        # digital é reutilizável com atribuição à fonte (vedicheritage.gov.in),
        # tipicamente para uso educacional. Ver docs/SOURCES_REVIEW.md.
        "gov-ind",
    }
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CORPUS = PROJECT_ROOT / "data/corpus.jsonl"
DEFAULT_TOKENIZER_DIR = PROJECT_ROOT / "artifacts/tokenizer"
DEFAULT_MODEL_DIR = PROJECT_ROOT / "artifacts/model"
DEFAULT_RAW_DIR = PROJECT_ROOT / "data/raw"
DEFAULT_EMBED_DIR = PROJECT_ROOT / "artifacts/embeddings"

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
