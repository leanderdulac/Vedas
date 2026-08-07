#!/usr/bin/env python3
"""
Ponto de entrada da aplicação Veda Knowledge.

CLI:
  python -m vedic_pipeline ingest --manifest fixtures/sources_local.json
  python vedic_knowledge_pipeline.py serve

API + UI:
  uvicorn vedic_knowledge_pipeline:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import sys

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from vedic_pipeline.api.app import create_app
from vedic_pipeline.cli import main

# Reexportações úteis
from vedic_pipeline.common.constants import (  # noqa: F401
    ALLOWED_LICENSES,
    DEFAULT_BASE_MODEL,
    DEFAULT_CORPUS,
    DEFAULT_EMBED_DIR,
    DEFAULT_MODEL_DIR,
    DEFAULT_RAW_DIR,
    DEFAULT_TOKENIZER_DIR,
    SANSKRIT_SPECIAL_TOKENS,
)
from vedic_pipeline.crawler.ingest import ingest_manifest, load_manifest  # noqa: F401
from vedic_pipeline.crawler.licenses import validate_source  # noqa: F401
from vedic_pipeline.etl.extractors import extract_text  # noqa: F401
from vedic_pipeline.train.model import train_causal_model  # noqa: F401
from vedic_pipeline.train.tokenizer import (  # noqa: F401
    evaluate_tokenizer_compression,
    train_bpe_tokenizer,
)

# Instância ASGI real (não pode ser None — uvicorn precisa do callable)
app = create_app()


if __name__ == "__main__":
    sys.exit(main())
