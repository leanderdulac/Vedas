"""Schemas Pydantic da API (nível de módulo — necessário para OpenAPI)."""

from typing import Optional

from pydantic import BaseModel, Field

from vedic_pipeline.common.constants import (
    DEFAULT_BASE_MODEL,
    DEFAULT_CORPUS,
    DEFAULT_EMBED_DIR,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_MODEL_DIR,
    DEFAULT_TOKENIZER_DIR,
)


class IngestRequest(BaseModel):
    manifest: str = Field(..., description="Caminho do manifesto JSON")
    corpus: str = Field(default=str(DEFAULT_CORPUS))
    min_chars: int = Field(default=80, ge=1)
    sync_db: bool = False


class TokenizeRequest(BaseModel):
    corpus: str = Field(default=str(DEFAULT_CORPUS))
    out_dir: str = Field(default=str(DEFAULT_TOKENIZER_DIR))
    vocab_size: int = Field(default=32000, ge=1000, le=256000)
    min_frequency: int = Field(default=2, ge=1)


class TrainRequest(BaseModel):
    corpus: str = Field(default=str(DEFAULT_CORPUS))
    tokenizer_dir: str = Field(default=str(DEFAULT_TOKENIZER_DIR))
    out_dir: str = Field(default=str(DEFAULT_MODEL_DIR))
    base_model: str = Field(default=DEFAULT_BASE_MODEL)
    epochs: int = Field(default=1, ge=1, le=100)
    block_size: int = Field(default=512, ge=64, le=4096)
    batch_size: int = Field(default=2, ge=1, le=64)
    learning_rate: float = Field(default=5e-5, gt=0)
    max_steps: Optional[int] = None
    fp16: bool = False


class BuildIndexRequest(BaseModel):
    corpus: str = Field(default=str(DEFAULT_CORPUS))
    out_dir: str = Field(default=str(DEFAULT_EMBED_DIR))
    model_name: str = Field(default=DEFAULT_EMBEDDING_MODEL)
    chunk_size: int = Field(default=800, ge=100, le=4000)
    overlap: int = Field(default=120, ge=0, le=1000)
    backend: str = Field(default="numpy")


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    index_dir: str = Field(default=str(DEFAULT_EMBED_DIR))
    top_k: int = Field(default=5, ge=1, le=50)
    tradition: Optional[str] = None
    language: Optional[str] = None
    backend: str = Field(default="auto")
    include_prompt: bool = False


class AskRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4000)
    index_dir: str = Field(default=str(DEFAULT_EMBED_DIR))
    top_k: int = Field(default=10, ge=1, le=30)
    tradition: Optional[str] = None
    language: Optional[str] = None
    backend: str = Field(default="auto")
    provider: str = Field(default="auto")
    model: Optional[str] = None
    include_hits: bool = True
    include_prompt: bool = False
    hybrid: bool = Field(
        default=True,
        description="Fusão semântica + lexical com expansão de consulta",
    )
