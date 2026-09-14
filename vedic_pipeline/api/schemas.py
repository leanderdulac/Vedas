"""Schemas Pydantic da API (nível de módulo — necessário para OpenAPI)."""

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from vedic_pipeline.api.request_policy import (
    api_index_dir,
    validate_index_dir,
    validate_model_name,
    validate_project_path,
)
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

    @field_validator("manifest", "corpus")
    @classmethod
    def _confine(cls, v: str) -> str:
        return validate_project_path(v, field="path")


class TokenizeRequest(BaseModel):
    corpus: str = Field(default=str(DEFAULT_CORPUS))
    out_dir: str = Field(default=str(DEFAULT_TOKENIZER_DIR))
    vocab_size: int = Field(default=32000, ge=1000, le=256000)
    min_frequency: int = Field(default=2, ge=1)

    @field_validator("corpus", "out_dir")
    @classmethod
    def _confine(cls, v: str) -> str:
        return validate_project_path(v, field="path")


class TrainRequest(BaseModel):
    corpus: str = Field(default=str(DEFAULT_CORPUS))
    tokenizer_dir: str = Field(default=str(DEFAULT_TOKENIZER_DIR))
    out_dir: str = Field(default=str(DEFAULT_MODEL_DIR))
    base_model: str = Field(default=DEFAULT_BASE_MODEL)
    epochs: int = Field(default=1, ge=1, le=100)
    block_size: int = Field(default=512, ge=64, le=4096)
    batch_size: int = Field(default=2, ge=1, le=64)
    learning_rate: float = Field(default=5e-5, gt=0)
    max_steps: int | None = None
    fp16: bool = False

    @field_validator("corpus", "tokenizer_dir", "out_dir")
    @classmethod
    def _confine(cls, v: str) -> str:
        return validate_project_path(v, field="path")

    @field_validator("base_model")
    @classmethod
    def _model(cls, v: str) -> str:
        return validate_model_name(v, field="base_model")

    @field_validator("max_steps")
    @classmethod
    def _max_steps(cls, v: int | None) -> int | None:
        if v is not None and not (1 <= v <= 100000):
            raise ValueError("max_steps deve estar entre 1 e 100000")
        return v


class BuildIndexRequest(BaseModel):
    corpus: str = Field(default=str(DEFAULT_CORPUS))
    out_dir: str = Field(default=str(DEFAULT_EMBED_DIR))
    model_name: str = Field(default=DEFAULT_EMBEDDING_MODEL)
    chunk_size: int = Field(default=800, ge=100, le=4000)
    overlap: int = Field(default=120, ge=0, le=1000)
    backend: Literal["numpy", "pgvector", "both"] = "numpy"
    embedding_dim: int | None = Field(
        default=None, ge=64, le=3072,
        description="Dimensão vector(N) pgvector (default: VEDIC_EMBEDDING_DIM ou dim do modelo)",
    )

    @model_validator(mode="after")
    def validate_overlap(self):
        if self.overlap >= self.chunk_size:
            raise ValueError("overlap deve ser menor que chunk_size")
        return self

    @field_validator("corpus", "out_dir")
    @classmethod
    def _confine(cls, v: str) -> str:
        return validate_project_path(v, field="path")

    @field_validator("model_name")
    @classmethod
    def _model(cls, v: str) -> str:
        return validate_model_name(v, field="model_name")


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4000)
    index_dir: str = Field(default_factory=api_index_dir, validate_default=True)

    @field_validator("index_dir")
    @classmethod
    def configured_index(cls, value: str) -> str:
        return validate_index_dir(value)

    @field_validator("query")
    @classmethod
    def nonempty_query(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("query não pode conter apenas espaços")
        return value


class SearchRequest(QueryRequest):
    top_k: int = Field(default=5, ge=1, le=50)
    tradition: str | None = None
    language: str | None = None
    backend: Literal["auto", "numpy", "pgvector"] = "auto"
    include_prompt: bool = False


class ExplainRequest(BaseModel):
    lang: Literal["pt", "en"] = "pt"
    provider: Literal["auto", "xai", "local", "extractive"] = "auto"
    model: str | None = None


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(..., min_length=1, max_length=6000)


class AskRequest(QueryRequest):
    top_k: int = Field(default=10, ge=1, le=30)
    tradition: str | None = None
    language: str | None = None
    backend: Literal["auto", "numpy", "pgvector"] = "auto"
    provider: Literal["auto", "xai", "local", "extractive"] = "auto"
    model: str | None = None
    include_hits: bool = True
    include_prompt: bool = False
    hybrid: bool = Field(
        default=True,
        description="Fusão semântica + lexical com expansão de consulta",
    )
    history: list[ChatTurn] = Field(
        default_factory=list,
        max_length=12,
        description="Turnos anteriores da conversa (sem a pergunta atual)",
    )
