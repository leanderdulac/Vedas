"""Persistência: catálogo PostgreSQL + vetores pgvector + objetos S3."""

from .catalog import count_documents, sync_corpus_to_db
from .db import (
    build_schema_sql,
    check_db,
    get_connection,
    get_database_url,
    get_embedding_column_dim,
    get_embedding_dim,
    init_schema,
)
from .objects import (
    get_s3_config,
    is_s3_enabled,
    pull_dir,
    push_dir,
)
from .objects import status as objects_status
from .vectors import search_pgvector, upsert_chunk_embeddings

__all__ = [
    "build_schema_sql",
    "check_db",
    "count_documents",
    "get_connection",
    "get_database_url",
    "get_embedding_column_dim",
    "get_embedding_dim",
    "get_s3_config",
    "init_schema",
    "is_s3_enabled",
    "objects_status",
    "pull_dir",
    "push_dir",
    "search_pgvector",
    "sync_corpus_to_db",
    "upsert_chunk_embeddings",
]
