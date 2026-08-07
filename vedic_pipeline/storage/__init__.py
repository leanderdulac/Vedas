"""Persistência: catálogo PostgreSQL + vetores pgvector."""

from .db import check_db, get_connection, get_database_url, init_schema
from .catalog import sync_corpus_to_db, count_documents
from .vectors import search_pgvector, upsert_chunk_embeddings

__all__ = [
    "check_db",
    "count_documents",
    "get_connection",
    "get_database_url",
    "init_schema",
    "search_pgvector",
    "sync_corpus_to_db",
    "upsert_chunk_embeddings",
]
