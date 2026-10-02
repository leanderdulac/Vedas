"""Híbrido honesto: pgvector sem sidecar numpy (M2)."""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from vedic_pipeline.llm.ask import retrieve_hits


class HonestHybridPgvectorTests(unittest.TestCase):
    def test_pgvector_loads_chunks_from_postgres_when_numpy_missing(self):
        pg_chunks = [
            {
                "chunk_id": "rv129",
                "doc_id": "rv",
                "title": "Rigveda RV 10.129 Nasadiya (Griffith)",
                "locator": "RV 10.129",
                "text": "Then was not non-existent nor existent.",
                "tradition": "vedic",
                "language": "en",
            },
        ]
        semantic = [
            {
                "chunk_id": "other",
                "doc_id": "x",
                "title": "Unrelated commentary",
                "text": "something else entirely",
                "score": 0.99,
            }
        ]
        missing = Path("/tmp/vedic-missing-numpy-index-m2")
        with (
            patch("vedic_pipeline.storage.vectors.search_pgvector", return_value=semantic),
            patch("vedic_pipeline.storage.vectors.list_chunks_for_hybrid", return_value=pg_chunks) as listed,
            patch("vedic_pipeline.llm.ask.get_index", side_effect=FileNotFoundError("no numpy")),
        ):
            hits, used = retrieve_hits(
                "Nasadiya hymn 10.129",
                backend="pgvector",
                hybrid=True,
                index_dir=missing,
            )
        listed.assert_called()
        self.assertEqual(used, "pgvector+hybrid")
        self.assertTrue(
            any(h.get("chunk_id") == "rv129" or "10.129" in (h.get("title") or "") for h in hits),
            hits,
        )

    def test_pgvector_reports_plain_backend_when_no_chunk_corpus(self):
        semantic = [
            {
                "chunk_id": "a",
                "doc_id": "x",
                "title": "Other",
                "text": "hi",
                "score": 0.9,
            }
        ]
        missing = Path("/tmp/vedic-missing-numpy-index-m2-empty")
        with (
            patch("vedic_pipeline.storage.vectors.search_pgvector", return_value=semantic),
            patch("vedic_pipeline.storage.vectors.list_chunks_for_hybrid", return_value=[]),
            patch("vedic_pipeline.llm.ask.get_index", side_effect=FileNotFoundError("no numpy")),
        ):
            hits, used = retrieve_hits(
                "Nasadiya",
                backend="pgvector",
                hybrid=True,
                index_dir=missing,
            )
        self.assertEqual(used, "pgvector")
        self.assertNotIn("+hybrid", used)
        self.assertEqual([h["chunk_id"] for h in hits], ["a"])
