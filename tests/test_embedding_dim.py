"""Dimensão de embedding configurável (pgvector vector(N))."""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from vedic_pipeline.storage.db import build_schema_sql, get_embedding_dim


class EmbeddingDimTests(unittest.TestCase):
    def test_default_is_384(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("VEDIC_EMBEDDING_DIM", None)
            self.assertEqual(get_embedding_dim(), 384)

    def test_env_override(self):
        with patch.dict(os.environ, {"VEDIC_EMBEDDING_DIM": "768"}):
            self.assertEqual(get_embedding_dim(), 768)

    def test_invalid_falls_back(self):
        for bad in ("abc", "12", "99999", "", "0"):
            with patch.dict(os.environ, {"VEDIC_EMBEDDING_DIM": bad}):
                self.assertEqual(get_embedding_dim(), 384)

    def test_build_schema_sql_uses_dim(self):
        sql = build_schema_sql(768)
        self.assertIn("vector(768)", sql)
        self.assertNotIn("vector(384)", sql)
        sql384 = build_schema_sql(384)
        self.assertIn("vector(384)", sql384)

    def test_build_schema_sql_rejects_bad_dim(self):
        for bad in (12, 0, -1, 4096, "768", None, True):
            with self.assertRaises(ValueError, msg=f"dim={bad!r}"):
                build_schema_sql(bad)  # type: ignore[arg-type]

    def test_build_pgvector_index_rejects_mismatch(self):
        import tempfile
        from pathlib import Path

        import numpy as np

        from vedic_pipeline.storage import vectors

        with tempfile.TemporaryDirectory() as tmp:
            corpus = Path(tmp) / "corpus.jsonl"
            corpus.write_text(
                '{"id": "d1", "title": "T", "text": "Om namo bhagavate vasudevaya Om shanti"}\n',
                encoding="utf-8",
            )

            class DummyModel:
                def encode(self, texts, **kwargs):
                    return np.ones((len(texts), 384), dtype=np.float32)

            with (
                patch.object(vectors, "init_schema"),
                patch(
                    "vedic_pipeline.search.embeddings._load_st_model",
                    return_value=DummyModel(),
                ),
                self.assertRaises(ValueError),
            ):
                vectors.build_pgvector_index(
                    corpus_path=corpus, model_name="dummy", embedding_dim=768
                )


if __name__ == "__main__":
    unittest.main()
