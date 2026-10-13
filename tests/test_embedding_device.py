"""O encode de embeddings segue VEDIC_DEVICE."""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from vedic_pipeline.search import embeddings


class EmbeddingDeviceTests(unittest.TestCase):
    def tearDown(self):
        embeddings._MODEL_CACHE.clear()

    def test_explicit_cpu(self):
        try:
            with patch.dict(os.environ, {"VEDIC_DEVICE": "cpu"}):
                self.assertEqual(embeddings.embedding_device(), "cpu")
        except ImportError:
            self.skipTest("torch ausente")

    def test_loader_receives_the_resolved_device(self):
        embeddings._MODEL_CACHE.clear()
        with patch.object(embeddings, "embedding_device", return_value="mps"), patch(
            "sentence_transformers.SentenceTransformer", return_value=object()
        ) as factory:
            first = embeddings._load_st_model("dummy/model")
            second = embeddings._load_st_model("dummy/model")
        factory.assert_called_once_with("dummy/model", device="mps")
        self.assertIs(first, second)


if __name__ == "__main__":
    unittest.main()
