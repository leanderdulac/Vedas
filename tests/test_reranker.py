"""Testes unitários para o módulo de Cross-Encoder Reranker."""

from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from vedic_pipeline.search.reranker import (
    DEFAULT_RERANKER_MODEL,
    get_reranker,
    get_reranker_model_name,
    invalidate_reranker,
    is_reranker_enabled,
    rerank_chunks,
    resolve_reranker_model_source,
)


class RerankerTests(unittest.TestCase):
    def setUp(self):
        invalidate_reranker()

    def tearDown(self):
        invalidate_reranker()

    def test_empty_chunks(self):
        self.assertEqual(rerank_chunks("query", []), [])

    def test_empty_model_env_falls_back_to_default(self):
        with patch.dict(os.environ, {"VEDIC_RERANKER_MODEL": ""}):
            self.assertEqual(get_reranker_model_name(), DEFAULT_RERANKER_MODEL)

    def test_resolve_local_dir_under_project_root(self):
        resolved = resolve_reranker_model_source("fixtures/rerank")
        self.assertTrue(Path(resolved).is_absolute())
        self.assertTrue(Path(resolved).is_dir())
        self.assertEqual(Path(resolved).name, "rerank")

    def test_env_change_reloads_without_explicit_invalidate(self):
        mock_a = MagicMock(name="ce-a")
        mock_b = MagicMock(name="ce-b")
        with (
            patch.dict(
                os.environ,
                {"VEDIC_ENABLE_RERANKER": "true", "VEDIC_RERANKER_MODEL": "model-a"},
            ),
            patch(
                "vedic_pipeline.search.reranker.load_cross_encoder",
                side_effect=[mock_a, mock_b],
            ) as loader,
        ):
            self.assertIs(get_reranker(), mock_a)
            os.environ["VEDIC_RERANKER_MODEL"] = "model-b"
            self.assertIs(get_reranker(), mock_b)
            self.assertEqual(loader.call_count, 2)

    def test_enable_after_disabled_reloads(self):
        mock_ce = MagicMock(name="ce-on")
        with (
            patch.dict(os.environ, {"VEDIC_ENABLE_RERANKER": "false"}, clear=False),
            patch(
                "vedic_pipeline.search.reranker.load_cross_encoder",
                return_value=mock_ce,
            ) as loader,
        ):
            self.assertIsNone(get_reranker())
            loader.assert_not_called()
            os.environ["VEDIC_ENABLE_RERANKER"] = "true"
            os.environ["VEDIC_RERANKER_MODEL"] = "artifacts/reranker"
            self.assertIs(get_reranker(), mock_ce)
            loader.assert_called_once()

    def test_default_is_disabled_when_env_unset(self):
        env = {k: v for k, v in os.environ.items() if k != "VEDIC_ENABLE_RERANKER"}
        with patch.dict(os.environ, env, clear=True):
            self.assertFalse(is_reranker_enabled())
            self.assertIsNone(get_reranker())

    def test_opt_in_values(self):
        for raw, enabled in (
            ("true", True),
            ("1", True),
            ("yes", True),
            ("on", True),
            ("false", False),
            ("0", False),
            ("off", False),
            ("", False),
            ("maybe", False),
        ):
            with self.subTest(raw=raw), patch.dict(os.environ, {"VEDIC_ENABLE_RERANKER": raw}):
                self.assertEqual(is_reranker_enabled(), enabled)

    def test_disabled_reranker_returns_original(self):
        with patch.dict(os.environ, {"VEDIC_ENABLE_RERANKER": "false"}):
            self.assertFalse(is_reranker_enabled())
            chunks = [
                {"chunk_id": "c1", "text": "Primeiro texto", "score": 0.9},
                {"chunk_id": "c2", "text": "Segundo texto", "score": 0.8},
            ]
            result = rerank_chunks("teste", chunks, top_k=2)
            self.assertEqual(len(result), 2)
            self.assertEqual(result[0]["chunk_id"], "c1")
            self.assertIsNone(get_reranker())

    def test_rerank_with_mock_model(self):
        # Mock de CrossEncoder que atribui score maior ao chunk 2
        mock_ce = MagicMock()
        # Logits: c1 ganha -2.0 (sigmoid baixa), c2 ganha 3.0 (sigmoid alta)
        mock_ce.predict.return_value = [-2.0, 3.0]

        with (
            patch.dict(os.environ, {"VEDIC_ENABLE_RERANKER": "true"}),
            patch("vedic_pipeline.search.reranker.get_reranker", return_value=mock_ce),
        ):
            chunks = [
                {"chunk_id": "c1", "text": "Texto neutro", "score": 0.5},
                {"chunk_id": "c2", "text": "Texto muito relevante", "score": 0.5},
            ]
            result = rerank_chunks("ātman", chunks, top_k=2)
            self.assertEqual(len(result), 2)
            # c2 deve ter sido promovido ao primeiro lugar pelo cross encoder
            self.assertEqual(result[0]["chunk_id"], "c2")
            self.assertIn("_cross_score", result[0])
            self.assertGreater(result[0]["score"], result[1]["score"])

    def test_graceful_fallback_on_exception(self):
        mock_ce = MagicMock()
        mock_ce.predict.side_effect = RuntimeError("CUDA OOM or model error")

        with (
            patch.dict(os.environ, {"VEDIC_ENABLE_RERANKER": "true"}),
            patch("vedic_pipeline.search.reranker.get_reranker", return_value=mock_ce),
        ):
            chunks = [
                {"chunk_id": "c1", "text": "Texto 1", "score": 0.9},
                {"chunk_id": "c2", "text": "Texto 2", "score": 0.8},
            ]
            result = rerank_chunks("ātman", chunks, top_k=2)
            # Deve retornar a lista sem quebrar a execução
            self.assertEqual(len(result), 2)
            self.assertEqual(result[0]["chunk_id"], "c1")


if __name__ == "__main__":
    unittest.main()
