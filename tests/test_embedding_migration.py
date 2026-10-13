"""Migração de embeddings: prefixos e5 e resolução de modelo no ask."""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from vedic_pipeline.search.embeddings import (
    _passage_texts,
    _query_text,
    build_embedding_index,
    needs_e5_prompts,
    search_index,
)


class E5PromptTests(unittest.TestCase):
    def test_needs_e5_prompts(self):
        self.assertTrue(needs_e5_prompts("intfloat/multilingual-e5-small"))
        self.assertTrue(needs_e5_prompts("intfloat/e5-base-v2"))
        self.assertFalse(needs_e5_prompts("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"))
        self.assertFalse(needs_e5_prompts("BAAI/bge-m3"))
        self.assertFalse(needs_e5_prompts("hkunlp/instructor-xl"))

    def test_prefix_helpers(self):
        self.assertEqual(_passage_texts("intfloat/multilingual-e5-small", ["om"]), ["passage: om"])
        self.assertEqual(_query_text("intfloat/multilingual-e5-small", "agni"), "query: agni")
        self.assertEqual(_passage_texts("x/outro", ["om"]), ["om"])
        self.assertEqual(_query_text("x/outro", "agni"), "agni")


class _RecorderModel:
    """Guarda o último lote codificado para inspeção."""

    def __init__(self, dim: int = 4):
        self.last_batch: list[str] = []
        self.dim = dim

    def encode(self, texts, **kwargs):
        import numpy as np

        self.last_batch = [str(t) for t in texts]
        return np.zeros((len(texts), self.dim), dtype=np.float32)


class E5IntegrationTests(unittest.TestCase):
    def test_search_index_prefixes_query_for_e5(self):
        import tempfile

        model = _RecorderModel()
        with tempfile.TemporaryDirectory() as tmp:
            from vedic_pipeline.common.corpus import utc_now_iso

            idx_dir = Path(tmp)
            (idx_dir / "chunks.jsonl").write_text(
                '{"chunk_id": "c1", "text": "agni", "doc_id": "d1"}\n', encoding="utf-8"
            )
            import json

            (idx_dir / "index_meta.json").write_text(
                json.dumps({"model_name": "intfloat/multilingual-e5-small", "n_chunks": 1, "dim": 4, "built_at": utc_now_iso()}),
                encoding="utf-8",
            )
            import numpy as np

            (idx_dir / "embeddings.npy").write_bytes(b"")
            np.save(idx_dir / "embeddings.npy", np.zeros((1, 4), dtype=np.float32))
            with patch("vedic_pipeline.search.embeddings._load_st_model", return_value=model):
                search_index("agni", {"vectors": np.zeros((1, 4), dtype=np.float32), "chunks": [{"chunk_id": "c1", "text": "agni"}], "meta": {"model_name": "intfloat/multilingual-e5-small"}, "index_dir": idx_dir})
            self.assertEqual(model.last_batch, ["query: agni"])

    def test_build_index_prefixes_passages_for_e5(self):
        import tempfile

        model = _RecorderModel()
        with tempfile.TemporaryDirectory() as tmp:
            corpus = Path(tmp) / "corpus.jsonl"
            corpus.write_text('{"id": "d1", "title": "T", "text": "अग्निमीळे पुरोहितं om"}\n', encoding="utf-8")
            out = Path(tmp) / "idx"
            with patch("vedic_pipeline.search.embeddings._load_st_model", return_value=model):
                build_embedding_index(corpus_path=corpus, out_dir=out, model_name="intfloat/multilingual-e5-small", chunk_size=100, overlap=0)
            self.assertTrue(model.last_batch)
            self.assertTrue(all(b.startswith("passage: ") for b in model.last_batch), model.last_batch)


class AskModelResolutionTests(unittest.TestCase):
    def _patches(self, index_meta: dict):
        return (
            patch("vedic_pipeline.llm.ask.get_database_url", return_value="postgresql://x"),
            patch(
                "vedic_pipeline.storage.db.check_db",
                return_value={"reachable": True, "counts": {"embeddings": 10}},
            ),
            patch("vedic_pipeline.llm.ask.get_index", return_value=index_meta),
            patch("vedic_pipeline.storage.vectors.search_pgvector", return_value=[]),
            patch("vedic_pipeline.llm.ask.hybrid_rerank", return_value=[]),
        )

    def test_retrieve_hits_uses_index_model_for_pgvector(self):
        import tempfile

        from vedic_pipeline.llm.ask import retrieve_hits

        index_meta = {"meta": {"model_name": "intfloat/multilingual-e5-small"}}
        with tempfile.TemporaryDirectory() as tmp:
            patches = self._patches(index_meta)
            with patches[0], patches[1], patches[2] as gi, patches[3] as pg, patches[4]:
                _hits, used = retrieve_hits(
                    "agni", backend="pgvector", top_k=2, hybrid=False, index_dir=Path(tmp)
                )
            self.assertEqual(used, "pgvector")
            self.assertTrue(gi.called)
            self.assertTrue(pg.called)
            self.assertEqual(pg.call_args.kwargs.get("model_name"), "intfloat/multilingual-e5-small")

    def test_missing_index_keeps_default_model(self):
        import tempfile

        from vedic_pipeline.common.constants import DEFAULT_EMBEDDING_MODEL
        from vedic_pipeline.llm.ask import retrieve_hits

        index_meta = {"meta": {"model_name": "intfloat/multilingual-e5-small"}}
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "ausente"
            patches = self._patches(index_meta)
            with patches[0], patches[1], patches[2] as gi, patches[3] as pg, patches[4]:
                retrieve_hits("agni", backend="pgvector", top_k=2, hybrid=False, index_dir=missing)
            self.assertFalse(gi.called)
            self.assertEqual(pg.call_args.kwargs.get("model_name"), DEFAULT_EMBEDDING_MODEL)


class MigrationStatusTests(unittest.TestCase):
    def test_smoke_failure_is_nonzero(self):
        from scripts.migrate_embeddings import migration_status

        self.assertEqual(migration_status({"smoke": {"exit": 1}}), 1)
        self.assertEqual(migration_status({"smoke": {"error": "timeout do smoke"}}), 1)
        self.assertEqual(migration_status({"smoke": {"exit": 0, "retrieval": "retrieval: 3/3"}}), 0)
        self.assertEqual(migration_status({}), 0)


if __name__ == "__main__":
    unittest.main()
