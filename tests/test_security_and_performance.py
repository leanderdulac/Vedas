import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from vedic_pipeline.api.app import create_app
from vedic_pipeline.api.catalog_service import (
    get_cached_corpus,
    invalidate_corpus_cache,
)
from vedic_pipeline.crawler.download import (
    canonicalize_source_url,
    download_source,
    is_safe_local_path,
    is_safe_url,
)
from vedic_pipeline.search.embeddings import build_embedding_index
from vedic_pipeline.search.rag import get_index, invalidate_index_cache
from vedic_pipeline.storage.db import _sanitize_db_error


class SecurityTests(unittest.TestCase):
    def test_sacred_texts_urls_use_static_archive(self):
        live = "https://www.sacred-texts.com/hin/rigveda/rv01001.htm"
        archived = canonicalize_source_url(live)
        self.assertEqual(archived, "https://archive.sacred-texts.com/hin/rigveda/rv01001.htm")
        self.assertEqual(
            canonicalize_source_url("https://www.gutenberg.org/cache/epub/2388/pg2388.txt"),
            "https://www.gutenberg.org/cache/epub/2388/pg2388.txt",
        )

    def test_ssrf_blocks_loopback_and_invalid_schemes(self):
        # Esquemas inválidos
        for invalid in ["ftp://example.com/file", "gopher://example.com", "file:///etc/passwd"]:
            ok, reason = is_safe_url(invalid)
            self.assertFalse(ok)
            self.assertIn("não suportado", reason)

        # Loopbacks diretos
        for host in ["http://localhost:8080/secret", "http://127.0.0.1/admin", "http://0.0.0.0/test"]:
            ok, reason = is_safe_url(host)
            self.assertFalse(ok)
            self.assertIn("loopback", reason)

        # Endereço link-local (cloud metadata)
        ok, reason = is_safe_url("http://169.254.169.254/latest/meta-data/")
        self.assertFalse(ok)
        self.assertTrue("restrito" in reason or "loopback" in reason)

    def test_lfi_blocks_path_traversal(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            secret_file = base_dir.parent / "secret_outside.txt"
            secret_file.write_text("sensível", encoding="utf-8")
            try:
                # Tentativa de traversal para fora da raiz permitida
                traversal_path = base_dir / ".." / "secret_outside.txt"
                ok, resolved, reason = is_safe_local_path(traversal_path, base_dir=base_dir)
                self.assertFalse(ok)
                self.assertIn("fora do diretório permitido", reason)
            finally:
                if secret_file.exists():
                    secret_file.unlink()

    def test_download_source_rejects_ssrf_and_lfi(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            raw_dir = Path(temp_dir)
            # Rejeita loopback
            with self.assertRaises(ValueError) as ctx:
                download_source({"url": "http://127.0.0.1:9999/test"}, raw_dir=raw_dir)
            self.assertIn("SSRF", str(ctx.exception))

            # Rejeita traversal fora do workspace
            with self.assertRaises(ValueError) as ctx:
                download_source({"url": "file:///etc/passwd"}, raw_dir=raw_dir)
            self.assertIn("segurança", str(ctx.exception))

    def test_sanitize_db_error_masks_credentials(self):
        raw_error = "connection to postgresql://vedas:super_secret_password@db.internal:5432/vedas failed"
        sanitized = _sanitize_db_error(Exception(raw_error))
        self.assertNotIn("super_secret_password", sanitized)
        self.assertIn("******", sanitized)

        conn_str_error = "failed with host=10.0.0.1 user=vedas password=another_secret dbname=vedas"
        sanitized2 = _sanitize_db_error(Exception(conn_str_error))
        self.assertNotIn("another_secret", sanitized2)
        self.assertIn("password=******", sanitized2)

    def test_api_500_errors_are_sanitized(self):
        with patch("vedic_pipeline.llm.ask.retrieve_hits", side_effect=RuntimeError("internal_db_syntax_at_0x1234")), \
             TestClient(create_app()) as client:
            res = client.post("/api/v1/search", json={"query": "atman"})
            self.assertEqual(res.status_code, 500)
            self.assertEqual(res.json()["detail"], "Erro interno ao processar a busca")
            self.assertNotIn("internal_db_syntax_at_0x1234", res.text)


class PerformanceAndCachingTests(unittest.TestCase):
    def setUp(self):
        invalidate_corpus_cache()
        invalidate_index_cache()

    def test_catalog_mtime_cache(self):
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
            f.write(json.dumps({"id": "doc1", "title": "Doc 1", "text": "Texto inicial"}) + "\n")
            temp_corpus = Path(f.name)

        try:
            records1 = get_cached_corpus(temp_corpus)
            self.assertEqual(len(records1), 1)
            self.assertEqual(records1[0]["title"], "Doc 1")

            # Segunda chamada deve reutilizar os mesmos objetos cacheados
            records2 = get_cached_corpus(temp_corpus)
            self.assertIs(records1, records2)

            # Atualização do arquivo no disco invalida o cache
            time.sleep(0.05)
            with temp_corpus.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"id": "doc2", "title": "Doc 2", "text": "Segundo texto"}) + "\n")

            records3 = get_cached_corpus(temp_corpus)
            self.assertEqual(len(records3), 2)
            self.assertIsNot(records1, records3)
        finally:
            if temp_corpus.exists():
                temp_corpus.unlink()

    def test_index_atomic_writes_and_cache_invalidation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            corpus_file = temp_path / "corpus.jsonl"
            corpus_file.write_text(
                json.dumps({"id": "sample1", "title": "Sample", "text": "Om namo bhagavate vasudevaya"}) + "\n",
                encoding="utf-8",
            )
            out_dir = temp_path / "index"

            # Mock do encoder para teste rápido sem baixar modelo pesado
            class DummyModel:
                def encode(self, texts, **kwargs):
                    import numpy as np
                    return np.ones((len(texts), 384), dtype=np.float32)

            with patch("vedic_pipeline.search.embeddings._load_st_model", return_value=DummyModel()):
                meta = build_embedding_index(corpus_path=corpus_file, out_dir=out_dir)
                self.assertIsNotNone(meta)
                self.assertTrue((out_dir / "embeddings.npy").exists())
                self.assertTrue((out_dir / "chunks.jsonl").exists())
                self.assertFalse((out_dir / "embeddings_tmp.npy").exists())
                self.assertFalse((out_dir / "chunks_tmp.jsonl").exists())

                # Carrega o índice e valida cache por mtime
                idx1 = get_index(out_dir)
                idx2 = get_index(out_dir)
                self.assertIs(idx1, idx2)


if __name__ == "__main__":
    unittest.main()
