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
    fetch_url_bytes,
    is_safe_local_path,
    is_safe_url,
    prepare_pinned_request,
    resolve_public_targets,
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

    def test_mixed_public_and_loopback_dns_is_rejected(self):
        import socket

        def mixed(_host, _port, *args, **kwargs):
            return [
                (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 0)),
                (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 0)),
            ]

        with patch("socket.getaddrinfo", mixed):
            ok, reason, ips = resolve_public_targets("http://evil.example/file")
        self.assertFalse(ok)
        self.assertIn("restrito", reason)
        self.assertEqual(ips, [])

    def test_prepare_pinned_request_uses_validated_ip(self):
        import socket

        def public(_host, _port, *args, **kwargs):
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 0))]

        with patch("socket.getaddrinfo", public):
            pinned, host, headers, extensions = prepare_pinned_request(
                "https://evil.example/path"
            )
        self.assertEqual(host, "evil.example")
        self.assertIn("93.184.216.34", pinned)
        self.assertNotIn("evil.example", pinned.split("/")[2])
        self.assertEqual(headers.get("Host"), "evil.example")
        self.assertEqual(extensions.get("sni_hostname"), "evil.example")

    def test_dns_rebinding_pins_first_ip_and_rejects_later_loopback(self):
        import socket

        calls = {"n": 0}

        def rebinding(_host, _port, *args, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 0))]
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 0))]

        class FakeStream:
            def __init__(self, status=200, content=b"hello-world", headers=None, location=None):
                self.status_code = status
                self.headers = dict(headers or {"content-type": "text/plain"})
                if location:
                    self.headers["location"] = location
                self._content = content

            def raise_for_status(self):
                return None

            def iter_bytes(self, chunk_size=65536):
                yield self._content

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        class FakeClient:
            urls: list[str] = []
            headers: list[dict] = []
            used_stream = False
            used_get = False
            responses: list[FakeStream] = []

            def __init__(self, *args, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def stream(self, method, url, headers=None, extensions=None):
                FakeClient.used_stream = True
                FakeClient.urls.append(url)
                FakeClient.headers.append(dict(headers or {}))
                if FakeClient.responses:
                    return FakeClient.responses.pop(0)
                return FakeStream()

            def get(self, *args, **kwargs):
                FakeClient.used_get = True
                raise AssertionError("client.get não deve ser usado (M6: stream)")

        FakeClient.urls = []
        FakeClient.headers = []
        FakeClient.used_stream = False
        FakeClient.used_get = False
        FakeClient.responses = []

        with patch("socket.getaddrinfo", rebinding), patch("httpx.Client", FakeClient):
            data, _ct, _final = fetch_url_bytes("http://evil.example/file")
        self.assertEqual(data, b"hello-world")
        self.assertTrue(FakeClient.used_stream)
        self.assertFalse(FakeClient.used_get)
        self.assertEqual(len(FakeClient.urls), 1)
        self.assertIn("93.184.216.34", FakeClient.urls[0])
        self.assertNotIn("127.0.0.1", FakeClient.urls[0])
        self.assertEqual(FakeClient.headers[0].get("Host"), "evil.example")

        FakeClient.urls = []
        FakeClient.headers = []
        FakeClient.responses = [
            FakeStream(status=302, location="http://evil.example/next"),
        ]
        calls["n"] = 0
        with (
            patch("socket.getaddrinfo", rebinding),
            patch("httpx.Client", FakeClient),
            self.assertRaises(ValueError) as ctx,
        ):
            fetch_url_bytes("http://evil.example/file")
        self.assertIn("SSRF", str(ctx.exception))
        self.assertIn("restrito", str(ctx.exception).lower() + str(ctx.exception))
        self.assertEqual(len(FakeClient.urls), 1)
        self.assertIn("93.184.216.34", FakeClient.urls[0])

    def test_stream_enforces_size_cap_while_reading(self):
        import socket

        def public(_host, _port, *args, **kwargs):
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 0))]

        class FakeStream:
            status_code = 200
            headers = {"content-type": "text/plain"}

            def raise_for_status(self):
                return None

            def iter_bytes(self, chunk_size=65536):
                yield b"x" * 80
                yield b"y" * 80

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        class FakeClient:
            def __init__(self, *args, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def stream(self, method, url, headers=None, extensions=None):
                return FakeStream()

        with (
            patch("socket.getaddrinfo", public),
            patch("httpx.Client", FakeClient),
            self.assertRaises(ValueError) as ctx,
        ):
            fetch_url_bytes("http://evil.example/big", limit=100)
        self.assertIn("excede limite", str(ctx.exception))

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
