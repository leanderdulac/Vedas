"""Controles anti-abuso: rate limiting, gate de mídia paga, versão e métricas."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from vedic_pipeline import __version__
from vedic_pipeline.api.app import create_app
from vedic_pipeline.api.metrics import METRICS
from vedic_pipeline.api.rate_limit import allow_request, reset_rate_limits
from vedic_pipeline.api.request_policy import authorize_media


class RateLimitTests(unittest.TestCase):
    def tearDown(self):
        reset_rate_limits()

    def test_read_routes_are_not_limited(self):
        reset_rate_limits()
        for path in ["/api/v1/documents", "/api/v1/health", "/health", "/metrics", "/"]:
            self.assertTrue(allow_request("9.9.9.9", path), path)

    def test_generation_routes_limited(self):
        reset_rate_limits()
        self.assertTrue(allow_request("1.2.3.4", "/api/v1/ask"))
        self.assertTrue(allow_request("1.2.3.4", "/ask/stream"))
        self.assertTrue(allow_request("1.2.3.4", "/api/v1/search"))
        # /api/v1/verses pertence ao grupo "media" (contador separado), não limita "generation".
        self.assertTrue(allow_request("1.2.3.4", "/api/v1/verses/RV.10.129.1"))

    def test_limit_exhausts_and_recovers(self):
        reset_rate_limits()
        with patch.dict(os.environ, {"VEDIC_RATE_LIMIT_PER_MINUTE": "3"}):
            for _ in range(3):
                self.assertTrue(allow_request("1.2.3.4", "/ask"))
            self.assertFalse(allow_request("1.2.3.4", "/ask"))
            # Outro cliente não é afetado
            self.assertTrue(allow_request("9.9.9.9", "/ask"))

    def test_can_be_disabled_by_env(self):
        reset_rate_limits()
        with patch.dict(os.environ, {"VEDIC_RATE_LIMIT_ENABLED": "false"}):
            for _ in range(50):
                self.assertTrue(allow_request("1.2.3.4", "/ask"))

    def test_http_429_when_exhausted(self):
        reset_rate_limits()
        with patch.dict(os.environ, {"VEDIC_RATE_LIMIT_PER_MINUTE": "2"}), \
             patch("vedic_pipeline.llm.ask.retrieve_hits", return_value=([], "numpy")), \
             TestClient(create_app()) as client:
            self.assertEqual(client.post("/api/v1/search", json={"query": "atman"}).status_code, 200)
            self.assertEqual(client.post("/api/v1/search", json={"query": "dharma"}).status_code, 200)
            self.assertEqual(client.post("/api/v1/search", json={"query": "karma"}).status_code, 429)


class MediaAuthTests(unittest.TestCase):
    def test_media_allowed_without_generation_token(self):
        # Em instância local (sem token) a geração de mídia continua liberada.
        with patch.dict(
            os.environ,
            {
                "VEDIC_GENERATION_API_TOKEN": "",
                "XAI_API_KEY": "k",
                "VEDIC_REQUIRE_GENERATION_TOKEN": "",
            },
        ):
            authorize_media(None)
            authorize_media("Bearer anything")

    def test_media_require_token_flag_is_fail_closed(self):
        from fastapi import HTTPException

        with patch.dict(
            os.environ,
            {
                "VEDIC_GENERATION_API_TOKEN": "",
                "XAI_API_KEY": "k",
                "VEDIC_REQUIRE_GENERATION_TOKEN": "true",
            },
        ):
            with self.assertRaises(HTTPException) as ctx:
                authorize_media(None)
            self.assertEqual(ctx.exception.status_code, 503)

    def test_media_requires_token_when_configured(self):
        from fastapi import HTTPException

        with patch.dict(os.environ, {"VEDIC_GENERATION_API_TOKEN": "secret", "XAI_API_KEY": "k"}):
            authorize_media("Bearer secret")
            with self.assertRaises(HTTPException):
                authorize_media(None)
            with self.assertRaises(HTTPException):
                authorize_media("Bearer wrong")

    def test_audio_endpoint_gated_by_token(self):
        with patch.dict(os.environ, {"VEDIC_GENERATION_API_TOKEN": "secret", "XAI_API_KEY": "k"}), \
             patch("vedic_pipeline.api.verse_service.get_verse", return_value=None), \
             TestClient(create_app()) as client:
            # Sem token -> 401 antes de checar o verso
            self.assertEqual(client.get("/api/v1/verses/RV.10.129.1/audio").status_code, 401)
            # Com token -> passa do gate (verso inexistente -> 404)
            self.assertEqual(
                client.get("/api/v1/verses/RV.10.129.1/audio", headers={"Authorization": "Bearer secret"}).status_code,
                404,
            )

    def test_image_generation_gated_but_cache_open(self):
        with tempfile.TemporaryDirectory() as tmp:
            cached = Path(tmp) / "RV.10.129.1.jpg"
            cached.write_bytes(b"\xff\xd8" + b"\x00" * 2000)  # "jpeg" com >1000 bytes
            with patch.dict(os.environ, {"VEDIC_GENERATION_API_TOKEN": "secret", "XAI_API_KEY": "k"}), \
                 patch("vedic_pipeline.llm.imagine.image_path", return_value=cached), \
                 TestClient(create_app()) as client:
                # Imagem em cache: leitura aberta mesmo com token configurado
                self.assertEqual(client.get("/api/v1/verses/RV.10.129.1/image").status_code, 200)


class ConsistencyTests(unittest.TestCase):
    def test_version_matches_package(self):
        with TestClient(create_app()) as client:
            health = client.get("/api/v1/health").json()
        self.assertEqual(health["version"], __version__)
        self.assertFalse(health["reranker"]["enabled"])
        self.assertFalse(health["reranker"]["default_enabled"])
        self.assertTrue(health["reranker"]["eligible_opt_in"])
        self.assertIn("require_token", health["generation"])
        self.assertIn("matched_profiles", health["corpus_profile"])

    def test_metric_paths_collapse_cardinality(self):
        METRICS.reset()
        for v in ["RV.10.129.1", "RV.10.129.2", "BG.2.47"]:
            METRICS.record_http_request("GET", f"/api/v1/verses/{v}/audio", 200, 0.01)
        for d in ["a", "b", "c"]:
            METRICS.record_http_request("GET", f"/api/v1/documents/{d}", 200, 0.01)
        text = METRICS.render_prometheus()
        series = [ln for ln in text.splitlines() if ln.startswith("vedas_http_requests_total{")]
        # Três versos + três docs viram apenas duas séries (sem rótulo por id)
        self.assertEqual(len(series), 2)
        self.assertIn("path=\"/api/v1/verses/:id/audio\"", text)
        self.assertIn("path=\"/api/v1/documents/:id\"", text)
        METRICS.reset()


if __name__ == "__main__":
    unittest.main()
