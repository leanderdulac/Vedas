"""Testes unitários para o módulo de métricas Prometheus e endpoint /metrics."""

from __future__ import annotations

import unittest

from fastapi.testclient import TestClient

from vedic_pipeline.api.app import create_app
from vedic_pipeline.api.metrics import METRICS


class MetricsTests(unittest.TestCase):
    def setUp(self):
        METRICS.reset()

    def tearDown(self):
        METRICS.reset()

    def test_record_http_request_and_render_prometheus(self):
        METRICS.record_http_request(method="GET", path="/api/v1/health", status=200, duration_s=0.012)
        METRICS.record_http_request(method="POST", path="/api/v1/search", status=200, duration_s=0.045)
        METRICS.record_search(backend="pgvector", duration_s=0.035)

        text = METRICS.render_prometheus(doc_count=12, chunk_count=15206)

        self.assertIn("vedas_corpus_documents_total 12", text)
        self.assertIn("vedas_corpus_chunks_total 15206", text)
        self.assertIn('vedas_http_requests_total{method="GET",path="/api/v1/health",status="200"} 1', text)
        self.assertIn('vedas_http_requests_total{method="POST",path="/api/v1/search",status="200"} 1', text)
        self.assertIn('vedas_search_requests_total{backend="pgvector"} 1', text)

    def test_metrics_endpoint_via_fastapi(self):
        app = create_app()
        client = TestClient(app)

        # Faz requisição de health
        res_health = client.get("/health")
        self.assertEqual(res_health.status_code, 200)

        # Consulta endpoint /metrics
        res_metrics = client.get("/metrics")
        self.assertEqual(res_metrics.status_code, 200)
        self.assertIn("text/plain", res_metrics.headers["content-type"])
        self.assertIn("vedas_uptime_seconds", res_metrics.text)
        self.assertIn("vedas_http_requests_total", res_metrics.text)


if __name__ == "__main__":
    unittest.main()
