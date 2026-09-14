"""Fila de trabalhos assíncronos para ops pesadas do pipeline."""

from __future__ import annotations

import os
import time
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from vedic_pipeline.api.app import create_app
from vedic_pipeline.api.jobs import get_job, list_jobs, reset_jobs, submit_job


def _wait_for(job_id: str, timeout: float = 5.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = get_job(job_id)
        assert job is not None
        if job["status"] in {"done", "error"}:
            return job
        time.sleep(0.02)
    raise AssertionError(f"job {job_id} não finalizou em {timeout}s")


class JobManagerTests(unittest.TestCase):
    def setUp(self):
        reset_jobs()

    def tearDown(self):
        reset_jobs()

    def test_submit_runs_and_returns_result(self):
        job = submit_job("demo", lambda: {"ok": True}, {"a": 1})
        self.assertEqual(job["status"], "queued")
        done = _wait_for(job["job_id"])
        self.assertEqual(done["status"], "done")
        self.assertEqual(done["result"], {"ok": True})
        self.assertIsNotNone(done["finished_at"])

    def test_error_captured_without_raise(self):
        def _boom():
            raise RuntimeError("quebrou de propósito")

        job = submit_job("demo", _boom)
        done = _wait_for(job["job_id"])
        self.assertEqual(done["status"], "error")
        self.assertIn("quebrou", done["error"])

    def test_get_and_list(self):
        self.assertIsNone(get_job("inexistente"))
        j1 = submit_job("a", lambda: 1)
        j2 = submit_job("b", lambda: 2)
        _wait_for(j1["job_id"])
        _wait_for(j2["job_id"])
        ids = {j["job_id"] for j in list_jobs()}
        self.assertIn(j1["job_id"], ids)
        self.assertIn(j2["job_id"], ids)


class JobApiTests(unittest.TestCase):
    def setUp(self):
        reset_jobs()

    def tearDown(self):
        reset_jobs()

    def test_async_requires_token(self):
        with patch.dict(os.environ, {"VEDIC_PIPELINE_API_TOKEN": ""}), TestClient(
            create_app()
        ) as client:
            self.assertEqual(client.post("/ingest/async", json={"manifest": "x"}).status_code, 503)
            self.assertEqual(client.get("/jobs").status_code, 503)
            self.assertEqual(client.get("/jobs/abc").status_code, 503)

    def test_ingest_async_flow(self):
        with patch.dict(os.environ, {"VEDIC_PIPELINE_API_TOKEN": "tok"}), patch(
            "vedic_pipeline.crawler.ingest.ingest_manifest", return_value={"ok": True, "added": 1}
        ), TestClient(create_app()) as client:
            headers = {"Authorization": "Bearer tok"}
            res = client.post("/ingest/async", json={"manifest": "fixtures/sources_local.json"}, headers=headers)
            self.assertEqual(res.status_code, 202, res.text)
            job_id = res.json()["job_id"]
            deadline = time.time() + 5.0
            final: dict | None = None
            while time.time() < deadline:
                got = client.get(f"/jobs/{job_id}", headers=headers)
                self.assertEqual(got.status_code, 200)
                if got.json()["status"] == "done":
                    final = got.json()
                    break
                time.sleep(0.02)
            self.assertIsNotNone(final)
            assert final is not None
            self.assertEqual(final["result"], {"ok": True, "added": 1})
            # lista contém o job; id inválido dá 404
            listed = client.get("/jobs", headers=headers).json()["items"]
            self.assertTrue(any(j["job_id"] == job_id for j in listed))
            self.assertEqual(client.get("/jobs/nope", headers=headers).status_code, 404)


if __name__ == "__main__":
    unittest.main()
