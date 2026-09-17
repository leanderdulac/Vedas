"""Fila de trabalhos assíncronos para ops pesadas do pipeline."""

from __future__ import annotations

import os
import tempfile
import time
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from vedic_pipeline.api.app import create_app
from vedic_pipeline.api.jobs import (
    get_job,
    job_counts,
    jobs_dir,
    list_jobs,
    load_jobs_from_disk,
    reset_jobs,
    submit_job,
)


def _wait_for(job_id: str, timeout: float = 5.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = get_job(job_id)
        assert job is not None
        if job["status"] in {"done", "error"}:
            return job
        time.sleep(0.02)
    raise AssertionError(f"job {job_id} não finalizou em {timeout}s")


class _TempJobsDirMixin:
    """Isola VEDIC_JOBS_DIR em diretório temporário para não tocar data/jobs real."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._env = patch.dict(os.environ, {"VEDIC_JOBS_DIR": self._tmp.name})
        self._env.start()
        reset_jobs()

    def tearDown(self):
        reset_jobs()
        self._env.stop()
        self._tmp.cleanup()


class JobManagerTests(_TempJobsDirMixin, unittest.TestCase):
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

    def test_jobs_persisted_to_disk(self):
        job = submit_job("demo", lambda: {"ok": True})
        _wait_for(job["job_id"])
        path = jobs_dir() / f"{job['job_id']}.json"
        self.assertTrue(path.exists())
        self.assertIn("demo", path.read_text(encoding="utf-8"))

    def test_restore_marks_interrupted(self):
        j1 = submit_job("demo", lambda: {"ok": True})
        _wait_for(j1["job_id"])
        # simula job em execução deixado por um restart
        (jobs_dir() / "stale.json").write_text(
            '{"job_id": "stale123", "kind": "train", "status": "running", '
            '"created_at": "2026-09-15T00:00:00+00:00", "finished_at": null, '
            '"params": {}, "result": null, "error": null}',
            encoding="utf-8",
        )
        restored = load_jobs_from_disk()
        self.assertGreaterEqual(restored, 2)
        stale = get_job("stale123")
        self.assertIsNotNone(stale)
        self.assertEqual(stale["status"], "error")
        self.assertIn("reinício", stale["error"] or "")
        # job concluído não muda de status no restore
        self.assertEqual(get_job(j1["job_id"])["status"], "done")

    def test_job_counts(self):
        j1 = submit_job("ingest", lambda: {"ok": True})
        j2 = submit_job("train", lambda: {"ok": True})
        _wait_for(j1["job_id"])
        _wait_for(j2["job_id"])
        counts = job_counts()
        self.assertEqual(counts["total"], 2)
        self.assertEqual(counts["by_status"], {"done": 2})
        self.assertEqual(counts["by_kind"], {"ingest": 1, "train": 1})


class JobApiTests(_TempJobsDirMixin, unittest.TestCase):
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

    def test_nullable_payloads_from_ui_accepted(self):
        """A UI envia embedding_dim/max_steps como null — o schema deve aceitar."""
        with patch.dict(os.environ, {"VEDIC_PIPELINE_API_TOKEN": "tok"}), patch(
            "vedic_pipeline.search.embeddings.build_embedding_index", return_value={"ok": True}
        ), patch(
            "vedic_pipeline.search.rag.get_index", return_value={"chunks": []}
        ), TestClient(create_app()) as client:
            headers = {"Authorization": "Bearer tok"}
            res = client.post(
                "/build-index/async",
                json={"backend": "numpy", "embedding_dim": None},
                headers=headers,
            )
            self.assertEqual(res.status_code, 202, res.text)
            res = client.post(
                "/train/async",
                json={"base_model": "gpt2", "max_steps": None},
                headers=headers,
            )
            self.assertEqual(res.status_code, 202, res.text)
            res = client.post("/db/init/async?dim=384", headers=headers)
            self.assertEqual(res.status_code, 202, res.text)


if __name__ == "__main__":
    unittest.main()
