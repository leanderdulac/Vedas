"""Backend de vídeo configurável: SVD local, xAI e Runway (HTTP mockado)."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx

from vedic_pipeline.llm import diffusion, imagine, video_providers

_MP4 = b"\x00\x00\x00\x18ftypmp42" + b"v" * 4000


def _mock_client(handler, calls):
    def factory(base_url, headers):
        def record(request: httpx.Request) -> httpx.Response:
            calls.append(request)
            return handler(request)

        return httpx.Client(base_url=base_url, headers=headers, transport=httpx.MockTransport(record))

    return factory


def _still(tmp: str) -> Path:
    path = Path(tmp) / "RV.1.1.5.jpg"
    path.write_bytes(b"\xff\xd8\xff" + b"j" * 2000)
    return path


class VideoBackendChoiceTests(unittest.TestCase):
    def test_explicit_backends(self):
        for name in ("svd", "xai", "runway"):
            with patch.dict(os.environ, {"VEDIC_VIDEO_BACKEND": name.upper()}):
                self.assertEqual(diffusion.video_backend(), name)
        with patch.dict(os.environ, {"VEDIC_VIDEO_BACKEND": "diffusion"}):
            self.assertEqual(diffusion.video_backend(), "svd")

    def test_empty_or_unknown_follows_media_backend(self):
        for value in ("", "auto", "bogus"):
            with patch.dict(os.environ, {"VEDIC_VIDEO_BACKEND": value, "VEDIC_MEDIA_BACKEND": "diffusion"}):
                self.assertEqual(diffusion.video_backend(), "svd")
            with patch.dict(os.environ, {"VEDIC_VIDEO_BACKEND": value, "VEDIC_MEDIA_BACKEND": "xai"}):
                self.assertEqual(diffusion.video_backend(), "xai")

    def test_models_and_seconds_from_env(self):
        with patch.dict(os.environ, {"RUNWAY_VIDEO_MODEL": "", "XAI_VIDEO_MODEL": "", "VEDIC_VIDEO_SECONDS": ""}):
            self.assertEqual(video_providers.runway_video_model(), "gen4.5")
            self.assertEqual(video_providers.xai_video_model(), "grok-imagine-video-1.5")
            self.assertEqual(video_providers.video_seconds(), 5)
        with patch.dict(os.environ, {"RUNWAY_VIDEO_MODEL": "gen4_turbo", "VEDIC_VIDEO_SECONDS": "99"}):
            self.assertEqual(video_providers.runway_video_model(), "gen4_turbo")
            self.assertEqual(video_providers.video_seconds(), 10)


class RunwayTests(unittest.TestCase):
    def test_submits_polls_and_downloads(self):
        calls: list[httpx.Request] = []
        polls = iter([{"status": "RUNNING", "progress": 0.4}, {"status": "SUCCEEDED", "output": ["https://cdn.example/v.mp4"], "cost": {"credits": 60}}])

        def handler(request: httpx.Request) -> httpx.Response:
            if request.method == "POST":
                return httpx.Response(200, json={"id": "task-1"})
            return httpx.Response(200, json={"id": "task-1", **next(polls)})

        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            os.environ, {"RUNWAYML_API_SECRET": "test-secret", "RUNWAY_VIDEO_MODEL": "", "VEDIC_VIDEO_SECONDS": ""}
        ), patch.object(video_providers, "_http_client", _mock_client(handler, calls)), patch.object(
            video_providers, "_download", return_value=_MP4
        ) as download:
            dest = Path(tmp) / "out.mp4"
            meta = video_providers.render_runway_video(_still(tmp), dest, "Agni, slow flames", sleep=lambda _s: None)
            self.assertEqual(dest.read_bytes(), _MP4)
        download.assert_called_once_with("https://cdn.example/v.mp4")
        self.assertEqual(meta, {"backend": "runway", "model": "gen4.5", "seconds": 5, "credits": 60, "cost_usd": 0.6})
        post = calls[0]
        self.assertEqual(post.url.path, "/v1/image_to_video")
        self.assertEqual(post.headers["X-Runway-Version"], video_providers.RUNWAY_VERSION)
        self.assertEqual(post.headers["Authorization"], "Bearer test-secret")
        body = json.loads(post.content)
        self.assertEqual((body["model"], body["ratio"], body["duration"]), ("gen4.5", "1280:720", 5))
        self.assertTrue(body["promptImage"].startswith("data:image/jpeg;base64,"))
        self.assertEqual(calls[-1].url.path, "/v1/tasks/task-1")

    def test_refusal_raises_provider_error(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(400, json={"error": "You do not have enough credits to run this task."})

        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"RUNWAYML_API_SECRET": "x"}), patch.object(
            video_providers, "_http_client", _mock_client(handler, [])
        ), self.assertRaisesRegex(video_providers.VideoProviderError, "enough credits"):
            video_providers.render_runway_video(_still(tmp), Path(tmp) / "o.mp4", "p", sleep=lambda _s: None)

    def test_failed_task_and_missing_key(self):
        def handler(request: httpx.Request) -> httpx.Response:
            if request.method == "POST":
                return httpx.Response(200, json={"id": "t"})
            return httpx.Response(200, json={"id": "t", "status": "FAILED", "failure": "moderation", "failureCode": "SAFETY"})

        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"RUNWAYML_API_SECRET": "x"}), patch.object(
                video_providers, "_http_client", _mock_client(handler, [])
            ), self.assertRaisesRegex(video_providers.VideoProviderError, "SAFETY"):
                video_providers.render_runway_video(_still(tmp), Path(tmp) / "o.mp4", "p", sleep=lambda _s: None)
            with patch.dict(os.environ, {"RUNWAYML_API_SECRET": ""}), self.assertRaises(video_providers.VideoProviderError):
                video_providers.render_runway_video(_still(tmp), Path(tmp) / "o.mp4", "p")

    def test_poll_times_out(self):
        def handler(request: httpx.Request) -> httpx.Response:
            if request.method == "POST":
                return httpx.Response(200, json={"id": "t"})
            return httpx.Response(200, json={"id": "t", "status": "PENDING"})

        ticks = iter(range(0, 10_000, 100))
        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            os.environ, {"RUNWAYML_API_SECRET": "x", "VEDIC_VIDEO_TIMEOUT": "60"}
        ), patch.object(video_providers, "_http_client", _mock_client(handler, [])), self.assertRaisesRegex(
            video_providers.VideoProviderError, "tempo esgotado"
        ):
            video_providers.render_runway_video(
                _still(tmp), Path(tmp) / "o.mp4", "p", sleep=lambda _s: None, clock=lambda: next(ticks)
            )


class XaiVideoTests(unittest.TestCase):
    def test_submits_polls_and_reports_cost(self):
        calls: list[httpx.Request] = []
        polls = iter(
            [
                {"status": "pending", "progress": 30},
                {
                    "status": "done",
                    "model": "grok-imagine-video-1.5",
                    "video": {"url": "https://vidgen.x.ai/v.mp4", "duration": 5},
                    "usage": {"cost_in_usd_ticks": 4_000_000_000},
                },
            ]
        )

        def handler(request: httpx.Request) -> httpx.Response:
            if request.method == "POST":
                return httpx.Response(200, json={"request_id": "req-1"})
            return httpx.Response(200, json=next(polls))

        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            os.environ, {"XAI_API_KEY": "test-key", "XAI_VIDEO_MODEL": "", "XAI_BASE_URL": "", "VEDIC_VIDEO_SECONDS": ""}
        ), patch.object(video_providers, "_http_client", _mock_client(handler, calls)), patch.object(
            video_providers, "_download", return_value=_MP4
        ):
            dest = Path(tmp) / "out.mp4"
            meta = video_providers.render_xai_video(_still(tmp), dest, "Agni", sleep=lambda _s: None)
            self.assertTrue(dest.exists())
        self.assertEqual(meta, {"backend": "xai", "model": "grok-imagine-video-1.5", "seconds": 5, "cost_usd": 0.4})
        body = json.loads(calls[0].content)
        self.assertEqual(calls[0].url.path, "/v1/videos/generations")
        self.assertEqual((body["duration"], body["aspect_ratio"], body["resolution"]), (5, "16:9", "720p"))
        self.assertTrue(body["image"]["url"].startswith("data:image/jpeg;base64,"))
        self.assertEqual(calls[-1].url.path, "/v1/videos/req-1")

    def test_failed_generation_raises(self):
        def handler(request: httpx.Request) -> httpx.Response:
            if request.method == "POST":
                return httpx.Response(200, json={"request_id": "r"})
            return httpx.Response(200, json={"status": "failed", "error": {"code": "invalid_argument", "message": "blocked"}})

        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"XAI_API_KEY": "k"}), patch.object(
            video_providers, "_http_client", _mock_client(handler, [])
        ), self.assertRaisesRegex(video_providers.VideoProviderError, "blocked"):
            video_providers.render_xai_video(_still(tmp), Path(tmp) / "o.mp4", "p", sleep=lambda _s: None)


class VideoJobTests(unittest.TestCase):
    bundle = {"locator": "RV 1.1.5", "verse_id": "RV.1.1.5", "witnesses": [{"role": "en", "text": "Agni, the invoker [RV 1.1.5]"}]}

    def _run(self, backend, renderer=None, *, installed=True, render_video=None):
        tmp = tempfile.mkdtemp()
        self.addCleanup(lambda: __import__("shutil").rmtree(tmp, ignore_errors=True))
        patches = [
            patch.object(imagine, "MEDIA_DIR", Path(tmp)),
            patch.object(imagine, "get_verse", return_value=self.bundle),
            patch("vedic_pipeline.llm.diffusion.diffusion_installed", return_value=installed),
            patch("vedic_pipeline.llm.diffusion.render_video", side_effect=render_video or (lambda s, d, m: d.write_bytes(_MP4))),
        ]
        if renderer is not None:
            patches.append(patch.dict(video_providers.RENDERERS, {backend: renderer}))
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        imagine._video_worker("RV.1.1.5", backend)
        return imagine._read_job("RV.1.1.5"), imagine.poll_verse_video("RV.1.1.5")

    def test_start_uses_the_configured_backend_in_a_thread(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(imagine, "MEDIA_DIR", Path(tmp)), patch(
            "vedic_pipeline.llm.diffusion.video_backend", return_value="runway"
        ), patch.object(imagine, "generate_verse_image"), patch.object(imagine, "threading") as threads:
            job = imagine.start_verse_video("RV.1.1.5")
            self.assertEqual((job["status"], job["backend"]), ("pending", "runway"))
            self.assertEqual(threads.Thread.call_args.kwargs["args"], ("RV.1.1.5", "runway"))
            again = imagine.start_verse_video("RV.1.1.5")
            self.assertEqual(threads.Thread.call_count, 1)
            self.assertEqual(again["backend"], "runway")
            status = imagine.poll_verse_video("RV.1.1.5")
            self.assertEqual((status["status"], status["ready"]), ("pending", False))

    def test_stale_pending_job_restarts(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(imagine, "MEDIA_DIR", Path(tmp)), patch(
            "vedic_pipeline.llm.diffusion.video_backend", return_value="svd"
        ), patch.object(imagine, "generate_verse_image"), patch.object(imagine, "threading") as threads:
            imagine._write_job("RV.1.1.5", {"verse_id": "RV.1.1.5", "status": "pending", "backend": "svd", "started_at": 1.0})
            imagine.start_verse_video("RV.1.1.5")
            threads.Thread.assert_called_once()

    def test_api_backend_success_records_cost(self):
        def renderer(still, dest, prompt):
            self.assertIn("Vedic verse RV 1.1.5", prompt)
            self.assertIn("Agni, the invoker", prompt)
            self.assertNotIn("[RV", prompt)
            dest.write_bytes(_MP4)
            return {"backend": "runway", "model": "gen4.5", "seconds": 5, "credits": 60, "cost_usd": 0.6}

        job, status = self._run("runway", renderer, render_video=AssertionError)
        self.assertTrue(status["ready"])
        self.assertEqual((job["backend"], job["cost_usd"]), ("runway", 0.6))
        self.assertNotIn("fallback_from", job)

    def test_api_failure_falls_back_to_svd(self):
        def renderer(still, dest, prompt):
            raise video_providers.VideoProviderError("Runway recusou o vídeo (400): not enough credits")

        job, status = self._run("runway", renderer)
        self.assertTrue(status["ready"])
        self.assertEqual((job["backend"], job["fallback_from"]), ("svd", "runway"))
        self.assertIn("credits", job["fallback_reason"])

    def test_api_failure_without_media_extra_is_visible(self):
        def renderer(still, dest, prompt):
            raise httpx.ConnectError("offline")

        job, status = self._run("xai", renderer, installed=False)
        self.assertEqual(status["status"], "failed")
        self.assertIn("offline", status["detail"])
        self.assertEqual(job["backend"], "xai")

    def test_svd_backend_never_calls_apis(self):
        with patch.dict(video_providers.RENDERERS, {"xai": AssertionError, "runway": AssertionError}):
            job, status = self._run("svd")
        self.assertTrue(status["ready"])
        self.assertEqual(job["backend"], "svd")


if __name__ == "__main__":
    unittest.main()
