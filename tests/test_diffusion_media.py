"""Stable Diffusion local: roteamento, refino do still, vídeo e drone de áudio."""

from __future__ import annotations

import base64
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from vedic_pipeline.llm import diffusion, imagine


class BackendTests(unittest.TestCase):
    def test_auto_uses_diffusion_only_when_installed(self):
        with patch.dict(os.environ, {"VEDIC_MEDIA_BACKEND": "auto"}), patch.object(
            diffusion, "diffusion_installed", return_value=False
        ):
            self.assertEqual(diffusion.media_backend(), "xai")
        with patch.dict(os.environ, {"VEDIC_MEDIA_BACKEND": "auto"}), patch.object(
            diffusion, "diffusion_installed", return_value=True
        ):
            self.assertEqual(diffusion.media_backend(), "diffusion")

    def test_explicit_backend(self):
        with patch.dict(os.environ, {"VEDIC_MEDIA_BACKEND": "xai"}):
            self.assertEqual(diffusion.media_backend(), "xai")
        with patch.dict(os.environ, {"VEDIC_MEDIA_BACKEND": "diffusion"}):
            self.assertEqual(diffusion.media_backend(), "diffusion")

    def test_turbo_sampling_disables_guidance(self):
        with patch.dict(os.environ, {"VEDIC_SD_STEPS": "4"}):
            self.assertEqual(diffusion.sampling_for("stabilityai/sd-turbo"), (4, 0.0))
        with patch.dict(os.environ, {"VEDIC_SD_STEPS": "20", "VEDIC_SD_GUIDANCE": "6.5"}):
            self.assertEqual(diffusion.sampling_for("stabilityai/stable-diffusion-xl-base-1.0"), (20, 6.5))

    def test_gentle_motion_uses_a_low_bucket(self):
        with patch.dict(os.environ, {"VEDIC_SVD_MOTION": ""}):
            self.assertEqual(diffusion.motion_bucket("Gentle sacred motion, slow push-in"), 30)
            self.assertEqual(diffusion.motion_bucket("abrupt cut"), 90)

    def test_mps_and_cuda_use_float16(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch ausente")

        with patch("vedic_pipeline.train.device.torch_device", return_value=torch.device("mps")):
            device, dtype = diffusion._device_and_dtype()
            self.assertEqual(device.type, "mps")
            self.assertEqual(dtype, torch.float16)
        with patch("vedic_pipeline.train.device.torch_device", return_value=torch.device("cpu")):
            _device, dtype = diffusion._device_and_dtype()
            self.assertEqual(dtype, torch.float32)

    def test_audio_bed_stays_off_without_the_extra(self):
        with patch.dict(os.environ, {"VEDIC_AUDIO_BED": "diffusion"}), patch.object(
            diffusion, "diffusion_installed", return_value=False
        ):
            self.assertFalse(diffusion.audio_bed_enabled())
        with patch.dict(os.environ, {"VEDIC_AUDIO_BED": "0"}):
            self.assertFalse(diffusion.audio_bed_enabled())


class MixTests(unittest.TestCase):
    def test_bed_loops_and_peak_stays_under_full_scale(self):
        speech = np.ones(10, dtype=np.float32)
        bed = np.ones(4, dtype=np.float32)
        mixed = diffusion.mix_waveforms(speech, bed, bed_gain=0.5)
        self.assertEqual(mixed.shape, (10,))
        self.assertLessEqual(float(mixed.max()), 0.98)

    def test_empty_bed_returns_speech(self):
        speech = np.array([0.2, -0.2], dtype=np.float32)
        out = diffusion.mix_waveforms(speech, np.array([], dtype=np.float32))
        self.assertEqual(out.tolist(), speech.tolist())

    def test_underlay_without_ffmpeg_keeps_the_voice(self):
        with patch.object(diffusion, "_ffmpeg", return_value=None):
            self.assertEqual(diffusion.underlay_speech(b"mp3-bytes", "tanpura"), b"mp3-bytes")


class _FakeResponse:
    def __init__(self, payload: dict):
        self.status_code = 200
        self._payload = payload
        self.text = ""

    def json(self):
        return self._payload


class _FakeClient:
    def __init__(self):
        self.posts: list[tuple[str, dict]] = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def post(self, url, json):
        self.posts.append((url, json))
        raw = base64.b64encode(b"j" * 1500).decode("ascii")
        return _FakeResponse({"data": [{"b64_json": raw}]})


class ImagineRoutingTests(unittest.TestCase):
    def test_diffusion_txt2img_then_refine(self):
        bundle = {"locator": "RV 1.1.1", "witnesses": [{"role": "en", "text": "I Laud Agni"}]}
        with tempfile.TemporaryDirectory() as tmp, patch.object(imagine, "MEDIA_DIR", Path(tmp)), patch(
            "vedic_pipeline.llm.diffusion.media_backend", return_value="diffusion"
        ), patch("vedic_pipeline.llm.diffusion.render_image", return_value=b"j" * 2000) as render, patch.object(
            imagine, "get_verse", return_value=bundle
        ):
            first = imagine.generate_verse_image("RV.1.1.1")
            self.assertTrue(first.exists())
            self.assertIsNone(render.call_args.kwargs.get("init_image"))
            second = imagine.generate_verse_image("RV.1.1.1", force=True)
            self.assertEqual(second, first)
            self.assertEqual(render.call_args.kwargs.get("init_image"), first)

    def test_xai_image_path_still_posts(self):
        bundle = {"locator": "RV 1.1.1", "witnesses": [{"role": "en", "text": "Agni"}]}
        client = _FakeClient()
        with tempfile.TemporaryDirectory() as tmp, patch.object(imagine, "MEDIA_DIR", Path(tmp)), patch(
            "vedic_pipeline.llm.diffusion.media_backend", return_value="xai"
        ), patch.object(imagine, "get_verse", return_value=bundle), patch.object(
            imagine, "_client", return_value=client
        ):
            path = imagine.generate_verse_image("RV.1.1.1")
            self.assertTrue(path.exists())
            self.assertEqual(client.posts[0][0], "/images/generations")
            self.assertIn("prompt", client.posts[0][1])

    def test_diffusion_video_is_local_and_polls_without_xai(self):
        bundle = {"locator": "RV 1.1.1", "verse_id": "RV.1.1.1", "witnesses": []}
        with tempfile.TemporaryDirectory() as tmp, patch.object(imagine, "MEDIA_DIR", Path(tmp)), patch(
            "vedic_pipeline.llm.diffusion.media_backend", return_value="diffusion"
        ), patch.object(imagine, "generate_verse_image", return_value=Path(tmp) / "RV.1.1.1.jpg"), patch.object(
            imagine, "threading"
        ) as threads, patch.object(imagine, "_client", side_effect=AssertionError("xAI não deve ser chamado")):
            threads.Thread.return_value.start.return_value = None
            job = imagine.start_verse_video("RV.1.1.1")
            self.assertEqual(job["status"], "pending")
            self.assertEqual(job["backend"], "diffusion")
            threads.Thread.assert_called_once()
            pending = imagine.poll_verse_video("RV.1.1.1")
            self.assertEqual(pending["status"], "pending")
            self.assertFalse(pending["ready"])

            def _fake_render(still, dest, motion):
                self.assertIn("Gentle", motion)
                dest.write_bytes(b"v" * 2000)

            with patch("vedic_pipeline.llm.diffusion.render_video", side_effect=_fake_render), patch.object(
                imagine, "get_verse", return_value=bundle
            ):
                (Path(tmp) / "RV.1.1.1.jpg").write_bytes(b"j" * 2000)
                imagine._diffusion_video_worker("RV.1.1.1")
            done = imagine.poll_verse_video("RV.1.1.1")
            self.assertTrue(done["ready"])
            self.assertTrue((Path(tmp) / "RV.1.1.1.mp4").exists())

    def test_diffusion_video_failure_is_visible(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(imagine, "MEDIA_DIR", Path(tmp)), patch(
            "vedic_pipeline.llm.diffusion.render_video", side_effect=RuntimeError("peso ausente")
        ), patch.object(imagine, "get_verse", return_value={}):
            imagine._diffusion_video_worker("RV.1.1.1")
            status = imagine.poll_verse_video("RV.1.1.1")
        self.assertEqual(status["status"], "failed")
        self.assertIn("peso ausente", status["detail"])


class AudioBedTests(unittest.TestCase):
    def test_cached_recitation_mixes_the_bed(self):
        from vedic_pipeline.llm import tts

        with tempfile.TemporaryDirectory() as tmp, patch.object(tts, "AUDIO_DIR", Path(tmp)), patch.object(
            tts, "synthesize", return_value=b"speech"
        ), patch("vedic_pipeline.llm.diffusion.audio_bed_enabled", return_value=True), patch(
            "vedic_pipeline.llm.diffusion.underlay_speech", return_value=b"mixed"
        ) as mix:
            path = tts.cached_verse_audio("RV.1.1.1", "agnim", language="sa")
            self.assertEqual(path.read_bytes(), b"mixed")
            mix.assert_called_once()
            self.assertIn("tanpura", mix.call_args.args[1])

    def test_bed_failure_keeps_the_voice(self):
        from vedic_pipeline.llm import tts

        with tempfile.TemporaryDirectory() as tmp, patch.object(tts, "AUDIO_DIR", Path(tmp)), patch.object(
            tts, "synthesize", return_value=b"speech"
        ), patch("vedic_pipeline.llm.diffusion.audio_bed_enabled", return_value=True), patch(
            "vedic_pipeline.llm.diffusion.underlay_speech", side_effect=RuntimeError("boom")
        ):
            path = tts.cached_verse_audio("RV.1.1.1", "agnim", language="sa")
            self.assertEqual(path.read_bytes(), b"speech")


if __name__ == "__main__":
    unittest.main()
