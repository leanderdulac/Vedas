"""Stable Diffusion local: roteamento, refino do still, vídeo e drone de áudio."""

from __future__ import annotations

import base64
import os
import tempfile
import unittest
from importlib.util import find_spec
from pathlib import Path
from unittest.mock import patch

import numpy as np

from vedic_pipeline.llm import diffusion, imagine

# Extra `[media]` (diffusers + torchsde): o CI instala só a API, então estes pulam lá.
_MEDIA_EXTRA = find_spec("diffusers") is not None and find_spec("torchsde") is not None
needs_media_extra = unittest.skipUnless(_MEDIA_EXTRA, 'requer o extra "media" (diffusers + torchsde)')


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
        with patch.dict(os.environ, {"VEDIC_SD_STEPS": "4", "VEDIC_SD_TURBO_GUIDANCE": "0"}):
            self.assertEqual(diffusion.sampling_for("stabilityai/sd-turbo"), (4, 0.0))
        with patch.dict(os.environ, {"VEDIC_SD_STEPS": "12", "VEDIC_SD_TURBO_GUIDANCE": "0"}):
            self.assertEqual(diffusion.sampling_for("stabilityai/sdxl-turbo"), (4, 0.0))

    def test_turbo_defaults_to_a_light_cfg(self):
        with patch.dict(os.environ, {"VEDIC_SD_STEPS": "4", "VEDIC_SD_TURBO_GUIDANCE": ""}):
            self.assertEqual(diffusion.sampling_for("stabilityai/sdxl-turbo"), (4, 1.5))
        with patch.dict(os.environ, {"VEDIC_SD_STEPS": "20", "VEDIC_SD_GUIDANCE": "6.5"}):
            self.assertEqual(diffusion.sampling_for("stabilityai/stable-diffusion-xl-base-1.0"), (20, 6.5))

    def test_turbo_light_cfg_turns_on_the_negative(self):
        with patch.dict(os.environ, {"VEDIC_SD_STEPS": "6", "VEDIC_SD_TURBO_GUIDANCE": "2"}):
            self.assertEqual(diffusion.sampling_for("stabilityai/sd-turbo"), (6, 2.0))
        with patch.dict(os.environ, {"VEDIC_SD_STEPS": "30", "VEDIC_SD_TURBO_GUIDANCE": "9"}):
            self.assertEqual(diffusion.sampling_for("stabilityai/sdxl-turbo"), (8, 3.0))

    def test_img2img_keeps_at_least_one_step(self):
        self.assertEqual(diffusion.img2img_steps(1, 0.35), 3)
        self.assertEqual(diffusion.img2img_steps(4, 0.35), 4)

    def test_image_model_falls_back_when_offline_and_missing(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            os.environ, {"HF_HOME": tmp, "HF_HUB_OFFLINE": "1", "VEDIC_SD_MODEL": "stabilityai/sdxl-turbo"}
        ):
            self.assertEqual(diffusion.image_model(), diffusion.DEFAULT_SD_MODEL)
            snap = Path(tmp) / "hub" / "models--stabilityai--sdxl-turbo" / "snapshots" / "abc"
            snap.mkdir(parents=True)
            (snap / "model_index.json").write_text("{}")
            (snap.parents[1] / "refs").mkdir()
            (snap.parents[1] / "refs" / "main").write_text("abc")
            self.assertEqual(diffusion.image_model(), "stabilityai/sdxl-turbo")
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"HF_HOME": tmp, "VEDIC_SD_MODEL": ""}):
            self.assertEqual(diffusion.image_model(), diffusion.DEFAULT_SD_MODEL)
            snap = Path(tmp) / "hub" / "models--stabilityai--sdxl-turbo" / "snapshots" / "abc"
            snap.mkdir(parents=True)
            (snap / "model_index.json").write_text("{}")
            (snap.parents[1] / "refs").mkdir()
            (snap.parents[1] / "refs" / "main").write_text("abc")
            self.assertEqual(diffusion.image_model(), diffusion.PREFERRED_SD_MODEL)

    @unittest.skipUnless(find_spec("PIL") is not None, "requer Pillow")
    def test_video_still_is_cropped_not_stretched(self):
        from PIL import Image

        square = Image.new("RGB", (512, 512))
        out = diffusion._cover(square, 1024, 576)
        self.assertEqual(out.size, (1024, 576))
        with patch.dict(os.environ, {"VEDIC_SVD_WIDTH": "768"}):
            self.assertEqual(diffusion.video_size(), (768, 432))
        with patch.dict(os.environ, {"VEDIC_SVD_WIDTH": ""}):
            self.assertEqual(diffusion.video_size(), (1024, 576))

    def test_gentle_motion_uses_a_low_bucket(self):
        with patch.dict(os.environ, {"VEDIC_SVD_MOTION": ""}):
            self.assertEqual(diffusion.motion_bucket("Gentle sacred motion, slow push-in"), 30)
            self.assertEqual(diffusion.motion_bucket("abrupt cut"), 90)

    def test_mps_and_cuda_use_float16(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch ausente")

        with patch.dict(os.environ, {"VEDIC_SD_DTYPE": ""}), patch(
            "vedic_pipeline.train.device.torch_device", return_value=torch.device("mps")
        ):
            device, dtype = diffusion._device_and_dtype()
            self.assertEqual(device.type, "mps")
            self.assertEqual(dtype, torch.float16)
        with patch.dict(os.environ, {"VEDIC_SD_DTYPE": ""}), patch(
            "vedic_pipeline.train.device.torch_device", return_value=torch.device("cpu")
        ):
            _device, dtype = diffusion._device_and_dtype()
            self.assertEqual(dtype, torch.float32)

    def test_place_keeps_the_vae_in_fp16(self):
        import torch

        class Vae(torch.nn.Linear):
            pass

        class Pipe:
            def __init__(self) -> None:
                self.vae = Vae(1, 1, dtype=torch.float16)

            def to(self, device):
                self.vae.to(device)
                return self

        pipe = Pipe()
        with patch("vedic_pipeline.train.device.torch_device", return_value=torch.device("cpu")):
            placed = diffusion._place(pipe)
        self.assertEqual(placed.vae.weight.dtype, torch.float16)

    def test_cached_snapshot_uses_the_local_model_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            snap = Path(tmp) / "hub" / "models--org--audio" / "snapshots" / "abc"
            snap.mkdir(parents=True)
            (snap / "model_index.json").write_text("{}")
            refs = snap.parents[1] / "refs"
            refs.mkdir()
            (refs / "main").write_text("abc\n")
            with patch.dict(os.environ, {"HF_HOME": tmp}):
                found = diffusion._cached_snapshot("org/audio")
                missing = diffusion._cached_snapshot("org/missing")
            self.assertEqual(Path(found or ""), snap)
            self.assertIsNone(missing)

    def test_fp16_load_prefers_the_fp16_variant(self):
        import torch

        calls: list[dict] = []

        class Pipe:
            def set_progress_bar_config(self, **_kwargs):
                return None

        class Fake:
            @classmethod
            def from_pretrained(cls, _model_id, **kwargs):
                calls.append(kwargs)
                if kwargs.get("variant") != "fp16":
                    raise OSError("sem variante")
                return Pipe()

        with patch("vedic_pipeline.train.device.torch_device", return_value=torch.device("mps")):
            diffusion._from_pretrained(Fake, "stabilityai/sd-turbo", torch_dtype=torch.float16)
        self.assertEqual(calls[0].get("variant"), "fp16")

    def test_float32_load_skips_the_fp16_variant(self):
        import torch

        calls: list[dict] = []

        class Pipe:
            def set_progress_bar_config(self, **_kwargs):
                return None

        class Fake:
            @classmethod
            def from_pretrained(cls, _model_id, **kwargs):
                calls.append(kwargs)
                return Pipe()

        with patch("vedic_pipeline.train.device.torch_device", return_value=torch.device("mps")):
            diffusion._from_pretrained(Fake, "org/audio-ausente", torch_dtype=torch.float32)
        self.assertEqual(len(calls), 1)
        self.assertNotIn("variant", calls[0])
        self.assertEqual(calls[0].get("torch_dtype"), torch.float32)

    @needs_media_extra
    def test_brownian_tree_is_built_on_cpu(self):
        import torch
        from diffusers.schedulers.scheduling_dpmsolver_sde import BatchedBrownianTree

        diffusion._use_cpu_brownian()
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        tree = BatchedBrownianTree(torch.zeros(2, 4, device=device), 0.3, 500.0, seed=1)
        noise = tree(1.0, 2.0)
        self.assertEqual(tree.trees[0]._device.type, "cpu")
        self.assertEqual(noise.device.type, "cpu")
        self.assertEqual(tuple(noise.shape), (2, 4))

    @needs_media_extra
    def test_audio_scheduler_does_not_end_at_zero(self):
        from diffusers import CosineDPMSolverMultistepScheduler

        sched = CosineDPMSolverMultistepScheduler(
            final_sigmas_type="zero",
            sigma_min=0.3,
            sigma_max=500.0,
        )
        fixed = diffusion._audio_scheduler(sched)
        fixed.set_timesteps(8)
        self.assertGreater(float(fixed.sigmas[-1]), 0.0)

    @needs_media_extra
    def test_zero_length_brownian_step_is_finite(self):
        import torch
        from diffusers.schedulers.scheduling_dpmsolver_sde import BrownianTreeNoiseSampler

        diffusion._use_cpu_brownian()
        sampler = BrownianTreeNoiseSampler(torch.zeros(2, 4), sigma_min=0.3, sigma_max=500.0, seed=1)
        noise = sampler(0.3, 0.3)
        self.assertFalse(torch.isnan(noise).any().item())
        self.assertEqual(float(noise.abs().sum()), 0.0)

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

    def test_quiet_bed_is_normalized_before_the_mix(self):
        quiet = np.array([0.0, 0.02, -0.01], dtype=np.float32)
        loud = diffusion.normalize_peak(quiet)
        self.assertAlmostEqual(float(np.max(np.abs(loud))), 1.0, places=5)
        silent = np.zeros(3, dtype=np.float32)
        self.assertEqual(diffusion.normalize_peak(silent).tolist(), [0.0, 0.0, 0.0])

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

    def test_figure_refresh_starts_over_with_the_negative(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(imagine, "MEDIA_DIR", Path(tmp)), patch(
            "vedic_pipeline.llm.diffusion.media_backend", return_value="diffusion"
        ), patch("vedic_pipeline.llm.diffusion.render_image", return_value=b"j" * 2000) as render:
            first = imagine.generate_figure_image("agni", "Agni, two heads", negative="blue skin")
            self.assertTrue(first.exists())
            imagine.generate_figure_image("agni", "Agni, two heads", negative="blue skin", force=True)
            self.assertEqual(render.call_count, 2)
            self.assertIsNone(render.call_args.kwargs.get("init_image"))
            self.assertEqual(render.call_args.kwargs.get("negative"), "blue skin")

    def test_compact_verse_prompt_puts_the_style_first(self):
        meaning = " ".join(["word"] * 200)
        bundle = {"locator": "RV 1.1.1", "witnesses": [{"role": "en", "text": meaning}]}
        compact = imagine.visual_prompt(bundle, compact=True)
        self.assertTrue(compact.startswith(imagine.COMPACT_STYLE))
        self.assertLessEqual(len(compact.split()), 50)
        self.assertNotIn("no Latin", compact)
        self.assertIn("no Latin", imagine.visual_prompt(bundle))

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
