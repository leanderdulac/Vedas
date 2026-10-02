"""Stable Diffusion local: roteamento, refino do still, vídeo e drone de áudio."""

from __future__ import annotations

import base64
import os
import shutil
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
        with patch.dict(os.environ, {"VEDIC_SVD_WIDTH": "1024"}):
            self.assertEqual(diffusion.video_size(), (1024, 576))
        with patch.dict(os.environ, {"VEDIC_SVD_WIDTH": ""}):
            # 25 quadros em 1024x576 não cabem nos 36 GB do MPS.
            self.assertEqual(diffusion.video_size(), (768, 432))

    def test_video_settings_target_five_seconds(self):
        env = {k: "" for k in ("VEDIC_SVD_FRAMES", "VEDIC_SVD_FPS", "VEDIC_VIDEO_INTERP_FPS", "VEDIC_VIDEO_ZOOM")}
        with patch.dict(os.environ, env):
            cfg = diffusion.video_settings()
        self.assertEqual(cfg["frames"], 25)
        self.assertEqual(cfg["fps"], 5)
        self.assertEqual(cfg["interp_fps"], 24)
        self.assertGreaterEqual(cfg["frames"] / cfg["fps"], 5.0)
        with patch.dict(os.environ, {"VEDIC_SVD_FRAMES": "99", "VEDIC_SVD_FPS": "7", "VEDIC_VIDEO_INTERP_FPS": "0"}):
            cfg = diffusion.video_settings()
        self.assertEqual(cfg["frames"], 25)  # máximo do img2vid-xt
        self.assertEqual(cfg["interp_fps"], 0)
        with patch.dict(os.environ, {"VEDIC_SVD_FPS": "6", "VEDIC_VIDEO_INTERP_FPS": "6", "VEDIC_VIDEO_ZOOM": "x"}):
            cfg = diffusion.video_settings()
        self.assertEqual(cfg["interp_fps"], 0)
        self.assertEqual(cfg["zoom"], 1.06)

    @unittest.skipUnless(find_spec("PIL") is not None, "requer Pillow")
    def test_square_still_is_padded_not_cropped(self):
        from PIL import Image

        # Quadro com uma faixa branca no topo (as chamas): tem de sobreviver.
        square = Image.new("RGB", (512, 512), (40, 40, 40))
        square.paste((255, 255, 255), (0, 0, 512, 20))
        with patch.dict(os.environ, {"VEDIC_SVD_FRAMING": ""}):
            out = diffusion._frame_for_video(square, 768, 432, zoom=1.0)
        self.assertEqual(out.size, (768, 432))
        self.assertGreater(out.getpixel((384, 2))[0], 200)
        with patch.dict(os.environ, {"VEDIC_SVD_FRAMING": "cover"}):
            cropped = diffusion._frame_for_video(square, 768, 432)
        self.assertLess(cropped.getpixel((384, 2))[0], 100)
        # Com zoom, sobra folga em cima para o Ken Burns consumir.
        with patch.dict(os.environ, {"VEDIC_SVD_FRAMING": ""}):
            padded = diffusion._frame_for_video(square, 768, 432, zoom=1.1)
        self.assertLess(padded.getpixel((384, 2))[0], 200)

    @unittest.skipUnless(find_spec("PIL") is not None, "requer Pillow")
    def test_ken_burns_zooms_smoothly_to_output_size(self):
        from PIL import Image

        frames = [Image.new("RGB", (768, 432), (i * 10, 0, 0)) for i in range(5)]
        out = diffusion.ken_burns(frames, (1024, 576), 1.06)
        self.assertEqual(len(out), 5)
        self.assertTrue(all(f.size == (1024, 576) for f in out))
        same = diffusion.ken_burns(frames[:1], (768, 432), 1.0)[0]
        self.assertEqual(same.getpixel((10, 10)), frames[0].getpixel((10, 10)))

    @unittest.skipUnless(find_spec("PIL") is not None, "requer Pillow")
    def test_render_video_runs_svd_then_interpolates_and_encodes(self):
        from PIL import Image

        calls: dict = {}

        class _Pipe:
            def __call__(self, image, **kw):
                calls.update(kw, size=image.size)
                frames = [Image.new("RGB", (kw["width"], kw["height"])) for _ in range(kw["num_frames"])]
                return type("R", (), {"frames": [frames]})()

        encoded: dict = {}
        with tempfile.TemporaryDirectory() as tmp:
            still = Path(tmp) / "s.jpg"
            Image.new("RGB", (64, 64), (200, 50, 0)).save(still)
            env = {k: "" for k in ("VEDIC_SVD_FRAMES", "VEDIC_SVD_FPS", "VEDIC_VIDEO_INTERP_FPS",
                                   "VEDIC_VIDEO_ZOOM", "VEDIC_SVD_WIDTH", "VEDIC_VIDEO_OUT_WIDTH")}
            with patch.dict(os.environ, env), patch.object(diffusion, "ensure_diffusion"), patch.object(
                diffusion, "_video_pipe", return_value=_Pipe()
            ), patch.object(
                diffusion, "interpolate_frames", side_effect=lambda f, a, b: f + f + f + f + f
            ), patch.object(
                diffusion, "encode_mp4", side_effect=lambda f, d, fps: encoded.update(n=len(f), fps=fps, size=f[0].size)
            ):
                diffusion.render_video(still, Path(tmp) / "v.mp4", "gentle flame")
        self.assertEqual(calls["num_frames"], 25)
        self.assertEqual(calls["fps"], 5)
        self.assertEqual(calls["size"], (768, 432))
        self.assertEqual(encoded["fps"], 24)
        self.assertEqual(encoded["size"], (1024, 576))
        self.assertEqual(encoded["n"], 125)

    @unittest.skipUnless(find_spec("PIL") is not None and shutil.which("ffmpeg"), "requer Pillow e ffmpeg")
    def test_interpolation_and_encoding_with_ffmpeg(self):
        from PIL import Image

        frames = [Image.new("RGB", (64, 36), (i * 20, 0, 0)) for i in range(6)]
        out = diffusion.interpolate_frames(frames, 5, 20)
        # 6 quadros a 5 fps = 1,2 s; a 20 fps, 24 quadros.
        self.assertEqual(len(out), 24)
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "v.mp4"
            diffusion.encode_mp4(out, dest, 20)
            self.assertGreater(dest.stat().st_size, 200)

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

    def test_bed_sits_well_below_the_voice(self):
        """Regressão da voz dupla: o drone normalizado ficava ~3 dB abaixo da voz."""
        sr = 44100
        t = np.arange(sr * 2) / sr
        speech = (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
        loud_bed = (0.9 * np.sin(2 * np.pi * 130 * t)).astype(np.float32)
        mixed = diffusion.mix_waveforms(speech, loud_bed)
        resid = mixed - speech * 0.9
        rel = 20 * np.log10(diffusion._rms(resid) / diffusion._rms(speech * 0.9))
        self.assertAlmostEqual(rel, diffusion.DEFAULT_BED_DB, places=3)
        with patch.dict(os.environ, {"VEDIC_AUDIO_BED_DB": "-30"}):
            mixed = diffusion.mix_waveforms(speech, loud_bed)
        rel = 20 * np.log10(diffusion._rms(mixed - speech * 0.9) / diffusion._rms(speech * 0.9))
        self.assertAlmostEqual(rel, -30.0, places=3)

    def test_bed_level_env_is_clamped(self):
        with patch.dict(os.environ, {"VEDIC_AUDIO_BED_DB": "0"}):
            self.assertEqual(diffusion.bed_level_db(), -12.0)
        with patch.dict(os.environ, {"VEDIC_AUDIO_BED_DB": "abc"}):
            self.assertEqual(diffusion.bed_level_db(), diffusion.DEFAULT_BED_DB)
        with patch.dict(os.environ, {"VEDIC_AUDIO_BED_DB": ""}):
            self.assertEqual(diffusion.bed_level_db(), -22.0)

    def test_soften_bed_drops_the_voice_band(self):
        sr = 44100
        t = np.arange(sr) / sr
        low = np.sin(2 * np.pi * 130 * t)
        high = np.sin(2 * np.pi * 3000 * t)
        out = diffusion.soften_bed((low + high).astype(np.float32), sr)
        spec = np.abs(np.fft.rfft(out))
        freqs = np.fft.rfftfreq(out.size, 1 / sr)
        at = lambda f: spec[np.argmin(np.abs(freqs - f))]  # noqa: E731
        self.assertLess(at(3000), at(130) * 0.01)
        self.assertEqual(out[0], 0.0)  # fade-in

    def test_silent_bed_leaves_the_voice_alone(self):
        speech = np.array([0.2, -0.2, 0.1], dtype=np.float32)
        out = diffusion.mix_waveforms(speech, np.zeros(3, dtype=np.float32))
        np.testing.assert_allclose(out, speech * 0.9, rtol=1e-6)

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
            "vedic_pipeline.llm.diffusion.image_backend", return_value="diffusion"
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
            "vedic_pipeline.llm.diffusion.image_backend", return_value="diffusion"
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
            "vedic_pipeline.llm.diffusion.image_backend", return_value="xai"
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


class QualityModeTests(unittest.TestCase):
    def test_image_backend_overrides_media_backend(self):
        with patch.dict(os.environ, {"VEDIC_IMAGE_BACKEND": "xai", "VEDIC_MEDIA_BACKEND": "diffusion"}):
            self.assertEqual(diffusion.image_backend(), "xai")
            self.assertEqual(diffusion.media_backend(), "diffusion")
        with patch.dict(os.environ, {"VEDIC_IMAGE_BACKEND": "", "VEDIC_MEDIA_BACKEND": "diffusion"}):
            self.assertEqual(diffusion.image_backend(), "diffusion")
        with patch.dict(os.environ, {"VEDIC_IMAGE_BACKEND": "bogus", "VEDIC_MEDIA_BACKEND": "xai"}):
            self.assertEqual(diffusion.image_backend(), "xai")

    def test_full_models_default_to_1024_and_30_steps(self):
        with patch.dict(os.environ, {"VEDIC_SD_SIZE": "", "VEDIC_SD_STEPS": "", "VEDIC_SD_GUIDANCE": "6.5"}):
            self.assertEqual(diffusion.image_size("stabilityai/stable-diffusion-xl-base-1.0"), 1024)
            self.assertEqual(diffusion.image_size("stabilityai/sdxl-turbo"), 512)
            self.assertEqual(diffusion.sampling_for("stabilityai/stable-diffusion-xl-base-1.0"), (30, 6.5))
            self.assertEqual(diffusion.sampling_for("stabilityai/sdxl-turbo")[0], 4)
        with patch.dict(os.environ, {"VEDIC_SD_SIZE": "4096"}):
            self.assertEqual(diffusion.image_size("x/y"), 1024)

    def test_refiner_and_upscale_are_opt_in(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            os.environ, {"HF_HOME": tmp, "HF_HUB_OFFLINE": "1", "VEDIC_SD_REFINER": "stabilityai/stable-diffusion-xl-refiner-1.0"}
        ):
            self.assertIsNone(diffusion.refiner_model())
        with patch.dict(os.environ, {"HF_HUB_OFFLINE": "0", "VEDIC_SD_REFINER": "off"}):
            self.assertIsNone(diffusion.refiner_model())
        with patch.dict(os.environ, {"HF_HUB_OFFLINE": "0", "VEDIC_SD_REFINER": "org/refiner"}):
            self.assertEqual(diffusion.refiner_model(), "org/refiner")
        with patch.dict(os.environ, {"VEDIC_SD_UPSCALE": ""}):
            self.assertEqual(diffusion.upscale_factor(), 1.0)
        with patch.dict(os.environ, {"VEDIC_SD_UPSCALE": "9"}):
            self.assertEqual(diffusion.upscale_factor(), 2.0)
        with patch.dict(os.environ, {"VEDIC_SD_REFINER_SPLIT": "abc"}):
            self.assertEqual(diffusion.refiner_split(), 0.8)

    def test_xai_prompt_carries_the_negative_and_bans_lettering(self):
        text = imagine.xai_image_prompt("Agni, two heads.", "blue skin, single head")
        self.assertIn("Avoid: blue skin, single head.", text)
        self.assertIn("No text, captions or lettering", text)
        already = imagine.xai_image_prompt("Scene, no Latin or Devanagari lettering in the frame.")
        self.assertEqual(already.count("lettering"), 1)

    def test_xai_figure_uses_the_quality_model(self):
        client = _FakeClient()
        with tempfile.TemporaryDirectory() as tmp, patch.object(imagine, "MEDIA_DIR", Path(tmp)), patch(
            "vedic_pipeline.llm.diffusion.image_backend", return_value="xai"
        ), patch.object(imagine, "_client", return_value=client), patch.dict(os.environ, {"XAI_IMAGE_MODEL": ""}):
            path = imagine.generate_figure_image("agni", "Agni, two heads", negative="blue skin")
            self.assertTrue(path.exists())
            body = client.posts[0][1]
            self.assertEqual(body["model"], "grok-imagine-image-quality")
            self.assertEqual(body["response_format"], "b64_json")
            self.assertIn("Avoid: blue skin.", body["prompt"])

    def test_xai_failure_falls_back_to_fast_turbo(self):
        bundle = {"locator": "RV 1.1.1", "witnesses": [{"role": "en", "text": "I Laud Agni"}]}
        with tempfile.TemporaryDirectory() as tmp, patch.object(imagine, "MEDIA_DIR", Path(tmp)), patch(
            "vedic_pipeline.llm.diffusion.image_backend", return_value="xai"
        ), patch("vedic_pipeline.llm.diffusion.diffusion_installed", return_value=True), patch.object(
            imagine, "_client", side_effect=RuntimeError("XAI_API_KEY não definida")
        ), patch("vedic_pipeline.llm.diffusion.render_image", return_value=b"j" * 2000) as render, patch.object(
            imagine, "get_verse", return_value=bundle
        ):
            path = imagine.generate_verse_image("RV.1.1.1")
            self.assertTrue(path.exists())
            self.assertTrue(render.call_args.kwargs.get("fast"))
            self.assertTrue(render.call_args.args[0].startswith(imagine.COMPACT_STYLE))

    def test_xai_failure_without_media_extra_raises(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(imagine, "MEDIA_DIR", Path(tmp)), patch(
            "vedic_pipeline.llm.diffusion.image_backend", return_value="xai"
        ), patch("vedic_pipeline.llm.diffusion.diffusion_installed", return_value=False), patch.object(
            imagine, "_client", side_effect=RuntimeError("XAI_API_KEY não definida")
        ), self.assertRaises(RuntimeError):
            imagine.generate_figure_image("agni", "Agni")


if __name__ == "__main__":
    unittest.main()
