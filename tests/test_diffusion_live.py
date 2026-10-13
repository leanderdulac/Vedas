"""Geração real com o Stable Diffusion já presente na máquina.

Não roda no CI. Os pesos de vídeo (SVD) e do drone (Stable Audio) não
fazem parte deste teste: eles não estão instalados aqui.

    VEDIC_DIFFUSION_LIVE=1 VEDIC_DEVICE=mps \\
      VEDIC_SD_MODEL=$HOME/models/sd/realvisxl-v4l-diffusers \\
      .venv/bin/python -m unittest tests.test_diffusion_live
"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from vedic_pipeline.llm import diffusion


def _live() -> bool:
    return (os.environ.get("VEDIC_DIFFUSION_LIVE") or "").strip() in {"1", "true", "on", "yes"}


def _image_model() -> str | None:
    explicit = (os.environ.get("VEDIC_SD_MODEL") or "").strip()
    if explicit and Path(explicit).joinpath("model_index.json").exists():
        return explicit
    for candidate in (
        Path.home() / "models/sd/realvisxl-v4l-diffusers",
        Path.home() / "Models/sd/realvisxl-v4l-diffusers",
    ):
        unet = candidate / "unet"
        if (candidate / "model_index.json").exists() and any(unet.glob("*.safetensors")):
            return str(candidate)
    return None


def _weights_ready(model_id: str) -> bool:
    path = Path(model_id)
    if path.is_dir():
        return any(file.stat().st_size > 50_000_000 for file in path.rglob("*.safetensors"))
    return False


@unittest.skipUnless(_live(), "defina VEDIC_DIFFUSION_LIVE=1 para gerar mídia de verdade")
class LiveDiffusionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        model = _image_model()
        if not model or not diffusion.diffusion_installed():
            raise unittest.SkipTest("Stable Diffusion local não encontrado")
        os.environ["VEDIC_SD_MODEL"] = model
        os.environ.setdefault("VEDIC_DEVICE", "mps")
        os.environ.setdefault("VEDIC_SD_STEPS", "4")
        os.environ.setdefault("VEDIC_SD_GUIDANCE", "2")
        os.environ.setdefault("VEDIC_SD_SIZE", "768")
        cls.model = model

    def test_text_to_image_then_refine(self):
        prompt = (
            "A brass oil lamp burning in a dark Vedic shrine, quiet flame, "
            "mineral reds and gold, no text, no letters"
        )
        first = diffusion.render_image(prompt)
        self.assertTrue(first.startswith(b"\xff\xd8"))
        self.assertGreater(len(first), 8_000)
        with tempfile.TemporaryDirectory() as tmp:
            still = Path(tmp) / "still.jpg"
            still.write_bytes(first)
            refined = diffusion.render_image(prompt, init_image=still)
        self.assertTrue(refined.startswith(b"\xff\xd8"))
        self.assertGreater(len(refined), 8_000)
        self.assertNotEqual(first, refined)
        pixels = np.frombuffer(first, dtype=np.uint8)
        self.assertGreater(int(pixels.std()), 8)
        out = Path("/tmp/vedas-sd-live")
        out.mkdir(parents=True, exist_ok=True)
        (out / "still.jpg").write_bytes(first)
        (out / "refined.jpg").write_bytes(refined)

    def test_video_when_svd_weights_exist(self):
        svd = os.environ.get("VEDIC_SVD_MODEL", diffusion.DEFAULT_SVD_MODEL)
        if not _weights_ready(svd):
            self.skipTest(f"pesos de vídeo ausentes ({svd})")
        still = Path("/tmp/vedas-sd-live/still.jpg")
        if not still.exists():
            self.skipTest("still da imagem ainda não foi gerado")
        dest = Path("/tmp/vedas-sd-live/motion.mp4")
        diffusion.render_video(still, dest, "Gentle sacred motion, slow flame")
        self.assertGreater(dest.stat().st_size, 1000)

    def test_audio_bed_when_weights_exist(self):
        audio = os.environ.get("VEDIC_AUDIO_MODEL", diffusion.DEFAULT_AUDIO_MODEL)
        if not _weights_ready(audio):
            self.skipTest(f"pesos de áudio ausentes ({audio})")
        samples, rate = diffusion.render_audio_bed(diffusion.audio_bed_prompt(), seconds=2.0)
        self.assertGreater(rate, 0)
        self.assertGreater(samples.size, 100)


if __name__ == "__main__":
    unittest.main()
