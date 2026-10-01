"""Mídia local com a família Stable Diffusion.

- Imagem: Stable Diffusion (default `stabilityai/sd-turbo`). Com um still
  já em disco e `force=True`, img2img refina a cena em vez de recomeçar.
- Vídeo: Stable Video Diffusion, a partir desse still, com pouco movimento.
- Áudio: Stable Audio Open gera só um drone de tanpura. O Stable Diffusion
  não recita sânscrito; a fala continua no TTS e o drone entra por baixo.

Os pesos baixam do Hugging Face na primeira geração. Com `HF_HUB_OFFLINE=1`
isso falha até o cache existir.
"""

from __future__ import annotations

import io
import logging
import os
import shutil
import subprocess
import threading
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger("vedic_pipeline.llm.diffusion")

DEFAULT_SD_MODEL = "stabilityai/sd-turbo"
DEFAULT_SVD_MODEL = "stabilityai/stable-video-diffusion-img2vid-xt"
DEFAULT_AUDIO_MODEL = "stabilityai/stable-audio-open-1.0"

_INSTALL_HINT = 'Stable Diffusion não está instalado. Rode: pip install -e ".[media]"'

_PIPES: dict[str, Any] = {}
_PIPE_LOCK = threading.Lock()


def diffusion_installed() -> bool:
    try:
        import diffusers  # noqa: F401
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


def media_backend() -> str:
    """`diffusion` (local) ou `xai` (Imagine). `auto` prefere o extra instalado."""
    choice = (os.environ.get("VEDIC_MEDIA_BACKEND") or "auto").strip().lower()
    if choice == "xai":
        return "xai"
    if choice == "diffusion":
        return "diffusion"
    if choice not in {"", "auto"}:
        logger.warning("VEDIC_MEDIA_BACKEND=%r desconhecido; usando auto", choice)
    return "diffusion" if diffusion_installed() else "xai"


def audio_bed_enabled() -> bool:
    """Drone de tanpura por baixo do TTS. `auto` liga só com o backend diffusion."""
    flag = (os.environ.get("VEDIC_AUDIO_BED") or "auto").strip().lower()
    if flag in {"0", "false", "off", "no"}:
        return False
    if flag in {"1", "true", "on", "yes", "diffusion"}:
        return diffusion_installed()
    return media_backend() == "diffusion" and diffusion_installed()


def ensure_diffusion() -> None:
    if not diffusion_installed():
        raise RuntimeError(_INSTALL_HINT)


def negative_prompt() -> str:
    return (
        "text, letters, caption, watermark, logo, signature, latin alphabet, "
        "devanagari script, deformed hands, extra fingers, modern clothing, "
        "camera, photograph of a phone, cartoon, lowres, blurry, frame"
    )


def sampling_for(model_name: str) -> tuple[int, float]:
    """SD-Turbo exige guidance 0 e poucos passos. Os outros modelos usam CFG."""
    steps = int(os.environ.get("VEDIC_SD_STEPS", "4"))
    steps = max(1, steps)
    if "turbo" in (model_name or "").lower():
        return min(steps, 4), 0.0
    guidance = float(os.environ.get("VEDIC_SD_GUIDANCE", "7.0"))
    return steps, guidance


def motion_bucket(prompt: str) -> int:
    """SVD: bucket baixo = movimento lento. O prompt do verso pede isso."""
    raw = (os.environ.get("VEDIC_SVD_MOTION") or "").strip()
    if raw.isdigit():
        return int(raw)
    text = (prompt or "").lower()
    if any(word in text for word in ("gentle", "slow", "reverent", "suave")):
        return 30
    return 90


def audio_bed_prompt(_text: str = "") -> str:
    """O modelo de áudio não recebe o mantra: ele tentaria cantar o texto."""
    return (
        "Solo tanpura drone in a quiet temple, four sympathetic strings, "
        "slow meditative pulse, warm and soft, no vocals, no percussion, no melody"
    )


def mix_waveforms(
    speech: np.ndarray,
    bed: np.ndarray,
    *,
    bed_gain: float = 0.18,
) -> np.ndarray:
    """Soma a voz com o drone, repetindo o drone se ele for mais curto."""
    voice = np.asarray(speech, dtype=np.float32).reshape(-1)
    drone = np.asarray(bed, dtype=np.float32).reshape(-1)
    if voice.size == 0 or drone.size == 0:
        return voice
    if drone.size < voice.size:
        reps = int(np.ceil(voice.size / drone.size))
        drone = np.tile(drone, reps)[: voice.size]
    else:
        drone = drone[: voice.size]
    mixed = voice * 0.9 + drone * bed_gain
    peak = float(np.max(np.abs(mixed))) if mixed.size else 0.0
    if peak > 0.98:
        mixed = mixed * (0.98 / peak)
    return mixed.astype(np.float32)


def _resample(samples: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr or samples.size == 0:
        return samples.astype(np.float32)
    length = int(len(samples) * dst_sr / src_sr)
    if length <= 1:
        return samples.astype(np.float32)
    idx = np.arange(length, dtype=np.float64) * (src_sr / dst_sr)
    i0 = np.floor(idx).astype(np.int64)
    i1 = np.minimum(i0 + 1, len(samples) - 1)
    frac = (idx - i0).astype(np.float32)
    base = samples.astype(np.float32)
    return (base[i0] * (1.0 - frac) + base[i1] * frac).astype(np.float32)


def _to_mono(audio: Any) -> np.ndarray:
    if hasattr(audio, "detach"):
        audio = audio.detach().float().cpu().numpy()
    arr = np.asarray(audio, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[0]
    if arr.ndim == 2:
        arr = arr.mean(axis=0)
    return arr.reshape(-1)


def _device_and_dtype() -> tuple[Any, Any]:
    import torch

    from vedic_pipeline.train.device import torch_device

    device = torch_device()
    raw = (os.environ.get("VEDIC_SD_DTYPE") or "").strip().lower()
    if raw in {"float16", "fp16", "half"}:
        dtype = torch.float16
    elif raw in {"float32", "fp32"} or device.type == "cpu":
        dtype = torch.float32
    else:
        # SDXL em fp32 não cabe na GPU da Apple. CUDA e MPS usam fp16.
        dtype = torch.float16
    return device, dtype


def _place(pipe: Any) -> Any:
    """Move o pipeline e deixa o VAE em fp32 para o still não sair preto."""
    import torch

    device, dtype = _device_and_dtype()
    pipe = pipe.to(device)
    vae = getattr(pipe, "vae", None)
    if dtype == torch.float16 and vae is not None:
        pipe.vae.to(dtype=torch.float32)
    return pipe


def _from_pretrained(cls: Any, model_id: str, **kwargs: Any) -> Any:
    try:
        pipe = cls.from_pretrained(model_id, **kwargs)
    except OSError as exc:
        raise RuntimeError(
            f"Não foi possível carregar {model_id}. Na primeira vez deixe "
            "HF_HUB_OFFLINE desligado para baixar os pesos."
        ) from exc
    if hasattr(pipe, "set_progress_bar_config"):
        pipe.set_progress_bar_config(disable=True)
    if hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing()
    if hasattr(pipe, "enable_vae_slicing"):
        pipe.enable_vae_slicing()
    return pipe


def _cached_pipe(key: str, factory: Any) -> Any:
    """O chamador já segura `_PIPE_LOCK` (MPS não aguenta dois modelos em paralelo)."""
    pipe = _PIPES.get(key)
    if pipe is None:
        pipe = factory()
        _PIPES[key] = pipe
    return pipe


def _image_pipes() -> tuple[Any, Any]:
    def load() -> tuple[Any, Any]:
        from diffusers import AutoPipelineForImage2Image, AutoPipelineForText2Image

        model = os.environ.get("VEDIC_SD_MODEL", DEFAULT_SD_MODEL)
        _device, dtype = _device_and_dtype()
        txt = _place(_from_pretrained(AutoPipelineForText2Image, model, torch_dtype=dtype))
        img = AutoPipelineForImage2Image.from_pipe(txt)
        return txt, img

    return _cached_pipe("image", load)


def render_image(
    prompt: str,
    *,
    negative: str | None = None,
    init_image: Path | None = None,
) -> bytes:
    """JPEG. `init_image` refina o still existente (img2img)."""
    ensure_diffusion()
    from PIL import Image

    model = os.environ.get("VEDIC_SD_MODEL", DEFAULT_SD_MODEL)
    steps, guidance = sampling_for(model)
    size = int(os.environ.get("VEDIC_SD_SIZE", "512"))
    size = max(256, min(size, 1024))
    kwargs: dict[str, Any] = {
        "prompt": prompt,
        "num_inference_steps": steps,
        "guidance_scale": guidance,
    }
    if guidance > 0:
        kwargs["negative_prompt"] = negative or negative_prompt()
    with _PIPE_LOCK:
        txt, img = _image_pipes()
        if init_image is not None:
            strength = float(os.environ.get("VEDIC_SD_STRENGTH", "0.35"))
            init = Image.open(init_image).convert("RGB").resize((size, size))
            frame = img(image=init, strength=strength, **kwargs).images[0]
        else:
            frame = txt(height=size, width=size, **kwargs).images[0]
        buf = io.BytesIO()
        frame.save(buf, format="JPEG", quality=90)
        return buf.getvalue()


def _video_pipe() -> Any:
    def load() -> Any:
        from diffusers import StableVideoDiffusionPipeline

        model = os.environ.get("VEDIC_SVD_MODEL", DEFAULT_SVD_MODEL)
        _device, dtype = _device_and_dtype()
        return _place(_from_pretrained(StableVideoDiffusionPipeline, model, torch_dtype=dtype))

    return _cached_pipe("video", load)


def render_video(still: Path, dest: Path, motion: str) -> None:
    """MP4 curto a partir do still. Movimento sai do texto de `motion`."""
    ensure_diffusion()
    from diffusers.utils import export_to_video
    from PIL import Image

    image = Image.open(still).convert("RGB").resize((1024, 576))
    frames_n = int(os.environ.get("VEDIC_SVD_FRAMES", "14"))
    fps = int(os.environ.get("VEDIC_SVD_FPS", "7"))
    with _PIPE_LOCK:
        result = _video_pipe()(
            image,
            num_frames=max(8, frames_n),
            decode_chunk_size=4,
            motion_bucket_id=motion_bucket(motion),
            fps=fps,
            noise_aug_strength=0.02,
        )
        dest.parent.mkdir(parents=True, exist_ok=True)
        export_to_video(result.frames[0], str(dest), fps=fps)


def _audio_sample_rate(pipe: Any) -> int:
    vae = getattr(pipe, "vae", None)
    config = getattr(vae, "config", None)
    rate = getattr(config, "sampling_rate", None)
    if isinstance(rate, int) and rate > 0:
        return rate
    return 44100


def _audio_pipe() -> Any:
    def load() -> Any:
        from diffusers import StableAudioPipeline

        model = os.environ.get("VEDIC_AUDIO_MODEL", DEFAULT_AUDIO_MODEL)
        _device, dtype = _device_and_dtype()
        return _place(_from_pretrained(StableAudioPipeline, model, torch_dtype=dtype))

    return _cached_pipe("audio", load)


def render_audio_bed(prompt: str, seconds: float = 12.0) -> tuple[np.ndarray, int]:
    """Drone sem voz. Devolve mono float32 e a taxa de amostragem."""
    ensure_diffusion()
    length = max(1.0, min(float(seconds), 47.0))
    steps = int(os.environ.get("VEDIC_AUDIO_STEPS", "80"))
    with _PIPE_LOCK:
        pipe = _audio_pipe()
        out = pipe(
            prompt,
            negative_prompt="vocals, singing, speech, words, chanting, drums, percussion",
            num_inference_steps=max(8, steps),
            audio_end_in_s=length,
            num_waveforms_per_prompt=1,
        )
        return _to_mono(out.audios), _audio_sample_rate(pipe)


def _ffmpeg() -> str | None:
    return shutil.which("ffmpeg")


def _decode_mp3(data: bytes) -> tuple[np.ndarray, int]:
    ffmpeg = _ffmpeg()
    if not ffmpeg:
        raise RuntimeError("ffmpeg ausente")
    proc = subprocess.run(
        [
            ffmpeg, "-hide_banner", "-loglevel", "error",
            "-i", "pipe:0", "-f", "f32le", "-ac", "1", "-ar", "44100", "pipe:1",
        ],
        input=data,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0 or not proc.stdout:
        raise RuntimeError("ffmpeg não decodificou a recitação")
    pcm = np.frombuffer(proc.stdout, dtype=np.float32).copy()
    return pcm, 44100


def _encode_mp3(samples: np.ndarray, sample_rate: int) -> bytes:
    ffmpeg = _ffmpeg()
    if not ffmpeg:
        raise RuntimeError("ffmpeg ausente")
    proc = subprocess.run(
        [
            ffmpeg, "-hide_banner", "-loglevel", "error",
            "-f", "f32le", "-ar", str(sample_rate), "-ac", "1", "-i", "pipe:0",
            "-f", "mp3", "pipe:1",
        ],
        input=np.asarray(samples, dtype=np.float32).tobytes(),
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0 or not proc.stdout:
        raise RuntimeError("ffmpeg não codificou a recitação")
    return proc.stdout


def underlay_speech(speech: bytes, prompt: str) -> bytes:
    """Mistura o MP3 do TTS com o drone. Sem ffmpeg, devolve a voz intacta."""
    if not _ffmpeg():
        logger.warning("ffmpeg ausente; recitação fica sem a cama de tanpura")
        return speech
    voice, voice_sr = _decode_mp3(speech)
    seconds = max(4.0, min(47.0, (len(voice) / voice_sr) + 0.5)) if voice_sr else 12.0
    bed, bed_sr = render_audio_bed(prompt, seconds)
    bed = _resample(bed, bed_sr, voice_sr)
    return _encode_mp3(mix_waveforms(voice, bed), voice_sr)
