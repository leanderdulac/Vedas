"""Mídia local com a família Stable Diffusion.

- Imagem: `VEDIC_IMAGE_BACKEND=xai` manda as figuras e os stills para o
  Grok Imagine (modo qualidade); `diffusion` gera local. Local,
  `VEDIC_SD_MODEL` escolhe o modelo; sem ele, o `stabilityai/sdxl-turbo` se
  estiver no cache, senão o `stabilityai/sd-turbo` (também o fallback offline
  quando o modelo pedido não tem pesos locais). Modelos completos (SDXL base,
  Playground) rodam a 1024 px com CFG cheio e aceitam refiner
  (`VEDIC_SD_REFINER`) e upscale 2x com img2img (`VEDIC_SD_UPSCALE`). O modo
  rápido (`fast=True`) é sempre o turbo a 512 px: é o fallback quando o xAI
  falha. Com um still já em disco e `force=True`, img2img refina a cena do
  verso em vez de recomeçar.
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
import time
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger("vedic_pipeline.llm.diffusion")

DEFAULT_SD_MODEL = "stabilityai/sd-turbo"
# Preferido quando já está no cache: desenha a iconografia (montaria, chamas,
# concha) muito melhor que o sd-turbo, no mesmo tempo no MPS (~0,7 s a 512 px).
PREFERRED_SD_MODEL = "stabilityai/sdxl-turbo"
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


def image_backend() -> str:
    """Backend só das imagens: `VEDIC_IMAGE_BACKEND`, senão o de mídia.

    Permite stills e figuras no Grok Imagine (melhor iconografia) com o vídeo
    e o drone ainda locais.
    """
    choice = (os.environ.get("VEDIC_IMAGE_BACKEND") or "").strip().lower()
    if choice in {"xai", "diffusion"}:
        return choice
    if choice not in {"", "auto"}:
        logger.warning("VEDIC_IMAGE_BACKEND=%r desconhecido; usando VEDIC_MEDIA_BACKEND", choice)
    return media_backend()


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


def image_model() -> str:
    """Modelo de imagem: `VEDIC_SD_MODEL`, senão o SDXL-Turbo se já estiver no
    cache, senão o sd-turbo.

    Offline (`HF_HUB_OFFLINE=1`) um id sem snapshot local só falharia na carga;
    o fallback mantém as figuras funcionando até o download.
    """
    model = (os.environ.get("VEDIC_SD_MODEL") or "").strip()
    if not model:
        return PREFERRED_SD_MODEL if _cached_snapshot(PREFERRED_SD_MODEL) else DEFAULT_SD_MODEL
    if model == DEFAULT_SD_MODEL:
        return model
    offline = (os.environ.get("HF_HUB_OFFLINE") or "").strip().lower() in {"1", "true", "yes", "on"}
    if offline and _cached_snapshot(model) is None:
        logger.warning("VEDIC_SD_MODEL=%s não está no cache offline; usando %s", model, DEFAULT_SD_MODEL)
        return DEFAULT_SD_MODEL
    return model


def is_turbo(model_name: str) -> bool:
    return "turbo" in (model_name or "").lower()


def fast_image_model() -> str:
    """Modelo do modo rápido: SDXL-Turbo se estiver no cache, senão o sd-turbo."""
    return PREFERRED_SD_MODEL if _cached_snapshot(PREFERRED_SD_MODEL) else DEFAULT_SD_MODEL


def image_size(model_name: str) -> int:
    """`VEDIC_SD_SIZE`; sem ele, 512 no turbo (resolução de treino) e 1024 no resto."""
    raw = (os.environ.get("VEDIC_SD_SIZE") or "").strip()
    size = int(raw) if raw.isdigit() else (512 if is_turbo(model_name) else 1024)
    return max(256, min(size, 1024)) // 8 * 8


def sampling_for(model_name: str) -> tuple[int, float]:
    """Turbo: poucos passos e um CFG leve (1,5) para o prompt negativo pesar.

    Com guidance 0 o turbo ignora o negativo e o Agni saía de pele azul.
    `VEDIC_SD_TURBO_GUIDANCE=0` volta ao modo sem CFG (metade do custo).
    Os outros modelos usam CFG cheio (`VEDIC_SD_GUIDANCE`, 7) e 30 passos
    por padrão (`VEDIC_SD_STEPS`).
    """
    raw_steps = (os.environ.get("VEDIC_SD_STEPS") or "").strip()
    steps = int(raw_steps) if raw_steps.isdigit() else (4 if is_turbo(model_name) else 30)
    steps = max(1, steps)
    if is_turbo(model_name):
        raw = (os.environ.get("VEDIC_SD_TURBO_GUIDANCE") or "").strip()
        guidance = float(raw) if raw else 1.5
        if guidance > 1.0:
            return min(steps, 8), min(guidance, 3.0)
        return min(steps, 4), 0.0
    guidance = float(os.environ.get("VEDIC_SD_GUIDANCE", "7.0"))
    return steps, guidance


def img2img_steps(steps: int, strength: float) -> int:
    """img2img roda `steps * strength` passos; abaixo de 1 o diffusers quebra."""
    import math

    strength = max(0.05, min(strength, 1.0))
    return max(steps, math.ceil(1.0 / strength))


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


DEFAULT_BED_DB = -22.0
BED_LOWPASS_HZ = 1200.0


def bed_level_db() -> float:
    """Nível RMS do drone relativo à voz. `VEDIC_AUDIO_BED_DB`, padrão -22 dB.

    Com o drone normalizado e ganho fixo de 0,18, ele ficava só ~3 dB abaixo
    da voz e os harmônicos de 300 Hz a 4 kHz soavam como um segundo narrador.
    """
    raw = (os.environ.get("VEDIC_AUDIO_BED_DB") or "").strip()
    try:
        value = float(raw) if raw else DEFAULT_BED_DB
    except ValueError:
        logger.warning("VEDIC_AUDIO_BED_DB=%r inválido; usando %s", raw, DEFAULT_BED_DB)
        value = DEFAULT_BED_DB
    return max(-40.0, min(value, -12.0))


def _rms(samples: np.ndarray) -> float:
    arr = np.asarray(samples, dtype=np.float64).reshape(-1)
    return float(np.sqrt(np.mean(arr * arr))) if arr.size else 0.0


def soften_bed(samples: np.ndarray, sample_rate: int, *, cutoff_hz: float = BED_LOWPASS_HZ) -> np.ndarray:
    """Passa-baixa suave e fades: tira do drone a faixa dos formantes da voz.

    O Stable Audio deixa harmônicos modulados de 1 a 4 kHz que lembram vogais
    cantadas; abaixo de ~1,2 kHz sobra o zumbido da tanpura.
    """
    arr = np.asarray(samples, dtype=np.float32).reshape(-1)
    if arr.size < 16 or sample_rate <= 0:
        return arr
    spec = np.fft.rfft(arr.astype(np.float64))
    freqs = np.fft.rfftfreq(arr.size, d=1.0 / sample_rate)
    lo, hi = cutoff_hz, cutoff_hz * 2.0
    gain = np.ones_like(freqs)
    band = (freqs > lo) & (freqs < hi)
    gain[band] = 0.5 * (1.0 + np.cos(np.pi * (freqs[band] - lo) / (hi - lo)))
    gain[freqs >= hi] = 0.0
    out = np.fft.irfft(spec * gain, n=arr.size).astype(np.float32)
    fade = min(int(0.4 * sample_rate), out.size // 4)
    if fade > 1:
        ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
        out[:fade] *= ramp
        out[-fade:] *= ramp[::-1]
    return out


def mix_waveforms(
    speech: np.ndarray,
    bed: np.ndarray,
    *,
    bed_gain: float | None = None,
    bed_db: float | None = None,
) -> np.ndarray:
    """Soma a voz com o drone, repetindo o drone se ele for mais curto.

    Sem `bed_gain`, o drone fica `bed_db` (padrão `bed_level_db()`) abaixo da
    voz em RMS, independente do volume que o Stable Audio devolveu.
    """
    voice = np.asarray(speech, dtype=np.float32).reshape(-1)
    drone = np.asarray(bed, dtype=np.float32).reshape(-1)
    if voice.size == 0 or drone.size == 0:
        return voice
    if drone.size < voice.size:
        reps = int(np.ceil(voice.size / drone.size))
        drone = np.tile(drone, reps)[: voice.size]
    else:
        drone = drone[: voice.size]
    voice = voice * 0.9
    if bed_gain is None:
        drone_rms = _rms(drone)
        if drone_rms < 1e-9:
            return voice.astype(np.float32)
        level = bed_level_db() if bed_db is None else float(bed_db)
        bed_gain = _rms(voice) * (10.0 ** (level / 20.0)) / drone_rms
    mixed = voice + drone * float(bed_gain)
    peak = float(np.max(np.abs(mixed))) if mixed.size else 0.0
    if peak > 0.98:
        mixed = mixed * (0.98 / peak)
    return mixed.astype(np.float32)


def normalize_peak(samples: np.ndarray, peak: float = 1.0) -> np.ndarray:
    """O Stable Audio devolve o drone com pico ~0,02 (-34 dBFS).

    Sem normalizar, o ganho de 0,18 da mistura o deixa inaudível sob a voz.
    """
    arr = np.asarray(samples, dtype=np.float32).reshape(-1)
    top = float(np.max(np.abs(arr))) if arr.size else 0.0
    if top < 1e-6:
        return arr
    return (arr * (peak / top)).astype(np.float32)


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
    """Move o pipeline e preserva o dtype de cada módulo.

    SDXL e o SVD sobem o VAE para fp32 só na decodificação, quando
    ``force_upcast`` está ligado, e devolvem o dtype original. Forçar fp32
    aqui deixa o latent em fp16 e o bias do VAE em float: o sd-turbo quebra
    no MPS com ``Input type (c10::Half) and bias type (float)``.
    """
    device, _dtype = _device_and_dtype()
    return pipe.to(device)


def _cached_snapshot(model_id: str) -> str | None:
    """Pasta local do snapshot quando `model_index.json` já está no cache.

    Um snapshot parcial (sem o `model.safetensors` monolítico) faz o Hub
    recusar o id em modo offline. A pasta local carrega só os módulos do índice.
    """
    direct = Path(model_id)
    if direct.is_dir() and (direct / "model_index.json").is_file():
        return str(direct)
    if model_id.count("/") != 1:
        return None
    home = (os.environ.get("HF_HOME") or "").strip()
    if home:
        root = Path(home) / "hub"
    else:
        try:
            from huggingface_hub.constants import HF_HUB_CACHE
        except ImportError:
            return None
        root = Path(HF_HUB_CACHE)
    repo = root / f"models--{model_id.replace('/', '--')}"
    try:
        revision = (repo / "refs" / "main").read_text().strip()
    except OSError:
        return None
    snapshot = repo / "snapshots" / revision
    if (snapshot / "model_index.json").is_file():
        return str(snapshot)
    return None


def _from_pretrained(cls: Any, model_id: str, **kwargs: Any) -> Any:
    """Carrega o pipeline. Em fp16 prefere os arquivos `*.fp16.safetensors`."""
    import torch

    source = _cached_snapshot(model_id) or model_id
    _, dtype = _device_and_dtype()
    requested = kwargs.get("torch_dtype", dtype)
    attempts: list[dict[str, Any]] = []
    if requested == torch.float16 and "variant" not in kwargs:
        attempts.append({"variant": "fp16"})
    attempts.append({})
    last: Exception | None = None
    pipe = None
    for extra in attempts:
        try:
            pipe = cls.from_pretrained(source, **kwargs, **extra)
            break
        except (OSError, ValueError) as exc:
            last = exc
            logger.info("Carga de %s %s falhou (%s); tentando o próximo conjunto de pesos", model_id, extra or "padrão", exc)
    if pipe is None:
        raise RuntimeError(
            f"Não foi possível carregar {model_id}. Na primeira vez deixe "
            "HF_HUB_OFFLINE desligado para baixar os pesos."
        ) from last
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


def _image_pipes(model: str | None = None) -> tuple[Any, Any]:
    model = model or image_model()

    def load() -> tuple[Any, Any]:
        from diffusers import AutoPipelineForImage2Image, AutoPipelineForText2Image

        _device, dtype = _device_and_dtype()
        logger.info("Carregando modelo de imagem %s", model)
        txt = _place(_from_pretrained(AutoPipelineForText2Image, model, torch_dtype=dtype))
        # A 1024 px o VAE decodifica em blocos: evita o pico de memória no MPS.
        vae = getattr(txt, "vae", None)
        if vae is not None and hasattr(vae, "enable_tiling") and not is_turbo(model):
            vae.enable_tiling()
        img = AutoPipelineForImage2Image.from_pipe(txt)
        img.set_progress_bar_config(disable=True)
        return txt, img

    return _cached_pipe(f"image:{model}", load)


def refiner_model() -> str | None:
    """`VEDIC_SD_REFINER` (ex.: stabilityai/stable-diffusion-xl-refiner-1.0).

    Só vale para modelos completos; no turbo e offline sem pesos é ignorado.
    """
    model = (os.environ.get("VEDIC_SD_REFINER") or "").strip()
    if not model or model.lower() in {"0", "off", "none", "false"}:
        return None
    offline = (os.environ.get("HF_HUB_OFFLINE") or "").strip().lower() in {"1", "true", "yes", "on"}
    if offline and _cached_snapshot(model) is None:
        logger.warning("VEDIC_SD_REFINER=%s não está no cache offline; seguindo sem refiner", model)
        return None
    return model


def refiner_split() -> float:
    """Fração do ruído que fica com o base antes do refiner (ensemble of experts)."""
    raw = (os.environ.get("VEDIC_SD_REFINER_SPLIT") or "").strip()
    try:
        value = float(raw) if raw else 0.8
    except ValueError:
        value = 0.8
    return max(0.5, min(value, 0.95))


def upscale_factor() -> float:
    """`VEDIC_SD_UPSCALE=2` amplia o quadro e refaz o detalhe com img2img (hires fix)."""
    raw = (os.environ.get("VEDIC_SD_UPSCALE") or "").strip()
    try:
        value = float(raw) if raw else 1.0
    except ValueError:
        value = 1.0
    return max(1.0, min(value, 2.0))


def _refiner_pipe(model: str, base: Any) -> Any:
    """O refiner reaproveita o segundo text encoder e o VAE do base."""

    def load() -> Any:
        from diffusers import StableDiffusionXLImg2ImgPipeline

        _device, dtype = _device_and_dtype()
        shared: dict[str, Any] = {"torch_dtype": dtype}
        for name in ("text_encoder_2", "vae"):
            module = getattr(base, name, None)
            if module is not None:
                shared[name] = module
        logger.info("Carregando refiner %s", model)
        return _place(_from_pretrained(StableDiffusionXLImg2ImgPipeline, model, **shared))

    return _cached_pipe(f"refiner:{model}", load)


def _warn_if_truncated(pipe: Any, prompt: str) -> None:
    tokenizer = getattr(pipe, "tokenizer", None)
    if tokenizer is None:
        return
    try:
        count = len(tokenizer(prompt).input_ids)
    except Exception:  # noqa: BLE001
        return
    limit = getattr(tokenizer, "model_max_length", 77) or 77
    if count > limit:
        logger.warning("Prompt com %d tokens; o CLIP lê só %d e corta o final", count, limit)


def _generator() -> Any:
    raw = (os.environ.get("VEDIC_SD_SEED") or "").strip()
    if not raw.lstrip("-").isdigit():
        return None
    import torch

    # MPS não tem gerador próprio estável; o ruído sai da CPU.
    return torch.Generator("cpu").manual_seed(int(raw))


def render_image(
    prompt: str,
    *,
    negative: str | None = None,
    init_image: Path | None = None,
    fast: bool = False,
) -> bytes:
    """JPEG. `init_image` refina o still existente (img2img).

    `fast=True` ignora o modo qualidade e usa o turbo a 512 px em 4 passos.
    """
    ensure_diffusion()
    from PIL import Image

    if fast:
        model = fast_image_model()
        steps, guidance, size = 4, 1.5, 512
        refiner, upscale = None, 1.0
    else:
        model = image_model()
        steps, guidance = sampling_for(model)
        size = image_size(model)
        refiner = None if is_turbo(model) else refiner_model()
        upscale = 1.0 if is_turbo(model) else upscale_factor()
    kwargs: dict[str, Any] = {
        "prompt": prompt,
        "num_inference_steps": steps,
        "guidance_scale": guidance,
    }
    if guidance > 1.0:
        kwargs["negative_prompt"] = negative or negative_prompt()
    with _PIPE_LOCK:
        txt, img = _image_pipes(model)
        _warn_if_truncated(txt, prompt)
        generator = _generator()
        if generator is not None:
            kwargs["generator"] = generator
        if init_image is not None:
            strength = float(os.environ.get("VEDIC_SD_STRENGTH", "0.35"))
            init = Image.open(init_image).convert("RGB").resize((size, size))
            frame = img(image=init, strength=strength, **{**kwargs, "num_inference_steps": img2img_steps(steps, strength)}).images[0]
        elif refiner:
            split = refiner_split()
            latents = txt(height=size, width=size, denoising_end=split, output_type="latent", **kwargs).images
            frame = _refiner_pipe(refiner, txt)(image=latents, denoising_start=split, **kwargs).images[0]
        else:
            frame = txt(height=size, width=size, **kwargs).images[0]
        if upscale > 1.0 and init_image is None:
            target = min(2048, int(size * upscale) // 8 * 8)
            strength = float(os.environ.get("VEDIC_SD_UPSCALE_STRENGTH", "0.3"))
            big = frame.resize((target, target), Image.LANCZOS)
            frame = img(image=big, strength=strength, **{**kwargs, "num_inference_steps": img2img_steps(steps, strength)}).images[0]
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


def _env_int(name: str, default: int, lo: int, hi: int) -> int:
    raw = (os.environ.get(name) or "").strip()
    try:
        value = int(float(raw)) if raw else default
    except ValueError:
        logger.warning("%s=%r inválido; usando %s", name, raw, default)
        value = default
    return max(lo, min(value, hi))


def _env_float(name: str, default: float, lo: float, hi: float) -> float:
    raw = (os.environ.get(name) or "").strip()
    try:
        value = float(raw) if raw else default
    except ValueError:
        logger.warning("%s=%r inválido; usando %s", name, raw, default)
        value = default
    return max(lo, min(value, hi))


def video_size() -> tuple[int, int]:
    """Tamanho em que o SVD roda. Padrão 768x432: com 25 quadros, 1024x576
    passa dos 36 GB de memória unificada do MPS. `VEDIC_SVD_WIDTH=1024` volta
    ao tamanho de treino (com 14 quadros cabe em ~26 GB)."""
    raw = (os.environ.get("VEDIC_SVD_WIDTH") or "").strip()
    width = int(raw) if raw.isdigit() else 768
    width = max(512, min(width, 1024)) // 64 * 64
    height = (width * 9 // 16) // 8 * 8
    return width, height


def video_output_size() -> tuple[int, int]:
    """Tamanho final do MP4 (upscale lanczos do SVD). `VEDIC_VIDEO_OUT_WIDTH`."""
    width = _env_int("VEDIC_VIDEO_OUT_WIDTH", 1024, 512, 1920) // 16 * 16
    height = (width * 9 // 16) // 2 * 2
    return width, height


def video_settings() -> dict[str, Any]:
    """Quadros e ritmo do vídeo do verso.

    - `VEDIC_SVD_FRAMES` (25, máx. do img2vid-xt): quadros gerados pelo SVD.
    - `VEDIC_SVD_FPS` (5): ritmo desses quadros; 25 / 5 = 5 s de vídeo.
    - `VEDIC_VIDEO_INTERP_FPS` (24): interpolação por movimento (ffmpeg
      minterpolate) até esse fps; 0 desliga.
    - `VEDIC_VIDEO_ZOOM` (1.06): Ken Burns lento sobre o movimento do SVD;
      1 desliga.
    """
    frames = _env_int("VEDIC_SVD_FRAMES", 25, 8, 25)
    fps = _env_int("VEDIC_SVD_FPS", 5, 2, 30)
    interp = _env_int("VEDIC_VIDEO_INTERP_FPS", 24, 0, 60)
    if interp and interp <= fps:
        interp = 0
    zoom = _env_float("VEDIC_VIDEO_ZOOM", 1.06, 1.0, 1.3)
    return {"frames": frames, "fps": fps, "interp_fps": interp, "zoom": zoom}


def _cover(image: Any, width: int, height: int) -> Any:
    """Recorta ao centro para 16:9 e redimensiona, sem esticar o still quadrado."""
    src_w, src_h = image.size
    target = width / height
    if src_w / src_h > target:
        crop_w = int(src_h * target)
        left = (src_w - crop_w) // 2
        box = (left, 0, left + crop_w, src_h)
    else:
        crop_h = int(src_w / target)
        # Retrato: mantém mais o alto do quadro, onde costumam estar rosto e chamas.
        top = max(0, (src_h - crop_h) // 3)
        box = (0, top, src_w, top + crop_h)
    return image.crop(box).resize((width, height))


def _pad_frame(image: Any, width: int, height: int, *, margin: float = 1.0) -> Any:
    """Still inteiro no centro, laterais com o próprio still desfocado.

    O still do Imagine é quadrado: recortar para 16:9 cortava o alto das
    chamas e o carneiro. `margin` > 1 deixa uma folga em cima e embaixo que o
    zoom lento consome sem cortar nada.
    """
    from PIL import ImageFilter

    src_w, src_h = image.size
    bg = _cover(image, width, height).filter(ImageFilter.GaussianBlur(radius=max(8, width // 40)))
    bg = bg.point(lambda v: int(v * 0.55))
    scale = min(width / src_w, height / (src_h * margin))
    fg = image.resize((max(1, round(src_w * scale)), max(1, round(src_h * scale))))
    left = (width - fg.width) // 2
    top = (height - fg.height) // 2
    bg.paste(fg, (left, top))
    return bg


def _frame_for_video(image: Any, width: int, height: int, zoom: float = 1.0) -> Any:
    """`VEDIC_SVD_FRAMING`: `pad` (padrão, still inteiro) ou `cover` (recorte)."""
    mode = (os.environ.get("VEDIC_SVD_FRAMING") or "pad").strip().lower()
    src_w, src_h = image.size
    if mode == "cover" or src_w / src_h >= width / height:
        return _cover(image, width, height)
    return _pad_frame(image, width, height, margin=zoom)


def ken_burns(frames: list[Any], size: tuple[int, int], zoom: float) -> list[Any]:
    """Zoom lento e contínuo (sub-pixel) e redimensiona para `size`."""
    from PIL import Image

    out_w, out_h = size
    n = len(frames)
    result = []
    for i, frame in enumerate(frames):
        t = i / (n - 1) if n > 1 else 0.0
        ease = 0.5 - 0.5 * np.cos(np.pi * t)
        z = 1.0 + (zoom - 1.0) * ease
        src_w, src_h = frame.size
        # Mapa saída -> entrada: janela central de (src/z), um pouco acima do centro.
        sx = src_w / (out_w * z)
        sy = src_h / (out_h * z)
        cx = src_w / 2.0
        cy = src_h * 0.48
        ox = cx - (out_w * sx) / 2.0
        oy = min(max(cy - (out_h * sy) / 2.0, 0.0), src_h - out_h * sy)
        result.append(
            frame.transform((out_w, out_h), Image.AFFINE, (sx, 0.0, ox, 0.0, sy, oy), resample=Image.BICUBIC)
        )
    return result


def interpolate_frames(frames: list[Any], src_fps: int, dst_fps: int) -> list[Any]:
    """Interpolação por movimento (minterpolate). Sem ffmpeg, devolve os quadros."""
    from PIL import Image

    ffmpeg = _ffmpeg()
    if not ffmpeg or not frames or dst_fps <= src_fps:
        return frames
    w, h = frames[0].size
    raw = b"".join(f.convert("RGB").tobytes() for f in frames)
    # Sem o tpad o minterpolate para no último quadro e perde ~0,4 s;
    # o trim fixa a duração em quadros / fps (25 / 5 = 5 s).
    duration = len(frames) / src_fps
    vf = (
        f"tpad=stop_mode=clone:stop_duration={2 / src_fps:.3f},"
        f"minterpolate=fps={dst_fps}:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1,"
        f"trim=duration={duration:.3f}"
    )
    proc = subprocess.run(
        [
            ffmpeg, "-hide_banner", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(src_fps), "-i", "pipe:0",
            "-vf", vf, "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
        ],
        input=raw,
        capture_output=True,
        check=False,
    )
    size = w * h * 3
    if proc.returncode != 0 or len(proc.stdout) < size:
        logger.warning("minterpolate falhou; vídeo fica a %s fps", src_fps)
        return frames
    data = proc.stdout
    return [Image.frombytes("RGB", (w, h), data[i : i + size]) for i in range(0, len(data) - size + 1, size)]


def encode_mp4(frames: list[Any], dest: Path, fps: int) -> None:
    """H.264 yuv420p com faststart: toca no Safari e no Chrome."""
    ffmpeg = _ffmpeg()
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.stem + ".tmp.mp4")
    if not ffmpeg:
        from diffusers.utils import export_to_video

        export_to_video(frames, str(tmp), fps=fps)
        tmp.replace(dest)
        return
    w, h = frames[0].size
    proc = subprocess.run(
        [
            ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps), "-i", "pipe:0",
            "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", str(tmp),
        ],
        input=b"".join(f.convert("RGB").tobytes() for f in frames),
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0 or not tmp.exists():
        raise RuntimeError(f"ffmpeg não codificou o vídeo: {proc.stderr.decode(errors='ignore')[-300:]}")
    tmp.replace(dest)


def render_video(still: Path, dest: Path, motion: str) -> None:
    """MP4 de ~5 s a partir do still: SVD (25 quadros a 5 fps), interpolação
    até 24 fps e um zoom lento. O still entra inteiro (laterais desfocadas)."""
    ensure_diffusion()
    from PIL import Image

    started = time.monotonic()
    cfg = video_settings()
    width, height = video_size()
    image = _frame_for_video(Image.open(still).convert("RGB"), width, height, cfg["zoom"])
    with _PIPE_LOCK:
        result = _video_pipe()(
            image,
            width=width,
            height=height,
            num_frames=cfg["frames"],
            decode_chunk_size=4,
            motion_bucket_id=motion_bucket(motion),
            fps=cfg["fps"],
            noise_aug_strength=0.02,
        )
        frames = list(result.frames[0])
        _empty_device_cache()
    out_fps = cfg["fps"]
    if cfg["interp_fps"]:
        frames = interpolate_frames(frames, cfg["fps"], cfg["interp_fps"])
        if len(frames) > cfg["frames"]:
            out_fps = cfg["interp_fps"]
    frames = ken_burns(frames, video_output_size(), cfg["zoom"])
    encode_mp4(frames, dest, out_fps)
    logger.info(
        "Vídeo %s: SVD %dx%d, %d quadros a %d fps -> %d quadros a %d fps (%.1f s) em %.0f s",
        dest.name, width, height, cfg["frames"], cfg["fps"], len(frames), out_fps,
        len(frames) / out_fps, time.monotonic() - started,
    )


def _audio_sample_rate(pipe: Any) -> int:
    vae = getattr(pipe, "vae", None)
    config = getattr(vae, "config", None)
    rate = getattr(config, "sampling_rate", None)
    if isinstance(rate, int) and rate > 0:
        return rate
    return 44100


def _use_cpu_brownian() -> None:
    """O ruído SDE do Stable Audio quebra no MPS e no sigma final.

    A árvore browniana recursa sem fim no MPS, e também quando o último sigma
    sai do intervalo (0). Com o sigma final igual ao mínimo, o passo tem
    duração zero e a divisão por ``sqrt(dt)`` vira NaN.
    """
    import torch
    from diffusers.schedulers import scheduling_dpmsolver_sde as sde

    if not getattr(sde.BatchedBrownianTree, "_vedic_cpu", False):
        original_init = sde.BatchedBrownianTree.__init__

        def _init(self, x, t0, t1, seed=None, **kwargs):
            original_init(self, x.detach().to("cpu"), t0, t1, seed, **kwargs)

        sde.BatchedBrownianTree.__init__ = _init  # type: ignore[method-assign]
        sde.BatchedBrownianTree._vedic_cpu = True

    if getattr(sde.BrownianTreeNoiseSampler, "_vedic_zero_step", False):
        return
    original_call = sde.BrownianTreeNoiseSampler.__call__

    def _call(self, sigma, sigma_next):
        t0 = self.transform(torch.as_tensor(sigma))
        t1 = self.transform(torch.as_tensor(sigma_next))
        if float((t1 - t0).abs()) == 0.0:
            return self.tree(t0, t0)
        return original_call(self, sigma, sigma_next)

    sde.BrownianTreeNoiseSampler.__call__ = _call  # type: ignore[method-assign]
    sde.BrownianTreeNoiseSampler._vedic_zero_step = True


def _audio_scheduler(scheduler: Any) -> Any:
    """O último sigma 0 fica fora da árvore browniana e a recursão não termina."""
    if type(scheduler).__name__ != "CosineDPMSolverMultistepScheduler":
        return scheduler
    from diffusers import CosineDPMSolverMultistepScheduler

    return CosineDPMSolverMultistepScheduler.from_config(
        scheduler.config,
        final_sigmas_type="sigma_min",
    )


def _audio_pipe() -> Any:
    def load() -> Any:
        import torch
        from diffusers import StableAudioPipeline

        model = os.environ.get("VEDIC_AUDIO_MODEL", DEFAULT_AUDIO_MODEL)
        # fp16 no MPS faz o BrownianInterval do scheduler recursar sem parar.
        pipe = _place(_from_pretrained(StableAudioPipeline, model, torch_dtype=torch.float32))
        pipe.scheduler = _audio_scheduler(pipe.scheduler)
        return pipe

    return _cached_pipe("audio", load)


def render_audio_bed(prompt: str, seconds: float = 12.0) -> tuple[np.ndarray, int]:
    """Drone sem voz. Devolve mono float32 e a taxa de amostragem."""
    ensure_diffusion()
    _use_cpu_brownian()
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


def _empty_device_cache() -> None:
    try:
        import torch

        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass


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
    bed = soften_bed(_resample(bed, bed_sr, voice_sr), voice_sr)
    return _encode_mp3(mix_waveforms(voice, bed), voice_sr)
