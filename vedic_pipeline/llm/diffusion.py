"""Mídia local com a família Stable Diffusion.

- Imagem: Stable Diffusion. `VEDIC_SD_MODEL` escolhe o modelo; sem ele, o
  `stabilityai/sdxl-turbo` se estiver no cache, senão o `stabilityai/sd-turbo`
  (também o fallback offline quando o modelo pedido não tem pesos locais). Com um still já em disco e `force=True`,
  img2img refina a cena do verso em vez de recomeçar.
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


def sampling_for(model_name: str) -> tuple[int, float]:
    """Turbo: poucos passos e um CFG leve (1,5) para o prompt negativo pesar.

    Com guidance 0 o turbo ignora o negativo e o Agni saía de pele azul.
    `VEDIC_SD_TURBO_GUIDANCE=0` volta ao modo sem CFG (metade do custo).
    Os outros modelos usam CFG cheio (`VEDIC_SD_GUIDANCE`).
    """
    steps = int(os.environ.get("VEDIC_SD_STEPS", "4"))
    steps = max(1, steps)
    if "turbo" in (model_name or "").lower():
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


def _image_pipes() -> tuple[Any, Any]:
    def load() -> tuple[Any, Any]:
        from diffusers import AutoPipelineForImage2Image, AutoPipelineForText2Image

        model = image_model()
        _device, dtype = _device_and_dtype()
        logger.info("Carregando modelo de imagem %s", model)
        txt = _place(_from_pretrained(AutoPipelineForText2Image, model, torch_dtype=dtype))
        img = AutoPipelineForImage2Image.from_pipe(txt)
        return txt, img

    return _cached_pipe(f"image:{image_model()}", load)


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
) -> bytes:
    """JPEG. `init_image` refina o still existente (img2img)."""
    ensure_diffusion()
    from PIL import Image

    model = image_model()
    steps, guidance = sampling_for(model)
    size = int(os.environ.get("VEDIC_SD_SIZE", "512"))
    size = max(256, min(size, 1024)) // 8 * 8
    kwargs: dict[str, Any] = {
        "prompt": prompt,
        "num_inference_steps": steps,
        "guidance_scale": guidance,
    }
    if guidance > 1.0:
        kwargs["negative_prompt"] = negative or negative_prompt()
    with _PIPE_LOCK:
        txt, img = _image_pipes()
        _warn_if_truncated(txt, prompt)
        generator = _generator()
        if generator is not None:
            kwargs["generator"] = generator
        if init_image is not None:
            strength = float(os.environ.get("VEDIC_SD_STRENGTH", "0.35"))
            kwargs["num_inference_steps"] = img2img_steps(steps, strength)
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


def video_size() -> tuple[int, int]:
    """SVD foi treinado em 1024x576. `VEDIC_SVD_WIDTH` menor (ex.: 768) poupa memória."""
    raw = (os.environ.get("VEDIC_SVD_WIDTH") or "").strip()
    width = int(raw) if raw.isdigit() else 1024
    width = max(512, min(width, 1024)) // 64 * 64
    height = (width * 9 // 16) // 8 * 8
    return width, height


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


def render_video(still: Path, dest: Path, motion: str) -> None:
    """MP4 curto a partir do still. Movimento sai do texto de `motion`."""
    ensure_diffusion()
    from diffusers.utils import export_to_video
    from PIL import Image

    width, height = video_size()
    image = _cover(Image.open(still).convert("RGB"), width, height)
    frames_n = int(os.environ.get("VEDIC_SVD_FRAMES", "14"))
    fps = int(os.environ.get("VEDIC_SVD_FPS", "7"))
    with _PIPE_LOCK:
        result = _video_pipe()(
            image,
            width=width,
            height=height,
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
    bed = normalize_peak(_resample(bed, bed_sr, voice_sr))
    return _encode_mp3(mix_waveforms(voice, bed), voice_sr)
