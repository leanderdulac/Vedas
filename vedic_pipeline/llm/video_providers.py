"""Vídeo do verso por API: Grok Imagine (xAI) e Runway, image-to-video.

As duas APIs são assíncronas: cria a tarefa, consulta até ficar pronta e baixa
o MP4 com a mesma validação SSRF das imagens. Quem chama roda isto na thread
de fundo do vídeo (`imagine._video_worker`) e cai no SVD local se falhar.

- xAI: `POST /v1/videos/generations` + `GET /v1/videos/{request_id}`;
  `XAI_VIDEO_MODEL` (padrão `grok-imagine-video-1.5`, US$ 0,08/s).
- Runway: `POST /v1/image_to_video` + `GET /v1/tasks/{id}` em
  `api.dev.runwayml.com` com `X-Runway-Version`; `RUNWAY_VIDEO_MODEL`
  (padrão `gen4.5`, 12 créditos/s; 1 crédito = US$ 0,01). A chave vem de
  `RUNWAYML_API_SECRET`.

O still quadrado entra inteiro num quadro 16:9 (laterais com o próprio still
desfocado), como no SVD: as APIs recortam o centro e cortariam as chamas.
"""

from __future__ import annotations

import base64
import io
import logging
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger("vedic_pipeline.llm.video_providers")

DEFAULT_XAI_VIDEO_MODEL = "grok-imagine-video-1.5"
DEFAULT_RUNWAY_VIDEO_MODEL = "gen4.5"
XAI_BASE = "https://api.x.ai/v1"
RUNWAY_BASE = "https://api.dev.runwayml.com/v1"
RUNWAY_VERSION = "2024-11-06"
# US$ por crédito da Runway (docs.dev.runwayml.com/guides/pricing).
RUNWAY_USD_PER_CREDIT = 0.01
# 1 US$ = 1e10 "ticks" no `usage.cost_in_usd_ticks` do xAI.
XAI_TICKS_PER_USD = 10_000_000_000
VIDEO_LIMIT_BYTES = 50 * 1024 * 1024
FRAME_SIZE = (1280, 720)


class VideoProviderError(RuntimeError):
    """Falha da API de vídeo (recusa, sem crédito, tarefa falhou, tempo esgotado)."""


def xai_video_model() -> str:
    return (os.environ.get("XAI_VIDEO_MODEL") or "").strip() or DEFAULT_XAI_VIDEO_MODEL


def runway_video_model() -> str:
    return (os.environ.get("RUNWAY_VIDEO_MODEL") or "").strip() or DEFAULT_RUNWAY_VIDEO_MODEL


def _env_number(name: str, default: float, lo: float, hi: float) -> float:
    raw = (os.environ.get(name) or "").strip()
    try:
        value = float(raw) if raw else default
    except ValueError:
        logger.warning("%s=%r inválido; usando %s", name, raw, default)
        value = default
    return max(lo, min(value, hi))


def video_seconds() -> int:
    """`VEDIC_VIDEO_SECONDS` (5): duração pedida às APIs, de 2 a 10 s."""
    return int(_env_number("VEDIC_VIDEO_SECONDS", 5, 2, 10))


def poll_timeout() -> float:
    """`VEDIC_VIDEO_TIMEOUT` (600 s): espera máxima pela tarefa da API."""
    return _env_number("VEDIC_VIDEO_TIMEOUT", 600, 30, 3600)


def still_data_uri(still: Path, size: tuple[int, int] = FRAME_SIZE) -> str:
    """Still como data URI JPEG 16:9 (até ~5 MB, limite da Runway)."""
    raw = still.read_bytes()
    try:
        from PIL import Image

        from vedic_pipeline.llm.diffusion import _pad_frame

        image = Image.open(io.BytesIO(raw)).convert("RGB")
        if abs(image.width / image.height - size[0] / size[1]) > 0.01:
            image = _pad_frame(image, size[0], size[1])
        else:
            image = image.resize(size)
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=90)
        raw = buf.getvalue()
    except (ImportError, OSError, ValueError):
        # Sem Pillow (CI) ou still que o Pillow não lê: vai como está.
        pass
    return "data:image/jpeg;base64," + base64.b64encode(raw).decode("ascii")


def _http_client(base_url: str, headers: dict[str, str]) -> httpx.Client:
    return httpx.Client(base_url=base_url, headers=headers, timeout=120.0)


def _error_detail(resp: httpx.Response) -> str:
    try:
        body = resp.json()
    except ValueError:
        return resp.text[:300]
    if isinstance(body, dict):
        err = body.get("error") or body.get("message") or body.get("failure") or body
        if isinstance(err, dict):
            err = err.get("message") or err
        return str(err)[:300]
    return str(body)[:300]


def _download(url: str) -> bytes:
    from vedic_pipeline.llm.imagine import _download_media_bytes

    return _download_media_bytes(url, timeout=180.0, limit=VIDEO_LIMIT_BYTES)


def _save(dest: Path, data: bytes) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.stem + ".tmp.mp4")
    tmp.write_bytes(data)
    tmp.replace(dest)


def render_xai_video(
    still: Path,
    dest: Path,
    prompt: str,
    *,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    """Grok Imagine image-to-video, 720p 16:9. Devolve modelo, segundos e custo."""
    key = os.environ.get("XAI_API_KEY")
    if not key:
        raise VideoProviderError("XAI_API_KEY não definida")
    base = (os.environ.get("XAI_BASE_URL") or XAI_BASE).rstrip("/")
    model = xai_video_model()
    seconds = video_seconds()
    started = clock()
    with _http_client(base, {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}) as client:
        resp = client.post(
            "/videos/generations",
            json={
                "model": model,
                "prompt": prompt,
                "image": {"url": still_data_uri(still)},
                "duration": seconds,
                "aspect_ratio": "16:9",
                "resolution": "720p",
            },
        )
        if resp.status_code >= 400:
            raise VideoProviderError(f"xAI recusou o vídeo ({resp.status_code}): {_error_detail(resp)}")
        request_id = (resp.json() or {}).get("request_id")
        if not request_id:
            raise VideoProviderError("xAI não devolveu request_id")
        deadline = started + poll_timeout()
        while True:
            sleep(5.0)
            poll = client.get(f"/videos/{request_id}")
            if poll.status_code >= 400:
                raise VideoProviderError(f"xAI falhou ao consultar o vídeo ({poll.status_code}): {_error_detail(poll)}")
            payload = poll.json() or {}
            status = str(payload.get("status") or "").lower()
            if status == "done":
                break
            if status in {"failed", "expired", "error"}:
                err = payload.get("error") or {}
                detail = err.get("message") if isinstance(err, dict) else err
                raise VideoProviderError(f"xAI: vídeo {status}: {detail or 'sem detalhe'}")
            if clock() > deadline:
                raise VideoProviderError("xAI: tempo esgotado à espera do vídeo")
    url = (payload.get("video") or {}).get("url") or payload.get("url")
    if not url:
        raise VideoProviderError("xAI: vídeo pronto sem URL")
    _save(dest, _download(url))
    ticks = (payload.get("usage") or {}).get("cost_in_usd_ticks")
    cost = round(ticks / XAI_TICKS_PER_USD, 4) if isinstance(ticks, (int, float)) else None
    return {"backend": "xai", "model": payload.get("model") or model, "seconds": seconds, "cost_usd": cost}


def render_runway_video(
    still: Path,
    dest: Path,
    prompt: str,
    *,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    """Runway image-to-video, 1280:720. Devolve modelo, segundos e custo."""
    key = os.environ.get("RUNWAYML_API_SECRET")
    if not key:
        raise VideoProviderError("RUNWAYML_API_SECRET não definida")
    base = (os.environ.get("RUNWAY_BASE_URL") or RUNWAY_BASE).rstrip("/")
    model = runway_video_model()
    seconds = video_seconds()
    started = clock()
    headers = {
        "Authorization": f"Bearer {key}",
        "X-Runway-Version": RUNWAY_VERSION,
        "Content-Type": "application/json",
    }
    with _http_client(base, headers) as client:
        resp = client.post(
            "/image_to_video",
            json={
                "model": model,
                "promptImage": still_data_uri(still),
                "promptText": prompt[:1000],
                "ratio": "1280:720",
                "duration": seconds,
            },
        )
        if resp.status_code >= 400:
            raise VideoProviderError(f"Runway recusou o vídeo ({resp.status_code}): {_error_detail(resp)}")
        task_id = (resp.json() or {}).get("id")
        if not task_id:
            raise VideoProviderError("Runway não devolveu o id da tarefa")
        deadline = started + poll_timeout()
        while True:
            sleep(5.0)
            poll = client.get(f"/tasks/{task_id}")
            if poll.status_code >= 400:
                raise VideoProviderError(f"Runway falhou ao consultar a tarefa ({poll.status_code}): {_error_detail(poll)}")
            task = poll.json() or {}
            status = str(task.get("status") or "").upper()
            if status == "SUCCEEDED":
                break
            if status in {"FAILED", "CANCELLED"}:
                code = task.get("failureCode") or ""
                raise VideoProviderError(f"Runway: tarefa {status.lower()} {code}: {task.get('failure') or 'sem detalhe'}")
            if clock() > deadline:
                raise VideoProviderError("Runway: tempo esgotado à espera do vídeo")
    output = task.get("output") or []
    if not output:
        raise VideoProviderError("Runway: tarefa pronta sem vídeo")
    _save(dest, _download(output[0]))
    credits = (task.get("cost") or {}).get("credits") if isinstance(task.get("cost"), dict) else None
    cost = round(credits * RUNWAY_USD_PER_CREDIT, 4) if isinstance(credits, (int, float)) else None
    return {"backend": "runway", "model": model, "seconds": seconds, "credits": credits, "cost_usd": cost}


RENDERERS: dict[str, Callable[..., dict[str, Any]]] = {
    "xai": render_xai_video,
    "runway": render_runway_video,
}
