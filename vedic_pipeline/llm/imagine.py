"""Ilustração de versos: Imagine (imagem) e vídeo a partir do still."""

from __future__ import annotations

import base64
import json
import logging
import os
import re
from pathlib import Path
from typing import Any

import httpx

from vedic_pipeline.api.verse_service import get_verse

logger = logging.getLogger("vedic_pipeline.llm.imagine")

MEDIA_DIR = Path("data/media")
IMAGE_MODEL = os.environ.get("XAI_IMAGE_MODEL", "grok-imagine-image-2.0")
VIDEO_MODEL = os.environ.get("XAI_VIDEO_MODEL", "grok-imagine-video-1.5")
DEFAULT_BASE = "https://api.x.ai/v1"

SCENE_STYLE = (
    "Classical Indian mural and Pahari miniature sensibility, sacred and restrained, "
    "soft mineral pigments, no photorealistic modern faces, no Latin or Devanagari "
    "lettering in the frame, no logos, no cameras, no contemporary clothing."
)


def _safe_id(verse_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", verse_id)[:80]


def image_path(verse_id: str) -> Path:
    return MEDIA_DIR / f"{_safe_id(verse_id)}.jpg"


def video_path(verse_id: str) -> Path:
    return MEDIA_DIR / f"{_safe_id(verse_id)}.mp4"


def job_path(verse_id: str) -> Path:
    return MEDIA_DIR / "jobs" / f"{_safe_id(verse_id)}.json"


def list_cached_media() -> dict[str, list[str]]:
    MEDIA_DIR.mkdir(parents=True, exist_ok=True)
    images = sorted(p.stem for p in MEDIA_DIR.glob("*.jpg") if p.stat().st_size > 1000)
    videos = sorted(p.stem for p in MEDIA_DIR.glob("*.mp4") if p.stat().st_size > 1000)
    return {"images": images, "videos": videos}


def visual_prompt(bundle: dict[str, Any]) -> str:
    locator = bundle.get("locator") or bundle.get("verse_id") or ""
    witnesses = bundle.get("witnesses") or []
    meaning = ""
    for role in ("en", "iast", "sa"):
        for w in witnesses:
            if w.get("role") == role and (w.get("text") or "").strip():
                meaning = w["text"].strip()
                break
        if meaning:
            break
    meaning = re.sub(r"\s+", " ", meaning)[:700]
    return (
        f"A single sacred scene illustrating Vedic verse {locator}. "
        f"The verse evokes: {meaning}. {SCENE_STYLE} "
        "One clear subject, cinematic still suitable as the first frame of a short film."
    )


def motion_prompt(bundle: dict[str, Any]) -> str:
    locator = bundle.get("locator") or ""
    return (
        f"Gentle sacred motion for verse {locator}: slow camera push-in, "
        "lamplight and incense drift, fabric and flame move slightly, "
        "reverent pace, no sudden cuts, no modern objects."
    )


def _client() -> httpx.Client:
    key = os.environ.get("XAI_API_KEY")
    if not key:
        raise RuntimeError("XAI_API_KEY não definida")
    base = os.environ.get("XAI_BASE_URL", DEFAULT_BASE).rstrip("/")
    return httpx.Client(
        base_url=base,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        timeout=120.0,
    )


def _save_image_bytes(dest: Path, raw: bytes) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(raw)
    return dest


def generate_verse_image(verse_id: str, *, force: bool = False) -> Path:
    dest = image_path(verse_id)
    if dest.exists() and dest.stat().st_size > 1000 and not force:
        return dest
    bundle = get_verse(verse_id)
    if not bundle:
        raise FileNotFoundError(f"Verso {verse_id} não encontrado")
    prompt = visual_prompt(bundle)
    with _client() as client:
        resp = client.post(
            "/images/generations",
            json={
                "model": IMAGE_MODEL,
                "prompt": prompt,
                "n": 1,
            },
        )
        if resp.status_code >= 400:
            detail = resp.text
            try:
                err = resp.json()
                detail = err.get("error") or err.get("message") or detail
            except Exception:
                pass
            raise RuntimeError(
                f"Imagine recusou a imagem ({resp.status_code}): {detail}. "
                "No console.x.ai, ative a permissão de Image/Video (Imagine) nesta chave."
            )
        payload = resp.json()
    data = (payload.get("data") or [payload])[0]
    if data.get("b64_json"):
        return _save_image_bytes(dest, base64.b64decode(data["b64_json"]))
    url = data.get("url")
    if not url:
        raise RuntimeError("Imagine não devolveu imagem")
    with httpx.Client(timeout=60.0) as dl:
        img = dl.get(url)
        img.raise_for_status()
        return _save_image_bytes(dest, img.content)


def start_verse_video(verse_id: str, *, duration: int = 6) -> dict[str, Any]:
    still = generate_verse_image(verse_id)
    bundle = get_verse(verse_id) or {}
    b64 = base64.b64encode(still.read_bytes()).decode("ascii")
    data_url = f"data:image/jpeg;base64,{b64}"
    with _client() as client:
        resp = client.post(
            "/videos/generations",
            json={
                "model": VIDEO_MODEL,
                "prompt": motion_prompt(bundle),
                "image": {"url": data_url},
                "duration": duration,
            },
        )
        if resp.status_code >= 400:
            detail = resp.text
            try:
                err = resp.json()
                detail = err.get("error") or err.get("message") or detail
            except Exception:
                pass
            raise RuntimeError(
                f"Imagine recusou o vídeo ({resp.status_code}): {detail}. "
                "Ative Image/Video nesta chave em console.x.ai."
            )
        payload = resp.json()
    request_id = payload.get("request_id") or payload.get("id")
    if not request_id:
        raise RuntimeError(f"Imagine vídeo sem request_id: {payload}")
    job = {"verse_id": verse_id, "request_id": request_id, "status": "pending"}
    job_path(verse_id).parent.mkdir(parents=True, exist_ok=True)
    job_path(verse_id).write_text(json.dumps(job), encoding="utf-8")
    return job


def poll_verse_video(verse_id: str) -> dict[str, Any]:
    dest = video_path(verse_id)
    if dest.exists() and dest.stat().st_size > 1000:
        return {"verse_id": verse_id, "status": "done", "ready": True}
    meta = job_path(verse_id)
    if not meta.exists():
        return {"verse_id": verse_id, "status": "missing", "ready": False}
    job = json.loads(meta.read_text(encoding="utf-8"))
    request_id = job.get("request_id")
    with _client() as client:
        resp = client.get(f"/videos/{request_id}")
        resp.raise_for_status()
        payload = resp.json()
    status = (payload.get("status") or "").lower()
    job["status"] = status
    meta.write_text(json.dumps(job), encoding="utf-8")
    if status in {"done", "completed", "succeeded"}:
        url = (payload.get("video") or {}).get("url") or payload.get("url")
        if not url:
            raise RuntimeError("Vídeo pronto sem URL")
        with httpx.Client(timeout=120.0) as dl:
            vid = dl.get(url)
            vid.raise_for_status()
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(vid.content)
        return {"verse_id": verse_id, "status": "done", "ready": True}
    if status in {"failed", "expired", "error"}:
        return {"verse_id": verse_id, "status": status, "ready": False, "detail": payload}
    return {"verse_id": verse_id, "status": status or "pending", "ready": False}
