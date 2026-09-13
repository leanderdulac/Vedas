"""Narração de versos via SpaceXAI TTS, com cache em disco."""

from __future__ import annotations

import hashlib
import logging
import os
import re
from pathlib import Path

logger = logging.getLogger("vedic_pipeline.llm.tts")

DEFAULT_VOICE = "orion"
DEFAULT_BASE_URL = "https://api.x.ai/v1"
AUDIO_DIR = Path("data/audio")


def tts_available() -> bool:
    return bool(os.environ.get("XAI_API_KEY"))


def _cache_path(verse_id: str, language: str, text: str) -> Path:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", verse_id)[:80]
    return AUDIO_DIR / f"{safe}_{language}_{digest}.mp3"


def synthesize(
    text: str,
    *,
    language: str = "sa",
    voice_id: str | None = None,
) -> bytes:
    api_key = os.environ.get("XAI_API_KEY")
    if not api_key:
        raise RuntimeError("XAI_API_KEY não definida")
    import httpx

    base = os.environ.get("XAI_BASE_URL", DEFAULT_BASE_URL).rstrip("/")
    voice = voice_id or os.environ.get("XAI_TTS_VOICE", DEFAULT_VOICE)
    payload = {"text": text[:4000], "voice_id": voice, "language": language}
    with httpx.Client(timeout=60.0) as client:
        resp = client.post(
            f"{base}/tts",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload,
        )
        if resp.status_code >= 400 and language not in {"auto", "hi"}:
            logger.warning("TTS language=%s falhou (%s); tentando auto", language, resp.status_code)
            payload["language"] = "auto"
            resp = client.post(
                f"{base}/tts",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json=payload,
            )
        resp.raise_for_status()
        return resp.content


def cached_verse_audio(
    verse_id: str,
    text: str,
    *,
    language: str = "sa",
) -> Path:
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    dest = _cache_path(verse_id, language, text)
    if dest.exists() and dest.stat().st_size > 100:
        return dest
    audio = synthesize(text, language=language)
    dest.write_bytes(audio)
    return dest
