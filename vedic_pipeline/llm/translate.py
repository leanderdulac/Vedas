"""Tradução de versos (sânscrito → português/inglês) com cache em disco.

A tradução usa LLM (xAI quando disponível) e o resultado é cacheado por
(texto fonte, idioma alvo, verso). Leitura do cache não tem custo; geração
nova passa pela política de token do /ask (authorize_generation).
Sem LLM, o modo extrativo devolve as testemunhas alinhadas (IAST/EN) como
textos de referência — nunca uma tradução inventada.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any

from vedic_pipeline.api.verse_service import get_verse

logger = logging.getLogger("vedic_pipeline.llm.translate")

TRANSLATION_DIR = Path("data/translations")

TRANSLATE_SYSTEM_PT = """Você é um tradutor de sânscrito clássico (védico).

Traduza o verso abaixo para português claro, natural e fiel ao original.
Regras:
- O texto sânscrito é a fonte; as testemunhas IAST/EN só apoiam a leitura.
- Mantenha termos técnicos em sânscrito transliterado quando útil (ex.: ātman, brahman, yajña).
- Não invente palavras, doutrina ou números de mantra ausentes no verso.
- Entregue apenas a tradução, sem comentários introdutórios.
"""

TRANSLATE_SYSTEM_EN = """You are a translator of classical (Vedic) Sanskrit.

Translate the verse below into clear, natural English, faithful to the original.
Rules:
- The Sanskrit text is the source; IAST/EN witnesses only support the reading.
- Keep technical terms in transliterated Sanskrit when useful (e.g. ātman, brahman, yajña).
- Do not invent words, doctrine, or mantra numbers absent from the verse.
- Output only the translation, without introductory remarks.
"""


def _source_text(bundle: dict[str, Any]) -> tuple[str, str]:
    """Escolhe o texto a traduzir: sânscrito > IAST > inglês."""
    for role in ("sa", "iast", "en"):
        for w in bundle.get("witnesses") or []:
            if w.get("role") == role and (w.get("text") or "").strip():
                return role, w["text"].strip()
    return "", ""


def _cache_path(bundle: dict[str, Any], lang: str) -> Path:
    role, text = _source_text(bundle)
    if not text:
        raise ValueError("Verso sem texto traduzível")
    digest = hashlib.sha256(f"{role}|{text}|{lang}".encode()).hexdigest()[:16]
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", bundle.get("verse_id") or "verse")[:80]
    return TRANSLATION_DIR / f"{safe}_{lang}_{digest}.json"


def load_cached_translation(bundle: dict[str, Any], lang: str) -> dict[str, Any] | None:
    try:
        path = _cache_path(bundle, lang)
    except ValueError:
        return None
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(payload, dict) or not payload.get("translation"):
        return None
    payload["cached"] = True
    return payload


def _reference_block(bundle: dict[str, Any], lang: str) -> dict[str, Any]:
    role, text = _source_text(bundle)
    references = [
        {
            "role": w.get("role"),
            "title": w.get("title"),
            "text": (w.get("text") or "").strip(),
        }
        for w in (bundle.get("witnesses") or [])
        if w.get("role") in {"iast", "en", "pt"} and (w.get("text") or "").strip()
    ]
    note = (
        "Sem LLM configurado: não há tradução automática. "
        "Testemunhas alinhadas como referência para estudo."
        if lang.startswith("pt")
        else "No LLM configured: no automatic translation. Aligned witnesses as study reference."
    )
    return {
        "verse_id": bundle.get("verse_id"),
        "locator": bundle.get("locator"),
        "lang": lang,
        "provider": "extractive",
        "model": None,
        "translation": None,
        "note": note,
        "source_role": role,
        "source_text": text,
        "references": references,
        "cached": False,
    }


def generate_translation(
    bundle: dict[str, Any],
    lang: str,
    *,
    provider: str = "auto",
    model: str | None = None,
) -> dict[str, Any]:
    """Gera a tradução (LLM) e persiste em cache. provider já deve vir resolvido."""
    from vedic_pipeline.llm.generate import generate_answer

    role, text = _source_text(bundle)
    if not text:
        raise ValueError("Verso sem texto traduzível")

    lang = "en" if (lang or "pt").lower().startswith("en") else "pt"
    if provider == "extractive":
        return _reference_block(bundle, lang)

    support = []
    for w in (bundle.get("witnesses") or []):
        if w.get("role") in {"iast", "en"} and (w.get("text") or "").strip():
            support.append(f"[{w['role']}] {w.get('title')}\n{w['text'].strip()}")
    user = (
        f"Localizador: {bundle.get('locator') or bundle.get('verse_id')}\n\n"
        f"Original ({role}):\n{text}\n"
    )
    if support:
        user += "\n\nTestemunhas de apoio:\n\n" + "\n\n---\n\n".join(support)

    system = TRANSLATE_SYSTEM_EN if lang == "en" else TRANSLATE_SYSTEM_PT
    generated = generate_answer(system, user, provider=provider, model=model, max_tokens=700)
    payload: dict[str, Any] = {
        "verse_id": bundle.get("verse_id"),
        "locator": bundle.get("locator"),
        "lang": lang,
        "provider": generated.get("provider"),
        "model": generated.get("model"),
        "translation": (generated.get("answer") or "").strip(),
        "source_role": role,
        "source_text": text,
        "references": [],
        "cached": False,
    }
    if payload["translation"]:
        try:
            path = _cache_path(bundle, lang)
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            tmp.replace(path)
        except (OSError, ValueError) as exc:  # noqa: BLE001
            logger.warning("Falha ao cachear tradução: %s", exc)
    return payload


def translate_verse(
    verse_id: str,
    *,
    lang: str = "pt",
    provider: str = "auto",
    model: str | None = None,
) -> dict[str, Any]:
    bundle = get_verse(verse_id)
    if not bundle:
        raise FileNotFoundError(f"Verso {verse_id} não encontrado")
    return generate_translation(bundle, lang, provider=provider, model=model)
