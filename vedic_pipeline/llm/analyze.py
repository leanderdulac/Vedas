"""Análise palavra-por-palavra do verso (vyākaraṇa) com cache em disco.

A segmentação em pāda é determinística (split pelos daṇḍas । ॥ e | ||).
A análise gramatical de cada palavra usa LLM (xAI quando disponível) e o
resultado é cacheado por (verso, texto, idioma). Leitura do cache não tem
custo; geração nova passa pela política de token do /ask
(authorize_generation). Sem LLM, devolve a segmentação em pāda + nota —
a divisão em pāda é a base da recitação tradicional e não depende de LLM.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any

from vedic_pipeline.api.verse_service import padas_of

logger = logging.getLogger("vedic_pipeline.llm.analyze")

ANALYSIS_DIR = Path("data/analyses")

ANALYZE_SYSTEM_PT = """Você é um paṇḍita de vyākaraṇa (gramática do saṃskṛta).

Analise o verso palavra por palavra. Para cada palavra (separe o sandhi e
apresente a forma contextual e a forma isolada):
- "pada": número do pāda (1-based)
- "form": palavra tal como aparece no texto (Devanāgarī)
- "iast": palavra isolada (sem sandhi) em IAST
- "grammar": análise concisa em português (ex.: "nom. sg. m. de deva-", "raiz kṛ, pres. 3ª sing.")
- "gloss": glosa em português (1-3 palavras)

Responda SOMENTE com JSON válido:
{"words": [{"pada": 1, "form": "...", "iast": "...", "grammar": "...", "gloss": "..."}]}

Não invente palavras ausentes no texto. Nada além do JSON.
"""

ANALYZE_SYSTEM_EN = """You are a paṇḍita of vyākaraṇa (Sanskrit grammar).

Analyze the verse word by word. For each word (split the sandhi and give the
in-context and isolated forms):
- "pada": pāda number (1-based)
- "form": word as it appears in the text (Devanāgarī)
- "iast": isolated word (sandhi split) in IAST
- "grammar": concise analysis in English (e.g. "nom. sg. m. of deva-", "root kṛ, pres. 3sg.")
- "gloss": short English gloss (1-3 words)

Answer ONLY with valid JSON:
{"words": [{"pada": 1, "form": "...", "iast": "...", "grammar": "...", "gloss": "..."}]}

Do not invent words absent from the text. Nothing beyond the JSON.
"""


def _cache_path(bundle: dict[str, Any], padas: list[dict[str, Any]], lang: str) -> Path:
    seed = json.dumps(
        {"v": bundle.get("verse_id"), "padas": padas, "lang": lang},
        ensure_ascii=False,
        sort_keys=True,
    )
    digest = hashlib.sha256(seed.encode()).hexdigest()[:16]
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", bundle.get("verse_id") or "verse")[:80]
    return ANALYSIS_DIR / f"{safe}_{lang}_{digest}.json"


def load_cached_analysis(
    bundle: dict[str, Any], padas: list[dict[str, Any]], lang: str
) -> dict[str, Any] | None:
    try:
        path = _cache_path(bundle, padas, lang)
    except (ValueError, TypeError):
        return None
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(payload, dict) or not isinstance(payload.get("padas"), list):
        return None
    payload["cached"] = True
    return payload


def _parse_words(answer: str) -> list[dict[str, Any]]:
    """Extrai {"words": [...]} da resposta do LLM (tolerante a cercas de código)."""
    if not answer:
        return []
    candidates = [answer.strip()]
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", answer, re.DOTALL)
    if fenced:
        candidates.insert(0, fenced.group(1))
    first = answer.find("{")
    last = answer.rfind("}")
    if first != -1 and last > first:
        candidates.insert(0, answer[first : last + 1])
    for cand in candidates:
        try:
            data = json.loads(cand)
        except json.JSONDecodeError:
            continue
        words = data.get("words") if isinstance(data, dict) else None
        if isinstance(words, list):
            return [w for w in words if isinstance(w, dict) and (w.get("form") or w.get("gloss"))]
    return []


def _user_prompt(bundle: dict[str, Any], padas: list[dict[str, Any]]) -> str:
    lines = [f"Localizador: {bundle.get('locator') or bundle.get('verse_id')}", ""]
    for p in padas:
        line = f"[pāda {p['index']}] {p['sa']}"
        if p.get("iast"):
            line += f"  ({p['iast']})"
        lines.append(line)
    witnesses = [
        f"[{w.get('role')}] {w.get('title')}\n{w.get('text')}"
        for w in bundle.get("witnesses") or []
        if w.get("role") in {"iast", "en"} and (w.get("text") or "").strip()
    ]
    if witnesses:
        lines.append("")
        lines.append("Testemunhas de apoio:")
        lines.extend(witnesses)
    return "\n".join(lines)


def _extractive(bundle: dict[str, Any], padas: list[dict[str, Any]], lang: str) -> dict[str, Any]:
    note = (
        "Análise palavra-por-palavra requer LLM configurado (VEDIC_GENERATION_API_TOKEN ou "
        "XAI_API_KEY). A divisão em pāda é a base da recitação tradicional e funciona sem LLM."
        if lang.startswith("pt")
        else "Word-by-word analysis needs a configured LLM (VEDIC_GENERATION_API_TOKEN or "
        "XAI_API_KEY). The pāda split is the basis of traditional recitation and works without an LLM."
    )
    return {
        "verse_id": bundle.get("verse_id"),
        "locator": bundle.get("locator"),
        "lang": lang,
        "provider": "extractive",
        "model": None,
        "padas": padas,
        "words": [],
        "note": note,
        "cached": False,
    }


def generate_analysis(
    bundle: dict[str, Any],
    padas: list[dict[str, Any]],
    lang: str,
    *,
    provider: str = "auto",
    model: str | None = None,
) -> dict[str, Any]:
    """Gera a análise (LLM) e persiste em cache. provider já deve vir resolvido."""
    from vedic_pipeline.llm.generate import generate_answer

    lang = "en" if (lang or "pt").lower().startswith("en") else "pt"
    if provider == "extractive":
        return _extractive(bundle, padas, lang)

    system = ANALYZE_SYSTEM_EN if lang == "en" else ANALYZE_SYSTEM_PT
    generated = generate_answer(
        system, _user_prompt(bundle, padas), provider=provider, model=model, max_tokens=1600
    )
    words = _parse_words(generated.get("answer") or "")
    payload: dict[str, Any] = {
        "verse_id": bundle.get("verse_id"),
        "locator": bundle.get("locator"),
        "lang": lang,
        "provider": generated.get("provider"),
        "model": generated.get("model"),
        "padas": padas,
        "words": words,
        "note": None if words else "O modelo não retornou análise estruturada válida.",
        "cached": False,
    }
    if words:
        try:
            path = _cache_path(bundle, padas, lang)
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            tmp.replace(path)
        except (OSError, ValueError) as exc:  # noqa: BLE001
            logger.warning("Falha ao cachear análise: %s", exc)
    return payload


def analyze_verse(
    verse_id: str,
    *,
    lang: str = "pt",
    provider: str = "auto",
    model: str | None = None,
) -> dict[str, Any]:
    bundle = get_verse_bundle(verse_id)
    if not bundle:
        raise FileNotFoundError(f"Verso {verse_id} não encontrado")
    padas = padas_of(bundle)
    if not padas:
        raise ValueError("Verso sem texto em sânscrito/IAST para análise")
    return generate_analysis(bundle, padas, lang, provider=provider, model=model)


def get_verse_bundle(verse_id: str) -> dict[str, Any] | None:
    from vedic_pipeline.api.verse_service import get_verse

    return get_verse(verse_id)
