"""Verso como unidade de estudo: sânscrito + testemunhas alinhadas + explicação."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from vedic_pipeline.common.constants import DEFAULT_EMBED_DIR
from vedic_pipeline.etl.structure import format_locator, parse_verse_id
from vedic_pipeline.storage.db import get_connection, get_database_url

EXPLAIN_SYSTEM_PT = """Você é um preceptor de estudos védicos.

Explique o verso em português claro, para leitura junto do original sânscrito.
Regras:
- O original sânscrito é a fonte; tradução e IAST só iluminam, não substituem.
- Cite o localizador (ex.: RV 10.129.1, BG 2.47) e a edição da testemunha.
- Não invente números de mantra nem doutrina que não esteja nas testemunhas.
- Estruture: (1) sentido do verso, (2) termos-chave, (3) limites do corpus.
"""

EXPLAIN_SYSTEM_EN = """You are a guide for Vedic study.

Explain the verse in clear English, to be read beside the Sanskrit original.
Rules:
- The Sanskrit original is the source; translation and IAST only illuminate it.
- Cite the locator (e.g. RV 10.129.1, BG 2.47) and the witness edition.
- Do not invent verse numbers or doctrine absent from the witnesses.
- Structure: (1) sense of the verse, (2) key terms, (3) corpus limits.
"""


def witness_role(row: dict[str, Any]) -> str:
    title = (row.get("title") or "").lower()
    lang = (row.get("language") or "").lower()
    if "vedaweb" in title or "zürich" in title or "zurich" in title or "iso-15919" in title:
        return "iast"
    if lang == "sa":
        return "sa"
    if lang in {"en", "eng"}:
        return "en"
    if lang.startswith("pt"):
        return "pt"
    return lang or "other"


def _covers_verse(row: dict[str, Any], parsed: dict[str, Any]) -> bool:
    if (row.get("verse_id") or "") == parsed.get("verse_id"):
        return True
    if (row.get("work") or "") != parsed.get("work"):
        return False
    if row.get("book") != parsed.get("book"):
        return False
    if parsed.get("hymn") is not None and row.get("hymn") != parsed.get("hymn"):
        return False
    target = parsed.get("verse")
    if target is None:
        return True
    start = row.get("verse")
    end = row.get("verse_end")
    if start is None:
        return False
    last = end if end is not None else start
    return int(start) <= int(target) <= int(last)


def _fetch_pg(parsed: dict[str, Any]) -> list[dict[str, Any]]:
    verse_id = parsed["verse_id"]
    sql = """
        SELECT chunk_id, doc_id, text, title, language, license, source_url,
               work, verse_id, locator, book, hymn, verse, verse_end, heading
        FROM chunks
        WHERE verse_id = %s
           OR (
                work = %s
                AND book IS NOT DISTINCT FROM %s
                AND hymn IS NOT DISTINCT FROM %s
                AND verse IS NOT NULL
                AND verse <= %s
                AND COALESCE(verse_end, verse) >= %s
           )
        LIMIT 60
    """
    target = parsed.get("verse") or 0
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(
            sql,
            (
                verse_id,
                parsed.get("work"),
                parsed.get("book"),
                parsed.get("hymn"),
                target,
                target,
            ),
        )
        return [dict(row) for row in cur.fetchall()]


def _fetch_numpy(parsed: dict[str, Any], index_dir: Path = DEFAULT_EMBED_DIR) -> list[dict[str, Any]]:
    from vedic_pipeline.search.rag import get_index

    index = get_index(index_dir)
    out: list[dict[str, Any]] = []
    for chunk in index.get("chunks") or []:
        if _covers_verse(chunk, parsed):
            out.append(dict(chunk))
        if len(out) >= 60:
            break
    return out


def load_witness_rows(verse_id: str) -> list[dict[str, Any]]:
    parsed = parse_verse_id(verse_id)
    if not parsed:
        return []
    parsed = {**parsed, "verse_id": verse_id.strip()}
    rows: list[dict[str, Any]] = []
    if get_database_url():
        try:
            rows = _fetch_pg(parsed)
        except Exception:
            rows = []
    if not rows:
        try:
            rows = _fetch_numpy(parsed)
        except Exception:
            rows = []
    return [row for row in rows if _covers_verse(row, parsed)]


def assemble_verse(verse_id: str, rows: list[dict[str, Any]] | None = None) -> dict[str, Any] | None:
    parsed = parse_verse_id(verse_id)
    if not parsed:
        return None
    parsed = {**parsed, "verse_id": verse_id.strip()}
    rows = rows if rows is not None else load_witness_rows(verse_id)
    if not rows:
        return None
    by_role: dict[str, dict[str, Any]] = {}
    for row in rows:
        role = witness_role(row)
        prev = by_role.get(role)
        exact = (row.get("verse_id") or "") == verse_id
        if prev is None or (exact and (prev.get("verse_id") or "") != verse_id):
            by_role[role] = row
    witnesses = []
    for role in ("sa", "iast", "en", "pt"):
        row = by_role.get(role)
        if not row:
            continue
        witnesses.append(
            {
                "role": role,
                "language": row.get("language"),
                "text": (row.get("text") or "").strip(),
                "title": row.get("title"),
                "doc_id": row.get("doc_id"),
                "license": row.get("license"),
                "source_url": row.get("source_url"),
                "locator": row.get("locator"),
            }
        )
    first = rows[0]
    locator = first.get("locator") or format_locator(
        parsed["work"],
        parsed.get("book"),
        parsed.get("hymn"),
        parsed.get("verse"),
    )
    return {
        "verse_id": verse_id.strip(),
        "locator": locator,
        "work": parsed.get("work"),
        "book": parsed.get("book"),
        "hymn": parsed.get("hymn"),
        "verse": parsed.get("verse"),
        "heading": first.get("heading"),
        "witnesses": witnesses,
        "has_sanskrit": any(w["role"] in {"sa", "iast"} for w in witnesses),
    }


def get_verse(verse_id: str) -> dict[str, Any] | None:
    return assemble_verse(verse_id)


def _extractive_explain(bundle: dict[str, Any], lang: str) -> str:
    locator = bundle.get("locator") or bundle.get("verse_id")
    lines: list[str] = []
    if lang.startswith("pt"):
        lines.append(f"Leitura do original ({locator}). Sem modelo de linguagem: só as testemunhas licenciadas.")
    else:
        lines.append(f"Reading the original ({locator}). No LLM: licensed witnesses only.")
    labels = {
        "sa": ("Sânscrito", "Sanskrit"),
        "iast": ("Transliteração (ISO-15919)", "Transliteration (ISO-15919)"),
        "en": ("Inglês", "English"),
        "pt": ("Português", "Portuguese"),
    }
    for w in bundle.get("witnesses") or []:
        pt_l, en_l = labels.get(w["role"], (w["role"], w["role"]))
        label = pt_l if lang.startswith("pt") else en_l
        lines.append("")
        lines.append(f"{label} — {w.get('title') or ''}")
        lines.append(w.get("text") or "")
    return "\n".join(lines).strip()


def explain_verse(
    verse_id: str,
    *,
    lang: str = "pt",
    provider: str = "auto",
    model: str | None = None,
) -> dict[str, Any]:
    bundle = get_verse(verse_id)
    if not bundle:
        raise FileNotFoundError(f"Verso {verse_id} não encontrado")
    lang = "en" if (lang or "pt").lower().startswith("en") else "pt"
    if provider == "extractive":
        return {
            "verse_id": verse_id,
            "locator": bundle.get("locator"),
            "lang": lang,
            "provider": "extractive",
            "model": None,
            "explanation": _extractive_explain(bundle, lang),
            "witnesses": bundle.get("witnesses"),
        }

    from vedic_pipeline.llm.generate import generate_answer

    system = EXPLAIN_SYSTEM_EN if lang == "en" else EXPLAIN_SYSTEM_PT
    blocks = []
    for w in bundle.get("witnesses") or []:
        blocks.append(f"[{w['role']}] {w.get('title')}\n{w.get('text')}")
    user = (
        f"Localizador: {bundle.get('locator')}\n"
        f"verse_id: {verse_id}\n\n"
        + "\n\n---\n\n".join(blocks)
    )
    generated = generate_answer(system, user, provider=provider, model=model, max_tokens=900)
    return {
        "verse_id": verse_id,
        "locator": bundle.get("locator"),
        "lang": lang,
        "provider": generated.get("provider"),
        "model": generated.get("model"),
        "explanation": generated.get("answer"),
        "witnesses": bundle.get("witnesses"),
    }


def recitation_text(bundle: dict[str, Any]) -> str | None:
    for role in ("sa", "iast", "en"):
        for w in bundle.get("witnesses") or []:
            if w.get("role") == role and (w.get("text") or "").strip():
                return w["text"].strip()
    return None
