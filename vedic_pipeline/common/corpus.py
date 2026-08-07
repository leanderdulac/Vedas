"""I/O do corpus JSONL e utilitários de identidade."""

from __future__ import annotations

import hashlib
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional

logger = logging.getLogger("vedic_pipeline.corpus")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_id(*parts: str) -> str:
    payload = "|".join(p or "" for p in parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def content_fingerprint(text: str) -> str:
    norm = re.sub(r"\s+", " ", text.lower()).strip()
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()


def load_corpus(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                logger.warning("Linha %d inválida em %s: %s", line_no, path, exc)
    return records


def append_corpus(path: Path, records: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("a", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            count += 1
    return count


def rewrite_corpus(path: Path, records: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    count = 0
    with tmp.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            count += 1
    tmp.replace(path)
    return count


def iter_corpus_texts(path: Path) -> Iterator[str]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = (rec.get("text") or "").strip()
            if text:
                yield text


def iter_corpus_records(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def deduplicate_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen_fp: set[str] = set()
    seen_url: set[str] = set()
    unique: list[dict[str, Any]] = []
    for rec in records:
        text = rec.get("text") or ""
        url = (rec.get("source_url") or "").strip()
        fp = rec.get("fingerprint") or content_fingerprint(text)
        if fp in seen_fp:
            continue
        if url and url in seen_url:
            continue
        seen_fp.add(fp)
        if url:
            seen_url.add(url)
        rec = dict(rec)
        rec["fingerprint"] = fp
        unique.append(rec)
    return unique


# Frases/padrões específicos (evitar substring acidente: "isha" ⊂ "upanishads")
_WORK_KEYS: list[tuple[str, tuple[str, ...]]] = [
    ("bhagavad-gita", (r"\bbhagavad[- ]?g[iī]t[aā]\b", r"\bsong celestial\b")),
    ("yoga-sutra", (r"\byoga[- ]s[uū]tras?\b", r"\bpatanjali\b", r"\bpatañjali\b")),
    ("isha-upanishad", (r"\b[iī]sha\b", r"\b[iī]ś[aā]\b", r"\bisa upanishad\b")),
    ("ramayana", (r"\br[aā]m[aā]yan", r"\bvalmiki\b", r"\bvālmīki\b")),
    ("mahabharata", (r"\bmah[aā]bh[aā]rat", r"\bganguli\b")),
]


def _is_fixture_url(url: str) -> bool:
    u = (url or "").strip()
    return u.startswith("fixtures/") or "/fixtures/" in u


def _work_key(title: str) -> Optional[str]:
    t = (title or "").lower()
    for key, patterns in _WORK_KEYS:
        if any(re.search(p, t, flags=re.IGNORECASE) for p in patterns):
            return key
    return None


def drop_superseded_fixtures(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove fixtures curtas quando já existe obra maior (web) da mesma família+idioma."""
    # best_web[(work_key, language)] = char_count
    best_web: dict[tuple[str, str], int] = {}
    for rec in records:
        url = (rec.get("source_url") or "").strip()
        if _is_fixture_url(url):
            continue
        wk = _work_key(rec.get("title") or "")
        if not wk:
            continue
        lang = (rec.get("language") or "en").lower()
        n = len(rec.get("text") or "")
        k = (wk, lang)
        if n > best_web.get(k, 0):
            best_web[k] = n

    kept: list[dict[str, Any]] = []
    for rec in records:
        url = (rec.get("source_url") or "").strip()
        wk = _work_key(rec.get("title") or "")
        lang = (rec.get("language") or "en").lower()
        web_chars = best_web.get((wk or "", lang), 0) if wk else 0
        if (
            _is_fixture_url(url)
            and wk
            and web_chars >= 3 * max(len(rec.get("text") or ""), 1)
        ):
            logger.info(
                "Removendo fixture superada: %s (obra web %s/%s com %d chars)",
                rec.get("title"),
                wk,
                lang,
                web_chars,
            )
            continue
        kept.append(rec)
    return kept

def dedupe_corpus_file(
    corpus_path: Path,
    *,
    drop_fixtures: bool = True,
) -> dict[str, Any]:
    records = load_corpus(corpus_path)
    before = len(records)
    unique = deduplicate_records(records)
    after_fp = len(unique)
    if drop_fixtures:
        unique = drop_superseded_fixtures(unique)
    after = rewrite_corpus(corpus_path, unique)
    return {
        "before": before,
        "after_fingerprint_dedupe": after_fp,
        "after": after,
        "removed": before - after,
        "dropped_fixtures": after_fp - after if drop_fixtures else 0,
    }