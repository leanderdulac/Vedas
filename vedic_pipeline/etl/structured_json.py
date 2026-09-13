"""Ingestão de JSON estruturado (DharmicData e formato compacto interno).

Só o texto sânscrito antigo entra no corpus, salvo traduções já em domínio
público (Griffith). Comentários e traduções modernas são descartados.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from typing import Any

from vedic_pipeline.common.corpus import content_fingerprint, stable_id, utc_now_iso
from vedic_pipeline.etl.structure import parse_int_or_roman

DHARMICDATA_REPO = "https://github.com/bhavykhatri/DharmicData"
DHARMICDATA_ATTR = (
    "Contains information from DharmicData "
    "(https://github.com/bhavykhatri/DharmicData), available under the Open Database License (ODbL). "
    "Sanskrit slokas/mantras only; modern commentaries and translations omitted."
)
VEDAWEB_REPO = "https://github.com/VedaWebProject/vedaweb-data"
VEDAWEB_ZURICH_XLSX = (
    "https://github.com/VedaWebProject/vedaweb-data/blob/main/rigveda/versions/zurich.xlsx"
)
VEDAWEB_ATTR = (
    "Contains the Zürich Rigveda layer from VedaWeb "
    "(https://github.com/VedaWebProject/vedaweb-data), University of Cologne / University of Zürich, "
    "distributed under Creative Commons Attribution 4.0 International (CC BY 4.0). "
    "ISO-15919 pada text only; morphology and NC-licensed layers omitted."
)

_DEV_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")
_VERSE_END = re.compile(r"(?:॥|।।)\s*([0-9०-९]+)\s*(?:॥|।।)")
_LEADING_COUNT = re.compile(r"^[0-9०-९]+\s*")
_BOOK_TOKEN = re.compile(r"(?:book|part|chapter|decade)\s+([IVXLCDM]+|\d+)", re.IGNORECASE)


def devanagari_int(token: str) -> int | None:
    digits = (token or "").translate(_DEV_DIGITS).strip()
    if digits.isdigit():
        value = int(digits)
        return value if value > 0 else None
    return None


def split_sukta_text(text: str) -> tuple[str | None, list[tuple[int, str]]]:
    """Separa anukramaṇī (ṛṣi/devatā/chanda) e mantras marcados com ॥n॥ / ।।n।।."""
    raw = (text or "").strip()
    if not raw:
        return None, []
    lines = raw.split("\n")
    heading: str | None = None
    body = raw
    first = lines[0].strip()
    if first and "।" in first and "॥" not in first and "।।" not in first:
        heading = _LEADING_COUNT.sub("", first).strip(" ।")
        body = "\n".join(lines[1:]).strip()

    verses: list[tuple[int, str]] = []
    pos = 0
    for match in _VERSE_END.finditer(body):
        number = devanagari_int(match.group(1))
        piece = body[pos : match.start()].strip()
        if number and piece:
            verses.append((number, piece))
        pos = match.end()
    return heading or None, verses


def compact_gita_chapter(payload: dict[str, Any] | list[Any]) -> dict[str, Any]:
    verses_in = []
    if isinstance(payload, dict) and isinstance(payload.get("BhagavadGitaChapter"), list):
        verses_in = payload["BhagavadGitaChapter"]
    elif isinstance(payload, dict) and payload.get("work") == "bhagavad-gita":
        return payload
    elif isinstance(payload, list):
        verses_in = payload
    verses = []
    chapter = None
    for item in verses_in:
        if not isinstance(item, dict):
            continue
        ch = item.get("chapter")
        n = item.get("verse")
        sa = (item.get("text") or "").strip()
        if ch is None or n is None or not sa:
            continue
        chapter = int(ch)
        verses.append({"verse": int(n), "text": sa})
    verses.sort(key=lambda v: v["verse"])
    return {
        "work": "bhagavad-gita",
        "chapter": chapter,
        "language": "sa",
        "license": "odbl",
        "attribution": DHARMICDATA_ATTR,
        "source_url": DHARMICDATA_REPO,
        "verses": verses,
    }


def compact_rigveda_mandala(payload: list[Any] | dict[str, Any]) -> dict[str, Any]:
    if isinstance(payload, dict) and payload.get("work") == "rigveda":
        return payload
    rows = payload if isinstance(payload, list) else payload.get("suktas") or []
    suktas: list[dict[str, Any]] = []
    mandala = None
    for item in rows:
        if not isinstance(item, dict):
            continue
        book = item.get("mandala")
        hymn = item.get("sukta") or item.get("hymn")
        if book is None or hymn is None:
            continue
        mandala = int(book)
        if item.get("verses"):
            verses = [(int(v["n"]), v["text"]) for v in item["verses"] if v.get("text")]
            heading = item.get("heading")
        else:
            heading, verses = split_sukta_text(item.get("text") or "")
        if not verses:
            continue
        suktas.append(
            {
                "sukta": int(hymn),
                "heading": heading,
                "verses": [{"n": n, "text": t} for n, t in verses],
            }
        )
    suktas.sort(key=lambda s: s["sukta"])
    return {
        "work": "rigveda",
        "mandala": mandala,
        "language": "sa",
        "license": "odbl",
        "attribution": DHARMICDATA_ATTR,
        "source_url": DHARMICDATA_REPO,
        "suktas": suktas,
    }


def compact_yajurveda_samhita(payload: list[Any] | dict[str, Any]) -> dict[str, Any]:
    if isinstance(payload, dict) and payload.get("work") == "yajurveda":
        return payload
    rows = payload if isinstance(payload, list) else payload.get("adhyayas") or []
    adhyayas: list[dict[str, Any]] = []
    samhita = "vajasaneyi-madhyandina"
    for item in rows:
        if not isinstance(item, dict):
            continue
        n = item.get("adhyaya")
        if n is None:
            continue
        samhita = item.get("samhita") or samhita
        heading, verses = split_sukta_text(item.get("text") or "")
        if not verses:
            continue
        adhyayas.append(
            {
                "adhyaya": int(n),
                "heading": heading,
                "verses": [{"n": vn, "text": t} for vn, t in verses],
            }
        )
    adhyayas.sort(key=lambda a: a["adhyaya"])
    return {
        "work": "yajurveda",
        "samhita": samhita,
        "language": "sa",
        "license": "odbl",
        "attribution": DHARMICDATA_ATTR,
        "source_url": DHARMICDATA_REPO,
        "adhyayas": adhyayas,
    }


def compact_atharvaveda_kaanda(payload: list[Any] | dict[str, Any]) -> dict[str, Any]:
    if isinstance(payload, dict) and payload.get("work") == "atharvaveda":
        return payload
    rows = payload if isinstance(payload, list) else payload.get("suktas") or []
    suktas: list[dict[str, Any]] = []
    kaanda = None
    for item in rows:
        if not isinstance(item, dict):
            continue
        book = item.get("kaanda") or item.get("kanda")
        hymn = item.get("sukta") or item.get("hymn")
        if book is None or hymn is None:
            continue
        kaanda = int(str(book).translate(_DEV_DIGITS))
        if item.get("verses"):
            verses = [(int(v["n"]), v["text"]) for v in item["verses"] if v.get("text")]
            heading = item.get("heading")
        else:
            heading, verses = split_sukta_text(item.get("text") or "")
        if not verses:
            continue
        suktas.append(
            {
                "sukta": int(str(hymn).translate(_DEV_DIGITS)),
                "heading": heading,
                "verses": [{"n": n, "text": t} for n, t in verses],
            }
        )
    suktas.sort(key=lambda s: s["sukta"])
    return {
        "work": "atharvaveda",
        "kaanda": kaanda,
        "language": "sa",
        "license": "odbl",
        "attribution": DHARMICDATA_ATTR,
        "source_url": DHARMICDATA_REPO,
        "suktas": suktas,
    }


def _roman_or_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value if value > 0 else None
    text = str(value).strip()
    match = _BOOK_TOKEN.search(text)
    token = match.group(1) if match else text
    return parse_int_or_roman(token)


def compact_samaveda(payload: dict[str, Any] | list[Any]) -> dict[str, Any]:
    if isinstance(payload, dict) and payload.get("work") == "samaveda":
        return payload
    rows = payload.get("verses") if isinstance(payload, dict) else payload
    hymns: dict[tuple[int, int, int, int], dict[str, Any]] = {}
    for item in rows or []:
        if not isinstance(item, dict):
            continue
        text = (item.get("text") or "").strip()
        verse_n = item.get("verse")
        if not text or verse_n is None:
            continue
        part_raw = str(item.get("part") or "1").upper()
        part = 2 if "SECOND" in part_raw or part_raw.strip() in {"2", "II"} else 1
        book = _roman_or_int(item.get("book")) or 1
        chapter = _roman_or_int(item.get("chapter")) or 1
        hymn = int(item.get("hymn") or 1)
        key = (part, book, chapter, hymn)
        slot = hymns.setdefault(
            key,
            {
                "part": part,
                "book": book,
                "chapter": chapter,
                "hymn": hymn,
                "heading": (item.get("deity") or item.get("section") or "").strip() or None,
                "verses": [],
            },
        )
        slot["verses"].append({"n": int(verse_n), "text": text})
    packed = []
    for key in sorted(hymns):
        item = hymns[key]
        item["verses"].sort(key=lambda v: v["n"])
        packed.append(item)
    return {
        "work": "samaveda",
        "language": "en",
        "license": "odbl",
        "edition": "Griffith, DharmicData",
        "attribution": DHARMICDATA_ATTR,
        "source_url": DHARMICDATA_REPO,
        "hymns": packed,
    }


def render_gita_chapter(compact: dict[str, Any]) -> str:
    chapter = compact.get("chapter")
    lines = [f"CHAPTER {chapter} — Sanskrit (DharmicData)", ""]
    for item in compact.get("verses") or []:
        lines.append(f"{chapter}.{item['verse']} {item['text']}")
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def render_rigveda_hymn(mandala: int, sukta: dict[str, Any]) -> str:
    heading = sukta.get("heading") or ""
    title = f"HYMN {mandala}.{sukta['sukta']}"
    if heading:
        title += f" — {heading}"
    lines = [title, ""]
    for item in sukta.get("verses") or []:
        lines.append(f"{item['n']}. {item['text']}")
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def render_yajurveda_adhyaya(compact: dict[str, Any], adhyaya: dict[str, Any]) -> str:
    n = adhyaya["adhyaya"]
    heading = adhyaya.get("heading") or compact.get("samhita") or ""
    title = f"ADHYAYA {n}"
    if heading:
        title += f" — {heading}"
    lines = [title, ""]
    for item in adhyaya.get("verses") or []:
        lines.append(f"{item['n']}. {item['text']}")
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def render_samaveda_hymn(hymn: dict[str, Any]) -> str:
    heading = hymn.get("heading") or ""
    title = f"HYMN {hymn['part']}.{hymn['book']}.{hymn['chapter']}.{hymn['hymn']}"
    if heading:
        title += f" — {heading}"
    lines = [title, ""]
    for item in hymn.get("verses") or []:
        lines.append(f"{item['n']}. {item['text']}")
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def load_json(path: Any) -> Any:
    from pathlib import Path

    return json.loads(Path(path).read_text(encoding="utf-8"))


def expand_structured_payload(
    payload: Any,
    *,
    source: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Expande JSON DharmicData/compacto em registros de corpus (um por hino ou capítulo)."""
    src = source or {}
    if _is_gita_payload(payload):
        compact = compact_gita_chapter(payload)
        text = render_gita_chapter(compact)
        chapter = compact.get("chapter")
        url = f"{DHARMICDATA_REPO}/blob/main/SrimadBhagvadGita/bhagavad_gita_chapter_{chapter}.json"
        return [
            _record(
                _merged(src, compact),
                text=text,
                title=f"Bhagavad-gītā {chapter} (Sanskrit, DharmicData)",
                url=url,
                tradition=src.get("tradition") or "vaishnava",
                language="sa",
            )
        ]
    if _is_rigveda_payload(payload):
        compact = compact_rigveda_mandala(payload)
        mandala = int(compact["mandala"])
        edition = compact.get("edition") or "Sanskrit, DharmicData"
        records = []
        for sukta in compact.get("suktas") or []:
            hymn = sukta["sukta"]
            records.append(
                _record(
                    _merged(src, compact),
                    text=render_rigveda_hymn(mandala, sukta),
                    title=f"Rigveda RV {mandala}.{hymn} ({edition})",
                    url=_rigveda_url(compact, mandala, hymn),
                    tradition=src.get("tradition") or "vedic",
                    language=compact.get("language") or "sa",
                )
            )
        return records
    if _is_yajurveda_payload(payload):
        compact = compact_yajurveda_samhita(payload)
        records = []
        for adhyaya in compact.get("adhyayas") or []:
            n = adhyaya["adhyaya"]
            records.append(
                _record(
                    _merged(src, compact),
                    text=render_yajurveda_adhyaya(compact, adhyaya),
                    title=f"Yajurveda VS {n} (Sanskrit, DharmicData)",
                    url=(
                        f"{DHARMICDATA_REPO}/blob/main/Yajurveda/"
                        f"vajasneyi_madhyadina_samhita.json#adhyaya-{n}"
                    ),
                    tradition=src.get("tradition") or "vedic",
                    language="sa",
                )
            )
        return records
    if _is_atharvaveda_payload(payload):
        compact = compact_atharvaveda_kaanda(payload)
        kaanda = int(compact["kaanda"])
        records = []
        for sukta in compact.get("suktas") or []:
            hymn = sukta["sukta"]
            records.append(
                _record(
                    _merged(src, compact),
                    text=render_rigveda_hymn(kaanda, sukta),
                    title=f"Atharvaveda AV {kaanda}.{hymn} (Sanskrit, DharmicData)",
                    url=(
                        f"{DHARMICDATA_REPO}/blob/main/AtharvaVeda/"
                        f"atharvaveda_kaanda_{kaanda}.json#sukta-{hymn}"
                    ),
                    tradition=src.get("tradition") or "vedic",
                    language="sa",
                )
            )
        return records
    if _is_samaveda_payload(payload):
        compact = compact_samaveda(payload)
        records = []
        for hymn in compact.get("hymns") or []:
            part, book, chapter, n = hymn["part"], hymn["book"], hymn["chapter"], hymn["hymn"]
            records.append(
                _record(
                    _merged(src, compact),
                    text=render_samaveda_hymn(hymn),
                    title=(
                        f"Sāmaveda SV {part}.{book}.{chapter}.{n} "
                        f"({compact.get('edition') or 'Griffith, DharmicData'})"
                    ),
                    url=(
                        f"{DHARMICDATA_REPO}/blob/main/Samaveda/Samaveda.json"
                        f"#p{part}-b{book}-c{chapter}-h{n}"
                    ),
                    tradition=src.get("tradition") or "vedic",
                    language="en",
                )
            )
        return records
    return []


def expand_structured_file(path: Any, source: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    return expand_structured_payload(load_json(path), source=source)


def _is_gita_payload(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False
    return payload.get("work") == "bhagavad-gita" or isinstance(
        payload.get("BhagavadGitaChapter"), list
    )


def _is_rigveda_payload(payload: Any) -> bool:
    if isinstance(payload, dict) and payload.get("work") == "rigveda":
        return True
    if isinstance(payload, list) and payload and isinstance(payload[0], dict):
        return payload[0].get("veda") == "rigveda" or (
            "mandala" in payload[0] and "sukta" in payload[0]
        )
    return False


def _is_yajurveda_payload(payload: Any) -> bool:
    if isinstance(payload, dict) and payload.get("work") == "yajurveda":
        return True
    if isinstance(payload, list) and payload and isinstance(payload[0], dict):
        return payload[0].get("veda") == "yajurveda" or "adhyaya" in payload[0]
    return False


def _is_atharvaveda_payload(payload: Any) -> bool:
    if isinstance(payload, dict) and payload.get("work") == "atharvaveda":
        return True
    if isinstance(payload, list) and payload and isinstance(payload[0], dict):
        first = payload[0]
        return first.get("veda") == "atharvaveda" or (
            "kaanda" in first and "sukta" in first
        )
    return False


def _is_samaveda_payload(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False
    if payload.get("work") == "samaveda":
        return True
    veda = str(payload.get("veda") or "").lower()
    return veda in {"samaveda", "sama-veda", "sāmaveda"} and isinstance(
        payload.get("verses"), list
    )


def looks_structured(path: Any) -> bool:
    from pathlib import Path

    path = Path(path)
    if path.suffix.lower() != ".json":
        return False
    try:
        payload = load_json(path)
    except (OSError, json.JSONDecodeError):
        return False
    return (
        _is_gita_payload(payload)
        or _is_rigveda_payload(payload)
        or _is_yajurveda_payload(payload)
        or _is_atharvaveda_payload(payload)
        or _is_samaveda_payload(payload)
    )


def _merged(src: dict[str, Any], compact: dict[str, Any]) -> dict[str, Any]:
    out = dict(src)
    if compact.get("license"):
        out["license"] = compact["license"]
    if compact.get("attribution"):
        out["attribution"] = compact["attribution"]
    return out


def _rigveda_url(compact: dict[str, Any], mandala: int, hymn: int) -> str:
    source = compact.get("source_url") or DHARMICDATA_REPO
    if "vedaweb" in source.lower() or "VedaWeb" in (compact.get("edition") or ""):
        return f"{VEDAWEB_ZURICH_XLSX}#{mandala:02d}.{hymn:03d}"
    return f"{DHARMICDATA_REPO}/blob/main/Rigveda/rigveda_mandala_{mandala}.json#sukta-{hymn}"


def _record(
    src: dict[str, Any],
    *,
    text: str,
    title: str,
    url: str,
    tradition: str,
    language: str,
) -> dict[str, Any]:
    fp = content_fingerprint(text)
    license_ = (src.get("license") or "odbl").lower()
    return {
        "id": stable_id(url, title, fp[:16]),
        "text": text,
        "source_url": url,
        "title": title,
        "tradition": tradition.lower(),
        "language": language.lower(),
        "license": license_,
        "retrieved_at": utc_now_iso(),
        "fingerprint": fp,
        "char_count": len(text),
        "attribution": src.get("attribution") or DHARMICDATA_ATTR,
    }


def compact_zurich_rows(rows: list[tuple[str, str, str]]) -> dict[int, dict[str, Any]]:
    """Agrupa linhas Zürich (stelle, pada, texto) em compactos por mandala."""
    padas: dict[tuple[int, int, int], dict[str, str]] = defaultdict(dict)
    for stelle, pada, text in rows:
        parsed = _parse_stelle(stelle)
        if parsed is None or not text:
            continue
        book, hymn, verse = parsed
        letter = (pada or "a").strip().lower()[:1] or "a"
        cleaned = text.strip().rstrip("|‖ ").strip()
        if letter not in padas[(book, hymn, verse)]:
            padas[(book, hymn, verse)][letter] = cleaned

    by_mandala: dict[int, dict[int, dict[str, Any]]] = defaultdict(dict)
    for (book, hymn, verse), pada_map in padas.items():
        joined = "\n".join(pada_map[k] for k in sorted(pada_map))
        slot = by_mandala[book].setdefault(hymn, {"sukta": hymn, "heading": None, "verses": {}})
        slot["verses"][verse] = joined

    compact_by_book: dict[int, dict[str, Any]] = {}
    for book, hymns in by_mandala.items():
        suktas = []
        for hymn in sorted(hymns):
            verses_map = hymns[hymn]["verses"]
            suktas.append(
                {
                    "sukta": hymn,
                    "heading": hymns[hymn].get("heading"),
                    "verses": [{"n": n, "text": verses_map[n]} for n in sorted(verses_map)],
                }
            )
        compact_by_book[book] = {
            "work": "rigveda",
            "mandala": book,
            "language": "sa",
            "script": "iso-15919",
            "license": "cc-by",
            "edition": "VedaWeb Zürich, ISO-15919",
            "attribution": VEDAWEB_ATTR,
            "source_url": VEDAWEB_REPO,
            "suktas": suktas,
        }
    return compact_by_book


def _parse_stelle(value: str) -> tuple[int, int, int] | None:
    parts = re.findall(r"\d+", value or "")
    if len(parts) < 3:
        return None
    book, hymn, verse = (int(parts[0]), int(parts[1]), int(parts[2]))
    if book < 1 or hymn < 1 or verse < 1:
        return None
    return book, hymn, verse
