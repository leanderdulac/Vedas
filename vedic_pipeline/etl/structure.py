"""Unidades canônicas: verso, hino e sūtra.

Parsers cobertos:
- Ṛgveda Griffith (página por hino em sacred-texts e fixture multi-hino)
- Bhagavad-gītā com numeração explícita (fixture) e Arnold (capítulo + âncoras)
- Yoga-sūtra numerado (pāda.sūtra)

Sem estrutura reconhecida o chunking por caracteres permanece o fallback.
Nunca inventa número de śloka: sem âncora, a Gītā Arnold cita só o capítulo.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

WORK_RIGVEDA = "rigveda"
WORK_GITA = "bhagavad-gita"
WORK_YOGA = "yoga-sutra"
WORK_YAJURVEDA = "yajurveda"
WORK_ATHARVAVEDA = "atharvaveda"
WORK_SAMAVEDA = "samaveda"

WORK_PREFIX = {
    WORK_RIGVEDA: "RV",
    WORK_GITA: "BG",
    WORK_YOGA: "YS",
    WORK_YAJURVEDA: "VS",
    WORK_ATHARVAVEDA: "AV",
    WORK_SAMAVEDA: "SV",
}
PREFIX_WORK = {prefix: work for work, prefix in WORK_PREFIX.items()}

# Āncoras de śloka na tradução em verso de Arnold (frases distintivas, não o cânone inteiro).
ARNOLD_VERSE_ANCHORS: dict[int, tuple[tuple[int, str], ...]] = {
    2: (
        (11, "thou grievest where no grief should be"),
        (12, "nor i, nor thou, nor any one of these"),
        (13, "infancy and youth and age"),
        (19, "lo! i have slain a man"),
        (20, "never the spirit was born"),
        (22, "as when one layeth"),
        (23, "weapons reach not the life"),
        (47, "let right deeds be"),
    ),
    3: (
        (4, "no man shall 'scape from act"),
        (5, "no jot of time, at any time"),
        (8, "work is more excellent than idleness"),
        (19, "thy task prescribed"),
        (21, "what the wise choose"),
        (35, "this is better, that one do"),
    ),
    4: (
        (7, "when righteousness"),
        (8, "succouring the good"),
        (37, "the flame of knowledge wastes"),
    ),
    6: (
        (16, "too much fasts"),
        (17, "moderate in eating"),
        (19, "a lamp burns sheltered from the wind"),
    ),
    9: (
        (26, "leaf, a flower, a fruit"),
    ),
    18: (
        (66, "fly unto me"),
        (78, "where krishna is"),
    ),
}

_HYMN_HEADER = re.compile(
    r"^HYMN\s+"
    r"(?:([IVXLCDM]+|\d+)\.(\d+)|([IVXLCDM]+|\d+)\.)"
    r"\s*(?:[—–:-]\s*)?(.*)$",
    re.IGNORECASE,
)
_VERSE_START = re.compile(
    r"^(\d{1,3})(?:[.)]|)\s+(.*)$",
)
_RV_TITLE = re.compile(r"\bRV\s+(\d{1,2})\.(\d{1,3})\b", re.IGNORECASE)
_RV_URL = re.compile(r"rv(\d{2})(\d{3})\.html?", re.IGNORECASE)
_AV_TITLE = re.compile(r"\bAV\s+(\d{1,2})\.(\d{1,3})\b", re.IGNORECASE)
_VS_TITLE = re.compile(r"\bVS\s+(\d{1,2})\b", re.IGNORECASE)
_SV_TITLE = re.compile(
    r"\bSV\s+(\d{1,2})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})\b",
    re.IGNORECASE,
)
_CHAPTER = re.compile(
    r"^\s*CHAPTER\s+([IVXLCDM]+|\d+)\b(?:\s*[.—–:-]+\s*(.*))?",
    re.IGNORECASE,
)
_ADHYAYA = re.compile(
    r"^\s*ADHYAYA\s+([IVXLCDM]+|\d+)\b(?:\s*[.—–:-]+\s*(.*))?",
    re.IGNORECASE,
)
_CHAPTER_END = re.compile(
    r"HERE\s+END(?:ETH|S)\s+CHAPTER\s+([IVXLCDM]+|\d+)",
    re.IGNORECASE,
)
_GITA_FULL = re.compile(r"^(\d{1,2})\.(\d{1,3})\s+(.*)$", re.DOTALL)
_GITA_NUM = re.compile(r"^(\d{1,3})[.)]\s+(.*)$", re.DOTALL)
_SUTRA = re.compile(r"^(\d)\.(\d{1,3})\s+(.*)$")
_PADA_HEADER = re.compile(
    r"^(SAMADHI|SADHANA|VIBHUTI|KAIVALYA)\s+PADA\b",
    re.IGNORECASE,
)
_NOTES = re.compile(r"^notes?\s+for\s+pipeline\b", re.IGNORECASE)
_GUTENBERG_END = re.compile(r"^\*\*\*\s*END OF (THE )?PROJECT GUTENBERG", re.IGNORECASE)
_SKIP_LINE = re.compile(
    r"^(click|next hymn|previous hymn|home\b|index\b|sacred[- ]texts)",
    re.IGNORECASE,
)
@dataclass(frozen=True)
class TextUnit:
    work: str
    book: int | None
    hymn: int | None
    verse: int | None
    verse_id: str
    locator: str
    text: str
    heading: str | None = None
    passage: int | None = None

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "work": self.work,
            "book": self.book,
            "hymn": self.hymn,
            "verse": self.verse,
            "verse_id": self.verse_id,
            "locator": self.locator,
            "heading": self.heading,
            "text": self.text,
        }

    def to_chunk_fields(self, verse_end: int | None = None, locator: str | None = None) -> dict[str, Any]:
        return {
            "work": self.work,
            "book": self.book,
            "hymn": self.hymn,
            "verse": self.verse,
            "verse_end": verse_end if verse_end is not None else self.verse,
            "verse_id": self.verse_id,
            "locator": locator or self.locator,
            "heading": self.heading,
        }


def parse_int_or_roman(token: str | None) -> int | None:
    if not token:
        return None
    raw = token.strip().upper().replace(" ", "")
    if not raw:
        return None
    if raw.isdigit():
        value = int(raw)
        return value if value > 0 else None
    roman = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
    if any(ch not in roman for ch in raw):
        return None
    total = 0
    prev = 0
    for ch in reversed(raw):
        value = roman[ch]
        if value < prev:
            total -= value
        else:
            total += value
            prev = value
    return total if total > 0 else None


def pack_samaveda(part: int, book: int, chapter: int, hymn: int) -> tuple[int, int]:
    return part * 100 + book, chapter * 1000 + hymn


def unpack_samaveda(book: int | None, hymn: int | None) -> tuple[int, int, int, int]:
    packed_book = book or 0
    packed_hymn = hymn or 0
    part, gbook = divmod(packed_book, 100)
    chapter, hymn_n = divmod(packed_hymn, 1000)
    return part, gbook, chapter, hymn_n


def format_verse_id(
    work: str,
    book: int | None,
    hymn: int | None = None,
    verse: int | None = None,
    passage: int | None = None,
) -> str:
    prefix = WORK_PREFIX.get(work, work.upper()[:2])
    if work in {WORK_RIGVEDA, WORK_ATHARVAVEDA}:
        parts = [prefix, str(book or 0), str(hymn or 0)]
        if verse is not None:
            parts.append(str(verse))
        return ".".join(parts)
    if work == WORK_SAMAVEDA:
        part, gbook, chapter, hymn_n = unpack_samaveda(book, hymn)
        parts = [prefix, str(part), str(gbook), str(chapter), str(hymn_n)]
        if verse is not None:
            parts.append(str(verse))
        return ".".join(parts)
    if work in {WORK_GITA, WORK_YAJURVEDA}:
        parts = [prefix, str(book or 0)]
        if verse is not None:
            parts.append(str(verse))
        elif passage is not None:
            parts.append(f"p{passage}")
        return ".".join(parts)
    if work == WORK_YOGA:
        return f"{prefix}.{book or 0}.{verse or 0}"
    return prefix


def format_locator(
    work: str,
    book: int | None,
    hymn: int | None = None,
    verse: int | None = None,
    verse_end: int | None = None,
) -> str:
    prefix = WORK_PREFIX.get(work, work.upper()[:2])
    if work in {WORK_RIGVEDA, WORK_ATHARVAVEDA}:
        base = f"{prefix} {book}.{hymn}"
        if verse is None:
            return base
        if verse_end is not None and verse_end != verse:
            return f"{base}.{verse}–{verse_end}"
        return f"{base}.{verse}"
    if work == WORK_SAMAVEDA:
        part, gbook, chapter, hymn_n = unpack_samaveda(book, hymn)
        base = f"{prefix} {part}.{gbook}.{chapter}.{hymn_n}"
        if verse is None:
            return base
        if verse_end is not None and verse_end != verse:
            return f"{base}.{verse}–{verse_end}"
        return f"{base}.{verse}"
    if work in {WORK_GITA, WORK_YAJURVEDA, WORK_YOGA}:
        if verse is None:
            return f"{prefix} {book}"
        if verse_end is not None and verse_end != verse:
            return f"{prefix} {book}.{verse}–{verse_end}"
        return f"{prefix} {book}.{verse}"
    return prefix


def detect_work(
    title: str | None = None,
    url: str | None = None,
    tradition: str | None = None,
) -> str | None:
    blob = f"{title or ''} {url or ''}".lower()
    if "atharva" in blob or _AV_TITLE.search(blob):
        return WORK_ATHARVAVEDA
    if (
        "yajur" in blob
        or "vajasaneyi" in blob
        or "vajasneyi" in blob
        or "mādhyandina" in blob
        or "madhyandina" in blob
        or _VS_TITLE.search(f"{title or ''} {url or ''}")
    ):
        return WORK_YAJURVEDA
    if (
        "samaveda" in blob
        or "sama-veda" in blob
        or "sāmaveda" in blob
        or "hymns of the sama" in blob
        or _SV_TITLE.search(f"{title or ''} {url or ''}")
    ):
        return WORK_SAMAVEDA
    if (
        "rigveda" in blob
        or "rig veda" in blob
        or "ṛgveda" in blob
        or "/hin/rigveda/" in blob
        or _RV_TITLE.search(blob)
        or _RV_URL.search(blob)
    ):
        return WORK_RIGVEDA
    if ("yoga" in blob and ("sutra" in blob or "sūtra" in blob or "patanjali" in blob or "patañjali" in blob)) or (
        (tradition or "").lower() == "yoga" and "sutra" in blob
    ):
        return WORK_YOGA
    if "bhagavad" in blob or "song celestial" in blob or re.search(r"\bg[iī]t[aā]\b", blob):
        return WORK_GITA
    return None


def rigveda_ref_from_meta(title: str | None, url: str | None) -> tuple[int | None, int | None]:
    blob_title = title or ""
    blob_url = url or ""
    match = _RV_TITLE.search(blob_title) or _RV_TITLE.search(blob_url)
    if match:
        return int(match.group(1)), int(match.group(2))
    match = _RV_URL.search(blob_url)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None, None


def _is_plausible_verse_number(number: int, *, max_n: int = 200) -> bool:
    return 1 <= number <= max_n


def _clean_lines(text: str) -> list[str]:
    return text.replace("\r\n", "\n").replace("\r", "\n").split("\n")


def _should_stop(line: str) -> bool:
    stripped = line.strip()
    return bool(_NOTES.match(stripped) or _GUTENBERG_END.match(stripped))


def parse_rigveda(
    text: str,
    *,
    book: int | None = None,
    hymn: int | None = None,
    work: str = WORK_RIGVEDA,
) -> list[TextUnit]:
    current_book, current_hymn = book, hymn
    heading: str | None = None
    verse_no: int | None = None
    verse_lines: list[str] = []
    prose_lines: list[str] = []
    units: list[TextUnit] = []
    body_started = False

    def make_unit(*, verse: int | None, body: str) -> TextUnit | None:
        cleaned = body.strip()
        if not cleaned or current_book is None or current_hymn is None:
            return None
        return TextUnit(
            work=work,
            book=current_book,
            hymn=current_hymn,
            verse=verse,
            verse_id=format_verse_id(work, current_book, current_hymn, verse),
            locator=format_locator(work, current_book, current_hymn, verse),
            text=cleaned,
            heading=heading,
        )

    def flush_verse() -> None:
        nonlocal verse_no, verse_lines
        unit = make_unit(verse=verse_no, body="\n".join(verse_lines))
        if unit:
            units.append(unit)
        verse_no = None
        verse_lines = []

    def flush_prose() -> None:
        nonlocal prose_lines
        if verse_no is not None:
            prose_lines = []
            return
        unit = make_unit(verse=None, body="\n".join(prose_lines))
        if unit:
            units.append(unit)
        prose_lines = []

    for raw in _clean_lines(text):
        stripped = raw.strip()
        if _should_stop(stripped):
            break
        if not stripped or _SKIP_LINE.match(stripped):
            continue

        header = _HYMN_HEADER.match(stripped)
        if header:
            flush_verse()
            flush_prose()
            if header.group(1) and header.group(2):
                current_book = parse_int_or_roman(header.group(1))
                current_hymn = parse_int_or_roman(header.group(2))
            else:
                hymn_token = parse_int_or_roman(header.group(3))
                if current_book is None and hymn_token is not None and hymn is None:
                    # Documento de um hino: "HYMN I. Agni" com livro vindo do título.
                    current_hymn = hymn_token
                elif hymn_token is not None:
                    current_hymn = hymn_token
            rest = (header.group(4) or "").strip(" —–-.").strip()
            heading = rest or heading
            body_started = True
            continue

        verse_match = _VERSE_START.match(stripped)
        if (
            verse_match
            and current_book is not None
            and current_hymn is not None
            and _is_plausible_verse_number(int(verse_match.group(1)))
            and (
                stripped[len(verse_match.group(1))] in ".)"
                or (verse_match.group(2)[:1].isalpha() or verse_match.group(2)[:1] in "\"“‘")
            )
        ):
            flush_verse()
            flush_prose()
            verse_no = int(verse_match.group(1))
            rest = verse_match.group(2).strip()
            verse_lines = [rest] if rest else []
            body_started = True
            continue

        if verse_no is not None:
            verse_lines.append(stripped)
        elif body_started and current_book is not None and current_hymn is not None:
            prose_lines.append(stripped)

    flush_verse()
    flush_prose()
    return units


def _gita_anchor_verse(chapter: int, text: str) -> int | None:
    blob = text.lower()
    for verse, needle in ARNOLD_VERSE_ANCHORS.get(chapter, ()):
        if needle in blob:
            return verse
    return None


def parse_gita(text: str) -> list[TextUnit]:
    chapter: int | None = None
    heading: str | None = None
    buf: list[str] = []
    units: list[TextUnit] = []
    passage = 0
    numbered_in_chapter = False

    def make_unit(*, verse: int | None, body: str, passage_n: int | None) -> TextUnit | None:
        cleaned = body.strip()
        if not cleaned or chapter is None:
            return None
        return TextUnit(
            work=WORK_GITA,
            book=chapter,
            hymn=None,
            verse=verse,
            verse_id=format_verse_id(WORK_GITA, chapter, verse=verse, passage=passage_n if verse is None else None),
            locator=format_locator(WORK_GITA, chapter, verse=verse),
            text=cleaned,
            heading=heading,
            passage=passage_n,
        )

    def flush_buffer() -> None:
        nonlocal buf, passage, numbered_in_chapter
        body = "\n".join(buf).strip()
        buf = []
        if not body or chapter is None:
            return
        full = _GITA_FULL.match(body)
        if full:
            chapter_n = int(full.group(1))
            verse_n = int(full.group(2))
            rest = full.group(3).strip()
            unit = TextUnit(
                work=WORK_GITA,
                book=chapter_n,
                hymn=None,
                verse=verse_n,
                verse_id=format_verse_id(WORK_GITA, chapter_n, verse=verse_n),
                locator=format_locator(WORK_GITA, chapter_n, verse=verse_n),
                text=rest or body,
                heading=heading,
            )
            units.append(unit)
            numbered_in_chapter = True
            return
        num = _GITA_NUM.match(body)
        if num and _is_plausible_verse_number(int(num.group(1)), max_n=80):
            verse_n = int(num.group(1))
            rest = num.group(2).strip()
            unit = make_unit(verse=verse_n, body=rest or body, passage_n=None)
            if unit:
                units.append(unit)
                numbered_in_chapter = True
            return
        anchored = _gita_anchor_verse(chapter, body)
        if numbered_in_chapter and anchored is None:
            return
        passage += 1
        unit = make_unit(verse=anchored, body=body, passage_n=None if anchored else passage)
        if unit:
            units.append(unit)

    for raw in _clean_lines(text):
        stripped = raw.strip()
        if _should_stop(stripped):
            break
        end = _CHAPTER_END.search(stripped)
        if end:
            flush_buffer()
            chapter = None
            continue
        header = _CHAPTER.match(stripped)
        if header:
            flush_buffer()
            chapter = parse_int_or_roman(header.group(1))
            rest = (header.group(2) or "").strip(" —–-")
            heading = rest or heading
            passage = 0
            numbered_in_chapter = False
            continue
        if not stripped:
            flush_buffer()
            continue
        inline_full = _GITA_FULL.match(stripped)
        if inline_full:
            flush_buffer()
            chapter = int(inline_full.group(1))
            buf = [stripped]
            continue
        if chapter is None:
            continue
        inline_num = _GITA_NUM.match(stripped)
        if inline_num and (
            numbered_in_chapter
            or (
                _is_plausible_verse_number(int(inline_num.group(1)), max_n=80)
                and len(inline_num.group(2)) > 8
            )
        ):
            flush_buffer()
            buf = [stripped]
            continue
        buf.append(stripped)

    flush_buffer()
    return units


def parse_yoga_sutra(text: str) -> list[TextUnit]:
    heading: str | None = None
    units: list[TextUnit] = []
    current: TextUnit | None = None
    tail: list[str] = []

    def flush() -> None:
        nonlocal current, tail
        if current is None:
            tail = []
            return
        extra = "\n".join(tail).strip()
        text_body = current.text if not extra else f"{current.text}\n{extra}"
        units.append(
            TextUnit(
                work=current.work,
                book=current.book,
                hymn=None,
                verse=current.verse,
                verse_id=current.verse_id,
                locator=current.locator,
                text=text_body,
                heading=current.heading,
            )
        )
        current = None
        tail = []

    for raw in _clean_lines(text):
        stripped = raw.strip()
        if _should_stop(stripped):
            break
        pada_header = _PADA_HEADER.match(stripped)
        if pada_header:
            flush()
            heading = stripped.title()
            continue
        sutra = _SUTRA.match(stripped)
        if sutra:
            flush()
            book_n = int(sutra.group(1))
            verse_n = int(sutra.group(2))
            rest = sutra.group(3).strip()
            current = TextUnit(
                work=WORK_YOGA,
                book=book_n,
                hymn=None,
                verse=verse_n,
                verse_id=format_verse_id(WORK_YOGA, book_n, verse=verse_n),
                locator=format_locator(WORK_YOGA, book_n, verse=verse_n),
                text=rest,
                heading=heading,
            )
            continue
        if current is not None and stripped:
            tail.append(stripped)

    flush()
    return units


def parse_numbered_verses(
    text: str,
    *,
    work: str,
    book: int | None = None,
    hymn: int | None = None,
    header_re: re.Pattern[str] | None = None,
    max_n: int = 400,
) -> list[TextUnit]:
    heading: str | None = None
    current_book = book
    current_hymn = hymn
    buf: list[str] = []
    units: list[TextUnit] = []

    def emit(verse_n: int, body: str) -> None:
        cleaned = body.strip()
        if not cleaned or current_book is None:
            return
        units.append(
            TextUnit(
                work=work,
                book=current_book,
                hymn=current_hymn,
                verse=verse_n,
                verse_id=format_verse_id(work, current_book, current_hymn, verse_n),
                locator=format_locator(work, current_book, current_hymn, verse_n),
                text=cleaned,
                heading=heading,
            )
        )

    def flush() -> None:
        nonlocal buf
        body = "\n".join(buf).strip()
        buf = []
        if not body:
            return
        full = _GITA_FULL.match(body)
        if full:
            emit(int(full.group(2)), full.group(3).strip() or body)
            return
        num = _GITA_NUM.match(body)
        if num and _is_plausible_verse_number(int(num.group(1)), max_n=max_n):
            emit(int(num.group(1)), num.group(2).strip() or body)

    for raw in _clean_lines(text):
        stripped = raw.strip()
        if _should_stop(stripped):
            break
        if header_re is not None:
            header = header_re.match(stripped)
            if header:
                flush()
                parsed = parse_int_or_roman(header.group(1))
                if parsed is not None:
                    current_book = parsed
                rest = (header.group(2) or "").strip(" —–-") if header.lastindex and header.lastindex >= 2 else ""
                heading = rest or heading
                continue
        if not stripped:
            flush()
            continue
        inline_full = _GITA_FULL.match(stripped)
        inline_num = _GITA_NUM.match(stripped)
        if inline_full or (
            inline_num and _is_plausible_verse_number(int(inline_num.group(1)), max_n=max_n)
        ):
            flush()
            buf = [stripped]
            continue
        if current_book is None:
            continue
        buf.append(stripped)

    flush()
    return units


def yajurveda_ref_from_meta(title: str | None, url: str | None) -> int | None:
    blob = f"{title or ''} {url or ''}"
    match = _VS_TITLE.search(blob)
    if match:
        return int(match.group(1))
    return None


def atharvaveda_ref_from_meta(title: str | None, url: str | None) -> tuple[int | None, int | None]:
    blob = f"{title or ''} {url or ''}"
    match = _AV_TITLE.search(blob)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None, None


def samaveda_ref_from_meta(
    title: str | None, url: str | None
) -> tuple[int, int, int, int] | None:
    blob = f"{title or ''} {url or ''}"
    match = _SV_TITLE.search(blob)
    if not match:
        return None
    return (
        int(match.group(1)),
        int(match.group(2)),
        int(match.group(3)),
        int(match.group(4)),
    )


def parse_document_units(record: dict[str, Any] | None = None, **meta: Any) -> list[TextUnit]:
    rec = dict(record or {})
    rec.update({k: v for k, v in meta.items() if v is not None})
    title = rec.get("title")
    url = rec.get("source_url") or rec.get("url")
    tradition = rec.get("tradition")
    text = rec.get("text") or ""
    work = rec.get("work") or detect_work(title, url, tradition)
    if not work or not text.strip():
        return []
    if work == WORK_RIGVEDA:
        book, hymn = rigveda_ref_from_meta(title, url)
        return parse_rigveda(text, book=book, hymn=hymn)
    if work == WORK_ATHARVAVEDA:
        book, hymn = atharvaveda_ref_from_meta(title, url)
        return parse_rigveda(text, book=book, hymn=hymn, work=WORK_ATHARVAVEDA)
    if work == WORK_YAJURVEDA:
        return parse_numbered_verses(
            text,
            work=WORK_YAJURVEDA,
            book=yajurveda_ref_from_meta(title, url),
            header_re=_ADHYAYA,
            max_n=500,
        )
    if work == WORK_SAMAVEDA:
        ref = samaveda_ref_from_meta(title, url)
        if not ref:
            return []
        packed_book, packed_hymn = pack_samaveda(*ref)
        return parse_numbered_verses(
            text,
            work=WORK_SAMAVEDA,
            book=packed_book,
            hymn=packed_hymn,
        )
    if work == WORK_GITA:
        return parse_gita(text)
    if work == WORK_YOGA:
        return parse_yoga_sutra(text)
    return []


def group_units(units: list[TextUnit], max_chars: int = 900, max_verses: int = 2) -> list[list[TextUnit]]:
    groups: list[list[TextUnit]] = []
    current: list[TextUnit] = []
    max_chars = max(200, max_chars)
    max_verses = max(1, max_verses)

    def compatible(prev: TextUnit, nxt: TextUnit) -> bool:
        if prev.work != nxt.work or prev.book != nxt.book or prev.hymn != nxt.hymn:
            return False
        if prev.verse is None or nxt.verse is None:
            return False
        return nxt.verse == prev.verse + 1

    for unit in units:
        if not current:
            current = [unit]
            continue
        joined = sum(len(item.text) for item in current) + len(unit.text)
        if compatible(current[-1], unit) and len(current) < max_verses and joined <= max_chars:
            current.append(unit)
        else:
            groups.append(current)
            current = [unit]
    if current:
        groups.append(current)
    return groups


def format_chunk_text(units: list[TextUnit]) -> str:
    parts: list[str] = []
    for unit in units:
        parts.append(f"[{unit.locator}] {unit.text}")
    return "\n\n".join(parts)


def parse_verse_id(verse_id: str) -> dict[str, Any] | None:
    """Interpreta RV.10.129.1 / BG.2.47 / VS.1.1 / AV.1.1.1 / SV.1.1.1.1.1."""
    raw = (verse_id or "").strip()
    if not raw:
        return None
    parts = raw.split(".")
    prefix = parts[0].upper()
    work = PREFIX_WORK.get(prefix)
    if not work or len(parts) < 2:
        return None
    nums: list[int] = []
    for token in parts[1:]:
        if token.isdigit():
            nums.append(int(token))
        elif token[:1].lower() == "p" and token[1:].isdigit():
            nums.append(int(token[1:]))
        else:
            return None
    if not nums:
        return None
    if work in {WORK_RIGVEDA, WORK_ATHARVAVEDA}:
        if len(nums) < 2:
            return None
        return {
            "work": work,
            "prefix": prefix,
            "book": nums[0],
            "hymn": nums[1],
            "verse": nums[2] if len(nums) > 2 else None,
        }
    if work == WORK_SAMAVEDA:
        if len(nums) < 4:
            return None
        packed_book, packed_hymn = pack_samaveda(nums[0], nums[1], nums[2], nums[3])
        return {
            "work": work,
            "prefix": prefix,
            "book": packed_book,
            "hymn": packed_hymn,
            "verse": nums[4] if len(nums) > 4 else None,
        }
    return {
        "work": work,
        "prefix": prefix,
        "book": nums[0],
        "hymn": None,
        "verse": nums[1] if len(nums) > 1 else None,
    }


def citation_label(hit: dict[str, Any]) -> str:
    locator = (hit.get("locator") or "").strip()
    title = (hit.get("title") or "").strip()
    if locator and title:
        return f"{locator} — {title}"
    return locator or title or "fonte"
