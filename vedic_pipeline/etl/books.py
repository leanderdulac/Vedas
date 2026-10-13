"""Livros inteiros (scan OCR do archive.org, Gutenberg, páginas do Sacred Texts).

Um livro baixado vira vários registros do corpus, um por seção citável
(livro/capítulo, grupo de sūtras, página), cada um com título no formato
"Obra — localizador (tradutor, ano)". O título define a obra para o teto
por obra do modo de entidade (``search.entity.work_key`` corta em " — "),
então todos os volumes e seções de uma obra contam como uma só.

O manifesto escolhe o recorte em ``split`` (ver ``SPLITTERS``) e a limpeza
em ``clean``:
  - ``ocr``: tira cabeçalho corrido, número de página, linhas-lixo do OCR,
    índice de capítulos, hifenização de fim de linha e quebra de página no
    meio da frase; junta as linhas de cada parágrafo;
  - ``gutenberg``: tira o preâmbulo/licença do Project Gutenberg;
  - ``sacred-texts``: HTML do Sacred Texts sem navegação, número de página
    e âncoras de nota (o itálico das letras com diacrítico não quebra linha).
"""

from __future__ import annotations

import difflib
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vedic_pipeline.common.corpus import content_fingerprint, stable_id, utc_now_iso
from vedic_pipeline.common.text import normalize_whitespace

BOOK_ETL = "book"


@dataclass(frozen=True)
class Section:
    locator: str  # "Book 3, ch. 25", "sūtras 15–20", "Title II: Deposits"
    anchor: str  # fragmento estável para a URL do registro ("book-3-ch-25")
    text: str


# ------------------------------------------------------------- boilerplate

_GUTENBERG_START = (
    "*** START OF THE PROJECT GUTENBERG EBOOK",
    "*** START OF THIS PROJECT GUTENBERG EBOOK",
    "*END*THE SMALL PRINT",
)
_GUTENBERG_END = (
    "*** END OF THE PROJECT GUTENBERG EBOOK",
    "*** END OF THIS PROJECT GUTENBERG EBOOK",
    "End of the Project Gutenberg",
    "End of Project Gutenberg",
)


def strip_gutenberg_boilerplate(text: str) -> str:
    """Remove cabeçalho/rodapé típicos do Project Gutenberg."""
    start = 0
    for marker in _GUTENBERG_START:
        i = text.find(marker)
        if i >= 0:
            nl = text.find("\n", i)
            start = nl + 1 if nl >= 0 else i + len(marker)
            break
    end = len(text)
    for marker in _GUTENBERG_END:
        i = text.find(marker, start)
        if i >= 0:
            end = i
            break
    return text[start:end].strip()


_GOOGLE_NOTICE = re.compile(
    r"This is a digital copy of a book that was preserved.*?books\s*\.\s*google\s*\.\s*com/?\s*",
    re.S,
)


def strip_scan_boilerplate(text: str) -> str:
    """Aviso do Google Books que abre os scans do archive.org."""
    return _GOOGLE_NOTICE.sub("", text, count=1)


# ------------------------------------------------------------- limpeza OCR

_PAGE_NUMBER = re.compile(r"^[\s\[\](){}.,:;'\"*•-]*(?:\d{1,4}|[ivxlc]{1,7})[\s\[\](){}.,:;'\"*•-]*$", re.I)
_OCR_JUNK_CHARS = set("■□▪●♦◆¥§¤|{}<>~^\\_=+#@$%&")
_TOC_ENTRY_START = re.compile(r"^\s*Chapter\s+[IVXLCivxlc1 ]+[.,:]?\s*[-—–]")
_TOC_ENTRY_END = re.compile(r"\bP\s*[.*,]\s*[\dIl ]{1,5}\s*[.,]?\s*$")


def _letters(line: str) -> str:
    return re.sub(r"[^A-Za-z]", "", line).upper()


def is_running_head(line: str, heads: Iterable[str], *, cutoff: float = 0.72) -> bool:
    """Cabeçalho corrido do scan ("SRIMADBHAGAVATAM. 82", "i6 NARADA sOtRA,").

    Compara só as letras (sem espaço, número e pontuação) com cada cabeçalho
    esperado, com tolerância a erro de OCR ("SR I M A DB HAG AB AT AM").
    """
    stripped = line.strip()
    if not stripped or len(stripped) > 48:
        return False
    letters = _letters(stripped)
    if len(letters) < 4:
        return False
    for head in heads:
        target = _letters(head)
        if not target:
            continue
        if letters == target:
            return True
        ratio = difflib.SequenceMatcher(None, letters, target).ratio()
        if ratio >= cutoff and abs(len(letters) - len(target)) <= max(3, len(target) // 3):
            return True
    return False


def is_ocr_junk(line: str) -> bool:
    """Linha de ruído de scan: sobra de carimbo, borda de página, marca d'água."""
    s = line.strip()
    if not s:
        return False
    alpha = sum(ch.isalpha() for ch in s)
    junk = sum(ch in _OCR_JUNK_CHARS for ch in s)
    if alpha / len(s) < 0.45:
        return True
    if junk and junk / len(s) >= 0.12:
        return True
    words = re.findall(r"[A-Za-z]{3,}", s)
    # linha curta sem nenhuma palavra de verdade ("v jfeJt Jhydfc")
    return len(s) <= 30 and not words


def drop_toc_entries(lines: list[str]) -> list[str]:
    """Remove o índice de capítulos ("Chapter XXV. — Kapila describes … — P. 119.")."""
    out: list[str] = []
    i = 0
    while i < len(lines):
        if _TOC_ENTRY_START.match(lines[i]):
            j = i
            # a entrada termina na linha com "P. 119" (no máx. 4 linhas adiante)
            while j < len(lines) and j - i < 5 and not _TOC_ENTRY_END.search(lines[j]):
                j += 1
            if j < len(lines) and _TOC_ENTRY_END.search(lines[j]):
                i = j + 1
                continue
        out.append(lines[i])
        i += 1
    return out


_TERMINAL = re.compile(r"[.!?:;\"'”’)\]]\s*$")


def join_paragraphs(text: str, protect: Callable[[str], bool] | None = None) -> str:
    """Junta linhas do parágrafo e emenda a frase cortada pela quebra de página.

    - "devo-\\ntion" → "devotion" (hífen de fim de linha antes de minúscula);
    - quebra simples de linha vira espaço;
    - parágrafo que termina sem pontuação seguido de outro que começa em
      minúscula é a mesma frase partida pelo fim da página: vira um só.
    """
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    merged: list[str] = []
    fixed = False  # último parágrafo é título protegido (não emenda)
    for p in paras:
        if protect and protect(p):
            merged.append(p)
            fixed = True
            continue
        if fixed:
            merged.append(re.sub(r"\s*\n\s*", " ", p))
            fixed = False
            continue
        p = re.sub(r"(\w)[-¬]\s*\n\s*([a-z])", r"\1\2", p)
        p = re.sub(r"\s*\n\s*", " ", p)
        if merged:
            prev = merged[-1]
            if prev.endswith(("-", "¬")) and p[:1].islower():
                merged[-1] = prev.rstrip("-¬") + p
                continue
            if not _TERMINAL.search(prev) and p[:1].islower():
                merged[-1] = f"{prev} {p}"
                continue
        merged.append(p)
    return "\n\n".join(merged)


_ZERO_AS_O = re.compile(r"(?<![\w.,])0(?= [A-Za-z])")


def clean_ocr_text(
    text: str,
    *,
    running_heads: Iterable[str] = (),
    keep: Callable[[str], bool] | None = None,
) -> str:
    """Limpa o djvu.txt de um scan; ``keep(linha)`` protege títulos usados no recorte."""
    heads = tuple(running_heads)
    lines = strip_scan_boilerplate(text).replace("\r\n", "\n").split("\n")
    lines = drop_toc_entries(lines)
    out: list[str] = []
    for line in lines:
        if keep and keep(line):
            out.append(line.strip())
            continue
        s = line.strip()
        if not s:
            out.append("")
            continue
        if _PAGE_NUMBER.match(s) or is_running_head(s, heads) or is_ocr_junk(s):
            # some como linha vazia: a junção de parágrafos emenda a frase
            out.append("")
            continue
        out.append(s)
    # título protegido vira parágrafo isolado para o recorte achá-lo
    joined = "\n".join(f"\n{ln}\n" if keep and ln and keep(ln) else ln for ln in out)
    cleaned = join_paragraphs(joined, protect=keep)
    cleaned = _ZERO_AS_O.sub("O", cleaned)  # "0 king" → "O king"
    return normalize_whitespace(cleaned)


# ------------------------------------------------------------- numerais

_ROMAN_OCR = str.maketrans({"l": "I", "1": "I", "i": "I", "|": "I", "!": "I", "L": "I", "f": "I",
                            "t": "I", "J": "I", "v": "V", "¥": "V", "x": "X"})
_ROMAN_VALUES = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}


def roman_to_int(token: str) -> int | None:
    token = (token or "").strip().upper()
    if not token or any(ch not in _ROMAN_VALUES for ch in token):
        return None
    total = 0
    for i, ch in enumerate(token):
        value = _ROMAN_VALUES[ch]
        nxt = _ROMAN_VALUES[token[i + 1]] if i + 1 < len(token) else 0
        total += -value if value < nxt else value
    return total if total > 0 else None


def ocr_roman(token: str) -> int | None:
    """Numeral romano lido de OCR ("XVH" → 17, "V 1 1 1" → 8, "XLII" → 13, "XXV ML" → 28).

    Em título de capítulo de scan, "L", "l", "1", "f" quase sempre são um "I"
    mal lido; "U", "H" e "M" são "II" colado.
    """
    raw = (token or "").strip()
    if raw.isdigit():
        return int(raw)
    t = raw.translate(_ROMAN_OCR)
    t = t.replace("U", "II").replace("H", "II").replace("M", "II")
    t = re.sub(r"[^IVXC]", "", t)
    return roman_to_int(t)


# ------------------------------------------------------------- recortes


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def _locator_anchor(locator: str) -> str:
    """"Book 8, ch. 23" → "book-8-ch-23"; "Book 8, ch. 3–4" → "book-8-ch-3"."""
    return _slug(re.sub(r"[–-]\d+$", "", locator.strip()))


def _chapter_label(start: int, end: int | None) -> str:
    return f"ch. {start}" if not end or end <= start else f"ch. {start}–{end}"


# Bhāgavata Purāṇa, M. N. Dutt (Calcutta, 1895–96): número de capítulos por
# skandha na vulgata, para fechar o intervalo do último capítulo achado.
BHAGAVATA_CHAPTERS = (19, 10, 33, 31, 26, 19, 15, 24, 24, 90, 31, 13)

_DUTT_BOOK = re.compile(r"^\s*book\s*[:.]?\s*([ivxl1]{1,5})[\s.,:;-]*$", re.I)
_DUTT_BOOK_END = re.compile(r"^\s*(?:the\s+)?end\s+of\s+book\s+([ivxl1]{1,5})\b", re.I)
# folha de rosto repetida no início de cada volume ("edited and published by … Beadon Street")
_DUTT_TITLE_PAGE = re.compile(r"edited\s+and\s+published\s+by.{0,700}?Beadon\s+Street[.,]?", re.I | re.S)


_NUMERAL_LIKE = re.compile(r"^[IVXLCivxlc1|!fHUMtJ¥\s.,:;*•-]+$")


def chapter_heading_number(line: str) -> int | None | bool:
    """Título "CHAPTER XIV." lido de scan → 14; ``None`` se é título com numeral ilegível.

    Tolera o OCR da palavra ("6HAPTER VI.", "CHAP I EU XIX.") e do numeral
    ("XVH", "V 1 1 1", "CHAPTER I. V" → 1, lixo depois do ponto ignorado).
    Devolve ``False`` quando a linha não é título de capítulo.
    """
    s = line.strip()
    if not s or len(s) > 26 or not re.match(r"^\S{0,2}[A-Z]", s):
        return False
    tokens = s.split()
    best: tuple[float, int | None] | None = None
    for j in range(1, min(3, len(tokens) - 1) + 1):
        word = _letters("".join(tokens[:j]))
        if len(word) < 4:
            continue
        ratio = difflib.SequenceMatcher(None, word, "CHAPTER").ratio()
        if ratio < 0.7:
            continue
        rest = " ".join(tokens[j:])
        rest = re.split(r"(?<=\S)[.,](?=\s*\S)", rest, maxsplit=1)[0]  # "I. V" → "I"
        if not rest.strip() or not _NUMERAL_LIKE.match(rest):
            continue
        if best is None or ratio > best[0]:
            best = (ratio, ocr_roman(rest))
    if best is not None:
        return best[1]
    # palavra lida bem e numeral ilegível ("CHAPTER ib"): é título, sem número
    word_ok = difflib.SequenceMatcher(None, _letters(tokens[0]), "CHAPTER").ratio() >= 0.85
    if word_ok and len(tokens) >= 2 and len(" ".join(tokens[1:])) <= 6:
        return None
    return False


def _is_dutt_heading(line: str) -> bool:
    s = line.strip()
    return bool(_DUTT_BOOK.match(s) or _DUTT_BOOK_END.match(s)) or chapter_heading_number(s) is not False


def repair_chapter_numbers(nums: list[int | None], max_ch: int) -> list[int]:
    """Sequência crescente de capítulos a partir dos numerais lidos (alguns errados).

    Numeral ilegível, repetido ou fora de ordem vira anterior+1; depois um
    passe de trás para frente garante que nenhum passe do teto da obra.
    """
    out: list[int] = []
    prev = 0
    for n in nums:
        if n is None or not (prev < n <= min(prev + 10, max_ch)):
            n = prev + 1
        out.append(n)
        prev = n
    if out and out[-1] > max_ch:
        out[-1] = max_ch
    for i in range(len(out) - 2, -1, -1):
        if out[i] >= out[i + 1]:
            out[i] = out[i + 1] - 1
    prev = 0
    for i, n in enumerate(out):  # nunca abaixo de 1 nem repetido
        if n <= prev:
            out[i] = prev + 1
        prev = out[i]
    return out


def split_dutt_bhagavata(text: str) -> list[Section]:
    """Scan do Śrīmad-Bhāgavatam de Dutt: um registro por capítulo achado.

    Livro vem de "BOOK III" / "book: iv" e de "End of Book II"; capítulo vem
    do numeral do título lido com tolerância a OCR. Título de capítulo que o
    OCR perdeu deixa o anterior com dois capítulos: o rótulo vira intervalo
    ("ch. 11–12") em vez de inventar a fronteira. O que fica entre o fim de
    um livro e o 1º capítulo do seguinte (folha de rosto, prefácio, sumário)
    vira "Book N, front matter" quando tem texto corrido.
    """
    text = _DUTT_TITLE_PAGE.sub("\n", text)
    cleaned = clean_ocr_text(
        text,
        running_heads=("SRIMADBHAGAVATAM", "SRIMADBHAGABATAM"),
        keep=_is_dutt_heading,
    )
    book = 1
    pending_book: int | None = None
    # blocos: (livro, numeral lido | None, é capítulo?, parágrafos)
    blocks: list[tuple[int, int | None, bool, list[str]]] = [(1, None, False, [])]
    for para in cleaned.split("\n\n"):
        s = para.strip()
        m_end = _DUTT_BOOK_END.match(s)
        m_book = _DUTT_BOOK.match(s)
        if m_end:
            n = ocr_roman(m_end.group(1))
            if n:
                pending_book = n + 1
                blocks.append((pending_book, None, False, []))
            continue
        if m_book:
            n = ocr_roman(m_book.group(1))
            if n and 1 <= n <= 12:
                pending_book = n
                if blocks[-1][2]:
                    blocks.append((n, None, False, []))
            continue
        num = chapter_heading_number(s)
        if num is not False:
            if pending_book is not None:
                book, pending_book = pending_book, None
            blocks.append((book, num, True, []))
            continue
        blocks[-1][3].append(s)
    # numera os capítulos de cada livro
    by_book: dict[int, list[int]] = {}
    for i, (bk, _n, is_ch, _ps) in enumerate(blocks):
        if is_ch:
            by_book.setdefault(bk, []).append(i)
    fixed: dict[int, int] = {}
    for bk, idxs in by_book.items():
        max_ch = BHAGAVATA_CHAPTERS[bk - 1] if bk <= len(BHAGAVATA_CHAPTERS) else 999
        for i, n in zip(idxs, repair_chapter_numbers([blocks[i][1] for i in idxs], max_ch), strict=True):
            fixed[i] = n
    sections: list[Section] = []
    for i, (bk, _n, is_ch, ps) in enumerate(blocks):
        body = "\n\n".join(ps).strip()
        if not body:
            continue
        if not is_ch:
            if len(body) < 1500:  # sobra de sumário/folha de rosto
                continue
            first_ch = fixed[by_book[bk][0]] if by_book.get(bk) else 1
            if first_ch > 1:
                # o OCR perdeu o título dos primeiros capítulos do livro
                label = f"Book {bk}, {_chapter_label(1, first_ch - 1)}"
                sections.append(Section(label, f"book-{bk}-ch-1", body))
            else:
                sections.append(Section(f"Book {bk}, front matter", f"book-{bk}-front", body))
            continue
        ch = fixed[i]
        idxs = by_book[bk]
        pos = idxs.index(i)
        max_ch = BHAGAVATA_CHAPTERS[bk - 1] if bk <= len(BHAGAVATA_CHAPTERS) else ch
        end = fixed[idxs[pos + 1]] - 1 if pos + 1 < len(idxs) else max_ch
        sections.append(Section(f"Book {bk}, {_chapter_label(ch, end)}", f"book-{bk}-ch-{ch}", body))
    return sections


_SUTRA = re.compile(r"^(\d{1,2})\.\s+\S")
_NBS_RUNNING_HEADS = ("NARADA SUTRA", "INTRODUCTION", "APPENDIX", "OR INQUIRY INTO LOVE")


def split_sturdy_narada_bhakti(text: str) -> list[Section]:
    """Nārada Bhakti Sūtra de Sturdy (1896): introdução + grupos de sūtras com comentário.

    Os 84 sūtras vêm numerados ("15. Definitions of love are now given…");
    cada grupo é uma sequência de sūtras seguida do comentário. O apêndice
    (notícias de jornal sobre Vivekananda em Londres) não é a obra e fica fora.
    """
    body = strip_scan_boilerplate(text)
    start = body.find("In a former Age")
    if start > 0:
        body = body[start:]
    cut = re.search(r"\n\s*APPENDIX\.\s*\n", body)
    m_s1 = re.search(r"\n\s*1\.\s+We will now explain", body)
    if cut and (not m_s1 or cut.start() > m_s1.start()):
        body = body[: cut.start()]
    cleaned = clean_ocr_text(body, running_heads=_NBS_RUNNING_HEADS)
    paras = cleaned.split("\n\n")
    first = next((i for i, p in enumerate(paras) if p.startswith("1. We will now explain")), None)
    sections: list[Section] = []
    if first is None:
        return [Section("text", "text", cleaned)] if cleaned else []
    intro = "\n\n".join(p for p in paras[:first] if not re.match(r"^N\w{2,4}A\s*S\w{3,4}A\W*;?$", p))
    if intro.strip():
        sections.append(Section("front matter and introduction", "introduction", intro.strip()))
    groups: list[tuple[int, list[str]]] = []
    last = 0
    commentary = 0  # caracteres de comentário desde o último sūtra
    for p in paras[first:]:
        m = _SUTRA.match(p)
        n = int(m.group(1)) if m else None
        is_sutra = n is not None and last < n <= last + 3
        # grupo novo só depois de comentário de verdade: sūtra que o OCR
        # partiu em dois parágrafos não abre grupo de 50 caracteres
        if is_sutra and (not groups or commentary >= 600):
            groups.append((n, []))
        if groups:
            groups[-1][1].append(p)
        if is_sutra:
            last = n
            commentary = 0
        else:
            commentary += len(p)
    for i, (n0, ps) in enumerate(groups):
        end = groups[i + 1][0] - 1 if i + 1 < len(groups) else 84
        label = f"sūtra {n0}" if end <= n0 else f"sūtras {n0}–{end}"
        sections.append(Section(label, f"sutras-{n0}-{end}", "\n\n".join(ps)))
    return sections


_HV_CHAPTER = re.compile(r"^CHAPTER\.?\s+([IVXLC]+)\.\s*(.*)$")
_HV_PARVA = re.compile(r"^BHAVISHYA PARVA\b")


def _title_case(heading: str) -> str:
    small = {"a", "an", "and", "the", "of", "to", "in", "on", "with", "by", "for", "his", "her", "at"}
    words = heading.strip().rstrip(".").lower().split()
    out = []
    for i, w in enumerate(words):
        out.append(w if i and w in small else w[:1].upper() + w[1:])
    return " ".join(out).replace("’S", "’s")


def split_dutt_harivamsa(text: str) -> list[Section]:
    """Harivaṃśa de Dutt (Gutenberg 61937): um registro por capítulo.

    A numeração de Dutt corre contínua (I–CCLXXVII) até o Bhaviṣya Parva,
    que recomeça em I; o título do capítulo pode quebrar em duas linhas.
    """
    body = strip_gutenberg_boilerplate(text)
    lines = body.split("\n")
    # pula o sumário: o corpo começa no segundo "CHAPTER I." (o primeiro é do índice)
    def is_heading(ln: str) -> bool:  # linha do sumário tem pontilhado até o número
        return ln.startswith("CHAPTER") and not re.search(r"\.{4,}\s*$", ln) and bool(_HV_CHAPTER.match(ln.strip()))

    starts = [i for i, ln in enumerate(lines) if is_heading(ln)]
    if not starts:
        return []
    prelude_at = next((i for i, ln in enumerate(lines) if ln.strip() == "THE PRELUDE." and i > 0), None)
    sections: list[Section] = []
    parva = ""
    heads: list[tuple[int, int, str, str]] = []  # (linha, nº, título, parva)
    i = 0
    while i < len(lines):
        ln = lines[i]
        if _HV_PARVA.match(ln.strip()) and i > starts[0]:
            parva = "Bhaviṣya Parva"
        m = _HV_CHAPTER.match(ln.strip()) if is_heading(ln) else None
        if m and i >= starts[0]:
            title = m.group(2).strip()
            j = i + 1
            if j < len(lines) and lines[j].strip() and lines[j].strip().upper() == lines[j].strip():
                title = f"{title} {lines[j].strip()}"
                j += 1
            heads.append((j, roman_to_int(m.group(1)) or 0, title, parva))
            i = j
            continue
        i += 1
    if prelude_at is not None and prelude_at < starts[0]:
        prelude = "\n".join(lines[prelude_at + 1 : starts[0]])
        prelude = _clean_gutenberg_body(prelude)
        if len(prelude) > 200:
            sections.append(Section("Prelude", "prelude", prelude))
    # erro isolado de digitação no numeral ("LVIII" entre LXII e LXIV): se o
    # vizinho seguinte é anterior+2, este é anterior+1. Saltos que o próprio
    # livro imprime (CXXVII → CCXVIII) ficam como estão: é o número da edição.
    nums = [h[1] for h in heads]
    for k in range(1, len(heads) - 1):
        same_parva = heads[k - 1][3] == heads[k][3] == heads[k + 1][3]
        if same_parva and nums[k] != nums[k - 1] + 1 and nums[k + 1] == nums[k - 1] + 2:
            nums[k] = nums[k - 1] + 1
    heads = [(h[0], n, h[2], h[3]) for h, n in zip(heads, nums, strict=True)]
    for k, (line_no, num, title, pv) in enumerate(heads):
        end = heads[k + 1][0] - 1 if k + 1 < len(heads) else len(lines)
        # o próximo título está na linha anterior ao corpo seguinte
        chunk_lines = lines[line_no:end]
        while chunk_lines and _HV_CHAPTER.match(chunk_lines[-1].strip() or "-"):
            chunk_lines.pop()
        body_text = _clean_gutenberg_body("\n".join(_drop_next_heading(chunk_lines)))
        if not body_text:
            continue
        where = f"{pv}, ch. {num}" if pv else f"ch. {num}"
        label = f"{where}: {_title_case(title)}" if title else where
        anchor = _slug(f"{'bhavishya-' if pv else ''}ch-{num}")
        sections.append(Section(label, anchor, body_text))
    return sections


def _drop_next_heading(chunk_lines: list[str]) -> list[str]:
    out = list(chunk_lines)
    # remove o(s) título(s) do capítulo seguinte que ficaram no fim do bloco
    while out and (not out[-1].strip() or out[-1].strip().upper() == out[-1].strip()):
        last = out[-1].strip()
        if last and len(last) > 80:
            break
        out.pop()
    return out


def _clean_gutenberg_body(text: str) -> str:
    text = re.sub(r"(?<!\w)_(.+?)_(?!\w)", r"\1", text, flags=re.S)  # _itálico_
    text = re.sub(r"\[Illustration[^\]]*\]", " ", text)
    return normalize_whitespace(join_paragraphs(text))


def split_single(text: str) -> list[Section]:
    return [Section("", "", text)] if text.strip() else []


# ------------------------------------------------------------- Subba Rau (OCR local)

_SR_PAGE = re.compile(r"^\[\[page (\d+)\]\]$")
# cabeçalho de página: "[Sk. 10. Adh. 41." lido como "Sx. 10, ApH. 41", "8k. 10, Aba, 6"…
_SR_HEAD = re.compile(r"(?:S|8|\$)\s*[kxeKX][eo]?\s*[.,]?\s*([0-9lIO]{1,2})\s*[.,]?\s*A?[A-Za-z]{1,3}\s*[.,]*\s*(\d{1,3})")
_SR_HEAD_LINE = re.compile(r"RIMAD|BHAGAVAT", re.I)
_SR_ORDINALS = {"EIGHTH": 8, "NINTH": 9, "TENTH": 10, "ELEVENTH": 11, "TWELFTH": 12}
_SR_ENDS = re.compile(r"thus\s+ends\s+the\s+(\w+)\s+skandha", re.I)
_SR_VERSE_ONE = re.compile(r"^\W{0,3}1\s*[.,]\s+(?!V\.|D\.|J|Note|AT\.)[A-Z‘'\"“(]")
_SR_VERSE = re.compile(r"^\W{0,3}(\d{1,3})\s*[.,]\s+\S")
_SR_TOC_HEADER = re.compile(r"\bP\s*[AaE]\s*[CcGgO]\s*[EeH]\b", re.I)
_SR_TOC_ENTRY = re.compile(r"^(?:\S{1,3}\s+){0,2}\W{0,3}\d{1,2}\s*[.,]\s.{8,}\s\d{1,3}\W{0,4}(?:\s\S{1,3}){0,3}\s*$")


_SR_SHORT_WORDS = frozenset(
    ["a", "i", "o", "an", "as", "at", "be", "by", "do", "go", "he", "if", "in", "is", "it", "me", "my", "no", "of", "on", "or", "so", "to", "up", "us", "we", "ye"]
)


def _sr_trim_line(line: str) -> str:
    """Tira das margens os restos de OCR da página escaneada ("| i ... a 4 |")."""
    toks = line.split()

    def junk(tok: str, edge_next: str | None) -> bool:
        if not any(c.isalnum() for c in tok):
            return True
        core = tok.strip("|!:;'\"‘’“”_~-—.,()[]{}")
        if len(core) > 2:
            return False
        if core.isdigit():  # número de verso no começo fica; no fim é resto
            return edge_next is None
        if core == "i":  # o pronome vem em maiúscula
            return True
        if edge_next is not None and edge_next[:1].isdigit():  # "a 6. Sri Suka…"
            return True
        return core.lower() not in _SR_SHORT_WORDS

    while toks and junk(toks[0], toks[1] if len(toks) > 1 else None):
        toks.pop(0)
    while toks and junk(toks[-1], None):
        toks.pop()
    return " ".join(toks)


def _sr_heading_number(line: str) -> int | None | bool:
    """"ADHYAYA 41." (com OCR ruim: "ADHVAYA 34", "ADAYAYA 80") → 41; False se não é título."""
    s = line.strip()
    if len(s) > 45 or _SR_TOC_HEADER.search(s) or "ijayadhva" in s or "ijayadhwa" in s:
        return False
    for m in re.finditer(r"[A-Za-z]{5,9}", s):
        # o título vem em caixa alta; "Adhyaya" em nota ("[84th Adhyaya ends…]") não conta
        if sum(c.isupper() for c in m.group(0)) < 5:
            continue
        if difflib.SequenceMatcher(None, m.group(0).upper(), "ADHYAYA").ratio() >= 0.7:
            rest = re.sub(r"[Il|](?=\d)|(?<=\d)[Il|]", "1", s[m.end():].replace(" ", ""))  # "I7" = 17
            digits = re.findall(r"\d+", rest)
            if not digits:
                return None
            n = int(digits[0])
            return n if 1 <= n <= 99 else None
    return False


def _sr_skandha_title(line: str) -> int | None:
    s = line.strip()
    up = re.sub(r"[^A-Z]", "", s.upper()).replace("SKANDUA", "SKANDHA")
    if "SKANDHA" not in up or len(s) > 40 or "ENDS" in up:
        return None
    for word, num in _SR_ORDINALS.items():
        if word + "SKANDHA" in up:
            return num
    return None


def repair_by_anchors(nums: list[int | None], max_ch: int) -> list[int]:
    """Numeração de capítulos a partir de numerais OCR, guiada pelos mais coerentes.

    Fica com o maior conjunto de numerais lidos que cabem juntos em ordem
    (entre os blocos i < j há espaço para j - i capítulos: n_j - n_i ≥ j - i)
    e preenche os demais pela posição. Um numeral mal lido ("9" no lugar de
    "2") não arrasta os seguintes, como faria uma correção só para a frente.
    """
    size = len(nums)
    cand = [
        (k, n) for k, n in enumerate(nums)
        if n is not None and k + 1 <= n <= max_ch - (size - 1 - k)
    ]
    # maior subsequência não decrescente de v = n - k (programação dinâmica)
    best: list[int] = []
    prev: list[int] = []
    for i, (k, n) in enumerate(cand):
        best.append(1)
        prev.append(-1)
        for j in range(i):
            kj, nj = cand[j]
            if nj - kj <= n - k and best[j] + 1 > best[i]:
                best[i], prev[i] = best[j] + 1, j
    anchors: dict[int, int] = {}
    if cand:
        i = max(range(len(cand)), key=lambda x: best[x])
        while i >= 0:
            anchors[cand[i][0]] = cand[i][1]
            i = prev[i]
    out: list[int] = []
    last: tuple[int, int] | None = None
    for k in range(size):
        if k in anchors:
            last = (k, anchors[k])
            out.append(anchors[k])
        elif last is None:
            out.append(k + 1)
        else:
            out.append(last[1] + (k - last[0]))
    return out


def split_subbarau_bhagavata(text: str) -> list[Section]:
    """Vol. II de S. Subba Rau (Tirupati, 1928) em OCR local com ``[[page N]]``.

    Capítulo começa no título "ADHYAYA N" e, quando o OCR perdeu o título, no
    verso "1." que o cabeçalho de página ("[Sk. 11. Adh. 14]", capítulo em
    curso no fim da página) confirma como capítulo novo. O numeral do último
    título de cada página é trocado pelo do cabeçalho dela quando os dois são
    próximos; numeral ilegível ou fora de ordem cede ao capítulo que se repete
    nos cabeçalhos das páginas do bloco; depois vale a sequência
    (``repair_by_anchors``). Sumários de cada skandha (da página
    "CONTENTS" até o título do skandha no texto) e o índice final ficam de fora.
    """
    pages: list[tuple[int, list[str]]] = []
    for line in text.splitlines():
        m = _SR_PAGE.match(line.strip())
        if m:
            pages.append((int(m.group(1)), []))
        elif pages:
            pages[-1][1].append(line)
    heads: list[tuple[int, int] | None] = []
    for _n, lines in pages:
        head = None
        for ln in lines[:3]:
            m = _SR_HEAD.search(ln) if _SR_HEAD_LINE.search(ln) else None
            if m:
                sk = int(m.group(1).replace("l", "1").replace("I", "1").replace("O", "0"))
                head = (sk, int(m.group(2)))
                break
        heads.append(head)

    def page_head_ch(pi: int, sk: int) -> int | None:
        h = heads[pi] if 0 <= pi < len(heads) else None
        if not h or sorted(str(h[0])) != sorted(str(sk)):  # "Sk. 21" = 12 com dígitos trocados
            return None
        return h[1] if 1 <= h[1] <= BHAGAVATA_CHAPTERS[sk - 1] else None

    skandha = 8
    toc = False
    finished = False
    # blocos: [skandha, número | None, página, linhas, implícito?]
    blocks: list[list[Any]] = []

    def has_verses(block: list[Any]) -> bool:
        return any(_SR_VERSE.match(x.strip()) for x in block[3])

    for pi, (_n, lines) in enumerate(pages):
        if finished:
            break
        top = " ".join(lines[:4]).upper()
        entries = sum(1 for ln in lines if _SR_TOC_ENTRY.match(ln.strip()))
        titled = any(_sr_skandha_title(ln) for ln in lines[:6])
        # sumário: página "CONTENTS", ou nome do skandha + entradas "N. assunto … página",
        # ou continuação de um sumário
        toc_page = "CONTENTS" in top or (entries >= 3 and (titled or toc))
        if toc_page:
            toc = True
        elif toc:
            # fim do sumário: o texto do skandha começa aqui (mesmo sem título)
            toc = False
            if any(len(ln.strip()) > 40 for ln in lines):
                blocks.append([skandha, 1, pi, [], True])
        page_titles = [i for i, ln in enumerate(lines) if _sr_heading_number(ln) is not False]
        last_title = page_titles[-1] if page_titles else None
        last_verse = 0
        for li, line in enumerate(lines):
            s = line.strip()
            if li < 3 and _SR_HEAD_LINE.search(s) and (_SR_HEAD.search(s) or len(s) < 30):
                continue
            sk_title = _sr_skandha_title(s)
            if sk_title:
                skandha = sk_title
                if not toc and not (blocks and blocks[-1][4] and blocks[-1][0] == skandha):
                    blocks.append([skandha, 1, pi, [], True])
                continue
            m_end = _SR_ENDS.search(s)
            if m_end and "part" not in s.lower():
                if m_end.group(1).lower() in {"twelfth", "12th"}:
                    finished = True
                    break
                continue
            num = _sr_heading_number(s)
            if num is not False:
                if toc:
                    continue
                head = page_head_ch(pi, skandha)
                if li == last_title and head and (num is None or abs(head - num) <= 3):
                    num = head
                if blocks and blocks[-1][4] and not has_verses(blocks[-1]):
                    blocks[-1][1] = num or blocks[-1][1]  # título logo após o nome do skandha
                    blocks[-1][4] = False
                else:
                    blocks.append([skandha, num, pi, [], False])
                continue
            if toc or not blocks:
                continue
            vm = _SR_VERSE.match(s)
            if _SR_VERSE_ONE.match(s) and last_verse >= 4:
                cur = blocks[-1][1]
                ahead = [c for c in (page_head_ch(pi, skandha), page_head_ch(pi + 1, skandha)) if c]
                if cur and any(cur < c <= cur + 2 for c in ahead) and pi != blocks[-1][2]:
                    blocks.append([skandha, cur + 1, pi, [line], False])
                    last_verse = 1
                    continue
            if vm:
                last_verse = int(vm.group(1))
            blocks[-1][3].append(line)
    by_sk: dict[int, list[int]] = {}
    for i, b in enumerate(blocks):
        by_sk.setdefault(b[0], []).append(i)
    # numeral ilegível ou fora de ordem: vale o primeiro capítulo que se repete
    # nos cabeçalhos das páginas inteiras do bloco
    for sk, idxs in by_sk.items():
        for pos, i in enumerate(idxs):
            num = blocks[i][1]
            prev = blocks[idxs[pos - 1]][1] if pos else None
            nxt = blocks[idxs[pos + 1]][1] if pos + 1 < len(idxs) else None
            ordered = (prev is None or num > prev) and (nxt is None or num < nxt) if num else False
            if ordered and num <= BHAGAVATA_CHAPTERS[sk - 1]:
                continue
            stop = blocks[idxs[pos + 1]][2] if pos + 1 < len(idxs) else len(pages)
            seen = [c for pi in range(blocks[i][2] + 1, stop) if (c := page_head_ch(pi, sk))]
            confirmed = next((c for c in seen if seen.count(c) >= 2), None)
            if confirmed:
                blocks[i][1] = confirmed
    fixed: dict[int, int] = {}
    for sk, idxs in by_sk.items():
        for i, n in zip(idxs, repair_by_anchors([blocks[i][1] for i in idxs], BHAGAVATA_CHAPTERS[sk - 1]), strict=True):
            fixed[i] = n
    sections: list[Section] = []
    for sk, idxs in by_sk.items():
        for pos, i in enumerate(idxs):
            raw = "\n".join(_sr_trim_line(x) for x in blocks[i][3])
            body = clean_ocr_text(raw, running_heads=("SRIMAD BHAGAVATAM",))
            if not body:
                continue
            ch = fixed[i]
            end = fixed[idxs[pos + 1]] - 1 if pos + 1 < len(idxs) else BHAGAVATA_CHAPTERS[sk - 1]
            sections.append(Section(f"Book {sk}, {_chapter_label(ch, end)}", f"book-{sk}-ch-{ch}", body))
    return sections


# ------------------------------------------------------------- GRETIL (sânscrito)

_GRETIL_REF = re.compile(r"//\s*bhp_(\d{2})\.(\d{2})\.(\d{3})(_\d+)?\*?\s*//")
_GRETIL_SPEAKER = re.compile(r"bhp_\d{2}\.\d{2}\.\d{3}(?:_\d+)?/\d+\s+")
# metro longo: GRETIL repete a primeira metade ("A $ B & A $ B & C % D")
_GRETIL_DUP_HALF = re.compile(r"([^$&%/]+?\$[^$&%/]+?&)\s*\1")


def _gretil_verse(raw: str) -> str:
    """Texto de um verso do GRETIL em linhas: "pāda a pāda b /" + "pāda c pāda d"."""
    t = raw.replace("&amp;", "&")
    t = _GRETIL_SPEAKER.sub("", t)
    for _ in range(2):
        t = _GRETIL_DUP_HALF.sub(r"\1", t)
    t = t.replace("$", " ").replace("%", " ").replace("&", " /\n")
    lines = [normalize_whitespace(x) for x in t.splitlines()]
    return "\n".join(x for x in lines if x)


def split_gretil_bhagavata(text: str) -> list[Section]:
    """Bhāgavata Purāṇa do GRETIL (texto puro): um registro por adhyāya.

    Cada verso termina em "// bhp_SS.AA.VVV //". O arquivo tem rótulos trocados
    (3.31.22–48 marcados como 3.32.x; um 8.8.3 no meio do 8.7), então o capítulo
    só avança quando o adhyāya seguinte começa (verso ≤ 3 e o próximo rótulo
    não volta ao corrente); rótulo fora de ordem fica no capítulo corrente,
    onde o texto de fato está, com o número de verso do rótulo.
    """
    body = text.split("# Text", 1)[1] if "# Text" in text else text
    units: list[tuple[int, int, int, str, str]] = []
    pos = 0
    for m in _GRETIL_REF.finditer(body):
        units.append((int(m.group(1)), int(m.group(2)), int(m.group(3)), m.group(4) or "", body[pos : m.start()]))
        pos = m.end()

    def following(s: int, a: int) -> tuple[int, int]:
        return (s, a + 1) if a < BHAGAVATA_CHAPTERS[s - 1] else (s + 1, 1)

    cur = (1, 1)
    chapters: dict[tuple[int, int], list[str]] = {}
    for i, (s, a, v, part, raw) in enumerate(units):
        nxt = units[i + 1][:2] if i + 1 < len(units) else None
        if (s, a) == following(*cur) and v <= 3 and nxt != cur:
            cur = (s, a)
        verse = _gretil_verse(raw)
        if verse:
            # capítulo pela posição, verso pelo rótulo
            chapters.setdefault(cur, []).append(f"{verse} // {cur[0]}.{cur[1]}.{v}{part} //")
    sections: list[Section] = []
    for (s, a), verses in chapters.items():
        sections.append(Section(f"Skandha {s}, adhyāya {a}", f"bhp-{s}-{a}", "\n\n".join(verses)))
    return sections


SPLITTERS: dict[str, Callable[[str], list[Section]]] = {
    "gretil-bhagavata": split_gretil_bhagavata,
    "subbarau-bhagavata": split_subbarau_bhagavata,
    "dutt-bhagavata": split_dutt_bhagavata,
    "sturdy-narada-bhakti": split_sturdy_narada_bhakti,
    "dutt-harivamsa": split_dutt_harivamsa,
    "single": split_single,
}


# ------------------------------------------------------------- Sacred Texts


def sacred_texts_html_to_text(html: str) -> str:
    """Página do Sacred Texts → texto corrido, sem navegação nem marcas de página.

    O site põe cada letra com diacrítico num ``<i>`` (Vai<i>s</i>ya); com
    separador de linha entre tags o nome quebra em três linhas. Aqui só
    bloco vira quebra de linha. Âncoras de nota ("248") e "p. 100" saem;
    as notas de rodapé ficam, com a referência "[n. 248]".
    """
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "noscript", "head"]):
        tag.decompose()
    for note in soup.select("span.margnote"):
        note.decompose()  # resumo de margem ("Supposed origin of the Code of Manu.") no meio da frase
    for a in soup.find_all("a"):
        href = a.get("href") or ""
        label = a.get_text(" ", strip=True)
        if re.fullmatch(r"p\.\s*\d+", label) or (a.get("name") or "").startswith("page_"):
            a.decompose()
        elif "#fn_" in href or href.startswith("#fn"):
            a.decompose()  # chamada de nota no corpo
        elif "#fr_" in href:
            m = re.fullmatch(r"\d+:(\d+)", label)
            a.replace_with(f"[n. {m.group(1)}] " if m else "")
        elif label in {"Next", "Previous", "Index", "Hinduism", "Sacred Texts"} or "amazon" in href:
            a.decompose()
    for br in soup.find_all("br"):
        br.replace_with("\n")
    for blk in soup.find_all(["p", "div", "h1", "h2", "h3", "h4", "h5", "h6", "hr", "table", "tr", "li", "center"]):
        blk.insert_before("\n\n")
        blk.insert_after("\n\n")
    text = soup.get_text("")
    lines = [ln.strip() for ln in text.split("\n")]
    out: list[str] = []
    for ln in lines:
        if re.fullmatch(r"p\.\s*\d+", ln) or re.fullmatch(r"\d{1,4}", ln):
            continue
        if re.match(r"^Next:\s", ln) or "Buy this Book" in ln or ln.endswith("at sacred-texts.com"):
            continue
        if re.fullmatch(r",?\s*by .*\[\d{4}\],? at sacred-texts\.com", ln):
            continue
        out.append(ln)
    text = "\n".join(out)
    # título da página e do livro repetidos no topo
    text = re.sub(r"^\s*The Minor Law Books \(SBE33\)[^\n]*\n", "", text)
    return normalize_whitespace(re.sub(r"\n\s*\n(\s*\n)+", "\n\n", text))


# ------------------------------------------------------------- registros


def is_book_source(src: dict[str, Any]) -> bool:
    return (src.get("etl") or "").lower() == BOOK_ETL


def book_title(src: dict[str, Any], locator: str) -> str:
    """"Bhāgavata Purāṇa — Book 3, ch. 25 (M. N. Dutt, 1896)"."""
    work = (src.get("work_title") or src.get("title") or "").strip()
    credit = src.get("credit") or ", ".join(str(x) for x in (src.get("translator"), src.get("year")) if x)
    head = f"{work} — {locator}" if locator else work
    return f"{head} ({credit})" if credit else head


def expand_book_file(path: Path | str, source: dict[str, Any]) -> list[dict[str, Any]]:
    """Arquivo baixado → registros do corpus, um por seção do recorte."""
    from vedic_pipeline.etl.extractors import extract_txt

    path = Path(path)
    if source.get("ocr") and path.suffix.lower() == ".pdf":
        from vedic_pipeline.etl.ocr import ocr_pdf

        raw = ocr_pdf(path, source["ocr"])  # scan sem camada de texto aproveitável
    else:
        raw = extract_txt(path)
    clean = (source.get("clean") or "").lower()
    if clean == "sacred-texts":
        text = sacred_texts_html_to_text(raw)
    elif clean == "gutenberg":
        text = raw  # o recorte tira o boilerplate e trata o itálico
    else:
        text = raw
    split = (source.get("split") or "single").lower()
    splitter = SPLITTERS.get(split)
    if splitter is None:
        raise ValueError(f"Recorte desconhecido no manifesto: {split!r} (opções: {sorted(SPLITTERS)})")
    min_chars = int(source.get("min_chars") or 200)
    base_url = (source.get("citation_url") or source.get("url") or "").strip()
    records: list[dict[str, Any]] = []
    seen_anchors: dict[str, int] = {}
    # volume que traz livros já cobertos por outra fonte: só os listados entram
    keep_books = {int(b) for b in source.get("keep_books") or []}
    # scan encadernado fora de ordem: o manifesto lista as seções que entram,
    # cada uma com o localizador certo (None = manter o do recorte)
    select: dict[str, str | None] | None = source.get("select")
    for sec in splitter(text):
        body = sec.text.strip()
        if len(body) < min_chars:
            continue
        if keep_books and not any(sec.anchor.startswith(f"book-{b}-") for b in keep_books):
            continue
        if select is not None:
            if sec.anchor not in select:
                continue
            new_locator = select[sec.anchor]
            if new_locator:
                sec = Section(new_locator, _locator_anchor(new_locator), sec.text)
        locator = sec.locator or (source.get("locator") or "")
        title = book_title(source, locator)
        anchor = sec.anchor
        if anchor:
            # número repetido na edição (dois "ch. 42") não pode colidir na URL,
            # senão a deduplicação por URL descartaria o segundo capítulo
            seen_anchors[anchor] = seen_anchors.get(anchor, 0) + 1
            if seen_anchors[anchor] > 1:
                anchor = f"{anchor}-{seen_anchors[anchor]}"
        url = f"{base_url}#{anchor}" if anchor else base_url
        fp = content_fingerprint(body)
        rec = {
            "id": stable_id(url, title, fp[:16]),
            "text": body,
            "source_url": url,
            "title": title,
            "tradition": (source.get("tradition") or "unknown").lower(),
            "language": (source.get("language") or "und").lower(),
            "license": (source.get("license") or "").lower(),
            "retrieved_at": utc_now_iso(),
            "fingerprint": fp,
            "char_count": len(body),
        }
        for key in ("translator", "year", "edition", "license_note", "work_title"):
            if source.get(key):
                rec[key] = source[key]
        if locator:
            rec["locator"] = locator
        if source.get("url") and source.get("url") != base_url:
            rec["retrieved_from"] = source["url"]
        records.append(rec)
    return records
