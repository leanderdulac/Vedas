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


SPLITTERS: dict[str, Callable[[str], list[Section]]] = {
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
    credit = ", ".join(str(x) for x in (src.get("translator"), src.get("year")) if x)
    head = f"{work} — {locator}" if locator else work
    return f"{head} ({credit})" if credit else head


def expand_book_file(path: Path | str, source: dict[str, Any]) -> list[dict[str, Any]]:
    """Arquivo baixado → registros do corpus, um por seção do recorte."""
    from vedic_pipeline.etl.extractors import extract_txt

    path = Path(path)
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
    for sec in splitter(text):
        body = sec.text.strip()
        if len(body) < min_chars:
            continue
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
