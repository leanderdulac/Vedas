"""Sāmaveda Kauthuma (sa.wikisource.org): wikitexto cru → arcas → compacto.

Fonte: Sanskrit Wikisource, सामवेदः/कौथुमीया/संहिता (CC BY-SA 4.0, permitida em
ALLOWED_LICENSES). Páginas-folha em wikitexto cru (`?action=raw`) trazem os
mantras dentro de `<poem><span …>…</span></poem>`, cada arca terminando com
numeração global em numerais devanāgarī (`॥ ६५१ ॥`; às vezes sem o ॥ de
abertura: `चर्षणीनां  ७१३ ॥`). Anotações (stobha com link externo, concordância
`ऋ. ९.११.१`, `<ref>`, `{{टिप्पणी}}`) são descartadas.

No Uttarārciká as seções de um ardha são marcadas por linhas que contêm só o
número da seção (numerais devanāgarī); no Pūrvārciká a daśatī inteira é um
grupo só. O pareamento com Griffith (`samaveda_griffith_en.json`) segue a
correspondência validada: daśatī do chanda ≡ hino Griffith (part1: livro =
prapāṭhaka); seção do ardha ≡ hino Griffith (part2: ardha = capítulo). O
pareamento só é feito quando a contagem de arcas bate com a de versos do hino
Griffith — divergência é registrada e a seção fica de fora (nunca inventa
alinhamento). Āraṇya (586–640) e Mahānāmnya (641–650) não existem em Griffith e
entram como hinos sa-only (chapter 3 e 4).
"""

from __future__ import annotations

import re
from typing import Any

_WIKISOURCE_BASE = "https://sa.wikisource.org/wiki/"
_WIKISOURCE_ROOT = "सामवेदः/कौथुमीया/संहिता"
WIKISOURCE_PAGE = f"{_WIKISOURCE_BASE}{_WIKISOURCE_ROOT}"

_ATTRIBUTION = (
    "Mantra text from Sanskrit Wikisource (sa.wikisource.org), Sāmaveda "
    "Kauthuma saṃhitā, licensed under Creative Commons Attribution-ShareAlike "
    "4.0 (CC BY-SA 4.0). Mantra text only; annotations, stobha labels, "
    "concordances and notes omitted."
)

_ARCA_END = re.compile(r"([०-९]+)\s*(?:॥|।।)")
# variante sem o ॥ de fecho: `…बृहस्पतिम्॥ ९१` (fim de linha, daṇḍa antes do nº)
_ARCA_END_OPEN = re.compile(r"(?:॥|।।)\s*([०-९]+)\s*$")
_SECTION_MARKER = re.compile(r"[（(]?[०-९]+[）)]?")
_TAG_REF = re.compile(r"<ref[^>]*/>|<ref[^>]*>.*?</ref>", re.S)
_TAG_COMMENT = re.compile(r"<!--.*?-->", re.S)
_TAG_TEMPLATE = re.compile(r"\{\{.*?\}\}", re.S)
_EXT_LINK_LABEL = re.compile(r"\[https?://\S+[ \t]+[^\]]*\]")
_EXT_LINK_BARE = re.compile(r"\[https?://\S+\]")
_INT_LINK_LABEL = re.compile(r"\[\[([^\]|]*)\|([^\]]*)\]\]")
_INT_LINK_BARE = re.compile(r"\[\[([^\]]*)\]\]")
_HTML_TAG = re.compile(r"</?[a-zA-Z][^>]*>")
_DEVA_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")
_ASCII_TO_DEVA = str.maketrans("0123456789", "०१२३४५६७८९")

# Nomes ordinais dos títulos de página (verbatten do allpages da API MediaWiki).
_PRAPATHAKA_NAMES = [
    "प्रथमप्रपाठकः",
    "द्वितीयप्रपाठकः",
    "तृतीयप्रपाठकः",
    "चतुर्थप्रपाठकः",
    "पञ्चमप्रपाठकः",
    "षष्ठप्रपाठकः",
    "सप्तमप्रपाठकः",
    "अष्टमप्रपाठकः",
    "नवमप्रपाठकः",
]
_DASATI_NAMES = [
    "प्रथमा दशतिः",
    "द्वितीया दशतिः",
    "तृतीया दशतिः",
    "चतुर्थी दशतिः",
    "पञ्चमी दशतिः",
    "षष्ठी दशतिः",
    "सप्तमी दशतिः",
    "अष्टमी दशतिः",
    "नवमी दशतिः",
    "दशमी दशतिः",
]
_ARDHA_NAMES = ["प्रथमोऽर्द्धः", "द्वितीयोऽर्द्धः", "तृतीयोऽर्द्धः"]
_DASATI_PER_PRAPATHAKA = [10, 10, 10, 10, 10, 9]  # chanda (arcas 1–585)
_ARDHAS_PER_PRAPATHAKA = [2, 2, 2, 2, 2, 3, 3, 3, 3]  # uttarārcika (651–1875)


def page_title(kind: str, *nums: int) -> str:
    """Título wiki de uma página-folha (daśatī, ardha ou mahānāmnya)."""
    if kind == "dasati":
        p, d = nums
        return (
            f"{_WIKISOURCE_ROOT}/पूर्वार्चिकः/छन्द आर्चिकः/"
            f"1.1.{p} {_PRAPATHAKA_NAMES[p - 1]}/1.1.{p}.{d} {_DASATI_NAMES[d - 1]}"
        )
    if kind == "aranya":
        d = nums[0]
        return f"{_WIKISOURCE_ROOT}/पूर्वार्चिकः/अथारण्यार्चिकः/1.2.{d} {_DASATI_NAMES[d - 1]}"
    if kind == "ardha":
        p, a = nums
        return (
            f"{_WIKISOURCE_ROOT}/उत्तरार्चिकः/2.{p} {_PRAPATHAKA_NAMES[p - 1]}/"
            f"2.{p}.{a} {_ARDHA_NAMES[a - 1]}"
        )
    if kind == "mahanamnya":
        return f"{_WIKISOURCE_ROOT}/पूर्वार्चिकः/महानाम्न्यार्चिकः"
    raise ValueError(f"kind inválido: {kind}")


def all_leaf_pages() -> list[tuple[str, str, tuple[int, ...]]]:
    """Todas as páginas-folha: (kind, título, nums). 59 daśatīs do chanda, 5 do
    āraṇya, 1 página do mahānāmnya e 22 ardhas do uttarārcika."""
    pages: list[tuple[str, str, tuple[int, ...]]] = []
    for p, nd in enumerate(_DASATI_PER_PRAPATHAKA, start=1):
        for d in range(1, nd + 1):
            pages.append(("dasati", page_title("dasati", p, d), (p, d)))
    for d in range(1, 6):
        pages.append(("aranya", page_title("aranya", d), (d,)))
    pages.append(("mahanamnya", page_title("mahanamnya"), ()))
    for p, na in enumerate(_ARDHAS_PER_PRAPATHAKA, start=1):
        for a in range(1, na + 1):
            pages.append(("ardha", page_title("ardha", p, a), (p, a)))
    return pages


def raw_url(title: str) -> str:
    from urllib.parse import quote

    return f"{_WIKISOURCE_BASE}{quote(title.replace(' ', '_'), safe='/_.')}?action=raw"


def deva_int(token: str) -> int | None:
    digits = (token or "").translate(_DEVA_DIGITS).strip()
    if digits.isdigit():
        value = int(digits)
        return value if value > 0 else None
    return None


def clean_wikitext(raw: str) -> str:
    """Remove refs, templates, links e tags do wikitexto, preservando linhas.

    Normaliza também daṇḍas ASCII (páginas em formato-tabela usam `|`, `|| N ||`).
    """
    text = _TAG_REF.sub("", raw or "")
    text = _TAG_COMMENT.sub("", text)
    text = _TAG_TEMPLATE.sub("", text)
    text = _EXT_LINK_LABEL.sub("", text)
    text = _EXT_LINK_BARE.sub("", text)
    text = _INT_LINK_LABEL.sub(r"\2", text)
    text = _INT_LINK_BARE.sub(r"\1", text)
    text = _HTML_TAG.sub("", text)
    text = text.replace("||", " ॥ ")
    text = re.sub(r"\|", " । ", text)
    return text


def parse_wikisource_sections(raw: str) -> list[tuple[int | None, list[tuple[int, str]]]]:
    """Divide o raw de uma página-folha em seções `(nº do marcador, arcas)`.

    Seções são separadas por linhas que contêm só numerais devanāgarī —
    opcionalmente entre parênteses (`(१)`) — e o valor do marcador é o número
    da seção (Uttarārciká). Páginas sem marcador (daśatīs) voltam uma seção
    única com marcador None. Texto do arca junta as linhas de mantras e fecha
    com ` ॥ N ॥` (nº global em numerais devanāgarī). Linhas sem daṇḍa fora de
    marcadores (rubricas, concordâncias, stobha) são descartadas.
    """
    sections: list[tuple[int | None, list[tuple[int, str]]]] = []
    current: list[tuple[int, str]] = []
    marker: int | None = None
    buf: list[str] = []
    for raw_line in clean_wikitext(raw).split("\n"):
        line = " ".join(raw_line.split())
        if not line:
            continue
        if _SECTION_MARKER.fullmatch(line):
            if current or buf:
                if buf:
                    # arca sem fechamento antes do marcador: descarta, é rubrica
                    buf = []
                sections.append((marker, current))
                current = []
            marker = deva_int("".join(re.findall(r"[०-९]", line)))
            continue
        if "।" in line or "॥" in line:
            match: re.Match[str] | None = None
            for m in _ARCA_END.finditer(line):
                match = m
            open_match: re.Match[str] | None = None
            if match is None:
                # variante `…॥ ९१` sem o ॥ de fecho do arca
                open_match = _ARCA_END_OPEN.search(line)
            if match is not None:
                head = line[: match.start()].strip().rstrip("।॥| ").strip()
                if head:
                    buf.append(head)
                body = " ".join(buf).strip()
                number = deva_int(match.group(1))
                if body and number:
                    # fechamento `॥ N ॥` com o nº global do arca, em numerais
                    # devanāgarī (o filtro de dígitos do padas_of só descarta ०-९)
                    deva_n = str(number).translate(_ASCII_TO_DEVA)
                    current.append((number, f"{body} ॥ {deva_n} ॥"))
                buf = []
            elif open_match is not None:
                number = deva_int(open_match.group(1))
                head = line[: open_match.start()].strip().rstrip("।॥| ").strip()
                if head:
                    buf.append(head)
                body = " ".join(buf).strip()
                if body and number:
                    deva_n = str(number).translate(_ASCII_TO_DEVA)
                    current.append((number, f"{body} ॥ {deva_n} ॥"))
                buf = []
            else:
                buf.append(line)
        # linhas sem daṇḍa/marcador: rubrica ou anotação — descartadas
    trailing = " ".join(buf).strip()
    if trailing and current:
        # arca final sem numeração: registra sem número (0) para contagem honesta
        current.append((0, trailing))
    if current or not sections:
        sections.append((marker, current))
    return [(m, sec) for m, sec in sections if sec]


def griffith_hymn_counts(griffith: dict) -> dict[tuple[int, int, int], dict[int, int]]:
    """(part, book, chapter) → {hymn: nº de versos}, na ordem dos hinos."""
    out: dict[tuple[int, int, int], dict[int, int]] = {}
    for hymn in griffith.get("hymns") or []:
        key = (int(hymn["part"]), int(hymn["book"]), int(hymn["chapter"]))
        out.setdefault(key, {})[int(hymn["hymn"])] = len(hymn.get("verses") or [])
    return out


def hymn_identity_sa(kind: str, nums: tuple[int, ...]) -> tuple[int, int, int, int]:
    """Identidade (part, book, chapter, hymn) de uma página no esquema sa.

    part1: daśatī D do prapāṭhaka P → hino (1, P, 1|2, D); āraṇya → chapter 3;
    mahānāmnya → chapter 4; part2: ardha A do prapāṭhaka P → (2, P, A, m) com
    m = seção numerada do ardha (resolvida no pareamento).
    """
    if kind == "dasati":
        p, d = nums
        # Griffith reinicia a numeração por capítulo: daśatīs 1–5 = hinos 1–5
        # do capítulo 1; 6–10 = hinos 1–5 do capítulo 2.
        return (1, p, (d - 1) // 5 + 1, (d - 1) % 5 + 1)
    if kind == "aranya":
        return (1, 1, 3, nums[0])
    if kind == "mahanamnya":
        return (1, 1, 4, 1)
    if kind == "ardha":
        p, a = nums
        return (2, p, a, 0)  # hymn definido por seção
    raise ValueError(f"kind inválido: {kind}")


def pair_ardha_sections(
    sections: list[tuple[int | None, list[tuple[int, str]]]],
    gr_counts: dict[int, int],
) -> tuple[list[tuple[int, list[tuple[int, str]]]], list[str]]:
    """Pareia seções do ardha com hinos Griffith, ancorado no marcador.

    O marcador da seção (nº do grupo de recitação no śākhā) é o número do hino
    Griffith equivalente — validado por conteúdo (s5 "राजन्…" ≡ g5 "O King…")
    e pelas lacunas reais do Griffith (b7c2 omite g4 e g9, presentes no sa).
    A contagem de arcas tem que bater exatamente. Seção com marcador sem hino
    Griffith de mesmo número sai sa-only (hymn = −marcador no retorno, para o
    caller distinguir; o caller decide o hymn final). Marcadores ausentes na
    página (lacunas no wikisource) são reportados.
    Retorna ([(hymn, arcas)], avisos).
    """
    paired: list[tuple[int, list[tuple[int, str]]]] = []
    notes: list[str] = []
    seen_markers: set[int] = set()
    for marker, sec in sections:
        if marker is None:
            notes.append(f"seção sem marcador ({len(sec)} arcas) — pulada")
            continue
        seen_markers.add(marker)
        expected = gr_counts.get(marker)
        if expected is None:
            notes.append(
                f"seção {marker} ({len(sec)} arcas): hino Griffith ausente — sai sa-only"
            )
            paired.append((-marker, sec))
            continue
        if expected != len(sec):
            notes.append(
                f"seção {marker}: {len(sec)} arcas vs hino {marker} Griffith {expected} — pulada"
            )
            continue
        paired.append((marker, sec))
    for hymn_n in sorted(gr_counts):
        if hymn_n not in seen_markers:
            notes.append(f"hino {hymn_n} Griffith sem seção correspondente no wikisource")
    return paired, notes


def pair_dasati(
    arcas: list[tuple[int, str]],
    gr_count: int | None,
) -> tuple[list[tuple[int, list[tuple[int, str]]]], list[str]]:
    """Pareia a daśatī inteira (um grupo) com o hino Griffith de mesmo número."""
    notes: list[str] = []
    if gr_count is None:
        notes.append("hino Griffith ausente — daśatī pulada")
        return [], notes
    if gr_count != len(arcas):
        notes.append(
            f"daśatī: {len(arcas)} arcas vs Griffith {gr_count} — daśatī pulada"
        )
        return [], notes
    return [(0, arcas)], notes


def build_hymns(
    fetched: list[tuple[str, tuple[int, ...], str]],
    griffith: dict,
) -> tuple[dict[str, Any], list[str]]:
    """Monta o payload compacto sa a partir dos raws buscados.

    `fetched`: [(kind, nums, raw)]. Retorna ({work…, hymns:[…]}, avisos).
    Hinos sa-only (āraṇya/mahānāmnya) entram com chapter/hymn próprios (3 e 4);
    no Uttarārciká a seção pareada vira o hymn do hino sa; versos recebem
    numeração relativa ao hino (n = ordem do arca dentro do hino).
    """
    gr_counts = griffith_hymn_counts(griffith)
    hymns: list[dict[str, Any]] = []
    notes: list[str] = []
    for kind, nums, raw in fetched:
        sections = parse_wikisource_sections(raw)
        part, book, chapter, hymn0 = hymn_identity_sa(kind, nums)
        page_url = f"{_WIKISOURCE_BASE}{page_title(kind, *nums).replace(' ', '_')}"
        if kind == "ardha":
            counts = gr_counts.get((2, book, chapter), {})
            paired, sec_notes = pair_ardha_sections(sections, counts)
            notes.extend(f"2.{book}.{chapter}: {n}" for n in sec_notes)
            for hymn, sec in paired:
                # hymn negativo = sa-only (hino que o Griffith omite); usa o
                # próprio marcador, que não colide com hinos Griffith do capítulo
                # fragmento #section-N no URL: o ingest dedupe por source_url e
                # vários hinos saem da mesma página de ardha
                hymns.append(
                    _hymn_entry(
                        2, book, chapter, abs(hymn), sec, f"{page_url}#section-{abs(hymn)}"
                    )
                )
        elif kind == "dasati":
            counts = gr_counts.get((1, book, chapter), {})
            arcas = sections[0][1] if sections else []
            paired, d_notes = pair_dasati(arcas, counts.get(hymn0))
            notes.extend(f"1.{book}.{chapter}.{hymn0}: {n}" for n in d_notes)
            for _, sec in paired:
                hymns.append(_hymn_entry(1, book, chapter, hymn0, sec, page_url))
        else:
            # āraṇya/mahānāmnya: sa-only, entra sempre (sem par Griffith)
            arcas = sections[0][1] if sections else []
            hymns.append(_hymn_entry(part, book, chapter, hymn0, arcas, page_url))
    hymns.sort(key=lambda h: (h["part"], h["book"], h["chapter"], h["hymn"]))
    return {
        "work": "samaveda",
        "language": "sa",
        "license": "cc-by-sa",
        "edition": "Kauthuma śākhā, sa.wikisource",
        "attribution": _ATTRIBUTION,
        "source_url": WIKISOURCE_PAGE,
        "hymns": hymns,
    }, notes


def _hymn_entry(
    part: int,
    book: int,
    chapter: int,
    hymn: int,
    arcas: list[tuple[int, str]],
    page_url: str = "",
) -> dict[str, Any]:
    start = arcas[0][0] if arcas and arcas[0][0] else None
    verses = []
    for offset, (_, text) in enumerate(arcas, start=1):
        verses.append({"n": offset, "text": text})
    entry: dict[str, Any] = {
        "part": part,
        "book": book,
        "chapter": chapter,
        "hymn": hymn,
        "heading": None,
        "verses": verses,
    }
    if page_url:
        entry["source_url"] = page_url
    if start:
        entry["arca_start"] = start
    return entry
